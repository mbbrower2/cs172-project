"""
bridge.py — autocomplete bridge between Node server and kwickChat.

Instead of generating replies to a speaking partner, this completes
the sentence the user has started typing, conditioned on their persona.

Input:  {"inputSoFar": "I really enjoy going"}
Output: {"suggestions": ["I really enjoy going hiking on weekends.",
                          "I really enjoy going to the park with my dog.", ...]}
"""

import sys
import os

KWICKCHAT_DIR = "/Users/maddiebrower/workspace/tufts/spring2026/cs172/kwickChat"
BNN_DIR       = "/Users/maddiebrower/workspace/tufts/spring2026/cs172/persona_bnn"

sys.path.insert(0, KWICKCHAT_DIR)
sys.path.insert(0, BNN_DIR)

import json
import random
import pickle
import warnings

import torch
import torch.nn.functional as F
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer

from utils import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import download_pretrained_model
from learner import PersonaLearner

MODEL_CHECKPOINT = os.environ.get("KWICKCHAT_MODEL", "")
BNN_CHECKPOINT   = os.environ.get("BNN_CHECKPOINT", "../models/bnn.pkl")
NUM_SUGGESTIONS  = int(os.environ.get("NUM_SUGGESTIONS", "4"))
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"


class Args:
    device      = DEVICE
    max_length  = 30    # longer — completing a full sentence
    min_length  = 3
    temperature = 0.85  # slightly higher for variety in completions
    top_k       = 50
    top_p       = 0.9
    no_sample   = False


def top_filtering(logits, top_k=0, top_p=0.9, threshold=-float("Inf"),
                  filter_value=-float("Inf")):
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        logits[sorted_indices[sorted_indices_to_remove]] = filter_value
    logits[logits < threshold] = filter_value
    return logits


def autocomplete(model, tokenizer, personality, partial_input, args):
    """
    Continue partial_input into full sentences conditioned on personality.
    Uses the same input format as the original kwickChat interact.py.
    """
    bos, eos, speaker1, speaker2, key = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    special_ids = set(tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS))

    partial_ids = tokenizer.encode(partial_input)
    # Use partial input as both the "history" (what was said) and the
    # seed for the reply — this prompts the model to continue the thought
    history     = [partial_ids]
    key_phrase  = [[key]]   # empty key placeholder required by build_input_from_segments

    suggestions = []

    for seed in range(NUM_SUGGESTIONS):
        random.seed(seed)
        torch.random.manual_seed(seed)

        current_output = list(partial_ids)  # start from what user typed

        with torch.no_grad():
            for _ in range(args.max_length):
                instance = build_input_from_segments(
                    personality,
                    history,
                    current_output,
                    tokenizer,
                    key_phrase,
                    with_eos=False,
                )
                input_ids      = torch.tensor(instance["input_ids"],      device=DEVICE).unsqueeze(0)
                token_type_ids = torch.tensor(instance["token_type_ids"], device=DEVICE).unsqueeze(0)

                logits = model(input_ids, token_type_ids=token_type_ids)[0]
                logits = logits[0, -1, :] / args.temperature
                logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
                probs  = F.softmax(logits, dim=-1)

                if args.no_sample:
                    prev = torch.topk(probs, 1)[1]
                else:
                    prev = torch.multinomial(probs, 1)

                token_id = prev.item()

                # stop at any special token
                if token_id in special_ids:
                    break

                current_output.append(token_id)

                # stop at sentence-ending punctuation once past min_length
                if len(current_output) - len(partial_ids) >= args.min_length:
                    decoded = tokenizer.decode(current_output, skip_special_tokens=True)
                    if decoded.rstrip().endswith((".", "!", "?")):
                        break

        completion = tokenizer.decode(current_output, skip_special_tokens=True)

        # only return if the model actually added something
        if completion.strip() != partial_input.strip():
            suggestions.append(completion)

    # deduplicate while preserving order
    seen = set()
    unique = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    return unique
    """
    Continue `partial_input` into a full sentence, conditioned on `personality`.
    Returns a list of NUM_SUGGESTIONS completions, each starting with the
    original partial input so the user sees their text preserved.
    """
    bos, eos, speaker1, speaker2, key = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    special_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    partial_ids = tokenizer.encode(partial_input)
    suggestions = []

    for seed in range(NUM_SUGGESTIONS):
        random.seed(seed)
        torch.random.manual_seed(seed)

        # Build input: [bos + persona] + [speaker1 + partial_ids_so_far]
        persona_seq   = [bos] + [tok for p in personality for tok in p]
        sequence_ids  = persona_seq + [speaker1] + partial_ids
        current_output = list(partial_ids)  # we'll extend this

        with torch.no_grad():
            for _ in range(args.max_length):
                input_tensor = torch.tensor([sequence_ids], device=DEVICE)
                # token_type_ids: speaker2 for persona, speaker1 for the reply
                token_types = (
                    [speaker2] * len(persona_seq) +
                    [speaker1] * len(current_output)
                )
                type_tensor = torch.tensor([token_types], device=DEVICE)

                logits = model(input_tensor, token_type_ids=type_tensor)[0]
                logits = logits[0, -1, :] / args.temperature
                logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
                probs  = F.softmax(logits, dim=-1)

                prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)

                # stop at EOS or special token
                if prev.item() in special_ids:
                    break

                current_output.append(prev.item())
                sequence_ids.append(prev.item())

                # stop if we hit a sentence-ending punctuation after min_length
                decoded_so_far = tokenizer.decode(current_output, skip_special_tokens=True)
                if len(current_output) >= args.min_length:
                    if decoded_so_far.rstrip().endswith((".", "!", "?")):
                        break

        completion = tokenizer.decode(current_output, skip_special_tokens=True)
        suggestions.append(completion)

    return suggestions


def load_kwickchat():
    checkpoint = MODEL_CHECKPOINT or download_pretrained_model()
    tokenizer  = OpenAIGPTTokenizer.from_pretrained(checkpoint)
    model      = OpenAIGPTLMHeadModel.from_pretrained(checkpoint)
    model.to(DEVICE)
    add_special_tokens_(model, tokenizer)
    return model, tokenizer


def load_bnn():
    learner = PersonaLearner()
    if os.path.exists(BNN_CHECKPOINT):
        with open(BNN_CHECKPOINT, "rb") as f:
            data = pickle.load(f)
        learner.vocab   = data["vocab"]
        learner.tfidf   = data["tfidf"]
        learner.model   = data["model"]
        learner._replay = data["replay"]
        sys.stderr.write(f"[bridge] BNN loaded ({len(learner.vocab)} tags)\n")
    else:
        sys.stderr.write("[bridge] No BNN checkpoint — persona tags disabled\n")
    return learner


def tag_to_sentence(tag):
    templates = {
        "interest":    "i enjoy {v}.",
        "job":         "i work as a {v}.",
        "edu":         "i attended {v}.",
        "trait":       "i am {v}.",
        "life":        "i am {v}.",
        "pet":         "i have a {v}.",
        "value":       "i believe in {v}.",
        "fav_food":    "my favorite food is {v}.",
        "fav_color":   "my favorite color is {v}.",
        "skill":       "i can {v}.",
        "achievement": "i achieved {v}.",
    }
    if ":" not in tag:
        return f"i am interested in {tag.replace('_', ' ')}."
    prefix, value = tag.split(":", 1)
    value = value.replace("_", " ")
    return templates.get(prefix, "i am associated with {v}.").format(v=value)


def main():
    sys.stderr.write("[bridge] Loading kwickChat model...\n")
    model, tokenizer = load_kwickchat()
    sys.stderr.write("[bridge] Loading BNN...\n")
    bnn = load_bnn()
    sys.stderr.write("[bridge] Ready.\n")
    sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
    sys.stdout.flush()

    args             = Args()
    persona_cache    = []   # grows as BNN discovers tags across the session
    seen_tags        = set()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps({"error": "invalid json"}) + "\n")
            sys.stdout.flush()
            continue

        input_so_far = req.get("inputSoFar", "").strip()

        if not input_so_far:
            sys.stdout.write(json.dumps({"suggestions": []}) + "\n")
            sys.stdout.flush()
            continue

        # ── update persona from BNN ───────────────────────────────────────────
        if bnn.model is not None:
            tags, uncertainty = bnn.infer(input_so_far)
            for tag in sorted(tags, key=lambda t: uncertainty.get(t, 1.0)):
                if tag not in seen_tags:
                    sentence = tag_to_sentence(tag)
                    encoded  = tokenizer.encode(sentence)
                    persona_cache.append(encoded)
                    seen_tags.add(tag)
                    sys.stderr.write(f"[bridge] New persona tag: {tag}\n")

        # ── autocomplete ──────────────────────────────────────────────────────
        suggestions = autocomplete(model, tokenizer, persona_cache, input_so_far, args)

        sys.stdout.write(json.dumps({"suggestions": suggestions}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()