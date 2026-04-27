import express from "express";
import cors from "cors";
import { spawn } from "child_process";
import readline from "readline";
import path from "path";
import { fileURLToPath } from "url";


const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

// ── spawn the Python bridge once at startup ───────────────────────────────────
const pythonBin   = process.env.PYTHON_BIN   || "python";
const bridgePath  = path.join(__dirname, "bridge.py");

const bridge = spawn(pythonBin, [bridgePath], {
  env: {
    ...process.env,
    KWICKCHAT_MODEL:  process.env.KWICKCHAT_MODEL  || "",
    BNN_CHECKPOINT:   process.env.BNN_CHECKPOINT   || "../models/bnn.pkl",
    NUM_SUGGESTIONS:  process.env.NUM_SUGGESTIONS  || "4",
  },
});

// queue of pending { resolve, reject } callbacks — one per request
const pendingRequests = [];
let bridgeReady = false;

// read newline-delimited JSON responses from Python
const rl = readline.createInterface({ input: bridge.stdout });
rl.on("line", (line) => {
  let msg;
  try { msg = JSON.parse(line); } catch { return; }

  if (msg.status === "ready") {
    bridgeReady = true;
    console.log("[server] Python bridge ready.");
    return;
  }

  const next = pendingRequests.shift();
  if (next) next.resolve(msg);
});

bridge.stderr.on("data", (d) => process.stderr.write(`[bridge] ${d}`));
bridge.on("close", (code) => {
  console.error(`[server] Python bridge exited with code ${code}`);
  // reject any waiting requests
  pendingRequests.forEach(({ reject }) =>
    reject(new Error("Bridge process died"))
  );
});

// send a request to Python and wait for one response line
function askBridge(payload) {
  return new Promise((resolve, reject) => {
    if (!bridgeReady) {
      return reject(new Error("Bridge not ready yet — try again in a moment"));
    }
    pendingRequests.push({ resolve, reject });
    bridge.stdin.write(JSON.stringify(payload) + "\n");
  });
}

// ── routes ────────────────────────────────────────────────────────────────────

// Health check
app.get("/health", (req, res) => {
  res.json({ ready: bridgeReady });
});

// Main prediction endpoint
// Body: { text: string, keywords: string, persona: string[] }
app.post("/predict", async (req, res) => {
  const { inputSoFar: text, keywords, persona } = req.body;

  if (!text || text.trim() === "") {
    return res.json({ suggestions: [] });
  }

  try {
    const result = await askBridge({
      inputSoFar: text.trim(),
      keywords: (keywords || "").trim(),
      persona:  Array.isArray(persona) ? persona : [],
    });

    if (result.error) {
      return res.status(500).json({ error: result.error });
    }

    res.json({ suggestions: result.suggestions || [] });
  } catch (err) {
    res.status(503).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`[server] API running on port ${PORT}`);
  console.log(`[server] Waiting for Python bridge to load models...`);
});

export default app;