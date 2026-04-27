// src/App.js
import React, { useState, useRef, useCallback } from "react";
import { getSuggestionsFromText } from "./model";
import "./App.css";

const DEBOUNCE_MS = 600; // wait 600ms after user stops typing before fetching

function App() {
  const [input, setInput]           = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [loading, setLoading]       = useState(false);
  const debounceTimer               = useRef(null);
  const lastFetchedText             = useRef("");  // track what we last fetched for

  const fetchSuggestions = useCallback(async (text) => {
    if (!text || text.trim() === "") {
      setSuggestions([]);
      return;
    }
    // Don't re-fetch if the text hasn't changed since last fetch
    if (text.trim() === lastFetchedText.current) return;

    lastFetchedText.current = text.trim();
    setLoading(true);
    const results = await getSuggestionsFromText(text);
    setLoading(false);
    setSuggestions(results);
  }, []);

  function handleChange(e) {
    const value = e.target.value;
    setInput(value);

    // Cancel any pending fetch
    if (debounceTimer.current) clearTimeout(debounceTimer.current);

    if (!value.trim()) {
      setSuggestions([]);
      lastFetchedText.current = "";
      return;
    }

    // Schedule a fetch only after the user pauses typing
    debounceTimer.current = setTimeout(() => {
      fetchSuggestions(value);
    }, DEBOUNCE_MS);
  }

  function handleSuggestionClick(sentence) {
    setInput(sentence);
    // Clear suggestions so they don't re-fetch just from the click
    setSuggestions([]);
    lastFetchedText.current = sentence;
  }

  function handleClear() {
    if (debounceTimer.current) clearTimeout(debounceTimer.current);
    setInput("");
    setSuggestions([]);
    lastFetchedText.current = "";
  }

  function handleSpeak() {
    if ("speechSynthesis" in window && input.trim().length > 0) {
      const utterance = new SpeechSynthesisUtterance(input);
      window.speechSynthesis.speak(utterance);
      // Refresh suggestions after speaking — new context
      lastFetchedText.current = "";
      fetchSuggestions(input);
    }
  }

  return (
    <div className="app">
      <h1>AAC UI Prototype</h1>

      <div className="composer">
        <textarea
          className="input"
          value={input}
          onChange={handleChange}
          placeholder="Type here (e.g., 'I am a farmer')..."
        />

        <div className="buttons-row">
          <button onClick={handleSpeak} disabled={!input.trim()}>
            Speak
          </button>
          <button onClick={handleClear}>Clear</button>
        </div>
      </div>

      <div className="suggestions">
        <h2>Suggestions</h2>
        {loading && <p className="hint">Generating suggestions…</p>}
        {!loading && suggestions.length === 0 && (
          <p className="hint">
            Start typing a keyword like "hungry", "tired", or "bathroom".
          </p>
        )}
        <div className="suggestion-list">
          {suggestions.map((s, i) => (
            <button
              key={i}
              className="suggestion-button"
              onClick={() => handleSuggestionClick(s)}
            >
              {s}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;