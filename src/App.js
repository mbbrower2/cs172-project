// src/App.js
import React, { useState } from "react";
import { getSuggestionsFromText } from "./model";
import "./App.css";

function App() {
  const [input, setInput] = useState("");
  const [suggestions, setSuggestions] = useState([]);

  async function handleChange(e) {
    const value = e.target.value;
    setInput(value);

    const newSuggestions = await getSuggestionsFromText(value);
    setSuggestions(newSuggestions);
  }

  function handleSuggestionClick(sentence) {
    setInput(sentence);

  }

  function handleClear() {
    setInput("");
    setSuggestions([]);
  }

  function handleSpeak() {
    if ("speechSynthesis" in window && input.trim().length > 0) {
      const utterance = new SpeechSynthesisUtterance(input);
      window.speechSynthesis.speak(utterance);
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
        {suggestions.length === 0 && (
          <p className="hint">Start typing a keyword like “hungry”, “tired”, or “bathroom”.</p>
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
