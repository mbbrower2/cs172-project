
export async function getSuggestionsFromText(text) {
  if (!text || text.trim() === "") {
    return [];
  }

  try {
    const response = await fetch("http://localhost:3001/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      console.error("Model API error:", response.statusText);
      return [];
    }

    const data = await response.json();

    // Expecting: { suggestions: [...] }
    if (!data || !Array.isArray(data.suggestions)) {
      console.error("Model API returned unexpected format:", data);
      return [];
    }

    return data.suggestions;
  } catch (err) {
    console.error("Error calling model API:", err);
    return [];
  }
}
