import express from "express";
import cors from "cors";

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

const phrases = [
  { id: 1, text: "I want water", tags: ["drink", "thirst", "water"] },
  { id: 2, text: "I need help", tags: ["help", "assist", "support"] },
  { id: 3, text: "I am hungry", tags: ["food", "eat", "hungry"] },
];

app.post("/predict", (req, res) => {
  const { text } = req.body;

  if (!text || text.trim() === "") {
    return res.json({ suggestions: [] });
  }

  const textLower = text.toLowerCase();
  const matches = phrases.filter(
    (phrase) =>
      phrase.text.toLowerCase().includes(textLower) ||
      phrase.tags?.some((tag) => tag.toLowerCase().includes(textLower))
  );

  res.json({ suggestions: matches.map((m) => m.text) });
});

app.listen(PORT, () => {
  console.log(`API server running on port ${PORT}`);
});

export default app;