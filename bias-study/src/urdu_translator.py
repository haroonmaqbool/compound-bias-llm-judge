# src/urdu_translator.py
# Translates 50 MT-Bench pairs to Urdu using Helsinki-NLP (local, free).
# No API key needed. Runs fully offline after model download.
# pip install transformers sentencepiece torch

import os, json
from transformers import MarianMTModel, MarianTokenizer

os.makedirs("data", exist_ok=True)

MODEL_NAME = "Helsinki-NLP/opus-mt-en-ur"
print(f"Loading {MODEL_NAME}...")
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model     = MarianMTModel.from_pretrained(MODEL_NAME)
print("  ✓ Model loaded")


def translate(text: str) -> str:
    if not text or not text.strip():
        return ""
    inputs  = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def translate_pairs(pairs_path: str, output_path: str, limit: int = 50):
    pairs = []
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    selected = pairs[:limit]
    print(f"Translating {len(selected)} pairs to Urdu...")

    urdu_pairs = []
    for i, pair in enumerate(selected):
        print(f"  [{i+1}/{len(selected)}] Translating...")
        urdu_pair = {
            "question": translate(pair["question"]),
            "short_a":  translate(pair["short_a"]),
            "long_a":   translate(pair["long_a"]),
            "short_b":  translate(pair["short_b"]),
            "long_b":   translate(pair["long_b"]),
        }
        urdu_pairs.append(urdu_pair)

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(urdu_pair, ensure_ascii=False) + "\n")

    print(f"\n✓ {len(urdu_pairs)} Urdu pairs saved to {output_path}")
    return urdu_pairs


if __name__ == "__main__":
    translate_pairs(
        pairs_path="data/pairs.jsonl",
        output_path="data/urdu_pairs.jsonl",
        limit=50,
    )