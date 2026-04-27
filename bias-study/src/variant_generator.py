# src/variant_generator.py
# Reads raw MT-Bench records and produces short/long variants of each response
# using Groq (LLaMA-3-70B-8192) — fast & free.
# Output: data/pairs.jsonl  (one JSON object per line)

import os, json, time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL  = os.getenv("GROQ_VARIANT_MODEL", "llama-3.3-70b-versatile")


# ── Core variant generator ────────────────────────────────────────────────────

def generate_variant(text: str, style: str) -> str:
    """
    Generate a short (~50%) or long (~150%) rewrite of `text` via Groq.
    style: 'short' | 'long'
    Returns the rewritten string.
    """
    if style == "short":
        instruction = (
            "Rewrite the following text at roughly half the length. "
            "Preserve all key facts and meaning. Return ONLY the rewritten text."
        )
    else:
        instruction = (
            "Rewrite the following text at roughly 1.5× the length. "
            "Add relevant detail and explanation without inventing new facts. "
            "Return ONLY the rewritten text."
        )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user",   "content": text},
        ],
        max_tokens=1024,
        temperature=0.3,
    )
    time.sleep(0.5)   # safe pacing — Groq free tier: ~30 req/min
    return response.choices[0].message.content.strip()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_pairs(
    input_path:  str = "data/mt_bench.json",
    output_path: str = "data/pairs.jsonl",
    limit:       int = 80,
):
    """
    Read raw MT-Bench records, extract question + response pair,
    generate short/long variants for each response, and save to output_path.

    Each output line:
    {
        "question": str,
        "short_a":  str,
        "long_a":   str,
        "short_b":  str,
        "long_b":   str,
    }
    """
    # Load raw records
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    records = records[:limit]
    pairs   = []

    for i, record in enumerate(records):
        question = record["conversation_a"][0]["content"]  # first user turn = question

        # Last assistant message = the model's full response
        resp_a_raw = next(
            (m["content"] for m in reversed(record["conversation_a"]) if m["role"] == "assistant"),
            ""
        )
        resp_b_raw = next(
            (m["content"] for m in reversed(record["conversation_b"]) if m["role"] == "assistant"),
            ""
        )

        if not resp_a_raw or not resp_b_raw:
            print(f"  Skipping record {i} — missing response")
            continue

        print(f"  [{i+1}/{len(records)}] Generating variants...")
        pair = {
            "question": question,
            "short_a":  generate_variant(resp_a_raw, "short"),
            "long_a":   generate_variant(resp_a_raw, "long"),
            "short_b":  generate_variant(resp_b_raw, "short"),
            "long_b":   generate_variant(resp_b_raw, "long"),
        }
        pairs.append(pair)

        with open(output_path, "a") as f:
            f.write(json.dumps(pair) + "\n")

    print(f"\n✓ {len(pairs)} pairs saved to {output_path}")
    return pairs


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_pairs(
        input_path  = "data/mt_bench.json",
        output_path = "data/pairs.jsonl",
        limit       = 80,
    )