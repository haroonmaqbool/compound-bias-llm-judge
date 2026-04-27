# src/variant_generator.py
# Uses Groq (llama3-70b, free) to generate short + long versions
# of each MT-Bench response, keeping factual content identical.

import os, time, json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

SHORT_PROMPT = (
    "Rewrite the following response to be significantly shorter while keeping "
    "all factual content identical. Output only the rewritten response, nothing else.\n\n"
    "Response:\n{response}"
)

LONG_PROMPT = (
    "Rewrite the following response to be significantly longer and more detailed while keeping "
    "all factual content identical. Do not add new facts — only elaborate. "
    "Output only the rewritten response, nothing else.\n\n"
    "Response:\n{response}"
)


def generate_variant(response: str, mode: str) -> str:
    """mode: 'short' or 'long'"""
    prompt = SHORT_PROMPT if mode == "short" else LONG_PROMPT
    result = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt.format(response=response)}],
        max_tokens=1024,
    )
    time.sleep(0.5)  # stay under Groq rate limit
    return result.choices[0].message.content.strip()


def build_pairs(mt_bench_path: str, output_path: str, limit: int = None):
    """
    Reads mt_bench.json, picks the first 'limit' pairs (or all if None),
    generates short/long variants for both responses in each pair,
    and saves to output_path as JSONL.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(mt_bench_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    if limit:
        records = records[:limit]

    pairs = []
    for i, record in enumerate(records):
        # MT-Bench stores model responses differently — adapt keys as needed
        question = record.get("question", record.get("turns", [""])[0])
        responses = record.get("model_judgments", {})

        # Grab first two model responses available
        model_names = list(responses.keys())
        if len(model_names) < 2:
            print(f"  Skipping record {i} — fewer than 2 responses")
            continue

        resp_a_raw = responses[model_names[0]].get("response", "")
        resp_b_raw = responses[model_names[1]].get("response", "")

        print(f"  [{i+1}/{len(records)}] Generating variants...")
        pair = {
            "question":  question,
            "short_a":   generate_variant(resp_a_raw, "short"),
            "long_a":    generate_variant(resp_a_raw, "long"),
            "short_b":   generate_variant(resp_b_raw, "short"),
            "long_b":    generate_variant(resp_b_raw, "long"),
        }
        pairs.append(pair)

        with open(output_path, "a") as f:
            f.write(json.dumps(pair) + "\n")

    print(f"\n✓ {len(pairs)} pairs saved to {output_path}")
    return pairs


if __name__ == "__main__":
    build_pairs(
        mt_bench_path="data/mt_bench.json",
        output_path="data/pairs.jsonl",
        limit=None,   # set to 10 for pilot run
    )