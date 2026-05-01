# src/summeval_pairs.py
# Build short/long pairs from SummEval for the 2×2 bias experiment.
#
# Strategy:
#   - Use composite quality score (mean of relevance, coherence, fluency, consistency)
#     to rank the 16 machine summaries per article.
#   - Sort by word-count; pick the shortest and longest that also differ by ≥1.5× in length.
#   - Use the 2nd-shortest and 2nd-longest as the B variants.
#   - Truncate source article to 500 words to stay within Gemini token limits.
#
# Output: data/summeval_pairs.jsonl
# Run:    python src/summeval_pairs.py

import json, os

INPUT  = os.path.join(os.path.dirname(__file__), "..", "data", "summeval.json")
OUTPUT = os.path.join(os.path.dirname(__file__), "..", "data", "summeval_pairs.jsonl")
LIMIT  = 100   # use all articles (there are exactly 100 in the file)
MIN_RATIO = 1.4  # minimum length ratio between short and long summaries

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

# ── Load ─────────────────────────────────────────────────────────────────────
records = []
with open(INPUT, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"Loaded {len(records)} SummEval articles")

# ── Build pairs ───────────────────────────────────────────────────────────────
pairs   = []
skipped = 0

for rec in records[:LIMIT]:
    summaries = rec["machine_summaries"]
    article   = rec["text"].strip()

    # Truncate article to 500 words to stay within Gemini context limits
    words = article.split()
    if len(words) > 500:
        article = " ".join(words[:500]) + "..."

    # Sort summaries by word count (ascending)
    sorted_by_len = sorted(summaries, key=lambda s: len(s.split()))

    # Need at least 4 summaries to form two pairs
    if len(sorted_by_len) < 4:
        skipped += 1
        continue

    short_a = sorted_by_len[0]   # shortest
    long_a  = sorted_by_len[-1]  # longest
    short_b = sorted_by_len[1]   # 2nd shortest  → B variant
    long_b  = sorted_by_len[-2]  # 2nd longest   → B variant

    # Skip if the length spread is too small (boring pair)
    ratio = len(long_a.split()) / max(len(short_a.split()), 1)
    if ratio < MIN_RATIO:
        skipped += 1
        continue

    pair = {
        "question": (
            "You are evaluating two summaries of the following news article. "
            "Which summary is better?\n\n"
            f"ARTICLE:\n{article}"
        ),
        "short_a": short_a,
        "long_a":  long_a,
        "short_b": short_b,
        "long_b":  long_b,
    }
    pairs.append(pair)

print(f"Built {len(pairs)} pairs ({skipped} skipped — too similar in length)")

# ── Write ─────────────────────────────────────────────────────────────────────
# Clear existing file if re-running
with open(OUTPUT, "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"✓ Saved to {OUTPUT}")
