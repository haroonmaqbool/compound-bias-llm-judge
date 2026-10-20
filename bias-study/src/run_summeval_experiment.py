# src/run_summeval_experiment.py
# Runs the 2×2 factorial experiment on SummEval pairs.
#
# Expected output: ~80-100 pairs × 4 conditions × 5 judges = ~1,600-2,000 rows
# Output file:     results/raw/summeval_results.jsonl
#
# Resume-safe: already-logged (qid, judge, position, length) combos are skipped.
# Run: python src/run_summeval_experiment.py

import json, os, sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment import run_experiment

PAIRS_FILE  = os.path.join(os.path.dirname(__file__), "..", "data", "summeval_pairs.jsonl")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "results", "raw", "summeval_results.jsonl")

# ── Load pairs ────────────────────────────────────────────────────────────────
if not os.path.exists(PAIRS_FILE):
    print(f"[ERROR] Pairs file not found: {PAIRS_FILE}")
    print("  Run `python src/summeval_pairs.py` first to generate it.")
    sys.exit(1)

pairs = []
with open(PAIRS_FILE, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            pairs.append(json.loads(line))

print(f"Loaded {len(pairs)} SummEval pairs")
print(f"Expected rows: {len(pairs)} pairs × 4 conditions × 5 judges = {len(pairs)*20} rows")
print(f"Output: {OUTPUT_FILE}")
print()

# ── Run experiment ────────────────────────────────────────────────────────────
run_experiment(
    pairs=pairs,
    output_file=OUTPUT_FILE,
    dataset_tag="summeval",
    language_tag="en",
)
