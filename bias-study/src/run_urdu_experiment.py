# src/run_urdu_experiment.py
# Runs the 2×2 factorial experiment on Urdu-translated MT-Bench pairs.
#
# Prerequisites:
#   1. Run `python src/urdu_translator.py` to generate data/urdu_pairs.jsonl
#
# Expected output: 50 pairs × 4 conditions × 5 judges = 1,000 rows
# Output file:     results/raw/urdu_results.jsonl
#
# Resume-safe: already-logged (qid, judge, position, length) combos are skipped.
# Run: python src/run_urdu_experiment.py

import json, os, sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment import run_experiment

PAIRS_FILE  = os.path.join(os.path.dirname(__file__), "..", "data", "urdu_pairs.jsonl")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "results", "raw", "urdu_results.jsonl")

# ── Load pairs ────────────────────────────────────────────────────────────────
if not os.path.exists(PAIRS_FILE):
    print(f"[ERROR] Urdu pairs file not found: {PAIRS_FILE}")
    print("  Run `python src/urdu_translator.py` first to translate the pairs.")
    sys.exit(1)

pairs = []
with open(PAIRS_FILE, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            pairs.append(json.loads(line))

print(f"Loaded {len(pairs)} Urdu pairs")
print(f"Expected rows: {len(pairs)} pairs × 4 conditions × 5 judges = {len(pairs)*20} rows")
print(f"Output: {OUTPUT_FILE}")
print()

# ── Run experiment ────────────────────────────────────────────────────────────
run_experiment(
    pairs=pairs,
    output_file=OUTPUT_FILE,
    dataset_tag="mt_bench",
    language_tag="ur",
)
