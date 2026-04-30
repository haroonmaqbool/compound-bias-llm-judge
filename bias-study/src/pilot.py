# src/pilot.py
# Pilot run on first 10 pairs only.
# Run this BEFORE the full experiment to catch API errors early.

import sys, os, json

# Ensure project root is on sys.path regardless of where this script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment import run_experiment

print("=" * 50)
print("PILOT RUN - 10 pairs only")
print("=" * 50)

pairs = []
with open("data/pairs.jsonl") as f:
    for line in f:
        if line.strip():
            pairs.append(json.loads(line))

pilot_pairs = pairs[:10]

df = run_experiment(
    pairs=pilot_pairs,
    output_file="results/raw/pilot_results.jsonl",
    dataset_tag="mt_bench",
    language_tag="en",
)

print("\nPilot results preview:")
print(df.groupby(["judge", "position", "length"])["winner"].value_counts())
print("\nPilot complete - check results/raw/pilot_results.jsonl")
print("  If everything looks right, run experiment.py for the full 80 pairs.")