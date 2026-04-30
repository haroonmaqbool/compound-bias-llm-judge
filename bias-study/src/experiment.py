# src/experiment.py
# 2×2 Factorial Experiment Runner
# Conditions: (position=first|second) × (length=short|long)
# Saves incrementally to results/results.jsonl — resume-safe.

import sys, os, json
import pandas as pd

# Ensure project root is on sys.path regardless of where this script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.judge import JUDGES, call_judge

os.makedirs("results/raw", exist_ok=True)


def run_experiment(
    pairs: list,
    output_file: str = "results/results.jsonl",
    dataset_tag: str = "mt_bench",
    language_tag: str = "en",
    judges_to_run: list = None,   # None = all 5
):
    """
    pairs: list of dicts with keys:
        question, short_a, long_a, short_b, long_b

    Each pair produces 4 conditions × N judges rows.
    """
    judge_names = judges_to_run or list(JUDGES.keys())
    results = []

    for i, pair in enumerate(pairs):
        q = pair["question"]

        # ── 4 conditions of the 2×2 design ───────────
        conditions = [
            ("first",  "short", pair["short_a"], pair["long_b"]),
            ("first",  "long",  pair["long_a"],  pair["short_b"]),
            ("second", "short", pair["long_b"],  pair["short_a"]),
            ("second", "long",  pair["short_b"], pair["long_a"]),
        ]

        for pos, length, resp_a, resp_b in conditions:
            for judge_name in judge_names:
                winner = call_judge(judge_name, q, resp_a, resp_b)
                row = {
                    "qid":      i,
                    "judge":    judge_name,
                    "position": pos,
                    "length":   length,
                    "winner":   winner,
                    "dataset":  dataset_tag,
                    "language": language_tag,
                }
                results.append(row)

                # Incremental save — never lose progress
                with open(output_file, "a") as f:
                    f.write(json.dumps(row) + "\n")

        print(f"  [OK] Pair {i+1}/{len(pairs)} done")

    df = pd.DataFrame(results)
    print(f"\nExperiment complete. {len(df)} rows saved to {output_file}")
    return df


if __name__ == "__main__":
    # Load pairs
    pairs = []
    with open("data/pairs.jsonl") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    run_experiment(
        pairs=pairs,
        output_file="results/results.jsonl",
        dataset_tag="mt_bench",
        language_tag="en",
    )