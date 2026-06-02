# src/experiment.py  — week3-fixes
# 2×2 Factorial Experiment Runner
# Conditions: (position=first|second) × (length=short|long)
# Saves incrementally to results/results.jsonl — resume-safe.
#
# Fix (week3): Added already-done key checking so re-running the script
# never appends duplicate (qid, judge, position, length) rows.

import sys, os, json
import pandas as pd

# Ensure project root is on sys.path regardless of where this script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.judge import JUDGES, call_judge

os.makedirs("results/raw", exist_ok=True)


def _load_done_keys(output_file: str) -> set:
    """Return set of (qid, judge, position, length) already saved in output_file."""
    done = set()
    if not os.path.exists(output_file):
        return done
    with open(output_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["qid"], r["judge"], r["position"], r["length"]))
            except Exception:
                pass
    return done


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

    # Fix (week3): load already-done keys to skip duplicate work on re-run
    done_keys = _load_done_keys(output_file)
    if done_keys:
        print(f"  ↻ Resume: {len(done_keys)} (qid, judge, position, length) combos already done, skipping.")

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
                key = (i, judge_name, pos, length)
                if key in done_keys:   # Fix (week3): skip already-recorded rows
                    continue

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
                done_keys.add(key)   # mark as done immediately

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