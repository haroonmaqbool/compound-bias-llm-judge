# src/consolidate.py
# Merges all raw JSONL result files into one clean master DataFrame.
# Run after all experiments (MT-Bench, SummEval, Urdu) are complete.
# Output: results/master_df.csv

import os, json
import pandas as pd

RESULT_FILES = [
    ("results/results.jsonl",            "mt_bench", "en"),
    ("results/raw/summeval_results.jsonl","summeval", "en"),
    ("results/raw/urdu_results.jsonl",   "mt_bench", "ur"),
]

all_rows = []

for filepath, dataset_tag, language_tag in RESULT_FILES:
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filepath} not found — skipping")
        continue

    count = 0
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                # Ensure consistent columns
                row.setdefault("dataset",  dataset_tag)
                row.setdefault("language", language_tag)
                all_rows.append(row)
                count += 1
            except json.JSONDecodeError as e:
                print(f"  [WARN] Bad JSON in {filepath}: {e}")

    print(f"  ✓ {count} rows loaded from {filepath}")

# ── Build master DataFrame ────────────────────────────
df = pd.DataFrame(all_rows)

# Enforce column types
df["qid"]      = df["qid"].astype(int)
df["position"] = df["position"].astype("category")
df["length"]   = df["length"].astype("category")
df["judge"]    = df["judge"].astype("category")
df["winner"]   = df["winner"].astype(str)
df["dataset"]  = df["dataset"].astype("category")
df["language"] = df["language"].astype("category")

# Drop rows where judge errored
before = len(df)
df = df[df["winner"].isin(["A", "B"])]
after  = len(df)
print(f"\n  Dropped {before - after} error/invalid rows")

# ── Save ─────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
df.to_csv("results/master_df.csv", index=False)

print(f"\n✓ Master DataFrame: {len(df)} rows × {len(df.columns)} columns")
print(f"  Saved to results/master_df.csv")
print(f"\nColumn summary:")
print(df.dtypes)
print(f"\nJudge counts:")
print(df["judge"].value_counts())
print(f"\nDataset counts:")
print(df["dataset"].value_counts())