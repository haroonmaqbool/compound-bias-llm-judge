# src/download_data.py
# Run once to download all datasets locally.
# Fixed dataset names — HuggingFaceH4/mt_bench_evaluated no longer exists.

import os, json
from datasets import load_dataset

os.makedirs("data", exist_ok=True)

# ── 1. MT-Bench Human Judgments (official LMSYS dataset) ──────────────
# Contains 80 questions, pairwise model responses, and human winner labels.
# Columns: question_id, model_a, model_b, winner, conversation_a, conversation_b, turn
print("Downloading MT-Bench...")
mt_bench = load_dataset("lmsys/mt_bench_human_judgments")

# We use the 'human' split — it has expert pairwise judgments
df_train = mt_bench["human"]
df_train.to_json("data/mt_bench.json", orient="records", lines=True)
print(f"  ✓ {len(df_train)} rows saved to data/mt_bench.json")

# Also save the gpt4 split for reference
mt_bench["gpt4_pair"].to_json("data/mt_bench_gpt4.json", orient="records", lines=True)
print(f"  ✓ {len(mt_bench['gpt4_pair'])} gpt4 rows saved to data/mt_bench_gpt4.json")

# ── 2. SummEval ───────────────────────────────────────────────────────
print("\nDownloading SummEval...")
summeval = load_dataset("mteb/summeval")
summeval["test"].to_json("data/summeval.json", orient="records", lines=True)
print(f"  ✓ {len(summeval['test'])} rows saved to data/summeval.json")

# ── 3. FairEval (manual clone) ────────────────────────────────────────
print("\nFairEval — clone manually by running this command in terminal:")
print("  git clone https://github.com/i-Eval/FairEval")
print("  (Then the file will be at: FairEval/data/faireval.json)")

print("\n✅ All downloads complete.")