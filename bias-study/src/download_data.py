# src/download_data.py
# Run once to download all datasets locally

import os
from datasets import load_dataset

os.makedirs("data", exist_ok=True)

# ── MT-Bench ──────────────────────────────────────────
print("Downloading MT-Bench...")
mt_bench = load_dataset("HuggingFaceH4/mt_bench_evaluated")
mt_bench["train"].to_json("data/mt_bench.json")
print(f"  ✓ {len(mt_bench['train'])} examples saved to data/mt_bench.json")

# ── SummEval ──────────────────────────────────────────
print("Downloading SummEval...")
summeval = load_dataset("mteb/summeval")
summeval["test"].to_json("data/summeval.json")
print(f"  ✓ {len(summeval['test'])} examples saved to data/summeval.json")

# ── FairEval ──────────────────────────────────────────
# Run this separately in terminal:
# git clone https://github.com/i-Eval/FairEval
print("\nFairEval: clone manually with:")
print("  git clone https://github.com/i-Eval/FairEval")
print("\nAll datasets ready.")