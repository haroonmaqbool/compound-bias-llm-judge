# Bias Study — LLM-as-Judge Bias Analysis

A systematic study of **position bias** and **verbosity bias** in LLM-as-a-judge evaluation
using a 2×2 factorial design across 5 judges and 3 datasets.

## Project Structure

```
bias-study/
├── data/           # MT-Bench, SummEval, Urdu pairs
├── FairEval/       # Cloned from github.com/i-Eval/FairEval
├── results/        # Experiment outputs
├── figures/        # Publication-ready PDFs
├── src/            # All source code
├── notebooks/      # Pilot exploration
└── paper/          # LaTeX manuscript
```

## Setup

```bash
# 1. Clone repo
git clone <your-repo-url> && cd bias-study

# 2. Clone FairEval
git clone https://github.com/i-Eval/FairEval.git FairEval

# 3. Create & activate environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 4. Install dependencies
pip install -r requirements.txt

# 5. Add API keys
cp .env.example .env   # then edit .env with your keys
```

## Reproducing Results

```bash
# Run experiments (saves incremental JSON to results/raw/)
python src/experiment.py

# Merge raw results into master CSV
python src/consolidate.py

# Run ANOVA + per-judge stats
python src/analysis.py

# Generate all figures (saved to figures/)
python src/figures.py
```

## Datasets

| Dataset | Source | Size |
|---------|--------|------|
| MT-Bench | HuggingFaceH4/mt_bench_evaluated | 80 questions |
| SummEval | mteb/summeval | 100 summaries |
| Urdu pairs | Self-constructed (translated via Helsinki-NLP) | TBD |

## Judges

- GPT-4o (OpenAI)
- Claude-3-Opus (Anthropic)
- Gemini-1.5-Pro (Google)
- Mistral-Large (Mistral AI)
- LLaMA-3-70B (via Groq)
