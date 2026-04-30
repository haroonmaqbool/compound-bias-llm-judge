<div align="center">

# Do Biases Stack?

### A Compound Bias Study for LLM-as-a-Judge Evaluation

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Groq](https://img.shields.io/badge/Groq-FF5A00?style=for-the-badge&logoColor=white)](https://groq.com/)
[![Gemini](https://img.shields.io/badge/Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![LaTeX](https://img.shields.io/badge/LaTeX-008080?style=for-the-badge&logo=latex&logoColor=white)](https://www.latex-project.org/)

**A research pipeline for measuring how position bias and verbosity bias interact inside automated LLM judges.**

[Overview](#overview) | [Features](#features) | [Quick Start](#quick-start) | [Methodology](#methodology) | [Outputs](#research-outputs)

---

</div>

## Overview

**Do Biases Stack?** investigates a subtle but important question in modern NLP:

> When an LLM judge evaluates two answers, what happens when more than one bias appears at the same time?

LLM-as-a-judge systems are widely used to compare responses, rank models, and automate evaluation. These judges can show **position bias** by preferring an answer based on where it appears, and **verbosity bias** by preferring longer responses. This project studies whether those biases simply add together, weaken each other, or amplify into a stronger compound effect.

The final system produces a structured experimental dataset, bias interaction metrics, cross-model comparisons, visual summaries, and paper-ready findings.

---

## Project Snapshot

| Category | Details |
|----------|---------|
| **Research Area** | LLM evaluation, NLP, automated judging, bias analysis |
| **Main Biases** | Position bias and verbosity bias |
| **Core Design** | 2x2 factorial experiment |
| **Datasets** | MT-Bench, SummEval, Urdu-translated subset |
| **Judges** | LLaMA, Qwen, Gemini, and related instruction-tuned judges |
| **Primary Output** | Bias interaction analysis across models, datasets, and languages |

---

## Features

### Core Research Features

| Feature | Description |
|---------|-------------|
| **Controlled Response Variants** | Generates short and long versions of model answers while preserving meaning. |
| **Position Swapping** | Tests whether judges prefer answers because they appear first or second. |
| **Verbosity Testing** | Measures whether longer answers receive unfair preference. |
| **Compound Bias Measurement** | Identifies whether position and verbosity effects are additive, sub-additive, or super-additive. |
| **Multi-Judge Evaluation** | Compares bias patterns across multiple LLM judges. |
| **Cross-Lingual Analysis** | Extends the study to Urdu to examine low-resource language behavior. |
| **Statistical Reporting** | Uses interaction terms, flip rates, win-rate shifts, and score drift to summarize results. |
| **Paper-Ready Artifacts** | Produces structured outputs for figures, tables, and manuscript writing. |

### Workflow Features

- **Resume-safe experiment logging** with incremental JSONL saves
- **API key rotation** for long Groq-based runs
- **Unified judge interface** for consistent pairwise evaluation
- **Dataset consolidation** into a clean master CSV
- **Reproducible command-line workflow**
- **Research manuscript folder** for LaTeX-based reporting

---

## Quick Demo: How It Feels to Use

1. **Download benchmark data** from MT-Bench and SummEval.
2. **Generate controlled variants** of responses in short and long forms.
3. **Create four comparison conditions** by changing answer order and answer length.
4. **Ask multiple LLM judges** to choose between Response A and Response B.
5. **Save every judgment** with judge name, dataset, language, position, length, and winner.
6. **Consolidate and analyze** results to reveal how biases interact.
7. **Export research outputs** as tables, plots, and paper-ready findings.

> In simple words: we carefully build biased answer pairs, ask AI judges to make decisions, and then measure whether the biases stack.

---

## Tech Stack

### Research & Analysis

| Technology | Purpose |
|------------|---------|
| **Python 3** | Main programming language |
| **Pandas** | Experiment tables and result consolidation |
| **SciPy** | Statistical analysis |
| **Statsmodels** | Factorial modeling and ANOVA-style analysis |
| **Matplotlib** | Static plots and research figures |
| **Seaborn** | Heatmaps and polished statistical visualizations |

### AI & Data

| Technology | Purpose |
|------------|---------|
| **Groq API** | LLM variant generation and judge calls |
| **Google Gemini API** | Gemini-based judge evaluation |
| **Hugging Face Datasets** | Benchmark dataset access |
| **Transformers** | Translation and local model support |
| **MarianMT** | English-to-Urdu translation |
| **PyTorch** | Model backend for translation workflows |
| **SentencePiece** | Tokenization support for translation models |

### Writing & Reporting

| Technology | Purpose |
|------------|---------|
| **LaTeX** | Final research paper |
| **BibTeX** | Academic references |
| **Jupyter Notebook** | Pilot checks and exploration |

---

## Project Structure

```text
Research/
|-- README.md                    # Main polished project overview
`-- bias-study/
    |-- requirements.txt         # Python dependencies
    |-- src/
    |   |-- download_data.py     # Downloads benchmark datasets
    |   |-- variant_generator.py # Creates short/long answer variants
    |   |-- judge.py             # Unified judge wrapper
    |   |-- pilot.py             # Small pilot run
    |   |-- experiment.py        # Full 2x2 factorial experiment
    |   |-- consolidate.py       # Builds master result CSV
    |   |-- analysis.py          # Statistical analysis pipeline
    |   |-- figures.py           # Publication figure generation
    |   `-- urdu_translator.py   # Urdu translation workflow
    |-- notebooks/               # Pilot checks and exploration
    |-- results/                 # Raw and consolidated outputs
    `-- paper/                   # LaTeX manuscript and references
```

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Groq API key**
- **Gemini API key**
- Internet access for dataset and model downloads

### 1. Enter the Project

```bash
cd bias-study
```

### 2. Create and Activate Environment

```bash
python -m venv .venv
```

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file inside `bias-study/`:

```env
GROQ_API_KEYS=your_groq_key_or_comma_separated_keys
GEMINI_API_KEY=your_gemini_key
```

Optional model configuration:

```env
GROQ_VARIANT_MODEL=llama-3.3-70b-versatile
GROQ_JUDGE_70B=llama-3.3-70b-versatile
GROQ_JUDGE_QWEN=qwen/qwen3-32b
GROQ_JUDGE_LLAMA4=meta-llama/llama-4-scout-17b-16e-instruct
GROQ_JUDGE_8B=llama-3.1-8b-instant
GEMINI_MODEL=gemini-2.0-flash-lite
```

### 5. Prepare Data

```bash
python src/download_data.py
python src/variant_generator.py
```

### 6. Run Experiments

```bash
# Pilot experiment
python src/pilot.py

# Full experiment
python src/experiment.py
```

### 7. Generate Results

```bash
python src/consolidate.py
python src/analysis.py
python src/figures.py
```

---

## Methodology

### Biases Studied

| Bias | Meaning |
|------|---------|
| **Position Bias** | The judge prefers an answer because it appears first or second. |
| **Verbosity Bias** | The judge prefers an answer because it is longer or more detailed. |
| **Compound Bias** | The combined effect when position and verbosity appear together. |

### Experimental Design

The study uses a **2x2 factorial setup**:

| Condition | Response Placement | Purpose |
|-----------|--------------------|---------|
| **Short + First** | Short response is shown first | Tests first-position effect on concise answers |
| **Long + First** | Long response is shown first | Tests whether length and first position reinforce each other |
| **Short + Second** | Short response is shown second | Tests second-position treatment of concise answers |
| **Long + Second** | Long response is shown second | Tests verbosity when the longer answer is not first |

This design makes it possible to estimate:

- **Main effect of position**
- **Main effect of verbosity**
- **Interaction effect between position and verbosity**
- **Model-specific vulnerability to compound bias**

---

## How It Works

### Step 1: Dataset Preparation

The pipeline loads benchmark examples from MT-Bench and SummEval, then prepares response pairs for controlled LLM-judge comparisons.

### Step 2: Variant Generation

Each answer is rewritten into:

- A **short variant** that preserves the core meaning with fewer words
- A **long variant** that preserves the same meaning with more explanation

### Step 3: Pair Construction

The system creates four carefully controlled pair layouts so that position and length can be tested separately and together.

### Step 4: LLM Judging

Each judge receives:

- The original question
- Response A
- Response B
- A strict instruction to return only `A` or `B`

### Step 5: Result Consolidation

Raw JSONL files are merged into a master dataset with consistent fields:

| Column | Meaning |
|--------|---------|
| `qid` | Question or pair id |
| `judge` | LLM judge name |
| `position` | Position condition |
| `length` | Length condition |
| `winner` | Selected response |
| `dataset` | Dataset source |
| `language` | Language tag |

### Step 6: Bias Interaction Analysis

The final analysis compares judge behavior across all conditions to identify where biases cancel, accumulate, or amplify.

---

## Evaluation Metrics

| Metric | What It Shows |
|--------|---------------|
| **Flip Rate** | How often a judge changes preference when response order changes |
| **Win-Rate Shift** | How much a response gains or loses from position or length |
| **Score Drift** | Direction and magnitude of bias-related judgment movement |
| **Interaction Term** | Whether two biases combine beyond their separate effects |
| **Agreement Consistency** | How stable judge decisions remain across controlled variants |

---

## Research Outputs

The final project produces:

- **Master result dataset** for all judge decisions
- **Bias interaction matrix** across judges
- **Per-judge vulnerability summaries**
- **Position-bias and verbosity-bias comparisons**
- **Cross-dataset analysis**
- **Cross-lingual Urdu analysis**
- **Publication-ready figures**
- **LaTeX research manuscript**

---

## Expected Insights

This study is designed to answer:

- Which judges are most affected by response order?
- Which judges reward longer responses more strongly?
- Do position and verbosity biases cancel each other out?
- Do they stack into a stronger compound bias?
- Are compound bias effects different in Urdu compared with English?
- What should researchers control for when using LLMs as evaluators?

---

## Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| API key error | Make sure `.env` is inside `bias-study/` and contains valid keys. |
| Groq rate limits | Add multiple comma-separated keys to `GROQ_API_KEYS`. |
| Missing data files | Run `python src/download_data.py`, then `python src/variant_generator.py`. |
| Slow Urdu translation | The translation model downloads the first time it runs. |
| Empty outputs | Run the pipeline from inside the `bias-study/` directory. |

---

## Academic Context

**Course:** Introduction to Natural Language Processing  
**Institution:** Forman Christian College  
**Project Type:** Research-oriented NLP evaluation study  
**Focus:** Reliable, bias-aware AI evaluation

---

<div align="center">

### Built to make AI evaluation more transparent, reliable, and bias-aware.

**Do Biases Stack?** turns a hidden evaluation problem into something measurable.

[Back to Top](#do-biases-stack)

</div>
