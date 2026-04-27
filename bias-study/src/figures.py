"""
figures.py — Week 2
Generate all publication-ready figures and save to figures/ as PDF.
Run: python src/figures.py
"""

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

MASTER_CSV  = Path(__file__).parent.parent / "results" / "master_df.csv"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def fig1_design_diagram():
    """Figure 1 — 2×2 factorial design diagram."""
    raise NotImplementedError


def fig2_bias_heatmap(df: pd.DataFrame):
    """Figure 2 — Bias heatmap across judges × conditions."""
    raise NotImplementedError


def fig3_fliprate_bar(df: pd.DataFrame):
    """Figure 3 — Position-bias flip-rate per judge."""
    raise NotImplementedError


def fig4_summeval_vulnerability(df: pd.DataFrame):
    """Figure 4 — SummEval verbosity vulnerability scatter."""
    raise NotImplementedError


if __name__ == "__main__":
    df = pd.read_csv(MASTER_CSV)
    fig1_design_diagram()
    fig2_bias_heatmap(df)
    fig3_fliprate_bar(df)
    fig4_summeval_vulnerability(df)
    print("All figures saved to", FIGURES_DIR)
