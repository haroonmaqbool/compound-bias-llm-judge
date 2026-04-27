"""
analysis.py — Week 2
ANOVA + per-judge statistics on master_df.csv.
Produces a summary statistics table and effect size estimates.
"""

import pandas as pd
import scipy.stats as stats
from pathlib import Path

MASTER_CSV = Path(__file__).parent.parent / "results" / "master_df.csv"


def load_master() -> pd.DataFrame:
    return pd.read_csv(MASTER_CSV)


def run_anova(df: pd.DataFrame):
    """
    Two-way ANOVA: score ~ order * length
    Prints F-statistics and p-values.
    """
    # TODO: implement using statsmodels OLS or pingouin
    raise NotImplementedError


def per_judge_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute flip-rate and mean score per judge per condition.
    Returns a summary DataFrame.
    """
    raise NotImplementedError


if __name__ == "__main__":
    df = load_master()
    print(df.describe())
    per_judge_stats(df)
