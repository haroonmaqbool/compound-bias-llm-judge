# src/analysis.py — Week 2
# Full statistical analysis on master_df.csv.
#
# Outputs (all written to results/tables/):
#   anova_overall.csv        — 2-way ANOVA pooled across all judges & datasets
#   anova_per_judge.csv      — 2-way ANOVA repeated for each judge
#   flip_rate.csv            — position-flip rate per judge (main bias metric)
#   interaction_class.csv    — super/sub/additive classification per judge
#   cohen_kappa.csv          — pairwise inter-judge agreement
#
# Run: python src/analysis.py

import os, itertools
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.metrics import cohen_kappa_score

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
MASTER_CSV = ROOT / "results" / "master_df.csv"
TABLES_DIR = ROOT / "results" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ── Load ─────────────────────────────────────────────────────────────────────
def load() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV)
    df = df[df["winner"].isin(["A", "B"])].copy()

    # ── Derived columns ───────────────────────────────────────────────────────
    # chose_A: 1 = judge chose Response A (always presented first in prompt)
    df["chose_A"] = (df["winner"] == "A").astype(int)

    # long_wins: 1 = the longer response won, regardless of which slot it was in
    # Condition layout from experiment.py:
    #   (first,  short): resp_A = short_a, resp_B = long_b  → long wins if B
    #   (first,  long):  resp_A = long_a,  resp_B = short_b → long wins if A
    #   (second, short): resp_A = long_b,  resp_B = short_a → long wins if A
    #   (second, long):  resp_A = short_b, resp_B = long_a  → long wins if B
    cond = {
        ("first",  "short"): "B",   # long is B
        ("first",  "long"):  "A",   # long is A
        ("second", "short"): "A",   # long is A
        ("second", "long"):  "B",   # long is B
    }
    df["long_wins"] = df.apply(
        lambda r: int(r["winner"] == cond[(r["position"], r["length"])]), axis=1
    )
    return df


# ── 2-way factorial ANOVA ─────────────────────────────────────────────────────
def run_anova(df: pd.DataFrame, label: str = "overall") -> pd.DataFrame:
    """
    DV: chose_A  (binary, 0/1)
    IVs: position (first/second) × length (short/long)

    Uses Type-II SS so each main effect is tested controlling for the other.
    Returns a tidy DataFrame with F, p, and η² for all three effects.
    """
    model = ols("chose_A ~ C(position) * C(length)", data=df).fit()
    tbl   = anova_lm(model, typ=2)

    ss_total       = tbl["sum_sq"].sum()
    tbl["eta_sq"]  = tbl["sum_sq"] / ss_total
    tbl["label"]   = label
    tbl.index.name = "effect"
    return tbl.reset_index()[["label", "effect", "F", "PR(>F)", "eta_sq"]] \
               .rename(columns={"PR(>F)": "p_value"})


# ── Position flip rate ────────────────────────────────────────────────────────
def compute_flip_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (qid, judge, length) triplet we have two rows:
      position='first'  and position='second'.

    A 'flip' = judge picks the same letter (A or B) in both conditions,
    meaning they consistently pick whichever response is presented FIRST
    (since the content is swapped between the two conditions).

    flip_rate = fraction of triplets where the verdict was the same letter,
    which equals the fraction driven purely by presentation order.
    """
    pivot = (
        df.pivot_table(
            index=["dataset", "language", "qid", "judge", "length"],
            columns="position",
            values="winner",
            aggfunc="first",
        )
        .dropna(subset=["first", "second"])
        .reset_index()
    )
    # Same letter → position-driven (flip)
    pivot["flip"] = (pivot["first"] == pivot["second"]).astype(int)

    per_judge = (
        pivot.groupby("judge")["flip"]
             .agg(flip_rate="mean", n_pairs="count")
             .reset_index()
    )
    return per_judge


# ── Interaction classification (OLS per judge) ────────────────────────────────
def classify_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fits chose_A ~ C(position) * C(length) per judge.
    Extracts the interaction coefficient and p-value.

    Classification:
      interaction coef > 0, p < 0.05 → super-additive
      interaction coef < 0, p < 0.05 → sub-additive
      p >= 0.05                       → additive (no significant interaction)
    """
    rows = []
    for judge, jdf in df.groupby("judge"):
        model  = ols("chose_A ~ C(position) * C(length)", data=jdf).fit()
        params = model.params
        pvals  = model.pvalues

        # statsmodels names the interaction term based on sorted levels
        interaction_key = [k for k in params.index if ":" in k]
        if not interaction_key:
            continue
        key   = interaction_key[0]
        coef  = params[key]
        pval  = pvals[key]

        if pval < 0.05:
            cls = "super-additive" if coef > 0 else "sub-additive"
        else:
            cls = "additive"

        rows.append({
            "judge":           judge,
            "interaction_coef": round(coef, 4),
            "interaction_p":    round(pval, 4),
            "classification":  cls,
        })
    return pd.DataFrame(rows)


# ── Cohen's Kappa (pairwise inter-judge agreement) ────────────────────────────
def compute_kappa(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each pair of judges, compute Cohen's Kappa on matching
    (qid, position, length, dataset, language) rows.
    """
    judges = sorted(df["judge"].unique())
    # Pivot so each judge is a column
    pivot = df.pivot_table(
        index=["dataset", "language", "qid", "position", "length"],
        columns="judge",
        values="winner",
        aggfunc="first",
    ).reset_index()

    rows = []
    for j1, j2 in itertools.combinations(judges, 2):
        sub = pivot[[j1, j2]].dropna()
        if len(sub) < 10:
            continue
        kappa = cohen_kappa_score(sub[j1], sub[j2])
        rows.append({"judge_a": j1, "judge_b": j2, "kappa": round(kappa, 4)})
    return pd.DataFrame(rows)


# ── Score drift ───────────────────────────────────────────────────────────────
def compute_score_drift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score drift = mean(chose_A | length=long) - mean(chose_A | length=short)
    per judge. Positive = judges favour the longer response more when it's
    presented as the long variant (length bias direction).
    """
    agg = df.groupby(["judge", "length"])["chose_A"].mean().unstack("length")
    agg["score_drift"] = agg["long"] - agg["short"]
    return agg.reset_index()[["judge", "score_drift"]]


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load()
    print(f"Loaded {len(df):,} rows | judges: {sorted(df['judge'].unique())}")
    print(f"Datasets: {df['dataset'].unique()} | Languages: {df['language'].unique()}\n")

    # 1. Overall ANOVA
    anova_overall = run_anova(df, label="overall")
    print("── Overall 2-way ANOVA ──────────────────────────────")
    print(anova_overall.to_string(index=False))

    # 2. Per-judge ANOVA
    per_judge_anova = []
    for judge, jdf in df.groupby("judge"):
        per_judge_anova.append(run_anova(jdf, label=judge))
    anova_per_judge = pd.concat(per_judge_anova, ignore_index=True)
    print("\n── Per-Judge ANOVA ──────────────────────────────────")
    print(anova_per_judge.to_string(index=False))

    # 3. Flip rate
    flip = compute_flip_rate(df)
    print("\n── Position Flip Rate ───────────────────────────────")
    print(flip.to_string(index=False))

    # 4. Interaction classification
    interaction_cls = classify_interactions(df)
    print("\n── Interaction Classification ────────────────────────")
    print(interaction_cls.to_string(index=False))

    # 5. Cohen's Kappa
    kappa = compute_kappa(df)
    print("\n── Pairwise Cohen's Kappa ───────────────────────────")
    print(kappa.to_string(index=False))

    # 6. Score drift
    drift = compute_score_drift(df)
    print("\n── Score Drift (long - short chose_A rate) ──────────")
    print(drift.to_string(index=False))

    # ── Save all tables ───────────────────────────────────────────────────────
    anova_overall.to_csv(TABLES_DIR / "anova_overall.csv", index=False)
    anova_per_judge.to_csv(TABLES_DIR / "anova_per_judge.csv", index=False)
    flip.to_csv(TABLES_DIR / "flip_rate.csv", index=False)
    interaction_cls.to_csv(TABLES_DIR / "interaction_class.csv", index=False)
    kappa.to_csv(TABLES_DIR / "cohen_kappa.csv", index=False)
    drift.to_csv(TABLES_DIR / "score_drift.csv", index=False)

    print(f"\n✓ All tables saved to {TABLES_DIR}/")
