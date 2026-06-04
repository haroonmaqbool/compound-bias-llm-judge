# src/analysis_v2.py — week4-hybrid
# Revised statistical analysis:
#
#  1. Primary model: GEE logistic regression with cluster-robust SEs (by qid)
#  2. Secondary: LPM ANOVA retained for η² variance decomposition only
#  3. Cloud-judge sensitivity: re-run excluding qwen3-32b (only Groq/API judge)
#     to verify the result holds with local Ollama judges only
#  4. Urdu truncation quantification: count long pairs that hit 512-token ceiling
#  5. LPM β>1 check: flag and report odds ratios (from logistic) for Urdu per-judge
#  6. Position flip rate and long-win rate per judge
#  7. Pairwise Cohen's κ inter-judge agreement
#  8. Bonferroni correction (α_adj = 0.01 for 5 per-judge tests)
#
# Outputs (results/tables/):
#   gee_overall.csv           — GEE interaction results (primary model)
#   gee_per_judge.csv         — GEE per-judge (primary)
#   sensitivity_no_cloud.csv  — GEE without qwen3-32b (local-only replication)
#   urdu_truncation.csv       — count of truncated long Urdu responses
#   anova_overall_v2.csv      — LPM ANOVA (secondary, for η²)
#   anova_per_judge_v2.csv    — LPM per-judge (secondary, for η²)
#   flip_rate.csv             — position flip rate and long-win rate per judge
#   cohen_kappa.csv           — pairwise Cohen's κ inter-judge agreement
#
# Run: python src/analysis_v2.py

import os, itertools, json
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
from sklearn.metrics import cohen_kappa_score

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
MASTER_CSV = ROOT / "results" / "master_df.csv"
TABLES_DIR = ROOT / "results" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

URDU_PAIRS = ROOT / "data" / "urdu_pairs.jsonl"

# ── Load & prepare data ───────────────────────────────────────────────────────
def load_df() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV)
    df = df[df["winner"].isin(["A", "B"])].copy()
    df["chose_A"]  = (df["winner"] == "A").astype(int)
    # Global unique item ID across all datasets+languages
    df["item_id"]  = df["dataset"].astype(str) + "_" + df["language"].astype(str) + "_" + df["qid"].astype(str)
    df["position"]  = df["position"].astype(str)
    df["length"]    = df["length"].astype(str)
    df["judge"]     = df["judge"].astype(str)
    return df


# ── 1. GEE logistic regression (PRIMARY model) ───────────────────────────────
def run_gee(df: pd.DataFrame, label: str = "overall") -> pd.DataFrame:
    """
    GEE with Binomial family and Exchangeable working correlation structure.
    Groups = item_id  → accounts for the 5 repeated measurements per item
    (one per judge) and the 4 conditions per item-judge pair.
    Cluster-robust (sandwich) SEs are the default in statsmodels GEE.
    """
    try:
        model = GEE.from_formula(
            "chose_A ~ C(position) * C(length)",
            groups="item_id",
            data=df,
            family=Binomial(),
            cov_struct=Exchangeable(),
        )
        res = model.fit(maxiter=60, first_dep_update=0)

        params = res.params
        cis    = res.conf_int()
        pvals  = res.pvalues

        rows = []
        for term in params.index:
            rows.append({
                "label":        label,
                "term":         term,
                "coef":         round(params[term], 4),
                "OR":           round(np.exp(params[term]), 4),
                "OR_lo95":      round(np.exp(cis.loc[term, 0]), 4),
                "OR_hi95":      round(np.exp(cis.loc[term, 1]), 4),
                "p_value":      pvals[term],
                "p_bonf_5":     min(pvals[term] * 5, 1.0),  # Bonferroni for 5 judges
                "sig_bonf":     pvals[term] * 5 < 0.05,
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"  [WARN] GEE failed for {label}: {e}")
        return pd.DataFrame()


# ── 2. LPM ANOVA (SECONDARY — for η² only) ───────────────────────────────────
def run_anova(df: pd.DataFrame, label: str = "overall") -> pd.DataFrame:
    model = ols("chose_A ~ C(position) * C(length)", data=df).fit()
    tbl   = anova_lm(model, typ=2)

    ss_total           = tbl["sum_sq"].sum()
    ss_residual        = tbl.loc["Residual", "sum_sq"]
    tbl["eta_sq"]      = tbl["sum_sq"] / ss_total
    tbl["partial_eta_sq"] = tbl["sum_sq"] / (tbl["sum_sq"] + ss_residual)
    tbl["label"]       = label
    tbl.index.name     = "effect"
    return tbl.reset_index()[["label","effect","F","PR(>F)","eta_sq","partial_eta_sq"]] \
               .rename(columns={"PR(>F)":"p_value"})


# ── 3. Cloud-judge sensitivity ───────────────────────────────────────────────
def cloud_judge_sensitivity(df: pd.DataFrame) -> tuple:
    """
    Returns (n_cloud_rows, df_without_cloud).
    Excludes qwen3-32b (the only Groq/API judge) and re-runs the analysis on
    the 4 local Ollama judges only. If the interaction survives, super-additivity
    is not an artefact of cloud API-specific behaviour or Groq rate-limit quirks.
    """
    n_cloud     = (df["judge"] == "qwen3-32b").sum()
    df_no_cloud = df[df["judge"] != "qwen3-32b"].copy()
    return n_cloud, df_no_cloud


# ── 4. Urdu truncation quantification ────────────────────────────────────────
def quantify_urdu_truncation() -> pd.DataFrame:
    """
    Estimate truncation without re-loading the heavy MarianMT tokenizer.
    Helsinki-NLP Marian uses BPE; on average Urdu BPE produces ~2-3 tokens
    per word.  We use word-count × 2.5 as a conservative estimate.
    512-token limit → safe word budget ≈ 512/2.5 = ~205 words.
    Responses exceeding 205 Urdu words are considered TRUNCATED.
    """
    if not URDU_PAIRS.exists():
        print(f"  [SKIP] {URDU_PAIRS} not found")
        return pd.DataFrame()

    WORD_BUDGET = 205   # conservative estimate for 512-token limit

    rows = []
    with open(URDU_PAIRS, encoding="utf-8") as f:
        for qid, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            pair = json.loads(line)
            for variant in ["long_a", "long_b", "short_a", "short_b"]:
                text = pair.get(variant, "")
                wc   = len(text.split())
                rows.append({
                    "qid":          qid,
                    "variant":      variant,
                    "word_count":   wc,
                    "est_tokens":   round(wc * 2.5),
                    "truncated":    wc > WORD_BUDGET,
                })

    df_trunc = pd.DataFrame(rows)
    long_rows = df_trunc[df_trunc["variant"].isin(["long_a", "long_b"])]
    summary = {
        "total_long_variants":      len(long_rows),
        "truncated_long_variants":  int(long_rows["truncated"].sum()),
        "truncation_pct":           round(100 * long_rows["truncated"].mean(), 1),
        "median_long_word_count":   int(long_rows["word_count"].median()),
        "token_budget":             512,
        "word_budget_used":         WORD_BUDGET,
    }
    print("\n── Urdu truncation quantification ───────────────────────────────")
    for k, v in summary.items():
        print(f"   {k}: {v}")
    return df_trunc, summary


# ── 5. LPM β > 1 check for Urdu ──────────────────────────────────────────────
def urdu_lpm_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag judges where the LPM interaction coefficient is outside [−1, +1].
    Report logistic OR as the recommended alternative.
    """
    df_ur = df[df["language"] == "ur"].copy()
    if len(df_ur) == 0:
        print("  [SKIP] No Urdu rows in master_df.csv")
        return pd.DataFrame()

    rows = []
    for judge, jdf in df_ur.groupby("judge"):
        if len(jdf) < 10:
            rows.append({"judge": judge, "note": "too few rows", "lpm_coef": None, "OR": None})
            continue

        # LPM
        try:
            lpm = ols("chose_A ~ C(position) * C(length)", data=jdf).fit()
            interaction_terms = [t for t in lpm.params.index if ":" in t]
            lpm_coef = lpm.params[interaction_terms[0]] if interaction_terms else None
        except Exception as e:
            lpm_coef = None

        # Logistic (for OR)
        try:
            from statsmodels.formula.api import logit
            lg = logit("chose_A ~ C(position) * C(length)", data=jdf).fit(disp=False)
            interaction_terms_lg = [t for t in lg.params.index if ":" in t]
            log_coef = lg.params[interaction_terms_lg[0]] if interaction_terms_lg else None
            OR       = np.exp(log_coef) if log_coef is not None else None
            p_val    = lg.pvalues[interaction_terms_lg[0]] if interaction_terms_lg else None
        except Exception:
            log_coef, OR, p_val = None, None, None

        rows.append({
            "judge":         judge,
            "n_valid":       len(jdf),
            "lpm_coef":      round(lpm_coef, 4) if lpm_coef is not None else None,
            "lpm_exceeds_1": abs(lpm_coef) > 1 if lpm_coef is not None else False,
            "logit_coef":    round(log_coef, 4) if log_coef is not None else None,
            "OR":            round(OR, 4) if OR is not None else None,
            "p_value":       p_val,
        })

    return pd.DataFrame(rows)


# ── 6. Position flip rate and long-win rate ──────────────────────────────────
def compute_flip_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flip rate = fraction of (qid, length, dataset, language) pairs where the
    judge's verdict differs between position=first and position=second.
    Long-win rate = fraction of decisions where the longer response wins.
    """
    rows = []
    for judge, jdf in df.groupby("judge"):
        idx = ["qid", "length", "dataset", "language"]
        first_s  = jdf[jdf["position"] == "first" ].set_index(idx)["winner"]
        second_s = jdf[jdf["position"] == "second"].set_index(idx)["winner"]
        common   = first_s.index.intersection(second_s.index)
        flip_rate = (first_s.loc[common].values != second_s.loc[common].values).mean() \
                    if len(common) > 0 else float("nan")

        # Long wins: position=first + length=long → A is long → A wins
        #            position=second + length=long → B is long → B wins
        long_rows = jdf[jdf["length"] == "long"]
        if len(long_rows) > 0:
            long_win = (
                ((long_rows["position"] == "first")  & (long_rows["winner"] == "A")) |
                ((long_rows["position"] == "second") & (long_rows["winner"] == "B"))
            ).mean()
        else:
            long_win = float("nan")

        rows.append({
            "judge":         judge,
            "flip_rate":     round(float(flip_rate), 3),
            "long_win_rate": round(float(long_win),  3),
        })
    return pd.DataFrame(rows).sort_values("flip_rate", ascending=False).reset_index(drop=True)


# ── 7. Pairwise Cohen's κ ────────────────────────────────────────────────────
def compute_kappa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise Cohen's κ on the subset of rows where ALL judges gave a valid A/B verdict.
    """
    judges = sorted(df["judge"].unique())
    idx    = ["qid", "position", "length", "dataset", "language"]
    pivot  = df.pivot_table(index=idx, columns="judge", values="winner", aggfunc="first")
    pivot  = pivot.dropna()   # require all judges present

    rows = []
    for j1, j2 in itertools.combinations(judges, 2):
        if j1 in pivot.columns and j2 in pivot.columns:
            k = cohen_kappa_score(pivot[j1], pivot[j2])
            rows.append({"judge_a": j1, "judge_b": j2, "kappa": round(k, 3)})
    return pd.DataFrame(rows).sort_values("kappa", ascending=False).reset_index(drop=True)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  analysis_v2.py — Reviewer-Response Statistical Analysis")
    print("=" * 65)

    print("\nLoading master_df.csv …")
    df = load_df()
    print(f"  Total valid rows: {len(df):,}")
    print(f"  Judges: {sorted(df['judge'].unique())}")
    print(f"  Languages: {sorted(df['language'].unique())}")

    # ── A. GEE PRIMARY MODEL — OVERALL ───────────────────────────────────────
    print("\n\n── A. GEE Logistic Regression (PRIMARY, pooled) ─────────────────")
    gee_overall = run_gee(df, label="overall")
    if not gee_overall.empty:
        print(gee_overall.to_string(index=False))
        gee_overall.to_csv(TABLES_DIR / "gee_overall.csv", index=False)
        print(f"  → Saved gee_overall.csv")

    # ── B. GEE PER-JUDGE ─────────────────────────────────────────────────────
    print("\n\n── B. GEE Per-Judge ─────────────────────────────────────────────")
    gee_per_judge_dfs = []
    BONF_ALPHA = 0.05 / 5
    for judge, jdf in df.groupby("judge"):
        res = run_gee(jdf, label=judge)
        if not res.empty:
            gee_per_judge_dfs.append(res)

    if gee_per_judge_dfs:
        gee_per_judge = pd.concat(gee_per_judge_dfs, ignore_index=True)
        print(gee_per_judge.to_string(index=False))
        gee_per_judge.to_csv(TABLES_DIR / "gee_per_judge.csv", index=False)

        # Interaction terms only
        interact = gee_per_judge[gee_per_judge["term"].str.contains(":")].copy()
        n_sig_bonf = (interact["p_value"] < BONF_ALPHA).sum()
        print(f"\n  [Bonferroni α={BONF_ALPHA:.3f}] "
              f"{n_sig_bonf}/{len(interact)} judges significant at corrected threshold")

    # ── C. LPM ANOVA SECONDARY — η² only ─────────────────────────────────────
    print("\n\n── C. LPM ANOVA (SECONDARY — for η² decomposition) ─────────────")
    anova_overall = run_anova(df, "overall")
    print(anova_overall.to_string(index=False))
    anova_overall.to_csv(TABLES_DIR / "anova_overall_v2.csv", index=False)

    anova_per_judge_dfs = []
    for judge, jdf in df.groupby("judge"):
        anova_per_judge_dfs.append(run_anova(jdf, label=judge))
    anova_pj = pd.concat(anova_per_judge_dfs, ignore_index=True)
    anova_pj.to_csv(TABLES_DIR / "anova_per_judge_v2.csv", index=False)

    # ── D. CLOUD JUDGE SENSITIVITY ────────────────────────────────────────────
    print("\n\n── D. Cloud-Judge Sensitivity (local Ollama judges only) ─────────")
    n_cloud, df_no_cloud = cloud_judge_sensitivity(df)
    print(f"  qwen3-32b rows excluded: {n_cloud:,} "
          f"({100*n_cloud/len(df):.1f}% of total)")
    print(f"  Remaining rows (4 local Ollama judges): {len(df_no_cloud):,}")

    gee_no_cloud = run_gee(df_no_cloud, label="overall_no_cloud")
    if not gee_no_cloud.empty:
        print("\n  GEE without qwen3-32b (local judges only):")
        print(gee_no_cloud[gee_no_cloud["term"].str.contains(":")].to_string(index=False))
        gee_no_cloud.to_csv(TABLES_DIR / "sensitivity_no_cloud.csv", index=False)
        print(f"  → Saved sensitivity_no_cloud.csv")

    # ── E. URDU TRUNCATION QUANTIFICATION ────────────────────────────────────
    print("\n\n── E. Urdu 512-Token Truncation Quantification ──────────────────")
    trunc_result = quantify_urdu_truncation()
    if isinstance(trunc_result, tuple):
        df_trunc, trunc_summary = trunc_result
        df_trunc.to_csv(TABLES_DIR / "urdu_truncation.csv", index=False)
        pd.DataFrame([trunc_summary]).to_csv(
            TABLES_DIR / "urdu_truncation_summary.csv", index=False)
        print(f"  → Saved urdu_truncation.csv")

    # ── F. URDU LPM β > 1 CHECK + ODDS RATIOS ────────────────────────────────
    print("\n\n── F. Urdu Per-Judge LPM β Check + Logistic Odds Ratios ─────────")
    urdu_or = urdu_lpm_check(df)
    if not urdu_or.empty:
        print(urdu_or.to_string(index=False))
        urdu_or.to_csv(TABLES_DIR / "urdu_odds_ratios.csv", index=False)
        print(f"  → Saved urdu_odds_ratios.csv")
        n_exceed = urdu_or["lpm_exceeds_1"].sum() if "lpm_exceeds_1" in urdu_or else 0
        if n_exceed > 0:
            print(f"\n  ⚠ {n_exceed} judge(s) have |LPM β| > 1 — report ORs instead (see urdu_odds_ratios.csv)")

    # ── G. POSITION FLIP RATE AND LONG-WIN RATE ──────────────────────────────
    print("\n\n── G. Position Flip Rate & Long-Win Rate ────────────────────────")
    flip_df = compute_flip_rates(df)
    print(flip_df.to_string(index=False))
    flip_df.to_csv(TABLES_DIR / "flip_rate.csv", index=False)
    print(f"  → Saved flip_rate.csv")

    # ── H. PAIRWISE COHEN'S κ ─────────────────────────────────────────────────
    print("\n\n── H. Pairwise Inter-Judge Agreement (Cohen's κ) ───────────────")
    kappa_df = compute_kappa(df)
    print(kappa_df.to_string(index=False))
    kappa_df.to_csv(TABLES_DIR / "cohen_kappa.csv", index=False)
    print(f"  → Saved cohen_kappa.csv")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("  All outputs saved to results/tables/")
    print("  Primary model:  gee_overall.csv / gee_per_judge.csv")
    print("  Sensitivity:    sensitivity_no_cloud.csv")
    print("  Flip rates:     flip_rate.csv")
    print("  Agreement:      cohen_kappa.csv")
    print("  η² only:        anova_*_v2.csv")
    print("=" * 65)
