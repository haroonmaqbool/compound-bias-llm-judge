# src/analysis_v2.py — week3-fixes
# Revised statistical analysis addressing peer-review feedback:
#
#  1. Primary model: GEE logistic regression with cluster-robust SEs (by qid)
#     → fixes pseudoreplication / non-independence of the 4-condition within-item design
#  2. Secondary: LPM ANOVA retained for η² variance decomposition only
#  3. Gemini contamination sensitivity: full analysis re-run excluding gemini-flash
#  4. Urdu truncation quantification: count long pairs that hit 512-token ceiling
#  5. LPM β>1 check: flag and report odds ratios (from logistic) for Urdu per-judge
#  6. llama3-70b Urdu invalid-rate sensitivity analysis
#  7. Bonferroni correction (α_adj = 0.01 for 5 per-judge tests)
#
# Outputs (results/tables/):
#   gee_overall.csv          — GEE interaction results (primary model)
#   gee_per_judge.csv        — GEE per-judge (primary)
#   sensitivity_no_gemini.csv — ANOVA without gemini-flash rows
#   urdu_truncation.csv      — count of truncated long Urdu responses
#   anova_overall_v2.csv     — LPM ANOVA (secondary, for η²)
#   anova_per_judge_v2.csv   — LPM per-judge (secondary, for η²)
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


# ── 3. Gemini contamination sensitivity ───────────────────────────────────────
def gemini_sensitivity(df: pd.DataFrame) -> tuple:
    """
    Returns (n_gemini_rows, results_without_gemini).
    We cannot know exactly which gemini rows fell back to llama3-8b,
    so the conservative approach is to exclude ALL gemini-flash rows.
    If the interaction survives without them, the finding is robust.
    """
    n_gemini = (df["judge"] == "gemini-flash").sum()
    df_no_gem = df[df["judge"] != "gemini-flash"].copy()
    return n_gemini, df_no_gem


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


# ── 6. llama3-70b Urdu invalid-rate sensitivity ───────────────────────────────
def llama70b_urdu_sensitivity(all_raw_rows: list) -> dict:
    """
    Count total attempted and invalid rows for llama3-70b on Urdu.
    Uses the raw JSONL (including ERROR/INVALID) to get the true invalid rate.
    """
    urdu_file = ROOT / "results" / "raw" / "urdu_results.jsonl"
    if not urdu_file.exists():
        return {}

    total = 0
    invalid = 0
    with open(urdu_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("judge") == "llama3-70b":
                    total += 1
                    if row.get("winner") not in ("A", "B"):
                        invalid += 1
            except Exception:
                pass

    result = {
        "judge": "llama3-70b",
        "dataset": "mt_bench_ur",
        "total_attempted": total,
        "invalid": invalid,
        "valid": total - invalid,
        "invalid_rate": round(invalid / total, 4) if total > 0 else None,
    }
    print(f"\n── llama3-70b Urdu invalid-rate ─────────────────────────────────")
    for k, v in result.items():
        print(f"   {k}: {v}")
    return result


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

    # ── D. GEMINI SENSITIVITY ─────────────────────────────────────────────────
    print("\n\n── D. Gemini-flash Contamination Sensitivity ────────────────────")
    n_gemini, df_no_gem = gemini_sensitivity(df)
    print(f"  gemini-flash rows excluded: {n_gemini:,} "
          f"({100*n_gemini/len(df):.1f}% of total)")
    print(f"  Remaining rows for sensitivity analysis: {len(df_no_gem):,}")

    gee_no_gem = run_gee(df_no_gem, label="overall_no_gemini")
    if not gee_no_gem.empty:
        print("\n  GEE without gemini-flash:")
        print(gee_no_gem[gee_no_gem["term"].str.contains(":")].to_string(index=False))
        gee_no_gem.to_csv(TABLES_DIR / "sensitivity_no_gemini.csv", index=False)
        print(f"  → Saved sensitivity_no_gemini.csv")

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

    # ── G. llama3-70b URDU INVALID RATE ─────────────────────────────────────
    print("\n\n── G. llama3-70b Urdu Invalid-Rate Sensitivity ──────────────────")
    llama70b_res = llama70b_urdu_sensitivity([])
    if llama70b_res:
        pd.DataFrame([llama70b_res]).to_csv(
            TABLES_DIR / "llama70b_urdu_invalid.csv", index=False)
        print(f"  → Saved llama70b_urdu_invalid.csv")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("  All outputs saved to results/tables/")
    print("  Use gee_*.csv as primary model for paper revision.")
    print("  Use anova_*_v2.csv for η² decomposition only.")
    print("=" * 65)
