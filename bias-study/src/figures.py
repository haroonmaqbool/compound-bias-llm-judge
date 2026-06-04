# src/figures.py — week3-fixes
# Generates all 4 publication-ready figures as PDF (vector, 300 DPI).
#
# Figure 1: 2×2 factorial design diagram
# Figure 2: Bias heatmap — chose_A rate per judge × condition
# Figure 3: Position flip-rate bar chart per judge
# Figure 4: Cross-dataset bias comparison (MT-Bench EN / SummEval / Urdu)
#
# Run: python src/figures.py
# Outputs: figures/fig1_*.pdf … fig4_*.pdf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
MASTER_CSV = ROOT / "results" / "master_df.csv"
TABLES_DIR = ROOT / "results" / "tables"
FIG_DIR    = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Academic colour palette (colour-blind friendly) ───────────────────────────
# week4-final: 4 Ollama local + 1 Groq cloud
PALETTE = {
    "qwen2.5":   "#2E86AB",   # steel blue  — Ollama 14B
    "llama3-8b": "#E84855",   # coral red   — Ollama 8B
    "gemma2-9b": "#3BB273",   # emerald     — Ollama 9B
    "mistral":   "#F18F01",   # amber       — Ollama 7B Mistral
    "qwen3-32b": "#7B2D8B",   # purple      — Groq 32B
}
JUDGE_ORDER = list(PALETTE.keys())

COND_LABELS = {
    ("first",  "short"): "Pos=1\nLen=Short",
    ("first",  "long"):  "Pos=1\nLen=Long",
    ("second", "short"): "Pos=2\nLen=Short",
    ("second", "long"):  "Pos=2\nLen=Long",
}
COND_ORDER = [("first","short"), ("first","long"),
              ("second","short"), ("second","long")]

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_df() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV)
    df = df[df["winner"].isin(["A", "B"])].copy()
    df["chose_A"] = (df["winner"] == "A").astype(int)
    cond_map = {
        ("first",  "short"): "B",
        ("first",  "long"):  "A",
        ("second", "short"): "A",
        ("second", "long"):  "B",
    }
    df["long_wins"] = df.apply(
        lambda r: int(r["winner"] == cond_map[(r["position"], r["length"])]), axis=1
    )
    return df


def savefig(fig, name: str, caption: str):
    path = FIG_DIR / name
    fig.savefig(path, format="pdf")
    print(f"  ✓ {name}")
    print(f"    Caption: {caption}\n")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — 2×2 Factorial Design Diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig1_design_diagram():
    # Zone map (coords 0-16, figure 9 inches wide = 0.5625 in/unit):
    #   x=0.0-0.5  : "POSITION BIAS FACTOR" rotated label
    #   x=0.8      : vertical arrow
    #   x=1.5-4.2  : row header text (ha="left", ~2.7 units wide)
    #   x=5.8-10.2 : left box column  (centre x=8)
    #   x=10.8-15.2: right box column (centre x=13)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(9, 8.55, "2x2 Factorial Design", ha="center", va="center",
            fontsize=13, fontweight="bold")

    for lbl, x in [("Short Response First", 8.0), ("Long Response First", 13.0)]:
        ax.text(x, 7.75, lbl, ha="center", va="center", fontsize=10,
                fontweight="bold", color="#2E86AB")

    # Row labels: ha="left" at x=1.5, extends right to ~x=4.5 — arrow is at x=0.8, no overlap
    for lbl, y in [("Position = First", 5.5), ("Position = Second", 3.0)]:
        ax.text(1.5, y, lbl, ha="left", va="center", fontsize=9.5,
                fontweight="bold", color="#E84855")

    # ── 4 condition boxes ─────────────────────────────────────────────────────
    box_data = [
        (8.0,  5.5, "short_a  |  long_b",  "Shorter is FIRST\nLonger is SECOND",  "#EBF5FB"),
        (13.0, 5.5, "long_a   |  short_b", "Longer is FIRST\nShorter is SECOND",  "#FEF9E7"),
        (8.0,  3.0, "long_b   |  short_a", "Longer is FIRST\nShorter is SECOND",  "#FEF9E7"),
        (13.0, 3.0, "short_b  |  long_a",  "Shorter is FIRST\nLonger is SECOND",  "#EBF5FB"),
    ]
    for x, y, content, note, colour in box_data:
        rect = mpatches.FancyBboxPatch(
            (x - 2.2, y - 1.05), 4.4, 2.1,
            boxstyle="round,pad=0.12", linewidth=1.2,
            edgecolor="#555", facecolor=colour,
        )
        ax.add_patch(rect)
        ax.text(x, y + 0.44, content, ha="center", va="center",
                fontsize=8.5, fontfamily="monospace", fontweight="bold")
        ax.text(x, y - 0.32, note, ha="center", va="center",
                fontsize=7.5, color="#444", style="italic")

    # LENGTH arrow — below boxes
    ax.annotate("", xy=(15.5, 1.4), xytext=(1.5, 1.4),
                arrowprops=dict(arrowstyle="->", color="#2E86AB", lw=1.8))
    ax.text(9.0, 1.0, "LENGTH BIAS FACTOR", ha="center", fontsize=9,
            color="#2E86AB", fontweight="bold")

    # POSITION arrow at x=0.8; rotated label at x=0.25 — both left of x=1.5 row labels
    ax.annotate("", xy=(0.8, 1.8), xytext=(0.8, 7.4),
                arrowprops=dict(arrowstyle="->", color="#E84855", lw=1.8))
    ax.text(0.25, 4.6, "POSITION\nBIAS\nFACTOR", ha="center", va="center",
            fontsize=8.5, color="#E84855", fontweight="bold", rotation=90)

    caption = (
        "Figure 1: The 2×2 factorial design crossing position (which response appears first) "
        "and length (which response is longer). Each cell shows the experimental condition "
        "applied to each judge during evaluation."
    )
    savefig(fig, "fig1_design_diagram.pdf", caption)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Bias Heatmap: chose_A rate per judge × condition
# ─────────────────────────────────────────────────────────────────────────────
def fig2_bias_heatmap(df: pd.DataFrame):
    # Build matrix: judges × 4 conditions
    cond_keys  = COND_ORDER
    cond_lbls  = [COND_LABELS[c] for c in cond_keys]
    judges     = [j for j in JUDGE_ORDER if j in df["judge"].unique()]

    matrix = np.zeros((len(judges), len(cond_keys)))
    for i, judge in enumerate(judges):
        jdf = df[df["judge"] == judge]
        for j, (pos, lng) in enumerate(cond_keys):
            sub = jdf[(jdf["position"] == pos) & (jdf["length"] == lng)]
            matrix[i, j] = sub["chose_A"].mean() if len(sub) else np.nan

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    im = ax.imshow(matrix, cmap="RdBu", vmin=0.3, vmax=0.7, aspect="auto")

    ax.set_xticks(range(len(cond_lbls)))
    ax.set_xticklabels(cond_lbls, fontsize=9)
    ax.set_yticks(range(len(judges)))
    ax.set_yticklabels(judges, fontsize=9)

    # Annotate cells
    for i in range(len(judges)):
        for j in range(len(cond_keys)):
            val = matrix[i, j]
            if not np.isnan(val):
                colour = "white" if abs(val - 0.5) > 0.12 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8.5, color=colour, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("P(choose Response A)", fontsize=9)
    cbar.ax.axhline(y=0.5, color="black", lw=1.2)

    ax.set_title("Response-A Selection Rate per Judge × Condition", pad=10)
    ax.set_xlabel("Experimental Condition")
    ax.set_ylabel("Judge Model")

    # Dashed lines separating position groups
    ax.axvline(x=1.5, color="black", lw=1.0, linestyle="--", alpha=0.5)

    fig.tight_layout()
    caption = (
        "Figure 2: Heatmap of P(choose Response A) across all 5 judges and 4 experimental "
        "conditions. Values above 0.5 (red) indicate A-preference; below 0.5 (blue) indicate "
        "B-preference. Diagonal asymmetry reveals compound bias effects."
    )
    savefig(fig, "fig2_bias_heatmap.pdf", caption)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Position Flip-Rate Bar Chart per Judge
# ─────────────────────────────────────────────────────────────────────────────
def fig3_fliprate_bar(df: pd.DataFrame):
    required = {"dataset", "language", "qid", "judge", "position", "winner", "long_wins"}
    missing = sorted(required - set(df.columns))
    if missing:
        print(f"  ! Figure 3 skipped: missing required columns: {missing}")
        print(f"    Available columns: {list(df.columns)}")
        return

    valid = df[df["winner"].isin(["A", "B"])].copy()
    if valid.empty:
        print("  ! Figure 3 skipped: no rows with winner in ['A', 'B'].")
        print(f"    Input rows: {len(df):,}")
        print(f"    Winner counts:\n{df['winner'].value_counts(dropna=False).to_string()}")
        return

    # Compute flip rate per judge. Do not include length in this pivot index:
    # the position-swapped first/second rows can carry different length labels,
    # so including length can keep the rows from matching.
    flip_index = ["dataset", "language", "qid", "judge"]
    duplicate_n = valid.duplicated(flip_index + ["position"]).sum()
    raw_pivot = valid.pivot_table(
        index=flip_index,
        columns="position",
        values="winner",
        aggfunc="first",
    )
    if raw_pivot.empty:
        print("  ! Figure 3 skipped: flip pivot is empty before matching positions.")
        print(f"    Valid rows: {len(valid):,}")
        print(f"    Position counts:\n{valid['position'].value_counts(dropna=False).to_string()}")
        return
    if not {"first", "second"}.issubset(raw_pivot.columns):
        print("  ! Figure 3 skipped: flip pivot does not contain both positions.")
        print(f"    Pivot columns: {list(raw_pivot.columns)}")
        print(f"    Position counts:\n{valid['position'].value_counts(dropna=False).to_string()}")
        return

    pivot = raw_pivot.dropna(subset=["first", "second"]).reset_index()
    if pivot.empty:
        print("  ! Figure 3 skipped: no matched first/second rows after pivot.")
        print(f"    Valid rows: {len(valid):,}")
        print(f"    Raw pivot rows: {len(raw_pivot):,}")
        print(f"    Duplicate index+position rows collapsed by aggfunc='first': {duplicate_n:,}")
        print(f"    Position counts:\n{valid['position'].value_counts(dropna=False).to_string()}")
        print(f"    Rows per judge:\n{valid['judge'].value_counts(dropna=False).to_string()}")
        return

    # Assumption: winner stores the winning displayed slot ("A" or "B"), not
    # the underlying response/model identity. With swapped presentation order,
    # equal slots across first/second mean the judge followed position, so this
    # counts as a position flip. If winner stores response identity, use !=.
    pivot["flip"] = (pivot["first"] == pivot["second"]).astype(int)
    flip_df = pivot.groupby("judge", as_index=False)["flip"].mean()
    flip_df.columns = ["judge", "flip_rate"]
    if flip_df.empty:
        print("  ! Figure 3 skipped: per-judge flip dataframe is empty.")
        print(f"    Matched pivot rows: {len(pivot):,}")
        print(f"    Pivot judges: {sorted(pivot['judge'].dropna().unique())}")
        return

    # Also compute long-response win rate per judge (length bias).
    long_df = valid.groupby("judge", as_index=False)["long_wins"].mean()
    long_df.columns = ["judge", "long_bias"]
    if long_df.empty:
        print("  ! Figure 3 skipped: per-judge long-win dataframe is empty.")
        print(f"    Valid rows: {len(valid):,}")
        print(f"    Rows per judge:\n{valid['judge'].value_counts(dropna=False).to_string()}")
        return

    merged = flip_df.merge(long_df, on="judge", how="inner")
    if merged.empty:
        print("  ! Figure 3 skipped: merged flip/long-win dataframe is empty.")
        print(f"    Flip judges: {sorted(flip_df['judge'].dropna().unique())}")
        print(f"    Long-win judges: {sorted(long_df['judge'].dropna().unique())}")
        print(f"    Matched pivot rows: {len(pivot):,}")
        print(f"    Duplicate index+position rows collapsed by aggfunc='first': {duplicate_n:,}")
        return

    ordered = [j for j in JUDGE_ORDER if j in set(merged["judge"])]
    extras = [j for j in merged["judge"] if j not in JUDGE_ORDER]
    merged = (
        merged.set_index("judge")
              .loc[ordered + extras]
              .reset_index()
    )
    merged = merged.dropna(subset=["flip_rate", "long_bias"])
    if merged.empty:
        print("  ! Figure 3 skipped: final plotting dataframe is empty after ordering/dropna.")
        print(f"    Flip dataframe:\n{flip_df.to_string(index=False)}")
        print(f"    Long-win dataframe:\n{long_df.to_string(index=False)}")
        return

    x    = np.arange(len(merged))
    w    = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))

    bars1 = ax.bar(x - w/2, merged["flip_rate"], w,
                   color=[PALETTE.get(j, "#666666") for j in merged["judge"]],
                   label="Position Flip Rate", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + w/2, merged["long_bias"], w,
                   color=[PALETTE.get(j, "#666666") for j in merged["judge"]],
                   label="Long-Response Win Rate", alpha=0.45, edgecolor="white",
                   hatch="//")

    # Reference line at 0.5 (chance)
    ax.axhline(0.5, color="#333", lw=1.0, linestyle="--", alpha=0.7, label="Chance (0.5)")

    ax.set_xticks(x)
    ax.set_xticklabels(merged["judge"], fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Position Flip Rate & Length Bias per Judge Model")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Value labels only when bars exist and have finite heights.
    if len(merged) > 0:
        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=7.5, color="#333")

    fig.tight_layout()
    caption = (
        "Figure 3: Per-judge position flip rate (solid bars) and long-response win rate "
        "(hatched bars). A flip rate >0.5 indicates systematic position bias; "
        "a long-win rate >0.5 indicates systematic verbosity bias."
    )
    savefig(fig, "fig3_fliprate_bar.pdf", caption)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Cross-Dataset Bias Comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig4_cross_dataset(df: pd.DataFrame):
    """
    Shows position flip rate and long-wins rate broken down by
    dataset × language combination, revealing whether bias is
    domain-specific (SummEval) or language-specific (Urdu).
    """
    df["subset"] = df["dataset"] + "_" + df["language"]
    subset_order = ["mt_bench_en", "summeval_en", "mt_bench_ur"]
    subset_labels = ["MT-Bench\n(English)", "SummEval\n(English)", "MT-Bench\n(Urdu)"]
    colours = ["#2E86AB", "#3BB273", "#E84855"]

    # Flip rate per subset
    pivot = (
        df.pivot_table(
            index=["subset", "qid", "judge", "length"],
            columns="position",
            values="winner",
            aggfunc="first",
        )
        .dropna(subset=["first", "second"])
        .reset_index()
    )
    pivot["flip"] = (pivot["first"] == pivot["second"]).astype(int)
    flip_by_subset = pivot.groupby("subset")["flip"].mean()
    long_by_subset = df.groupby("subset")["long_wins"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=False)

    for ax, (series, title, ylabel) in zip(axes, [
        (flip_by_subset, "Position Flip Rate by Dataset", "Flip Rate"),
        (long_by_subset, "Long-Response Win Rate by Dataset", "Win Rate"),
    ]):
        vals = [series.get(s, np.nan) for s in subset_order]
        bars = ax.bar(subset_labels, vals, color=colours, alpha=0.85,
                      edgecolor="white", width=0.5)
        ax.axhline(0.5, color="#333", lw=1.0, linestyle="--", alpha=0.7, label="Chance")
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Bias Magnitude Across Datasets and Languages", fontsize=11, y=1.01)
    fig.tight_layout()
    caption = (
        "Figure 4: Comparison of position flip rate (left) and verbosity bias (right) "
        "across three evaluation subsets: MT-Bench (English), SummEval (English), and "
        "MT-Bench translated to Urdu. Higher values indicate stronger bias."
    )
    savefig(fig, "fig4_cross_dataset.pdf", caption)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading master_df.csv …")
    df = load_df()
    print(f"  {len(df):,} rows loaded\n")

    print("Generating figures …")
    fig1_design_diagram()
    fig2_bias_heatmap(df)
    fig3_fliprate_bar(df)
    fig4_cross_dataset(df)
    print(f"All figures saved to {FIG_DIR}/")
