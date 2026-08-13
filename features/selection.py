"""
Step 2B (EDA half) -- per-feature statistical evidence for Chapter 3.

Dataset-agnostic (Dataset Dependency Rule): every function here takes
(X, y, ...) or plain arrays, never a dataset name. A thin runner script
calls these once per loader; nothing in this module knows which dataset
produced its input.

Chapter 3 rubric point (PROJECT_PLAN.md Step 2B): "Broad, hand-wavy
theoretical justifications without data-driven evidence will not be
accepted." Every one of the 11 unified columns gets a real statistic per
dataset, not an assertion -- including an honest "untestable" verdict for
the four B_ONLY columns on Dataset A, which are entirely NaN by construction
and cannot be tested, rather than silently skipping them.

EFFECT SIZE CHOICE: Cliff's delta, not Cohen's d. Our features are not
normally distributed (that's exactly why Mann-Whitney U was chosen over a
t-test), and Cliff's delta is the non-parametric effect size that pairs with
Mann-Whitney U -- it is in fact derived directly from the same U statistic
(delta = 2U/(n1*n2) - 1), so no extra distributional assumption is smuggled
in through the effect-size measure after avoiding one for the test itself.
With ~100k-250k rows per class, every p-value here will be astronomically
significant regardless of whether the effect is real or trivial; Cliff's
delta is what actually separates a meaningful behavioural difference from a
large-sample artifact, and the report says exactly this rather than reading
significance alone as evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless -- this module writes files, never shows a window

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats


def _save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------

def cliffs_delta_from_u(u_statistic: float, n1: int, n2: int) -> float:
    """Cliff's delta derived directly from the Mann-Whitney U statistic --
    exact, and O(1) given U rather than the O(n1*n2) pairwise-comparison
    definition, which matters once n1/n2 reach ~1e5.

    `u_statistic` must be the U scipy.stats.mannwhitneyu(sample1, sample2)
    returns for sample1 (its documented convention): the count of pairs
    (x in sample1, y in sample2) with x > y, plus 0.5 per tie. Positive
    delta then means sample1 tends to have the larger values.
    """
    return (2.0 * u_statistic) / (n1 * n2) - 1.0


def effect_size_verdict(abs_delta: float) -> str:
    """Romano et al. (2006) magnitude bands for Cliff's delta -- the
    standard reference thresholds, not an invented cutoff."""
    if abs_delta < 0.147:
        return "negligible"
    if abs_delta < 0.33:
        return "small"
    if abs_delta < 0.474:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# 2B.1 -- class distribution
# ---------------------------------------------------------------------------

def class_distribution_counts(y, labels: dict[int, str] | None = None) -> dict[str, int]:
    labels = labels or {0: "benign", 1: "attack"}
    counts = pd.Series(y).value_counts().to_dict()
    return {labels.get(k, str(k)): int(v) for k, v in counts.items()}


def plot_class_distribution(counts: dict[str, int], title: str, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    keys, values = list(counts.keys()), list(counts.values())
    ax.bar(keys, values, color=sns.color_palette("Set2", len(keys)))
    ax.set_ylabel("row count")
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout()
    _save(fig, path)


# ---------------------------------------------------------------------------
# 2B.2 -- per-feature class-conditional distributions
# ---------------------------------------------------------------------------

def plot_feature_distributions(X: pd.DataFrame, y, out_dir: str | Path, dataset_label: str) -> list[Path]:
    """One histogram+box combo figure per feature, benign vs. attack overlay.
    A column that is entirely NaN for BOTH classes (never happens under the
    current schema, but would mean nothing to plot) is skipped rather than
    rendered as two empty axes."""
    out_dir = Path(out_dir)
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)
    written = []

    for col in X.columns:
        benign = X.loc[y == 0, col].dropna()
        attack = X.loc[y == 1, col].dropna()
        if len(benign) == 0 and len(attack) == 0:
            continue

        fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(9, 4))
        if len(benign):
            ax_hist.hist(benign, bins=50, alpha=0.5, label="benign", density=True, color="tab:blue")
        if len(attack):
            ax_hist.hist(attack, bins=50, alpha=0.5, label="attack", density=True, color="tab:red")
        ax_hist.set_title(f"{col} -- distribution")
        ax_hist.legend()

        box_data = [d for d in [benign, attack] if len(d)]
        box_labels = [lbl for lbl, d in [("benign", benign), ("attack", attack)] if len(d)]
        ax_box.boxplot(box_data, tick_labels=box_labels, showfliers=False)
        ax_box.set_title(f"{col} -- box plot (outliers hidden)")

        fig.suptitle(f"{dataset_label}: {col}")
        fig.tight_layout()
        path = out_dir / f"{dataset_label}_{col}_distribution.png"
        _save(fig, path)
        written.append(path)

    return written


# ---------------------------------------------------------------------------
# 2B.3 -- per-feature statistical test table (the graded part)
# ---------------------------------------------------------------------------

def feature_significance_table(
    X: pd.DataFrame, y, columns: list[str] | None = None
) -> list[dict[str, Any]]:
    """Mann-Whitney U + Cliff's delta per column, benign (y=0) vs. attack
    (y=1). NaN is dropped per-column, not per-row -- so a column that is
    entirely NaN on this dataset (every B_ONLY column on Dataset A) is
    reported as honestly untestable rather than silently dropping every row
    from every other column's test too."""
    columns = columns or list(X.columns)
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    rows = []
    for col in columns:
        series = X[col]
        benign = series[y == 0].dropna()
        attack = series[y == 1].dropna()

        if len(benign) == 0 or len(attack) == 0:
            rows.append(
                {
                    "feature": col,
                    "median_benign": None,
                    "median_attack": None,
                    "u_statistic": None,
                    "p_value": None,
                    "cliffs_delta": None,
                    "verdict": "untestable (one class entirely NaN on this column)",
                }
            )
            continue

        u_stat, p_value = stats.mannwhitneyu(attack, benign, alternative="two-sided")
        delta = cliffs_delta_from_u(u_stat, len(attack), len(benign))
        rows.append(
            {
                "feature": col,
                "median_benign": float(benign.median()),
                "median_attack": float(attack.median()),
                "u_statistic": float(u_stat),
                "p_value": float(p_value),
                "cliffs_delta": float(delta),
                "verdict": effect_size_verdict(abs(delta)),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 2B.4 -- correlation heatmap + multicollinearity flags
# ---------------------------------------------------------------------------

def correlation_heatmap(
    X: pd.DataFrame, dataset_label: str, path: str | Path, threshold: float = 0.9
) -> tuple[pd.DataFrame, list[tuple[str, str, float]]]:
    """Pearson correlation matrix + heatmap, plus every |r| > threshold pair
    flagged as a Ch 4 multicollinearity candidate. Columns that are entirely
    NaN produce NaN correlations (pandas' default), which are excluded from
    flagging rather than compared against threshold."""
    corr = X.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"Feature correlation -- {dataset_label}")
    fig.tight_layout()
    _save(fig, path)

    flagged = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) > threshold:
                flagged.append((cols[i], cols[j], float(r)))
    return corr, flagged


# ---------------------------------------------------------------------------
# 2B.5 -- three-way breakdown, Dataset A only: benign vs. heavy vs. light
# ---------------------------------------------------------------------------

def three_way_breakdown(
    X: pd.DataFrame, attack_subclass, columns: list[str] | None = None
) -> list[dict[str, Any]]:
    """Dataset A only. Kruskal-Wallis (Mann-Whitney's >2-group analogue)
    plus the two pairwise Cliff's deltas against benign, so the report can
    show numerically whether light sits closer to benign than heavy does --
    the setup Step 3A's error analysis depends on."""
    columns = columns or list(X.columns)
    attack_subclass = pd.Series(attack_subclass).reset_index(drop=True)
    X = X.reset_index(drop=True)

    rows = []
    for col in columns:
        benign = X.loc[attack_subclass == "benign", col].dropna()
        heavy = X.loc[attack_subclass == "heavy_attack", col].dropna()
        light = X.loc[attack_subclass == "light_attack", col].dropna()

        groups = [g for g in [benign, heavy, light] if len(g) > 0]
        if len(benign) == 0 or len(groups) < 2:
            rows.append({"feature": col, "verdict": "untestable"})
            continue

        h_stat, p_kw = stats.kruskal(*groups)
        entry: dict[str, Any] = {
            "feature": col,
            "median_benign": float(benign.median()) if len(benign) else None,
            "median_heavy": float(heavy.median()) if len(heavy) else None,
            "median_light": float(light.median()) if len(light) else None,
            "kruskal_h": float(h_stat),
            "kruskal_p": float(p_kw),
        }
        if len(heavy) and len(benign):
            u, _ = stats.mannwhitneyu(heavy, benign, alternative="two-sided")
            entry["cliffs_delta_heavy_vs_benign"] = float(cliffs_delta_from_u(u, len(heavy), len(benign)))
        if len(light) and len(benign):
            u, _ = stats.mannwhitneyu(light, benign, alternative="two-sided")
            entry["cliffs_delta_light_vs_benign"] = float(cliffs_delta_from_u(u, len(light), len(benign)))
        rows.append(entry)

    return rows


# ---------------------------------------------------------------------------
# 2B.6 -- near-constant / redundancy audit
# ---------------------------------------------------------------------------

def near_constant_report(X: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-column NaN fraction and a near-constant flag (<=1 distinct
    non-NaN value). Correlated-pair redundancy is reported separately by
    correlation_heatmap()'s flagged-pairs output -- this covers the other
    kind of redundancy (a column carrying no information at all)."""
    report = []
    for col in X.columns:
        series = X[col]
        nan_fraction = float(series.isna().mean())
        finite = series.dropna()
        near_constant = bool(finite.nunique() <= 1) if len(finite) else True
        report.append(
            {
                "feature": col,
                "nan_fraction": nan_fraction,
                "near_constant": near_constant,
                "n_unique_observed": int(finite.nunique()),
            }
        )
    return report
