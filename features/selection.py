"""
Step 2B (EDA half) + Step 2C (feature-ranking half) -- per-feature
statistical evidence for Chapter 3, and gain-based ranking + the deliberate
leakage demonstration + VIF for Chapter 4.

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

IMPORTANCE CHOICE (Step 2C): gain-based, not the XGBoost default
weight-based importance. Weight counts how many times a feature is split
on, which is biased toward high-cardinality features that offer more
distinct split points -- exactly the kind of artifact this project's own
leakage demo exists to catch, so using it for the "legitimate" ranking
figure would be self-undermining. Gain measures the actual average
loss-reduction a feature's splits contribute, which is what "useful"
should mean here.
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


# ---------------------------------------------------------------------------
# 2C.1/2C.2 -- gain-based feature importance ranking (Figure 4.1, and reused
# for the before/after leakage demo importance charts)
# ---------------------------------------------------------------------------

def gain_importance(fitted_estimator, feature_names: list[str]) -> dict[str, float]:
    """Gain-based importance from a fitted xgboost.XGBClassifier, normalised
    to sum to 1 across `feature_names`. Handles both cases the booster can
    be in: real column names (fit directly on a DataFrame) or generic
    "f0","f1",... keys (fit on a bare ndarray -- what happens inside an
    sklearn Pipeline unless pandas output is explicitly configured, which
    this project's Pipeline does not do). Either way the result is keyed by
    the real feature name, resolved positionally in the f-N case."""
    booster = fitted_estimator.get_booster()
    raw = booster.get_score(importance_type="gain")

    resolved: dict[str, float] = {}
    for key, gain in raw.items():
        if key in feature_names:
            resolved[key] = gain
        elif key.startswith("f") and key[1:].isdigit():
            idx = int(key[1:])
            if idx < len(feature_names):
                resolved[feature_names[idx]] = gain

    total = sum(resolved.values())
    if total <= 0:
        return {name: 0.0 for name in feature_names}
    return {name: resolved.get(name, 0.0) / total for name in feature_names}


def plot_feature_importance(importance: dict[str, float], title: str, path: str | Path) -> None:
    items = sorted(importance.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(items) + 1.5))
    ax.barh([k for k, _ in items], [v for _, v in items], color="tab:blue")
    ax.set_xlabel("gain-based importance (normalized)")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, path)


# ---------------------------------------------------------------------------
# 2C.3/2C.4 -- deliberate leakage demonstration support
# ---------------------------------------------------------------------------

def factorize_leakage_column(X: pd.DataFrame, leakage_col: str) -> pd.DataFrame:
    """Label/ordinal-encode a high-cardinality categorical leakage column
    (e.g. Dataset A's raw-text `_leakage_sld`) into arbitrary integer codes,
    NOT one-hot: `sld` alone spans 11K+ distinct values in benign traffic,
    so one-hot would blow up dimensionality for a column that exists only to
    demonstrate leakage and then get dropped. The codes carry no ordinal
    meaning, which is fine -- what the model exploits in this demonstration
    is a pure identity lookup ("this exact code always means attack"), and
    an arbitrary numeric label supports that exactly as well as any
    semantically meaningful encoding would."""
    X = X.copy()
    codes, _ = pd.factorize(X[leakage_col])
    X[leakage_col] = codes
    return X


# ---------------------------------------------------------------------------
# 2C.5 -- multicollinearity via VIF
# ---------------------------------------------------------------------------

def compute_vif(X: pd.DataFrame, columns: list[str] | None = None) -> list[dict[str, Any]]:
    """Variance Inflation Factor per column: VIF_i = 1 / (1 - R^2_i), where
    R^2_i is from an OLS regression of column i on every other usable
    column (sklearn's LinearRegression -- the formula is direct enough that
    pulling in statsmodels as a new project dependency isn't worth it).
    VIF is undefined for a column with zero variance or that is entirely
    NaN (nothing to regress against or onto); those are reported explicitly
    as `vif: None` rather than raising or silently coercing to a sentinel
    value that could be misread as a real (low) VIF."""
    from sklearn.linear_model import LinearRegression

    columns = columns or list(X.columns)
    usable = [c for c in columns if X[c].notna().any() and X[c].dropna().nunique() > 1]
    sub = X[usable].dropna()

    vifs: dict[str, float] = {}
    if len(usable) >= 2 and len(sub) > len(usable):
        for col in usable:
            others = [c for c in usable if c != col]
            reg = LinearRegression().fit(sub[others], sub[col])
            r2 = reg.score(sub[others], sub[col])
            vifs[col] = float("inf") if r2 >= 1.0 else 1.0 / (1.0 - r2)

    rows = []
    for col in columns:
        if col in vifs:
            rows.append({"feature": col, "vif": vifs[col], "flagged": bool(vifs[col] > 10)})
        else:
            rows.append(
                {
                    "feature": col,
                    "vif": None,
                    "flagged": False,
                    "note": "undefined (constant or all-NaN column)",
                }
            )
    return rows
