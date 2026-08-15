"""
Step 2G -- cross-dataset transfer, distribution shift, and the F1-only
ablation (D3) -- Chapter 5's centrepiece experiment.

Dataset-agnostic (Dataset Dependency Rule) at the leaf-function level: every
function takes already-loaded (X, y, meta) pairs. The one thing this module
necessarily knows, unlike every other models/* or evaluation/* module, is
that there are exactly two datasets and it is comparing them -- that is the
entire point of Step 2G, not a violation of the rule.

mode="intersection" (F1-F3, 7 columns) throughout the transfer matrix and the
in-domain cells it is compared against: B_ONLY columns (F4/F5) cannot
participate in transfer by construction (schema/unified.py), and scoring the
in-domain cells on a different column set than the transfer cells would make
the matrix internally incomparable -- the entire point of a 2x2 matrix is
that all four cells are measured the same way.

TRANSFER CELLS vs. IN-DOMAIN CELLS -- two different evaluation procedures,
deliberately:
  - In-domain cells reuse each model's own run_*() (5-fold stratified CV via
    evaluation.metrics.evaluate()), same discipline as every other step.
  - Transfer cells fit ONCE on the full source dataset and score once against
    the full target dataset -- CV has no meaning when train and test are two
    different datasets. The scaler (inside build_pipeline) is fit on the
    source dataset only and applied as-is to the target, which is the honest
    simulation of deploying a model into a new environment (point 1 of the
    Step 2G spec) rather than an oracle that has seen the target's own scale.

Per-model construction mirrors each model's own module exactly (use_smote
flag, hyperparameter source) so a transfer cell is not silently trained
differently from that model's in-domain results elsewhere in the project:
  - xgboost:          use_smote=False, scale_pos_weight from the source y
                       (models/supervised.py convention)
  - isolation_forest:  use_smote=False, unsupervised fit (models/unsupervised.py)
  - cnn:               use_smote=True (models/deep.py convention)
  - autoencoder:        use_smote=False, fits on benign SOURCE rows only
                       (AutoencoderDetector's own fit() already restricts to
                       y_train==0 internally -- see models/deep.py)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from scipy.stats import ks_2samp

from evaluation.metrics import _compute_fold_metrics, _majority_baseline, _positive_proba
from models.deep import build_autoencoder, build_cnn
from models.supervised import build_xgboost, compute_scale_pos_weight
from models.unsupervised import build_isolation_forest
from models.deep import run_autoencoder, run_cnn
from models.supervised import run_xgboost
from models.unsupervised import run_isolation_forest
from preprocessing.pipeline import build_pipeline
from schema.unified import INTERSECTION_COLUMNS, project

# ---------------------------------------------------------------------------
# Per-model spec: how to build an untrained estimator (some need y_train to
# compute a hyperparameter, e.g. XGBoost's scale_pos_weight) and whether that
# model's own convention uses SMOTE inside build_pipeline(). Single source of
# truth so a transfer cell trains a model exactly the way its own module does.
# ---------------------------------------------------------------------------
_BUILDERS: dict[str, Callable[[dict, pd.Series], Any]] = {
    "xgboost": lambda config, y_train: build_xgboost(config, compute_scale_pos_weight(y_train)),
    "isolation_forest": lambda config, y_train: build_isolation_forest(config),
    "cnn": lambda config, y_train: build_cnn(config),
    "autoencoder": lambda config, y_train: build_autoencoder(config),
}
_USE_SMOTE: dict[str, bool] = {
    "xgboost": False,
    "isolation_forest": False,
    "cnn": True,
    "autoencoder": False,
}
_RUN_FNS: dict[str, Callable] = {
    "xgboost": run_xgboost,
    "isolation_forest": run_isolation_forest,
    "cnn": run_cnn,
    "autoencoder": run_autoencoder,
}
VALID_MODELS: tuple[str, ...] = tuple(_BUILDERS)


def transfer_cell(
    model_name: str,
    config: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    families: str = "intersection",
) -> dict[str, Any]:
    """Fit `model_name` once on the full (X_train, y_train), score once
    against the full (X_test, y_test). No CV -- see module docstring for why
    that would be meaningless across two different datasets."""
    if model_name not in VALID_MODELS:
        raise ValueError(f"model_name must be one of {VALID_MODELS}, got {model_name!r}")

    X_train_proj = project(X_train, families)
    X_test_proj = project(X_test, families)

    estimator = _BUILDERS[model_name](config, y_train)
    pipe = build_pipeline(estimator, use_smote=_USE_SMOTE[model_name])
    pipe.fit(X_train_proj, y_train)

    y_pred = pipe.predict(X_test_proj)
    y_proba = _positive_proba(pipe, X_test_proj)
    fold_metrics = _compute_fold_metrics(y_test, y_pred, y_proba)

    return {
        "model_name": model_name,
        "families_mode": families,
        "n_train": len(X_train_proj),
        "n_test": len(X_test_proj),
        "metrics": {k: fold_metrics[k] for k in ("precision", "recall", "f1", "pr_auc", "roc_auc", "fpr")},
        "confusion_matrix": fold_metrics["confusion_matrix"],
        "majority_baseline": _majority_baseline(y_test),
    }


def indomain_cell(
    model_name: str,
    config: dict,
    X: pd.DataFrame,
    y: pd.Series,
    meta: dict,
    families: str = "intersection",
) -> dict[str, Any]:
    """5-fold CV in-domain result via that model's own run_*(), restricted to
    `families` (normally "intersection") so it is directly comparable to the
    transfer cells in the same matrix."""
    if model_name not in VALID_MODELS:
        raise ValueError(f"model_name must be one of {VALID_MODELS}, got {model_name!r}")

    result = _RUN_FNS[model_name](X, y, config, meta, families=families)
    return {
        "model_name": model_name,
        "families_mode": families,
        "n_rows": len(X),
        "metrics": result["mean"],
        "majority_baseline": result["majority_baseline"],
    }


def build_transfer_matrix(
    model_name: str,
    config: dict,
    X_a: pd.DataFrame,
    y_a: pd.Series,
    meta_a: dict,
    X_b: pd.DataFrame,
    y_b: pd.Series,
    meta_b: dict,
    families: str = "intersection",
) -> dict[str, Any]:
    """The full 2x2 matrix from PROJECT_PLAN.md Step 2G point 1 for one model:
    train_A_test_A / train_B_test_B (in-domain, CV) and train_A_test_B /
    train_B_test_A (transfer, single fit/score), all under the same
    `families` mode so every cell is comparable to every other."""
    return {
        "model_name": model_name,
        "families_mode": families,
        "train_A_test_A": indomain_cell(model_name, config, X_a, y_a, meta_a, families),
        "train_B_test_B": indomain_cell(model_name, config, X_b, y_b, meta_b, families),
        "train_A_test_B": transfer_cell(model_name, config, X_a, y_a, X_b, y_b, families),
        "train_B_test_A": transfer_cell(model_name, config, X_b, y_b, X_a, y_a, families),
    }


def ablation_f1_only_vs_intersection(
    model_name: str,
    config: dict,
    X_a: pd.DataFrame,
    y_a: pd.Series,
    X_b: pd.DataFrame,
    y_b: pd.Series,
    intersection_matrix: dict[str, Any],
) -> dict[str, Any]:
    """Step 2G point 2 / D3: rerun ONLY the two transfer cells (in-domain
    performance is not what the hypothesis is about) with families="F1_only"
    and compare against the already-computed families="intersection" transfer
    cells passed in via `intersection_matrix` (from build_transfer_matrix()),
    rather than recomputing them -- avoids retraining slow models (CNN/AE)
    twice for numbers we already have.

    Reports the result whichever way it comes out (module docstring / plan):
    the F1-only-transfers-better hypothesis is what D3 predicted, not what
    this function assumes."""
    f1_only_ab = transfer_cell(model_name, config, X_a, y_a, X_b, y_b, families="F1_only")
    f1_only_ba = transfer_cell(model_name, config, X_b, y_b, X_a, y_a, families="F1_only")

    def _better(f1_only_cell, intersection_cell) -> bool:
        return f1_only_cell["metrics"]["f1"] > intersection_cell["metrics"]["f1"]

    return {
        "model_name": model_name,
        "train_A_test_B": {
            "F1_only": f1_only_ab,
            "intersection": intersection_matrix["train_A_test_B"],
            "f1_only_transfers_better": _better(f1_only_ab, intersection_matrix["train_A_test_B"]),
        },
        "train_B_test_A": {
            "F1_only": f1_only_ba,
            "intersection": intersection_matrix["train_B_test_A"],
            "f1_only_transfers_better": _better(f1_only_ba, intersection_matrix["train_B_test_A"]),
        },
    }


def distribution_shift(
    X_a: pd.DataFrame, X_b: pd.DataFrame, columns: list[str] | None = None
) -> list[dict[str, Any]]:
    """Step 2G point 3: per intersection column, a Kolmogorov-Smirnov
    two-sample statistic (magnitude of the largest CDF gap between A and B,
    0=identical distributions, 1=fully disjoint support) plus
    mean/variance/skew per side. Ranked descending by KS statistic so the
    ranking can be checked against the ablation result (module docstring
    D3 cross-check)."""
    columns = columns if columns is not None else INTERSECTION_COLUMNS
    rows = []
    for col in columns:
        a = X_a[col].dropna()
        b = X_b[col].dropna()
        stat, p = ks_2samp(a, b)
        rows.append(
            {
                "feature": col,
                "ks_statistic": float(stat),
                "p_value": float(p),
                "mean_a": float(a.mean()),
                "mean_b": float(b.mean()),
                "var_a": float(a.var()),
                "var_b": float(b.var()),
                "skew_a": float(a.skew()),
                "skew_b": float(b.skew()),
            }
        )
    rows.sort(key=lambda r: r["ks_statistic"], reverse=True)
    return rows


def plot_distribution_shift(
    X_a: pd.DataFrame, X_b: pd.DataFrame, columns: list[str] | None = None, path: str | Path = "runs/figures/distribution_shift.png"
) -> None:
    """One grid figure, overlaid density per intersection column, A vs. B --
    a single file rather than one-per-feature so Ch 5.2 has one figure
    reference, not seven."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    columns = columns if columns is not None else INTERSECTION_COLUMNS
    n = len(columns)
    ncols = 3
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for ax, col in zip(axes, columns):
        sns.kdeplot(X_a[col].dropna(), ax=ax, label="Dataset A", fill=True, alpha=0.3, warn_singular=False)
        sns.kdeplot(X_b[col].dropna(), ax=ax, label="Dataset B", fill=True, alpha=0.3, warn_singular=False)
        ax.set_title(col)
        ax.legend(fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Distribution shift, intersection columns -- Dataset A vs. Dataset B")
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_cross_dataset_result(name: str, results: dict, out_dir: str | Path = "runs/metrics") -> Path:
    """Same one-JSON-per-experiment convention as evaluation.metrics.save_metrics,
    kept as its own function (rather than importing that one directly) so this
    module's file-naming convention (cross_dataset_<...>) is documented here
    where it is used."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cross_dataset_{name}.json"
    path.write_text(json.dumps(results, indent=2))
    return path
