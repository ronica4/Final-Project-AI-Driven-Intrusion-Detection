"""
Step 2D -- XGBoost, the project's primary supervised detector.

Dataset-agnostic by construction (Dataset Dependency Rule): every function
here takes (X, y, config, ...) already produced by a loader. Nothing in this
module knows or asks which dataset it is scoring.

use_smote=False, scale_pos_weight instead (PROJECT_PLAN.md Step 2A point 3):
resampling the training fold to 1:1 AND reweighting the loss for the same
imbalance is double-compensation -- pick one, and the plan's theoretical
defence (Ch 7.2) is "SMOTE for density-based models that have no native
weighting knob (Isolation Forest can't accept it at all); scale_pos_weight
for XGBoost, which has one built in and doesn't need synthetic rows."

scale_pos_weight is computed once per dataset+framing from the FULL y, not
recomputed per CV fold. This is safe: unlike SMOTE or StandardScaler, it does
not touch feature values or fit any transform on data a fold hasn't seen --
it only encodes the dataset's known label ratio, a property of the raw
labels that is already public before any modelling decision is made.

In-domain runs use families="full" (all 11 unified columns) for both
datasets, matching Step 2F's CNN convention so every model in the Ch 8
comparison table is scored on the same input shape per dataset. On Dataset A
the four B_ONLY columns are entirely NaN -> imputed to a constant, so the
effective input dimensionality is 7, not 11; this is reported, not hidden
(see the note attached to every exf2021 result below). Cross-dataset
transfer and the F1-only ablation (Step 2G) are the only places
families="intersection" / "F1_only" get used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless -- this module writes files, never shows a window

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

from evaluation.metrics import evaluate
from preprocessing.pipeline import build_pipeline, get_cv
from schema.unified import B_ONLY_COLUMNS, project


def compute_scale_pos_weight(y: pd.Series) -> float:
    """neg/pos ratio -- XGBoost's own documented definition of scale_pos_weight.
    Raises rather than silently returning inf/nan if a class is entirely absent,
    since that means the CV split (or the dataset itself) is broken upstream."""
    y = pd.Series(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"scale_pos_weight is undefined: n_pos={n_pos}, n_neg={n_neg}."
        )
    return n_neg / n_pos


def build_xgboost(
    config: dict, scale_pos_weight: float, max_depth: int | None = None
) -> xgb.XGBClassifier:
    """XGBClassifier with every hyperparameter read from config.yaml -- none
    hardcoded here (PROJECT_PLAN.md Step 0D / Ch 7.3 rubric item: no library
    defaults). max_depth may be overridden for the Step 2D.4 sensitivity
    sweep; every other value stays at its locked config value."""
    cfg = config["models"]["xgboost"]
    return xgb.XGBClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=max_depth if max_depth is not None else cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        min_child_weight=cfg["min_child_weight"],
        scale_pos_weight=scale_pos_weight,
        random_state=cfg["random_state"],
        eval_metric="logloss",
        n_jobs=-1,
    )


def run_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
    meta: dict,
    families: str = "full",
    max_depth: int | None = None,
) -> dict[str, Any]:
    """One full stratified-CV evaluation of XGBoost on (X, y), scored via the
    shared 2A' harness (evaluation.metrics.evaluate) -- no bespoke scoring."""
    X_proj = project(X, families)
    cv = get_cv(y, config)
    spw = compute_scale_pos_weight(y)
    model = build_pipeline(build_xgboost(config, spw, max_depth=max_depth), use_smote=False)

    result = evaluate(model, X_proj, y, cv, meta, families=families)
    # Extra fields appended alongside, not through meta (evaluate()'s frozen
    # schema only pulls dataset_name/framing out of meta -- see Step 2A').
    result["scale_pos_weight"] = spw
    if families == "full" and set(B_ONLY_COLUMNS) & set(X_proj.columns):
        result["b_only_columns_all_nan"] = bool(X_proj[B_ONLY_COLUMNS].isna().all().all())
    return result


def sensitivity_sweep(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
    meta: dict,
    families: str = "full",
) -> list[dict[str, Any]]:
    """Vary max_depth alone (Step 2D.4); everything else stays locked. Returns
    one summary dict per depth plus the full evaluate() result, so both the
    plot and the raw numbers are available from a single call."""
    depths = config["models"]["xgboost_sensitivity_sweep"]["max_depth"]
    sweep = []
    for depth in depths:
        result = run_xgboost(X, y, config, meta, families=families, max_depth=depth)
        sweep.append(
            {
                "max_depth": depth,
                "f1": result["mean"]["f1"],
                "fpr": result["mean"]["fpr"],
                "result": result,
            }
        )
    return sweep


def plot_sensitivity(sweep: list[dict[str, Any]], dataset_label: str, path: str | Path) -> None:
    """F1 and FPR on twin y-axes against max_depth -- one consistent figure
    style, reused by Step 2E's contamination sweep."""
    depths = [r["max_depth"] for r in sweep]
    f1s = [r["f1"] for r in sweep]
    fprs = [r["fpr"] for r in sweep]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(depths, f1s, marker="o", color="tab:blue", label="F1")
    ax1.set_xlabel("max_depth")
    ax1.set_ylabel("F1", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(depths, fprs, marker="s", color="tab:red", label="FPR")
    ax2.set_ylabel("FPR", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(0, max(0.5, max(fprs) * 1.2))

    ax1.set_title(f"XGBoost max_depth sensitivity -- {dataset_label}")
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
