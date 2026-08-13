"""
Shared evaluation harness -- the ONE scoring function every model, from both
teammates, is measured by. This exists so Chapter 8's comparison tables are
built from directly comparable numbers, not hand-reconciled at the last
minute from five different ad-hoc scoring scripts.

Dataset-agnostic by construction (Dataset Dependency Rule): this module only
ever sees (model, X, y, cv, meta). It has no idea which dataset produced its
input, and must never be changed to find out.

FROZEN JSON SCHEMA -- the return value of evaluate() / the file written by
save_metrics(). Both teammates code against this shape from Step 2A' onward;
changing a field name after models are trained means re-running experiments,
so treat this as an interface contract, not an implementation detail.

{
  "dataset_name":   str,             # from meta["dataset_name"] -- self-describing,
  "framing":        str,             # from meta["framing"]      -- so a metrics file
  "families_mode":  str,             # "full" | "intersection" | "F1_only" -- can't be
                                      #   misattributed when results get merged in Ch 8
  "n_folds":        int,
  "majority_baseline": {
      "majority_class":     0 | 1,   # whichever class is >=50% of y
      "f1":                 float,   # F1 of predicting majority_class for everyone
      "always_positive_f1": float,   # both given regardless of which is "majority",
      "always_negative_f1": float,   #   so the report can show the full trivial-baseline picture
  },
  "per_fold": [
      {
        "fold": int,
        "precision": float, "recall": float, "f1": float,
        "pr_auc": float, "roc_auc": float, "fpr": float,
        "confusion_matrix": [[tn, fp], [fn, tp]],
      }, ...
  ],
  "mean": {"precision": ..., "recall": ..., "f1": ..., "pr_auc": ..., "roc_auc": ..., "fpr": ...},
  "std":  {"precision": ..., "recall": ..., "f1": ..., "pr_auc": ..., "roc_auc": ..., "fpr": ...},
  "aggregate_confusion_matrix": [[TN, FP], [FN, TP]]   # summed across folds (out-of-fold)
}

The majority-class baseline is included in EVERY result, always, not just
where it looks interesting. It is cheap to compute and it is the single
thing that makes a headline number interpretable instead of impressive-
sounding -- an "always predict malicious" baseline of F1=0.96 next to a
model's F1=0.97 says something very different than the same 0.97 sitting
alone. This is exactly the check that would have caught the 93%-positive
DoHBrw hard-framing bug (D8) before it ever reached a report table.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless -- this module writes files, never shows a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

_METRIC_KEYS = ["precision", "recall", "f1", "pr_auc", "roc_auc", "fpr"]


def _positive_proba(fitted_estimator, X) -> np.ndarray:
    """Best-effort ranking score for the positive class. predict_proba is
    preferred; decision_function is an acceptable fallback (PR/ROC-AUC only
    need correct rank order, not a [0,1]-bounded score). Falling all the way
    back to hard predictions is a last resort that degrades AUC quality and
    is always flagged, never silent."""
    if hasattr(fitted_estimator, "predict_proba"):
        return fitted_estimator.predict_proba(X)[:, 1]
    if hasattr(fitted_estimator, "decision_function"):
        return fitted_estimator.decision_function(X)
    warnings.warn(
        f"{type(fitted_estimator).__name__} has neither predict_proba nor "
        f"decision_function -- falling back to hard predict() for PR/ROC-AUC, "
        f"which degrades AUC quality. Prefer an estimator that scores.",
        stacklevel=2,
    )
    return fitted_estimator.predict(X).astype(float)


def _compute_fold_metrics(y_true, y_pred, y_proba) -> dict[str, Any]:
    """Pure function, independently unit-tested against hand-derived values
    (tests/test_metrics.py) -- the thing every other part of this module
    trusts to be correct."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "fpr": fpr,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _majority_baseline(y) -> dict[str, Any]:
    """The trivial baseline every real result must be read alongside."""
    y = np.asarray(y)
    positive_rate = float(y.mean())
    majority_class = 1 if positive_rate >= 0.5 else 0

    always_positive_f1 = float(
        f1_score(y, np.ones_like(y), pos_label=1, zero_division=0)
    )
    always_negative_f1 = float(
        f1_score(y, np.zeros_like(y), pos_label=1, zero_division=0)
    )

    return {
        "majority_class": majority_class,
        "f1": always_positive_f1 if majority_class == 1 else always_negative_f1,
        "always_positive_f1": always_positive_f1,
        "always_negative_f1": always_negative_f1,
    }


def evaluate(model, X: pd.DataFrame, y: pd.Series, cv, meta: dict, families: str = "full") -> dict[str, Any]:
    """Fit-and-score `model` (typically a preprocessing.pipeline.build_pipeline()
    output) under stratified CV, fold by fold. `model` is CLONED per fold via
    sklearn.base.clone -- the same discipline as the Pipeline itself: nothing
    fit on one fold's data is ever reused or leaked into another fold.
    """
    per_fold = []
    agg_cm = np.zeros((2, 2), dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        estimator = clone(model)
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        y_proba = _positive_proba(estimator, X_test)

        fold_metrics = _compute_fold_metrics(y_test, y_pred, y_proba)
        fold_metrics["fold"] = fold_idx
        per_fold.append(fold_metrics)
        agg_cm += np.array(fold_metrics["confusion_matrix"])

    mean = {k: float(np.mean([f[k] for f in per_fold])) for k in _METRIC_KEYS}
    std = {k: float(np.std([f[k] for f in per_fold])) for k in _METRIC_KEYS}

    return {
        "dataset_name": meta.get("dataset_name", "unknown"),
        "framing": meta.get("framing", "n/a"),
        "families_mode": families,
        "n_folds": cv.n_splits,
        "majority_baseline": _majority_baseline(y),
        "per_fold": per_fold,
        "mean": mean,
        "std": std,
        "aggregate_confusion_matrix": agg_cm.tolist(),
    }


def save_metrics(name: str, results: dict, out_dir: str | Path = "runs/metrics") -> Path:
    """One JSON per experiment. `name` should be descriptive and collision-free,
    e.g. 'xgboost_exf2021_full' or 'isoforest_dohbrw2020_hard' -- the plan's
    convention throughout Phase 2/3."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(results, indent=2))
    return path


def plot_confusion(cm, title: str, path: str | Path) -> None:
    """Used by every model, both datasets -- one consistent visual style for
    all of Chapter 8's confusion matrices."""
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Benign", "Malicious"],
        yticklabels=["Benign", "Malicious"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
