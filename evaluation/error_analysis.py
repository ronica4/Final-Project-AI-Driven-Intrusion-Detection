"""
Step 3A -- sample-level forensic error analysis for Chapter 8.1.

Dataset-agnostic (Dataset Dependency Rule): every function takes an
already-computed (y_true, y_pred, y_proba, X, meta) or index sets -- nothing
here knows which dataset or which model produced its input.

Works from OUT-OF-FOLD predictions, not evaluate()'s aggregated metrics.
evaluate()'s frozen JSON schema (Step 2A') deliberately never exposes
per-row predictions -- that is the right contract for a scoring harness, but
forensics needs the actual row-level verdicts, so this module adds
out_of_fold_predictions() alongside evaluate() rather than changing its
schema.

CROSS-MODEL COMPARISON CAVEAT: cross_model_failure_overlap() compares FN/FP
index sets across models. Those indices are only meaningful across models
if every model was scored against literally the same row ordering -- the
same loader call, same config, same sampling. Comparing index sets computed
from two different loader calls (even of the "same" dataset) is meaningless
and this module has no way to detect that mistake, so callers must ensure it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import clone

from evaluation.metrics import _positive_proba


def out_of_fold_predictions(model, X: pd.DataFrame, y: pd.Series, cv) -> tuple[pd.Series, pd.Series]:
    """Same fold structure and leakage discipline as evaluation.metrics.evaluate()
    (clone per fold, fit on the train fold only, score the held-out fold
    only) but returns per-row predictions/scores aligned to X's index
    instead of aggregated metrics -- what sample-level forensics needs."""
    y_pred = pd.Series(index=X.index, dtype="int64")
    y_proba = pd.Series(index=X.index, dtype="float64")

    for train_idx, test_idx in cv.split(X, y):
        estimator = clone(model)
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test = X.iloc[test_idx]

        estimator.fit(X_train, y_train)
        y_pred.iloc[test_idx] = estimator.predict(X_test)
        y_proba.iloc[test_idx] = _positive_proba(estimator, X_test)

    return y_pred, y_proba


def per_subclass_recall(y_true, y_pred, attack_subclass) -> dict[str, dict[str, Any]]:
    """Recall per attack_subclass value, restricted to rows where y_true==1
    -- recall is only meaningful for the positive-class subclasses, so a
    'benign' subclass value (y_true always 0 there) naturally produces no
    entry rather than a meaningless one."""
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    attack_subclass = pd.Series(attack_subclass).reset_index(drop=True)

    result: dict[str, dict[str, Any]] = {}
    for subclass in sorted(attack_subclass.dropna().unique()):
        mask = (attack_subclass == subclass) & (y_true == 1)
        n = int(mask.sum())
        if n == 0:
            continue
        n_detected = int((y_pred[mask] == 1).sum())
        result[subclass] = {"n": n, "n_detected": n_detected, "recall": n_detected / n}
    return result


def top_confident_errors(
    y_true, y_pred, y_proba, X: pd.DataFrame, n: int = 20
) -> dict[str, list[dict[str, Any]]]:
    """Top-n highest-confidence false negatives (lowest y_proba among missed
    attacks -- the model was most sure these were benign) and false
    positives (highest y_proba among flagged benign rows -- the model was
    most sure these were attacks)."""
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    y_proba = pd.Series(y_proba).reset_index(drop=True)
    X = X.reset_index(drop=True)

    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)

    fn_idx = y_proba[fn_mask].sort_values(ascending=True).index[:n]
    fp_idx = y_proba[fp_mask].sort_values(ascending=False).index[:n]

    def _rows(idx) -> list[dict[str, Any]]:
        out = []
        for i in idx:
            row: dict[str, Any] = {"row_index": int(i), "y_proba": float(y_proba.loc[i])}
            for col in X.columns:
                val = X.loc[i, col]
                row[col] = float(val) if pd.notna(val) else None
            out.append(row)
        return out

    return {"false_negatives": _rows(fn_idx), "false_positives": _rows(fp_idx)}


def class_medians(X: pd.DataFrame, y) -> dict[str, dict[str, float | None]]:
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)
    medians: dict[str, dict[str, float | None]] = {}
    for col in X.columns:
        benign = X.loc[y == 0, col].dropna()
        attack = X.loc[y == 1, col].dropna()
        medians[col] = {
            "benign": float(benign.median()) if len(benign) else None,
            "attack": float(attack.median()) if len(attack) else None,
        }
    return medians


def nearest_class_by_features(
    row: dict[str, float], medians: dict[str, dict[str, float | None]]
) -> dict[str, str]:
    """For each feature in `row`, which class median it sits numerically
    closer to -- a cheap, honest way to say 'what does this misclassified
    row have in common with the class it was confused for' without
    inventing a similarity metric. Ties resolve to 'benign'."""
    verdict: dict[str, str] = {}
    for col, value in row.items():
        if col in ("row_index", "y_proba") or value is None:
            continue
        col_medians = medians.get(col, {})
        benign_median, attack_median = col_medians.get("benign"), col_medians.get("attack")
        if benign_median is None or attack_median is None:
            continue
        verdict[col] = "benign" if abs(value - benign_median) <= abs(value - attack_median) else "attack"
    return verdict


def cross_model_failure_overlap(
    errors_by_model: dict[str, dict[str, set[int]]],
) -> dict[str, Any]:
    """Set-intersection analysis across >=2 models' false-negative /
    false-positive index sets. `errors_by_model` is
    {model_name: {"false_negatives": set[int], "false_positives": set[int]}}.
    Non-overlapping failure sets are the empirical justification for the
    cascade (links Ch 8.1 to Ch 8.4) -- if two models miss the same rows,
    stacking them buys nothing; low overlap means each model catches what
    the other misses."""
    names = list(errors_by_model.keys())
    pairwise: list[dict[str, Any]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            model_a, model_b = names[i], names[j]
            for kind in ("false_negatives", "false_positives"):
                set_a = errors_by_model[model_a][kind]
                set_b = errors_by_model[model_b][kind]
                union = set_a | set_b
                overlap = set_a & set_b
                pairwise.append(
                    {
                        "model_a": model_a,
                        "model_b": model_b,
                        "error_type": kind,
                        "n_model_a": len(set_a),
                        "n_model_b": len(set_b),
                        "n_overlap": len(overlap),
                        "overlap_fraction_of_union": (len(overlap) / len(union)) if union else 0.0,
                    }
                )
    return {"models": names, "pairwise": pairwise}


def analyse_errors(
    model_name: str,
    y_true,
    y_pred,
    y_proba,
    X: pd.DataFrame,
    meta: dict,
    n_top: int = 20,
) -> dict[str, Any]:
    """Orchestrates the Step 3A forensics for one model's out-of-fold
    predictions: per-subclass recall, top-n most-confident FN/FP with
    feature values against class medians, and confusion-matrix context.
    Also returns the FULL false_negative/false_positive index sets (not
    just the top-n) so cross_model_failure_overlap() has something to
    intersect against once a second model's predictions exist."""
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    y_proba = pd.Series(y_proba).reset_index(drop=True)
    X = X.reset_index(drop=True)

    medians = class_medians(X, y_true)
    top_errors = top_confident_errors(y_true, y_pred, y_proba, X, n=n_top)
    for kind in ("false_negatives", "false_positives"):
        for row in top_errors[kind]:
            row["nearest_class_per_feature"] = nearest_class_by_features(row, medians)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    result: dict[str, Any] = {
        "model_name": model_name,
        "dataset_name": meta.get("dataset_name", "unknown"),
        "framing": meta.get("framing", "n/a"),
        "n_rows": len(y_true),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "overall_recall": tp / (tp + fn) if (tp + fn) > 0 else None,
        "overall_fpr": fp / (fp + tn) if (fp + tn) > 0 else None,
        "class_medians": medians,
        "top_errors": top_errors,
        "false_negative_indices": sorted(int(i) for i in y_true[(y_true == 1) & (y_pred == 0)].index),
        "false_positive_indices": sorted(int(i) for i in y_true[(y_true == 0) & (y_pred == 1)].index),
    }

    if "attack_subclass" in meta:
        result["per_subclass_recall"] = per_subclass_recall(y_true, y_pred, meta["attack_subclass"])

    return result


def save_error_analysis(
    model_name: str, dataset_name: str, results: dict, out_dir: str | Path = "runs/metrics"
) -> Path:
    """Naming convention locked by the plan: error_analysis_<model>_<dataset>.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"error_analysis_{model_name}_{dataset_name}.json"
    path.write_text(json.dumps(results, indent=2))
    return path
