"""
Step 3C -- the hybrid cascade -> Chapter 8.4.

Chain the three models so each does what it is cheapest/best at: Isolation
Forest (Step 2E) reads everything and discards the obviously-normal
majority; XGBoost (Step 2D) scores the survivors and resolves anything it is
confident about; only the genuinely ambiguous handful escalates further.

Dataset-agnostic (Dataset Dependency Rule): every function takes an
already-loaded (X, y, meta), same as every other models/*, evaluation/*
module. `families` defaults to "full" (each stage's own in-domain
convention, matching Step 2D/2E/2F), not "intersection" -- the cascade is a
single-dataset deployment pipeline, not a cross-dataset comparison (that is
Step 2G).

**3D (LLM arbiter) IS DEFERRED BY DEFAULT (D7) and NOT RUN HERE.** Stage 3 in
the plan's spec is "escalate to the LLM." With no arbiter available, escalated
rows fall back to XGBoost's own verdict -- this is not an ad-hoc shortcut,
it is exactly Step 3D's own documented graceful-degradation rule ("if the
API fails, fall back to the XGBoost verdict and log the failure"), applied
because the API was never called in the first place rather than because it
failed mid-call. The consequence, stated plainly rather than buried: with
this fallback, the cascade's PREDICTIVE metrics reduce algebraically to
"Isolation-Forest-filter then XGBoost" -- identical to skipping stage 3
outright. The number that demonstrates the cascade is doing real triage work
is therefore the **escalation count** in the funnel table (how many rows
WOULD have gone to the LLM), not a metrics delta that stage 3 cannot produce
without an arbiter. If 3D is later un-deferred, escalated-row predictions
here are exactly the set the arbiter should replace.

Latency and the funnel are measured on a single held-out stratified test
split (fit once on train, time inference once on test) rather than through
5-fold CV aggregation used for headline metrics elsewhere in the project:
per-stage wall-clock latency is a property of one fitted model serving one
batch of rows, and CV's repeated refit-per-fold has no equivalent meaning
here. The end-to-end cascade metrics and each individual model's own metrics
are computed on that SAME test split and the SAME fitted models, so the
"cascade vs. individual model" comparison in Chapter 8.4 is apples-to-apples.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evaluation.metrics import _compute_fold_metrics, _majority_baseline
from models.deep import build_autoencoder
from models.supervised import build_xgboost, compute_scale_pos_weight
from models.unsupervised import build_isolation_forest, select_cascade_contamination, sensitivity_sweep
from preprocessing.pipeline import build_pipeline
from schema.unified import project


def run_cascade(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
    meta: dict,
    families: str = "full",
    stage1_contamination: float | None = None,
    test_frac: float = 0.2,
) -> dict[str, Any]:
    """Fits all three stage models once on a stratified train split, runs the
    cascade forward on the held-out test split, and returns the funnel,
    per-stage latency, escalation diagnostics, and end-to-end metrics
    compared against each individual model on the identical test rows."""
    X_proj = project(X, families)
    random_state = config["cv"]["random_state"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_proj, y, test_size=test_frac, stratify=y, random_state=random_state
    )

    # --- Stage 1: Isolation Forest, tuned for recall (Step 2E.3's cascade
    # contamination choice is passed in by the caller; see run_cascade_from_sweep). ---
    iso_pipe = build_pipeline(build_isolation_forest(config, contamination=stage1_contamination), use_smote=False)
    t0 = time.perf_counter()
    iso_pipe.fit(X_train, y_train)
    t_iso_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    iso_pred_test = iso_pipe.predict(X_test)
    t_iso_predict = time.perf_counter() - t0

    survivors_mask = iso_pred_test == 1
    n_in_stage1 = len(X_test)
    n_discarded_stage1 = int((~survivors_mask).sum())
    n_survivors = int(survivors_mask.sum())

    # --- Stage 2: XGBoost on survivors. ---
    spw = compute_scale_pos_weight(y_train)
    xgb_pipe = build_pipeline(build_xgboost(config, spw), use_smote=False)
    t0 = time.perf_counter()
    xgb_pipe.fit(X_train, y_train)
    t_xgb_fit = time.perf_counter() - t0

    X_survivors = X_test[survivors_mask]
    t0 = time.perf_counter()
    if n_survivors:
        xgb_proba_survivors = xgb_pipe.predict_proba(X_survivors)[:, 1]
    else:
        xgb_proba_survivors = np.array([])
    t_xgb_predict = time.perf_counter() - t0
    xgb_pred_survivors = (xgb_proba_survivors >= 0.5).astype(int)

    # --- Autoencoder, fit alongside (needed for Stage 3's disagreement check). ---
    ae_pipe = build_pipeline(build_autoencoder(config), use_smote=False)
    t0 = time.perf_counter()
    ae_pipe.fit(X_train, y_train)
    t_ae_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    if n_survivors:
        ae_pred_survivors = ae_pipe.predict(X_survivors)
    else:
        ae_pred_survivors = np.array([])
    t_ae_predict = time.perf_counter() - t0

    # --- Stage 3: escalation band OR XGBoost/Autoencoder disagreement. ---
    lower = config["cascade"]["llm_lower"]
    upper = config["cascade"]["llm_upper"]
    in_band = (xgb_proba_survivors >= lower) & (xgb_proba_survivors <= upper)
    disagree = xgb_pred_survivors != ae_pred_survivors
    escalate_mask = in_band | disagree
    n_escalated = int(escalate_mask.sum())

    # 3D deferred (module docstring): escalated rows fall back to XGBoost's
    # own verdict, so final survivor predictions are simply xgb_pred_survivors.
    final_pred_survivors = xgb_pred_survivors

    y_pred_test = pd.Series(0, index=X_test.index, dtype="int64")
    y_pred_test.loc[X_survivors.index] = final_pred_survivors
    # Positive-class score for AUC purposes: discarded rows score 0 (Isolation
    # Forest called them normal with no XGBoost score to fall back on);
    # survivors carry XGBoost's own probability.
    y_score_test = pd.Series(0.0, index=X_test.index, dtype="float64")
    y_score_test.loc[X_survivors.index] = xgb_proba_survivors

    cascade_metrics = _compute_fold_metrics(y_test, y_pred_test, y_score_test)

    max_llm_calls = config["cascade"]["max_llm_calls"]

    funnel = {
        "stage1_isolation_forest": {
            "rows_in": n_in_stage1,
            "rows_discarded": n_discarded_stage1,
            "rows_passed": n_survivors,
            "fit_latency_s": t_iso_fit,
            "predict_latency_s": t_iso_predict,
            "predict_latency_s_per_row": t_iso_predict / n_in_stage1 if n_in_stage1 else None,
        },
        "stage2_xgboost": {
            "rows_in": n_survivors,
            "rows_resolved_confident": n_survivors - n_escalated,
            "fit_latency_s": t_xgb_fit,
            "predict_latency_s": t_xgb_predict,
            "predict_latency_s_per_row": t_xgb_predict / n_survivors if n_survivors else None,
        },
        "stage3_escalation": {
            "rows_in": n_survivors,
            "rows_escalated": n_escalated,
            "escalated_in_band": int(in_band.sum()),
            "escalated_by_disagreement": int(disagree.sum()),
            "escalated_by_both": int((in_band & disagree).sum()),
            "within_max_llm_calls_budget": n_escalated <= max_llm_calls,
            "max_llm_calls_budget": max_llm_calls,
            "note": "3D deferred by default (D7) -- escalated rows fall back to XGBoost's own verdict (module docstring), not scored by an LLM here.",
            "autoencoder_fit_latency_s": t_ae_fit,
            "autoencoder_predict_latency_s": t_ae_predict,
        },
    }

    return {
        "dataset_name": meta.get("dataset_name", "unknown"),
        "framing": meta.get("framing", "n/a"),
        "families_mode": families,
        "n_test": len(X_test),
        "stage1_contamination": iso_pipe.named_steps["estimator"].contamination,
        "funnel": funnel,
        "cascade_metrics": {k: cascade_metrics[k] for k in ("precision", "recall", "f1", "pr_auc", "roc_auc", "fpr")},
        "cascade_confusion_matrix": cascade_metrics["confusion_matrix"],
        "majority_baseline": _majority_baseline(y_test),
        "_fitted": {"iso_pipe": iso_pipe, "xgb_pipe": xgb_pipe, "ae_pipe": ae_pipe},
        "_test_split": {"X_test": X_test, "y_test": y_test},
    }


def individual_model_metrics_on_same_split(cascade_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Scores each of the three already-fitted stage models (from
    run_cascade()'s `_fitted`/`_test_split` payload) on the IDENTICAL test
    rows the cascade was scored on, so Chapter 8.4's "cascade vs. individual
    model" comparison is not confounded by a different split or refit."""
    from evaluation.metrics import _positive_proba

    fitted = cascade_result["_fitted"]
    X_test = cascade_result["_test_split"]["X_test"]
    y_test = cascade_result["_test_split"]["y_test"]

    out: dict[str, dict[str, Any]] = {}
    for name, pipe in fitted.items():
        model_name = name.replace("_pipe", "")
        y_pred = pipe.predict(X_test)
        y_proba = _positive_proba(pipe, X_test)
        m = _compute_fold_metrics(y_test, y_pred, y_proba)
        out[model_name] = {k: m[k] for k in ("precision", "recall", "f1", "pr_auc", "roc_auc", "fpr")}
    return out


def save_cascade_result(name: str, results: dict, out_dir: str | Path = "runs/metrics") -> Path:
    """Same one-JSON-per-experiment convention as the rest of the project.
    Strips the `_fitted`/`_test_split` payload (fitted estimators and raw
    DataFrames are not JSON-serialisable and don't belong in a results file
    -- individual_model_metrics_on_same_split() must be called before saving
    if that comparison is wanted in the JSON)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cascade_{name}.json"
    serialisable = {k: v for k, v in results.items() if not k.startswith("_")}
    path.write_text(json.dumps(serialisable, indent=2))
    return path
