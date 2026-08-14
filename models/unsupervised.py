"""
Step 2E -- Isolation Forest, the project's unsupervised first-stage detector.

Dataset-agnostic by construction (Dataset Dependency Rule): every function
here takes (X, y, config, ...) already produced by a loader. Nothing in this
module knows or asks which dataset it is scoring.

use_smote=False, and not merely as a default (PROJECT_PLAN.md Step 2E point 1
/ preprocessing/pipeline.py's docstring): resampling the training fold to a
synthetic 1:1 balance before fitting a density estimator is incoherent.
Isolation Forest's whole mechanism is "learn what the majority density looks
like, flag what sits outside it" -- feeding it a fold where SMOTE has
manufactured a fake 50% minority density would teach it that the attack
class IS part of the normal density, defeating the model's premise before
it fits a single tree. XGBoost (Step 2D) uses scale_pos_weight instead of
SMOTE because it has a native reweighting knob; Isolation Forest has no such
knob to misuse, and unlike XGBoost, has no supervised objective to reweight
in the first place -- so the honest answer for this model is simply "no
resampling at all," not "resampling via a different mechanism."

IsolationForestDetector wraps sklearn's IsolationForest to bridge two
mismatches with the rest of this codebase:
  1. sklearn's IsolationForest.predict() returns {-1: outlier, 1: inlier},
     not this project's {1: attack, 0: benign} convention. Left unmapped,
     evaluate()'s confusion-matrix and F1/precision/recall calls (which
     assume labels {0, 1}) would silently score against the wrong classes.
  2. sklearn's decision_function() returns HIGHER = more normal (inlier).
     evaluation.metrics._positive_proba() expects higher = more likely
     POSITIVE (attack) for PR-AUC/ROC-AUC to rank correctly, so the sign is
     flipped once here rather than risk every downstream AUC being silently
     inverted.
Fitting on y=None is intentional, not an oversight: this is the one place in
the codebase where the label is available at fit time but structurally must
not be used -- the entire point of Step 2E is a detector that has never seen
a label. Pipeline.fit(X, y) forwards y positionally to every step regardless,
so the wrapper accepts and discards it rather than breaking the Pipeline
contract build_pipeline() (preprocessing/pipeline.py) relies on for every
model in this project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest

from evaluation.metrics import evaluate
from models.supervised import plot_sensitivity as _plot_sensitivity
from preprocessing.pipeline import build_pipeline, get_cv
from schema.unified import B_ONLY_COLUMNS, project


class IsolationForestDetector(BaseEstimator):
    """sklearn-compatible adapter: fits an unsupervised IsolationForest but
    exposes predict()/decision_function() in this project's {0, 1} /
    higher-is-more-attack conventions, so it drops into evaluate() and
    build_pipeline() exactly like every supervised model in models/supervised.py.

    Subclasses BaseEstimator (not IsolationForest itself) so get_params()/
    set_params() are derived automatically from __init__'s signature -- the
    same discipline sklearn.base.clone() relies on for every fold in
    evaluate(), and deliberately does not inherit IsolationForest's own
    predict()/decision_function(), which is exactly what needs overriding.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int | float | str = "auto",
        contamination: float | str = "auto",
        max_features: int | float = 1.0,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None) -> "IsolationForestDetector":
        # y is accepted only to satisfy the Pipeline/estimator fit(X, y)
        # contract and is deliberately unused -- see module docstring.
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X)
        return self

    def predict(self, X) -> np.ndarray:
        """Remaps sklearn's {-1: outlier, 1: inlier} to this project's
        {1: attack, 0: benign}."""
        raw = self.model_.predict(X)
        return np.where(raw == -1, 1, 0)

    def decision_function(self, X) -> np.ndarray:
        """Sign-flipped so higher = more anomalous = more likely attack,
        matching evaluation.metrics._positive_proba()'s expectation for the
        positive class (see module docstring, mismatch 2)."""
        return -self.model_.decision_function(X)


def build_isolation_forest(
    config: dict, contamination: float | None = None
) -> IsolationForestDetector:
    """Every hyperparameter read from config.yaml -- none hardcoded here
    (PROJECT_PLAN.md Step 0D / Ch 7.3 rubric item: no library defaults).
    contamination may be overridden for the Step 2E.3 sensitivity sweep;
    every other value stays at its locked config value."""
    cfg = config["models"]["isolation_forest"]
    return IsolationForestDetector(
        n_estimators=cfg["n_estimators"],
        max_samples=cfg["max_samples"],
        contamination=contamination if contamination is not None else cfg["contamination"],
        max_features=cfg["max_features"],
        random_state=cfg["random_state"],
        n_jobs=cfg["n_jobs"],
    )


def run_isolation_forest(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
    meta: dict,
    families: str = "full",
    contamination: float | None = None,
) -> dict[str, Any]:
    """One full stratified-CV evaluation of Isolation Forest on (X, y),
    scored via the shared 2A' harness (evaluation.metrics.evaluate) -- no
    bespoke scoring. use_smote=False throughout (module docstring)."""
    X_proj = project(X, families)
    cv = get_cv(y, config)
    model = build_pipeline(
        build_isolation_forest(config, contamination=contamination), use_smote=False
    )

    result = evaluate(model, X_proj, y, cv, meta, families=families)
    # Extra field appended alongside, not through meta (evaluate()'s frozen
    # schema only pulls dataset_name/framing out of meta -- see Step 2A').
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
    """Vary contamination alone (Step 2E.3); everything else stays locked.
    Returns one summary dict per contamination value plus the full
    evaluate() result -- recall is carried alongside f1/fpr because the
    cascade selection criterion (select_cascade_contamination below)
    optimises recall, not F1."""
    contaminations = config["models"]["isolation_forest_sensitivity_sweep"]["contamination"]
    sweep = []
    for c in contaminations:
        result = run_isolation_forest(X, y, config, meta, families=families, contamination=c)
        sweep.append(
            {
                "contamination": c,
                "f1": result["mean"]["f1"],
                "fpr": result["mean"]["fpr"],
                "recall": result["mean"]["recall"],
                "result": result,
            }
        )
    return sweep


def plot_sensitivity(sweep: list[dict[str, Any]], dataset_label: str, path: str | Path) -> None:
    """Thin wrapper over models.supervised.plot_sensitivity's shared twin-axis
    style (PROJECT_PLAN.md Step 2E: "plot F1 and FPR... reused by Step 2E's
    contamination sweep") -- no plotting code duplicated here."""
    _plot_sensitivity(
        sweep,
        dataset_label,
        path,
        param_key="contamination",
        param_label="contamination",
        model_label="Isolation Forest",
    )


def select_cascade_contamination(
    sweep: list[dict[str, Any]], max_tolerable_fpr: float = 0.5
) -> dict[str, Any]:
    """Step 2E.3's cascade-threshold decision rule, made explicit and testable
    rather than left as a sentence in the report: among contamination values
    whose mean FPR is within `max_tolerable_fpr`, pick the one that MAXIMISES
    RECALL, not the one that maximises F1.

    Why recall over F1 here specifically: this model is the cascade's FIRST
    stage (Step 3C), whose entire job is "never let an attack through
    unexamined." A first-stage filter that misses attacks (low recall) is a
    fatal, unrecoverable failure -- nothing downstream ever sees that row
    again. A first-stage filter that over-flags (high FPR / low precision) is
    merely expensive -- it costs the second-stage model (XGBoost) and, in the
    worst case, the LLM arbiter some extra work, but it does not lose an
    attack. F1 penalises exactly the over-flagging that this stage is
    supposed to tolerate, so optimising F1 here would pick a *worse* cascade
    threshold than optimising recall directly.

    If no candidate meets `max_tolerable_fpr`, falls back to the
    highest-recall point in the whole sweep and flags that explicitly via
    met_fpr_tolerance=False -- the caller (and the report) must not read a
    contamination value as "meets the FPR budget" when it does not.
    """
    if not sweep:
        raise ValueError("sweep must be non-empty")

    candidates = [r for r in sweep if r["fpr"] <= max_tolerable_fpr]
    pool = candidates if candidates else sweep
    best = max(pool, key=lambda r: r["recall"])

    return {
        "contamination": best["contamination"],
        "recall": best["recall"],
        "fpr": best["fpr"],
        "f1": best["f1"],
        "max_tolerable_fpr": max_tolerable_fpr,
        "met_fpr_tolerance": bool(candidates),
    }
