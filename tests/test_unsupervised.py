"""
Step 2E verification. Synthetic data only (deterministic, no dependency on
real datasets) -- same discipline as tests/test_supervised.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from models.unsupervised import (
    IsolationForestDetector,
    build_isolation_forest,
    plot_sensitivity,
    run_isolation_forest,
    select_cascade_contamination,
    sensitivity_sweep,
)
from schema.unified import B_ONLY_COLUMNS, UNIFIED_COLUMNS


@pytest.fixture(scope="module")
def config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def _synthetic_frame(n_majority=300, n_minority=30, random_state=42):
    """90/9 imbalance (10:1), with the B_ONLY columns entirely NaN -- mirrors
    exactly what Exf2021Loader hands downstream code (same helper shape as
    tests/test_supervised.py's, kept independent per-file rather than shared
    so each model's test suite has no cross-file dependency)."""
    rng = np.random.default_rng(random_state)
    n = n_majority + n_minority

    X = pd.DataFrame(
        {col: rng.normal(loc=0.0, scale=1.0, size=n) for col in UNIFIED_COLUMNS}
    )
    for col in B_ONLY_COLUMNS:
        X[col] = np.nan

    signal = X["vol_primary"] + rng.normal(scale=1.5, size=n)
    y = pd.Series((signal > np.quantile(signal, 1 - n_minority / n)).astype(int))

    shuffle_idx = rng.permutation(n)
    X = X.iloc[shuffle_idx].reset_index(drop=True)
    y = y.iloc[shuffle_idx].reset_index(drop=True)
    return X, y


class _StubForest:
    """Fake sklearn IsolationForest exposing only what
    IsolationForestDetector calls, so the remapping/sign-flip logic can be
    hand-verified without depending on what a real IsolationForest actually
    fits (which has no closed-form expected output to assert against)."""

    def __init__(self, predict_return, decision_return):
        self._predict_return = np.asarray(predict_return)
        self._decision_return = np.asarray(decision_return)

    def fit(self, X):
        return self

    def predict(self, X):
        return self._predict_return

    def decision_function(self, X):
        return self._decision_return


def test_predict_remaps_sklearn_convention_to_project_convention():
    # sklearn: -1 = outlier, 1 = inlier. Project: 1 = attack, 0 = benign.
    det = IsolationForestDetector(random_state=42)
    det.model_ = _StubForest(predict_return=[-1, 1, 1, -1], decision_return=[0, 0, 0, 0])

    preds = det.predict(np.zeros((4, 1)))

    assert list(preds) == [1, 0, 0, 1]


def test_decision_function_sign_is_flipped_so_higher_means_more_attack():
    # sklearn: higher decision_function = more normal (inlier). Project needs
    # higher = more likely POSITIVE/attack for evaluate()'s AUC calls.
    det = IsolationForestDetector()
    det.model_ = _StubForest(predict_return=[1, 1], decision_return=[0.3, -0.2])

    scores = det.decision_function(np.zeros((2, 1)))

    assert list(scores) == [pytest.approx(-0.3), pytest.approx(0.2)]


def test_build_isolation_forest_reads_locked_hyperparameters_from_config(config):
    model = build_isolation_forest(config)
    cfg = config["models"]["isolation_forest"]
    assert model.n_estimators == cfg["n_estimators"]
    assert model.max_samples == cfg["max_samples"]
    assert model.contamination == cfg["contamination"]
    assert model.max_features == cfg["max_features"]
    assert model.random_state == cfg["random_state"]
    assert model.n_jobs == cfg["n_jobs"]


def test_build_isolation_forest_contamination_override_for_sweep(config):
    model = build_isolation_forest(config, contamination=0.05)
    assert model.contamination == 0.05
    # Every other hyperparameter is untouched by the override.
    assert model.n_estimators == config["models"]["isolation_forest"]["n_estimators"]


def test_run_isolation_forest_end_to_end_structure_and_no_smote(config):
    X, y = _synthetic_frame()
    meta = {"dataset_name": "synthetic", "framing": "n/a"}

    result = run_isolation_forest(X, y, config, meta, families="full")

    assert result["dataset_name"] == "synthetic"
    assert result["families_mode"] == "full"
    assert set(result["mean"]) == {"precision", "recall", "f1", "pr_auc", "roc_auc", "fpr"}
    assert "majority_baseline" in result
    assert result["b_only_columns_all_nan"] is True


def test_run_isolation_forest_uses_no_smote_step():
    """Resampling an unsupervised density estimator is incoherent (module
    docstring) -- assert the built pipeline structurally has no smote step."""
    from models.unsupervised import build_isolation_forest as _build
    from preprocessing.pipeline import build_pipeline

    pipe = build_pipeline(
        _build({"models": {"isolation_forest": {
            "n_estimators": 10, "max_samples": 16, "contamination": 0.1,
            "max_features": 1.0, "random_state": 42, "n_jobs": -1,
        }}}),
        use_smote=False,
    )
    assert "smote" not in dict(pipe.steps)


def test_sensitivity_sweep_covers_all_configured_contaminations(config):
    X, y = _synthetic_frame()
    meta = {"dataset_name": "synthetic", "framing": "n/a"}

    sweep = sensitivity_sweep(X, y, config, meta, families="full")

    expected = config["models"]["isolation_forest_sensitivity_sweep"]["contamination"]
    assert [r["contamination"] for r in sweep] == expected
    for r in sweep:
        assert 0.0 <= r["f1"] <= 1.0
        assert 0.0 <= r["fpr"] <= 1.0
        assert 0.0 <= r["recall"] <= 1.0
        assert r["result"]["families_mode"] == "full"


def test_plot_sensitivity_writes_a_file(tmp_path, config):
    X, y = _synthetic_frame(n_majority=100, n_minority=10)
    meta = {"dataset_name": "synthetic", "framing": "n/a"}
    small_config = {
        **config,
        "models": {
            **config["models"],
            "isolation_forest_sensitivity_sweep": {"contamination": [0.1, 0.2]},
        },
    }

    sweep = sensitivity_sweep(X, y, small_config, meta, families="full")
    out_path = tmp_path / "sweep.png"
    plot_sensitivity(sweep, "synthetic", out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_select_cascade_contamination_picks_max_recall_within_fpr_budget():
    sweep = [
        {"contamination": 0.05, "f1": 0.50, "fpr": 0.10, "recall": 0.60},
        {"contamination": 0.10, "f1": 0.60, "fpr": 0.30, "recall": 0.75},
        {"contamination": 0.20, "f1": 0.55, "fpr": 0.60, "recall": 0.90},
        {"contamination": 0.30, "f1": 0.50, "fpr": 0.55, "recall": 0.95},
    ]
    # Only the first two are within max_tolerable_fpr=0.5; of those, 0.10
    # has the higher recall (0.75 > 0.60) despite NOT having the higher F1
    # (0.60 > 0.50 here, but the point is the selection key is recall, not F1).
    result = select_cascade_contamination(sweep, max_tolerable_fpr=0.5)

    assert result["contamination"] == 0.10
    assert result["recall"] == pytest.approx(0.75)
    assert result["met_fpr_tolerance"] is True


def test_select_cascade_contamination_falls_back_when_no_candidate_meets_budget():
    sweep = [
        {"contamination": 0.05, "f1": 0.50, "fpr": 0.60, "recall": 0.60},
        {"contamination": 0.10, "f1": 0.60, "fpr": 0.70, "recall": 0.75},
    ]
    result = select_cascade_contamination(sweep, max_tolerable_fpr=0.5)

    # No candidate meets the budget -> fall back to global max recall, but
    # flag that the fallback happened rather than silently reporting it as met.
    assert result["contamination"] == 0.10
    assert result["recall"] == pytest.approx(0.75)
    assert result["met_fpr_tolerance"] is False


def test_select_cascade_contamination_raises_on_empty_sweep():
    with pytest.raises(ValueError):
        select_cascade_contamination([])
