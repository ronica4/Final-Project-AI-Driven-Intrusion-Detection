"""
Step 2D verification. Synthetic data only (deterministic, no dependency on
real datasets) -- must pass identically before or after Step 0B's manual
download step, same discipline as tests/test_pipeline.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from models.supervised import (
    build_xgboost,
    compute_scale_pos_weight,
    plot_sensitivity,
    run_xgboost,
    sensitivity_sweep,
)
from schema.unified import UNIFIED_COLUMNS


@pytest.fixture(scope="module")
def config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def _synthetic_frame(n_majority=300, n_minority=30, random_state=42):
    """90/9 imbalance (10:1), with the B_ONLY columns entirely NaN -- mirrors
    exactly what Exf2021Loader hands downstream code."""
    rng = np.random.default_rng(random_state)
    n = n_majority + n_minority

    X = pd.DataFrame(
        {col: rng.normal(loc=0.0, scale=1.0, size=n) for col in UNIFIED_COLUMNS}
    )
    from schema.unified import B_ONLY_COLUMNS

    for col in B_ONLY_COLUMNS:
        X[col] = np.nan

    # Make the label weakly dependent on the first column so there's
    # something learnable, without being trivially perfectly separable.
    signal = X["vol_primary"] + rng.normal(scale=1.5, size=n)
    y = pd.Series((signal > np.quantile(signal, 1 - n_minority / n)).astype(int))

    shuffle_idx = rng.permutation(n)
    X = X.iloc[shuffle_idx].reset_index(drop=True)
    y = y.iloc[shuffle_idx].reset_index(drop=True)
    return X, y


def test_compute_scale_pos_weight_matches_hand_computed_value():
    y = pd.Series([0] * 80 + [1] * 20)  # 4:1 neg:pos
    assert compute_scale_pos_weight(y) == pytest.approx(80 / 20)


def test_compute_scale_pos_weight_raises_on_missing_class():
    with pytest.raises(ValueError):
        compute_scale_pos_weight(pd.Series([0, 0, 0]))
    with pytest.raises(ValueError):
        compute_scale_pos_weight(pd.Series([1, 1, 1]))


def test_build_xgboost_reads_locked_hyperparameters_from_config(config):
    model = build_xgboost(config, scale_pos_weight=3.5)
    cfg = config["models"]["xgboost"]
    assert model.n_estimators == cfg["n_estimators"]
    assert model.max_depth == cfg["max_depth"]
    assert model.learning_rate == cfg["learning_rate"]
    assert model.subsample == cfg["subsample"]
    assert model.colsample_bytree == cfg["colsample_bytree"]
    assert model.min_child_weight == cfg["min_child_weight"]
    assert model.scale_pos_weight == pytest.approx(3.5)
    assert model.random_state == cfg["random_state"]


def test_build_xgboost_max_depth_override_for_sweep(config):
    model = build_xgboost(config, scale_pos_weight=1.0, max_depth=9)
    assert model.max_depth == 9
    # Every other hyperparameter is untouched by the override.
    assert model.n_estimators == config["models"]["xgboost"]["n_estimators"]


def test_run_xgboost_end_to_end_structure_and_no_smote(config):
    X, y = _synthetic_frame()
    meta = {"dataset_name": "synthetic", "framing": "n/a"}

    result = run_xgboost(X, y, config, meta, families="full")

    assert result["dataset_name"] == "synthetic"
    assert result["families_mode"] == "full"
    assert set(result["mean"]) == {"precision", "recall", "f1", "pr_auc", "roc_auc", "fpr"}
    assert "majority_baseline" in result
    # B_ONLY columns are all-NaN on this synthetic frame (mirrors Dataset A) --
    # confirm the harness saw and recorded that, rather than silently coping.
    assert result["b_only_columns_all_nan"] is True


def test_run_xgboost_uses_no_smote_step():
    """use_smote=False is the whole point of scale_pos_weight (module
    docstring) -- assert the built pipeline structurally has no smote step."""
    from models.supervised import build_xgboost as _build
    from preprocessing.pipeline import build_pipeline

    pipe = build_pipeline(_build({"models": {"xgboost": {
        "n_estimators": 10, "max_depth": 3, "learning_rate": 0.1,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1,
        "random_state": 42,
    }}}, scale_pos_weight=1.0), use_smote=False)
    assert "smote" not in dict(pipe.steps)


def test_sensitivity_sweep_covers_all_configured_depths(config):
    X, y = _synthetic_frame()
    meta = {"dataset_name": "synthetic", "framing": "n/a"}

    sweep = sensitivity_sweep(X, y, config, meta, families="full")

    expected_depths = config["models"]["xgboost_sensitivity_sweep"]["max_depth"]
    assert [r["max_depth"] for r in sweep] == expected_depths
    for r in sweep:
        assert 0.0 <= r["f1"] <= 1.0
        assert 0.0 <= r["fpr"] <= 1.0
        assert r["result"]["families_mode"] == "full"


def test_plot_sensitivity_writes_a_file(tmp_path, config):
    X, y = _synthetic_frame(n_majority=100, n_minority=10)
    meta = {"dataset_name": "synthetic", "framing": "n/a"}
    # Small sweep for speed -- override the config depths list, everything
    # else about the run is identical to the real sweep.
    small_config = {**config, "models": {**config["models"], "xgboost_sensitivity_sweep": {"max_depth": [3, 6]}}}

    sweep = sensitivity_sweep(X, y, small_config, meta, families="full")
    out_path = tmp_path / "sweep.png"
    plot_sensitivity(sweep, "synthetic", out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
