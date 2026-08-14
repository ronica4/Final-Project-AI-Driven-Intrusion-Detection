"""
Step 2F verification. Synthetic data only (deterministic, no dependency on
real datasets) -- same discipline as tests/test_supervised.py and
tests/test_unsupervised.py. Training hyperparameters are overridden to tiny
values in a copied config so the suite stays fast; the real, locked values
only ever come from config/config.yaml itself (test_build_* below).
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from models.deep import (
    AutoencoderDetector,
    CNNClassifier,
    build_autoencoder,
    build_cnn,
    fit_full_for_training_curve,
    plot_training_curve,
    run_autoencoder,
    run_cnn,
)
from schema.unified import B_ONLY_COLUMNS, UNIFIED_COLUMNS


@pytest.fixture(scope="module")
def config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def fast_config(config):
    """Same locked architecture/loss/optimizer as config.yaml, but training
    budgets slashed so CV + a handful of epochs finishes in seconds on a
    ~250-row synthetic frame."""
    cfg = copy.deepcopy(config)
    cfg["cv"]["n_splits"] = 3
    cfg["models"]["cnn"]["max_epochs"] = 3
    cfg["models"]["cnn"]["early_stopping_patience"] = 2
    cfg["models"]["cnn"]["batch_size"] = 32
    cfg["models"]["cnn"]["val_frac"] = 0.3
    cfg["models"]["autoencoder"]["max_epochs"] = 3
    cfg["models"]["autoencoder"]["batch_size"] = 32
    return cfg


def _synthetic_frame(n_majority=200, n_minority=50, random_state=42):
    """80/20-ish imbalance, with the B_ONLY columns entirely NaN -- mirrors
    exactly what Exf2021Loader hands downstream code (same helper shape as
    tests/test_supervised.py's and tests/test_unsupervised.py's, kept
    independent per-file so each model's test suite has no cross-file
    dependency)."""
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


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------
def test_build_cnn_reads_locked_hyperparameters_from_config(config):
    model = build_cnn(config)
    cfg = config["models"]["cnn"]
    assert tuple(model.conv_filters) == tuple(cfg["conv_filters"])
    assert model.kernel_size == cfg["kernel_size"]
    assert model.padding == cfg["padding"]
    assert model.dropout == cfg["dropout"]
    assert model.dense_units == cfg["dense_units"]
    assert model.learning_rate == cfg["learning_rate"]
    assert model.batch_size == cfg["batch_size"]
    assert model.early_stopping_patience == cfg["early_stopping_patience"]
    assert model.max_epochs == cfg["max_epochs"]
    assert model.val_frac == cfg["val_frac"]
    assert model.random_state == cfg["random_state"]


def test_build_autoencoder_reads_locked_hyperparameters_from_config(config):
    model = build_autoencoder(config)
    cfg = config["models"]["autoencoder"]
    assert tuple(model.layers) == tuple(cfg["layers"])
    assert model.learning_rate == cfg["learning_rate"]
    assert model.batch_size == cfg["batch_size"]
    assert model.max_epochs == cfg["max_epochs"]
    assert model.threshold_percentile == cfg["threshold_percentile"]
    assert model.random_state == cfg["random_state"]


# ---------------------------------------------------------------------------
# Pipeline structure (SMOTE on/off)
# ---------------------------------------------------------------------------
def test_run_cnn_uses_smote_step():
    from preprocessing.pipeline import build_pipeline

    pipe = build_pipeline(
        CNNClassifier(max_epochs=1, batch_size=16, val_frac=0.3), use_smote=True
    )
    assert "smote" in dict(pipe.steps)


def test_run_autoencoder_uses_no_smote_step():
    """Resampling a model fit exclusively on benign rows is incoherent
    (module docstring) -- assert the built pipeline structurally has no
    smote step."""
    from preprocessing.pipeline import build_pipeline

    pipe = build_pipeline(AutoencoderDetector(max_epochs=1, batch_size=16), use_smote=False)
    assert "smote" not in dict(pipe.steps)


# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------
def test_cnn_adapts_to_actual_input_width_not_hardcoded_11():
    """families="intersection" hands the estimator 7 columns, not 11 --
    the network must be built from X.shape[1] at fit time."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 7)).astype(np.float32)
    y = np.array([0] * 45 + [1] * 15)

    model = CNNClassifier(max_epochs=2, batch_size=16, val_frac=0.3, random_state=42)
    model.fit(X, y)

    assert model.model_.flatten_dim == model.conv_filters[-1] * 7
    proba = model.predict_proba(X)
    assert proba.shape == (60, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_cnn_reproducible_across_two_runs_same_seed():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 5)).astype(np.float32)
    y = np.array([0] * 60 + [1] * 20)

    m1 = CNNClassifier(max_epochs=3, batch_size=16, val_frac=0.3, random_state=42).fit(X, y)
    m2 = CNNClassifier(max_epochs=3, batch_size=16, val_frac=0.3, random_state=42).fit(X, y)

    p1 = m1.predict_proba(X)
    p2 = m2.predict_proba(X)
    assert np.allclose(p1, p2, atol=1e-6)


def test_run_cnn_end_to_end_structure_and_smote(fast_config):
    X, y = _synthetic_frame()
    meta = {"dataset_name": "synthetic", "framing": "n/a"}

    result = run_cnn(X, y, fast_config, meta, families="full")

    assert result["dataset_name"] == "synthetic"
    assert result["families_mode"] == "full"
    assert set(result["mean"]) == {"precision", "recall", "f1", "pr_auc", "roc_auc", "fpr"}
    assert "majority_baseline" in result
    assert result["b_only_columns_all_nan"] is True


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------
def test_autoencoder_adapts_to_actual_input_width_not_hardcoded_11():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 7)).astype(np.float32)
    y = np.array([0] * 45 + [1] * 15)

    model = AutoencoderDetector(max_epochs=2, batch_size=16, random_state=42)
    model.fit(X, y)

    scores = model.decision_function(X)
    assert scores.shape == (60,)
    assert (scores >= 0).all()  # mean-squared reconstruction error is non-negative


def test_autoencoder_fits_on_benign_rows_only():
    """The central leakage guard for this model (PROJECT_PLAN.md Step 2F
    point 2): changing ONLY the attack rows must not change the fitted
    model or its threshold at all, since fit() must never train on them."""
    rng = np.random.default_rng(7)
    n_benign, n_attack = 80, 20
    X_benign = rng.normal(loc=0.0, scale=1.0, size=(n_benign, 6)).astype(np.float32)
    y = np.array([0] * n_benign + [1] * n_attack)

    X_attack_a = np.full((n_attack, 6), 100.0, dtype=np.float32)
    X_attack_b = np.full((n_attack, 6), -999.0, dtype=np.float32)

    Xa = np.vstack([X_benign, X_attack_a])
    Xb = np.vstack([X_benign, X_attack_b])

    model_a = AutoencoderDetector(max_epochs=3, batch_size=16, random_state=42).fit(Xa, y)
    model_b = AutoencoderDetector(max_epochs=3, batch_size=16, random_state=42).fit(Xb, y)

    assert model_a.threshold_ == pytest.approx(model_b.threshold_)
    for p_a, p_b in zip(model_a.model_.parameters(), model_b.model_.parameters()):
        assert torch.allclose(p_a, p_b, atol=1e-6)


def test_autoencoder_threshold_is_95th_percentile_of_benign_training_errors():
    rng = np.random.default_rng(3)
    n_benign, n_attack = 100, 20
    X_benign = rng.normal(size=(n_benign, 5)).astype(np.float32)
    X_attack = rng.normal(loc=10.0, size=(n_attack, 5)).astype(np.float32)
    X = np.vstack([X_benign, X_attack])
    y = np.array([0] * n_benign + [1] * n_attack)

    model = AutoencoderDetector(
        max_epochs=3, batch_size=16, threshold_percentile=95, random_state=42
    ).fit(X, y)

    benign_errors = model._reconstruction_error(X_benign)
    expected = np.percentile(benign_errors, 95)
    assert model.threshold_ == pytest.approx(expected)


def test_autoencoder_reproducible_across_two_runs_same_seed():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 5)).astype(np.float32)
    y = np.array([0] * 60 + [1] * 20)

    m1 = AutoencoderDetector(max_epochs=3, batch_size=16, random_state=42).fit(X, y)
    m2 = AutoencoderDetector(max_epochs=3, batch_size=16, random_state=42).fit(X, y)

    assert m1.threshold_ == pytest.approx(m2.threshold_)
    s1 = m1.decision_function(X)
    s2 = m2.decision_function(X)
    assert np.allclose(s1, s2, atol=1e-6)


def test_run_autoencoder_end_to_end_structure_and_no_smote(fast_config):
    X, y = _synthetic_frame()
    meta = {"dataset_name": "synthetic", "framing": "n/a"}

    result = run_autoencoder(X, y, fast_config, meta, families="full")

    assert result["dataset_name"] == "synthetic"
    assert result["families_mode"] == "full"
    assert set(result["mean"]) == {"precision", "recall", "f1", "pr_auc", "roc_auc", "fpr"}
    assert "majority_baseline" in result
    assert result["b_only_columns_all_nan"] is True


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
def test_fit_full_for_training_curve_returns_history_for_both_models(fast_config):
    X, y = _synthetic_frame()

    cnn_history = fit_full_for_training_curve(X, y, fast_config, model_name="cnn")
    ae_history = fit_full_for_training_curve(X, y, fast_config, model_name="autoencoder")

    assert len(cnn_history) > 0
    assert all("val_pr_auc" in h for h in cnn_history)
    assert len(ae_history) == fast_config["models"]["autoencoder"]["max_epochs"]
    assert all("train_mse" in h for h in ae_history)


def test_fit_full_for_training_curve_rejects_unknown_model_name(fast_config):
    X, y = _synthetic_frame()
    with pytest.raises(ValueError):
        fit_full_for_training_curve(X, y, fast_config, model_name="not_a_model")


def test_plot_training_curve_writes_a_file(tmp_path):
    history = [{"epoch": i, "val_pr_auc": 0.5 + i * 0.01} for i in range(5)]
    out_path = tmp_path / "curve.png"

    plot_training_curve(history, "val_pr_auc", "PR-AUC", "CNN", "synthetic", out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
