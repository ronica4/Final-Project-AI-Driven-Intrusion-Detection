"""
Step 2G verification. Synthetic data only (deterministic, no dependency on
real datasets), two independent synthetic frames standing in for Dataset A
and Dataset B -- same discipline as every other tests/test_*.py in this
project. Training hyperparameters are overridden to tiny values in a copied
config for the deep-model tests, same pattern as tests/test_deep.py's
fast_config.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from evaluation.cross_dataset import (
    VALID_MODELS,
    ablation_f1_only_vs_intersection,
    build_transfer_matrix,
    distribution_shift,
    indomain_cell,
    plot_distribution_shift,
    save_cross_dataset_result,
    transfer_cell,
)
from schema.unified import B_ONLY_COLUMNS, INTERSECTION_COLUMNS, UNIFIED_COLUMNS


@pytest.fixture(scope="module")
def config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def fast_config(config):
    cfg = copy.deepcopy(config)
    cfg["cv"]["n_splits"] = 3
    cfg["models"]["cnn"]["max_epochs"] = 3
    cfg["models"]["cnn"]["early_stopping_patience"] = 2
    cfg["models"]["cnn"]["batch_size"] = 32
    cfg["models"]["cnn"]["val_frac"] = 0.3
    cfg["models"]["autoencoder"]["max_epochs"] = 3
    cfg["models"]["autoencoder"]["batch_size"] = 32
    cfg["models"]["isolation_forest"]["max_samples"] = 32
    return cfg


def _synthetic_frame(n_majority, n_minority, loc_shift=0.0, random_state=42):
    """B_ONLY columns all-NaN, mirroring Exf2021Loader's shape -- both "A" and
    "B" stand-ins share this shape since Step 2G's mode="intersection"
    projection only ever looks at the 7 shared columns anyway. loc_shift
    displaces the signal column so the two frames are NOT identically
    distributed, which is what distribution_shift() needs to have something
    to detect."""
    rng = np.random.default_rng(random_state)
    n = n_majority + n_minority

    X = pd.DataFrame(
        {col: rng.normal(loc=loc_shift, scale=1.0, size=n) for col in UNIFIED_COLUMNS}
    )
    for col in B_ONLY_COLUMNS:
        X[col] = np.nan

    signal = X["vol_primary"] + rng.normal(scale=1.5, size=n)
    y = pd.Series((signal > np.quantile(signal, 1 - n_minority / n)).astype(int))

    shuffle_idx = rng.permutation(n)
    X = X.iloc[shuffle_idx].reset_index(drop=True)
    y = y.iloc[shuffle_idx].reset_index(drop=True)
    return X, y


@pytest.fixture
def dataset_a():
    return _synthetic_frame(200, 50, loc_shift=0.0, random_state=1)


@pytest.fixture
def dataset_b():
    return _synthetic_frame(200, 50, loc_shift=1.5, random_state=2)


# ---------------------------------------------------------------------------
# transfer_cell / indomain_cell -- structure, one fast model (isolation_forest)
# ---------------------------------------------------------------------------
def test_transfer_cell_rejects_unknown_model(config, dataset_a, dataset_b):
    X_a, y_a = dataset_a
    X_b, y_b = dataset_b
    with pytest.raises(ValueError):
        transfer_cell("not_a_model", config, X_a, y_a, X_b, y_b)


def test_transfer_cell_fits_on_source_only_and_scores_on_target(fast_config, dataset_a, dataset_b):
    X_a, y_a = dataset_a
    X_b, y_b = dataset_b

    result = transfer_cell("isolation_forest", fast_config, X_a, y_a, X_b, y_b, families="intersection")

    assert result["model_name"] == "isolation_forest"
    assert result["families_mode"] == "intersection"
    assert result["n_train"] == len(X_a)
    assert result["n_test"] == len(X_b)
    assert set(result["metrics"]) == {"precision", "recall", "f1", "pr_auc", "roc_auc", "fpr"}
    assert "majority_baseline" in result
    # majority_baseline must reflect the TARGET labels, not the source's.
    assert result["majority_baseline"]["majority_class"] in (0, 1)


def test_indomain_cell_uses_cv_and_intersection_columns(fast_config, dataset_a):
    X_a, y_a = dataset_a
    meta = {"dataset_name": "synthetic_a", "framing": "n/a"}

    result = indomain_cell("isolation_forest", fast_config, X_a, y_a, meta, families="intersection")

    assert result["families_mode"] == "intersection"
    assert result["n_rows"] == len(X_a)
    assert set(result["metrics"]) == {"precision", "recall", "f1", "pr_auc", "roc_auc", "fpr"}


def test_build_transfer_matrix_has_all_four_cells(fast_config, dataset_a, dataset_b):
    X_a, y_a = dataset_a
    X_b, y_b = dataset_b
    meta_a = {"dataset_name": "synthetic_a", "framing": "n/a"}
    meta_b = {"dataset_name": "synthetic_b", "framing": "n/a"}

    matrix = build_transfer_matrix(
        "isolation_forest", fast_config, X_a, y_a, meta_a, X_b, y_b, meta_b, families="intersection"
    )

    assert set(matrix) == {
        "model_name", "families_mode",
        "train_A_test_A", "train_B_test_B", "train_A_test_B", "train_B_test_A",
    }
    # In-domain cells came from CV (n_rows == full dataset); transfer cells
    # from a single fit/score (n_train/n_test, not n_rows).
    assert "n_rows" in matrix["train_A_test_A"]
    assert "n_train" in matrix["train_A_test_B"] and "n_test" in matrix["train_A_test_B"]


def test_build_transfer_matrix_cnn_wires_through(fast_config, dataset_a, dataset_b):
    """One deep-model smoke test (not exhaustive over all four models) --
    confirms the CNN's SMOTE-on / different builder path also fits the
    transfer_cell/indomain_cell contract, using tiny epochs for speed."""
    X_a, y_a = dataset_a
    X_b, y_b = dataset_b
    meta_a = {"dataset_name": "synthetic_a", "framing": "n/a"}
    meta_b = {"dataset_name": "synthetic_b", "framing": "n/a"}

    matrix = build_transfer_matrix(
        "cnn", fast_config, X_a, y_a, meta_a, X_b, y_b, meta_b, families="intersection"
    )
    assert matrix["model_name"] == "cnn"
    assert 0.0 <= matrix["train_A_test_B"]["metrics"]["f1"] <= 1.0


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------
def test_ablation_reuses_passed_in_intersection_cells_not_recomputed(fast_config, dataset_a, dataset_b):
    X_a, y_a = dataset_a
    X_b, y_b = dataset_b
    meta_a = {"dataset_name": "synthetic_a", "framing": "n/a"}
    meta_b = {"dataset_name": "synthetic_b", "framing": "n/a"}

    matrix = build_transfer_matrix(
        "isolation_forest", fast_config, X_a, y_a, meta_a, X_b, y_b, meta_b, families="intersection"
    )
    ablation = ablation_f1_only_vs_intersection(
        "isolation_forest", fast_config, X_a, y_a, X_b, y_b, matrix
    )

    assert ablation["train_A_test_B"]["intersection"] is matrix["train_A_test_B"]
    assert ablation["train_A_test_B"]["F1_only"]["families_mode"] == "F1_only"
    assert isinstance(ablation["train_A_test_B"]["f1_only_transfers_better"], bool)


# ---------------------------------------------------------------------------
# Distribution shift
# ---------------------------------------------------------------------------
def test_distribution_shift_ks_zero_for_identical_distributions():
    X_same, _ = _synthetic_frame(200, 0, loc_shift=0.0, random_state=99)
    rows = distribution_shift(X_same, X_same.copy(), columns=INTERSECTION_COLUMNS)
    for r in rows:
        assert r["ks_statistic"] == pytest.approx(0.0, abs=1e-9)


def test_distribution_shift_detects_a_shifted_column_and_ranks_descending(dataset_a, dataset_b):
    X_a, _ = dataset_a
    X_b, _ = dataset_b
    rows = distribution_shift(X_a, X_b, columns=INTERSECTION_COLUMNS)

    assert len(rows) == len(INTERSECTION_COLUMNS)
    ks_values = [r["ks_statistic"] for r in rows]
    assert ks_values == sorted(ks_values, reverse=True)
    # dataset_b's loc_shift=1.5 vs dataset_a's 0.0 must show up as a > 0 shift.
    assert rows[0]["ks_statistic"] > 0.0


def test_plot_distribution_shift_writes_a_file(tmp_path, dataset_a, dataset_b):
    X_a, _ = dataset_a
    X_b, _ = dataset_b
    out_path = tmp_path / "shift.png"

    plot_distribution_shift(X_a, X_b, columns=INTERSECTION_COLUMNS, path=out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_save_cross_dataset_result_writes_json(tmp_path):
    path = save_cross_dataset_result("unit_test", {"a": 1}, out_dir=tmp_path)
    assert path.exists()
    assert path.name == "cross_dataset_unit_test.json"


def test_valid_models_matches_all_four_project_models():
    assert set(VALID_MODELS) == {"xgboost", "isolation_forest", "cnn", "autoencoder"}
