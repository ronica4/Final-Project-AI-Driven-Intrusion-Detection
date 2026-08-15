"""
Step 3C verification. Synthetic data only, same discipline as every other
tests/test_*.py -- deterministic, no dependency on real datasets.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from ensemble.cascade import individual_model_metrics_on_same_split, run_cascade, save_cascade_result
from schema.unified import B_ONLY_COLUMNS, UNIFIED_COLUMNS


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


def _synthetic_frame(n_majority, n_minority, random_state=42):
    rng = np.random.default_rng(random_state)
    n = n_majority + n_minority

    X = pd.DataFrame({col: rng.normal(size=n) for col in UNIFIED_COLUMNS})
    for col in B_ONLY_COLUMNS:
        X[col] = np.nan

    signal = X["vol_primary"] + rng.normal(scale=1.5, size=n)
    y = pd.Series((signal > np.quantile(signal, 1 - n_minority / n)).astype(int))

    shuffle_idx = rng.permutation(n)
    X = X.iloc[shuffle_idx].reset_index(drop=True)
    y = y.iloc[shuffle_idx].reset_index(drop=True)
    return X, y


@pytest.fixture
def synthetic_dataset():
    return _synthetic_frame(400, 100, random_state=7)


def test_run_cascade_funnel_accounts_for_every_row(fast_config, synthetic_dataset):
    X, y = synthetic_dataset
    meta = {"dataset_name": "synthetic", "framing": "n/a"}

    result = run_cascade(X, y, fast_config, meta, families="full", test_frac=0.3)

    funnel = result["funnel"]
    n_test = result["n_test"]
    assert funnel["stage1_isolation_forest"]["rows_in"] == n_test
    assert (
        funnel["stage1_isolation_forest"]["rows_discarded"] + funnel["stage1_isolation_forest"]["rows_passed"]
        == n_test
    )
    assert funnel["stage2_xgboost"]["rows_in"] == funnel["stage1_isolation_forest"]["rows_passed"]
    assert funnel["stage3_escalation"]["rows_in"] == funnel["stage2_xgboost"]["rows_in"]
    assert (
        funnel["stage2_xgboost"]["rows_resolved_confident"] + funnel["stage3_escalation"]["rows_escalated"]
        == funnel["stage2_xgboost"]["rows_in"]
    )


def test_run_cascade_escalation_is_subset_of_survivors(fast_config, synthetic_dataset):
    X, y = synthetic_dataset
    meta = {"dataset_name": "synthetic", "framing": "n/a"}

    result = run_cascade(X, y, fast_config, meta, families="full", test_frac=0.3)
    stage3 = result["funnel"]["stage3_escalation"]

    assert 0 <= stage3["rows_escalated"] <= stage3["rows_in"]
    assert stage3["escalated_in_band"] <= stage3["rows_escalated"]
    assert stage3["escalated_by_disagreement"] <= stage3["rows_escalated"]
    assert stage3["escalated_by_both"] <= min(stage3["escalated_in_band"], stage3["escalated_by_disagreement"])


def test_run_cascade_metrics_and_confusion_matrix_present(fast_config, synthetic_dataset):
    X, y = synthetic_dataset
    meta = {"dataset_name": "synthetic", "framing": "n/a"}

    result = run_cascade(X, y, fast_config, meta, families="full", test_frac=0.3)

    assert set(result["cascade_metrics"]) == {"precision", "recall", "f1", "pr_auc", "roc_auc", "fpr"}
    for v in result["cascade_metrics"].values():
        assert 0.0 <= v <= 1.0
    assert "majority_baseline" in result


def test_run_cascade_respects_explicit_stage1_contamination(fast_config, synthetic_dataset):
    X, y = synthetic_dataset
    meta = {"dataset_name": "synthetic", "framing": "n/a"}

    result = run_cascade(X, y, fast_config, meta, families="full", stage1_contamination=0.2, test_frac=0.3)
    assert result["stage1_contamination"] == pytest.approx(0.2)


def test_individual_model_metrics_on_same_split_matches_cascade_test_rows(fast_config, synthetic_dataset):
    X, y = synthetic_dataset
    meta = {"dataset_name": "synthetic", "framing": "n/a"}

    result = run_cascade(X, y, fast_config, meta, families="full", test_frac=0.3)
    individual = individual_model_metrics_on_same_split(result)

    assert set(individual) == {"iso", "xgb", "ae"}
    for m in individual.values():
        assert set(m) == {"precision", "recall", "f1", "pr_auc", "roc_auc", "fpr"}


def test_save_cascade_result_strips_unserialisable_payload(tmp_path, fast_config, synthetic_dataset):
    X, y = synthetic_dataset
    meta = {"dataset_name": "synthetic", "framing": "n/a"}

    result = run_cascade(X, y, fast_config, meta, families="full", test_frac=0.3)
    path = save_cascade_result("unit_test", result, out_dir=tmp_path)

    assert path.exists()
    assert path.name == "cascade_unit_test.json"
    import json

    saved = json.loads(path.read_text())
    assert "_fitted" not in saved
    assert "_test_split" not in saved
    assert "funnel" in saved
