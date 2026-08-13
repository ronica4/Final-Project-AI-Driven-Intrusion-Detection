"""
Step 3A verification. Synthetic data only, hand-computed where possible --
same discipline as tests/test_metrics.py and tests/test_selection.py.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from evaluation.error_analysis import (
    analyse_errors,
    class_medians,
    cross_model_failure_overlap,
    nearest_class_by_features,
    out_of_fold_predictions,
    per_subclass_recall,
    save_error_analysis,
    top_confident_errors,
)


def test_per_subclass_recall_matches_hand_computed_values():
    # heavy: 4 positives, indices 0-3, predicted [1,1,1,0] -> 3/4 = 0.75
    # light: 5 positives, indices 4-8, predicted [1,0,0,0,0] -> 1/5 = 0.2
    # benign: 2 negatives, indices 9-10 -> excluded entirely (y_true=0 there)
    y_true = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    y_pred = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    attack_subclass = ["heavy"] * 4 + ["light"] * 5 + ["benign"] * 2

    result = per_subclass_recall(y_true, y_pred, attack_subclass)

    assert result["heavy"] == {"n": 4, "n_detected": 3, "recall": pytest.approx(0.75)}
    assert result["light"] == {"n": 5, "n_detected": 1, "recall": pytest.approx(0.2)}
    assert "benign" not in result


def test_top_confident_errors_hand_verified_ordering():
    # Rows 0,1 are false negatives (y_true=1, y_pred=0): row 0 has the
    # LOWEST proba (0.05) -> model was MOST confident it was benign ->
    # must rank first. Rows 2,3 are false positives (y_true=0, y_pred=1):
    # row 3 has the HIGHEST proba (0.95) -> must rank first.
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [0, 0, 1, 1, 1, 0]
    y_proba = [0.05, 0.20, 0.60, 0.95, 0.99, 0.01]
    X = pd.DataFrame({"f1": [10, 11, 12, 13, 14, 15]})

    result = top_confident_errors(y_true, y_pred, y_proba, X, n=2)

    assert [r["row_index"] for r in result["false_negatives"]] == [0, 1]
    assert [r["y_proba"] for r in result["false_negatives"]] == [pytest.approx(0.05), pytest.approx(0.20)]
    assert [r["row_index"] for r in result["false_positives"]] == [3, 2]
    assert [r["y_proba"] for r in result["false_positives"]] == [pytest.approx(0.95), pytest.approx(0.60)]
    # feature values carried through onto each error row
    assert result["false_negatives"][0]["f1"] == pytest.approx(10.0)


def test_top_confident_errors_respects_n_limit():
    y_true = [1] * 5
    y_pred = [0] * 5
    y_proba = [0.1, 0.2, 0.3, 0.4, 0.5]
    X = pd.DataFrame({"f1": range(5)})
    result = top_confident_errors(y_true, y_pred, y_proba, X, n=2)
    assert len(result["false_negatives"]) == 2


def test_class_medians_matches_hand_computed_values():
    X = pd.DataFrame({"f1": [1, 2, 3, 100, 200, 300]})
    y = [0, 0, 0, 1, 1, 1]
    medians = class_medians(X, y)
    assert medians["f1"]["benign"] == pytest.approx(2.0)
    assert medians["f1"]["attack"] == pytest.approx(200.0)


def test_nearest_class_by_features_picks_closer_median_and_ties_to_benign():
    medians = {"f1": {"benign": 0.0, "attack": 10.0}, "f2": {"benign": 0.0, "attack": 10.0}}
    row = {"row_index": 0, "y_proba": 0.5, "f1": 1.0, "f2": 5.0}  # f2 is an exact tie
    verdict = nearest_class_by_features(row, medians)
    assert verdict["f1"] == "benign"  # closer to 0 than to 10
    assert verdict["f2"] == "benign"  # tie -> resolves to benign


def test_cross_model_failure_overlap_hand_computed():
    errors_by_model = {
        "model_a": {"false_negatives": {1, 2, 3, 4}, "false_positives": {10, 11}},
        "model_b": {"false_negatives": {3, 4, 5, 6}, "false_positives": {20, 21}},
    }
    result = cross_model_failure_overlap(errors_by_model)

    fn_row = next(p for p in result["pairwise"] if p["error_type"] == "false_negatives")
    # {1,2,3,4} vs {3,4,5,6} -> overlap {3,4} (n=2), union size 6 -> 2/6
    assert fn_row["n_overlap"] == 2
    assert fn_row["overlap_fraction_of_union"] == pytest.approx(2 / 6)

    fp_row = next(p for p in result["pairwise"] if p["error_type"] == "false_positives")
    # {10,11} vs {20,21} -> completely disjoint
    assert fp_row["n_overlap"] == 0
    assert fp_row["overlap_fraction_of_union"] == pytest.approx(0.0)


def test_out_of_fold_predictions_covers_every_row_with_no_leakage_reuse():
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series((X["f1"] + rng.normal(scale=0.5, size=n) > 0).astype(int))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=1000)

    y_pred, y_proba = out_of_fold_predictions(model, X, y, cv)

    assert not y_pred.isna().any()
    assert not y_proba.isna().any()
    assert list(y_pred.index) == list(X.index)
    assert set(y_pred.unique()) <= {0, 1}
    assert (y_proba.between(0, 1)).all()


def test_analyse_errors_end_to_end_structure_and_save(tmp_path):
    y_true = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    y_pred = pd.Series([1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    y_proba = pd.Series([0.9, 0.8, 0.7, 0.4, 0.6, 0.3, 0.2, 0.1, 0.05, 0.1, 0.2])
    X = pd.DataFrame({"f1": range(11)})
    meta = {
        "dataset_name": "synthetic",
        "framing": "n/a",
        "attack_subclass": pd.Series(["heavy"] * 4 + ["light"] * 5 + ["benign"] * 2),
    }

    result = analyse_errors("test_model", y_true, y_pred, y_proba, X, meta, n_top=5)

    assert result["model_name"] == "test_model"
    assert result["dataset_name"] == "synthetic"
    assert result["n_rows"] == 11
    assert result["per_subclass_recall"]["heavy"]["recall"] == pytest.approx(0.75)
    assert result["per_subclass_recall"]["light"]["recall"] == pytest.approx(0.2)
    assert len(result["false_negative_indices"]) == 5  # rows 3,5,6,7,8
    assert "nearest_class_per_feature" in result["top_errors"]["false_negatives"][0]

    path = save_error_analysis("test_model", "synthetic", result, out_dir=tmp_path)
    assert path.name == "error_analysis_test_model_synthetic.json"
    assert json.loads(path.read_text())["model_name"] == "test_model"
