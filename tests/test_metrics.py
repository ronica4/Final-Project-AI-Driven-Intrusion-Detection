"""
Step 2A' verification: every metric checked against hand-derived values on a
synthetic, fully-known confusion matrix -- not cross-checked against sklearn
calling sklearn (that would test nothing). y_pred and y_proba are
deliberately DECOUPLED in the hand-computed test below: y_pred is chosen to
produce a clean, hand-countable confusion matrix, and y_proba is chosen
SEPARATELY to perfectly rank-separate the classes, giving exact AUC values
(1.0) by definition. This isolates each metric's correctness independently
rather than simulating a single realistic classifier.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from evaluation.metrics import (
    _compute_fold_metrics,
    _majority_baseline,
    evaluate,
    plot_confusion,
    save_metrics,
)


def test_compute_fold_metrics_matches_hand_computed_values():
    # 4 negatives (idx 0-3), 4 positives (idx 4-7).
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    # TN=2 (idx0,1), FP=2 (idx2,3), FN=2 (idx6,7), TP=2 (idx4,5) -- by hand:
    #   precision = TP/(TP+FP) = 2/4 = 0.5
    #   recall    = TP/(TP+FN) = 2/4 = 0.5
    #   f1        = 2*0.5*0.5/(0.5+0.5) = 0.5
    #   fpr       = FP/(FP+TN) = 2/4 = 0.5
    y_pred = [0, 0, 1, 1, 1, 1, 0, 0]
    # Deliberately independent of y_pred: perfectly rank-separates the two
    # classes (every positive scores higher than every negative) -> both
    # ROC-AUC and PR-AUC are exactly 1.0 by definition of a perfect ranking.
    y_proba = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]

    result = _compute_fold_metrics(y_true, y_pred, y_proba)

    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)
    assert result["fpr"] == pytest.approx(0.5)
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["pr_auc"] == pytest.approx(1.0)
    assert result["confusion_matrix"] == [[2, 2], [2, 2]]


def test_majority_baseline_matches_hand_computed_value():
    # 4 positives, 1 negative -> majority_class = 1 (positive_rate = 0.8).
    # always-positive: TP=4, FP=1, FN=0 -> precision=4/5, recall=1
    #   f1 = 2*(4/5)*1 / ((4/5)+1) = (8/5)/(9/5) = 8/9
    y = [1, 1, 1, 1, 0]
    result = _majority_baseline(y)

    assert result["majority_class"] == 1
    assert result["always_positive_f1"] == pytest.approx(8 / 9)
    assert result["always_negative_f1"] == pytest.approx(0.0)
    assert result["f1"] == pytest.approx(8 / 9)  # majority_class==1 -> uses always_positive


def test_majority_baseline_picks_negative_when_majority_is_negative():
    y = [0, 0, 0, 0, 1]  # positive_rate = 0.2 -> majority_class = 0
    result = _majority_baseline(y)
    assert result["majority_class"] == 0
    assert result["f1"] == result["always_negative_f1"]


def _synthetic_classification_frame(n=200, n_features=11, random_state=42):
    rng = np.random.default_rng(random_state)
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    # Make the label weakly dependent on f0 so the classifier has *something*
    # to learn, but not perfectly separable -- realistic enough to exercise
    # every code path without being a trivial degenerate case.
    y = pd.Series((X["f0"] + rng.normal(scale=1.5, size=n) > 0).astype(int))
    return X, y


def test_evaluate_end_to_end_structure_and_self_consistency():
    X, y = _synthetic_classification_frame()
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=1000)
    meta = {"dataset_name": "synthetic", "framing": "n/a"}

    result = evaluate(model, X, y, cv, meta, families="full")

    # Schema shape.
    assert result["dataset_name"] == "synthetic"
    assert result["framing"] == "n/a"
    assert result["families_mode"] == "full"
    assert result["n_folds"] == 4
    assert len(result["per_fold"]) == 4
    assert set(result["mean"]) == {"precision", "recall", "f1", "pr_auc", "roc_auc", "fpr"}

    # mean/std are genuinely the mean/std of the per-fold values (self-consistency).
    for key in ["precision", "recall", "f1", "pr_auc", "roc_auc", "fpr"]:
        fold_values = [f[key] for f in result["per_fold"]]
        assert result["mean"][key] == pytest.approx(np.mean(fold_values))
        assert result["std"][key] == pytest.approx(np.std(fold_values))

    # aggregate_confusion_matrix is exactly the sum of the per-fold matrices.
    expected_agg = np.zeros((2, 2), dtype=int)
    for f in result["per_fold"]:
        expected_agg += np.array(f["confusion_matrix"])
    assert result["aggregate_confusion_matrix"] == expected_agg.tolist()

    # majority_baseline matches calling the pure function directly on y.
    assert result["majority_baseline"] == _majority_baseline(y)


def test_evaluate_clones_model_no_state_leaks_across_folds():
    """If clone() were missing, refitting the SAME estimator object across
    folds could leak state (e.g. LogisticRegression's warm-start behaviour,
    or any estimator that accumulates state in fit()). This isn't easily
    observable for LogisticRegression directly, so we assert the more
    direct claim: the model passed in is never itself fitted (still raises
    NotFittedError-equivalent behaviour), proving evaluate() only ever
    fits clones."""
    from sklearn.exceptions import NotFittedError

    X, y = _synthetic_classification_frame()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=1000)
    meta = {"dataset_name": "synthetic", "framing": "n/a"}

    evaluate(model, X, y, cv, meta)

    with pytest.raises(NotFittedError):
        model.predict(X)


def test_save_metrics_round_trips(tmp_path):
    results = {"dataset_name": "synthetic", "mean": {"f1": 0.5}}
    path = save_metrics("unit_test_experiment", results, out_dir=tmp_path)
    assert path.exists()
    assert json.loads(path.read_text()) == results


def test_plot_confusion_writes_a_file(tmp_path):
    out_path = tmp_path / "cm.png"
    plot_confusion([[10, 2], [3, 15]], "Unit test confusion matrix", out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
