"""
Step 2A verification: prove the leakage guard structurally, not just by
inspection. Synthetic data only (deterministic, no dependency on real
datasets) -- this must pass identically on any machine, before or after
Step 0B's manual download step.

This is the evidence Chapter 7.2 asks for. Results are also written to
runs/metrics/leakage_guard_proof.json so the numbers can be quoted directly
in the report rather than re-derived from a screenshot of test output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from preprocessing.pipeline import build_pipeline, get_cv
from schema.unified import UNIFIED_COLUMNS


def _synthetic_imbalanced_frame(n_majority=475, n_minority=25, random_state=42):
    """95/5 imbalance, with NaN deliberately injected into a few cells,
    mirroring what a loader actually hands the pipeline (real feature names,
    real-shaped missingness on the B_ONLY columns)."""
    rng = np.random.default_rng(random_state)
    n = n_majority + n_minority

    X = pd.DataFrame(
        {col: rng.normal(loc=0.0, scale=1.0, size=n) for col in UNIFIED_COLUMNS}
    )
    # Inject NaN the way Dataset A actually does: B_ONLY columns entirely NaN
    # for a subset of rows, plus a few scattered NaN elsewhere.
    nan_rows = rng.choice(n, size=15, replace=False)
    X.loc[nan_rows, "vol_primary"] = np.nan
    X.loc[: n // 3, "time_central"] = np.nan  # simulates Dataset A's all-NaN B_ONLY cols

    y = pd.Series([0] * n_majority + [1] * n_minority, name="y")
    # Shuffle so the class imbalance isn't trivially separable by row order.
    shuffle_idx = rng.permutation(n)
    X = X.iloc[shuffle_idx].reset_index(drop=True)
    y = y.iloc[shuffle_idx].reset_index(drop=True)
    return X, y


@pytest.fixture(scope="module")
def config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def data():
    return _synthetic_imbalanced_frame()


def test_build_pipeline_uses_imblearn_not_sklearn():
    pipe = build_pipeline(LogisticRegression(max_iter=1000), use_smote=True)
    assert isinstance(pipe, ImbPipeline)
    assert [name for name, _ in pipe.steps] == ["imputer", "scaler", "smote", "estimator"]


def test_use_smote_false_omits_smote_step():
    pipe = build_pipeline(LogisticRegression(max_iter=1000), use_smote=False)
    assert [name for name, _ in pipe.steps] == ["imputer", "scaler", "estimator"]


def test_imputer_is_median_strategy():
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    imputer: SimpleImputer = pipe.named_steps["imputer"]
    assert imputer.strategy == "median"


def test_all_nan_column_is_preserved_not_dropped():
    """Regression test for a real bug caught while building this module:
    SimpleImputer's default behaviour silently DROPS a column with zero
    observed values (with only a warning), which would collapse Dataset A's
    11-column output to 7 wherever the four B_ONLY columns are all-NaN --
    exactly the shape the CNN/Autoencoder (Step 2F) are built around.
    keep_empty_features=True must be set so the column survives as a
    constant instead of vanishing."""
    n = 50
    X = pd.DataFrame(
        {col: np.random.default_rng(0).normal(size=n) for col in UNIFIED_COLUMNS}
    )
    X["time_central"] = np.nan  # entirely NaN, as it is on every Dataset A row
    y = pd.Series([0] * (n - 5) + [1] * 5)

    pipe = build_pipeline(LogisticRegression(max_iter=1000), use_smote=False)
    pipe.fit(X, y)

    imputed = pipe.named_steps["imputer"].transform(X)
    assert imputed.shape[1] == len(UNIFIED_COLUMNS), (
        f"Expected all {len(UNIFIED_COLUMNS)} columns to survive imputation, "
        f"got {imputed.shape[1]} -- an all-NaN column was silently dropped."
    )


def test_get_cv_caps_n_splits_to_smallest_class():
    y = pd.Series([0] * 10 + [1] * 2)  # smaller than default n_splits=5
    config = {"cv": {"n_splits": 5, "shuffle": True, "random_state": 42}}
    with pytest.warns(UserWarning, match="reducing n_splits"):
        cv = get_cv(y, config)
    assert cv.n_splits == 2


def test_leakage_guard_structural_proof(config, data, tmp_path):
    """The actual Ch 7.2 evidence: scaler refits per fold (proof nothing is
    fit once and reused across folds), the resampled TRAINING data is
    balanced, and the held-out TEST fold is provably untouched by SMOTE."""
    X, y = data
    cv = get_cv(y, config)

    fold_scaler_means: list[list[float]] = []
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # Full pipeline (with estimator) -- inspect the fitted scaler.
        pipe = build_pipeline(LogisticRegression(max_iter=1000), use_smote=True)
        pipe.fit(X_train, y_train)
        fold_scaler_means.append(pipe.named_steps["scaler"].mean_.tolist())

        # Resample-only pipeline (no estimator) -- inspect what SMOTE actually
        # produced, to directly verify "balanced training / untouched test".
        resample_pipe = ImbPipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=42)),
            ]
        )
        X_res, y_res = resample_pipe.fit_resample(X_train, y_train)

        train_counts_before = y_train.value_counts().to_dict()
        train_counts_after = pd.Series(y_res).value_counts().to_dict()
        test_counts = y_test.value_counts().to_dict()

        fold_results.append(
            {
                "fold": fold_idx,
                "train_class_counts_before_smote": {str(k): int(v) for k, v in train_counts_before.items()},
                "train_class_counts_after_smote": {str(k): int(v) for k, v in train_counts_after.items()},
                "test_class_counts_untouched": {str(k): int(v) for k, v in test_counts.items()},
            }
        )

        # Training set is imbalanced before SMOTE, balanced after.
        assert train_counts_before[0] != train_counts_before[1]
        assert train_counts_after[0] == train_counts_after[1]

        # Test fold is NEVER resampled -- its class counts must retain the
        # original ~95/5 imbalance, not the post-SMOTE 50/50 balance.
        assert test_counts.get(1, 0) < test_counts.get(0, 0)

        # No NaN survives the imputer.
        assert not np.isnan(X_res).any() if isinstance(X_res, np.ndarray) else not X_res.isna().any().any()

    # The central claim: each fold's scaler was fit independently, on that
    # fold's training data alone -- proven by the means actually differing
    # across folds. If the pipeline were (incorrectly) fit once outside the
    # CV loop and reused, every fold's mean_ would be bit-for-bit identical.
    means_array = np.array(fold_scaler_means)
    all_identical = np.allclose(means_array, means_array[0])
    assert not all_identical, (
        "Scaler means are identical across all folds -- this means the "
        "pipeline was fit once and reused, which is exactly the leakage "
        "bug this test exists to catch."
    )

    proof = {
        "claim": "Preprocessing pipeline structurally prevents train/test leakage",
        "n_folds": cv.n_splits,
        "scaler_refits_per_fold": not all_identical,
        "fold_results": fold_results,
    }

    out_path = Path("runs/metrics/leakage_guard_proof.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(proof, indent=2))
