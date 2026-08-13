"""
Preprocessing pipeline -- the structural guarantee against data leakage.

Every scaling, imputation, and resampling step lives inside a single
imblearn Pipeline, which is then handed whole to cross-validation. This
makes leakage structurally impossible rather than merely avoided by
discipline: StratifiedKFold refits the entire Pipeline from scratch inside
each fold, so the scaler's mean/std and SMOTE's synthetic samples are always
computed from that fold's training data alone, never from the fold's test
data or from data outside the fold entirely.

Dataset-agnostic by construction (Dataset Dependency Rule, PROJECT_PLAN.md
Step 0D): this module only ever sees the 11 unified schema columns. It has
no idea whether X came from Dataset A or Dataset B, and must never be
changed to find out.
"""

from __future__ import annotations

import warnings

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def build_pipeline(estimator, use_smote: bool = True, random_state: int = 42) -> ImbPipeline:
    """Impute -> scale -> (optionally) SMOTE -> estimator, as ONE fittable unit.

    MUST use imblearn.pipeline.Pipeline, never sklearn.pipeline.Pipeline.
    sklearn's Pipeline does not understand resamplers (steps that change the
    number of rows) -- handed a SMOTE step, it would silently apply it at
    *transform* time on the test fold too, synthesizing fake test-fold
    minority samples and inflating every metric computed against it. This
    is a leakage bug that produces no error, no warning, and a plausible-
    looking score -- exactly the kind that survives to the final report
    undetected. imblearn's Pipeline is resampler-aware: it applies SMOTE
    during fit() only, and is a pure passthrough at predict()/transform()
    time.

    use_smote=False is the correct choice, not a fallback, in two cases:
      - The Autoencoder (Step 2F): it trains on benign rows only, so there
        is no minority class within its training data for SMOTE to act on.
        Passing use_smote=True here would either error or synthesize
        nonsense from a single-class fit.
      - XGBoost when using scale_pos_weight instead (Step 2D): cost-sensitive
        reweighting and synthetic oversampling are two different, largely
        redundant answers to the same imbalance problem. Chapter 7.2 asks
        for a defended choice, not a reflexive "always SMOTE" -- the honest
        answer is "we used both, and here is when each applies."

    Median imputation, not mean: the volume features (vol_primary, vol_total,
    ...) are heavily right-skewed by the heavy-exfiltration tail, and a mean
    imputed from a skewed distribution is itself skewed. Median is the
    robust choice here (Chapter 5.3).

    keep_empty_features=True is required, not optional: on Dataset A, the
    four B_ONLY columns (time_central, time_dispersion, time_skew,
    disp_uniqueness) are 100% NaN by design (D2 -- unobservable in stateless
    DNS telemetry). SimpleImputer's default behaviour for a column with zero
    observed values is to DROP it silently and warn -- confirmed empirically:
    without this flag, a Dataset-A fold goes in with 11 columns and comes out
    with 7. That collapses the fixed-width input the CNN and Autoencoder
    (Step 2F) are built around, and does so without raising, so it would
    surface three steps downstream as a confusing shape mismatch rather than
    here where the cause is obvious. With the flag, an all-NaN column is
    imputed to a constant (0, since no other fill_value is given), which
    StandardScaler then leaves at a constant 0 too (zero-variance features
    get scale_=1, avoiding a divide-by-zero) -- i.e. exactly the "these four
    features carry no signal on Dataset A" behaviour the schema intends,
    made structurally true instead of merely documented.
    """
    steps = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("estimator", estimator))
    return ImbPipeline(steps)


def get_cv(y: pd.Series, config: dict) -> StratifiedKFold:
    """StratifiedKFold sized from config, defensively capped to the smallest
    class actually present in y.

    Takes y (not just config) because StratifiedKFold requires n_splits <=
    the smallest class count or it raises an opaque sklearn error at
    .split() time. On real data (light-attack alone is tens of thousands of
    rows) this cap never engages; it exists so a tiny synthetic test frame,
    or an unexpectedly small class in a subsampled run, fails loudly and
    specifically here rather than obscurely three calls downstream.
    """
    cv_cfg = config["cv"]
    n_splits = cv_cfg["n_splits"]

    min_class_count = int(pd.Series(y).value_counts().min())
    if min_class_count < n_splits:
        warnings.warn(
            f"n_splits={n_splits} exceeds the smallest class count "
            f"({min_class_count}); reducing n_splits to {min_class_count} "
            f"for this call. This should never happen on real data -- if "
            f"you see this warning outside a unit test, something upstream "
            f"produced an unexpectedly small class.",
            stacklevel=2,
        )
        n_splits = max(2, min_class_count)

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=cv_cfg["shuffle"],
        random_state=cv_cfg["random_state"],
    )
