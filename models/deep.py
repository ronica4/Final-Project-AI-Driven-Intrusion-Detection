"""
Step 2F -- 1D-CNN and Autoencoder, the project's deep detectors (PyTorch CPU per D10).

Dataset-agnostic by construction (Dataset Dependency Rule): every function
here takes (X, y, config, ...) already produced by a loader. Nothing in this
module knows or asks which dataset it is scoring.

Family order matters for the CNN specifically: schema.unified.UNIFIED_COLUMNS
is fixed in family order (F1 volume, F2 randomness, F3 structure, F4 timing,
F5 dispersion) precisely so that a kernel-3 convolution sliding across the
vector sees physically adjacent, behaviourally related features -- "high
volume AND high randomness" is learnable as one local pattern only because
those two families sit next to each other in the vector, not because of
anything the model does on its own. That is the Ch 6.1 structural
justification for using a CNN on an 11-wide tabular vector at all.

use_smote=True for the CNN (the Pipeline's default, preprocessing/pipeline.py):
unlike XGBoost (has scale_pos_weight) or Isolation Forest (structurally
cannot accept any resampling -- see models/unsupervised.py), the CNN has no
native reweighting knob wired up here, so SMOTE is the correct choice, not a
special case that needs its own justification.

use_smote=False for the Autoencoder, same reasoning as Isolation Forest but
sharper: the Autoencoder is fit EXCLUSIVELY on benign training-fold rows (see
AutoencoderDetector.fit below), so by the time SMOTE would run there is no
minority class left in its training data to resample -- passing
use_smote=True here would either error or synthesize nonsense from a
single-class fit. This is also the leakage trap this model is built around:
the reconstruction-error threshold is the 95th percentile of BENIGN
TRAINING-FOLD reconstruction error, computed inside AutoencoderDetector.fit
so it is a property of that fold's training split alone, never of the test
fold or of the full dataset.

Both estimators are built dynamically from the actual input width observed
at fit() time (X.shape[1]), not a hardcoded 11: Step 2F's headline runs use
families="full" (11 columns, matching Step 2D/2E's convention so every model
in the Ch 8 comparison table shares one input shape per dataset), but
run_cnn/run_autoencoder accept families="intersection"/"F1_only" too (used
by Step 2G's transfer/ablation experiments), and a hardcoded 11-wide first
layer would silently break on a 7- or 3-column input instead of adapting to
it. The Autoencoder's layers config ([11, 8, 4, 8, 11]) is read as "compress
to this sequence of hidden bottleneck widths," with the first/last entries
replaced by the real input width -- the intended compression ratio survives,
the literal "11" does not.

Both wrappers subclass BaseEstimator (not nn.Module) for the same reason
IsolationForestDetector does (models/unsupervised.py): get_params()/
set_params() derived from __init__'s signature is what sklearn.base.clone()
relies on for every fold in evaluation.metrics.evaluate().
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless -- this module writes files, never shows a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from evaluation.metrics import evaluate
from preprocessing.pipeline import build_pipeline, get_cv
from schema.unified import B_ONLY_COLUMNS, project


# ---------------------------------------------------------------------------
# 1D-CNN
# ---------------------------------------------------------------------------
class _CNNModule(nn.Module):
    """Conv block x2 (BatchNorm + ReLU after each) -> dropout -> dense -> dense(1) -> sigmoid.
    "same" padding at stride 1 keeps the sequence length fixed at n_features
    through both conv blocks, so the flattened width is conv_filters[-1] * n_features
    regardless of how many features this particular families= mode produced."""

    def __init__(
        self,
        n_features: int,
        conv_filters: tuple[int, int],
        kernel_size: int,
        padding: str,
        dropout: float,
        dense_units: int,
    ) -> None:
        super().__init__()
        f1, f2 = conv_filters
        self.conv1 = nn.Conv1d(1, f1, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(f1)
        self.conv2 = nn.Conv1d(f1, f2, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(f2)
        self.dropout = nn.Dropout(dropout)
        self.flatten_dim = f2 * n_features
        self.dense1 = nn.Linear(self.flatten_dim, dense_units)
        self.dense2 = nn.Linear(dense_units, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.dense1(x))
        return self.sigmoid(self.dense2(x))


class CNNClassifier(BaseEstimator):
    """sklearn-compatible adapter around _CNNModule: exposes fit()/predict()/
    predict_proba() so it drops into evaluate() and build_pipeline() exactly
    like every other model in this project.

    Early stopping on validation PR-AUC (patience configurable) requires an
    internal train/validation split, carved from whatever (X, y) fit()
    receives. Because this estimator sits as the LAST step of the Pipeline
    (preprocessing/pipeline.py), and SMOTE (if enabled) runs as the step
    immediately before it, fit() necessarily receives already-resampled data
    when use_smote=True -- there is no way for the final Pipeline step to
    "reach back" to pre-SMOTE data. This means the internal validation split
    used for early stopping may contain synthetic minority rows. Documented
    here explicitly rather than hidden: it does NOT leak into any reported
    metric (evaluate()'s CV test folds are computed entirely outside this
    class, on real held-out rows only) -- it only affects which epoch's
    weights get selected as "best," a training-time decision, not a scoring
    one.
    """

    def __init__(
        self,
        conv_filters: tuple[int, int] = (32, 64),
        kernel_size: int = 3,
        padding: str = "same",
        dropout: float = 0.3,
        dense_units: int = 64,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        early_stopping_patience: int = 5,
        max_epochs: int = 100,
        val_frac: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.dropout = dropout
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.max_epochs = max_epochs
        self.val_frac = val_frac
        self.random_state = random_state

    def fit(self, X, y) -> "CNNClassifier":
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=self.val_frac, stratify=y, random_state=self.random_state
        )

        n_features = X.shape[1]
        self.model_ = _CNNModule(
            n_features=n_features,
            conv_filters=self.conv_filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            dropout=self.dropout,
            dense_units=self.dense_units,
        )
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = nn.BCELoss()

        train_ds = TensorDataset(
            torch.from_numpy(X_tr).unsqueeze(1), torch.from_numpy(y_tr)
        )
        gen = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, generator=gen
        )
        X_val_t = torch.from_numpy(X_val).unsqueeze(1)

        best_val_pr_auc = -np.inf
        best_state = None
        epochs_no_improve = 0
        history: list[dict[str, Any]] = []

        for epoch in range(self.max_epochs):
            self.model_.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model_(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_pred = self.model_(X_val_t).squeeze(-1).numpy()
            val_pr_auc = float(average_precision_score(y_val, val_pred))
            history.append({"epoch": epoch, "val_pr_auc": val_pr_auc})

            if val_pr_auc > best_val_pr_auc + 1e-6:
                best_val_pr_auc = val_pr_auc
                best_state = copy.deepcopy(self.model_.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.early_stopping_patience:
                    break

        self.model_.load_state_dict(best_state)
        self.training_history_ = history
        self.best_val_pr_auc_ = best_val_pr_auc
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            p = self.model_(torch.from_numpy(X).unsqueeze(1)).squeeze(-1).numpy()
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_cnn(config: dict) -> CNNClassifier:
    """Every hyperparameter read from config.yaml -- none hardcoded here
    (PROJECT_PLAN.md Step 0D / Ch 7.3 rubric item: no library defaults)."""
    cfg = config["models"]["cnn"]
    return CNNClassifier(
        conv_filters=tuple(cfg["conv_filters"]),
        kernel_size=cfg["kernel_size"],
        padding=cfg["padding"],
        dropout=cfg["dropout"],
        dense_units=cfg["dense_units"],
        learning_rate=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        early_stopping_patience=cfg["early_stopping_patience"],
        max_epochs=cfg["max_epochs"],
        val_frac=cfg["val_frac"],
        random_state=cfg["random_state"],
    )


def run_cnn(
    X: pd.DataFrame, y: pd.Series, config: dict, meta: dict, families: str = "full"
) -> dict[str, Any]:
    """One full stratified-CV evaluation of the CNN on (X, y), scored via the
    shared 2A' harness (evaluation.metrics.evaluate) -- no bespoke scoring.
    use_smote=True (module docstring)."""
    X_proj = project(X, families)
    cv = get_cv(y, config)
    model = build_pipeline(build_cnn(config), use_smote=True)

    result = evaluate(model, X_proj, y, cv, meta, families=families)
    if families == "full" and set(B_ONLY_COLUMNS) & set(X_proj.columns):
        result["b_only_columns_all_nan"] = bool(X_proj[B_ONLY_COLUMNS].isna().all().all())
    return result


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------
class _AEModule(nn.Module):
    """Symmetric encoder/decoder built from `layers` (e.g. [11, 8, 4, 8, 11]).
    ReLU between every layer except the final reconstruction output, which
    stays linear -- the input is standardized (StandardScaler, mean 0) so a
    bounded activation like sigmoid would clip negative feature values."""

    def __init__(self, layers: list[int]) -> None:
        super().__init__()
        dims = list(zip(layers[:-1], layers[1:]))
        modules: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(dims):
            modules.append(nn.Linear(in_dim, out_dim))
            if i < len(dims) - 1:
                modules.append(nn.ReLU())
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AutoencoderDetector(BaseEstimator):
    """sklearn-compatible adapter around _AEModule. Fits on benign
    training-fold rows only (module docstring); decision_function() returns
    the raw reconstruction error, already higher-is-more-attack (matching
    evaluation.metrics._positive_proba()'s convention) with no sign flip
    needed, unlike IsolationForestDetector. Has no predict_proba, same as
    IsolationForestDetector -- _positive_proba() falls back to
    decision_function() for both.
    """

    def __init__(
        self,
        layers: tuple[int, ...] = (11, 8, 4, 8, 11),
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 100,
        threshold_percentile: float = 95,
        random_state: int = 42,
    ) -> None:
        self.layers = layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state

    def fit(self, X, y) -> "AutoencoderDetector":
        # y IS used here (unlike IsolationForestDetector) -- selecting the
        # benign-only training subset is the entire point of this model; see
        # module docstring for why that is not a leakage risk (it is a
        # property of the training fold's own labels, computed before any
        # test-fold data is touched).
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        X_benign = X[y == 0]

        n_features = X.shape[1]
        real_layers = [n_features, *list(self.layers[1:-1]), n_features]
        self.model_ = _AEModule(real_layers)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        ds = TensorDataset(torch.from_numpy(X_benign))
        gen = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, generator=gen)

        history: list[dict[str, Any]] = []
        self.model_.train()
        for epoch in range(self.max_epochs):
            epoch_loss, n_seen = 0.0, 0
            for (xb,) in loader:
                optimizer.zero_grad()
                recon = self.model_(xb)
                loss = loss_fn(recon, xb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
                n_seen += xb.size(0)
            history.append({"epoch": epoch, "train_mse": epoch_loss / n_seen})
        self.training_history_ = history

        # Threshold computed from benign TRAINING-fold reconstruction error
        # only -- never from test-fold data (the leakage trap this model is
        # built around, PROJECT_PLAN.md Step 2F point 2).
        self.model_.eval()
        with torch.no_grad():
            recon_benign = self.model_(torch.from_numpy(X_benign)).numpy()
        benign_errors = np.mean((X_benign - recon_benign) ** 2, axis=1)
        self.threshold_ = float(np.percentile(benign_errors, self.threshold_percentile))
        return self

    def _reconstruction_error(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            recon = self.model_(torch.from_numpy(X)).numpy()
        return np.mean((X - recon) ** 2, axis=1)

    def decision_function(self, X) -> np.ndarray:
        return self._reconstruction_error(X)

    def predict(self, X) -> np.ndarray:
        return (self._reconstruction_error(X) > self.threshold_).astype(int)


def build_autoencoder(config: dict) -> AutoencoderDetector:
    """Every hyperparameter read from config.yaml -- none hardcoded here
    (PROJECT_PLAN.md Step 0D / Ch 7.3 rubric item: no library defaults)."""
    cfg = config["models"]["autoencoder"]
    return AutoencoderDetector(
        layers=tuple(cfg["layers"]),
        learning_rate=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        max_epochs=cfg["max_epochs"],
        threshold_percentile=cfg["threshold_percentile"],
        random_state=cfg["random_state"],
    )


def run_autoencoder(
    X: pd.DataFrame, y: pd.Series, config: dict, meta: dict, families: str = "full"
) -> dict[str, Any]:
    """One full stratified-CV evaluation of the Autoencoder on (X, y), scored
    via the shared 2A' harness (evaluation.metrics.evaluate) -- no bespoke
    scoring. use_smote=False throughout (module docstring)."""
    X_proj = project(X, families)
    cv = get_cv(y, config)
    model = build_pipeline(build_autoencoder(config), use_smote=False)

    result = evaluate(model, X_proj, y, cv, meta, families=families)
    if families == "full" and set(B_ONLY_COLUMNS) & set(X_proj.columns):
        result["b_only_columns_all_nan"] = bool(X_proj[B_ONLY_COLUMNS].isna().all().all())
    return result


# ---------------------------------------------------------------------------
# Training curves (Step 2F verification: "training curves saved")
# ---------------------------------------------------------------------------
def fit_full_for_training_curve(
    X: pd.DataFrame, y: pd.Series, config: dict, model_name: str, families: str = "full"
):
    """One-off fit on the FULL projected dataset, purely to produce a curve
    to plot for the report. Deliberately separate from run_cnn/run_autoencoder:
    evaluate() clones and refits a fresh estimator inside every CV fold and
    never exposes a fold's internal training_history_, so a single full-data
    fit is the only way to get a curve at all -- it feeds no reported metric.
    Returns the fitted estimator's training_history_.
    """
    X_proj = project(X, families)
    if model_name == "cnn":
        pipe = build_pipeline(build_cnn(config), use_smote=True)
    elif model_name == "autoencoder":
        pipe = build_pipeline(build_autoencoder(config), use_smote=False)
    else:
        raise ValueError(f"Unknown model_name {model_name!r}; expected 'cnn' or 'autoencoder'")
    pipe.fit(X_proj, y)
    return pipe.named_steps["estimator"].training_history_


def plot_training_curve(
    history: list[dict[str, Any]],
    metric_key: str,
    metric_label: str,
    model_label: str,
    dataset_label: str,
    path: str | Path,
) -> None:
    """One consistent single-axis training-curve style, shared by the CNN's
    val_pr_auc-per-epoch curve and the Autoencoder's train_mse-per-epoch
    curve."""
    epochs = [h["epoch"] for h in history]
    values = [h[metric_key] for h in history]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, values, marker="o", color="tab:blue")
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{model_label} training curve -- {dataset_label}")
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
