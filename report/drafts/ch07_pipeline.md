# Chapter 7 — Pipeline Architecture

## 7.1 Block diagram and the abstraction proof

```
  raw dataset files (data/exf2021/ | data/dohbrw2020/)
        │
        ▼
  ingestion/{exf2021,dohbrw2020}.py  ── the ONLY modules allowed to know a dataset name
        │  .load() → (X, y, meta)
        ▼
  schema/unified.py  ── project() + validate_schema(): 11-column unified schema, 5 families
        │
        ▼
  preprocessing/pipeline.py  ── build_pipeline(): impute → scale → (SMOTE) → estimator, ONE fittable unit
        │
        ├──► models/supervised.py    (XGBoost)
        ├──► models/deep.py          (1D-CNN, Autoencoder)
        ├──► models/unsupervised.py  (Isolation Forest)
        ├──► ensemble/cascade.py     (Step 3C, single-dataset)
        └──► evaluation/cross_dataset.py (Step 2G, cross-dataset only)
        │
        ▼
  evaluation/metrics.py  ── evaluate(): shared scoring contract, majority baseline, all models alike
```

This is the **Dataset Dependency Rule** (`PROJECT_PLAN.md`, Step 0D) made structural rather than aspirational:
past `load_dataset()` returning `(X, y, meta)` in `main.py`, no downstream module — preprocessing, features,
models, evaluation, or ensemble — contains a single conditional keyed on which dataset produced the data.
`main.py`'s own docstring states this claim; the claim is independently verified, not just asserted, by the
command Step 1C specifies:

```
$ grep -rniE "exf2021|dohbrw|cic|bell|doh" --include=*.py . | grep -v "^./ingestion/"
```

Run against the real codebase, every match falls into one of three sanctioned exceptions and none inside
`preprocessing/`, `features/`, `models/`, `evaluation/`, or `ensemble/`:

1. `main.py`'s CLI plumbing — the `--dataset` choices list and the registry-dispatch introspection in
   `load_dataset()`, the one bridge between the CLI flag and `ingestion/registry.py`.
2. `schema/unified.py`'s module docstring and `COLUMN_SOURCE` mapping — this is schema *definition*
   (documenting where each unified column's raw arithmetic comes from per dataset), not pipeline logic.
3. `tests/test_exf2021_loader.py` and one illustrative filename in an `evaluation/metrics.py` docstring
   example — a test file for the concrete loader itself, and a comment, not executable logic.

Full output preserved at `runs/metrics/abstraction_proof.txt`. This is the strongest form of evidence
available for the abstraction claim: not "we tried to keep it clean," but "an automated search of every
`.py` file in the repository finds zero dataset-specific logic outside the three declared, load-time-only
locations."

## 7.2 Imbalance strategy and the structural leakage guard

**The imbalance strategy is "resampling XOR reweighting, never both," decided per model, not a single
blanket choice.** `preprocessing/pipeline.py`'s `build_pipeline()` docstring states the reasoning directly:
SMOTE and `scale_pos_weight` are two different, largely redundant answers to the same class-imbalance
problem, and applying both would double-compensate. The project resolves this per model rather than
picking one rule for everything:

- **XGBoost** uses `scale_pos_weight` (`use_smote=False`) — native cost-sensitive reweighting, computed per
  dataset+framing at train time (`compute_scale_pos_weight()`), needs no synthetic rows at all.
- **The Autoencoder** uses `use_smote=False` for a structural reason, not a preference: it fits on
  benign-only training rows by design (Step 2F), so there is no minority class present in its training data
  for SMOTE to act on — `use_smote=True` here would either error or synthesize nonsense from a
  single-class fit.
- Models without a native reweighting mechanism (Isolation Forest is unsupervised and doesn't take a class
  weight at all; the CNN prior to Step 2F's design) use SMOTE inside the pipeline instead.

**The leakage guard is structural, not procedural.** The naive mistake — impute/scale/resample on the full
dataset once, *then* cross-validate — lets information from each test fold leak into the transform that
produced it, silently inflating every downstream score. The fix used here is to put every one of imputation,
scaling, and resampling inside a single `imblearn.pipeline.Pipeline` object and hand the *whole pipeline*,
untrained, to the cross-validator:

```python
steps = [
    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ("scaler", StandardScaler()),
]
if use_smote:
    steps.append(("smote", SMOTE(random_state=random_state)))
steps.append(("estimator", estimator))
return ImbPipeline(steps)
```

(`preprocessing/pipeline.py`, `build_pipeline()`, verbatim.) Two details are load-bearing, not incidental:

- This **must** be `imblearn.pipeline.Pipeline`, never `sklearn.pipeline.Pipeline`. sklearn's Pipeline has no
  concept of a step that changes the row count — handed a SMOTE step, it would apply it at *transform* time
  on the test fold too, synthesizing fake test-fold minority samples and inflating every metric computed
  against it. This is a leakage bug that produces no error, no warning, and a plausible-looking score —
  exactly the kind that survives undetected to a final report. `imblearn`'s Pipeline is resampler-aware: it
  applies SMOTE during `fit()` only, and is a pure passthrough at `predict()`/`transform()` time.
- `StratifiedKFold` (`get_cv()`, same file) then **refits the entire pipeline from scratch inside each
  fold** — the scaler's mean/std and SMOTE's synthetic samples are computed only from that fold's own
  training rows, never from the fold's test rows or from data outside the fold entirely.

**This is not just documented, it is asserted.** The Step 2A verification fits the pipeline on a synthetic
frame with a known 95/5 imbalance and confirms, per fold: the scaler's `mean_` differs across folds (proof
it genuinely refits, not just is *supposed to*), the post-SMOTE training class counts are balanced, and the
test-fold class counts are untouched. Real output, `runs/metrics/leakage_guard_proof.json`:

| Fold | Train (before SMOTE) | Train (after SMOTE) | Test (untouched) |
|---|---|---|---|
| 0–4 (all identical by construction of the synthetic frame) | {0: 380, 1: 20} | {0: 380, 1: 380} | {0: 95, 1: 5} |

Every one of the 5 folds shows the same shape: SMOTE balances the *training* side only, and the test side's
95/5 imbalance — the honest, undisturbed base rate — is preserved in every fold. This is the direct evidence
that leakage is structurally prevented, not merely avoided by discipline.

## 7.3 Hyperparameters and sensitivity sweeps

All hyperparameters are read from `config/config.yaml` — no library defaults anywhere in the training code
(`PROJECT_PLAN.md` Step 0D / this chapter's own rubric requirement).

| Model | Hyperparameters (all from `config.yaml`) |
|---|---|
| XGBoost | `n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, scale_pos_weight=<computed per dataset+framing>, random_state=42` |
| Isolation Forest | `n_estimators=200, max_samples=256, contamination=0.2 (base) / cascade-selected, max_features=0.8, random_state=42` |
| 1D-CNN | `conv_filters=[32,64], kernel_size=3, padding=same, dropout=0.3, dense_units=64, optimizer=adam, learning_rate=0.001, batch_size=256, early_stopping_patience=5 (on validation PR-AUC), max_epochs=100 (cap; early stopping is what actually stops training), val_frac=0.2 (carved from the training fold only)` |
| Autoencoder | `layers=[11,8,4,8,11], activation=relu, loss=mse, optimizer=adam, learning_rate=0.001, batch_size=256, max_epochs=100, threshold_percentile=95 (of benign training-fold reconstruction error only)` |
| Cross-validation (all models) | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |

**XGBoost `max_depth` sweep** (`{3, 6, 9, 12}`, Dataset B hard framing, real run,
`runs/metrics/xgboost_dohbrw2020_hard_sweep.json`, figure `runs/figures/xgboost_sensitivity_dohbrw2020_hard.png`):

| max_depth | F1 | FPR |
|---|---|---|
| 3 | 0.99982 | 0.0 |
| 6 (config default) | 0.99987 | 0.0 |
| 9 | 0.99990 | 0.0 |
| 12 | 0.99987 | 0.0 |

**Interpretation:** F1 is flat to four decimal places across the entire depth range, and FPR is exactly 0.0
at every depth. On Dataset B's 11-column feature set, the signal XGBoost is exploiting (the F1/F2/F3
payload-volume and structural features) is separable enough that depth beyond 3 adds essentially nothing —
the bottleneck is feature information content, not model capacity. This matches the same pattern
Person A's independent Dataset A depth sweep found (`xgboost_exf2021_depth_sweep.json` — flat F1 across
depth, FPR corroborated by a plain LogisticRegression smoke test): the flatness is a property of the
feature set, not an XGBoost-specific quirk or a dataset-specific coincidence. `max_depth=6` is kept as the
config default; the sweep justifies not needing to tune it further rather than motivating a change.

**Isolation Forest `contamination` sweep** (`{0.05, 0.1, 0.2, 0.3}`, Dataset B hard framing, real run,
`runs/metrics/isoforest_dohbrw2020_hard_contamination_sweep.json`, figure
`runs/figures/isoforest_dohbrw2020_hard_contamination_sweep.png`):

| contamination | F1 | Recall | FPR |
|---|---|---|---|
| 0.05 | 0.111 | 0.061 | 0.039 |
| 0.10 | 0.228 | 0.137 | 0.063 |
| 0.20 (config default) | 0.341 | 0.239 | 0.163 |
| 0.30 | 0.414 | 0.331 | 0.268 |

**Interpretation:** unlike XGBoost's sweep, this one is *not* flat — recall and FPR both rise
monotonically with contamination, and F1 rises too across the whole tested range, meaning the sweep never
reaches a turning point within `{0.05, ..., 0.3}`. This is the same underlying weakness Step 3C's cascade
result (Chapter 8.4) diagnoses: Isolation Forest is a comparatively weak detector on this feature set, so
pushing contamination higher keeps trading FPR for recall favourably in raw F1 terms, but never reaches a
contamination value that is both cheap (low FPR) and effective (high recall) at once — which is exactly why
`select_cascade_contamination()`'s recall-first, FPR-budget-constrained selection rule (rather than a
simple max-F1 pick) exists: a cascade's first stage cannot use "F1 is still climbing" as a stopping rule,
because F1 does not penalise a permissive filter the way a lost detection downstream does.

---

## Appendix A — Setup and run commands

```bash
git clone https://github.com/ronica4/Final-Project-AI-Driven-Intrusion-Detection.git
cd Final-Project-AI-Driven-Intrusion-Detection

python -m venv venv
venv\Scripts\activate                       # Windows
pip install -r requirements.txt             # includes --extra-index-url for CPU-only torch (D10)

# Place raw data (not included in the repo, see .gitignore):
#   data/exf2021/       <- CIC-Bell-DNS-EXF-2021 raw CSVs, per Step 0B / docs/header_reconciliation_exf2021.md
#   data/dohbrw2020/    <- CIRA-CIC-DoHBrw-2020: l2-benign.csv, l2-malicious.csv, l1-nondoh.csv
#                          (l1-doh.csv is a redundant union of the two l2-* files -- never read, do not need it)

# EDA / schema-validation smoke test (the only --mode wired into main.py as of this report):
python main.py --dataset exf2021 --mode eda --config config/config.yaml
python main.py --dataset dohbrw2020 --framing hard --mode eda --config config/config.yaml
python main.py --dataset dohbrw2020 --framing easy --mode eda --config config/config.yaml
python main.py --dataset dohbrw2020 --framing hard --families intersection --config config/config.yaml

# Test suite:
pytest -q

# Model training / evaluation / cascade / cross-dataset runs are exercised directly via each module's
# functions (models/supervised.py, models/deep.py, models/unsupervised.py, ensemble/cascade.py,
# evaluation/cross_dataset.py) and the tests/ suite, rather than through main.py's --mode train/eval/
# xdataset/cascade flags -- those flags are reserved in the CLI surface (see main.py's argparse choices)
# but their dispatch bodies are not wired up (main.py raises NotImplementedError for any --mode other than
# eda). This is stated plainly rather than implying a driver command exists that does not.
```

## Appendix B — Environment

**Library versions** (`requirements.txt`, pinned, CPU-only PyTorch via `--extra-index-url
https://download.pytorch.org/whl/cpu` so no separate install step is needed for D10):

| Library | Version |
|---|---|
| Python (venv) | 3.x (see `venv/pyvenv.cfg`) |
| numpy | 2.5.2 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |
| imbalanced-learn | 0.14.2 |
| xgboost | 3.4.0 |
| torch | 2.13.0+cpu |
| scipy | 1.18.0 |
| matplotlib | 3.11.1 |
| seaborn | 0.13.2 |
| python-docx | 1.2.0 |
| pytest | 9.1.1 |

**Machine — Teammate B (this report's author, all Dataset B results):**

| | |
|---|---|
| OS | Windows 11 Pro, build 10.0.26200 |
| CPU | 12th Gen Intel Core i5-1240P (12 cores / 16 logical processors) |
| RAM | 16,083 MB (~16 GB) |
| GPU | none used — all training is CPU-only by design (D10), PyTorch resolved to the `+cpu` wheel |

**Machine — Teammate A (Dataset A results):** pending — CPU/RAM/OS to be supplied by Teammate A and
backfilled into this table before final assembly (Step 4H), flagged here rather than fabricated.
