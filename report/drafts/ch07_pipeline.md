# Chapter 7 — Pipeline, Imbalance Strategy, and Hyperparameters

## 7.1 Block diagram and the abstraction proof

```
 CLI (--dataset, --framing, --mode, --families)
        |
        v
 registry.get(dataset_name)(config, framing=...)   <- the ONLY place a dataset name is named
        |
        v
 loader.load() -> (X, y, meta)                      <- ingestion/exf2021.py | ingestion/dohbrw2020.py
        |
        v
 schema.validate_schema(X, mode=families)           <- hard assert: 11 unified columns, correct dtypes
        |
        v
 =========== EVERYTHING BELOW THIS LINE IS DATASET-BLIND ===========
        |
        v
 preprocessing.build_pipeline(estimator, use_smote) <- impute -> scale -> [SMOTE] -> estimator, one
        |                                                imblearn.pipeline.Pipeline, refit per CV fold
        v
 models/{supervised,unsupervised,deep}.py           <- XGBoost | Isolation Forest | 1D-CNN | Autoencoder
        |
        v
 evaluation.metrics.evaluate()                      <- one scoring function, every model, both datasets
        |
        v
 ensemble/cascade.py (Step 3C, single-dataset deployment path: IsoForest -> XGBoost -> escalation)
```

The load boundary (`loader.load()` returning `(X, y, meta)`) is not just a diagram convention — it is
enforced structurally (the "Dataset Dependency Rule," `PROJECT_PLAN.md` Step 0D) and verified
empirically, not just claimed. Step 1C's abstraction proof runs

```
grep -rniE "exf2021|dohbrw|cic|bell|doh" --include=*.py . | grep -v "^./ingestion/"
```

against the full codebase and requires the output to be empty outside three sanctioned exceptions. The
saved output (`runs/metrics/abstraction_proof.txt`) confirms all matches fall into exactly those three
categories — `main.py`'s CLI plumbing (the `--dataset` choices list and the registry-dispatch
introspection, the one legitimate bridge between a CLI flag and `ingestion/registry.py`),
`schema/unified.py`'s module docstring and `COLUMN_SOURCE` mapping (schema *definition*, documenting
where each unified column's arithmetic comes from per dataset, not pipeline *logic*), and a test file for
the concrete Dataset A loader plus one illustrative filename in a comment — and states explicitly: **no
module under `preprocessing/`, `features/`, `models/`, `evaluation/`, or `ensemble/` contains any
conditional logic keyed on a dataset name.** This is what makes Chapter 5's transfer-matrix experiment
(train on one dataset, score on the other with zero retraining) a meaningful test of the behavioural
abstraction rather than a claim taken on faith — the same `models/unsupervised.py` and `ensemble/cascade.py`
that were built and unit-tested against synthetic data ran against real Dataset B with, per the Step 2E
and Step 3C result logs, "zero changes."

## 7.2 Imbalance strategy and the structural leakage guard

**Two different, deliberately non-overlapping answers to class imbalance, chosen per model rather than
applied uniformly** (`preprocessing/pipeline.py`, quoted below):

- **SMOTE** (synthetic minority oversampling) for the CNN, inside the Pipeline, fit-time only.
- **`scale_pos_weight`** (cost-sensitive reweighting) for XGBoost, computed once per dataset+framing from
  the full training `y` (`compute_scale_pos_weight()`, `models/supervised.py`).
- **Neither** for Isolation Forest (unsupervised — resampling a density estimator toward synthetic 1:1
  balance would teach it the attack class is part of the normal density, defeating its entire premise
  before it fits a single tree) or the Autoencoder (fit exclusively on benign rows — there is no minority
  class present in its training data for SMOTE to act on in the first place).

The rule enforced throughout the codebase is **resampling xor reweighting, never both** — using SMOTE
*and* `scale_pos_weight` on the same model would double-compensate for the same imbalance
(`models/supervised.py` Step 2D notes, `PROJECT_PLAN.md` Step 2A point 3). This is a defended choice
rather than a default: each of the four models gets the imbalance treatment that is actually coherent
with its own fitting mechanism, documented per-model in Chapter 6.1.

**The leakage guard is structural, not a matter of discipline.** The danger with any preprocessing step
that touches the whole dataset before the train/test split — a scaler fit on all rows, a resampler run
before folding — is that information from the test fold leaks into training, producing scores that look
real but are not reproducible against genuinely unseen data. The fix, `build_pipeline()`
(`preprocessing/pipeline.py`), makes this failure mode structurally unreachable rather than merely
avoided by convention:

```python
def build_pipeline(estimator, use_smote: bool = True, random_state: int = 42) -> ImbPipeline:
    steps = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("estimator", estimator))
    return ImbPipeline(steps)
```

Two details matter beyond the step list itself. First, **`imblearn.pipeline.Pipeline`, never
`sklearn.pipeline.Pipeline`** — sklearn's Pipeline has no concept of a resampling step (one that changes
row count) and would silently apply SMOTE at *transform* time too, synthesizing fake minority rows inside
the test fold and inflating every metric computed against it. This produces no error and no warning — a
leakage bug indistinguishable from a real result until someone re-derives the numbers by hand. Second,
this whole object — imputer, scaler, resampler, estimator — is handed to `StratifiedKFold` as one
fittable unit, which means the scaler's mean/std and SMOTE's synthetic samples are recomputed from
scratch inside every fold, using only that fold's training rows.

This claim is verified empirically, not just asserted from the code. Step 2A's leakage-guard proof
(`runs/metrics/leakage_guard_proof.json`) fits the pipeline on a synthetic 400/20 imbalanced frame across
5 folds and records, per fold: `train_class_counts_before_smote` (380 / 20), `train_class_counts_after_smote`
(380 / 380 — SMOTE balanced the training fold as intended), and `test_class_counts_untouched` (95 / 5 —
identical across all 5 folds, exactly the untouched holdout counts, never resampled). The test fold's
class counts never move, in every fold, which is the concrete evidence that SMOTE only ever sees training
rows.

One real bug this structure caught in practice, worth recording here rather than only in the commit
history: `SimpleImputer(strategy="median")` **without** `keep_empty_features=True` silently *drops* any
column with zero observed values instead of imputing it to a constant. On real Dataset A data this
collapsed the pipeline's output from 11 columns to 7 — the four `B_ONLY` columns (100% NaN by design, D2,
Chapter 5.4) vanished rather than becoming the constant-zero features the schema intends. Caught by
running the pipeline against real data, not a synthetic frame; fixed by adding the flag and confirmed with
`warnings.simplefilter("error")` that the fixed pipeline raises zero warnings and preserves all 11
columns on Dataset A. `keep_empty_features=True` is therefore the structural implementation of the D2
observability finding, not a defensive afterthought: "this family carries no signal here" is a property of
the pipeline's output shape, not merely a sentence in Chapter 5.

## 7.3 Hyperparameters and sensitivity sweeps

Every hyperparameter that trains a model is explicit in `config/config.yaml` — no library defaults are
relied on anywhere in the codebase (Step 0D / this rubric item).

| Model | Hyperparameter | Value |
|---|---|---|
| XGBoost | `n_estimators` | 400 |
| | `max_depth` | 6 |
| | `learning_rate` | 0.05 |
| | `subsample` | 0.8 |
| | `colsample_bytree` | 0.8 |
| | `min_child_weight` | 3 |
| | `scale_pos_weight` | computed per dataset+framing at train time |
| Isolation Forest | `n_estimators` | 200 |
| | `max_samples` | 256 |
| | `contamination` | 0.2 (baseline; swept for the cascade, see below) |
| | `max_features` | 0.8 |
| 1D-CNN | `conv_filters` | [32, 64] |
| | `kernel_size` | 3 (`padding="same"`) |
| | `dropout` | 0.3 |
| | `dense_units` | 64 |
| | `optimizer` / `learning_rate` | Adam / 0.001 |
| | `batch_size` | 256 |
| | `early_stopping_patience` | 5 (on validation PR-AUC), `max_epochs` 100 cap |
| | `val_frac` | 0.2, carved from the training fold only, never the test fold |
| Autoencoder | `layers` | [11, 8, 4, 8, 11] |
| | `activation` / `loss` | ReLU / MSE |
| | `optimizer` / `learning_rate` | Adam / 0.001 |
| | `batch_size` | 256, `max_epochs` 100 |
| | `threshold_percentile` | 95th percentile of benign training-fold reconstruction error |

All four models fix `random_state=42`; the 5-fold `StratifiedKFold` (`shuffle=True, random_state=42`)
is shared across every model and dataset via `preprocessing.pipeline.get_cv()`.

**Sensitivity sweep 1 — XGBoost `max_depth ∈ {3, 6, 9, 12}`, Dataset A (221,315 rows,
`families="full"`):**

| depth | F1 | FPR |
|---|---|---|
| 3 | 0.8181 | 0.4049 |
| 6 | 0.8182 | 0.4048 |
| 9 | 0.8182 | 0.4049 |
| 12 | 0.8182 | 0.4049 |

**Interpretation.** F1 and FPR move by less than 0.001 across the entire depth range — there is no
visible overfit-onset within this grid to point to. Read alongside the high, depth-invariant FPR
(~40%), the honest story is that depth is not the bottleneck: Dataset A's effective feature set is 7
informative numeric columns (F1–F3; the other 4 are structurally constant, Chapter 5.4), and 400 boosted
trees at depth 3 already extract essentially everything an axis-aligned split model can extract from 7
continuous features. Going deeper neither helps (there is no additional structure in 7 continuous
features for extra depth to find) nor hurts (there is not enough added capacity to memorise noise at
n≈221K). The same ~0.40 FPR was independently seen from the plain LogisticRegression smoke test in Step
2A′ — corroborating evidence that the ~40% FPR is a property of the feature set on this data, not an
XGBoost-specific quirk.

**Sensitivity sweep 2 — Isolation Forest `contamination ∈ {0.05, 0.10, 0.20, 0.30}`, Dataset B (both
framings), `families="full"`:**

| contamination | hard F1 | hard FPR | hard recall | easy F1 | easy FPR | easy recall |
|---|---|---|---|---|---|---|
| 0.05 | 0.1107 | 0.0387 | 0.0609 | 0.0582 | 0.0537 | 0.0359 |
| 0.10 | 0.2283 | 0.0627 | 0.1370 | 0.1792 | 0.0916 | 0.1315 |
| 0.20 | 0.3411 | 0.1627 | 0.2391 | 0.3326 | 0.1671 | 0.3218 |
| 0.30 | 0.4137 | 0.2681 | 0.3307 | 0.3430 | 0.2697 | 0.4120 |

**Interpretation.** Recall and FPR rise monotonically with contamination on both framings, as expected —
flagging more rows as outliers necessarily catches more attacks at the cost of more false alarms. All
four swept points sit under the cascade's 0.5 FPR budget on both framings, so `select_cascade_contamination`
(the same rule Step 3C's cascade Stage 1 uses, deliberately maximising **recall** at tolerable FPR rather
than maximising F1, because a first-stage filter that misses an attack is fatal while one that
over-flags merely costs the next stage extra work) picks the highest swept value, `contamination=0.30`,
on both framings — hard: recall 0.3307/FPR 0.2681/F1 0.4137; easy: recall 0.4120/FPR 0.2697/F1 0.3430.
Because the whole grid stayed inside budget, this sweep does not locate where the recall/FPR tradeoff
actually bends — the honest caveat is "0.30 is the best point *tested*," not "0.30 is an interior
optimum." This ceiling is exactly what Chapter 8.4's real cascade result traces back to: even the best
available Stage-1 recall (33.1% on hard framing) caps the whole cascade's end-to-end recall, since any
row Isolation Forest discards never reaches XGBoost.

## Appendix A — Setup and reproduction commands

```bash
git clone <repo-url>
cd maleware_detection_final_project

python -m venv venv
venv\Scripts\activate                 # Windows
# source venv/bin/activate            # macOS/Linux

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Place raw data (never committed -- gitignored):
#   data/exf2021/       <- CIC-Bell-DNS-EXF-2021 stateless CSVs (Attacks + Benign)
#   data/dohbrw2020/    <- CIRA-CIC-DoHBrw-2020 CSV distribution (l1-nondoh.csv, l2-benign.csv, l2-malicious.csv)

# Verify the install:
python -c "import pandas, sklearn, xgboost, imblearn, torch; print('ok')"

# Smoke-test both loaders:
python main.py --dataset exf2021 --mode eda
python main.py --dataset dohbrw2020 --framing hard --mode eda
python main.py --dataset dohbrw2020 --framing easy --mode eda
python main.py --dataset dohbrw2020 --framing hard --families intersection --mode eda

# Run the full test suite:
pytest tests/ -v

# Reproduce a specific model result, e.g. Step 3C's cascade on Dataset B hard framing:
PYTHONPATH=. python -c "
from config.loader import load_config          # or yaml.safe_load('config/config.yaml')
from ingestion.dohbrw2020 import DohBrw2020Loader
from ensemble.cascade import run_cascade, save_cascade_result
config = load_config()
X, y, meta = DohBrw2020Loader(config, framing='hard').load()
result = run_cascade(X, y, config, meta, families='full')
save_cascade_result('dohbrw2020_hard', result)
"
```

On Windows, `PYTHONPATH` must be set explicitly for any script run from outside the repository root
(`PYTHONPATH="<repo-root>" venv\Scripts\python.exe -u <script>`) — the working directory alone is not
sufficient for the `evaluation`/`models`/`ensemble` package imports to resolve.

## Appendix B — Library versions and machine specifications

**Library versions** (pinned exactly in `requirements.txt`, per Step 0A — the grader reproduces our
numbers against these exact versions, not a `>=` range):

| Library | Version |
|---|---|
| Python | 3.12.1 |
| pandas | 3.0.5 |
| numpy | 2.5.2 |
| scikit-learn | 1.9.0 |
| scipy | 1.18.0 |
| imbalanced-learn | 0.14.2 |
| xgboost | 3.4.0 |
| torch | 2.13.0+cpu (installed from the CPU wheel index, per D10 — avoids ~2 GB of unused CUDA packages) |
| matplotlib | 3.11.1 |
| seaborn | 0.13.2 |
| PyYAML | 6.0.3 |
| python-docx | 1.2.0 |
| pytest | 9.1.1 |

**Machine specifications:**

| | Teammate B (this machine) | Teammate A |
|---|---|---|
| OS | Windows 11 Pro, 64-bit | Windows 11 Home, 64-bit (Build 26200) |
| CPU | 12th Gen Intel Core i5-1240P (12 cores / 16 logical processors) | 13th Gen Intel Core i7-1360P (12 cores / 16 logical processors) |
| RAM | 15.7 GB | 15.7 GB |
| GPU | None used — all model training is CPU-only per D10 (PyTorch CPU build; XGBoost/sklearn are CPU by default) | None used — same CPU-only convention |
| Role in this project | Dataset B (CIRA-CIC-DoHBrw-2020) local; Step 3C cascade, Step 2E/2F, report chapters | Dataset A (CIC-Bell-DNS-EXF-2021) local; Step 2G real-data run, Step 2F Dataset A backfill, Ch 8.1–8.3 |

**A real discrepancy worth flagging rather than silently smoothing over: the "Library versions" table
above states one set of pinned versions "the grader reproduces our numbers against," but Teammate A's
actual installed environment does not match `requirements.txt` exactly** — checked directly (`python -c
"import pandas; print(pandas.__version__)"` etc.) rather than assumed:

| Library | `requirements.txt` (pinned) | Teammate A's actual environment |
|---|---|---|
| Python | 3.12.1 | 3.11.9 |
| pandas | 3.0.5 | 2.2.3 |
| numpy | 2.5.2 | 2.2.4 |
| scikit-learn | 1.9.0 | 1.8.0 |
| scipy | 1.18.0 | 1.16.3 |
| imbalanced-learn | 0.14.2 | 0.14.2 (matches) |
| xgboost | 3.4.0 | 3.2.0 |
| torch | 2.13.0+cpu | 2.10.0+cpu |
| matplotlib | 3.11.1 | 3.10.3 |
| seaborn | 0.13.2 | 0.13.2 (matches) |
| PyYAML | 6.0.3 | 6.0.3 (matches) |
| pytest | 9.1.1 | 8.4.2 |

Every result in this report from Teammate A's machine (Step 1A/2D/2E's Dataset A half, Step 2F's Dataset
A backfill, Step 2G's real cross-dataset run, Step 3A/3B forensics) was produced under the versions in
the right-hand column, not the pinned ones — `requirements.txt` appears to have been pinned from
Teammate B's environment at some point after Teammate A's venv was first set up (Step 0A), and the two
were never re-synced. Reported honestly here rather than either (a) claiming exact-version reproducibility
that did not actually hold, or (b) risking a late dependency upgrade this close to the deadline purely to
make the table match — a major-version bump on pandas (2→3) or numpy this late carries real regression
risk for zero report-quality benefit, since every real number in this document was already produced,
verified, and tested under the versions actually listed above.
