# Chapter 7 — Pipeline, Imbalance Strategy, and Hyperparameters

## 7.1 Block diagram and the abstraction proof

```
 CLI --> registry.get(dataset_name) --> loader.load() -> (X, y, meta)
      --> schema.validate_schema(X, mode=families)
      === EVERYTHING BELOW IS DATASET-BLIND ===
      --> preprocessing.build_pipeline (impute -> scale -> [SMOTE] -> estimator)
      --> models/{supervised,unsupervised,deep}.py (XGBoost | IsoForest | CNN | AE)
      --> evaluation.metrics.evaluate()
      --> ensemble/cascade.py (Step 3C: IsoForest -> XGBoost -> escalation)
```

Load boundary (`loader.load()` → `(X, y, meta)`) enforced structurally (Dataset Dependency Rule, Step 0D),
verified empirically: Step 1C's abstraction proof greps for dataset-name literals, requires empty output
outside three sanctioned exceptions (CLI plumbing, `schema/unified.py`'s `COLUMN_SOURCE`, one test file).
No module under `preprocessing/`, `features/`, `models/`, `evaluation/`, `ensemble/` contains dataset-keyed
logic — what makes Ch. 5's transfer matrix a meaningful test, not a claim on faith.

## 7.2 Imbalance strategy and the structural leakage guard

**Two non-overlapping answers, per model:** SMOTE for the CNN (fit-time, inside the Pipeline);
`scale_pos_weight` for XGBoost; neither for Isolation Forest (resampling a density estimator defeats its
premise) or the Autoencoder (fit exclusively on benign rows). Rule: **resampling xor reweighting, never
both**.

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

Two details matter. **`imblearn.pipeline.Pipeline`, never `sklearn.pipeline.Pipeline`** — sklearn's would
silently apply SMOTE at *transform* time too, synthesizing fake minority rows inside the test fold with no
warning. **The whole object goes to `StratifiedKFold` as one fittable unit**, so scaler/SMOTE stats
recompute from scratch inside every fold, training rows only — verified via a synthetic 400/20 frame
across 5 folds confirming test-fold class counts (95/5) never move.

Bug caught by this structure: `SimpleImputer(strategy="median")` without `keep_empty_features=True`
silently drops any all-NaN column instead of imputing to a constant — on real Dataset A this collapsed 11
columns to 7 (the four B_ONLY columns vanished). Fixed by adding the flag.

## 7.3 Hyperparameters and sensitivity sweeps

Every hyperparameter explicit in `config/config.yaml` — no library defaults relied on. Full table in
Appendix C. All models fix `random_state=42`; 5-fold `StratifiedKFold` shared across every model/dataset.
Two sensitivity sweeps (Appendix C) confirm headline results aren't artifacts of an untuned hyperparameter:
XGBoost F1/FPR move under 0.001 across `max_depth ∈ {3,6,9,12}`; Isolation Forest recall/FPR rise
monotonically with `contamination` (0.30 chosen for the cascade — the ceiling Ch. 8.4 traces back to).

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
#   data/dohbrw2020/    <- CIRA-CIC-DoHBrw-2020 CSV distribution

# Verify install, then smoke-test both loaders:
python -c "import pandas, sklearn, xgboost, imblearn, torch; print('ok')"
python main.py --dataset exf2021 --mode eda
python main.py --dataset dohbrw2020 --framing hard --mode eda

# Run the full test suite:
pytest tests/ -v
```

On Windows, `PYTHONPATH` must be set explicitly for any script run from outside the repository root.

## Appendix B — Library versions and machine specifications

| Library | Pinned (`requirements.txt`) | Teammate A's actual env |
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

**Discrepancy flagged rather than smoothed over:** every Teammate-A-machine result in this report was
produced under the right-hand column, not the pinned one — `requirements.txt` was pinned from Teammate B's
environment after A's venv was already set up, and the two were never re-synced. A major-version bump this
close to the deadline carries real regression risk for zero benefit, since every number was already
produced under the versions listed.

| | Teammate B (this machine) | Teammate A |
|---|---|---|
| OS | Windows 11 Pro, 64-bit | Windows 11 Home, 64-bit (Build 26200) |
| CPU | 12th Gen Intel Core i5-1240P (12c/16t) | 13th Gen Intel Core i7-1360P (12c/16t) |
| RAM | 15.7 GB | 15.7 GB |
| GPU | None — CPU-only per D10 | None — same CPU-only convention |
| Role | Dataset B local; Step 3C cascade, 2E/2F, report chapters | Dataset A local; Step 2G real-data run, Ch 8.1-8.3 |

## Appendix C — Full hyperparameter table and sensitivity sweeps

| Model | Hyperparameter | Value |
|---|---|---|
| XGBoost | `n_estimators`/`max_depth`/`learning_rate` | 400 / 6 / 0.05 |
| | `subsample`/`colsample_bytree`/`min_child_weight` | 0.8 / 0.8 / 3 |
| | `scale_pos_weight` | computed per dataset+framing |
| Isolation Forest | `n_estimators`/`max_samples`/`max_features` | 200 / 256 / 0.8 |
| | `contamination` | 0.2 baseline (swept below) |
| 1D-CNN | `conv_filters`/`kernel_size`/`dropout`/`dense_units` | [32,64] / 3 / 0.3 / 64 |
| | `optimizer`/`learning_rate`/`batch_size` | Adam / 0.001 / 256 |
| | `early_stopping`/`max_epochs`/`val_frac` | 5 (val PR-AUC) / 100 / 0.2 |
| Autoencoder | `layers` | [11, 8, 4, 8, 11] |
| | `activation`/`loss`/`optimizer`/`learning_rate` | ReLU / MSE / Adam / 0.001 |
| | `batch_size`/`max_epochs`/`threshold` | 256 / 100 / 95th pct. benign error |

**XGBoost `max_depth ∈ {3, 6, 9, 12}`, Dataset A (221,315 rows, `families="full"`):**

| depth | F1 | FPR |
|---|---|---|
| 3 | 0.8181 | 0.4049 |
| 6 | 0.8182 | 0.4048 |
| 9 | 0.8182 | 0.4049 |
| 12 | 0.8182 | 0.4049 |

F1/FPR move under 0.001 across the grid — 400 boosted trees at depth 3 already extract essentially
everything an axis-aligned split model can from 7 informative columns. The same ~0.40 FPR was
independently seen from a plain LogisticRegression smoke test (Step 2A′), corroborating that ~40% FPR is a
property of the feature set, not an XGBoost quirk.

**Isolation Forest `contamination ∈ {0.05, 0.10, 0.20, 0.30}`, Dataset B (both framings):**

| contamination | hard F1 | hard FPR | hard recall | easy F1 | easy FPR | easy recall |
|---|---|---|---|---|---|---|
| 0.05 | 0.1107 | 0.0387 | 0.0609 | 0.0582 | 0.0537 | 0.0359 |
| 0.10 | 0.2283 | 0.0627 | 0.1370 | 0.1792 | 0.0916 | 0.1315 |
| 0.20 | 0.3411 | 0.1627 | 0.2391 | 0.3326 | 0.1671 | 0.3218 |
| 0.30 | 0.4137 | 0.2681 | 0.3307 | 0.3430 | 0.2697 | 0.4120 |

Recall/FPR rise monotonically on both framings; all four points sit under the cascade's 0.5 FPR budget, so
`select_cascade_contamination` picks `contamination=0.30` on both. This ceiling is exactly what Ch. 8.4's
cascade result traces back to: even the best available Stage-1 recall (33.1%, hard framing) caps the
whole cascade's end-to-end recall.
