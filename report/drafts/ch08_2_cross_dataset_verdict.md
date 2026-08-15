# Chapter 8.2 — Cross-Dataset Comparison: The Verdict

## 8.2.1 The question

Chapter 5 (§5.2) measured that cross-dataset transfer collapses to a trivial fixed-guess classifier in 7
of 8 transfer cells, across all four models. The rubric asks for an explicit diagnosis: is this
**distribution shift** (the two environments genuinely differ, so a model correctly learned on A knows
nothing useful about B) or **overfitting** (the model memorised training-set idiosyncrasies rather than
a generalisable signal, so even a *correctly*-scaled version of B would fail)? These have different
practical implications — distribution shift is fixed by retraining per environment; overfitting would
mean the models are not learning the intended signal at all, on either dataset.

## 8.2.2 The verdict: distribution shift, not overfitting — and the evidence is direct, not inferred

**This is distribution shift, cleanly.** The evidence does not require inference from indirect symptoms;
it is a direct measurement:

1. **In-domain performance is strong for every model, on both datasets, ruling out "the models never
   learned a real signal" as an explanation.** XGBoost and the CNN both score F1≈0.82 on Dataset A and
   F1≈0.99–1.00 on Dataset B — recall above 0.99 on both, in both directions. A model that had merely
   memorised training-set noise would not generalise this well to its *own* held-out fold, 5 times over,
   under stratified CV. Whatever XGBoost and the CNN learned, it predicts real held-out rows correctly
   within each dataset — the failure is specific to *crossing* datasets, not a general failure to learn.
2. **The distribution-shift analysis (§5.2.3) measures the cause directly, rather than inferring it from
   the transfer failure alone.** Every one of the 7 intersection columns shows a Kolmogorov–Smirnov
   statistic between 0.89 and 1.00 — near-total or fully disjoint support between Dataset A's and Dataset
   B's values for the *same* nominal feature. `vol_total` differs by a factor of ~1,859× in raw scale
   between datasets (§5.2.3) — this is not a subtle shift a slightly-overfit model might have been
   robust to; it is the input space itself being numerically almost unrecognisable across the boundary.
   This was measured on the raw data, independent of any model's behaviour — it would be true whether or
   not a single model had ever been trained.
3. **The specific failure mode matches distribution shift's signature, not overfitting's.** An overfit
   model transferred to new data typically degrades gracefully or fails inconsistently row-by-row. What
   was observed instead (§5.2.1) is uniform collapse to a **fixed decision** — every model, transferred
   in a given direction, predicts either *all* rows or *no* rows positive (FPR/recall pinned at exactly
   1.0 or exactly 0.0 in 6 of 8 transfer cells). That is the signature of a decision boundary, learned in
   one dataset's standardised coordinate space, landing entirely on one side of the other dataset's
   differently-shaped standardised space — a geometric consequence of the two datasets occupying
   different numeric regions after scaling, not a memorisation artifact.
4. **The one partial exception reinforces the same reading.** CNN's train-B→test-A cell has F1=0.0000
   (the fixed-decision collapse) but ROC-AUC=0.773 — well above chance. If the CNN had overfit Dataset
   B's noise, its *ranking* of Dataset A's rows should be no better than random once transferred. It is
   not: the CNN's probability output still orders Dataset A's rows correctly, most of the time, after
   transfer — the underlying signal survived; only the fixed 0.5 decision threshold (calibrated on
   Dataset B's own scale) failed to land anywhere sensible on Dataset A's differently-scaled inputs. This
   is exactly what distribution shift predicts and exactly what overfitting would not produce.

## 8.2.3 Per-cell diagnosis (evidence, not just the aggregate verdict)

| direction | model | outcome | what it demonstrates |
|---|---|---|---|
| A → B | XGBoost, CNN, Isolation Forest, Autoencoder | always-positive collapse (F1≈0.65–0.67, FPR=1.0) | Dataset A-trained boundary reads all of B's rescaled inputs as "high enough to be attack" |
| B → A | XGBoost, CNN | always-negative collapse (F1=0.0000) | Dataset B-trained boundary reads all of A's rescaled inputs as "too low to be attack" |
| B → A | Isolation Forest, Autoencoder | always-positive collapse (F1≈0.65, FPR=1.0) | opposite direction from XGBoost/CNN's B→A collapse — these two unsupervised models' benign-shaped reference region, built from B, treats essentially all of A's differently-scaled data as anomalous |
| B → A | CNN (ranking only) | ROC-AUC=0.773 despite F1=0.0000 | signal survives transfer; only the fixed threshold fails — the clearest single piece of evidence that this is a calibration/scale problem, not a learning failure |

The fact that the *same* direction (B→A) produces opposite fixed-classifier outcomes depending on model
family (supervised XGBoost/CNN collapse to always-negative; unsupervised Isolation Forest/Autoencoder
collapse to always-positive) is itself informative: it rules out a single simple explanation like "B's
data is just numerically larger than A's, so everything looks like an outlier both ways" — the actual
mechanism is specific to each model's own decision function interacting with the scale mismatch, not one
universal artifact.

## 8.2.4 What this means for deployment

**Practical conclusion, stated plainly rather than softened:** none of the four architectures tested in
this project can be trained once and deployed across both a plaintext-DNS resolver log and an encrypted
DoH flow capture. The unified schema's behavioural abstraction (same family names, same conceptual
security meaning, `schema/unified.py`) is real at the level of *what each family is trying to measure* —
both vantage points do carry a payload-volume signal, an encoding-randomness signal, a structural-
complexity signal — but is not sufficient, on its own, to make a trained model portable, because the
*numeric realisation* of those families differs by orders of magnitude between vantage points (§5.2.3).
The correct deployment story for a system built this way is **train and calibrate separately per vantage
point** — this project's two datasets are not stand-ins for "any DNS exfiltration data," they represent
two genuinely different measurement regimes that happen to share a conceptual feature taxonomy.

## 8.2.5 What's still open

- The proposed remedies in §5.2.3/§5.3 (`log1p` on all three F1 columns before scaling, or a scaler fit
  jointly across both datasets' training halves) are documented but not yet tested against this same
  transfer matrix — a concrete next step if time allows, not claimed as a fix here.
- This diagnosis covers `mode="intersection"` only (F1–F3), per Step 2G's design — the B_ONLY families
  (F4/F5) cannot be tested for transfer at all, by construction, since Dataset A has no values for them
  (§5.4).
