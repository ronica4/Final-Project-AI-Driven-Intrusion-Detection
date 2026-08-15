# Chapter 5 — Harmonisation Across the Encryption Boundary

## 5.1 Unified schema specification

Every loader (`ingestion/exf2021.py`, `ingestion/dohbrw2020.py`) emits the same 11 columns, in the
same order, grouped into five behavioural families (`schema/unified.py`). Column order matters beyond
readability: the 1D-CNN's receptive field (Chapter 6.1) is only meaningful because related features sit
physically adjacent in the vector.

| Family | Columns | Role | Security meaning |
|---|---|---|---|
| **F1** Payload volume | `vol_primary`, `vol_secondary`, `vol_total` | INTERSECTION | Data volume moving through a channel that should only carry small lookups — exfiltration must push more bytes per unit than normal traffic. |
| **F2** Encoding randomness | `rand_entropy`, `rand_dispersion` | INTERSECTION | Entropy signature of base64URL-encoded payload; encoded binary looks statistically random, legitimate hostnames do not. |
| **F3** Structural complexity | `struct_segments`, `struct_max_segment` | INTERSECTION | Carrier segmentation forced by DNS label-length limits (63 B/label, 253 B/FQDN); chunking a file into many labels produces structurally unusual names. |
| **F4** Temporal rhythm | `time_central`, `time_dispersion`, `time_skew` | B_ONLY | Machine-paced vs. human-paced timing — automated exfiltration has regular inter-arrival timing. |
| **F5** Endpoint dispersion | `disp_uniqueness` | B_ONLY | Fan-out and destination rarity — exfiltration channels concentrate traffic toward few attacker-controlled endpoints. |

**INTERSECTION vs. B_ONLY is a scientific claim, not an implementation detail.** Three families (F1–F3)
have a genuine, if imperfect, counterpart on both sides of the encryption boundary. Two (F4–F5) exist
only in Dataset B, because Dataset A is restricted to *stateless* per-query features (decision D1) and
therefore has no temporal or endpoint telemetry at all to draw on — not a gap we failed to fill, but a
structural consequence of what a stateless DNS resolver log can observe versus what a full DoH flow
capture can observe. `COLUMN_SOURCE` (`schema/unified.py`) documents the exact per-dataset arithmetic
behind every column, including that F2/F3's Dataset-B realisations (`PacketLengthCoefficientofVariation`,
`log1p(PacketLengthVariance)`, `PacketLengthMode`, `PacketLengthStandardDeviation`) are **statistical
proxies for**, not equivalents of, Dataset A's direct string-level entropy and label-count measures — a
distinction that motivates the D3 ablation below.

## 5.2 Cross-dataset transfer, ablation, and distribution shift

**Status (15 Aug 2026): run against real data, all four models, by Teammate A** — Teammate B built and
unit-tested `evaluation/cross_dataset.py` (11/11 passing on synthetic data) but could not run it for real,
lacking Dataset A locally. Once Teammate B sent the three raw Dataset B CSVs (`l2-benign.csv`,
`l2-malicious.csv`, `l1-nondoh.csv`), both datasets were available on Teammate A's machine and the module
was run as designed, no code changes needed. Full numbers: `runs/metrics/cross_dataset_transfer_matrix_
{model}.json`, `cross_dataset_ablation_{model}.json`, `cross_dataset_distribution_shift.json`.

**Headline finding: transfer collapses to a trivial classifier in 7 of the 8 transfer cells tested, for
every one of the four models.** This is not a per-model quirk — it recurs identically across XGBoost,
Isolation Forest, the CNN, and the Autoencoder, four structurally unrelated architectures.

The module itself implements exactly what the design calls for and is documented here so the eventual
numbers slot into a pre-built frame rather than requiring restructuring:

- **Transfer matrix** (`build_transfer_matrix`): all four cells — train-A/test-A, train-B/test-B
  (in-domain, via each model's own 5-fold CV), train-A/test-B, train-B/test-A (transfer, single
  fit-on-source/score-on-target pass) — every cell restricted to `mode="intersection"` (F1–F3, the only
  columns both datasets share) so the four cells are directly comparable. The scaler is fit on the
  **training/source dataset only** and applied as-is to the target — the honest simulation of deploying
  a model into a new environment rather than an oracle that has seen the target's own scale.
- **Ablation (D3)** (`ablation_f1_only_vs_intersection`): both transfer directions rerun with
  `families="F1_only"` against the already-computed `families="intersection"` cells. The hypothesis
  under test is that F1 (volume) means the same thing on both sides of the encryption boundary while
  F2/F3 are lookalikes that should *hurt* transfer — reported whichever way it actually comes out, not
  forced to confirm the prediction.
- **Distribution shift** (`distribution_shift`): a Kolmogorov–Smirnov two-sample statistic per
  intersection column, ranked descending, cross-checked against the ablation result — if the
  worst-transferring features are also the ones with the largest KS distance, that is two independent
  lines of evidence for the same conclusion.

### 5.2.1 Transfer matrix — real results, all four models

`mode="intersection"` (F1–F3, 7 columns) throughout. In-domain cells via 5-fold CV; transfer cells via a
single fit-on-source/score-on-target pass, scaler fit on the source only.

| model | train A→test A (in-domain) | train B→test B (in-domain) | train A→test B (transfer) | train B→test A (transfer) |
|---|---|---|---|---|
| XGBoost | F1=0.8182, recall=0.999, FPR=0.405 | F1=0.9999, recall=1.000, FPR=0.000 | F1=0.6667, recall=1.000, FPR=1.000, ROC-AUC=0.707 | F1=0.0000, recall=0.000, FPR=0.000, ROC-AUC=0.490 |
| Isolation Forest | F1=0.1051, recall=0.075, FPR=0.313 | F1=0.3954, recall=0.276, FPR=0.121 | F1=0.6667, recall=1.000, FPR=1.000, ROC-AUC=0.579 | F1=0.6460, recall=1.000, FPR=1.000, ROC-AUC=0.379 |
| CNN | F1=0.8176, recall=0.998, FPR=0.404 | F1=0.9927, recall=0.993, FPR=0.008 | F1=0.6667, recall=1.000, FPR=1.000, ROC-AUC=0.708 | F1=0.0000, recall=0.000, FPR=0.000, **ROC-AUC=0.773** |
| Autoencoder | F1=0.0474, recall=0.026, FPR=0.049 | F1=0.6773, recall=0.560, FPR=0.051 | F1=0.6667, recall=1.000, FPR=1.000, ROC-AUC=0.671 | F1=0.6460, recall=1.000, FPR=1.000, ROC-AUC=0.359 |

**Every in-domain cell is broadly consistent with that model's own single-dataset result elsewhere in
this report** (XGBoost/CNN both land near F1≈0.82 on A and F1≈0.99–1.00 on B; Isolation Forest's weak
in-domain numbers match Step 2E; the Autoencoder's below-chance ROC-AUC=0.335 on Dataset A echoes Step
2E's Isolation Forest finding — see §5.4). **Every transfer cell collapses to (or very near) a trivial
fixed-guess classifier** — F1=0.6667/recall=1.0/FPR=1.0 is exactly the always-positive baseline for a
dataset whose positive rate is ~0.50 (Dataset B); F1=0.6460 at the same recall/FPR is the identical
always-positive baseline for Dataset A (positive rate 0.4772); F1=0.0000/recall=0/FPR=0 is the
mirror-image always-negative baseline. **No model, of four structurally different ones, produces a
working cross-dataset detector.**

**One genuine exception worth flagging, not smoothing over:** CNN's train-B→test-A cell has F1=0.0000
(zero rows crossed the 0.5 decision threshold) but **ROC-AUC=0.773 — well above chance**, unlike every
other trivial-classifier cell in the table (whose ROC-AUC sits at or near 0.50, or even below it for
Isolation Forest's B→A cell at 0.379). This means the CNN's raw probability *ranking* still carries real
signal after transfer — the model has not forgotten how to distinguish attack from benign on Dataset A's
rescaled inputs, it has just landed a decision threshold (0.5, calibrated implicitly during training on
Dataset B's own scale) that happens to call zero of Dataset A's rows positive. This suggests **threshold
recalibration on a small target-domain sample**, not a full retrain, might partially rescue this one
transfer direction for the CNN specifically — a concrete idea for future work, not tested here given time.

### 5.2.2 The ablation (D3) — real results, all four models: hypothesis falsified, unanimously

| model | direction | intersection F1 | F1-only F1 | F1-only transfers better? |
|---|---|---|---|---|
| XGBoost | A→B | 0.6667 | 0.0000 | **No** |
| XGBoost | B→A | 0.0000 | 0.0000 | No (tied at the floor) |
| Isolation Forest | A→B | 0.6667 | 0.6667 | No (tied — same trivial classifier) |
| Isolation Forest | B→A | 0.6460 | 0.6460 | No (tied — same trivial classifier) |
| CNN | A→B | 0.6667 | 0.6667 | No (tied — same trivial classifier) |
| CNN | B→A | 0.0000 | 0.0000 | No (tied — same trivial classifier) |
| Autoencoder | A→B | 0.6667 | 0.6667 | No (tied — same trivial classifier) |
| Autoencoder | B→A | 0.6460 | 0.6460 | No (tied — same trivial classifier) |

**F1-only transfers better in zero of eight cells, across all four models.** The plan's hypothesis — that
F1 (payload volume) is the one family that survives the encryption boundary intact, and dropping the
"lookalike" F2/F3 families should therefore *help* transfer — is not merely unconfirmed, it is
**unanimously falsified**. Where the two feature sets are distinguishable at all (XGBoost's A→B cell),
F1-only is strictly *worse*; everywhere else, both feature sets degenerate to the identical trivial
classifier, meaning F1 alone provides no rescue either. §5.2.3 explains why: F1's own columns show among
the *largest* distributional shifts of the entire intersection set, contradicting the premise that F1 is
scale-stable across the boundary.

### 5.2.3 Distribution shift — why transfer collapses

Real data, Kolmogorov–Smirnov two-sample statistic per intersection column, Dataset A vs. Dataset B:

| feature | family | KS statistic | mean (A) | mean (B) | ratio (B/A) |
|---|---|---|---|---|---|
| `vol_primary` | F1 | 1.0000 | 12.39 | 173.03 | ×14.0 |
| `vol_secondary` | F1 | 1.0000 | 5.83 | 95.19 | ×16.3 |
| `vol_total` | F1 | 1.0000 | 21.81 | 40,523.60 | ×1,858.6 |
| `struct_segments` | F3 | 1.0000 | 4.68 | 70.79 | ×15.1 |
| `rand_dispersion` | F2 | 0.9897 | 0.89 | 9.34 | ×10.5 |
| `struct_max_segment` | F3 | 0.9011 | 8.17 | 220.02 | ×26.9 |
| `rand_entropy` | F2 | 0.8916 | 2.48 | 1.00 | ×0.40 |

Every column — **including every one of F1's three columns** — shows a KS statistic between 0.89 and
1.00: near-total or fully disjoint support between the two datasets. `vol_total` alone differs by a
factor of ~1,859× in raw scale, because Dataset A's realisation (`FQDN_count`, a small per-query integer)
and Dataset B's (`FlowBytesSent`, cumulative bytes in a flow) are behaviourally analogous but not
numerically comparable quantities (`schema/unified.py` `COLUMN_SOURCE`). **This directly contradicts
§5.4's original premise that F1 is "the family with a genuinely direct counterpart on both sides"** — at
the raw-distribution level it is not, and z-scoring (fit on the training/source dataset only, §5.3) does
not repair this: standardising to mean 0 / std 1 preserves each dataset's own distribution *shape*, and
the two shapes differ enough that a decision boundary learned in one dataset's standardised space does
not carve up the other's the same way. Concretely: on Dataset A, "high z-scored volume" reliably signals
attack; the same z-score region in Dataset B's differently-shaped standardised space behaves completely
differently, so a model trained on A calls nearly everything in B's space "high" (the observed
always-positive collapse), and the reverse direction collapses the opposite way.

**What §5.2.1's ranking check (the module's own stated goal — cross-check the KS ranking against the
ablation result) actually shows:** with every column at KS≥0.89, there is no meaningful ranking signal
left to check against the ablation — the ablation's uniform, unanimous failure (§5.2.2) is fully
consistent with a uniform, near-total distribution shift across every intersection column, not a
graded one where some features transfer better than others. That uniformity is itself the finding: this
is not "F2/F3 are worse proxies than F1," it is "none of the 7 raw-scale intersection columns are
numerically comparable across this specific encryption boundary, F1 included."

**What was not tried, given today's deadline:** a `log1p` transform applied uniformly to all three F1
columns before scaling (already used for `rand_dispersion`'s Dataset-B realisation, §5.3, but not
extended to F1), or a scaler fit jointly across both datasets' training halves rather than per-source —
proposed remedies, not yet tested against this same transfer matrix.

## 5.3 Scaling remedies

Three remedies are applied, each for a distinct mathematical reason, all inside `preprocessing/pipeline.py`'s
`build_pipeline()` so every model shares one preprocessing contract:

- **Z-scoring within dataset** (`StandardScaler`, fit per-fold in-domain / fit-on-source for transfer).
  The unified schema's 11 columns span wildly different native scales (byte counts vs. entropy bits vs.
  segment counts) — z-scoring puts every feature on a comparable footing before it reaches a
  distance-sensitive estimator (Isolation Forest's isolation depth, the CNN's gradient-based training),
  and doing it **per fold/per source dataset only** — never on the full combined data — is what keeps the
  cross-dataset transfer cells an honest "deploy into an unseen environment" simulation rather than an
  oracle fit.
- **`log1p` on heavy-tailed volume features.** `rand_dispersion`'s Dataset-B realisation is defined
  directly as `log1p(PacketLengthVariance)` (`schema/unified.py` `COLUMN_SOURCE`) precisely because
  packet-length variance is heavily right-skewed by the rare, very large exfiltration payloads — a raw
  variance feature would let a handful of extreme rows dominate any distance- or gradient-based model's
  gradient signal; the log transform compresses that tail into a shape closer to the bulk of the benign
  distribution while preserving rank order.
- **Median imputation, not mean** (`SimpleImputer(strategy="median", keep_empty_features=True)`).
  Median is the robust choice for the same right-skewed volume features `log1p` targets — a mean
  computed from a skewed distribution is itself skewed, so imputing with it would bias every row that
  needed imputation toward the tail. `keep_empty_features=True` is the structural implementation of the
  D2 observability finding below: an entirely-NaN column (Dataset A's four B_ONLY columns) is imputed to
  a constant rather than silently dropped, which `StandardScaler` then leaves at a stable zero — "this
  family carries no signal here" becomes a structural property of the pipeline, not merely a sentence in
  this report.

## 5.4 The observability finding (D2)

**Three of five behavioural families are unobservable, or observable only as a weak statistical proxy,
once traffic crosses the plaintext-DNS-to-encrypted-DoH boundary — and measuring exactly how much
information that boundary destroys is this project's central empirical claim, not a limitation to
apologise for.**

- **F4 (temporal rhythm) and F5 (endpoint dispersion) are entirely unobservable on Dataset A.** Every row
  of `time_central`, `time_dispersion`, `time_skew`, and `disp_uniqueness` is NaN by construction on
  Dataset A — not a data-quality defect, but the direct consequence of Dataset A being restricted to
  stateless, single-query features (decision D1): a resolver log of independent queries has no
  client-session key to compute inter-arrival rhythm or destination fan-out over. Dataset B, built from
  full DoH flow captures, observes both natively. This is verified structurally in `runs/metrics/near_constant_report_exf2021.json`
  (Chapter 3.6): the four B_ONLY columns are exactly 100% NaN, confirmed `near_constant: true` only
  *after* imputation, never silently coerced beforehand.
- **F2 and F3 survive the boundary only as proxies, not equivalents.** Dataset A measures encoding
  randomness and structural complexity directly from the query string itself (`entropy`, `labels`,
  `labels_max` — string-level measures available because Dataset A observes the plaintext query).
  Dataset B has no plaintext query to inspect at all — DoH's entire premise is that the query is
  encrypted in transit — so its F2/F3 columns are reconstructed from packet-length statistics instead
  (`PacketLengthCoefficientofVariation`, `PacketLengthMode`, `PacketLengthStandardDeviation`). These are
  *correlated with*, not *identical to*, the plaintext-side signal: a base64-encoded payload's
  string-level entropy and the packet-length variability it produces on the wire are related but
  distinct measurements, one direct and one inferred. This is exactly the distinction the D3 ablation
  (§5.2) is designed to quantify — if F2/F3 genuinely only proxy the plaintext signal, models trained on
  F1+F2+F3 should transfer worse across the boundary than models trained on F1 alone.
- **F1 (payload volume) was predicted to be the one family with a genuinely direct counterpart on both
  sides.** Encrypted or not, a payload of N bytes still produces N bytes of DNS/DoH traffic — volume is a
  property of the channel, not of whether the channel's content is legible, so F1 was the family
  predicted to transfer best.

**Update (15 Aug 2026, once §5.2's real transfer numbers landed): the F1 prediction above did not
hold, and the ablation falsifies it directly (§5.2.2).** Conceptual equivalence ("both sides measure
payload volume") turned out not to imply numerical equivalence: §5.2.3's distribution-shift analysis
shows F1's own three columns among the *most* distributionally shifted of the whole intersection set
(KS 0.90–1.00; `vol_total` differs ×1,859 in raw scale, because Dataset A's realisation is a small
per-query integer count and Dataset B's is cumulative flow bytes — related in concept, not in units).
The corrected reading: **"can be computed on both sides" and "means the same numeric thing on both
sides" are different claims, and this project's data says F1 satisfies only the first one** — no
better than F2/F3, and on `vol_total` specifically, considerably worse. This sharpens, rather than
undermines, the chapter's central claim: even the family assumed safest from the encryption boundary's
effects still fails to transfer, so the boundary's practical cost is *larger* than the original D2
argument (which only priced in F4/F5's total loss and F2/F3's degradation) accounted for.

**Framed as a result about what encrypted telemetry can and cannot reveal:** an operator who only has
access to encrypted DoH flow captures — the realistic case for anyone downstream of the resolver once
DoH is in wide use — retains conceptual visibility into payload volume, encoding randomness, and
structural complexity, and zero visibility into anything Dataset A's stateless schema was ever able to
see about rhythm or fan-out (because Dataset A never had that telemetry either). But §5.2's real numbers
show that visibility does not equal **transferability**: a detector trained on one vantage point's raw
feature scale cannot be deployed as-is on the other, for any of the four models tested, even on the
families both sides can nominally compute. The practical implication for a defender is therefore not
"invest in F1 over F2/F3" (the original hypothesis) but **"train and calibrate separately per vantage
point — plaintext-DNS and encrypted-DoH require their own models, not one model deployed twice."**
