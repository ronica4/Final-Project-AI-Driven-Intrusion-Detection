# Chapter 8.1 — Sample-Level Forensic Error Analysis

> Status (15 Aug 2026): **All four models complete on Dataset A**, all real, all out-of-fold (XGBoost and
> Isolation Forest from 14 Aug; CNN and Autoencoder added today, replacing the LogReg stand-in that
> previously filled the CNN's slot in the cross-model comparison — dropped now that real numbers exist,
> per the standing rule not to keep a stand-in alongside the real result it was substituting for). Dataset
> B summary metrics are from Teammate B's backfills. `evaluation/error_analysis.py`'s Step 3A harness has
> now run against all four architectures with zero code changes — confirming it really is model-agnostic
> by construction, not just convenient for the two models it happened to be built against first.

## 8.1.1 Confusion matrices

**Dataset A — exact counts, all four models** (real out-of-fold predictions, 5-fold CV, 221,315 rows):

| model | TN | FP | FN | TP | recall | FPR | F1 |
|---|---|---|---|---|---|---|---|
| XGBoost (t=0.50, Step 2D/3B baseline) | 68,868 | 46,846 | 58 | 105,543 | 99.95% | 40.48% | 81.82% |
| XGBoost (t=0.70, Step 3B optimised) | 69,747 | 45,967 | 2,081 | 103,520 | 98.03% | 39.72% | 81.16% |
| CNN | 68,896 | 46,818 | 188 | 105,413 | 99.82% | 40.46% | 81.77% |
| Isolation Forest (contamination=0.20, locked config) | 79,354 | 36,360 | 97,738 | 7,863 | 7.45% | 31.42% | 10.50% |
| Autoencoder (benign-only fit, 95th-pct threshold) | 110,007 | 5,707 | 103,131 | 2,470 | 2.34% | 4.93% | 4.34% |

**The two supervised models (XGBoost, CNN) land almost on top of each other** — F1 within 0.05 points,
recall both >99.8%, FPR both ~40.5% — despite being structurally unrelated architectures (gradient-boosted
trees vs. convolution). **The two unsupervised models (Isolation Forest, Autoencoder) both fail badly, in
the same direction:** both have recall under 8%, and the Autoencoder is worse than Isolation Forest on
every axis except FPR (which is low only because it flags almost nothing as positive at all — a low FPR
next to a 2.34% recall is not a favourable trade, it means the model is close to a constant-negative
classifier).

**Dataset B — reported metrics** (Teammate B's backfills; exact integer confusion matrices not in hand
for XGBoost, only the aggregated precision/recall/FPR/F1 — noted here as metrics, not reconstructed into
counts, to avoid presenting a derived number as an original one):

| model | framing | precision | recall | FPR | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| XGBoost | hard | — | — | — | ≈0.9999 | — |
| CNN | hard | 0.9956 | 0.9920 | 0.0044 | 0.9938 | 0.9995 |
| Isolation Forest | hard | 0.5951 | 0.2391 | 0.1627 | 0.3411 | 0.4611 |
| Isolation Forest | easy | 0.3441 | 0.3218 | 0.1671 | 0.3326 | 0.5572 |
| Autoencoder | hard | 0.8400 | 0.2681 | 0.0508 | 0.4064 | 0.7903 |

*(XGBoost's Dataset B row still only has F1 recorded — precision/recall/FPR/ROC-AUC weren't kept in the
tracker's one-line backfill summary. Same open item as Ch 8.2b's missing Dataset B FPR column.)*

**Dataset B tells almost the opposite story from Dataset A on the supervised/unsupervised split**: XGBoost
and CNN both remain excellent (F1 0.994–1.00), but Isolation Forest and the Autoencoder are *also*
meaningfully better on B than on A (F1 0.34–0.41 vs. 0.10–0.04) even though neither clears a useful
threshold. §8.1.2 and §8.1.4 dig into why the unsupervised gap is so much worse specifically on Dataset A.

## 8.1.2 Per-subclass recall — heavy vs. light (Dataset A)

| model | heavy_attack recall | light_attack recall | overall recall | gap (heavy − light) |
|---|---|---|---|---|
| XGBoost | 99.94% (62,881/62,918) | 99.95% (42,662/42,683) | 99.95% | **−0.01 pp** (light marginally higher) |
| CNN | 99.79% (62,784/62,918) | 99.87% (42,629/42,683) | 99.82% | **−0.08 pp** (light marginally higher) |
| Isolation Forest | 7.57% (4,764/62,918) | 7.26% (3,099/42,683) | 7.45% | +0.31 pp |
| Autoencoder | 2.41% (1,516/62,918) | 2.24% (954/42,683) | 2.34% | +0.17 pp |

**All four models tell the same story on the heavy-vs-light question, now with a real CNN result replacing
the LogReg stand-in: there is no meaningful heavy/light recall gap on Dataset A, in either direction, for
any model tested.** This is now confirmed a fourth independent way (two supervised architectures, two
unsupervised architectures), and continues to confirm Chapter 3's finding that light and heavy attacks sit
at statistically equal distance from benign on Dataset A's F1–F3 features (§3.5) — the light-class-blind-
spot hypothesis from Chapter 2 does not hold on this dataset, on these features, under any of the four
architectures tested.

## 8.1.3 False negative / false positive forensics

Pulled `sld` for interpretability only, never as a feature, same discipline throughout.

**XGBoost** (Step 3A, `runs/metrics/error_analysis_xgboost_exf2021.json`):

- **Confident false negatives**: `msftncsi`, `gstatic`, `googleapis`, `office`, `wireshark` — well-known
  Microsoft/Google service domains, exactly the kind of ubiquitous, trusted-looking name an attacker would
  deliberately pick to camouflage a channel. Both subclasses appear here.
- **Confident false positives**: `microsoft`, `windows`, `192` (numeric/IP-shaped), `atester`.

**CNN** (new today, `runs/metrics/error_analysis_cnn_exf2021.json`):

- **Confident false negatives**: `bing` (4 of the top 8), `microsoft` (×2), `gstatic` — the same
  trusted-service-name camouflage pattern XGBoost's FNs already showed, with `microsoft`/`gstatic`
  appearing in **both** models' confident-FN lists — a second independent architecture converging on the
  same camouflage domains as its most-confident mistakes.
- **Confident false positives**: `atester` and `local` (both also in XGBoost's FP list — see §8.1.4's
  near-total FP overlap between the two supervised models), plus a genuinely new pattern XGBoost's own FP
  list did not surface: four of the CNN's top 8 false positives are long NetBIOS-style encoded strings
  (`FHEPFCELEHFCEPFFFACACACACACACABN`, `EEEFFDELFEEPFACNDDEKEGDADEFEEDBM`, `EGEDECENEBEBEIFEFCEKEDFCEBCACAAA`)
  — legitimate NetBIOS-over-TCP/IP name-resolution traffic, which encodes hostnames into a half-ASCII
  scheme that is structurally dense and high-entropy-looking by design, for reasons that have nothing to
  do with exfiltration. This is a plausible, specific explanation for part of the CNN's FPR: a convolution
  over F2/F3 (randomness/structural-complexity) features is exactly the kind of model that would treat
  "looks encoded" as attack-like, and legitimate NetBIOS names genuinely do look encoded.

**Isolation Forest** (`runs/metrics/error_analysis_isoforest_exf2021.json`):

- **Confident false negatives**: every one of the top 8 has `sld = "192"` — a numeric/IP-shaped
  subdomain, the same pattern that showed up in XGBoost's false-*positive* list, now on the opposite side
  of a different model's error. Spans both subclasses.
- **Confident false positives**: `gov`, `microsoft` (×4), `local`, `wordpress`, `blogspot`. `microsoft`
  and `local` both also appear in the CNN's and/or XGBoost's FP lists — see §8.1.4.

**Autoencoder** (new today, `runs/metrics/error_analysis_autoencoder_exf2021.json`):

- **Confident false negatives**: every one of the top 8 has `sld = "224"` — the *same* numeric/IP-shaped
  pattern Isolation Forest's top false negatives showed (there, `sld = "192"`; here, `sld = "224"` — not
  the identical value, but the identical *shape*). **Two independent unsupervised architectures now each
  have their single most-confident false-negative pattern be a numeric-looking subdomain**, strong,
  triangulated evidence that numeric-shaped `sld` values read as "unremarkable" to density/reconstruction-
  based detectors specifically, regardless of the underlying algorithm.
- **Confident false positives**: `town` (×3), `city`, `gov`, `112` (numeric-shaped — notably on the
  opposite side from the FN pattern above, a reminder real data does not always fit a single clean
  narrative), `lww`, `blogspot`. `gov` and `blogspot` both also appear in Isolation Forest's FP list.

**Cross-model narrative, now with four models' worth of evidence:** `microsoft` appears in the confident-FP
lists of **three of four** models (XGBoost, CNN, Isolation Forest); `atester`/`local` link XGBoost and CNN
specifically (the two supervised models); numeric-shaped `sld` values appear on *both* sides of the
confusion matrix depending on which kind of model is looking at them — false positives for the supervised
models (XGBoost's `192`), false negatives for the unsupervised ones (Isolation Forest's `192`, the
Autoencoder's `224`). The same superficial feature (a numeric-looking hostname label) reads as suspicious
to a decision boundary trained to associate structural weirdness with attack, and as unremarkable to a
density/reconstruction method that has no attack examples to learn from and instead measures "how unusual
does this look relative to the bulk of the data" — two opposite verdicts from the same superficial cue,
depending entirely on which side of the supervised/unsupervised split the model sits.

## 8.1.4 Cross-model failure comparison

Every pairwise comparison across all four Dataset-A models, all reported against a **chance baseline** —
the overlap expected if the two models' errors were statistically independent, computed as
`n_model_a × (n_model_b / N)` where `N` is the total attack count (105,601) for false negatives or the
total benign count (115,714) for false positives. A raw overlap fraction alone does not distinguish
"these models fail for a related reason" from "these models both fail on a lot of rows, so some overlap
is inevitable" — exactly the same discipline as this report's large-*n* p-value caveat (§3.3).

| pair | error type | actual overlap | expected under independence | ratio |
|---|---|---|---|---|
| XGBoost vs. CNN | false negatives | 42 | 0.1 | **407× higher** |
| XGBoost vs. CNN | false positives | 46,775 | 18,954 | **2.47× higher** |
| XGBoost vs. Isolation Forest | false negatives | 22 | 54 | **2.4× lower** |
| XGBoost vs. Isolation Forest | false positives | 3,489 | 14,720 | **4.2× lower** |
| XGBoost vs. Autoencoder | false negatives | 40 | 57 | **1.4× lower** |
| XGBoost vs. Autoencoder | false positives | 1,128 | 2,310 | **2.1× lower** |
| Isolation Forest vs. CNN | false negatives | 15 | 174 | **11.6× lower** |
| Isolation Forest vs. CNN | false positives | 3,457 | 14,711 | **4.3× lower** |
| Isolation Forest vs. Autoencoder | false negatives | 97,731 | 95,452 | 1.02× (≈ chance) |
| Isolation Forest vs. Autoencoder | false positives | 5,327 | 1,793 | **3.0× higher** |
| CNN vs. Autoencoder | false negatives | 92 | 184 | **2.0× lower** |
| CNN vs. Autoencoder | false positives | 1,109 | 2,309 | **2.1× lower** |

**Four clear patterns emerge, each with a distinct explanation:**

1. **The two supervised models (XGBoost, CNN) fail on almost exactly the same rows, far beyond what their
   individually-tiny false-negative counts would predict by chance (407×).** Their false positives also
   agree well beyond chance (2.47×), matching the earlier XGBoost-vs-LogReg finding (99.2% raw FP overlap)
   with a *real* CNN result instead of a stand-in. Ensembling two models of this supervised, feature-
   weighted kind will not touch the FPR problem — it lives in what the F1–F3 feature set represents on
   this dataset, not in either model's individual decision boundary.
2. **Every pairing that crosses the supervised/unsupervised boundary (XGBoost/IsoForest, XGBoost/AE,
   IsoForest/CNN, CNN/AE) shows overlap *below* chance, often dramatically so** (11.6× lower for Isolation
   Forest vs. CNN). This is the clearest, most consistent evidence in this chapter that the two failure
   *sources* are structurally different: a learned decision boundary over 7 informative features misses
   different rows than a density-or-reconstruction method with no attack examples to learn from at all.
3. **The two unsupervised models' false negatives sit almost exactly at the chance baseline (1.02×) —
   not meaningfully correlated, once the baseline correction is applied.** The *raw* fraction here
   (97,731 of 97,738 Isolation Forest misses are also Autoencoder misses — 99.99%) looks like near-total
   agreement and would be easy to over-read as "these two models fail identically." The honest,
   chance-adjusted reading is different: both models miss such a large fraction of all attacks (92.6% and
   97.7% respectively) that this much raw overlap is close to what independence alone would produce. This
   is the sharpest illustration in the whole report of why the chance-baseline correction matters — the
   same raw number supports two very different conclusions depending on whether it is read against the
   right denominator.
4. **The two unsupervised models' false *positives*, by contrast, agree meaningfully more than chance
   (3.0×)** — unlike their false negatives. Both flag a shared subset of benign rows as unusual, which is
   consistent with them both ultimately relying on some notion of "distance from the bulk of the data,"
   even though one measures isolation depth and the other reconstruction error.

**The catch, carried over from the original XGBoost/Isolation-Forest finding and now confirmed against two
more models:** none of Isolation Forest's or the Autoencoder's favourable overlap statistics make either
one a usable peer detector on Dataset A — their absolute recall (7.45% and 2.34%) is far too low, and even
Isolation Forest's most permissive contamination sweep point (0.30) only reaches 11.88% recall (Step 2E).
The right reading stays what Step 2E and the original XGBoost/Isolation-Forest comparison already
concluded: these architectures' errors are usefully *different* where they do succeed, which is exactly
the first-stage-filter role Step 3C's cascade design assumes for them — and Chapter 8.4's real cascade
result on Dataset B (not Dataset A) already shows the limits of that assumption when the filter's own
recall is too low to carry the cascade.

## 8.1.5 Step 3B before/after table

Carried over from Step 3B (Dataset A, XGBoost, threshold optimisation for FPR reduction):

| | threshold | light recall | heavy recall | overall recall | overall F1 | FPR |
|---|---|---|---|---|---|---|
| Before | 0.50 | 99.95% | 99.94% | 99.95% | 81.82% | 40.48% |
| After | 0.70 | 98.04% | 98.02% | 98.03% | 81.16% | 39.72% |

FPR moved by well under one percentage point for a real cost in recall (2,023 additional false
negatives) — the honest negative result Step 3B reported, now sitting directly alongside all four models'
confusion matrices above for context: even the *best*-performing models on this dataset (XGBoost and CNN,
essentially tied) cannot threshold their way out of the FPR problem, which is the concrete evidence
Chapter 8.4's cascade rationale needs.

## 8.1.6 What's still open

- Exact Dataset B confusion-matrix integers for XGBoost (only F1 is recorded), and, once available, the
  same cross-model chance-baseline overlap analysis run on Dataset B — worth doing since Dataset B's
  near-perfect XGBoost/CNN separation makes for a very different starting point than Dataset A's, and
  §8.1.1 already shows the supervised/unsupervised gap is much narrower there.
- A chance-baseline recomputation for the original Step 3A XGBoost-vs-LogReg pair, now superseded by the
  real XGBoost-vs-CNN numbers above — the LogReg run itself is no longer part of this chapter's
  comparison set (dropped per the standing rule against keeping a stand-in alongside its real
  replacement), so this is a historical note rather than an open task.
