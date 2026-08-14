# Chapter 8.1 — Sample-Level Forensic Error Analysis

> Status (14 Aug 2026): XGBoost and Isolation Forest are complete on Dataset A (both real, both
> out-of-fold), plus Dataset B summary metrics for both from Teammate B's backfills. A LogReg run stands
> in for the CNN in the cross-model comparison until Step 2F's real CNN/Autoencoder results land on this
> machine — clearly labelled every place it appears, per Step 3A's original convention, and to be
> **replaced, not merely supplemented**, once real CNN/AE forensics exist. Isolation Forest forensics
> (per-subclass recall, FN/FP pull, cross-model overlap vs. XGBoost) are new since the last plan version —
> added independently while blocked on B, using the same `evaluation/error_analysis.py` harness Step 3A
> already built, which turned out to need zero changes to run against a second model.

## 8.1.1 Confusion matrices

**Dataset A — exact counts** (both from real out-of-fold predictions, 5-fold CV, 221,315 rows):

| model | TN | FP | FN | TP | recall | FPR | F1 |
|---|---|---|---|---|---|---|---|
| XGBoost (t=0.50, Step 2D/3B baseline) | 68,868 | 46,846 | 58 | 105,543 | 99.95% | 40.48% | 81.82% |
| XGBoost (t=0.70, Step 3B optimised) | 69,747 | 45,967 | 2,081 | 103,520 | 98.03% | 39.72% | 81.16% |
| Isolation Forest (contamination=0.20, locked config) | 79,354 | 36,360 | 97,738 | 7,863 | 7.45% | 31.42% | 10.50% |

**Dataset B — reported metrics** (Teammate B's backfills; exact integer confusion matrices not yet in
hand, only the aggregated precision/recall/FPR/F1 recorded in the plan — noted here as metrics, not
reconstructed into counts, to avoid presenting a derived number as an original one):

| model | framing | precision | recall | FPR | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| XGBoost | hard | — | — | — | ≈0.9999 | — |
| Isolation Forest | hard | 0.5951 | 0.2391 | 0.1627 | 0.3411 | 0.4611 |
| Isolation Forest | easy | 0.3441 | 0.3218 | 0.1671 | 0.3326 | 0.5572 |

*(XGBoost's Dataset B row only has F1 recorded so far — precision/recall/FPR/ROC-AUC weren't kept in the
tracker's one-line backfill summary, only F1≈0.9999. Same open item as Ch 8.2b's missing Dataset B FPR
column.)*

**CNN / Autoencoder — PENDING, both datasets.** Not run on this machine; Step 2F's Dataset B write-up is
in progress on B's end, Dataset A backfill deliberately on hold per her request until that lands.

## 8.1.2 Per-subclass recall — heavy vs. light (Dataset A)

| model | heavy_attack recall | light_attack recall | overall recall | gap (heavy − light) |
|---|---|---|---|---|
| XGBoost | 99.94% (62,881/62,918) | 99.95% (42,662/42,683) | 99.95% | **−0.01 pp** (light marginally higher) |
| LogReg (CNN stand-in) | 99.61% (62,670/62,918) | 99.72% (42,565/42,683) | 99.65% | −0.11 pp |
| Isolation Forest | 7.57% (4,764/62,918) | 7.26% (3,099/42,683) | 7.45% | +0.31 pp |

All three models tell the **same story** on the heavy-vs-light question, despite being wildly different
in overall performance: **there is no meaningful heavy/light recall gap on Dataset A, in either
direction, for any model tested.** This is now triangulated a third way (Step 3A: two supervised models;
here: one unsupervised model added), and confirms Chapter 3's own finding that light and heavy attacks
sit at statistically equal distance from benign on Dataset A's F1–F3 features (§3.5) — three independent
methods (two classifiers' recall, one anomaly detector's recall, and the underlying univariate effect
sizes) now agree the light-class-blind-spot hypothesis from Chapter 2 simply does not hold on this
dataset, on these features.

## 8.1.3 False negative / false positive forensics

**XGBoost** (Step 3A, `runs/metrics/error_analysis_xgboost_exf2021.json`) — pulled `sld` for
interpretability only, never as a feature:

- **Confident false negatives** (real attacks the model was ≥99.9% sure were benign): `msftncsi`,
  `gstatic`, `googleapis`, `office`, `wireshark` — well-known Microsoft/Google service domains, exactly
  the kind of ubiquitous, trusted-looking name an attacker would deliberately pick to camouflage a
  channel. Both subclasses appear here, so this isn't a light-only phenomenon.
- **Confident false positives** (benign traffic the model was ≥83% sure was attack): `microsoft`,
  `windows`, `192` (a numeric/IP-shaped subdomain), `atester`.

**Isolation Forest** (added today, `runs/metrics/error_analysis_isoforest_exf2021.json`):

- **Confident false negatives** (real attacks scored as the *least* anomalous in the entire dataset):
  every one of the 8 highest-confidence misses has `sld = "192"` — the same numeric/IP-shaped subdomain
  pattern that showed up in XGBoost's false-*positive* list above, now on the opposite side of a
  different model's error. Spans both `heavy_attack` and `light_attack` rows. A numeric-looking `sld`
  apparently reads as unremarkable to a density-based detector regardless of which class it actually
  belongs to — consistent with Isolation Forest's core failure mode here (§2E: attack is not a minority
  density on Dataset A, so "looks statistically ordinary" doesn't imply "is benign").
- **Confident false positives** (benign traffic scored as *most* anomalous): `gov`, `microsoft` (×4),
  `local`, `wordpress`, `blogspot`. **`microsoft` appears in both models' top false-positive lists** —
  one concrete, specific point of agreement between two structurally very different models, worth naming
  even though their false-positive sets barely overlap in aggregate (§8.1.4).

## 8.1.4 Cross-model failure comparison

Two independent pairwise comparisons now exist for Dataset A. Both are reported against a **chance
baseline** — the overlap you'd expect if the two models' errors were statistically independent — because
a raw overlap fraction alone doesn't say whether two models are failing on the *same* rows for a
*related* reason, or coincidentally.

| pair | error type | n (model A) | n (model B) | actual overlap | overlap vs. chance |
|---|---|---|---|---|---|
| XGBoost vs. LogReg (Step 3A) | false negatives | 58 | 366 | 40 (10.4% of union) | *(chance baseline not computed in 3A)* |
| XGBoost vs. LogReg (Step 3A) | false positives | 46,846 | 46,901 | 46,692 (99.2% of union) | *(chance baseline not computed in 3A)* |
| XGBoost vs. Isolation Forest (new) | false negatives | 58 | 97,738 | 22 | **2.4× lower** than the ~54 expected if XGBoost's rare misses were a random subset of all attacks |
| XGBoost vs. Isolation Forest (new) | false positives | 46,846 | 36,360 | 3,489 | **4.2× lower** than the ~14,720 expected under independence |

**This is a real, quantified, and non-obvious result, not just a repeat of Step 3A's finding under a new
model name:**

- **XGBoost vs. LogReg** (two supervised, feature-weighted models): false negatives barely overlap
  (10%) — they miss largely different attacks, good news for an ensemble's recall. False positives
  overlap almost totally (99%) — they flag nearly the same benign rows as suspicious, so ensembling two
  models of this kind will not touch the FPR problem, because the issue lives upstream in what the
  F1–F3 feature set represents, not in either model's individual decision boundary (Step 3A's original
  conclusion).
- **XGBoost vs. Isolation Forest** (supervised vs. unsupervised/density-based): both failure modes
  overlap **less than chance would predict**, not merely "not much." The 58 attacks XGBoost misses are
  disproportionately attacks Isolation Forest actually flags as anomalous — genuinely complementary
  recall behaviour, more so than the XGBoost/LogReg pairing. And the two models' false positives are
  substantially *different* benign rows, unlike the XGBoost/LogReg near-total FP agreement — a
  structurally different error source (density-based outlierness vs. a learned decision boundary) really
  does produce a structurally different set of mistakes.
- **The catch, and it's a real one:** Isolation Forest's favourable overlap statistics don't make it a
  usable peer detector on Dataset A — its absolute recall is 7.45% against XGBoost's 99.95%, and even at
  the most permissive contamination in Step 2E's sweep (0.30), sweep recall only reaches 11.88%, meaning
  ~88% of real attacks are discarded before reaching a second stage. The right reading is not "add
  Isolation Forest as a voting member," it's "Isolation Forest's errors are at least *usefully
  different* where it does succeed" — consistent with, and now with sharper numbers behind, Step 3C's
  planned role for it as a first-stage filter rather than a peer classifier, and consistent with Step
  2E's own conclusion that its outlier premise is structurally broken on Dataset A specifically. Step 3C
  should account for this directly: Dataset A's cascade cannot lean on Isolation Forest's recall the way
  the cascade design implicitly assumes for Dataset B's easier framing.

**CNN vs. XGBoost / CNN vs. Isolation Forest — PENDING**, once Step 2F's real Dataset A results exist;
the LogReg-standin comparison above should be dropped from the final chapter at that point, not kept
alongside the real one.

## 8.1.5 Step 3B before/after table

Carried over from Step 3B (Dataset A, XGBoost, threshold optimisation for FPR reduction):

| | threshold | light recall | heavy recall | overall recall | overall F1 | FPR |
|---|---|---|---|---|---|---|
| Before | 0.50 | 99.95% | 99.94% | 99.95% | 81.82% | 40.48% |
| After | 0.70 | 98.04% | 98.02% | 98.03% | 81.16% | 39.72% |

FPR moved by well under one percentage point for a real cost in recall (2,023 additional false
negatives) — the honest negative result Step 3B reported, now sitting directly alongside the two other
models' confusion matrices above for context: even the *best*-performing model on this dataset cannot
threshold its way out of the FPR problem, which is the concrete evidence Chapter 8.4's cascade rationale
needs.

## 8.1.6 What's still open

- CNN and Autoencoder confusion matrices, per-subclass recall, and FN/FP forensics — Dataset A backfill
  intentionally on hold until B's Dataset B write-up for Step 2F lands (to avoid both of us editing the
  same PROJECT_PLAN.md section simultaneously).
- Exact Dataset B confusion-matrix integers for XGBoost (only F1 is recorded) and, once available, the
  same cross-model chance-baseline overlap analysis run on Dataset B — worth doing once all four models
  exist there, since Dataset B's near-perfect XGBoost separation (F1≈0.9999) makes for a very different
  starting point than Dataset A's.
- A chance-baseline recomputation for the two Step 3A pairs (XGBoost vs. LogReg) for consistency with
  the new pairs above — not done in the original 3A writeup, worth adding once assembling the final
  chapter so all four rows in §8.1.4's table are on equal footing.
