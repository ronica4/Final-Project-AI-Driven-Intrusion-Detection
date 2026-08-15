# Chapter 8.1 — Sample-Level Forensic Error Analysis

`evaluation/error_analysis.py` ran against all four architectures, both datasets, zero code changes —
harness is model-agnostic by construction.

## 8.1.1 Confusion matrices

**Dataset A — all four models** (out-of-fold, 5-fold CV, 221,315 rows):

| model | TN | FP | FN | TP | recall | FPR | F1 |
|---|---|---|---|---|---|---|---|
| XGBoost (t=0.50) | 68,868 | 46,846 | 58 | 105,543 | 99.95% | 40.48% | 81.82% |
| XGBoost (t=0.70, optimised) | 69,747 | 45,967 | 2,081 | 103,520 | 98.03% | 39.72% | 81.16% |
| CNN | 68,896 | 46,818 | 188 | 105,413 | 99.82% | 40.46% | 81.77% |
| Isolation Forest (contam.=0.20) | 79,354 | 36,360 | 97,738 | 7,863 | 7.45% | 31.42% | 10.50% |
| Autoencoder (benign-only, 95th-pct) | 110,007 | 5,707 | 103,131 | 2,470 | 2.34% | 4.93% | 4.34% |

Supervised models land almost on top of each other (F1 within 0.05, recall >99.8%, FPR ~40.5%) despite
unrelated architectures. Both unsupervised models fail the same direction (recall <8%); Autoencoder's
lower FPR isn't favourable — next to 2.34% recall, close to constant-negative.

**Dataset B — XGBoost:** TN=19,807, FP=0, FN=4, TP=19,803 — recall 99.98%, FPR **0.0000%**, F1 0.9999. 2
of 4 FNs route through `SourceIP=1.1.1.1` (Cloudflare's public DoH resolver), not the fixed testbed IPs
every other malicious row uses.

| model | framing | precision | recall | FPR | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| CNN | hard | 0.9956 | 0.9920 | 0.0044 | 0.9938 | 0.9995 |
| Isolation Forest | hard | 0.5951 | 0.2391 | 0.1627 | 0.3411 | 0.4611 |
| Isolation Forest | easy | 0.3441 | 0.3218 | 0.1671 | 0.3326 | 0.5572 |
| Autoencoder | hard | 0.8400 | 0.2681 | 0.0508 | 0.4064 | 0.7903 |

Dataset B: XGBoost/CNN stay excellent; unsupervised models better than on A (F1 0.34–0.41 vs. 0.10–0.04),
neither clears a useful threshold.

## 8.1.2 Per-subclass recall — heavy vs. light (Dataset A)

| model | heavy recall | light recall | overall recall | gap |
|---|---|---|---|---|
| XGBoost | 99.94% | 99.95% | 99.95% | −0.01 pp |
| CNN | 99.79% | 99.87% | 99.82% | −0.08 pp |
| Isolation Forest | 7.57% | 7.26% | 7.45% | +0.31 pp |
| Autoencoder | 2.41% | 2.24% | 2.34% | +0.17 pp |

All four agree: no meaningful heavy/light recall gap — fourth independent confirmation of Ch. 3's finding
that light/heavy sit at statistically equal distance from benign.

## 8.1.3 False negative / false positive forensics

`sld` pulled for interpretability only, never as a feature.

- **XGBoost** FNs: `msftncsi`, `gstatic`, `googleapis`, `office`, `wireshark` (trusted-name camouflage).
  FPs: `microsoft`, `windows`, `192` (numeric-shaped), `atester`.
- **CNN** FNs: `bing`, `microsoft`, `gstatic` (overlaps XGBoost). FPs: `atester`/`local` (shared), plus
  NetBIOS-style strings — legitimate, high-entropy by design.
- **Isolation Forest** FNs: top-8 all `sld="192"` — an XGBoost false *positive*. FPs: `gov`, `microsoft`,
  `local`, `wordpress`, `blogspot`.
- **Autoencoder** FNs: top-8 all `sld="224"` — same numeric pattern. FPs: `town`, `city`, `gov`, `112`,
  `blogspot`.

`microsoft` in 3/4 models' confident-FP lists; numeric-shaped `sld` sits on *both* sides of the confusion
matrix by model type — FP for supervised, FN for unsupervised: suspicious to a boundary trained on
structural weirdness, unremarkable to a density/reconstruction method with no attack examples.

## 8.1.4 Cross-model failure comparison

Every pairwise comparison (Dataset A) vs. a **chance baseline** (`n_a × n_b / N`) — raw overlap alone
can't distinguish "related" from "both fail a lot." Full 12-row table in Appendix D.4. **(1)** Supervised
models fail on nearly the same rows, far beyond chance (407×/2.47×) — ensembling won't fix FPR. **(2)**
every supervised/unsupervised pairing shows below-chance overlap — structurally different failure
sources. **(3)** unsupervised FNs sit almost exactly at chance (1.02×) despite deceptive 99.99% raw
overlap — both simply miss so much (92.6%/97.7%) this is near what independence alone produces. **(4)**
unsupervised FPs, unlike FNs, agree more than chance (3.0×) — both rely on "distance from the bulk of the
data." Neither is a usable peer detector on A (recall 7.45%, 2.34%), but their errors are usefully
*different* where they succeed — the first-stage-filter role the cascade (8.4) assumes.

## 8.1.5 Step 3B before/after table

| | threshold | light recall | heavy recall | overall recall | overall F1 | FPR |
|---|---|---|---|---|---|---|
| Before | 0.50 | 99.95% | 99.94% | 99.95% | 81.82% | 40.48% |
| After | 0.70 | 98.04% | 98.02% | 98.03% | 81.16% | 39.72% |

FPR moved under one point for a real cost (2,023 more FNs) — even A's best models can't threshold their
way out of the FPR problem, the evidence Ch. 8.4's cascade rationale needs.
