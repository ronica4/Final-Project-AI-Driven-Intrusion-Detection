# Chapter 8.3 — Benchmark Comparison Against Published Results

Four papers, one per model architecture (Ch. 6.2). Discrepancies attributed to specific named causes.

| Model | Published | Published metrics | This project | Direction |
|---|---|---|---|---|
| XGBoost | Abrahim et al. (2026), stacking ensemble, DoHBrw-2020 | Recall 0.9996, Precision 0.9967, F1 0.9981 | Dataset B hard: Recall 0.9998, F1 0.9999 (standalone) | matches/exceeds |
| CNN | Li et al. (2024), ITransformer-CNN, CIC-Bell-DNS | Accuracy 95.67%, Recall 83.21%, F1 88.43% | Dataset A: Recall 99.80%, F1 81.77% | recall higher, F1 lower |
| Isolation Forest | Wang et al. (2022), KRTunnel, mobile DNS *(unverified)* | Accuracy 98.1% | A: Recall 7.45%, F1 0.1050 · B hard: Recall 23.91%, Precision 59.51%, F1 0.3411 | far below |
| Autoencoder | De Bernardi et al. (2025) *(unverified)* | — | A: Recall 2.34%, Precision 30.06%, F1 0.0434 (AUC 0.2643) · B hard: Recall 26.81%, Precision 84.00%, F1 0.4064 | not comparable |

**XGBoost.** Our standalone model on B slightly exceeds Abrahim et al.'s full stacking ensemble — not the
same system (theirs: XGBoost as meta-learner atop LSTM/GRU, not standalone; corpus adds
DoH-Tunnel-Traffic-HKD). The plan's predicted pattern ("hard framing looks worse than published easy
framing") doesn't hold — hard framing already F1≈0.9999.

**CNN.** Recall dramatically higher (99.80% vs. 83.21%); F1 somewhat lower (81.77% vs. 88.43%) —
diagnosable: this CNN carries the same ~40% FPR wall every supervised model hits on A (Ch. 8.1); Li et al.
don't disclose Precision/FPR. Their model fuses a Transformer over domain-name sequences with a CNN over
traffic features — not a standalone CNN over an 11-column vector — the Transformer branch likely carries
most of their precision gain from sequence patterns this project's stateless schema (D1) can't see.

**Isolation Forest.** Starkest gap. Wang et al.'s 98.1% not independently verified (every fetch HTTP 403;
number from a search snippet) — but the gap's *direction* isn't in question: far below usable on both
datasets, diagnosed by Step 2E's core result and confirmed by the Autoencoder and Ch. 5's transfer collapse
— exfiltration isn't a minority density on either dataset (A's IsoForest ROC-AUC=0.2610, below chance).
KRTunnel's mobile context is plausibly a genuine low-base-rate setting these datasets (built for
statistical power, D5/D8) don't supply.

**Autoencoder.** No numeric comparison possible — every fetch (MDPI, ProQuest mirror) returned 403 or no
extractable body; citation confirmed via two sources, but dataset/config/metrics are not. Our own
Autoencoder is genuinely poor on both — below majority baseline on B, below-chance on A — echoing Isolation
Forest's diagnosis: two structurally unrelated unsupervised architectures both invert on Dataset A
specifically.

**Summary.** The plan's anticipated single explanation ("hard framing looks worse than published
easy-framing work") explains none of the four gaps. Real causes are more specific: an
ensemble-vs-standalone comparison, a feature-access asymmetry, a dataset-specific structural property
already diagnosed in Ch. 5/8.1, and one unverifiable benchmark — consistent with this report's D4
dual-framing design, built to let such a claim be checked rather than asserted.
