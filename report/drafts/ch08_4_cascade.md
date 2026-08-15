# Chapter 8.4 — The Hybrid Cascade

## 8.4.1 Rationale

A single supervised model scores every row with the same machinery regardless of how obvious it is.
`ensemble/cascade.py` triages instead: **Stage 1 — Isolation Forest**, tuned for **recall**, discards
whatever it scores unambiguously normal (a first-stage miss is unrecoverable downstream, so contamination
maximises recall within an FPR budget, not F1). **Stage 2 — XGBoost** scores survivors; confident
predictions resolve directly. **Stage 3 — escalation** triggers when XGBoost's probability falls in
`0.35≤P≤0.65` **or** XGBoost/Autoencoder disagree. Step 3D (LLM arbiter) deferred by default (D7);
escalated rows fall back to XGBoost's own verdict.

## 8.4.2 Funnel and results

Real Dataset B, hard framing (39,614 rows, 7,923-row held-out split; single stratified 80/20, not CV,
since per-stage latency is being measured):

| Stage | Rows in | Rows out | Fit latency | Predict latency (per row) |
|---|---|---|---|---|
| 1 — Isolation Forest (contam.=0.30) | 7,923 | 2,413 survive (5,510 discarded, 69.5%) | 1.87 s | 27.7 μs |
| 2 — XGBoost | 2,413 | 1,732 confident, 681 escalate (28.2%) | 2.36 s | 5.7 μs |
| 3 — Escalation | 2,413 | 681 escalated: 0 via probability band, 681 via disagreement | AE fit 31.9 s | 23.9 μs (AE) |

Contamination (0.30) selected by `select_cascade_contamination(max_tolerable_fpr=0.5)`, recall
0.331/FPR 0.268; 681 escalations sits within the "few hundred" target.

| | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR |
|---|---|---|---|---|---|---|
| **Cascade (end-to-end)** | 1.0 | 0.3380 | **0.5053** | 0.6690 | 0.5793 | 0.0 |
| Isolation Forest (alone) | 0.5549 | 0.3380 | 0.4201 | 0.5258 | 0.4591 | 0.2711 |
| **XGBoost (alone)** | 1.0 | 0.9997 | **0.9999** | 1.0 | 1.0 | 0.0 |
| Autoencoder (alone) | 0.8144 | 0.2492 | 0.3816 | 0.7614 | 0.7850 | 0.0568 |
| Majority-class baseline | — | — | 0.6667 | — | — | — |

**The cascade loses to standalone XGBoost by 0.49 F1, and to its own majority-class baseline.**

## 8.4.3 Diagnosis

Cascade end-to-end recall (0.3380) is numerically identical to standalone Isolation Forest's recall on the
same split: Stage 1 is irreversible, so the cascade's ceiling equals Stage 1's own recall regardless of
Stage 2's quality. Fails on B's hard framing because Isolation Forest is structurally weak on a near-1:1
balanced framing — "attack" isn't a minority anomaly here, so an unsupervised outlier detector has no
principled way to prefer one half of an evenly-weighted, bimodal distribution. Latency premise fails too:
Stage 1 predicts at ~27.7 μs/row vs. Stage 2's ~5.7 μs/row — not even cheaper than the model it shields.

**What this result is evidence for.** Isolation Forest's cheap-anomaly-detection assumption — attacks are
rare and isolate faster than a dense normal majority — does not hold on B's artificially balanced hard
framing; the cascade's collapse is the clearest demonstration of that limit here. Practical value would
depend on a framing closer to production's real base rate (Ch. 8.2b), which this submission's hard framing
was deliberately constructed *not* to be (D4/D8).

## Bonus (B.1–B.4)

Not applicable. Step 3D (LLM arbiter) was deferred (D7); no LLM-arbitrated subset exists to report bonus
criteria against.
