# Chapter 6 — Model Architectures

## 6.1 Per-model justification

Four architectures span the relevant inductive biases — axis-aligned rules, local-pattern convolution,
density-based isolation, reconstruction-based novelty.

**XGBoost — axis-aligned splits match threshold-like exfiltration signals.** `feature ≤ threshold` splits
match F1/F3's physics (a payload does or doesn't exceed a normal lookup; a label does or doesn't approach
the 63-byte limit). `scale_pos_weight` gives cost-sensitive reweighting without synthesising rows.
Primary supervised detector — tree ensembles are the natural fit for tabular, threshold-driven telemetry,
matching Mahdavifar et al.'s own best classical model (Random Forest).

**1D-CNN — local receptive field over a family-ordered vector.** `UNIFIED_COLUMNS` groups the 11 features
by family, so a kernel-3 convolution (two blocks, 32/64 filters) learns within-family interactions as
*local* structure — an MLP would need to learn each as a global weight pattern from scratch. BatchNorm +
dropout (0.3) regularise against 11 inputs being a narrow signal.

**Isolation Forest — cheap, high-recall isolation depth as a first-stage filter.** Scores by how few random
splits isolate a point; no labels needed, only that anomalies are few and different. Fits exactly one
role: cascade Stage 1 — a cheap pre-filter before a supervised model, deliberately not the primary
detector. Recall-first tuning (`select_cascade_contamination`) exists because a first-stage miss is
unrecoverable downstream; Ch. 8.4's Dataset B result (best recall 33%) is the honest limit case.

**Autoencoder — benign-only reconstruction manifold.** `11→8→4→8→11`, fit exclusively on benign rows;
reconstruction error above the 95th-percentile benign threshold needs no attack examples at fit time —
matters for detecting a tool the training data never saw. Kept in the cascade (Stage 3) as an independent
signal for the XGBoost/Autoencoder disagreement check.

## 6.2 Literature benchmark table

See `ch06_2_benchmark_papers.md` — four architecture-specific papers, reused in Ch. 8.3.
