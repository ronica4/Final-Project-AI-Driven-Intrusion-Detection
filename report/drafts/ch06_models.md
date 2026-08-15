# Chapter 6 — Model Architectures

## 6.1 Per-model mathematical justification

Four architectures were chosen to span the space of relevant inductive biases — axis-aligned rule
learning, local-pattern convolution, density-based isolation, and reconstruction-based novelty — rather
than four arbitrary points in model-space. Each is justified below against the specific signal the
unified schema (Chapter 5.1) is designed to expose.

**XGBoost — axis-aligned splits match threshold-like exfiltration signals.**
Gradient-boosted trees partition feature space with axis-aligned splits (`feature ≤ threshold`), which is
exactly the functional form the unified schema's features were engineered to produce a signal in: F1's
volume features and F3's structural-complexity features are threshold-like by the physics of the attack
itself (a payload of size N either does or does not exceed what a benign lookup plausibly needs; a DNS
label either does or does not approach the 63-byte carrier limit), not smoothly varying signals that need
a learned nonlinear boundary to separate. `scale_pos_weight` (rather than SMOTE — see Chapter 7.2) gives
XGBoost native cost-sensitive reweighting for class imbalance without synthesising rows, matching the
project's "resampling xor reweighting, never both" discipline (`models/supervised.py`). This is the
project's primary supervised detector precisely because tree ensembles are the natural match for
tabular, mixed-scale, threshold-driven security telemetry — the same reasoning the two Chapter 2 papers'
own best-performing classical models (Random Forest in Mahdavifar et al.) independently arrive at.

**1D-CNN — local receptive field over a family-ordered vector.**
`schema/unified.py`'s column order is not incidental: `UNIFIED_COLUMNS` groups the 11 features by
family (F1's three volume columns adjacent, then F2's two entropy columns, then F3, F4, F5), specifically
so a 1D convolution's local receptive field (kernel size 3, two conv blocks at 32/64 filters,
`padding="same"`) can learn within-family interaction patterns — e.g. a volume/dispersion relationship
inside F1, or a rhythm/dispersion relationship spanning F4 into F5 — as *local* structure, the way a CNN
over a 1D signal is designed to exploit. A supervised model with no notion of spatial locality (e.g. a
plain MLP over the same 11 numbers) would have to learn every such interaction as a global weight
pattern from scratch; ordering the input by family gives the convolution's inductive bias something real
to latch onto. BatchNorm and dropout (0.3) regularise against the comparatively small feature count (11
inputs is a narrow signal for a two-block conv architecture) overfitting to noise in any single fold.

**Isolation Forest — cheap, high-recall isolation depth as a first-stage filter.**
Isolation Forest's mechanism — recursively partition feature space with random splits, and score a point
by how few splits it takes to isolate it — requires no labels and no assumption about what the attack
class looks like, only that anomalies are *few and different* and therefore isolate faster than the
dense majority. This is the right fit for exactly one specific role in this project: Step 3C's cascade
Stage 1, a cheap pre-filter whose job is to discard the "obviously normal" bulk of traffic before a more
expensive supervised model examines the survivors (`models/unsupervised.py`). It is deliberately *not*
the project's primary detector — `select_cascade_contamination`'s own recall-first selection criterion
exists because a first-stage filter's failure mode (missing an attack, which no downstream stage can
ever recover) is categorically worse than its success mode being merely permissive (over-flagging costs
the next stage some extra work, not a lost detection). Step 3C's real result on Dataset B hard framing
(Chapter 8.4) is the honest limit case of this reasoning: when even the best-available recall/FPR
tradeoff point tops out at 33% recall, the cascade's entire premise fails, which is itself evidence about
where Isolation Forest's cheap-anomaly-detection assumption does and does not hold.

**Autoencoder — benign-only reconstruction manifold for detecting novel tools.**
The `11→8→4→8→11` autoencoder (`models/deep.py`) is fit exclusively on benign training-fold rows,
learning a compressed manifold of what *normal* traffic looks like rather than what *attack* traffic
looks like. Its detection signal — reconstruction error above the 95th-percentile benign threshold — is
therefore structurally different from every other model in the project: it requires no attack examples
at fit time at all, which is precisely the property that matters for detecting a tool the training data
never saw. XGBoost and the CNN can only ever recognise attack patterns present (or interpolatable) in
their training distribution; a benign-only reconstruction model's failure mode is instead "did this row
look like normal traffic," which generalises differently and is the reason it is retained in the cascade
(Step 3C, Stage 3) as a second, structurally independent signal for the XGBoost/Autoencoder disagreement
check, rather than folded into the same supervised-vs-supervised comparison as XGBoost and the CNN.

## 6.2 Literature benchmark table

See `report/drafts/ch06_2_benchmark_papers.md` for the four architecture-specific papers (one per model:
XGBoost, 1D-CNN, Isolation Forest, Autoencoder) applied to DNS tunneling/exfiltration, with their
reported Recall/FPR/F1, collected once here for reuse in Chapter 8.3's benchmark comparison.
