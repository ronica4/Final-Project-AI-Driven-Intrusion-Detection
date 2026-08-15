# Chapter 8.4 — The Hybrid Cascade

## 8.4.1 Rationale

A single supervised model scores every row with the same expensive machinery, whether that row is
obviously benign or genuinely ambiguous. A real SOC does not work that way: cheap signals triage the
bulk of traffic, and only the hard cases reach an analyst (or, in this project's design, the most
expensive stage). `ensemble/cascade.py` implements that triage as three stages, chained so each does
what it is individually best at:

- **Stage 1 — Isolation Forest**, tuned for **recall** (Chapter 6.1, Chapter 7.3 sweep 2), reads every
  row and discards whatever it scores as unambiguously normal. A first-stage filter's failure mode
  (missing an attack, which no downstream stage can ever recover) is categorically worse than its
  success mode being merely permissive, so the contamination threshold is chosen by
  `select_cascade_contamination` to maximise recall within an FPR budget, not to maximise F1.
- **Stage 2 — XGBoost** scores the survivors. Confident predictions resolve directly.
- **Stage 3 — escalation** triggers when XGBoost's probability falls in the ambiguous band
  (`0.35 ≤ P_xgb ≤ 0.65`) **or** XGBoost and the Autoencoder disagree — two structurally independent
  models (Chapter 6.1) landing on different verdicts is itself a signal worth flagging, even when neither
  probability is individually uncertain.

Step 3D (an LLM arbiter that would resolve Stage 3's escalated rows) is deferred by default for this
submission (decision D7) given the deadline. This is not a gap left silently in the pipeline: per Step
3D's own documented graceful-degradation rule, escalated rows fall back to XGBoost's own verdict, which
`ensemble/cascade.py`'s module docstring states explicitly. Every metric reported below already reflects
that fallback.

## 8.4.2 Block diagram and funnel

![Cascade funnel, Dataset B hard framing](../../runs/figures/cascade_funnel_dohbrw2020_hard.png)

Run against real Dataset B, hard framing (39,614 rows total, `positive_rate=0.5000`, `families="full"`,
7,923-row held-out test split, single stratified 80/20 split rather than cross-validation — deliberate,
since per-stage wall-clock latency is the property being measured here, and CV's repeated refit-per-fold
has no equivalent meaning for a latency claim):

| Stage | Rows in | Rows out | Discarded/Resolved | Fit latency | Predict latency (per row) |
|---|---|---|---|---|---|
| 1 — Isolation Forest (contamination=0.30) | 7,923 | 2,413 survive | 5,510 discarded (69.5%) | 1.87 s | 27.7 μs |
| 2 — XGBoost | 2,413 | 1,732 confident | 681 escalate (28.2% of survivors) | 2.36 s | 5.7 μs |
| 3 — Escalation check | 2,413 | — | 681 escalated: **0 via probability band, 681 via XGBoost/Autoencoder disagreement** | (Autoencoder fit 31.9 s) | 23.9 μs (AE) |

Stage 1's contamination (0.30) was selected by `select_cascade_contamination(max_tolerable_fpr=0.5)` —
reusing Step 2E's own cascade-threshold rule rather than hand-picking a value — at recall 0.331 / FPR
0.268. 681 escalations sits inside the plan's target of "a few hundred, not tens of thousands," though
above the 200-call budget `config.yaml` reserves for a Step 3D LLM arbiter; moot, since 3D did not run
this submission, and every escalated row's final verdict is XGBoost's own prediction regardless. Every
escalation in this run came from model disagreement rather than probability-band ambiguity, meaning
XGBoost's own confidence was rarely in the [0.35, 0.65] band at all — consistent with its near-perfect
standalone performance (below).

## 8.4.3 Results — cascade vs. individual models, same held-out rows

`individual_model_metrics_on_same_split()` scores each of the three already-fitted stage models
(Isolation Forest, XGBoost, Autoencoder) on the *identical* 7,923 test rows the cascade itself was
scored on, so this is not confounded by different splits.

| | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR |
|---|---|---|---|---|---|---|
| **Cascade (end-to-end)** | 1.0 | 0.3380 | **0.5053** | 0.6690 | 0.5793 | 0.0 |
| Isolation Forest (alone) | 0.5549 | 0.3380 | 0.4201 | 0.5258 | 0.4591 | 0.2711 |
| **XGBoost (alone)** | 1.0 | 0.9997 | **0.9999** | 1.0 | 1.0 | 0.0 |
| Autoencoder (alone) | 0.8144 | 0.2492 | 0.3816 | 0.7614 | 0.7850 | 0.0568 |
| Majority-class baseline | — | — | 0.6667 | — | — | — |

**The cascade does not beat standalone XGBoost — it loses to it by 0.49 F1, and it loses to its own
majority-class baseline (0.5053 vs. 0.6667).** Reported honestly rather than reframed around a favourable
metric, per this chapter's own instruction (Step 3C point 4: "if the cascade does not beat the best
single model, report that honestly and diagnose why").

## 8.4.4 Diagnosis

The cascade's end-to-end recall (0.3380) is **numerically identical** to standalone Isolation Forest's
recall on the same split (0.3380). This is not a coincidence, and it is algebraic, not statistical: Stage
1 is an irreversible filter. Every row Isolation Forest discards is permanently gone from every
downstream stage — Stage 2's near-perfect XGBoost (F1 0.9999 alone) never gets the chance to correctly
classify a row Stage 1 already threw away. The cascade's ceiling is therefore exactly Stage 1's own
recall, no matter how good Stage 2 is.

The cascade's entire premise (§8.4.1) is a *cheap, high-recall* first-stage filter. On Dataset B's hard
framing specifically, that premise does not hold: Chapter 7.3's sensitivity sweep shows
`select_cascade_contamination`'s own recall-maximising selection rule tops out at 33% recall within a
50%-FPR budget — not because a better contamination value was overlooked (the sweep's best-available
point was used), but because Isolation Forest is a structurally weak detector on this near-1:1 balanced
framing (Chapter 6.1, Chapter 7.3: "attack" is not a minority anomaly here, so an unsupervised outlier
detector has no principled way to prefer one half of an evenly-weighted, roughly-bimodal distribution as
"the" anomalies). This is a limitation of the cascade design under this specific framing, not a bug to
patch or a tuning mistake to fix.

**The latency premise fares no better, on an entirely independent axis.** Stage 1 predicts at ~27.7
μs/row versus Stage 2 XGBoost's ~5.7 μs/row on survivors — Isolation Forest is not even cheaper per row
than the model it is supposed to be shielding, so there is no compensating throughput win to offset the
lost recall either. A cascade that traded some F1 for a real latency reduction would be a legitimate
engineering result even without beating standalone XGBoost on accuracy; this cascade does not clear that
bar, and saying so plainly is the honest reading of the numbers rather than a favourable-metric-only
summary.

**What this result is evidence for.** Isolation Forest's cheap-anomaly-detection assumption — that
attacks are rare and isolate faster than a dense normal majority — genuinely does not hold on Dataset B's
hard, artificially-balanced framing, and the cascade's collapse is the clearest demonstration of that
limit anywhere in this project (compare Chapter 6.1's and Chapter 7.3's sensitivity-sweep interpretation
of the same underlying cause). The cascade's practical value, if any, would depend on a framing closer to
production's real base rate (Chapter 8.2b) — where "attack" genuinely is a minority anomaly — which this
submission's hard framing was deliberately constructed *not* to be (D4/D8).

## Bonus (B.1–B.4)

Not applicable. Step 3D (the LLM arbiter) was deferred by default (D7) given the submission deadline, so
there is no LLM-arbitrated subset to report bonus criteria against. Escalated rows fall back to
XGBoost's own verdict throughout §8.4.3's numbers.
