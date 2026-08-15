## 8.4 The hybrid cascade

### Rationale

A single high-recall detector and a single high-precision detector solve different halves of the same
operational problem: a detector tuned purely for recall (Isolation Forest, unsupervised, no labels needed)
floods an analyst with false positives at any usable recall; a detector tuned for precision (XGBoost,
supervised, near-perfect on this feature set — Chapter 8.1/8.3) offers no protection against attack
patterns absent from its training distribution. A cascade's premise is to combine them so each stage does
only the part of the job it is cheap and reliable at: a fast, permissive first-stage filter discards the
bulk of obviously-benign traffic, a stronger second-stage model examines only the survivors, and a narrow
band of genuinely ambiguous cases escalates further rather than forcing a low-confidence guess. This is
also the project's designated slot for a future LLM arbiter (Step 3D) — deferred today per the plan's own
"skip unless everything else is done with real time to spare" instruction — so the cascade is built with
that third stage present in the code path, gracefully degrading to Stage 2's own verdict when 3D is absent
rather than requiring a structural change later.

### Architecture

```
  Test rows (n=7,923)
        │
        ▼
  Stage 1 — Isolation Forest (recall-tuned via select_cascade_contamination, max_tolerable_fpr=0.5)
        │  discard "obviously benign" (5,510 rows, 69.5%) ──► final verdict: benign, no further stage
        ▼
  survivors (2,413 rows)
        │
        ▼
  Stage 2 — XGBoost
        │  confident verdict (1,732 rows) ──► final verdict: XGBoost's own prediction
        ▼
  Stage 3 — escalation check: 0.35 ≤ P_xgb ≤ 0.65  OR  XGBoost/Autoencoder disagree
        │
        ▼
  escalated (681 rows) ──► Step 3D LLM arbiter (deferred) ──► falls back to XGBoost's own verdict
```

`ensemble/cascade.py`'s `run_cascade()` fits all three stage models once on an 80/20 stratified split
(`test_frac=0.2`), instruments rows-in/out/discarded and wall-clock fit/predict latency at every stage, and
reuses `models.unsupervised.select_cascade_contamination()` — the same recall-first, FPR-budget-constrained
selection rule already built for Step 2E.3 — to choose Stage 1's contamination rather than hand-picking one.
`individual_model_metrics_on_same_split()` then re-scores each of the three already-fitted stage models on
the *identical* held-out rows the cascade itself was scored on, so the cascade-vs-individual-model
comparison below is not confounded by a different split. 6/6 tests pass on synthetic data
(`tests/test_cascade.py`).

### Funnel table

Real run, Dataset B hard framing, 39,614 rows total, 7,923-row held-out test split
(`runs/metrics/cascade_dohbrw2020_hard.json`, diagram `runs/figures/cascade_funnel_dohbrw2020_hard.png`):

| Stage | Rows in | Outcome | Rows out | Fit latency | Predict latency (per row) |
|---|---|---|---|---|---|
| 1 — Isolation Forest (contamination=0.3, chosen by sweep) | 7,923 | discard 5,510 (69.5%) | 2,413 survive | 1.870 s | 27.68 μs |
| 2 — XGBoost | 2,413 | resolve confidently | 1,732 (71.8% of survivors) | 2.359 s | 5.72 μs |
| 3 — escalation (band or disagreement) | 2,413 | escalate | 681 (28.2% of survivors) | — | — |

Of the 681 escalations: **0 via the probability band, 681 via XGBoost/Autoencoder disagreement** — the
band contributed nothing on this run; every escalation came from the two models disagreeing outright. 681
is within the plan's qualitative target ("a few hundred, not tens of thousands") though above the
`max_llm_calls=200` budget reserved in `config.yaml` for Step 3D — moot since 3D is deferred, and noted
here rather than silently absorbed.

### Results — honest, including where the cascade loses

| | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR |
|---|---|---|---|---|---|---|
| **Cascade (end-to-end)** | 1.0 | 0.3380 | **0.5053** | 0.6690 | 0.5793 | 0.0 |
| Majority baseline | — | — | 0.6667 | — | — | — |
| Isolation Forest (standalone, same split) | 0.5549 | 0.3380 | 0.4201 | 0.5258 | 0.4591 | 0.2711 |
| **XGBoost (standalone, same split)** | 1.0 | 0.9997 | **0.9999** | 1.0 | 1.0 | 0.0 |
| Autoencoder (standalone, same split) | 0.8144 | 0.2492 | 0.3816 | 0.7614 | 0.7850 | 0.0568 |

**The cascade does not beat standalone XGBoost, and it sits below its own majority baseline (F1 0.505 vs.
0.667).** This is reported as the honest result rather than reframed — the plan's own instruction for this
step is to "report cascade F1/PR-AUC/FPR against each individual model honestly, even if the cascade
doesn't beat the best single model on its own."

### Diagnosis

The cascade's end-to-end recall (0.3380) is numerically identical to standalone Isolation Forest's recall on
the same split (0.3380) — not a coincidence, but the direct mechanism: any attack row Stage 1 discards is
gone forever, since no downstream stage ever sees it. The cascade's entire premise is a *cheap, high-recall*
first-stage filter; on this dataset and framing, even `select_cascade_contamination()`'s own recall-first
rule tops out at 33% recall within a 50%-FPR budget, because Isolation Forest itself is a weak detector here
(consistent with Chapter 6.1's design rationale and the Chapter 7.3 contamination sweep, which never reaches
a favourable recall/FPR turning point in the tested range). This is not a tuning mistake correctable by
picking a different contamination — the sweep's best-available point within budget was used — and not a
bug; it is a limitation of the cascade design under this specific framing, reported as such.

The latency premise fares no better: Stage 1 predicts at **27.68 μs/row**, nearly 5× *slower* than Stage 2
XGBoost's **5.72 μs/row** on survivors. Isolation Forest is not even cheaper than the model it is meant to
be shielding, so there is no compensating throughput win to offset the lost recall either — both of the
cascade's two justifications (cheap filtering, safety-net recall) fail to materialise on this run, and both
are stated plainly rather than one being highlighted and the other omitted.

### Bonus (B.1–B.4)

Not attempted. Step 3D (LLM arbiter) was deferred by design per the plan's own submission-day priority
(skip unless everything else is done with real time to spare), so the bonus criteria that depend on 3D
having run do not apply this submission.
