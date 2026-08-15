# Chapter 8.2b — Base-Rate Honesty

## 8.2b.1 Benchmark balance vs. production reality

| Setting | Attack base rate |
|---|---|
| Dataset A (EXF-2021) | 47.72% |
| Dataset B, hard framing | ~50% |
| Dataset B, easy framing | 21.41% |
| Realistic enterprise DNS traffic | ~0.01% |

Every other metric in this report is measured near 1:1 for statistical power (D5/D8) — ~5,000× denser in
attacks than a real network. Precision (unlike recall) is base-rate-dependent: at low prevalence, false
positives from the benign majority swamp true positives, regardless of how strong recall/AUC looked in
training.

## 8.2b.2 Production-scale extrapolation — Dataset A

10,000,000 queries/day, 1:10,000 base rate (1,000 attacks/day):

| | Before threshold tuning | After threshold tuning |
|---|---|---|
| False alerts/day | ≈4,048,000 | ≈3,972,000 |
| Analyst-facing precision | ≈0.025% | ≈0.025% |

Threshold tuning (Step 3B) barely moves the outcome — ~4M false alerts/day for ~1,000 true ones. A
detector with a merely 1% FPR at this scale would still generate ~100,000/day; this system runs at
roughly **40×** that FPR. Since before/after are nearly identical, the fix must be structural (Ch. 8.4).

## 8.2b.3 Dataset B tells a different story

Same extrapolation, Dataset B hard-framing rates (rule-of-three 95% upper-bound FPR ≤0.0151%, measured FP
count zero):

| | Dataset A — before | Dataset A — after | Dataset B — hard |
|---|---|---|---|
| False alerts/day | ≈4,048,000 | ≈3,972,000 | ≈1,515 |
| Analyst-facing precision | ≈0.025% | ≈0.025% | ≈39.8% |

At the same production scale, Dataset A is unusable while Dataset B hard-framing is close to SOC-viable —
the sharpest quantitative contrast in this report, consistent with every other chapter's finding that DoH
flow telemetry carries far more discriminating signal than plaintext per-query features. 10M/day and
1:10,000 are illustrative assumptions, not measured production data.
