# Chapter 8.2b — Base-Rate Honesty

## 8.2b.1 Benchmark balance vs. production reality

Every other metric in this report is measured near 1:1 for statistical power (D5/D8) — ~5,000× denser in
attacks than a real network. Precision (unlike recall) is base-rate-dependent: at low prevalence, false
positives from the benign majority swamp true positives, regardless of how strong recall/AUC looked in
training. 10M/day and 1:10,000 below are illustrative assumptions, not measured production data.

| | A (EXF-2021) | B, hard | B, easy | Realistic enterprise |
|---|---|---|---|---|
| Attack base rate | 47.72% | ~50% | 21.41% | ~0.01% |

## 8.2b.2 Production-scale extrapolation (10,000,000 queries/day, 1:10,000 base rate, 1,000 attacks/day)

| | A — before tuning | A — after tuning | B — hard |
|---|---|---|---|
| False alerts/day | ≈4,048,000 | ≈3,972,000 | ≈1,515 |
| Analyst-facing precision | ≈0.025% | ≈0.025% | ≈39.8% |

Threshold tuning (Step 3B) barely moves A's outcome — ~4M false alerts/day for ~1,000 true ones; a
detector with a merely 1% FPR at this scale would still generate ~100,000/day, this system runs at
roughly **40×** that FPR. Since before/after are nearly identical, the fix must be structural (Ch. 8.4).
Dataset B hard-framing (rule-of-three 95% upper-bound FPR ≤0.0151%, measured FP count zero) tells a
different story: at the same production scale, A is unusable while B hard-framing is close to SOC-viable —
the sharpest quantitative contrast in this report, consistent with every other chapter's finding that DoH
flow telemetry carries far more discriminating signal than plaintext per-query features.
