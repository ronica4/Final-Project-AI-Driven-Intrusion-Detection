# Chapter 3 — Exploratory Data Analysis

## 3.1 Class distribution

| Dataset | Framing | Benign | Attack | Ratio |
|---|---|---|---|---|
| A (EXF-2021) | n/a | 115,714 | 105,601 (62,918 heavy + 42,683 light) | ~1.10:1 |
| B (DoHBrw-2020) | hard | 19,807 | ~19,807 (subsampled from 249,836) | ~1:1 |
| B (DoHBrw-2020) | easy | 19,807 | 249,836 + ~98% non-DoH negatives | heavily imbalanced |

Light class retained at 100% (D5) — the analytical point. Easy framing's non-DoH traffic (ordinary HTTPS,
trivially separable) excluded from hard framing so it can't inflate the headline score (D8). Resampling
happens per-fold via SMOTE, never on the full dataset.

## 3.2 Statistical significance — Cliff's delta, not p-value

At n in the hundreds of thousands, Mann-Whitney p≈0 for nearly any feature, including ones with no
practical separation — **a p-value at this n proves nothing about effect size.** Every feature reported
with **Cliff's delta** (Romano et al. 2006: negligible <0.147, small <0.33, medium <0.474, large ≥0.474).

Dataset A (221,315 rows; 4 B_ONLY columns untestable): full table in Appendix D.1. **5 of 7 features carry
a large real effect (δ ≥0.474: vol_secondary +0.551, vol_total +0.564, rand_dispersion +0.591,
struct_segments +0.533); 2 (`rand_entropy` −0.125, `struct_max_segment` −0.145) do not**, despite both
being "significant" by p-value alone — corroborated in Ch. 4's gain importance. Dataset B, hard framing:
all 11 columns testable, 4 large, 6 medium — a richer profile, consistent with richer flow telemetry.

## 3.3 Correlation and multicollinearity

Any |r|>0.9 flags a Ch. 4 VIF candidate. Dataset A: `vol_secondary`↔`struct_segments` (r=0.926),
`rand_dispersion`↔`struct_segments` (r=0.905). Dataset B: 4 pairs flagged, including
`vol_primary`↔`struct_max_segment` (r=0.94) — a different pair than A's, noted in Ch. 5: the two vantage
points differ not just in what they observe but in which features are redundant with each other.

## 3.4 Three-way breakdown — benign vs. heavy vs. light (Dataset A)

Hypothesis: light attacks sit closer to benign (throttling to look normal). **Data does not bear this
out:**

| feature | Cliff's δ (light vs. benign) | Cliff's δ (heavy vs. benign) |
|---|---|---|
| vol_total | +0.567 | +0.561 |
| rand_dispersion | +0.590 | +0.592 |

Light and heavy sit at essentially **equal** distance from benign on 5 of 7 features, light's delta
marginally *larger* on several — opposite the throughput/stealth prediction. **Carried into Ch. 8:** on
stateless features, "light" describes payload weight, not stealth. Reframes Ch. 8.1's question from "why
do light attacks look more like benign" (they don't) to "why are any light attacks missed at all" (mostly
they aren't — light/heavy recall statistically indistinguishable).

## 3.5 Near-constant / redundancy audit

Dataset A: 4 B_ONLY columns exactly 100% NaN (`near_constant: true`, post-imputation only); all 7
testable columns have `nan_fraction: 0.0`, genuine variety (26–1,161 unique values). No dead columns. The
multicollinearity pairs above feed directly into Ch. 4's VIF analysis, independently corroborating this
chapter's finding.
