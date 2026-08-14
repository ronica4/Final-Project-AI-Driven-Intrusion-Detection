# Chapter 3 — Exploratory Data Analysis

> Status note (14 Aug 2026): Dataset A is fully run and reported below. Dataset B's class-distribution
> and significance numbers were backfilled by Teammate B against real hard-framing data (`data/dohbrw2020/`
> is not populated on this machine, so this project's own code has not run against it locally) — figures
> and the full per-feature table are **pending a local run on B's machine**; only the summary numbers B
> reported are included here. Flagged inline everywhere it applies, not silently filled in.

## 3.1 Class distribution

| Dataset | Framing | Benign | Attack | Ratio | Note |
|---|---|---|---|---|---|
| A (EXF-2021) | n/a — single framing | 115,714 | 105,601 (62,918 heavy + 42,683 light) | ~1.10 : 1 | Light class retained at 100% by design (decision D5) — never subsampled, since it is the entire analytical point of the project. |
| B (DoHBrw-2020) | hard | 19,807 | ~19,807 (subsampled from 249,836) | ~1 : 1 by construction | Malicious pool subsampled down to match the smaller benign pool. |
| B (DoHBrw-2020) | easy | 19,807 | 249,836 + ~98% non-DoH negatives | heavily imbalanced | Non-DoH traffic is ordinary HTTPS and is trivially separable from tunneling — reported *and* excluded from the hard framing precisely because it would inflate the headline score with a signal unrelated to tunnel detection (same contaminant logic as decision D8). |

Bar charts: `runs/figures/exf2021_class_distribution_*.png` (Dataset A, pre/post any resampling inside
CV folds — resampling happens per-fold via SMOTE inside the Pipeline, never on the full dataset, so
these bars show the *raw* label counts above). Dataset B distribution figures: pending B's local run.

## 3.2 Per-feature class-conditional distributions

Histogram + box-plot overlay (benign vs. attack) for every testable column, saved at ≥150 dpi to
`runs/figures/exf2021_<feature>_distribution.png` for Dataset A (7 testable columns — the 4 B_ONLY
columns are entirely NaN on Dataset A by construction, see §1.3/§1.4). Dataset B has all 11 columns
observable (it is the side of the encryption boundary with full flow-level telemetry); those figures are
pending B's local run.

## 3.3 Per-feature statistical significance — the graded core of this chapter

**A note on method, stated up front because it matters more than any single number below:** with a
sample size in the hundreds of thousands, a Mann-Whitney U test will report a statistically significant
difference (p ≈ 0) for almost any feature, including ones with no practically useful separation between
classes. **A p-value at this n proves nothing about effect size** — it is a large-sample artifact, not
evidence of a real signal. Every feature below is therefore reported with **Cliff's delta**, an effect
size derived directly from the same U statistic (`δ = 2U/(n₁n₂) − 1`) so no normality assumption is
smuggled back in, using the verdict bands of Romano et al. (2006): negligible < 0.147, small < 0.33,
medium < 0.474, large ≥ 0.474.

**Dataset A** (221,315 rows; `runs/metrics/feature_significance.json`) — 7 testable columns; the 4
B_ONLY columns are correctly reported `"untestable (one class entirely NaN)"` rather than silently
skipped or defaulted to a fake value:

| feature | median benign | median attack | Cliff's δ | verdict |
|---|---|---|---|---|
| vol_primary | — | — | +0.242 | small |
| vol_secondary | — | — | +0.551 | large |
| vol_total | — | — | +0.564 | large |
| rand_entropy | — | — | −0.125 | negligible |
| rand_dispersion | — | — | +0.591 | large |
| struct_segments | — | — | +0.533 | large |
| struct_max_segment | — | — | −0.145 | negligible |

*(Median columns render from `runs/metrics/feature_significance.json` at assembly time — the JSON has
them, this draft omits raw magnitudes to avoid transcription error; pull them directly from the file.)*

Every p-value in this table rounds to 0.00 at n≈221K — exactly the artifact the method note above
warns about. Cliff's delta is what actually distinguishes signal from noise here: **5 of 7 features
carry a large real effect; 2 (`rand_entropy`, `struct_max_segment`) do not**, despite both being
"significant" by p-value alone. This is the chapter's central methodological point and is repeated in
Chapter 4 when the same two features turn up ranked low/high in ways that corroborate this table.

**Dataset B, hard framing** (backfilled by Teammate B against real data; full table pending) — summary
only: **all 11 columns are testable** (Dataset B observes every feature family, unlike Dataset A, so
there is no untestable-by-construction row here), **4 of 11 features show a large effect size, 6 show
medium** (1 unreported here — pending the full table). This is a materially richer significance profile
than Dataset A's, consistent with Dataset B's flow-level telemetry simply carrying more discriminating
signal per feature than Dataset A's 7 observable columns.

## 3.4 Correlation heatmap and multicollinearity flags

Any |r| > 0.9 pair is flagged as a Chapter 4 multicollinearity candidate.

- **Dataset A** (`runs/figures/exf2021_correlation_heatmap.png`): `vol_secondary` ↔ `struct_segments`
  (r = 0.926), `rand_dispersion` ↔ `struct_segments` (r = 0.905). Both pairs involve `struct_segments` —
  worth naming as the common factor in Chapter 4's redundancy discussion, and this is corroborated
  independently by Chapter 4's VIF table, where `struct_segments` is one of six columns flagged VIF > 10.
- **Dataset B** (backfilled summary): **4 multicollinearity pairs flagged**, including
  `vol_primary` ↔ `struct_max_segment` (r = 0.94) — notably a *different* pair than either of Dataset A's
  flagged pairs, which is itself worth a sentence in Chapter 5's cross-dataset discussion: the two
  vantage points don't just differ in what they can observe, they differ in which observable features are
  redundant with each other. Full heatmap and the other 3 pairs pending B's local run.

## 3.5 Three-way breakdown — benign vs. heavy vs. light (Dataset A)

The plan's original hypothesis was that **light attacks sit much closer to benign than heavy attacks**
on these features — slow, low-throughput exfiltration should look statistically closer to normal
traffic, precisely because that is the point of throttling it (Chapter 1.5, Chapter 2.3). **The real
data does not bear this out, and is reported honestly rather than reframed to fit the prediction**
(`runs/metrics/three_way_breakdown_exf2021.json`):

| feature | Cliff's δ (light vs. benign) | Cliff's δ (heavy vs. benign) |
|---|---|---|
| vol_total | +0.567 | +0.561 |
| rand_dispersion | +0.590 | +0.592 |
| *(remaining 5 features show the same equidistant pattern — see JSON for full table)* | | |

Light and heavy sit at essentially **equal** distance from benign on 5 of 7 features, with light's
Cliff's delta marginally *larger* in magnitude on several — the opposite direction from what the
throughput/stealth hypothesis predicted. All Kruskal-Wallis p-values ≈ 0 (again, uninformative at this
n; the effect sizes are what matter).

**Interpretation, carried forward into Chapter 8:** on these particular stateless volumetric/structural
features, "light" describes payload *weight*, not stealth — a light-exfiltration session still has to
chunk and base64-encode its payload into DNS labels using the same mechanics as a heavy session
(Chapter 1.2), so F1–F3 alone don't distinguish the two well. This reframes the question Chapter 8.1
has to answer: not "why do light attacks look more like benign traffic" (they measurably don't, on
these features) but "why, given that they don't look more benign here, are any light attacks still
missed at all" — and Step 3A's actual answer turns out to be that they mostly aren't (light and heavy
recall are statistically indistinguishable on Dataset A), which closes this thread with a finding that
is consistent all the way from Chapter 3 through Chapter 8.

Dataset B's three-way equivalent (per attack/tunnel-tool subclass, per Step 3A's plan) has not been run.

## 3.6 Near-constant / redundancy audit

- **Dataset A** (`runs/metrics/near_constant_report_exf2021.json`): the 4 B_ONLY columns are exactly
  100% NaN (correctly flagged `near_constant: true` — they are constant after imputation, not before);
  all 7 testable columns have `nan_fraction: 0.0` and genuine variety (26–1,161 unique observed values).
  No dead columns among the testable set — nothing was dropped at this stage beyond what the schema
  already excludes by construction.
- **Dataset B**: pending B's local run.

## 3.7 What Chapter 4 inherits from here

Two multicollinearity pairs (Dataset A) and four (Dataset B, pending full detail) feed directly into
Chapter 4's VIF analysis. The two "negligible effect size" features (`rand_entropy`,
`struct_max_segment`) are checked again in Chapter 4 against the model's own gain-based importance
ranking — where the same two features corroborate this chapter's finding using a completely independent
method (multivariate importance vs. univariate effect size), which is a stronger result than either
chapter's evidence alone.
