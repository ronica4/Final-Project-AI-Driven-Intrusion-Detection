# Chapter 4 — Feature Ranking and the Leakage Demonstration

We deliberately cheat, catch ourselves, and report the honest number, via two structurally different
leakage mechanisms: Dataset A's `sld` (partial-overlap identifier) and Dataset B's `SourceIP`
(genuinely-exclusive identifier) — same practical conclusion, different routes.

## 4.1 Gain-based feature importance (Dataset A, clean run, `sld`/timestamp excluded)

| feature | gain share |
|---|---|
| rand_dispersion | 0.5112 |
| vol_total | 0.2767 |
| struct_segments | 0.1046 |
| vol_secondary | 0.0728 |
| struct_max_segment | 0.0187 |
| vol_primary | 0.0105 |
| rand_entropy | 0.0055 |
| all 4 B_ONLY columns | 0.0000 |

Gain (not weight-based) importance used deliberately — weight is biased toward high-cardinality features,
the artifact this chapter's leakage demo exists to catch. B_ONLY columns receive exactly zero gain —
XGBoost never splits on a post-imputation constant, third independent confirmation that effective
dimensionality is 7, not 11.

## 4.2 Leakage demo #1 — Dataset A, `sld`

`sld` takes 22–33 distinct values in attack traffic but 11,134–40,599 in benign — same trap class as
`SourceIP` (§4.3). Two encodings tested:

| | F1 (clean) | F1 (dirty) | dominant feature, dirty |
|---|---|---|---|
| Label-encoded `sld` | 0.8182 | 0.8184 | `_leakage_sld`, 75.5% of gain |
| Binary known-attack indicator | 0.8182 | 0.8184 | `_leakage_sld`, 95.4% of gain |

Importance dominates in both encodings, but score barely moves (+0.0002 F1) — muted vs. a
`SourceIP`-style memorised lookup. Direct check: of 33 attack-side `sld` values, **30 also appear in
benign traffic** — only 3 attack-exclusive. Low attack-side cardinality is real, but the value set is
mostly *shared*, a structurally weaker leakage mechanism than `SourceIP` — dropped by the production
loader regardless.

## 4.3 Leakage demo #2 — Dataset B, `SourceIP`

Real data, hard framing (39,614 rows). `SourceIP` has only **21 distinct values** — `192.168.20.111` alone
27.7% of rows; top 5 IPs over half.

| variant | F1 | top-3 gain importance |
|---|---|---|
| with `_leakage_sourceip` | 0.9998 | `_leakage_sourceip` 0.3818, `struct_segments` 0.1321, `time_dispersion` 0.1053 |
| clean (production default) | **0.9999** | `struct_segments` 0.3057, `time_dispersion` 0.1923, `time_central` 0.1304 |

Sharper than §4.2: leakage column dominates gain (0.3818) yet the clean model is marginally *better*. Step
2D already established near-perfect separability on packet-shape features alone, so no residual error even
an exclusive lookup can recover. High gain importance measures "how much splits routed through this
column," not "how much removing it would cost" — why both are checked, both datasets.

## 4.4 Multicollinearity — Variance Inflation Factor (Dataset A, all 11 columns)

Full table in Appendix D.2. **6 of 7 testable columns heavily collinear** (VIF>10: vol_primary 128.3,
vol_secondary 113.6, vol_total 136.3, rand_dispersion 14.4, struct_segments 100.5, struct_max_segment
168.0) — F1/F3 almost entirely redundant, consistent with Ch. 3's correlation pairs. Not acted on for
XGBoost (tree splits robust to collinearity); flagged for Ch. 6's CNN/Autoencoder discussion. `rand_entropy`
lowest VIF (2.33) — least redundant, sharpening its near-zero gain importance: genuinely little
independent signal (third corroboration, after Ch. 3's Cliff's delta).

## 4.5 Discrepancy analysis

| Surprise | Classification | Evidence |
|---|---|---|
| `rand_entropy`'s near-zero importance | Genuine latent pattern | VIF=2.33 (rules out collinearity); Cliff's δ=−0.125, negligible (Ch. 3) |
| `sld` dominates importance but barely moves score | Refinement of leakage, not straightforward | 30/33 attack `sld` values also appear in benign (§4.2) |
| Near-universal high VIF across F1/F3 | Multicollinearity, confirmed directly | Consistent with Ch. 3; larger in scope than predicted |

Two independent methods (Ch. 3's effect size, this chapter's gain importance) agree `rand_entropy`
underperforms its security-literature reputation on this dataset — stated plainly in Ch. 8.3.
