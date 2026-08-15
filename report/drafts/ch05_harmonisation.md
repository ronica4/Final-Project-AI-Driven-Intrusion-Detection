# Chapter 5 — Harmonisation Across the Encryption Boundary

## 5.1 Unified schema specification

Every loader (`ingestion/exf2021.py`, `ingestion/dohbrw2020.py`) emits the same 11 columns, in the
same order, grouped into five behavioural families (`schema/unified.py`). Column order matters beyond
readability: the 1D-CNN's receptive field (Chapter 6.1) is only meaningful because related features sit
physically adjacent in the vector.

| Family | Columns | Role | Security meaning |
|---|---|---|---|
| **F1** Payload volume | `vol_primary`, `vol_secondary`, `vol_total` | INTERSECTION | Data volume moving through a channel that should only carry small lookups — exfiltration must push more bytes per unit than normal traffic. |
| **F2** Encoding randomness | `rand_entropy`, `rand_dispersion` | INTERSECTION | Entropy signature of base64URL-encoded payload; encoded binary looks statistically random, legitimate hostnames do not. |
| **F3** Structural complexity | `struct_segments`, `struct_max_segment` | INTERSECTION | Carrier segmentation forced by DNS label-length limits (63 B/label, 253 B/FQDN); chunking a file into many labels produces structurally unusual names. |
| **F4** Temporal rhythm | `time_central`, `time_dispersion`, `time_skew` | B_ONLY | Machine-paced vs. human-paced timing — automated exfiltration has regular inter-arrival timing. |
| **F5** Endpoint dispersion | `disp_uniqueness` | B_ONLY | Fan-out and destination rarity — exfiltration channels concentrate traffic toward few attacker-controlled endpoints. |

**INTERSECTION vs. B_ONLY is a scientific claim, not an implementation detail.** Three families (F1–F3)
have a genuine, if imperfect, counterpart on both sides of the encryption boundary. Two (F4–F5) exist
only in Dataset B, because Dataset A is restricted to *stateless* per-query features (decision D1) and
therefore has no temporal or endpoint telemetry at all to draw on — not a gap we failed to fill, but a
structural consequence of what a stateless DNS resolver log can observe versus what a full DoH flow
capture can observe. `COLUMN_SOURCE` (`schema/unified.py`) documents the exact per-dataset arithmetic
behind every column, including that F2/F3's Dataset-B realisations (`PacketLengthCoefficientofVariation`,
`log1p(PacketLengthVariance)`, `PacketLengthMode`, `PacketLengthStandardDeviation`) are **statistical
proxies for**, not equivalents of, Dataset A's direct string-level entropy and label-count measures — a
distinction that motivates the D3 ablation below.

## 5.2 Cross-dataset transfer, ablation, and distribution shift

**Status: the transfer matrix, F1-only-vs-intersection ablation, and per-feature KS distribution-shift
analysis (`evaluation/cross_dataset.py`, Step 2G) are built and unit-tested (11/11 passing on synthetic
data) but have not been run against real data on this machine** — this machine never had Dataset A's raw
files downloaded (Teammate B worked Dataset B locally throughout Phase 2), and the transfer matrix
requires both datasets loaded in the same process. Dataset B's three required raw files
(`l2-benign.csv`, `l2-malicious.csv`, `l1-nondoh.csv`) were handed to Teammate A, who has Dataset A
locally, to run this step; results are pending and will be backfilled into this section once available,
flagged inline rather than fabricated.

The module itself implements exactly what the design calls for and is documented here so the eventual
numbers slot into a pre-built frame rather than requiring restructuring:

- **Transfer matrix** (`build_transfer_matrix`): all four cells — train-A/test-A, train-B/test-B
  (in-domain, via each model's own 5-fold CV), train-A/test-B, train-B/test-A (transfer, single
  fit-on-source/score-on-target pass) — every cell restricted to `mode="intersection"` (F1–F3, the only
  columns both datasets share) so the four cells are directly comparable. The scaler is fit on the
  **training/source dataset only** and applied as-is to the target — the honest simulation of deploying
  a model into a new environment rather than an oracle that has seen the target's own scale.
- **Ablation (D3)** (`ablation_f1_only_vs_intersection`): both transfer directions rerun with
  `families="F1_only"` against the already-computed `families="intersection"` cells. The hypothesis
  under test is that F1 (volume) means the same thing on both sides of the encryption boundary while
  F2/F3 are lookalikes that should *hurt* transfer — reported whichever way it actually comes out, not
  forced to confirm the prediction.
- **Distribution shift** (`distribution_shift`): a Kolmogorov–Smirnov two-sample statistic per
  intersection column, ranked descending, cross-checked against the ablation result — if the
  worst-transferring features are also the ones with the largest KS distance, that is two independent
  lines of evidence for the same conclusion.

## 5.3 Scaling remedies

Three remedies are applied, each for a distinct mathematical reason, all inside `preprocessing/pipeline.py`'s
`build_pipeline()` so every model shares one preprocessing contract:

- **Z-scoring within dataset** (`StandardScaler`, fit per-fold in-domain / fit-on-source for transfer).
  The unified schema's 11 columns span wildly different native scales (byte counts vs. entropy bits vs.
  segment counts) — z-scoring puts every feature on a comparable footing before it reaches a
  distance-sensitive estimator (Isolation Forest's isolation depth, the CNN's gradient-based training),
  and doing it **per fold/per source dataset only** — never on the full combined data — is what keeps the
  cross-dataset transfer cells an honest "deploy into an unseen environment" simulation rather than an
  oracle fit.
- **`log1p` on heavy-tailed volume features.** `rand_dispersion`'s Dataset-B realisation is defined
  directly as `log1p(PacketLengthVariance)` (`schema/unified.py` `COLUMN_SOURCE`) precisely because
  packet-length variance is heavily right-skewed by the rare, very large exfiltration payloads — a raw
  variance feature would let a handful of extreme rows dominate any distance- or gradient-based model's
  gradient signal; the log transform compresses that tail into a shape closer to the bulk of the benign
  distribution while preserving rank order.
- **Median imputation, not mean** (`SimpleImputer(strategy="median", keep_empty_features=True)`).
  Median is the robust choice for the same right-skewed volume features `log1p` targets — a mean
  computed from a skewed distribution is itself skewed, so imputing with it would bias every row that
  needed imputation toward the tail. `keep_empty_features=True` is the structural implementation of the
  D2 observability finding below: an entirely-NaN column (Dataset A's four B_ONLY columns) is imputed to
  a constant rather than silently dropped, which `StandardScaler` then leaves at a stable zero — "this
  family carries no signal here" becomes a structural property of the pipeline, not merely a sentence in
  this report.

## 5.4 The observability finding (D2)

**Three of five behavioural families are unobservable, or observable only as a weak statistical proxy,
once traffic crosses the plaintext-DNS-to-encrypted-DoH boundary — and measuring exactly how much
information that boundary destroys is this project's central empirical claim, not a limitation to
apologise for.**

- **F4 (temporal rhythm) and F5 (endpoint dispersion) are entirely unobservable on Dataset A.** Every row
  of `time_central`, `time_dispersion`, `time_skew`, and `disp_uniqueness` is NaN by construction on
  Dataset A — not a data-quality defect, but the direct consequence of Dataset A being restricted to
  stateless, single-query features (decision D1): a resolver log of independent queries has no
  client-session key to compute inter-arrival rhythm or destination fan-out over. Dataset B, built from
  full DoH flow captures, observes both natively. This is verified structurally in `runs/metrics/near_constant_report_exf2021.json`
  (Chapter 3.6): the four B_ONLY columns are exactly 100% NaN, confirmed `near_constant: true` only
  *after* imputation, never silently coerced beforehand.
- **F2 and F3 survive the boundary only as proxies, not equivalents.** Dataset A measures encoding
  randomness and structural complexity directly from the query string itself (`entropy`, `labels`,
  `labels_max` — string-level measures available because Dataset A observes the plaintext query).
  Dataset B has no plaintext query to inspect at all — DoH's entire premise is that the query is
  encrypted in transit — so its F2/F3 columns are reconstructed from packet-length statistics instead
  (`PacketLengthCoefficientofVariation`, `PacketLengthMode`, `PacketLengthStandardDeviation`). These are
  *correlated with*, not *identical to*, the plaintext-side signal: a base64-encoded payload's
  string-level entropy and the packet-length variability it produces on the wire are related but
  distinct measurements, one direct and one inferred. This is exactly the distinction the D3 ablation
  (§5.2) is designed to quantify — if F2/F3 genuinely only proxy the plaintext signal, models trained on
  F1+F2+F3 should transfer worse across the boundary than models trained on F1 alone.
- **F1 (payload volume) is the one family with a genuinely direct counterpart on both sides.** Encrypted
  or not, a payload of N bytes still produces N bytes of DNS/DoH traffic — volume is a property of the
  channel, not of whether the channel's content is legible, so F1 is the family predicted to transfer
  best.

**Framed as a result about what encrypted telemetry can and cannot reveal:** an operator who only has
access to encrypted DoH flow captures — the realistic case for anyone downstream of the resolver once
DoH is in wide use — retains full visibility into payload volume, degraded-but-present visibility into
encoding randomness and structural complexity, and zero visibility into anything Dataset A's stateless
schema was ever able to see about rhythm or fan-out (because Dataset A never had that telemetry either).
The practical implication for a defender choosing which signal families to invest detection effort in is
therefore already implied by this table, independent of whatever the transfer matrix numbers eventually
show.
