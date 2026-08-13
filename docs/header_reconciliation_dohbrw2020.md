# Header Reconciliation — CIRA-CIC-DoHBrw-2020 (Step 0C)

Verified against real downloaded files on 13 Aug 2026. Spec = `PROJECT_PLAN.md` D2/D4/D8 proposal.
Discrepancies documented below rather than silently adapted, per Step 0C instructions.

## 1. File layout differs from a single combined CSV

**Downloaded:** `CSVs.zip` → `CSVs/Total_CSVs.zip` → four pre-aggregated, pre-labeled files (the
official site also ships `BenignDoH-NonDoH-CSVs.zip` and `MaliciousDoH-CSVs.zip`, which are the
same traffic broken into thousands of raw per-flow dumps; skipped as redundant for our purposes):

```
data/dohbrw2020/
├── l1-doh.csv        # Layer 1: DoH traffic (Benign-DoH + Malicious-DoH combined)
├── l1-nondoh.csv      # Layer 1: ordinary HTTPS, not DoH at all
├── l2-benign.csv      # Layer 2: Benign-DoH only
└── l2-malicious.csv   # Layer 2: Malicious-DoH only
```

MD5 of each zip verified against the CIC-supplied `.md5` files before extraction — all match.

**Verdict:** classes are already cleanly pre-separated into files, same as Dataset A. `y` and the
DoH/non-DoH split come from **which file a row was read from**, confirmed redundantly by the
in-file `Label` column (`DoH` / `NonDoH` / `Benign` / `Malicious`, one constant value per file —
verified, not assumed).

- **Hard framing (D4/D8):** `Label == Benign` from `l2-benign.csv` vs. `Label == Malicious` from
  `l2-malicious.csv`, `non-DoH` rows never touched.
- **Easy framing:** positives = `l2-malicious.csv`, negatives = `l2-benign.csv` + `l1-nondoh.csv`.

## 2. Column set — all 34 raw columns confirmed, zero renames needed

```
SourceIP, DestinationIP, SourcePort, DestinationPort, TimeStamp, Duration, FlowBytesSent,
FlowSentRate, FlowBytesReceived, FlowReceivedRate, PacketLengthVariance,
PacketLengthStandardDeviation, PacketLengthMean, PacketLengthMedian, PacketLengthMode,
PacketLengthSkewFromMedian, PacketLengthSkewFromMode, PacketLengthCoefficientofVariation,
PacketTimeVariance, PacketTimeStandardDeviation, PacketTimeMean, PacketTimeMedian, PacketTimeMode,
PacketTimeSkewFromMedian, PacketTimeSkewFromMode, PacketTimeCoefficientofVariation,
ResponseTimeTimeVariance, ResponseTimeTimeStandardDeviation, ResponseTimeTimeMean,
ResponseTimeTimeMedian, ResponseTimeTimeMode, ResponseTimeTimeSkewFromMedian,
ResponseTimeTimeSkewFromMode, ResponseTimeTimeCoefficientofVariation, Label
```

Identical across all four files (confirmed programmatically). Every column `schema/unified.py`
proposed for Dataset B (`PacketLengthMean`, `PacketLengthMedian`, `FlowBytesSent`,
`PacketLengthCoefficientofVariation`, `PacketLengthVariance`, `PacketLengthMode`,
`PacketLengthStandardDeviation`, `PacketTimeMean`, `PacketTimeStandardDeviation`,
`PacketTimeSkewFromMedian`, `FlowSentRate`) is present **verbatim** — unlike Dataset A, **no
renames were needed here.** The `ResponseTimeTime*` family (8 columns) and `PacketTimeMode` /
`PacketTimeVariance` / `PacketTimeCoefficientofVariation` / `PacketLengthSkewFrom*` are unused by
the unified schema; left in the raw frame only long enough to be dropped by `project()`.

## 3. Row counts — match the plan's D8 figures exactly

| Source | Rows | Positive rate |
|---|---|---|
| `l1-doh.csv` | 269,643 | — |
| `l1-nondoh.csv` | 897,493 | — |
| `l2-benign.csv` | **19,807** | — |
| `l2-malicious.csv` | **249,836** | — |

`l2-benign` / `l2-malicious` counts match D8's cited 19,807 / 249,836 exactly — the balancing plan
(keep all 19,807 benign, subsample malicious to ~19,807, `random_state=42`) needs no adjustment.
Hard framing after balancing: ~39,614 rows, positive_rate = 0.50. Easy framing: positives =
249,836, negatives = 19,807 + 897,493 = 917,300, positive_rate ≈ 0.214 — matches D4's "~98% of
negatives are ordinary HTTPS" and the plan's stated ~0.21 easy-framing rate.

## 4. Leakage audit — `SourceIP`/`DestinationIP` confirmed, cardinality quantified

The plan's suspicion (testbed used fixed IPs) is confirmed and now quantified, mirroring the `sld`
finding on Dataset A:

| File | Unique SourceIP | Unique DestinationIP |
|---|---|---|
| `l2-benign` (19,807 rows) | **10** | **10** |
| `l2-malicious` (249,836 rows) | **14** | **16** |
| `l1-nondoh` (897,493 rows, real web traffic) | 6,755 | 33,718 |

A closed set of 10-16 IPs across a quarter-million rows is a pure lookup key — a model trained
with `SourceIP` left in memorizes the testbed rather than learning tunneling behaviour. Contrast
with `l1-nondoh`'s thousands of unique IPs (ordinary browsing), confirming the artifact is specific
to the lab setup, not a property of DoH traffic in general. `SourcePort`/`DestinationPort`/
`TimeStamp` dropped for the same testbed-artifact reason (D2 rationale unchanged).

## 5. Missing values

Only `ResponseTimeTimeMedian` and `ResponseTimeTimeSkewFromMedian` contain NaN (0.1-1.3% of rows
per file — likely flows with too few request/response pairs to compute a median/skew). Neither
column is part of `UNIFIED_COLUMNS`, so this has zero effect on the schema; noted here only for
completeness of the header audit. All 11 unified-schema source columns are fully populated in
Dataset B — confirmed zero NaN in `B_ONLY_COLUMNS` sources, matching Step 1B's planned assertion.

## 6. Locked per-column arithmetic (final — Dataset B side of the Step 0C table)

| Unified column | Dataset B source | Verified |
|---|---|---|
| `vol_primary` | `PacketLengthMean` | ✅ present, float64, 0 NaN |
| `vol_secondary` | `PacketLengthMedian` | ✅ present, float64, 0 NaN |
| `vol_total` | `FlowBytesSent` | ✅ present, int64, 0 NaN |
| `rand_entropy` | `PacketLengthCoefficientofVariation` | ✅ present, float64, 0 NaN |
| `rand_dispersion` | `log1p(PacketLengthVariance)` | ✅ `PacketLengthVariance` present, float64, 0 NaN |
| `struct_segments` | `PacketLengthMode` | ✅ present, int64, 0 NaN |
| `struct_max_segment` | `PacketLengthStandardDeviation` | ✅ present, float64, 0 NaN |
| `time_central` | `PacketTimeMean` | ✅ present, float64, 0 NaN |
| `time_dispersion` | `PacketTimeStandardDeviation` | ✅ present, float64, 0 NaN |
| `time_skew` | `PacketTimeSkewFromMedian` | ✅ present, float64, 0 NaN |
| `disp_uniqueness` | `FlowSentRate` | ✅ present, float64, 0 NaN |

No proposed mapping needs revision. **Step 0C is now complete for both datasets.**

Dropped columns: `SourceIP`, `DestinationIP`, `SourcePort`, `DestinationPort`, `TimeStamp`
(leakage/testbed-artifact, retained only behind `include_leakage_columns=True` for the Step 2C
demo) — matches `schema/unified.py`'s existing `DROPPED_COLUMNS["dataset_b"]`, no change needed.
