# Header Reconciliation — CIC-Bell-DNS-EXF-2021 (Step 0C)

Verified against real downloaded files on 13 Aug 2026. Spec = `project_idea_description.md` §3.
Discrepancies documented below rather than silently adapted, per Step 0C instructions.

## 1. File layout differs from the spec

**Spec assumed:** stateless/stateful ship as paired files per class
(`heavy-stateful` / `heavy-stateless` / `light-stateful` / `light-stateless`), each internally
60/40 benign/attack.

**Actual layout (UNB `/CSV` index):**

```
CSV/
├── Attack_heavy_Benign/
│   ├── Attacks.zip   → 6 files: stateless_features-heavy_{audio,compressed,exe,image,text,video}.pcap.csv
│   │                     (+ matching 6 stateful_features-heavy_*.pcap.csv, unused per D1)
│   └── Benign.zip    → 3 files: stateless_features-benign_heavy_{1,2,3}.pcap.csv
├── Attack_Light_Benign/
│   ├── Attacks.zip   → 6 files: stateless_features-light_{audio,compressed,exe,image,text,video}.pcap.csv
│   └── Benign.zip    → 1 file:  stateless_features-light_benign.pcap.csv
└── Benign.zip (top-level, sibling to both Attack_* folders)
                       → 2 files: stateless_features-benign_{1,2}.pcap.csv
```

**Verdict:** attacks and benign are **already cleanly separated into different files** — better
than the spec's assumption, since it means no 60/40 in-file split needs to be disentangled. The
label (`y`) and `attack_subclass` (`heavy_attack` / `light_attack` / `benign`) come entirely from
**which file/folder a row was read from**, not from any column in the CSV.

⚠ The top-level `Benign.zip` was easy to miss — it sits as a sibling to the two attack folders in
the browser UI, not nested inside either. **It must be included**: without it, benign totals are
roughly half the expected count (see §3). `ingestion/exf2021.py` must read benign rows from
**three** locations: `heavy_benign/`, `light_benign/`, and `top_level_benign/`.

## 2. Column set — confirmed, one rename needed

All 16 stateless files (6 heavy-attack + 6 light-attack + 6 benign, across all three benign
sources) have **identical columns**, confirmed programmatically:

```
timestamp, FQDN_count, subdomain_length, upper, lower, numeric, entropy, special,
labels, labels_max, labels_average, longest_word, sld, len, subdomain
```

That's the spec's 14 stateless features plus `timestamp` (not in the spec's list; not part of
`UNIFIED_COLUMNS`, drop at load time — it's wall-clock capture time, not a security-relevant
per-query feature, and Dataset B doesn't have an equivalent to intersect against).

## 3. Row counts — top-level benign was the missing piece

| Source | Rows |
|---|---|
| heavy_attacks (6 payload types summed) | 251,670 |
| light_attacks (6 payload types summed) | 42,683 |
| heavy_benign (3 files) | 181,694 |
| light_benign (1 file) | 60,091 |
| **top_level_benign (2 files)** | **221,073** |
| **Total** | **757,211** |

Spec stated 1,019,318 total (heavy 323,698 / light 53,978 / benign 641,642). Our reconciled
totals are lower across the board (attacks: 294,353 vs 377,676; benign: 462,858 vs 641,642) —
likely the spec's figures came from a paper/documentation snapshot that doesn't exactly match
this file distribution. **We proceed with the real, verified counts** rather than chasing the
documented ones; this discrepancy itself goes in Chapter 5 as evidence of why header verification
against real files matters. Positive rate at these counts ≈ 38.9% (294,353 / 757,211) — closer to
balanced than the "needle in a haystack" framing implies, which is a separate honest note for
Ch 7.2 alongside the DoH base-rate paragraph.

## 4. Leakage audit — `sld` confirmed as raw text, AND a second leakage signal found

- **`subdomain`** is `int64`, values `{0, 1}` — a boolean "has-subdomain" flag, **not** raw
  subdomain text. Spec's leakage concern doesn't apply to this column. Keep as-is.
- **`sld`** is `object` dtype containing genuine raw strings: literal hostnames
  (`DESKTOP-3JF04TC`), NetBIOS-encoded name-service strings
  (`FHFAEBEECACACACACACACACACACACAAA`), and short tokens (`local`, `dns`, `192`). Confirms the
  spec's concern — **drop per the leakage rule.**
- **New finding, not anticipated by the spec:** `sld` cardinality is wildly different by class —
  **22 unique values** in heavy-attack traffic vs. **11,134** in `heavy_benign` and **22,153** in
  `top_level_benign`. A model trained with `sld` left in could trivially separate attack from
  benign by memorizing a closed set of ~22 testbed values, structurally identical to the
  `SourceIP` leakage already planned for Dataset B. **Recommend a matching leakage demonstration
  for Dataset A in Step 2C**, run once with `sld` included (expect near-perfect, artificially
  inflated score) and once with it dropped (the honest number) — mirrors the existing `SourceIP`
  demo and strengthens the Ch 4 discrepancy analysis with a second, independent example of the
  same lesson.

## 5. Missing values

Only `longest_word` has NaN, and only in benign files (some benign queries have no word-like
segment to compute a longest-word length from — e.g. purely numeric or IP-literal subdomains).
Attack files have zero NaN across all 14 features. Handled by the Pipeline's median imputer
(Step 2A) — not a loader concern.

## 6. Locked per-column arithmetic (final — supersedes the draft table in Step 0C)

| Unified column | Dataset A source | Notes |
|---|---|---|
| `vol_primary` | `len` | |
| `vol_secondary` | `subdomain_length` | |
| `vol_total` | `FQDN_count` | |
| `rand_entropy` | `entropy` | |
| `rand_dispersion` | `(numeric + special + upper) / len` | guard divide-by-zero where `len == 0` |
| `struct_segments` | `labels` | |
| `struct_max_segment` | `labels_max` | |
| `time_central` … `disp_uniqueness` | `NaN` | unobservable in Dataset A, per D2 |

Dropped columns: `timestamp` (not in schema), `sld` (leakage — raw text + class-skewed
cardinality, kept only behind an `include_leakage_columns` flag for the Step 2C demo, mirroring
Dataset B's `SourceIP` handling).
