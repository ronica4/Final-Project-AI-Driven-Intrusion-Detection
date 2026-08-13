# PROJECT_PLAN.md

**Course 3917 — Using AI for Malware and Intrusion Detection, Reichman University, Semester 2 2026**
**Project:** AI-Driven Detection of DNS-Based Data Exfiltration — T1048.003
**Team:** 2 people (Teammate **A**, Teammate **B**)
**Working folder:** `C:\Users\guyha\OneDrive\Desktop\dns_exfil_project`
**Deadline:** Saturday 15 August 2026 · **Plan written:** Thursday 13 August 2026
**Submission artifact:** `Group_X1_X2_Final_Project.zip` (both student IDs — per the official PDF, §"Final Project Submission Overview")

---

## What Is This Project?

When an attacker steals data from a company, the hard part isn't grabbing the file — it's getting it out
past the firewall. One clever trick is to smuggle it out through **DNS**, the internet's phone book.

Every time any computer looks up a website name, that lookup is allowed out of almost every network on
earth, because if you block DNS nothing works. So the attacker hides the stolen file *inside the names
being looked up*. They chop the file into small chunks, scramble each chunk into letters and numbers,
and ask the network to look up names like `k3jf9dk2n4.7.exfil.attacker.com`. Each lookup carries a few
more bytes of the stolen file to a server the attacker controls. Nothing ever connects *in*. To a normal
firewall it looks like a computer that is just very curious about domain names.

**We are building a system that learns to spot this.** It reads network logs and decides, per record,
whether this looks like ordinary browsing or like a file being smuggled out.

**The interesting part — and the actual scientific question of this project:**

We test the same attack from **two completely different vantage points**:

- **Dataset A — plaintext DNS.** We can literally read the lookup names. The smuggled payload is visible:
  it's long, it's random-looking, it's chopped into weird segments.
- **Dataset B — DNS-over-HTTPS (DoH).** The same attack, but now everything is encrypted. We cannot read
  a single character of any name. All we can see is the *shape* of the traffic: how big the encrypted
  packets are, how fast they come, how long the conversation lasts.

So the question is: **which clues survive encryption, and which ones die with it?**

Our answer, which we test rather than assert: **only the volume clue truly survives.** "A lot of data is
moving through a channel that should only carry tiny lookups" is measurable whether or not you can read
the payload. But "the text looks random" and "the name is chopped into many segments" have *no real
equivalent* once encrypted — the best you can do is a statistical lookalike, and we predict those
lookalikes actively *hurt* when you try to carry a detector across the encryption boundary.

We also do something most student projects skip: **we try hard to catch ourselves cheating.** Twice.

1. Dataset B ships an IP-address column that published papers have accidentally trained on to reach
   "100% accuracy." We train on it *deliberately*, show the fake perfect score, name it as leakage, then
   throw it out and report the honest number.
2. Dataset B's easy negatives (ordinary web traffic that isn't DoH at all) make up ~98% of the negative
   class and are trivially separable. Including them inflates the headline score with a signal that has
   nothing to do with tunneling. We report **both** framings and show the gap.

Same lesson, two different mechanisms. That contrast is a large part of the grade.

Finally we chain the models into a **cascade** — a cheap fast model filters the firehose, a precise model
judges what survives, and only the genuinely ambiguous handful get escalated to a **large language model**
acting as a tier-2 analyst that must explain its reasoning.

---

## Key Design Decisions (locked — do not re-litigate)

These were decided at plan review on 13 Aug. They are recorded here so the report and the code stay
consistent, and so neither teammate silently reverts one.

| # | Decision | Rationale |
|---|---|---|
| **D1** | Dataset A = **stateless** files only | Lexical features are what we're testing for survival under encryption; avoids fabricating a join key between files never meant to be joined |
| **D2** | Schema is **INTERSECTION (F1–F3) + B-ONLY EXTENSION (F4–F5)**, not 5 symmetric families | `ttl_mean`/`ttl_variance` measure *cache lifetime*, not inter-arrival timing. Pairing them with `PacketTime*` would fake a temporal correspondence that does not exist. Switching A to stateful hides the flaw behind a non-null column rather than fixing it |
| **D3** | Ablation is **F1-only vs. F1+F2+F3** | Follows from D2. The original F1+F4 design was untestable — F4 is unobservable in A |
| **D4** | Dataset B evaluated under **two framings**, both reported; hard framing **balanced to ~1:1** | Non-DoH is ordinary HTTPS, trivially separable, and ~98% of the negative class. Same contaminant class as `SourceIP` leakage. See D8 for why 1:1 |
| **D5** | **Light-attack class never subsampled** | Smallest class *and* the entire analytical point of the project |
| **D6** | ZIP name follows the **PDF**, not the TA email | PDF is the official spec; sources genuinely conflict |
| **D7** | Bonus chapter (Step 3D) is **deferred by default**, added back only if Phase 3 finishes early Saturday | ~90% schedule utilisation assuming zero bugs is not how a 48-hour build goes. Costs bonus points only, never base points |
| **D8** | Hard framing subsamples **Malicious-DoH down to ~19,807** (1:1 vs Benign-DoH) | Unbalanced, "hard" framing is **93% positive**: always-malicious scores F1 = 0.962, PR-AUC baseline is 0.93, SMOTE would oversample *benign*, and `scale_pos_weight` drops below 1. The needle-in-a-haystack framing Ch 7.2 requires is inverted |
| **D9** | F1 columns renamed `vol_primary` / `vol_secondary` / `vol_total` | The original `vol_mean`/`vol_max`/`vol_dispersion` names were aspirational — neither dataset supplies mean/max/dispersion. `FQDN_count` and `FlowBytesSent` are both cumulative, so `vol_total` is an honest pairing |
| **D10** | Deep learning framework: **PyTorch (CPU)** | ~200 MB, no Windows/CUDA friction, explicit training loops that read well in the report. Both models are tiny |
| **D11** | EXF-2021 real file layout **locked** (13 Aug, header-verified) — see full writeup below and in `docs/header_reconciliation_exf2021.md` | Real files diverge from the spec in three ways: attacks/benign are pre-separated (no 60/40 unmixing needed); benign comes from **three** sources incl. an easy-to-miss top-level `Benign.zip`; `sld` is a second, worse leakage column than `SourceIP` (22 unique values in attack traffic vs. 11K–22K in benign) |
| **D12** | DoHBrw-2020 headers **locked** (13 Aug, header-verified by B) — see `docs/header_reconciliation_dohbrw2020.md` | Clean, pre-aggregated CSVs (`l1-doh`/`l1-nondoh`/`l2-benign`/`l2-malicious`). All 11 proposed column names matched exactly — zero renames. Raw counts confirmed: Benign-DoH 19,807 / Malicious-DoH 249,836, exactly matching D8's assumption. **`SourceIP` leakage confirmed independently**: 10–16 unique IPs across DoH classes vs. 6,755 in `l1-nondoh` — the same lookup-table failure mode as `sld` (D11), now verified on both datasets separately rather than assumed by analogy |

### D2 in full — the revised family map

| Family | Dataset A (plaintext, stateless) | Dataset B (encrypted, flow) | Role |
|---|---|---|---|
| **F1 Payload volume** | `len`, `subdomain_length`, `FQDN_count` | `FlowBytesSent`, `PacketLengthMean`, `PacketLengthMedian` | **INTERSECTION** — semantically equivalent |
| **F2 Encoding randomness** | `entropy`, `numeric`, `special`, `upper` | `PacketLengthVariance`, `PacketLengthCoefficientofVariation`, `PacketLengthSkewFromMode` | **INTERSECTION** — B side is a *proxy*, not an equivalent |
| **F3 Structural complexity** | `labels`, `labels_max`, `labels_average`, `longest_word` | `PacketLengthMode`, `PacketLengthStandardDeviation` | **INTERSECTION** — B side is a *weak* proxy |
| **F4 Temporal rhythm** | *not observable* | `Duration`, `PacketTime*` (8), `ResponseTimeTime*` (8) | **B_ONLY** extension |
| **F5 Endpoint dispersion** | *not observable* | `FlowSentRate`, `FlowReceivedRate` | **B_ONLY** extension |

**Evaluation rules that follow:**

| Run | Families used |
|---|---|
| In-domain, Dataset B | F1–F5 (all) |
| In-domain, Dataset A | F1–F3 |
| Cross-dataset transfer (either direction) | **F1–F3 only** (the intersection) |
| Ablation | **F1-only** vs. **F1+F2+F3** |

**Revised falsifiable hypothesis (goes in the report verbatim):**

> Only **F1** carries semantically equivalent meaning across the encryption boundary. **F2** and **F3**
> have B-side realisations that are statistical proxies rather than equivalents, so including them should
> **degrade** cross-dataset transfer rather than help it.

That 3 of 5 families are either unobservable in Dataset A or non-equivalent across the boundary is
**itself a Chapter 5 finding about telemetry observability.** Write it as a result. Not as an apology.

---

## Project Structure

```
dns_exfil_project/
├── PROJECT_PLAN.md              # this file — the tracker is the source of truth
├── main.py                      # CLI entry point: --dataset --mode --framing --families
├── requirements.txt             # exact pinned versions (== not >=)
├── README.md                    # quick-start; verified against a clean checkout at Sync 5
├── .gitignore                   # must exclude data/, .env, *.pkl, runs/
├── .env.example                 # shows HF_TOKEN=... ; the real .env is NEVER committed
│
├── config/
│   └── config.yaml              # paths, hyperparameters, sample_frac, framing, thresholds
│
├── data/                        # ⬅ GITIGNORED. Raw CSVs live here, never in the repo
│   ├── exf2021/                 #    CIC-Bell-DNS-EXF-2021 stateless CSVs
│   └── dohbrw2020/              #    CIRA-CIC-DoHBrw-2020 CSVs
│
├── ingestion/                   # ⬅ THE ONLY DATASET-AWARE CODE IN THE REPO
│   ├── base.py                  #    AbstractLoader ABC — the enforcing contract
│   ├── exf2021.py               #    knows CIC-Bell layout; light-class-preserving subsample
│   ├── dohbrw2020.py            #    knows DoHBrw CSVs; drops 5 identifiers; dual framing
│   └── registry.py              #    "exf2021" -> ExfLoader, "dohbrw2020" -> DohLoader
│
├── schema/
│   └── unified.py               # UNIFIED_COLUMNS, FAMILY_MAP, FAMILY_ROLE, validate_schema(mode)
│
├── preprocessing/
│   └── pipeline.py              # imblearn Pipeline: impute -> scale -> SMOTE (train fold ONLY)
│
├── features/
│   └── selection.py             # tree importance, VIF, correlation, leakage demonstration
│
├── models/
│   ├── supervised.py            # XGBoost
│   ├── deep.py                  # 1D-CNN + Autoencoder
│   └── unsupervised.py          # Isolation Forest
│
├── evaluation/
│   ├── metrics.py               # F1, PR-AUC, FPR, confusion matrices, CV harness
│   ├── error_analysis.py        # sample-level FN/FP forensics + the before/after experiment
│   └── cross_dataset.py         # train-A/test-B transfer matrix + F1-only ablation
│
├── ensemble/
│   ├── cascade.py               # IsolationForest -> XGBoost -> LLM routing
│   └── llm_arbiter.py           # HF InferenceClient; HF_TOKEN from env, NEVER committed
│
├── runs/                        # ⬅ GITIGNORED. Metrics JSON, figures, fitted models
│   ├── figures/                 #    every plot that goes in the report
│   └── metrics/                 #    one JSON per experiment — the frozen numbers
│
├── report/
│   └── Final_Report.docx        # 15 pages max, 1.5 spacing, Calibri 11/12
│
└── ai_logs/                     # ⬅ REQUIRED DELIVERABLE — full, unedited
    ├── claude_code_teammate_A.json
    └── claude_code_teammate_B.json
```

---

## Progress Tracker

Legend — Status: ✅ done · 🟡 in progress · ⬜ not started · ⛔ blocked
Gate: **BLOCK** = gates the other person's work, gets priority · **PAR** = parallel, safe to run concurrently

| Step | Owner | Gate | Status | Notes |
|---|---|---|---|---|
| **PHASE 0 — SHARED CONTRACT** | | | | |
| 0A — Repo skeleton, venv, pinned requirements | A+B | BLOCK | ✅ | PyTorch CPU per D10 |
| 0B — Dataset acquisition | A+B | BLOCK | ✅ | Both datasets downloaded and verified |
| 0C — Header verification & schema finalisation | A+B | BLOCK | ✅ | Both datasets done — `docs/header_reconciliation_exf2021.md` (D11) and `docs/header_reconciliation_dohbrw2020.md` (D12). Two independent leakage findings (`sld`, `SourceIP`) |
| 0D — `schema/unified.py`, `ingestion/base.py`, `registry.py`, `config.yaml` | A+B | BLOCK | ✅ | Fully LOCKED for both datasets. `tests/test_schema.py` 6/6 passing on both machines |
| **SYNC 1** — contract agreed, skeleton pushed, both pull | A+B | BLOCK | ✅ | Both pulled `3314098`, both ran `pytest` green independently |
| **PHASE 1 — INGESTION** | | | | |
| 1A — `ingestion/exf2021.py` | A | PAR | ✅ | Light class retained at 100% (D5). 12/12 tests passing incl. real-data integration tests |
| 1B — `ingestion/dohbrw2020.py` | B | PAR | ⬜ | Drops 5 identifiers; dual framing flag (D4) |
| 1C — `main.py` CLI wiring + end-to-end smoke test | B | PAR | ⬜ | B owns `main.py` from here on |
| **SYNC 2** — both loaders pass `validate_schema()` on real data | A+B | BLOCK | ⬜ | |
| **PHASE 2 — ANALYSIS & MODELS** | | | | |
| 2A — `preprocessing/pipeline.py` | A | BLOCK | ⬜ | Gates every model |
| **2A′ — `evaluation/metrics.py` shared harness + JSON schema** | A | **BLOCK** | ⬜ | **Also gates B** — 2E, 2F *and* 2G all import it. Do immediately after 2A, before 2B/2C |
| 2B — EDA + statistical evidence per feature → **Ch 3** | A | PAR | ⬜ | |
| 2C — Feature ranking + **`SourceIP` leakage demo** → **Ch 4** | A | PAR | ⬜ | |
| 2D — XGBoost + `max_depth` sweep | A | PAR | ⬜ | Harness moved to 2A′ — 2D is no longer on B's critical path |
| 2E — Isolation Forest + `contamination` sweep | B | PAR | ⬜ | |
| 2F — 1D-CNN + Autoencoder | B | PAR | ⬜ | |
| 2G — Cross-dataset transfer + shift plots + **ablation** → **Ch 5** | B | PAR | ⬜ | Needs 2A′ harness + A's trained XGBoost (Sync 3). Uses `families=F1_only` and `intersection` |
| **SYNC 3** — four models trained, both datasets, both framings | A+B | BLOCK | ⬜ | |
| **PHASE 3 — ENSEMBLE & FORENSICS** | | | | |
| 3A — Sample-level FN/FP forensics → **Ch 8.1** | A | PAR | ⬜ | |
| 3B — **Iterative optimisation: before/after experiment** | A | PAR | ⬜ | Light-class recall. One variable. One table |
| 3C — `ensemble/cascade.py` → **Ch 8.4** | B | PAR | ⬜ | |
| 3D — `ensemble/llm_arbiter.py` → **Bonus** | B | PAR | ⬜ **DEFERRED** | **Not started by default** (D7). Add-back decision is an explicit agenda item at Sync 4 |
| **SYNC 4** — all numbers frozen, no further experiments | A+B | BLOCK | ⬜ | |
| **PHASE 4 — REPORT** | | | | |
| 4A — Ch 1 threat characterisation + Ch 2 literature matrix | A | PAR | ⬜ | Must open by distinguishing exfil from DGA |
| 4B — Ch 3 EDA + Ch 4 feature ranking | A | PAR | ⬜ | |
| 4C — Ch 5 harmonisation + Ch 6 model justification | B | PAR | ⬜ | Needs 4 papers, one per model |
| 4D — Ch 7 pipeline + Appendices A & B | B | PAR | ⬜ | |
| 4E — Ch 8.1–8.3 error analysis, variance, benchmarking | A | PAR | ⬜ | |
| 4F — Ch 8.4 cascade + Bonus chapter | B | PAR | ⬜ | |
| 4G — Executive Summary | A | PAR | ⬜ | Written last, max 1 page |
| 4H — **Final assembly**, formatting, captions, 15-page cut | **B** | BLOCK | ⬜ | Single owner — a doc stitched by two people reads like it |
| **PHASE 5 — SUBMISSION** | | | | |
| 5A — AI log export, both teammates | A+B | BLOCK | ⬜ | Full and unedited. Curated = treated as missing |
| 5B — Clean-checkout verification + ZIP assembly | A+B | BLOCK | ⬜ | |
| **SYNC 5** — ZIP verified on both machines | A+B | BLOCK | ⬜ | |

---

## Open Items

**Resolved at plan review (13 Aug):** deep-learning framework → **PyTorch CPU** (D10). All schema and
framing questions → D1–D9. Nothing blocks Step 0A.

**Deferred, non-blocking:**

- **Teammate names.** A and B stay as placeholders for now. Fill in before Step 5A, since the `ai_logs/`
  filenames and the report title page both need them. Not on the critical path.

Everything else is locked in "Key Design Decisions" above. Do not re-litigate those without updating the
table, so the code and the report cannot silently diverge from each other.

---

# Step-by-Step Plan

---

## PHASE 0 — SHARED CONTRACT (`A+B`, joint, BLOCKING everything)

Both teammates sit together for this phase. It is the interface every later step depends on, and
splitting it causes rework that costs more than the hour it saves.

---

### Step 0A — Repo skeleton, virtualenv, pinned requirements
**Owner: A+B · Gate: BLOCKING · Est: 0.75 h · Files: `requirements.txt`, `.gitignore`, `.env.example`, `README.md` (stub), all `__init__.py`**

**What this means (plain English).**
Before anyone writes real code, we agree on the folder layout and the exact library versions. The reason
to pin versions *exactly* (`==1.5.3`, not `>=1.5`) is that the grader has to be able to reproduce our
numbers, and a library that silently changed its default between versions can move a metric. It is also
worth a rubric point in Appendix B. We also set up `.gitignore` first, because the fastest way to ruin
this project is to accidentally commit a 2 GB dataset or an API token.

**What Claude should do.**
1. Create the full directory tree from "Project Structure" above, with an empty `__init__.py` in every
   Python package directory.
2. Create the venv and install, then freeze exact versions into `requirements.txt`:
   - Core: `pandas`, `numpy`, `scikit-learn`, `scipy`, `pyyaml`
   - Imbalance: `imbalanced-learn` — **note:** SMOTE must live in `imblearn.pipeline.Pipeline`, *not*
     `sklearn.pipeline.Pipeline`. sklearn's Pipeline does not support resamplers and will silently
     apply SMOTE at transform time on the test fold. This is a leakage trap; call it out in a comment.
   - Models: `xgboost`, and **`torch` (CPU build)** per D10 — install from the CPU index
     (`--index-url https://download.pytorch.org/whl/cpu`) to avoid pulling ~2 GB of unused CUDA wheels
   - Plots: `matplotlib`, `seaborn`
   - LLM: `huggingface_hub`, `python-dotenv`
   - Report: `python-docx` (optional, only if we generate tables programmatically)
3. Write `.gitignore` with at minimum: `data/`, `runs/`, `.env`, `venv/`, `__pycache__/`, `*.pkl`, `*.pt`.
4. Write `.env.example` containing exactly `HF_TOKEN=hf_your_token_here` and nothing else.
5. Verify: `python -c "import pandas, sklearn, xgboost, imblearn; print('ok')"` and confirm
   `git status` shows no data files after dropping a dummy file into `data/`.

---

### Step 0B — Dataset acquisition ⚠ MANUAL STEP
**Owner: A+B · Gate: BLOCKING · Est: 1.0 h (mostly waiting) · Files: `data/` (gitignored)**

**What this means (plain English).**
Both datasets come from the Canadian Institute for Cybersecurity and are behind a registration form,
not a direct link. Downloads are large and their server is not fast. **Start this before anything else
and let it run in the background** — this is the single most likely thing to eat the schedule, and
unlike everything else on this plan it cannot be sped up by working harder.

**Manual checklist — a human must do these:**

- [ ] **A:** Go to `https://www.unb.ca/cic/datasets/dns-exf-2021.html` (landing page; the direct index is
      `http://cicresearch.ca/CICDataset/CICBellEXFDNS2021/`) and download the **stateless** CSVs
      (heavy-stateless and light-stateless, plus benign). Skip the stateful files entirely — per D1 we
      are not using them. Extract to `data/exf2021/`.
      *Good news for the schedule:* the full EXF-2021 distribution is only ~270 MB, so this download
      should be minutes, not hours.
- [ ] **B:** Go to `https://www.unb.ca/cic/datasets/dohbrw-2020.html`, complete the request form,
      download the **CSV** distribution. **Do not download the PCAPs** — they are enormous and §11 of
      the brief rules out raw PCAP parsing. Extract to `data/dohbrw2020/`.
- [ ] Both: record file sizes and SHA256 of each CSV in a scratch note — goes in Appendix B for
      reproducibility.
- [ ] Both: confirm `git status` is still clean.

**Contingency — mirrors pre-located, do not go searching at midnight.**

If a UNB form doesn't respond or a download stalls, switch immediately rather than waiting. Verified
Kaggle mirrors, in preference order:

| Dataset | Mirror | Preference |
|---|---|---|
| EXF-2021 | `https://www.kaggle.com/datasets/humera11/cicbelldnsexf2021` | **First choice** — appears to be a verbatim re-upload |
| EXF-2021 | `https://www.kaggle.com/datasets/dhoogla/cicbelldnsexf2021` | Fallback — ⚠ see caveat |
| DoHBrw-2020 | `https://www.kaggle.com/datasets/kwalite09/ciracicdohbrw2020` | **First choice** |
| DoHBrw-2020 | `https://www.kaggle.com/datasets/supplejade/bccc-cira-cic-dohbrw-2020-dns-over-http` | Second choice |
| DoHBrw-2020 | `https://www.kaggle.com/datasets/dhoogla/cicdohbrw2020` | Fallback — ⚠ see caveat |

⚠ **Caveat that matters more than it looks.** The `dhoogla` uploads are that author's *cleaned,
ML-ready* republications, not verbatim copies — they routinely drop identifier columns and recast dtypes.
For us that is actively dangerous: **`SourceIP` is the column Step 2C needs in order to run the
deliberate leakage demonstration.** A pre-cleaned mirror may have already removed it, silently killing a
10-point set-piece. Prefer a verbatim mirror; if we end up on a cleaned one, verify at Step 0C that
`SourceIP` survived and flag it at Sync 1 if not.

**If any mirror is used at all:** re-run Step 0C header verification against it from scratch — do not
assume a mirror matches UNB's column set — and document the provenance change in Chapter 5 rather than
quietly substituting. CIC's licence permits redistribution provided the dataset and its paper are cited,
so mirror use is legitimate; concealing it is not.

---

### Step 0C — Header verification & schema finalisation ✅ COMPLETE
**Owner: A+B · Gate: BLOCKING · Est: 1.25 h · Files: `docs/header_reconciliation_*.md`**
**Status: Dataset A ✅ VERIFIED 13 Aug (A) · Dataset B ✅ VERIFIED 13 Aug (B) — both halves done, `schema/unified.py` fully LOCKED**

**What this means (plain English).**
This is the step that saves the project. My written spec lists what the columns *should* be — but that
spec was assembled from papers and documentation, not from the actual files. Real datasets rename
things, add columns, use different capitalisation, and ship surprises. If we write loaders against the
spec and the spec is wrong, every downstream step is built on sand.

So: open the files, print what's really there, and reconcile against the plan **before writing a single
loader**. Then we lock the exact arithmetic for each of the 11 output columns.

**✅ Dataset B (DoHBrw-2020) — DONE (verified by B, 13 Aug). Full findings in
`docs/header_reconciliation_dohbrw2020.md`.** Much cleaner than Dataset A's check:

1. **File layout:** four pre-aggregated, pre-labeled CSVs (`l1-doh`, `l1-nondoh`, `l2-benign`,
   `l2-malicious`), MD5-verified against CIC's own checksums. Classes are pre-separated by file, same
   pattern as Dataset A, and the split is confirmed redundantly by an in-file `Label` column
   (`DoH`/`NonDoH`/`Benign`/`Malicious`) — not assumed from filename alone.
2. **Columns: zero renames needed.** All 34 raw columns confirmed identical across all four files, and
   **every single one of the 11 proposed `dataset_b` column names in `schema/unified.py` matched the
   real header verbatim** — `PacketLengthMean`, `FlowBytesSent`, `PacketTimeSkewFromMedian`, all of it,
   exactly as originally proposed. Unlike Dataset A, nothing needed correcting.
3. **Row counts match D8 exactly:** `l2-benign` 19,807 / `l2-malicious` 249,836 — the hard-framing
   balancing plan (keep all 19,807 benign, subsample malicious to ~19,807) needs zero adjustment.
   Easy framing positive rate ≈ 0.214, matching D4's estimate.
4. **`SourceIP` leakage confirmed and quantified — independently mirrors the `sld` finding (D11):**
   only **10–16 unique `SourceIP`/`DestinationIP` values** across the DoH classes (19,807–249,836 rows
   each), versus **6,755 unique IPs** in `l1-nondoh`'s real web traffic. Same lookup-table failure mode
   as `sld`, now confirmed on both datasets independently rather than assumed by analogy — a stronger
   basis for the Chapter 4 discrepancy analysis than either finding alone.
5. Only two non-schema columns (`ResponseTimeTimeMedian`, `ResponseTimeTimeSkewFromMedian`) have any
   NaN; all 11 unified-schema source columns are fully populated — zero NaN in the B-side data feeding
   `B_ONLY_COLUMNS`.

`schema/unified.py` is now **fully LOCKED** for both datasets — every `# PROPOSED` comment is gone.

---

**✅ Dataset A (EXF-2021) — DONE. Full findings in `docs/header_reconciliation_exf2021.md`.**
Summary of what changed from the spec, so 1A and 2C can be written against reality rather than the
original assumptions:

1. **File layout differs from the spec.** Attacks and benign live in **already-separate files** (better
   than the assumed 60/40 in-file mix — no unmixing needed), but attacks are further split by exfiltrated
   payload type (`audio`/`compressed`/`exe`/`image`/`text`/`video`, 6 files each for heavy and light), and
   **benign comes from three separate sources**: `heavy_benign/` (3 files), `light_benign/` (1 file), and
   an easy-to-miss **top-level `Benign.zip`** that sits as a sibling to both attack folders in the UNB
   browser UI rather than nested inside either (2 files, ~221K rows — over a third of total benign volume;
   skipping it silently would have halved the benign population). `y` and `attack_subclass` are derived
   entirely from **which file a row came from**, not from any column in the CSV.
2. **Columns match the spec exactly**: the 14 stateless features, plus `timestamp` (not in the unified
   schema — wall-clock capture time, no B-side equivalent to intersect against — dropped at load time).
   Identical across all 16 real files, verified programmatically.
3. **Real row counts are lower than the spec's stated totals** (757,211 vs. 1,019,318) — documented as a
   genuine discrepancy rather than chased to match; we proceed with the real, verified counts. Positive
   rate at these counts ≈ 38.9%, closer to balanced than the "needle in a haystack" framing implies —
   noted for Ch 7.2 alongside the DoH base-rate paragraph (Ch 8.2b).
4. **Leakage audit resolved, plus one new finding (D11):**
   - `subdomain` is `int64 ∈ {0,1}` — a boolean has-subdomain flag, **not** raw text. Keep as-is.
   - `sld` is confirmed raw text (real hostnames, NetBIOS-encoded strings) — **drop it**, per the
     original leakage rule.
   - **New:** `sld` cardinality is class-skewed — **22 unique values** in attack traffic vs.
     **11,134–22,153** in the three benign sources. Structurally identical to the `SourceIP` leakage
     already planned for Dataset B. **Step 2C should run a matching leakage demonstration for `sld`**
     (once included, once dropped), mirroring the `SourceIP` demo — see the updated Step 2C below.
5. Only `longest_word` has NaN, and only in benign rows. Handled by the Pipeline's median imputer — not
   a loader concern.

**Locked per-column arithmetic — supersedes the draft table this section originally had:**

| Unified column | Family | Role | Dataset A (**locked**) | Dataset B (**locked**) |
|---|---|---|---|---|
| `vol_primary` | F1 | INTERSECTION | `len` | `PacketLengthMean` |
| `vol_secondary` | F1 | INTERSECTION | `subdomain_length` | `PacketLengthMedian` |
| `vol_total` | F1 | INTERSECTION | `FQDN_count` | `FlowBytesSent` |
| `rand_entropy` | F2 | INTERSECTION | `entropy` | `PacketLengthCoefficientofVariation` |
| `rand_dispersion` | F2 | INTERSECTION | `(numeric + special + upper) / len` — guard `len == 0` | `log1p(PacketLengthVariance)` |
| `struct_segments` | F3 | INTERSECTION | `labels` | `PacketLengthMode` |
| `struct_max_segment` | F3 | INTERSECTION | `labels_max` | `PacketLengthStandardDeviation` |
| `time_central` | F4 | B_ONLY | `NaN` | `PacketTimeMean` |
| `time_dispersion` | F4 | B_ONLY | `NaN` | `PacketTimeStandardDeviation` |
| `time_skew` | F4 | B_ONLY | `NaN` | `PacketTimeSkewFromMedian` |
| `disp_uniqueness` | F5 | B_ONLY | `NaN` | `FlowSentRate` |

Dropped from Dataset A: `timestamp` (not in schema), `sld` (leakage — kept only behind
`include_leakage_columns=True` for the Step 2C demo, mirroring Dataset B's `SourceIP` handling).
Dropped from Dataset B: `SourceIP`, `DestinationIP`, `SourcePort`, `DestinationPort`, `TimeStamp`
(leakage/testbed-artifact, same `include_leakage_columns=True` gate).

✅ **The F1 naming question (D9) is resolved** — see the Key Design Decisions table above.
✅ **Both header reconciliations are complete and pushed** — `docs/header_reconciliation_exf2021.md` and
`docs/header_reconciliation_dohbrw2020.md`. `schema/unified.py` is fully LOCKED, `ingestion/base.py`,
`ingestion/registry.py`, `config/config.yaml`, `main.py`, and `tests/test_schema.py` (6/6 passing, on
both teammates' machines independently) are in place. **Step 0D is complete.**

---

### Step 0D — The contract: schema, loader ABC, registry, config
**Owner: A+B · Gate: BLOCKING · Est: 1.0 h · Files: `schema/unified.py`, `ingestion/base.py`, `ingestion/registry.py`, `config/config.yaml`, `main.py` (stub only)**

**What this means (plain English).**
This is the single most important architectural rule in the whole assignment: **only the ingestion code
is allowed to know which dataset it's looking at.** Everything after it — cleaning, feature selection,
training, evaluation — must be blind to the source. A grader should be able to swap Dataset A for
Dataset B and have the entire rest of the pipeline run untouched.

We enforce it with a hard contract: every loader must hand back a table with *exactly* the same 11
columns in exactly the same order, and a function that crashes loudly if it doesn't.

**What Claude should do.**

1. **`schema/unified.py`** — implement:
   ```python
   UNIFIED_COLUMNS: list[str]      # exactly 11, in family order, as finalised in 0C

   FAMILY_MAP: dict[str, list[str]]
   # "F1_payload_volume": ["vol_primary", "vol_secondary", "vol_total"], ...

   FAMILY_ROLE: dict[str, str]
   # "F1_payload_volume": "INTERSECTION",  "F4_temporal_rhythm": "B_ONLY", ...

   INTERSECTION_COLUMNS: list[str]   # derived: the 7 F1-F3 columns
   B_ONLY_COLUMNS: list[str]         # derived: the 4 F4-F5 columns
   F1_ONLY_COLUMNS: list[str]        # derived: the 3 F1 columns -- the ablation arm

   VALID_MODES = ("full", "intersection", "F1_only")

   def validate_schema(X: pd.DataFrame, mode: str = "full") -> None:
       """Hard assert on the loader contract. Raises SchemaViolation, never warns.

       mode="full"         -> X.columns must equal UNIFIED_COLUMNS exactly, in order
       mode="intersection" -> X.columns must equal INTERSECTION_COLUMNS exactly, in order
       mode="F1_only"      -> X.columns must equal F1_ONLY_COLUMNS exactly, in order

       Also asserts: no duplicate columns, all dtypes numeric, X is not empty, and
       that `mode` is in VALID_MODES (a typo'd mode must raise, never silently pass).
       Does NOT assert absence of NaN -- NaN is a legitimate, reportable signal here.
       """

   def project(X: pd.DataFrame, mode: str) -> pd.DataFrame:
       """Return the column subset for a given evaluation mode. The ONLY sanctioned
       way downstream code selects families -- no ad-hoc column lists anywhere else."""
   ```

   **`F1_only` is the ablation arm (D3)** and must exist from day one. Step 2G compares `F1_only`
   against `intersection` on Friday evening; discovering the mode is missing at that point costs an
   unplanned edit to a file A owns while B is mid-experiment.
   Every family and every column gets a docstring stating its **security meaning**, its source columns
   in both datasets, and its INTERSECTION/B_ONLY role. This file is quoted directly in Chapter 5, so
   write the comments as report prose, not as reminders to yourself.

2. **`ingestion/base.py`** — the ABC:
   ```python
   class AbstractLoader(ABC):
       @abstractmethod
       def load(self) -> tuple[pd.DataFrame, pd.Series, dict]:
           """Returns (X, y, meta).

           X    MUST have exactly schema.unified.UNIFIED_COLUMNS, in order.
           y    MUST be binary int (1 = exfiltration).
           meta carries provenance ONLY -- never consumed by any downstream module.
                Required keys: dataset_name, n_rows_raw, n_rows_after_sampling,
                class_counts_raw, class_counts_sampled, framing, dropped_columns,
                attack_subclass (Series aligned to X.index, or None)
           """
   ```
   Note `attack_subclass` in `meta`: it carries heavy/light for Dataset A. It lives in `meta` precisely
   *because* it must never influence training — it is used only by the error-analysis module, after
   predictions exist. Say that in the docstring.

3. **`ingestion/registry.py`** — a plain `{name: LoaderClass}` dict plus `get(name)`. No dynamic imports,
   no plugin magic. A grader must be able to read it in five seconds.

4. **`config/config.yaml`** — sections:
   ```yaml
   paths:        { exf2021: data/exf2021, dohbrw2020: data/dohbrw2020, runs: runs }
   sampling:     { sample_frac: 0.25, preserve_classes: [light_attack], random_state: 42 }
   framing:      { dohbrw2020: hard }        # hard | easy  -- see D4
   cv:           { n_splits: 5, shuffle: true, random_state: 42 }
   models:       { xgboost: {...}, cnn: {...}, isolation_forest: {...}, autoencoder: {...} }
   cascade:      { llm_lower: 0.35, llm_upper: 0.65, max_llm_calls: 200 }
   llm:          { model: meta-llama/Llama-3.1-8B-Instruct, timeout_s: 30 }
   ```
   Every hyperparameter from the model table goes here explicitly. **No library defaults anywhere** —
   that is a rubric requirement in Ch 7.3, not a style preference.

5. **`main.py` stub** — argument parsing only, no logic yet:
   `--dataset {exf2021,dohbrw2020} --mode {eda,train,eval,xdataset,cascade} --framing {hard,easy} --families {full,intersection} --config config/config.yaml`

6. **Verification:** write `tests/test_schema.py` with five cases — a correct frame passes; a
   column-reordered frame raises; a missing-column frame raises; `project()` returns exactly 7 columns
   for `intersection` and exactly 3 for `F1_only`; and an invalid mode string raises. It must be
   possible to run this before any real data exists.

---

### 🔄 SYNC 1 — end of Phase 0 ✅ COMPLETE (13 Aug)
**Owner: A+B · Gate: BLOCKING**

- [x] Both datasets downloaded and extracted
- [x] Header reconciliation table produced and reviewed **by both people** — `docs/header_reconciliation_exf2021.md` (A) + `docs/header_reconciliation_dohbrw2020.md` (B)
- [x] Per-column arithmetic locked, including the `vol_primary`/`vol_secondary`/`vol_total` mapping (D9)
- [x] `validate_schema()` supports all three modes: `full`, `intersection`, `F1_only`
- [x] `schema/unified.py` contract agreed; `test_schema.py` passes (6/6)
- [x] Repo pushed (`3314098`); **both teammates pulled and confirmed tests pass on their own machine**
- [x] Mirror URLs were not needed — both UNB downloads succeeded directly

**From this point on the two teammates work on disjoint files and do not edit each other's.**
**Phase 1 (ingestion) can now start in parallel:** A → Step 1A (`ingestion/exf2021.py`), B → Step 1B
(`ingestion/dohbrw2020.py`). Both loaders now have a fully locked, verified schema to build against.

---

## PHASE 1 — INGESTION (parallel, disjoint files)

---

### Step 1A — Dataset A loader
**Owner: A · Gate: PARALLEL · Est: 2.0 h · Files: `ingestion/exf2021.py` (A owns exclusively)**

**What this means (plain English).**
Teammate A writes the code that reads the plaintext-DNS files and converts them into our 11 standard
columns. Two things make this more than a file read.

First, **the light-attack rows are sacred.** The light class — slow, patient exfiltration — is the
smallest group in the data *and* is the whole point of the project's error analysis. If we shrink the
data for speed by taking a proportional random sample, we shrink the one population we cannot afford to
have noisy. So: sample the heavy-attack and benign rows, keep **100% of the light rows**.

Second, the four F4/F5 columns don't exist in this dataset at all. We emit them as `NaN` rather than
inventing values. That's not laziness — the missingness *is* the finding.

**What Claude should do.**
1. Implement `Exf2021Loader(AbstractLoader)`.
2. **Read per the real layout locked in Step 0C** (not the spec's original assumption of pre-mixed
   60/40 files):
   - Attack rows: glob `stateless_features-heavy_*.pcap.csv` (6 payload-type files) and
     `stateless_features-light_*.pcap.csv` (6 payload-type files); concatenate each group, tagging
     `attack_subclass = "heavy_attack"` / `"light_attack"`. The payload type (audio/exe/text/...) is not
     part of `UNIFIED_COLUMNS` — keep it in `meta` only, as a bonus dimension for error analysis if time
     allows, never as a training feature.
   - Benign rows: concatenate **all three** sources — `heavy_benign/Benign/*.pcap.csv` (3 files),
     `light_benign/Benign/*.pcap.csv` (1 file), and `top_level_benign/Benign/*.pcap.csv` (2 files).
     **Do not skip the top-level source** — it's ~221K rows, over a third of total benign volume, and
     it's the one that's easy to miss because it sits as a folder sibling rather than nested inside
     either attack folder.
   - `y = 1` for both attack groups, `y = 0` for all benign rows regardless of source.
3. **Leakage guard (locked at 0C):** drop `sld` unconditionally (raw text + class-skewed cardinality —
   22 unique values in attack traffic vs. 11K–22K in benign, see D11) unless
   `include_leakage_columns=True` for the Step 2C demo. Drop `timestamp` unconditionally (not part of
   the unified schema). Keep `subdomain` — confirmed to be a boolean flag, not raw text. Record every
   drop in `meta["dropped_columns"]`.
4. **Subsampling (D5) — inside the loader, so nothing downstream knows it happened:**
   ```python
   # Light attack is never sampled: it is the smallest class AND the analytical
   # centre of the project. Proportional sampling would add noise exactly where
   # we can least afford it. Heavy and benign are sampled by config.sample_frac.
   ```
   - `light_attack` → retained at 100%, always
   - `heavy_attack`, `benign` → sampled at `config.sampling.sample_frac`, `random_state=42`
   - Record `class_counts_raw` and `class_counts_sampled` in `meta`
   - `sample_frac: 1.0` must be a valid setting that skips sampling entirely (the stretch-goal full run)
5. Build the 11 unified columns per the 0C-locked arithmetic. Emit `NaN` for all four F4/F5 columns —
   as `np.nan` floats, so the column dtype stays numeric and `validate_schema` passes.
6. Build `y`: `1` for heavy **and** light attack, `0` for benign. Keep the heavy/light distinction only
   in `meta["attack_subclass"]`.
7. **Do not scale here.** Scaling is the Pipeline's job, fit on the training fold only. A loader that
   z-scores is a loader that leaks.
8. **Verification:**
   - `validate_schema(X, mode="full")` passes
   - assert `X[B_ONLY_COLUMNS].isna().all().all()` — all four are fully NaN, as designed
   - assert light-attack count in `meta["class_counts_sampled"]` equals the raw count exactly
   - print the pre/post sampling class-count table → save to `runs/metrics/exf2021_sampling.json`
     (this table goes in the report per D5)

---

### Step 1B — Dataset B loader, with dual framing
**Owner: B · Gate: PARALLEL · Est: 2.5 h · Files: `ingestion/dohbrw2020.py` (B owns exclusively)**

**What this means (plain English).**
Teammate B writes the encrypted-side loader. This one carries the project's second self-audit.

The dataset has three groups: malicious DoH (the attack), benign DoH (innocent encrypted DNS), and
non-DoH (ordinary web traffic that isn't DNS at all). The tempting move is to call everything that isn't
malicious a "negative" — which gives you ~917k negatives. But ~98% of those are ordinary HTTPS, which a
model can separate from DoH almost trivially on packet sizes alone. You'd get a beautiful F1 score that
mostly measures "can you tell DNS traffic from web traffic," which is *not the question we're asking*.

So we build both:
- **Hard framing (primary):** benign DoH vs. malicious DoH. Both sides are DoH. The only difference is
  tunneling. This is the real problem.
- **Easy framing (secondary):** everything collapsed, explicitly labelled as inflated.

One catch that has to be handled here rather than later: in the raw data there are 12× more malicious
DoH records than benign ones, so "hard" framing as originally specified would be **93% attack** — and a
model that blindly shouts "attack!" at everything would score an F1 of 0.96 without having learned
anything. That's the same disease as the easy framing, just pointing the other way. So we balance it:
keep every benign-DoH record and randomly draw an equal number of malicious ones (D8).

The gap between those two numbers is a genuine result, and it pairs with the `SourceIP` leakage demo as
the same lesson via a different mechanism.

**What Claude should do.**
1. Implement `DohBrw2020Loader(AbstractLoader)` taking `framing: str` and
   `include_leakage_columns: bool = False`.
2. **Identifier drop (leakage audit):** drop `SourceIP`, `DestinationIP`, `SourcePort`,
   `DestinationPort`, `TimeStamp` — *unless* `include_leakage_columns=True`, which Step 2C needs to run
   the deliberate leakage demonstration. Record every drop in `meta["dropped_columns"]`. Comment:
   ```python
   # Published analyses of this dataset report ~100% accuracy driven by SourceIP
   # alone -- a pure testbed artifact, since attacker and victim had fixed IPs.
   # Retained only behind an explicit flag, used once, to demonstrate the failure.
   ```
3. **Framing logic (D4 + D8):**
   ```python
   # framing="hard" (PRIMARY): Benign-DoH vs Malicious-DoH, BALANCED ~1:1.
   #   non-DoH rows are DROPPED entirely. Both classes are DoH; the only
   #   difference is tunneling, which is the actual detection question.
   #
   #   Raw counts are Benign-DoH 19,807 vs Malicious-DoH 249,836 -- i.e. 93%
   #   POSITIVE. Left unbalanced this framing is broken, not hard: an
   #   always-malicious classifier scores F1 = 0.962, the PR-AUC baseline is
   #   0.93, SMOTE would oversample *benign*, and scale_pos_weight falls below
   #   1. So we keep all 19,807 Benign-DoH (the scarce class -- same logic as
   #   D5) and subsample Malicious-DoH to ~19,807, random_state=42.
   #   Result: ~40k rows at positive_rate ~= 0.50.
   #
   # framing="easy" (SECONDARY, reported as inflated): positives = Malicious-DoH,
   #   negatives = Benign-DoH + non-DoH. ~98% of negatives are ordinary HTTPS and
   #   are trivially separable on packet-length statistics. Reported ONLY as a
   #   contrast case in Ch 8 -- never as the headline number.
   ```
   Set `meta["framing"]`, `meta["positive_rate"]`, and **both** raw and balanced class counts.
4. **Do NOT apply a blanket `sample_frac` under hard framing** — at 0.25 it lands at ~76% positive,
   which is still broken. Hard framing uses the explicit 1:1 balance above and ignores `sample_frac`
   entirely; make that override loud in the code and record it in `meta`, so nobody later "fixes" the
   inconsistency by re-enabling it. `sample_frac` still applies to the majority classes under easy
   framing.

   Side benefit worth noting: ~40k rows trains far faster than 270k, which buys back schedule.
5. Build the 11 unified columns per the 0C arithmetic — all 11 are populated here; Dataset B is the
   dataset that can fill everything.
6. `meta["attack_subclass"]` carries the original three-class label so error analysis can ask "which
   tunnel tool did we miss" later.
7. **Verification:**
   - `validate_schema(X, mode="full")` passes for **both** framings
   - assert zero NaN in `B_ONLY_COLUMNS` (B observes all families)
   - assert hard framing has no non-DoH rows, `positive_rate ≈ 0.50 ± 0.02`, and total rows ≈ 40k
   - assert hard framing retained **all 19,807** Benign-DoH rows (none sampled away)
   - assert easy framing `positive_rate ≈ 0.21`
   - assert the 5 identifiers are absent when the flag is off, present when on
   - save both class-count tables to `runs/metrics/dohbrw2020_sampling.json`

---

### Step 1C — `main.py` wiring and end-to-end smoke test
**Owner: B · Gate: PARALLEL · Est: 0.75 h · Files: `main.py` (B owns exclusively from here on)**

**What this means (plain English).**
Wire the CLI so one command loads either dataset and validates it. This is also where we produce the
"proof of abstraction" that Chapter 7 requires: after the loader returns, no line of code anywhere
mentions a dataset name.

**What Claude should do.**
1. Implement the load path:
   ```python
   loader = registry.get(args.dataset)(config, framing=args.framing)
   X, y, meta = loader.load()
   schema.validate_schema(X, mode=args.families)   # hard assert
   # from here on, NOTHING knows which dataset this is
   ```
2. Print a provenance banner from `meta` — dataset, framing, raw/sampled counts, positive rate,
   dropped columns, NaN count per family.
3. **The abstraction proof:** run
   `grep -rniE "exf2021|dohbrw|cic|bell|doh" --include=*.py . | grep -v "^./ingestion/"`
   The output must be **empty** (excluding `registry.py`'s string keys and this plan file). Save the
   command and its empty output to `runs/metrics/abstraction_proof.txt` — it goes in Chapter 7 as
   verifiable evidence.
4. **Verification:** all four of these run clean:
   `--dataset exf2021 --mode eda`, `--dataset dohbrw2020 --framing hard --mode eda`,
   `--dataset dohbrw2020 --framing easy --mode eda`, and one `--families intersection` run.

---

### 🔄 SYNC 2 — end of Phase 1
**Owner: A+B · Gate: BLOCKING**

- [ ] Both loaders pass `validate_schema()` on real data, both modes
- [ ] Sampling tables saved; light-class count verified untouched
- [ ] Both framings of Dataset B produce the expected positive rates
- [ ] Abstraction grep is empty and saved
- [ ] Both teammates run all four smoke commands **on their own machine**

---

## PHASE 2 — ANALYSIS & MODELS (parallel, disjoint files)

---

### Step 2A — Preprocessing pipeline 🔴 BLOCKING
**Owner: A · Gate: BLOCKING (B cannot train until this lands) · Est: 1.25 h · Files: `preprocessing/pipeline.py`**

**What this means (plain English).**
Before a model sees data we need to fill missing values, put every feature on a comparable scale, and
fix the fact that attacks are rare. The danger is doing any of that *before* splitting into train and
test — because then information from the test set has bled into the training process, and your score is
fiction. The fix is structural: put every one of those steps inside a Pipeline object, and hand the
whole Pipeline to the cross-validator. Then it is *impossible* to fit on the test fold, because the
Pipeline refits from scratch inside each fold.

A does this first because both teammates' models depend on it.

**What Claude should do.**
1. `build_pipeline(estimator, use_smote: bool) -> imblearn.pipeline.Pipeline` with steps:
   `SimpleImputer(strategy="median")` → `StandardScaler()` → `SMOTE(random_state=42)` (optional) →
   estimator.
2. **Use `imblearn.pipeline.Pipeline`, never `sklearn.pipeline.Pipeline`.** Comment why: sklearn's
   version does not understand resamplers and will apply SMOTE at transform time on the test fold.
3. `use_smote=False` for the Autoencoder (trained on benign only — SMOTE would be incoherent) and for
   XGBoost when using `scale_pos_weight` instead. Document that choice; Ch 7.2 asks for a theoretical
   defence of the imbalance strategy, and "we used both and here's when each applies" is a stronger
   answer than picking one.
4. **Median imputation, not mean** — the volume features are heavily right-skewed and the mean is
   dragged by the heavy-exfiltration tail. Note this in Ch 5.3.
5. `get_cv(y)` returning `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
6. **Verification:** fit on a synthetic frame with known NaN and a 95/5 imbalance; assert the scaler's
   `mean_` differs between two different folds (proof it refits per fold), and assert the resampled
   training set is balanced while the test fold is untouched. This assertion *is* the Ch 7.2 evidence —
   save it to `runs/metrics/leakage_guard_proof.json`.

---

### Step 2A′ — Shared evaluation harness 🔴 BLOCKING
**Owner: A · Gate: BLOCKING (B cannot score anything until this lands) · Est: 0.75 h · Files: `evaluation/metrics.py`**

**What this means (plain English).**
One scoring function that every model, from both teammates, is measured by — so that when we build the
comparison tables in Chapter 8 the numbers are directly comparable instead of needing hand-reconciliation
at midnight on Saturday.

**This was originally folded into Step 2D and that was a scheduling bug.** Steps 2E, 2F **and** 2G all
import this module, so B is blocked on it — but 2D sat third in A's Friday queue behind 2B (2.5 h) and
2C (2.0 h), which would have stalled B for most of Friday afternoon waiting on a harness A hadn't reached
yet. It moves here, immediately after 2A, so **B is unblocked the moment 2A′ lands** and A is then free
to sequence 2B / 2C / 2D in whatever order is convenient.

**What Claude should do.**
1. `evaluate(model, X, y, cv, meta) -> dict` returning per-fold and mean ± std for Precision,
   Recall/DR, **F1**, **PR-AUC**, ROC-AUC, **FPR**, plus the confusion matrix.
2. `save_metrics(name, dict)` → one JSON per experiment under `runs/metrics/`.
3. `plot_confusion(cm, title, path)` — used by every model, both datasets.
4. **Freeze the JSON schema here and write it into the module docstring.** Every field name, every unit.
   Both teammates code against it from this point; changing it later means re-running experiments.
5. Record `meta["framing"]`, `meta["dataset_name"]`, and the families mode in every result JSON, so a
   metrics file is self-describing and can't be misattributed when the two teammates merge results.
6. **Include the majority-class baseline in every result** — the F1 a trivial always-positive (or
   always-negative) classifier would score on that exact split. Cheap to compute, and it is the thing
   that makes a headline number interpretable rather than impressive-sounding. It is also precisely what
   would have caught the 93%-positive framing bug on its own.
7. **Verification:** run on a synthetic frame with a known confusion matrix and assert every metric
   matches hand-computed values. Both teammates import it and confirm identical output before Sync 3.

---

### Step 2B — EDA and hard statistical evidence → **Chapter 3**
**Owner: A · Gate: PARALLEL · Est: 2.5 h · Files: `features/selection.py` (EDA half), `runs/figures/`**

**What this means (plain English).**
Chapter 3 is worth 15 points and the brief is blunt about what earns them: *"Broad, hand-wavy
theoretical justifications without data-driven evidence will not be accepted."* For every one of our 11
features we must show, with a number, that it actually behaves differently for attacks than for normal
traffic. Not "entropy should be higher" — an actual statistic with an actual p-value and an actual plot.

**What Claude should do.**
1. Class distribution bar charts, both datasets, both framings — pre- and post-sampling.
2. Per-feature class-conditional histograms and box plots, benign vs. attack, both datasets.
3. **Per-feature statistical test — this is the graded part.** For each of the 11 columns compute:
   - **Mann-Whitney U** (non-parametric; our features are not normal — do not use a t-test and claim
     they are)
   - **Cliff's delta** or **Cohen's d** as an effect size. With ~250k rows *every* p-value will be
     astronomically significant, so p alone proves nothing. Effect size is what separates a real signal
     from a large-sample artifact. Say exactly this in the chapter — it demonstrates statistical
     literacy and it is true.
   - Output one table: feature | median benign | median attack | U statistic | p | effect size | verdict
   - Save to `runs/metrics/feature_significance.json` and render as a report table
4. Correlation heatmap per dataset. Flag any |r| > 0.9 pair as a multicollinearity candidate for Ch 4.
5. **Three-way breakdown on Dataset A: benign vs. heavy vs. light.** Expect the light class to sit much
   closer to benign than heavy does. That gap, quantified here, is the setup for the entire error
   analysis in Step 3A — establish it now with numbers.
6. Noise/redundancy reduction: document which features are near-constant or redundant, and what we did.
7. **Verification:** every figure saved to `runs/figures/` at ≥150 dpi with a descriptive filename;
   every figure gets a caption written now, while you remember what it shows.

---

### Step 2C — Feature ranking + the deliberate leakage demonstration (×2) → **Chapter 4**
**Owner: A · Gate: PARALLEL · Est: 2.5 h (+0.5 h vs. original estimate — now two leakage demos, not one) · Files: `features/selection.py` (ranking half)**

**What this means (plain English).**
Two jobs. First, let a tree model rank the features by usefulness and compare that ranking against what
we *expected* as security analysts — agreements and surprises both need explaining.

Second, the set-piece: **we cheat on purpose, then catch ourselves — twice, once per dataset.** We train
a model on Dataset B with the `SourceIP` column left in. It will score nearly perfectly, because in the
lab that produced this data the attacker always used the same IP — so the model just memorises "this IP
= attack." Dataset A has its own version of the same trap, found during Step 0C's header check: `sld`
takes only **22 distinct values** in attack traffic but **11,134–22,153** in benign — a model could
trivially separate the classes by memorising which of 22 fixed testbed values `sld` holds, no different
in kind from `SourceIP`. Neither is real detection, both are lookup tables, and both are exactly the
artifact the rubric asks us to diagnose. We show the fake score, show the feature-importance chart
dominated by one identifier, name it, then re-run clean and report the honest number — for both.

**What Claude should do.**
1. Fit XGBoost inside the Pipeline; extract **gain-based** importance (not the default weight-based —
   weight counts splits and is biased toward high-cardinality features; say so in the chapter).
2. Horizontal bar chart of all 11 features, both datasets. This is Figure 4.1.
3. **Leakage demonstration #1 (Dataset B, `include_leakage_columns=True`):**
   - Train, report F1 / PR-AUC — expect ≈0.99+
   - Plot importance; expect `SourceIP` to dominate
   - Re-run with the identifiers dropped; report the honest F1 / PR-AUC
   - Produce a two-row before/after table and one paragraph naming it as a testbed artifact
   - Save to `runs/metrics/leakage_demo_dohbrw2020.json`
4. **Leakage demonstration #2 (Dataset A, `include_leakage_columns=True`) — added after the 0C header
   check surfaced the `sld` cardinality skew (D11), not in the original spec:**
   - Train with `sld` included (one-hot or target-encoded — note the encoding choice, since `sld` isn't
     numeric); report F1 / PR-AUC and expect a similarly inflated score
   - Plot importance; expect `sld` to dominate
   - Re-run with `sld` dropped (the production loader default); report the honest F1 / PR-AUC
   - Same two-row before/after table format as #1 — the parallel structure is itself the point, it shows
     the *same lesson* recurring under a *different mechanism*, which is a stronger Ch 4 narrative than
     either demo alone
   - Save to `runs/metrics/leakage_demo_exf2021.json`
5. **Multicollinearity:** compute VIF across the 11 columns. Expect `vol_primary` / `vol_secondary` to be
   correlated on both sides. Report VIF > 10 as flagged, and discuss whether to act (for tree models,
   usually not — trees are robust to collinearity; for the CNN and AE it matters more). That nuance is
   worth a point.
6. **Discrepancy analysis — answer the three rubric questions explicitly:** did the algorithm agree with
   our intuition; what surprised us; is each surprise leakage, multicollinearity, or a genuine latent
   pattern. Answer per surprise, not in general.

---

### Step 2D — XGBoost and sensitivity sweep
**Owner: A · Gate: PARALLEL · Est: 1.25 h · Files: `models/supervised.py`**

**What this means (plain English).**
Train the main supervised model, scoring it through the 2A′ harness. Then deliberately vary one knob
(tree depth) and plot what it does to accuracy and false alarms — the rubric wants evidence that we
understand our hyperparameters rather than having copied them.

Report **PR-AUC**, not raw accuracy. With a rare positive class, accuracy is close to meaningless — a
model that says "benign" every single time scores 95% on an imbalanced set while detecting nothing.

**What Claude should do.**
1. `models/supervised.py`: XGBoost with the locked, non-default hyperparameters —
   `n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
   min_child_weight=3, scale_pos_weight=<computed per dataset+framing>, random_state=42`.
   All values read from `config.yaml`, none hardcoded.
2. Score everything through `evaluation.metrics.evaluate()` from Step 2A′ — no bespoke scoring here.
3. Run XGBoost: Dataset A (full-families = F1–F3 in practice), Dataset B hard framing, Dataset B easy
   framing. Three result sets.
4. **Sensitivity sweep:** `max_depth ∈ {3, 6, 9, 12}`; plot F1 and FPR against depth for both datasets.
   Interpret it — where does it start overfitting, and how do you know?
5. **Verification:** metrics JSONs exist for all three runs; the sweep figure is saved and captioned.

---

### Step 2E — Isolation Forest + sensitivity sweep
**Owner: B · Gate: PARALLEL · Est: 1.25 h · Files: `models/unsupervised.py`**

**What this means (plain English).**
An unsupervised model that doesn't learn what "attack" means at all — it just learns what *typical*
looks like and flags outliers. It's fast and catches things it has never seen, which makes it the right
first stage of the cascade: cheap, high-recall, tolerant of false alarms because a second model will
clean up behind it.

**What Claude should do.**
1. `IsolationForest(n_estimators=200, max_samples=256, contamination=0.2, max_features=0.8,
   random_state=42, n_jobs=-1)`, inside the Pipeline, **`use_smote=False`** (resampling an
   unsupervised density estimator is incoherent — note that in the report).
2. Evaluate with A's `evaluation/metrics.py` harness, all three dataset/framing combinations.
3. **Sensitivity sweep:** `contamination ∈ {0.05, 0.1, 0.2, 0.3}`; plot F1 and FPR. This directly
   informs the cascade threshold in Step 3C — pick the value that maximises **recall** at tolerable FPR,
   not the one that maximises F1, and explain why: a first-stage filter that misses attacks is fatal,
   while one that over-flags is merely expensive.

---

### Step 2F — 1D-CNN and Autoencoder
**Owner: B · Gate: PARALLEL (needs 2A pipeline + 2A′ harness) · Est: 3.0 h · Files: `models/deep.py` — PyTorch CPU per D10**

**What this means (plain English).**
Two deep models. The **CNN** slides small filters across the feature vector to learn *combinations* of
neighbouring features — which is why we order the vector by family, so "high volume AND high randomness"
sits physically adjacent and is learnable as one pattern.

The **Autoencoder** is trained only on normal traffic. It learns to compress and rebuild benign records
accurately. Show it an attack and it rebuilds it badly, because it has never seen anything like it — and
that reconstruction error becomes the detector. The point is that it can catch tunnel tools that didn't
exist when we trained.

**What Claude should do.**
1. **1D-CNN:** input shaped `(11, 1)` in family order. 2 conv blocks (32 and 64 filters, kernel 3,
   `padding="same"`), BatchNorm after each, dropout 0.3, dense 64, sigmoid output. Adam `lr=1e-3`,
   batch 256, early stopping on validation PR-AUC with patience 5, `random_state=42`.
   With only 11 features, kernel 3 covers a meaningful fraction of the vector — state that the
   family-ordered layout is what makes the convolution meaningful rather than arbitrary. That is the
   Ch 6.1 structural justification for this model.
2. **Autoencoder:** `11 → 8 → 4 → 8 → 11`, ReLU, MSE, Adam `lr=1e-3`.
   **Fit on benign training-fold rows only.** Threshold = 95th percentile of *benign training-fold*
   reconstruction error — computed inside the fold, never on test data. This is the leakage trap
   specific to this model; guard it explicitly and assert it.
3. Both must accept NaN-containing input via the Pipeline's imputer — on Dataset A four columns are
   fully NaN and will impute to a constant. **Note the consequence honestly:** four constant inputs
   contribute nothing, so the effective input dimensionality on Dataset A is 7, not 11. Report it.
4. Evaluate through A's harness so results are comparable.
5. **Verification:** training curves saved; assert the AE threshold was computed from training-fold
   benign rows only; assert reproducibility across two runs with the same seed.

---

### Step 2G — Cross-dataset transfer, distribution shift, and the ablation → **Chapter 5**
**Owner: B · Gate: PARALLEL (needs 2A′ harness, plus A's trained XGBoost from 2D — do after Sync 3) · Est: 2.5 h · Files: `evaluation/cross_dataset.py`**

**What this means (plain English).**
The centrepiece experiment. Train a detector on plaintext DNS, then — without retraining — point it at
encrypted DoH traffic and see whether it still works. Then the reverse. If the behavioural abstraction
is real, something transfers. If it collapses, that is *also* a result, and the rubric explicitly asks
us to diagnose which of two causes is responsible: the environments genuinely differ (distribution
shift), or the model memorised its training set (overfitting).

Then the ablation that tests our hypothesis. We predicted that only **F1** (volume) means the same thing
on both sides, and that F2 and F3 are lookalikes that should *hurt* transfer. So train on F1 alone,
train on F1+F2+F3, and compare transfer. **If F1-only transfers better, the hypothesis is confirmed.**

**What Claude should do.**
1. **Transfer matrix**, all four cells, every model, using `mode="intersection"` (F1–F3) throughout —
   the B-only families cannot participate in transfer by construction:

   | | test A | test B (hard) |
   |---|---|---|
   | **train A** | in-domain | transfer |
   | **train B (hard)** | transfer | in-domain |

   Report F1, PR-AUC, Recall, FPR per cell. **Fit the scaler on the training dataset only** and apply it
   to the target — that is the honest simulation of deploying a model into a new environment.
2. **The ablation (D3):** rerun both transfer directions with `families="F1_only"` vs.
   `families="intersection"` (F1+F2+F3). Produce one comparison table. Interpret against the revised
   hypothesis, and **report the result whichever way it comes out.** If F1-only does *not* transfer
   better, say so plainly and diagnose why — a falsified hypothesis honestly reported scores; a quietly
   buried one does not.
3. **Distribution shift plots (Ch 5.2):** for each of the 7 intersection columns, overlaid density plots
   A vs. B, plus a table of mean/variance/skew per dataset and a **Kolmogorov–Smirnov statistic** per
   feature quantifying the shift. Rank the features by KS — and check whether the ranking matches the
   ablation result. If the features that transfer worst are also the ones with the largest KS distance,
   those are two independent lines of evidence for the same conclusion, which is a genuinely strong
   result. Say so if it holds.
4. **Scaling remedies (Ch 5.3):** document z-scoring within dataset, `log1p` on the heavy-tailed volume
   features, and median imputation — with the mathematical reason for each.
5. **Observability finding (D2):** write the section stating that 3 of 5 families are unobservable or
   non-equivalent across the boundary, with the NaN counts as evidence. **Frame as a result about what
   encrypted telemetry can and cannot reveal.** This is a Chapter 5 contribution, not a limitation
   paragraph.

---

### 🔄 SYNC 3 — end of Phase 2
**Owner: A+B · Gate: BLOCKING**

- [ ] All four models trained on: Dataset A, Dataset B hard, Dataset B easy
- [ ] Every result in `runs/metrics/` using the agreed JSON schema
- [ ] Leakage demo complete with before/after numbers
- [ ] Both sensitivity sweeps plotted
- [ ] Transfer matrix and ablation complete
- [ ] **Both teammates run `main.py` end to end on both datasets on their own machine**

---

## PHASE 3 — ENSEMBLE & FORENSICS (parallel, disjoint files)

---

### Step 3A — Sample-level forensic error analysis → **Chapter 8.1**
**Owner: A · Gate: PARALLEL · Est: 2.5 h · Files: `evaluation/error_analysis.py`**

**What this means (plain English).**
Chapter 8 is 20 points and this is its core. Aggregate metrics say *how often* a model was wrong. This
says *which* records it got wrong and *what they had in common*. We pull the actual misclassified rows
and look at them.

The expected headline: **light exfiltration is where the models fail.** Slow, patient attacks look
statistically like normal traffic, because that is the entire design goal of running slow. Proving that
with per-subclass recall numbers is a much stronger finding than any aggregate F1.

**What Claude should do.**
1. `analyse_errors(model_name, y_true, y_pred, y_proba, X, meta) -> dict`.
2. **Per-subclass recall using `meta["attack_subclass"]`** — heavy vs. light on Dataset A; per tunnel
   tool on Dataset B if that granularity survived. Expect heavy recall ≫ light recall. Quantify the gap.
3. Pull the 20 highest-confidence **false negatives** and 20 highest-confidence **false positives** per
   model. Tabulate their 11 feature values against the class medians. Answer: what do they share?
4. **Cross-model failure comparison** — the rubric asks specifically why one architecture fails where
   another succeeds. Build the set intersection: which samples did XGBoost miss that the CNN caught,
   and vice versa. Non-overlapping failure modes are the empirical justification for the cascade —
   if the models failed on identical samples, ensembling them would be pointless. Make that argument
   explicitly; it links Ch 8.1 to Ch 8.4.
5. **False positives:** what benign traffic looks like exfiltration? Expect CDN domains with long
   random-looking subdomains and DNS service-discovery patterns. Name concrete examples.
6. Save everything to `runs/metrics/error_analysis_<model>_<dataset>.json`.

---

### Step 3B — Iterative optimisation: the before/after experiment 🔴
**Owner: A · Gate: PARALLEL · Est: 1.5 h · Files: `evaluation/error_analysis.py` (optimisation half)**

**What this means (plain English).**
The brief requires that we don't just *diagnose* a failure — we fix one and prove the fix worked with
numbers. Not prose. The structure is rigid on purpose: diagnose one specific failure in 8.1, change
**exactly one thing**, re-run, put both numbers in one table.

Our target is fixed: **light-exfiltration false negatives.** And we report the cost honestly — pushing
recall up on a rare class always buys false positives somewhere else. Reporting that trade openly is
worth more than a table that only shows the improvement.

**What Claude should do.**
1. **Diagnosis (from 3A):** light-class recall is materially below heavy-class recall.
2. **Change exactly one variable.** Pick one, run it, do not stack them:
   - **Option 1 (recommended):** lower the XGBoost decision threshold from 0.5 to the value maximising
     F1 **on the light subclass only**, chosen on validation folds, never on test.
   - **Option 2:** raise `scale_pos_weight`, or apply per-subclass sample weights upweighting light rows.

   Recommend Option 1: it is a pure post-hoc threshold move, so the model is identical and the
   comparison is clean — any metric change is attributable to the threshold and nothing else. Option 2
   retrains and confounds the comparison.
3. **One table, two rows:**

   | Configuration | Light recall | Heavy recall | Overall F1 | PR-AUC | **FPR** |
   |---|---|---|---|---|---|
   | Before (threshold 0.50) | | | | | |
   | After (threshold `t*`) | | | | | |

4. **One paragraph of defence** stating: what we changed, why it targets the diagnosed failure, what it
   improved, **what it cost in false positives**, and whether a real SOC would accept that trade. If
   light recall improves by 8 points and FPR triples, say that and argue whether it's worth it. Do not
   present the improvement alone.
5. Save to `runs/metrics/optimisation_before_after.json`.

---

### Step 3C — Hybrid cascade → **Chapter 8.4**
**Owner: B · Gate: PARALLEL · Est: 2.0 h · Files: `ensemble/cascade.py`**

**What this means (plain English).**
Chain the models so each does what it's best at. A cheap anomaly detector reads everything and throws
away the obviously-normal majority. Whatever it flags goes to the precise supervised model. And only
the genuinely ambiguous handful — where the model is near 50/50, or where two models disagree — gets
escalated to the expensive LLM. That's how a real SOC triages, and it's what keeps the LLM call count
in the hundreds rather than the hundreds of thousands.

**What Claude should do.**
1. Implement the three stages:
   - **Stage 1** — Isolation Forest, tuned for **recall** (per 2E). Anything scored normal is discarded.
   - **Stage 2** — XGBoost on survivors. Confident benign → BENIGN. Confident malicious → MALICIOUS.
   - **Stage 3** — escalate only when `0.35 ≤ P_xgb ≤ 0.65` **OR** XGBoost and the Autoencoder disagree.
2. **Instrument every stage:** rows in, rows out, rows discarded, per-stage latency. Report the funnel
   as a table — it demonstrates the cascade is doing real work rather than being decorative.
3. **Assert the escalation count is a few hundred, not tens of thousands.** If it blows up, tighten the
   band and document the change rather than silently capping it.
4. Report end-to-end cascade F1 / PR-AUC / FPR against each individual model. **If the cascade does not
   beat the best single model, report that honestly** and diagnose why — a cascade that trades a little
   F1 for a large latency reduction is a legitimate engineering result, and saying so is better than
   quietly reporting only the favourable metric.
5. Produce the block diagram for Ch 8.4.

---

### Step 3D — LLM arbiter → **Bonus Chapter** 🟨 DEFERRED BY DEFAULT
**Owner: B · Gate: PARALLEL · Est: 2.5 h · Files: `ensemble/llm_arbiter.py`**

**What this means (plain English).**
The ambiguous cases go to a language model prompted to act as a tier-2 SOC analyst: here are the
behavioural features as z-scores, here's what the other models thought, reason it through and give a
verdict with a justification. Worth up to +10 bonus points.

**🟨 This step is DEFERRED BY DEFAULT (D7) — do not start it on Friday.** The plan assumes it does not
happen. It is added back only if Phase 3 finishes early on Saturday morning, and that add-back is an
explicit agenda item at Sync 4 rather than a drift decision someone makes alone at 2am.

The reasoning: 27.5 h + 28.0 h against roughly 31 available hours each (Thu evening ~5 h, Fri ~14 h,
Sat ~12 h) is ~90% utilisation *assuming zero bugs, no download failure, and normal sleep*. That is not
how a 48-hour build goes. Deferring this costs bonus points only, never base points, and buys the buffer
we will almost certainly need. **Second cut if we need more:** reduce both sensitivity sweeps from four
values to three (Step 2D `max_depth`, Step 2E `contamination`) — that is ~0.5 h back with minimal
rubric cost, since the sweep's purpose is demonstrating the trend, not resolving it finely.

**What Claude should do.**
1. `HF_TOKEN` from environment via `python-dotenv`. **Never commit it.** Fail with a clear message if
   absent rather than crashing obscurely. Verify `.env` is gitignored *before* the first run.
2. `huggingface_hub.InferenceClient` with `meta-llama/Llama-3.1-8B-Instruct`, timeout 30s, retry with
   backoff, and **graceful degradation**: if the API fails, fall back to the XGBoost verdict and log the
   failure. A rate limit must never take down the evaluation run.
3. Use the prompt template from the spec: 11 features as z-scores grouped by family name, both model
   probabilities, `NaN` rendered explicitly as `not observable in this telemetry` (the model should know
   the difference between "zero" and "unmeasurable"), then the forced three-step chain-of-thought and
   the strict output line `VERDICT: ... | CONFIDENCE: ... | REASON: ...`.
4. Parse strictly; count and report parse failures rather than silently dropping them.
5. **Cap at 200 samples** (`config.cascade.max_llm_calls`). Sample the escalated set stratified by true
   label so the accuracy estimate isn't dominated by one class.
6. **Measure and report:** wall-clock latency per call (mean, median, p95), total call count, total wall
   time, parse failure rate.
7. **B.3 — reasoning accuracy:** compare LLM verdicts against ground truth *and* against XGBoost on the
   same ambiguous subset. The honest question is whether the LLM beats the model it was brought in to
   arbitrate. Report it either way.
8. **B.4 — SOC viability verdict.** If 200 calls take 15 minutes, extrapolate to production volume and
   state plainly whether this is deployable. The brief explicitly rewards candour here and penalises
   hand-waving.

---

### 🔄 SYNC 4 — end of Phase 3 · 🔒 NUMBERS FREEZE
**Owner: A+B · Gate: BLOCKING**

- [ ] Error analysis complete for all models, both datasets
- [ ] Before/after optimisation table produced
- [ ] Cascade evaluated with funnel table
- [ ] **Step 3D add-back decision made explicitly** — 3D is deferred by default (D7). Add it back only
      if everything above is genuinely finished and it is still Saturday morning. Decide out loud, record
      it, move on. Do not leave it ambiguous
- [ ] **All numbers frozen. No further experiments.** From here it is writing only.
- [ ] Every figure needed by the report exists in `runs/figures/` with a caption

---

## PHASE 4 — REPORT (parallel drafting, single-owner assembly)

Format: **.docx, max 15 pages, 1.5 spacing, Calibri or Arial 11/12, every table and figure captioned,
supporting graphs in appendices.** 15 pages is tight for 8 chapters — write dense, use tables over
prose, push exploratory plots to appendices.

---

### Step 4A — Chapter 1 (threat) + Chapter 2 (literature) — 25 points
**Owner: A · Gate: PARALLEL · Est: 3.0 h**

**Ch 1 (15 pts).** Open by **distinguishing exfiltration from DGA** — the professor rejected a DGA
proposal previously, so address it in the first paragraph: DGA is a *rendezvous* problem (does this
domain look machine-generated?) under TA0011; exfiltration is a *carrier capacity* problem (is data
being smuggled inside query names?) under **TA0010**. Different tactic, mechanism, feature space, and
failure mode. Then: the MITRE table (T1048.003 primary, T1572.002, T1071.004, T1132.001); the attack
mechanics (base64URL encode → chunk to the 63-byte label / 253-byte FQDN limits → emit as subdomains →
resolver forwards to attacker NS, no inbound connection); the throughput/stealth tradeoff that produces
the light class; the **Telemetry & Feature Mapping table in exactly the 5-column format the brief
prescribes** (behaviour | telemetry source | raw fields | derived feature | explanation); and §1.3
theoretical rationale per feature.

**Ch 2 (10 pts).** Extraction matrix from Nadler et al. (2019) and Mahdavifar et al. (2021). Contrast
them on the three axes: the semantic meaning each assigns to **entropy**; whether they model
**per-query or per-session**; and which features we **adopted, modified, or rejected** — with reasons.
Nadler's low-throughput focus is the direct ancestor of our light-class analysis; make that lineage
explicit, it strengthens both chapters.

---

### Step 4B — Chapter 3 (EDA) + Chapter 4 (ranking) — 25 points
**Owner: A · Gate: PARALLEL · Est: 2.5 h**

From Steps 2B and 2C. Lead Ch 3.3 with the significance table and the **effect-size caveat about large
n**. Lead Ch 4 with the leakage demonstration — it is the strongest single narrative beat in the report.

---

### Step 4C — Chapter 5 (harmonisation) + Chapter 6 (models) — 15 points
**Owner: B · Gate: PARALLEL · Est: 3.0 h**

**Ch 5.** The schema spec with the INTERSECTION/B_ONLY table, the observability finding (D2), the shift
plots with KS statistics, and the scaling remedies.

**Ch 6.1.** Per-model mathematical justification: XGBoost's axis-aligned splits match threshold-like
exfiltration signals; the CNN's local receptive field over a **family-ordered** vector; Isolation
Forest's cheap high-recall isolation depth; the Autoencoder's benign-only manifold for novel tools.

**Ch 6.2 — needs four papers, one per model** (per the PDF's Step 6, over and above the two from Ch 2).
For each: how the authors configured that architecture, on what dataset, and what Recall / FPR / F1 they
reported. Search IEEE Xplore, ACM DL, Google Scholar for that architecture applied to DNS
tunneling/exfiltration. These numbers are reused for benchmarking in Ch 8.3 — collect them in one table
now and cite that table twice rather than re-researching later.

---

### Step 4D — Chapter 7 (pipeline) + Appendices — 10 points
**Owner: B · Gate: PARALLEL · Est: 2.0 h**

**7.1** Block diagram + the **abstraction proof** (the empty grep output from Step 1C) as verifiable
evidence. **7.2** Imbalance strategy and the structural leakage guard — quote the actual Pipeline
construction code and the fold-refit assertion from Step 2A. **7.3** Full hyperparameter table plus both
sensitivity sweeps with interpretation.

**Appendix A:** concrete step-by-step commands — clone, venv, install, place data, run each mode.
**Appendix B:** exact library versions from `requirements.txt`, plus CPU/RAM/OS of both machines.

---

### Step 4E — Chapter 8.1–8.3 — part of 20 points
**Owner: A · Gate: PARALLEL · Est: 2.5 h**

**8.1** Confusion matrices for all 4 models × both datasets, the per-subclass recall breakdown, the
FN/FP forensics, the cross-model failure comparison, **and the Step 3B before/after table**.
**8.2** The cross-dataset comparison table, with an explicit verdict per drop: distribution shift or
overfitting, and the evidence for that call.

**8.2b — the base-rate honesty paragraph. Do not skip this; it is short and it is worth real points.**
State plainly that **neither framing has an operationally realistic base rate.** Real-world DoH traffic
is overwhelmingly benign, and this dataset cannot supply that ratio at all — the easy framing is 21%
positive, our balanced hard framing is 50%, and production is closer to 1 in 10,000. Then extrapolate:
take our measured FPR and compute what it does to an analyst's queue at a ~1:10,000 base rate. A 1% FPR
against 10 million daily queries is 100,000 false alerts a day per true positive — which is not a
detector, it is a denial-of-service against your own SOC. Report the arithmetic honestly, including if
it is unflattering. This is exactly the "explain when and why your system fails" analysis the brief
closes on, and it connects directly to the cascade's purpose in Ch 8.4.

**8.3** Our numbers against the four papers' benchmarks —
attribute discrepancies to normalisation, feature set, framing, or hyperparameters. **Our hard-framing
number will likely look worse than published results; that is because published results often use the
easy framing.** Say that. It is the strongest possible use of the D4 dual-framing work.

---

### Step 4F — Chapter 8.4 (cascade) + Bonus — 10 points + up to 10 bonus
**Owner: B · Gate: PARALLEL · Est: 1.5 h**

Cascade rationale, block diagram, funnel table, code explanation. Bonus per B.1–B.4 if 3D survived.

---

### Step 4G — Executive Summary — 5 points
**Owner: A · Gate: PARALLEL · Est: 0.75 h**

Max 1 page, written last. Threat; both datasets; four models with headline results; ensemble
performance; and the one-line scientific finding about which feature families survive encryption.

---

### Step 4H — Final assembly 🔴 SINGLE OWNER
**Owner: B · Gate: BLOCKING · Est: 2.0 h · Files: `report/Final_Report.docx` (B owns exclusively)**

**What this means (plain English).**
One person merges everything. A document stitched together by two people reads like it — inconsistent
tense, duplicated explanations, three different names for the same thing. B does the merge; A reviews
the merged result but does not edit the file directly.

**Checklist:**
- [ ] Title page: course name, project title with **T1048.003**, group number, both names + emails, date
- [ ] ≤ 15 pages, 1.5 spacing, Calibri/Arial 11/12
- [ ] Every table and figure numbered and captioned; every one referenced in the text
- [ ] Consistent terminology — one name per concept throughout
- [ ] Exploratory plots moved to appendices to protect the page budget
- [ ] Cross-references correct (Ch 8.3 cites the Ch 6.2 benchmark table)
- [ ] **A reads the assembled document end to end and sends comments to B**

---

## PHASE 5 — SUBMISSION

### Step 5A — AI conversation logs ⚠ MANUAL, REQUIRED
**Owner: A+B · Gate: BLOCKING · Est: 0.5 h each**

The brief is explicit: **full and unedited. Truncated, summarised, or curated logs are treated as
missing**, and AI fingerprints without logs are penalised as undisclosed use.

- [ ] **A** exports their complete session history → `ai_logs/claude_code_teammate_A.json`
- [ ] **B** exports their complete session history → `ai_logs/claude_code_teammate_B.json`
- [ ] Multiple sessions concatenated in chronological order with clear separators
- [ ] Coverage spans planning, debugging, error analysis, and report writing
- [ ] One additional file per any other tool used (ChatGPT, Gemini, …)
- [ ] **Scan the logs for a pasted `HF_TOKEN` before zipping.** Logs are submitted verbatim; if a token
      was ever pasted into a conversation it is now in the deliverable. If found, **rotate the token**
      rather than editing the log — editing it violates the unedited requirement.

### Step 5B — Clean-checkout verification and ZIP
**Owner: A+B · Gate: BLOCKING · Est: 1.0 h**

1. Clone the repo to a **fresh directory**, create a new venv, `pip install -r requirements.txt`, and
   follow the README literally with no prior knowledge. Anything that requires unwritten knowledge is a
   README bug — fix the README, not your memory.
2. `git log -p | grep -i "hf_"` → must be empty. Confirm `.env` was never committed at any point in
   history, not merely absent now.
3. Confirm `data/` and `runs/` are excluded and the ZIP is a sane size.
4. Assemble `Group_X1_X2_Final_Project.zip` (**both student IDs**, per D6) containing:
   - the code repository (`main.py`, `requirements.txt`, `README.md`, all modules)
   - `report/Final_Report.docx`
   - `ai_logs/` with **both** teammates' files
5. **Both teammates unzip it on their own machine and confirm the contents.**

### 🔄 SYNC 5 — pre-submission
- [ ] ZIP verified independently by both teammates
- [ ] Both AI log files present and full-length
- [ ] README verified against a genuinely clean checkout
- [ ] No token in code, history, or logs
- [ ] Submitted **before** Saturday 15 August

---

## Workload Balance

**Baseline = Step 3D deferred (D7).** The "+3D" column shows what happens if we add the bonus back.

| Phase | A (hours) | B (hours) | B if +3D | Notes |
|---|---|---|---|---|
| Phase 0 (joint) | 4.0 | 4.0 | 4.0 | Identical — worked together |
| Phase 1 — Ingestion | 2.0 | 3.25 | 3.25 | B carries `main.py` + dual framing |
| Phase 2 — Analysis & models | 8.25 | 6.75 | 6.75 | A: 2A 1.25 + **2A′ 0.75** + 2B 2.5 + **2C 2.5** (two leakage demos) + 2D 1.25. B carries three models |
| Phase 3 — Ensemble & forensics | 4.0 | **2.0** | 4.5 | 3A + 3B for A; 3C only for B unless 3D returns |
| Phase 4 — Report | 8.75 | 8.5 | 8.5 | A: Ch 1,2,3,4,8.1–8.3, Exec. B: Ch 5,6,7,8.4,App + assembly |
| Phase 5 — Submission | 1.0 | 1.0 | 1.0 | Joint |
| **Total** | **28.0 h** | **25.5 h** | 28.0 h | **52.5 / 47.5 split** at baseline |

Well inside the 60/40 requirement either way.

**How the balance was struck.** A owns more report chapters (6.5 vs 5.5) because B owns **final assembly**,
which is heavy, unglamorous, and must be one person's job. A owns the blocking preprocessing pipeline
*and* the blocking evaluation harness because both gate B's models — so A's Friday morning front-loads
hard and A must resist the temptation to start EDA first. B owns three of the four models but they are
individually smaller than A's EDA + leakage + ranking package.

**The 2 h of slack in B's baseline is deliberate**, not an imbalance to correct. B owns final assembly
(Step 4H), which is the one task on this plan that cannot be parallelised, cannot start early, and sits
directly on the submission deadline. If anything upstream slips, it lands on 4H. Leave the slack there.

**If we fall further behind:** cut the sensitivity sweeps from four values to three (~0.5 h, minimal
rubric cost). After that, A hands the Executive Summary to B or B hands Ch 8.4 drafting to A —
whichever of them is actually ahead at Sync 4, decided then rather than now.

---

## Schedule — ~48 hours, Thursday 13 → Saturday 15 August

| When | A | B |
|---|---|---|
| **Thu evening** | Phase 0 joint (0A → 0D) · **start 0B downloads FIRST** | same |
| | **Sync 1 before bed** | |
| **Fri morning** | 1A loader → **2A pipeline → 2A′ harness** — BLOCKING, do before EDA, B is waiting | 1B loader → 1C main.py |
| | **Sync 2 ~midday** — *2A′ must be pushed by here or B stalls* | |
| **Fri afternoon** | 2B EDA → 2C leakage → 2D XGBoost | 2E IsoForest → 2F deep models |
| **Fri evening** | 3A error analysis | 2G transfer + ablation |
| | **Sync 3 Friday night** | |
| **Sat morning** | 3B before/after → start Ch 1,2 | 3C cascade → **buffer / catch-up** (3D stays deferred unless genuinely ahead) |
| | **Sync 4 ~midday — NUMBERS FREEZE + 3D add-back decision** | |
| **Sat afternoon** | Ch 3,4,8.1–8.3, Exec Summary | Ch 5,6,7,8.4,Bonus, Appendices |
| **Sat evening** | Review B's assembled draft | **4H final assembly** → 5A/5B → submit |

**Critical path:** 0B downloads → 0C headers → 0D schema → 2A pipeline → **2A′ harness** → everything else.
All five are owned by A or joint, and every one of them gates B. Protect them.

**Biggest schedule risk:** Step 0B. Dataset downloads are the one thing on this plan that working harder
cannot accelerate. Start them before Step 0A, not after.
