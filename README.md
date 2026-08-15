# AI-Driven Detection of DNS-Based Data Exfiltration (T1048.003)

Course 3917 — Using AI for Malware and Intrusion Detection, Reichman University, Semester 2 2026.

Guy Hasson (211430046) · Ronie Shmulevich (211897574)

Detects DNS-based exfiltration across the encryption boundary using two datasets — plaintext DNS
(CIC-Bell-DNS-EXF-2021) and encrypted DNS-over-HTTPS (CIRA-CIC-DoHBrw-2020) — mapped onto one unified
11-column feature schema, with four models (XGBoost, Isolation Forest, 1D-CNN, Autoencoder), a hybrid
cascade, and a cross-dataset transfer analysis. Findings are written up in
`report/Final_Report.docx`; per-chapter source drafts are under `report/drafts/`.

## Install

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

`requirements.txt` pins `--extra-index-url https://download.pytorch.org/whl/cpu`, so `torch` resolves to
the CPU-only build automatically — no separate install step, and no CUDA wheels are pulled. Everything in
this project runs on CPU. Developed and verified on Python 3.11.

## Datasets

Raw data is **not** included — the CIC licence permits use but not redistribution, so `data/` is
gitignored. Download both datasets and place them in the exact layout below.

**CIC-Bell-DNS-EXF-2021** (~270 MB) — https://www.unb.ca/cic/datasets/dns-exf-2021.html
(direct index: http://cicresearch.ca/CICDataset/CICBellEXFDNS2021/). Download the **stateless** CSVs —
not the stateful ones. The archive already ships the five directories below; extract them so they sit
directly under `data/exf2021/`, keeping the nested `Attacks/` and `Benign/` subfolders intact:

```
data/exf2021/
  heavy_attacks/Attacks/       stateless_features-heavy_<payload_type>.pcap.csv
  light_attacks/Attacks/       stateless_features-light_<payload_type>.pcap.csv
  heavy_benign/Benign/         stateless_features-benign_heavy_*.pcap.csv
  light_benign/Benign/         stateless_features-light_benign.pcap.csv
  top_level_benign/Benign/     stateless_features-benign_*.pcap.csv
```

All three benign sources are required — the loader concatenates them.

**CIRA-CIC-DoHBrw-2020** — https://www.unb.ca/cic/datasets/dohbrw-2020.html
Complete the request form and download the **CSV** distribution (not the PCAPs — they are enormous and
unnecessary here). Three files go directly in `data/dohbrw2020/`:

```
data/dohbrw2020/
  l1-nondoh.csv        ordinary HTTPS, not DoH
  l2-benign.csv        Benign-DoH only      (19,807 rows)
  l2-malicious.csv     Malicious-DoH only  (249,836 rows)
```

The distribution also contains `l1-doh.csv`; this project never reads it, because its 269,643 rows are
an exact union of the two `l2-` files. You can leave it out.

Exact column names and the header-reconciliation decisions for both datasets are documented in
`docs/header_reconciliation_exf2021.md` and `docs/header_reconciliation_dohbrw2020.md`. If a loader
raises a "no files matched" error, compare your directory against those documents first.

No API keys or tokens are required. `.env.example` exists only for an LLM-arbiter stage that was
scoped out of this submission; you do not need to create a `.env`.

## Running

```bash
python main.py --dataset exf2021 --mode eda
python main.py --dataset dohbrw2020 --framing hard --mode eda
```

Full CLI surface:

```
--dataset  {exf2021,dohbrw2020}              required
--mode     {eda,train,eval,xdataset,cascade} default: eda
--framing  {hard,easy}                       default: hard   (only meaningful for dohbrw2020)
--families {full,intersection,F1_only}       default: full
--config   PATH                              default: config/config.yaml
```

`--framing hard` balances Malicious-DoH down to roughly 1:1 against Benign-DoH and is the headline
framing throughout the report; `easy` is the inflated three-class collapse, reported only as a contrast
case. Every hyperparameter is explicit in `config/config.yaml` — no library defaults are relied on
anywhere in the codebase. Outputs (metrics JSON, figures) are written under `runs/`, also gitignored.

## Tests

```bash
pytest tests/ -q
```

115 tests, all on synthetic fixtures — they run without the raw datasets present and take ~3.5 minutes.

## Project structure

```
main.py               CLI entry point, dispatches on --dataset/--mode
config/               config.yaml — all paths and hyperparameters
schema/               unified.py — the 11-column cross-dataset feature schema
ingestion/            per-dataset loaders (exf2021, dohbrw2020) + registry
preprocessing/        pipeline.py — cleaning, framing, splitting
features/             selection.py — feature-family selection and ranking
models/               supervised.py (XGBoost), unsupervised.py (Isolation Forest),
                      deep.py (1D-CNN, Autoencoder)
ensemble/             cascade.py — the three-stage hybrid cascade
evaluation/           metrics.py, error_analysis.py, cross_dataset.py
tests/                pytest suite
docs/                 header reconciliation notes for both datasets
report/               Final_Report.docx + per-chapter drafts
ai_use.md             GenAI use disclosure
```
