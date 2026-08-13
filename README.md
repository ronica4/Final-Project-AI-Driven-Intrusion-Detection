# AI-Driven Detection of DNS-Based Data Exfiltration (T1048.003)

Course 3917 — Using AI for Malware and Intrusion Detection, Reichman University, Semester 2 2026.

See `PROJECT_PLAN.md` for the full design, decisions log, and progress tracker.

## Quick start

> This section is verified against a clean checkout at Sync 5 — do not trust it blindly until then.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Datasets are **not** included in the repo (see `.gitignore`). Download per Step 0B of
`PROJECT_PLAN.md` and place under:

```
data/exf2021/
data/dohbrw2020/
```

Copy `.env.example` to `.env` and fill in `HF_TOKEN` (required only for the LLM cascade stage).

## Running

```bash
python main.py --dataset exf2021 --mode eda --config config/config.yaml
python main.py --dataset dohbrw2020 --framing hard --mode eda --config config/config.yaml
```

Full CLI surface: `--dataset {exf2021,dohbrw2020} --mode {eda,train,eval,xdataset,cascade}
--framing {hard,easy} --families {full,intersection} --config config/config.yaml`

## Project structure

See "Project Structure" in `PROJECT_PLAN.md` for the full file layout and ownership map.
