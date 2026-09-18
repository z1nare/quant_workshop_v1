# Build Your First Quant Trading Strategy

A two-hour, hands-on workshop by the **Quantitative Finance Society** ([quant-soc.com](https://quant-soc.com)) for
first-year students who know basic Python. You investigate a hypothesis on a controlled synthetic market, train a
small predictive model, turn forecasts into risk-limited portfolio weights, evaluate with costs, refactor the research
into two reusable functions, connect them to a supplied trading engine and run an accelerated simulation, with an
optional, supervised Alpaca **paper**-trading demonstration. You leave with a working local application.

> Teaching project. The market is synthetic, the pattern in it is planted, and nothing here is a validated
> real-market investment strategy or investment advice. Systematic trading is one part of quantitative finance.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/z1nare/quant_workshop_v1/blob/v1.0.0/notebooks/workshop_student.ipynb)

## Quickstart A: browser only (Google Colab, no installs, no account beyond Google)

1. Open the student notebook in Colab with the badge above (or `File → Open notebook → GitHub`, paste the repository URL, pick `notebooks/workshop_student.ipynb`).
2. `File → Save a copy in Drive` so your edits persist.
3. Runtime: the default CPU runtime is enough. Run the **Setup** cell: it downloads the pinned workshop release into
   `/content/workshop`, installs the two extra packages, and is safe to re-run.
4. Work through the notebook. Six short exercises (5–15 lines each) with hints and labelled reference solutions.
5. **Export checkpoint** (section 6) downloads a ZIP with your `strategy.py`, model, configuration and the runner.
   Colab runtimes are temporary; saving the notebook does not save those files.
6. Optional: for the paper-trading demo, add Alpaca **paper** keys as Colab Secrets `APCA_API_KEY_ID` and
   `APCA_API_SECRET_KEY` (🔑 panel). Never paste keys into cells.

## Quickstart A2: Kaggle (browser only, alternative to Colab)

Instructor, once: download this repository (Code → Download ZIP) and upload it as a Kaggle **dataset**
(Datasets → New Dataset). Import `notebooks/workshop_student.ipynb` as a Kaggle notebook (File → Import Notebook),
attach the dataset with **Add Input**, make the notebook public and share its link.

1. Open the link and press **Copy & Edit** (free Kaggle account; this route needs no phone verification).
2. Run the **Setup** cell: it copies the workshop from the attached dataset into `/kaggle/working/workshop`. With no
   dataset attached it downloads the pinned release from GitHub instead, which needs **Internet** switched on in the
   notebook settings (Kaggle asks for phone verification before it allows that).
3. **Export checkpoint** (section 6) saves the ZIP under `/kaggle/working`: download it from the **Output** section of
   the right-hand panel before you stop. Kaggle sessions are temporary too.

## Quickstart B: local (Windows, macOS, Linux)

Requirements: Python 3.11 or 3.12 (3.12 is what we test in CI and in Colab), about 500 MB of disk, no GPU.

**Windows (PowerShell)**
```powershell
# 1. on the repository page: Code → Download ZIP, unzip it (no Git needed), then go to the folder that contains
#    requirements.txt. Windows "Extract All" nests it one level deep:
cd quant_workshop_v1-main\quant_workshop_v1-main
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1        # "running scripts is disabled"? run: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
jupyter lab notebooks/workshop_student.ipynb
```

**macOS / Linux**
```bash
cd quant_workshop_v1-main          # the unzipped folder, the one that contains requirements.txt
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/workshop_student.ipynb
```

Then, in two terminals (with the venv active):
```bash
python run_trader.py --mode replay --speed 20      # offline replay of the final-test window; Ctrl+C stops cleanly
streamlit run dashboard.py                         # http://localhost:8501 (reads ./state, never trades)
```
`run_trader.py` without `--bundle` builds the reference bundle from the checked-in data; pass `--bundle workspace/exports/<your_fund>` to run yours.

Paper trading (optional): copy `.env.example` to `.env`, add paper keys, then
```bash
python run_trader.py --mode paper-preview            # real data + account, proposes orders, submits nothing
python run_trader.py --mode paper --i-understand-paper-orders
```

## What is in the repository

| Path | Purpose |
|---|---|
| `notebooks/workshop_student.ipynb` | the workshop (exercises as stubs) |
| `notebooks/workshop_instructor.ipynb` | completed version with presenter notes (generated) |
| `notebooks/workshop_source.py`, `build_notebooks.py` | single source for both notebooks; `--check` keeps them in sync |
| `quantsoc/` | small package: market, features, modeling, portfolio, backtest, artifacts, engine, broker, exercises, viz, widgets |
| `strategy.py` | reference implementation of the learner-facing interface (`predict_returns`, `allocate`) |
| `run_trader.py` | local application: replay / paper-preview / paper |
| `dashboard.py` | Streamlit monitor bound to localhost; reads state, never trades |
| `data/` | checked-in synthetic market + stress scenario + data card; `generate_data.py` regenerates them |
| `docs/` | instructor guide, run sheet, student prep checklist, troubleshooting, VPS tutorial, timing contract, verification report |
| `tests/`, `.github/workflows/ci.yml` | 90+ tests incl. clean-kernel notebook execution; CI needs no credentials |
| `tools/` | `set_release.py`, `calibrate_market.py`, `run_notebook.py` |

## Instructor: publishing configuration you must supply

The Colab route (and the Kaggle route without an attached dataset) downloads a **pinned release** of this repository,
currently `z1nare/quant_workshop_v1` at tag `v1.0.0`. The values (`OWNER`, `REPO`, `REVISION`) live in
`quantsoc/release.py` and the notebook setup cell. To publish a new release, pick a new tag and run:

```bash
python tools/set_release.py --owner <org> --repo <repo> --revision <tag>
git commit -am "Pin workshop release <tag>" && git tag <tag> && git push origin main <tag>
```

This rewrites `quantsoc/release.py`, the notebook source, rebuilds both notebooks and updates the badge above. Notebooks
that students already copied keep downloading the tag they were copied with, so send the new link.

## Documentation

* `docs/instructor_guide.md` and `docs/run_sheet.md`: timed 120-minute plan, misconceptions, recovery checkpoints, abbreviated route
* `docs/student_prep_checklist.md`: what to do before the session (5 minutes)
* `docs/troubleshooting.md`
* `docs/vps_deployment.md`: optional Linux server deployment
* `docs/timing_contract.md`: exactly when data is available and when orders can fill
* `docs/verification_report.md`: what was actually tested, timings, and what remains manual

## Development

```bash
pip install -r requirements-dev.txt
pytest -q -m "not slow"          # unit + engine tests with a mocked broker (~15 s)
pytest -q -m slow                # executes both notebooks from a clean kernel (~2–3 min)
python notebooks/build_notebooks.py --check
python generate_data.py --check
```

License: MIT (see `LICENSE`).
