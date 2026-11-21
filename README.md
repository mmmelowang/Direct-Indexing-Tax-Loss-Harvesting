# Tax-Loss Harvesting in a Direct-Indexed S&P 500 Portfolio, 2000–2025

Backtest of tax-aware portfolio optimization (Moehle et al., 2021) on survivorship-bias-free CRSP data.
The report is `report/report.pdf`; this file explains how to run the code and reproduce its results.

## Setup

Python 3.10 or later. From the `code/` folder:

```bash
pip install -r requirements.txt
```

Downloading the data needs a WRDS account with CRSP access (you are asked for your WRDS username and password).
The CRSP data are licensed by WRDS and are not included.

## Reproduce the results

All commands are run inside `code/`.

```bash
# 1. Data: download CRSP from WRDS and build the dataset (a few minutes; about 800 MB)
python -m tlh.data.wrds_loader --start 1999-01-01 --end 2025-12-31 --out data/crsp_sp500.pkl

# 2. Backtests: every run used in the report, saved to results/
python run_experiments.py all

# 3. Figures and tables: saved to ../report/figures/
jupyter nbconvert --to notebook --execute --inplace report_figures.ipynb

# Unit tests (no data needed)
python -m pytest tests -q
```

Step 2 can also be run in parts, in this order:

| Command | What it runs | Time (approx.) |
|---|---|---|
| `python run_experiments.py select` | in-sample (2000–2012) choice of k, S0 trim band, gamma_risk and S1 matched | 5 min |
| `python run_experiments.py main` | full period 2000–2025: S0, S1, S2 | 2 min |
| `python run_experiments.py oos` | out-of-sample 2013–2025: S0, S1, S2 | 1 min |
| `python run_experiments.py windows` | 21 six-year windows: S0, S1, S2, S1 matched | 30 min |
| `python run_experiments.py te_grid` | frontier: S1 and S2 for six tracking-error caps (3 processes) | 30–60 min |

Finished runs are skipped, so an interrupted step can simply be restarted.
The saved results are included in `code/results/`, so step 3 can be run without steps 1–2.

## Folder layout

See `PROJECT_STRUCTURE.md` for a description of every file.
