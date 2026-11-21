"""Run all backtests used in the report and save them under results/.

    python run_experiments.py select     # in-sample (2000-2012) choice of k and the S0 trim band
    python run_experiments.py main       # 2000-2025: S0, S1, S2 with gamma_risk = 400
    python run_experiments.py oos        # out-of-sample 2013-2025, new account, same settings
    python run_experiments.py windows    # 6-year windows: S0, S1, S2, and S1 matched (gamma_risk = 800)
    python run_experiments.py te_grid    # frontier: S1 and S2 with a tracking-error cap, 2000-2025
    python run_experiments.py all        # select, main, oos, windows, te_grid

Main results, out-of-sample and windows control risk with the penalty gamma_risk = 400, chosen
in-sample (2000-2012) as the value among 100, 200, 400 whose realized tracking error is closest
to 0.5%. S1 matched uses gamma_risk = 800, chosen in-sample so that its tracking error is close
to S2's. Also chosen in-sample and then held fixed:
  - k = 15 factors: forecast accuracy is the same for k = 5 ... 40;
  - S0 trim band 10%: lowest tracking error among none / 25% / 10%.
The frontier is a separate experiment: S1 and S2 with a cap on ex-ante tracking error (TE_GRID)
and a small gamma_risk = 10, so that the cap sets the risk level.
Finished runs are skipped, so an interrupted job can simply be restarted.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one thread per process; parallel runs are much faster this way
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pickle
import sys
import time

import numpy as np
import pandas as pd

from tlh.backtest import run_backtest
from tlh.config import Config
from tlh.metrics import summary
from tlh.risk import fit_pca_model
from tlh.strategies import S0, TAM

DATA = "data/crsp_sp500.pkl"
OUT = "results"
IS_END = "2012-12-31"
OOS_START = "2013-01-01"
WINDOW_STARTS = range(2000, 2021)

FACTOR_GRID = [5, 10, 15, 25, 40]
TRIM_GRID = [None, 0.25, 0.10]
GAMMA_GRID = [100, 200, 400]        # candidates for gamma_risk
MATCH_GRID = [600, 800, 1000]       # candidates for S1 matched (gamma_risk above GAMMA_GRID)
TE_TARGET = 0.005                   # tracking-error target used to choose gamma_risk

TRIM_BAND = 0.10        # chosen in-sample
GAMMA = 400.0           # chosen in-sample: tracking error closest to 0.5%
GAMMA_MATCHED = 800.0   # chosen in-sample: S1 tracking error closest to S2's
FRONTIER_GAMMA = 10.0   # small penalty inside the tracking-error cap (frontier only)
TE_GRID = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02]   # frontier caps (annualized, ex-ante)


def _save(obj, name):
    with open(os.path.join(OUT, name), "wb") as f:
        pickle.dump(obj, f)


def _load(name):
    with open(os.path.join(OUT, name), "rb") as f:
        return pickle.load(f)


def _done(name):
    return os.path.exists(os.path.join(OUT, name))


def strategies(end):
    """S0, S1 and S2 with the in-sample gamma_risk; S2 liquidates at ``end``."""
    return {"S0": S0(trim_band=TRIM_BAND), "S1": TAM(gamma_risk=GAMMA),
            "S2": TAM(gamma_risk=GAMMA, horizon=end)}


def window_strategies(end):
    """Strategies of the 6-year windows: the main ones plus S1 matched to S2's tracking error."""
    return {**strategies(end), "S1_matched": TAM(gamma_risk=GAMMA_MATCHED)}


def frontier_strategies(end, te_max):
    """S1 and S2 with a cap on ex-ante tracking error and a small gamma_risk (frontier only)."""
    return {"S1": TAM(gamma_risk=FRONTIER_GAMMA, te_max=te_max),
            "S2": TAM(gamma_risk=FRONTIER_GAMMA, te_max=te_max, horizon=end)}


def factor_check(md, horizon=63):
    """In-sample forecast accuracy of the risk model for each k (random 100-stock active portfolios)."""
    q = pd.Series(md.dates, index=md.dates).groupby([md.dates.year, md.dates.quarter]).first()
    end = int(np.searchsorted(md.dates, pd.Timestamp(IS_END), side="right"))
    days = [int(np.searchsorted(md.dates, d)) for d in q.values]
    days = [t for t in days if t >= 252 and t + horizon <= end]
    rows = []
    for k in FACTOR_GRID:
        rng = np.random.default_rng(0)
        pairs = []
        for t in days:
            members = np.where(md.in_index[t - 1])[0]
            model = fit_pca_model(md.ret, t, members, k=k, sector=md.sector)
            wb = md.bench_weights(t - 1)[members]
            pick = rng.choice(len(members), 100, replace=False)
            wp = np.zeros(len(members)); wp[pick] = wb[pick]; wp /= wp.sum()
            realized = (np.nan_to_num(md.ret[t:t + horizon][:, members]) @ (wp - wb)).std(ddof=1) * np.sqrt(252)
            pairs.append((model.vol(wp - wb), realized))
        p = np.array(pairs)
        rows.append(dict(k=k, realized_over_forecast=(p[:, 1] / p[:, 0]).mean(), corr=np.corrcoef(p.T)[0, 1]))
    return pd.DataFrame(rows).set_index("k")


def run_select(md):
    """In-sample (2000-2012) choice of the S0 trim band, gamma_risk, S1 matched and k."""
    cfg = Config(end=IS_END)
    runs = {f"S0_trim{t}": S0(trim_band=t) for t in TRIM_GRID}
    runs.update({f"S1_g{g}": TAM(gamma_risk=float(g)) for g in GAMMA_GRID + MATCH_GRID})
    runs["S2"] = TAM(gamma_risk=GAMMA, horizon=cfg.end)
    for key, strat in runs.items():
        name = f"select_{key}.pkl"
        if not _done(name):
            start = time.time()
            _save(summary(run_backtest(md, strat, cfg)), name)
            print(f"select {key}: {time.time() - start:.0f}s", flush=True)
    if not _done("select_factors.pkl"):
        _save(factor_check(md), "select_factors.pkl")
    te = {k: _load(f"select_{k}.pkl")["te_realized"] for k in runs}
    best_trim = min(TRIM_GRID, key=lambda t: te[f"S0_trim{t}"])
    best_gamma = min(GAMMA_GRID, key=lambda g: abs(te[f"S1_g{g}"] - TE_TARGET))
    best_match = min(MATCH_GRID, key=lambda g: abs(te[f"S1_g{g}"] - te["S2"]))
    print(f"in-sample choice: trim band {best_trim}, gamma_risk {best_gamma}, matched gamma {best_match}")
    print(pd.Series(te).mul(100).round(2).rename("te_realized (%)"))
    print(_load("select_factors.pkl").round(3))


def run_main(md):
    cfg = Config()
    for key, strat in strategies(cfg.end).items():
        if not _done(f"main_{key}.pkl"):
            start = time.time()
            _save(run_backtest(md, strat, cfg), f"main_{key}.pkl")
            print(f"main {key}: {time.time() - start:.0f}s", flush=True)


def run_windows(md):
    for y in WINDOW_STARTS:
        cfg = Config(start=f"{y}-01-01", end=f"{y + 5}-12-31")
        for key, strat in window_strategies(cfg.end).items():
            name = f"window_{y}_{key}.pkl"
            if not _done(name):
                s = summary(run_backtest(md, strat, cfg))
                s.update(start=y, strategy=key)
                _save(s, name)
        print(f"window {y}-{y + 5} done", flush=True)


def run_oos(md):
    cfg = Config(start=OOS_START)
    for key, strat in strategies(cfg.end).items():
        if not _done(f"oos_{key}.pkl"):
            start = time.time()
            _save(run_backtest(md, strat, cfg), f"oos_{key}.pkl")
            print(f"oos {key}: {time.time() - start:.0f}s", flush=True)


_MD = None


def _init_worker():
    global _MD
    with open(DATA, "rb") as f:
        _MD = pickle.load(f)


def _te_job(job):
    """One frontier run in a worker process."""
    period, key, te = job
    cfg = Config() if period == "full" else Config(start=OOS_START)
    name = f"te_{period}_{key}_{te:.4f}"
    start = time.time()
    res = run_backtest(_MD, frontier_strategies(cfg.end, te)[key], cfg)
    _save(summary(res), name + ".pkl")
    return f"{name}: {time.time() - start:.0f}s"


def run_te_grid(processes=3):
    """Frontier: S1 and S2 for each tracking-error cap over the full period, run in parallel."""
    from multiprocessing import Pool
    jobs = [(period, key, te) for period in ("full",) for te in TE_GRID
            for key in ("S1", "S2") if not _done(f"te_{period}_{key}_{te:.4f}.pkl")]
    if not jobs:
        print("te_grid: all runs done")
        return
    with Pool(processes, initializer=_init_worker) as pool:
        for msg in pool.imap_unordered(_te_job, jobs):
            print(f"te_grid {msg}", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUT, exist_ok=True)
    if what == "te_grid":
        run_te_grid()
        sys.exit()
    with open(DATA, "rb") as f:
        md = pickle.load(f)
    if what in ("select", "all"):
        run_select(md)
    if what in ("main", "all"):
        run_main(md)
    if what in ("oos", "all"):
        run_oos(md)
    if what in ("windows", "all"):
        run_windows(md)
    if what == "all":
        run_te_grid()
