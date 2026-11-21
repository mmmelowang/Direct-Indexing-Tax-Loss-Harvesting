"""MarketData: the backtest-ready dataset built by tlh/data/wrds_loader.py.

All panel fields are (T days x N stocks) numpy arrays aligned on ``dates`` and ``ids``.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class MarketData:
    dates: pd.DatetimeIndex      # trading days, length T
    ids: np.ndarray              # asset ids (e.g. CRSP permno), length N
    ret: np.ndarray              # (T, N) total return incl. dividends (and delisting return on last day); NaN if not trading
    retx: np.ndarray             # (T, N) price return (ex-dividend); NaN if not trading
    mcap: np.ndarray             # (T, N) market cap (any consistent unit; only relative sizes are used); NaN if not trading
    in_index: np.ndarray         # (T, N) bool, point-in-time index membership
    sector: np.ndarray           # (N,) int sector code (see wrds_loader.sic_to_sector)
    names: np.ndarray | None = None              # (N,) tickers, for display
    bench_ret_official: np.ndarray | None = None  # optional (T,) official index total return for validation

    def __post_init__(self):
        T, N = self.ret.shape
        assert self.retx.shape == (T, N) and self.mcap.shape == (T, N)
        assert self.in_index.shape == (T, N) and len(self.dates) == T and len(self.ids) == N
        valid = ~np.isnan(self.ret)
        self.in_index = self.in_index & valid
        # last trading day index per asset (for delisting); -1 if never trades
        any_valid = valid.any(axis=0)
        last = T - 1 - np.argmax(valid[::-1], axis=0)
        self.last_day = np.where(any_valid, last, -1)
        self.delists = any_valid & (self.last_day < T - 1)
        # On a delisting day the delisting return lives in ``ret``; fold it into the
        # price so that the forced liquidation happens at the delisting value.
        self.retx = self.retx.copy()
        cols = np.where(self.delists)[0]
        self.retx[self.last_day[cols], cols] = self.ret[self.last_day[cols], cols]
        # price-return index (split adjusted); NaN outside listing period
        first = np.argmax(valid, axis=0)
        tt = np.arange(T)[:, None]
        self.alive = any_valid & (tt >= first) & (tt <= self.last_day)   # interior gaps are forward-filled
        r = np.where(valid, np.nan_to_num(self.retx), 0.0)
        p = np.cumprod(1.0 + r, axis=0)
        self.price = np.where(self.alive, p, np.nan)
        # cash dividend per $ held, paid at the close of day t
        self.div = np.where(valid, np.nan_to_num(self.ret - self.retx), 0.0)

    @property
    def T(self) -> int:
        return len(self.dates)

    @property
    def N(self) -> int:
        return len(self.ids)

    def bench_weights(self, t: int) -> np.ndarray:
        """Cap weights of index members at the close of day t (sum to 1)."""
        w = np.where(self.in_index[t], np.nan_to_num(self.mcap[t]), 0.0)
        s = w.sum()
        return w / s if s > 0 else w

    def bench_returns(self) -> np.ndarray:
        """Daily total return of the reconstructed cap-weighted index (day 0 = 0)."""
        out = np.zeros(self.T)
        for t in range(1, self.T):
            w = self.bench_weights(t - 1)
            out[t] = np.nansum(w * np.nan_to_num(self.ret[t]))
        return out
