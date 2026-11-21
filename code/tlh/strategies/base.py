"""Strategy interface.

At each rebalance the engine builds a Context and calls ``strategy.trade(ctx)``,
which returns a length-N vector of dollar trades (positive = buy, negative =
sell). The engine sells first (lots chosen by LTFO; loss lots are skipped when
the name was bought in the last 30 days), then buys (only buyable names,
scaled down if cash is short).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Context:
    t: int                     # row index into MarketData
    date: np.datetime64
    price: np.ndarray          # (N,) close prices; NaN if not trading
    shares: np.ndarray         # (N,) shares held
    cash: float
    nav: float
    bench_w: np.ndarray        # (N,) cap weights of index members (sum to 1)
    buyable: np.ndarray        # (N,) bool: in the index, trading, not blocked by a wash sale
    no_loss_sell: np.ndarray   # (N,) bool: bought within 30 days, so loss lots cannot be sold
    book: object               # LotBook
    cfg: object                # Config
    risk: object = None        # RiskModel over ``active`` (only for strategies with needs_risk)
    active: np.ndarray | None = None
    diag: dict = field(default_factory=dict)   # diagnostics returned to the engine

    @property
    def value(self) -> np.ndarray:
        return np.nan_to_num(self.shares * self.price)

    @property
    def target(self) -> np.ndarray:
        """Target dollar holdings: benchmark weights on the invested fraction (1 - cash buffer)."""
        return (1.0 - self.cfg.cash_buffer) * self.bench_w * self.nav


class Strategy:
    name = "base"
    needs_risk = False

    def trade(self, ctx: Context) -> np.ndarray:
        raise NotImplementedError


def drop_small(u: np.ndarray, nav: float, min_w: float) -> np.ndarray:
    """Zero out trades smaller than ``min_w`` of NAV."""
    u = u.copy()
    u[np.abs(u) < min_w * nav] = 0.0
    return u
