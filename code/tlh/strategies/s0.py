"""S0: direct index without loss harvesting (the baseline for tax alpha).

At each rebalance:
  1. sell every holding that is no longer in the index;
  2. sell down positions above (1 + trim_band) times their target weight;
  3. if cash is below the buffer (e.g. after a tax payment), sell holdings pro rata;
  4. invest the cash above the buffer in the names below their index weight, in
     proportion to the shortfall; any remainder pro rata to index weights.

S0 never sells to realize a loss.  Sales use LTFO lot relief, so the tax alpha of
the other strategies comes from harvesting, not from better lot accounting.

S0 returns u: vector of trades in units of NAV.
"""
from __future__ import annotations

import numpy as np

from .base import Context, Strategy, drop_small


class S0(Strategy): #inherits from Strategy, overrides the trade method to implement the S0 strategy
    name = "S0"

    def __init__(self, trim_band: float | None = 0.10):
        self.trim_band = trim_band      # None disables trimming

    def trade(self, ctx: Context) -> np.ndarray:
        kappa, eta = ctx.cfg.tc, ctx.cfg.cash_buffer
        value = ctx.value
        u = np.zeros_like(value)

        removed = (value > 0) & (ctx.bench_w == 0)
        u[removed] = -value[removed]
        if self.trim_band is not None:
            over = ~removed & (value > (1 + self.trim_band) * ctx.target)
            u[over] = ctx.target[over] - value[over]
        removed = u < 0

        available = ctx.cash + (1 - kappa) * (-u[removed]).sum() - eta * ctx.nav
        if available < 0:
            keep = (value > 0) & ~removed
            u[keep] -= (-available / (1 - kappa)) * value[keep] / value[keep].sum()
        elif available > 0:
            budget = available / (1 + kappa)
            shortfall = np.where(ctx.buyable, np.maximum(ctx.target - value, 0.0), 0.0)
            if shortfall.sum() >= budget:
                u += budget * shortfall / shortfall.sum()
            else:
                u += shortfall
                w = np.where(ctx.buyable, ctx.bench_w, 0.0)
                u += (budget - shortfall.sum()) * w / w.sum()
            # concentrate the budget on purchases above the minimum trade size,
            # so that small allocations are not dropped and cash does not pile up
            buys = np.maximum(u, 0.0)
            keep = buys >= ctx.cfg.min_trade_weight * ctx.nav
            if keep.any():
                u = np.minimum(u, 0.0) + np.where(keep, buys, 0.0) * buys.sum() / buys[keep].sum()
        return drop_small(u, ctx.nav, ctx.cfg.min_trade_weight)
