"""S1 / S2: tax-aware Markowitz (TAM) strategy of Moehle et al. (2021), with a deferral-aware tax term.

Each rebalance solves the TAM problem (tlh/optimizer.py) over the index members
and current holdings, with one sale variable per tax lot. The tax per dollar
sold of lot l is

    T_l = (rho_now - delta * rho_lt) * (1 - basis_l / price)

S1 (no horizon): delta = 0; a harvested loss is worth the current rate.
S2 (horizon given): delta_t = (1 + rate) ** -(days to the horizon / 365) subtracts the
present value of the extra long-term tax paid at the horizon: harvesting does not change the
total gain taxed over the life of the account, so each dollar of loss realized now is a dollar
of extra gain at liquidation. delta rises to 1 as the horizon approaches.
"""
from __future__ import annotations

import time

import numpy as np

from ..optimizer import TamInputs, TamParams, loss_names, solve_tam
from .base import Context, Strategy, drop_small


class TAM(Strategy):
    needs_risk = True

    def __init__(self, gamma_risk: float = 400.0, gamma_tc: float = 1.0, gamma_tax: float = 1.0,
                 horizon: str | None = None, rate: float = 0.05, te_max: float | None = None,
                 name: str | None = None):
        """horizon: liquidation date for S2 (None gives S1); rate: annual discount rate of the future tax;
        te_max: cap on the ex-ante tracking error after trading (None = no cap)."""
        self.delta = 0.0
        self.horizon = None if horizon is None else np.datetime64(horizon, "D")
        self.rate = rate
        self.params = TamParams(gamma_risk=gamma_risk, gamma_tc=gamma_tc, gamma_tax=gamma_tax, te_max=te_max)
        self.name = name or ("S1" if self.horizon is None else "S2")

    def inputs(self, ctx: Context) -> TamInputs:
        """Optimizer inputs over ctx.active, in weight space."""
        A, nav, cfg = ctx.active, ctx.nav, ctx.cfg
        w = ctx.value[A] / nav
        target = (1 - cfg.cash_buffer) * ctx.bench_w[A]
        # Tax lots' owners, saleable capital, and tax per dollar sold
        owner, cap, T = [], [], []
        sell_cap = np.zeros(len(A))
        for k, i in enumerate(A):
            if ctx.shares[i] <= 0:
                continue
            vals, Tl, _, gain_frac = ctx.book.lot_table(i, ctx.price[i], ctx.date, self.delta)
            #wash sale rule
            if ctx.no_loss_sell[i]:                       # bought within 30 days: loss lots are locked
                vals = np.where(gain_frac >= 0, vals, 0.0)
            for v, t in zip(vals, Tl):
                if v > 0:
                    owner.append(k); cap.append(v / nav); T.append(t)
            sell_cap[k] = vals.sum() / nav
        return TamInputs(x=w - target, sell_cap=sell_cap, T=np.zeros(len(A)), buyable=ctx.buyable[A],
                         B=ctx.risk.loadings(), D=ctx.risk.D, cash=ctx.cash / nav, kappa=cfg.tc,
                         eta=cfg.cash_buffer, lot_asset=np.array(owner, int), lot_cap=np.array(cap),
                         lot_T=np.array(T))

    def trade(self, ctx: Context) -> np.ndarray:
        """Update delta (S2 only), solve the TAM problem, and return dollar trades per name.
        Solver diagnostics go to ctx.diag; if the solve fails, no trade is made."""
        start = time.time()
        if self.horizon is not None:
            years = max((self.horizon - ctx.date).astype(int), 0) / 365
            self.delta = (1 + self.rate) ** -years
        P = self.inputs(ctx)
        res = solve_tam(P, self.params)
        ctx.diag.update(delta=self.delta, status=res.status, n_loss_names=int(loss_names(P).sum()), n_opt_lots=len(P.lot_asset),
                        objective=res.objective, lower_bound=res.lower_bound, gap_bp=res.gap * 1e4,
                        solve_s=time.time() - start, te_planned=np.nan, te_slack=res.te_slack)
        if res.status == "ok":
            ctx.diag["te_planned"] = ctx.risk.vol(P.x + res.u)   # ex-ante TE of the planned trade list
        u = np.zeros(len(ctx.price))
        if res.status == "ok":
            u[ctx.active] = res.u * ctx.nav #convert to dollar trades
        return drop_small(u, ctx.nav, ctx.cfg.min_trade_weight)
