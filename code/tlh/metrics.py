"""Performance and tax metrics computed from a backtest Result.

Conventions:
  pre-tax return     daily NAV return + year-end tax payment (or refund);
  after-tax return   from NAV, which already includes the annual tax settlements;
                     "no liquidation": final NAV (step-up at death or donation),
                     "liquidation":    final NAV minus the tax on all unrealized gains;
  tax alpha          after-tax annualized return of a strategy minus that of S0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def tax_flows(res) -> pd.Series:
    """Daily tax paid from the account"""
    d = res.daily #daily tax series
    tax = pd.Series(0.0, index=d.index)
    #only fill in the last trading day with the tax paid for that year
    for year, row in res.annual.iterrows():
        tax[d.index[d.index.year == year][-1]] = row.tax
    return tax


def pretax_returns(res) -> pd.Series:
    d = res.daily #pre_tax return = (nav+tax)/nav(t-1) -1
    return ((d.nav + tax_flows(res)) / d.nav.shift(1) - 1).fillna(0.0)


def monthly(r: pd.Series) -> pd.Series: #monthly returns
    return (1 + r).resample("ME").prod() - 1


def annualized(growth: float, years: float) -> float:
    return growth ** (1 / years) - 1


def after_tax_liquidation_value(res) -> pd.Series:
    """nav-tax paid EOY - unrealized tax on all gains (liquidation)"""
    rb = res.rebal
    return rb.nav - rb.ytd_tax - rb.liq_tax


def summary(res) -> dict:
    d, rb, real = res.daily, res.rebal, res.realized
    years = len(d) / 252
    init = res.cfg.initial_cash
    r_pre, r_b = pretax_returns(res), d.bench_ret
    losses = real[real.gain < 0] if len(real) else real
    out = dict(
        pretax=annualized((1 + r_pre).prod(), years),
        index=annualized((1 + r_b).prod(), years),
        after_tax_noliq=annualized(d.nav.iloc[-1] / init, years),
        after_tax_liq=annualized((d.nav.iloc[-1] - res.liq_tax_end) / init, years),
        te_realized=(monthly(r_pre) - monthly(r_b)).std() * np.sqrt(12), #std(monthly ret - bmk monthly ret) * sqrt(12)
        te_ex_ante=rb.te_ex_ante.mean() if "te_ex_ante" in rb else np.nan, #from covariance of daily returns and benchmark returns
        sell_turnover=rb.sells.iloc[1:].sum() / d.nav.mean() / years,
        tcost_bp=1e4 * rb.tcost.iloc[1:].sum() / d.nav.mean() / years,
        st_losses=-losses[~losses.long_term].gain.sum() / init if len(losses) else 0.0, #sum of short term losses / initial cash
        lt_losses=-losses[losses.long_term].gain.sum() / init if len(losses) else 0.0,
        taxes_paid=res.annual.tax.sum(),
        final_nav=d.nav.iloc[-1],
        liq_tax=res.liq_tax_end,
        avg_holdings=rb.n_hold.mean(),
    )
    if "gap_bp" in rb:
        gap = rb.gap_bp.iloc[1:]
        out.update(gap_median_bp=gap.median(), gap_max_bp=gap.max(), share_gap_below_1bp=(gap < 1).mean())
    return out


def compare(results: dict, base: str = "S0") -> pd.DataFrame:
    """Summary table, one column per strategy, with tax alpha relative to ``base``."""
    t = pd.DataFrame({k: summary(v) for k, v in results.items()})
    t.loc["tax_alpha_noliq"] = t.loc["after_tax_noliq"] - t.loc["after_tax_noliq", base]
    t.loc["tax_alpha_liq"] = t.loc["after_tax_liq"] - t.loc["after_tax_liq", base]
    t.loc["pretax_vs_base"] = t.loc["pretax"] - t.loc["pretax", base]
    return t
