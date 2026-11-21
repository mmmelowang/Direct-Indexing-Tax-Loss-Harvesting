"""Capital-gains tax: annual settlement and liquidation tax.

Net short-term and net long-term gains are each taxed at their own rate:
tax = rho_st * net_ST + rho_lt * net_LT. A negative tax is a refund: the
losses offset the investor's gains elsewhere. 

The engine settles the tax in cash on the last trading day of each calendar
year: a positive tax is paid from the account, a refund is added to it.
"""
from __future__ import annotations


def net_gains(realizations) -> tuple[float, float]:
    """(net short-term gain, net long-term gain) of a list of lot realizations."""
    st = sum(r.gain for r in realizations if not r.long_term)
    lt = sum(r.gain for r in realizations if r.long_term)
    return st, lt


def settle_year(net_st: float, net_lt: float, cfg) -> float:
    """Tax due for one year (negative = refund)."""
    return cfg.rho_st * net_st + cfg.rho_lt * net_lt #tax = 0.408 * net_st + 0.238 * net_lt


def liquidation_tax(lot_rows, cfg) -> float:
    """assume every lot were sold at each period end. ``lot_rows`` iterates (gain, long_term).

    Nothing is traded; the number is used for the liquidation convention of
    after-tax return.
    """
    rows = list(lot_rows)
    st = sum(g for g, is_lt in rows if not is_lt)
    lt = sum(g for g, is_lt in rows if is_lt)
    return settle_year(st, lt, cfg)
