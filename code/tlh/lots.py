"""Tax-lot book with LTFO (least-tax-first-out) relief and wash-sale ledger.

Prices are split-adjusted *price-return* indices (built from CRSP ``retx``),
so share counts are in "adjusted share" units and splits need no special
handling. Dividends are paid to cash by the engine and never reinvested.

Lot relief (Moehle et al. 2021, Sec. 3.2): for a sale of a given size, sort
the asset's lots by the per-dollar tax rate

    T_j = rho_j * (1 - basis_j / price),   rho_j in {rho_st, rho_lt}

and sell from the smallest (most negative) T first. This minimizes the tax
of the sale; with a single tax rate it is HIFO.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Lot:
    shares: float
    basis: float          # cost per share (same units as price)
    date: np.datetime64   # acquisition date


@dataclass
class Realization:
    asset: int
    date: np.datetime64
    shares: float
    proceeds: float
    cost: float
    long_term: bool
    acq_date: np.datetime64

    @property
    def gain(self) -> float:
        return self.proceeds - self.cost


class LotBook:
    def __init__(self, n_assets: int, rho_st: float, rho_lt: float, lt_days: int = 365):
        self.n = n_assets
        self.rho_st = rho_st
        self.rho_lt = rho_lt
        self.lt_days = lt_days
        self.lots: dict[int, list[Lot]] = {}
        self.last_buy = np.full(n_assets, np.datetime64("NaT"), dtype="datetime64[D]")
        self.last_loss_sale = np.full(n_assets, np.datetime64("NaT"), dtype="datetime64[D]")
        self.realized: list[Realization] = []
        self._shares = np.zeros(n_assets)

    # ------------------------------------------------------------------ basics
    def shares(self, i: int) -> float:
        return float(self._shares[i])

    def share_vector(self) -> np.ndarray:
        return self._shares.copy()

    def _resync(self, i: int) -> None: #recompute shares for asset i from lots
        self._shares[i] = sum(l.shares for l in self.lots.get(i, ()))

    def is_long_term(self, acq: np.datetime64, today: np.datetime64) -> bool: #>365days
        return (today - acq).astype(int) > self.lt_days

    def total_basis(self) -> float: #total cost 
        return float(sum(l.shares * l.basis for ls in self.lots.values() for l in ls))

    # ------------------------------------------------------------------ views
    def _rate(self, long_term, delta: float):
        """Effective tax rate per dollar of gain: rho_now - delta * rho_lt (delta = 0: current rate only)."""
        return np.where(long_term, self.rho_lt, self.rho_st) - delta * self.rho_lt

    def lot_table(self, i: int, price: float, today: np.datetime64, delta: float = 0.0):
        """Return (values, T, long_term, gain_frac) arrays for asset i's lots.

        T = (rho_now - delta * rho_lt) * (1 - basis / price) is the tax per dollar sold.
        """
        ls = self.lots.get(i, [])
        if not ls:
            z = np.zeros(0)
            return z, z, z.astype(bool), z
        sh = np.array([l.shares for l in ls])
        bs = np.array([l.basis for l in ls])
        lt = np.array([self.is_long_term(l.date, today) for l in ls])
        gain_frac = 1.0 - bs / price
        return sh * price, self._rate(lt, delta) * gain_frac, lt, gain_frac

    # ------------------------------------------------------------------ trades
    def buy(self, i: int, shares: float, price: float, today: np.datetime64) -> None:
        #add a new lot for asset i with new basis and date(today)
        if shares <= 0:
            return
        self.lots.setdefault(i, []).append(Lot(shares, price, today))
        self.last_buy[i] = today
        self._resync(i)

    def sell(self, i: int, shares: float, price: float, today: np.datetime64,
             allow_loss: bool = True, delta: float = 0.0) -> list[Realization]:
        """Sell ``shares`` of asset i using LTFO. Returns realizations.

        If ``allow_loss`` is False, loss lots are skipped (used when a purchase
        happened in the last 30 days, so a loss would be a wash sale).
        ``delta`` orders the lots by the deferral-aware tax (same T as lot_table).
        """
        ls = self.lots.get(i, [])
        if shares <= 0 or not ls:
            return []
        lt = np.array([self.is_long_term(l.date, today) for l in ls])
        T = self._rate(lt, delta) * (1.0 - np.array([l.basis for l in ls]) / price)
        order = np.argsort(T, kind="stable") #sort by T, smallest tax first
        remaining = min(shares, sum(l.shares for l in ls)) #cap betwn shares and total shares
        out: list[Realization] = []
        for j in order:
            if remaining <= 1e-12:
                break
            lot = ls[j]
            if not allow_loss and lot.basis > price: #if not allow loss, and basis>price, skip this lot
                continue
            q = lot.shares if lot.shares - remaining < 1e-9 else remaining   #prevent tiny leftover lots
            r = Realization(i, today, q, q * price, q * lot.basis, #build a sell realization
                            self.is_long_term(lot.date, today), lot.date)
            out.append(r)
            lot.shares -= q #deduct the sell shares
            remaining -= q #deduct the remaining shares to sell
        self.lots[i] = [l for l in ls if l.shares > 1e-9]
        if not self.lots[i]:
            del self.lots[i]
        self._resync(i) #recompute shares for asset i
        if any(r.gain < 0 for r in out): #if any of the realizations is a loss, update last_loss_sale
            self.last_loss_sale[i] = today
        self.realized.extend(out)
        return out

    def sell_all(self, i: int, price: float, today: np.datetime64) -> list[Realization]:
        return self.sell(i, self.shares(i) + 1.0, price, today) #sell all shares of asset i, +1.0 means sell all

    # ------------------------------------------------------------------ wash sale
    def wash_masks(self, today: np.datetime64, window: int = 30):
        """(no_buy, no_loss_sell) boolean masks for today's trading.

        no_buy:       a loss sale happened within the last ``window`` days.
        no_loss_sell: a purchase happened within the last ``window`` days.
        """
        d_ls = (today - self.last_loss_sale).astype("timedelta64[D]")
        d_b = (today - self.last_buy).astype("timedelta64[D]")
        no_buy = ~np.isnat(d_ls) & (d_ls.astype("int64") <= window)
        no_loss_sell = ~np.isnat(d_b) & (d_b.astype("int64") <= window)
        return no_buy, no_loss_sell
