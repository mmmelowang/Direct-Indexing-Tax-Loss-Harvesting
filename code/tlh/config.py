"""Backtest configuration.

Monetary quantities are in dollars. Inside the optimizer every quantity is
expressed as a fraction of NAV ("weight space"), so gamma_risk does not need
rescaling with account size.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # account
    initial_cash: float = 1_000_000.0
    cash_buffer: float = 0.005          # eta: target cash weight after each trade

    # schedule
    start: str = "2000-01-03"
    end: str = "2025-12-31"
    min_days_between_trades: int = 31   # first trading day >= 31 calendar days after the last trade
    wash_sale_days: int = 30

    # taxes
    rho_st: float = 0.408               # short-term rate: 37% + 3.8% NIIT
    rho_lt: float = 0.238               # long-term rate: 20% + 3.8% NIIT
    lt_holding_days: int = 365          # held more than this many days -> long-term
    # Losses offset the investor's outside gains of the same character immediately,
    # so tax = rho_st * net ST + rho_lt * net LT (negative = refund).

    # trading costs
    tc_bps: float = 5.0                 # one-way cost per dollar traded
    min_trade_weight: float = 2e-5      # trades below 0.2 bp of NAV are dropped

    # risk model
    risk_lookback: int = 252
    risk_min_obs: int = 126
    risk_n_factors: int = 15

    @property
    def tc(self) -> float:
        return self.tc_bps / 1e4
