"""Daily backtest engine.

Each trading day, in this order:
  1. dividends of the day are paid into cash (not reinvested);
  2. holdings that delist today are liquidated at the delisting value (no cost);
  3. on a rebalance day (the start, then the first trading day at least
     ``min_days_between_trades`` calendar days after the last trade) the strategy
     returns dollar trades, which are executed at the close: sells first via the
     LotBook (LTFO), then buys, each paying the one-way cost ``tc``;
  4. on the last trading day of a calendar year, the tax on that year's net
     short-term and long-term gains is paid from cash (a refund if negative);
  5. NAV is recorded.

"""
from __future__ import annotations

from dataclasses import dataclass
import os
import pickle
import time

import numpy as np
import pandas as pd

from .config import Config
from .lots import LotBook
from .risk import fit_pca_model
from .strategies.base import Context, Strategy
from .tax import liquidation_tax, net_gains, settle_year


@dataclass
class Result:
    name: str
    daily: pd.DataFrame       # nav, cash, bench_ret (reconstructed index total return)
    rebal: pd.DataFrame       # one row per rebalance, after trading
    annual: pd.DataFrame      # one row per calendar year: realized gains and tax
    realized: pd.DataFrame    # every lot sale
    liq_tax_end: float        # tax if all holdings were sold on the last day
    cfg: Config


def _lot_gains(book: LotBook, price: np.ndarray, today):
    """output (unrealized gain, long_term) for every open lot; used to compute liquidation tax"""
    for i, lots in book.lots.items():
        for lot in lots:
            yield lot.shares * (price[i] - lot.basis), book.is_long_term(lot.date, today)


def _split(realizations) -> dict:
    """Sum lot-sale gains into st_gain, st_loss, lt_gain, lt_loss."""
    out = dict(st_gain=0.0, st_loss=0.0, lt_gain=0.0, lt_loss=0.0)
    for r in realizations:
        key = "lt" if r.long_term else "st"
        out[f"{key}_gain" if r.gain >= 0 else f"{key}_loss"] += r.gain
    return out


def run_backtest(md, strategy: Strategy, cfg: Config | None = None, verbose: bool = False,
                 checkpoint: str | None = None, time_limit: float | None = None) -> Result | None:
    """Run one backtest. With ``checkpoint`` and ``time_limit`` (seconds), a run that takes longer
    saves its state to ``checkpoint`` and returns None; calling again resumes from that point."""
    cfg = cfg or Config()
    started = time.time()
    dates = md.dates.values.astype("datetime64[D]")
    t0 = int(np.searchsorted(md.dates, pd.Timestamp(cfg.start))) #fist trading day
    t1 = int(np.searchsorted(md.dates, pd.Timestamp(cfg.end), side="right")) - 1 # last trading day

    book = LotBook(md.N, cfg.rho_st, cfg.rho_lt, cfg.lt_holding_days)
    cash = cfg.initial_cash
    nav_hist = np.zeros(t1 - t0 + 1) #array to store the NAV each day
    cash_hist = np.zeros(t1 - t0 + 1) 
    rebal_rows, annual_rows = [], []
    last_trade = None
    year_start = 0          # index into book.realized where the current tax year starts
    period_start = 0        # index into book.realized where the current rebalance period starts
    t_start = t0
    state_keys = ["book", "cash", "nav_hist", "cash_hist", "rebal_rows", "annual_rows", "last_trade",
                  "year_start", "period_start", "strategy"]
    if checkpoint and os.path.exists(checkpoint):   # resume an interrupted run
        with open(checkpoint, "rb") as f:
            state = pickle.load(f)
        (book, cash, nav_hist, cash_hist, rebal_rows, annual_rows, last_trade,
         year_start, period_start, strategy) = (state[k] for k in state_keys)
        t_start = state["t"]

    for t in range(t_start, t1 + 1): #within each year loop through each trading day
        if checkpoint and time_limit and time.time() - started > time_limit:
            values = (book, cash, nav_hist, cash_hist, rebal_rows, annual_rows, last_trade,
                      year_start, period_start, strategy)
            with open(checkpoint, "wb") as f:
                pickle.dump(dict(zip(state_keys, values), t=t), f)
            return None
        today = dates[t]
        px = md.price[t]

        # 1. dividends: add to cash for div = shares * price * div_yield
        if t > t0:
            sh = book.share_vector() #share  vector
            held = sh > 0
            cash += float(np.nansum(sh[held] * md.price[t - 1, held] * md.div[t, held]))

        # 2. delistings: delist currently hold, will delist today stock
        sh = book.share_vector()
        for i in np.where((sh > 0) & md.delists & (md.last_day == t))[0]:
            cash += sum(r.proceeds for r in book.sell_all(i, px[i], today))

        # 3. rebalance: after 31 days since last trade
        if last_trade is None or (today - last_trade).astype(int) >= cfg.min_days_between_trades:
            sh = book.share_vector()
            value = np.nan_to_num(sh * px)
            nav = value.sum() + cash
            bw = md.bench_weights(t)
            alive = ~np.isnan(px) #today has price
            dying = md.delists & (md.last_day <= t)
            no_buy, no_loss_sell = book.wash_masks(today, cfg.wash_sale_days) #buy/sold in past 30 days -> can't buy/sell at loss
            buyable = md.in_index[t] & alive & ~dying & ~no_buy

            risk = active = None
            if strategy.needs_risk: #s1, s2 stategy
                active = np.where(((sh > 0) & alive) | (bw > 0))[0] #active stock = already hold&in benchmark
                risk = fit_pca_model(md.ret, t, active, cfg.risk_lookback, cfg.risk_min_obs,
                                     cfg.risk_n_factors, sector=md.sector)
            ctx = Context(t, today, px, sh, cash, nav, bw, buyable, no_loss_sell, book, cfg, risk, active)
            u = np.where(alive, np.nan_to_num(strategy.trade(ctx)), 0.0) #trade vector for each stock, positive = buy, negative = sell
           
            #sell first at LTFO, then buy; each paying the one-way cost tc
            sell_val = 0.0
            for i in np.where(u < 0)[0]:
                q = min(-u[i] / px[i], sh[i]) #sell amount <= current holding
                #realized G/L from the sale
                rs = book.sell(i, q, px[i], today, allow_loss=not no_loss_sell[i],
                               delta=getattr(strategy, "delta", 0.0))
                sell_val += sum(r.proceeds for r in rs)
            cash += (1 - cfg.tc) * sell_val

            #then buy
            sold_at_loss_today = book.last_loss_sale == today #can't buy stock was sold today
            want = np.where((u > 0) & buyable & ~sold_at_loss_today, u, 0.0)
            need = (1 + cfg.tc) * want.sum() #total cash needed
            scale = 1.0 if need <= cash else max(cash, 0.0) / need #if cash is not enough, scaled down
            buy_val = 0.0
            for i in np.where(want > 0)[0]:
                amt = want[i] * scale
                book.buy(i, amt / px[i], px[i], today) #buy a new lot
                buy_val += amt 
            cash -= (1 + cfg.tc) * buy_val
            #record rebalance info
            sh = book.share_vector()
            value = np.nan_to_num(sh * px)
            mv = value.sum()
            nav_post = mv + cash
            basis = book.total_basis()
            ytd_st, ytd_lt = net_gains(book.realized[year_start:])
            row = dict(date=md.dates[t], nav=nav_post, mv=mv, cash=cash, buys=buy_val, sells=sell_val,
                       tcost=cfg.tc * (buy_val + sell_val), n_hold=int((sh > 0).sum()),
                       n_lots=sum(len(v) for v in book.lots.values()), basis=basis,
                       unrealized=mv - basis, ytd_tax=settle_year(ytd_st, ytd_lt, cfg),
                       liq_tax=liquidation_tax(_lot_gains(book, px, today), cfg), buy_scale=scale)
            row.update(_split(book.realized[period_start:]))
            period_start = len(book.realized)
            if risk is not None:
                row["te_ex_ante"] = risk.vol(value[active] / nav_post - bw[active])
            row.update(ctx.diag)
            rebal_rows.append(row)
            last_trade = today
            if verbose and len(rebal_rows) % 60 == 0:
                print(f"{strategy.name} {md.dates[t].date()} nav={nav_post:,.0f} holdings={row['n_hold']}")

        # 4. year-end tax, paid from cash
        if t == t1 or md.dates[t + 1].year != md.dates[t].year:
            rs = book.realized[year_start:]
            year_start = len(book.realized)
            st, lt = net_gains(rs)
            tax = settle_year(st, lt, cfg)
            cash -= tax
            annual_rows.append(dict(year=md.dates[t].year, net_st=st, net_lt=lt, tax=tax, **_split(rs)))

        # 5. record everyday NAV and cash
        nav_hist[t - t0] = float(np.nansum(book.share_vector() * px)) + cash
        cash_hist[t - t0] = cash

    if checkpoint and os.path.exists(checkpoint):
        os.remove(checkpoint)
    bench = md.bench_returns()[t0:t1 + 1].copy()
    bench[0] = 0.0
    daily = pd.DataFrame(dict(nav=nav_hist, cash=cash_hist, bench_ret=bench), index=md.dates[t0:t1 + 1])
    realized = pd.DataFrame([dict(date=pd.Timestamp(r.date), asset=r.asset, shares=r.shares,
                                  proceeds=r.proceeds, cost=r.cost, gain=r.gain, long_term=r.long_term)
                             for r in book.realized])
    liq_end = liquidation_tax(_lot_gains(book, md.price[t1], dates[t1]), cfg)
    return Result(strategy.name, daily, pd.DataFrame(rebal_rows).set_index("date"),
                  pd.DataFrame(annual_rows).set_index("year"), realized, liq_end, cfg)
