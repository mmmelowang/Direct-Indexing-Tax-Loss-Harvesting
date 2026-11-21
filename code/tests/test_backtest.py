"""Engine tests on tiny markets whose results can be computed by hand."""
import numpy as np
import pandas as pd
import pytest

from tlh.backtest import run_backtest
from tlh.config import Config
from tlh.data.market import MarketData
from tlh.strategies import S0

KAPPA = 5e-4


def market(start, T, N, mcap):
    dates = pd.bdate_range(start, periods=T)
    ret = np.zeros((T, N))
    return dates, ret, ret.copy(), np.tile(np.asarray(mcap, float), (T, 1)), np.ones((T, N), bool)


def build(dates, ret, retx, mcap, in_index):
    N = ret.shape[1]
    return MarketData(dates, np.arange(N), ret, retx, mcap, in_index, np.zeros(N, int),
                      np.array([f"S{i}" for i in range(N)]))


def run(md):
    cfg = Config(start=str(md.dates[0].date()), end=str(md.dates[-1].date()))
    return run_backtest(md, S0(), cfg)


def initial_buy():
    return (1e6 * 0.995) / (1 + KAPPA)


def test_flat_prices_nav_only_loses_initial_cost():
    md = build(*market("2010-01-04", 60, 3, [3, 2, 1]))
    res = run(md)
    assert res.daily.nav.iloc[-1] == pytest.approx(1e6 - KAPPA * initial_buy(), abs=1e-6)
    assert res.realized.empty and res.annual.tax.abs().sum() == 0


def test_dividend_goes_to_cash():
    dates, ret, retx, mcap, idx = market("2010-01-04", 60, 3, [3, 2, 1])
    ret[10, 0] = 0.01                                    # 1% dividend on stock 0, price unchanged
    res = run(build(dates, ret, retx, mcap, idx))
    value0 = initial_buy() * 3 / 6
    jump = res.daily.nav.iloc[10] - res.daily.nav.iloc[9]
    assert jump == pytest.approx(0.01 * value0)


def test_delisting_is_liquidated_at_delisting_return():
    dates, ret, retx, mcap, idx = market("2010-01-04", 60, 3, [3, 2, 1])
    ret[40, 2] = retx[40, 2] = -0.30                     # delisting return on the last day
    ret[41:, 2] = retx[41:, 2] = np.nan
    res = run(build(dates, ret, retx, mcap, idx))
    value2 = initial_buy() * 1 / 6
    assert res.daily.nav.iloc[40] - res.daily.nav.iloc[39] == pytest.approx(-0.30 * value2)
    r = res.realized[res.realized.asset == 2]
    assert len(r) == 1 and r.gain.iloc[0] == pytest.approx(-0.30 * value2) and not r.long_term.iloc[0]


def test_index_change_sells_deletion_and_buys_addition():
    dates, ret, retx, mcap, idx = market("2010-01-04", 60, 4, [3, 2, 1, 1])
    idx[:, 3] = False
    idx[35:, 3] = True                                   # stock 3 joins
    idx[35:, 1] = False                                  # stock 1 leaves (keeps trading)
    res = run(build(dates, ret, retx, mcap, idx))
    reb = res.rebal
    after = reb.index[reb.index >= dates[35]][0]          # first rebalance after the change
    sold = res.realized[res.realized.date == after]
    assert set(sold.asset) == {1}
    assert reb.loc[after, "n_hold"] == 3
    assert reb.loc[after, "cash"] == pytest.approx(0.005 * reb.loc[after, "nav"], rel=1e-3)


def test_year_end_tax_on_realized_gain():
    dates, ret, retx, mcap, idx = market("2010-11-01", 80, 3, [3, 2, 1])
    ret[5, 1] = retx[5, 1] = 0.10                        # stock 1 gains 10%
    idx[10:, 1] = False                                  # then leaves the index
    res = run(build(dates, ret, retx, mcap, idx))
    gain = initial_buy() * 2 / 6 * 0.10
    assert res.realized.gain.sum() == pytest.approx(gain)
    assert res.annual.loc[2010, "tax"] == pytest.approx(0.408 * gain)
    t = list(dates).index(pd.Timestamp("2010-12-31"))
    assert res.daily.cash.iloc[t] - res.daily.cash.iloc[t - 1] == pytest.approx(-0.408 * gain)
