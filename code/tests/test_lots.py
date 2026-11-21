"""Unit tests for tlh.lots.LotBook (hand-computed cases)."""
import numpy as np
import pytest

from tlh.lots import LotBook

D = np.datetime64
RHO_ST, RHO_LT = 0.408, 0.238


def book():
    return LotBook(3, RHO_ST, RHO_LT)


def test_buy_creates_separate_lots():
    b = book()
    b.buy(0, 100, 100.0, D("2020-01-02"))
    b.buy(0, 50, 120.0, D("2020-06-01"))
    assert len(b.lots[0]) == 2
    assert b.shares(0) == 150
    assert b.total_basis() == pytest.approx(100 * 100 + 50 * 120)


def test_sell_picks_lowest_tax_lot_first():
    b = book()
    b.buy(0, 100, 100.0, D("2020-01-02"))   # long-term by 2021-03-01, gain
    b.buy(0, 100, 120.0, D("2020-06-01"))   # short-term, loss
    out = b.sell(0, 100, 110.0, D("2021-03-01"))
    assert len(out) == 1
    r = out[0]
    assert r.cost == pytest.approx(100 * 120)
    assert r.gain == pytest.approx(-1000.0)
    assert not r.long_term
    assert b.lots[0][0].basis == 100.0


def test_ltfo_differs_from_hifo():
    # price 110: X long-term basis 100 -> T = 0.0216; Y short-term basis 103 -> T = 0.0260.
    # HIFO would sell Y (higher basis); LTFO sells X (lower tax per dollar).
    b = book()
    b.buy(0, 10, 100.0, D("2020-01-02"))
    b.buy(0, 10, 103.0, D("2020-12-01"))
    out = b.sell(0, 10, 110.0, D("2021-03-01"))
    assert out[0].cost == pytest.approx(10 * 100.0)
    assert out[0].long_term


def test_partial_lot_sale():
    b = book()
    b.buy(0, 100, 100.0, D("2020-01-02"))
    out = b.sell(0, 40, 90.0, D("2020-03-01"))
    assert out[0].shares == 40
    assert b.shares(0) == pytest.approx(60)
    assert out[0].gain == pytest.approx(40 * (90 - 100))


def test_long_term_threshold_is_more_than_365_days():
    b = book()
    b.buy(0, 1, 100.0, D("2020-01-02"))
    assert not b.is_long_term(D("2020-01-02"), D("2021-01-01"))   # 365 days
    assert b.is_long_term(D("2020-01-02"), D("2021-01-02"))       # 366 days


def test_wash_sale_masks():
    b = book()
    b.buy(0, 100, 100.0, D("2020-01-02"))
    b.sell(0, 100, 80.0, D("2020-02-10"))                          # loss sale
    no_buy, _ = b.wash_masks(D("2020-03-11"))                      # 30 days later
    assert no_buy[0] and not no_buy[1]
    no_buy, _ = b.wash_masks(D("2020-03-12"))                      # 31 days later
    assert not no_buy[0]


def test_recent_buy_blocks_loss_sale():
    b = book()
    b.buy(0, 100, 100.0, D("2020-01-02"))
    b.buy(0, 100, 90.0, D("2020-02-20"))
    _, no_loss_sell = b.wash_masks(D("2020-03-01"))
    assert no_loss_sell[0]
    out = b.sell(0, 200, 95.0, D("2020-03-01"), allow_loss=False)
    assert all(r.gain >= 0 for r in out)                           # only the basis-90 lot sold
    assert b.shares(0) == pytest.approx(100)


def test_sell_more_than_held_sells_everything():
    b = book()
    b.buy(0, 10, 50.0, D("2020-01-02"))
    b.sell_all(0, 60.0, D("2020-02-01"))
    assert b.shares(0) == 0 and 0 not in b.lots


def test_delta_changes_lot_order():
    # a long-term loss lot (basis 120) and a short-term loss lot (basis 110), price 100.
    # delta = 0: the long-term lot has the lower T (-0.0476 vs -0.0408) and is sold first.
    # delta = 1: the long-term loss is worth nothing (rho_lt - rho_lt = 0), so the short-term lot goes first.
    b = LotBook(1, RHO_ST, RHO_LT)
    b.buy(0, 10, 120.0, D("2019-01-02"))     # long-term loss at 100
    b.buy(0, 10, 110.0, D("2020-12-01"))     # short-term loss at 100
    _, T0, lt, _ = b.lot_table(0, 100.0, D("2021-03-01"), delta=0.0)
    _, T1, _, _ = b.lot_table(0, 100.0, D("2021-03-01"), delta=1.0)
    assert T0 == pytest.approx([0.238 * (1 - 1.2), 0.408 * (1 - 1.1)])
    assert T1 == pytest.approx([0.0, (0.408 - 0.238) * (1 - 1.1)])
    out = b.sell(0, 10, 100.0, D("2021-03-01"), delta=1.0)
    assert out[0].cost == pytest.approx(10 * 110.0)          # short-term loss lot first
