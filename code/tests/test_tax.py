"""Unit tests for tlh.tax (hand-computed cases)."""
import pytest

from tlh.config import Config
from tlh.tax import liquidation_tax, net_gains, settle_year
from tlh.lots import Realization

CFG = Config()


def test_gain_and_loss():
    assert settle_year(1000, 0, CFG) == pytest.approx(408)
    assert settle_year(-1000, 0, CFG) == pytest.approx(-408)       # refund
    assert settle_year(-1000, 2000, CFG) == pytest.approx(-408 + 476)


def test_net_gains_splits_by_character():
    rs = [Realization(0, None, 1, 120, 100, False, None),    # ST +20
          Realization(1, None, 1, 80, 100, False, None),     # ST -20
          Realization(2, None, 1, 150, 100, True, None)]     # LT +50
    assert net_gains(rs) == (pytest.approx(0.0), pytest.approx(50.0))


def test_liquidation_tax():
    rows = [(1000, True), (-400, False)]
    assert liquidation_tax(rows, CFG) == pytest.approx(0.238 * 1000 - 0.408 * 400)
