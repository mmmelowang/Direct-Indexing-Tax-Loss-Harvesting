"""Unit tests for tlh.risk (simulated one-factor world with known covariance)."""
import numpy as np
import pytest

from tlh.risk import fit_pca_model


def simulate(T=1000, n=60, seed=0):
    rng = np.random.default_rng(seed)
    beta = rng.uniform(0.6, 1.4, n)
    f = rng.normal(0, 0.01, T)                      # daily factor vol 1%
    e = rng.normal(0, 0.012, (T, n))                # daily specific vol 1.2%
    return f[:, None] * beta + e, beta


def test_one_factor_recovers_variance():
    ret, beta = simulate()
    m = fit_pca_model(ret, 900, np.arange(60), lookback=800, min_obs=100, k=1)
    w = np.ones(60) / 60
    true_var = (0.01 * beta.mean()) ** 2 + (0.012 ** 2) / 60
    assert m.variance(w) / 252 == pytest.approx(true_var, rel=0.15)


def test_cov_matches_variance():
    ret, _ = simulate()
    m = fit_pca_model(ret, 900, np.arange(60), lookback=500, min_obs=100, k=3)
    w = np.random.default_rng(1).normal(size=60)
    assert w @ m.cov() @ w == pytest.approx(m.variance(w))


def test_no_lookahead():
    ret, _ = simulate()
    ret2 = ret.copy()
    ret2[900:] = 99.0                               # future data must not matter
    a = fit_pca_model(ret, 900, np.arange(60), lookback=500, min_obs=100, k=3)
    b = fit_pca_model(ret2, 900, np.arange(60), lookback=500, min_obs=100, k=3)
    assert np.allclose(a.D, b.D) and np.allclose(a.factor_var, b.factor_var)


def test_short_history_stock_gets_conservative_specific_variance():
    ret, _ = simulate()
    ret[:850, 5] = np.nan                           # stock 5 has only 50 observations
    m = fit_pca_model(ret, 900, np.arange(60), lookback=500, min_obs=100, k=3)
    others = np.delete(m.D, 5)
    assert m.D[5] == pytest.approx(2.0 * np.median(others), rel=0.2)


def test_explained_variance_sums_below_one():
    ret, _ = simulate()
    m = fit_pca_model(ret, 900, np.arange(60), lookback=500, min_obs=100, k=5)
    assert 0 < m.explained.sum() < 1 and np.all(np.diff(m.explained) <= 1e-12)
