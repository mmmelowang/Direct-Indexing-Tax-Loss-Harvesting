"""Statistical (PCA) factor risk model: V = X Sigma X' + D, annualized.

Stand-in for the commercial Barra model used by Moehle et al. (2021). The model
is re-estimated at each rebalance date from the trailing ``lookback`` daily
returns, using only data strictly before that date.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RiskModel:
    assets: np.ndarray        # (n,) column indices into MarketData
    X: np.ndarray             # (n, k) factor exposures
    factor_var: np.ndarray    # (k,) annualized factor variances (Sigma is diagonal for PCA factors)
    D: np.ndarray             # (n,) annualized specific variances
    explained: np.ndarray     # (k,) fraction of total variance explained by each factor

    @property
    def k(self) -> int:
        return self.X.shape[1]

    def loadings(self) -> np.ndarray:
        """B (n, k) with X Sigma X' = B B'. Used by the second-order-cone formulation."""
        return self.X * np.sqrt(self.factor_var)

    def cov(self) -> np.ndarray:
        """covariance matrix V = B B' + diag(D)"""
        B = self.loadings()
        return B @ B.T + np.diag(self.D)

    def variance(self, w: np.ndarray) -> float:
        """Ex-ante variance of a weight vector over assets."""
        b = self.loadings().T @ w # portfolio's exposure to each factor
        return float(b @ b + np.sum(self.D * w ** 2)) #factor risk + specific risk

    def vol(self, w: np.ndarray) -> float:
        """Ex-ante annualized volatility (tracking error if ``w`` is an active-weight vector)."""
        return float(np.sqrt(self.variance(w)))


def fit_pca_model(ret: np.ndarray, t: int, assets: np.ndarray, lookback: int = 252,
                  min_obs: int = 126, k: int = 15, ann: int = 252,
                  sector: np.ndarray | None = None, d_floor_frac: float = 0.1) -> RiskModel:
    """Fit the model on rows ``[t - lookback, t)`` of ``ret`` for the given ``assets``.

    Stocks with at least ``min_obs`` observations in the window(half year) enter the PCA.
    Younger stocks(just IPO) get the average exposure of their sector peers and a
    conservative specific variance (twice the median). Specific variances are
    floored at ``d_floor_frac`` times the median.
    """
    #window period; choose stock with >=126 days and enter PCA
    R = ret[max(0, t - lookback):t][:, assets]
    n = len(assets)
    obs = (~np.isnan(R)).sum(axis=0)
    good = obs >= min_obs
    if good.sum() <= k:
        raise ValueError(f"only {good.sum()} stocks with >= {min_obs} observations; need more than k={k}")

    Rg = R[:, good]
    Rg = np.where(np.isnan(Rg), 0.0, Rg - np.nanmean(Rg, axis=0))   # missing days contribute zero; and demean stock return
    U, S, Vt = np.linalg.svd(Rg, full_matrices=False) #SVD decompose Rg
    Vk = Vt[:k].T                                # (keep the top $k$ directions (stocks × $k$). This is the exposure matrix X, each stock's coefficient on each factor
    fac = Rg @ Vk                                # daily return of each factor (T × k).
    resid = Rg - fac @ Vk.T

    X = np.zeros((n, k))
    D = np.zeros(n)
    X[good] = Vk #good stock get their PCA exposures
    D[good] = resid.var(axis=0, ddof=1) * ann #good stock get specific variance
    factor_var = fac.var(axis=0, ddof=1) * ann #var of each factor return, annualized
    explained = S[:k] ** 2 / np.sum(S ** 2) #each factor's contribution to total variance; S^2 is var of each factor

    '''
    fill in young stocks(<126 days data) with sector peer's avg exposure, var = 2*med(good stock)
    '''
    med = np.median(D[good])
    if (~good).any():
        for i in np.where(~good)[0]:
            peers = good if sector is None else good & (sector[assets] == sector[assets[i]])
            X[i] = X[peers].mean(axis=0) if peers.any() else X[good].mean(axis=0) #stock i's exposure = avg of its sector peers, or avg of all good stocks if no peers
        D[~good] = 2.0 * med #young stock's variance = 2*median of good stocks
    D = np.maximum(D, d_floor_frac * med) #floor = 0.1: specific variance can't go below 10% of median of good stocks
    return RiskModel(assets, X, factor_var, D, explained)
