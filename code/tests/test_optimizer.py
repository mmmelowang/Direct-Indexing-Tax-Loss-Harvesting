"""Tests for tlh.optimizer: compare the two-step method with exhaustive search over trade directions."""
import itertools

import numpy as np
import pytest

from tlh.optimizer import (TamInputs, TamParams, loss_names, solve_fixed, solve_relaxation,
                           solve_tam)


def random_problem(seed, n=6, k=2, n_loss=3):
    rng = np.random.default_rng(seed)
    B = rng.normal(0, 0.15, (n, k))
    D = rng.uniform(0.02, 0.06, n)
    hold = rng.dirichlet(np.ones(n)) * 0.995            # fully invested, cash = 0.5%
    hb = rng.dirichlet(np.ones(n)) * 0.995
    T = np.where(np.arange(n) < n_loss, -rng.uniform(0.05, 0.4, n), rng.uniform(0.0, 0.2, n))
    return TamInputs(x=hold - hb, sell_cap=hold, T=T, buyable=np.ones(n, bool), B=B, D=D,
                     cash=1.0 - hold.sum(), kappa=5e-4, eta=0.005)


def brute_force(P, prm):
    """Best objective over all buy-only / sell-only assignments of the loss names."""
    L = np.where(loss_names(P))[0]
    best = np.inf
    for signs in itertools.product([1.0, -1.0], repeat=len(L)):
        z = np.zeros(len(P.x))
        z[L] = signs
        out = solve_fixed(P, prm, z)
        if out is not None:
            best = min(best, out[2])
    return best


@pytest.mark.parametrize("seed", range(5))
def test_relaxation_is_lower_bound_and_two_step_is_near_optimal(seed):
    P = random_problem(seed)
    prm = TamParams(gamma_risk=50.0)
    exact = brute_force(P, prm)
    res = solve_tam(P, prm)
    assert res.status == "ok"
    assert res.lower_bound <= exact + 1e-9          # relaxation never exceeds the true optimum
    assert res.objective >= exact - 1e-9            # two-step solution is feasible, so not better
    assert res.gap >= -1e-9
    assert res.objective - exact <= res.gap + 1e-9  # certificate: distance to optimum <= gap


def test_cash_constraint_and_bounds():
    P = random_problem(0)
    res = solve_tam(P, TamParams())
    cost = P.kappa * (res.buy.sum() + res.sell.sum())
    assert res.buy.sum() - res.sell.sum() + cost == pytest.approx(P.cash - P.eta, abs=1e-7)
    assert np.all(res.sell <= P.sell_cap + 1e-9) and np.all(res.buy >= -1e-9)


def test_no_loss_names_gives_exact_solution():
    P = random_problem(1, n_loss=0)
    res = solve_tam(P, TamParams())
    assert res.gap == pytest.approx(0.0, abs=1e-9)


def test_forbidden_purchase_is_respected():
    P = random_problem(2)
    P.buyable[0] = False
    res = solve_tam(P, TamParams())
    assert res.buy[0] == pytest.approx(0.0, abs=1e-9)


def test_loss_name_is_sold_not_round_tripped():
    P = random_problem(3)
    res = solve_tam(P, TamParams(gamma_risk=50.0))
    assert np.all(np.minimum(res.buy, res.sell) < 1e-6)


def multi_lot_problem(seed, n=5, lots_per_name=3):
    """Like random_problem, but each held name is split into several lots with different T."""
    P = random_problem(seed, n=n)
    rng = np.random.default_rng(100 + seed)
    owner, cap, T = [], [], []
    for i in range(n):
        shares = rng.dirichlet(np.ones(lots_per_name)) * P.sell_cap[i]
        for k in range(lots_per_name):
            owner.append(i)
            cap.append(shares[k])
            T.append(P.T[i] + rng.normal(0, 0.1))
    P.lot_asset, P.lot_cap, P.lot_T = np.array(owner), np.array(cap), np.array(T)
    return P


@pytest.mark.parametrize("seed", range(4))
def test_multi_lot_two_step_against_exhaustive_search(seed):
    P = multi_lot_problem(seed)
    prm = TamParams(gamma_risk=50.0)
    exact = brute_force(P, prm)
    res = solve_tam(P, prm)
    tol = 1e-7                                      # solver accuracy
    assert res.status == "ok"
    assert res.lower_bound <= exact + tol
    assert res.objective >= exact - tol
    assert res.objective - exact <= res.gap + tol


def test_single_lot_detail_matches_default():
    P = random_problem(0)
    a = solve_tam(P, TamParams())
    owner = np.where(P.sell_cap > 0)[0]
    P.lot_asset, P.lot_cap, P.lot_T = owner, P.sell_cap[owner], P.T[owner]
    b = solve_tam(P, TamParams())
    assert a.objective == pytest.approx(b.objective, abs=1e-9)


def test_lowest_tax_lot_is_sold_first():
    # one name with a loss lot and a gain lot of equal size; it must be cut to half its weight
    n = 2
    P = TamInputs(x=np.array([0.25, -0.25]), sell_cap=np.array([0.5, 0.0]), T=np.zeros(n),
                  buyable=np.array([False, True]), B=np.zeros((n, 1)), D=np.array([0.04, 0.04]),
                  cash=0.0, kappa=0.0, eta=0.0,
                  lot_asset=np.array([0, 0]), lot_cap=np.array([0.25, 0.25]), lot_T=np.array([0.1, -0.2]))
    res = solve_tam(P, TamParams(gamma_risk=1000.0, bmax=1.0))
    assert res.sell[0] == pytest.approx(0.25, abs=1e-4)
    # objective = tax only (risk is zero after the trade): -0.2 * 0.25 if the loss lot was sold
    assert res.objective == pytest.approx(-0.2 * 0.25, abs=1e-4)


def ex_ante_te(P, u):
    w = P.x + u
    return float(np.sqrt(np.sum((P.B.T @ w) ** 2) + np.sum(P.D * w ** 2)))


def test_te_cap_is_respected_when_feasible():
    P = random_problem(0)
    free = solve_tam(P, TamParams(gamma_risk=1.0, bmax=1.0))
    cap = 0.5 * ex_ante_te(P, free.u)                 # tighter than the unconstrained solution
    res = solve_tam(P, TamParams(gamma_risk=1.0, bmax=1.0, te_max=cap))
    assert res.status == "ok"
    assert ex_ante_te(P, res.u) <= cap * (1 + 1e-4)
    assert res.te_slack == pytest.approx(0.0, abs=1e-8)


def test_te_cap_infeasible_uses_slack():
    P = random_problem(1)
    P.buyable[:] = False                              # nothing can be bought, so the active risk cannot be removed
    res = solve_tam(P, TamParams(gamma_risk=1.0, te_max=1e-4))
    assert res.status == "ok" and res.te_slack > 0


def test_te_cap_relaxation_is_lower_bound():
    P = multi_lot_problem(0)
    prm = TamParams(gamma_risk=1.0, te_max=0.05)
    exact = brute_force(P, prm)
    res = solve_tam(P, prm)
    assert res.lower_bound <= exact + 1e-7 and res.objective >= exact - 1e-7


def test_no_cap_matches_default():
    P = random_problem(2)
    assert solve_tam(P, TamParams()).objective == pytest.approx(solve_tam(P, TamParams(te_max=None)).objective)
