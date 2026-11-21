"""Tax-aware Markowitz (TAM) optimizer 
    minimize   gamma_risk * (x+u)' V (x+u)         risk relative to the benchmark
             + gamma_tc  * kappa * sum(b + s)      trading cost
             + gamma_tax * sum(T_i s_i)            tax on the sales

    subject to u = b - s,  b, s >= 0,  s <= sell_cap,  b <= bmax(10% of NAV)
               b_i = 0 for names that cannot be bought (wash sale, not in index),
               sum(b) - sum(s) + kappa * sum(b + s) = cash - eta     (cash ends at the buffer eta)

x = h - h_b are the current active weights, V = B B' + diag(D) the risk model,
T_i = rho_i (1 - basis_i / price_i) the tax per dollar sold. 
A name whose lot is below cost has T_i < 0: selling it earns a tax credit. Then it makes the model want to buy
and sell on the same stock, which makes the problem non-convex ("loss names"). To removes this:

  step 1  convex relaxation (second-order cone program). For each loss name the
          trade is written as a convex combination of a buy-only point (weight
          theta) and a sell-only point (weight 1 - theta); the specific-risk
          term D_i (x_i + u_i)^2 becomes its perspective form
              D_i [ x_i^2 + 2 x_i u_i + b_i^2 / theta_i + s_i^2 / (1 - theta_i) ].

  step 2  fix each loss name's trade direction (buy-only or sell-only) from the
          relaxed solution and solve the remaining convex QP.

gap = (step 2 objective, real cost) - (step 1 objective, lower bound) >= 0 certifies how far the trade
list can be from the true optimum.

Several lots per name: each lot l has its own sale variable s_l <= cap_l and tax
T_l, and the name's sale is the sum over its lots. Because the lots differ only
in T_l, the optimizer sells the lowest-T lots first (LTFO). A name is a loss name
if any of its sellable lots has T_l < 0.

Tracking-error cap (optional, TamParams.te_max): the ex-ante variance of the
active weights after trading must stay below te_max**2. A penalized slack lets
the cap be exceeded only when no trade can meet it (e.g. wash-sale rules block
the needed purchases). Both steps cap the exact TE; in step 1 this is looser
than capping the relaxed risk, so step 1 remains a lower bound.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cvxpy as cp
import scipy.sparse as sp


@dataclass
class TamInputs:
    x: np.ndarray          # (n,) active weights h - h_b
    sell_cap: np.ndarray   # (n,) maximum weight that can be sold (holding, 0 if sale is not allowed)
    T: np.ndarray          # (n,) tax per dollar sold (negative for loss lots)
    buyable: np.ndarray    # (n,) bool, False if purchases are forbidden
    B: np.ndarray          # (n, k) factor loadings, X Sigma X' = B B'
    D: np.ndarray          # (n,) specific variances
    cash: float            # current cash weight
    kappa: float           # one-way proportional trading cost
    eta: float             # target cash weight after trading
    # optional lot detail; if omitted, each name is a single lot given by sell_cap and T
    lot_asset: np.ndarray | None = None   # (m,) name index of each lot
    lot_cap: np.ndarray | None = None     # (m,) weight of each lot that can be sold
    lot_T: np.ndarray | None = None       # (m,) tax per dollar sold of each lot


@dataclass
class TamParams:
    gamma_risk: float = 400.0
    gamma_tc: float = 1.0
    gamma_tax: float = 1.0
    bmax: float = 0.1          # largest single purchase, as a fraction of NAV
    solver: str = "CLARABEL"
    te_max: float | None = None   # cap on ex-ante tracking error (annualized); None = no cap
    te_penalty: float = 10.0      # cost per unit of relative excess TE / te_max - 1 (slack); large, so the cap is exceeded only if unavoidable


@dataclass
class TamResult:
    u: np.ndarray              # (n,) net trade in weights (buys positive)
    buy: np.ndarray
    sell: np.ndarray
    objective: float           # step 2 objective
    lower_bound: float         # step 1 objective (equals objective if there are no loss names)
    sign: np.ndarray           # (n,) +1 buy-only, -1 sell-only, 0 free
    status: str
    te_slack: float = 0.0      # relative excess TE / te_max - 1 of the executed trade list (0 if the cap holds)

    @property
    def gap(self) -> float:
        return self.objective - self.lower_bound


def lots(P: TamInputs):
    """(owner, cap, T) of the sellable lots; without lot detail, one lot per held name."""
    if P.lot_asset is None:
        owner = np.where(P.sell_cap > 1e-12)[0]
        return owner, P.sell_cap[owner], P.T[owner]
    keep = P.lot_cap > 1e-12
    return P.lot_asset[keep], P.lot_cap[keep], P.lot_T[keep]


def loss_names(P: TamInputs) -> np.ndarray:
    """find the trade which buying and selling in one trade is possible, treat it as a loss."""
    owner, cap, T = lots(P)
    has_loss_lot = np.zeros(len(P.x), bool)
    has_loss_lot[owner[T < 0]] = True
    return has_loss_lot & P.buyable


def _common(P: TamInputs, prm: TamParams):
    '''
    constaints, objective, and variables common to both the relaxation and the fixed-sign problems
    returns b (per name), s (per lot, None if no lots), sell (per name), u, cons, objective, owner
    '''
    n = len(P.x)
    owner, cap, T = lots(P)
    m = len(owner)
    b = cp.Variable(n, nonneg=True)
    cons = [b <= prm.bmax]
    if m:
        s = cp.Variable(m, nonneg=True)
        M = sp.csr_matrix((np.ones(m), (owner, np.arange(m))), shape=(n, m))   # lot -> name
        sell = M @ s
        cons.append(s <= cap)
        tax = T @ s
    else:
        s, sell, tax = None, np.zeros(n), 0.0
    u = b - sell
    if (~P.buyable).any():
        cons.append(b[~P.buyable] == 0)
    tb, ts = cp.sum(b), cp.sum(sell)
    cons.append(tb - ts + P.kappa * (tb + ts) == P.cash - P.eta)
    linear = prm.gamma_tc * P.kappa * (tb + ts) + prm.gamma_tax * tax
    sys_var = cp.sum_squares(P.B.T @ (P.x + u)) #systematic (factor) variance of the active weights
    return b, s, sell, u, cons, linear + prm.gamma_risk * sys_var, owner, sys_var


def _te_cap(P: TamInputs, prm: TamParams, u, cons, obj):
    """ex-ante TE of the active weights after trading <= te_max (1 + slack), slack >= 0 and penalized.
    TE = || [B'(x+u); sqrt(D) (x+u)] ||, a second-order cone; dividing by te_max keeps it of order 1.
    Returns (objective, slack)."""
    if prm.te_max is None:
        return obj, None
    w = P.x + u
    slack = cp.Variable(nonneg=True)
    te = cp.norm(cp.hstack([P.B.T @ w, cp.multiply(np.sqrt(P.D), w)]))
    cons.append(te / prm.te_max <= 1 + slack)
    return obj + prm.te_penalty * slack, slack


def _solve(problem, prm: TamParams) -> bool:
    """Solve with the chosen solver, falling back to SCS; True if a solution was found."""
    for solver in (prm.solver, "SCS"):
        try:
            problem.solve(solver=solver)
        except cp.error.SolverError:
            continue
        if problem.status in ("optimal", "optimal_inaccurate"):
            return True
    return False


def _value(expr, n):
    return np.zeros(n) if isinstance(expr, np.ndarray) else np.asarray(expr.value).ravel()


def solve_fixed(P: TamInputs, prm: TamParams, z: np.ndarray):
    """solve the non-convex issue: 
    Step 2. z[i] = +1 buy-only, -1 sell-only, 0 free. Returns (b, s, objective, slack) or None.
    objective is the real cost of the trade"""
    b, s, sell, u, cons, base, owner, sys_var = _common(P, prm) #base includes the linear and systematic part of the objective
    if (z < 0).any():
        cons.append(b[z < 0] == 0)
    if s is not None:
        buy_only_lots = np.isin(owner, np.where(z > 0)[0])
        if buy_only_lots.any():
            cons.append(s[buy_only_lots] == 0)
    spec_var = cp.sum(cp.multiply(P.D, cp.square(P.x + u))) #idiosyncratic variance (gain or loss)
    obj = base + prm.gamma_risk * spec_var
    obj, slack = _te_cap(P, prm, u, cons, obj) #tracking-error cap, if any
    problem = cp.Problem(cp.Minimize(obj), cons) #optimizer
    if not _solve(problem, prm):
        return None
    #output: b(optimal buy amount, faction of NAV), s(optimal sell amount per name), objective value, slack above the cap
    slack_value = 0.0 if slack is None else float(slack.value)
    return np.asarray(b.value).ravel(), _value(sell, len(P.x)), float(problem.value), slack_value


def solve_relaxation(P: TamInputs, prm: TamParams):
    """Step 1. Returns (b, s, theta, objective) with theta indexed over the loss names, or None."""
    b, s, sell, u, cons, base, owner, sys_var = _common(P, prm)
    L = np.where(loss_names(P))[0] #loss name index
    F = np.setdiff1d(np.arange(len(P.x)), L) #not loss name index
    spec_var = cp.sum(cp.multiply(P.D[F], cp.square(P.x[F] + u[F]))) #idiosyn variance for non-loss names
    theta = np.zeros(0) #initialize theta for loss names
    if len(L):
        th = cp.Variable(len(L)) #initialize a variable theta with len(L), weight for buy-only stock
        pv = cp.Variable(len(L))
        qv = cp.Variable(len(L))
        # map each lot of a loss name to its theta
        pos = -np.ones(len(P.x), int)
        pos[L] = np.arange(len(L))
        lot_pos = pos[owner]
        in_L = lot_pos >= 0
        Q = sp.csr_matrix((np.ones(in_L.sum()), (np.arange(in_L.sum()), lot_pos[in_L])), shape=(in_L.sum(), len(L)))
        _, cap, _ = lots(P)
        cons += [th >= 0, th <= 1, #constraint
                 b[L] <= prm.bmax * th, #buy amount b[L] ≤ bmax·θ
                 s[in_L] <= cp.multiply(cap[in_L], 1 - Q @ th), #each lot's sale <= lot cap·(1-θ)
                 #constraint: b^2 / θ ≤ pv, s^2 / (1 - θ) ≤ qv, which is a second-order cone constraint
                 #and when you do the optimizer(min), it will make pv = b^2 / θ, qv = s^2 / (1 - θ)
                 cp.SOC(pv + th, cp.vstack([2 * b[L], pv - th]), axis=0),
                 cp.SOC(qv + 1 - th, cp.vstack([2 * sell[L], qv - (1 - th)]), axis=0)]
        #idiosyn risk for loss names
        spec_var += cp.sum(cp.multiply(
            P.D[L], P.x[L] ** 2 + 2 * cp.multiply(P.x[L], u[L]) + pv + qv))
    obj, slack = _te_cap(P, prm, u, cons, base + prm.gamma_risk * spec_var) #cap on the exact TE
    if slack is not None and len(L):
        # also cap the relaxed variance: equal to the true variance for pure buys and sells, larger for mixed
        # trades, so mixed trades are limited by the cap itself. 1 + 3 slack >= (1 + slack)^2 for slack <= 1,
        # so every true trade list with TE up to twice the cap stays feasible and step 1 remains a lower bound.
        cons.append((sys_var + spec_var) / prm.te_max ** 2 <= 1 + 3 * slack)
     #optimize
    problem = cp.Problem(cp.Minimize(obj), cons)
    if not _solve(problem, prm):
        return None #if it returns infeasible or unbounded --> none
    if len(L):
        theta = np.clip(np.asarray(th.value).ravel(), 0.0, 1.0) #clip theta to [0, 1]
    return np.asarray(b.value).ravel(), _value(sell, len(P.x)), theta, float(problem.value)


def guess_signs(P: TamInputs, b, s, theta, tolerance: float = 1e-9) -> np.ndarray:
    """Trade direction of each loss name: z[i] = +1 buy-only, -1 sell-only, 0 free. if u~0, check theta, if theta>=0.5, buy"""
    z = np.zeros(len(P.x))
    L = np.where(loss_names(P))[0] #loss name index
    u = (b - s)[L] #get the net trade for loss names
    z[L] = np.where(u > tolerance, 1.0, np.where(u < -tolerance, -1.0, np.where(theta >= 0.5, 1.0, -1.0)))
    return z


def solve_tam(P: TamInputs, prm: TamParams | None = None) -> TamResult:
    """Two-step TAM solve: convex relaxation, guess signs, convex QP."""
    prm = prm or TamParams() #params, if not provided, use default
    relax = solve_relaxation(P, prm)
    if relax is None:
        return TamResult(np.zeros(len(P.x)), np.zeros(len(P.x)), np.zeros(len(P.x)),
                         np.nan, np.nan, np.zeros(len(P.x)), "relaxation_failed")
    b_r, s_r, theta, lower = relax #lower is the objective(cost) of the relaxation problem
    z = guess_signs(P, b_r, s_r, theta)
    fixed = solve_fixed(P, prm, z) #z limit the buy/sell side, and function is convex, so resolve the problem
    if fixed is None:
        return TamResult(np.zeros(len(P.x)), np.zeros(len(P.x)), np.zeros(len(P.x)),
                         np.nan, lower, z, "step2_failed")
    b, s, obj, slack = fixed
    return TamResult(b - s, b, s, obj, lower, z, "ok", slack)
