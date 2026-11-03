"""The Dutch book: bets at the forecaster's own prices that lose money in every world.

Setup. An item has worlds w = 1..W and claims i = 1..n. The forecaster said claim i has
probability p_i. A contract on claim i pays $1 if the claim is true and costs p_i. A
bookie may buy (s_i > 0) or sell (s_i < 0) any amount of each contract at that price.
The bookie's payoff in world w is

    sum_i s_i * g_i(w),   g_i(w) = e_i(w) - p_i

for an unconditional claim with indicator e_i, and

    g_i(w) = b_i(w) * (e_i(w) - p_i)

for a conditional claim P(A | B), whose bet is called off (money back) when B is false.
That is de Finetti's conditional bet.

The Dutch book is the stake vector that maximises the worst-case payoff,

    maximise t   subject to   sum_i s_i g_i(w) >= t  for every world w,
                              sum_i |s_i| <= 1.

The stake bound is $1 of total contract face value, so t is the guaranteed profit per
$1 staked. t = 0 (bet nothing) is always feasible, and t > 0 exactly when the prices
are incoherent: de Finetti's theorem, and for conditionals the version in Walley,
Pelessoni and Vicig (2004).

For unconditional claims, linear programming duality gives a second reading of t. It
equals the L-infinity distance from p to the nearest coherent assignment: the largest
amount any single price would have to move to make the set coherent. A test checks
this identity against a separate LP. The Euclidean distance to the nearest coherent
assignment, also reported, has its own guarantee (Predd et al. 2009): moving to that
point lowers the Brier score in every world by at least the squared distance.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog
from scipy.optimize import minimize as _minimize

from dutchbook.items import Item

# scipy-stubs types `constraints` and `options` more narrowly than scipy accepts.
minimize: Any = _minimize

Array = NDArray[np.float64]

TOL = 1e-9


@dataclass(frozen=True)
class Book:
    """The best Dutch book against one set of prices."""

    profit: float  # guaranteed profit per $1 of total stake, >= 0
    stakes: tuple[float, ...]  # > 0 buy, < 0 sell, in claim order; sum of |s| <= 1
    payoffs: tuple[float, ...]  # the bookie's payoff in each world

    @property
    def coherent(self) -> bool:
        return self.profit <= 1e-7


def payoff_matrix(item: Item, prices: Sequence[float]) -> Array:
    """g[i, w]: the bookie's payoff in world w from buying one contract on claim i."""
    if len(prices) != len(item.claims):
        raise ValueError(f"{item.id}: {len(prices)} prices for {len(item.claims)} claims")
    g = np.zeros((len(item.claims), len(item.worlds)))
    for i, (claim, p) in enumerate(zip(item.claims, prices, strict=True)):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{item.id}: price {p} for {claim.prop} is outside [0, 1]")
        e = np.asarray(claim.event, dtype=float)
        b = np.ones_like(e) if claim.given is None else np.asarray(claim.given, dtype=float)
        g[i] = b * (e - p)
    return g


def dutch_book(
    item: Item, prices: Sequence[float], sell_prices: Sequence[float] | None = None
) -> Book:
    """The best book against `prices`.

    With `sell_prices`, contracts are bought at `prices` (an ask) and sold at
    `sell_prices` (a bid), which is how a market's executable book is priced. Without
    it, both sides trade at one price, which is the forecaster's case.
    """
    g_buy = payoff_matrix(item, prices)
    # Selling one contract at the bid pays bid - e, the negative of buying at the bid.
    g_sell = -payoff_matrix(item, sell_prices) if sell_prices is not None else -g_buy
    n, w = g_buy.shape
    # Variables: s_plus (n), s_minus (n), t. linprog minimises, so the objective is -t.
    c = np.zeros(2 * n + 1)
    c[-1] = -1.0
    # t - sum_i (s+_i g_buy_i(w) + s-_i g_sell_i(w)) <= 0 for each world.
    a_worlds = np.hstack([-g_buy.T, -g_sell.T, np.ones((w, 1))])
    a_stake = np.concatenate([np.ones(2 * n), [0.0]])[None, :]
    a_ub = np.vstack([a_worlds, a_stake])
    b_ub = np.concatenate([np.zeros(w), [1.0]])
    bounds = [(0.0, None)] * (2 * n) + [(None, None)]
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success or res.x is None or res.fun is None:  # pragma: no cover
        raise RuntimeError(f"{item.id}: Dutch book LP failed: {res.message}")
    plus, minus = res.x[:n], res.x[n : 2 * n]
    profit = max(0.0, -float(res.fun))
    if profit < TOL:
        profit, plus, minus = 0.0, np.zeros(n), np.zeros(n)
    stakes = plus - minus
    payoffs = plus @ g_buy + minus @ g_sell
    return Book(profit, tuple(float(s) for s in stakes), tuple(float(v) for v in payoffs))


def profit(item: Item, prices: Sequence[float]) -> float:
    """The guaranteed profit alone, with closed forms where the structure allows.

    The noise analysis solves thousands of books per item, so the simple shapes skip
    the LP. All three formulas are the L-infinity distance to the coherent set, and
    the tests check each against the LP:

    - claims that partition the worlds (complements, range bins, who-wins lists): the
      smallest d with sum max(0, p_i - d) = 1 when the prices sum to more than 1, and
      the mirror image when they sum to less. That is |sum p - 1| / n unless a price
      would have to cross 0 or 1;
    - two claims with the same indicator (paraphrases): |p1 - p2| / 2;
    - a strictly nested chain (ladders): the largest inversion, max over i < j of
      (p_j - p_i) / 2, where claim j is the narrower event.
    """
    shape = _shape(item)
    p = list(prices)
    if shape == "partition":
        total = sum(p)
        if total > 1.0:
            return _shrink(p, 1.0)
        return _shrink([1.0 - x for x in p], len(p) - 1.0)
    if shape == "same":
        return abs(p[0] - p[1]) / 2.0
    if shape is not None and shape.startswith("chain:"):
        order = [int(i) for i in shape[6:].split(",")]
        q = [p[i] for i in order]  # widest event first, so q should not increase
        worst, top = 0.0, q[0]
        for x in q[1:]:
            worst = max(worst, x - top)
            top = min(top, x)
        return max(worst, 0.0) / 2.0
    return dutch_book(item, prices).profit


def _shrink(values: list[float], target: float) -> float:
    """The smallest d >= 0 with sum max(0, v - d) <= target."""
    v = sorted(values, reverse=True)
    if sum(v) <= target:
        return 0.0
    run = 0.0
    for m, x in enumerate(v, 1):
        run += x
        d = (run - target) / m
        nxt = v[m] if m < len(v) else 0.0
        if d >= nxt - 1e-15:
            return max(d, 0.0)
    return max((run - target) / len(v), 0.0)  # pragma: no cover


_SHAPES: dict[str, str | None] = {}


def _shape(item: Item) -> str | None:
    if item.id in _SHAPES:
        return _SHAPES[item.id]
    shape: str | None = None
    if not any(c.conditional for c in item.claims):
        e = np.array([c.event for c in item.claims], dtype=int)
        sizes = e.sum(axis=1)
        if np.all(e.sum(axis=0) == 1):
            shape = "partition"
        elif len(e) == 2 and np.array_equal(e[0], e[1]) and 0 < sizes[0] < e.shape[1]:
            shape = "same"
        else:
            order = [int(i) for i in np.argsort(-sizes, kind="stable")]
            nested = all(
                sizes[a] > sizes[b] and np.all(e[b] <= e[a]) for a, b in itertools.pairwise(order)
            )
            if nested and sizes[order[0]] < e.shape[1] and sizes[order[-1]] > 0:
                shape = "chain:" + ",".join(str(i) for i in order)
    _SHAPES[item.id] = shape
    return shape


def linf_distance(item: Item, prices: Sequence[float]) -> float:
    """Smallest d such that some coherent assignment is within d of every price.

    Only defined here for items without conditional claims. Used to check the duality
    with `dutch_book` in the tests and as a cross-check in the analysis.
    """
    if any(c.conditional for c in item.claims):
        raise ValueError("linf_distance only handles unconditional claims")
    e = np.array([c.event for c in item.claims], dtype=float)  # n x W
    p = np.asarray(prices, dtype=float)
    n, w = e.shape
    # Variables: lambda (W), d. Minimise d s.t. |E lambda - p| <= d, lambda in simplex.
    c = np.concatenate([np.zeros(w), [1.0]])
    a_ub = np.vstack([np.hstack([e, -np.ones((n, 1))]), np.hstack([-e, -np.ones((n, 1))])])
    b_ub = np.concatenate([p, -p])
    a_eq = np.concatenate([np.ones(w), [0.0]])[None, :]
    res = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=[1.0],
        bounds=[(0.0, None)] * (w + 1),
        method="highs",
    )
    if not res.success or res.fun is None:  # pragma: no cover
        raise RuntimeError(res.message)
    return float(res.fun)


@dataclass(frozen=True)
class Projection:
    """The nearest coherent assignment in Euclidean distance."""

    prices: tuple[float, ...]
    distance: float
    world_probs: tuple[float, ...]


def project(item: Item, prices: Sequence[float]) -> Projection:
    """Euclidean projection of the prices onto the item's coherent set.

    The coherent set is every assignment some probability distribution over the worlds
    produces, with conditional claims given by the ratio P(A and B) / P(B). Without
    conditionals the problem is a convex QP and the solver's answer is the global
    optimum. With them it is not convex, so the solver starts from several points and
    keeps the best.
    """
    p = np.asarray(prices, dtype=float)
    payoff_matrix(item, prices)  # validates
    w = len(item.worlds)
    e = np.array([c.event for c in item.claims], dtype=float)
    cond = [i for i, c in enumerate(item.claims) if c.conditional]
    uncond = [i for i, c in enumerate(item.claims) if not c.conditional]
    if not cond:
        lam = _simplex_qp(e, p)
        q = e @ lam
        return Projection(tuple(float(x) for x in q), float(np.linalg.norm(q - p)), _tup(lam))

    gmat = np.array([item.claims[i].given for i in cond], dtype=float)  # k x W
    joint = e[cond] * gmat  # A and B
    k = len(cond)

    def split(x: Array) -> tuple[Array, Array]:
        return x[:w], x[w:]

    def assignment(x: Array) -> Array:
        lam, q = split(x)
        out = np.empty(len(item.claims))
        out[uncond] = e[uncond] @ lam
        out[cond] = q
        return out

    def objective(x: Array) -> float:
        d = assignment(x) - p
        return float(d @ d)

    def gradient(x: Array) -> Array:
        _lam, _ = split(x)
        d = assignment(x) - p
        grad_lam = 2.0 * e[uncond].T @ d[uncond] if uncond else np.zeros(w)
        grad_q = 2.0 * d[cond]
        return np.concatenate([grad_lam, grad_q])

    constraints = [
        {"type": "eq", "fun": lambda x: np.array([np.sum(split(x)[0]) - 1.0])},
        {
            "type": "eq",
            "fun": lambda x: joint @ split(x)[0] - split(x)[1] * (gmat @ split(x)[0]),
        },
    ]
    bounds = [(0.0, 1.0)] * (w + k)
    starts = [np.full(w, 1.0 / w)]
    if uncond:
        starts.append(np.clip(_simplex_qp(e[uncond], p[uncond]), 1e-6, 1.0))
    rng = np.random.default_rng(0)
    starts += [rng.dirichlet(np.ones(w)) for _ in range(4)]
    best: tuple[float, Array] | None = None
    for start in starts:
        lam0 = start / start.sum()
        q0 = (joint @ lam0) / np.maximum(gmat @ lam0, 1e-12)
        res = minimize(
            objective,
            np.concatenate([lam0, q0]),
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-14},
        )
        x = np.asarray(res.x, dtype=float)
        lam, q = split(x)
        if abs(lam.sum() - 1.0) > 1e-6 or np.max(np.abs(joint @ lam - q * (gmat @ lam))) > 1e-6:
            continue
        val = objective(x)
        if best is None or val < best[0]:
            best = (val, x)
    if best is None:  # pragma: no cover - the uniform start always converges in practice
        raise RuntimeError(f"{item.id}: projection did not converge")
    x = best[1]
    q = assignment(x)
    return Projection(tuple(float(v) for v in q), float(np.linalg.norm(q - p)), _tup(split(x)[0]))


def _simplex_qp(e: Array, p: Array) -> Array:
    """argmin over the simplex of ||E lambda - p||^2, by an active-set method.

    Small and exact enough for a handful of worlds: solve the equality-constrained
    least-squares problem on the current support, drop the most negative weight, and
    repeat. Falls back to SLSQP if the active set cycles.
    """
    w = e.shape[1]
    support = list(range(w))
    for _ in range(4 * w):
        sub = e[:, support]
        m = len(support)
        # KKT system for min ||sub l - p||^2 s.t. sum l = 1.
        kkt = np.zeros((m + 1, m + 1))
        kkt[:m, :m] = 2.0 * sub.T @ sub
        kkt[:m, m] = 1.0
        kkt[m, :m] = 1.0
        rhs = np.concatenate([2.0 * sub.T @ p, [1.0]])
        sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
        lam_s = sol[:m]
        if np.all(lam_s >= -1e-12):
            lam = np.zeros(w)
            lam[support] = np.clip(lam_s, 0.0, None)
            lam /= lam.sum()
            if _kkt_ok(e, p, lam):
                return lam
            break
        support.pop(int(np.argmin(lam_s)))
    return _simplex_slsqp(e, p)


def _kkt_ok(e: Array, p: Array, lam: Array) -> bool:
    grad = 2.0 * e.T @ (e @ lam - p)
    on = lam > 1e-10
    mu = float(np.mean(grad[on]))
    return bool(np.all(np.abs(grad[on] - mu) < 1e-7) and np.all(grad[~on] >= mu - 1e-7))


def _simplex_slsqp(e: Array, p: Array) -> Array:
    w = e.shape[1]
    res = minimize(
        lambda lam: float(np.sum((e @ lam - p) ** 2)),
        np.full(w, 1.0 / w),
        jac=lambda lam: 2.0 * e.T @ (e @ lam - p),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * w,
        constraints=[{"type": "eq", "fun": lambda lam: np.array([lam.sum() - 1.0])}],
        options={"maxiter": 1000, "ftol": 1e-15},
    )
    lam: Array = np.clip(np.asarray(res.x, dtype=float), 0.0, None)
    out: Array = lam / lam.sum()
    return out


def _tup(x: Array) -> tuple[float, ...]:
    return tuple(float(v) for v in x)
