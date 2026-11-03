"""The Dutch book solver against cases worked out by hand.

Each case says which bets win, and the profit is checked to the cent fraction. The
property tests then check the solver against two facts that hold for any prices:
coherent prices admit no book, and for unconditional claims the guaranteed profit per
$1 equals the L-infinity distance to the coherent set.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from dutchbook.book import dutch_book, linf_distance, payoff_matrix, profit, project
from dutchbook.items import Claim, Item, indicator


def complement() -> Item:
    worlds = ("A", "not A")
    return Item(
        "c",
        "complement",
        "t",
        worlds,
        (Claim("a", indicator(worlds, ["A"])), Claim("na", indicator(worlds, ["not A"]))),
    )


def ladder() -> Item:
    # X > 100 and X > 120 over the intervals they cut.
    worlds = ("<=100", "100-120", ">120")
    return Item(
        "l",
        "monotonicity",
        "t",
        worlds,
        (
            Claim("gt100", indicator(worlds, ["100-120", ">120"])),
            Claim("gt120", indicator(worlds, [">120"])),
        ),
    )


def partition(n: int = 3) -> Item:
    worlds = tuple(f"o{i}" for i in range(n))
    return Item(
        "p",
        "partition",
        "t",
        worlds,
        tuple(Claim(f"o{i}", indicator(worlds, [f"o{i}"])) for i in range(n)),
    )


def conjunction() -> Item:
    worlds = ("AB", "A~B", "~AB", "~A~B")
    return Item(
        "j",
        "conjunction",
        "t",
        worlds,
        (
            Claim("a", indicator(worlds, ["AB", "A~B"])),
            Claim("b", indicator(worlds, ["AB", "~AB"])),
            Claim("ab", indicator(worlds, ["AB"])),
            Claim("aorb", indicator(worlds, ["AB", "A~B", "~AB"])),
        ),
    )


def conditional() -> Item:
    worlds = ("AB", "A~B", "~AB", "~A~B")
    b = indicator(worlds, ["AB", "~AB"])
    return Item(
        "k",
        "conditional",
        "t",
        worlds,
        (
            Claim("a|b", indicator(worlds, ["AB", "A~B"]), given=b),
            Claim("ab", indicator(worlds, ["AB"])),
            Claim("b", b),
        ),
    )


def paraphrase() -> Item:
    worlds = ("A", "not A")
    return Item(
        "x",
        "paraphrase",
        "t",
        worlds,
        (Claim("a", indicator(worlds, ["A"])), Claim("a2", indicator(worlds, ["A"]))),
    )


def test_complement_overround() -> None:
    # Sell half a contract on each side at 0.60: take in 0.60, pay out 0.50.
    book = dutch_book(complement(), [0.6, 0.6])
    assert book.profit == pytest.approx(0.1)
    assert book.stakes == pytest.approx((-0.5, -0.5))
    assert min(book.payoffs) == pytest.approx(0.1)


def test_complement_underround() -> None:
    # Buy half of each at 0.30 and 0.50: pay 0.40, collect 0.50.
    book = dutch_book(complement(), [0.3, 0.5])
    assert book.profit == pytest.approx(0.1)
    assert book.stakes == pytest.approx((0.5, 0.5))


def test_coherent_complement_has_no_book() -> None:
    book = dutch_book(complement(), [0.3, 0.7])
    assert book.profit == 0.0
    assert book.coherent
    assert book.stakes == (0.0, 0.0)


def test_ladder_inversion() -> None:
    # P(X > 120) = 0.7 above P(X > 100) = 0.5. Buy the wider event, sell the narrower:
    # every world pays at least 0.1, and the middle one pays 0.6.
    book = dutch_book(ladder(), [0.5, 0.7])
    assert book.profit == pytest.approx(0.1)
    assert book.stakes == pytest.approx((0.5, -0.5))
    assert book.payoffs == pytest.approx((0.1, 0.6, 0.1))


def test_partition_overround() -> None:
    book = dutch_book(partition(), [0.5, 0.4, 0.4])
    assert book.profit == pytest.approx(0.1)
    assert book.stakes == pytest.approx((-1 / 3, -1 / 3, -1 / 3))


def test_partition_profit_is_gap_over_n() -> None:
    for n in (2, 4, 7):
        prices = [0.9 / n] * n  # sums to 0.9
        assert dutch_book(partition(n), prices).profit == pytest.approx(0.1 / n)


def test_conjunction_above_conjunct() -> None:
    # P(A and B) = 0.6 > P(A) = 0.5: buy A, sell A-and-B, half each.
    book = dutch_book(conjunction(), [0.5, 0.5, 0.6, 0.6])
    assert book.profit == pytest.approx(0.05)


def test_conjunction_disjunction_identity() -> None:
    # P(A) + P(B) = P(A and B) + P(A or B) is the only constraint broken here:
    # 0.5 + 0.5 vs 0.2 + 0.6. All four bets at 1/4: profit 0.2 / 4.
    book = dutch_book(conjunction(), [0.5, 0.5, 0.2, 0.6])
    assert book.profit == pytest.approx(0.05)
    assert book.stakes == pytest.approx((-0.25, -0.25, 0.25, 0.25))


def test_conditional_hand_solved() -> None:
    # P(A|B) = 0.8, but P(A and B) / P(B) = 0.2 / 0.5 = 0.4.
    # Sell a units of the conditional, buy a of A-and-B, sell 0.8a of B. In the three
    # distinct worlds (AB, not-A B, not-B) the payoffs are 0.6a - 0.4a, 0.8a - 0.2a -
    # 0.4a and -0.2a + 0.4a, all 0.2a. With a + a + 0.8a = 1 the profit is 1/14. The
    # world weights (0.271, 0.157, 0.571) make every bet's expectation 1/14 in size,
    # which caps the LP at the same value, so 1/14 is optimal.
    book = dutch_book(conditional(), [0.8, 0.2, 0.5])
    assert book.profit == pytest.approx(1 / 14)
    assert book.stakes == pytest.approx((-1 / 2.8, 1 / 2.8, -0.8 / 2.8))


def test_conditional_bet_is_called_off_when_condition_fails() -> None:
    g = payoff_matrix(conditional(), [0.8, 0.2, 0.5])
    # Worlds A~B and ~A~B: the conditional contract returns the stake.
    assert g[0].tolist() == pytest.approx([0.2, 0.0, -0.8, 0.0])


def test_coherent_conditional_has_no_book() -> None:
    assert dutch_book(conditional(), [0.4, 0.2, 0.5]).profit == 0.0
    proj = project(conditional(), [0.4, 0.2, 0.5])
    assert proj.distance == pytest.approx(0.0, abs=1e-6)


def test_paraphrase_gap() -> None:
    book = dutch_book(paraphrase(), [0.7, 0.5])
    assert book.profit == pytest.approx(0.1)


def test_projection_complement() -> None:
    proj = project(complement(), [0.6, 0.6])
    assert proj.prices == pytest.approx((0.5, 0.5))
    assert proj.distance == pytest.approx(math.sqrt(0.02))


def test_projection_ladder_pools_the_inversion() -> None:
    proj = project(ladder(), [0.5, 0.7])
    assert proj.prices == pytest.approx((0.6, 0.6))
    assert proj.distance == pytest.approx(math.sqrt(0.02))


def test_projection_conditional() -> None:
    proj = project(conditional(), [0.8, 0.2, 0.5])
    q = proj.prices
    # The answer must be coherent, and closer than the trivial fixes.
    assert q[0] * q[2] == pytest.approx(q[1], abs=1e-6)
    assert dutch_book(conditional(), list(q)).profit < 1e-6
    trivial = math.dist([0.8, 0.2, 0.5], [0.4, 0.2, 0.5])
    assert proj.distance < trivial


def test_rejects_bad_prices() -> None:
    with pytest.raises(ValueError, match="outside"):
        dutch_book(complement(), [1.2, 0.0])
    with pytest.raises(ValueError, match="prices for"):
        dutch_book(complement(), [0.5])
    with pytest.raises(ValueError, match="unconditional"):
        linf_distance(conditional(), [0.4, 0.2, 0.5])


ITEMS = [complement, ladder, partition, conjunction, paraphrase]


@pytest.mark.parametrize("make", ITEMS)
def test_coherent_prices_admit_no_book(make: Callable[[], Item]) -> None:
    item = make()
    rng = np.random.default_rng(1)
    e = np.array([c.event for c in item.claims], dtype=float)
    for _ in range(50):
        lam = rng.dirichlet(np.ones(len(item.worlds)))
        prices = (e @ lam).tolist()
        assert dutch_book(item, prices).profit == 0.0
        assert project(item, prices).distance == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("make", ITEMS)
def test_profit_equals_linf_distance(make: Callable[[], Item]) -> None:
    item = make()
    rng = np.random.default_rng(2)
    for _ in range(50):
        prices = rng.uniform(0, 1, len(item.claims)).tolist()
        book = dutch_book(item, prices)
        assert book.profit == pytest.approx(linf_distance(item, prices), abs=1e-8)
        assert min(book.payoffs) >= book.profit - 1e-8
        assert sum(abs(s) for s in book.stakes) <= 1 + 1e-8


@pytest.mark.parametrize("make", ITEMS)
def test_projection_is_coherent_and_no_further_than_any_world_mix(make: Callable[[], Item]) -> None:
    item = make()
    rng = np.random.default_rng(3)
    e = np.array([c.event for c in item.claims], dtype=float)
    for _ in range(20):
        prices = rng.uniform(0, 1, len(item.claims))
        proj = project(item, prices.tolist())
        assert dutch_book(item, list(proj.prices)).profit < 1e-6
        for _ in range(20):
            other = e @ rng.dirichlet(np.ones(len(item.worlds)))
            assert proj.distance <= np.linalg.norm(other - prices) + 1e-7


def test_projection_lowers_brier_in_every_world() -> None:
    # Predd et al. (2009): the projection beats the original prices in every world by
    # at least the squared distance.
    item = partition(4)
    e = np.array([c.event for c in item.claims], dtype=float)
    prices = np.array([0.5, 0.4, 0.3, 0.2])
    proj = project(item, prices.tolist())
    q = np.array(proj.prices)
    for w in range(len(item.worlds)):
        truth = e[:, w]
        gain = np.sum((prices - truth) ** 2) - np.sum((q - truth) ** 2)
        assert gain >= proj.distance**2 - 1e-9


def test_conditional_profit_bounded_by_random_search() -> None:
    # No random stake vector should beat the LP.
    item = conditional()
    prices = [0.8, 0.2, 0.5]
    g = payoff_matrix(item, prices)
    best = dutch_book(item, prices).profit
    rng = np.random.default_rng(4)
    for _ in range(2000):
        s = rng.normal(size=3)
        s /= np.abs(s).sum()
        assert float(np.min(s @ g)) <= best + 1e-9


@pytest.mark.parametrize(
    "make", [complement, ladder, partition, conjunction, conditional, paraphrase]
)
def test_fast_profit_matches_lp(make: Callable[[], Item]) -> None:
    item = make()
    rng = np.random.default_rng(5)
    for _ in range(200):
        prices = rng.uniform(0, 1, len(item.claims)).round(2).tolist()
        assert profit(item, prices) == pytest.approx(dutch_book(item, prices).profit, abs=1e-8)


def test_fast_profit_on_a_long_ladder() -> None:
    worlds = tuple(f"w{i}" for i in range(6))
    item = Item(
        "long",
        "monotonicity",
        "t",
        worlds,
        # Listed out of order on purpose: the chain is found from the indicators.
        tuple(Claim(f"c{i}", indicator(worlds, worlds[i:])) for i in (3, 1, 5, 2, 4)),
    )
    rng = np.random.default_rng(6)
    for _ in range(300):
        prices = rng.uniform(0, 1, 5).tolist()
        assert profit(item, prices) == pytest.approx(dutch_book(item, prices).profit, abs=1e-8)


def test_bid_ask_book() -> None:
    # Asks sum to 0.95 on a three-way partition: buy a third of each at the ask.
    item = partition()
    book = dutch_book(item, [0.30, 0.30, 0.35], sell_prices=[0.28, 0.28, 0.33])
    assert book.profit == pytest.approx(0.05 / 3)
    # A wide market is coherent even though its mids are not.
    wide = dutch_book(item, [0.40, 0.40, 0.40], sell_prices=[0.30, 0.30, 0.30])
    assert wide.profit == 0.0
    assert dutch_book(item, [0.35, 0.35, 0.35]).profit > 0
