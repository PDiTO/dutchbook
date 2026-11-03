"""The seeded generator for propositions and items.

Everything is templated, so every logical relation holds by construction:

- **Numeric ladders.** One quantity (the BTC price on a date, the S&P 500 close, the
  unemployment rate) cut at five thresholds. "Above t" for each threshold is a
  monotone ladder. The six intervals between the cuts form a partition. "At or below
  t" is the complement of "above t", and a reworded "above t" is a paraphrase.
- **Date ladders.** "X happens between now and D" for three dates D, which can only
  grow with D, plus the negated "the period ends without X".
- **Who-wins lists.** Five named candidates and "someone other than" those five form a
  partition; "someone other than X" is the complement of "X wins".
- **Pairs.** Two propositions from unrelated topics give A and B, A or B, and
  P(A | B), asked with the bet void if B fails.

The seed picks thresholds from a grid around a reference value, the dates, the named
candidates and the pairs, so a new seed gives new questions with the same structure.
Proposition ids encode their meaning (`btc.gt.80000`, `btc.bin.80000_to_90000`,
`starship.by.2028-12-31`), which the tests use to check every indicator against
simulated outcomes.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable
from dataclasses import dataclass

from dutchbook.elicit import AS_OF
from dutchbook.items import Bank, Claim, Item, Proposition, indicator

DEFAULT_SEED = 20261103
START = AS_OF  # events count from the day the questions were asked


@dataclass(frozen=True)
class Quantity:
    key: str
    subject: str  # noun phrase for the quantity
    when: str  # the measurement date and source, appended to every question
    alt: str  # a reworded question template for the paraphrase, with {x}
    reference: float  # roughly the value when the bank was built
    steps: tuple[float, ...]  # threshold grid: multipliers, or offsets if additive
    additive: bool
    rounding: float
    fmt: Callable[[float], str]


def _usd(x: float) -> str:
    return f"${x:,.0f}"


def _plain(x: float) -> str:
    return f"{x:,.0f}"


def _pct(x: float) -> str:
    return f"{x:.2f}%"


def _rate(x: float) -> str:
    return f"{x:.1f}%"


def _fx(x: float) -> str:
    return f"{x:.2f}"


# Reference values were read on 3 November 2026: BTC $76.7k, ETH $2,250, S&P 500
# 7,541, gold $3,956, Brent $104, EUR/USD 1.119, US 10-year 4.89%.
MULT = (0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.3, 1.5, 1.75)
QUANTITIES: tuple[Quantity, ...] = (
    Quantity(
        "btc",
        "the price of one bitcoin in US dollars",
        "at the CoinGecko daily close on 30 June 2027",
        "On 30 June 2027, will CoinGecko's daily closing price for bitcoin exceed {x}?",
        76_700,
        MULT,
        additive=False,
        rounding=5_000,
        fmt=_usd,
    ),
    Quantity(
        "eth",
        "the price of one ether (ETH) in US dollars",
        "at the CoinGecko daily close on 30 June 2027",
        "On 30 June 2027, will CoinGecko's daily closing price for ether exceed {x}?",
        2_250,
        MULT,
        additive=False,
        rounding=100,
        fmt=_usd,
    ),
    Quantity(
        "spx",
        "the S&P 500 index",
        "at its official close on 30 June 2027",
        "When US markets close on 30 June 2027, will the S&P 500 stand higher than {x}?",
        7_541,
        (0.7, 0.78, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.17, 1.25),
        additive=False,
        rounding=100,
        fmt=_plain,
    ),
    Quantity(
        "gold",
        "the LBMA Gold Price PM in US dollars per troy ounce",
        "on 30 June 2027",
        "On 30 June 2027, will the afternoon LBMA gold price exceed {x} an ounce?",
        3_956,
        (0.7, 0.78, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3),
        additive=False,
        rounding=100,
        fmt=_usd,
    ),
    Quantity(
        "brent",
        "the front-month Brent crude oil futures settlement price in US dollars per barrel",
        "on 30 June 2027",
        "Will front-month Brent crude settle above {x} a barrel on 30 June 2027?",
        104,
        (0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.3, 1.45),
        additive=False,
        rounding=5,
        fmt=_usd,
    ),
    Quantity(
        "us10y",
        "the US 10-year Treasury par yield (constant maturity, as published by the US Treasury)",
        "for 30 June 2027",
        "Will the US Treasury's published 10-year par yield for 30 June 2027 be higher than {x}?",
        4.89,
        (-1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0),
        additive=True,
        rounding=0.25,
        fmt=_pct,
    ),
    Quantity(
        "unemployment",
        "the US unemployment rate (U-3, seasonally adjusted) for June 2027",
        "as first published by the Bureau of Labor Statistics",
        "When the BLS first publishes the June 2027 jobs report, will the headline "
        "seasonally adjusted unemployment rate be higher than {x}?",
        4.5,
        (-1.0, -0.7, -0.4, -0.2, 0.0, 0.2, 0.4, 0.7, 1.0, 1.5),
        additive=True,
        rounding=0.1,
        fmt=_rate,
    ),
    Quantity(
        "eurusd",
        "the ECB euro reference rate against the US dollar (US dollars per euro)",
        "on 30 June 2027",
        "On 30 June 2027, will the ECB's euro reference rate be more than {x} US dollars?",
        1.119,
        (-0.12, -0.08, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.08, 0.12),
        additive=True,
        rounding=0.01,
        fmt=_fx,
    ),
)


@dataclass(frozen=True)
class Event:
    key: str
    question: str  # "a SpaceX Starship spacecraft carry ..." after "Will"
    clause: str  # indicative: "a SpaceX Starship spacecraft carries ..."
    noun: str  # gerund: "a SpaceX Starship spacecraft carrying ..."


EVENTS: tuple[Event, ...] = (
    Event(
        "starship",
        "a SpaceX Starship spacecraft carry at least one person into orbit",
        "a SpaceX Starship spacecraft carries at least one person into orbit",
        "a SpaceX Starship spacecraft carrying at least one person into orbit",
    ),
    Event(
        "moon",
        "astronauts land on the surface of the Moon",
        "astronauts land on the surface of the Moon",
        "astronauts landing on the surface of the Moon",
    ),
    Event(
        "btc150",
        "the price of one bitcoin reach $150,000 or more in CoinGecko's daily data",
        "the price of one bitcoin reaches $150,000 or more in CoinGecko's daily data",
        "the price of one bitcoin reaching $150,000 or more in CoinGecko's daily data",
    ),
    Event(
        "bear",
        "the S&P 500 close at least 20% below its highest previous close",
        "the S&P 500 closes at least 20% below its highest previous close",
        "the S&P 500 closing at least 20% below its highest previous close",
    ),
    Event(
        "pheic",
        "the World Health Organization declare a new Public Health Emergency of "
        "International Concern",
        "the World Health Organization declares a new Public Health Emergency of "
        "International Concern",
        "the World Health Organization declaring a new Public Health Emergency of "
        "International Concern",
    ),
    Event(
        "rejoin",
        "the United Kingdom formally apply to rejoin the European Union",
        "the United Kingdom formally applies to rejoin the European Union",
        "the United Kingdom formally applying to rejoin the European Union",
    ),
)

DATES = (
    ("2027-06-30", "30 June 2027"),
    ("2027-12-31", "31 December 2027"),
    ("2028-12-31", "31 December 2028"),
    ("2029-12-31", "31 December 2029"),
    ("2030-12-31", "31 December 2030"),
)


@dataclass(frozen=True)
class Contest:
    key: str
    win: str  # "{name} win the 2028 United States presidential election"
    clause: str  # "{name} wins the 2028 United States presidential election"
    other: str  # "the 2028 ... election be won by someone other than {names}"
    pool: tuple[str, ...]
    named: int  # how many of the pool to name; 0 means all, with no "other" bucket
    fixed: int = 2  # the first `fixed` entrants in the pool are always named


CONTESTS: tuple[Contest, ...] = (
    Contest(
        "us2028",
        "{name} win the 2028 United States presidential election",
        "{name} wins the 2028 United States presidential election",
        "the 2028 United States presidential election be won by someone other than {names}",
        (
            "JD Vance",
            "Gavin Newsom",
            "Josh Shapiro",
            "Alexandria Ocasio-Cortez",
            "Marco Rubio",
            "Pete Buttigieg",
            "Wes Moore",
            "Andy Beshear",
        ),
        5,
    ),
    Contest(
        "france2027",
        "{name} win the 2027 French presidential election",
        "{name} wins the 2027 French presidential election",
        "the 2027 French presidential election be won by someone other than {names}",
        (
            "Jordan Bardella",
            "Marine Le Pen",
            "Édouard Philippe",
            "Gabriel Attal",
            "Jean-Luc Mélenchon",
            "Bruno Retailleau",
            "Raphaël Glucksmann",
        ),
        5,
    ),
    Contest(
        "f1_2027",
        "{name} win the 2027 Formula 1 World Drivers' Championship",
        "{name} wins the 2027 Formula 1 World Drivers' Championship",
        "the 2027 Formula 1 World Drivers' Championship be won by someone other than {names}",
        (
            "Max Verstappen",
            "Lando Norris",
            "Oscar Piastri",
            "George Russell",
            "Charles Leclerc",
            "Lewis Hamilton",
            "Kimi Antonelli",
        ),
        5,
    ),
    Contest(
        "cwc2027",
        "{name} win the 2027 ICC Men's Cricket World Cup",
        "{name} wins the 2027 ICC Men's Cricket World Cup",
        "the 2027 ICC Men's Cricket World Cup be won by a team other than {names}",
        ("India", "Australia", "England", "South Africa", "New Zealand", "Pakistan"),
        5,
    ),
    Contest(
        "wwc2027",
        "{name} win the 2027 FIFA Women's World Cup",
        "{name} wins the 2027 FIFA Women's World Cup",
        "the 2027 FIFA Women's World Cup be won by a team other than {names}",
        ("Spain", "the United States", "England", "Germany", "France", "Brazil", "Sweden"),
        5,
    ),
    Contest(
        "party2028",
        "the {name} win the 2028 United States presidential election",
        "the {name} wins the 2028 United States presidential election",
        "",
        (
            "Republican Party's nominee",
            "Democratic Party's nominee",
            "candidate of neither the Republican nor the Democratic Party",
        ),
        0,
    ),
)

N_THRESHOLDS = 5
N_DATES = 3
N_PAIRS = 6


@dataclass
class _Builder:
    props: dict[str, Proposition]
    clauses: dict[str, str]
    items: list[Item]
    pair_candidates: dict[str, str]

    def prop(self, pid: str, text: str, topic: str, clause: str | None = None) -> str:
        if pid in self.props:
            if self.props[pid].text != text:
                raise ValueError(f"{pid} defined twice with different text")
        else:
            self.props[pid] = Proposition(pid, text, topic)
        if clause is not None:
            self.clauses[pid] = clause
        return pid


def build(seed: int = DEFAULT_SEED) -> Bank:
    rng = random.Random(seed)
    b = _Builder({}, {}, [], {})
    for q in QUANTITIES:
        _quantity(b, q, rng)
    for e in EVENTS:
        _event(b, e, rng)
    for c in CONTESTS:
        _contest(b, c, rng)
    _pairs(b, rng)
    return Bank(seed, AS_OF, tuple(b.props.values()), tuple(b.items))


def thresholds(q: Quantity, rng: random.Random) -> list[float]:
    """Five distinct rounded thresholds from the quantity's grid."""
    jitter = rng.uniform(-0.04, 0.04)
    for _ in range(100):
        steps = sorted(rng.sample(q.steps, N_THRESHOLDS))
        if q.additive:
            raw = [q.reference + s + jitter for s in steps]
        else:
            raw = [q.reference * s * (1 + jitter) for s in steps]
        cuts = [round(x / q.rounding) * q.rounding for x in raw]
        cuts = [round(x, 6) for x in cuts]
        if len(set(cuts)) == N_THRESHOLDS and all(c > 0 for c in cuts):
            return cuts
    raise ValueError(f"{q.key}: could not draw distinct thresholds")  # pragma: no cover


def _num(x: float) -> str:
    return f"{x:g}"


def _quantity(b: _Builder, q: Quantity, rng: random.Random) -> None:
    cuts = thresholds(q, rng)
    k = q.key
    worlds = [f"<= {q.fmt(cuts[0])}"]
    worlds += [f"{q.fmt(lo)} to {q.fmt(hi)}" for lo, hi in itertools.pairwise(cuts)]
    worlds.append(f"> {q.fmt(cuts[-1])}")

    def above(i: int) -> str:
        x = q.fmt(cuts[i])
        return b.prop(
            f"{k}.gt.{_num(cuts[i])}",
            f"Will {q.subject} be above {x} {q.when}?",
            k,
            clause=f"{q.subject} is above {x} {q.when}",
        )

    def at_or_below(i: int) -> str:
        return b.prop(
            f"{k}.le.{_num(cuts[i])}",
            f"Will {q.subject} be at or below {q.fmt(cuts[i])} {q.when}?",
            k,
        )

    ladder = [above(i) for i in range(N_THRESHOLDS)]
    b.pair_candidates[k] = ladder[2]
    b.items.append(
        Item(
            f"{k}.ladder",
            "monotonicity",
            k,
            tuple(worlds),
            tuple(Claim(p, indicator(worlds, worlds[i + 1 :])) for i, p in enumerate(ladder)),
            note="P(above t) can only fall as t rises",
        )
    )
    bins = [at_or_below(0)]
    for i in range(N_THRESHOLDS - 1):
        lo, hi = cuts[i], cuts[i + 1]
        bins.append(
            b.prop(
                f"{k}.bin.{_num(lo)}_to_{_num(hi)}",
                f"Will {q.subject} be above {q.fmt(lo)} and at or below {q.fmt(hi)} {q.when}?",
                k,
            )
        )
    bins.append(ladder[-1])
    b.items.append(
        Item(
            f"{k}.bins",
            "partition",
            k,
            tuple(worlds),
            tuple(Claim(p, indicator(worlds, [worlds[i]])) for i, p in enumerate(bins)),
            note="the six ranges are exhaustive and mutually exclusive",
        )
    )
    yes_no = ("yes", "no")
    for i in (0, 2):
        b.items.append(
            Item(
                f"{k}.complement.{_num(cuts[i])}",
                "complement",
                k,
                yes_no,
                (
                    Claim(ladder[i], indicator(yes_no, ["yes"])),
                    Claim(at_or_below(i), indicator(yes_no, ["no"])),
                ),
                note="above t, and at or below t",
            )
        )
    alt = b.prop(f"{k}.gt.{_num(cuts[2])}.alt", q.alt.format(x=q.fmt(cuts[2])), k)
    b.items.append(
        Item(
            f"{k}.paraphrase",
            "paraphrase",
            k,
            yes_no,
            (Claim(ladder[2], indicator(yes_no, ["yes"])), Claim(alt, indicator(yes_no, ["yes"]))),
            note="the same event in different words",
        )
    )


def _event(b: _Builder, e: Event, rng: random.Random) -> None:
    dates = sorted(rng.sample(DATES, N_DATES))
    worlds = [f"by {dates[0][1]}"]
    worlds += [f"{d0[1]} to {d1[1]}" for d0, d1 in itertools.pairwise(dates)]
    worlds.append(f"not by {dates[-1][1]}")
    ladder = []
    for iso, label in dates:
        ladder.append(
            b.prop(
                f"{e.key}.by.{iso}",
                f"Will {e.question} between {START} and {label}?",
                e.key,
                clause=f"{e.clause} between {START} and {label}",
            )
        )
    b.items.append(
        Item(
            f"{e.key}.ladder",
            "monotonicity",
            e.key,
            tuple(worlds),
            tuple(Claim(p, indicator(worlds, worlds[: i + 1])) for i, p in enumerate(ladder)),
            note="P(happens by D) can only rise with D",
        )
    )
    b.pair_candidates[e.key] = ladder[1]
    iso, label = dates[1]
    neg = b.prop(
        f"{e.key}.notby.{iso}",
        f"Will the period from {START} to {label} pass without {e.noun}?",
        e.key,
    )
    yes_no = ("yes", "no")
    b.items.append(
        Item(
            f"{e.key}.complement.{iso}",
            "complement",
            e.key,
            yes_no,
            (Claim(ladder[1], indicator(yes_no, ["yes"])), Claim(neg, indicator(yes_no, ["no"]))),
            note="happens by D, and the period ends without it",
        )
    )


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")


def _join(names: list[str]) -> str:
    return ", ".join(names[:-1]) + " or " + names[-1]


def _contest(b: _Builder, c: Contest, rng: random.Random) -> None:
    if c.named == 0:
        named = list(c.pool)
    else:
        named = [*c.pool[: c.fixed], *rng.sample(c.pool[c.fixed :], c.named - c.fixed)]
    claims = []
    worlds = [*named] if c.named == 0 else [*named, "other"]
    for name in named:
        pid = b.prop(
            f"{c.key}.win.{_slug(name)}",
            f"Will {c.win.format(name=name)}?",
            c.key,
            clause=c.clause.format(name=name),
        )
        claims.append(Claim(pid, indicator(worlds, [name])))
    if c.named:
        pid = b.prop(
            f"{c.key}.other",
            f"Will {c.other.format(names=_join(sorted(named)))}?",
            c.key,
        )
        claims.append(Claim(pid, indicator(worlds, ["other"])))
    b.items.append(
        Item(
            f"{c.key}.field",
            "partition",
            c.key,
            tuple(worlds),
            tuple(claims),
            note="exactly one entrant wins",
        )
    )
    if not c.named:
        return
    b.pair_candidates[c.key] = claims[0].prop
    first = named[0]
    yes_no = ("yes", "no")
    neg = b.prop(
        f"{c.key}.notwin.{_slug(first)}",
        f"Will {c.other.format(names=first)}?",
        c.key,
    )
    b.items.append(
        Item(
            f"{c.key}.complement.{_slug(first)}",
            "complement",
            c.key,
            yes_no,
            (
                Claim(claims[0].prop, indicator(yes_no, ["yes"])),
                Claim(neg, indicator(yes_no, ["no"])),
            ),
            note="X wins, and someone other than X wins",
        )
    )


# Topics that are logically linked, so a pair drawn from both could rule out one of
# the four combinations of A and B.
RELATED = ({"btc", "btc150"}, {"spx", "bear"}, {"us2028", "party2028"})


def _pairs(b: _Builder, rng: random.Random) -> None:
    # Candidates: the middle rung of each ladder and the first named entrant of each
    # contest. Pairs never share a topic or a related topic, so all four combinations
    # of A and B are possible.
    topics = sorted(b.pair_candidates)
    rng.shuffle(topics)
    worlds = ("A and B", "A only", "B only", "neither")
    pairs: list[tuple[str, str]] = []
    while topics and len(pairs) < N_PAIRS:
        ta = topics.pop(0)
        tb = next((t for t in topics if not _related(ta, t)), None)
        if tb is None:
            continue
        topics.remove(tb)
        pairs.append((ta, tb))
    for ta, tb in pairs:
        a, bb = b.pair_candidates[ta], b.pair_candidates[tb]
        ca, cb = b.clauses[a], b.clauses[bb]
        topic = f"pair.{ta}.{tb}"
        both = b.prop(
            f"{topic}.and",
            f"Will both of these be true: {ca}; and {cb}?",
            topic,
        )
        either = b.prop(
            f"{topic}.or",
            f"Will at least one of these be true: {ca}; or {cb}?",
            topic,
        )
        given = b.prop(
            f"{topic}.given",
            f"Suppose that {cb}. In that case, will it also be true that {ca}? "
            "(If the first part turns out false, the question is void.)",
            topic,
        )
        ind_a = indicator(worlds, ["A and B", "A only"])
        ind_b = indicator(worlds, ["A and B", "B only"])
        ind_ab = indicator(worlds, ["A and B"])
        b.items.append(
            Item(
                f"{topic}.conjunction",
                "conjunction",
                topic,
                worlds,
                (
                    Claim(a, ind_a),
                    Claim(bb, ind_b),
                    Claim(both, ind_ab),
                    Claim(either, indicator(worlds, ["A and B", "A only", "B only"])),
                ),
                note="Frechet bounds, and P(A) + P(B) = P(A and B) + P(A or B)",
            )
        )
        b.items.append(
            Item(
                f"{topic}.conditional",
                "conditional",
                topic,
                worlds,
                (Claim(given, ind_a, given=ind_b), Claim(both, ind_ab), Claim(bb, ind_b)),
                note="P(A | B) P(B) = P(A and B)",
            )
        )


def _related(a: str, b: str) -> bool:
    return any(a in group and b in group for group in RELATED)
