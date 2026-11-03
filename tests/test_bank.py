"""The generated bank: deterministic, well-formed, and logically correct.

The strongest check is semantic. Proposition ids encode what each question means
(`btc.gt.80000`, `starship.by.2028-12-31`, `us2028.win.jd_vance`), so the test draws
random outcomes for the world, works out whether each proposition is true by reading
its id, and checks that the item's indicators agree in some world. If a template or a
world list were wrong, some outcome would fit no world.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pytest

from dutchbook.bank import CONTESTS, DATES, EVENTS, QUANTITIES, build
from dutchbook.book import dutch_book
from dutchbook.items import Bank, Item

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Outcome:
    values: dict[str, float]  # quantity key -> value
    happened: dict[str, str | None]  # event key -> ISO date it first happened, or None
    winners: dict[str, str]  # contest key -> slug of the winner


def draw_outcome(rng: random.Random, bank: Bank) -> Outcome:
    values = {}
    for q in QUANTITIES:
        cuts = sorted(
            float(p.id.split(".gt.")[1].removesuffix(".alt"))
            for p in bank.propositions
            if p.topic == q.key and ".gt." in p.id
        )
        # Land exactly on a cut sometimes, to test the at-or-below boundary.
        if rng.random() < 0.3:
            values[q.key] = rng.choice(cuts)
        else:
            values[q.key] = rng.uniform(cuts[0] * 0.5, cuts[-1] * 1.5)
    happened = {}
    for e in EVENTS:
        options = [d for d, _ in DATES] + ["2026-12-01", "2028-06-01", None]
        happened[e.key] = rng.choice(options)
    winners = {}
    for c in CONTESTS:
        slugs = [
            p.id.split(".win.")[1]
            for p in bank.propositions
            if p.topic == c.key and ".win." in p.id
        ]
        winners[c.key] = rng.choice([*slugs, "somebody_else"]) if c.named else rng.choice(slugs)
    return Outcome(values, happened, winners)


def truth(pid: str, o: Outcome, bank: Bank) -> bool:
    topic, rest = pid.split(".", 1)
    if topic in o.values:
        x = o.values[topic]
        kind, *args = rest.removesuffix(".alt").split(".", 1)
        if kind == "gt":
            return x > float(args[0])
        if kind == "le":
            return x <= float(args[0])
        lo, hi = _two_floats(args[0])
        return lo < x <= hi
    if topic in o.happened:
        kind, iso = rest.split(".", 1)
        when = o.happened[topic]
        by = when is not None and when <= iso
        return by if kind == "by" else not by
    if topic in o.winners:
        winner = o.winners[topic]
        if rest == "other":
            named = [
                p.id.split(".win.")[1]
                for p in bank.propositions
                if p.topic == topic and ".win." in p.id
            ]
            return winner not in named
        kind, slug = rest.split(".", 1)
        return (winner == slug) if kind == "win" else (winner != slug)
    raise AssertionError(f"no semantics for {pid}")


def _two_floats(s: str) -> tuple[float, float]:
    lo, hi = s.split("_to_")
    return float(lo), float(hi)


def pair_truths(item: Item, o: Outcome, bank: Bank) -> dict[str, bool]:
    """Truth of every claim in a pair item, from the truths of A and B."""
    pair = next(i for i in bank.items if i.topic == item.topic and i.family == "conjunction")
    a = truth(pair.claims[0].prop, o, bank)
    b = truth(pair.claims[1].prop, o, bank)
    out = {pair.claims[0].prop: a, pair.claims[1].prop: b}
    out[f"{item.topic}.and"] = a and b
    out[f"{item.topic}.or"] = a or b
    out[f"{item.topic}.given"] = a  # only scored when b holds
    out["B"] = b
    return out


def matching_worlds(item: Item, o: Outcome, bank: Bank) -> list[int]:
    if item.topic.startswith("pair."):
        t = pair_truths(item, o, bank)
    else:
        t = {c.prop: truth(c.prop, o, bank) for c in item.claims}
    hits = []
    for w in range(len(item.worlds)):
        ok = True
        for c in item.claims:
            if c.given is not None:
                b = t["B"]
                if c.given[w] != b or (b and c.event[w] != t[c.prop]):
                    ok = False
            elif c.event[w] != t[c.prop]:
                ok = False
        if ok:
            hits.append(w)
    return hits


@pytest.fixture(scope="module")
def bank() -> Bank:
    return build()


def test_same_seed_same_bank() -> None:
    assert build(7).to_json() == build(7).to_json()


def test_new_seed_new_questions() -> None:
    a = {p.text for p in build(1).propositions}
    b = {p.text for p in build(2).propositions}
    assert len(a - b) > 40


def test_committed_bank_matches_generator() -> None:
    committed = ROOT / "data" / "bank.json"
    if not committed.exists():
        pytest.skip("no committed bank")
    assert Bank.load(committed).to_json() == build(Bank.load(committed).seed).to_json()


def test_round_trips_through_json(bank: Bank, tmp_path: Path) -> None:
    bank.save(tmp_path / "bank.json")
    assert Bank.load(tmp_path / "bank.json") == bank


def test_every_family_present(bank: Bank) -> None:
    families = {i.family for i in bank.items}
    assert families == {
        "complement",
        "monotonicity",
        "partition",
        "conjunction",
        "conditional",
        "paraphrase",
    }


def test_every_proposition_is_scored(bank: Bank) -> None:
    used = {p for item in bank.items for p in item.props}
    assert used == {p.id for p in bank.propositions}


@pytest.mark.parametrize("seed", [1, 2, 3, 20261103])
def test_indicators_match_meaning(seed: int) -> None:
    bank = build(seed)
    rng = random.Random(seed)
    seen: dict[str, set[int]] = {i.id: set() for i in bank.items}
    for _ in range(400):
        o = draw_outcome(rng, bank)
        for item in bank.items:
            hits = matching_worlds(item, o, bank)
            # A conditional item can't tell "A only" from "neither": with B false,
            # the conditional bet is void and A alone is not asked.
            assert len(hits) == 1 or (item.family == "conditional" and len(hits) == 2), item.id
            seen[item.id].update(hits)
    # Every world of every item came up, so no world is impossible.
    for item in bank.items:
        assert seen[item.id] == set(range(len(item.worlds))), item.id


def test_true_answers_are_coherent(bank: Bank) -> None:
    # Answering 1 for true and 0 for false, in any world, is coherent.
    rng = random.Random(0)
    for _ in range(20):
        o = draw_outcome(rng, bank)
        for item in bank.items:
            w = matching_worlds(item, o, bank)[0]
            prices = [
                float(c.event[w]) if c.given is None or c.given[w] else 0.5 for c in item.claims
            ]
            if any(c.given is not None and not c.given[w] for c in item.claims):
                continue
            assert dutch_book(item, prices).profit == 0.0


def test_pairs_avoid_related_topics(bank: Bank) -> None:
    for item in bank.items:
        if item.family == "conjunction":
            _, ta, tb = item.topic.split(".")
            assert {ta, tb} not in ({"btc", "btc150"}, {"spx", "bear"}, {"us2028", "party2028"})


def test_contests_name_the_front_runners(bank: Bank) -> None:
    ids = {p.id for p in bank.propositions}
    assert "us2028.win.jd_vance" in ids
    assert "us2028.win.gavin_newsom" in ids


def test_items_reject_malformed_structure() -> None:
    from dutchbook.items import Claim, Item, Proposition

    w = ("yes", "no")
    with pytest.raises(ValueError, match="unknown family"):
        Item("x", "nonsense", "t", w, ())
    with pytest.raises(ValueError, match="two worlds"):
        Item("x", "complement", "t", ("only",), ())
    with pytest.raises(ValueError, match="wrong number"):
        Item("x", "complement", "t", w, (Claim("a", (1,)),))
    with pytest.raises(ValueError, match="0/1"):
        Item("x", "complement", "t", w, (Claim("a", (2, 0)),))
    with pytest.raises(ValueError, match="impossible"):
        Item("x", "conditional", "t", w, (Claim("a", (1, 0), given=(0, 0)),))
    with pytest.raises(ValueError, match="twice"):
        Item("x", "complement", "t", w, (Claim("a", (1, 0)), Claim("a", (0, 1))))
    p = Proposition("a", "A?", "t")
    with pytest.raises(ValueError, match="duplicate"):
        Bank(1, "now", (p, p), ())
    with pytest.raises(ValueError, match="same text"):
        Bank(1, "now", (p, Proposition("b", "A?", "t")), ())
    item = Item("x", "complement", "t", w, (Claim("zz", (1, 0)), Claim("a", (0, 1))))
    with pytest.raises(ValueError, match="unknown propositions"):
        Bank(1, "now", (p,), (item,))
    assert Bank(1, "now", (p,), ()).prop("a") == p
    with pytest.raises(KeyError):
        Bank(1, "now", (p,), ()).prop("b")
