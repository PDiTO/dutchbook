"""Propositions, and the items that tie them together with a logical structure.

A *proposition* is one yes/no question, asked in its own conversation. An *item* is a
small set of propositions together with the possible worlds they live in. Each
proposition enters an item as a *claim*: a 0/1 indicator over the item's worlds, and,
for a conditional probability, a second indicator for the condition.

Writing every relation as indicators over an explicit list of worlds is what makes
the constraints correct by construction. "BTC above $100k" and "BTC above $120k" are
not related by a rule someone typed in; they are two indicators over the same
intervals of the BTC price, and the Dutch book solver finds whatever bets the
indicators allow.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FAMILIES = (
    "complement",
    "monotonicity",
    "partition",
    "conjunction",
    "conditional",
    "paraphrase",
)


@dataclass(frozen=True)
class Proposition:
    id: str
    text: str
    topic: str


@dataclass(frozen=True)
class Claim:
    """How one proposition's probability enters an item.

    `event[w]` is 1 if the proposition is true in world w. For a conditional
    probability P(A | B), `event` is A and `given` is B; the bet on it is called off
    in worlds where B is false.
    """

    prop: str
    event: tuple[int, ...]
    given: tuple[int, ...] | None = None

    @property
    def conditional(self) -> bool:
        return self.given is not None


@dataclass(frozen=True)
class Item:
    id: str
    family: str
    topic: str
    worlds: tuple[str, ...]
    claims: tuple[Claim, ...]
    note: str = ""
    tags: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"{self.id}: unknown family {self.family!r}")
        n = len(self.worlds)
        if n < 2:
            raise ValueError(f"{self.id}: an item needs at least two worlds")
        for c in self.claims:
            if len(c.event) != n or (c.given is not None and len(c.given) != n):
                raise ValueError(f"{self.id}: claim {c.prop} has the wrong number of worlds")
            if any(v not in (0, 1) for v in (*c.event, *(c.given or ()))):
                raise ValueError(f"{self.id}: claim {c.prop} is not a 0/1 indicator")
            if c.given is not None and not any(c.given):
                raise ValueError(f"{self.id}: claim {c.prop} conditions on an impossible event")
        if len({c.prop for c in self.claims}) != len(self.claims):
            raise ValueError(f"{self.id}: a proposition appears twice")

    @property
    def props(self) -> tuple[str, ...]:
        return tuple(c.prop for c in self.claims)


@dataclass(frozen=True)
class Bank:
    """Every proposition to ask about and every item to score."""

    seed: int
    as_of: str
    propositions: tuple[Proposition, ...]
    items: tuple[Item, ...]

    def __post_init__(self) -> None:
        ids = [p.id for p in self.propositions]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate proposition id")
        texts = [p.text for p in self.propositions]
        if len(set(texts)) != len(texts):
            raise ValueError("two propositions share the same text")
        known = set(ids)
        for item in self.items:
            missing = [p for p in item.props if p not in known]
            if missing:
                raise ValueError(f"{item.id} refers to unknown propositions {missing}")

    def prop(self, prop_id: str) -> Proposition:
        for p in self.propositions:
            if p.id == prop_id:
                return p
        raise KeyError(prop_id)

    def to_json(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "as_of": self.as_of,
            "propositions": [asdict(p) for p in self.propositions],
            "items": [_item_json(i) for i in self.items],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Bank:
        return cls(
            seed=int(data["seed"]),
            as_of=str(data["as_of"]),
            propositions=tuple(Proposition(**p) for p in data["propositions"]),
            items=tuple(_item_from_json(i) for i in data["items"]),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> Bank:
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))


def _item_json(item: Item) -> dict[str, Any]:
    return {
        "id": item.id,
        "family": item.family,
        "topic": item.topic,
        "worlds": list(item.worlds),
        "claims": [
            {"prop": c.prop, "event": list(c.event)}
            | ({"given": list(c.given)} if c.given is not None else {})
            for c in item.claims
        ],
        "note": item.note,
        "tags": list(item.tags),
    }


def _item_from_json(data: dict[str, Any]) -> Item:
    return Item(
        id=data["id"],
        family=data["family"],
        topic=data["topic"],
        worlds=tuple(data["worlds"]),
        claims=tuple(
            Claim(
                prop=c["prop"],
                event=tuple(c["event"]),
                given=tuple(c["given"]) if "given" in c else None,
            )
            for c in data["claims"]
        ),
        note=data.get("note", ""),
        tags=tuple(data.get("tags", ())),
    )


def indicator(worlds: Sequence[str], true_in: Iterable[str]) -> tuple[int, ...]:
    true = set(true_in)
    unknown = true - set(worlds)
    if unknown:
        raise ValueError(f"unknown worlds {sorted(unknown)}")
    return tuple(int(w in true) for w in worlds)
