"""Polymarket books, from a hand-built snapshot in the shape Gamma returns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dutchbook.markets import _threshold, books, ladder_items, latest, partition_item


def market(label: str, bid: float | None, ask: float | None, **extra: Any) -> dict[str, Any]:
    return {
        "question": extra.pop("question", f"Will {label}?"),
        "groupItemTitle": label,
        "bestBid": bid,
        "bestAsk": ask,
        "active": extra.pop("active", True),
        "closed": False,
        "negRiskOther": extra.pop("other", False),
    }


def event(eid: int, title: str, markets: list[dict[str, Any]], **flags: Any) -> dict[str, Any]:
    return {"id": eid, "title": title, "markets": markets, "volume24hr": 1000.0, **flags}


SNAPSHOT = {
    "source": "test",
    "taken": "2026-11-03T00:00:00+00:00",
    "events": [
        # Mids sum to 1.05; asks to 1.08, bids to 1.02, so selling every leg at the bid
        # locks in 2 cents over three contracts.
        event(
            1,
            "Who wins?",
            [market("A", 0.49, 0.51), market("B", 0.29, 0.31), market("C", 0.24, 0.26)],
            negRisk=True,
        ),
        # Augmented with no live Other: an extra "none" world, so only sum <= 1 binds.
        event(
            2,
            "Who else?",
            [
                market("A", 0.3, 0.32),
                market("B", 0.3, 0.32),
                market("Party A", None, None, active=False),
            ],
            negRisk=True,
            negRiskAugmented=True,
        ),
        # An "above" ladder with one inversion at the mid: 72k quoted above 70k.
        event(
            3,
            "Bitcoin above ___ on November 8?",
            [
                market(
                    "68,000", 0.80, 0.82, question="Will the price of Bitcoin be above $68,000?"
                ),
                market(
                    "70,000", 0.60, 0.62, question="Will the price of Bitcoin be above $70,000?"
                ),
                market(
                    "72,000", 0.63, 0.65, question="Will the price of Bitcoin be above $72,000?"
                ),
                market(
                    "74,000", 0.20, 0.22, question="Will the price of Bitcoin be above $74,000?"
                ),
            ],
            negRisk=False,
        ),
        # Reach and dip chains in one event, each coherent.
        event(
            4,
            "What price will Bitcoin hit in November?",
            [
                market("↑ 90,000", 0.30, 0.32),
                market("↑ 100,000", 0.10, 0.12),
                market("↑ 110,000", 0.04, 0.05),
                market("↓ 70,000", 0.40, 0.42),
                market("↓ 60,000", 0.15, 0.17),
                market("↓ 50,000", 0.05, 0.06),
            ],
            negRisk=False,
        ),
    ],
}


@pytest.fixture
def snap(tmp_path: Path) -> Path:
    path = tmp_path / "polymarket-2026-11-03.json"
    path.write_text(json.dumps(SNAPSHOT))
    return path


def test_books(snap: Path) -> None:
    by = {(b.event, b.kind, b.n): b for b in books(snap)}
    who = by[("Who wins?", "partition", 3)]
    assert who.total_mid == pytest.approx(1.05)
    assert who.mid_profit == pytest.approx(0.05 / 3)
    assert who.executable_profit == pytest.approx(0.02 / 3)
    other = by[("Who else?", "partition", 2)]
    assert other.mid_profit == 0.0  # 0.62 total is fine when "none" is possible
    ladder = by[("Bitcoin above ___ on November 8?", "ladder", 4)]
    assert ladder.mid_profit == pytest.approx(0.015)  # (0.64 - 0.61) / 2
    assert ladder.executable_profit == pytest.approx(0.005)  # (0.63 - 0.62) / 2
    hits = [b for b in books(snap) if b.event.startswith("What price")]
    assert len(hits) == 2
    assert all(b.mid_profit == 0.0 for b in hits)


def test_partition_needs_every_live_quote() -> None:
    e = event(9, "x", [market("A", 0.5, 0.52), market("B", None, None)], negRisk=True)
    assert partition_item(e) is None
    assert partition_item(event(9, "x", [market("A", 0.5, 0.52)], negRisk=False)) is None


def test_ladders_skip_negrisk_and_short_chains() -> None:
    assert ladder_items(event(9, "x", [market("↑ 1", 0.1, 0.2)], negRisk=True)) == []
    short = event(9, "x", [market("↑ 1", 0.1, 0.2), market("↑ 2", 0.05, 0.1)], negRisk=False)
    assert ladder_items(short) == []


@pytest.mark.parametrize(
    ("label", "value"),
    [("70,000", 70000), ("↑ $100M", 1e8), ("↓ 2.5k", 2500), ("$790", 790), ("none", None)],
)
def test_threshold(label: str, value: float | None) -> None:
    assert _threshold(label) == value


def test_latest(tmp_path: Path) -> None:
    assert latest(tmp_path) is None
    for d in ("2026-11-01", "2026-11-03"):
        (tmp_path / f"polymarket-{d}.json").write_text("{}")
    assert latest(tmp_path) == tmp_path / "polymarket-2026-11-03.json"


def test_write_snapshot_keeps_usable_events(tmp_path: Path, snap: Path) -> None:
    from dutchbook.markets import write_snapshot

    junk = event(99, "Will it rain?", [market("Yes", 0.4, 0.5)], negRisk=False)
    path = tmp_path / "polymarket-2026-11-04.json"
    write_snapshot(path, [*SNAPSHOT["events"], junk], "2026-11-04T00:00:00+00:00")  # type: ignore[list-item]
    data = json.loads(path.read_text())
    assert data["listed"] == 5
    assert [e["id"] for e in data["events"]] == [1, 2, 3, 4]
    assert books(path) == books(snap)
