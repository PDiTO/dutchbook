"""The same Dutch book, priced against live Polymarket markets.

A market is a forecaster too, and Polymarket lists the two shapes the benchmark uses
most: negRisk events, whose outcomes are mutually exclusive (who wins, which range),
and threshold ladders ("Bitcoin above ___ on November 8?", "What price will Bitcoin
hit in November?"). Each becomes an `Item`, and the solver runs twice:

- at the **mid** of each market, which is the market's own probability and the
  number comparable to a model's answer;
- at the **bid and ask**, which is what a trader could actually lock in. Buying
  happens at the ask and selling at the bid, before fees.

Only Gamma's public `/events` endpoint is used, without a key. `snapshot` saves the
fields it needs to `data/markets/`, so the analysis rebuilds offline.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dutchbook.book import dutch_book, profit
from dutchbook.items import Claim, Item, indicator

GAMMA = "https://gamma-api.polymarket.com/events"
KEEP = (
    "question",
    "groupItemTitle",
    "bestBid",
    "bestAsk",
    "lastTradePrice",
    "active",
    "closed",
    "negRiskOther",
    "liquidityNum",
    "volumeNum",
)


def _get(params: dict[str, Any]) -> list[dict[str, Any]]:  # pragma: no cover - network
    url = f"{GAMMA}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "dutchbook/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data if isinstance(data, list) else []


def snapshot(out: Path, pages: int = 20, page_size: int = 100) -> Path:  # pragma: no cover
    """Active events by 24-hour volume, trimmed to what the analysis reads."""
    events: dict[str, dict[str, Any]] = {}
    for page in range(pages):
        batch = _get(
            {
                "closed": "false",
                "active": "true",
                "limit": page_size,
                "offset": page * page_size,
                "order": "volume24hr",
                "ascending": "false",
            }
        )
        if not batch:
            break
        for e in batch:
            events[str(e["id"])] = {
                "id": e["id"],
                "title": e.get("title"),
                "negRisk": e.get("negRisk"),
                "negRiskAugmented": e.get("negRiskAugmented"),
                "volume24hr": e.get("volume24hr"),
                "markets": [{k: m.get(k) for k in KEEP} for m in e.get("markets", [])],
            }
        time.sleep(0.5)
    taken = datetime.now(UTC)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"polymarket-{taken:%Y-%m-%d}.json"
    write_snapshot(path, list(events.values()), taken.isoformat(timespec="seconds"))
    return path


def usable(event: dict[str, Any]) -> bool:
    """Whether an event yields at least one partition or ladder item."""
    return partition_item(event) is not None or bool(ladder_items(event))


def write_snapshot(path: Path, events: list[dict[str, Any]], taken: str) -> None:
    """Keep only the events the analysis can use, one per line."""
    kept = [e for e in events if usable(e)]
    body = ",\n".join(json.dumps(e, ensure_ascii=False) for e in kept)
    path.write_text(
        f'{{"source": "{GAMMA}", "taken": "{taken}", "listed": {len(events)}, '
        f'"events": [\n{body}\n]}}\n'
    )
    print(f"{len(events)} events listed, {len(kept)} usable -> {path}")


@dataclass(frozen=True)
class Quote:
    label: str
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(frozen=True)
class MarketBook:
    event: str
    kind: str  # partition | ladder
    n: int
    mid_profit: float
    executable_profit: float
    max_spread: float
    total_mid: float  # sum of mids, for partitions
    volume24hr: float


def _quote(m: dict[str, Any]) -> Quote | None:
    bid, ask = m.get("bestBid"), m.get("bestAsk")
    if not m.get("active") or m.get("closed") or bid is None or ask is None:
        return None
    bid, ask = float(bid), float(ask)
    if not 0.0 < ask <= 1.0 or not 0.0 <= bid < ask:
        return None
    return Quote(str(m.get("groupItemTitle") or m.get("question")), bid, ask)


def partition_item(event: dict[str, Any]) -> tuple[Item, list[Quote]] | None:
    """A negRisk event as a partition, or None if some live outcome has no quote.

    A plain negRisk event lists every outcome. An augmented one can add outcomes
    later, so it is only treated as exhaustive when it has a live "Other" market;
    otherwise there is an extra world in which none of the listed outcomes wins.
    """
    if not event.get("negRisk"):
        return None
    live = [m for m in event["markets"] if m.get("active") and not m.get("closed")]
    quotes = [_quote(m) for m in live]
    if len(live) < 2 or any(q is None for q in quotes):
        return None
    qs = [q for q in quotes if q is not None]
    exhaustive = not event.get("negRiskAugmented") or any(m.get("negRiskOther") for m in live)
    worlds = [f"o{i}" for i in range(len(qs))] + ([] if exhaustive else ["none"])
    claims = tuple(Claim(f"o{i}", indicator(worlds, [f"o{i}"])) for i in range(len(qs)))
    item = Item(f"pm.{event['id']}", "partition", "polymarket", tuple(worlds), claims)
    return item, qs


THRESHOLD = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?\s*[kKmMbB]?")


def _threshold(label: str) -> float | None:
    m = THRESHOLD.search(label.replace("$", ""))
    if m is None:
        return None
    text = m.group(0).replace(",", "").strip()
    mult = {"k": 1e3, "m": 1e6, "b": 1e9}.get(text[-1].lower(), 1.0)
    if text[-1].isalpha():
        text = text[:-1]
    try:
        return float(text) * mult
    except ValueError:  # pragma: no cover
        return None


def ladder_items(event: dict[str, Any]) -> list[tuple[Item, list[Quote]]]:
    """Threshold ladders: "above X", "reach X" (up) and "dip to X" (down) chains.

    Each chain is a set of nested events. "Above 70k" contains "above 72k"; "reach
    100k" contains "reach 110k"; "dip to 60k" contains "dip to 55k". A chain becomes
    an item whose worlds are the intervals between thresholds.
    """
    if event.get("negRisk"):
        return []
    chains: dict[str, list[tuple[float, Quote]]] = {}
    for m in event["markets"]:
        q = _quote(m)
        label = str(m.get("groupItemTitle") or "")
        question = str(m.get("question") or "").lower()
        x = _threshold(label)
        if q is None or x is None:
            continue
        if "↓" in label or " dip" in question or "(low)" in question:
            direction = "down"
        elif "↑" in label or " reach" in question or "(high)" in question or " above" in question:
            direction = "up"
        else:
            continue
        chains.setdefault(direction, []).append((x, q))
    out = []
    for direction, found in chains.items():
        rows = sorted({x: (x, q) for x, q in found}.values(), key=lambda r: r[0])
        if len(rows) < 3:
            continue
        if direction == "down":
            rows = rows[::-1]  # widest event (highest dip level) first
        n = len(rows)
        worlds = [f"w{i}" for i in range(n + 1)]
        # Claim i is true in worlds i+1..n: nested, widest first.
        claims = tuple(Claim(f"t{i}", indicator(worlds, worlds[i + 1 :])) for i in range(n))
        item = Item(
            f"pm.{event['id']}.{direction}", "monotonicity", "polymarket", tuple(worlds), claims
        )
        out.append((item, [q for _, q in rows]))
    return out


def books(snapshot_path: Path) -> list[MarketBook]:
    data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    out = []
    for event in data["events"]:
        found = []
        part = partition_item(event)
        if part is not None:
            found.append(("partition", *part))
        found += [("ladder", item, qs) for item, qs in ladder_items(event)]
        for kind, item, qs in found:
            mids = [q.mid for q in qs]
            out.append(
                MarketBook(
                    event=str(event.get("title")),
                    kind=kind,
                    n=len(qs),
                    mid_profit=profit(item, mids),
                    executable_profit=dutch_book(
                        item, [q.ask for q in qs], sell_prices=[q.bid for q in qs]
                    ).profit,
                    max_spread=max(q.spread for q in qs),
                    total_mid=sum(mids),
                    volume24hr=float(event.get("volume24hr") or 0.0),
                )
            )
    return out


def latest(root: Path) -> Path | None:
    files = sorted(root.glob("polymarket-*.json"))
    return files[-1] if files else None
