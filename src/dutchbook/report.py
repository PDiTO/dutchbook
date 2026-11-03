"""Rebuild every number and chart from the recordings: results/ and docs/img/.

`dutchbook report` reads `data/bank.json`, every `data/recordings/*.jsonl` and the
newest `data/markets/polymarket-*.json`, and makes no network calls. The seeds are
fixed, so running it twice gives the same files.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from dutchbook import baselines, plots
from dutchbook.analysis import (
    Calls,
    ModelScore,
    Samples,
    load_samples,
    paired_difference,
    score_model,
)
from dutchbook.bank import EVENTS, QUANTITIES
from dutchbook.items import FAMILIES, Bank
from dutchbook.markets import MarketBook, books, latest

QUANTITY_KEYS = {q.key for q in QUANTITIES}
EVENT_KEYS = {e.key for e in EVENTS}
TIGHT = 0.05  # a market counts as liquid if every leg's spread is at most 5 cents
MODEL_ORDER = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "gpt-6.1-sol")


def build_report(
    bank: Bank,
    recordings: Path,
    results: Path,
    figures: Path,
    markets_dir: Path,
    *,
    draws: int = 1000,
    boot: int = 2000,
    k: int = 5,
    model_order: Sequence[str] = MODEL_ORDER,
    quiet: bool = False,
) -> dict[str, Any]:
    results.mkdir(parents=True, exist_ok=True)
    img = figures / "img"
    files = sorted(recordings.glob("*.jsonl"))
    order = {m: i for i, m in enumerate(model_order)}
    files.sort(key=lambda f: (order.get(f.stem, len(order)), f.stem))

    models: list[ModelScore] = []
    model_samples: list[Samples] = []
    for n, f in enumerate(files):
        samples, calls = load_samples(f)
        if not quiet:
            print(f"scoring {f.stem}")
        models.append(
            score_model(f.stem, bank, samples, draws=draws, boot=boot, seed=n, calls=calls)
        )
        model_samples.append(samples)

    # Baselines go through exactly the same pipeline.
    coherent = baselines.coherent_forecaster(bank, seed=0)
    noise_sd = float(np.median([m.retest.pooled_sd for m in models])) if models else 0.05
    rng = np.random.default_rng(7)
    base_samples = {
        "coherent": baselines.repeat(coherent, k),
        "coherent + noise": baselines.repeat(coherent, k, baselines.noisy(noise_sd, rng)),
        "random": baselines.random_forecaster(bank, k, seed=0),
    }
    base_scores = [
        score_model(name, bank, s, draws=min(draws, 300), boot=boot, seed=100 + i)
        for i, (name, s) in enumerate(base_samples.items())
    ]

    snap = latest(markets_dir)
    market_books = books(snap) if snap else []

    summary = _summary(bank, models, base_scores, market_books, noise_sd, snap, boot)
    (results / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    _items_csv(models + base_scores, results / "items.csv")
    _answers_csv(bank, models, model_samples, results / "answers.csv")
    (results / "report.md").write_text(_markdown(summary, models, base_scores))

    if models:
        # The random forecaster is ten times further out and would flatten the chart;
        # its numbers are in the tables.
        plots.headline(
            models, [b for b in base_scores if b.model == "coherent + noise"], img / "headline.svg"
        )
        plots.families(models, img / "families.svg")
        plots.null_panels(models, img / "noise-null.svg")
        plots.retest(models, model_samples, img / "retest.svg")
        plots.partition_sums(models, _partition_labels(bank), img / "partition-sums.svg")
        _ladder_charts(bank, models, model_samples, img)
        plots.bins_vs_ladder(
            models, [bin_points(bank, s) for s in model_samples], img / "bins-vs-ladder.svg"
        )
        if market_books:
            plots.markets(
                {
                    "models asked once, all four": [
                        s.single
                        for m in models
                        for s in m.items
                        if s.family in ("partition", "monotonicity")
                    ]
                },
                {
                    f"Polymarket, every leg within {TIGHT * 100:.0f}c": [
                        b.mid_profit for b in market_books if b.max_spread <= TIGHT
                    ],
                    "Polymarket, all quoted": [b.mid_profit for b in market_books],
                },
                img / "markets.svg",
            )
    if not quiet:
        print((results / "report.md").read_text())
    return summary


def _calls(c: Calls | None) -> dict[str, Any]:
    if c is None:
        return {}
    return {
        "recorded_calls": c.calls,
        "transport_errors": c.transport_errors,
        "replies": c.replies,
        "status": dict(c.status),
        "tool_events": c.tool_events,
        "seconds": round(c.seconds, 1),
        "cost_usd_reported": round(c.cost_usd, 2),
    }


def _model_json(m: ModelScore) -> dict[str, Any]:
    return {
        "model": m.model,
        "items": len(m.items),
        "excluded": m.excluded,
        "calls": _calls(m.calls),
        "retest": asdict(m.retest),
        "headline": {k: asdict(v) for k, v in m.headline.items()},
        "by_family": {f: {k: asdict(v) for k, v in d.items()} for f, d in m.by_family.items()},
        "pooled_p_noise_null": m.pooled_p,
        "pooled_p_pooled_sd_null": m.pooled_p_alt,
        "significant_items_bh05": {f: list(v) for f, v in m.significant.items()},
        "sums": sums(m),
        "share_of_single_explained_by_noise": (
            m.headline["null_single"].value / m.headline["single"].value
            if m.headline and m.headline["single"].value > 0
            else None
        ),
    }


def _summary(
    bank: Bank,
    models: Sequence[ModelScore],
    base: Sequence[ModelScore],
    market_books: Sequence[MarketBook],
    noise_sd: float,
    snap: Path | None,
    boot: int,
) -> dict[str, Any]:
    pairs = {}
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            for metric in ("single", "averaged", "excess"):
                iv = paired_difference(a, b, metric, boot=boot)
                pairs[f"{a.model} - {b.model}: {metric}"] = asdict(iv)
    mk: dict[str, Any] = {}
    if snap is not None:
        mk["snapshot"] = snap.name
        for kind in ("partition", "ladder"):
            for label, rows in (
                ("all", [b for b in market_books if b.kind == kind]),
                ("tight", [b for b in market_books if b.kind == kind and b.max_spread <= TIGHT]),
            ):
                if not rows:
                    continue
                mid = np.array([b.mid_profit for b in rows])
                ex = np.array([b.executable_profit for b in rows])
                mk[f"{kind}/{label}"] = {
                    "n": len(rows),
                    "mid_mean": float(mid.mean()),
                    "mid_median": float(np.median(mid)),
                    "mid_share_incoherent": float(np.mean(mid > 1e-9)),
                    "executable_n": int(np.sum(ex > 1e-9)),
                    "executable_mean": float(ex.mean()),
                    "executable_max": float(ex.max()),
                }
    return {
        "bank": {
            "seed": bank.seed,
            "propositions": len(bank.propositions),
            "items": len(bank.items),
            "families": {f: sum(1 for i in bank.items if i.family == f) for f in FAMILIES},
        },
        "models": [_model_json(m) for m in models],
        "baselines": [_model_json(b) for b in base],
        "baseline_noise_sd": noise_sd,
        "paired_differences": pairs,
        "markets": mk,
    }


def _kind(topic: str) -> str:
    if topic in QUANTITY_KEYS:
        return "numeric"
    if topic in EVENT_KEYS:
        return "dated event"
    return "who wins"


def sums(m: ModelScore) -> dict[str, dict[str, float]]:
    """Mean of the averaged answers summed over each complement or partition item.

    Coherent answers sum to exactly 1, so this shows which way the errors lean, by
    family and by kind of topic.
    """
    out: dict[str, dict[str, float]] = {}
    for family in ("complement", "partition"):
        for kind in ("numeric", "dated event", "who wins"):
            rows = [sum(s.means) for s in m.items if s.family == family and _kind(s.topic) == kind]
            if rows:
                out[f"{family}/{kind}"] = {
                    "n": len(rows),
                    "mean_sum": float(np.mean(rows)),
                    "min_sum": float(np.min(rows)),
                    "max_sum": float(np.max(rows)),
                    "share_above_1": float(np.mean(np.array(rows) > 1.0 + 1e-9)),
                }
    return out


def _items_csv(scores: Sequence[ModelScore], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "item",
                "family",
                "topic",
                "k",
                "single",
                "averaged",
                "l2",
                "null_single",
                "null_averaged",
                "null_p",
                "pooled_null_p",
                "means",
            ]
        )
        for m in scores:
            for s in m.items:
                w.writerow(
                    [
                        m.model,
                        s.item,
                        s.family,
                        s.topic,
                        s.k,
                        f"{s.single:.5f}",
                        f"{s.averaged:.5f}",
                        f"{s.l2:.5f}",
                        f"{s.null_single:.5f}",
                        f"{s.null_averaged:.5f}",
                        f"{s.null_p:.4f}",
                        f"{s.pooled_null_p:.4f}",
                        " ".join(f"{x:.3f}" for x in s.means),
                    ]
                )


def _answers_csv(
    bank: Bank, models: Sequence[ModelScore], samples: Sequence[Samples], path: Path
) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["proposition", *[m.model for m in models], "text"])
        for p in bank.propositions:
            row = [p.id]
            for s in samples:
                row.append(" ".join(f"{x:g}" for x in s.get(p.id, [])))
            w.writerow([*row, p.text])


TOPIC_LABELS = {
    "btc": "bitcoin price, 6 ranges",
    "eth": "ether price, 6 ranges",
    "spx": "S&P 500, 6 ranges",
    "gold": "gold price, 6 ranges",
    "brent": "Brent crude, 6 ranges",
    "us10y": "US 10-year yield, 6 ranges",
    "unemployment": "US unemployment, 6 ranges",
    "eurusd": "EUR/USD, 6 ranges",
    "us2028": "2028 US president, 5 + other",
    "france2027": "2027 French president, 5 + other",
    "f1_2027": "2027 F1 champion, 5 + other",
    "cwc2027": "2027 Cricket World Cup, 5 + other",
    "wwc2027": "2027 Women's World Cup, 5 + other",
    "party2028": "2028 winning party, 3",
}


def _partition_labels(bank: Bank) -> dict[str, str]:
    return {
        item.id: TOPIC_LABELS.get(item.topic, item.topic)
        for item in bank.items
        if item.family == "partition"
    }


def _ladder_charts(
    bank: Bank, models: Sequence[ModelScore], samples: Sequence[Samples], img: Path
) -> None:
    item = next(i for i in bank.items if i.id == "btc.ladder")
    xs = [float(p.split(".gt.")[1]) / 1000 for p in item.props]
    plots.ladder(
        models,
        samples,
        item.props,
        xs,
        "threshold ($k)",
        "P(bitcoin above $X on 30 June 2027): each dot one ask, line the mean",
        img / "ladder-btc.svg",
    )


def bin_points(bank: Bank, samples: Samples) -> list[tuple[float, float]]:
    """(implied by ladder, asked) for every range bin except the top one.

    The top bin is the same question as the top rung, so it is left out. The bottom
    bin, "at or below t1", is implied as 1 - P(above t1).
    """
    mean = {p: float(np.mean(v)) for p, v in samples.items() if v}
    out = []
    for item in bank.items:
        if item.family != "partition" or item.topic not in QUANTITY_KEYS:
            continue
        ladder = next(i for i in bank.items if i.id == f"{item.topic}.ladder")
        rungs = [mean.get(p) for p in ladder.props]
        for claim in item.claims[:-1]:
            asked = mean.get(claim.prop)
            world = claim.event.index(1)
            above_lo = 1.0 if world == 0 else rungs[world - 1]
            above_hi = rungs[world]
            if asked is None or above_lo is None or above_hi is None:
                continue
            out.append((above_lo - above_hi, asked))
    return out


def _row(m: ModelScore, metrics: Sequence[str]) -> list[str]:
    return [m.headline[x].fmt() if x in m.headline else "" for x in metrics]


def _markdown(
    summary: dict[str, Any], models: Sequence[ModelScore], base: Sequence[ModelScore]
) -> str:
    lines = ["# dutchbook results", ""]
    b = summary["bank"]
    lines += [
        f"Bank seed {b['seed']}: {b['propositions']} propositions, {b['items']} items "
        f"({', '.join(f'{k} {v}' for k, v in b['families'].items())}).",
        "",
        "All arbitrage numbers are cents of guaranteed profit per $1 of total stake, "
        "mean over the six families, with 95% topic-cluster bootstrap intervals.",
        "",
        "## Headline",
        "",
        "| forecaster | asked once | noise only, asked once | averaged | noise only, averaged "
        "| excess over noise | L2 distance | noise p |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    metrics = ("single", "null_single", "averaged", "null_averaged", "excess", "l2")
    for m in [*models, *base]:
        if not m.headline:
            continue
        lines.append(f"| {m.model} | " + " | ".join(_row(m, metrics)) + f" | {m.pooled_p:.3f} |")
    lines += ["", "## Calls", ""]
    lines += [
        "| model | calls | transport errors | replies | ok | refusal | no tag | multiple tags "
        "| bad number | tool events |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in models:
        c = m.calls or Calls()
        st = c.status
        lines.append(
            f"| {m.model} | {c.calls} | {c.transport_errors} | {c.replies} | {st['ok']} | "
            f"{st['refusal']} | {st['no_tag']} | {st['multiple_tags']} | {st['bad_number']} | "
            f"{c.tool_events} |"
        )
    lines += ["", "## Test-retest", ""]
    lines += [
        "| model | mean SD | median SD | identical every time | ICC | mean abs diff |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in [*models, *base]:
        r = m.retest
        lines.append(
            f"| {m.model} | {r.mean_sd * 100:.1f} | {r.median_sd * 100:.1f} | "
            f"{r.identical:.0%} | {r.icc:.2f} | {r.mean_abs_diff * 100:.1f} |"
        )
    lines += ["", "## By family", ""]
    for metric, title in (
        ("single", "asked once"),
        ("averaged", "averaged"),
        ("excess", "averaged minus noise only"),
    ):
        lines += [f"### {title}", ""]
        lines += ["| family | " + " | ".join(m.model for m in models) + " |"]
        lines += ["| :--- | " + " | ".join("---:" for _ in models) + " |"]
        for f in FAMILIES:
            cells = [m.by_family[f][metric].fmt() if f in m.by_family else "" for m in models]
            lines.append(f"| {f} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Items beyond noise (Benjamini-Hochberg, q = 0.05)", ""]
    lines += ["| family | " + " | ".join(m.model for m in models) + " |"]
    lines += ["| :--- | " + " | ".join("---:" for _ in models) + " |"]
    for f in [*FAMILIES, "all"]:
        cells = [
            f"{m.significant[f][0]} / {m.significant[f][1]}" if f in m.significant else ""
            for m in models
        ]
        lines.append(f"| {f} | " + " | ".join(cells) + " |")
    lines += ["", "## Which way the sums lean", ""]
    lines += [
        "Mean of the averaged answers summed over each complement pair or partition. "
        "Coherent answers sum to 1. Range in brackets; share of items summing above 1 after.",
        "",
    ]
    lines += ["| family / topic | " + " | ".join(m.model for m in models) + " |"]
    lines += ["| :--- | " + " | ".join("---:" for _ in models) + " |"]
    keys = sorted({k for m in models for k in sums(m)})
    for key in keys:
        cells = []
        for m in models:
            v = sums(m).get(key)
            cells.append(
                f"{v['mean_sum']:.2f} [{v['min_sum']:.2f}, {v['max_sum']:.2f}], "
                f"{v['share_above_1']:.0%}"
                if v
                else ""
            )
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired differences (cents per $1)", ""]
    lines += ["| comparison | difference |", "| :--- | ---: |"]
    for k, v in summary["paired_differences"].items():
        lines.append(f"| {k} | {v['value'] * 100:.1f} [{v['lo'] * 100:.1f}, {v['hi'] * 100:.1f}] |")
    if summary["markets"]:
        lines += ["", f"## Polymarket ({summary['markets']['snapshot']})", ""]
        lines += [
            "| shape | markets | mid: mean | mid: median | mid: share incoherent "
            "| executable books | executable: max |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for k, v in summary["markets"].items():
            if k == "snapshot":
                continue
            lines.append(
                f"| {k} | {v['n']} | {v['mid_mean'] * 100:.2f} | {v['mid_median'] * 100:.2f} | "
                f"{v['mid_share_incoherent']:.0%} | {v['executable_n']} | "
                f"{v['executable_max'] * 100:.2f} |"
            )
    return "\n".join(lines) + "\n"


def compare(a: Any, b: Any, *, tol: float = 1e-6, path: str = "") -> list[str]:
    """Every difference between two summaries, with numbers equal within `tol`.

    Used in CI to check that the committed results rebuild from the recordings on a
    clean machine, where the last digits of a solver's answer can differ.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}/{k}: missing" for k in sorted(set(a) ^ set(b))]
        for k in sorted(set(a) & set(b)):
            out += compare(a[k], b[k], tol=tol, path=f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: {len(a)} vs {len(b)} entries"]
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in compare(x, y, tol=tol, path=f"{path}/{i}")
        ]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return [] if abs(a - b) <= tol * max(1.0, abs(a), abs(b)) else [f"{path}: {a} vs {b}"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]
