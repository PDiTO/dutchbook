"""From recorded answers to scores, with noise separated from incoherence.

Each proposition was asked k times, each time in a fresh conversation. So for every
item there are three different questions to ask of the answers:

1. **Single-ask arbitrage.** Take one answer per proposition (repeat r of each, for
   r = 1..k) and price the Dutch book. This is what someone who asked each question
   once would face. It mixes real incoherence with sampling noise, because two noisy
   answers to logically linked questions will rarely line up exactly.
2. **Arbitrage after averaging.** Average the k answers per proposition first, then
   price the book. Averaging cancels most of the noise, so what is left is closer to
   the model's systematic incoherence.
3. **The noise null.** Would a forecaster with *coherent* beliefs and this model's
   own noise leave that much arbitrage after averaging? To find out, take the nearest
   coherent assignment to the averaged answers, add Gaussian noise with each
   proposition's own spread, average k draws, and price the book; repeat many times.
   The observed arbitrage is compared with that distribution, per item and pooled.
   The difference, observed minus null, is the excess: incoherence that noise does
   not explain.

Five answers give a poor estimate of one proposition's spread, and plugging the raw
estimate in makes the null too narrow (the usual z-instead-of-t mistake): on
simulated coherent forecasters it flagged pure noise far more often than 5% of the
time. So each proposition's variance is shrunk towards the model's pooled variance,
as in empirical-Bayes moderated t-tests, with the pooled value worth `PRIOR_DF`
degrees of freedom against the proposition's own k - 1. A second null uses the
pooled spread alone for every proposition, as a robustness check.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dutchbook.book import profit, project
from dutchbook.elicit import parse
from dutchbook.items import FAMILIES, Bank, Item
from dutchbook.run import answered, load_records

Samples = dict[str, list[float]]
Array = NDArray[np.float64]

PRIOR_DF = 4.0


@dataclass
class Calls:
    """What happened to every call for one model."""

    calls: int = 0  # recorded attempts
    transport_errors: int = 0
    replies: int = 0
    status: Counter[str] = field(default_factory=Counter)
    tool_events: int = 0
    seconds: float = 0.0
    cost_usd: float = 0.0


def load_samples(path: Path) -> tuple[Samples, Calls]:
    """Valid probabilities per proposition, in repeat order, plus call accounting."""
    records = load_records(path)
    calls = Calls(calls=len(records))
    for r in records:
        calls.seconds += r.seconds
        calls.cost_usd += r.cost_usd or 0.0
        if r.error is not None:
            calls.transport_errors += 1
    samples: Samples = {}
    for (prop, _rep), r in sorted(answered(records).items()):
        calls.replies += 1
        calls.tool_events += r.tool_events
        parsed = parse(r.text or "")
        calls.status[parsed.status] += 1
        if parsed.probability is not None:
            samples.setdefault(prop, []).append(parsed.probability)
    return samples, calls


@dataclass(frozen=True)
class ItemScore:
    item: str
    family: str
    topic: str
    k: int  # smallest number of valid answers among the item's propositions
    means: tuple[float, ...]
    single: float  # mean single-ask profit over aligned repeats
    averaged: float  # profit against the averaged answers
    l2: float  # Euclidean distance from the averaged answers to coherence
    null_single: float  # expected single-ask profit under the noise null
    null_averaged: float  # expected averaged profit under the noise null
    null_p: float  # P(null averaged profit >= observed)
    pooled_null_averaged: float
    pooled_null_p: float
    null_draws: tuple[float, ...] = field(repr=False, default=())
    pooled_null_draws: tuple[float, ...] = field(repr=False, default=())

    @property
    def excess(self) -> float:
        return self.averaged - self.null_averaged


def score_item(
    item: Item,
    samples: Samples,
    *,
    draws: int,
    rng: np.random.Generator,
    pooled_sd: float,
) -> ItemScore | None:
    """None when some proposition has fewer than two valid answers."""
    vals = [samples.get(p, []) for p in item.props]
    if min(len(v) for v in vals) < 2:
        return None
    k = min(len(v) for v in vals)
    means = [float(np.mean(v)) for v in vals]
    reps = max(len(v) for v in vals)
    single = float(np.mean([profit(item, [v[r % len(v)] for v in vals]) for r in range(reps)]))
    averaged = profit(item, means)
    proj = project(item, means)
    centre = np.array(proj.prices)
    sds = [_moderated_sd(v, pooled_sd) for v in vals]
    sizes = [len(v) for v in vals]

    null_avg = np.empty(draws)
    null_one = np.empty(draws)
    pooled = np.empty(draws)
    for b in range(draws):
        avg, one, pool = [], [], []
        for c, sd, n in zip(centre, sds, sizes, strict=True):
            x = np.clip(c + rng.normal(0.0, sd, size=n), 0.0, 1.0)
            avg.append(float(x.mean()))
            one.append(float(x[0]))
            g = np.clip(c + rng.normal(0.0, pooled_sd, size=n), 0.0, 1.0)
            pool.append(float(g.mean()))
        null_avg[b] = profit(item, avg)
        null_one[b] = profit(item, one)
        pooled[b] = profit(item, pool)
    return ItemScore(
        item=item.id,
        family=item.family,
        topic=item.topic,
        k=k,
        means=tuple(means),
        single=single,
        averaged=averaged,
        l2=proj.distance,
        null_single=float(null_one.mean()),
        null_averaged=float(null_avg.mean()),
        null_p=_p_value(null_avg, averaged),
        pooled_null_averaged=float(pooled.mean()),
        pooled_null_p=_p_value(pooled, averaged),
        null_draws=tuple(float(x) for x in null_avg),
        pooled_null_draws=tuple(float(x) for x in pooled),
    )


def _moderated_sd(values: Sequence[float], pooled_sd: float) -> float:
    """One proposition's spread, shrunk towards the model's pooled spread."""
    n = len(values)
    own = float(np.var(values, ddof=1)) if n > 1 else 0.0
    return math.sqrt((PRIOR_DF * pooled_sd**2 + (n - 1) * own) / (PRIOR_DF + n - 1))


def _p_value(null: Array, observed: float) -> float:
    if observed <= 1e-9:
        return 1.0
    return float((1 + np.sum(null >= observed - 1e-12)) / (len(null) + 1))


@dataclass(frozen=True)
class Retest:
    """Test-retest noise across the k repeats of each proposition."""

    props: int
    mean_sd: float
    median_sd: float
    pooled_sd: float  # sqrt of the mean within-proposition variance
    identical: float  # share of propositions answered identically every time
    icc: float  # between-proposition share of total variance
    mean_abs_diff: float  # mean |answer_r - answer_s| over pairs of repeats


def retest(samples: Samples) -> Retest:
    sds, variances, diffs, groups = [], [], [], []
    identical = 0
    for v in samples.values():
        if len(v) < 2:
            continue
        a = np.array(v)
        sds.append(float(a.std(ddof=1)))
        variances.append(float(a.var(ddof=1)))
        identical += int(np.all(a == a[0]))
        diffs += [abs(x - y) for i, x in enumerate(a) for y in a[i + 1 :]]
        groups.append(a)
    if not groups:
        return Retest(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    within = float(np.mean(variances))
    between = float(np.var([g.mean() for g in groups], ddof=1)) if len(groups) > 1 else 0.0
    k_bar = float(np.mean([len(g) for g in groups]))
    between_adj = max(between - within / k_bar, 0.0)
    icc = between_adj / (between_adj + within) if between_adj + within > 0 else 1.0
    return Retest(
        props=len(groups),
        mean_sd=float(np.mean(sds)),
        median_sd=float(np.median(sds)),
        pooled_sd=math.sqrt(within),
        identical=identical / len(groups),
        icc=icc,
        mean_abs_diff=float(np.mean(diffs)),
    )


@dataclass(frozen=True)
class Interval:
    value: float
    lo: float
    hi: float

    def fmt(self, scale: float = 100.0, digits: int = 1) -> str:
        return (
            f"{self.value * scale:.{digits}f} "
            f"[{self.lo * scale:.{digits}f}, {self.hi * scale:.{digits}f}]"
        )


@dataclass
class ModelScore:
    model: str
    items: list[ItemScore]
    excluded: list[str]
    retest: Retest
    calls: Calls | None = None
    headline: dict[str, Interval] = field(default_factory=dict)
    by_family: dict[str, dict[str, Interval]] = field(default_factory=dict)
    pooled_p: float = 1.0  # all items together vs the noise null
    pooled_p_alt: float = 1.0  # same, pooled-sd null
    significant: dict[str, tuple[int, int]] = field(default_factory=dict)  # family -> (sig, n)


METRICS = ("single", "averaged", "null_single", "null_averaged", "excess", "l2")


def score_model(
    model: str,
    bank: Bank,
    samples: Samples,
    *,
    draws: int = 500,
    boot: int = 2000,
    seed: int = 0,
    calls: Calls | None = None,
) -> ModelScore:
    rng = np.random.default_rng(seed)
    rt = retest(samples)
    items, excluded = [], []
    for item in bank.items:
        s = score_item(item, samples, draws=draws, rng=rng, pooled_sd=rt.pooled_sd)
        if s is None:
            excluded.append(item.id)
        else:
            items.append(s)
    out = ModelScore(model, items, excluded, rt, calls)
    if not items:
        return out
    out.headline, out.by_family = summarise(items, boot=boot, rng=rng)
    out.pooled_p = _pooled_p(items, alt=False)
    out.pooled_p_alt = _pooled_p(items, alt=True)
    out.significant = significant(items)
    return out


def _value(s: ItemScore, metric: str) -> float:
    return float(getattr(s, metric))


def summarise(
    items: Sequence[ItemScore], *, boot: int, rng: np.random.Generator
) -> tuple[dict[str, Interval], dict[str, dict[str, Interval]]]:
    """The headline (mean over families) and per-family means, with cluster CIs.

    The bootstrap resamples topics, not items: the items on one topic share
    propositions, so they are not independent. Each family counts once in the
    headline, so the 27 complement items don't outweigh the 6 conditional ones.
    """
    topics = sorted({s.topic for s in items})
    by_topic: dict[str, list[ItemScore]] = {t: [] for t in topics}
    for s in items:
        by_topic[s.topic].append(s)
    families = [f for f in FAMILIES if any(s.family == f for s in items)]

    def stats(sample: Sequence[ItemScore]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        fam: dict[str, dict[str, float]] = {}
        for f in families:
            rows = [s for s in sample if s.family == f]
            if rows:
                fam[f] = {m: float(np.mean([_value(s, m) for s in rows])) for m in METRICS}
        head = {m: float(np.mean([fam[f][m] for f in fam])) for m in METRICS}
        return head, fam

    head, fam = stats(items)
    boot_head: dict[str, list[float]] = {m: [] for m in METRICS}
    boot_fam: dict[str, dict[str, list[float]]] = {f: {m: [] for m in METRICS} for f in families}
    for _ in range(boot):
        pick = rng.choice(len(topics), size=len(topics), replace=True)
        sample = [s for i in pick for s in by_topic[topics[i]]]
        h, fm = stats(sample)
        for m in METRICS:
            boot_head[m].append(h[m])
            for f in families:
                if f in fm:
                    boot_fam[f][m].append(fm[f][m])

    def interval(value: float, draws: list[float]) -> Interval:
        lo, hi = np.percentile(draws, [2.5, 97.5]) if draws else (value, value)
        return Interval(value, float(lo), float(hi))

    headline = {m: interval(head[m], boot_head[m]) for m in METRICS}
    by_family = {f: {m: interval(fam[f][m], boot_fam[f][m]) for m in METRICS} for f in fam}
    return headline, by_family


def _pooled_p(items: Sequence[ItemScore], *, alt: bool) -> float:
    observed = sum(s.averaged for s in items)
    draws = np.sum([np.array(s.pooled_null_draws if alt else s.null_draws) for s in items], axis=0)
    return float((1 + np.sum(draws >= observed - 1e-12)) / (len(draws) + 1))


def significant(items: Sequence[ItemScore], q: float = 0.05) -> dict[str, tuple[int, int]]:
    """Items whose averaged arbitrage beats the noise null, Benjamini-Hochberg at q."""
    ps = sorted((s.null_p, i) for i, s in enumerate(items))
    m = len(ps)
    cutoff = 0
    for rank, (p, _) in enumerate(ps, 1):
        if p <= q * rank / m:
            cutoff = rank
    sig = {i for _, i in ps[:cutoff]}
    out: dict[str, tuple[int, int]] = {}
    for f in FAMILIES:
        idx = [i for i, s in enumerate(items) if s.family == f]
        if idx:
            out[f] = (sum(1 for i in idx if i in sig), len(idx))
    out["all"] = (len(sig), m)
    return out


def paired_difference(
    a: ModelScore, b: ModelScore, metric: str, *, boot: int = 2000, seed: int = 0
) -> Interval:
    """Headline difference a - b, with both resampled on the same topics."""
    rng = np.random.default_rng(seed)
    common = sorted({s.item for s in a.items} & {s.item for s in b.items})
    ia = [s for s in a.items if s.item in common]
    ib = [s for s in b.items if s.item in common]
    topics = sorted({s.topic for s in ia})

    def head(rows: Sequence[ItemScore]) -> float:
        fams = {s.family for s in rows}
        return float(
            np.mean([np.mean([_value(s, metric) for s in rows if s.family == f]) for f in fams])
        )

    value = head(ia) - head(ib)
    diffs = []
    for _ in range(boot):
        pick = [topics[i] for i in rng.choice(len(topics), size=len(topics), replace=True)]
        ra = [s for t in pick for s in ia if s.topic == t]
        rb = [s for t in pick for s in ib if s.topic == t]
        diffs.append(head(ra) - head(rb))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return Interval(value, float(lo), float(hi))
