"""Reference forecasters, answering the same bank the models answer.

- **coherent**: a forecaster with one joint distribution over every topic. Each
  quantity gets a log-normal (or normal, for rates) centred on its reference value,
  each dated event a constant hazard, each contest Dirichlet shares, and the topics in
  a pair are independent. Every answer is read off that distribution, so it admits no
  Dutch book anywhere. It scores 0 by construction, and the tests check that it does.
- **noisy coherent**: the same answers plus independent noise on every ask, with the
  spread chosen to match a model's measured test-retest noise. This is what pure noise
  around coherent beliefs looks like through the full pipeline.
- **random**: every answer uniform on [0, 1], independently.

The coherent forecaster needs to know what each proposition means, which it reads from
the proposition ids the generator writes (`btc.gt.80000`, `moon.by.2029-12-31`).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from dutchbook.bank import QUANTITIES
from dutchbook.items import Bank

Samples = dict[str, list[float]]


@dataclass(frozen=True)
class Meaning:
    """What one proposition says, decoded from its id."""

    topic: str
    kind: str  # gt, le, bin, by, notby, win, notwin, other, and, or, given
    args: tuple[str, ...]


def meaning(pid: str) -> Meaning:
    if pid.startswith("pair."):
        _, ta, tb, kind = pid.split(".", 3)
        return Meaning(f"pair.{ta}.{tb}", kind, (ta, tb))
    topic, rest = pid.split(".", 1)
    if rest == "other":
        return Meaning(topic, "other", ())
    kind, arg = rest.split(".", 1)
    if kind == "gt" and arg.endswith(".alt"):
        arg = arg.removesuffix(".alt")
    if kind == "bin":
        lo, hi = arg.split("_to_")
        return Meaning(topic, kind, (lo, hi))
    return Meaning(topic, kind, (arg,))


AS_OF_DATE = date(2026, 11, 3)


def coherent_forecaster(bank: Bank, seed: int = 0) -> dict[str, float]:
    """One coherent answer per proposition."""
    rng = np.random.default_rng(seed)
    quantities = {q.key: q for q in QUANTITIES}
    vol = {k: float(rng.uniform(0.1, 0.5)) for k in quantities}
    hazard: dict[str, float] = {}
    shares: dict[str, dict[str, float]] = {}
    out: dict[str, float] = {}

    def p_above(topic: str, x: float) -> float:
        q = quantities[topic]
        if q.additive:
            return float(norm.sf(x, loc=q.reference, scale=vol[topic] * 2))
        return float(norm.sf(math.log(x / q.reference) / vol[topic]))

    def p_by(topic: str, iso: str) -> float:
        h = hazard.setdefault(topic, float(rng.uniform(0.02, 0.6)))
        years = (date.fromisoformat(iso) - AS_OF_DATE).days / 365.25
        return 1.0 - math.exp(-h * years)

    def contest_share(topic: str, slug: str) -> float:
        if topic not in shares:
            entrants = sorted(
                {
                    meaning(p.id).args[0]
                    for p in bank.propositions
                    if p.topic == topic and meaning(p.id).kind == "win"
                }
            )
            has_other = any(p.id == f"{topic}.other" for p in bank.propositions)
            names = [*entrants, "__other__"] if has_other else entrants
            draw = rng.dirichlet(np.ones(len(names)))
            shares[topic] = dict(zip(names, (float(x) for x in draw), strict=True))
        return shares[topic][slug]

    def base(pid: str) -> float:
        m = meaning(pid)
        if m.kind == "gt":
            return p_above(m.topic, float(m.args[0]))
        if m.kind == "le":
            return 1.0 - p_above(m.topic, float(m.args[0]))
        if m.kind == "bin":
            return p_above(m.topic, float(m.args[0])) - p_above(m.topic, float(m.args[1]))
        if m.kind == "by":
            return p_by(m.topic, m.args[0])
        if m.kind == "notby":
            return 1.0 - p_by(m.topic, m.args[0])
        if m.kind == "win":
            return contest_share(m.topic, m.args[0])
        if m.kind == "notwin":
            return 1.0 - contest_share(m.topic, m.args[0])
        if m.kind == "other":
            return contest_share(m.topic, "__other__")
        raise ValueError(pid)

    pairs: dict[str, tuple[str, str]] = {}
    for item in bank.items:
        if item.family == "conjunction":
            pairs[item.topic] = (item.claims[0].prop, item.claims[1].prop)
    for p in bank.propositions:
        m = meaning(p.id)
        if m.topic.startswith("pair."):
            continue
        out[p.id] = min(1.0, max(0.0, base(p.id)))
    for p in bank.propositions:
        m = meaning(p.id)
        if not m.topic.startswith("pair."):
            continue
        a_id, b_id = pairs[m.topic]
        a, b = out[a_id], out[b_id]
        out[p.id] = {"and": a * b, "or": a + b - a * b, "given": a}[m.kind]
    return out


def repeat(
    answers: dict[str, float], k: int, noise: Callable[[float], float] | None = None
) -> Samples:
    return {pid: [noise(v) if noise else v for _ in range(k)] for pid, v in answers.items()}


def noisy(sd: float, rng: np.random.Generator) -> Callable[[float], float]:
    """Gaussian noise on the log-odds scale, matched to `sd` near p = 0.5."""
    scale = 4.0 * sd  # d p / d logit = p (1 - p) = 1/4 at p = 0.5

    def draw(p: float) -> float:
        p = min(max(p, 1e-4), 1 - 1e-4)
        z = math.log(p / (1 - p)) + rng.normal(0.0, scale)
        return round(1.0 / (1.0 + math.exp(-z)), 2)

    return draw


def random_forecaster(bank: Bank, k: int, seed: int = 0) -> Samples:
    rng = np.random.default_rng(seed)
    draws: NDArray[np.float64] = rng.uniform(0, 1, size=(len(bank.propositions), k))
    return {
        p.id: [round(float(x), 2) for x in row]
        for p, row in zip(bank.propositions, draws, strict=True)
    }
