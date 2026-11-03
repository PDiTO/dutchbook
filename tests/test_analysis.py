from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dutchbook import baselines
from dutchbook.analysis import (
    Interval,
    load_samples,
    paired_difference,
    retest,
    score_item,
    score_model,
    significant,
)
from dutchbook.bank import build
from dutchbook.book import dutch_book
from dutchbook.items import Bank, Claim, Item, indicator
from dutchbook.run import Record


@pytest.fixture(scope="module")
def bank() -> Bank:
    return build()


def complement() -> Item:
    w = ("yes", "no")
    return Item(
        "c",
        "complement",
        "t",
        w,
        (Claim("a", indicator(w, ["yes"])), Claim("b", indicator(w, ["no"]))),
    )


def test_a_coherent_forecaster_scores_zero(bank: Bank) -> None:
    answers = baselines.coherent_forecaster(bank)
    for item in bank.items:
        assert dutch_book(item, [answers[p] for p in item.props]).profit == pytest.approx(
            0, abs=1e-9
        )
    m = score_model("coherent", bank, baselines.repeat(answers, 5), draws=20, boot=50)
    assert m.headline["single"].value == pytest.approx(0, abs=1e-9)
    assert m.headline["averaged"].value == pytest.approx(0, abs=1e-9)
    assert m.retest.identical == 1.0
    assert m.significant["all"][0] == 0


def test_steady_incoherence_beats_the_noise_null() -> None:
    # 0.6 and 0.6 every time, with a little noise: the overround can't be noise.
    samples = {"a": [0.6, 0.61, 0.6, 0.59, 0.6], "b": [0.6, 0.6, 0.62, 0.6, 0.58]}
    s = score_item(complement(), samples, draws=300, rng=np.random.default_rng(0), pooled_sd=0.01)
    assert s is not None
    assert s.averaged == pytest.approx(0.1, abs=0.002)
    assert s.null_averaged < 0.01
    assert s.null_p < 0.01
    assert s.excess > 0.09


def test_zero_variance_incoherence_is_still_significant() -> None:
    # Identical answers every time, from a model that is never noisy.
    samples = {"a": [0.7] * 5, "b": [0.4] * 5}
    s = score_item(complement(), samples, draws=100, rng=np.random.default_rng(0), pooled_sd=0.0)
    assert s is not None
    assert s.null_averaged == 0.0
    assert s.null_p == pytest.approx(1 / 101)


def test_pure_noise_is_not_flagged() -> None:
    # Coherent beliefs (0.3, 0.7) with large noise: single asks look incoherent,
    # averaging removes most of it, and the null explains what is left.
    rng = np.random.default_rng(1)
    ps = []
    for _ in range(40):
        samples = {
            "a": list(np.clip(0.3 + rng.normal(0, 0.05, 5), 0, 1)),
            "b": list(np.clip(0.7 + rng.normal(0, 0.05, 5), 0, 1)),
        }
        s = score_item(complement(), samples, draws=200, rng=rng, pooled_sd=0.05)
        assert s is not None
        assert s.single >= s.averaged - 1e-12
        ps.append(s.null_p)
    # Roughly uniform p-values, not piled up near zero.
    assert np.mean(np.array(ps) < 0.05) <= 0.1


def test_null_is_calibrated_on_small_samples() -> None:
    # Many noisy coherent forecasters, five asks each, on a five-bin partition: the
    # share of p-values under 0.05 should be near 5%, not inflated by thin variances.
    w = tuple(f"o{i}" for i in range(5))
    item = Item("p", "partition", "t", w, tuple(Claim(x, indicator(w, [x])) for x in w))
    truth = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    rng = np.random.default_rng(2)
    ps = []
    for _ in range(150):
        samples = {
            x: list(np.clip(t + rng.normal(0, 0.04, 5), 0, 1))
            for x, t in zip(w, truth, strict=False)
        }
        s = score_item(item, samples, draws=200, rng=rng, pooled_sd=0.04)
        assert s is not None
        ps.append(s.null_p)
    assert np.mean(np.array(ps) < 0.05) < 0.1


def test_items_with_too_few_answers_are_excluded(bank: Bank) -> None:
    samples = baselines.repeat(baselines.coherent_forecaster(bank), 5)
    first = bank.items[0]
    samples[first.props[0]] = [0.5]
    m = score_model("m", bank, samples, draws=5, boot=10)
    hit = [i.id for i in bank.items if first.props[0] in i.props]
    assert set(m.excluded) == set(hit)
    assert len(m.items) == len(bank.items) - len(hit)


def test_retest() -> None:
    r = retest({"a": [0.5, 0.5, 0.5], "b": [0.2, 0.4, 0.3], "c": [0.9]})
    assert r.props == 2
    assert r.identical == 0.5
    assert r.mean_sd == pytest.approx(0.05)
    assert r.mean_abs_diff == pytest.approx((0 + 0 + 0 + 0.2 + 0.1 + 0.1) / 6)
    assert retest({}).props == 0


def test_benjamini_hochberg() -> None:
    rng = np.random.default_rng(0)
    w = ("yes", "no")
    item = Item(
        "c",
        "complement",
        "t",
        w,
        (Claim("a", indicator(w, ["yes"])), Claim("b", indicator(w, ["no"]))),
    )
    strong = score_item(item, {"a": [0.7] * 5, "b": [0.5] * 5}, draws=50, rng=rng, pooled_sd=0)
    weak = score_item(item, {"a": [0.3] * 5, "b": [0.7] * 5}, draws=50, rng=rng, pooled_sd=0)
    assert strong is not None
    assert weak is not None
    assert significant([strong, weak]) == {"complement": (1, 2), "all": (1, 2)}


def test_paired_difference_with_itself_is_zero(bank: Bank) -> None:
    noise = baselines.noisy(0.03, np.random.default_rng(0))
    samples = baselines.repeat(baselines.coherent_forecaster(bank), 5, noise)
    m = score_model("m", bank, samples, draws=5, boot=20)
    d = paired_difference(m, m, "single", boot=20)
    assert d == Interval(0.0, 0.0, 0.0)


def test_random_forecaster_is_far_worse_than_noisy_coherent(bank: Bank) -> None:
    rng = np.random.default_rng(0)
    coherent = baselines.coherent_forecaster(bank)
    noisy = score_model(
        "noisy", bank, baselines.repeat(coherent, 5, baselines.noisy(0.02, rng)), draws=20, boot=50
    )
    rand = score_model("random", bank, baselines.random_forecaster(bank, 5), draws=20, boot=50)
    assert rand.headline["single"].value > 10 * noisy.headline["single"].value
    assert noisy.pooled_p > 0.01
    assert rand.retest.icc < 0.1


def test_load_samples_counts_every_status(tmp_path: Path) -> None:
    def rec(prop: str, rep: int, text: str | None, error: str | None = None) -> str:
        r = Record("m", prop, rep, 0, "s", "p", text, error, 1.0, cost_usd=0.01)
        return r.to_json()

    lines = [
        rec("a", 0, "<probability>0.2</probability>"),
        rec("a", 1, None, error="timeout"),
        rec("a", 1, "<probability>0.3</probability>"),
        rec("b", 0, "I can't predict that."),
        rec("b", 1, "<probability>30%</probability>"),
    ]
    path = tmp_path / "m.jsonl"
    path.write_text("\n".join(lines) + "\n")
    samples, calls = load_samples(path)
    assert samples == {"a": [0.2, 0.3]}
    assert calls.calls == 5
    assert calls.transport_errors == 1
    assert calls.replies == 4
    assert dict(calls.status) == {"ok": 2, "refusal": 1, "bad_number": 1}
    assert calls.cost_usd == pytest.approx(0.05)
    assert json.loads(lines[0])["prop"] == "a"


def test_meaning_round_trips_every_bank_id(bank: Bank) -> None:
    for p in bank.propositions:
        m = baselines.meaning(p.id)
        assert m.topic == p.topic
