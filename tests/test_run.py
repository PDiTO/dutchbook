from __future__ import annotations

from pathlib import Path

import pytest

from dutchbook.bank import build
from dutchbook.elicit import SYSTEM_PROMPT
from dutchbook.items import Bank
from dutchbook.run import ReplayProvider, answered, load_records, recording_path, run
from tests.fakes import FakeProvider


@pytest.fixture
def small_bank() -> Bank:
    full = build()
    items = tuple(i for i in full.items if i.topic == "btc")
    props = {p for i in items for p in i.props}
    return Bank(full.seed, full.as_of, tuple(p for p in full.propositions if p.id in props), items)


def test_records_every_call(small_bank: Bank, tmp_path: Path) -> None:
    out = tmp_path / "fake.jsonl"
    stats = run(small_bank, FakeProvider(), out, k=3, workers=2, log=lambda s: None)
    records = load_records(out)
    assert len(records) == 3 * len(small_bank.propositions)
    assert stats.replies == len(records)
    assert stats.parse == {"ok": len(records)}
    r = records[0]
    assert r.system == SYSTEM_PROMPT
    assert r.prompt.startswith("Question: ")
    assert r.argv == ("fake", "<prompt>")
    assert {(r.prop, r.rep) for r in records} == {
        (p.id, rep) for p in small_bank.propositions for rep in range(3)
    }


def test_resumes_without_asking_again(small_bank: Bank, tmp_path: Path) -> None:
    out = tmp_path / "fake.jsonl"
    run(small_bank, FakeProvider(), out, k=2, limit=5, log=lambda s: None)
    assert len(load_records(out)) == 5
    fake = FakeProvider()
    stats = run(small_bank, fake, out, k=2, log=lambda s: None)
    assert stats.skipped == 5
    assert fake.calls == 2 * len(small_bank.propositions) - 5
    assert len(answered(load_records(out))) == 2 * len(small_bank.propositions)


def test_transport_errors_are_recorded_and_retried(small_bank: Bank, tmp_path: Path) -> None:
    out = tmp_path / "fake.jsonl"
    waits: list[float] = []
    stats = run(
        small_bank,
        FakeProvider(fail_first=2),
        out,
        k=1,
        workers=1,
        sleep=waits.append,
        log=lambda s: None,
    )
    records = load_records(out)
    errors = [r for r in records if r.error]
    assert len(errors) == 2
    assert all(r.text is None for r in errors)
    assert stats.errors == 0
    assert waits == [10.0, 30.0]
    assert len(answered(records)) == len(small_bank.propositions)


def test_stops_after_too_many_errors(small_bank: Bank, tmp_path: Path) -> None:
    out = tmp_path / "fake.jsonl"
    logs: list[str] = []
    stats = run(
        small_bank,
        FakeProvider(fail_first=10_000),
        out,
        k=1,
        workers=1,
        retries=1,
        max_consecutive_errors=3,
        sleep=lambda s: None,
        log=logs.append,
    )
    assert stats.replies == 0
    assert len(load_records(out)) == 3
    assert any("stopped" in line for line in logs)


def test_parse_failures_are_scored_not_retried(small_bank: Bank, tmp_path: Path) -> None:
    out = tmp_path / "fake.jsonl"
    fake = FakeProvider(answer=lambda prompt, n: "I'd say about 40%.")
    stats = run(small_bank, fake, out, k=1, log=lambda s: None)
    assert stats.parse == {"no_tag": len(small_bank.propositions)}
    assert fake.calls == len(small_bank.propositions)


def test_replay_reproduces_a_recording(small_bank: Bank, tmp_path: Path) -> None:
    first = tmp_path / "a" / "fake.jsonl"
    fake = FakeProvider(answer=lambda prompt, n: f"<probability>0.{n + 1}</probability>")
    run(small_bank, fake, first, k=3, workers=3, log=lambda s: None)
    replay = ReplayProvider("fake-model", load_records(first))
    second = tmp_path / "b" / "fake.jsonl"
    run(small_bank, replay, second, k=3, workers=3, log=lambda s: None)

    def texts(path: Path) -> set[tuple[str, str | None]]:
        return {(r.prompt, r.text) for r in load_records(path)}

    assert texts(first) == texts(second)
    with pytest.raises(Exception, match="no recorded reply"):
        replay.complete(SYSTEM_PROMPT, "Question: unknown")


def test_replay_checks_the_system_prompt(small_bank: Bank, tmp_path: Path) -> None:
    out = tmp_path / "fake.jsonl"
    run(small_bank, FakeProvider(), out, k=1, log=lambda s: None)
    rec = load_records(out)[0]
    with pytest.raises(Exception, match="system prompt"):
        ReplayProvider("x", [rec]).complete("another system prompt", rec.prompt)


def test_recording_path() -> None:
    assert recording_path(Path("data"), "m") == Path("data/m.jsonl")
