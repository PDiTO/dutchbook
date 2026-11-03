"""The whole offline pipeline: fake recordings in, results and charts out."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dutchbook.bank import build
from dutchbook.cli import main
from dutchbook.report import build_report
from dutchbook.run import run
from tests.fakes import FakeProvider, stable_unit
from tests.test_markets import SNAPSHOT


def incoherent(prompt: str, n: int) -> str:
    # Answers that ignore logic: a stable number per question, plus a little noise.
    p = min(max(stable_unit(prompt) + 0.02 * ((n % 3) - 1), 0.0), 1.0)
    return f"Reasoning.\n<probability>{p:.2f}</probability>"


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, object]]:
    root = tmp_path_factory.mktemp("report")
    bank = build()
    rec = root / "recordings"
    run(bank, FakeProvider("fake-a", incoherent), rec / "fake-a.jsonl", k=3, log=lambda s: None)
    run(bank, FakeProvider("fake-b"), rec / "fake-b.jsonl", k=3, log=lambda s: None)
    markets = root / "markets"
    markets.mkdir()
    (markets / "polymarket-2026-11-03.json").write_text(json.dumps(SNAPSHOT))
    summary = build_report(
        bank, rec, root / "results", root / "docs", markets, draws=10, boot=20, k=3, quiet=True
    )
    return root, summary


def test_writes_every_output(built: tuple[Path, dict[str, object]]) -> None:
    root, _ = built
    for name in ("summary.json", "items.csv", "answers.csv", "report.md"):
        assert (root / "results" / name).stat().st_size > 0
    for name in (
        "headline",
        "families",
        "noise-null",
        "retest",
        "partition-sums",
        "ladder-btc",
        "markets",
    ):
        svg = (root / "docs" / "img" / f"{name}.svg").read_text()
        assert svg.startswith("<?xml")
        assert "<dc:date>" not in svg  # no timestamp, so rebuilds are byte-identical


def test_summary_contents(built: tuple[Path, dict[str, object]]) -> None:
    root, summary = built
    data = json.loads((root / "results" / "summary.json").read_text())
    assert data == json.loads(json.dumps(summary))
    names = [m["model"] for m in data["models"]]
    assert names == ["fake-a", "fake-b"]
    assert [b["model"] for b in data["baselines"]] == ["coherent", "coherent + noise", "random"]
    coherent = data["baselines"][0]["headline"]["averaged"]["value"]
    assert coherent == pytest.approx(0.0, abs=1e-9)
    fake_a = data["models"][0]
    assert fake_a["headline"]["single"]["value"] > 0.01
    assert fake_a["calls"]["status"] == {"ok": 3 * data["bank"]["propositions"]}
    assert "fake-a - fake-b: single" in data["paired_differences"]
    assert data["markets"]["partition/all"]["n"] == 2
    md = (root / "results" / "report.md").read_text()
    assert "| fake-a |" in md
    assert "## Polymarket" in md


def test_report_is_deterministic(built: tuple[Path, dict[str, object]], tmp_path: Path) -> None:
    root, _ = built
    bank = build()
    build_report(
        bank,
        root / "recordings",
        tmp_path / "results",
        tmp_path / "docs",
        root / "markets",
        draws=10,
        boot=20,
        k=3,
        quiet=True,
    )
    for name in ("summary.json", "items.csv"):
        assert (tmp_path / "results" / name).read_text() == (root / "results" / name).read_text()
    a = (tmp_path / "docs" / "img" / "headline.svg").read_text()
    assert a == (root / "docs" / "img" / "headline.svg").read_text()


def test_cli_bank_and_report(built: tuple[Path, dict[str, object]], tmp_path: Path) -> None:
    root, _ = built
    bank_path = tmp_path / "bank.json"
    assert main(["bank", "--out", str(bank_path)]) == 0
    assert json.loads(bank_path.read_text())["seed"] == build().seed
    code = main(
        [
            "report",
            "--bank",
            str(bank_path),
            "--recordings",
            str(root / "recordings"),
            "--results",
            str(tmp_path / "results"),
            "--figures",
            str(tmp_path / "docs"),
            "--markets",
            str(root / "markets"),
            "--draws",
            "5",
            "--boot",
            "10",
        ]
    )
    assert code == 0
    assert (tmp_path / "results" / "summary.json").exists()


def test_compare(tmp_path: Path) -> None:
    from dutchbook.report import compare

    a = {"x": 1.0, "y": [1, 2, {"z": "s"}], "w": True}
    assert compare(a, {"x": 1.0 + 1e-9, "y": [1, 2, {"z": "s"}], "w": True}) == []
    diffs = compare(a, {"x": 1.1, "y": [1, 2], "v": 0})
    assert "/x: 1.0 vs 1.1" in diffs
    assert "/y: 3 vs 2 entries" in diffs
    assert "/v: missing" in diffs
    assert compare({"a": "s"}, {"a": "t"}) == ["/a: 's' vs 't'"]
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(a))
    assert main(["compare", str(pa), str(pb)]) == 0
    pb.write_text(json.dumps({**a, "x": 2.0}))
    assert main(["compare", str(pa), str(pb)]) == 1
