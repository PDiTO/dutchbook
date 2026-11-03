"""dutchbook bank | run | probe | report | markets"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from dutchbook import bank as bank_mod
from dutchbook.elicit import SYSTEM_PROMPT, build_prompt, parse
from dutchbook.items import Bank
from dutchbook.markets import snapshot
from dutchbook.providers import make
from dutchbook.report import build_report, compare
from dutchbook.run import recording_path, run

DATA = Path("data")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dutchbook", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("bank", help="generate the propositions and items")
    p.add_argument("--seed", type=int, default=bank_mod.DEFAULT_SEED)
    p.add_argument("--out", type=Path, default=DATA / "bank.json")

    p = sub.add_parser("run", help="ask every proposition k times and record the replies")
    p.add_argument("--model", required=True, help="backend:model, e.g. claude:claude-opus-5-5")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--effort", default="medium")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--bank", type=Path, default=DATA / "bank.json")
    p.add_argument("--out", type=Path, default=DATA / "recordings")
    p.add_argument("--seed", type=int, default=0, help="shuffles the order of calls")

    p = sub.add_parser("probe", help="ask one question and print the raw reply")
    p.add_argument("--model", required=True)
    p.add_argument("--effort", default="medium")
    p.add_argument("question", nargs="?", default="Will it snow in London on 25 December 2027?")

    p = sub.add_parser("report", help="rebuild results/ and docs/ charts from the recordings")
    p.add_argument("--bank", type=Path, default=DATA / "bank.json")
    p.add_argument("--recordings", type=Path, default=DATA / "recordings")
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--figures", type=Path, default=Path("docs"))
    p.add_argument("--markets", type=Path, default=DATA / "markets")
    p.add_argument("--draws", type=int, default=1000, help="noise-null draws per item")
    p.add_argument("--boot", type=int, default=2000, help="bootstrap draws")

    p = sub.add_parser("compare", help="check two summary.json files agree within a tolerance")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--tol", type=float, default=1e-6)

    p = sub.add_parser("markets", help="snapshot Polymarket partitions and ladders")
    p.add_argument("--out", type=Path, default=DATA / "markets")

    args = parser.parse_args(argv)
    if args.cmd == "bank":
        b = bank_mod.build(args.seed)
        b.save(args.out)
        fams = Counter(i.family for i in b.items)
        print(f"{len(b.propositions)} propositions, {len(b.items)} items -> {args.out}")
        print(", ".join(f"{k} {v}" for k, v in sorted(fams.items())))
        return 0
    if args.cmd == "run":
        b = Bank.load(args.bank)
        provider = make(args.model, effort=args.effort)
        out = recording_path(args.out, provider.label)
        n = len(b.propositions) * args.k
        print(f"{provider.label}: {len(b.propositions)} propositions x {args.k} = {n} calls")
        stats = run(
            b,
            provider,
            out,
            k=args.k,
            workers=args.workers,
            seed=args.seed,
            limit=args.limit,
            backend=provider.backend,
        )
        print(
            f"asked {stats.asked}, replies {stats.replies}, failed {stats.errors}, "
            f"already recorded {stats.skipped}; {stats.parse}"
        )
        return 0 if stats.errors == 0 else 1
    if args.cmd == "probe":
        provider = make(args.model, effort=args.effort)
        reply = provider.complete(SYSTEM_PROMPT, build_prompt(args.question))
        print(reply.text)
        print(f"--- {reply.seconds:.1f}s, tool events {reply.tool_events}, {parse(reply.text)}")
        return 0
    if args.cmd == "report":
        build_report(
            Bank.load(args.bank),
            args.recordings,
            args.results,
            args.figures,
            args.markets,
            draws=args.draws,
            boot=args.boot,
        )
        return 0
    if args.cmd == "compare":
        diffs = compare(
            json.loads(args.a.read_text()), json.loads(args.b.read_text()), tol=args.tol
        )
        for d in diffs[:50]:
            print(d)
        print(f"{len(diffs)} differences")
        return 1 if diffs else 0
    if args.cmd == "markets":
        snapshot(args.out)
        return 0
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
