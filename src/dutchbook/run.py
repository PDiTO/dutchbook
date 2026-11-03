"""Ask every proposition k times, record every call, and replay recordings.

Each call is one JSON line in `data/recordings/<model>.jsonl` with the system prompt,
the user prompt, the CLI arguments, the reply text and the CLI's own output. The
analysis reads only these files, so every number can be rebuilt offline. Runs are
resumable: a (proposition, repeat) pair with a recorded reply is not asked again.

Transport errors (a crashed CLI, a timeout, a rate limit) are recorded too, with the
error, and retried with backoff. They are never scored. A reply that arrives but
breaks the output format is scored as a parse failure, and is not retried, because
retrying would hide exactly the failures the benchmark counts.
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dutchbook.elicit import SYSTEM_PROMPT, build_prompt, parse
from dutchbook.items import Bank
from dutchbook.providers import Provider, ProviderError, Reply


@dataclass(frozen=True)
class Record:
    model: str
    prop: str
    rep: int
    attempt: int
    system: str
    prompt: str
    text: str | None
    error: str | None
    seconds: float
    tool_events: int = 0
    cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    argv: tuple[str, ...] = ()
    raw: Any = None
    ts: str = ""
    backend: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> Record:
        data = json.loads(line)
        data["argv"] = tuple(data.get("argv") or ())
        return cls(**data)


def recording_path(root: Path, model: str) -> Path:
    return root / f"{model}.jsonl"


def load_records(path: Path) -> list[Record]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [Record.from_json(line) for line in f if line.strip()]


def answered(records: Iterable[Record]) -> dict[tuple[str, int], Record]:
    """The reply for each (proposition, repeat): the last attempt that got one."""
    out: dict[tuple[str, int], Record] = {}
    for r in records:
        if r.error is None and r.text is not None:
            out[(r.prop, r.rep)] = r
    return out


@dataclass
class RunStats:
    asked: int = 0
    replies: int = 0
    errors: int = 0
    skipped: int = 0
    parse: dict[str, int] = field(default_factory=dict)


def run(
    bank: Bank,
    provider: Provider,
    out: Path,
    *,
    k: int,
    workers: int = 4,
    seed: int = 0,
    limit: int | None = None,
    retries: int = 3,
    backoff: float = 10.0,
    max_consecutive_errors: int = 12,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
    backend: str = "",
) -> RunStats:
    """Ask each proposition `k` times through `provider`, appending to `out`."""
    out.parent.mkdir(parents=True, exist_ok=True)
    done = answered(load_records(out))
    tasks = [(p.id, p.text, rep) for p in bank.propositions for rep in range(k)]
    stats = RunStats(skipped=sum(1 for t in tasks if (t[0], t[2]) in done))
    todo = [t for t in tasks if (t[0], t[2]) not in done]
    # Shuffle so repeats of one question are spread out and no family is asked at a
    # different time of day than another.
    random.Random(seed).shuffle(todo)
    if limit is not None:
        todo = todo[:limit]
    lock = threading.Lock()
    stop = threading.Event()
    consecutive = [0]

    def write(rec: Record) -> None:
        with lock, out.open("a", encoding="utf-8") as f:
            f.write(rec.to_json() + "\n")

    def ask(prop: str, text: str, rep: int) -> Record | None:
        prompt = build_prompt(text)
        for attempt in range(retries + 1):
            if stop.is_set():
                return None
            start = time.monotonic()
            reply: Reply | None = None
            error: str | None = None
            try:
                reply = provider.complete(SYSTEM_PROMPT, prompt)
            except ProviderError as e:
                error = str(e)
            rec = Record(
                model=provider.label,
                prop=prop,
                rep=rep,
                attempt=attempt,
                system=SYSTEM_PROMPT,
                prompt=prompt,
                text=reply.text if reply else None,
                error=error,
                seconds=round(reply.seconds if reply else time.monotonic() - start, 3),
                tool_events=reply.tool_events if reply else 0,
                cost_usd=reply.cost_usd if reply else None,
                usage=reply.usage if reply else None,
                argv=reply.argv if reply else (),
                raw=reply.raw if reply else None,
                ts=datetime.now(UTC).isoformat(timespec="seconds"),
                backend=backend,
            )
            write(rec)
            with lock:
                if error is None:
                    consecutive[0] = 0
                else:
                    consecutive[0] += 1
                    if consecutive[0] >= max_consecutive_errors:
                        stop.set()
            if error is None:
                return rec
            log(f"  {prop} rep {rep} attempt {attempt}: {error}")
            if attempt < retries:
                sleep(backoff * (3**attempt))
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(ask, *t) for t in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            stats.asked += 1
            if rec is None:
                stats.errors += 1
            else:
                stats.replies += 1
                status = parse(rec.text or "").status
                stats.parse[status] = stats.parse.get(status, 0) + 1
            if n % 25 == 0 or n == len(todo):
                log(f"{provider.label}: {n}/{len(todo)} done, {stats.errors} failed, {stats.parse}")
    if stop.is_set():
        log(f"{provider.label}: stopped after {max_consecutive_errors} errors in a row")
    return stats


@dataclass
class ReplayProvider:
    """Plays back recorded replies, in recorded order, for each prompt.

    Running the pipeline against a replay of a real recording reproduces it exactly,
    which is how the tests exercise the runner without a network.
    """

    label: str
    records: list[Record]
    _queues: dict[str, list[Record]] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        for r in sorted(self.records, key=lambda r: (r.prop, r.rep, r.attempt)):
            if r.text is not None:
                self._queues.setdefault(r.prompt, []).append(r)

    def complete(self, system: str, prompt: str) -> Reply:
        with self._lock:
            queue = self._queues.get(prompt)
            if not queue:
                raise ProviderError(f"replay: no recorded reply left for {prompt[:60]!r}")
            r = queue.pop(0)
        if r.system != system:
            raise ProviderError("replay: the system prompt differs from the recording")
        return Reply(r.text or "", r.seconds, r.tool_events, r.cost_usd, r.usage, r.raw, r.argv)
