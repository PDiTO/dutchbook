"""A scripted provider for the tests. Nothing here touches the network."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from dutchbook.providers import ProviderError, Reply


def stable_unit(text: str) -> float:
    """A deterministic number in [0, 1) from a string."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) / 2**32


@dataclass
class FakeProvider:
    """Answers from a function of the prompt and the call number for that prompt."""

    label: str = "fake-model"
    answer: Callable[[str, int], str] = lambda prompt, n: (
        f"Thinking.\n<probability>{stable_unit(prompt):.2f}</probability>"
    )
    fail_first: int = 0  # raise a transport error on the first n calls overall
    calls: int = 0
    seen: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def complete(self, system: str, prompt: str) -> Reply:
        with self._lock:
            self.calls += 1
            if self.calls <= self.fail_first:
                raise ProviderError("fake: rate limited")
            n = self.seen.get(prompt, 0)
            self.seen[prompt] = n + 1
        return Reply(self.answer(prompt, n), 0.01, argv=("fake", "<prompt>"))
