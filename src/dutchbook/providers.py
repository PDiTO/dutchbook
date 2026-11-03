"""Model access through headless agent CLIs, set up as plain chat.

There are no API keys here. Each provider shells out to an agent CLI that is already
logged in, the way retro does, but with everything that would make it an agent turned
off: no tools, no file access, no web search, no MCP servers, no skills, no project or
user instruction files. Each call runs in a fresh empty directory, with stdin closed
and no session saved.

| backend | how tools and context are switched off |
| :--- | :--- |
| claude | `--safe-mode` (no CLAUDE.md, skills, plugins, hooks or MCP), `--tools ""`, |
|        | `--strict-mcp-config`, `--disable-slash-commands`, `--system-prompt` |
|        | replaces the default prompt |
| codex  | a private `CODEX_HOME` holding only the login and the model provider, so no |
|        | AGENTS.md, skills, memories, plugins or MCP; shell, code mode, apps, |
|        | browser, image and agent features disabled; `web_search="disabled"`; |
|        | `-s read-only`; `model_instructions_file` replaces the base prompt |
| cursor | `--mode ask --sandbox enabled` in an empty workspace (untested) |
| opencode | an inline agent with every tool denied (untested) |

What is left over is documented in the README: Claude Code still prefixes one line of
its own and an environment block (working directory, platform, date), and Codex
still lists a few tool names that fail closed when called. Every provider counts the
tool events in its output so a call that used a tool anyway would show up.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class Completed:
    returncode: int
    stdout: str
    stderr: str


Exec = Callable[[list[str], Path, float, dict[str, str] | None], Completed]


def _subprocess(
    args: list[str], cwd: Path, timeout: float, env: dict[str, str] | None
) -> Completed:  # pragma: no cover - exercised only against the real CLIs
    proc = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        stdin=subprocess.DEVNULL,
        env={**os.environ, **env} if env else None,
    )
    return Completed(proc.returncode, proc.stdout, proc.stderr)


class ProviderError(RuntimeError):
    """A transport failure: the CLI crashed, timed out or hit a rate limit.

    These are retried and never scored. A reply that arrives but breaks the format is
    not an error; it is a parse failure and is scored as one.
    """


@dataclass(frozen=True)
class Reply:
    text: str
    seconds: float
    tool_events: int = 0
    cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    raw: Any = None  # the CLI's output, as recorded
    argv: tuple[str, ...] = ()


class Provider(Protocol):
    @property
    def label(self) -> str: ...

    def complete(self, system: str, prompt: str) -> Reply: ...


@dataclass
class CLIProvider:
    """Base class for the agent CLI backends."""

    model: str
    effort: str = "medium"
    timeout: float = 300.0
    exec: Exec = field(default=_subprocess, repr=False)
    binary: str = ""
    backend: str = ""
    tested: bool = False

    @property
    def label(self) -> str:
        return self.model

    def complete(self, system: str, prompt: str) -> Reply:
        with tempfile.TemporaryDirectory(prefix="dutchbook-") as tmp:
            scratch = Path(tmp)
            args, env = self._args(system, prompt, scratch)
            start = time.monotonic()
            try:
                done = self.exec([self.binary, *args], scratch, self.timeout, env)
            except subprocess.TimeoutExpired as e:
                raise ProviderError(f"{self.backend} timed out after {self.timeout:.0f}s") from e
            except FileNotFoundError as e:
                raise ProviderError(f"{self.binary} is not installed") from e
            seconds = time.monotonic() - start
            if done.returncode != 0:
                tail = (done.stderr or done.stdout).strip().splitlines()[-3:]
                raise ProviderError(
                    f"{self.backend} exited {done.returncode}: {' | '.join(tail)[:300]}"
                )
            reply = self._parse(done, scratch)
        return Reply(
            reply.text,
            seconds,
            reply.tool_events,
            reply.cost_usd,
            reply.usage,
            reply.raw,
            tuple(_redact_argv(args, prompt, system)),
        )

    def _args(
        self, system: str, prompt: str, scratch: Path
    ) -> tuple[list[str], dict[str, str] | None]:
        raise NotImplementedError

    def _parse(self, done: Completed, scratch: Path) -> Reply:
        raise NotImplementedError


@dataclass
class ClaudeCLI(CLIProvider):
    binary: str = "claude"
    backend: str = "claude"
    tested: bool = True

    def _args(
        self, system: str, prompt: str, scratch: Path
    ) -> tuple[list[str], dict[str, str] | None]:
        return [
            "-p",
            prompt,
            "--model",
            self.model,
            "--effort",
            self.effort,
            "--system-prompt",
            system,
            "--output-format",
            "json",
            "--safe-mode",
            "--tools",
            "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-session-persistence",
        ], None

    def _parse(self, done: Completed, scratch: Path) -> Reply:
        out = _last_json_object(done.stdout)
        if out.get("is_error"):
            raise ProviderError(f"claude reported an error: {str(out.get('result'))[:200]}")
        usage = out.get("usage") if isinstance(out.get("usage"), dict) else None
        server_tools = (usage or {}).get("server_tool_use") or {}
        tool_events = (
            int(out.get("num_turns", 1))
            - 1
            + sum(int(v) for v in server_tools.values() if isinstance(v, int))
        )
        keep = ("result", "session_id", "num_turns", "total_cost_usd", "usage", "modelUsage")
        return Reply(
            text=str(out.get("result") or ""),
            seconds=0.0,
            tool_events=max(tool_events, 0),
            cost_usd=out.get("total_cost_usd"),
            usage=usage,
            raw={k: out[k] for k in keep if k in out},
        )


# Codex features that add tools or context. All are switched off for every call.
CODEX_DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "code_mode_host",
    "apps",
    "plugins",
    "multi_agent",
    "multi_agent_v2",
    "image_generation",
    "view_image",
    "browser_use",
    "browser_use_external",
    "in_app_browser",
    "computer_use",
    "goals",
    "sleep_tool",
    "skill_search",
    "tool_suggest",
    "hooks",
    "memories",
)

BASE_INSTRUCTIONS = "You answer the user's question directly in plain text. You have no tools."


@dataclass
class CodexCLI(CLIProvider):
    binary: str = "codex"
    backend: str = "codex"
    tested: bool = True
    home: Path | None = None  # private CODEX_HOME; built on first use if None

    def _args(
        self, system: str, prompt: str, scratch: Path
    ) -> tuple[list[str], dict[str, str] | None]:
        home = self.home or codex_home()
        (scratch / "instructions.md").write_text(f"{BASE_INSTRUCTIONS}\n\n{system}\n")
        args = [
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-rules",
            "-s",
            "read-only",
            "-C",
            str(scratch),
            "-m",
            self.model,
            "-c",
            f"model_reasoning_effort={self.effort}",
            "-c",
            f'model_instructions_file="{scratch / "instructions.md"}"',
            "-c",
            'web_search="disabled"',
            "-o",
            str(scratch / "last.txt"),
        ]
        for feature in CODEX_DISABLED_FEATURES:
            args += ["--disable", feature]
        return [*args, prompt], {"CODEX_HOME": str(home)}

    def _parse(self, done: Completed, scratch: Path) -> Reply:
        events = [e for e in (_maybe_json(line) for line in done.stdout.splitlines()) if e]
        texts, tool_events, usage = [], 0, None
        for event in events:
            if not isinstance(event, dict):
                continue
            item = event.get("item") if isinstance(event.get("item"), dict) else None
            if item and event.get("type") == "item.completed":
                kind = item.get("type")
                if kind == "agent_message":
                    texts.append(str(item.get("text", "")))
                elif kind not in ("reasoning", "error"):
                    tool_events += 1
            if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
                usage = event["usage"]
            if event.get("type") in ("turn.failed", "error"):
                raise ProviderError(f"codex: {json.dumps(event)[:300]}")
        last = scratch / "last.txt"
        text = last.read_text(encoding="utf-8") if last.exists() else "\n".join(texts)
        return Reply(text.strip(), 0.0, tool_events, None, usage, raw=events)


@dataclass
class CursorCLI(CLIProvider):
    binary: str = "cursor-agent"
    backend: str = "cursor"

    def _args(
        self, system: str, prompt: str, scratch: Path
    ) -> tuple[list[str], dict[str, str] | None]:
        # cursor-agent has no system prompt flag, so the system text leads the prompt.
        return [
            "-p",
            "--output-format",
            "json",
            "--mode",
            "ask",
            "--sandbox",
            "enabled",
            "--trust",
            "--workspace",
            str(scratch),
            "--model",
            self.model,
            f"{system}\n\n{prompt}",
        ], None

    def _parse(self, done: Completed, scratch: Path) -> Reply:
        out = _last_json_object(done.stdout)
        if out.get("is_error"):
            raise ProviderError(f"cursor reported an error: {str(out.get('result'))[:200]}")
        return Reply(str(out.get("result") or ""), 0.0, raw=out)


@dataclass
class OpenCodeCLI(CLIProvider):
    binary: str = "opencode"
    backend: str = "opencode"

    def _args(
        self, system: str, prompt: str, scratch: Path
    ) -> tuple[list[str], dict[str, str] | None]:
        config = {
            "agent": {
                "chat": {
                    "mode": "primary",
                    "prompt": system,
                    "permission": {"*": "deny"},
                    "tools": {"*": False},
                }
            },
            "mcp": {},
        }
        args = ["run", "--pure", "--format", "json", "--agent", "chat", "--dir", str(scratch)]
        return [*args, "-m", self.model, prompt], {"OPENCODE_CONFIG_CONTENT": json.dumps(config)}

    def _parse(self, done: Completed, scratch: Path) -> Reply:
        texts, tool_events, cost = [], 0, 0.0
        events = [e for e in (_maybe_json(line) for line in done.stdout.splitlines()) if e]
        for event in events:
            if not isinstance(event, dict):
                continue
            part = event.get("part") or {}
            if event.get("type") == "text":
                texts.append(str(part.get("text", "")))
            elif event.get("type") == "tool_use":
                tool_events += 1
            elif event.get("type") == "step_finish":
                cost += float(part.get("cost") or 0)
        return Reply("".join(texts).strip(), 0.0, tool_events, cost, raw=events)


BACKENDS: dict[str, type[CLIProvider]] = {
    "claude": ClaudeCLI,
    "codex": CodexCLI,
    "cursor": CursorCLI,
    "opencode": OpenCodeCLI,
}


def make(spec: str, *, effort: str = "medium", exec: Exec | None = None) -> CLIProvider:
    """`backend:model`, for example `claude:claude-opus-5-5` or `codex:gpt-6.1-sol`."""
    backend, sep, model = spec.partition(":")
    if not sep or not model:
        raise ValueError(f"expected backend:model, got {spec!r}")
    try:
        cls = BACKENDS[backend]
    except KeyError:
        raise ValueError(
            f"unknown backend {backend!r}; choose from {', '.join(BACKENDS)}"
        ) from None
    provider = cls(model=model, effort=effort)
    if exec is not None:
        provider.exec = exec
    return provider


def codex_home(root: Path | None = None, user_home: Path | None = None) -> Path:
    """A private CODEX_HOME that holds only the login and the model provider.

    Codex loads `$CODEX_HOME/AGENTS.md`, skills, memories and plugins into every
    session. Pointing it at a directory with none of those keeps the user's own
    instructions out of the prompt. The model provider block is copied from the
    user's config, because some setups route codex through a proxy.
    """
    user_home = user_home or Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    root = root or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    home = root / "dutchbook" / "codex-home"
    home.mkdir(parents=True, exist_ok=True)
    auth = home / "auth.json"
    if not auth.exists() and (user_home / "auth.json").exists():
        auth.symlink_to(user_home / "auth.json")
    lines = ['web_search = "disabled"', "", "[features]"]
    lines += [f"{f} = false" for f in CODEX_DISABLED_FEATURES]
    config_path = user_home / "config.toml"
    if config_path.exists():
        user = tomllib.loads(config_path.read_text(encoding="utf-8"))
        provider = user.get("model_provider")
        table = user.get("model_providers", {}).get(provider) if provider else None
        if isinstance(provider, str) and isinstance(table, dict):
            lines.insert(0, f"model_provider = {json.dumps(provider)}")
            lines += ["", f"[model_providers.{provider}]"]
            lines += [f"{k} = {json.dumps(v)}" for k, v in table.items()]
    (home / "config.toml").write_text("\n".join(lines) + "\n")
    return home


def available(backend: str) -> str | None:
    cls = BACKENDS[backend]
    return shutil.which(cls.binary)


def _redact_argv(args: list[str], prompt: str, system: str) -> list[str]:
    """The argv as recorded: prompt and system text are stored separately."""
    out = []
    for a in args:
        if a == prompt or a.endswith(prompt):
            out.append("<prompt>")
        elif a == system:
            out.append("<system>")
        else:
            out.append(a)
    return out


def _maybe_json(line: str) -> Any:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _last_json_object(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.strip().splitlines()):
        value = _maybe_json(line)
        if isinstance(value, dict):
            return value
    value = _maybe_json(stdout)
    if isinstance(value, dict):
        return value
    raise ProviderError("expected a JSON object on stdout")
