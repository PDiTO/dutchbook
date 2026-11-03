"""The CLI backends, against output shapes recorded from the real CLIs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from dutchbook.providers import (
    CODEX_DISABLED_FEATURES,
    ClaudeCLI,
    CodexCLI,
    Completed,
    CursorCLI,
    OpenCodeCLI,
    ProviderError,
    codex_home,
    make,
)

CLAUDE_OUT = json.dumps(
    {
        "type": "result",
        "is_error": False,
        "result": "Rough estimate.\n\n<probability>0.45</probability>",
        "num_turns": 1,
        "session_id": "s",
        "total_cost_usd": 0.0009,
        "usage": {
            "input_tokens": 2,
            "output_tokens": 50,
            "server_tool_use": {"web_search_requests": 0},
        },
        "duration_ms": 6000,
    }
)

CODEX_EVENTS = [
    {"thread_id": "t", "type": "thread.started"},
    {
        "item": {"id": "item_0", "message": "Code Mode is unavailable", "type": "error"},
        "type": "item.completed",
    },
    {"type": "turn.started"},
    {
        "item": {
            "id": "item_1",
            "text": "Toss-up.\n\n<probability>0.48</probability>",
            "type": "agent_message",
        },
        "type": "item.completed",
    },
    {"type": "turn.completed", "usage": {"input_tokens": 6411, "output_tokens": 127}},
]


class Recorder:
    def __init__(self, stdout: str, returncode: int = 0, write_last: str | None = None) -> None:
        self.stdout, self.returncode, self.write_last = stdout, returncode, write_last
        self.args: list[str] = []
        self.env: dict[str, str] | None = None

    def __call__(
        self, args: list[str], cwd: Path, timeout: float, env: dict[str, str] | None
    ) -> Completed:
        self.args, self.env = args, env
        if self.write_last is not None:
            (cwd / "last.txt").write_text(self.write_last)
        return Completed(self.returncode, self.stdout, "boom" if self.returncode else "")


def test_claude_is_pure_chat() -> None:
    rec = Recorder(CLAUDE_OUT)
    reply = ClaudeCLI(model="claude-haiku-5-5", exec=rec).complete("SYS", "PROMPT")
    assert reply.text.endswith("<probability>0.45</probability>")
    assert reply.cost_usd == 0.0009
    assert reply.tool_events == 0
    args = rec.args
    assert args[0] == "claude"
    assert args[args.index("--tools") + 1] == ""
    assert args[args.index("--system-prompt") + 1] == "SYS"
    for flag in ("--safe-mode", "--strict-mcp-config", "--disable-slash-commands"):
        assert flag in args
    assert "<prompt>" in reply.argv
    assert "<system>" in reply.argv
    assert "PROMPT" not in reply.argv


def test_claude_counts_tool_turns_and_errors() -> None:
    out = json.loads(CLAUDE_OUT) | {"num_turns": 3}
    reply = ClaudeCLI(model="m", exec=Recorder(json.dumps(out))).complete("s", "p")
    assert reply.tool_events == 2
    bad = json.loads(CLAUDE_OUT) | {"is_error": True, "result": "rate limited"}
    with pytest.raises(ProviderError, match="rate limited"):
        ClaudeCLI(model="m", exec=Recorder(json.dumps(bad))).complete("s", "p")
    with pytest.raises(ProviderError, match="exited 1"):
        ClaudeCLI(model="m", exec=Recorder("", returncode=1)).complete("s", "p")
    with pytest.raises(ProviderError, match="JSON object"):
        ClaudeCLI(model="m", exec=Recorder("not json")).complete("s", "p")


def test_codex_is_pure_chat(tmp_path: Path) -> None:
    stdout = "\n".join(json.dumps(e) for e in CODEX_EVENTS)
    rec = Recorder(stdout, write_last="Toss-up.\n\n<probability>0.48</probability>\n")
    reply = CodexCLI(model="gpt-6.1-sol", home=tmp_path, exec=rec).complete("SYS", "PROMPT")
    assert reply.text == "Toss-up.\n\n<probability>0.48</probability>"
    assert reply.tool_events == 0  # the code-mode error item is not a tool call
    assert reply.usage == {"input_tokens": 6411, "output_tokens": 127}
    assert rec.env == {"CODEX_HOME": str(tmp_path)}
    args = rec.args
    assert args[args.index("-s") + 1] == "read-only"
    assert 'web_search="disabled"' in args
    disabled = {args[i + 1] for i, a in enumerate(args) if a == "--disable"}
    assert disabled == set(CODEX_DISABLED_FEATURES)
    assert args[-1] == "PROMPT"


def test_codex_flags_tool_items_and_failed_turns(tmp_path: Path) -> None:
    events = [*CODEX_EVENTS[:3], {"item": {"type": "command_execution"}, "type": "item.completed"}]
    events.append(CODEX_EVENTS[3])
    stdout = "\n".join(json.dumps(e) for e in events)
    reply = CodexCLI(model="m", home=tmp_path, exec=Recorder(stdout)).complete("s", "p")
    assert reply.tool_events == 1
    assert reply.text.endswith("<probability>0.48</probability>")
    failed = json.dumps({"type": "turn.failed", "error": {"message": "usage limit"}})
    with pytest.raises(ProviderError, match="usage limit"):
        CodexCLI(model="m", home=tmp_path, exec=Recorder(failed)).complete("s", "p")


def test_codex_home_keeps_only_login_and_provider(tmp_path: Path) -> None:
    user = tmp_path / "user"
    user.mkdir()
    (user / "auth.json").write_text("{}")
    (user / "AGENTS.md").write_text("private instructions")
    (user / "config.toml").write_text(
        'model = "x"\nmodel_provider = "proxy"\n\n[model_providers.proxy]\n'
        'name = "Proxy"\nbase_url = "https://example.invalid/v1"\n\n'
        '[mcp_servers.browser]\ncommand = "browser"\n'
    )
    home = codex_home(root=tmp_path / "cache", user_home=user)
    assert (home / "auth.json").resolve() == (user / "auth.json").resolve()
    assert not (home / "AGENTS.md").exists()
    config = (home / "config.toml").read_text()
    assert 'model_provider = "proxy"' in config
    assert "[model_providers.proxy]" in config
    assert "mcp_servers" not in config
    assert "shell_tool = false" in config
    # Building it again is harmless.
    assert codex_home(root=tmp_path / "cache", user_home=user) == home


def test_cursor_and_opencode_args() -> None:
    rec = Recorder(json.dumps({"result": "<probability>0.3</probability>"}))
    reply = CursorCLI(model="m", exec=rec).complete("SYS", "PROMPT")
    assert reply.text == "<probability>0.3</probability>"
    assert rec.args[rec.args.index("--mode") + 1] == "ask"
    assert rec.args[-1] == "SYS\n\nPROMPT"
    events = [
        {"type": "text", "part": {"text": "<probability>0.2</probability>"}},
        {"type": "tool_use", "part": {}},
        {"type": "step_finish", "part": {"cost": 0.01}},
    ]
    rec = Recorder("\n".join(json.dumps(e) for e in events))
    reply = OpenCodeCLI(model="p/m", exec=rec).complete("SYS", "PROMPT")
    assert reply.text == "<probability>0.2</probability>"
    assert reply.tool_events == 1
    assert rec.env is not None
    config = json.loads(rec.env["OPENCODE_CONFIG_CONTENT"])
    assert config["agent"]["chat"]["permission"] == {"*": "deny"}


def test_timeouts_and_missing_binaries_are_transport_errors() -> None:
    def slow(args: list[str], cwd: Path, timeout: float, env: dict[str, str] | None) -> Completed:
        raise subprocess.TimeoutExpired(args, timeout)

    def missing(
        args: list[str], cwd: Path, timeout: float, env: dict[str, str] | None
    ) -> Completed:
        raise FileNotFoundError(args[0])

    with pytest.raises(ProviderError, match="timed out"):
        ClaudeCLI(model="m", exec=slow).complete("s", "p")
    with pytest.raises(ProviderError, match="not installed"):
        ClaudeCLI(model="m", exec=missing).complete("s", "p")


def test_make() -> None:
    p = make("claude:claude-opus-5-5", effort="low")
    assert isinstance(p, ClaudeCLI)
    assert p.label == "claude-opus-5-5"
    assert p.effort == "low"
    rec = Recorder(CLAUDE_OUT)
    assert make("claude:m", exec=rec).exec is rec
    with pytest.raises(ValueError, match="backend:model"):
        make("claude")
    with pytest.raises(ValueError, match="unknown backend"):
        make("gemini:x")
