import json
import stat
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

from noisefloor import client as cl
from noisefloor.client import (
    BackendError,
    ChatRequest,
    Client,
    Ledger,
    ModelCallRefusedError,
    billing_guard_findings,
    effective_caps,
    parse_stream_json,
)


class Handler(BaseHTTPRequestHandler):
    hits: ClassVar[list[dict[str, Any]]] = []
    reply: ClassVar[Any] = None

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.hits.append(body)
        payload = (
            Handler.reply
            if Handler.reply is not None
            else {
                "choices": [
                    {
                        "message": {"content": "hello", "tool_calls": [{"id": "1"}]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            }
        )
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def server() -> Iterator[str]:
    Handler.hits.clear()
    Handler.reply = None
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture
def no_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(cl.os.environ):
        if name.startswith(("ANTHROPIC_", "CLAUDE_CODE_USE_")):
            monkeypatch.delenv(name)
    monkeypatch.delenv("NOISEFLOOR_NO_MODELS", raising=False)
    monkeypatch.setattr(cl, "billing_guard_findings", lambda: [])


def make_client(tmp_path: Path, base: str = "", **caps: int) -> Client:
    defaults = {
        "claude_agent_runs_per_day": 6,
        "claude_single_calls_per_day": 2,
        "claude_review_runs_per_day": 1,
    }
    return Client(
        ledger=Ledger(tmp_path / "ledger.jsonl"),
        cache_dir=tmp_path / "cache",
        caps={**defaults, **caps},
        base_urls={"*": base} if base else {},
    )


def req(**kw: Any) -> ChatRequest:
    base = {
        "backend": "openai-local",
        "model": "qwen",
        "messages": ({"role": "user", "content": "hi"},),
        "seed": 1,
    }
    return ChatRequest(**{**base, **kw})


def test_local_call_cache_and_ledger(tmp_path: Path, server: str, no_guard: None) -> None:
    c = make_client(tmp_path, server)
    first = c.chat(req(stop=("X",), tools=({"type": "function"},)))
    assert first.text == "hello" and first.tool_calls == ({"id": "1"},) and not first.cached
    assert (first.input_tokens, first.output_tokens, first.finish_reason) == (7, 3, "stop")
    assert Handler.hits[0]["seed"] == 1 and Handler.hits[0]["stop"] == ["X"]
    second = c.chat(req(stop=("X",), tools=({"type": "function"},)))
    assert second.cached and second.text == "hello" and len(Handler.hits) == 1
    c.chat(req(seed=2))
    assert len(Handler.hits) == 2
    entries = c.ledger.entries()
    assert [e["cached"] for e in entries] == [False, True, False]
    assert all(e["cash_usd"] == 0 and e["kind"] == "local" for e in entries)


def test_local_errors(tmp_path: Path, server: str, no_guard: None) -> None:
    with pytest.raises(BackendError, match="no base URL"):
        make_client(tmp_path).chat(req())
    remote = Client(
        ledger=Ledger(tmp_path / "l"),
        cache_dir=tmp_path / "c",
        caps={},
        base_urls={"qwen": "https://example.com"},
    )
    with pytest.raises(ModelCallRefusedError, match="loopback"):
        remote.chat(req())
    Handler.reply = {"nope": 1}
    with pytest.raises(BackendError, match="malformed"):
        make_client(tmp_path, server).chat(req(seed=5))
    Handler.reply = b"not json"
    with pytest.raises(BackendError, match="local backend failed"):
        make_client(tmp_path, server).chat(req(seed=6))
    dead = make_client(tmp_path, "http://127.0.0.1:9")
    with pytest.raises(BackendError):
        dead.chat(req(seed=7), timeout=2)
    with pytest.raises(ValueError, match="unknown backend"):
        make_client(tmp_path, server).chat(req(backend="carrier-pigeon"))


def test_kill_switch(tmp_path: Path, server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOISEFLOOR_NO_MODELS", "1")
    with pytest.raises(ModelCallRefusedError, match="disabled"):
        make_client(tmp_path, server).chat(req())


def test_claude_refused_when_guard_tripped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOISEFLOOR_NO_MODELS", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://x")
    with pytest.raises(ModelCallRefusedError, match="billing guard"):
        make_client(tmp_path).chat(req(backend="claude", model="haiku"), kind="single_call")


def fake_claude(tmp_path: Path, events: list[dict[str, Any]]) -> str:
    script = tmp_path / "claude"
    lines = "\n".join(json.dumps(e) for e in events)
    script.write_text(f"#!/bin/sh\ncat > /dev/null\ncat <<'EOF'\n{lines}\nnot json\nEOF\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def test_claude_single_call_and_caps(tmp_path: Path, no_guard: None) -> None:
    events = [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "4"}]}},
        {
            "type": "result",
            "subtype": "success",
            "result": "4",
            "total_cost_usd": 0.01,
            "usage": {"input_tokens": 10, "output_tokens": 1},
        },
    ]
    c = make_client(tmp_path)
    c.claude_bin = fake_claude(tmp_path, events)
    messages = ({"role": "system", "content": "math"}, {"role": "user", "content": "2+2"})
    out = c.chat(req(backend="claude", model="haiku", messages=messages), kind="single_call")
    assert out.text == "4" and out.plan_equivalent_usd == 0.01 and out.input_tokens == 10
    entry = c.ledger.entries()[-1]
    assert entry["plan_equivalent_usd"] == 0.01 and entry["cash_usd"] == 0
    c.chat(req(backend="claude", model="haiku", messages=messages, seed=2), kind="single_call")
    with pytest.raises(ModelCallRefusedError, match="daily cap"):
        c.chat(req(backend="claude", model="haiku", seed=3), kind="single_call")
    with pytest.raises(ModelCallRefusedError, match="unknown Claude call kind"):
        c.chat(req(backend="claude", model="haiku", seed=4), kind="banter")
    cmd = c.claude_command("haiku", "sp")
    assert "--bare" not in cmd and "--restricted" in cmd and "--strict-mcp-config" in cmd
    with pytest.raises(ValueError):
        c.claude_command("gpt", "sp")


def test_claude_backend_failures(tmp_path: Path, no_guard: None) -> None:
    c = make_client(tmp_path)
    c.claude_bin = str(tmp_path / "missing-claude")
    with pytest.raises(BackendError, match="failed to run"):
        c.chat(req(backend="claude", model="haiku"), kind="single_call")


def test_parse_stream_json_variants() -> None:
    text = [
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "a"}, {"type": "tool_use"}]},
            }
        ),
        json.dumps({"type": "result", "result": "", "usage": {}}),
    ]
    assert parse_stream_json(text, 1.0).text == "a"
    with pytest.raises(BackendError, match="no result"):
        parse_stream_json(["", "garbage"], 1.0, 1, "boom")
    with pytest.raises(BackendError, match="error"):
        parse_stream_json([json.dumps({"type": "result", "is_error": True, "result": "x"})], 1)


def test_guard_and_caps_helpers(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"apiKeyHelper": "x"}))
    (tmp_path / "bad.json").write_text("{")
    found = billing_guard_findings(
        {"ANTHROPIC_API_KEY": "k", "CLAUDE_CODE_USE_VERTEX": "1"},
        [settings, tmp_path / "bad.json", tmp_path / "none.json"],
    )
    assert found == ["ANTHROPIC_API_KEY", "CLAUDE_CODE_USE_VERTEX", f"apiKeyHelper in {settings}"]
    assert isinstance(billing_guard_findings({}), list)
    budget = tmp_path / "budget.toml"
    budget.write_text(
        "cash_usd = 0\nclaude_agent_runs_per_day = 30\ndisk_floor_gb = 8\ncap_share = 0.2\n"
    )
    assert effective_caps(budget, {})["claude_agent_runs_per_day"] == 6
    assert effective_caps(budget, {"CAP_SHARE": "0.1"})["claude_agent_runs_per_day"] == 3
    assert effective_caps(budget, {})["disk_floor_gb"] == 8


def test_ledger_count_today(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "l.jsonl")
    assert ledger.entries() == [] and ledger.count_today("agent_run") == 0
    ledger.append({"kind": "agent_run"})
    ledger.append({"kind": "agent_run", "cached": True})
    ledger.append({"kind": "single_call"})
    assert ledger.count_today("agent_run") == 1
    assert ledger.count_today("agent_run", day="1999-01-01") == 0
