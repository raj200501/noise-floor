"""Both local harnesses classify the same events the same way (D-021).

A scripted OpenAI-compatible server stands in for llama-server: a reply can be delayed
past the episode's wall clock, or be llama-server's context-size error. minimal-agent
and Inspect's basic agent must then report the same end reason and status.
"""

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from noisefloor import inspect_harness as ih
from noisefloor.client import Client, Ledger
from noisefloor.proxy import serve_proxy
from noisefloor.runner import RunSpec, run_minimal
from noisefloor.suite import generate

CONTEXT_ERROR = {
    "error": {
        "code": 400,
        "message": "request (40000 tokens) exceeds the available context size (32768 tokens)",
        "type": "exceed_context_size_error",
    }
}
LIST_ROOT = {"tool": "list_dir", "args": {"path": "/"}}
NUMERIC_SEED = 950000  # config d1 with a numeric answer


@contextmanager
def scripted(replies: list[dict[str, Any]]) -> Iterator[str]:
    """Serve ``replies`` in order (the last repeats); yields the base URL."""
    state = {"i": 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: Any) -> None:
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            self._send(200, {"object": "list", "data": [{"id": "local", "object": "model"}]})

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            with lock:
                reply = replies[min(state["i"], len(replies) - 1)]
                state["i"] += 1
            time.sleep(reply.get("delay", 0))
            if "status" in reply:
                self._send(reply["status"], reply["body"])
                return
            message: dict[str, Any] = {"role": "assistant", "content": reply.get("content", "")}
            calls = reply.get("tool_calls")
            if calls:
                message["tool_calls"] = [
                    {
                        "id": f"call{state['i']}_{n}",
                        "type": "function",
                        "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])},
                    }
                    for n, c in enumerate(calls)
                ]
            finish = "tool_calls" if calls else "stop"
            choice = {"index": 0, "message": message, "finish_reason": finish}
            self._send(
                200,
                {
                    "id": "x",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "local",
                    "choices": [choice],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                },
            )

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def client_for(url: str, tmp_path: Path, name: str) -> Client:
    return Client(
        ledger=Ledger(tmp_path / f"{name}.jsonl"),
        cache_dir=tmp_path / f"{name}-cache",
        caps={},
        base_urls={"*": url},
    )


def minimal_run(url: str, tmp_path: Path, timeout: float, seed: int = 950001) -> dict[str, Any]:
    spec = RunSpec(
        run_id="parity-min",
        phase="test",
        family="config",
        task_seed=seed,
        difficulty=1,
        model="qwen3-4b",
        harness="minimal",
        temperature=0.0,
        seed=1,
        timeout_s=timeout,
        max_steps=15,
        threads=1,
        concurrency=1,
    )
    return run_minimal(spec, client_for(url, tmp_path, "min"))


def inspect_run(url: str, tmp_path: Path, timeout: float, seed: int = 950001) -> dict[str, Any]:
    with serve_proxy(client_for(url, tmp_path, "insp")) as proxy:
        settings = ih.InspectSettings(
            run_id="parity-insp",
            base_url=proxy,
            temperature=0.0,
            seed=1,
            max_steps=15,
            time_limit_s=timeout,
            max_tokens=64,
            log_dir=tmp_path / "inspect-logs",
        )
        return ih.run(generate("config", seed, 1), settings)


@pytest.fixture(autouse=True)
def models_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOISEFLOOR_NO_MODELS", raising=False)


def test_wall_clock_during_a_model_call_is_a_timeout_in_both_harnesses(tmp_path: Path) -> None:
    minimal = [{"content": json.dumps(LIST_ROOT)}, {"content": json.dumps(LIST_ROOT), "delay": 5}]
    with scripted(minimal) as url:
        out = minimal_run(url, tmp_path, timeout=2.0)
    assert (out["end_reason"], out["status"]) == ("wall_clock", "timeout"), out
    tool = {"name": "list_dir", "arguments": {"path": "/"}}
    inspect = [{"tool_calls": [tool]}, {"tool_calls": [tool], "delay": 5}]
    with scripted(inspect) as url:
        out = inspect_run(url, tmp_path, timeout=2.0)
    assert (out["end_reason"], out["status"]) == ("wall_clock", "timeout"), out


def test_context_overflow_is_a_context_limit_in_both_harnesses(tmp_path: Path) -> None:
    overflow = {"status": 400, "body": CONTEXT_ERROR}
    with scripted([{"content": json.dumps(LIST_ROOT)}, overflow]) as url:
        out = minimal_run(url, tmp_path, timeout=30.0)
    assert (out["end_reason"], out["status"]) == ("context_limit", "timeout"), out
    tool = {"name": "list_dir", "arguments": {"path": "/"}}
    with scripted([{"tool_calls": [tool]}, overflow]) as url:
        out = inspect_run(url, tmp_path, timeout=30.0)
    assert (out["end_reason"], out["status"]) == ("context_limit", "timeout"), out


def test_other_server_errors_stay_backend_errors(tmp_path: Path) -> None:
    broken = {"status": 500, "body": {"error": {"message": "boom", "type": "server_error"}}}
    with scripted([broken]) as url:
        out = minimal_run(url, tmp_path, timeout=30.0)
    assert (out["end_reason"], out["status"]) == ("backend_error", "infra_error"), out


def test_numeric_submit_is_accepted_by_minimal_and_rejected_by_inspect(tmp_path: Path) -> None:
    """The documented harness difference: Inspect's submit takes only a string."""
    task = generate("config", NUMERIC_SEED, 1)
    number = int(task.expected.expected)
    reply = {"content": json.dumps({"tool": "submit", "args": {"answer": number}})}
    with scripted([reply]) as url:
        out = minimal_run(url, tmp_path, timeout=30.0, seed=NUMERIC_SEED)
    assert out["end_reason"] == "submitted" and out["outcome"] == 1.0
    tool = {"name": "submit", "arguments": {"answer": number}}
    with scripted([{"tool_calls": [tool]}]) as url:
        out = inspect_run(url, tmp_path, timeout=30.0, seed=NUMERIC_SEED)
    assert out["end_reason"] == "stopped" and out["submit_rejected"] is True, out
    assert out["answer"] is None and out["outcome"] == 0.0 and out["rejected_calls"] == 1
