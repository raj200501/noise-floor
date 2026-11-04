import asyncio
import json
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from fake_llm import write_fake_server
from noisefloor import inspect_harness as ih
from noisefloor.client import Client, Ledger
from noisefloor.models import wait_ready
from noisefloor.proxy import serve_proxy
from noisefloor.suite import generate


def call(fn: str, **args: object) -> SimpleNamespace:
    return SimpleNamespace(function=fn, arguments=args)


def submit(call_id: str, args: object) -> SimpleNamespace:
    return SimpleNamespace(function="submit", arguments=args, id=call_id)


def result(call_id: str, error: object = None) -> SimpleNamespace:
    return SimpleNamespace(role="tool", tool_call_id=call_id, error=error, tool_calls=None)


def test_submitted_answer_counts_only_accepted_submits() -> None:
    read = SimpleNamespace(function="read_file", arguments={"path": "/a"}, id="c0")
    accepted = [
        SimpleNamespace(role="assistant", tool_calls=[read]),
        result("c0"),
        SimpleNamespace(role="assistant", tool_calls=[submit("c1", {"answer": "42"})]),
        result("c1"),
    ]
    assert ih.submitted_answer(accepted) == ("42", False)
    assert ih.submitted_answer(accepted[:2]) == (None, False)
    # Inspect rejected a numeric answer: no answer, and the rejection is recorded.
    rejected = [
        SimpleNamespace(role="assistant", tool_calls=[submit("c2", {"answer": 42})]),
        result("c2", error=SimpleNamespace(message="answer: 42 is not of type 'string'")),
    ]
    assert ih.submitted_answer(rejected) == (None, True)
    assert ih.submitted_answer(accepted + rejected) == (None, True)
    assert ih.submitted_answer(rejected + accepted) == ("42", False)
    # A submit call that never ran (no tool result) is not an answer either.
    unrun = [SimpleNamespace(role="assistant", tool_calls=[submit("c3", {"answer": "7"})])]
    assert ih.submitted_answer(unrun) == (None, False)
    odd = [SimpleNamespace(role="assistant", tool_calls=[submit("c4", "raw")]), result("c4")]
    assert ih.submitted_answer(odd) == ("raw", False)
    assert ih.rejected_calls(accepted + rejected) == 1


@pytest.mark.parametrize(
    ("sample", "answer", "expected"),
    [
        (
            SimpleNamespace(
                error=SimpleNamespace(message="APIConnectionError: refused"), limit=None
            ),
            None,
            "backend_error",
        ),
        (
            SimpleNamespace(error=SimpleNamespace(message="KeyError: x"), limit=None),
            None,
            "harness_error",
        ),
        (SimpleNamespace(error=None, limit=SimpleNamespace(type="time")), None, "wall_clock"),
        (SimpleNamespace(error=None, limit=SimpleNamespace(type="message")), None, "step_limit"),
        (SimpleNamespace(error=None, limit=None), "7", "submitted"),
        (SimpleNamespace(error=None, limit=None), None, "stopped"),
        (
            SimpleNamespace(
                error=None, limit=None, output=SimpleNamespace(stop_reason="model_length")
            ),
            None,
            "context_limit",
        ),
    ],
)
def test_end_reason(sample: SimpleNamespace, answer: str | None, expected: str) -> None:
    assert ih.end_reason(sample, answer) == expected


def test_tool_defs_bind_to_env() -> None:
    task = generate("config", 900321, 1)
    env = task.start(max_calls=5)
    tools = ih.tool_defs(env)
    from inspect_ai.tool import ToolDef

    names = [ToolDef(t).name for t in tools]
    assert "submit" not in names and "read_file" in names and "list_dir" in names
    listing = asyncio.run(tools[names.index("list_dir")](path="/"))
    assert isinstance(listing, str) and env.calls == 1


@pytest.fixture
def upstream(tmp_path: Path) -> Iterator[str]:
    server = write_fake_server(tmp_path)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([str(server), "--port", str(port)])
    try:
        assert wait_ready(port, timeout=20)
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_run_end_to_end_through_proxy(
    tmp_path: Path, upstream: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NOISEFLOOR_NO_MODELS", raising=False)
    client = Client(
        ledger=Ledger(tmp_path / "ledger.jsonl"),
        cache_dir=tmp_path / "cache",
        caps={},
        base_urls={"*": upstream},
    )
    task = generate("pointer", 900321, 1)
    with serve_proxy(client) as url:
        settings = ih.InspectSettings(
            run_id="t1",
            base_url=url,
            temperature=0.0,
            seed=3,
            max_steps=4,
            time_limit_s=60,
            max_tokens=64,
            log_dir=tmp_path / "logs",
        )
        start = time.monotonic()
        out = ih.run(task, settings)
    assert out["end_reason"] == "submitted" and out["status"] == "ok"
    assert out["answer"] == "nope" and out["outcome"] == 0.0 and out["steps"] == 1
    assert out["input_tokens"] == 11 and out["output_tokens"] == 5
    assert time.monotonic() - start < 60
    assert (tmp_path / "ledger.jsonl").exists()


def test_main_writes_result(tmp_path: Path, upstream: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOISEFLOOR_NO_MODELS", raising=False)
    client = Client(
        ledger=Ledger(tmp_path / "l.jsonl"),
        cache_dir=tmp_path / "c",
        caps={},
        base_urls={"*": upstream},
    )
    with serve_proxy(client) as url:
        spec = {
            "run_id": "t2",
            "family": "config",
            "task_seed": 900321,
            "difficulty": 1,
            "base_url": url,
            "temperature": 0.0,
            "seed": 1,
            "max_steps": 3,
            "timeout_s": 60,
            "max_tokens": 64,
            "log_dir": str(tmp_path / "logs"),
        }
        (tmp_path / "spec.json").write_text(json.dumps(spec))
        assert ih.main([str(tmp_path / "spec.json"), str(tmp_path / "out.json")]) == 0
    assert json.loads((tmp_path / "out.json").read_text())["status"] == "ok"


def test_run_without_a_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task = generate("config", 900321, 1)
    settings = ih.InspectSettings(
        run_id="t3",
        base_url="http://127.0.0.1:9",
        temperature=0.0,
        seed=1,
        max_steps=2,
        time_limit_s=10,
        max_tokens=8,
        log_dir=tmp_path,
    )
    monkeypatch.setattr(ih, "inspect_eval", lambda *a, **k: [])
    assert ih.run(task, settings)["error"] == "no log"
    monkeypatch.setattr(
        ih, "inspect_eval", lambda *a, **k: [SimpleNamespace(samples=[], error="model refused")]
    )
    out = ih.run(task, settings)
    assert out["status"] == "infra_error" and out["error"] == "model refused"
