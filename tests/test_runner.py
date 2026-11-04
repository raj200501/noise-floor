import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from fake_llm import write_fake_server
from noisefloor import models, runner
from noisefloor.client import ChatRequest, ChatResponse, Client, Ledger
from noisefloor.models import LocalModel
from noisefloor.runner import RunSpec, batch_key, run_batch, run_minimal
from noisefloor.suite import AnswerSpec, generate


def spec(**kw: Any) -> RunSpec:
    base: dict[str, Any] = dict(
        run_id="r1",
        phase="test",
        family="config",
        task_seed=900123,
        difficulty=1,
        model="qwen2.5-0.5b",
        harness="minimal",
        temperature=0.0,
        seed=1,
        timeout_s=60,
        max_steps=6,
        threads=1,
        concurrency=1,
    )
    return RunSpec(**{**base, **kw})


@pytest.fixture
def fake_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    weights = tmp_path / "models" / "fake.gguf"
    weights.parent.mkdir()
    weights.write_bytes(b"fake weights")
    model = LocalModel(
        "fake",
        "fakefam",
        0.1,
        "x/y",
        "0" * 40,
        "fake.gguf",
        hashlib.sha256(b"fake weights").hexdigest(),
        "mit",
    )
    monkeypatch.setitem(models.MODELS, "fake", model)
    monkeypatch.setenv("NOISEFLOOR_MODELS", str(weights.parent))
    monkeypatch.setenv("NOISEFLOOR_LLAMA_SERVER", str(write_fake_server(tmp_path)))
    monkeypatch.delenv("NOISEFLOOR_NO_MODELS", raising=False)
    return "fake"


def test_runspec_validation_and_round_trip() -> None:
    s = spec(factors={"harness": "minimal"}, block="d1-am")
    assert RunSpec.from_json(json.loads(json.dumps(s.to_json()))) == s
    assert batch_key(s) == ("qwen2.5-0.5b", 1, 1)
    with pytest.raises(ValueError, match="harness"):
        spec(harness="claude-code")
    with pytest.raises(ValueError, match="model"):
        spec(model="gpt-9")


class OneShot(Client):
    """Client that answers every request with a fixed reply."""

    def __init__(self, reply: str, tmp: Path, fail: bool = False) -> None:
        super().__init__(ledger=Ledger(tmp / "l.jsonl"), cache_dir=tmp / "c", caps={})
        self.reply = reply
        self.fail = fail

    def chat(
        self, request: ChatRequest, *, kind: str = "local", timeout: float = 600.0
    ) -> ChatResponse:
        if self.fail:
            from noisefloor.client import BackendError

            raise BackendError("server down")
        return ChatResponse(text=self.reply, input_tokens=3, output_tokens=2, wall_s=0.01)


def test_run_minimal_correct_wrong_and_failing(tmp_path: Path) -> None:
    task = generate("config", 900123, 1)
    assert isinstance(task.expected, AnswerSpec)
    good = json.dumps({"tool": "submit", "args": {"answer": task.expected.expected}})
    out = run_minimal(spec(), OneShot(good, tmp_path))
    assert out["status"] == "ok" and out["end_reason"] == "submitted" and out["outcome"] == 1.0
    wrong = run_minimal(spec(), OneShot('{"tool": "submit", "args": {"answer": "?"}}', tmp_path))
    assert wrong["outcome"] == 0.0 and wrong["status"] == "ok"
    loop = run_minimal(spec(), OneShot('{"tool": "list_dir", "args": {"path": "/"}}', tmp_path))
    assert loop["end_reason"] == "step_limit" and loop["status"] == "timeout"
    talk = run_minimal(spec(max_steps=3), OneShot("thinking...", tmp_path))
    assert talk["status"] == "timeout" and talk["parse_errors"] == 3
    down = run_minimal(spec(), OneShot("", tmp_path, fail=True))
    assert down["status"] == "infra_error" and down["end_reason"] == "backend_error"


def test_run_batch_with_fake_server(tmp_path: Path, fake_model: str) -> None:
    specs = [
        spec(run_id="m1", model=fake_model),
        spec(run_id="i1", model=fake_model, harness="inspect", family="pointer"),
    ]
    batch = {"batch_id": "b1", "runs": [s.to_json() for s in specs]}
    assert run_batch(batch, tmp_path / "out") == 2
    rows = {
        r["run_id"]: r
        for r in map(json.loads, (tmp_path / "out" / "results.jsonl").read_text().splitlines())
    }
    assert rows["m1"]["status"] == "ok" and rows["m1"]["end_reason"] == "submitted"
    assert rows["i1"]["end_reason"] == "submitted", rows["i1"]
    assert rows["i1"]["status"] == "ok" and rows["i1"]["answer"] == "nope"
    assert rows["m1"]["outcome"] == 0.0 and rows["m1"]["batch_id"] == "b1"
    assert json.loads((tmp_path / "out" / "server_command.json").read_text())[0].endswith(
        "llama-server"
    )
    ledger = (tmp_path / "out" / "ledger.jsonl").read_text().splitlines()
    assert len(ledger) >= 2


def test_run_batch_rejects_mixed_and_bad_models(
    tmp_path: Path, fake_model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mixed = {
        "batch_id": "b",
        "runs": [
            spec(model=fake_model).to_json(),
            spec(run_id="r2", model=fake_model, threads=2).to_json(),
        ],
    }
    with pytest.raises(ValueError, match="share"):
        run_batch(mixed, tmp_path / "x")
    assert run_batch({"batch_id": "e", "runs": []}, tmp_path / "e") == 0
    bad = LocalModel("bad", "f", 0.1, "x/y", "0" * 40, "fake.gguf", "0" * 64, "mit")
    monkeypatch.setitem(models.MODELS, "bad", bad)
    # A model that fails its hash check records no run: the batch fails as a whole.
    assert run_batch({"batch_id": "c", "runs": [spec(model="bad").to_json()]}, tmp_path / "c") == 0
    note = json.loads((tmp_path / "c" / "aborted.json").read_text())
    assert "hash" in note["reason"] and note["drop"] == []
    assert not (tmp_path / "c" / "results.jsonl").exists()


def test_a_server_that_does_not_start_records_nothing(
    tmp_path: Path, fake_model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager

    @contextmanager
    def no_server(*args: object, **kwargs: object) -> Any:
        raise RuntimeError("llama-server did not become ready")
        yield  # pragma: no cover

    monkeypatch.setattr(runner, "serve", no_server)
    batch = {"batch_id": "s", "runs": [spec(model=fake_model).to_json()]}
    assert run_batch(batch, tmp_path / "s") == 0
    note = json.loads((tmp_path / "s" / "aborted.json").read_text())
    assert "did not start" in note["reason"] and "ready" in note["reason"]


def scripted(statuses: dict[str, str]) -> Any:
    def fake(s: RunSpec, client: object) -> dict[str, Any]:
        return {"status": statuses[s.run_id], "end_reason": "x", "outcome": 0.0}

    return fake


def test_three_infra_errors_in_a_row_abort_the_batch(
    tmp_path: Path, fake_model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    order = ["a", "b", "c", "d", "e", "f", "g"]
    statuses = dict(
        zip(
            order,
            ["infra_error", "ok", "infra_error", "infra_error", "infra_error", "ok", "ok"],
            strict=True,
        )
    )
    monkeypatch.setattr(runner, "run_minimal", scripted(statuses))
    batch = {"batch_id": "k", "runs": [spec(run_id=r, model=fake_model).to_json() for r in order]}
    assert run_batch(batch, tmp_path / "k") == 5  # f and g never start
    note = json.loads((tmp_path / "k" / "aborted.json").read_text())
    # The isolated infra_error (a) stays bankable; the streak is dropped.
    assert note["drop"] == ["c", "d", "e"] and "in a row" in note["reason"]


def test_a_failed_health_check_aborts_and_drops_the_failing_run(
    tmp_path: Path, fake_model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    statuses = {"a": "ok", "b": "timeout", "c": "ok"}
    monkeypatch.setattr(runner, "run_minimal", scripted(statuses))
    probes: list[int] = []

    def health(port: int, timeout: float) -> bool:
        probes.append(port)
        return False

    monkeypatch.setattr(runner, "wait_ready", health)
    batch = {"batch_id": "h", "runs": [spec(run_id=r, model=fake_model).to_json() for r in "abc"]}
    assert run_batch(batch, tmp_path / "h") == 2
    note = json.loads((tmp_path / "h" / "aborted.json").read_text())
    assert note["drop"] == ["b"] and "health" in note["reason"] and len(probes) == 1


def test_watch_drops_failures_that_finish_after_the_abort(tmp_path: Path) -> None:
    watch = runner._Watch(tmp_path, healthy=None)
    assert not watch.settle("a", "timeout")  # no probe without a server: a timeout is an outcome
    assert [watch.settle(r, "infra_error") for r in "bcd"] == [False, False, True]
    assert watch.settle("e", "ok") and watch.settle("f", "infra_error")
    note = json.loads((tmp_path / "aborted.json").read_text())
    assert note["drop"] == ["b", "c", "d", "f"]


def test_inspect_child_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    def boom(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired("x", 1)

    monkeypatch.setattr(runner.subprocess, "run", boom)
    out = runner.run_inspect(spec(harness="inspect"), "http://127.0.0.1:9", tmp_path)
    assert out["status"] == "infra_error" and "hard timeout" in out["error"]

    def quiet(*args: object, **kwargs: object) -> object:
        return subprocess.CompletedProcess([], 3, "", "crash")

    monkeypatch.setattr(runner.subprocess, "run", quiet)
    out2 = runner.run_inspect(spec(run_id="r9", harness="inspect"), "http://127.0.0.1:9", tmp_path)
    assert out2["status"] == "infra_error" and "exited 3" in out2["error"]


def test_harness_crash_is_recorded(
    tmp_path: Path, fake_model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def crash(*args: object, **kwargs: object) -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(runner, "run_minimal", crash)
    run_batch({"batch_id": "k", "runs": [spec(model=fake_model).to_json()]}, tmp_path / "k")
    row = json.loads((tmp_path / "k" / "results.jsonl").read_text())
    assert row["status"] == "infra_error" and "boom" in row["error"]


def test_main_usage(tmp_path: Path, fake_model: str) -> None:
    assert runner.main([]) == 2
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps({"batch_id": "z", "runs": [spec(model=fake_model).to_json()]}))
    assert runner.main([str(batch), str(tmp_path / "zout")]) == 0


def test_host_readings_name_load_memory_and_the_server() -> None:
    import os

    here = runner.host_readings(os.getpid())
    assert len(here["loadavg"]) == 3 and here["mem_available_mb"] > 0
    assert here["server_rss_mb"] > 0 and here["server_peak_rss_mb"] >= here["server_rss_mb"]
    assert "server_rss_mb" not in runner.host_readings()
    assert runner._proc_kb(Path("/nonexistent")) == {}


def test_run_records_carry_host_readings(tmp_path: Path, fake_model: str) -> None:
    batch = {"batch_id": "hr", "runs": [spec(model=fake_model).to_json()]}
    assert run_batch(batch, tmp_path / "hr") == 1
    row = json.loads((tmp_path / "hr" / "results.jsonl").read_text())
    assert row["host"]["server_peak_rss_mb"] >= 0 and len(row["host"]["loadavg"]) == 3
