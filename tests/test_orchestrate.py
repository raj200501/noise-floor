import json
from pathlib import Path
from typing import Any

import pytest

from noisefloor import orchestrate
from noisefloor.orchestrate import Batch, banked_ids, config_hash, execute, make_batches
from noisefloor.runner import RunSpec


def spec(run_id: str, model: str = "qwen2.5-0.5b", threads: int = 4) -> RunSpec:
    return RunSpec(
        run_id=run_id,
        phase="test",
        family="config",
        task_seed=900001,
        difficulty=1,
        model=model,
        harness="minimal",
        temperature=0.0,
        seed=1,
        timeout_s=60,
        max_steps=5,
        threads=threads,
        concurrency=1,
    )


def test_make_batches_keeps_order_and_splits_on_config_and_size() -> None:
    specs = [spec("a"), spec("b"), spec("c", threads=2), spec("d"), spec("e"), spec("f")]
    batches = make_batches(specs, "p", max_size=2)
    assert [[r.run_id for r in b.runs] for b in batches] == [["a", "b"], ["c"], ["d", "e"], ["f"]]
    assert all(b.batch_id.startswith(f"p-b{i:04d}-") for i, b in enumerate(batches))
    assert len({b.batch_id for b in batches}) == 4
    assert make_batches([], "p") == []
    with pytest.raises(ValueError, match="positive"):
        make_batches(specs, "p", max_size=0)


def fake_executor(log: list[str]) -> orchestrate.Executor:
    def run(batch: Batch, workspace: Path) -> list[dict[str, Any]]:
        log.append(batch.batch_id)
        (workspace / "out").mkdir(parents=True, exist_ok=True)
        (workspace / "out" / "ledger.jsonl").write_text('{"kind": "local"}\n')
        return [{**s.to_json(), "status": "ok", "outcome": 1.0} for s in batch.runs]

    return run


def test_execute_banks_with_provenance_and_resumes(tmp_path: Path) -> None:
    artifact = tmp_path / "art" / "runs.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    calls: list[str] = []
    specs = [spec("a"), spec("b", threads=2)]
    n = execute(
        specs,
        artifact=artifact,
        ledger=ledger,
        prefix="t",
        executor=fake_executor(calls),
        work_root=tmp_path / "work",
        log=lambda m: None,
    )
    assert n == 2 and [c.split("-")[1] for c in calls] == ["b0000", "b0001"]
    rows = [json.loads(ln) for ln in artifact.read_text().splitlines()]
    prov = rows[0]["provenance"]
    assert prov["model_sha256"] and prov["config_sha256"] == config_hash(specs[0])
    assert prov["git"]["commit"] and "runtime" in prov and rows[0]["batch_started_at"]
    assert len(ledger.read_text().splitlines()) == 2
    assert banked_ids(artifact) == {"a", "b"}
    again = execute(
        [*specs, spec("c")],
        artifact=artifact,
        ledger=ledger,
        prefix="t2",
        executor=fake_executor(calls),
        work_root=tmp_path / "work",
        log=lambda m: None,
    )
    assert again == 1 and banked_ids(artifact) == {"a", "b", "c"}
    assert banked_ids(tmp_path / "none.jsonl") == set()


def test_sandboxed_executor_leaves_missing_runs_unbanked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    def fake_run(
        cmd: list[str], workspace: Path, **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        out = Path(workspace) / "out"
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.jsonl").write_text(
            json.dumps({"run_id": "a", "status": "ok"}) + '\n{"run_id": "b", "sta'
        )
        assert "noisefloor.runner" in cmd and kwargs["env"]["NOISEFLOOR_MODELS"]
        return subprocess.CompletedProcess(cmd, 1, "", "killed")

    monkeypatch.setattr(orchestrate.sandbox, "run", fake_run)
    rows = orchestrate.sandboxed_executor()(Batch("x", (spec("a"), spec("b"))), tmp_path / "w")
    assert [r["run_id"] for r in rows] == ["a"]  # b's record was cut: b stays unbanked
    note = json.loads((tmp_path / "w" / "out" / "aborted.json").read_text())
    assert "exited 1" in note["reason"] and note["drop"] == []
    assert (tmp_path / "w" / "runner.stderr").read_text() == "killed"


def failing_executor(drop: list[str], skip: set[str]) -> orchestrate.Executor:
    def run(batch: Batch, workspace: Path) -> list[dict[str, Any]]:
        out = workspace / "out"
        out.mkdir(parents=True, exist_ok=True)
        (out / "ledger.jsonl").write_text('{"kind": "local"}\n{"kind": "local"}\n')
        if drop:
            note = {"reason": "3 infra_error runs in a row", "drop": drop, "at": "t"}
            (out / "aborted.json").write_text(json.dumps(note))
        rows = [
            {**s.to_json(), "status": "infra_error" if s.run_id in drop else "ok"}
            for s in batch.runs
            if s.run_id not in skip
        ]
        return rows

    return run


def test_a_failed_batch_banks_only_untouched_runs_and_raises(tmp_path: Path) -> None:
    artifact = tmp_path / "runs.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    messages: list[str] = []
    specs = [spec(r) for r in "abcde"]
    with pytest.raises(orchestrate.BatchFailedError, match="in a row") as info:
        execute(
            specs,
            artifact=artifact,
            ledger=ledger,
            prefix="f",
            executor=failing_executor(["b", "c", "d"], skip={"e"}),
            work_root=tmp_path / "work",
            log=messages.append,
        )
    assert info.value.banked == 1 and banked_ids(artifact) == {"a"}
    assert len(ledger.read_text().splitlines()) == 2  # usage of every call is kept
    assert "1 runs banked, 0 infra_error" in messages[0] and "4 runs not banked" in messages[1]
    assert not any("ok" in m.split() for m in messages)  # no outcome counts in the log
    # A runner that stops early without a note is a failed batch too.
    with pytest.raises(orchestrate.BatchFailedError, match="before every run"):
        execute(
            specs,
            artifact=artifact,
            ledger=ledger,
            prefix="g",
            executor=failing_executor([], skip={"e"}),
            work_root=tmp_path / "work",
            log=messages.append,
        )
    assert banked_ids(artifact) == {"a", "b", "c", "d"}


def test_execute_cuts_a_partial_record_before_appending(tmp_path: Path) -> None:
    artifact = tmp_path / "runs.jsonl"
    artifact.write_text(json.dumps({"run_id": "z"}) + '\n{"run_id": "a", "pro')
    messages: list[str] = []
    calls: list[str] = []
    n = execute(
        [spec("a")],
        artifact=artifact,
        ledger=tmp_path / "ledger.jsonl",
        prefix="p",
        executor=fake_executor(calls),
        work_root=tmp_path / "work",
        log=messages.append,
    )
    assert n == 1 and banked_ids(artifact) == {"z", "a"}
    assert "cut by an interrupted write" in messages[0]
    assert all(json.loads(ln) for ln in artifact.read_text().splitlines())


def test_salvage_banks_completed_runs_of_an_interrupted_batch(tmp_path: Path) -> None:
    artifact = tmp_path / "runs.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    old = tmp_path / "work" / "s-b0000-deadbeef" / "out"
    old.mkdir(parents=True)
    a = spec("a")
    (old / "results.jsonl").write_text(
        json.dumps({**a.to_json(), "status": "ok", "outcome": 1.0, "batch_id": "s-b0000-x"})
        + "\n"
        + json.dumps({"run_id": "stranger", "status": "ok"})
        + "\n"
    )
    (old / "ledger.jsonl").write_text('{"kind": "local"}\n')
    (old.parent / ".started").write_text("2026-11-05T03:00:00+00:00\n")
    dropped = tmp_path / "work" / "s-b0001-cafecafe" / "out"
    dropped.mkdir(parents=True)
    (dropped / "results.jsonl").write_text(json.dumps({**spec("b").to_json(), "status": "x"}))
    (dropped / "aborted.json").write_text(json.dumps({"reason": "r", "drop": ["b"], "at": "t"}))
    calls: list[str] = []
    n = execute(
        [a, spec("b")],
        artifact=artifact,
        ledger=ledger,
        prefix="s",
        executor=fake_executor(calls),
        work_root=tmp_path / "work",
        log=lambda m: None,
    )
    rows = {json.loads(ln)["run_id"]: json.loads(ln) for ln in artifact.read_text().splitlines()}
    assert n == 2 and set(rows) == {"a", "b"} and rows["a"]["salvaged"] is True
    assert "salvaged" not in rows["b"] and len(calls) == 1
    # A salvaged run keeps its batch's recorded start; a new batch records its own.
    assert rows["a"]["batch_started_at"] == "2026-11-05T03:00:00+00:00"
    old_ws = {"s-b0000-deadbeef", "s-b0001-cafecafe"}
    new_ws = [w for w in (tmp_path / "work").iterdir() if w.name not in old_ws]
    assert (
        len(new_ws) == 1
        and (new_ws[0] / ".started").read_text().strip() == (rows["b"]["batch_started_at"])
    )
    assert (old.parent / ".banked").exists()
    again = execute(
        [a, spec("b")],
        artifact=artifact,
        ledger=ledger,
        prefix="s",
        executor=fake_executor(calls),
        work_root=tmp_path / "work",
        log=lambda m: None,
    )
    assert again == 0 and len(ledger.read_text().splitlines()) == 2
