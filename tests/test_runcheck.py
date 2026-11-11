"""Tests for tools/runcheck.py: the operating checks of the long runs (D-036)."""

import importlib.util
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tools_runcheck", TOOLS / "runcheck.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rc = _load()
PY = str(rc.REPO / ".venv/bin/python")
ARM = (10, [PY, "-m", "noisefloor.e1", "local", "prereg/0004-e1-factorial.md"])
RUNNER = (11, [PY, "-m", "noisefloor.runner", "batch.json", "out"])
SERVER = (12, ["/x/llama-server", "-m", "model.gguf"])
SHELL = (13, ["bash", "-c", "pgrep -f 'noisefloor.e1 local' && tail runs/e0g6/driver.sh"])


def _ts(text: str) -> float:
    return datetime.fromisoformat(text).replace(tzinfo=UTC).timestamp()


def _log(tmp_path: Path, line: str, now: float, age: float = 30.0) -> Path:
    log = tmp_path / "log.txt"
    log.write_text(f"e1 local arm: batch e1-day1-b1-b0000 banked 10 runs\n{line}\n")
    os.utime(log, (now - age, now - age))
    return log


def test_processes_leaves_out_this_process() -> None:
    procs = rc.processes()
    assert procs
    assert os.getpid() not in {pid for pid, _ in procs}


def test_study_and_arm_processes_match_commands_not_mentions() -> None:
    procs = [
        (1, ["bash", "runs/e0g6/driver.sh"]),
        (2, ["bash", "tools/e0g6_driver.sh"]),
        (3, [PY, "-m", "noisefloor.recovery", "g6-run", "prereg/0006-g6-recovery.md"]),
        (4, [PY, "-c", "from multiprocessing.spawn import spawn_main", "--multiprocessing-fork"]),
        SHELL,
        ARM,
        RUNNER,
        SERVER,
    ]
    busy = rc.study_processes(procs)
    assert [b.split(" (")[0] for b in busy] == ["driver 1", "driver 2", "study 3", "pool worker 4"]
    assert rc.arm_processes(procs) == {"arm": [10], "runner": [11], "server": [12]}
    assert rc.study_processes([SHELL]) == []
    assert rc.arm_processes([SHELL]) == {"arm": [], "runner": [], "server": []}


def test_gap_none_open_short_and_busy(tmp_path: Path) -> None:
    now = _ts("2026-11-14T10:00:00")
    gap_b1 = "e1 local arm: nothing eligible in block b1, day 1; wait 7200 s"
    assert rc.gap([], tmp_path / "missing", now)[0] == 0
    log = _log(tmp_path, gap_b1, now)
    code, line = rc.gap([ARM], log, now)
    assert (code, line) == (0, "E1-GAP: OPEN 6600 s")  # 2 h to 12:00, less the 10-minute margin
    assert rc.gap([ARM], log, now, need=7000)[1] == "E1-GAP: SHORT 6600 s (need 7000)"
    assert rc.gap([ARM], log, now, need=7000)[0] == 1
    assert rc.gap([ARM, RUNNER], log, now)[0] == 1
    assert rc.gap([ARM, SERVER], log, now)[0] == 1
    stale = _log(tmp_path, gap_b1, now, age=700)
    assert "old" in rc.gap([ARM], stale, now)[1]


def test_gap_reads_the_block_and_the_first_batch_wait(tmp_path: Path) -> None:
    # Just after 12:00 the arm starts a block-2 batch without logging: the b1 wait line is old news.
    now = _ts("2026-11-14T12:00:20")
    log = _log(tmp_path, "e1 local arm: nothing eligible in block b1, day 1; wait 37 s", now)
    code, line = rc.gap([ARM], log, now)
    assert code == 1 and "block b1, now b2" in line
    # A batch has ended and the next one is starting: no wait line, so no gap.
    busy = _log(tmp_path, "e1 local arm: batch e1-day1-b2-b0001 banked 10 runs", now)
    assert rc.gap([ARM], busy, now)[0] == 1
    # Launched in the evening, the arm waits for 00:00 UTC.
    evening = _ts("2026-11-13T22:00:00")
    first = _log(tmp_path, "e1 local arm: the first batch starts at 00:00 UTC, in 7200 s", evening)
    assert rc.gap([ARM], first, evening) == (0, "E1-GAP: OPEN 6600 s")


def _workspace(work: Path, name: str, rows: list[dict[str, object]], **notes: object) -> Path:
    ws = work / name
    (ws / "out" / "cache").mkdir(parents=True)
    if rows:
        (ws / "out" / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for key, value in notes.items():
        (ws / "out" / f"{key}.json").write_text(json.dumps(value))
    return ws


def test_relaunch_rotates_replay_prone_outputs_only(tmp_path: Path) -> None:
    work = tmp_path / "work"
    now = _ts("2026-11-14T03:00:00")
    done = _workspace(work, "e1-day1-b1-b0000-b0000-aaaa", [{"run_id": "r1"}])
    (done / ".banked").write_text("x\n")
    empty = _workspace(work, "e1-day1-b1-b0001-b0000-bbbb", [])
    keep = _workspace(work, "e1-day1-b1-b0002-b0000-cccc", [{"run_id": "r5"}, {"run_id": "r6"}])
    dropped = _workspace(
        work,
        "e1-day1-b1-b0003-b0000-dddd",
        [{"run_id": "r7"}],
        aborted={"reason": "three infra_error runs in a row", "drop": ["r7"], "at": "t"},
    )
    rep = rc.relaunch([], work, {"r1"}, now)
    assert len(rep.problems) == 2  # two workspaces still to rotate
    assert any("2 runs to salvage" in ln for ln in rep.lines)
    assert any("aborted.json" in ln and "note local" in ln for ln in rep.lines)
    rep = rc.relaunch([], work, {"r1"}, now, rotate=True)
    assert rep.problems == []
    assert (empty / "out.0").is_dir() and not (empty / "out").exists()
    assert (dropped / "out.0" / "aborted.json").exists()
    assert (keep / "out" / "results.jsonl").exists()
    assert (done / "out").exists()
    assert rc.relaunch([ARM], work, set(), now).problems  # never while an arm is alive


def test_relaunch_warns_about_the_first_batch(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _workspace(work, "e1-day1-b1-b0000-b0000-aaaa", [{"run_id": "r1"}])
    late = rc.relaunch([], work, set(), _ts("2026-11-14T00:40:00"))
    assert any("FIRST BATCH" in ln for ln in late.lines)
    early = rc.relaunch([], work, set(), _ts("2026-11-14T00:10:00"))
    assert not any("FIRST BATCH" in ln for ln in early.lines)


def test_preflight_refuses_then_repairs_and_checks_the_environment(tmp_path: Path) -> None:
    good = tmp_path / "a.jsonl"
    good.write_text('{"i": 0}\n{"i": 1}\n{"i": 2, "x"')
    env = {"env": "e1", "src_dirty": False}

    def envs() -> set[str]:
        return {"e1"}

    busy = rc.preflight([(1, ["bash", "runs/e0g6/driver.sh"])], "", [good], envs, lambda: env)
    assert busy.problems and good.read_text().endswith('"x"')  # nothing touched while busy
    dirty = rc.preflight([], " M src/noisefloor/e1.py\n", [good], envs, lambda: env)
    assert dirty.problems and good.read_text().endswith('"x"')
    rep = rc.preflight([], "", [good], envs, lambda: env)
    assert rep.problems == [] and good.read_text() == '{"i": 0}\n{"i": 1}\n'
    assert any("repaired" in ln for ln in rep.lines)
    other = rc.preflight([], "", [good], envs, lambda: {"env": "e2", "src_dirty": False})
    assert any("runs e2" in p for p in other.problems)
    mixed = rc.preflight([], "", [good], lambda: {"e1", "e2"}, lambda: env)
    assert any("span" in p for p in mixed.problems)
    bad = tmp_path / "b.jsonl"
    bad.write_text('{"i": 0}\nnot json\n{"i": 2}\n')
    assert any(
        "not a JSON record" in p for p in rc.preflight([], "", [bad], envs, lambda: env).problems
    )


def test_recorded_envs_reads_stamps_and_environment_files(tmp_path: Path) -> None:
    (tmp_path / "artifacts/g6").mkdir(parents=True)
    (tmp_path / "artifacts/e0/raw").mkdir(parents=True)
    (tmp_path / "artifacts/g6/raw.jsonl").write_text(json.dumps({"stamp": {"env": "e1"}}) + "\n")
    (tmp_path / "artifacts/g6/environments.json").write_text(json.dumps({"e1": {}}))
    assert rc.recorded_envs(tmp_path) == {"e1"}
    (tmp_path / "artifacts/e0/raw/environments.json").write_text(json.dumps({"e2": {}}))
    assert rc.recorded_envs(tmp_path) == {"e1", "e2"}
    (tmp_path / "artifacts/e0/raw/environments.json").write_text("{")
    try:
        rc.recorded_envs(tmp_path)
    except ValueError as exc:
        assert "does not parse" in str(exc)
    else:
        raise AssertionError("a broken environments.json must raise")
