"""Tests for tools/runcheck.py: the operating checks of the long runs (D-036)."""

import importlib.util
import json
import os
import subprocess
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
    # At 00:00 it wakes and starts the first batch before any runner process exists.
    midnight = _ts("2026-11-14T00:00:01")
    due = _log(tmp_path, "e1 local arm: the first batch starts at 00:00 UTC, in 600 s", midnight)
    code, line = rc.gap([ARM], due, midnight)
    assert code == 1 and "first batch is due" in line


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
    cut = _workspace(work, "e1-day1-b1-b0004-b0000-eeee", [])
    (cut / "out" / "results.jsonl").write_text('{"run_id": "r9", "outc')  # no whole record
    rep = rc.relaunch([], work, False, now)
    assert len(rep.problems) == 3  # three workspaces still to rotate
    assert any("cccc: 2 whole records" in ln for ln in rep.lines)
    assert any("aborted.json" in ln and "note local" in ln for ln in rep.lines)
    rep = rc.relaunch([], work, False, now, rotate=True)
    assert rep.problems == []
    assert (empty / "out.0").is_dir() and not (empty / "out").exists()
    assert (dropped / "out.0" / "aborted.json").exists()
    assert (cut / "out.0" / "results.jsonl").exists()
    assert (keep / "out" / "results.jsonl").exists()
    assert (done / "out").exists()
    assert rc.relaunch([ARM], work, True, now).problems  # never while an arm is alive
    dirty = rc.relaunch([], work, False, now, porcelain="?? src/noisefloor/.e1.py.swp\n")
    assert any("never commit" in p for p in dirty.problems)


def test_relaunch_counts_lines_and_never_parses_records(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text("not json at all\nstill not json\n{cut")
    assert rc.complete_lines(results) == 2
    assert rc.complete_lines(tmp_path / "missing.jsonl") == 0


def test_relaunch_warns_about_the_first_batch(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _workspace(work, "e1-day1-b1-b0000-b0000-aaaa", [{"run_id": "r1"}])
    late = rc.relaunch([], work, True, _ts("2026-11-14T00:40:00"))
    assert any("FIRST BATCH" in ln for ln in late.lines)
    early = rc.relaunch([], work, True, _ts("2026-11-14T00:10:00"))
    assert not any("FIRST BATCH" in ln for ln in early.lines)
    banked = rc.relaunch([], work, False, _ts("2026-11-14T00:40:00"))
    assert not any("FIRST BATCH" in ln for ln in banked.lines)


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


STAMP = {"env": "e1", "analysis": "a1", "src_dirty": False, "at": "2026-11-12T00:00:00Z"}


def _fence(name: str, value: object) -> str:
    return f"```json {name}\n{json.dumps(value)}\n```\n"


def _complete_repo(tmp_path: Path) -> Path:
    """A finished run in miniature: 2 G6 seeds, a grouped and an ungrouped 0005 cell."""
    repo = tmp_path / "repo"
    (repo / "prereg").mkdir(parents=True)
    (repo / "artifacts/g6").mkdir(parents=True)
    (repo / "artifacts/e0/raw").mkdir(parents=True)
    (repo / "runs/e0g6").mkdir(parents=True)
    (repo / rc.P6).write_text("# 0006\n" + _fence("g6-seeds", [7, 8]))
    grid = [
        {"cell": "G-x", "replicates": 2, "params": {"group": "g", "component": "task"}},
        {"cell": "S-y", "replicates": 2, "params": {}},
    ]
    (repo / rc.P5).write_text("# 0005\n" + _fence("e0-grid", grid))
    g6 = [{"seed": s, "components": {}, "stamp": STAMP} for s in (7, 8)]
    (repo / "artifacts/g6/raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in g6))
    grouped = [{"i": i, "seed": rc.glmm_seed("g", i), "stamp": STAMP} for i in (1, 0)]
    (repo / "artifacts/e0/raw/G-x.jsonl").write_text("".join(json.dumps(r) + "\n" for r in grouped))
    cells = [{"i": i, "stamp": STAMP} for i in (0, 1)]
    (repo / "artifacts/e0/raw/S-y.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cells))
    log = repo / rc.DRIVER_LOG
    log.write_text("start 2026-11-11T21:07:01Z workers 4 head abc\ng6 done x\nall done x\n")
    return repo


def test_complete_passes_a_finished_run_and_reads_no_outcome(tmp_path: Path) -> None:
    repo = _complete_repo(tmp_path)
    rep = rc.complete([], repo)
    assert rep.problems == []
    text = "\n".join(rep.lines)
    assert "G6: 2 of 2 preregistered seeds" in text
    assert "g 1 of 1" in text and "ungrouped 1 of 1" in text
    assert "environment e1, analysis code a1" in text
    # Only the index, the seed, and the stamp are read: outcomes may be anything.
    path = repo / "artifacts/e0/raw/S-y.jsonl"
    path.write_text(path.read_text().replace('"stamp"', '"hit": "unread", "stamp"'))
    assert rc.complete([], repo).problems == []


def test_complete_refuses_missing_duplicate_and_foreign_rows(tmp_path: Path) -> None:
    repo = _complete_repo(tmp_path)
    g6 = repo / "artifacts/g6/raw.jsonl"
    g6.write_text(g6.read_text().splitlines()[0] + "\n")  # seed 8 has no row
    assert any(
        "G6: no row for 1 of 2 preregistered seeds" in p for p in rc.complete([], repo).problems
    )
    row = json.dumps({"seed": 7, "stamp": STAMP})
    g6.write_text(f"{row}\n{row}\n" + json.dumps({"seed": 8, "stamp": STAMP}) + "\n")
    assert any("two rows" in p for p in rc.complete([], repo).problems)
    g6.write_text("".join(json.dumps({"seed": s, "stamp": STAMP}) + "\n" for s in (7, 8, 9)))
    assert any("not preregistered" in p for p in rc.complete([], repo).problems)

    repo = _complete_repo(tmp_path / "b")
    cell = repo / "artifacts/e0/raw/S-y.jsonl"
    cell.write_text(json.dumps({"i": 0, "stamp": STAMP}) + "\n")
    rep = rc.complete([], repo)
    assert any("1 of 2 cells incomplete (first S-y 1/2)" in p for p in rep.problems)
    assert "ungrouped 0 of 1" in "\n".join(rep.lines)
    (repo / "artifacts/e0/raw/G-x.jsonl").unlink()
    assert any("2 of 2 cells incomplete" in p for p in rc.complete([], repo).problems)

    repo = _complete_repo(tmp_path / "c")
    grouped = repo / "artifacts/e0/raw/G-x.jsonl"
    grouped.write_text(
        "".join(json.dumps({"i": i, "seed": 5, "stamp": STAMP}) + "\n" for i in (0, 1))
    )
    assert any("does not carry its dataset's seed" in p for p in rc.complete([], repo).problems)


def test_complete_requires_one_clean_environment_and_whole_lines(tmp_path: Path) -> None:
    repo = _complete_repo(tmp_path)
    cell = repo / "artifacts/e0/raw/S-y.jsonl"
    other = {**STAMP, "analysis": "a2"}
    cell.write_text(
        json.dumps({"i": 0, "stamp": STAMP}) + "\n" + json.dumps({"i": 1, "stamp": other}) + "\n"
    )
    assert any("2 environment" in p for p in rc.complete([], repo).problems)
    dirty = {**STAMP, "src_dirty": True}
    for path in (repo / "artifacts/g6/raw.jsonl", *sorted((repo / "artifacts/e0/raw").iterdir())):
        path.write_text(path.read_text().replace(json.dumps(STAMP), json.dumps(dirty)))
    cell.write_text("".join(json.dumps({"i": i, "stamp": dirty}) + "\n" for i in (0, 1)))
    assert any("dirty" in p for p in rc.complete([], repo).problems)

    repo = _complete_repo(tmp_path / "b")
    cell = repo / "artifacts/e0/raw/S-y.jsonl"
    cell.write_text(cell.read_text() + '{"i": 2, "st')
    assert any("last line incomplete" in p for p in rc.complete([], repo).problems)
    cell.write_text(json.dumps({"i": 0, "stamp": STAMP}) + "\nnot json\n")
    assert any("not a JSON object" in p for p in rc.complete([], repo).problems)


def test_complete_reads_the_driver_log_since_its_last_start(tmp_path: Path) -> None:
    repo = _complete_repo(tmp_path)
    log = repo / rc.DRIVER_LOG
    log.write_text("start a\ne0-run exit 1\nstart b\ng6 done x\ne0 cells done x\n")
    assert rc.complete([], repo).problems == []  # the old driver's last line counts as done
    log.write_text("start a\ng6 done x\ne0-run exit 1\ne0 grouped done x\ne0 cells done x\n")
    assert any("e0-run exit 1" in p for p in rc.complete([], repo).problems)
    log.write_text("start a\ng6-run done x\n")
    assert any("not finished" in p for p in rc.complete([], repo).problems)
    log.unlink()
    assert any("no driver log" in p for p in rc.complete([], repo).problems)
    alive = rc.complete([(1, ["bash", "runs/e0g6/driver.sh"])], _complete_repo(tmp_path / "b"))
    assert any(p.startswith("alive:") for p in alive.problems)


def test_complete_restates_the_seed_and_block_rules_of_src() -> None:
    from noisefloor.preregs import prereg_block
    from noisefloor.recovery import e0_glmm_seed

    for group, i in (("e1", 0), ("e1-het", 499), ("e1-local-lo2", 1999)):
        assert rc.glmm_seed(group, i) == e0_glmm_seed(group, i)
    for path, name in ((rc.P5, "e0-grid"), (rc.P6, "g6-seeds")):
        assert rc.prereg_json(rc.REPO / path, name) == prereg_block(rc.REPO / path, name)


def test_alive_names_the_arm_by_its_argv_never_a_launching_shell() -> None:
    launch = ".venv/bin/python -m noisefloor.e1 local prereg/0004-e1-factorial.md"
    wrapper = (20, ["/bin/bash", "-c", f"{{ eval 'mkdir -p runs/e1 && setsid nohup {launch} &' }}"])
    subshell = (21, ["/bin/bash", "-c", "{ \\builtin typeset +x '__CLAUDE_CC_COMMAND_TEXT' }"])
    code, line = rc.alive([wrapper, subshell, RUNNER, SERVER])
    assert (code, line) == (1, "no E1 local arm")
    code, line = rc.alive([wrapper, subshell, ARM, RUNNER, SERVER])
    assert (code, line) == (0, "10")
    code, line = rc.alive([ARM, (30, [PY, "-m", "noisefloor.e1", "local", "x.md"])])
    assert (code, line) == (0, "10 30")  # never more than one in practice; both are named


def _summaries(repo: Path, cells: list[dict[str, object]], comps: list[dict[str, object]]) -> None:
    (repo / "artifacts/e0").mkdir(parents=True, exist_ok=True)
    (repo / "artifacts/g6").mkdir(parents=True, exist_ok=True)
    (repo / "artifacts/e0/summary-0005.json").write_text(json.dumps({"cells": cells}))
    (repo / "artifacts/g6/summary.json").write_text(json.dumps({"components": comps}))


GOOD_CHECKS = {
    "frozen plan": True,
    "validation": True,
    "server": "frozen pin",
    "run code": {"dirty": False},
    "sandbox": True,
}


def _preflight(
    repo: Path,
    checks: dict[str, object] | None = None,
    *,
    procs: list[tuple[int, list[str]]] | None = None,
    now: str = "2026-11-13T21:05:00",
    weights: bool = False,
    mem_kib: int = 14 * 1024 * 1024,
    disk_free: int = 12 * 1024**3,
) -> rc.Report:
    calls: list[bool] = []

    def run(weights: bool) -> dict[str, object]:
        calls.append(weights)
        return dict(GOOD_CHECKS if checks is None else checks)

    rep = rc.preflight_e1(procs or [], repo, _ts(now), run, weights, mem_kib, disk_free)
    rep.lines.append(f"calls={calls}")
    return rep


def test_e1_preflight_passes_a_ready_launch_and_names_failing_extras(tmp_path: Path) -> None:
    _summaries(
        tmp_path,
        [
            {"cell": "G-e1-task", "pass": True, "headline": True},
            {"cell": "S-c-t-i25", "pass": False, "headline": False},
        ],
        [{"component": "task", "pass": True}],
    )
    rep = _preflight(tmp_path)
    assert rep.problems == []
    text = "\n".join(rep.lines)
    assert "failing reported extras (no hold, D-043): S-c-t-i25" in text
    assert "calls=[False]" in text


def test_e1_preflight_refuses_near_the_start_after_banking_and_beside_runs(tmp_path: Path) -> None:
    _summaries(tmp_path, [{"cell": "c", "pass": True, "headline": True}], [])
    for clock in ("2026-11-13T23:45:00", "2026-11-14T00:00:00", "2026-11-14T00:29:59"):
        rep = _preflight(tmp_path, now=clock)
        assert any("23:45 to 00:30 UTC" in p for p in rep.problems)
        assert "calls=[]" in rep.lines  # nothing ran
    assert _preflight(tmp_path, now="2026-11-14T00:30:00").problems == []
    assert _preflight(tmp_path, now="2026-11-13T23:44:59").problems == []
    (tmp_path / "artifacts/e1").mkdir(parents=True)
    (tmp_path / "artifacts/e1/local.jsonl").write_text("")
    rep = _preflight(tmp_path)
    assert any("e1-relaunch" in p for p in rep.problems) and "calls=[]" in rep.lines
    (tmp_path / "artifacts/e1/local.jsonl").unlink()
    for proc in (ARM, RUNNER, SERVER, (1, ["bash", "runs/e0g6/driver.sh"])):
        rep = _preflight(tmp_path, procs=[proc])
        assert any(p.startswith("alive:") for p in rep.problems) and "calls=[]" in rep.lines


def test_e1_preflight_holds_on_a_failing_headline_cell_or_component(tmp_path: Path) -> None:
    rep = _preflight(tmp_path)
    assert any("summary-0005.json is not banked" in p for p in rep.problems)
    assert any("g6/summary.json is not banked" in p for p in rep.problems)
    _summaries(
        tmp_path,
        [{"cell": "H-e1-task", "pass": False, "headline": True}],
        [{"component": "day", "pass": False}, {"component": "task", "pass": True}],
    )
    rep = _preflight(tmp_path)
    assert any(
        "failing headline cells (G5 fails; hold, D-036 item 1): H-e1-task" in p
        for p in rep.problems
    )
    assert any("failing G6 components (hold, D-036 item 1): day" in p for p in rep.problems)


def test_e1_preflight_reports_each_start_up_check(tmp_path: Path) -> None:
    _summaries(tmp_path, [{"cell": "c", "pass": True, "headline": True}], [])
    bad = {
        "frozen plan": {"error": "ValueError: the run plan from this code differs"},
        "validation": {"error": "ValueError: summary not rebuilt"},
        "server": {"error": "ValueError: the llama-server build (x) is not the frozen one"},
        "run code": {"dirty": True},
        "sandbox": False,
        "weights": {"qwen3-4b": False, "qwen3-1.7b": True},
    }
    rep = _preflight(tmp_path, checks=bad, weights=True)
    text = "\n".join(rep.problems)
    for needle in ("frozen plan", "validation", "server", "uncommitted", "sandbox", "qwen3-4b"):
        assert needle in text, needle
    assert "calls=[True]" in rep.lines
    amended = {**GOOD_CHECKS, "server": "prereg/0007-llama-server.md"}
    rep = _preflight(tmp_path, checks=amended)
    assert rep.problems == [] and any("0007" in ln for ln in rep.lines)
    assert any("no result" in p for p in _preflight(tmp_path, checks={}).problems)
    low = _preflight(tmp_path, mem_kib=4 * 1024 * 1024, disk_free=1024**3)
    assert any("MemAvailable" in p for p in low.problems)
    assert any("free disk" in p for p in low.problems)


def test_e1_preflight_checks_run_the_real_start_up_code() -> None:
    found = rc.run_checks(False)
    assert set(found) == {"frozen plan", "validation", "server", "run code", "sandbox"}, found
    assert found["frozen plan"] is True  # the frozen 0004 plan, pins, and tasks match the code


def test_wait_until_refuses_a_time_that_is_not_hh_mm() -> None:
    for bad in ("25:00", "9:00", "21:5", "", "now"):
        proc = subprocess.run(
            ["bash", str(TOOLS / "wait_until.sh"), bad], capture_output=True, text=True, check=False
        )
        assert proc.returncode == 2 and "usage" in proc.stdout, bad
