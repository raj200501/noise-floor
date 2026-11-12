#!/usr/bin/env python3
"""Operating checks for the long runs (DECISIONS.md D-036): G6 and E0 part 2, then E1.

It never starts, stops, or signals a run, and it reads nothing a run reports as an outcome.
Standard library only, except that it imports :mod:`noisefloor.jsonl` from ``src/`` for the
one repair it makes, so the repair is the one the runs themselves use.

``e0g6-preflight``
    Before the G6 and E0 part 2 driver is launched or relaunched: no driver or study process is
    alive; ``src/``, ``uv.lock`` and ``pyproject.toml`` have no uncommitted changes (rows from a
    dirty tree are refused only at banking); every raw row file ends cleanly (a cut final line
    is removed and printed: prereg 0005 and 0006 let that dataset run again; note it in
    docs/LOG.md) and every line is a JSON object; and the environment of this checkout is the
    one the rows recorded. Prints ``PREFLIGHT: PASS env <hash>`` or ``PREFLIGHT: FAIL``.
``e0g6-complete``
    Before G6 and E0 part 2 are banked (D-036 item 8): completeness read from the raw rows, not
    the log. The driver has ended; every seed of prereg 0006 has exactly one row and no other
    seed has one; every replicate index of every cell of prereg 0005 has a line, and each line of
    a grouped cell carries its dataset's seed; every line is whole; the rows of both studies share
    one environment and one analysis code from a clean ``src/``; and the driver log shows no step
    exit since its last start and a finished last step. Only indices, seeds, and stamps are read.
    Prints ``COMPLETE: YES`` or ``COMPLETE: NO`` (exit 1).
``e1-gap [--need SECONDS]``
    Whether CPU-heavy work (tests, ``make prepush``, the fast ship-check, R, report builds,
    workflows) may run now beside the E1 local arm: ``E1-GAP: NONE`` when no arm runs,
    ``E1-GAP: OPEN <s>`` when the arm waits between blocks, with the seconds left before the
    next 00:00 or 12:00 UTC boundary minus a 10-minute margin, otherwise ``E1-GAP: BUSY <why>``
    (exit 1), or ``E1-GAP: SHORT <s>`` (exit 1) when fewer than SECONDS are left.
``e1-relaunch [--rotate]``
    Before the arm is relaunched after an interruption: refuses while an arm, a batch runner, or
    a model server is alive; lists each unbanked E1 workspace with the number of whole records
    it holds (counted by lines, never parsed: prereg 0004 lets only the runners read E1 records
    before an arm ends) and any ``aborted.json``; and, with ``--rotate``, moves aside the output
    of a workspace from which nothing would be salvaged, so the retry does not replay the cached
    model replies of the killed attempt (prereg 0004: interrupted runs run again from the
    beginning).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from noisefloor import jsonl  # noqa: E402 - the path is set just above

BLOCK_S = 43_200  # E1 blocks: 00:00 to 12:00 and 12:00 to 24:00 UTC (prereg 0004)
MARGIN_S = 600  # CPU-heavy work ends this long before the next block boundary
LOG_FRESH_S = 660  # a waiting arm logs at least every 600 s (noisefloor.e1.run_local)
GRACE_S = 900  # noisefloor.e1.START_GRACE_S: the first batch starts by 00:15 UTC
GAP_LINE = re.compile(
    r"^e1 local arm: (?:nothing eligible in block b(?P<block>[12]), day \d+; wait \d+ s"
    r"|the first batch starts at 00:00 UTC, in \d+ s)$"
)
STUDY_MODULES = ("noisefloor.recovery", "noisefloor.e0_run")
DRIVERS = ("e0g6/driver.sh", "e0g6_driver.sh")
P5 = "prereg/0005-e0-glmm-cells.md"
P6 = "prereg/0006-g6-recovery.md"
DRIVER_LOG = "runs/e0g6/log.txt"
STEP_EXIT = re.compile(r"^[\w-]+ exit \d+")  # both drivers: "<step> exit <code>"
FINISHED = re.compile(r"^(?:all done|e0 cells done) ")  # tools/e0g6_driver.sh; the old driver

Proc = tuple[int, list[str]]


def processes() -> list[Proc]:
    """Every other process (pid, argv), leaving out this one and its ancestors."""
    mine, pid = set(), os.getpid()
    while pid > 1:
        mine.add(pid)
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            break
        pid = int(stat.rsplit(")", 1)[1].split()[1])
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in mine:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        argv = [a.decode(errors="replace") for a in raw.split(b"\0") if a]
        if argv:
            found.append((int(entry.name), argv))
    return found


def _module(argv: Sequence[str]) -> str | None:
    """The module a ``python -m <module>`` command runs, if any."""
    for i, arg in enumerate(argv[:-1]):
        if arg == "-m":
            return argv[i + 1]
    return None


def study_processes(procs: Sequence[Proc]) -> list[str]:
    """The G6 and E0 drivers, study runs, and their pool workers among ``procs``."""
    out = []
    for pid, argv in procs:
        script = argv[1] if len(argv) > 1 and Path(argv[0]).name in ("bash", "sh") else ""
        if script.endswith(DRIVERS):
            out.append(f"driver {pid} ({script})")
        elif _module(argv) in STUDY_MODULES:
            out.append(f"study {pid} ({_module(argv)} {' '.join(argv[-3:])})")
        elif "multiprocessing.spawn" in " ".join(argv) and str(REPO) in argv[0]:
            out.append(f"pool worker {pid}")
    return out


def arm_processes(procs: Sequence[Proc]) -> dict[str, list[int]]:
    """The E1 local arm, its batch runners, and model servers among ``procs``."""
    out: dict[str, list[int]] = {"arm": [], "runner": [], "server": []}
    for pid, argv in procs:
        module = _module(argv)
        if module == "noisefloor.e1" and "local" in argv:
            out["arm"].append(pid)
        elif module == "noisefloor.runner":
            out["runner"].append(pid)
        elif Path(argv[0]).name == "llama-server":
            out["server"].append(pid)
    return out


def _git(*args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
    return proc.stdout.strip()


def raw_files(repo: Path = REPO) -> list[Path]:
    """The raw row files of G6 and of E0 (part 1 and part 2)."""
    return [repo / "artifacts/g6/raw.jsonl", *sorted((repo / "artifacts/e0/raw").glob("*.jsonl"))]


def recorded_envs(repo: Path = REPO) -> set[str]:
    """Environment hashes G6 and E0 part 2 recorded: G6 row stamps, both environments.json.

    Raises ``ValueError`` when a file does not parse.
    """
    found: set[str] = set()
    for row in jsonl.read(repo / "artifacts/g6/raw.jsonl"):
        stamp = row.get("stamp")
        if isinstance(stamp, dict) and stamp.get("env"):
            found.add(str(stamp["env"]))
    for path in (
        repo / "artifacts/g6/environments.json",
        repo / "artifacts/e0/raw/environments.json",
    ):
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except ValueError as exc:
                raise ValueError(f"{_show(path)} does not parse") from exc
            found.update(str(k) for k in data)
    return found


def _show(path: Path) -> str:
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def current_env(repo: Path = REPO) -> dict[str, Any]:
    """This checkout's environment hash and dirty flag, as the study processes compute them."""
    code = (
        "import json; from noisefloor.provenance import environment as e; d = e(); "
        "print(json.dumps({'env': d['env'], 'src_dirty': d['src_dirty']}))"
    )
    proc = subprocess.run(
        [str(repo / ".venv/bin/python"), "-c", code],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {"error": proc.stderr.strip().splitlines()[-1:] or ["no output"]}
    return dict(json.loads(proc.stdout.strip().splitlines()[-1]))


@dataclass
class Report:
    """Lines to print and problems that fail the check."""

    lines: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def preflight(
    procs: Sequence[Proc],
    porcelain: str,
    files: Sequence[Path],
    envs: Callable[[], set[str]],
    env: Callable[[], dict[str, Any]],
) -> Report:
    """The e0g6-preflight checks; repairs a cut final line only when no study can be writing."""
    rep = Report()
    busy = study_processes(procs)
    if busy:
        rep.problems.append("alive: " + "; ".join(busy))
    if porcelain.strip():
        rep.problems.append("uncommitted changes under src/, uv.lock or pyproject.toml")
    if rep.problems:
        return rep  # never touch row files while a study may still write them
    for path in files:
        cut = jsonl.repair_tail(path)
        if cut is not None:
            rep.lines.append(f"repaired {_show(path)}: removed a cut final line ({len(cut)} chars)")
        try:
            jsonl.read(path)
        except ValueError as exc:
            rep.problems.append(str(exc))
    try:
        seen = envs()
    except ValueError as exc:
        rep.problems.append(str(exc))
        return rep
    here = env()
    if "error" in here:
        rep.problems.append(f"cannot compute the environment: {here['error']}")
        return rep
    if here.get("src_dirty"):
        rep.problems.append("the environment reports uncommitted changes under src/")
    if len(seen) > 1:
        rep.problems.append(f"rows already span environments {sorted(seen)}")
    elif seen and here.get("env") not in seen:
        rep.problems.append(f"this checkout runs {here.get('env')}; the rows ran {min(seen)}")
    rep.lines.append(f"env {here.get('env')}")
    return rep


def gap(procs: Sequence[Proc], log: Path, now: float, need: float = 0.0) -> tuple[int, str]:
    """The e1-gap verdict: (exit code, line)."""
    found = arm_processes(procs)
    if not found["arm"]:
        return 0, "E1-GAP: NONE (no local arm runs)"
    if found["runner"] or found["server"]:
        return 1, "E1-GAP: BUSY (a batch runner or model server is alive)"
    if not log.exists():
        return 1, f"E1-GAP: BUSY (no log at {log})"
    age = now - log.stat().st_mtime
    if age >= LOG_FRESH_S:
        return 1, f"E1-GAP: BUSY (the log is {age:.0f} s old; is the arm alive?)"
    lines = [ln for ln in log.read_text(errors="replace").splitlines() if ln.strip()]
    match = GAP_LINE.match(lines[-1]) if lines else None
    if match is None:
        return 1, "E1-GAP: BUSY (the last log line is not a wait between blocks)"
    block = 1 if datetime.fromtimestamp(now, UTC).hour < 12 else 2
    if match.group("block") is None and int(now) % 86_400 <= GRACE_S:
        # The arm wakes at 00:00 for its first batch and starts it without a log line.
        return 1, "E1-GAP: BUSY (the first batch is due now: it starts by 00:15 UTC)"
    if match.group("block") is not None and int(match.group("block")) != block:
        return 1, f"E1-GAP: BUSY (the log waits in block b{match.group('block')}, now b{block})"
    left = (int(now) // BLOCK_S + 1) * BLOCK_S - now - MARGIN_S
    if left < need:
        return 1, f"E1-GAP: SHORT {max(left, 0):.0f} s (need {need:.0f})"
    return 0, f"E1-GAP: OPEN {left:.0f} s"


def complete_lines(results: Path) -> int:
    """How many records a runner's results.jsonl holds, by counting whole lines only.

    Prereg 0004 lets only the runners read E1 records before an arm ends, so this never parses
    one (a cut final line has no newline and is not counted).
    """
    return results.read_bytes().count(b"\n") if results.exists() else 0


def relaunch(
    procs: Sequence[Proc],
    work: Path,
    nothing_banked: bool,
    now: float,
    rotate: bool = False,
    porcelain: str = "",
) -> Report:
    """The e1-relaunch checks and, with ``rotate``, the rotation of replay-prone outputs.

    A relaunch reuses the workspace of an interrupted batch when it salvages nothing from it:
    when the workspace holds no whole record, or when ``aborted.json`` drops every record. The
    retry would then replay the killed attempt's cached replies, so its output is moved aside.
    """
    rep = Report()
    found = arm_processes(procs)
    alive = [f"{kind} {pid}" for kind, pids in found.items() for pid in pids]
    if alive:
        rep.problems.append("alive: " + ", ".join(alive) + "; do not relaunch")
        return rep
    if porcelain.strip():
        # The arm exits at its next batch boundary while src/ is dirty, and committing the
        # change records a code change: restore it instead (DECISIONS.md D-036 item 7, D-038).
        rep.problems.append(
            "uncommitted changes under src/, uv.lock or pyproject.toml: restore them, never "
            "commit (git restore --staged --worktree -- src uv.lock pyproject.toml; git clean "
            "-n -- src, then remove what it lists)"
        )
    for ws in sorted(work.glob("e1-day*")):
        out = ws / "out"
        if (ws / ".banked").exists() or not out.is_dir():
            continue
        records = complete_lines(out / "results.jsonl")
        note = out / "aborted.json"
        failed = json.loads(note.read_text()) if note.exists() else None
        dropped = len(set(failed.get("drop", []))) if failed else 0
        text = f"{ws.name}: {records} whole records"
        if failed:
            text += f", {dropped} of them dropped; aborted.json: {failed.get('reason')!r} at "
            text += f"{failed.get('at')} (record it with `python -m noisefloor.e1 note local`)"
        if records <= dropped:
            n = sum(1 for _ in ws.glob("out.*"))
            if rotate:
                out.rename(ws / f"out.{n}")
                text += f"; nothing to salvage, so out/ moved to out.{n}"
            else:
                text += "; nothing to salvage: rotate it (--rotate) before relaunching"
                rep.problems.append(f"{ws.name}: not rotated")
        elif nothing_banked and ws.name.startswith("e1-day1-b1-") and int(now) % 86_400 > GRACE_S:
            text += (
                "; FIRST BATCH: nothing is banked and it is past 00:15 UTC, so a relaunch waits "
                "for the next 00:00 UTC and these runs become a day of their own"
            )
        rep.lines.append(text)
    return rep


def prereg_json(path: Path, name: str) -> Any:
    """A preregistration's ```json <name>``` block, read as ``noisefloor.preregs.prereg_block``."""
    found = re.search(rf"```json[ \t]+{re.escape(name)}[ \t]*\n(.*?)\n```", path.read_text(), re.S)
    if found is None:
        raise ValueError(f"{_show(path)} has no {name} block")
    return json.loads(found.group(1))


def glmm_seed(group: str, replicate: int) -> int:
    """``noisefloor.recovery.e0_glmm_seed`` with the standard library (a test pins the two)."""
    digest = hashlib.sha256(f"e0-glmm:{group}:{replicate}".encode()).hexdigest()
    return int(digest[:12], 16)


def _lines(path: Path, rep: Report) -> list[dict[str, Any]]:
    """The whole lines of a raw row file as objects; a cut or unreadable line is a problem."""
    data = path.read_bytes().decode()
    if data and not data.endswith("\n"):
        rep.problems.append(f"{_show(path)}: last line incomplete")
    rows: list[dict[str, Any]] = []
    for n, line in enumerate(data.split("\n")[:-1], 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            row = None
        if not isinstance(row, dict):
            rep.problems.append(f"{_show(path)}: line {n} is not a JSON object")
            continue
        rows.append(row)
    return rows


def _stamp(row: dict[str, Any]) -> tuple[str, str, Any]:
    stamp = row.get("stamp")
    if not isinstance(stamp, dict):
        stamp = {}
    return str(stamp.get("env")), str(stamp.get("analysis")), stamp.get("src_dirty")


def complete(procs: Sequence[Proc], repo: Path = REPO) -> Report:
    """The e0g6-complete checks; reads only indices, seeds, and stamps of the raw rows."""
    rep = Report()
    busy = study_processes(procs)
    if busy:
        rep.problems.append("alive: " + "; ".join(busy))
    stamps: set[tuple[str, str, Any]] = set()

    seeds = [int(s) for s in prereg_json(repo / P6, "g6-seeds")]
    raw = repo / "artifacts/g6/raw.jsonl"
    rows = _lines(raw, rep) if raw.exists() else []
    count: dict[int, int] = {}
    for row in rows:
        count[int(row.get("seed", -1))] = count.get(int(row.get("seed", -1)), 0) + 1
        stamps.add(_stamp(row))
    missing = [s for s in seeds if s not in count]
    rep.lines.append(f"G6: {len(seeds) - len(missing)} of {len(seeds)} preregistered seeds")
    if missing:
        rep.problems.append(f"G6: no row for {len(missing)} of {len(seeds)} preregistered seeds")
    twice = sorted(s for s, n in count.items() if n > 1)
    if twice:
        rep.problems.append(f"G6: seeds with two rows: {twice[:5]}")
    extra = sorted(set(count) - set(seeds))
    if extra:
        rep.problems.append(f"G6: rows for seeds not preregistered: {extra[:5]}")

    grid = prereg_json(repo / P5, "e0-grid")
    tally: dict[str, list[int]] = {}
    short: list[str] = []
    for cell in grid:
        group = (cell.get("params") or {}).get("group")
        counts = tally.setdefault("ungrouped" if group is None else str(group), [0, 0])
        counts[1] += 1
        reps = int(cell["replicates"])
        path = repo / "artifacts/e0/raw" / f"{cell['cell']}.jsonl"
        lines = _lines(path, rep) if path.exists() else []
        have = {int(row.get("i", -1)) for row in lines}
        done = sum(1 for i in range(reps) if i in have)
        if done == reps:
            counts[0] += 1
        else:
            short.append(f"{cell['cell']} {done}/{reps}")
        for row in lines:
            stamps.add(_stamp(row))
            i = int(row.get("i", -1))
            if group is not None and 0 <= i < reps and row.get("seed") != glmm_seed(str(group), i):
                rep.problems.append(
                    f"{cell['cell']}: replicate {i} does not carry its dataset's seed"
                )
                break
    rep.lines.append(
        "E0 part 2: " + ", ".join(f"{g} {done} of {n}" for g, (done, n) in tally.items()) + " cells"
    )
    if short:
        rep.problems.append(
            f"E0 part 2: {len(short)} of {len(grid)} cells incomplete (first {short[0]})"
        )

    if len(stamps) > 1:
        rep.problems.append(
            f"rows span {len(stamps)} environment, analysis code, or dirty-flag combinations"
        )
    elif stamps:
        env, code, dirty = next(iter(stamps))
        if dirty is not False:
            rep.problems.append(f"rows ran from a dirty src/ (src_dirty {dirty})")
        rep.lines.append(f"one stamp in both studies: environment {env}, analysis code {code}")

    log = repo / DRIVER_LOG
    if not log.exists():
        rep.problems.append(f"no driver log at {DRIVER_LOG}")
        return rep
    text = log.read_text().splitlines()
    starts = [n for n, line in enumerate(text) if line.startswith("start ")]
    since = text[starts[-1] :] if starts else text
    exits = [line for line in since if STEP_EXIT.match(line)]
    if exits:
        rep.problems.append(f"driver log: {exits[0]} since the last start")
    if not any(FINISHED.match(line) for line in since):
        rep.problems.append("driver log: the last step has not finished since the last start")
    return rep


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("e0g6-preflight")
    sub.add_parser("e0g6-complete")
    p_gap = sub.add_parser("e1-gap")
    p_gap.add_argument("--need", type=float, default=0.0, help="seconds the job needs")
    p_gap.add_argument("--log", type=Path, default=REPO / "runs/e1/log.txt")
    p_rel = sub.add_parser("e1-relaunch")
    p_rel.add_argument("--rotate", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "e1-gap":
        code, line = gap(processes(), args.log, time.time(), args.need)
        print(line)
        return code
    if args.cmd == "e0g6-complete":
        rep = complete(processes())
        verdict = "COMPLETE: NO" if rep.problems else "COMPLETE: YES"
    elif args.cmd == "e1-relaunch":
        work = Path(os.environ.get("NOISEFLOOR_WORK", Path.home() / ".cache/noisefloor/work"))
        artifact = REPO / "artifacts/e1/local.jsonl"  # its size only: no record is read
        empty = not artifact.exists() or artifact.stat().st_size == 0
        dirty = _git("status", "--porcelain", "--", "src", "uv.lock", "pyproject.toml")
        rep = relaunch(processes(), work, empty, time.time(), args.rotate, dirty)
        verdict = "RELAUNCH: FAIL" if rep.problems else "RELAUNCH: OK"
    else:
        rep = preflight(
            processes(),
            _git("status", "--porcelain", "--", "src", "uv.lock", "pyproject.toml"),
            raw_files(),
            recorded_envs,
            current_env,
        )
        verdict = "PREFLIGHT: FAIL" if rep.problems else "PREFLIGHT: PASS"
    for line in rep.lines:
        print(line)
    for problem in rep.problems:
        print(f"problem: {problem}")
    print(verdict)
    return 1 if rep.problems else 0


if __name__ == "__main__":
    sys.exit(main())
