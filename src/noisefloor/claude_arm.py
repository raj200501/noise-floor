"""E1 Claude arm (D-019): Claude tiers as test subjects through Raj's Claude Code plan.

Two harnesses run Claude tiers:

* ``claude_code`` (this module): Claude Code's own agent loop. The task's mock files are
  written into a disposable workspace outside the repository (the mock path
  ``/etc/app.cfg`` becomes ``./etc/app.cfg``). Claude Code works on them with its file
  tools only (:data:`CLAUDE_CODE_TOOLS`: no shell, no code execution, no web tools) under
  ``--restricted``, which also confines the file tools to the workspace, and writes its
  answer to ``ANSWER.txt``. The suite's grader reads that answer and the final files.
* ``minimal``: this project's minimal-agent scaffold with one headless single call per
  step (:func:`noisefloor.runner.run_minimal`).

Every run is one ``agent_run`` against ``claude_agent_runs_per_day`` and is limited to
``claude_agent_max_turns`` turns, both times the cap share (:func:`turn_cap`), through
:class:`noisefloor.client.Client`. The Claude Code process needs the network to reach the
model, so it runs on the host rather than in :mod:`noisefloor.sandbox`; it has no tool
that runs code or reaches the network. Results are model results on synthetic tasks.

``python -m noisefloor.claude_arm smoke`` runs one exploratory run on a pilot task and
prints what the child loaded (model, login source, tools, MCP servers, memory): the
check docs/PLAN.md asks Raj to run once on his machine before the arm starts.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from noisefloor.client import REPO, ClaudeRun, Client
from noisefloor.models import CLAUDE_TIERS
from noisefloor.suite import Snapshot, episode_status, generate
from noisefloor.suite.fs import FileView

if TYPE_CHECKING:  # the runner imports this module
    from noisefloor.runner import RunSpec

ANSWER_FILE = "ANSWER.txt"
CLAUDE_CODE_TOOLS = ("Read", "Glob", "Grep", "Write", "Edit")
MAX_FILE_BYTES = 1 << 20  # larger workspace files are read only up to this size
PREAMBLE = (
    "The current directory is the task's file system: a path such as /data/notes.txt in "
    "the task means ./data/notes.txt here. Use your file tools to read, search, and write "
    "files. To submit an answer, write it alone to ./ANSWER.txt (where the task says to "
    "call `submit`, write ./ANSWER.txt instead); the answer is final once written.\n\n"
    "Task:\n"
)
SMOKE_SEED = 900002  # a pilot seed (900000 to 900999), unused by the pilot plans


def turn_cap(max_steps: int, caps: Mapping[str, Any]) -> int:
    """The turn budget of a Claude run: the run's step limit, at most the turn cap."""
    cap = caps.get("claude_agent_max_turns")
    return max_steps if cap is None else max(1, min(max_steps, int(cap)))


def materialize(files: FileView, root: Path) -> None:
    """Write a task's mock files under ``root`` (``/a/b.txt`` becomes ``root/a/b.txt``)."""
    for path, content in files.to_dict().items():
        relative = path.lstrip("/")
        if relative == ANSWER_FILE:
            raise ValueError(f"task file {path} collides with the answer file")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def collect(root: Path) -> tuple[Snapshot, str | None]:
    """The final workspace as a mock file system, and the answer (None if not written).

    Symbolic links and Claude Code's own ``.claude`` directory are left out, and so is
    ``ANSWER.txt``, which becomes the answer.
    """
    files: dict[str, str] = {}
    answer: str | None = None
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink() or not path.is_file() or relative.parts[0] == ".claude":
            continue
        with path.open("rb") as fh:
            text = fh.read(MAX_FILE_BYTES).decode("utf-8", errors="replace")
        if relative.as_posix() == ANSWER_FILE:
            answer = text
        else:
            files["/" + relative.as_posix()] = text
    return Snapshot(files), answer


def end_reason(run: ClaudeRun, answer: str | None) -> str:
    """How a Claude Code episode ended, in the suite's terms.

    Writing ``ANSWER.txt`` is submitting, whatever happened after it. Otherwise the turn
    cap is ``step_limit``, the wall clock is ``wall_clock``, a clean finish is
    ``stopped``, and anything else is a ``backend_error``.
    """
    result = run.stream.result
    if answer is not None:
        reason = "submitted"
    elif run.ended == "turn_cap" or (result or {}).get("subtype") == "error_max_turns":
        reason = "step_limit"
    elif run.ended == "wall_clock":
        reason = "wall_clock"
    elif result is not None and result.get("subtype") == "success" and not result.get("is_error"):
        reason = "stopped"
    else:
        reason = "backend_error"
    return reason


def child_record(run: ClaudeRun, max_turns: int) -> dict[str, Any]:
    """What the child loaded and how it ended, for the run record and provenance.

    ``clean`` is true when the child reported the plan login, only the allowed tools, no
    MCP servers, and no memory files.
    """
    init = run.stream.init
    result = run.stream.result or {}
    servers = [str(s.get("name")) for s in init.get("mcp_servers") or [] if isinstance(s, dict)]
    tools = [str(t) for t in init.get("tools") or []]
    memory = init.get("memoryPaths") or {}
    return {
        "command": run.command,
        "model_id": init.get("model"),
        "cli_version": init.get("claude_code_version"),
        "api_key_source": init.get("apiKeySource"),
        "permission_mode": init.get("permissionMode"),
        "tools": tools,
        "mcp_servers": servers,
        "memory_paths": memory,
        "clean": bool(init)
        and init.get("apiKeySource") in (None, "none")
        and set(tools) <= set(CLAUDE_CODE_TOOLS)
        and not servers
        and not memory,
        "ended": run.ended,
        "returncode": run.returncode,
        "result_subtype": result.get("subtype"),
        "num_turns": result.get("num_turns"),
        "max_turns": max_turns,
        "plan_equivalent_usd": run.stream.plan_equivalent_usd(),
        "rate_limit_status": run.stream.rate_limit.get("status"),
    }


def run_claude_code(spec: RunSpec, client: Client) -> dict[str, Any]:
    """One Claude Code episode on ``spec``'s task; returns the run record's outcome fields.

    Raises :class:`noisefloor.client.ModelCallRefusedError` when the run must not
    count (guard, daily cap, plan limit), leaving it to be run again later.
    """
    task = generate(spec.family, spec.task_seed, spec.difficulty)
    max_turns = turn_cap(spec.max_steps, client.caps)
    with tempfile.TemporaryDirectory(prefix="nf-claude-run-") as tmp:
        root = Path(tmp)
        materialize(task.files, root)
        run = client.agent_run(
            CLAUDE_TIERS[spec.model],
            PREAMBLE + task.instruction,
            cwd=root,
            tools=CLAUDE_CODE_TOOLS,
            max_turns=max_turns,
            timeout=spec.timeout_s,
        )
        files, answer = collect(root)
    reason = end_reason(run, answer)
    grade = task.grade(answer, files)
    input_tokens, output_tokens = run.stream.usage()
    error = ""
    if reason == "backend_error":
        result = run.stream.result or {}
        error = str(result.get("result") or run.stderr or f"exit {run.returncode}")[:500]
    return {
        "end_reason": reason,
        "status": episode_status(reason),
        "outcome": 1.0 if grade.passed else 0.0,
        "grade_reason": grade.reason,
        "answer": answer,
        "steps": len(run.stream.turn_ids),
        "tool_calls": run.stream.tool_uses,
        "parse_errors": 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "wall_s": round(run.wall_s, 3),
        "error": error,
        "claude": child_record(run, max_turns),
    }


def smoke_spec(tier: str, now: datetime | None = None) -> RunSpec:
    """The exploratory smoke run: one easy pilot task with the Claude Code harness."""
    from noisefloor.runner import RunSpec  # noqa: PLC0415 - the runner imports this module

    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S")
    return RunSpec(
        run_id=f"claude-smoke-{tier}-{stamp}",
        phase="smoke",
        family="pointer",
        task_seed=SMOKE_SEED,
        difficulty=1,
        model=tier,
        harness="claude_code",
        temperature=0.0,
        seed=0,
        timeout_s=300.0,
        max_steps=15,
        threads=1,
        concurrency=1,
    )


def smoke_report(record: Mapping[str, Any]) -> list[str]:
    """Lines that say whether a smoke run's child was clean; the last is the verdict."""
    child = record.get("claude") or {}
    checks = [
        ("login", child.get("api_key_source") in (None, "none"), child.get("api_key_source")),
        ("tools", set(child.get("tools") or []) <= set(CLAUDE_CODE_TOOLS), child.get("tools")),
        ("mcp servers", not child.get("mcp_servers"), child.get("mcp_servers")),
        ("memory", not child.get("memory_paths"), child.get("memory_paths")),
        ("ended", record.get("status") != "infra_error", record.get("end_reason")),
    ]
    lines = [f"model: {child.get('model_id')} (cli {child.get('cli_version')})"]
    lines += [f"{name}: {'OK' if ok else 'CHECK'} {value!r}" for name, ok, value in checks]
    lines.append(f"outcome: {record.get('outcome')} after {record.get('steps')} turns")
    failed = [name for name, ok, _ in checks if not ok]
    lines.append("SMOKE: PASS" if not failed else "SMOKE: CHECK " + ", ".join(failed))
    return lines


def main(
    argv: Sequence[str] | None = None,
    *,
    execute: Callable[..., int] | None = None,
    artifact: Path | None = None,
) -> int:
    """``python -m noisefloor.claude_arm smoke [--tier claude-haiku]``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke"])
    parser.add_argument("--tier", choices=sorted(CLAUDE_TIERS), default="claude-haiku")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if execute is None:
        from noisefloor.orchestrate import execute as run_plan  # noqa: PLC0415

        execute = run_plan
    out = artifact or REPO / "artifacts" / "pilot" / "claude-smoke.jsonl"
    spec = smoke_spec(args.tier)
    execute(
        [spec],
        artifact=out,
        ledger=REPO / "artifacts" / "usage" / "ledger.jsonl",
        prefix="claude-smoke",
        log=lambda m: print(m, flush=True),
    )
    rows = [json.loads(ln) for ln in out.read_text().splitlines()] if out.exists() else []
    record = next((r for r in rows if r.get("run_id") == spec.run_id), None)
    if record is None:
        print("SMOKE: NOT RUN (refused or interrupted; see the log above)")
        return 1
    for line in smoke_report(record):
        print(line)
    return 0 if smoke_report(record)[-1] == "SMOKE: PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
