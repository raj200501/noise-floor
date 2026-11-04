"""Orchestrator for subject runs: batching, sandboxed execution, banking with provenance.

Design shape (split plot): a batch is a whole plot that fixes the model server
configuration (model, threads, concurrency); harness, timeout, sampling, and task vary
inside it in randomized order. ``execute`` runs batches in the order given, each via
:mod:`noisefloor.runner` (local models in :mod:`noisefloor.sandbox`; Claude tiers as a
host process, because Claude Code needs the network, D-019), then appends every run
record, with provenance, to a JSON Lines artifact, and the batch's ledger lines to the
usage ledger. Runs whose ``run_id`` is already in the artifact are skipped, so an
interrupted plan resumes where it stopped and a banked run is never re-run. When a
batch reports a refused model call (``refused.json``), ``execute`` banks what finished
and stops; the rest of the plan runs on a later call.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from noisefloor import sandbox
from noisefloor.client import claude_child_env
from noisefloor.models import CLAUDE_TIERS, MODELS, RUNTIME, is_claude, server_binary
from noisefloor.runner import RunSpec, batch_key

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WORK = Path.home() / ".cache" / "noisefloor" / "work"


@dataclass(frozen=True)
class Batch:
    """Runs that share one server configuration."""

    batch_id: str
    runs: tuple[RunSpec, ...]


def git_state(repo: Path = REPO) -> dict[str, Any]:
    """Commit and cleanliness of the checkout that launched the runs."""

    def git(*args: str) -> str:
        proc = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)
        return proc.stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def config_hash(spec: RunSpec) -> str:
    """SHA-256 of the run spec's canonical JSON."""
    return hashlib.sha256(json.dumps(spec.to_json(), sort_keys=True).encode()).hexdigest()


CLAUDE_RUNTIME = {
    "name": "Claude Code, headless claude -p",
    "billing": "Raj's Claude plan; total_cost_usd is logged as plan_equivalent_usd, never cash",
}
CLAUDE_SANDBOX = (
    "host process; Claude Code --restricted with file tools only, confined to a disposable "
    "workspace outside the repository; no shell, code execution, or web tools (D-019)"
)


def provenance(spec: RunSpec, batch: Batch, base: Mapping[str, Any]) -> dict[str, Any]:
    """Provenance fields attached to every banked run record.

    A Claude run records its model alias here; the model ID that served it, the CLI
    version, and the exact command line are in the run record's ``claude`` field.
    """
    if is_claude(spec.model):
        model_fields: dict[str, Any] = {
            "model_alias": CLAUDE_TIERS[spec.model],
            "model_family": "claude",
            "runtime": dict(CLAUDE_RUNTIME),
            "sandbox": CLAUDE_SANDBOX,
        }
    else:
        model = MODELS[spec.model]
        model_fields = {
            "model_repo": model.repo,
            "model_revision": model.revision,
            "model_file": model.filename,
            "model_sha256": model.sha256,
            "model_family": model.family,
        }
    return {
        **base,
        **model_fields,
        "config_sha256": config_hash(spec),
        "batch_size": len(batch.runs),
    }


def base_provenance() -> dict[str, Any]:
    """Run-independent provenance: code version, environment, runtime."""
    return {
        "git": git_state(),
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "cpus": os.cpu_count(),
        "runtime": dict(RUNTIME),
        "sandbox": "unshare user+net+mount namespaces, read-only root, chroot (D-012)",
    }


def make_batches(specs: Sequence[RunSpec], prefix: str, max_size: int = 10) -> list[Batch]:
    """Group consecutive runs that share a server configuration, at most ``max_size`` each.

    Order is kept, and a change of ``block`` (E1 plans name each planned batch there)
    always starts a new batch. Batch IDs include a hash of their run IDs, so a resumed
    plan never reuses the workspace of a different batch.
    """
    if max_size < 1:
        raise ValueError("max_size must be positive")
    groups: list[list[RunSpec]] = []
    for spec in specs:
        last = groups[-1][-1] if groups else None
        same = last is not None and batch_key(last) == batch_key(spec) and last.block == spec.block
        if same and len(groups[-1]) < max_size:
            groups[-1].append(spec)
        else:
            groups.append([spec])
    batches = []
    for i, group in enumerate(groups):
        digest = hashlib.sha256(",".join(s.run_id for s in group).encode()).hexdigest()[:8]
        batches.append(Batch(f"{prefix}-b{i:04d}-{digest}", tuple(group)))
    return batches


def banked_ids(artifact: Path) -> set[str]:
    """Run IDs already in an artifact."""
    if not artifact.exists():
        return set()
    return {json.loads(ln)["run_id"] for ln in artifact.read_text().splitlines() if ln.strip()}


Executor = Callable[[Batch, Path], list[dict[str, Any]]]


def _prepare(batch: Batch, workspace: Path, **extra: str) -> Path:
    """Write ``batch.json`` into a fresh workspace and return its path.

    A workspace already banked by an earlier call (possible when a refusal left a
    batch's runs unrecorded) keeps its old output under ``out.<n>``.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    banked = workspace / ".banked"
    if banked.exists():
        out = workspace / "out"
        if out.exists():
            n = sum(1 for _ in workspace.glob("out.*"))
            out.rename(workspace / f"out.{n}")
        banked.unlink()
    batch_file = workspace / "batch.json"
    payload = {"batch_id": batch.batch_id, "runs": [r.to_json() for r in batch.runs], **extra}
    batch_file.write_text(json.dumps(payload, sort_keys=True))
    return batch_file


def _rows(batch: Batch, out: Path, returncode: int | None) -> list[dict[str, Any]]:
    """The batch's run records, with a harness error for each run that has none.

    After a refusal (``refused.json``), runs without a record are left out instead, so
    they stay unbanked and run again later.
    """
    results = out / "results.jsonl"
    rows = (
        [json.loads(ln) for ln in results.read_text().splitlines() if ln.strip()]
        if results.exists()
        else []
    )
    if (out / "refused.json").exists():
        return rows
    done = {r["run_id"] for r in rows}
    for spec in batch.runs:
        if spec.run_id not in done:
            rows.append(
                {
                    **spec.to_json(),
                    "end_reason": "harness_error",
                    "status": "infra_error",
                    "outcome": 0.0,
                    "error": f"runner exited {returncode} before this run",
                }
            )
    return rows


def host_executor(ledger: Path, timeout_per_run: float = 900.0) -> Executor:
    """Run a Claude batch with :mod:`noisefloor.runner` as a host process (D-019).

    The runner logs Claude usage straight to ``ledger`` and reads caps from the
    repository's budget.toml; its environment is the Claude child environment
    (:func:`noisefloor.client.claude_child_env`).
    """

    def execute_batch(batch: Batch, workspace: Path) -> list[dict[str, Any]]:
        batch_file = _prepare(
            batch, workspace, ledger=str(ledger), budget=str(REPO / "budget.toml")
        )
        out = workspace / "out"
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "noisefloor.runner", str(batch_file), str(out)],
                cwd=workspace,
                env=claude_child_env(),
                capture_output=True,
                text=True,
                timeout=timeout_per_run * len(batch.runs) + 300,
                check=False,
            )
            stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout, stderr, code = str(exc.stdout or ""), str(exc.stderr or ""), None
        (workspace / "runner.stdout").write_text(stdout)
        (workspace / "runner.stderr").write_text(stderr[-50000:])
        return _rows(batch, out, code)

    return execute_batch


def default_executor(ledger: Path) -> Executor:
    """Claude batches on the host, local-model batches in the sandbox."""
    host, boxed = host_executor(ledger), sandboxed_executor()

    def execute_batch(batch: Batch, workspace: Path) -> list[dict[str, Any]]:
        claude = is_claude(batch.runs[0].model)
        return (host if claude else boxed)(batch, workspace)

    return execute_batch


def sandboxed_executor(timeout_per_run: float = 900.0) -> Executor:
    """Run a batch with :mod:`noisefloor.runner` inside the sandbox; return its records."""

    def execute_batch(batch: Batch, workspace: Path) -> list[dict[str, Any]]:
        batch_file = _prepare(batch, workspace)
        out = workspace / "out"
        env = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "NOISEFLOOR_MODELS": os.environ.get(
                "NOISEFLOOR_MODELS", str(Path.home() / ".cache" / "noisefloor" / "models")
            ),
            "NOISEFLOOR_LLAMA_SERVER": str(server_binary()),
        }
        proc = sandbox.run(
            [sys.executable, "-m", "noisefloor.runner", str(batch_file), str(out)],
            workspace,
            timeout=timeout_per_run * len(batch.runs) + 300,
            env=env,
        )
        (workspace / "runner.stdout").write_text(proc.stdout)
        (workspace / "runner.stderr").write_text(proc.stderr[-50000:])
        return _rows(batch, out, proc.returncode)

    return execute_batch


def _bank(
    rows: Sequence[Mapping[str, Any]],
    batch: Batch,
    workspace: Path,
    *,
    artifact: Path,
    ledger: Path,
    base: Mapping[str, Any],
    started: str,
    salvaged: bool,
    stamp: Mapping[str, Any] | None = None,
) -> None:
    """Append records with provenance, merge the batch ledger, and mark the workspace."""
    by_id = {s.run_id: s for s in batch.runs}
    with artifact.open("a") as fh:
        for row in rows:
            record = {
                **row,
                **(stamp or {}),
                "provenance": provenance(by_id[row["run_id"]], batch, base),
                "batch_started_at": started,
            }
            if salvaged:
                record["salvaged"] = True
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    batch_ledger = workspace / "out" / "ledger.jsonl"
    if batch_ledger.exists():
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a") as fh:
            fh.write(batch_ledger.read_text())
    (workspace / ".banked").write_text(started + "\n")


def salvage(
    root: Path, prefix: str, specs: Mapping[str, RunSpec], done: set[str]
) -> list[tuple[Path, list[dict[str, Any]]]]:
    """Completed but unbanked runs left in workspaces of an interrupted plan."""
    found = []
    for workspace in sorted(root.glob(f"{prefix}-b*")):
        results = workspace / "out" / "results.jsonl"
        if (workspace / ".banked").exists() or not results.exists():
            continue
        rows = [json.loads(ln) for ln in results.read_text().splitlines() if ln.strip()]
        rows = [r for r in rows if r.get("run_id") in specs and r["run_id"] not in done]
        if rows:
            found.append((workspace, rows))
    return found


def execute(
    specs: Iterable[RunSpec],
    *,
    artifact: Path,
    ledger: Path,
    prefix: str,
    executor: Executor | None = None,
    work_root: Path | None = None,
    log: Callable[[str], None] = print,
    max_batch: int = 10,
    stamp: Mapping[str, Any] | None = None,
) -> int:
    """Run every spec not yet banked in ``artifact``; returns the number banked now.

    Runs that an interrupted earlier call completed but did not bank are salvaged from
    their workspaces first (marked ``"salvaged": true``) and never re-run. ``stamp``
    fields (such as the preregistration a confirmatory run follows) go on every record.
    """
    spec_list = list(specs)
    by_id = {s.run_id: s for s in spec_list}
    done = banked_ids(artifact)
    run_batch = executor or default_executor(ledger)
    root = work_root or Path(os.environ.get("NOISEFLOOR_WORK", DEFAULT_WORK))
    base = base_provenance()
    artifact.parent.mkdir(parents=True, exist_ok=True)
    banked = 0
    for workspace, rows in salvage(root, prefix, by_id, done):
        batch = Batch(workspace.name, tuple(by_id[r["run_id"]] for r in rows))
        # The batch's own start, recorded when it began (salvage time only as a fallback).
        stamp_file = workspace / ".started"
        started = (
            stamp_file.read_text().strip()
            if stamp_file.exists()
            else datetime.now(UTC).isoformat(timespec="seconds")
        )
        _bank(
            rows,
            batch,
            workspace,
            artifact=artifact,
            ledger=ledger,
            base=base,
            started=started,
            salvaged=True,
            stamp=stamp,
        )
        done |= {r["run_id"] for r in rows}
        banked += len(rows)
        log(f"{workspace.name}: salvaged {len(rows)} completed runs")
    todo = [s for s in spec_list if s.run_id not in done]
    for batch in make_batches(todo, prefix, max_batch):
        workspace = root / batch.batch_id
        started = datetime.now(UTC).isoformat(timespec="seconds")
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / ".started").write_text(started + "\n")
        rows = run_batch(batch, workspace)
        _bank(
            rows,
            batch,
            workspace,
            artifact=artifact,
            ledger=ledger,
            base=base,
            started=started,
            salvaged=False,
            stamp=stamp,
        )
        banked += len(rows)
        ok = sum(r.get("status") == "ok" for r in rows)
        log(f"{batch.batch_id}: {len(rows)} runs banked ({ok} ok)")
        refused = workspace / "out" / "refused.json"
        if refused.exists():
            note = json.loads(refused.read_text())
            log(f"{batch.batch_id}: model calls refused ({note.get('reason')}); stopping")
            break
    return banked
