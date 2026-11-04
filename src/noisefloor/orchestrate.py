"""Orchestrator for subject runs: batching, sandboxed execution, banking with provenance.

Design shape (split plot): a batch is a whole plot that fixes the model server
configuration (model, threads, concurrency); harness, timeout, sampling, and task vary
inside it in randomized order. ``execute`` runs batches in the order given, each in
:mod:`noisefloor.sandbox` via :mod:`noisefloor.runner`, then appends every run record,
with provenance, to a JSON Lines artifact, and the batch's ledger lines to the usage
ledger. Runs whose ``run_id`` is already in the artifact are skipped, so an
interrupted plan resumes where it stopped and a banked run is never re-run.
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
from noisefloor.models import MODELS, RUNTIME, server_binary
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


def provenance(spec: RunSpec, batch: Batch, base: Mapping[str, Any]) -> dict[str, Any]:
    """Provenance fields attached to every banked run record."""
    model = MODELS[spec.model]
    return {
        **base,
        "config_sha256": config_hash(spec),
        "model_repo": model.repo,
        "model_revision": model.revision,
        "model_file": model.filename,
        "model_sha256": model.sha256,
        "model_family": model.family,
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

    Order is kept. Batch IDs include a hash of their run IDs, so a resumed plan never
    reuses the workspace of a different batch.
    """
    if max_size < 1:
        raise ValueError("max_size must be positive")
    groups: list[list[RunSpec]] = []
    for spec in specs:
        if groups and batch_key(groups[-1][-1]) == batch_key(spec) and len(groups[-1]) < max_size:
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


def sandboxed_executor(timeout_per_run: float = 900.0) -> Executor:
    """Run a batch with :mod:`noisefloor.runner` inside the sandbox; return its records."""

    def execute_batch(batch: Batch, workspace: Path) -> list[dict[str, Any]]:
        workspace.mkdir(parents=True, exist_ok=True)
        batch_file = workspace / "batch.json"
        batch_file.write_text(
            json.dumps(
                {"batch_id": batch.batch_id, "runs": [r.to_json() for r in batch.runs]},
                sort_keys=True,
            )
        )
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
        results = out / "results.jsonl"
        rows = (
            [json.loads(ln) for ln in results.read_text().splitlines() if ln.strip()]
            if results.exists()
            else []
        )
        done = {r["run_id"] for r in rows}
        for spec in batch.runs:
            if spec.run_id not in done:
                rows.append(
                    {
                        **spec.to_json(),
                        "end_reason": "harness_error",
                        "status": "infra_error",
                        "outcome": 0.0,
                        "error": f"runner exited {proc.returncode} before this run",
                    }
                )
        return rows

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
) -> None:
    """Append records with provenance, merge the batch ledger, and mark the workspace."""
    by_id = {s.run_id: s for s in batch.runs}
    with artifact.open("a") as fh:
        for row in rows:
            record = {
                **row,
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
) -> int:
    """Run every spec not yet banked in ``artifact``; returns the number banked now.

    Runs that an interrupted earlier call completed but did not bank are salvaged from
    their workspaces first (marked ``"salvaged": true``) and never re-run.
    """
    spec_list = list(specs)
    by_id = {s.run_id: s for s in spec_list}
    done = banked_ids(artifact)
    run_batch = executor or sandboxed_executor()
    root = work_root or Path(os.environ.get("NOISEFLOOR_WORK", DEFAULT_WORK))
    base = base_provenance()
    artifact.parent.mkdir(parents=True, exist_ok=True)
    banked = 0
    for workspace, rows in salvage(root, prefix, by_id, done):
        batch = Batch(workspace.name, tuple(by_id[r["run_id"]] for r in rows))
        started = datetime.now(UTC).isoformat(timespec="seconds")
        _bank(
            rows,
            batch,
            workspace,
            artifact=artifact,
            ledger=ledger,
            base=base,
            started=started,
            salvaged=True,
        )
        done |= {r["run_id"] for r in rows}
        banked += len(rows)
        log(f"{workspace.name}: salvaged {len(rows)} completed runs")
    todo = [s for s in spec_list if s.run_id not in done]
    for batch in make_batches(todo, prefix, max_batch):
        workspace = root / batch.batch_id
        started = datetime.now(UTC).isoformat(timespec="seconds")
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
        )
        banked += len(rows)
        ok = sum(r.get("status") == "ok" for r in rows)
        log(f"{batch.batch_id}: {len(rows)} runs banked ({ok} ok)")
    return banked
