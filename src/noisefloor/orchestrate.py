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


def make_batches(specs: Sequence[RunSpec], prefix: str) -> list[Batch]:
    """Group consecutive runs that share a server configuration (order is kept)."""
    batches: list[Batch] = []
    current: list[RunSpec] = []
    for spec in specs:
        if current and batch_key(current[-1]) != batch_key(spec):
            batches.append(Batch(f"{prefix}-b{len(batches):04d}", tuple(current)))
            current = []
        current.append(spec)
    if current:
        batches.append(Batch(f"{prefix}-b{len(batches):04d}", tuple(current)))
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


def execute(
    specs: Iterable[RunSpec],
    *,
    artifact: Path,
    ledger: Path,
    prefix: str,
    executor: Executor | None = None,
    work_root: Path | None = None,
    log: Callable[[str], None] = print,
) -> int:
    """Run every spec not yet banked in ``artifact``; returns the number banked now."""
    done = banked_ids(artifact)
    todo = [s for s in specs if s.run_id not in done]
    run_batch = executor or sandboxed_executor()
    root = work_root or Path(os.environ.get("NOISEFLOOR_WORK", DEFAULT_WORK))
    base = base_provenance()
    artifact.parent.mkdir(parents=True, exist_ok=True)
    banked = 0
    for batch in make_batches(todo, prefix):
        workspace = root / batch.batch_id
        started = datetime.now(UTC).isoformat(timespec="seconds")
        rows = run_batch(batch, workspace)
        by_id = {s.run_id: s for s in batch.runs}
        with artifact.open("a") as fh:
            for row in rows:
                spec = by_id[row["run_id"]]
                fh.write(
                    json.dumps(
                        {
                            **row,
                            "provenance": provenance(spec, batch, base),
                            "batch_started_at": started,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        batch_ledger = workspace / "out" / "ledger.jsonl"
        if batch_ledger.exists():
            ledger.parent.mkdir(parents=True, exist_ok=True)
            with ledger.open("a") as fh:
                fh.write(batch_ledger.read_text())
        banked += len(rows)
        ok = sum(r.get("status") == "ok" for r in rows)
        log(f"{batch.batch_id}: {len(rows)} runs banked ({ok} ok)")
    return banked
