"""Batch runner for E1 subject runs.

A batch is a set of runs that share one model configuration (model, threads,
concurrency). ``python -m noisefloor.runner BATCH.json OUT_DIR`` runs it.

For a local model (inside :mod:`noisefloor.sandbox`), the runner:

1. checks the model file against its pinned SHA-256;
2. starts llama-server on loopback with the batch's threads and ``concurrency``
   parallel slots, and the project proxy in front of it;
3. runs the batch's episodes with ``concurrency`` in flight at once, each with its
   harness (minimal-agent in this process, Inspect's basic agent in a child
   process through the proxy);
4. appends one JSON line per run to ``OUT_DIR/results.jsonl`` as runs finish; every
   model call is in ``OUT_DIR/ledger.jsonl``.

For a Claude tier (on the host, D-019), it runs the episodes one at a time with the
``claude_code`` or ``minimal`` harness, under the caps in the batch's ``budget`` file,
and logs usage straight to the batch's ``ledger`` (the project ledger), so daily caps
count every run, banked or not.

A run record holds the run spec, the outcome (1 or 0 from the suite grader), the
status (``ok``, ``timeout``, ``infra_error``), the end reason, steps, tool calls,
tokens, wall time, and UTC start and end times. When a model call is refused (billing
guard, daily cap, plan limit, kill switch), the runner records nothing for that run,
writes the reason to ``OUT_DIR/refused.json``, and starts no further runs; unrecorded
runs stay unbanked and run again later.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from noisefloor.claude_arm import run_claude_code, turn_cap
from noisefloor.client import REPO, Client, Ledger, ModelCallRefusedError, effective_caps
from noisefloor.minimal_agent import AgentConfig, run_episode
from noisefloor.models import CLAUDE_TIERS, MODELS, is_claude, serve
from noisefloor.proxy import serve_proxy
from noisefloor.suite import Grade, ToolEnv, episode_status, generate

HARNESSES = ("minimal", "inspect", "claude_code")


@dataclass(frozen=True)
class RunSpec:
    """One subject run: a task, a model configuration, and every factor level."""

    run_id: str
    phase: str
    family: str
    task_seed: int
    difficulty: int
    model: str
    harness: str
    temperature: float
    seed: int
    timeout_s: float
    max_steps: int
    threads: int
    concurrency: int
    max_tokens: int = 384
    factors: Mapping[str, str] = field(default_factory=dict)
    block: str = ""

    def __post_init__(self) -> None:
        """Reject unknown harnesses and models, and harnesses a model cannot run."""
        if self.harness not in HARNESSES:
            raise ValueError(f"unknown harness {self.harness!r}")
        if self.model not in MODELS and not is_claude(self.model):
            raise ValueError(f"unknown model {self.model!r}")
        if self.harness == "claude_code" and not is_claude(self.model):
            raise ValueError("the claude_code harness runs Claude tiers only")
        if self.harness == "inspect" and is_claude(self.model):
            raise ValueError("the inspect harness runs local models only")

    def to_json(self) -> dict[str, Any]:
        """Plain-data form."""
        return {**self.__dict__, "factors": dict(self.factors)}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> RunSpec:
        """Inverse of :meth:`to_json`."""
        return cls(**{**data, "factors": dict(data.get("factors", {}))})


def batch_key(spec: RunSpec) -> tuple[str, int, int]:
    """Runs that can share one server: same model, threads, and concurrency."""
    return (spec.model, spec.threads, spec.concurrency)


def free_port() -> int:
    """An unused loopback port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run_minimal(spec: RunSpec, client: Client) -> dict[str, Any]:
    """One minimal-agent episode (a Claude tier counts as one ``agent_run``)."""
    task = generate(spec.family, spec.task_seed, spec.difficulty)
    claude = is_claude(spec.model)
    max_steps = turn_cap(spec.max_steps, client.caps) if claude else spec.max_steps
    if claude:
        client.start_agent_run(CLAUDE_TIERS[spec.model])
    env = task.start(max_calls=max_steps)
    config = AgentConfig(
        backend="claude" if claude else "openai-local",
        model=CLAUDE_TIERS[spec.model] if claude else spec.model,
        temperature=spec.temperature,
        max_steps=max_steps,
        max_tokens=spec.max_tokens,
        wall_limit_s=spec.timeout_s,
        seed=spec.seed,
        call_kind="agent_step" if claude else "local",
    )
    episode = run_episode(env, client, config)
    reason = episode.end_reason or "step_limit"
    grade = task.grade_env(env)
    extra = {"max_steps_used": max_steps, "model_ids": episode.model_ids} if claude else {}
    return {
        **extra,
        "end_reason": reason,
        "status": episode_status(reason),
        "outcome": 1.0 if grade.passed else 0.0,
        "outcome_strict": strict_outcome(grade),
        "grade_reason": grade.reason,
        "answer": env.answer,
        "steps": episode.steps,
        "tool_calls": tool_calls(env),
        "rejected_calls": episode.rejected_calls,
        "parse_errors": episode.parse_errors,
        "input_tokens": episode.input_tokens,
        "output_tokens": episode.output_tokens,
        "wall_s": round(episode.wall_s, 3),
        "error": episode.error,
    }


def strict_outcome(grade: Grade) -> float:
    """The outcome under the strict comparison (D-021); file grades have only one."""
    return float(grade.score if grade.strict is None else grade.strict)


def tool_calls(env: ToolEnv) -> int:
    """Tool calls that reached the episode's tools, not counting ``submit`` (all harnesses)."""
    return sum(1 for entry in env.trace if entry.tool != "submit")


def run_inspect(spec: RunSpec, proxy_url: str, out_dir: Path) -> dict[str, Any]:
    """One Inspect basic-agent episode in a child process."""
    run_dir = out_dir / "inspect" / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **spec.to_json(),
        "task_seed": spec.task_seed,
        "base_url": proxy_url,
        "log_dir": str(run_dir / "logs"),
    }
    (run_dir / "spec.json").write_text(json.dumps(payload, sort_keys=True))
    result_path = run_dir / "result.json"
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "noisefloor.inspect_harness",
                str(run_dir / "spec.json"),
                str(result_path),
            ],
            capture_output=True,
            text=True,
            timeout=spec.timeout_s + 180,
            check=False,
        )
        (run_dir / "stderr.txt").write_text(proc.stderr[-20000:])
    except subprocess.TimeoutExpired:
        return {
            "end_reason": "harness_error",
            "status": "infra_error",
            "outcome": 0.0,
            "error": "inspect child exceeded its hard timeout",
            "wall_s": round(time.monotonic() - start, 3),
        }
    if not result_path.exists():
        return {
            "end_reason": "harness_error",
            "status": "infra_error",
            "outcome": 0.0,
            "error": f"inspect child exited {proc.returncode}: {proc.stderr[-300:]}",
            "wall_s": round(time.monotonic() - start, 3),
        }
    return dict(json.loads(result_path.read_text()))


class _Recorder:
    """Appends run records to ``results.jsonl``; thread-safe."""

    def __init__(self, out_dir: Path, batch_id: str) -> None:
        """Write to ``out_dir/results.jsonl`` for batch ``batch_id``."""
        self.path = out_dir / "results.jsonl"
        self.batch_id = batch_id
        self.count = 0
        self.lock = threading.Lock()

    def __call__(self, spec: RunSpec, fields: Mapping[str, Any], started: str) -> None:
        """Append one run record."""
        row = {
            **spec.to_json(),
            **fields,
            "started_at": started,
            "finished_at": _now(),
            "batch_id": self.batch_id,
            "sandboxed": os.environ.get("NOISEFLOOR_SANDBOXED") == "1",
        }
        with self.lock, self.path.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            self.count += 1


def _run_all(
    specs: list[RunSpec],
    client: Client,
    record: _Recorder,
    out_dir: Path,
    *,
    proxy_url: str = "",
    workers: int = 1,
) -> int:
    """Run the episodes; a refused model call stops the batch and records nothing more."""
    refused = threading.Event()

    def one(spec: RunSpec) -> None:
        if refused.is_set():
            return
        started = _now()
        try:
            if spec.harness == "minimal":
                fields = run_minimal(spec, client)
            elif spec.harness == "claude_code":
                fields = run_claude_code(spec, client)
            else:
                fields = run_inspect(spec, proxy_url, out_dir)
        except ModelCallRefusedError as exc:
            refused.set()
            note = {
                "reason": str(exc),
                "resets_at": getattr(exc, "resets_at", None),
                "run_id": spec.run_id,
                "at": _now(),
            }
            (out_dir / "refused.json").write_text(json.dumps(note, sort_keys=True))
            return
        except Exception as exc:  # a harness crash must not lose the other runs
            fields = {
                "end_reason": "harness_error",
                "status": "infra_error",
                "outcome": 0.0,
                "error": f"{type(exc).__name__}: {exc}"[:500],
            }
        record(spec, fields, started)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, specs))
    return record.count


def run_batch(batch: Mapping[str, Any], out_dir: Path) -> int:
    """Run every spec in a batch; returns the number of runs recorded."""
    specs = [RunSpec.from_json(s) for s in batch["runs"]]
    if not specs:
        return 0
    keys = {batch_key(s) for s in specs}
    if len(keys) != 1:
        raise ValueError("a batch must share model, threads, and concurrency")
    model_key, threads, concurrency = keys.pop()
    out_dir.mkdir(parents=True, exist_ok=True)
    record = _Recorder(out_dir, str(batch["batch_id"]))
    if is_claude(model_key):
        client = Client(
            ledger=Ledger(Path(batch.get("ledger") or out_dir / "ledger.jsonl")),
            cache_dir=out_dir / "cache",
            caps=effective_caps(Path(batch.get("budget") or REPO / "budget.toml")),
            label=str(batch["batch_id"]),
        )
        return _run_all(specs, client, record, out_dir)
    model = MODELS[model_key]
    if not model.verify():
        for spec in specs:
            record(
                spec,
                {
                    "end_reason": "harness_error",
                    "status": "infra_error",
                    "outcome": 0.0,
                    "error": "model file missing or hash mismatch",
                },
                _now(),
            )
        return record.count
    port = free_port()
    with serve(
        model, port, threads, parallel=concurrency, log_path=out_dir / "server.log"
    ) as server_cmd:
        (out_dir / "server_command.json").write_text(json.dumps(server_cmd))
        client = Client(
            ledger=Ledger(out_dir / "ledger.jsonl"),
            cache_dir=out_dir / "cache",
            caps={},
            label=str(batch["batch_id"]),
            base_urls={"*": f"http://127.0.0.1:{port}"},
        )
        with serve_proxy(client) as proxy_url:
            return _run_all(
                specs, client, record, out_dir, proxy_url=proxy_url, workers=concurrency
            )


def main(argv: list[str] | None = None) -> int:
    """``python -m noisefloor.runner BATCH.json OUT_DIR``."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: python -m noisefloor.runner BATCH.json OUT_DIR", file=sys.stderr)
        return 2
    batch = json.loads(Path(args[0]).read_text())
    n = run_batch(batch, Path(args[1]))
    print(f"recorded {n} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
