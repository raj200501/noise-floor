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
runs stay unbanked and run again later. A batch-level failure (the model file fails its
hash check, the server does not start or stops answering its health check, or three
infra_error runs in a row) writes ``OUT_DIR/aborted.json``, which names the recorded runs
the orchestrator must not bank, and starts no further runs.
"""

from __future__ import annotations

import fcntl
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from noisefloor import jsonl
from noisefloor.claude_arm import planned_turns, run_claude_code
from noisefloor.client import REPO, Client, Ledger, ModelCallRefusedError, effective_caps
from noisefloor.minimal_agent import AgentConfig, run_episode
from noisefloor.models import CLAUDE_TIERS, MODELS, is_claude, serve, wait_ready
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
    max_steps = planned_turns(spec.max_steps, client.caps) if claude else spec.max_steps
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
    extra = (
        {
            "max_steps_used": max_steps,
            "model_ids": episode.model_ids,
            "cli_versions": episode.cli_versions,
        }
        if claude
        else {}
    )
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


def _proc_kb(path: Path) -> dict[str, int]:
    """The ``Name: <n> kB`` fields of a /proc file, in kB (empty if unreadable)."""
    try:
        text = path.read_text()
    except OSError:
        return {}
    out = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[name.strip()] = int(parts[0])
    return out


def host_readings(server_pid: int | None = None) -> dict[str, Any]:
    """Load average, available memory, and the model server's memory, now.

    These go on every run record (``host``), so CPU contention and memory pressure can be
    checked run by run; fields that cannot be read are left out.
    """
    out: dict[str, Any] = {}
    with suppress(OSError):
        out["loadavg"] = [round(x, 2) for x in os.getloadavg()]
    available = _proc_kb(Path("/proc/meminfo")).get("MemAvailable")
    if available is not None:
        out["mem_available_mb"] = available // 1024
    if server_pid is not None:
        status = _proc_kb(Path(f"/proc/{server_pid}/status"))
        for field_name, key in (("VmRSS", "server_rss_mb"), ("VmHWM", "server_peak_rss_mb")):
            if field_name in status:
                out[key] = status[field_name] // 1024
    return out


class _Recorder:
    """Appends run records to ``results.jsonl``; thread-safe."""

    def __init__(self, out_dir: Path, batch_id: str) -> None:
        """Write to ``out_dir/results.jsonl`` for batch ``batch_id``."""
        self.path = out_dir / "results.jsonl"
        self.batch_id = batch_id
        self.count = 0
        self.lock = threading.Lock()
        self.server_pid: int | None = None

    def __call__(self, spec: RunSpec, fields: Mapping[str, Any], started: str) -> None:
        """Append one run record."""
        row = {
            **spec.to_json(),
            **fields,
            "started_at": started,
            "finished_at": _now(),
            "batch_id": self.batch_id,
            "sandboxed": os.environ.get("NOISEFLOOR_SANDBOXED") == "1",
            "host": host_readings(self.server_pid),
        }
        with self.lock:
            jsonl.append(self.path, [row])
            self.count += 1


ABORT_STREAK = 3
HEALTH_WAIT_S = 30.0


def abort(out_dir: Path, reason: str, drop: Sequence[str] = ()) -> None:
    """Mark the batch as failed as a whole (``OUT_DIR/aborted.json``).

    ``drop`` names recorded runs the orchestrator must not bank. Those runs and the runs
    that never started stay unbanked and run again, under their planned run ids, once the
    fault is repaired.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    note = {"reason": reason, "drop": list(drop), "at": _now()}
    (out_dir / "aborted.json").write_text(json.dumps(note, sort_keys=True))


class _Watch:
    """Batch-level failure detection over a batch's recorded runs (thread-safe).

    The batch aborts on :data:`ABORT_STREAK` infra_error runs in a row, or when a run ends
    other than ``ok`` and the server then fails its health check (a crashed or hung
    server). The failing runs are dropped, and so is every run recorded after the abort
    (runs already in flight), whatever its status, so the rule never depends on an outcome.
    """

    def __init__(self, out_dir: Path, healthy: Callable[[], bool] | None) -> None:
        """Watch the batch writing to ``out_dir``; ``healthy`` probes its server."""
        self.out_dir = out_dir
        self.healthy = healthy
        self.streak: list[str] = []
        self.drop: list[str] | None = None
        self.reason = ""
        self.lock = threading.Lock()

    def settle(self, run_id: str, status: str) -> bool:
        """Account for one recorded run; True once the batch has aborted."""
        server_ok = status == "ok" or self.healthy is None or self.healthy()
        with self.lock:
            if self.drop is not None:
                self.drop.append(run_id)
                abort(self.out_dir, self.reason, self.drop)
                return True
            self.streak = [*self.streak, run_id] if status == "infra_error" else []
            if len(self.streak) >= ABORT_STREAK:
                self.reason = f"{ABORT_STREAK} infra_error runs in a row"
            elif not server_ok:
                self.reason = "the model server failed its health check after a run"
            else:
                return False
            self.drop = list(dict.fromkeys([*self.streak, run_id]))
            abort(self.out_dir, self.reason, self.drop)
            return True


def _run_all(
    specs: list[RunSpec],
    client: Client,
    record: _Recorder,
    out_dir: Path,
    *,
    proxy_url: str = "",
    workers: int = 1,
    healthy: Callable[[], bool] | None = None,
) -> int:
    """Run the episodes; a refusal or a batch-level failure stops the batch.

    A refused model call records nothing for its run and writes ``refused.json``. A
    batch-level failure (see :class:`_Watch`) writes ``aborted.json``. Either way no
    further run starts, and the runs left out run again later.
    """
    stop = threading.Event()
    watch = _Watch(out_dir, healthy)

    def one(spec: RunSpec) -> None:
        if stop.is_set():
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
            stop.set()
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
        if watch.settle(spec.run_id, str(fields.get("status"))):
            stop.set()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, specs))
    return record.count


def run_batch(batch: Mapping[str, Any], out_dir: Path) -> int:
    """Run every spec in a batch; returns the number of runs recorded.

    A local batch whose model file fails its hash check, or whose server does not start,
    records no run and writes ``aborted.json``.
    """
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
        abort(out_dir, "model file missing or hash mismatch")
        return 0
    port = free_port()
    server: dict[str, Any] = {}
    with ExitStack() as stack:
        try:
            server_cmd = stack.enter_context(
                serve(
                    model,
                    port,
                    threads,
                    parallel=concurrency,
                    log_path=out_dir / "server.log",
                    info=server,
                )
            )
        except Exception as exc:  # no run started, so nothing is recorded
            abort(out_dir, f"server did not start: {type(exc).__name__}: {exc}"[:500])
            return 0
        record.server_pid = server.get("pid")
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
                specs,
                client,
                record,
                out_dir,
                proxy_url=proxy_url,
                workers=concurrency,
                healthy=lambda: wait_ready(port, timeout=HEALTH_WAIT_S),
            )


def main(argv: list[str] | None = None) -> int:
    """``python -m noisefloor.runner BATCH.json OUT_DIR``."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: python -m noisefloor.runner BATCH.json OUT_DIR", file=sys.stderr)
        return 2
    batch = json.loads(Path(args[0]).read_text())
    out = Path(args[1])
    with hold_running(out.parent):
        n = run_batch(batch, out)
    print(f"recorded {n} runs")
    return 0


RUNNING = ".running"  # held by a live runner for its whole life (review ops-6)


@contextmanager
def hold_running(workspace: Path) -> Iterator[None]:
    """Hold an exclusive lock on ``workspace/.running`` while the batch runs.

    The orchestrator refuses to salvage or start a batch while any runner of the plan holds
    its lock, so a relaunch never runs beside a live runner left by a killed orchestrator.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / RUNNING).open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        yield


def running(workspace: Path) -> bool:
    """True while a runner holds ``workspace/.running``."""
    path = workspace / RUNNING
    if not path.exists():
        return False
    with path.open("a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fh, fcntl.LOCK_UN)
    return False


if __name__ == "__main__":
    raise SystemExit(main())
