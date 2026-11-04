"""E1 factorial study: design, run plan, and schedule (docs/BUILD_BRIEF.md section 5).

Local arm: every model is crossed with every task, and each (model, task) cell runs the
half fraction 2^(5-1) (``runs_per_cell = 16``) or one of its two blocks (8) of five
two-level factors (:func:`noisefloor.recovery.half_fraction_blocks`, the design the G6
recovery study simulates): harness, wall-clock timeout, step budget, sampling
temperature, and concurrency. Within a cell every factor is balanced (half the cell's
runs at each level). Runs are
grouped into batches of one model and one concurrency level (one llama-server
configuration, at most ``batch_size`` runs), and batches are spread evenly, in random
order, over ``days`` x ``blocks_per_day`` time slots. Each run's ``block`` field is
``<slot>/<batch>``, so the orchestrator keeps the planned batches.

Claude slice: each Claude tier runs every task ``claude_runs_per_task`` times, crossing
harness (minimal-agent or Claude Code) with timeout; the step budget is the turn cap and
temperature is the CLI's default (neither can vary for Claude here, D-019).

All tasks are procedurally generated mock tasks (nf-tools suite version 2).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import random
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from noisefloor import jsonl, preregs
from noisefloor.client import CLAUDE_LEDGER
from noisefloor.models import CLAUDE_TIERS, MODELS
from noisefloor.preregs import REPO, prereg_block, repo_path
from noisefloor.recovery import TWO_LEVEL, cell_runs, half_fraction_blocks
from noisefloor.runner import RunSpec
from noisefloor.suite import FAMILY_NAMES, Task, check_disjoint, generate, suite_manifest

PILOT_SEEDS = range(900000, 901000)
CONFIRMATORY_SEEDS = range(100000, 101000)
NATIVE = {"local": "inspect", "claude": "claude_code"}
DEFAULT_LEVELS: Mapping[str, tuple[Any, Any]] = {
    "harness": ("minimal", "native"),
    "timeout": (150.0, 300.0),
    "steps": (8, 15),
    "temperature": (0.0, 0.7),
    "concurrency": (1, 2),
}


@dataclass(frozen=True)
class E1Design:
    """Everything that fixes the E1 run plan (the preregistration states each value)."""

    models: tuple[str, ...]
    claude_tiers: tuple[str, ...] = ("claude-haiku", "claude-sonnet")
    families: tuple[str, ...] = FAMILY_NAMES
    difficulties: tuple[int, ...] = (1, 2, 3)
    tasks_per_family: int = 8
    levels: Mapping[str, tuple[Any, Any]] = field(default_factory=lambda: dict(DEFAULT_LEVELS))
    days: int = 3
    blocks_per_day: int = 2
    batch_size: int = 10
    threads: int = 4
    claude_runs_per_task: int = 2
    order_seed: int = 1
    max_tokens: int = 384
    runs_per_cell: int = 8

    def __post_init__(self) -> None:
        """Reject designs E1 cannot run."""
        if self.runs_per_cell not in (8, 16):
            raise ValueError("runs_per_cell must be 8 (one block) or 16 (the half fraction)")
        unknown = [m for m in self.models if m not in MODELS]
        if unknown:
            raise ValueError(f"not pinned local models: {unknown}")
        bad_tiers = [t for t in self.claude_tiers if t not in CLAUDE_TIERS]
        if bad_tiers:
            raise ValueError(f"not Claude tiers: {bad_tiers}")
        if set(self.levels) != set(TWO_LEVEL):
            raise ValueError(f"levels must name exactly {TWO_LEVEL}")
        if self.tasks_per_family > len(CONFIRMATORY_SEEDS) or self.tasks_per_family < 1:
            raise ValueError("tasks_per_family out of range")
        check_disjoint(PILOT_SEEDS, CONFIRMATORY_SEEDS)

    @property
    def slots(self) -> list[str]:
        """Time slots in order: ``day1-b1``, ``day1-b2``, ..."""
        return [
            f"day{d + 1}-b{b + 1}" for d in range(self.days) for b in range(self.blocks_per_day)
        ]

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> E1Design:
        """Inverse of :meth:`to_json` (the ``e1-plan`` block of the preregistration)."""
        return cls(
            models=tuple(data["models"]),
            claude_tiers=tuple(data["claude_tiers"]),
            families=tuple(data["families"]),
            difficulties=tuple(int(d) for d in data["difficulties"]),
            tasks_per_family=int(data["tasks_per_family"]),
            levels={str(k): (v[0], v[1]) for k, v in data["levels"].items()},
            days=int(data["days"]),
            blocks_per_day=int(data["blocks_per_day"]),
            batch_size=int(data["batch_size"]),
            threads=int(data["threads"]),
            claude_runs_per_task=int(data["claude_runs_per_task"]),
            order_seed=int(data["order_seed"]),
            max_tokens=int(data["max_tokens"]),
            runs_per_cell=int(data.get("runs_per_cell", 8)),
        )

    def to_json(self) -> dict[str, Any]:
        """Plain-data form (for the preregistration and the manifest)."""
        return {
            "models": list(self.models),
            "claude_tiers": list(self.claude_tiers),
            "families": list(self.families),
            "difficulties": list(self.difficulties),
            "tasks_per_family": self.tasks_per_family,
            "levels": {k: list(v) for k, v in self.levels.items()},
            "days": self.days,
            "blocks_per_day": self.blocks_per_day,
            "batch_size": self.batch_size,
            "threads": self.threads,
            "claude_runs_per_task": self.claude_runs_per_task,
            "order_seed": self.order_seed,
            "max_tokens": self.max_tokens,
            "runs_per_cell": self.runs_per_cell,
        }


def tasks(design: E1Design) -> list[Task]:
    """The confirmatory tasks: per family, seeds in order with difficulties in rotation."""
    out = []
    for family in design.families:
        for j in range(design.tasks_per_family):
            difficulty = design.difficulties[j % len(design.difficulties)]
            out.append(generate(family, CONFIRMATORY_SEEDS[j], difficulty))
    return out


def manifest(design: E1Design) -> dict[str, Any]:
    """The task manifest (hash of every task) for the preregistration."""
    return suite_manifest(tasks(design))


def plan_sha256(design: E1Design) -> str:
    """SHA-256 of the canonical JSON of every planned run (local, then Claude)."""
    specs = [s.to_json() for s in (*local_plan(design), *claude_plan(design))]
    text = json.dumps(specs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def model_pins(design: E1Design) -> dict[str, dict[str, str]]:
    """Repository, revision, file, and SHA-256 of every local model's weights."""
    return {
        m: {
            "repo": MODELS[m].repo,
            "revision": MODELS[m].revision,
            "filename": MODELS[m].filename,
            "sha256": MODELS[m].sha256,
        }
        for m in design.models
    }


def check_frozen_plan(prereg: Path, design: E1Design) -> None:
    """Refuse when the code would give another plan or other weights than the frozen ones.

    The ``e1-design`` block records ``plan_sha256`` and ``model_pins``. For a
    preregistration in prereg/CHAIN.jsonl both are required, so a frozen file without them
    cannot run; a file outside the chain (a test fixture) is checked only for the keys it
    has.
    """
    frozen = preregs.chain_entry(prereg) is not None
    try:
        block = prereg_block(prereg, "e1-design")
    except ValueError:
        if frozen:
            raise
        return
    expected = {
        "plan_sha256": (plan_sha256(design), "the run plan from this code differs from"),
        "model_pins": (model_pins(design), "the local model pins differ from"),
    }
    for key, (value, differs) in expected.items():
        if key not in block:
            if frozen:
                raise ValueError(f"the frozen e1-design block has no {key}")
            continue
        if block[key] != value:
            raise ValueError(f"{differs} the frozen e1-design block ({key})")


def _stamp(prereg: Path, design: E1Design) -> dict[str, Any]:
    """Fields every confirmatory record carries, after the frozen-prereg checks pass."""
    entry = preregs.require_frozen(prereg)
    check_frozen_plan(prereg, design)
    return {"prereg": repo_path(prereg), "prereg_sha256": entry["sha256"], "confirmatory": True}


def _level(design: E1Design, name: str, bit: int) -> Any:
    return design.levels[name][bit]


def _local_spec(
    design: E1Design, model: str, task: Task, bits: Sequence[int], seed: int
) -> RunSpec:
    levels = dict(zip(TWO_LEVEL, bits, strict=True))
    harness = _level(design, "harness", levels["harness"])
    factors = {name: str(_level(design, name, levels[name])) for name in TWO_LEVEL}
    return RunSpec(
        run_id=f"e1-{model}-{task.id}-{''.join(str(b) for b in bits)}",
        phase="e1",
        family=task.family,
        task_seed=task.seed,
        difficulty=task.difficulty,
        model=model,
        harness=NATIVE["local"] if harness == "native" else "minimal",
        temperature=float(_level(design, "temperature", levels["temperature"])),
        seed=seed,
        timeout_s=float(_level(design, "timeout", levels["timeout"])),
        max_steps=int(_level(design, "steps", levels["steps"])),
        threads=design.threads,
        concurrency=int(_level(design, "concurrency", levels["concurrency"])),
        max_tokens=design.max_tokens,
        factors=factors,
    )


def local_plan(design: E1Design) -> list[RunSpec]:
    """Every local-arm run, in execution order (slot by slot, batch by batch)."""
    rng = random.Random(f"e1-local:{design.order_seed}")
    first, second = half_fraction_blocks()
    task_list = tasks(design)
    runs: list[RunSpec] = []
    for mi, model in enumerate(design.models):
        for ti, task in enumerate(task_list):
            half = cell_runs(first, second, mi + ti, design.runs_per_cell)
            for bits in half:
                runs.append(_local_spec(design, model, task, bits, rng.randrange(2**31)))
    groups: dict[tuple[str, int], list[RunSpec]] = {}
    for spec in runs:
        groups.setdefault((spec.model, spec.concurrency), []).append(spec)
    slots = design.slots
    planned: list[tuple[str, list[RunSpec]]] = []
    for key in sorted(groups):
        members = groups[key]
        rng.shuffle(members)
        size = design.batch_size
        chunks = [members[i : i + size] for i in range(0, len(members), size)]
        offset = rng.randrange(len(slots))
        # Each (model, concurrency) group spreads its batches evenly over the slots.
        for j, chunk in enumerate(chunks):
            planned.append((slots[(offset + j) % len(slots)], chunk))
    rng.shuffle(planned)
    plan: list[RunSpec] = []
    for slot in slots:
        for b, (where, chunk) in enumerate(planned):
            if where == slot:
                plan += [_with_block(spec, f"{slot}/b{b:04d}") for spec in chunk]
    return plan


def _with_block(spec: RunSpec, block: str) -> RunSpec:
    return RunSpec.from_json({**spec.to_json(), "block": block})


DRY_SEEDS = range(900600, 900700)  # pilot seeds of the E1 dry run, unused by earlier pilots
DRY_CONFIGS = ((4, 1), (4, 2), (3, 1), (3, 2))  # (server threads, concurrency)


def dry_plan(
    design: E1Design,
    runs_per_config: int = 4,
    configs: Sequence[tuple[int, int]] = DRY_CONFIGS,
) -> list[RunSpec]:
    """The exploratory E1 dry run: every model at each (threads, concurrency) setting.

    Each (model, setting) is one batch of ``runs_per_config`` runs on the same pilot tasks
    (families in turn, difficulties in rotation, seeds from :data:`DRY_SEEDS`). A batch
    takes the half-fraction rows of its concurrency level whose temperature level
    alternates by model, so harness, timeout, and step budget are balanced in it, and a
    run matches its counterpart at the other thread count (same task, levels, and seed)
    for paired wall-time comparisons. Phase ``e1-dry``: never confirmatory, never pooled.
    """
    first, second = half_fraction_blocks()
    c_at, t_at = TWO_LEVEL.index("concurrency"), TWO_LEVEL.index("temperature")
    chosen = [
        generate(
            design.families[k % len(design.families)],
            DRY_SEEDS[k],
            design.difficulties[k % len(design.difficulties)],
        )
        for k in range(runs_per_config)
    ]
    plan: list[RunSpec] = []
    for mi, model in enumerate(design.models):
        for threads, concurrency in configs:
            c_bit = list(design.levels["concurrency"]).index(concurrency)
            rows = [b for b in first + second if b[c_at] == c_bit and b[t_at] == mi % 2]
            for k, task in enumerate(chosen):
                bits = rows[k % len(rows)]
                seed = random.Random(f"e1-dry:{design.order_seed}:{model}:{concurrency}:{k}")
                spec = _local_spec(design, model, task, bits, seed.randrange(2**31)).to_json()
                code = "".join(str(b) for b in bits)
                spec.update(
                    run_id=f"e1dry-{model}-t{threads}-c{concurrency}-{task.id}-{code}",
                    phase="e1-dry",
                    threads=threads,
                    block=f"dry/{model}-t{threads}-c{concurrency}",
                )
                plan.append(RunSpec.from_json(spec))
    return plan


def dry_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Infrastructure readings of a dry run, per batch (no outcomes).

    Per (model, threads, concurrency): runs, infra_error and timeout counts, median and
    largest wall time, the largest 1-minute load average, the least available memory,
    and the model server's peak resident memory, from the runs' ``host`` readings.
    """
    groups: dict[tuple[str, int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["model"]), int(row["threads"]), int(row["concurrency"]))
        groups.setdefault(key, []).append(row)
    out = []
    for (model, threads, concurrency), members in sorted(groups.items()):
        walls = sorted(float(r.get("wall_s") or 0.0) for r in members)
        hosts = [r.get("host") or {} for r in members]

        def most(key: str, hosts: list[Mapping[str, Any]] = hosts) -> Any:
            values = [h[key] for h in hosts if key in h]
            return max(values) if values else None

        loads = [h["loadavg"][0] for h in hosts if h.get("loadavg")]
        free = [h["mem_available_mb"] for h in hosts if "mem_available_mb" in h]
        out.append(
            {
                "model": model,
                "threads": threads,
                "concurrency": concurrency,
                "runs": len(members),
                "infra_error": sum(1 for r in members if r.get("status") == "infra_error"),
                "timeout": sum(1 for r in members if r.get("status") == "timeout"),
                "wall_median_s": walls[len(walls) // 2] if walls else None,
                "wall_max_s": walls[-1] if walls else None,
                "load_max": max(loads) if loads else None,
                "mem_available_min_mb": min(free) if free else None,
                "server_peak_rss_mb": most("server_peak_rss_mb"),
            }
        )
    return out


def claude_plan(design: E1Design) -> list[RunSpec]:
    """Every Claude-slice run: harness crossed with timeout, balanced within each task.

    Each task's runs come in pairs that differ in both harness and timeout: tasks
    alternate between the pairs (minimal, short) + (native, long) and (minimal, long) +
    (native, short), offset by tier, so harness and timeout are balanced in every tier
    and within every task. Runs are ordered by complete (tier, task) groups with the tiers
    interleaved, so a slice that ends early leaves complete pairs balanced over tiers.
    """
    if design.claude_runs_per_task % 2:
        raise ValueError("claude_runs_per_task must be even")
    rng = random.Random(f"e1-claude:{design.order_seed}")
    pairs = (((0, 0), (1, 1)), ((0, 1), (1, 0)))
    steps = design.levels["steps"][0]
    task_list = tasks(design)
    groups: dict[tuple[int, int], list[RunSpec]] = {}
    for ci, tier in enumerate(design.claude_tiers):
        for ti, task in enumerate(task_list):
            combos = pairs[(ci + ti) % 2] * (design.claude_runs_per_task // 2)
            group = groups.setdefault((ci, ti), [])
            for k, (h, t) in enumerate(combos):
                harness = _level(design, "harness", h)
                timeout = float(_level(design, "timeout", t))
                group.append(
                    RunSpec(
                        run_id=f"e1c-{tier}-{task.id}-h{h}t{t}-{k}",
                        phase="e1-claude",
                        family=task.family,
                        task_seed=task.seed,
                        difficulty=task.difficulty,
                        model=tier,
                        harness=NATIVE["claude"] if harness == "native" else "minimal",
                        temperature=0.0,
                        seed=rng.randrange(2**31),
                        timeout_s=timeout,
                        max_steps=int(steps),
                        threads=1,
                        concurrency=1,
                        max_tokens=design.max_tokens,
                        factors={
                            "harness": str(harness),
                            "timeout": str(timeout),
                            "steps": str(steps),
                            "temperature": "cli_default",
                            "concurrency": "1",
                        },
                    )
                )
    # Tasks in a fixed random order; for each task the tiers take turns, the first tier
    # rotating with the task, and a tier's runs of the task are back to back in random
    # order. A slice cut anywhere leaves complete (tier, task) groups but one, and tiers
    # within one group of each other.
    order = list(range(len(task_list)))
    rng.shuffle(order)
    plan: list[RunSpec] = []
    n_tiers = len(design.claude_tiers)
    for position, ti in enumerate(order):
        for j in range(n_tiers):
            group = groups[((position + j) % n_tiers, ti)]
            rng.shuffle(group)
            plan.extend(group)
    return plan


# ---------------------------------------------------------------------------
# running the preregistered plan

ARTIFACTS = REPO / "artifacts" / "e1"
DRY_ARTIFACT = REPO / "artifacts" / "pilot" / "e1-dry.jsonl"
LEDGER = REPO / "artifacts" / "usage" / "ledger.jsonl"
BLOCK_HOURS = ((0, 12), (12, 24))  # UTC time-of-day blocks b1 and b2


def current_block(now: datetime, block_hours: Sequence[Sequence[int]] = BLOCK_HOURS) -> int:
    """1-based time-of-day block of a UTC time."""
    for index, (start, end) in enumerate(block_hours):
        if start <= now.hour < end:
            return index + 1
    raise ValueError(f"hour {now.hour} is in no block")


def seconds_to_next_block(
    now: datetime, block_hours: Sequence[Sequence[int]] = BLOCK_HOURS
) -> float:
    """Seconds from ``now`` until the next block starts."""
    end = block_hours[current_block(now, block_hours) - 1][1]
    boundary = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=end - now.hour)
    return max(1.0, (boundary - now).total_seconds())


def slot_day(label: str) -> int:
    """Planned day (1-based) of a batch labeled ``day<k>-b<n>/b<NNNN>``."""
    slot = label.split("/", maxsplit=1)[0]
    return int(slot.split("-", maxsplit=1)[0].removeprefix("day"))


def arm_start(artifact: Path) -> date | None:
    """UTC date on which the earliest banked run started (the arm's day 1), if any."""
    days = [
        date.fromisoformat(str(started)[:10])
        for row in jsonl.read(artifact, tail_ok=True)
        if (started := row.get("started_at"))
    ]
    return min(days) if days else None


def eligible_batches(pending: Sequence[str], block: int, day: int) -> list[str]:
    """Pending batches that may start now, earliest planned first.

    A batch of slot ``day<k>-b<n>`` may start only in time-of-day block ``n`` of the arm's
    day ``k`` or a later day, so the arm spans at least as many UTC days as the plan has,
    and a slot that overruns its block continues in the same block of the next day.
    """
    return [k for k in pending if k.split("/")[0].endswith(f"-b{block}") and slot_day(k) <= day]


def planned_batches(plan: Sequence[RunSpec]) -> dict[str, list[RunSpec]]:
    """Planned batches by ``block`` label, in plan order."""
    batches: dict[str, list[RunSpec]] = {}
    for spec in plan:
        batches.setdefault(spec.block, []).append(spec)
    return batches


ARM_DAYS = 7  # the local arm starts no batch after day 7 of the arm
CLAUDE_END_DAYS = 45  # the Claude slice starts no run 45 days or more after the freeze
INFRA_STOP_SHARE = 0.05  # an arm stops for repair above this share of infra_error runs ...
INFRA_STOP_MIN = 5  # ... once at least this many are banked since its start or last resume
STOP_KINDS = ("batch_failure", "infra_share", "cli_changed", "defect", "day_limit", "end_point")
FINAL_STOPS = ("day_limit", "end_point")  # stops that end an arm for good
RESUME_HINT = "after a repair: python -m noisefloor.e1 resume %s --note '<what was repaired>'"


class ArmBusyError(RuntimeError):
    """Another runner holds the arm's lock."""


def stops_path(artifact: Path) -> Path:
    """The arm's stop log, next to its artifact (``stops-local.jsonl``, ``stops-claude.jsonl``)."""
    return artifact.with_name(f"stops-{artifact.stem}.jsonl")


@contextmanager
def arm_lock(artifact: Path) -> Iterator[None]:
    """Hold the arm's exclusive lock (a git-ignored ``.<arm>.lock`` beside its artifact)."""
    path = artifact.with_name(f".{artifact.stem}.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ArmBusyError(f"another runner holds {path}") from exc
        yield


def open_stop(stops: Path) -> dict[str, Any] | None:
    """The arm's latest stop, unless a resume follows it."""
    rows = jsonl.read(stops)
    return rows[-1] if rows and rows[-1]["kind"] != "resume" else None


def record_stop(stops: Path, kind: str, reason: str, at: datetime, **fields: Any) -> None:
    """Append a stop of ``kind`` (one of :data:`STOP_KINDS`) to the arm's stop log."""
    if kind not in STOP_KINDS:
        raise ValueError(f"unknown stop kind {kind!r}")
    row = {"at": at.isoformat(timespec="seconds"), "kind": kind, "reason": reason, **fields}
    jsonl.append(stops, [row])


def resume(stops: Path, note: str, at: datetime, amendment: Path | None = None) -> None:
    """Record that the arm resumes after its open stop was repaired.

    A day-limit stop never resumes. A defect stop resumes only under a frozen amendment
    (``amendment``) that says how the affected runs are handled.
    """
    stop = open_stop(stops)
    if stop is None:
        raise ValueError(f"{stops.name}: the arm is not stopped")
    if stop["kind"] in FINAL_STOPS:
        raise ValueError(
            f"the arm has ended ({stop['kind']}: {stop['reason']}); it does not resume"
        )
    row: dict[str, Any] = {"at": at.isoformat(timespec="seconds"), "kind": "resume", "note": note}
    if stop["kind"] == "defect":
        if amendment is None:
            raise ValueError("a defect stop resumes only under a frozen amendment")
        preregs.require_frozen(amendment)
        row["amendment"] = repo_path(amendment)
    jsonl.append(stops, [{**row, "after": stop["at"]}])


def infra_counts(artifact: Path, stops: Path) -> tuple[int, int]:
    """(runs, infra_error runs) banked since the arm started or last resumed.

    A run counts by its batch's start; a record without a start time always counts.
    """
    resumes = [r["at"] for r in jsonl.read(stops) if r["kind"] == "resume"]
    since = datetime.fromisoformat(resumes[-1]) if resumes else None

    def counts(row: Mapping[str, Any]) -> bool:
        started = row.get("batch_started_at") or row.get("started_at")
        return since is None or not started or datetime.fromisoformat(str(started)) >= since

    rows = [r for r in jsonl.read(artifact, tail_ok=True) if counts(r)]
    return len(rows), sum(1 for r in rows if r.get("status") == "infra_error")


def _infra_stop(artifact: Path, stops: Path, at: datetime, log: Callable[[str], None]) -> bool:
    """Stop the arm (and say so) if its infra_error share is over the limit."""
    runs, infra = infra_counts(artifact, stops)
    if infra < INFRA_STOP_MIN or infra <= INFRA_STOP_SHARE * runs:
        return False
    reason = f"{infra} of {runs} runs banked since the arm started or resumed are infra_error"
    record_stop(stops, "infra_share", reason, at, runs=runs, infra_error=infra)
    log(f"e1: stopped for repair: {reason}")
    return True


def run_local(
    prereg: Path,
    *,
    artifact: Path = ARTIFACTS / "local.jsonl",
    ledger: Path = LEDGER,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
    wait: bool = True,
    execute: Callable[..., int] | None = None,
) -> int:
    """Run the local arm of the preregistered plan; returns the number of runs banked now.

    The arm's first batch starts in time-of-day block 1 (00:00 to 12:00 UTC), and day 1 is
    the UTC date on which it starts (before any run is banked, today). A planned batch of
    slot ``day<k>-b<n>`` then starts only in block ``n`` of the arm's day ``k`` or later
    (:func:`eligible_batches`); the earliest unfinished eligible batch runs next, so a slot
    that overruns its block continues in the same block of the next day. Without ``wait``,
    the call returns when no batch is eligible instead of sleeping. Resumable.

    The arm stops, and records why in its stop log (:func:`stops_path`), when a batch
    fails as a whole, when its infra_error share is over the limit (:func:`_infra_stop`),
    or when a batch would start after day :data:`ARM_DAYS`. A stopped arm starts nothing
    until :func:`resume` records a repair. One runner at a time holds the arm's lock.
    """
    from noisefloor.orchestrate import BatchFailedError  # noqa: PLC0415
    from noisefloor.orchestrate import execute as default_execute  # noqa: PLC0415

    run = execute or default_execute
    design = E1Design.from_json(prereg_block(prereg, "e1-plan"))
    stamp = _stamp(prereg, design)
    batches = planned_batches(local_plan(design))
    stops = stops_path(artifact)
    banked = 0
    with arm_lock(artifact):
        while True:
            step = _local_next(batches, artifact, stops, now(), log)
            if step is None:
                return banked
            if isinstance(step, float):
                if not wait:
                    return banked
                sleep(min(step, 600.0))
                continue
            try:
                added = run(
                    batches[step],
                    artifact=artifact,
                    ledger=ledger,
                    prefix="e1-" + step.replace("/", "-"),
                    stamp=stamp,
                    log=log,
                )
            except BatchFailedError as exc:
                banked += exc.banked
                record_stop(stops, "batch_failure", exc.reason, now(), batch=step)
                log(f"e1 local arm: batch {step} failed ({exc.reason}); stopped for repair")
                return banked
            banked += added
            if added == 0:
                reason = "a batch banked nothing (a refused model call)"
                record_stop(stops, "batch_failure", reason, now(), batch=step)
                log(f"e1 local arm: batch {step} banked nothing; stopped for repair")
                return banked


def _local_next(
    batches: Mapping[str, Sequence[RunSpec]],
    artifact: Path,
    stops: Path,
    clock: datetime,
    log: Callable[[str], None],
) -> str | float | None:
    """The next local batch (its label), seconds to wait for one, or None: stop now."""
    from noisefloor.orchestrate import banked_ids  # noqa: PLC0415 - orchestrate imports runner

    done = banked_ids(artifact)
    pending = [k for k, specs in batches.items() if any(s.run_id not in done for s in specs)]
    if not pending:
        log("e1 local arm: every planned run is banked")
        return None
    stop = open_stop(stops)
    start = arm_start(artifact)
    day = 1 if start is None else (clock.date() - start).days + 1
    if day > ARM_DAYS:  # the arm ends, stopped or not
        left = sum(1 for k in pending for s in batches[k] if s.run_id not in done)
        reason = f"day {day} of the arm; no batch starts after day {ARM_DAYS}"
        if stop is None or stop["kind"] != "day_limit":
            record_stop(stops, "day_limit", reason, clock, unbanked=left)
        log(f"e1 local arm: {reason}; {left} planned runs not banked")
        return None
    if stop is not None:
        log(f"e1 local arm: stopped ({stop['kind']}: {stop['reason']}); {RESUME_HINT % 'local'}")
        return None
    if _infra_stop(artifact, stops, clock, log):
        return None
    block = current_block(clock)
    eligible = eligible_batches(pending, block, day)
    if start is None and block != 1:
        eligible = []  # the arm starts at the beginning of a day's first block
    if eligible:
        return eligible[0]
    pause = seconds_to_next_block(clock)
    log(f"e1 local arm: nothing eligible in block b{block}, day {day}; next block in {pause:.0f} s")
    return pause


def frozen_at(prereg: Path) -> datetime | None:
    """When the preregistration was frozen (its chain entry), if it is in the chain."""
    entry = preregs.chain_entry(prereg)
    if entry is None:
        return None
    return datetime.fromisoformat(str(entry["frozen_at"]).replace("Z", "+00:00"))


def served_models(record: Mapping[str, Any]) -> list[str]:
    """Model IDs the CLI reported for a Claude run (either harness)."""
    ids = [str(m) for m in record.get("model_ids") or []]
    child = record.get("claude") or {}
    if child.get("model_id"):
        ids.append(str(child["model_id"]))
    return list(dict.fromkeys(ids))


def claude_changes(artifact: Path, stops: Path) -> tuple[str | None, str | None]:
    """(model change, new CLI version) in the banked Claude runs.

    A tier's first banked run fixes its model ID for the slice; the change is returned as a
    reason. The CLI version of the first banked run, and each version that a
    ``cli_changed`` stop recorded (accepted once resumed), are allowed; the first other
    version is returned.
    """
    rows = jsonl.read(artifact, tail_ok=True)
    first: dict[str, str] = {}
    model_change = None
    for row in rows:
        for model_id in served_models(row):
            known = first.setdefault(str(row["model"]), model_id)
            if model_id != known and model_change is None:
                model_change = f"{row['model']} was served by {model_id} after {known}"
    versions = [str(v) for r in rows if (v := (r.get("claude") or {}).get("cli_version"))]
    accepted = {versions[0]} if versions else set()
    accepted |= {str(s.get("to")) for s in jsonl.read(stops) if s["kind"] == "cli_changed"}
    return model_change, next((v for v in versions if v not in accepted), None)


def _claude_ended_or_changed(
    prereg: Path, artifact: Path, stops: Path, at: datetime, log: Callable[[str], None]
) -> bool:
    """Record and report why the Claude slice must not run now, if it must not."""
    from noisefloor.orchestrate import banked_ids  # noqa: PLC0415 - orchestrate imports runner

    frozen = frozen_at(prereg)
    model_change, new_cli = claude_changes(artifact, stops)
    if frozen is not None and at >= frozen + timedelta(days=CLAUDE_END_DAYS):
        reason = f"{CLAUDE_END_DAYS} days after the freeze"
        record_stop(stops, "end_point", reason, at, banked=len(banked_ids(artifact)))
        log(f"e1 Claude slice: ended ({reason})")
        return True
    if model_change is not None:
        record_stop(stops, "end_point", model_change, at, banked=len(banked_ids(artifact)))
        log(f"e1 Claude slice: ended ({model_change})")
        return True
    if new_cli is not None:
        record_stop(stops, "cli_changed", f"Claude Code {new_cli} in use", at, to=new_cli)
        log(f"e1 Claude slice: stopped (Claude Code {new_cli}); a resume accepts it")
        return True
    return _infra_stop(artifact, stops, at, log)


def run_claude(
    prereg: Path,
    *,
    artifact: Path = ARTIFACTS / "claude.jsonl",
    ledger: Path = CLAUDE_LEDGER,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    log: Callable[[str], None] = print,
    execute: Callable[..., int] | None = None,
) -> int:
    """Run the Claude slice until it is done or a cap, the guard, or a plan limit stops it.

    A batch-level failure or an infra_error share over the limit stops the slice, as in
    :func:`run_local`, until :func:`resume` records a repair; so does a change of the
    Claude Code version. The slice ends for good :data:`CLAUDE_END_DAYS` days after the
    freeze, or when a tier is served by another model ID than its first run's.
    """
    from noisefloor.orchestrate import BatchFailedError  # noqa: PLC0415
    from noisefloor.orchestrate import execute as default_execute  # noqa: PLC0415

    run = execute or default_execute
    design = E1Design.from_json(prereg_block(prereg, "e1-plan"))
    stamp = _stamp(prereg, design)
    stops = stops_path(artifact)
    with arm_lock(artifact):
        stop = open_stop(stops)
        if stop is not None:
            hint = RESUME_HINT % "claude"
            log(f"e1 Claude slice: stopped ({stop['kind']}: {stop['reason']}); {hint}")
            return 0
        if _claude_ended_or_changed(prereg, artifact, stops, now(), log):
            return 0
        try:
            return int(
                run(
                    claude_plan(design),
                    artifact=artifact,
                    ledger=ledger,
                    prefix="e1c",
                    stamp=stamp,
                    log=log,
                )
            )
        except BatchFailedError as exc:
            record_stop(stops, "batch_failure", exc.reason, now(), batch=exc.batch_id)
            log(f"e1 Claude slice: batch {exc.batch_id} failed ({exc.reason}); stopped for repair")
            return exc.banked


def run_dry(
    plan_file: Path,
    *,
    runs: int = 4,
    artifact: Path = DRY_ARTIFACT,
    ledger: Path = LEDGER,
    log: Callable[[str], None] = print,
    execute: Callable[..., int] | None = None,
) -> list[dict[str, Any]]:
    """Run the exploratory dry run (:func:`dry_plan`) batch by batch; returns each batch's fate.

    A batch that fails as a whole is a finding, not a stop: it is logged, listed in
    ``<artifact stem>-batches.jsonl`` with its reason, and the next batch runs.
    """
    from noisefloor.orchestrate import BatchFailedError  # noqa: PLC0415
    from noisefloor.orchestrate import execute as default_execute  # noqa: PLC0415

    run = execute or default_execute
    block = prereg_block(plan_file, "e1-plan")
    design = E1Design.from_json(block)
    digest = hashlib.sha256(json.dumps(block, sort_keys=True).encode()).hexdigest()
    stamp = {"exploratory": True, "dry_run": "e1", "e1_plan_sha256": digest}
    fates = []
    for label, specs in planned_batches(dry_plan(design, runs)).items():
        fate: dict[str, Any] = {"block": label, "runs": len(specs), "failed": None}
        try:
            run(specs, artifact=artifact, ledger=ledger, prefix="e1dry", stamp=stamp, log=log)
        except BatchFailedError as exc:
            fate["failed"] = exc.reason
            log(f"e1 dry run: {label} failed ({exc.reason}); next batch")
        fates.append(fate)
    jsonl.append(artifact.with_name(f"{artifact.stem}-batches.jsonl"), fates)
    return fates


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m noisefloor.e1 plan|local|claude PREREG``, ``stop|resume ARM``, or ``dry-run PLAN``.

    Exit codes: 0 done or waiting, 3 the arm is stopped (``resume`` after a repair), 4
    another runner holds the arm's lock.
    """
    parser = argparse.ArgumentParser(description="E1 factorial study")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "local", "claude"):
        cmd = sub.add_parser(name)
        cmd.add_argument("prereg", type=Path)
        if name == "local":
            cmd.add_argument("--no-wait", action="store_true", help="do not wait for a block")
    stop_cmd = sub.add_parser("stop", help="stop an arm for a defect found while it runs")
    stop_cmd.add_argument("arm", choices=["local", "claude"])
    stop_cmd.add_argument("--reason", required=True)
    resume_cmd = sub.add_parser("resume", help="record a repair; the arm then runs again")
    resume_cmd.add_argument("arm", choices=["local", "claude"])
    resume_cmd.add_argument("--note", required=True, help="what was repaired")
    resume_cmd.add_argument("--amendment", type=Path, help="frozen amendment (defect stops)")
    dry_cmd = sub.add_parser("dry-run", help="exploratory dry run at the E1 server settings")
    dry_cmd.add_argument("plan", type=Path, help="a file with the e1-plan block (a draft works)")
    dry_cmd.add_argument("--runs", type=int, default=4, help="runs per model and setting")
    dry_cmd.add_argument("--summary", action="store_true", help="summarize the banked runs only")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "dry-run":
        if not args.summary:
            run_dry(args.plan, runs=args.runs, log=lambda m: print(m, flush=True))
        for line in dry_summary(jsonl.read(DRY_ARTIFACT, tail_ok=True)):
            print(json.dumps(line, sort_keys=True))
        return 0
    if args.command in ("stop", "resume"):
        stops = stops_path(ARTIFACTS / f"{args.arm}.jsonl")
        at = datetime.now(UTC)
        if args.command == "stop":
            record_stop(stops, "defect", args.reason, at)
        else:
            resume(stops, args.note, at, args.amendment)
        print(f"{stops.name}: {args.command} recorded")
        return 0
    design = E1Design.from_json(prereg_block(args.prereg, "e1-plan"))
    if args.command == "plan":
        lp, cp = local_plan(design), claude_plan(design)
        print(f"local runs {len(lp)} in {len(planned_batches(lp))} batches; Claude runs {len(cp)}")
        print(f"task manifest {manifest(design)['sha256']}")
        print(f"plan sha256 {plan_sha256(design)}")
        return 0
    artifact = ARTIFACTS / f"{args.command}.jsonl"
    say = lambda m: print(m, flush=True)  # noqa: E731
    try:
        if args.command == "local":
            banked = run_local(args.prereg, artifact=artifact, wait=not args.no_wait, log=say)
        else:
            banked = run_claude(args.prereg, artifact=artifact, log=say)
    except ArmBusyError as exc:
        print(f"e1: {exc}", file=sys.stderr)
        return 4
    print(f"banked {banked} runs")
    return 3 if open_stop(stops_path(artifact)) is not None else 0


if __name__ == "__main__":
    sys.exit(main())
