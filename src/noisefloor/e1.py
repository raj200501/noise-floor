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

Claude slice: each Claude tier runs the first ``claude_tasks_per_family`` tasks of each family
``claude_runs_per_task`` times, crossing harness (minimal-agent or Claude Code) with timeout;
the step budget is the turn cap and temperature is the CLI's default (neither can vary for
Claude here, D-019).

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
from noisefloor.client import CLAUDE_LEDGER, billing_guard_findings
from noisefloor.models import CLAUDE_TIERS, DEFAULT_CTX, MODELS, RUNTIME, server_build_sha256
from noisefloor.preregs import REPO, prereg_block, repo_path
from noisefloor.recovery import TWO_LEVEL, cell_runs, deal_batches, half_fraction_blocks
from noisefloor.runner import ABORT_STREAK, RunSpec
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
    # Configurations (0/1 codes in TWO_LEVEL order) that every (model, task) cell runs a
    # second time, phase "e1-rep", to measure run-to-run sampling (D-028 item 3).
    replicates: tuple[tuple[int, ...], ...] = ()
    # The Claude slice runs the first this many tasks of each family (None: all of them),
    # because its caps bind while the local arm's size comes from the power analysis.
    claude_tasks_per_family: int | None = None

    def __post_init__(self) -> None:
        """Reject designs E1 cannot run."""
        per = self.claude_tasks_per_family
        if per is not None and not 1 <= per <= self.tasks_per_family:
            raise ValueError("claude_tasks_per_family must be between 1 and tasks_per_family")
        if self.runs_per_cell not in (8, 16):
            raise ValueError("runs_per_cell must be 8 (one block) or 16 (the half fraction)")
        for bits in self.replicates:
            if len(bits) != len(TWO_LEVEL) or set(bits) - {0, 1} or sum(bits) % 2:
                raise ValueError(f"replicate {bits} is not a run of the half fraction")
        if self.replicates and self.runs_per_cell != 16:
            raise ValueError("replicates need the whole half fraction in every cell")
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
            replicates=tuple(tuple(int(b) for b in r) for r in data.get("replicates", [])),
            claude_tasks_per_family=(
                None if (per := data.get("claude_tasks_per_family")) is None else int(per)
            ),
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
            "replicates": [list(r) for r in self.replicates],
            "claude_tasks_per_family": self.claude_tasks_per_family,
        }


def _family_tasks(design: E1Design, per_family: int) -> list[Task]:
    out = []
    for family in design.families:
        for j in range(per_family):
            difficulty = design.difficulties[j % len(design.difficulties)]
            out.append(generate(family, CONFIRMATORY_SEEDS[j], difficulty))
    return out


def tasks(design: E1Design) -> list[Task]:
    """The confirmatory tasks: per family, seeds in order with difficulties in rotation."""
    return _family_tasks(design, design.tasks_per_family)


def claude_tasks(design: E1Design) -> list[Task]:
    """The Claude slice's tasks: the first ``claude_tasks_per_family`` tasks of each family.

    They are a subset of :func:`tasks` (the same seeds and difficulties), so every Claude
    task is also a local-arm task.
    """
    return _family_tasks(design, design.claude_tasks_per_family or design.tasks_per_family)


def manifest(design: E1Design) -> dict[str, Any]:
    """The task manifest (hash of every task) for the preregistration."""
    return suite_manifest(tasks(design))


def plan_sha256(design: E1Design) -> str:
    """SHA-256 of the canonical JSON of every planned run (local, then Claude)."""
    specs = [s.to_json() for s in (*local_plan(design), *claude_plan(design))]
    text = json.dumps(specs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def model_pins(design: E1Design) -> dict[str, dict[str, Any]]:
    """Weights (repository, revision, file, SHA-256) and server settings of every local model.

    The server settings are the model's own llama-server arguments (for the Qwen3 models,
    thinking off through the chat template) and the context per parallel slot; every server
    also runs with ``--jinja`` and ``--seed 0`` (:func:`noisefloor.models.server_command`).
    """
    return {
        m: {
            "repo": MODELS[m].repo,
            "revision": MODELS[m].revision,
            "filename": MODELS[m].filename,
            "sha256": MODELS[m].sha256,
            "server_args": list(MODELS[m].server_args),
            "ctx_per_slot": DEFAULT_CTX,
        }
        for m in design.models
    }


def server_pin() -> dict[str, Any]:
    """The llama-server build (:data:`noisefloor.models.RUNTIME`) and its SHA-256.

    The digest covers the binary and the shared libraries beside it
    (:func:`noisefloor.models.server_build_sha256`).
    """
    return {**RUNTIME, "sha256": server_build_sha256()}


def run_code(local: bool) -> dict[str, Any]:
    """The run code a batch would use now (:func:`noisefloor.orchestrate.code_provenance`)."""
    from noisefloor.orchestrate import code_provenance  # noqa: PLC0415

    found = code_provenance()
    if not local:
        found.pop("server_sha256", None)  # the Claude slice starts no model server
    return found


def code_key(code: Mapping[str, Any], local: bool) -> str:
    """One string for the run code an arm compares across its batches."""
    parts = [code.get("src_tree"), code.get("uv_lock")]
    if local:
        parts.append(code.get("server_sha256"))
    return ":".join(str(p) for p in parts)


def code_change(artifact: Path, stops: Path, current: str, local: bool) -> str | None:
    """``current`` if it differs from the run code of the arm's first banked run.

    Codes a resumed ``code_changed`` stop recorded are accepted, as for CLI versions.
    """
    keys = [
        code_key(code, local)
        for row in jsonl.read(artifact, tail_ok=True)
        if (code := (row.get("provenance") or {}).get("code"))
    ]
    if not keys:
        return None
    accepted = {keys[0]} | {
        str(s.get("to")) for s in jsonl.read(stops) if s["kind"] == "code_changed"
    }
    return None if current in accepted else current


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
    tasks = manifest(design)
    expected = {
        "plan_sha256": (plan_sha256(design), "the run plan from this code differs from"),
        "model_pins": (model_pins(design), "the local model pins differ from"),
        "task_manifest_sha256": (tasks["sha256"], "the generated tasks differ from"),
        "suite_version": (tasks["version"], "the task suite version differs from"),
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


REPLICATE_PHASE = "e1-rep"


def local_plan(design: E1Design) -> list[RunSpec]:
    """Every local-arm run, in execution order (slot by slot, batch by batch).

    A replicate run (phase ``e1-rep``, run id ``<planned run id>-rep``) repeats one cell
    configuration with its own sampling seed and is batched like any other run.
    """
    rng = random.Random(f"e1-local:{design.order_seed}")
    first, second = half_fraction_blocks()
    task_list = tasks(design)
    runs: list[RunSpec] = []
    for mi, model in enumerate(design.models):
        for ti, task in enumerate(task_list):
            half = cell_runs(first, second, mi + ti, design.runs_per_cell)
            for bits in half:
                runs.append(_local_spec(design, model, task, bits, rng.randrange(2**31)))
            for bits in design.replicates:
                rep = _local_spec(design, model, task, bits, rng.randrange(2**31)).to_json()
                rep.update(run_id=f"{rep['run_id']}-rep", phase=REPLICATE_PHASE)
                runs.append(RunSpec.from_json(rep))
    groups: dict[tuple[str, int], list[RunSpec]] = {}
    for spec in runs:
        groups.setdefault((spec.model, spec.concurrency), []).append(spec)
    slots = design.slots
    # Each (model, concurrency) group spreads its batches evenly over the slots, by the
    # rule the G6 and E0 part 2 simulator uses too.
    dealt = deal_batches(groups, len(slots), design.batch_size, rng.shuffle, rng.randrange)
    planned = [(slots[index], chunk) for index, chunk in dealt]
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

    The slice runs :func:`claude_tasks`. Each task's runs come in pairs that differ in
    both harness and timeout: tasks alternate between the pairs (minimal, short) +
    (native, long) and (minimal, long) + (native, short), the same for every tier, so
    harness and timeout are balanced in every tier and within every task, and two tiers
    under one harness always ran the same timeout on a task. Runs are ordered by
    complete (tier, task) groups with the tiers interleaved, so a slice that ends early
    leaves complete pairs balanced over tiers. ``max_tokens`` is 0 on every Claude run:
    the reply cap reaches only the local server, and the Claude CLI sets its own.
    """
    if design.claude_runs_per_task % 2:
        raise ValueError("claude_runs_per_task must be even")
    rng = random.Random(f"e1-claude:{design.order_seed}")
    pairs = (((0, 0), (1, 1)), ((0, 1), (1, 0)))
    steps = design.levels["steps"][0]
    task_list = claude_tasks(design)
    groups: dict[tuple[int, int], list[RunSpec]] = {}
    for ci, tier in enumerate(design.claude_tiers):
        for ti, task in enumerate(task_list):
            combos = pairs[ti % 2] * (design.claude_runs_per_task // 2)
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
                        max_tokens=0,  # not applied: the Claude CLI takes no reply cap
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


START_GRACE_S = 900.0  # the arm's first batch starts at most this long after 00:00 UTC


def seconds_since_midnight(now: datetime) -> float:
    """Seconds since the last 00:00 of ``now``'s date (UTC for the E1 clock)."""
    return (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()


def seconds_to_midnight(now: datetime) -> float:
    """Seconds from ``now`` until the next 00:00 (at least 1)."""
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    return max(1.0, (midnight - now).total_seconds())


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
STOP_KINDS = (
    "batch_failure",
    "infra_share",
    "cli_changed",
    "code_changed",
    "contaminated",
    "defect",
    "day_limit",
    "end_point",
    "tier_ended",
    "note",
)
FINAL_STOPS = ("day_limit", "end_point")  # stops that end an arm for good
NOT_STOPS = ("tier_ended", "note")  # rows of the stop log that stop nothing
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
    """The arm's latest stop, unless a resume follows it (:data:`NOT_STOPS` rows are not stops).

    A ``note`` records an event that stops nothing, such as banked runs lost with a
    container before their push, which then run again under their planned run ids (a
    deviation, prereg 0004).
    """
    rows = [r for r in jsonl.read(stops) if r["kind"] not in NOT_STOPS]
    return rows[-1] if rows and rows[-1]["kind"] != "resume" else None


def ended_tiers(stops: Path) -> set[str]:
    """Claude tiers that start no further run (another model ID served them)."""
    return {str(r["tier"]) for r in jsonl.read(stops) if r["kind"] == "tier_ended"}


def record_stop(stops: Path, kind: str, reason: str, at: datetime, **fields: Any) -> None:
    """Append a stop of ``kind`` (one of :data:`STOP_KINDS`) to the arm's stop log."""
    if kind not in STOP_KINDS:
        raise ValueError(f"unknown stop kind {kind!r}")
    row = {"at": at.isoformat(timespec="seconds"), "kind": kind, "reason": reason, **fields}
    jsonl.append(stops, [row])


E1_PREREG = "0004"  # the preregistration this module runs (amendments must amend it)
SMOKE_ARTIFACT = REPO / "artifacts" / "pilot" / "claude-smoke.jsonl"


def smoke_passed(version: str | None = None, artifact: Path = SMOKE_ARTIFACT) -> bool:
    """True if a banked smoke run printed SMOKE: PASS (for ``version`` of the CLI, if given)."""
    from noisefloor.claude_arm import smoke_report  # noqa: PLC0415

    for record in jsonl.read(artifact, tail_ok=True):
        same = version is None or (record.get("claude") or {}).get("cli_version") == version
        if same and smoke_report(record)[-1] == "SMOKE: PASS":
            return True
    return False


def resume(
    stops: Path,
    note: str,
    at: datetime,
    amendment: Path | None = None,
    smoke: Callable[[str | None], bool] = smoke_passed,
) -> None:
    """Record that the arm resumes after its open stop was repaired.

    A day-limit or end-point stop never resumes. A defect stop resumes only under a frozen
    amendment of this preregistration (``amendment``) that says how the affected runs are
    handled and names the stop it answers: its ```json defect-resume``` block gives the
    stop's ``at`` (one time, or a list). A new Claude Code version is accepted only after a
    smoke run on it printed SMOKE: PASS (``python -m noisefloor.claude_arm smoke``).
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
        if preregs.amends(amendment) != E1_PREREG:
            raise ValueError(f"{amendment.name} does not amend prereg {E1_PREREG}")
        answers = prereg_block(amendment, "defect-resume").get("at")
        if stop["at"] not in (answers if isinstance(answers, list) else [answers]):
            raise ValueError(
                f"{amendment.name} answers the defect stop at {answers}, not the one at "
                f"{stop['at']}"
            )
        row["amendment"] = repo_path(amendment)
    if stop["kind"] == "cli_changed":
        if not smoke(str(stop.get("to"))):
            raise ValueError(f"no SMOKE: PASS on Claude Code {stop.get('to')}; run the smoke first")
        row["smoke"] = "pass"
    jsonl.append(stops, [{**row, "after": stop["at"]}])


def require_validation(prereg: Path) -> None:
    """Refuse to start an arm before G6 and E0 part 2 are banked (prereg 0004).

    Then a kind of interval that fails its cells can still change method by a dated
    amendment before any E1 run exists. A file outside the chain (a fixture) is not checked.
    """
    if preregs.chain_entry(prereg) is None:
        return
    from noisefloor.analysis.e1 import validation_artifacts  # noqa: PLC0415 - imports e1

    for name, path in validation_artifacts(prereg_block(prereg, "e1-analysis")).items():
        if not path.exists():
            raise ValueError(f"{name}: {path.name} is not banked; G6 and E0 part 2 come first")


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


def _code_stop(
    artifact: Path, stops: Path, clock: datetime, log: Callable[[str], None], local: bool
) -> bool:
    """True when no batch may start: uncommitted run code, or code the arm did not start with.

    Uncommitted changes under ``src/`` (or to the lock or project file) block every batch
    until they are committed; run code other than the arm's first banked run's (and, for
    the local arm, another llama-server binary) stops the arm until a resume accepts it,
    reported as a deviation (prereg 0004).
    """
    arm = "local" if local else "claude"
    code = run_code(local)
    if code["dirty"]:
        log(f"e1 {arm}: src/, uv.lock, or pyproject.toml has uncommitted changes; commit first")
        return True
    key = code_key(code, local)
    changed = code_change(artifact, stops, key, local)
    if changed is None:
        return False
    record_stop(stops, "code_changed", f"the run code changed to {changed}", clock, to=changed)
    log(f"e1 {arm}: stopped (the run code changed); {RESUME_HINT % arm}")
    return True


def check_server(prereg: Path, artifact: Path) -> str | None:
    """Before the local arm's first run, the llama-server build must be the frozen one.

    The build's digest (its binary and the shared libraries beside it,
    :func:`noisefloor.models.server_build_sha256`) must equal the frozen ``llama_server``
    pin, or the ``sha256`` of a ```json llama-server``` block in a frozen amendment of this
    preregistration (a rebuilt server, a deviation); returns that amendment, if one was
    needed. After the first run, a changed build is a ``code_changed`` stop instead.
    """
    if arm_start(artifact) is not None or preregs.chain_entry(prereg) is None:
        return None
    pinned = prereg_block(prereg, "e1-design").get("llama_server", {}).get("sha256")
    actual = server_pin()["sha256"]
    if pinned == actual and actual is not None:
        return None
    for amendment in preregs.amendments_with(E1_PREREG, "llama-server"):
        preregs.require_frozen(amendment)
        if actual is not None and prereg_block(amendment, "llama-server").get("sha256") == actual:
            return repo_path(amendment)
    raise ValueError(f"the llama-server build ({actual}) is not the frozen one ({pinned})")


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

    The arm's first batch starts at the beginning of time-of-day block 1, at most
    :data:`START_GRACE_S` after 00:00 UTC (launched later, the runner waits for the next
    00:00 UTC), and day 1 is the UTC date on which it starts. A planned batch of
    slot ``day<k>-b<n>`` then starts only in block ``n`` of the arm's day ``k`` or later
    (:func:`eligible_batches`); the earliest unfinished eligible batch runs next, so a slot
    that overruns its block continues in the same block of the next day. Without ``wait``,
    the call returns when no batch is eligible instead of sleeping. Resumable.

    The arm stops, and records why in its stop log (:func:`stops_path`), when a batch
    fails as a whole, when its infra_error share is over the limit (:func:`_infra_stop`),
    or when a batch would start after day :data:`ARM_DAYS`. A stopped arm starts nothing
    until :func:`resume` records a repair. One runner at a time holds the arm's lock.
    """
    from noisefloor.orchestrate import BatchFailedError, RunnerBusyError  # noqa: PLC0415
    from noisefloor.orchestrate import execute as default_execute  # noqa: PLC0415

    run = execute or default_execute
    design = E1Design.from_json(prereg_block(prereg, "e1-plan"))
    stamp = _stamp(prereg, design)
    require_validation(prereg)
    allowed = check_server(prereg, artifact)
    if allowed:
        log(f"e1 local arm: the llama-server build is the one {allowed} names (a deviation)")
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
            except RunnerBusyError as exc:  # a runner left by a killed orchestrator
                log(f"e1 local arm: {exc}; waiting for it to finish")
                if not wait:
                    return banked
                sleep(60.0)
                continue
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
    if _infra_stop(artifact, stops, clock, log) or _code_stop(artifact, stops, clock, log, True):
        return None
    block = current_block(clock)
    # The arm starts at the beginning of a day's first block, so day 1 has a full one.
    first = start is None and seconds_since_midnight(clock) > START_GRACE_S
    eligible = [] if first else eligible_batches(pending, block, day)
    if eligible:
        return eligible[0]
    if first:
        pause = seconds_to_midnight(clock)
        log(f"e1 local arm: the first batch starts at 00:00 UTC, in {pause:.0f} s")
    else:
        pause = seconds_to_next_block(clock)
        log(f"e1 local arm: nothing eligible in block b{block}, day {day}; wait {pause:.0f} s")
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


def cli_versions(record: Mapping[str, Any]) -> list[str]:
    """Claude Code versions a Claude run reported (minimal-agent calls or the Claude child)."""
    found = [str(v) for v in record.get("cli_versions") or []]
    child = (record.get("claude") or {}).get("cli_version")
    if child:
        found.append(str(child))
    return list(dict.fromkeys(found))


def claude_changes(
    artifact: Path, stops: Path, smoke: Callable[[str | None], bool] | None = None
) -> tuple[dict[str, str], str | None]:
    """(model changes by tier, new CLI version) in the banked Claude runs.

    A tier's first banked run fixes its model ID for the slice; each tier later served by
    another ID maps to a reason. The CLI version of the first banked run is allowed when a
    smoke run on that version passed (``smoke``), and so is each version that a
    ``cli_changed`` stop recorded (accepted once resumed, which needs a smoke run on it);
    the first other version is returned.
    """
    rows = jsonl.read(artifact, tail_ok=True)
    first: dict[str, str] = {}
    changes: dict[str, str] = {}
    for row in rows:
        tier = str(row.get("model", ""))
        for model_id in served_models(row):
            known = first.setdefault(tier, model_id)
            if model_id != known and tier not in changes:
                changes[tier] = f"{tier} was served by {model_id} after {known}"
    versions = [v for r in rows for v in cli_versions(r)]
    accepted = {str(s.get("to")) for s in jsonl.read(stops) if s["kind"] == "cli_changed"}
    passed = smoke_passed if smoke is None else smoke
    if versions and versions[0] not in accepted and passed(versions[0]):
        accepted.add(versions[0])
    return changes, next((v for v in versions if v not in accepted), None)


def since_resume(artifact: Path, stops: Path) -> list[dict[str, Any]]:
    """Banked runs, in banking order, from the arm's start or last resume on."""
    resumes = [r["at"] for r in jsonl.read(stops) if r["kind"] == "resume"]
    since = datetime.fromisoformat(resumes[-1]) if resumes else None
    rows = jsonl.read(artifact, tail_ok=True)
    if since is None:
        return rows

    def after(row: Mapping[str, Any]) -> bool:
        started = row.get("batch_started_at") or row.get("started_at")
        return not started or datetime.fromisoformat(str(started)) >= since

    return [r for r in rows if after(r)]


def claude_problem(
    prereg: Path, artifact: Path, stops: Path, at: datetime
) -> tuple[str, str, dict[str, Any]] | None:
    """(stop kind, reason, fields) of the first rule that stops the Claude slice now.

    In order: the end point (:data:`CLAUDE_END_DAYS` days after the freeze), a new
    Claude Code version (or a first one no smoke run passed on), a child that did not
    start clean, and three infra_error runs in a row anywhere in banking order since the
    slice started or resumed (the batch-level rule of the local arm, across the two-run
    Claude batches; those runs stay banked). A tier served by another model ID ends that
    tier only (:func:`note_tier_changes`).
    """
    from noisefloor.orchestrate import banked_ids  # noqa: PLC0415 - orchestrate imports runner

    frozen = frozen_at(prereg)
    _, new_cli = claude_changes(artifact, stops)
    count = {"banked": len(banked_ids(artifact))}
    if frozen is not None and at >= frozen + timedelta(days=CLAUDE_END_DAYS):
        return "end_point", f"{CLAUDE_END_DAYS} days after the freeze", count
    if new_cli is not None:
        return "cli_changed", f"Claude Code {new_cli} in use", {"to": new_cli}
    recent = since_resume(artifact, stops)
    dirty = [r["run_id"] for r in recent if (r.get("claude") or {}).get("clean") is False]
    if dirty:
        return "contaminated", f"run {dirty[0]} did not start clean", {"runs": dirty}
    streak = longest = 0
    for row in recent:  # banking order: a streak may end inside a pair
        streak = streak + 1 if row.get("status") == "infra_error" else 0
        longest = max(longest, streak)
    if longest >= ABORT_STREAK:
        return "batch_failure", f"{longest} infra_error runs in a row", {}
    return None


def note_tier_changes(
    artifact: Path, stops: Path, at: datetime, log: Callable[[str], None]
) -> set[str]:
    """Record a ``tier_ended`` note for each tier newly served by another model ID.

    That tier starts no further run, its runs on the other ID are left out of the analyses
    (counted per tier), and the other tiers go on (prereg 0004). Returns every ended tier.
    """
    changes, _ = claude_changes(artifact, stops)
    ended = ended_tiers(stops)
    for tier, reason in sorted(changes.items()):
        if tier not in ended:
            record_stop(stops, "tier_ended", reason, at, tier=tier)
            log(f"e1 Claude slice: {reason}; {tier} starts no further run")
            ended.add(tier)
    return ended


def _claude_ended_or_changed(
    prereg: Path, artifact: Path, stops: Path, at: datetime, log: Callable[[str], None]
) -> bool:
    """Record and report why the Claude slice must not run now, if it must not."""
    problem = claude_problem(prereg, artifact, stops, at)
    if problem is None:
        return _infra_stop(artifact, stops, at, log)
    kind, reason, fields = problem
    record_stop(stops, kind, reason, at, **fields)
    ended = kind in FINAL_STOPS
    log(f"e1 Claude slice: {'ended' if ended else 'stopped'} ({reason})")
    return True


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

    The plan runs one (tier, task) group at a time, and before each group the slice's
    stops are checked again (:func:`_claude_may_start`): a batch-level failure or an
    infra_error share over the limit stops the slice, as in :func:`run_local`, until
    :func:`resume` records a repair; so do a change of the Claude Code version or of the
    run code. The slice ends for good :data:`CLAUDE_END_DAYS` days after the freeze, when
    a tier is served by another model ID than its first run's, or when every planned run
    is banked (an ``end_point`` stop the analysis reads). A group left unfinished (a cap or
    the guard refused a call) ends the call; the next call starts with that group's
    remaining runs.
    """
    from noisefloor.orchestrate import banked_ids  # noqa: PLC0415
    from noisefloor.orchestrate import execute as default_execute  # noqa: PLC0415

    run = execute or default_execute
    design = E1Design.from_json(prereg_block(prereg, "e1-plan"))
    stamp = _stamp(prereg, design)
    require_validation(prereg)
    if preregs.chain_entry(prereg) is not None and not smoke_passed():
        raise ValueError(
            "the slice starts after SMOKE: PASS (python -m noisefloor.claude_arm smoke)"
        )
    stops = stops_path(artifact)
    banked = 0
    with arm_lock(artifact):
        if _claude_day_end(prereg, artifact, stops, now(), log):
            return banked
        for group in claude_groups(claude_plan(design)):
            if all(s.run_id in banked_ids(artifact) for s in group):
                continue
            if group[0].model in note_tier_changes(artifact, stops, now(), log):
                continue
            if not _claude_may_start(prereg, artifact, stops, now(), log):
                return banked
            added, go_on = _claude_group(
                run, group, artifact=artifact, ledger=ledger, stamp=stamp, now=now, log=log
            )
            banked += added
            if not go_on:
                return banked
        last = open_stop(stops)
        if last is None or last["kind"] not in FINAL_STOPS:
            reason = "every planned run of the tiers still served is banked"
            record_stop(stops, "end_point", reason, now(), banked=len(banked_ids(artifact)))
        log("e1 Claude slice: every planned run is banked")
        return banked


def _claude_day_end(
    prereg: Path, artifact: Path, stops: Path, at: datetime, log: Callable[[str], None]
) -> bool:
    """True (after recording the end point once) from :data:`CLAUDE_END_DAYS` days on.

    The slice ends then whether or not it is stopped, as the local arm ends on its day
    limit (prereg 0004), so a stop open at that time does not keep the analysis waiting.
    """
    from noisefloor.orchestrate import banked_ids  # noqa: PLC0415 - orchestrate imports runner

    frozen = frozen_at(prereg)
    if frozen is None or at < frozen + timedelta(days=CLAUDE_END_DAYS):
        return False
    stop = open_stop(stops)
    if stop is None or stop["kind"] not in FINAL_STOPS:
        reason = f"{CLAUDE_END_DAYS} days after the freeze"
        record_stop(stops, "end_point", reason, at, banked=len(banked_ids(artifact)))
    log(f"e1 Claude slice: ended ({CLAUDE_END_DAYS} days after the freeze, stopped or not)")
    return True


def _claude_may_start(
    prereg: Path, artifact: Path, stops: Path, at: datetime, log: Callable[[str], None]
) -> bool:
    """False (after saying why) when the next Claude pair must not start."""
    if _claude_day_end(prereg, artifact, stops, at, log):
        return False
    stop = open_stop(stops)
    if stop is not None:
        hint = RESUME_HINT % "claude"
        log(f"e1 Claude slice: stopped ({stop['kind']}: {stop['reason']}); {hint}")
        return False
    if _claude_ended_or_changed(prereg, artifact, stops, at, log):
        return False
    return not _code_stop(artifact, stops, at, log, False)


def _claude_group(
    run: Callable[..., int],
    group: Sequence[RunSpec],
    *,
    artifact: Path,
    ledger: Path,
    stamp: Mapping[str, Any],
    now: Callable[[], datetime],
    log: Callable[[str], None],
) -> tuple[int, bool]:
    """Run one (tier, task) group: (runs banked, whether the slice goes on)."""
    from noisefloor.orchestrate import (  # noqa: PLC0415
        BatchFailedError,
        RunnerBusyError,
        banked_ids,
    )

    stops = stops_path(artifact)
    try:
        added = int(
            run(group, artifact=artifact, ledger=ledger, prefix="e1c", stamp=stamp, log=log)
        )
    except BatchFailedError as exc:
        record_stop(stops, "batch_failure", exc.reason, now(), batch=exc.batch_id)
        log(f"e1 Claude slice: batch {exc.batch_id} failed ({exc.reason}); stopped")
        return exc.banked, False
    except RunnerBusyError as exc:  # a runner left by a killed orchestrator
        log(f"e1 Claude slice: {exc}; run again when it has finished")
        return 0, False
    if not all(s.run_id in banked_ids(artifact) for s in group):
        log("e1 Claude slice: a model call was refused (cap or guard); next call resumes")
        return added, False
    return added, True


def claude_groups(plan: Sequence[RunSpec]) -> list[list[RunSpec]]:
    """Consecutive runs of one tier and task in :func:`claude_plan` order."""
    groups: list[list[RunSpec]] = []
    for spec in plan:
        key = (spec.model, spec.family, spec.task_seed, spec.difficulty)
        last = groups[-1][-1] if groups else None
        if last is not None and (last.model, last.family, last.task_seed, last.difficulty) == key:
            groups[-1].append(spec)
        else:
            groups.append([spec])
    return groups


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


def _parser() -> argparse.ArgumentParser:
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
    end_cmd = sub.add_parser("end", help="end the Claude slice early (a deviation)")
    end_cmd.add_argument("arm", choices=["claude"])
    end_cmd.add_argument("--reason", required=True)
    note_cmd = sub.add_parser("note", help="record an event that stops nothing (a deviation)")
    note_cmd.add_argument("arm", choices=["local", "claude"])
    note_cmd.add_argument("--reason", required=True)
    dry_cmd = sub.add_parser("dry-run", help="exploratory dry run at the E1 server settings")
    dry_cmd.add_argument("plan", type=Path, help="a file with the e1-plan block (a draft works)")
    dry_cmd.add_argument("--runs", type=int, default=4, help="runs per model and setting")
    dry_cmd.add_argument("--summary", action="store_true", help="summarize the banked runs only")
    return parser


def _stop_command(args: argparse.Namespace) -> int:
    """``stop``, ``resume``, ``end``, or ``note`` of an arm (the Claude slice's where it runs)."""
    if args.arm == "claude" and billing_guard_findings():
        print("e1: the Claude slice's stops are recorded where it runs", file=sys.stderr)
        return 5
    stops = stops_path(ARTIFACTS / f"{args.arm}.jsonl")
    at = datetime.now(UTC)
    if args.command == "stop":
        record_stop(stops, "defect", args.reason, at)
    elif args.command == "end":
        record_stop(stops, "end_point", f"ended early: {args.reason}", at, early=True)
    elif args.command == "note":
        record_stop(stops, "note", args.reason, at)
    else:
        resume(stops, args.note, at, args.amendment)
    print(f"{stops.name}: {args.command} recorded")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m noisefloor.e1 COMMAND``: plan, local, claude, stop, resume, end, note, dry-run.

    ``plan|local|claude PREREG``, ``stop|resume|end|note ARM``, and ``dry-run PLAN``. Exit
    codes: 0 done or waiting, 3 the arm is stopped (``resume`` after a repair), 4 another
    runner holds the arm's lock, 5 a Claude-slice stop, resume, end, or note refused on a
    machine where the billing guard is not clean (the slice's files are written only on
    the machine that runs it).
    """
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "dry-run":
        if not args.summary:
            run_dry(args.plan, runs=args.runs, log=lambda m: print(m, flush=True))
        for line in dry_summary(jsonl.read(DRY_ARTIFACT, tail_ok=True)):
            print(json.dumps(line, sort_keys=True))
        return 0
    if args.command in ("stop", "resume", "end", "note"):
        return _stop_command(args)
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
