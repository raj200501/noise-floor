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
import json
import random
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

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


def claude_plan(design: E1Design) -> list[RunSpec]:
    """Every Claude-slice run: harness crossed with timeout, balanced within each task.

    Each task's runs come in pairs that differ in both harness and timeout: tasks
    alternate between the pairs (minimal, short) + (native, long) and (minimal, long) +
    (native, short), offset by tier, so harness and timeout are balanced in every tier
    and within every task.
    """
    if design.claude_runs_per_task % 2:
        raise ValueError("claude_runs_per_task must be even")
    rng = random.Random(f"e1-claude:{design.order_seed}")
    pairs = (((0, 0), (1, 1)), ((0, 1), (1, 0)))
    steps = design.levels["steps"][0]
    plan = []
    for ci, tier in enumerate(design.claude_tiers):
        for ti, task in enumerate(tasks(design)):
            combos = pairs[(ci + ti) % 2] * (design.claude_runs_per_task // 2)
            for k, (h, t) in enumerate(combos):
                harness = _level(design, "harness", h)
                timeout = float(_level(design, "timeout", t))
                plan.append(
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
    rng.shuffle(plan)
    return plan


# ---------------------------------------------------------------------------
# running the preregistered plan

ARTIFACTS = REPO / "artifacts" / "e1"
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
    if not artifact.exists():
        return None
    days = [
        date.fromisoformat(str(started)[:10])
        for ln in artifact.read_text().splitlines()
        if ln.strip() and (started := json.loads(ln).get("started_at"))
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

    A planned batch of slot ``day<k>-b<n>`` starts only in time-of-day block ``n`` of the
    arm's day ``k`` or later (:func:`eligible_batches`; day 1 is the UTC date of the first
    banked run, or today before any); the earliest unfinished eligible batch runs next, so a
    slot that overruns its block continues in the same block of the next day, and the
    analysis uses each run's actual start. Without ``wait``, the call returns when no batch
    is eligible instead of sleeping. Resumable.
    """
    from noisefloor.orchestrate import banked_ids  # noqa: PLC0415 - orchestrate imports runner
    from noisefloor.orchestrate import execute as default_execute  # noqa: PLC0415

    run = execute or default_execute
    design = E1Design.from_json(prereg_block(prereg, "e1-plan"))
    batches = planned_batches(local_plan(design))
    stamp = {"prereg": repo_path(prereg), "confirmatory": True}
    banked = 0
    first_day = now().date()
    while True:
        done = banked_ids(artifact)
        pending = [k for k, specs in batches.items() if any(s.run_id not in done for s in specs)]
        if not pending:
            log("e1 local arm: every planned run is banked")
            return banked
        clock = now()
        day = (clock.date() - (arm_start(artifact) or first_day)).days + 1
        block = current_block(clock)
        eligible = eligible_batches(pending, block, day)
        if not eligible:
            if not wait:
                return banked
            pause = seconds_to_next_block(clock)
            log(f"e1 local arm: nothing eligible in block b{block}, day {day}; wait {pause:.0f} s")
            sleep(min(pause, 600.0))
            continue
        label = eligible[0]
        added = run(
            batches[label],
            artifact=artifact,
            ledger=ledger,
            prefix="e1-" + label.replace("/", "-"),
            stamp=stamp,
            log=log,
        )
        banked += added
        if added == 0:
            log(f"e1 local arm: batch {label} banked nothing (refused or failed); stopping")
            return banked


def run_claude(
    prereg: Path,
    *,
    artifact: Path = ARTIFACTS / "claude.jsonl",
    ledger: Path = LEDGER,
    log: Callable[[str], None] = print,
    execute: Callable[..., int] | None = None,
) -> int:
    """Run the Claude slice until it is done or a cap, the guard, or a plan limit stops it."""
    from noisefloor.orchestrate import execute as default_execute  # noqa: PLC0415

    run = execute or default_execute
    design = E1Design.from_json(prereg_block(prereg, "e1-plan"))
    stamp = {"prereg": repo_path(prereg), "confirmatory": True}
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


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m noisefloor.e1 plan|local|claude PREREG [--no-wait]``."""
    parser = argparse.ArgumentParser(description="E1 factorial study")
    parser.add_argument("command", choices=["plan", "local", "claude"])
    parser.add_argument("prereg", type=Path)
    parser.add_argument("--no-wait", action="store_true", help="local: do not wait for a block")
    args = parser.parse_args(list(argv) if argv is not None else None)
    design = E1Design.from_json(prereg_block(args.prereg, "e1-plan"))
    if args.command == "plan":
        lp, cp = local_plan(design), claude_plan(design)
        print(f"local runs {len(lp)} in {len(planned_batches(lp))} batches; Claude runs {len(cp)}")
        print(f"task manifest {manifest(design)['sha256']}")
        return 0
    if args.command == "local":
        banked = run_local(args.prereg, wait=not args.no_wait, log=lambda m: print(m, flush=True))
    else:
        banked = run_claude(args.prereg, log=lambda m: print(m, flush=True))
    print(f"banked {banked} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
