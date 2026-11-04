"""E1 factorial study: design, run plan, and schedule (docs/BUILD_BRIEF.md section 5).

Local arm: every model is crossed with every task, and each (model, task) cell runs one
block of a half fraction 2^(5-1) of five two-level factors
(:func:`noisefloor.recovery.half_fraction_blocks`, the same blocks the G6 recovery study
simulates): harness, wall-clock timeout, step budget, sampling temperature, and
concurrency. Within a cell every factor is balanced (4 runs at each level). Runs are
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

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from noisefloor.models import CLAUDE_TIERS, MODELS
from noisefloor.recovery import TWO_LEVEL, half_fraction_blocks
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

    def __post_init__(self) -> None:
        """Reject designs E1 cannot run."""
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
            half = first if (mi + ti) % 2 == 0 else second
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
