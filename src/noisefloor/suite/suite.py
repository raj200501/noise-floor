"""Build nf-tools task sets, split pilot from confirmatory seeds, and pin them in a manifest.

The pilot (which sets difficulty) and the confirmatory study use disjoint seed
ranges chosen by the caller; :func:`split` raises if they overlap.
:func:`suite_manifest` lists the task ids and a SHA-256 over a canonical JSON
serialization of every task, so a preregistration can pin the exact
confirmatory task set and anyone can check it by rebuilding the tasks.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from noisefloor.suite.tasks import DIFFICULTIES, FAMILIES, FAMILY_NAMES, Task, generate

SUITE_NAME = "nf-tools"
SUITE_VERSION = "2"  # D-021: grader, pointer, derive, and tool fixes


def _seed_list(seeds: range | Sequence[int]) -> list[int]:
    values = list(seeds)
    for seed in values:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"seeds must be non-negative integers, got {seed!r}")
    if len(set(values)) != len(values):
        raise ValueError("seeds contain duplicates")
    return values


def build_suite(
    seeds: range | Sequence[int],
    families: Sequence[str] = FAMILY_NAMES,
    difficulties: Sequence[int] = DIFFICULTIES,
) -> list[Task]:
    """Every task for the given seeds, families, and difficulties.

    Tasks are ordered by family, then difficulty, then seed, each in the order
    given. Raises ValueError for empty, duplicate, or invalid arguments.
    """
    seed_list = _seed_list(seeds)
    if not seed_list or not families or not difficulties:
        raise ValueError("seeds, families, and difficulties must be non-empty")
    if len(set(families)) != len(families) or len(set(difficulties)) != len(difficulties):
        raise ValueError("families and difficulties must not repeat")
    unknown = [family for family in families if family not in FAMILIES]
    if unknown:
        raise ValueError(f"unknown families: {', '.join(unknown)}")
    return [
        generate(family, seed, difficulty)
        for family in families
        for difficulty in difficulties
        for seed in seed_list
    ]


def check_disjoint(
    pilot_seeds: range | Sequence[int], confirmatory_seeds: range | Sequence[int]
) -> None:
    """Raise ValueError if any seed is in both the pilot and the confirmatory set."""
    overlap = sorted(set(_seed_list(pilot_seeds)) & set(_seed_list(confirmatory_seeds)))
    if overlap:
        shown = ", ".join(str(seed) for seed in overlap[:10])
        more = f" and {len(overlap) - 10} more" if len(overlap) > 10 else ""
        raise ValueError(f"pilot and confirmatory seeds overlap: {shown}{more}")


@dataclass(frozen=True)
class Split:
    """Pilot and confirmatory task sets built from disjoint seeds."""

    pilot: tuple[Task, ...]
    confirmatory: tuple[Task, ...]


def split(
    pilot_seeds: range | Sequence[int],
    confirmatory_seeds: range | Sequence[int],
    families: Sequence[str] = FAMILY_NAMES,
    difficulties: Sequence[int] = DIFFICULTIES,
) -> Split:
    """Build pilot and confirmatory suites; raises ValueError if the seed sets overlap."""
    check_disjoint(pilot_seeds, confirmatory_seeds)
    return Split(
        pilot=tuple(build_suite(pilot_seeds, families, difficulties)),
        confirmatory=tuple(build_suite(confirmatory_seeds, families, difficulties)),
    )


def canonical_json(task: Task) -> str:
    """Canonical JSON of a task: sorted keys, no extra whitespace, ASCII only."""
    return json.dumps(task.canonical(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def task_sha256(task: Task) -> str:
    """SHA-256 (hex) of :func:`canonical_json`."""
    return hashlib.sha256(canonical_json(task).encode("ascii")).hexdigest()


def suite_manifest(tasks: Sequence[Task]) -> dict[str, Any]:
    """Manifest that pins a task set.

    ``sha256`` covers the canonical JSON of every task, sorted by id, so it does
    not depend on the order of ``tasks``. Raises ValueError for an empty set or
    repeated task ids.
    """
    if not tasks:
        raise ValueError("no tasks")
    ordered = sorted(tasks, key=lambda task: task.id)
    ids = [task.id for task in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("task ids repeat")
    body = "[" + ",".join(canonical_json(task) for task in ordered) + "]"
    return {
        "suite": SUITE_NAME,
        "version": SUITE_VERSION,
        "procedurally_generated": True,
        "n_tasks": len(ordered),
        "families": sorted({task.family for task in ordered}),
        "difficulties": sorted({task.difficulty for task in ordered}),
        "seeds": sorted({task.seed for task in ordered}),
        "task_ids": ids,
        "task_sha256": {task.id: task_sha256(task) for task in ordered},
        "sha256": hashlib.sha256(body.encode("ascii")).hexdigest(),
    }
