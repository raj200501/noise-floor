"""E1 pilot (exploratory, never pooled with confirmatory data).

The pilot chooses model sizes and the task difficulty mix so that every model
configuration's mean success lies between 0.2 and 0.8 (docs/BUILD_BRIEF.md section 4),
measures run times, and gives variance estimates for the E1 power analysis. Pilot
tasks use seeds 900000 to 900999, disjoint from the confirmatory seed range fixed in
the E1 preregistration. Results go to ``artifacts/pilot/<plan>.jsonl`` with
``"exploratory": true``.

``python -m noisefloor.pilot <plan>`` runs a plan in the foreground (use the
background wrapper in docs/PLAN.md for long plans); it resumes if interrupted.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from noisefloor.orchestrate import REPO, execute
from noisefloor.runner import RunSpec
from noisefloor.suite import FAMILY_NAMES

PILOT_SEEDS = range(900000, 901000)


@dataclass(frozen=True)
class PilotPlan:
    """A grid of pilot runs: every model, harness, family, and difficulty crossed."""

    models: tuple[str, ...]
    harnesses: tuple[str, ...] = ("minimal", "inspect")
    difficulties: tuple[int, ...] = (1, 2, 3, 4, 5)
    seeds_per_cell: int = 1
    seed_offset: int = 0
    temperature: float = 0.0
    timeout_s: float = 300.0
    threads: int = 4
    concurrency: int = 1
    max_steps: int = 15


PLANS: dict[str, PilotPlan] = {
    "p1": PilotPlan(models=("qwen2.5-0.5b", "qwen2.5-1.5b", "granite-3.3-2b", "smollm2-1.7b")),
}


def plan_runs(name: str, order_seed: int = 7) -> list[RunSpec]:
    """Run specs for a pilot plan: models in random order, runs shuffled within each model."""
    plan = PLANS[name]
    rng = random.Random(f"pilot:{name}:{order_seed}")
    models = list(plan.models)
    rng.shuffle(models)
    specs: list[RunSpec] = []
    for model in models:
        cell: list[RunSpec] = []
        for family in FAMILY_NAMES:
            for difficulty in plan.difficulties:
                for k in range(plan.seeds_per_cell):
                    task_seed = PILOT_SEEDS[plan.seed_offset + k]
                    for harness in plan.harnesses:
                        cell.append(
                            RunSpec(
                                run_id=f"{name}-{model}-{harness}-{family}-d{difficulty}-s{task_seed}",
                                phase="pilot",
                                family=family,
                                task_seed=task_seed,
                                difficulty=difficulty,
                                model=model,
                                harness=harness,
                                temperature=plan.temperature,
                                seed=rng.randrange(2**31),
                                timeout_s=plan.timeout_s,
                                max_steps=plan.max_steps,
                                threads=plan.threads,
                                concurrency=plan.concurrency,
                            )
                        )
        rng.shuffle(cell)
        specs.extend(cell)
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    """Run a pilot plan (resumable)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", choices=sorted(PLANS))
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args(list(argv) if argv is not None else None)
    specs = plan_runs(args.plan)
    if args.dry_run:
        print(f"{len(specs)} runs; first: {specs[0].run_id}")
        return 0
    artifact = REPO / "artifacts" / "pilot" / f"{args.plan}.jsonl"
    ledger = REPO / "artifacts" / "usage" / "ledger.jsonl"
    banked = execute(
        specs,
        artifact=artifact,
        ledger=ledger,
        prefix=f"pilot-{args.plan}",
        log=lambda m: print(m, flush=True),
    )
    print(f"banked {banked} runs in {artifact.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
