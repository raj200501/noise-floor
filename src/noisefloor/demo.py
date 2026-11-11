"""Offline demo (`make demo`): evalstats on a recorded synthetic fixture.

Runs with no network and no model calls. It reads the committed fixture
``fixtures/demo/synthetic_results.csv`` (synthetic data with planted effects,
labeled as such), checks its hash against ``fixtures/demo/MANIFEST.json``, and
prints a deterministic report that ``tests/golden/demo.txt`` pins byte for byte.

``python -m noisefloor.demo --write-fixture`` regenerates the fixture and manifest
from the planted spec below (only needed if the spec changes).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

from evalstats import (
    EvalResults,
    design_from_results,
    mde,
    paired_difference,
    read_csv,
    score_interval,
)
from noisefloor.synthetic import PlantedSpec, simulate

REPO = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO / "fixtures" / "demo"
FIXTURE = FIXTURE_DIR / "synthetic_results.csv"
MANIFEST = FIXTURE_DIR / "MANIFEST.json"
SEED = 20261023
SPEC = PlantedSpec(
    models={"synth-alpha": 0.55, "synth-beta": 0.40, "synth-gamma": 0.30, "synth-delta": -0.35},
    n_tasks=120,
    runs=3,
    intercept=0.1,
    task_sd=1.2,
    model_task_sd=0.5,
)
BOOT_SEED = 2026
N_BOOT = 2000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fixture() -> None:
    """Regenerate the synthetic fixture and its manifest."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    simulate(SPEC, SEED).to_csv(FIXTURE, index=False, lineterminator="\n")
    manifest = {
        "synthetic": True,
        "description": "Synthetic eval results with planted effects; not model results.",
        "generator": "python -m noisefloor.demo --write-fixture",
        "seed": SEED,
        "spec": {
            "models": SPEC.models,
            "n_tasks": SPEC.n_tasks,
            "runs": SPEC.runs,
            "intercept": SPEC.intercept,
            "task_sd": SPEC.task_sd,
            "model_task_sd": SPEC.model_task_sd,
        },
        "files": {FIXTURE.name: _sha256(FIXTURE)},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_fixture() -> EvalResults:
    """Load the fixture after checking its hash against the manifest."""
    manifest = json.loads(MANIFEST.read_text())
    expected = manifest["files"][FIXTURE.name]
    if _sha256(FIXTURE) != expected:
        raise SystemExit(f"fixture hash mismatch: {FIXTURE}")
    return read_csv(FIXTURE)


def report(results: EvalResults) -> list[str]:
    """Lines of the demo report."""
    runs = int(results.run_counts(results.models[0]).iloc[0])
    lines = [
        "noise-floor offline demo",
        "data: fixtures/demo/synthetic_results.csv (SYNTHETIC, planted effects, not model results)",
        f"sha256 {_sha256(FIXTURE)[:16]} matches fixtures/demo/MANIFEST.json",
        f"{len(results)} task runs: {len(results.models)} models, {len(results.tasks)} tasks, "
        f"{runs} runs per task",
        "",
        "1. Benchmark scores (mean over tasks of task means), 95% task-clustered t intervals",
        f"   {'model':<12} {'score':>6}  95% CI",
    ]
    scores = {m: score_interval(results, m) for m in results.models}
    order = sorted(results.models, key=lambda m: -scores[m].interval.estimate)
    for m in order:
        iv = scores[m].interval
        lines.append(f"   {m:<12} {iv.estimate:6.3f}  [{iv.lo:.3f}, {iv.hi:.3f}]")
    lines += ["", "2. Adjacent pairs in score order (paired on shared tasks)"]
    for a, b in itertools.pairwise(order):
        cmp = paired_difference(results, a, b)
        d = cmp.difference
        verdict = "supported" if cmp.supported else "not supported"
        p_text = "p < 0.001" if cmp.p_value < 0.001 else f"p = {cmp.p_value:.3f}"
        lines.append(
            f"   {a} - {b}: {d.estimate:+.3f} [{d.lo:+.3f}, {d.hi:+.3f}], "
            f"{p_text}, order {verdict} at 95%"
        )
    top_a, top_b = order[0], order[1]
    lines += [
        "",
        f"3. {top_a} - {top_b} by interval method "
        f"({N_BOOT} task resamples, seed {BOOT_SEED} for bootstraps)",
    ]
    for method in ("cluster_t", "bootstrap", "bca"):
        d = paired_difference(
            results, top_a, top_b, method=method, n_boot=N_BOOT, seed=BOOT_SEED
        ).difference
        lines.append(f"   {method:<10} [{d.lo:+.3f}, {d.hi:+.3f}]")
    design = design_from_results(results, top_a, top_b)
    lines += [
        "",
        f"4. Minimum detectable effect for {top_a} vs {top_b} (alpha 0.05, power 0.8)",
        f"   variances: between-task {design.between_task_var:.4f}, "
        f"within-task {design.within_task_var:.4f}",
        f"   {'tasks':>6} {'runs':>5} {'mde':>7}",
    ]
    for n_tasks in (120, 250, 500, 1000):
        for k in (1, 3, 10):
            lines.append(f"   {n_tasks:>6} {k:>5} {mde(design, n_tasks, k):7.3f}")
    lines += ["", "Every number above comes from the synthetic fixture; no model was called."]
    return lines


def main(argv: list[str] | None = None) -> int:
    """Run the demo, or regenerate the fixture with ``--write-fixture``."""
    parser = argparse.ArgumentParser(description="noise-floor offline demo")
    parser.add_argument("--write-fixture", action="store_true", help="regenerate the fixture")
    args = parser.parse_args(argv)
    if args.write_fixture:
        write_fixture()
        print(f"wrote {FIXTURE} and {MANIFEST}")
        return 0
    print("\n".join(report(load_fixture())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
