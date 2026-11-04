"""Planted-component recovery for the E1 analysis (G6), on synthetic data only.

Each dataset mirrors the E1 local-arm design: models crossed with tasks, a balanced
two-level fractional factorial of run-level factors inside every (model, task) cell,
runs grouped into batches that share a server configuration (model and concurrency),
and batches spread over days and time-of-day blocks. Outcomes are Bernoulli draws from

    logit p = intercept + model + factors + day + block + task + model:task + batch

with ``task``, ``model:task``, and ``batch`` drawn from normal distributions with the
planted variances. The E1 analysis (:func:`analyze`) fits the logit GLMM with
:func:`evalstats.decompose` and a parametric bootstrap; :func:`recover` reports, per
component, the planted value, the estimate, and the interval.

Planted values: a random component's planted value is its variance parameter; a fixed
factor's is the variance over runs of its planted contribution to the linear predictor
under the design's level shares (for a balanced two-level factor with effect ``b``,
``b**2 / 4``). Every row carries ``synthetic = True``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evalstats import EvalResults
from evalstats.glmm import BOOT_INTERVALS, Decomposition, boot_variance_interval, decompose

# Run-level two-level factors of the half fraction 2^(5-1) (defining relation I = ABCDE),
# split into two blocks of 8 runs by the ABC interaction so that every main effect is
# balanced within each block (4 runs at each level).
RUN_FACTORS = ("harness", "timeout", "steps", "temperature")
BATCH_FACTOR = "concurrency"
TWO_LEVEL = (*RUN_FACTORS, BATCH_FACTOR)


@dataclass(frozen=True)
class Design:
    """Shape of one synthetic E1-like dataset."""

    n_models: int = 4
    n_tasks: int = 40
    runs_per_cell: int = 8
    batch_size: int = 10
    days: int = 3
    blocks_per_day: int = 2

    def __post_init__(self) -> None:
        """Reject shapes the factorial cannot fill."""
        if self.runs_per_cell != 8:
            raise ValueError("runs_per_cell must be 8 (one block of the half fraction)")
        if min(self.n_models, self.n_tasks, self.batch_size, self.days, self.blocks_per_day) < 1:
            raise ValueError("every design size must be positive")


@dataclass(frozen=True)
class Planted:
    """Planted effects (logit scale) and variance parameters."""

    intercept: float = 0.0
    model: tuple[float, ...] = (0.6, 0.2, -0.2, -0.6)
    factors: Mapping[str, float] = field(
        default_factory=lambda: {
            "harness": 0.5,
            "timeout": 0.4,
            "steps": 0.4,
            "temperature": 0.1,
            "concurrency": 0.0,
        }
    )
    day: tuple[float, ...] = (0.0, 0.1, -0.1)
    block: tuple[float, ...] = (0.0, 0.05)
    task_var: float = 2.0
    model_task_var: float = 0.5
    batch_var: float = 0.1

    def to_json(self) -> dict[str, Any]:
        """Plain-data form."""
        return {**asdict(self), "factors": dict(self.factors)}


def half_fraction_blocks() -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """The 16 runs of 2^(5-1) with I = ABCDE, split into two blocks by ABC (0/1 coding)."""
    runs = [bits for bits in itertools.product((0, 1), repeat=5) if sum(bits) % 2 == 0]
    first = [r for r in runs if (r[0] + r[1] + r[2]) % 2 == 0]
    second = [r for r in runs if (r[0] + r[1] + r[2]) % 2 == 1]
    return first, second


def fixed_contribution(levels: Sequence[float], shares: Sequence[float]) -> float:
    """Variance over runs of a factor's contribution, given level effects and shares."""
    effects = np.asarray(levels, dtype=np.float64)
    weights = np.asarray(shares, dtype=np.float64)
    return float(weights @ (effects - weights @ effects) ** 2)


def simulate(design: Design, planted: Planted, seed: int) -> pd.DataFrame:
    """One synthetic dataset (deterministic given ``seed``); see the module docstring."""
    if len(planted.model) != design.n_models or len(planted.day) != design.days:
        raise ValueError("planted model and day effects must match the design")
    if len(planted.block) != design.blocks_per_day:
        raise ValueError("planted block effects must match the design")
    rng = np.random.default_rng(seed)
    first, second = half_fraction_blocks()
    task_eff = rng.normal(0.0, math.sqrt(planted.task_var), design.n_tasks)
    cell_eff = rng.normal(0.0, math.sqrt(planted.model_task_var), (design.n_models, design.n_tasks))
    runs = []
    for m in range(design.n_models):
        for t in range(design.n_tasks):
            half = first if (m + t) % 2 == 0 else second
            for bits in half:
                runs.append((m, t, dict(zip(TWO_LEVEL, bits, strict=True))))
    order = rng.permutation(len(runs))
    # Batches: consecutive runs (in random order) of one model and one concurrency level.
    groups: dict[tuple[int, int], list[int]] = {}
    for index in order:
        m, _, levels = runs[int(index)]
        groups.setdefault((m, levels[BATCH_FACTOR]), []).append(int(index))
    batches: list[list[int]] = []
    for members in groups.values():
        batches += [
            members[i : i + design.batch_size] for i in range(0, len(members), design.batch_size)
        ]
    slots = design.days * design.blocks_per_day
    batch_slot = rng.permutation(np.arange(len(batches)) % slots)
    batch_eff = rng.normal(0.0, math.sqrt(planted.batch_var), len(batches))
    rows = []
    for b, members in enumerate(batches):
        day, tod = divmod(int(batch_slot[b]), design.blocks_per_day)
        for index in members:
            m, t, levels = runs[index]
            eta = (
                planted.intercept
                + planted.model[m]
                + sum(planted.factors[name] * levels[name] for name in TWO_LEVEL)
                + planted.day[day]
                + planted.block[tod]
                + task_eff[t]
                + cell_eff[m, t]
                + batch_eff[b]
            )
            rows.append(
                {
                    "model": f"m{m}",
                    "task": f"t{t:03d}",
                    "run": f"r{index}",
                    "outcome": float(rng.random() < 1.0 / (1.0 + math.exp(-eta))),
                    **{name: str(levels[name]) for name in TWO_LEVEL},
                    "day": f"d{day}",
                    "block": f"b{tod}",
                    "batch": f"B{b:04d}",
                    "synthetic": True,
                }
            )
    return pd.DataFrame(rows)


RANDOM_TERMS = ("task", "model:task", "batch")
FIXED_TERMS = ("model", *TWO_LEVEL, "day", "block")


def planted_values(design: Design, planted: Planted, data: pd.DataFrame) -> dict[str, float]:
    """The planted value of every component, with fixed shares taken from ``data``."""
    values = {
        "task": planted.task_var,
        "model:task": planted.model_task_var,
        "batch": planted.batch_var,
    }

    def shares(column: str, names: Sequence[str]) -> list[float]:
        return [float(np.mean(data[column] == name)) for name in names]

    models = [f"m{m}" for m in range(design.n_models)]
    values["model"] = fixed_contribution(planted.model, shares("model", models))
    for name in TWO_LEVEL:
        values[name] = fixed_contribution((0.0, planted.factors[name]), shares(name, ["0", "1"]))
    days = [f"d{d}" for d in range(design.days)]
    values["day"] = fixed_contribution(planted.day, shares("day", days))
    blocks = [f"b{b}" for b in range(design.blocks_per_day)]
    values["block"] = fixed_contribution(planted.block, shares("block", blocks))
    return values


@dataclass(frozen=True)
class AnalysisSettings:
    """The E1 decomposition settings (fixed in the preregistration)."""

    n_boot: int = 200
    nagq: int = 0
    boot_interval: str = "basic"
    level: float = 0.95
    random: tuple[str, ...] = RANDOM_TERMS


def analyze(data: pd.DataFrame, settings: AnalysisSettings, seed: int) -> Decomposition:
    """The E1 decomposition of one table of runs (a factor held constant is left out)."""
    fixed = tuple(name for name in FIXED_TERMS if data[name].nunique() > 1)
    return decompose(
        EvalResults(data),
        random=settings.random,
        fixed=fixed,
        level=settings.level,
        backend="lme4",
        method="parametric_bootstrap",
        n_boot=settings.n_boot,
        nagq=settings.nagq,
        boot_interval=settings.boot_interval,
        seed=seed,
    )


def recover(
    design: Design, planted: Planted, settings: AnalysisSettings, seed: int
) -> dict[str, Any]:
    """Simulate one dataset, analyze it, and return each component's row for G6."""
    data = simulate(design, planted, seed)
    truth = planted_values(design, planted, data)
    decomp = analyze(data, settings, seed)
    components = {}
    for comp in decomp.components:
        if comp.kind == "residual":
            continue
        components[comp.name] = {
            "planted": truth[comp.name],
            "estimate": comp.variance,
            "lo": comp.lo,
            "hi": comp.hi,
        }
    return {
        "seed": seed,
        "components": components,
        "converged": decomp.converged,
        "notes": list(decomp.messages)[-1:],
    }


def coverage_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """Per component: coverage, mean bias, and interval width over datasets."""
    names = sorted({name for row in rows for name in row["components"]})
    out = {}
    for name in names:
        comps = [row["components"][name] for row in rows if name in row["components"]]
        ok = [c for c in comps if c["lo"] is not None and c["hi"] is not None]
        covered = sum(1 for c in ok if c["lo"] <= c["planted"] <= c["hi"])
        out[name] = {
            "n": len(comps),
            "with_interval": len(ok),
            "coverage": covered / len(ok) if ok else math.nan,
            "bias": float(np.mean([c["estimate"] - c["planted"] for c in comps])),
            "mean_width": float(np.mean([c["hi"] - c["lo"] for c in ok])) if ok else math.nan,
        }
    return out


def dumps(obj: Any) -> str:
    """Canonical JSON (NaN and infinities as null)."""

    def clean(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [clean(v) for v in value]
        return value

    return json.dumps(clean(obj), sort_keys=True)


PILOT_SEEDS = range(990000, 991000)  # exploratory method pilots; never used by G6


def pilot_row(
    design: Design, planted: Planted, settings: AnalysisSettings, seed: int
) -> dict[str, Any]:
    """One exploratory dataset scored under every random-effect interval rule.

    The bootstrap runs once; ``percentile``, ``basic``, and ``bc`` intervals are all
    computed from its replicates. Fixed factors use the box interval.
    """
    data = simulate(design, planted, seed)
    truth = planted_values(design, planted, data)
    decomp = analyze(data, settings, seed)
    rows: dict[str, Any] = {}
    for comp in decomp.components:
        if comp.kind == "residual":
            continue
        entry: dict[str, Any] = {"planted": truth[comp.name], "estimate": comp.variance}
        if comp.kind == "random":
            reps = np.asarray(decomp.replicates[comp.name], dtype=np.float64)
            for kind in BOOT_INTERVALS:
                entry[kind] = boot_variance_interval(comp.variance, reps, settings.level, kind)
        else:
            entry["box"] = (comp.lo, comp.hi)
        rows[comp.name] = entry
    return {"seed": seed, "components": rows, "notes": list(decomp.messages)[-1:]}


def pilot_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Coverage of each interval rule per component, with bias, over pilot datasets."""
    out: dict[str, dict[str, Any]] = {}
    names = sorted({n for row in rows for n in row["components"]})
    for name in names:
        comps = [row["components"][name] for row in rows if name in row["components"]]
        summary: dict[str, Any] = {
            "n": len(comps),
            "bias": float(np.mean([c["estimate"] - c["planted"] for c in comps])),
            "sd": float(np.std([c["estimate"] for c in comps])),
        }
        for kind in ("percentile", "basic", "bc", "box"):
            ok = [c for c in comps if c.get(kind) and None not in c[kind]]
            if ok:
                hit = sum(1 for c in ok if c[kind][0] <= c["planted"] <= c[kind][1])
                summary[kind] = hit / len(ok)
        out[name] = summary
    return out


# ---------------------------------------------------------------------------
# confirmatory runs, configured only by frozen preregistration blocks

REPO = Path(__file__).resolve().parents[2]
CONFIG_BLOCK = "recovery-config"  # design, planted values, and analysis settings
E0_DIR = REPO / "artifacts" / "e0"
G6_DIR = REPO / "artifacts" / "g6"


def prereg_block(path: Path, name: str) -> Any:
    """The JSON of the ```json <name>``` block of a preregistration file."""
    match = re.search(rf"```json[ \t]+{re.escape(name)}[ \t]*\n(.*?)\n```", path.read_text(), re.S)
    if match is None:
        raise ValueError(f"{path}: no ```json {name}``` block")
    return json.loads(match.group(1))


def config_from(block: Mapping[str, Any]) -> tuple[Design, Planted, AnalysisSettings]:
    """Design, planted values, and settings from a ``recovery-config`` block."""
    p = block["planted"]
    planted = Planted(
        intercept=float(p["intercept"]),
        model=tuple(float(v) for v in p["model"]),
        factors={str(k): float(v) for k, v in p["factors"].items()},
        day=tuple(float(v) for v in p["day"]),
        block=tuple(float(v) for v in p["block"]),
        task_var=float(p["task_var"]),
        model_task_var=float(p["model_task_var"]),
        batch_var=float(p["batch_var"]),
    )
    st = block["settings"]
    settings = AnalysisSettings(
        n_boot=int(st["n_boot"]),
        nagq=int(st["nagq"]),
        boot_interval=str(st["boot_interval"]),
        level=float(st["level"]),
        random=tuple(str(r) for r in st["random"]),
    )
    return Design(**block["design"]), planted, settings


def _recover_job(job: tuple[dict[str, Any], int]) -> dict[str, Any]:
    block, seed = job
    design, planted, settings = config_from(block)
    return recover(design, planted, settings, seed)


def _pool_map(jobs: list[Any], func: Any, workers: int) -> Iterator[Any]:
    if workers <= 1:
        yield from map(func, jobs)
        return
    with mp.get_context("spawn").Pool(workers, initializer=os.nice, initargs=(10,)) as pool:
        yield from pool.imap_unordered(func, jobs)


def run_g6(prereg: Path, raw: Path, workers: int = 1, log: Any = print) -> int:
    """Append a recovery row for every preregistered seed not yet in ``raw`` (resumable)."""
    block = prereg_block(prereg, CONFIG_BLOCK)
    seeds = [int(s) for s in prereg_block(prereg, "g6-seeds")]
    done = set()
    if raw.exists():
        done = {json.loads(ln)["seed"] for ln in raw.read_text().splitlines() if ln.strip()}
    jobs = [(block, s) for s in seeds if s not in done]
    raw.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    with raw.open("a") as fh:
        for row in _pool_map(jobs, _recover_job, workers):
            fh.write(dumps(row) + "\n")
            fh.flush()
            added += 1
            if added % 10 == 0:
                log(f"g6: {len(done) + added}/{len(seeds)} datasets")
    return added


def bank_g6(prereg: Path, raw: Path, out: Path) -> dict[str, Any]:
    """Write the G6 artifact from the raw rows of every preregistered seed."""
    block = prereg_block(prereg, CONFIG_BLOCK)
    seeds = [int(s) for s in prereg_block(prereg, "g6-seeds")]
    rows = {}
    for line in raw.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[int(row["seed"])] = row
    missing = [s for s in seeds if s not in rows]
    if missing:
        raise ValueError(f"{len(missing)} preregistered datasets are not run yet")
    datasets = [
        {"seed": s, "components": rows[s]["components"], "converged": rows[s].get("converged")}
        for s in seeds
    ]
    names = sorted({n for d in datasets for n in d["components"]})
    bias = {
        n: float(
            np.mean(
                [d["components"][n]["estimate"] - d["components"][n]["planted"] for d in datasets]
            )
        )
        for n in names
    }
    nominal = float(block["settings"]["level"])
    artifact = {
        "prereg": prereg.relative_to(REPO).as_posix()
        if prereg.is_relative_to(REPO)
        else str(prereg),
        "synthetic": True,
        "confirmatory": True,
        "nominal": nominal,
        "config": block,
        "datasets": datasets,
        "bias": bias,
        "coverage": coverage_summary(datasets),
        "generator": "python -m noisefloor.recovery g6-bank " + prereg.name,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps(artifact) + "\n")
    return artifact


def _repo_path(path: Path) -> str:
    absolute = path.resolve()
    return absolute.relative_to(REPO).as_posix() if absolute.is_relative_to(REPO) else str(path)


def e0_glmm_seed(group: str, replicate: int) -> int:
    """Dataset seed of one replicate of a group of E0 GLMM cells (all components share it)."""
    digest = hashlib.sha256(f"e0-glmm:{group}:{replicate}".encode()).hexdigest()
    return int(digest[:12], 16)


def run_e0_glmm(amendment: Path, workers: int = 1, root: Path = E0_DIR, log: Any = print) -> int:
    """Fill the raw files of the amendment's ``glmm_component`` cells (resumable).

    Each replicate index is one simulated dataset shared by every component cell of its
    group (seed :func:`e0_glmm_seed`); a cell's raw line holds that component's planted
    value as ``truth``, its estimate, and its interval.
    """
    grid = prereg_block(amendment, "e0-grid")
    block = prereg_block(amendment, CONFIG_BLOCK)
    cells = [c for c in grid if c["method"] == "glmm_component"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for c in cells:
        groups.setdefault(str(c["params"]["group"]), []).append(c)
    added = 0
    for group, members in groups.items():
        reps = max(int(c["replicates"]) for c in members)
        paths = {
            str(c["params"]["component"]): root / "raw" / f"{c['cell']}.jsonl" for c in members
        }
        done = None
        for path in paths.values():
            have = set()
            if path.exists():
                have = {json.loads(ln)["i"] for ln in path.read_text().splitlines() if ln.strip()}
            done = have if done is None else done & have
        todo = [i for i in range(reps) if i not in (done or set())]
        jobs = [(block, e0_glmm_seed(group, i)) for i in todo]
        index = {e0_glmm_seed(group, i): i for i in todo}
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        for row in _pool_map(jobs, _recover_job, workers):
            i = index[int(row["seed"])]
            for component, path in paths.items():
                comp = row["components"].get(component)
                line = {
                    "i": i,
                    "seed": row["seed"],
                    "truth": None if comp is None else comp["planted"],
                    "estimate": None if comp is None else comp["estimate"],
                    "lo": None if comp is None else comp["lo"],
                    "hi": None if comp is None else comp["hi"],
                    "hit": None,
                }
                with path.open("a") as fh:
                    fh.write(dumps(line) + "\n")
            added += 1
            if added % 10 == 0:
                log(f"e0 {group}: {added}/{len(todo)} datasets")
    return added


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m noisefloor.recovery pilot OUT.jsonl FIRST COUNT [--n-boot B]``.

    Runs exploratory method-pilot datasets (seeds from :data:`PILOT_SEEDS`) with the
    default design and planted values, appending one JSON line per dataset (resumable),
    then prints the coverage summary.
    """
    args_list = list(argv) if argv is not None else sys.argv[1:]
    if args_list and args_list[0] in ("g6-run", "g6-bank", "e0-run"):
        return _confirmatory_main(args_list)
    parser = argparse.ArgumentParser(description="planted-component recovery (synthetic)")
    parser.add_argument("command", choices=["pilot"])
    parser.add_argument("out", type=Path)
    parser.add_argument("first", type=int)
    parser.add_argument("count", type=int)
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument(
        "--no-batch", action="store_true", help="plant no batch variance and fit no batch term"
    )
    args = parser.parse_args(args_list)
    seeds = [s for s in range(args.first, args.first + args.count)]
    if any(s not in PILOT_SEEDS for s in seeds):
        raise SystemExit("pilot seeds must come from PILOT_SEEDS")
    done = set()
    if args.out.exists():
        done = {json.loads(ln)["seed"] for ln in args.out.read_text().splitlines() if ln.strip()}
    settings = AnalysisSettings(n_boot=args.n_boot)
    planted = Planted()
    if args.no_batch:
        settings = AnalysisSettings(n_boot=args.n_boot, random=("task", "model:task"))
        planted = dataclasses.replace(planted, batch_var=0.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        if seed in done:
            continue
        row = pilot_row(Design(), planted, settings, seed)
        with args.out.open("a") as fh:
            fh.write(dumps(row) + "\n")
    rows = [json.loads(ln) for ln in args.out.read_text().splitlines() if ln.strip()]
    print(json.dumps(pilot_summary(rows), indent=1, sort_keys=True))
    return 0


def _confirmatory_main(argv: list[str]) -> int:
    """``g6-run PREREG [--workers N]``, ``g6-bank PREREG``, ``e0-run AMENDMENT [--workers N]``."""
    parser = argparse.ArgumentParser(description="confirmatory recovery runs from a prereg")
    parser.add_argument("command", choices=["g6-run", "g6-bank", "e0-run"])
    parser.add_argument("prereg", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    raw = G6_DIR / "raw.jsonl"
    if args.command == "g6-run":
        print(f"added {run_g6(args.prereg, raw, args.workers)} datasets")
    elif args.command == "g6-bank":
        artifact = bank_g6(args.prereg, raw, G6_DIR / "recovery.json")
        print(json.dumps(artifact["coverage"], indent=1, sort_keys=True))
    else:
        print(f"added {run_e0_glmm(args.prereg, args.workers)} datasets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
