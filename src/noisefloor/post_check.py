"""Post-E1 check at the fitted values (D-028 item 4; synthetic data only).

G6 and E0 part 2 validate the primary analysis at planted values chosen before E1. This
check asks whether the analysis also holds where E1 landed: it simulates
:data:`POST_DATASETS` datasets (seeds :data:`POST_SEEDS`, fixed in the E1 preregistration)
from the E1 fit, analyzes each with the unchanged primary analysis
(:func:`noisefloor.recovery.recover`), and holds every component's coverage to the G5 rule
(coverage at least nominal minus 0.015 minus twice its Monte Carlo standard error).

The planted values come from one auxiliary fit of the primary model with task-by-harness,
task-by-steps, and task-by-timeout random terms added (:func:`auxiliary_fit`): its
intercept, model, factor, day, and block effects, its task and model:task variances, and
its three task-by-factor variances, planted together (if that fit fails, the primary fit
in ``artifacts/e1/variance_components.json`` without interactions, labeled in the
config). The task truth the check scores is the planted task variance plus half of each
task-by-factor variance, which the primary model's task term targets
(:func:`noisefloor.recovery.planted_values`). The batch variance planted is the batch-term
sensitivity fit's ``batch_var_upper`` (``sensitivity.json``): the upper percentile of its
bootstrap replicate batch variances, since its point estimate runs low (D-022). A
component whose coverage fails the rule is reported as resting on an unsupported method;
if that batch variance exceeds :data:`BATCH_RULE`, or the batch-term fit failed or gave
no such value, the concurrency, day, and block intervals are also reported as not
validated (and the check plants no batch variance).

``python -m noisefloor.post_check run PREREG`` appends one row per dataset to
``artifacts/e1/post_check/raw.jsonl`` (resumable, stamped as G6 rows are);
``bank PREREG`` writes ``artifacts/e1/post_check.json``. Both take ``--amendment`` when a
frozen amendment of prereg 0004 allows other analysis code.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from evalstats import EvalResults
from evalstats.glmm import decompose
from noisefloor import jsonl, preregs
from noisefloor.e0_run import record_environment
from noisefloor.e1 import E1Design
from noisefloor.preregs import prereg_block
from noisefloor.provenance import (
    check_same_environment,
    single_analysis_code,
    single_environment,
    stamp,
)
from noisefloor.recovery import TWO_LEVEL, AnalysisSettings, Design, Planted, recover

REPO = Path(__file__).resolve().parents[2]
E1_DIR = REPO / "artifacts" / "e1"
POST_SEEDS = range(980000, 980200)  # never used by G6 (970000 on) or the pilots (990000 on)
POST_DATASETS = len(POST_SEEDS)
BATCH_RULE = 0.05  # batch variance above which concurrency, day, and block are not validated
BATCH_FACTORS = ("concurrency", "day", "block")
INTERACTION_FACTORS = ("harness", "steps", "timeout")


def g5_rule(covered: int, n: int, nominal: float) -> tuple[bool, float, float]:
    """(passes, coverage, threshold) under the G5 rule."""
    coverage = covered / n
    threshold = nominal - 0.015 - 2 * math.sqrt(coverage * (1 - coverage) / n)
    return coverage >= threshold - 1e-12, coverage, threshold


def settings_from(analysis: Mapping[str, Any]) -> AnalysisSettings:
    """The primary analysis settings of the ``e1-analysis`` block, for :func:`recover`."""
    return AnalysisSettings(
        n_boot=int(analysis["n_boot"]),
        nagq=int(analysis["nagq"]),
        boot_interval=str(analysis["boot_interval"]),
        level=float(analysis["level"]),
        random=tuple(analysis["random"]),
        box_interval=str(analysis["box_interval"]),
        share_interval=str(analysis.get("share_interval", "percentile")),
        fixed_interval=str(analysis.get("fixed_interval", "box")),
        mde_designs=tuple((int(t), int(k)) for t, k in analysis["mde"]["designs"]),
        mde_interval=str(analysis["mde"]["interval"]),
    )


def auxiliary_fit(data: pd.DataFrame, analysis: Mapping[str, Any]) -> dict[str, Any]:
    """One fit of the primary model plus task-by-factor random terms (no intervals).

    Returns its fixed effects, its random variances (task and model:task net of the
    task-by-factor terms), and the task-by-factor variances, all planted together.
    """
    fixed = tuple(f for f in analysis["fixed"] if data[f].nunique() > 1)
    random = (*analysis["random"], *(f"task:{f}" for f in INTERACTION_FACTORS))
    fit = decompose(
        EvalResults(data),
        random=random,
        fixed=fixed,
        intervals=False,
        backend="lme4",
        nagq=int(analysis["nagq"]),
    )
    found = {c.name: c.variance for c in fit.components if c.kind == "random"}
    return {
        "fixed_effects": {str(k): float(v) for k, v in fit.fixed_effects.items()},
        "variances": {k: float(found[k]) for k in analysis["random"]},
        "interactions": {f: float(found[f"task:{f}"]) for f in INTERACTION_FACTORS},
    }


def planted_from_fit(
    fixed_effects: Mapping[str, float],
    variances: Mapping[str, float],
    *,
    batch_var: float,
    interactions: Mapping[str, float],
    design: E1Design,
    data: pd.DataFrame,
) -> tuple[Design, Planted]:
    """The simulation design and planted values that reproduce the E1 fit.

    The analysis uses treatment coding with the first sorted level as reference; the
    simulator codes each two-level factor 0 and 1 in the order of the design's levels, so
    a factor's planted effect is the coefficient of its level 1 minus that of its level 0,
    and the planted intercept is the fit's linear predictor with every factor at level 0,
    the first sorted model, and the first day and block.
    """
    models = sorted(str(m) for m in data["model"].unique())
    days = sorted(str(d) for d in data["day"].unique())
    blocks = sorted(str(b) for b in data["block"].unique())

    def coef(name: str, level: Any) -> float:
        return float(fixed_effects.get(f"{name}{level}", 0.0))

    intercept = float(fixed_effects["(Intercept)"])
    factors = {}
    for name in TWO_LEVEL:
        level0, level1 = (str(v) for v in design.levels[name])
        intercept += coef(name, level0)
        factors[name] = coef(name, level1) - coef(name, level0)
    simulated = Design(
        n_models=len(models),
        n_tasks=int(data["task"].nunique()),
        runs_per_cell=design.runs_per_cell,
        batch_size=design.batch_size,
        days=len(days),
        blocks_per_day=len(blocks),
        # The replicate placeholders are dealt into batches and then left out, as in E1.
        replicates=tuple(tuple(int(b) for b in r) for r in design.replicates),
    )
    planted = Planted(
        intercept=intercept,
        model=tuple(coef("model", m) for m in models),
        factors=factors,
        day=tuple(coef("day", d) for d in days),
        block=tuple(coef("block", b) for b in blocks),
        task_var=float(variances["task"]),
        model_task_var=float(variances["model:task"]),
        batch_var=float(batch_var),
        interactions=dict(interactions),
    )
    return simulated, planted


def _config_path(directory: Path) -> Path:
    return directory / "post_check" / "config.json"


def prepare(prereg: Path, directory: Path = E1_DIR) -> dict[str, Any]:
    """Derive the planted configuration from the banked E1 results and write it once.

    The configuration is written to ``post_check/config.json`` beside the E1 artifacts and
    reused by every later call, so a resumed run simulates from the same values.
    """
    from noisefloor.analysis import e1 as analysis  # noqa: PLC0415 - analysis imports e1

    path = _config_path(directory)
    if path.exists():
        return dict(json.loads(path.read_text()))
    preregs.require_frozen(prereg)
    settings = prereg_block(prereg, "e1-analysis")
    design = E1Design.from_json(prereg_block(prereg, "e1-plan"))
    rows = [json.loads(ln) for ln in (directory / "runs.jsonl").read_text().splitlines()]
    data = analysis.analysis_frame(rows, settings["exclude_status"])
    variance = json.loads((directory / "variance_components.json").read_text())
    sensitivity = json.loads((directory / "sensitivity.json").read_text())
    batch_fit = sensitivity["batch_random_effect"]
    # The batch variance planted, and compared with the 0.05 rule, is the batch-term fit's
    # batch_var_upper (the upper percentile of its bootstrap replicates; the point estimate
    # runs low, D-022). A failed batch-term fit, or one without that value, plants 0, and
    # then concurrency, day, and block are reported as not validated.
    batch_error, batch = batch_term(batch_fit)
    # The auxiliary fit's fixed effects and variances are planted with its task-by-factor
    # variances; if it fails, the check runs with the primary fit and no interactions.
    try:
        aux: dict[str, Any] = auxiliary_fit(data, settings)
        aux_error = None
    except Exception as exc:  # a failed auxiliary fit is reported, not fatal
        aux_error = f"{type(exc).__name__}: {exc}"[:300]
        aux = {
            "fixed_effects": variance["fixed_effects"],
            "variances": {c["name"]: c["estimate"] for c in variance["components"]},
            "interactions": {},
        }
    simulated, planted = planted_from_fit(
        aux["fixed_effects"],
        aux["variances"],
        batch_var=batch,
        interactions=aux["interactions"],
        design=design,
        data=data,
    )
    config = {
        "design": simulated.__dict__,
        "planted": planted.to_json(),
        "settings": settings_from(settings).__dict__,
        "seeds": [POST_SEEDS[0], POST_SEEDS[-1]],
        "batch_rule": BATCH_RULE,
        "batch_fit_error": batch_error,
        "auxiliary_fit_error": aux_error,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=1, sort_keys=True) + "\n")
    return dict(json.loads(path.read_text()))  # the same value every later call reads


def batch_term(batch_fit: Mapping[str, Any]) -> tuple[str | None, float]:
    """(why the batch-term fit counts as failed, or None; the batch variance to plant).

    The batch-term fit counts as failed when it raised, or when it gave its batch variance
    no ``batch_var_upper`` (no batch term, or more than 10 percent of the replicates
    failed); the check then plants 0 (prereg 0004 items 9 and 10).
    """
    if batch_fit.get("error"):
        return str(batch_fit["error"]), 0.0
    upper = batch_fit.get("batch_var_upper")
    if upper is None or not math.isfinite(float(upper)):
        return "the batch-term fit gave its batch variance no upper end", 0.0
    return None, float(upper)


def _job(job: tuple[dict[str, Any], int]) -> dict[str, Any]:
    from noisefloor.recovery import config_from  # noqa: PLC0415 - spawned workers

    block, seed = job
    design, planted, settings = config_from(block)
    return {**recover(design, planted, settings, seed), "stamp": stamp()}


def run(
    prereg: Path,
    directory: Path = E1_DIR,
    *,
    workers: int = 1,
    log: Any = print,
    seeds: Sequence[int] = POST_SEEDS,
    amendment: Path | None = None,
) -> int:
    """Append a row for every post-check seed not yet run (resumable)."""
    from noisefloor.analysis import e1 as analysis  # noqa: PLC0415 - analysis imports e1
    from noisefloor.recovery import _pool_map, dumps  # noqa: PLC0415

    # The check runs the analysis code G6 and E0 part 2 ran (prereg 0004), or code a frozen
    # amendment of 0004 allows, in one environment.
    analysis.check_analysis_code(prereg, prereg_block(prereg, "e1-analysis"), amendment)
    config = prepare(prereg, directory)
    raw = directory / "post_check" / "raw.jsonl"
    jsonl.repair_tail(raw)
    present = jsonl.read(raw)
    check_same_environment((r.get("stamp") for r in present), "post-E1 check")
    done = {int(r["seed"]) for r in present}
    block = {k: config[k] for k in ("design", "planted", "settings")}
    jobs = [(block, s) for s in seeds if s not in done]
    record_environment(raw.parent)
    added = 0
    for row in _pool_map(jobs, _job, workers):
        jsonl.append(raw, [json.loads(dumps(row))])
        added += 1
        if added % 10 == 0:
            log(f"post-E1 check: {len(done) + added}/{len(seeds)} datasets")
    return added


def summarize(
    rows: Sequence[Mapping[str, Any]],
    nominal: float,
    batch_var: float,
    batch_failed: bool = False,
) -> dict[str, Any]:
    """Per component, coverage under the G5 rule; and which intervals are not validated."""
    names = sorted({n for r in rows for n in r["components"]})
    components = {}
    for name in names:
        comps = [r["components"][name] for r in rows if name in r["components"]]
        covered = sum(
            1
            for c in comps
            if c["lo"] is not None and c["hi"] is not None and c["lo"] <= c["planted"] <= c["hi"]
        )
        ok, coverage, threshold = g5_rule(covered, len(comps), nominal)
        components[name] = {
            "datasets": len(comps),
            "coverage": coverage,
            "threshold": threshold,
            "passes": ok,
        }
    failing = sorted(n for n, c in components.items() if not c["passes"])
    batch = batch_var > BATCH_RULE
    unsupported = sorted({*failing, *(BATCH_FACTORS if batch or batch_failed else ())})
    return {
        "components": components,
        "failing": failing,
        "batch_var": batch_var,
        "batch_rule_exceeded": batch,
        "batch_fit_failed": batch_failed,
        "not_validated": unsupported,
    }


def bank(
    prereg: Path,
    directory: Path = E1_DIR,
    seeds: Sequence[int] = POST_SEEDS,
    *,
    amendment: Path | None = None,
) -> dict[str, Any]:
    """Write ``post_check.json`` from the rows of every post-check seed.

    The rows must share one environment and have run the analysis code the E1 analysis
    accepts (that of G6 and E0 part 2, or one a frozen ``amendment`` of 0004 allows).
    """
    from noisefloor.analysis import e1 as analysis  # noqa: PLC0415 - analysis imports e1

    entry = preregs.require_frozen(prereg)
    code = analysis.check_analysis_code(prereg, prereg_block(prereg, "e1-analysis"), amendment)
    config = prepare(prereg, directory)
    rows = {int(r["seed"]): r for r in jsonl.read(directory / "post_check" / "raw.jsonl")}
    missing = [s for s in seeds if s not in rows]
    if missing:
        raise ValueError(f"{len(missing)} post-check datasets are not run yet")
    ordered = [rows[s] for s in seeds]
    env = single_environment(r.get("stamp") for r in ordered)
    ran = single_analysis_code(r.get("stamp") for r in ordered)
    if ran != code["hash"]:
        raise ValueError(f"the rows ran analysis code {ran}, not {code['hash']}")
    artifact = {
        "prereg": preregs.repo_path(prereg),
        "prereg_sha256": entry["sha256"],
        "synthetic": True,
        "label": "post-E1 check at the fitted values (D-028 item 4)",
        "environment": env,
        "analysis_code": ran,
        "amendment": code["amendment"],
        "config": config,
        **summarize(
            ordered,
            float(config["settings"]["level"]),
            config["planted"]["batch_var"],
            batch_failed=config.get("batch_fit_error") is not None,
        ),
    }
    (directory / "post_check.json").write_text(
        json.dumps(artifact, indent=1, sort_keys=True) + "\n"
    )
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m noisefloor.post_check run|bank PREREG [--workers N]``."""
    parser = argparse.ArgumentParser(description="post-E1 check at the fitted values")
    parser.add_argument("command", choices=["run", "bank"])
    parser.add_argument("prereg", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--amendment", type=Path, help="a frozen amendment of 0004 that allows other analysis code"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "run":
        added = run(args.prereg, workers=args.workers, amendment=args.amendment)
        print(f"added {added} datasets")
    else:
        result = bank(args.prereg, amendment=args.amendment)
        print(f"not validated: {', '.join(result['not_validated']) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
