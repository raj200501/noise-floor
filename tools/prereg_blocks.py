#!/usr/bin/env python3
"""Generate the JSON blocks of preregistrations 0004 (E1), 0005 (E0 part 2), and 0006 (G6).

``uv run python tools/prereg_blocks.py OUTDIR [--threads N]`` writes one ``<block>.json``
per block. The prose around the blocks is written by hand. Every value here is a design
parameter set by a DECISIONS.md entry (cited beside it); none is a result. The interval rules
and their headline status are read from the banked exploratory method pilots
(``artifacts/method_pilots/summary.json``, D-023, D-028, D-029), never typed here.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from evalstats import PairedDesign, design_from_logit, power_at
from noisefloor.analysis.mde import DESIGNS
from noisefloor.e1 import E1Design, claude_plan, local_plan, manifest, model_pins, plan_sha256
from noisefloor.models import CLAUDE_TIERS, MODELS
from noisefloor.recovery import FIXED_TERMS, Design, Planted

REPO = Path(__file__).resolve().parent.parent
SUMMARY = REPO / "artifacts" / "method_pilots" / "summary.json"
LOCAL_MODELS = ("qwen3-1.7b", "qwen3-4b", "granite-4.0-micro")  # D-025
CLAUDE = ("claude-haiku", "claude-sonnet", "claude-opus")  # D-019, D-030
REPLICATES = ((0, 1, 1, 0, 0), (0, 1, 1, 1, 1))  # D-028 item 3: minimal, 300 s, 15 steps
SEED = 20261105
G6_SEEDS = list(range(970000, 970200))
REFERENCE = {
    "harness": "minimal",
    "timeout": "300.0",
    "steps": "15",
    "temperature": "0.0",
    "concurrency": "1",
}
TASK_VAR, MODEL_TASK_VAR = 4.0, 0.5  # D-028 item 5
INTERACTIONS = {"harness": 0.25, "steps": 0.25, "timeout": 0.25}  # D-028 item 5
GLMM_REPLICATES, OTHER_REPLICATES = 500, 2000  # G5: 500 for GLMM intervals, else 2,000
N_TASKS = 30
# Local-arm groups at other intercepts, so the validated range spans low and high scores
# (review writing-5); Claude score cells from below one half to near the ceiling.
LOCAL_GROUPS = (
    ("e1-local-lo2", -3.0),
    ("e1-local-lo", -1.5),
    ("e1-local", None),
    ("e1-local-hi", 1.5),
)
CLAUDE_SCORE_INTERCEPTS = (-2.0, 0.0, 2.5, 3.5, 4.5, 5.5)
CLAUDE_DGP = {"task_var": 2.0, "model_task_var": 0.5}


def pilot_rules() -> dict[str, Any]:
    """The D-023 rule choices and the D-029 jackknife check, from the banked pilots."""
    summary = json.loads(SUMMARY.read_text())
    choice = summary["pools"]["nagq1-e1-shape"]["choice"]
    checks = [
        summary["pools"][p]["check"]["fixed"] for p in ("pilot7a-jackknife", "pilot7b-jackknife")
    ]
    return {
        "random": choice["random"]["rule"],
        "box": choice["fixed"]["rule"].removeprefix("box_"),
        "share": choice["share"]["rule"],
        "mde": choice["mde"]["rule"],
        "headline": {
            "random": choice["random"]["headline"],
            "fixed": all(c["headline"] for c in checks),
            "share": choice["share"]["headline"],
            "mde": choice["mde"]["headline"],
        },
    }


def e1_blocks(threads: int, rules: dict[str, Any]) -> dict[str, Any]:
    """The ``e1-plan``, ``e1-design``, and ``e1-analysis`` blocks of 0004."""
    design = E1Design(
        models=LOCAL_MODELS,
        claude_tiers=CLAUDE,
        tasks_per_family=6,
        runs_per_cell=16,
        order_seed=SEED,
        threads=threads,
        replicates=REPLICATES,
    )
    lp, cp = local_plan(design), claude_plan(design)
    tasks = manifest(design)
    levels: dict[str, set[str]] = {}
    for spec in [*lp, *cp]:
        for k, v in spec.factors.items():
            levels.setdefault(k, set()).add(v)
    models = [
        {
            "id": m,
            "family": MODELS[m].family,
            "tier": f"{MODELS[m].params_b}B {MODELS[m].quant}",
            "backend": "local",
        }
        for m in LOCAL_MODELS
    ]
    models += [
        {"id": t, "family": "claude", "tier": CLAUDE_TIERS[t], "backend": "claude"} for t in CLAUDE
    ]
    e1_design = {
        "factors": {k: sorted(v) for k, v in sorted(levels.items())},
        "models": models,
        "local_runs": len(lp),
        "replicate_runs": sum(1 for s in lp if s.phase == "e1-rep"),
        "claude_runs": len(cp),
        "tasks": tasks["n_tasks"],
        "task_manifest_sha256": tasks["sha256"],
        "suite_version": tasks["version"],
        "plan_sha256": plan_sha256(design),
        "model_pins": model_pins(design),
    }
    analysis = {
        "random": ["task", "model:task"],
        "fixed": list(FIXED_TERMS),
        "level": 0.95,
        "n_boot": 200,
        "nagq": 1,
        "boot_interval": rules["random"],
        "box_interval": rules["box"],
        "share_interval": rules["share"],
        "fixed_interval": "jackknife",  # D-028 item 1, D-029
        "score_interval": {"local": "cluster_t", "claude": "cluster_wilson"},
        "contrast_interval": {
            "harness-local": "cluster_t",
            "harness-claude": "mover_wilson",
            "harness-diff-local": "cluster_t",
            "models-local": "cluster_t",
            "tiers-claude": "mover_wilson",
            "models-claude": "mover_wilson",
        },
        "sampling_interval": "cluster_wilson",  # D-031 item 1
        "seed": SEED,
        "exclude_status": ["infra_error"],
        "block_hours": [[0, 12], [12, 24]],
        "mde": {
            "designs": [list(d) for d in DESIGNS],
            "reference": REFERENCE,
            "averaged": ["day", "block"],
            "interval": rules["mde"],
            "alpha": 0.05,
            "power": 0.8,
            "gaps": [0.02, 0.05, 0.1],
        },
    }
    return {"e1-plan": design.to_json(), "e1-design": e1_design, "e1-analysis": analysis}


def recovery_config(rules: dict[str, Any]) -> dict[str, Any]:
    """The ``recovery-config`` block (G6 and E0 part 2): the E1 local arm's shape."""
    n = len(LOCAL_MODELS)
    spread = tuple(round(0.6 - 1.2 * i / (n - 1), 4) for i in range(n))
    design = Design(n_models=n, n_tasks=N_TASKS, runs_per_cell=16, batch_size=10)
    planted = Planted(model=spread, task_var=TASK_VAR, model_task_var=MODEL_TASK_VAR, batch_var=0.0)
    return {
        "design": dict(design.__dict__),
        "planted": planted.to_json(),
        "settings": {
            "n_boot": 200,
            "nagq": 1,
            "boot_interval": rules["random"],
            "box_interval": rules["box"],
            "share_interval": rules["share"],
            "fixed_interval": "jackknife",
            "level": 0.95,
            "random": ["task", "model:task"],
            "mde_designs": [list(d) for d in DESIGNS],
            "mde_interval": rules["mde"],
            "score_interval": "cluster_t",
            "contrast_interval": "cluster_t",
        },
    }


def g6_components() -> list[str]:
    """Every component a G6 dataset reports (:func:`noisefloor.recovery.recover`)."""
    random = ["task", "model:task"]
    fixed = list(FIXED_TERMS)
    models = [f"m{i}" for i in range(len(LOCAL_MODELS))]
    pairs = [f"{a}-{b}" for i, a in enumerate(models) for b in models[i + 1 :]]
    names = [*random, *fixed]
    names += [f"share:{c}" for c in [*names, "residual"]]
    names += [f"mde:T{t}:k{k}" for t, k in DESIGNS]
    names += [f"score:{m}" for m in models] + [f"harness:{m}" for m in models]
    names += [f"hdiff:{p}" for p in pairs] + [f"pair:{p}" for p in pairs]
    return sorted(names)


def _tag(x: float) -> str:
    return str(x).replace(".", "").replace("-", "m")


def e0_grid(rules: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str]]:
    """The 0005 grid, its headline map, and the kind of every headline cell."""
    grid: list[dict[str, Any]] = []
    headline: dict[str, str] = {}
    kinds: dict[str, str] = {}

    def cell(name: str, method: str, interval: str, params: dict[str, Any], reps: int) -> None:
        grid.append(
            {
                "cell": name,
                "method": method,
                "interval_type": interval,
                "nominal": 0.95,
                "replicates": reps,
                "params": {"n_tasks": N_TASKS, **params},
            }
        )

    def head(label: str, name: str, kind: str) -> None:
        headline[label] = name
        kinds[name] = kind

    # E1-shaped GLMM cells: group e1 (the planted values of recovery-config) and group
    # e1-het (task-by-factor variation outside the analysis model, D-028 item 5).
    for group, planted, fixed_only in (
        ("e1", {}, False),
        ("e1-het", {"interactions": INTERACTIONS}, True),
    ):
        tag = "e1" if group == "e1" else "het"
        terms = list(FIXED_TERMS) if fixed_only else ["task", "model:task", *FIXED_TERMS]
        for comp in terms:
            name = f"G-{tag}-{comp.replace(':', '_')}"
            kind = "fixed" if comp in FIXED_TERMS else "random"
            params = {"group": group, "component": comp, "planted": planted}
            cell(name, "glmm_component", "parametric_bootstrap", params, GLMM_REPLICATES)
            if rules["headline"][kind]:
                head(f"e1 {kind} component {comp} ({group})", name, kind)
        for comp in [*terms, *([] if fixed_only else ["residual"])]:
            name = f"H-{tag}-{comp.replace(':', '_')}"
            params = {"group": group, "component": f"share:{comp}", "planted": planted}
            cell(name, "glmm_component", "parametric_bootstrap", params, GLMM_REPLICATES)
            if rules["headline"]["share"]:
                head(f"e1 share of {comp} ({group})", name, "share")
        if not fixed_only:
            for t, k in DESIGNS:
                name = f"M-e1-T{t}-k{k}"
                params = {"group": group, "component": f"mde:T{t}:k{k}", "planted": planted}
                cell(name, "glmm_component", "parametric_bootstrap", params, GLMM_REPLICATES)
                if rules["headline"]["mde"]:
                    head(f"e1 MDE {t} tasks x {k} runs", name, "mde")
    # Local-arm scores and contrasts on E1-shaped datasets (no GLMM fit), at the planted
    # intercept and at intercepts below and above it, so local scores from about 0.25 to
    # 0.75 lie inside the validated range (review writing-5).
    models = [f"m{i}" for i in range(len(LOCAL_MODELS))]
    pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1 :]]
    for group, intercept in LOCAL_GROUPS:
        planted = {} if intercept is None else {"intercept": intercept}
        tag = group.removeprefix("e1-local").lstrip("-") or "mid"
        comps = [f"score:{m}" for m in models] + [f"harness:{m}" for m in models]
        comps += [f"hdiff:{a}-{b}" for a, b in pairs] + [f"pair:{a}-{b}" for a, b in pairs]
        for comp in comps:
            name = f"L-{tag}-{comp.replace(':', '-')}"
            method = "score_interval" if comp.startswith("score") else "paired_difference"
            params = {"group": group, "analysis": "local", "component": comp, "planted": planted}
            cell(name, method, "cluster_t", params, OTHER_REPLICATES)
            kind = comp.split(":")[0]
            head(f"e1 local {kind} {comp.split(':')[1]} ({tag})", name, f"local-{kind}")
    dgp = {
        "task_sd": round(math.sqrt(CLAUDE_DGP["task_var"]), 4),
        "model_task_sd": round(math.sqrt(CLAUDE_DGP["model_task_var"]), 4),
    }
    # Claude tiers: scores (2 runs per task), tier pairs under one harness (1 run per
    # side), within-tier harness contrasts with the timeout split (stats-4), and
    # Claude-local comparisons (1 run against 4).
    for c in CLAUDE_SCORE_INTERCEPTS:
        name = f"S-c-w-i{_tag(c)}"
        cell(
            name,
            "score_interval",
            "cluster_wilson",
            {"runs": 2, **dgp, "intercept": c},
            OTHER_REPLICATES,
        )
        head(f"e1 Claude score, intercept {c}", name, "claude-score")
    for c, effect in ((-2.0, 0.4), (0.0, 0.4), (2.5, 0.4), (3.5, 0.4), (4.5, 0.4), (3.5, 0.0)):
        name = f"P-c-w-i{_tag(c)}-e{_tag(effect)}"
        params = {"runs": 1, **dgp, "intercept": c, "effect_a": effect, "effect_b": 0.0}
        cell(name, "paired_difference", "mover_wilson", params, OTHER_REPLICATES)
        head(f"e1 Claude tier pair, intercept {c}, effect {effect}", name, "claude-tiers")
        name = f"P-cw-w-i{_tag(c)}-e{_tag(effect)}"
        within = {**params, "within": True, "split_shift": 0.4}
        cell(name, "paired_difference", "mover_wilson", within, OTHER_REPLICATES)
        head(f"e1 Claude harness contrast, intercept {c}, effect {effect}", name, "claude-harness")
    for effect in (0.4, 2.5, 3.5, 4.5):
        name = f"P-cl-w-e{_tag(effect)}"
        params = {
            "runs": 1,
            "runs_b": 4,
            **dgp,
            "intercept": 0.0,
            "effect_a": effect,
            "effect_b": 0.0,
        }
        cell(name, "paired_difference", "mover_wilson", params, OTHER_REPLICATES)
        head(f"e1 Claude-local comparison, effect {effect}", name, "claude-local")
    # The sampling table: discordance of replicate pairs, 3 models per task (D-031 item 1).
    for c in (0.0, 2.5, 3.5):
        name = f"S-r-w-i{_tag(c)}"
        cell(
            name,
            "score_interval",
            "cluster_wilson",
            {"runs": 3, **dgp, "intercept": c},
            OTHER_REPLICATES,
        )
        head(f"e1 sampling table, intercept {c}", name, "sampling")
    # E3 bounds on local scores: ambiguous runs among true failures and among true
    # successes (the truth at an end of the set, review stats-10); completely at random as
    # reported extras.
    bound = {"runs": 16, "intercept": 0.0, "task_sd": round(math.sqrt(TASK_VAR), 4)}
    for rate in (0.05, 0.4):
        for among in ("fail", "success"):
            name = f"B-e1-{among}-a{_tag(rate)}"
            params = {**bound, "ambiguous_rate": rate, "ambiguous_among": among}
            cell(name, "bounded_score", "imbens_manski", params, OTHER_REPLICATES)
            head(f"e3 bounds on E1 local scores, ambiguous {among} share {rate}", name, "bounds")
        name = f"B-e1-any-a{_tag(rate)}"
        cell(
            name,
            "bounded_score",
            "imbens_manski",
            {**bound, "ambiguous_rate": rate},
            OTHER_REPLICATES,
        )
    # Reported extras (not headline): the t interval at 2 runs per task near the ceiling,
    # beside the Wilson cells it is compared with (review writing-25).
    for c in (2.5, 3.5, 4.5):
        params = {"runs": 2, **dgp, "intercept": c}
        cell(f"S-c-t-i{_tag(c)}", "score_interval", "cluster_t", params, OTHER_REPLICATES)
    return grid, headline, kinds


def _gh(nodes: int = 40) -> tuple[Any, Any]:
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    return x, w / w.sum()


def within_design(eta: float, effect: float, sd: float) -> tuple[float, Any]:
    """Probability-scale effect and paired-design inputs of a within-model contrast.

    Both sides share one random effect with SD ``sd``; ``within`` sums the two sides'
    run-level Bernoulli variances (divided by runs per side in the power formula).
    """
    x, w = _gh()
    p1 = 1 / (1 + np.exp(-(eta + effect + sd * x)))
    p0 = 1 / (1 + np.exp(-(eta + sd * x)))
    diff = p1 - p0
    mean = float(np.sum(w * diff))
    between = float(np.sum(w * diff**2)) - mean**2
    within = float(np.sum(w * (p1 * (1 - p1) + p0 * (1 - p0))))
    return mean, PairedDesign(between_task_var=max(0.0, between), within_task_var=within)


def e1_power(rules: dict[str, Any]) -> dict[str, Any]:
    """The ``e1-power`` block: the preregistered power analysis (writing-14, stats-12).

    Power to read each fixed factor as non-zero comes from the banked pilot 7 (the E1
    shape, the task-jackknife rule; exploratory), with the files' SHA-256. Power of each
    confirmatory contrast family is the normal-approximation power of
    :func:`evalstats.power_at` at stated, assumed effects and the planted variances.
    """
    summary = json.loads(SUMMARY.read_text())

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    pilots = REPO / "artifacts" / "method_pilots"
    factor_power = {}
    for name in FIXED_TERMS:
        factor_power[name] = {
            pool: summary["pools"][f"pilot7{pool}-jackknife"]["components"][name]["above_zero"][
                "jackknife"
            ]
            for pool in ("a", "b")
        }
    planted = Planted(model=(0.6, 0.0, -0.6), task_var=TASK_VAR, model_task_var=MODEL_TASK_VAR)
    others = sum(planted.factors[f] for f in planted.factors if f != "harness") / 2
    base = planted.intercept + others + sum(planted.day) / 3 + sum(planted.block) / 2
    sd_cell = math.sqrt(planted.task_var + planted.model_task_var)
    families = []
    for i, effect in enumerate(planted.model):
        diff, design = within_design(base + effect, planted.factors["harness"], sd_cell)
        families.append(
            {
                "family": "harness-local",
                "comparison": f"harness contrast, model position {i + 1}",
                "assumed_logit_effect": planted.factors["harness"],
                "tasks": N_TASKS,
                "runs_per_side": 8,
                "effect": diff,
                "power": power_at(diff, design, N_TASKS, 8),
            }
        )
    for i, a in enumerate(planted.model):
        for j in range(i + 1, len(planted.model)):
            b = planted.model[j]
            design = design_from_logit(base + a, base + b, planted.task_var, planted.model_task_var)
            gap = within_design(base + b, a - b, sd_cell)[0]
            families.append(
                {
                    "family": "models-local",
                    "comparison": f"model positions {i + 1} and {j + 1}",
                    "assumed_logit_effect": round(a - b, 4),
                    "tasks": N_TASKS,
                    "runs_per_side": 8,
                    "effect": gap,
                    "power": power_at(gap, design, N_TASKS, 8),
                }
            )
    # A harness difference between two local models: one model's harness effect is 0.5
    # on the logit scale larger than the other's (an assumed interaction, not planted).
    # The two contrasts share each task's effect, which the sum of variances below ignores,
    # so this power is conservative.
    diff_hi, design_hi = within_design(base, 0.5, sd_cell)
    diff_lo, design_lo = within_design(base, 0.0, sd_cell)
    both = PairedDesign(
        between_task_var=design_hi.between_task_var + design_lo.between_task_var,
        within_task_var=design_hi.within_task_var + design_lo.within_task_var,
    )
    families.append(
        {
            "family": "harness-diff-local",
            "comparison": "harness contrasts of two models differing by 0.5 on the logit scale",
            "assumed_logit_effect": 0.5,
            "tasks": N_TASKS,
            "runs_per_side": 8,
            "effect": diff_hi - diff_lo,
            "power": power_at(diff_hi - diff_lo, both, N_TASKS, 8),
        }
    )
    claude_sd = math.sqrt(2.0 + 0.5)
    for c in (0.0, 2.5, 3.5):
        diff, design = within_design(c, 0.4, claude_sd)
        families.append(
            {
                "family": "harness-claude",
                "comparison": f"Claude Code minus minimal-agent, tier intercept {c}",
                "assumed_logit_effect": 0.4,
                "tasks": N_TASKS,
                "runs_per_side": 1,
                "effect": diff,
                "power": power_at(diff, design, N_TASKS, 1),
            }
        )
        pair = design_from_logit(c + 0.4, c, 2.0, 0.5)
        gap = within_design(c, 0.4, claude_sd)[0]
        families.append(
            {
                "family": "tiers-claude",
                "comparison": f"two tiers 0.4 apart on the logit scale, intercept {c}",
                "assumed_logit_effect": 0.4,
                "tasks": N_TASKS,
                "runs_per_side": 1,
                "effect": gap,
                "power": power_at(gap, pair, N_TASKS, 1),
            }
        )
    return {
        "label": (
            "design calculations at assumed effects and planted variances (task variance "
            f"{TASK_VAR}, model:task {MODEL_TASK_VAR}; Claude cells task 2.0, model:task 0.5), "
            "not estimates; normal approximation, two-sided alpha 0.05"
        ),
        "fixed_factor_power": {
            "rule": "task jackknife box lower end above 0 (D-028 item 1)",
            "source": "artifacts/method_pilots/summary.json pools pilot7a-jackknife (no "
            "task-by-factor variation) and pilot7b-jackknife (variances 0.25), "
            "components.<factor>.above_zero.jackknife",
            "summary_sha256": digest(SUMMARY),
            "pilot7a_sha256": digest(pilots / "pilot7a.jsonl"),
            "pilot7b_sha256": digest(pilots / "pilot7b.jsonl"),
            "power": factor_power,
        },
        "families": families,
    }


def validated_range(config: dict[str, Any]) -> dict[str, Any]:
    """Planted success rates of the local groups and the Claude score cells.

    These are what the headline cells span: an E1 score outside the range of its kind's
    cells is reported as outside the validated range (prereg 0004).
    """
    from noisefloor.e0 import expected_sigmoid
    from noisefloor.recovery import config_from, planted_success, simulate

    design, planted, _ = config_from(config)
    data = simulate(design, planted, 1)
    local = {}
    for group, intercept in LOCAL_GROUPS:
        changed = (
            planted if intercept is None else dataclasses.replace(planted, intercept=intercept)
        )
        success = planted_success(data, changed)
        local[group] = [
            float(success[(data["model"] == f"m{i}").to_numpy()].mean())
            for i in range(design.n_models)
        ]
    sd = math.sqrt(CLAUDE_DGP["task_var"] + CLAUDE_DGP["model_task_var"])
    claude = {str(c): expected_sigmoid(c, sd) for c in CLAUDE_SCORE_INTERCEPTS}
    return {
        "label": "planted success rates, design values (not results)",
        "local_scores_by_group": local,
        "claude_score_by_intercept": claude,
    }


def main() -> int:
    """Write every block into OUTDIR."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    rules = pilot_rules()
    grid, headline, kinds = e0_grid(rules)
    blocks = {
        **e1_blocks(args.threads, rules),
        "recovery-config": recovery_config(rules),
        "g6-seeds": G6_SEEDS,
        "g6-components": g6_components(),
        "e1-power": e1_power(rules),
        "e0-grid": grid,
        "headline": headline,
        "headline-kinds": kinds,
        "validated-range": validated_range(recovery_config(rules)),
        "rules": rules,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, value in blocks.items():
        (args.outdir / f"{name}.json").write_text(json.dumps(value, indent=1) + "\n")
    design = blocks["e1-design"]
    print(
        json.dumps(
            {
                "local_runs": design["local_runs"],
                "claude_runs": design["claude_runs"],
                "plan_sha256": design["plan_sha256"],
                "cells": len(grid),
                "headline": len(headline),
                "rules": rules,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
