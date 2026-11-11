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

from evalstats import PairedDesign, design_from_logit, power_at, tasks_needed
from noisefloor.analysis.mde import DESIGNS
from noisefloor.e1 import (
    E1Design,
    claude_plan,
    local_plan,
    manifest,
    model_pins,
    plan_sha256,
    planned_batches,
    server_pin,
)
from noisefloor.models import CLAUDE_TIERS, MODELS
from noisefloor.recovery import FIXED_TERMS, Design, Planted
from noisefloor.suite import FAMILY_NAMES

REPO = Path(__file__).resolve().parent.parent
SUMMARY = REPO / "artifacts" / "method_pilots" / "summary.json"
DRY_RUN = REPO / "artifacts" / "pilot" / "e1-dry.jsonl"
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
# D-033: the power analysis sets the local sample (the smallest tasks per family at which
# every local harness contrast and model pair has power 0.8 under the Holm reading at the
# assumed effects, with the t statistics of their cluster_t tests, D-034; e1_power checks
# it); the Claude slice keeps the first 6 tasks per family, as its caps bind; the local
# arm spreads over 4 days at the dry run's rates.
LOCAL_TASKS_PER_FAMILY, CLAUDE_TASKS_PER_FAMILY, DAYS = 17, 6, 4
N_LOCAL_TASKS = len(FAMILY_NAMES) * LOCAL_TASKS_PER_FAMILY
N_CLAUDE_TASKS = len(FAMILY_NAMES) * CLAUDE_TASKS_PER_FAMILY
DAY_EFFECTS = (0.0, 0.1, -0.1, 0.05)  # D-032 values for days 1 to 3; day 4 added by D-033
POWER_TARGET = 0.8
# Members of each Holm family in E1 (prereg 0004 item 7): 3 local models, 3 local pairs,
# 3 local harness differences, 3 tiers, 3 tier pairs under each of 2 harnesses, and every
# tier against every local model.
FAMILY_SIZE = {
    "harness-local": 3,
    "models-local": 3,
    "harness-diff-local": 3,
    "harness-claude": 3,
    "tiers-claude": 6,
    "models-claude": 9,
}
SIZING_FAMILIES = ("harness-local", "models-local")  # the families that set the sample
HOLM_DRAWS = 200_000
# Each contrast family's interval (prereg 0004 item 7). The power rows of a cluster_t
# family use t statistics on tasks - 1 degrees of freedom, as its test does (review
# final-stats-5); the Wilson families use normal statistics.
CONTRAST_INTERVAL = {
    "harness-local": "cluster_t",
    "harness-claude": "mover_wilson",
    "harness-diff-local": "cluster_t",
    "models-local": "cluster_t",
    "tiers-claude": "mover_wilson",
    "models-claude": "mover_wilson",
}
# Cells that plant no difference, so a kind's validated span starts at 0 (final-stats-7):
# model m0 of e1-local-mh has harness effect 0.5 - 0.5, and e1-local-eq plants m0 = m1.
NULL_CELLS = {
    "local-harness": ("e1-local-mh", "harness:m0"),
    "local-hdiff": ("e1-local-eq", "hdiff:m0-m1"),
    "local-pair": ("e1-local-eq", "pair:m0-m1"),
}
# Local-arm groups: at other intercepts, so the validated range spans low and high scores
# (review writing-5); with model-by-harness effects, so it spans harness differences of
# the size E1 can detect; and with two equal models, so it spans pairs near 0 (brief-10).
LOCAL_GROUPS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("e1-local-lo2", {"intercept": -3.0}),
    ("e1-local-lo", {"intercept": -1.5}),
    ("e1-local", {}),
    ("e1-local-hi", {"intercept": 1.5}),
    ("e1-local-mh", {"model_factors": {"harness": [-0.5, 0.0, 0.5]}}),
    ("e1-local-eq", {"model": [0.3, 0.3, -0.3]}),
)
DIFFERENCE_KINDS = (  # compared in absolute value: a difference and its reverse mirror
    "local-harness",
    "local-hdiff",
    "local-pair",
    "claude-tiers",
    "claude-harness",
    "claude-local",
)
CLAUDE_SCORE_INTERCEPTS = (-2.0, 0.0, 2.5, 3.5, 4.5, 5.5)
# (intercept, logit effect) of the tier-pair and harness cells, up to large gaps (brief-10).
CLAUDE_PAIR_CELLS = (
    (-2.0, 0.4),
    (0.0, 0.4),
    (2.5, 0.4),
    (3.5, 0.4),
    (4.5, 0.4),
    (3.5, 0.0),
    (0.0, 1.0),
    (0.0, 2.0),
    (2.5, 1.0),
    (2.5, 2.0),
)
# (local intercept, tier offset) of the Claude-local cells: from no gap to large gaps over
# low local rates (review stats-9).
CLAUDE_LOCAL_CELLS = (
    (0.0, 0.0),
    (0.0, 0.4),
    (0.0, 2.5),
    (0.0, 3.5),
    (0.0, 4.5),
    (-1.5, 3.5),
    (-1.5, 6.0),
    (-3.0, 4.5),
    (-3.0, 6.0),
)
# Sampling-table cells: the local planted values, reference intercepts from E[p (1 - p)]
# near its largest value down to about 0.02 (review resolution-12).
SAMPLING_INTERCEPTS = (0.0, 2.0, 4.0, 6.0)
# E3 bound cells: (intercept, ambiguous rate, among), up to the timeout shares of the dry run.
BOUND_CELLS = (
    (0.0, 0.05, "fail"),
    (0.0, 0.05, "success"),
    (0.0, 0.4, "fail"),
    (0.0, 0.4, "success"),
    (-1.5, 0.8, "fail"),
    (1.5, 0.8, "fail"),
)
CLAUDE_DGP = {"task_var": 2.0, "model_task_var": 0.5}


def pilot_rules() -> dict[str, Any]:
    """The D-023 rule choices and the D-029 jackknife check, from the banked pilots."""
    summary = json.loads(SUMMARY.read_text())
    choice = summary["pools"]["nagq1-e1-shape"]["choice"]
    checks = [summary["pools"][p]["check"] for p in ("pilot7a-jackknife", "pilot7b-jackknife")]
    return {
        "random": choice["random"]["rule"],
        "box": choice["fixed"]["rule"].removeprefix("box_"),
        "share": choice["share"]["rule"],
        "mde": choice["mde"]["rule"],
        "headline": {
            "random": choice["random"]["headline"],
            "fixed": all(c["fixed"]["headline"] for c in checks),
            # Pilot 6 scored the shares mapped from bootstrap boxes; pilot 7 the fixed-factor
            # shares mapped from the task-jackknife boxes E1 uses (D-033).
            "share": choice["share"]["headline"]
            and all(c["fixed-share"]["headline"] for c in checks),
            "mde": choice["mde"]["headline"],
        },
    }


def e1_blocks(threads: int, rules: dict[str, Any]) -> dict[str, Any]:
    """The ``e1-plan``, ``e1-design``, and ``e1-analysis`` blocks of 0004."""
    design = make_design(threads)
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
        "claude_tasks": len({(s.family, s.task_seed, s.difficulty) for s in cp}),
        "task_manifest_sha256": tasks["sha256"],
        "suite_version": tasks["version"],
        "plan_sha256": plan_sha256(design),
        "model_pins": model_pins(design),
        "llama_server": server_pin(),
        "duration_estimate": duration_estimate(design),
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
        "contrast_interval": dict(CONTRAST_INTERVAL),
        "sampling_interval": "cluster_wilson",  # D-031 item 1
        "validated_range": "prereg/0005-e0-glmm-cells.md",  # its validated-range block
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


def make_design(threads: int) -> E1Design:
    """The E1 design of prereg 0004 (D-025, D-028, D-031, D-033)."""
    return E1Design(
        models=LOCAL_MODELS,
        claude_tiers=CLAUDE,
        tasks_per_family=LOCAL_TASKS_PER_FAMILY,
        claude_tasks_per_family=CLAUDE_TASKS_PER_FAMILY,
        days=DAYS,
        runs_per_cell=16,
        order_seed=SEED,
        threads=threads,
        replicates=REPLICATES,
    )


def dry_rates(rows: list[dict[str, Any]], threads: int) -> dict[str, float]:
    """Seconds per run of each (model, concurrency) batch of the dry run at ``threads``.

    A batch's time runs from its start (server start included) to its last run's end.
    """
    from datetime import datetime

    batches: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if int(row["threads"]) == threads:
            batches.setdefault(str(row["batch_id"]), []).append(row)
    rates: dict[str, float] = {}
    for members in batches.values():
        start = datetime.fromisoformat(str(members[0]["batch_started_at"]))
        end = max(datetime.fromisoformat(str(r["finished_at"])) for r in members)
        key = f"{members[0]['model']}/c{members[0]['concurrency']}"
        rates[key] = (end - start).total_seconds() / len(members)
    return dict(sorted(rates.items()))


def duration_estimate(design: E1Design) -> dict[str, Any]:
    """Hours each slot of the local plan takes at the dry run's rates (a planning estimate).

    Also the hours if every run used its full wall-clock limit (two at a time at
    concurrency 2), an upper bound for runs that do not fail early.
    """
    rows = [json.loads(ln) for ln in DRY_RUN.read_text().splitlines() if ln.strip()]
    rates = dry_rates(rows, design.threads)
    slots: dict[str, dict[str, float]] = {}
    for label, specs in planned_batches(local_plan(design)).items():
        slot = label.split("/")[0]
        entry = slots.setdefault(slot, {"runs": 0, "hours": 0.0, "hours_at_limits": 0.0})
        key = f"{specs[0].model}/c{specs[0].concurrency}"
        entry["runs"] += len(specs)
        entry["hours"] += len(specs) * rates[key] / 3600
        in_flight = specs[0].concurrency  # runs at a time
        entry["hours_at_limits"] += sum(s.timeout_s for s in specs) / in_flight / 3600
    return {
        "label": "planning estimate from the exploratory dry run (pilot seeds), not a result",
        "source": "artifacts/pilot/e1-dry.jsonl",
        "source_sha256": hashlib.sha256(DRY_RUN.read_bytes()).hexdigest(),
        "seconds_per_run": rates,
        "slots": {k: {n: round(v, 2) for n, v in e.items()} for k, e in slots.items()},
        "max_slot_hours": round(max(e["hours"] for e in slots.values()), 2),
        "block_hours": 12,
    }


def recovery_config(rules: dict[str, Any]) -> dict[str, Any]:
    """The ``recovery-config`` block (G6 and E0 part 2): the E1 local arm's shape."""
    n = len(LOCAL_MODELS)
    spread = tuple(round(0.6 - 1.2 * i / (n - 1), 4) for i in range(n))
    design = Design(
        n_models=n,
        n_tasks=N_LOCAL_TASKS,
        runs_per_cell=16,
        batch_size=10,
        days=DAYS,
        blocks_per_day=2,
        replicates=REPLICATES,
    )
    planted = Planted(
        model=spread,
        day=DAY_EFFECTS,
        task_var=TASK_VAR,
        model_task_var=MODEL_TASK_VAR,
        batch_var=0.0,
    )
    shape = dict(design.__dict__)
    shape["replicates"] = [list(r) for r in design.replicates]
    return {
        "design": shape,
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
    """The 0005 grid, its headline map, and the kind of every headline cell.

    Each cell's method and interval type name the library function and interval label its
    validity check uses (noisefloor.e0_validity.FUNCTIONS), so a failing cell makes the
    function E1 calls warn: ``glmm_component`` cells check ``decompose`` (random variances
    ``parametric_bootstrap``, fixed contributions ``glmm_task_jackknife``, shares
    ``glmm_plugin_share``), ``glmm_mde`` cells the MDE replicates, and the harness
    differences ``contrast_difference``.
    """
    grid: list[dict[str, Any]] = []
    headline: dict[str, str] = {}
    kinds: dict[str, str] = {}

    def cell(
        name: str, method: str, interval: str, params: dict[str, Any], reps: int, tasks: int
    ) -> None:
        grid.append(
            {
                "cell": name,
                "method": method,
                "interval_type": interval,
                "nominal": 0.95,
                "replicates": reps,
                "params": {"n_tasks": tasks, **params},
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
            interval = "glmm_task_jackknife" if kind == "fixed" else "parametric_bootstrap"
            params = {"group": group, "component": comp, "planted": planted}
            cell(name, "glmm_component", interval, params, GLMM_REPLICATES, N_LOCAL_TASKS)
            if rules["headline"][kind]:
                head(f"e1 {kind} component {comp} ({group})", name, kind)
        for comp in [*terms, *([] if fixed_only else ["residual"])]:
            name = f"H-{tag}-{comp.replace(':', '_')}"
            params = {"group": group, "component": f"share:{comp}", "planted": planted}
            cell(
                name, "glmm_component", "glmm_plugin_share", params, GLMM_REPLICATES, N_LOCAL_TASKS
            )
            if rules["headline"]["share"]:
                head(f"e1 share of {comp} ({group})", name, "share")
        if not fixed_only:
            for t, k in DESIGNS:
                name = f"M-e1-T{t}-k{k}"
                params = {"group": group, "component": f"mde:T{t}:k{k}", "planted": planted}
                cell(
                    name, "glmm_mde", "parametric_bootstrap", params, GLMM_REPLICATES, N_LOCAL_TASKS
                )
                if rules["headline"]["mde"]:
                    head(f"e1 MDE {t} tasks x {k} runs", name, "mde")
    # Local-arm scores and contrasts on E1-shaped datasets (no GLMM fit), at the planted
    # intercept, at intercepts below and above it, and with model-by-harness effects.
    models = [f"m{i}" for i in range(len(LOCAL_MODELS))]
    pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1 :]]
    methods = {
        "score": "score_interval",
        "harness": "paired_difference",
        "hdiff": "contrast_difference",
        "pair": "paired_difference",
    }
    for group, planted in LOCAL_GROUPS:
        tag = group.removeprefix("e1-local").lstrip("-") or "mid"
        comps = [f"score:{m}" for m in models] + [f"harness:{m}" for m in models]
        comps += [f"hdiff:{a}-{b}" for a, b in pairs] + [f"pair:{a}-{b}" for a, b in pairs]
        for comp in comps:
            name = f"L-{tag}-{comp.replace(':', '-')}"
            kind = comp.split(":")[0]
            params = {"group": group, "analysis": "local", "component": comp, "planted": planted}
            cell(name, methods[kind], "cluster_t", params, OTHER_REPLICATES, N_LOCAL_TASKS)
            head(f"e1 local {kind} {comp.split(':')[1]} ({tag})", name, f"local-{kind}")
    dgp = {
        "task_sd": round(math.sqrt(CLAUDE_DGP["task_var"]), 4),
        "model_task_sd": round(math.sqrt(CLAUDE_DGP["model_task_var"]), 4),
    }
    # Claude tiers: scores (2 runs per task), tier pairs under one harness (1 run per
    # side, the same timeout on both sides, D-033), within-tier harness contrasts with the
    # timeout split (stats-4), and Claude-local comparisons (1 run against 4).
    for c in CLAUDE_SCORE_INTERCEPTS:
        name = f"S-c-w-i{_tag(c)}"
        params = {"runs": 2, **dgp, "intercept": c}
        cell(name, "score_interval", "cluster_wilson", params, OTHER_REPLICATES, N_CLAUDE_TASKS)
        head(f"e1 Claude score, intercept {c}", name, "claude-score")
    for c, effect in CLAUDE_PAIR_CELLS:
        name = f"P-c-w-i{_tag(c)}-e{_tag(effect)}"
        params = {"runs": 1, **dgp, "intercept": c, "effect_a": effect, "effect_b": 0.0}
        cell(name, "paired_difference", "mover_wilson", params, OTHER_REPLICATES, N_CLAUDE_TASKS)
        head(f"e1 Claude tier pair, intercept {c}, effect {effect}", name, "claude-tiers")
        name = f"P-cw-w-i{_tag(c)}-e{_tag(effect)}"
        within = {**params, "within": True, "split_shift": 0.4}
        cell(name, "paired_difference", "mover_wilson", within, OTHER_REPLICATES, N_CLAUDE_TASKS)
        head(f"e1 Claude harness contrast, intercept {c}, effect {effect}", name, "claude-harness")
    for c, effect in CLAUDE_LOCAL_CELLS:
        name = f"P-cl-w-i{_tag(c)}-e{_tag(effect)}"
        params = {
            "runs": 1,
            "runs_b": 4,
            **dgp,
            "intercept": c,
            "effect_a": effect,
            "effect_b": 0.0,
        }
        cell(name, "paired_difference", "mover_wilson", params, OTHER_REPLICATES, N_CLAUDE_TASKS)
        label = f"e1 Claude-local comparison, local intercept {c}, tier {effect} above"
        head(label, name, "claude-local")
    # The sampling table: two independent runs per model and local task at one
    # configuration, the discordance indicator, at the local planted values (D-031 item 1).
    local_dgp = {
        "task_sd": round(math.sqrt(TASK_VAR), 4),
        "model_task_sd": round(math.sqrt(MODEL_TASK_VAR), 4),
        "effects": [0.6, 0.0, -0.6],
    }
    for c in SAMPLING_INTERCEPTS:
        name = f"S-r-w-i{_tag(c)}"
        params = {**local_dgp, "intercept": c}
        cell(
            name, "sampling_discordance", "cluster_wilson", params, OTHER_REPLICATES, N_LOCAL_TASKS
        )
        head(f"e1 sampling table, intercept {c}", name, "sampling")
    # Less spread across tasks puts more runs near one half: a larger E[p (1 - p)].
    params = {**local_dgp, "task_sd": 1.0, "intercept": 0.0}
    name = "S-r-w-i00-t1"
    cell(name, "sampling_discordance", "cluster_wilson", params, OTHER_REPLICATES, N_LOCAL_TASKS)
    head("e1 sampling table, intercept 0.0, task variance 1.0", name, "sampling")
    # E3 bounds on local scores: ambiguous runs among true failures and among true
    # successes (the truth at an end of the set, review stats-10), up to large ambiguous
    # shares (review brief-10); among all runs as reported extras.
    bound = {"runs": 16, "task_sd": round(math.sqrt(TASK_VAR), 4)}
    for c, rate, among in BOUND_CELLS:
        name = f"B-e1-{among}-a{_tag(rate)}" + ("" if c == 0.0 else f"-i{_tag(c)}")
        params = {**bound, "intercept": c, "ambiguous_rate": rate, "ambiguous_among": among}
        cell(name, "bounded_score", "imbens_manski", params, OTHER_REPLICATES, N_LOCAL_TASKS)
        side = "failures" if among == "fail" else "successes"
        label = f"e3 bounds on E1 local scores, ambiguous rate {rate} among {side}, intercept {c}"
        head(label, name, "bounds")
    for rate in (0.05, 0.4):
        params = {**bound, "intercept": 0.0, "ambiguous_rate": rate}
        cell(
            f"B-e1-any-a{_tag(rate)}",
            "bounded_score",
            "imbens_manski",
            params,
            OTHER_REPLICATES,
            N_LOCAL_TASKS,
        )
    # Reported extras (not headline): the t interval at 2 runs per task near the ceiling,
    # beside the Wilson cells it is compared with (review writing-25).
    for c in (2.5, 3.5, 4.5):
        params = {"runs": 2, **dgp, "intercept": c}
        cell(
            f"S-c-t-i{_tag(c)}",
            "score_interval",
            "cluster_t",
            params,
            OTHER_REPLICATES,
            N_CLAUDE_TASKS,
        )
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


def unequal_design(
    eta_a: float, eta_b: float, task_var: float, model_task_var: float, runs_a: int, runs_b: int
) -> PairedDesign:
    """Paired-design inputs for ``runs_a`` runs of model a against ``runs_b`` of model b.

    As :func:`evalstats.design_from_logit`, with each side's Bernoulli variance divided by
    its own runs per task; use the result with one run per task.
    """
    x, w = _gh()
    u = math.sqrt(task_var) * x[:, None, None]
    s = math.sqrt(model_task_var)
    pa = 1 / (1 + np.exp(-(eta_a + u + s * x[None, :, None])))
    pb = 1 / (1 + np.exp(-(eta_b + u + s * x[None, None, :])))
    weight = w[:, None, None] * w[None, :, None] * w[None, None, :]
    diff = pa - pb
    mean = float(np.sum(weight * diff))
    between = max(0.0, float(np.sum(weight * diff**2)) - mean**2)
    within = float(np.sum(weight * (pa * (1 - pa) / runs_a + pb * (1 - pb) / runs_b)))
    return PairedDesign(between_task_var=between, within_task_var=within)


def _row(
    family: str,
    comparison: str,
    logit: float,
    effect: float,
    design: PairedDesign,
    tasks: int,
    runs: int,
) -> dict[str, Any]:
    needed = tasks_needed(abs(effect), design, runs, power=POWER_TARGET) if effect else None
    shift = abs(effect) / design.se(tasks, runs) if effect else 0.0
    df = tasks - 1 if CONTRAST_INTERVAL[family] == "cluster_t" else None
    return {
        "family": family,
        "comparison": comparison,
        "assumed_logit_effect": logit,
        "tasks": tasks,
        "runs_per_side": runs,
        "effect": effect,
        "test": f"t on {df} df" if df is not None else "normal",
        "power": power_at(effect, design, tasks, runs) if df is None else t_power(shift, df),
        "power_holm": holm_power(shift, FAMILY_SIZE[family], df),
        "tasks_for_power_0.8": needed,
    }


def t_power(shift: float, df: int) -> float:
    """Power of one two-sided t test at alpha 0.05 on ``df`` degrees of freedom."""
    from scipy import stats

    crit = float(stats.t.ppf(0.975, df))
    return float(stats.nct.sf(crit, df, shift) + stats.nct.cdf(-crit, df, shift))


def holm_power(shift: float, members: int, df: int | None = None) -> float:
    """Power of the Holm reading for one comparison of a family.

    Every member of the family has the comparison's standardized effect ``shift``; the
    smaller of the powers with independent test statistics and with correlation 0.5
    (a positive correlation lowers it here) is returned, from fixed-seed simulation. With
    ``df``, each member's statistic is a t statistic on ``df`` degrees of freedom (its
    own chi-square draw for the estimated standard error), as in a cluster_t test;
    otherwise a normal one.
    """
    from scipy import stats

    if shift == 0.0:
        return 0.05
    out = []
    for rho in (0.0, 0.5):
        rng = np.random.default_rng(SEED)
        common = rng.standard_normal((HOLM_DRAWS, 1))
        own = rng.standard_normal((HOLM_DRAWS, members))
        z = shift + math.sqrt(rho) * common + math.sqrt(1 - rho) * own
        if df is None:
            p = 2 * stats.norm.sf(np.abs(z))
        else:
            scale = np.sqrt(rng.chisquare(df, (HOLM_DRAWS, members)) / df)
            p = 2 * stats.t.sf(np.abs(z / scale), df)
        order = np.argsort(p, axis=1)
        steps = np.take_along_axis(p, order, axis=1) <= 0.05 / (members - np.arange(members))
        rejected = np.cumprod(steps, axis=1).astype(bool)
        first = np.zeros_like(rejected)
        np.put_along_axis(first, order, rejected, axis=1)
        out.append(float(first[:, 0].mean()))
    return min(out)


def local_rows(planted: Planted, tasks: int) -> list[dict[str, Any]]:
    """Power rows of the local families at ``tasks`` tasks (harness, pairs, differences)."""
    others = sum(planted.factors[f] for f in planted.factors if f != "harness") / 2
    base = planted.intercept + others + sum(planted.day) / len(planted.day)
    base += sum(planted.block) / len(planted.block)
    sd_cell = math.sqrt(planted.task_var + planted.model_task_var)
    rows = []
    for i, effect in enumerate(planted.model):
        diff, design = within_design(base + effect, planted.factors["harness"], sd_cell)
        label = f"harness contrast, model position {i + 1}"
        rows.append(
            _row("harness-local", label, planted.factors["harness"], diff, design, tasks, 8)
        )
    for i, a in enumerate(planted.model):
        for j in range(i + 1, len(planted.model)):
            b = planted.model[j]
            design = design_from_logit(base + a, base + b, planted.task_var, planted.model_task_var)
            gap = within_design(base + b, a - b, sd_cell)[0]
            label = f"model positions {i + 1} and {j + 1}"
            rows.append(_row("models-local", label, round(a - b, 4), gap, design, tasks, 8))
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
    label = "harness contrasts of two models differing by 0.5 on the logit scale"
    rows.append(_row("harness-diff-local", label, 0.5, diff_hi - diff_lo, both, tasks, 8))
    return rows


def planted_for_power() -> Planted:
    """The planted values of the power analysis (those of the recovery-config block)."""
    return Planted(
        model=(0.6, 0.0, -0.6), day=DAY_EFFECTS, task_var=TASK_VAR, model_task_var=MODEL_TASK_VAR
    )


def smallest_tasks_per_family() -> int:
    """The smallest tasks per family at which every row of the sizing families has Holm
    power of at least 0.8 (the D-033 sample-size rule)."""
    planted = planted_for_power()
    for per in range(1, 41):
        rows = local_rows(planted, per * len(FAMILY_NAMES))
        if all(r["power_holm"] >= POWER_TARGET for r in rows if r["family"] in SIZING_FAMILIES):
            return per
    raise ValueError("no task count up to 40 per family reaches the target")


def e1_power(rules: dict[str, Any]) -> dict[str, Any]:
    """The ``e1-power`` block: the preregistered power analysis (writing-14, stats-12).

    Families: the power of one comparison and of the Holm reading for every confirmatory
    contrast family at stated, assumed effects and the planted variances, with each
    family's test statistic (t on tasks - 1 degrees of freedom for the cluster_t families,
    normal for the Wilson families, :func:`evalstats.power_at`), the tasks one comparison
    needs for power 0.8, and a frozen verdict per family (underpowered when its smallest
    ``power_holm`` is below 0.8). Fixed factors: power to read a
    factor as non-zero comes from G6 at the E1 shape (prereg 0006, before any E1 data);
    the exploratory pilot 7 shares at 30 tasks are planning information.
    """
    summary = json.loads(SUMMARY.read_text())

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    pilots = REPO / "artifacts" / "method_pilots"
    planted = planted_for_power()
    pilot_shares = {}
    for name in FIXED_TERMS:
        pilot_shares[name] = {
            pool: summary["pools"][f"pilot7{pool}-jackknife"]["components"][name]["above_zero"][
                "jackknife"
            ]
            for pool in ("a", "b")
        }
    rows = local_rows(planted, N_LOCAL_TASKS)
    claude_sd = math.sqrt(CLAUDE_DGP["task_var"] + CLAUDE_DGP["model_task_var"])
    claude = N_CLAUDE_TASKS
    for c in (0.0, 2.5, 3.5):
        diff, design = within_design(c, 0.4, claude_sd)
        label = f"Claude Code minus minimal-agent, tier intercept {c}"
        rows.append(_row("harness-claude", label, 0.4, diff, design, claude, 1))
        pair = design_from_logit(c + 0.4, c, CLAUDE_DGP["task_var"], CLAUDE_DGP["model_task_var"])
        gap = within_design(c, 0.4, claude_sd)[0]
        label = f"two tiers 0.4 apart on the logit scale, intercept {c}"
        rows.append(_row("tiers-claude", label, 0.4, gap, pair, claude, 1))
    # Claude-local comparisons, as the P-cl-w cells: 1 Claude run against 4 local
    # minimal-agent runs at the Claude step budget, the local model at intercept 0.
    for effect in (0.4, 2.5, 3.5, 4.5):
        design = unequal_design(
            effect, 0.0, CLAUDE_DGP["task_var"], CLAUDE_DGP["model_task_var"], 1, 4
        )
        gap = within_design(0.0, effect, claude_sd)[0]
        label = f"a tier {effect} above the local model on the logit scale"
        rows.append(_row("models-claude", label, effect, gap, design, claude, 1))
    verdicts = {}
    for family in dict.fromkeys(r["family"] for r in rows):
        least = min(r["power_holm"] for r in rows if r["family"] == family)
        verdicts[family] = {"min_power_holm": least, "underpowered": least < POWER_TARGET}
    smallest = smallest_tasks_per_family()
    if smallest != LOCAL_TASKS_PER_FAMILY:
        raise ValueError(f"the sizing rule gives {smallest} tasks per family, not the design's")
    return {
        "label": (
            "design calculations at assumed effects and planted variances (task variance "
            f"{TASK_VAR}, model:task {MODEL_TASK_VAR}; Claude cells task "
            f"{CLAUDE_DGP['task_var']}, model:task {CLAUDE_DGP['model_task_var']}), not "
            "estimates; two-sided alpha 0.05, with t statistics on tasks - 1 df for the "
            "cluster_t families and normal statistics for the Wilson families (test)"
        ),
        "power_target": POWER_TARGET,
        "power_columns": (
            "power: one comparison at two-sided alpha 0.05 (an upper bound for the Holm "
            "reading); power_holm: the Holm reading within the family when every member has "
            "this effect, the smaller over test correlations 0 and 0.5 (fixed-seed simulation "
            "of the row's test statistics); tasks_for_power_0.8: the tasks one comparison "
            "needs for power 0.8 at two-sided alpha 0.05 (normal approximation; the Holm "
            "reading needs more)"
        ),
        "sizing": {
            "rule": (
                "the smallest tasks per family at which every harness-local and models-local "
                "row has power_holm of at least 0.8, with the t statistics of their cluster_t "
                "tests (D-033, D-034)"
            ),
            "families": list(SIZING_FAMILIES),
            "tasks_per_family": LOCAL_TASKS_PER_FAMILY,
        },
        "families": rows,
        "family_verdicts": verdicts,
        "fixed_factor_power": {
            "rule": (
                "a fixed factor is read as non-zero when its task-jackknife box's lower end "
                "is above 0 (D-028 item 1); its power is the G6 above_zero share at the "
                "planted values (prereg 0006), banked before any E1 data, and a factor "
                "below 0.8 there is reported as underpowered. Concurrency is planted at 0, "
                "so its G6 share is the false-positive rate of that reading."
            ),
            "planted_logit_effects": dict(planted.factors),
            "pilot7": {
                "label": (
                    "synthetic simulation estimates (exploratory pilot 7: 3 models, 30 "
                    "tasks, 60 datasets per pool), planning information only"
                ),
                "source": "artifacts/method_pilots/summary.json pools pilot7a-jackknife (no "
                "task-by-factor variation) and pilot7b-jackknife (variances 0.25), "
                "components.<factor>.above_zero.jackknife",
                "summary_sha256": digest(SUMMARY),
                "pilot7a_sha256": digest(pilots / "pilot7a.jsonl"),
                "pilot7b_sha256": digest(pilots / "pilot7b.jsonl"),
                "n_datasets": {
                    pool: summary["pools"][f"pilot7{pool}-jackknife"]["datasets"]
                    for pool in ("a", "b")
                },
                "above_zero_share": {k: v for k, v in pilot_shares.items() if k != "concurrency"},
                "false_positive_rate": {"concurrency": pilot_shares["concurrency"]},
            },
        },
    }


def _span(values: list[float]) -> list[float]:
    return [min(values), max(values)]


def validated_range(config: dict[str, Any], grid: list[dict[str, Any]]) -> dict[str, Any]:
    """The planted values each headline kind's cells span (prereg 0004 and 0005).

    An E1 interval whose point estimate lies outside its kind's span is reported as
    outside the validated range (``noisefloor.analysis.e1`` applies the check). A
    difference is compared in absolute value (each paired interval mirrors when its two
    sides swap), the sampling table by its within-configuration variance, and an E3 bound
    by the model's ambiguous share of runs.
    """
    from noisefloor.e0 import _true_discordance, _true_score, _true_within, expected_sigmoid
    from noisefloor.recovery import (
        AnalysisSettings,
        config_from,
        local_intervals,
        simulate,
    )

    design, base, _ = config_from(config)
    data = simulate(design, base, 1)
    local: dict[str, dict[str, list[float]]] = {}
    truths_by_group: dict[str, dict[str, Any]] = {}
    for group, overrides in LOCAL_GROUPS:
        changes = dict(overrides)
        if "model_factors" in changes:
            changes["model_factors"] = {
                k: tuple(float(x) for x in v) for k, v in changes["model_factors"].items()
            }
        if "model" in changes:
            changes["model"] = tuple(float(x) for x in changes["model"])
        planted = dataclasses.replace(base, **changes)
        group_data = simulate(design, planted, 1)
        truths = local_intervals(group_data, planted, AnalysisSettings())
        truths_by_group[group] = truths
        for name, entry in sorted(truths.items()):
            local.setdefault(f"local-{name.split(':')[0]}", {}).setdefault(group, []).append(
                float(entry["planted"])
            )
    del data
    kinds: dict[str, Any] = {}
    for kind, groups in sorted(local.items()):
        values = [v for vs in groups.values() for v in vs]
        kinds[kind] = {"by_group": groups, "values": values}
    by_kind: dict[str, list[float]] = {}
    for cell in grid:
        p = cell["params"]
        if cell["cell"].startswith("S-c-w"):
            by_kind.setdefault("claude-score", []).append(_true_score(0.0, p))
        elif cell["cell"].startswith("P-c-w"):
            by_kind.setdefault("claude-tiers", []).append(
                _true_score(p["effect_a"], p) - _true_score(p["effect_b"], p)
            )
        elif cell["cell"].startswith("P-cw-w"):
            by_kind.setdefault("claude-harness", []).append(
                _true_within(p["effect_a"], p["effect_b"], p)
            )
        elif cell["cell"].startswith("P-cl-w"):
            by_kind.setdefault("claude-local", []).append(
                _true_score(p["effect_a"], p) - _true_score(p["effect_b"], p)
            )
        elif cell["cell"].startswith("S-r-w"):
            by_kind.setdefault("sampling", []).append(_true_discordance(p))
        elif cell["cell"].startswith("B-e1-") and "any" not in cell["cell"]:
            success = expected_sigmoid(float(p["intercept"]), float(p["task_sd"]))
            side = 1 - success if p["ambiguous_among"] == "fail" else success
            by_kind.setdefault("bounds", []).append(float(p["ambiguous_rate"]) * side)
    for kind, values in sorted(by_kind.items()):
        kinds[kind] = {"values": values}
    for kind, (group, name) in NULL_CELLS.items():
        truth = float(truths_by_group[group][name]["planted"])
        if abs(truth) > 0.005:  # a null cell's truth is 0 up to the batch dealing of one seed
            raise ValueError(f"{group} {name} is not a null difference ({truth})")
        kinds[kind]["null_cells"] = [f"{group} {name}"]
    for kind, entry in kinds.items():
        if kind in DIFFERENCE_KINDS:
            entry["compare"] = (
                "absolute value of the estimated difference; the span starts at 0 when a "
                "cell of the kind plants no difference (null_cells, or an effect of 0)"
            )
            span = _span([abs(v) for v in entry["values"]])
            entry["span"] = [0.0, span[1]] if entry.get("null_cells") else span
        elif kind == "sampling":
            entry["compare"] = (
                "the estimated within-configuration variance (half the discordant share)"
            )
            entry["span"] = _span(entry["values"])
        elif kind == "bounds":
            entry["compare"] = "the model's ambiguous share of runs"
            entry["span"] = _span(entry["values"])
        else:
            entry["compare"] = "the estimated success rate"
            entry["span"] = _span(entry["values"])
    return {
        "label": "planted values of the headline cells, design values (not results)",
        "compare": "the point estimate of each E1 interval against its kind's span",
        "kinds": kinds,
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
        "validated-range": validated_range(recovery_config(rules), grid),
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
