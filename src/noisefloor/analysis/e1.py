"""E1 analysis (gate G7): runs table, variance decomposition, and MDE table.

``python -m noisefloor.analysis.e1 build PREREG`` reads the banked run records of the local
arm (``artifacts/e1/local.jsonl``) and of the Claude slice (``artifacts/e1/claude.jsonl``,
when present) and writes under ``artifacts/e1/``:

* ``runs.jsonl``: one row per task run (realized UTC day and time-of-day block from the
  run's start, factor levels, outcome, status, and whether it is confirmatory: in the
  preregistered plan and run under that plan's phase);
* ``variance_components.json``: the logit GLMM decomposition of the confirmatory local-arm
  runs, settings from the ``e1-analysis`` block of the preregistration;
* ``mde.json``: minimum detectable effects of paired comparisons for leaderboard designs,
  and the tasks needed for given gaps, derived from the fitted model;
* ``scores.json`` and ``contrasts.json``: each model's success rate and the paired
  contrasts, with the preregistered intervals and Holm-adjusted p-values;
* ``sampling.json``: the within-configuration outcome variance from the replicate runs,
  beside the variance the MDE assumes at its reference configuration;
* ``sensitivity.json``: the preregistered sensitivity analyses, reported beside these;
* ``e3_bounds.json``: E3, each local model's score under every policy for ambiguous runs,
  the ranking under each policy, and bounds on each adjacent pair;
* ``manifest.json``: the preregistration and the SHA-256 of every input and output.

``python -m noisefloor.analysis.e1 --verify`` recomputes every output from the inputs and
fails on any difference. Results are model results on procedurally generated mock tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evalstats import (
    EvalResults,
    contrast_difference,
    paired_difference,
    policy_bounds,
    score_interval,
    tasks_needed,
    validity,
)
from evalstats.glmm import (
    LOGISTIC_RESIDUAL_VAR,
    Decomposition,
    FitFailedError,
    boot_coef_interval,
    boot_variance_interval,
    decompose,
    enough,
)
from noisefloor import jsonl
from noisefloor.analysis import mde as mde_lib
from noisefloor.e1 import (
    ARM_DAYS,
    REPLICATE_PHASE,
    E1Design,
    check_frozen_plan,
    claude_plan,
    claude_tasks,
    local_plan,
    served_models,
)
from noisefloor.models import MODELS, is_claude
from noisefloor.preregs import amends, chain_entry, prereg_block, require_frozen
from noisefloor.suite import SUITE_VERSION

REPO = Path(__file__).resolve().parents[3]
E1_DIR = REPO / "artifacts" / "e1"
INPUTS = ("local.jsonl", "claude.jsonl")
OUTPUTS = (
    "runs.jsonl",
    "variance_components.json",
    "mde.json",
    "scores.json",
    "contrasts.json",
    "sampling.json",
    "sensitivity.json",
    "e3_bounds.json",
)
STOP_LOGS = ("stops-local.jsonl", "stops-claude.jsonl")  # hashed into the manifest too
MOCK_TASKS = (
    f"every task is a procedurally generated mock task (nf-tools suite version {SUITE_VERSION}); "
    "results are model results on these mock tasks, not on real-world work"
)
GAPS = (0.02, 0.05, 0.10)
CLAUDE_LOCAL_CONFOUNDS = (  # prereg 0004 item 5: how a Claude-local pair differs besides the model
    "the Claude side runs at the CLI's default temperature",
    "no reply cap reaches the CLI (the reply cap applies only to the local server)",
    "each Claude step is sent as one role-tagged prompt",
    "the Claude timeout alternates between 150 s and 300 s by task",
)


def _rel(path: Path) -> str:
    absolute = path.resolve()
    return absolute.relative_to(REPO).as_posix() if absolute.is_relative_to(REPO) else str(path)


def read_records(directory: Path) -> list[dict[str, Any]]:
    """Banked run records of both arms (a missing file contributes nothing)."""
    rows: list[dict[str, Any]] = []
    for name in INPUTS:
        path = directory / name
        if path.exists():
            rows += [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    return rows


def claude_exclusions(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Banked Claude runs the analyses leave out, by run id, with the reason.

    ``model_changed``: the CLI reported a model ID other than the one that served the
    tier's first banked run (runs in order of start). ``contaminated``: a Claude Code
    child that did not report a clean start (plan login, only the allowed tools, no MCP
    servers, no memory files). ``step_budget``: the run's turn cap (``max_steps_used`` or
    ``claude.max_turns``) differs from its ``steps`` factor.
    """
    out: dict[str, str] = {}
    first: dict[str, str] = {}
    ordered = sorted(records, key=lambda r: (str(r.get("started_at") or ""), str(r["run_id"])))
    for rec in ordered:
        tier = str(rec["model"])
        if not is_claude(tier):
            continue
        ids = served_models(rec)
        if ids:
            first.setdefault(tier, ids[0])
        if any(model_id != first[tier] for model_id in ids):
            out[str(rec["run_id"])] = "model_changed"
            continue
        if (rec.get("claude") or {}).get("clean") is False:
            out[str(rec["run_id"])] = "contaminated"
            continue
        used = rec.get("max_steps_used") or (rec.get("claude") or {}).get("max_turns")
        planned = str((rec.get("factors") or {}).get("steps", ""))
        if used is not None and planned.isdigit() and int(used) != int(planned):
            out[str(rec["run_id"])] = "step_budget"
    return out


def arms_not_ended(
    records: Sequence[Mapping[str, Any]],
    planned: Mapping[str, str],
    directory: Path,
    frozen_at: datetime | None,
    now: datetime,
) -> list[str]:
    """Why an arm has not ended (empty once both have), so no outcome is read early.

    The local arm ends when every planned local run is banked, its day-7 stop is recorded,
    or the clock has passed day :data:`noisefloor.e1.ARM_DAYS` of the arm (it starts no
    batch after that day). The Claude slice ends only when every planned Claude run is
    banked or its own runner recorded an end point, on the machine that runs it (the
    analysis machine's clock does not end it; ``frozen_at`` is kept for the caller).
    """
    del frozen_at  # the Claude slice's end is its runner's end_point line (prereg 0004)
    banked = {str(r["run_id"]) for r in records}
    out = []
    local = {rid for rid, phase in planned.items() if phase in ("e1", REPLICATE_PHASE)}
    local_stops = jsonl.read(directory / "stops-local.jsonl")
    starts = [
        _utc(str(r["started_at"]))
        for r in records
        if r.get("started_at") and not is_claude(str(r["model"]))
    ]
    past = bool(starts) and (now.date() - min(starts).date()).days + 1 > ARM_DAYS
    if not local <= banked and not past and not any(s["kind"] == "day_limit" for s in local_stops):
        out.append(f"the local arm has not ended ({len(local & banked)} of {len(local)} banked)")
    claude = {rid for rid, phase in planned.items() if phase == "e1-claude"}
    claude_stops = jsonl.read(directory / "stops-claude.jsonl")
    over = any(s["kind"] == "end_point" for s in claude_stops)
    if not claude <= banked and not over:
        out.append(
            f"the Claude slice has not ended ({len(claude & banked)} of {len(claude)} banked)"
        )
    return out


def time_block(started_at: str, block_hours: Sequence[Sequence[int]]) -> tuple[str, str]:
    """(UTC date, block label ``b1``, ``b2``, ...) of a run's start time."""
    day, _, clock = started_at.partition("T")
    hour = int(clock[:2]) if clock[:2].isdigit() else 0
    for index, (start, end) in enumerate(block_hours):
        if start <= hour < end:
            return day, f"b{index + 1}"
    raise ValueError(f"hour {hour} is in no time-of-day block")


def runs_table(
    records: Sequence[Mapping[str, Any]],
    *,
    prereg: str,
    planned: Mapping[str, str],
    block_hours: Sequence[Sequence[int]],
    frozen_at: str | None = None,
    prereg_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """One row per run; ``planned`` maps each preregistered run id to its phase.

    A run is confirmatory when its id and phase are planned, its record cites ``prereg``
    (and, when ``prereg_sha256`` is given, the preregistration's chain SHA-256), and it
    started no earlier than ``frozen_at`` (the preregistration's freeze time; not checked
    when None). A record without ``started_at`` takes its batch's start, flagged
    ``start_from_batch`` (the runner records a start for every run it starts and banks no
    run it did not start, so this marks a malformed record). Day and block are those of
    the batch's start (``batch_started_at``), so every run of a batch shares them, as
    planned. A run id banked more than once raises ``ValueError``.
    """
    counts: dict[str, int] = {}
    for rec in records:
        counts[str(rec["run_id"])] = counts.get(str(rec["run_id"]), 0) + 1
    dups = sorted(k for k, n in counts.items() if n > 1)
    if dups:
        raise ValueError(f"{len(dups)} run ids banked more than once, first {dups[0]}")
    left_out = claude_exclusions(records)
    rows = []
    for rec in sorted(records, key=lambda r: str(r["run_id"])):
        started = str(rec.get("started_at") or rec.get("batch_started_at") or "")
        batch_start = str(rec.get("batch_started_at") or started)
        day, block = time_block(batch_start, block_hours)
        model = str(rec["model"])
        cites = rec.get("prereg") == prereg and (
            prereg_sha256 is None or rec.get("prereg_sha256") == prereg_sha256
        )
        in_time = frozen_at is None or _utc(started) >= _utc(frozen_at)
        planned_run = planned.get(str(rec["run_id"])) == rec.get("phase")
        rows.append(
            {
                "run_id": rec["run_id"],
                "prereg": rec.get("prereg"),
                "prereg_sha256": rec.get("prereg_sha256"),
                "started_at": started,
                "batch_started_at": batch_start,
                "start_from_batch": not rec.get("started_at"),
                "model_id": model,
                "backend": "claude" if is_claude(model) else "local",
                "family": "claude" if is_claude(model) else MODELS[model].family,
                "task": f"{rec['family']}-d{rec['difficulty']}-s{rec['task_seed']}",
                "harness": rec["harness"],
                "factors": dict(rec.get("factors", {})),
                "day": day,
                "block": block,
                "planned_slot": str(rec.get("block", "")).split("/")[0],
                "planned_batch": str(rec.get("block") or "") or None,
                "batch_id": rec.get("batch_id"),
                "outcome": int(float(rec.get("outcome", 0.0)) >= 0.5),
                "outcome_strict": rec.get("outcome_strict"),
                "status": rec.get("status"),
                "end_reason": rec.get("end_reason"),
                "confirmatory": bool(planned_run and cites and in_time),
                "excluded": left_out.get(str(rec["run_id"])),
                "replicate": rec.get("phase") == REPLICATE_PHASE,
            }
        )
    return rows


def _utc(stamp: str) -> datetime:
    """A timestamp as an aware UTC datetime (``Z`` and offsets accepted)."""
    value = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def analysis_frame(
    rows: Sequence[Mapping[str, Any]],
    exclude: Sequence[str],
    *,
    outcome: str = "outcome",
    failures: Sequence[str] = (),
) -> pd.DataFrame:
    """Confirmatory local-arm runs for the decomposition (excluded statuses dropped).

    ``outcome`` names the graded column (``outcome`` or ``outcome_strict``); runs whose
    status is in ``failures`` are kept with outcome 0.
    """
    keep = [r for r in _usable(rows, exclude) if r["backend"] == "local"]
    if not keep:
        raise ValueError("no confirmatory local-arm runs to analyze")
    days = {d: f"d{i + 1}" for i, d in enumerate(sorted({r["day"] for r in keep}))}

    def graded(r: Mapping[str, Any]) -> float:
        if r["status"] in failures:
            return 0.0
        return float(float(r[outcome] or 0.0) >= 0.5)

    return pd.DataFrame(
        [
            {
                "model": r["model_id"],
                "task": r["task"],
                "run": r["run_id"],
                "outcome": graded(r),
                **{name: str(value) for name, value in r["factors"].items()},
                "day": days[r["day"]],
                "block": r["block"],
                # The planned batch (prereg 0004 item 9): a planned batch finished after a
                # stop is one batch here, though its pieces ran as two executed batches.
                "batch": str(r.get("planned_batch") or r["batch_id"]),
            }
            for r in keep
        ]
    )


def decomposition(data: pd.DataFrame, settings: Mapping[str, Any]) -> Decomposition:
    """The preregistered decomposition (a fixed factor held constant is left out).

    The same call as :func:`noisefloor.recovery.analyze`, so G6 and E0 part 2 validate
    exactly this procedure: ``fixed_interval`` ``jackknife`` gives the fixed contributions
    task-jackknife boxes (D-028 item 1).
    """
    fixed = [f for f in settings["fixed"] if data[f].nunique() > 1]
    return decompose(
        EvalResults(data),
        random=tuple(settings["random"]),
        fixed=tuple(fixed),
        level=float(settings["level"]),
        backend="lme4",
        method="parametric_bootstrap",
        n_boot=int(settings["n_boot"]),
        nagq=int(settings["nagq"]),
        boot_interval=str(settings["boot_interval"]),
        box_interval=str(settings["box_interval"]),
        share_interval=str(settings.get("share_interval", "percentile")),
        fixed_interval=str(settings.get("fixed_interval", "box")),
        jackknife_group="task",
        seed=int(settings["seed"]),
    )


def components_json(decomp: Decomposition) -> list[dict[str, Any]]:
    """Component rows; the residual is the fixed logistic variance (its own interval)."""
    out = []
    for c in decomp.components:
        residual = c.kind == "residual"
        out.append(
            {
                "name": c.name,
                "kind": c.kind,
                "estimate": c.variance,
                "lo": LOGISTIC_RESIDUAL_VAR if residual else c.lo,
                "hi": LOGISTIC_RESIDUAL_VAR if residual else c.hi,
                "share": c.share,
                "share_lo": c.share_lo,
                "share_hi": c.share_hi,
                "interval_method": c.interval_method,
            }
        )
    return out


def mde_tables(
    decomp: Decomposition, data: pd.DataFrame, settings: Mapping[str, Any]
) -> dict[str, Any]:
    """MDE rows for each design and tasks needed for each gap, with intervals."""
    cfg = settings["mde"]
    designs = [(int(t), int(k)) for t, k in cfg["designs"]]
    models = sorted(data["model"].unique())
    if "task" in data.columns:  # the E0 MDE cells (prereg 0005, method glmm_mde)
        validity.check("replicate_mdes", "parametric_bootstrap", int(data["task"].nunique()))
    variances = {c.name: c.variance for c in decomp.components if c.kind == "random"}
    etas = mde_lib.reference_eta(
        decomp.fixed_effects, data, models, cfg["reference"], cfg["averaged"]
    )
    pooled = mde_lib.pooled_design(etas, variances["task"], variances["model:task"])
    alpha, power = float(cfg.get("alpha", 0.05)), float(cfg.get("power", 0.8))
    point = mde_lib.mde_values(pooled, designs, alpha=alpha, power=power)
    reps = mde_lib.replicate_mdes(
        decomp.coefficient_replicates,
        {k: decomp.replicates[k] for k in ("task", "model:task")},
        data,
        models=models,
        reference=cfg["reference"],
        averaged=cfg["averaged"],
        designs=designs,
        alpha=alpha,
        power=power,
    )
    level = float(settings["level"])
    rows = []
    first: dict[int, tuple[int, float | None, float | None]] = {}
    for t, k in designs:
        # More than MAX_FAILED_SHARE failed replicates: no interval, as for the components.
        ci = (
            boot_coef_interval(point[(t, k)], reps[(t, k)], level, str(cfg["interval"]))
            if enough(np.asarray(reps[(t, k)], dtype=float))
            else None
        )
        lo = None if ci is None else max(0.0, ci[0])
        hi = None if ci is None else ci[1]
        first.setdefault(k, (t, lo, hi))
        rows.append(
            {
                "tasks": t,
                "runs": k,
                "mde": point[(t, k)],
                "lo": lo,
                "hi": hi,
                "interval_method": f"parametric_bootstrap_{cfg['interval']}",
            }
        )
    # Tasks needed for a gap is ceil(T * MDE(T, k)^2 / gap^2) for any design row with k
    # runs, a monotone map of that row, so the row's interval maps to the count's interval.
    needed = []
    for gap in [float(g) for g in cfg.get("gaps", GAPS)]:
        for k, (t, lo, hi) in sorted(first.items()):
            needed.append(
                {
                    "gap": gap,
                    "runs": k,
                    "tasks": tasks_needed(gap, pooled, k, alpha=alpha, power=power),
                    "lo": _tasks_for(t, lo, gap),
                    "hi": _tasks_for(t, hi, gap),
                    "interval_method": "mapped from the MDE interval",
                }
            )
    by_model = {
        m: mde_lib.bernoulli_variance(eta, variances["task"], variances["model:task"])
        for m, eta in etas.items()
    }
    return {
        "reference": dict(cfg["reference"]),
        "averaged": list(cfg["averaged"]),
        "between_task_var": pooled.between_task_var,
        "within_task_var": pooled.within_task_var,
        "within_var_by_model": by_model,
        "rows": rows,
        "tasks_needed": needed,
    }


def _tasks_for(tasks: int, mde_value: float | None, gap: float) -> int | None:
    """Tasks whose MDE equals ``gap``, from the MDE of a design with ``tasks`` tasks."""
    if mde_value is None:
        return None
    return max(1, math.ceil(tasks * mde_value * mde_value / (gap * gap) - 1e-9))


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values (step-down, monotone), in the input order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, i in enumerate(order):
        value = p_values[i]
        candidate = min(1.0, (len(p_values) - rank) * value) if math.isfinite(value) else 1.0
        running = max(running, candidate)
        adjusted[i] = running
    return adjusted


def _usable(
    rows: Sequence[Mapping[str, Any]], exclude: Sequence[str], *, replicates: bool = False
) -> list[Mapping[str, Any]]:
    """Confirmatory runs an analysis uses: no excluded status and not left out.

    Replicate runs serve only the sampling table (``replicates=True`` keeps them).
    """
    return [
        r
        for r in rows
        if r["confirmatory"]
        and r["status"] not in exclude
        and not r.get("excluded")
        and (replicates or not r.get("replicate"))
    ]


def sampling_table(
    rows: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
    reference: Mapping[str, str] | None = None,
    assumed: float | None = None,
    assumed_by_model: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Run-to-run sampling, from the replicate runs, on the probability scale (D-028 item 3).

    Each replicate pairs with the planned run of the same model, task, and configuration
    (run id without ``-rep``), when both are usable. Per configuration, the mean over
    tasks of the per-task mean of ``(y1 - y2)**2 / 2`` (half the task-clustered estimate of
    the share of discordant pairs) estimates the within-configuration outcome variance,
    which equals ``E[p (1 - p)]`` when repeated runs are independent; its interval is the
    task-clustered Wilson interval (``settings["sampling_interval"]``, default
    ``cluster_wilson``) of the share of discordant pairs, halved. The row of the
    ``reference`` configuration (the MDE's) also gives ``assumed``, the mean over models of
    ``E[p (1 - p)]`` that the MDE uses there (half its pooled within-task variance), and
    each model's value (``assumed_by_model``). The frozen reading (prereg 0004 item 6): when
    the mean lies above the upper end of the reference row's interval, repeated runs at
    the reference are not independent draws (``independence``: ``violated``), and the
    MDE rows with more than one run per task overstate what extra runs buy. The reading
    treats the implied mean as known (its estimation error is ignored), so it has no
    stated false-positive rate.
    """
    usable = {
        str(r["run_id"]): r for r in _usable(rows, settings["exclude_status"], replicates=True)
    }
    pairs: dict[tuple[tuple[str, str], ...], list[tuple[Mapping[str, Any], int]]] = {}
    for run_id, rep in usable.items():
        original = usable.get(run_id.removesuffix("-rep"))
        if not rep.get("replicate") or original is None or original is rep:
            continue
        config = tuple(sorted((str(k), str(v)) for k, v in rep["factors"].items()))
        pairs.setdefault(config, []).append((rep, int(rep["outcome"] != original["outcome"])))
    method = str(settings.get("sampling_interval", "cluster_wilson"))
    level = float(settings["level"])
    out = []
    for config, members in sorted(pairs.items()):
        label = ",".join(f"{k}={v}" for k, v in config)
        frame = pd.DataFrame(
            [
                {"model": label, "task": r["task"], "run": r["model_id"], "outcome": float(d)}
                for r, d in members
            ]
        )
        si = score_interval(EvalResults(frame), label, method=method, level=level)
        factors = dict(config)
        is_reference = reference is not None and all(
            factors.get(k) == str(v) for k, v in reference.items()
        )
        hi = None if si.interval.hi is None else si.interval.hi / 2
        reading = None
        if is_reference and assumed is not None and hi is not None:
            reading = "violated" if assumed > hi else "not rejected"
        out.append(
            {
                "configuration": factors,
                "pairs": len(members),
                "discordant": sum(d for _, d in members),
                "within_var": si.interval.estimate / 2,
                "lo": None if si.interval.lo is None else si.interval.lo / 2,
                "hi": hi,
                "interval_method": f"half of {si.interval.method}",
                "mde_reference": is_reference,
                "mde_assumed_within_var": assumed if is_reference else None,
                "mde_assumed_within_var_by_model": (
                    dict(assumed_by_model) if is_reference and assumed_by_model else None
                ),
                "independence": reading,
            }
        )
    return out


def mark_independence(mde: dict[str, Any], sampling: Sequence[Mapping[str, Any]]) -> None:
    """Label the MDE rows with more than one run per task by the independence reading.

    The k-run rows (and their tasks-needed rows) rest on independent repeated runs; when
    the sampling table's reference row reads ``violated``, they are labeled as resting on
    a violated assumption (prereg 0004 item 6); otherwise as consistent with the table.
    """
    reading = next((r["independence"] for r in sampling if r.get("mde_reference")), None)
    label = {
        "violated": "rests on a violated independence assumption (sampling table)",
        "not rejected": "independence not rejected by the sampling table",
        None: "independence not checked (no reference row in the sampling table)",
    }[reading]
    for row in [*mde.get("rows", []), *mde.get("tasks_needed", [])]:
        if int(row["runs"]) > 1:
            row["independence"] = label


def schedule_check(rows: Sequence[Mapping[str, Any]], days: int = 3) -> dict[str, Any]:
    """Whether the local arm kept the simulated schedule: ``days`` days, batches on their slots.

    G6 and E0 part 2 simulate exactly the planned days x 2 blocks with each batch on its
    planned slot. A batch runs off its slot when its realized day (the rank of its start
    date among the arm's dates) or block differs from the planned ``day<k>-b<n>``. If the
    arm used another number of days or any batch ran off its slot, the day and block
    intervals rest on a design the validation did not simulate (prereg 0004).
    """
    local = [r for r in rows if r["confirmatory"] and r["backend"] == "local"]
    dates = sorted({str(r["day"]) for r in local})
    rank = {d: i + 1 for i, d in enumerate(dates)}
    batches: dict[str, tuple[str, str, str]] = {}
    for r in local:
        batches[str(r["batch_id"])] = (str(r["planned_slot"]), str(r["day"]), str(r["block"]))
    off = sorted(
        b for b, (slot, day, block) in batches.items() if slot != f"day{rank[day]}-{block}"
    )
    return {
        "days": len(dates),
        "batches": len(batches),
        "batches_off_plan": len(off),
        "simulated_design": len(dates) == days and not off,
    }


def exclusion_counts(
    rows: Sequence[Mapping[str, Any]], exclude: Sequence[str]
) -> dict[str, dict[str, int]]:
    """Confirmatory runs left out of the primary analyses, per arm and reason."""
    out: dict[str, dict[str, int]] = {"local": {}, "claude": {}}
    for r in rows:
        if not r["confirmatory"]:
            continue
        reason = r.get("excluded") or (r["status"] if r["status"] in exclude else None)
        if reason:
            arm = out[str(r["backend"])]
            arm[str(reason)] = arm.get(str(reason), 0) + 1
    return out


def _results(rows: Sequence[Mapping[str, Any]], key: str) -> EvalResults:
    """Runs as an EvalResults table with ``model`` from ``key``.

    ``key`` names a row field, or ``factor:<name>`` for that factor's level.
    """
    factor = key.removeprefix("factor:") if key.startswith("factor:") else None
    frame = pd.DataFrame(
        [
            {
                "model": str(r["factors"][factor] if factor else r[key]),
                "task": r["task"],
                "run": r["run_id"],
                "outcome": float(r["outcome"]),
            }
            for r in rows
        ]
    )
    return EvalResults(frame)


def interval_method(settings: Mapping[str, Any], name: str, key: str, default: str) -> str:
    """The preregistered interval method for one arm or Holm family.

    ``settings[name]`` is one method for all, or a mapping from arm (``local``, ``claude``)
    or Holm family to a method.
    """
    value = settings.get(name, default)
    return str(value[key]) if isinstance(value, Mapping) else str(value)


def scores_table(
    rows: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Each model's success rate with its preregistered interval (confirmatory runs).

    The interval method is ``settings["score_interval"]`` (default ``cluster_t``), one
    method or one per arm. Each row also counts the model's runs that ended at a limit
    (status ``timeout``) and how many of them have outcome 1 (D-021: a derive run is graded on
    its final file whatever the end reason).
    """
    usable = _usable(rows, settings["exclude_status"])
    if not usable:
        return []
    results = _results(usable, "model_id")
    out = []
    for model in sorted({r["model_id"] for r in usable}):
        mine = [r for r in usable if r["model_id"] == model]
        backend = mine[0]["backend"]
        method = interval_method(settings, "score_interval", backend, "cluster_t")
        si = score_interval(results, model, method=method, level=float(settings["level"]))
        out.append(
            {
                "model": model,
                "backend": backend,
                "score": si.interval.estimate,
                "lo": si.interval.lo,
                "hi": si.interval.hi,
                "n_tasks": si.n_tasks,
                "runs_per_task": si.runs_per_task,
                "interval_method": si.interval.method,
                "timeout_runs": sum(1 for r in mine if r["status"] == "timeout"),
                "timeout_runs_with_outcome_1": sum(
                    1 for r in mine if r["status"] == "timeout" and int(r["outcome"]) == 1
                ),
            }
        )
    return out


def _paired_row(
    results: EvalResults, a: str, b: str, level: float, *, label: str, method: str, family: str
) -> dict[str, Any]:
    pc = paired_difference(results, a, b, level=level, method=method)
    return {
        "contrast": label,
        "family": family,
        "a": a,
        "b": b,
        "difference": pc.difference.estimate,
        "lo": pc.difference.lo,
        "hi": pc.difference.hi,
        "p_value": pc.p_value,
        "n_tasks": pc.n_tasks,
        "interval_method": pc.difference.method,
    }


def contrasts_table(
    rows: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Paired harness contrasts per model and model comparisons, Holm-adjusted per family.

    Harness: native minus minimal for every model, on its own confirmatory runs. Harness
    differences (D-028 item 6): for every pair of local models, the first model's harness
    contrast minus the second's, paired by task (:func:`evalstats.contrast_difference`).
    Models: every pair of local models on minimal-agent runs; every pair of Claude tiers
    under each harness; and every Claude tier against every local model on minimal-agent
    runs with the Claude step budget (``scaffold-confounded``, with every known difference
    in ``confounds``, prereg 0004 item 5). Holm families: local harness
    contrasts, Claude harness contrasts, local harness differences, local model pairs,
    Claude tier pairs, and Claude-local pairs. The interval method is
    ``settings["contrast_interval"]`` (default ``cluster_t``), one method or one per family.
    """
    level = float(settings["level"])

    def method(family: str) -> str:
        return interval_method(settings, "contrast_interval", family, "cluster_t")

    usable = _usable(rows, settings["exclude_status"])
    harness: list[dict[str, Any]] = []
    for model in sorted({r["model_id"] for r in usable}):
        mine = [r for r in usable if r["model_id"] == model]
        if {r["factors"].get("harness") for r in mine} >= {"native", "minimal"}:
            results = _results(mine, "factor:harness")
            family = f"harness-{mine[0]['backend']}"
            harness.append(
                _paired_row(
                    results,
                    "native",
                    "minimal",
                    level,
                    label=f"harness:{model}",
                    method=method(family),
                    family=family,
                )
            )
    differences = _harness_differences(usable, level, method("harness-diff-local"))
    minimal = [r for r in usable if r["factors"].get("harness") == "minimal"]
    local_models = sorted({r["model_id"] for r in minimal if r["backend"] == "local"})
    claude_models = sorted({r["model_id"] for r in minimal if r["backend"] == "claude"})
    models: list[dict[str, Any]] = []
    local_runs = _results([r for r in minimal if r["backend"] == "local"], "model_id")
    for i, a in enumerate(local_models):
        for b in local_models[i + 1 :]:
            models.append(
                _paired_row(
                    local_runs,
                    a,
                    b,
                    level,
                    label=f"model:{a}-{b}",
                    method=method("models-local"),
                    family="models-local",
                )
            )
    models += _tier_pairs(usable, level, method("tiers-claude"))
    claude_steps = {r["factors"].get("steps") for r in minimal if r["backend"] == "claude"}
    matched = [
        r for r in minimal if r["backend"] == "claude" or r["factors"].get("steps") in claude_steps
    ]
    if claude_models:
        matched_runs = _results(matched, "model_id")
        for a in claude_models:
            for b in local_models:
                row = _paired_row(
                    matched_runs,
                    a,
                    b,
                    level,
                    label=f"model:{a}-{b}",
                    method=method("models-claude"),
                    family="models-claude",
                )
                row["label"] = "scaffold-confounded"
                row["confounds"] = list(CLAUDE_LOCAL_CONFOUNDS)
                models.append(row)
    every = (*harness, *differences, *models)
    for name in sorted({r["family"] for r in every}):
        members = [r for r in every if r["family"] == name]
        for row, adj in zip(members, holm([r["p_value"] for r in members]), strict=True):
            row["p_holm"] = adj
            # The confirmatory reading: Holm-adjusted p below 0.05 within the family. The
            # 95% interval is per comparison (unadjusted) and reported beside it.
            row["supported"] = bool(adj < 0.05)
            row["interval_excludes_zero"] = bool(row["lo"] > 0 or row["hi"] < 0)
    return {"harness": harness, "harness_differences": differences, "models": models}


def _tier_pairs(
    usable: Sequence[Mapping[str, Any]], level: float, method: str
) -> list[dict[str, Any]]:
    """Every pair of Claude tiers under each harness, paired by task (D-028 item 6)."""
    out: list[dict[str, Any]] = []
    for harness_level in ("minimal", "native"):
        tier_runs = [
            r
            for r in usable
            if r["backend"] == "claude" and r["factors"].get("harness") == harness_level
        ]
        tiers = sorted({r["model_id"] for r in tier_runs})
        if len(tiers) < 2:
            continue
        tier_results = _results(tier_runs, "model_id")
        for i, a in enumerate(tiers):
            for b in tiers[i + 1 :]:
                out.append(
                    _paired_row(
                        tier_results,
                        a,
                        b,
                        level,
                        label=f"tiers:{harness_level}:{a}-{b}",
                        method=method,
                        family="tiers-claude",
                    )
                )
    return out


def _harness_differences(
    usable: Sequence[Mapping[str, Any]], level: float, method: str
) -> list[dict[str, Any]]:
    """For each pair of local models, harness contrast of the first minus the second's."""
    local = [r for r in usable if r["backend"] == "local"]
    frame = pd.DataFrame(
        [
            {
                "model": f"{r['model_id']}:{r['factors'].get('harness')}",
                "task": r["task"],
                "run": r["run_id"],
                "outcome": float(r["outcome"]),
            }
            for r in local
        ]
    )
    if frame.empty:
        return []
    results = EvalResults(frame)
    models = sorted({str(r["model_id"]) for r in local})
    out = []
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            cd = contrast_difference(
                results,
                f"{a}:native",
                f"{a}:minimal",
                f"{b}:native",
                f"{b}:minimal",
                method=method,
                level=level,
            )
            out.append(
                {
                    "contrast": f"harness-difference:{a}-{b}",
                    "family": "harness-diff-local",
                    "a": a,
                    "b": b,
                    "difference": cd.difference.estimate,
                    "lo": cd.difference.lo,
                    "hi": cd.difference.hi,
                    "p_value": cd.p_value,
                    "n_tasks": cd.n_tasks,
                    "interval_method": cd.difference.method,
                }
            )
    return out


def _dump(obj: Any) -> str:
    def clean(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [clean(v) for v in value]
        return value

    return json.dumps(clean(obj), sort_keys=True, indent=1) + "\n"


G6_ARTIFACT = REPO / "artifacts" / "g6" / "recovery.json"


def validation_artifacts(settings: Mapping[str, Any]) -> dict[str, Path]:
    """The banked G6 and E0 part 2 artifacts that validated this analysis."""
    source = str(settings.get("validated_range", "prereg/0005-e0-glmm-cells.md"))
    number = Path(source).name[:4]
    return {"g6": G6_ARTIFACT, "e0": REPO / "artifacts" / "e0" / f"coverage-{number}.json"}


def check_analysis_code(
    prereg: Path, settings: Mapping[str, Any], amendment: Path | None = None
) -> dict[str, Any]:
    """The analysis code in use, refused unless it is the code G6 and E0 part 2 ran.

    Prereg 0004 pins :data:`noisefloor.provenance.ANALYSIS_CODE`, the lock, and the
    libraries from G6 and E0 part 2 to the E1 analysis: for a preregistration in the chain,
    both artifacts must record the analysis-code hash in use now, and the files must have
    no uncommitted changes. A frozen ``amendment`` of this preregistration may allow other
    code (a deviation, recorded in the outputs): its ```json analysis-code``` block must
    name the hash in use (``{"allows": "<16 hex>"}``), so an amendment written for another
    purpose allows nothing. A file outside the chain (a test fixture) is not checked.
    """
    from noisefloor import provenance  # noqa: PLC0415 - provenance loads R lazily

    current = provenance.analysis_code()
    found: dict[str, Any] = {"hash": current["hash"], "validated_by": {}, "amendment": None}
    if chain_entry(prereg) is None:
        return found
    if current["dirty"]:
        raise ValueError("the analysis code has uncommitted changes")
    for name, path in validation_artifacts(settings).items():
        if not path.exists():
            raise ValueError(f"{_rel(path)} is not banked; G6 and E0 part 2 come first")
        found["validated_by"][name] = json.loads(path.read_text()).get("analysis_code")
    if all(v == current["hash"] for v in found["validated_by"].values()):
        return found
    if amendment is None:
        raise ValueError(
            f"the analysis code ({current['hash']}) is not the code G6 and E0 part 2 ran "
            f"({found['validated_by']}); a frozen amendment must allow it"
        )
    require_frozen(amendment)
    if amends(amendment) != Path(prereg).name[:4]:
        raise ValueError(f"{amendment} does not amend {Path(prereg).name}")
    allows = prereg_block(amendment, "analysis-code").get("allows")
    if allows != current["hash"]:
        raise ValueError(
            f"{_rel(amendment)} allows analysis code {allows}, not the code in use "
            f"({current['hash']})"
        )
    found["amendment"] = _rel(amendment)
    return found


def compute(
    prereg: Path,
    directory: Path = E1_DIR,
    now: datetime | None = None,
    amendment: Path | None = None,
) -> dict[str, str]:
    """Every output file's text, recomputed from the inputs in ``directory``.

    Raises ``ValueError`` while either arm is still running (:func:`arms_not_ended`), or
    when the analysis code is not the code G6 and E0 part 2 ran
    (:func:`check_analysis_code`).
    """
    settings = prereg_block(prereg, "e1-analysis")
    design = E1Design.from_json(prereg_block(prereg, "e1-plan"))
    planned = {s.run_id: s.phase for s in [*local_plan(design), *claude_plan(design)]}
    check_frozen_plan(prereg, design)
    code = check_analysis_code(prereg, settings, amendment)
    entry = chain_entry(prereg)
    frozen = None if entry is None else _utc(str(entry["frozen_at"]))
    records = read_records(directory)
    running = arms_not_ended(records, planned, directory, frozen, now or datetime.now(UTC))
    if running:
        raise ValueError("; ".join(running) + "; outcomes are not analyzed before then")
    rows = runs_table(
        records,
        prereg=_rel(prereg),
        planned=planned,
        block_hours=settings["block_hours"],
        frozen_at=None if entry is None else str(entry["frozen_at"]),
        prereg_sha256=None if entry is None else str(entry["sha256"]),
    )
    data = analysis_frame(rows, settings["exclude_status"])
    left_out = exclusion_counts(rows, settings["exclude_status"])
    variance: dict[str, Any] = {
        "prereg": _rel(prereg),
        "n_runs": len(data),
        "excluded_runs": sum(left_out["local"].values()),
        "exclusions": left_out,
        "schedule": schedule_check(rows, design.days),
        "settings": settings,
        "analysis_code": code,
    }
    try:
        decomp: Decomposition | None = decomposition(data, settings)
    except FitFailedError as exc:  # prereg 0004: reported as not estimable; G7 fails
        decomp = None
        failure = {"estimable": False, "message": str(exc)[:2000], "components": []}
        variance.update(failure)
        mde: dict[str, Any] = {"prereg": _rel(prereg), **failure, "rows": [], "tasks_needed": []}
        del mde["components"]
    if decomp is not None:
        variance.update(
            {
                "estimable": True,
                "backend": decomp.backend,
                "formula": decomp.formula,
                "converged": decomp.converged,
                "messages": list(decomp.messages),
                "components": components_json(decomp),
                "fixed_effects": {str(k): float(v) for k, v in decomp.fixed_effects.items()},
            }
        )
        mde = {"prereg": _rel(prereg), "estimable": True, **mde_tables(decomp, data, settings)}
    sampling_rows = sampling_table(
        rows,
        settings,
        reference=settings["mde"]["reference"],
        assumed=None if decomp is None else float(mde["within_task_var"]) / 2,
        assumed_by_model=None if decomp is None else mde["within_var_by_model"],
    )
    if decomp is not None:
        mark_independence(mde, sampling_rows)
    sampling = {
        "prereg": _rel(prereg),
        "label": "within-configuration outcome variance from replicate runs (D-028 item 3)",
        "rows": sampling_rows,
    }
    scores = {
        "prereg": _rel(prereg),
        "excluded": left_out,
        "rows": scores_table(rows, settings),
    }
    contrasts = {"prereg": _rel(prereg), "excluded": left_out, **contrasts_table(rows, settings)}
    sensitivity = {
        "prereg": _rel(prereg),
        "label": "sensitivity analyses, reported beside the primary results",
        **sensitivity_tables(rows, settings),
    }
    bounds = {
        "prereg": _rel(prereg),
        "source": "e1",
        "label": "E3 policy-sensitivity bounds on the local models' scores and ranking",
        **bounds_table(rows, settings),
    }
    ranges = validated_ranges(settings)
    if ranges is not None:
        mark_ranges(ranges, scores, contrasts, sampling, bounds)
    pairs = complete_pairs(rows, settings["exclude_status"], design.claude_runs_per_task)
    mark_complete_pairs(pairs, len(claude_tasks(design)), scores, contrasts)
    tables = {
        "variance_components.json": variance,
        "mde.json": mde,
        "scores.json": scores,
        "contrasts.json": contrasts,
        "sampling.json": sampling,
        "sensitivity.json": sensitivity,
        "e3_bounds.json": bounds,
    }
    marked = [{**r, "mock_tasks": True} for r in rows]
    return {
        "runs.jsonl": "".join(json.dumps(r, sort_keys=True) + "\n" for r in marked),
        **{name: _dump({**table, "tasks": MOCK_TASKS}) for name, table in tables.items()},
    }


FLAGGED_NOTE = "E1 has no flagging step, so no run has status flagged"
FAMILY_KINDS = {  # contrast family -> kind of the validated-range block (prereg 0005)
    "harness-local": "local-harness",
    "harness-claude": "claude-harness",
    "harness-diff-local": "local-hdiff",
    "models-local": "local-pair",
    "tiers-claude": "claude-tiers",
    "models-claude": "claude-local",
}


def validated_ranges(settings: Mapping[str, Any]) -> dict[str, Any] | None:
    """The ``validated-range`` block of the preregistration ``settings`` names, if any."""
    source = settings.get("validated_range")
    if not source:
        return None
    block: dict[str, Any] = prereg_block(REPO / str(source), "validated-range")
    return dict(block["kinds"])


def _inside(ranges: Mapping[str, Any], kind: str, value: float | None) -> bool | None:
    """Whether ``value`` lies in the span of ``kind`` (None when there is no value)."""
    if value is None or not math.isfinite(value) or kind not in ranges:
        return None
    entry = ranges[kind]
    lo, hi = entry["span"]
    if entry.get("compare", "").startswith("absolute"):
        value = abs(value)
    return bool(lo - 1e-12 <= value <= hi + 1e-12)


def mark_ranges(
    ranges: Mapping[str, Any],
    scores: Mapping[str, Any],
    contrasts: Mapping[str, Any],
    sampling: Mapping[str, Any],
    bounds: Mapping[str, Any],
) -> None:
    """Flag every interval whose point estimate lies outside its kind's validated span.

    Each row gains ``outside_validated_range`` (True, False, or None without an estimate):
    scores by arm, contrasts by family, the sampling table by its discordance share, and
    the E3 bounds by each model's ambiguous share (prereg 0004 and 0005).
    """
    for row in scores["rows"]:
        kind = f"{row['backend']}-score"
        row["outside_validated_range"] = _outside(_inside(ranges, kind, row["score"]))
    for table in ("harness", "harness_differences", "models"):
        for row in contrasts[table]:
            kind = FAMILY_KINDS[str(row["family"])]
            row["outside_validated_range"] = _outside(_inside(ranges, kind, row["difference"]))
    for row in sampling["rows"]:
        found = _inside(ranges, "sampling", float(row["within_var"]))
        row["outside_validated_range"] = _outside(found)
    for row in bounds["entries"]:
        found = _inside(ranges, "bounds", row["ambiguous_share"])
        row["outside_validated_range"] = _outside(found)


def _outside(inside: bool | None) -> bool | None:
    return None if inside is None else not inside


def complete_pairs(
    rows: Sequence[Mapping[str, Any]], exclude: Sequence[str], runs_per_task: int
) -> dict[str, int]:
    """Per Claude tier, the tasks with every planned run of that tier analyzed."""
    counts: dict[tuple[str, str], int] = {}
    for r in _usable(rows, exclude):
        if r["backend"] == "claude":
            key = (str(r["model_id"]), str(r["task"]))
            counts[key] = counts.get(key, 0) + 1
    out: dict[str, int] = {}
    for (tier, _task), n in sorted(counts.items()):
        out[tier] = out.get(tier, 0) + int(n >= runs_per_task)
    return out


def _claude_tiers(row: Mapping[str, Any]) -> list[str]:
    """The Claude tiers a score or contrast row rests on."""
    if row.get("family") == "harness-claude":
        return [str(row["contrast"]).removeprefix("harness:")]
    names = [row["model"]] if "model" in row else [row["a"], row["b"]]
    return [str(m) for m in names if is_claude(str(m))]


def mark_complete_pairs(
    pairs: Mapping[str, int],
    planned: int,
    scores: Mapping[str, Any],
    contrasts: Mapping[str, Any],
) -> None:
    """Label the Claude rows whose realized design differs from their E0 cells.

    The Claude cells of prereg 0005 simulate ``planned`` tasks with every run of a tier
    present. Each score and contrast row that rests on a Claude tier records each tier's
    complete pairs (tasks with both runs analyzed, :func:`complete_pairs`) and
    ``matches_e0_cells``: False when a tier has fewer than ``planned`` (prereg 0004).
    """
    rows = [*scores["rows"], *contrasts["harness"], *contrasts["models"]]
    for row in rows:
        tiers = _claude_tiers(row)
        if not tiers:
            continue
        row["complete_pairs"] = {t: int(pairs.get(t, 0)) for t in tiers}
        row["matches_e0_cells"] = all(pairs.get(t, 0) >= planned for t in tiers)


def bounds_table(rows: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]) -> dict[str, Any]:
    """E3 on E1: the local models' scores and ranking under every policy for ambiguous runs.

    Ambiguous runs are those with status ``infra_error``, ``timeout``, or ``flagged``
    (``evalstats.policy_bounds``; E1 flags no run), on the confirmatory local-arm runs.
    ``entries``: each model's score under every policy, its worst-case bounds, and their
    Imbens and Manski interval. ``rankings``: the order under each policy, and how many
    policies order the models otherwise than ``as_scored``. ``adjacent``: for each
    adjacent pair in the ``as_scored`` order, the worst-case bounds on the difference with
    their Imbens and Manski interval and the identified and robust readings; no E0 cell
    has this shape, so these intervals are reported as unvalidated.
    """
    local = [
        r for r in rows if r["confirmatory"] and r["backend"] == "local" and not r.get("replicate")
    ]
    empty: dict[str, Any] = {"entries": [], "rankings": {}, "ranking_changes": 0, "adjacent": []}
    if not local:
        return {**empty, "flagged": FLAGGED_NOTE}
    frame = pd.DataFrame(
        [
            {
                "model": r["model_id"],
                "task": r["task"],
                "run": r["run_id"],
                "outcome": float(r["outcome"]),
                "status": str(r["status"]),
            }
            for r in local
        ]
    )
    report = policy_bounds(EvalResults.from_frame(frame), level=float(settings["level"]))
    entries = [
        {
            "model": b.model,
            "scores": dict(b.scores),
            "lower": b.lower,
            "upper": b.upper,
            "ci_lo": b.ci_lo,
            "ci_hi": b.ci_hi,
            "ambiguous_runs": b.ambiguous_runs,
            "ambiguous_share": b.ambiguous_share,
            "interval_method": "imbens_manski",
        }
        for b in report.models
    ]
    adjacent = [
        {
            "higher": pair.higher,
            "lower": pair.lower,
            "difference": pair.difference,
            "bound_lo": pair.bound_lo,
            "bound_hi": pair.bound_hi,
            "ci_lo": pair.ci_lo,
            "ci_hi": pair.ci_hi,
            "identified": pair.identified,
            "robust": pair.robust,
            "interval_method": "imbens_manski",
            "validated": False,
        }
        for pair in report.adjacent
    ]
    return {
        "entries": entries,
        "reference_policy": report.policy,
        "rankings": {name: list(order) for name, order in report.rankings.items()},
        "ranking_changes": report.ranking_changes,
        "adjacent": adjacent,
        "flagged": FLAGGED_NOTE,
    }


def sensitivity_tables(
    rows: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]
) -> dict[str, Any]:
    """The sensitivity decompositions of prereg 0004 item 6, beside the primary one.

    With a random intercept for the planned batch, with the strict grade as the outcome,
    and with ``infra_error`` runs counted as failures; each uses the primary settings. The
    batch-term fit also records ``batch_var_upper`` (:func:`batch_upper`), the value the
    batch rule of prereg 0004 items 9 and 10 uses.
    """
    exclude = list(settings["exclude_status"])
    batch = {**settings, "random": [*settings["random"], "batch"]}
    frames = {
        "batch_random_effect": (analysis_frame(rows, exclude), batch),
        "outcome_strict": (analysis_frame(rows, exclude, outcome="outcome_strict"), settings),
        "infra_error_as_failure": (
            analysis_frame(rows, [s for s in exclude if s != "infra_error"], failures=exclude),
            settings,
        ),
    }
    out: dict[str, Any] = {}
    for name, (data, use) in frames.items():
        try:
            decomp = decomposition(data, use)
        except Exception as exc:  # a failed sensitivity fit is reported, not fatal
            out[name] = {"n_runs": len(data), "error": f"{type(exc).__name__}: {exc}"[:300]}
            continue
        out[name] = {
            "n_runs": len(data),
            "formula": decomp.formula,
            "converged": decomp.converged,
            "components": components_json(decomp),
        }
        if name == "batch_random_effect":
            out[name]["batch_var_upper"] = batch_upper(decomp, float(settings["level"]))
            out[name]["batch_var_upper_method"] = BATCH_UPPER_METHOD
    return out


BATCH_UPPER_METHOD = (
    "upper percentile of the batch variance's finite bootstrap replicates at the primary "
    "level (97.5 percent at 0.95); none when more than 10 percent of them failed"
)


def batch_upper(decomp: Decomposition, level: float) -> float | None:
    """The batch variance the batch rule uses (prereg 0004 items 9 and 10).

    The upper ``(1 + level) / 2`` quantile of the batch-term fit's finite bootstrap
    replicate batch variances: bounded by the replicates, unlike the ``basic_log`` upper
    end ``estimate**2 / q_lo``, which grows without bound as the lower quantile nears 0.
    None (the batch-term fit then counts as failed) when the fit has no batch term or more
    than ``MAX_FAILED_SHARE`` of its replicates failed.
    """
    draws = np.asarray(decomp.replicates.get("batch", ()), dtype=float)
    if not enough(draws):
        return None
    ci = boot_variance_interval(0.0, draws, level, "percentile")
    return None if ci is None else ci[1]


def sha256_text(text: str) -> str:
    """SHA-256 of a text's UTF-8 bytes."""
    return hashlib.sha256(text.encode()).hexdigest()


def build(prereg: Path, directory: Path = E1_DIR, amendment: Path | None = None) -> dict[str, Any]:
    """Write every output and the manifest; returns the manifest."""
    outputs = compute(prereg, directory, amendment=amendment)
    files = {}
    for name in (*INPUTS, *STOP_LOGS):
        path = directory / name
        if path.exists():
            files[_rel(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, text in outputs.items():
        (directory / name).write_text(text)
        files[_rel(directory / name)] = sha256_text(text)
    manifest: dict[str, Any] = {"prereg": _rel(prereg), "files": files}
    if amendment is not None:
        manifest["amendment"] = _rel(amendment)
    (directory / "manifest.json").write_text(_dump(manifest))
    return manifest


def verify(directory: Path = E1_DIR) -> list[str]:
    """Differences between the banked outputs and a recomputation (empty if none)."""
    manifest = json.loads((directory / "manifest.json").read_text())
    prereg = REPO / manifest["prereg"]
    problems = []
    for path, digest in manifest["files"].items():
        actual = REPO / path
        if not actual.exists() or hashlib.sha256(actual.read_bytes()).hexdigest() != digest:
            problems.append(f"{path}: missing or hash differs from the manifest")
    amendment = REPO / manifest["amendment"] if manifest.get("amendment") else None
    for name, text in compute(prereg, directory, amendment=amendment).items():
        if (directory / name).read_text() != text:
            problems.append(f"{name}: recomputation differs")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    """``build PREREG`` or ``--verify``."""
    parser = argparse.ArgumentParser(description="E1 analysis (gate G7)")
    parser.add_argument("command", nargs="?", choices=["build"])
    parser.add_argument("prereg", nargs="?", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--amendment", type=Path, help="a frozen amendment that allows other analysis code"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.verify:
        problems = verify()
        for p in problems:
            print(p)
        print("E1-VERIFY: " + ("FAIL" if problems else "PASS"))
        return 1 if problems else 0
    if args.command != "build" or args.prereg is None:
        parser.error("use build PREREG or --verify")
    manifest = build(args.prereg, amendment=args.amendment)
    print(f"wrote {len(manifest['files'])} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
