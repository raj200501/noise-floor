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
* ``manifest.json``: the preregistration and the SHA-256 of every input and output.

``python -m noisefloor.analysis.e1 --verify`` recomputes every output from the inputs and
fails on any difference. Results are model results on procedurally generated mock tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from evalstats import EvalResults, tasks_needed
from evalstats.glmm import LOGISTIC_RESIDUAL_VAR, Decomposition, boot_coef_interval, decompose
from noisefloor.analysis import mde as mde_lib
from noisefloor.e1 import E1Design, claude_plan, local_plan
from noisefloor.models import MODELS, is_claude

REPO = Path(__file__).resolve().parents[3]
E1_DIR = REPO / "artifacts" / "e1"
INPUTS = ("local.jsonl", "claude.jsonl")
OUTPUTS = ("runs.jsonl", "variance_components.json", "mde.json")
GAPS = (0.02, 0.05, 0.10)


def prereg_block(path: Path, name: str) -> Any:
    """The JSON of the ```json <name>``` block of a preregistration file."""
    match = re.search(rf"```json[ \t]+{re.escape(name)}[ \t]*\n(.*?)\n```", path.read_text(), re.S)
    if match is None:
        raise ValueError(f"{path}: no ```json {name}``` block")
    return json.loads(match.group(1))


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
) -> list[dict[str, Any]]:
    """One row per run; ``planned`` maps each preregistered run id to its phase."""
    rows = []
    for rec in sorted(records, key=lambda r: str(r["run_id"])):
        day, block = time_block(str(rec.get("started_at", "")), block_hours)
        model = str(rec["model"])
        rows.append(
            {
                "run_id": rec["run_id"],
                "prereg": prereg,
                "model_id": model,
                "backend": "claude" if is_claude(model) else "local",
                "family": "claude" if is_claude(model) else MODELS[model].family,
                "task": f"{rec['family']}-d{rec['difficulty']}-s{rec['task_seed']}",
                "harness": rec["harness"],
                "factors": dict(rec.get("factors", {})),
                "day": day,
                "block": block,
                "planned_slot": str(rec.get("block", "")).split("/")[0],
                "batch_id": rec.get("batch_id"),
                "outcome": int(float(rec.get("outcome", 0.0)) >= 0.5),
                "outcome_strict": rec.get("outcome_strict"),
                "status": rec.get("status"),
                "end_reason": rec.get("end_reason"),
                "confirmatory": planned.get(str(rec["run_id"])) == rec.get("phase"),
            }
        )
    return rows


def analysis_frame(rows: Sequence[Mapping[str, Any]], exclude: Sequence[str]) -> pd.DataFrame:
    """Confirmatory local-arm runs for the decomposition (excluded statuses dropped)."""
    keep = [
        r
        for r in rows
        if r["confirmatory"] and r["backend"] == "local" and r["status"] not in exclude
    ]
    if not keep:
        raise ValueError("no confirmatory local-arm runs to analyze")
    days = {d: f"d{i + 1}" for i, d in enumerate(sorted({r["day"] for r in keep}))}
    return pd.DataFrame(
        [
            {
                "model": r["model_id"],
                "task": r["task"],
                "run": r["run_id"],
                "outcome": float(r["outcome"]),
                **{name: str(value) for name, value in r["factors"].items()},
                "day": days[r["day"]],
                "block": r["block"],
            }
            for r in keep
        ]
    )


def decomposition(data: pd.DataFrame, settings: Mapping[str, Any]) -> Decomposition:
    """The preregistered decomposition (a fixed factor held constant is left out)."""
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
    variances = {c.name: c.variance for c in decomp.components if c.kind == "random"}
    etas = mde_lib.reference_eta(
        decomp.fixed_effects, data, models, cfg["reference"], cfg["averaged"]
    )
    pooled = mde_lib.pooled_design(etas, variances["task"], variances["model:task"])
    point = mde_lib.mde_values(pooled, designs)
    reps = mde_lib.replicate_mdes(
        decomp.coefficient_replicates,
        {k: decomp.replicates[k] for k in ("task", "model:task")},
        data,
        models=models,
        reference=cfg["reference"],
        averaged=cfg["averaged"],
        designs=designs,
    )
    level = float(settings["level"])
    rows = []
    for t, k in designs:
        ci = boot_coef_interval(point[(t, k)], reps[(t, k)], level, str(cfg["interval"]))
        rows.append(
            {
                "tasks": t,
                "runs": k,
                "mde": point[(t, k)],
                "lo": None if ci is None else max(0.0, ci[0]),
                "hi": None if ci is None else ci[1],
                "interval_method": f"parametric_bootstrap_{cfg['interval']}",
            }
        )
    needed = []
    for gap in GAPS:
        for k in sorted({k for _, k in designs}):
            needed.append({"gap": gap, "runs": k, "tasks": tasks_needed(gap, pooled, k)})
    return {
        "reference": dict(cfg["reference"]),
        "averaged": list(cfg["averaged"]),
        "between_task_var": pooled.between_task_var,
        "within_task_var": pooled.within_task_var,
        "rows": rows,
        "tasks_needed": needed,
    }


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


def compute(prereg: Path, directory: Path = E1_DIR) -> dict[str, str]:
    """Every output file's text, recomputed from the inputs in ``directory``."""
    settings = prereg_block(prereg, "e1-analysis")
    design = E1Design.from_json(prereg_block(prereg, "e1-plan"))
    planned = {s.run_id: s.phase for s in [*local_plan(design), *claude_plan(design)]}
    rows = runs_table(
        read_records(directory),
        prereg=_rel(prereg),
        planned=planned,
        block_hours=settings["block_hours"],
    )
    data = analysis_frame(rows, settings["exclude_status"])
    decomp = decomposition(data, settings)
    excluded = sum(
        1
        for r in rows
        if r["confirmatory"]
        and r["backend"] == "local"
        and r["status"] in settings["exclude_status"]
    )
    variance = {
        "prereg": _rel(prereg),
        "n_runs": len(data),
        "excluded_runs": excluded,
        "backend": decomp.backend,
        "formula": decomp.formula,
        "converged": decomp.converged,
        "messages": list(decomp.messages),
        "settings": settings,
        "components": components_json(decomp),
    }
    mde = {"prereg": _rel(prereg), **mde_tables(decomp, data, settings)}
    return {
        "runs.jsonl": "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
        "variance_components.json": _dump(variance),
        "mde.json": _dump(mde),
    }


def sha256_text(text: str) -> str:
    """SHA-256 of a text's UTF-8 bytes."""
    return hashlib.sha256(text.encode()).hexdigest()


def build(prereg: Path, directory: Path = E1_DIR) -> dict[str, Any]:
    """Write every output and the manifest; returns the manifest."""
    outputs = compute(prereg, directory)
    files = {}
    for name in INPUTS:
        path = directory / name
        if path.exists():
            files[_rel(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, text in outputs.items():
        (directory / name).write_text(text)
        files[_rel(directory / name)] = sha256_text(text)
    manifest = {"prereg": _rel(prereg), "files": files}
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
    for name, text in compute(prereg, directory).items():
        if (directory / name).read_text() != text:
            problems.append(f"{name}: recomputation differs")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    """``build PREREG`` or ``--verify``."""
    parser = argparse.ArgumentParser(description="E1 analysis (gate G7)")
    parser.add_argument("command", nargs="?", choices=["build"])
    parser.add_argument("prereg", nargs="?", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.verify:
        problems = verify()
        for p in problems:
            print(p)
        print("E1-VERIFY: " + ("FAIL" if problems else "PASS"))
        return 1 if problems else 0
    if args.command != "build" or args.prereg is None:
        parser.error("use build PREREG or --verify")
    manifest = build(args.prereg)
    print(f"wrote {len(manifest['files'])} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
