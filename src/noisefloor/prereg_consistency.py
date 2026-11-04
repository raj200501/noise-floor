"""Consistency of the E1, E0 part 2, and G6 preregistrations (reviews ops-8 and stats-6).

G6 (0006) and E0 part 2 (0005) validate the E1 analysis (0004) only if they simulate the E1
shape and analyze it with the E1 settings. :func:`problems` lists every mismatch between
the three files' blocks; the test suite runs it on the frozen files, so a mismatch fails
G1, and it runs before the three are frozen.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from noisefloor.e1 import E1Design
from noisefloor.preregs import prereg_block
from noisefloor.recovery import FIXED_TERMS, MDE_AVERAGED, MDE_REFERENCE, TWO_LEVEL

LOCAL_FAMILIES = ("harness-local", "harness-diff-local", "models-local")


def _same(out: list[str], what: str, a: Any, b: Any) -> None:
    if a != b:
        out.append(f"{what}: {a!r} != {b!r}")


def design_problems(plan: E1Design, config: Mapping[str, Any]) -> list[str]:
    """Mismatches between the E1 plan and a recovery-config's design and planted values."""
    out: list[str] = []
    design, planted = config["design"], config["planted"]
    _same(out, "n_models vs the E1 models", design["n_models"], len(plan.models))
    _same(
        out,
        "n_tasks vs the E1 tasks",
        design["n_tasks"],
        len(plan.families) * plan.tasks_per_family,
    )
    for key in ("runs_per_cell", "batch_size", "days", "blocks_per_day"):
        _same(out, key, design[key], getattr(plan, key))
    _same(out, "planted model effects per model", len(planted["model"]), design["n_models"])
    _same(out, "planted day effects per day", len(planted["day"]), design["days"])
    _same(out, "planted block effects per block", len(planted["block"]), design["blocks_per_day"])
    return out


def settings_problems(
    plan: E1Design, settings: Mapping[str, Any], analysis: Mapping[str, Any]
) -> list[str]:
    """Mismatches between a recovery-config's settings and the E1 analysis settings."""
    out: list[str] = []
    for key in ("n_boot", "nagq", "boot_interval", "box_interval", "share_interval", "level"):
        _same(out, key, settings[key], analysis[key])
    _same(
        out,
        "fixed_interval",
        settings.get("fixed_interval", "box"),
        analysis.get("fixed_interval", "box"),
    )
    _same(out, "random terms", list(settings["random"]), list(analysis["random"]))
    _same(out, "fixed terms", list(FIXED_TERMS), list(analysis["fixed"]))
    mde = analysis["mde"]
    _same(
        out,
        "MDE designs",
        [list(d) for d in settings["mde_designs"]],
        [list(d) for d in mde["designs"]],
    )
    _same(out, "MDE interval", settings["mde_interval"], mde["interval"])
    _same(out, "MDE averaged factors", list(MDE_AVERAGED), list(mde["averaged"]))
    coded = {name: str(plan.levels[name][int(MDE_REFERENCE[name])]) for name in TWO_LEVEL}
    reference = {name: str(value) for name, value in mde["reference"].items()}
    _same(out, "MDE reference (simulation codes mapped to E1 levels)", coded, reference)
    score = analysis["score_interval"]
    local_score = score["local"] if isinstance(score, Mapping) else score
    _same(out, "local score interval", settings["score_interval"], local_score)
    contrast = analysis["contrast_interval"]
    for family in LOCAL_FAMILIES:
        method = contrast[family] if isinstance(contrast, Mapping) else contrast
        _same(out, f"{family} interval", settings["contrast_interval"], method)
    return out


def problems(e1: Path, e0: Path, g6: Path) -> list[str]:
    """Every mismatch among preregistrations 0004 (``e1``), 0005 (``e0``), and 0006 (``g6``)."""
    plan = E1Design.from_json(prereg_block(e1, "e1-plan"))
    analysis = prereg_block(e1, "e1-analysis")
    e0_config = prereg_block(e0, "recovery-config")
    g6_config = prereg_block(g6, "recovery-config")
    out: list[str] = []
    if e0_config != g6_config:
        out.append("the recovery-config blocks of 0005 and 0006 differ")
    out += design_problems(plan, g6_config)
    out += settings_problems(plan, g6_config["settings"], analysis)
    return out
