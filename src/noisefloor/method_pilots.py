"""Exploratory interval-method pilots for the E1 decomposition (synthetic data only).

Each pilot simulated datasets in the shape of the E1 local arm with planted variance
components (``noisefloor.recovery pilot``, seeds from ``PILOT_SEEDS``, never used by G6 or
E0) and scored every candidate interval rule on each one. The raw rows are banked in
``artifacts/method_pilots/<label>.jsonl`` and ``manifest.json`` records each pilot's
command and the code it ran. ``python -m noisefloor.method_pilots`` recomputes
``summary.json`` from them; ``--check`` exits 1 if the banked summary differs.

Coverage counts a missing interval as a miss, as G6 does.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from noisefloor.recovery import TWO_LEVEL

REPO = Path(__file__).resolve().parents[2]
DIR = REPO / "artifacts" / "method_pilots"
RULE_KEYS = ("estimate", "planted")
SD_RATIO_MIN_PLANTED = 0.01  # contributions large enough that the box rarely reaches 0
Z975 = 1.959963984540054


def read_rows(path: Path) -> list[dict[str, Any]]:
    """The pilot's rows, one per dataset, in seed order."""
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    return sorted(rows, key=lambda r: int(r["seed"]))


def _covers(interval: Any, value: float) -> bool:
    if not interval or interval[0] is None or interval[1] is None:
        return False
    return bool(interval[0] <= value <= interval[1])


def rules_of(rows: Sequence[Mapping[str, Any]], component: str) -> list[str]:
    """Interval rules scored for a component in every row that has it."""
    sets = [
        {k for k in row["components"][component] if k not in RULE_KEYS}
        for row in rows
        if component in row["components"]
    ]
    return sorted(set.intersection(*sets)) if sets else []


def component_summary(rows: Sequence[Mapping[str, Any]], component: str) -> dict[str, Any]:
    """Coverage of each rule, bias, and spread of one component over the pilot's rows."""
    comps = [row["components"][component] for row in rows if component in row["components"]]
    errors = [c["estimate"] - c["planted"] for c in comps if c["estimate"] is not None]
    out: dict[str, Any] = {
        "n": len(comps),
        "planted_mean": sum(c["planted"] for c in comps) / len(comps),
        "bias": sum(errors) / len(errors) if errors else None,
        "coverage": {},
        "missing": {},
        "above_zero": {},
    }
    for rule in rules_of(rows, component):
        hits = sum(_covers(c[rule], c["planted"]) for c in comps)
        out["coverage"][rule] = hits / len(comps)
        out["missing"][rule] = sum(1 for c in comps if not c[rule] or None in c[rule])
        # Power of "supported as non-zero": the interval's lower end is above 0.
        above = sum(1 for c in comps if c[rule] and c[rule][0] is not None and c[rule][0] > 0)
        out["above_zero"][rule] = above / len(comps)
    return out


def two_level_pooled(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Coverage of each box rule pooled over the two-level factors (harness and the rest)."""
    names = [n for n in TWO_LEVEL if any(n in row["components"] for row in rows)]
    rules = sorted(set.intersection(*(set(rules_of(rows, n)) for n in names))) if names else []
    out: dict[str, Any] = {"factors": names, "coverage": {}}
    for rule in rules:
        hits = total = 0
        for name in names:
            for row in rows:
                comp = row["components"].get(name)
                if comp is not None:
                    total += 1
                    hits += _covers(comp[rule], comp["planted"])
        out["coverage"][rule] = hits / total
        out["n"] = total
    return out


def sd_ratio(rows: Sequence[Mapping[str, Any]], component: str, rule: str) -> dict[str, Any] | None:
    """Empirical sd of a two-level coefficient over the sd its intervals imply.

    For a balanced two-level factor the contribution is ``beta**2 / 4``, so the
    coefficient's estimate is ``2 * sqrt(estimate)`` and a box ``[lo, hi]`` with
    ``lo > 0`` is the coefficient interval ``[2 sqrt(lo), 2 sqrt(hi)]``, whose half-width
    over 1.96 is the implied sd. A ratio above 1 means the intervals are too narrow.
    """
    comps = [row["components"][component] for row in rows if component in row["components"]]
    if not comps or min(c["planted"] for c in comps) < SD_RATIO_MIN_PLANTED:
        return None
    betas = [2 * math.sqrt(c["estimate"]) for c in comps if c["estimate"] is not None]
    widths = [
        (2 * math.sqrt(c[rule][1]) - 2 * math.sqrt(c[rule][0])) / (2 * Z975)
        for c in comps
        if c.get(rule) and None not in c[rule] and c[rule][0] > 0
    ]
    if len(betas) < 2 or not widths:
        return None
    mean = sum(betas) / len(betas)
    sd = math.sqrt(sum((b - mean) ** 2 for b in betas) / len(betas))
    implied = sum(widths) / len(widths)
    planted_beta = 2 * math.sqrt(sum(c["planted"] for c in comps) / len(comps))
    return {
        "rule": rule,
        "planted_beta": planted_beta,
        "mean_beta": mean,
        "empirical_sd": sd,
        "implied_sd": implied,
        "ratio": sd / implied,
        "n_intervals": len(widths),
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Everything the method decisions cite, for one pilot or one pool of pilots."""
    names = sorted({n for row in rows for n in row["components"]})
    ratios = {}
    for name in TWO_LEVEL:
        if name in names:
            found = sd_ratio(rows, name, "box_basic")
            if found is not None:
                ratios[name] = found
    return {
        "datasets": len(rows),
        "seeds": [int(rows[0]["seed"]), int(rows[-1]["seed"])] if rows else [],
        "components": {n: component_summary(rows, n) for n in names},
        "two_level": two_level_pooled(rows),
        "sd_ratio": ratios,
    }


def build(directory: Path = DIR) -> dict[str, Any]:
    """The summary of every pilot and pool listed in the manifest."""
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = {p["label"]: read_rows(directory / p["file"]) for p in manifest["pilots"]}
    out: dict[str, Any] = {
        "synthetic": True,
        "exploratory": True,
        "manifest": "artifacts/method_pilots/manifest.json",
        "pilots": {label: summarize(r) for label, r in rows.items()},
        "pools": {},
    }
    for pool in manifest.get("pools", []):
        joined = sorted(
            (row for label in pool["pilots"] for row in rows[label]),
            key=lambda r: int(r["seed"]),
        )
        out["pools"][pool["label"]] = {"pilots": pool["pilots"], **summarize(joined)}
    return out


def dumps(obj: Any) -> str:
    """Stable JSON text for the banked summary."""
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m noisefloor.method_pilots [--check] [--dir DIR]``."""
    parser = argparse.ArgumentParser(description="summarize the exploratory method pilots")
    parser.add_argument("--check", action="store_true", help="fail if summary.json is stale")
    parser.add_argument("--dir", type=Path, default=DIR)
    args = parser.parse_args(argv)
    text = dumps(build(args.dir))
    target = args.dir / "summary.json"
    if args.check:
        same = target.exists() and target.read_text() == text
        print("method pilots: summary current" if same else "method pilots: summary STALE")
        return 0 if same else 1
    target.write_text(text)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
