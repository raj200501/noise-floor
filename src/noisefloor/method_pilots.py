"""Exploratory interval-method pilots for the E1 decomposition (synthetic data only).

Each pilot simulated datasets in the shape of the E1 local arm with planted variance
components (``noisefloor.recovery pilot``, seeds from ``PILOT_SEEDS``, never used by G6 or
E0) and scored every candidate interval rule on each one. The raw rows are banked in
``artifacts/method_pilots/<label>.jsonl`` and ``manifest.json`` records each pilot's
command and the code it ran. ``python -m noisefloor.method_pilots`` recomputes
``summary.json`` from them; ``--check`` exits 1 if the banked summary differs. A pool
either chooses a rule per kind (``choose``, D-023) or scores a rule fixed in advance
(``check``, as pilot 7 does for the task jackknife of D-028, and for the plugin shares
built from it, ``plugin_from``, D-033).

Coverage counts a missing interval as a miss, as G6 does.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evalstats.glmm import LOGISTIC_RESIDUAL_VAR
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
        "median_width": {},
    }
    for rule in rules_of(rows, component):
        hits = sum(_covers(c[rule], c["planted"]) for c in comps)
        out["coverage"][rule] = hits / len(comps)
        out["missing"][rule] = sum(1 for c in comps if not c[rule] or None in c[rule])
        # Power of "supported as non-zero": the interval's lower end is above 0.
        above = sum(1 for c in comps if c[rule] and c[rule][0] is not None and c[rule][0] > 0)
        out["above_zero"][rule] = above / len(comps)
        widths = [c[rule][1] - c[rule][0] for c in comps if c[rule] and None not in c[rule]]
        out["median_width"][rule] = statistics.median(widths) if widths else None
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


KINDS = {
    "random": ("task", "model:task"),
    "fixed": ("model", *TWO_LEVEL, "day", "block"),
}
TIE = 0.01  # rules within this much of the best minimum coverage are tied (D-023)
FLOOR = 0.90  # every component's coverage must reach this for a headline kind (D-028)
POOLED_FLOOR = 0.935  # and the kind's pooled coverage this (D-028)


def kind_members(names: Sequence[str], kind: str) -> list[str]:
    """Components of one kind of result: random, fixed, share, fixed-share, or mde.

    ``fixed-share`` is the shares of the fixed factors only (pilot 7 scores them: its two
    bootstrap replicates give the random components no interval worth scoring).
    """
    if kind in KINDS:
        return [n for n in KINDS[kind] if n in names]
    if kind == "fixed-share":
        return [f"share:{n}" for n in KINDS["fixed"] if f"share:{n}" in names]
    return [n for n in names if n.startswith(f"{kind}:")]


def choose_rule(rows: Sequence[Mapping[str, Any]], kind: str) -> dict[str, Any]:
    """The D-023 rule for one kind of result.

    The rule with the highest minimum coverage over the kind's components wins; rules
    within ``TIE`` of that minimum are broken by the smaller median width, summed over
    the components, each component's widths divided by its median planted value
    (``1`` if that is 0) so that no single scale dominates.
    """
    names = sorted({n for row in rows for n in row["components"]})
    members = kind_members(names, kind)
    rules = sorted(set.intersection(*(set(rules_of(rows, m)) for m in members)))
    summaries = {m: component_summary(rows, m) for m in members}
    minimum = {r: min(summaries[m]["coverage"][r] for m in members) for r in rules}
    best = max(minimum.values())
    tied = [r for r in rules if minimum[r] >= best - TIE]

    def width(rule: str) -> float:
        total = 0.0
        for m in members:
            scale = statistics.median(
                [row["components"][m]["planted"] for row in rows if m in row["components"]]
            )
            w = summaries[m]["median_width"][rule]
            total += math.inf if w is None else w / (scale or 1.0)
        return total

    rule = min(tied, key=lambda r: (width(r), r))
    status = rule_status(rows, kind, rule)
    return {
        **status,
        "min_coverage_by_rule": minimum,
        "tied": tied,
    }


def rule_status(rows: Sequence[Mapping[str, Any]], kind: str, rule: str) -> dict[str, Any]:
    """Coverage of one fixed rule over a kind's components, and the D-028 headline test.

    A kind is headline when its coverage pooled over components and datasets is at least
    ``POOLED_FLOOR`` and every component's is at least ``FLOOR``.
    """
    names = sorted({n for row in rows for n in row["components"]})
    members = kind_members(names, kind)
    coverage = {m: component_summary(rows, m)["coverage"][rule] for m in members}
    hits = total = 0
    for m in members:
        for row in rows:
            comp = row["components"].get(m)
            if comp is not None:
                total += 1
                hits += _covers(comp.get(rule), comp["planted"])
    pooled = hits / total if total else 0.0
    minimum = min(coverage.values())
    return {
        "kind": kind,
        "components": members,
        "rule": rule,
        "coverage": coverage,
        "min_coverage": minimum,
        "pooled_coverage": pooled,
        "headline": minimum >= FLOOR and pooled >= POOLED_FLOOR,
    }


def with_plugin_shares(
    rows: Sequence[Mapping[str, Any]], random_rule: str, box_rule: str
) -> list[dict[str, Any]]:
    """Rows whose share entries also carry the ``plugin`` share interval.

    As ``evalstats.glmm.plugin_share``: each component's interval under the rule chosen for
    its kind, mapped through ``v / (v + rest)`` with the rest of the estimated total held at
    its estimate; the residual share takes its percentile interval.
    """
    out = []
    for row in rows:
        comps = {k: dict(v) for k, v in row["components"].items()}
        names = [k for k in comps if not k.startswith(("share:", "mde:"))]
        total = sum(comps[n]["estimate"] for n in names) + LOGISTIC_RESIDUAL_VAR
        for n in names:
            key = f"share:{n}"
            if key not in comps:
                continue
            ci = comps[n].get(random_rule if n in KINDS["random"] else box_rule)
            if not ci or None in ci:
                comps[key]["plugin"] = None
                continue
            rest = total - comps[n]["estimate"]
            comps[key]["plugin"] = [ci[0] / (ci[0] + rest), ci[1] / (ci[1] + rest)]
        if "share:residual" in comps:
            comps["share:residual"]["plugin"] = comps["share:residual"].get("percentile")
        out.append({**row, "components": comps})
    return out


def choices(rows: Sequence[Mapping[str, Any]], kinds: Sequence[str]) -> dict[str, Any]:
    """D-023 choices for the listed kinds.

    Shares are chosen last, with the plugin candidate built from the rules chosen for the
    random and fixed components.
    """
    out = {k: choose_rule(rows, k) for k in kinds if k != "share"}
    if "share" in kinds:
        aug = with_plugin_shares(rows, out["random"]["rule"], out["fixed"]["rule"])
        out["share"] = choose_rule(aug, "share")
    return out


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
        entry = {"pilots": pool["pilots"], **summarize(joined)}
        if pool.get("plugin_from"):  # plugin shares from these rules, before any check
            source = pool["plugin_from"]
            joined = with_plugin_shares(joined, source["random"], source["fixed"])
        if pool.get("choose"):
            entry["choice"] = choices(joined, pool["choose"])
        if pool.get("check"):
            entry["check"] = {k: rule_status(joined, k, r) for k, r in pool["check"].items()}
        out["pools"][pool["label"]] = entry
    return out


def dumps(obj: Any) -> str:
    """Stable JSON text for the banked summary."""
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"


def same_summary(a: Any, b: Any) -> bool:
    """Equal JSON values, with floats equal to 1e-9 relative (or 1e-12 absolute).

    The pinned numpy and scipy differ between Python 3.11 and 3.12 (uv.lock), and they
    round some sums differently in the last digit, so a bit-exact text comparison would
    call a summary stale on one interpreter and current on the other.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_summary(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_summary(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return bool(a == b) and type(a) is type(b)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m noisefloor.method_pilots [--check] [--dir DIR]``."""
    parser = argparse.ArgumentParser(description="summarize the exploratory method pilots")
    parser.add_argument("--check", action="store_true", help="fail if summary.json is stale")
    parser.add_argument("--dir", type=Path, default=DIR)
    args = parser.parse_args(argv)
    text = dumps(build(args.dir))
    target = args.dir / "summary.json"
    if args.check:
        same = target.exists() and same_summary(json.loads(target.read_text()), json.loads(text))
        print("method pilots: summary current" if same else "method pilots: summary STALE")
        return 0 if same else 1
    target.write_text(text)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
