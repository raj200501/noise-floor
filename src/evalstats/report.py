"""Confidence reports for a leaderboard: which of its orderings the data support.

:func:`confidence_report` turns one row per run into a Markdown report (``evalstats report``
on the command line): each entry's score with a simultaneous rank interval, each adjacent
ordering with its paired interval and whether the data support it, and, when the rows carry a
``status`` column (the harness adapters in :mod:`evalstats.adapters` add one), each score's
bounds over every way of scoring runs that ended in timeouts, infrastructure errors, or flags.
Every section uses the tasks every entry attempted, and the bounds follow the ranking's order,
so a pair of neighbours is the same pair in each section.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats

from evalstats.bounds import PolicyReport, policy_bounds
from evalstats.rank import RankConfidence, rank_confidence
from evalstats.schema import EvalResults

SYNTHETIC_NOTE = "**Synthetic data.** The rows are marked synthetic; these are not model results."


def report_inputs(
    results: EvalResults,
    *,
    level: float = 0.95,
    n_boot: int = 2000,
    seed: int | None = 0,
) -> tuple[RankConfidence, PolicyReport | None]:
    """The rank confidence and the policy bounds that :func:`confidence_report` renders.

    The bounds are ``None`` when ``results`` has no ``status`` column. Otherwise they use the
    tasks every entry attempted (the tasks the ranking uses) and follow the ranking's order.
    """
    ranks = rank_confidence(results, level=level, n_boot=n_boot, seed=seed)
    if "status" not in results.frame.columns:
        return ranks, None
    shared = results.score_matrix().dropna(axis=0, how="any").index
    order = [r.model for r in sorted(ranks.intervals, key=lambda r: r.rank)]
    bounds = policy_bounds(results.subset(tasks=list(shared)), level=level, reference_order=order)
    return ranks, bounds


def is_synthetic(results: EvalResults) -> bool:
    """True if a ``synthetic`` column marks any row as synthetic."""
    if "synthetic" not in results.frame.columns:
        return False
    return any(_truthy(v) for v in results.frame["synthetic"])


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(pd.notna(value) and value)


def confidence_report(
    results: EvalResults,
    *,
    title: str = "leaderboard",
    level: float = 0.95,
    n_boot: int = 2000,
    seed: int | None = 0,
) -> str:
    """A Markdown confidence report for every entry in ``results``.

    Ranks and adjacent orderings come from :func:`evalstats.rank_confidence` on the tasks every
    entry attempted (simultaneous rank intervals at ``level`` from ``n_boot`` task-bootstrap
    replicates, seeded by ``seed``); an ordering counts as supported when its paired interval
    excludes zero. When ``results`` has a ``status`` column, a section from
    :func:`evalstats.policy_bounds` bounds each score over every scoring of ambiguous runs, on
    the same tasks and in the same order. Rows marked ``synthetic`` get a label saying so.
    Raises ``ValueError`` (as ``rank_confidence`` does) for fewer than two entries.
    """
    ranks, bounds = report_inputs(results, level=level, n_boot=n_boot, seed=seed)
    return render_report(ranks, bounds, title=title, synthetic=is_synthetic(results))


def _pct(level: float) -> str:
    return f"{100 * level:g}%"


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def _cell(name: str) -> str:
    """An entry name as a Markdown table cell (a ``|`` would end the cell)."""
    return name.replace("|", "\\|")


def render_report(
    ranks: RankConfidence,
    bounds: PolicyReport | None = None,
    *,
    title: str = "leaderboard",
    synthetic: bool = False,
) -> str:
    """Render a rank-confidence result, and optionally policy bounds, as Markdown.

    ``synthetic=True`` adds a line saying that the rows are synthetic, not model results.
    """
    pct = _pct(ranks.level)
    kind = "simultaneous" if ranks.simultaneous else "marginal"
    dropped = (
        f" ({_count(ranks.dropped_tasks, 'other task')} left out)" if ranks.dropped_tasks else ""
    )
    lines = [f"# Confidence report: {title}", ""]
    if synthetic:
        lines += [SYNTHETIC_NOTE, ""]
    lines += [
        f"{len(ranks.intervals)} entries on the {ranks.n_tasks} tasks every entry attempted"
        f"{dropped}; {pct} {kind} rank intervals from {ranks.n_boot} task-bootstrap replicates.",
        "",
        "| Rank | Entry | Score | Rank interval | Could rank first |",
        "|---:|---|---:|:---:|:---:|",
    ]
    for r in sorted(ranks.intervals, key=lambda x: x.rank):
        first = "yes" if r.could_be_best else "no"
        lines.append(
            f"| {r.rank} | {_cell(r.model)} | {r.score:.3f} | {r.best_rank} to {r.worst_rank} "
            f"| {first} |"
        )
    supported = sum(1 for p in ranks.adjacent if p.supported)
    lines += [
        "",
        "## Adjacent orderings",
        "",
        f"{supported} of {len(ranks.adjacent)} adjacent orderings are supported: their paired "
        f"{pct} interval excludes zero.",
        "",
        "| Higher | Lower | Gap | Interval | Supported |",
        "|---|---|---:|:---:|:---:|",
    ]
    for p in ranks.adjacent:
        mark = "yes" if p.supported else "no"
        lines.append(
            f"| {_cell(p.higher)} | {_cell(p.lower)} | {p.difference:+.3f} | "
            f"[{p.lo:+.3f}, {p.hi:+.3f}] | {mark} |"
        )
    lines += _conflict_lines(ranks)
    if bounds is not None:
        lines += _bounds_lines(bounds)
    return "\n".join(lines) + "\n"


def _conflict_lines(ranks: RankConfidence) -> list[str]:
    """A caution when rank intervals order a pair that its paired interval does not.

    Simultaneous intervals should be the wider ones; with few tasks the task bootstrap's
    critical value can fall below the paired t value, and the rank intervals then separate
    entries that the paired interval does not.
    """
    conflicts = [p for p in ranks.adjacent if p.supported_simultaneous and not p.supported]
    if not conflicts:
        return []
    names = ", ".join(f"{p.higher} over {p.lower}" for p in conflicts)
    those = "that ordering" if len(conflicts) == 1 else "those orderings"
    t_crit = float(stats.t.ppf(0.5 + ranks.level / 2, df=ranks.n_tasks - 1))
    why = ""
    if ranks.critical_value < t_crit:
        why = (
            f" With {ranks.n_tasks} tasks the bootstrap critical value "
            f"({ranks.critical_value:.2f}) is below the paired t value ({t_crit:.2f})."
        )
    return [
        "",
        f"Caution: the rank intervals order {_count(len(conflicts), 'adjacent pair')} whose "
        f"paired interval includes zero ({names}); read {those} as not supported.{why}",
    ]


def _bounds_lines(bounds: PolicyReport) -> list[str]:
    statuses = ", ".join(bounds.ambiguous) or "none"
    lines = ["", "## Ambiguous runs", ""]
    if not any(m.ambiguous_runs for m in bounds.models):
        return [
            *lines,
            f"No run has an ambiguous status ({statuses}), so no score depends on how "
            "such runs are scored.",
        ]
    held = sum(1 for p in bounds.adjacent if p.identified)
    robust = sum(1 for p in bounds.adjacent if p.robust)
    lines += [
        f"Runs with an ambiguous status ({statuses}) can be scored more than one way; each "
        "score's bounds cover every way, on the same tasks as the ranking. "
        f"{held} of {len(bounds.adjacent)} adjacent orderings hold under every way before "
        f"sampling error, and {robust} also with it (their {_pct(bounds.level)} Imbens and "
        "Manski interval excludes zero; that interval has no coverage check of its own).",
        "",
        "| Entry | As scored | Bounds | Ambiguous runs |",
        "|---|---:|:---:|---:|",
    ]
    for m in bounds.models:
        lines.append(
            f"| {_cell(m.model)} | {m.scores['as_scored']:.3f} | [{m.lower:.3f}, {m.upper:.3f}] "
            f"| {m.ambiguous_runs} |"
        )
    return lines
