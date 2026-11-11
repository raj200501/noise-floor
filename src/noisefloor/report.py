"""Figures for README.md and docs/REPORT.md, generated from banked artifacts (``make report``).

``python -m noisefloor.report`` writes static SVG figures into ``docs/figures/`` and a
``manifest.json`` with the SHA-256 of every artifact each figure was drawn from;
``--check`` exits 1 when a figure is missing or differs from what the artifacts give now.
Figures carry their own light and dark colors (CSS custom properties with a
``prefers-color-scheme`` query) and an SVG title and description for screen readers. A
figure whose artifact does not exist yet is skipped and listed as skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
FIGURES = REPO / "docs" / "figures"
BOARDS = {"tb2": "Terminal-Bench 2.0", "epoch-swebench": "SWE-bench Verified (Epoch AI)"}

# Reference palette roles (light, dark): chart surface, inks, hairlines, and series slot 1.
STYLE = """
svg { --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; }
@media (prefers-color-scheme: dark) {
  svg { --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --s1: #3987e5; --s2: #d95926; --s3: #199e70; }
}
text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.bg { fill: var(--surface); }
.title { fill: var(--ink); font-size: 15px; font-weight: 600; }
.sub { fill: var(--ink2); font-size: 12px; }
.label { fill: var(--ink2); font-size: 11px; }
.note { fill: var(--ink2); font-size: 11px; font-variant-numeric: tabular-nums; }
.tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.grid { stroke: var(--grid); stroke-width: 1; }
.axis { stroke: var(--axis); stroke-width: 1; }
.ci { stroke: var(--s1); stroke-width: 2; stroke-linecap: round; }
.dot { fill: var(--s1); stroke: var(--surface); stroke-width: 2; }
.line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.c1 { stroke: var(--s1); } .c2 { stroke: var(--s2); } .c3 { stroke: var(--s3); }
.f1 { fill: var(--s1); } .f2 { fill: var(--s2); } .f3 { fill: var(--s3); }
.ring { stroke: var(--surface); stroke-width: 2; }
""".strip()


@dataclass(frozen=True)
class Row:
    """One labeled estimate with its interval (``lo``/``hi`` may be missing)."""

    label: str
    estimate: float
    lo: float | None
    hi: float | None
    note: str = ""


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _fit(text: str, width: float, px_per_char: float = 7.0) -> str:
    """Shorten a label to fit ``width`` pixels at 11px (an ellipsis marks the cut)."""
    room = max(4, int(width / px_per_char))
    return text if len(text) <= room else text[: room - 1] + "…"


def nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    """Round tick values covering ``[lo, hi]`` (steps of 1, 2, or 5 times a power of 10)."""
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / count
    power = 10 ** math.floor(math.log10(raw))
    step = next(m * power for m in (1, 2, 5, 10) if m * power >= raw)
    first = math.ceil(lo / step - 1e-9) * step
    ticks = []
    value = first
    while value <= hi + step * 1e-9:
        ticks.append(round(value, 12))
        value += step
    return ticks


def interval_chart(
    rows: Sequence[Row],
    *,
    title: str,
    subtitle: str,
    axis_label: str,
    domain: tuple[float, float],
    tick_format: Callable[[float], str] = lambda v: f"{v:g}",
    width: int = 760,
    row_height: int = 16,
    label_width: int = 260,
    note_width: int = 120,
) -> str:
    """A dot-and-interval chart: one row per estimate on a shared horizontal axis."""
    top, bottom = 58, 46
    height = top + row_height * len(rows) + bottom
    x0, x1 = label_width + 12, width - note_width - 12
    lo_d, hi_d = domain

    def x(v: float) -> float:
        return x0 + (min(max(v, lo_d), hi_d) - lo_d) / (hi_d - lo_d) * (x1 - x0)

    desc = f"{subtitle} One row per item; the dot is the estimate and the line its interval."
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">',
        f'<title id="t">{_esc(title)}</title>',
        f'<desc id="d">{_esc(desc)}</desc>',
        f"<style>{STYLE}</style>",
        f'<rect class="bg" x="0" y="0" width="{width}" height="{height}"/>',
        f'<text class="title" x="16" y="24">{_esc(title)}</text>',
        f'<text class="sub" x="16" y="42">{_esc(subtitle)}</text>',
    ]
    plot_bottom = top + row_height * len(rows)
    for tick in nice_ticks(lo_d, hi_d):
        tx = x(tick)
        out.append(
            f'<line class="grid" x1="{tx:.1f}" y1="{top - 4}" x2="{tx:.1f}" y2="{plot_bottom}"/>'
        )
        out.append(
            f'<text class="tick" x="{tx:.1f}" y="{plot_bottom + 16}" text-anchor="middle">'
            f"{_esc(tick_format(tick))}</text>"
        )
    out.append(f'<line class="axis" x1="{x0}" y1="{plot_bottom}" x2="{x1}" y2="{plot_bottom}"/>')
    out.append(
        f'<text class="tick" x="{(x0 + x1) / 2:.1f}" y="{plot_bottom + 34}" '
        f'text-anchor="middle">{_esc(axis_label)}</text>'
    )
    for i, row in enumerate(rows):
        cy = top + row_height * i + row_height / 2
        out.append(
            f'<text class="label" x="{label_width}" y="{cy + 4:.1f}" text-anchor="end">'
            f"{_esc(_fit(row.label, label_width - 16))}</text>"
        )
        if row.lo is not None and row.hi is not None:
            out.append(
                f'<line class="ci" x1="{x(row.lo):.1f}" y1="{cy:.1f}" x2="{x(row.hi):.1f}" '
                f'y2="{cy:.1f}"/>'
            )
        out.append(f'<circle class="dot" cx="{x(row.estimate):.1f}" cy="{cy:.1f}" r="4"/>')
        if row.note:
            out.append(f'<text class="note" x="{x1 + 12}" y="{cy + 4:.1f}">{_esc(row.note)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def short_label(entry: str) -> str:
    """An entry name with a long bracketed run id cut to its first 4 characters."""
    name, sep, rest = entry.partition(" [")
    if not sep or not rest.endswith("]") or len(rest) <= 6:
        return entry
    return f"{name} [{rest[:4]}]"


@dataclass(frozen=True)
class Series:
    """One line: points ``(x, y, lo, hi)`` with optional vertical intervals."""

    label: str
    points: Sequence[tuple[float, float, float | None, float | None]]


def line_chart(
    series: Sequence[Series],
    *,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    x_ticks: Sequence[float],
    y_domain: tuple[float, float],
    y_format: Callable[[float], str] = lambda v: f"{v:g}",
    width: int = 760,
    height: int = 380,
) -> str:
    """Up to three series on a log x axis, with point intervals, a legend, and end labels."""
    if not 1 <= len(series) <= 3:
        raise ValueError("line_chart draws one to three series (the palette's all-pairs cap)")
    left, right, top, bottom = 64, 96, 78, 52
    x0, x1, y0, y1 = left, width - right, height - bottom, top
    lx0, lx1 = math.log(min(x_ticks)), math.log(max(x_ticks))
    lo_d, hi_d = y_domain

    def x(v: float) -> float:
        return x0 + (math.log(v) - lx0) / (lx1 - lx0) * (x1 - x0)

    def y(v: float) -> float:
        return y0 - (min(max(v, lo_d), hi_d) - lo_d) / (hi_d - lo_d) * (y0 - y1)

    names = ", ".join(s.label for s in series)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">',
        f'<title id="t">{_esc(title)}</title>',
        f'<desc id="d">{_esc(subtitle)} Lines: {_esc(names)}; each point has its interval.</desc>',
        f"<style>{STYLE}</style>",
        f'<rect class="bg" x="0" y="0" width="{width}" height="{height}"/>',
        f'<text class="title" x="16" y="24">{_esc(title)}</text>',
        f'<text class="sub" x="16" y="42">{_esc(subtitle)}</text>',
    ]
    for tick in nice_ticks(lo_d, hi_d):
        ty = y(tick)
        out.append(f'<line class="grid" x1="{x0}" y1="{ty:.1f}" x2="{x1}" y2="{ty:.1f}"/>')
        out.append(
            f'<text class="tick" x="{x0 - 8}" y="{ty + 4:.1f}" text-anchor="end">'
            f"{_esc(y_format(tick))}</text>"
        )
    out.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}"/>')
    for tick in x_ticks:
        out.append(
            f'<text class="tick" x="{x(tick):.1f}" y="{y0 + 16}" text-anchor="middle">'
            f"{tick:g}</text>"
        )
    out.append(
        f'<text class="tick" x="{(x0 + x1) / 2:.1f}" y="{y0 + 36}" text-anchor="middle">'
        f"{_esc(x_label)}</text>"
    )
    out.append(
        f'<text class="tick" x="16" y="{(y0 + y1) / 2:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 16 {(y0 + y1) / 2:.1f})">{_esc(y_label)}</text>'
    )
    legend_x = x0
    for i, s in enumerate(series, start=1):
        out.append(
            f'<line class="line c{i}" x1="{legend_x}" y1="58" x2="{legend_x + 18}" y2="58"/>'
        )
        out.append(f'<text class="label" x="{legend_x + 24}" y="62">{_esc(s.label)}</text>')
        legend_x += 30 + int(len(s.label) * 6.5)
    ends = [y(max(s.points)[1]) for s in series]
    # End labels only where the lines end apart; converging lines keep the legend alone.
    labelled = all(abs(a - b) >= 14 for i, a in enumerate(ends) for b in ends[i + 1 :])
    for i, s in enumerate(series, start=1):
        pts = sorted(s.points)
        path = " ".join(f"{x(px):.1f},{y(py):.1f}" for px, py, _, _ in pts)
        out.append(f'<polyline class="line c{i}" points="{path}"/>')
        for px, py, lo, hi in pts:
            if lo is not None and hi is not None:
                out.append(
                    f'<line class="ci c{i}" x1="{x(px):.1f}" y1="{y(lo):.1f}" '
                    f'x2="{x(px):.1f}" y2="{y(hi):.1f}"/>'
                )
            out.append(f'<circle class="f{i} ring" cx="{x(px):.1f}" cy="{y(py):.1f}" r="4"/>')
        if labelled:
            end_x, end_y = pts[-1][0], pts[-1][1]
            out.append(
                f'<text class="label" x="{x(end_x) + 10:.1f}" y="{y(end_y) + 4:.1f}">'
                f"{_esc(s.label)}</text>"
            )
    out.append("</svg>")
    return "\n".join(out) + "\n"


COMPONENT_NAMES = {"model:task": "model x task", "residual": "residual (logistic)"}


def e1_shares() -> tuple[str, list[Path]] | None:
    """Each variance component's share of the latent total, with its interval (E1)."""
    path = REPO / "artifacts" / "e1" / "variance_components.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    comps = sorted(data["components"], key=lambda c: -float(c["share"]))
    rows = []
    for c in comps:
        lo, hi = c.get("share_lo"), c.get("share_hi")
        note = f"{c['share']:.1%}" + ("" if lo is None or hi is None else f" [{lo:.1%}, {hi:.1%}]")
        rows.append(
            Row(
                label=COMPONENT_NAMES.get(str(c["name"]), str(c["name"])),
                estimate=float(c["share"]),
                lo=None if lo is None else float(lo),
                hi=None if hi is None else float(hi),
                note=note,
            )
        )
    top = max([r.hi if r.hi is not None else r.estimate for r in rows] + [0.05])
    domain_hi = nice_ticks(0.0, top)[-1]
    domain_hi = domain_hi if domain_hi >= top else domain_hi + nice_ticks(0.0, top)[1]
    level = round(100 * float(data["settings"]["level"]))
    svg = interval_chart(
        rows,
        title=f"E1: share of latent-scale variance in task success, with {level}% intervals",
        subtitle=(
            f"Logit GLMM on {data['n_runs']} local-arm runs (synthetic mock tasks); "
            f"analysis plan {data['prereg']}."
        ),
        axis_label="share of the total latent variance",
        domain=(0.0, domain_hi),
        tick_format=lambda v: f"{v:.0%}",
        row_height=22,
        label_width=170,
        note_width=150,
    )
    return svg, [path]


def e1_mde() -> tuple[str, list[Path]] | None:
    """Minimum detectable difference of a paired comparison by design (E1)."""
    path = REPO / "artifacts" / "e1" / "mde.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    by_runs: dict[int, list[tuple[float, float, float | None, float | None]]] = {}
    for r in data["rows"]:
        by_runs.setdefault(int(r["runs"]), []).append(
            (float(r["tasks"]), float(r["mde"]), r.get("lo"), r.get("hi"))
        )
    series = [
        Series(f"{k} run{'s' if k > 1 else ''} per task", pts) for k, pts in sorted(by_runs.items())
    ][:3]
    tasks = sorted({float(r["tasks"]) for r in data["rows"]})
    top = max(float(r["hi"] if r.get("hi") is not None else r["mde"]) for r in data["rows"])
    ticks = nice_ticks(0.0, top)
    domain_hi = ticks[-1] if ticks[-1] >= top else ticks[-1] + ticks[1]
    svg = line_chart(
        series,
        title="E1: smallest success-rate gap a paired comparison detects (power 0.8, 5% level)",
        subtitle=(
            "Implied by the E1 fit at the reference configuration (synthetic mock tasks); "
            f"analysis plan {data['prereg']}."
        ),
        x_label="tasks in the benchmark (log scale)",
        y_label="minimum detectable difference",
        x_ticks=tasks,
        y_domain=(0.0, domain_hi),
        y_format=lambda v: f"{v:.2f}",
    )
    return svg, [path]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def e2_scores(board: str) -> tuple[str, list[Path]] | None:
    """Every entry's score with its 95% task-clustered interval, in rank order."""
    path = REPO / "artifacts" / "e2" / board / "results.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    primary = data["sets"]["primary"]
    rows = [
        Row(
            label=short_label(str(e["entry"])),
            estimate=float(e["score"]),
            lo=float(e["lo"]),
            hi=float(e["hi"]),
            note=f"{e['score']:.3f} [{e['lo']:.3f}, {e['hi']:.3f}]",
        )
        for e in primary["entries"]
    ]
    level = round(100 * float(primary["level"]))
    tasks = {int(e["n_tasks"]) for e in primary["entries"]}
    task_text = f"{min(tasks)} tasks" if len(tasks) == 1 else f"{min(tasks)} to {max(tasks)} tasks"
    svg = interval_chart(
        rows,
        title=f"{BOARDS.get(board, board)}: score of each entry with its {level}% interval",
        subtitle=(
            f"{len(rows)} entries, {task_text}; task-clustered t intervals; "
            f"analysis plan {data['prereg']}."
        ),
        axis_label="success rate (mean over tasks of the per-task mean)",
        domain=(0.0, 1.0),
    )
    return svg, [path]


def _signed(value: float) -> str:
    """A difference to 3 decimals with its sign, and 0.000 for one that rounds to zero."""
    return "0.000" if abs(value) < 0.0005 else f"{value:+.3f}"


def e2_gaps(board: str) -> tuple[str, list[Path]] | None:
    """Every adjacent pair's gap with its 95% paired, task-clustered interval, in rank order.

    A pair is separated (prereg 0001, question 1) when its interval excludes zero; the zero
    line is the plot's reference.
    """
    path = REPO / "artifacts" / "e2" / board / "results.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    primary = data["sets"]["primary"]
    pairs = (primary.get("q1") or {}).get("pairs") or []
    if not pairs:  # results without adjacent pairs have no gap figure
        return None
    rows = [
        Row(
            label=f"{i + 1} vs {i + 2}: {short_label(str(p['higher']))}",
            estimate=float(p["difference"]),
            lo=float(p["lo"]),
            hi=float(p["hi"]),
            note=_signed(float(p["difference"])) + (" separated" if p["supported"] else ""),
        )
        for i, p in enumerate(pairs)
    ]
    level = round(100 * float(primary["level"]))
    reach = max(max(abs(r.lo or 0.0), abs(r.hi or 0.0)) for r in rows)
    bound = math.ceil(reach * 20) / 20  # the next multiple of 0.05
    separated = sum(1 for p in pairs if p["supported"])
    svg = interval_chart(
        rows,
        title=f"{BOARDS.get(board, board)}: each adjacent gap with its {level}% paired interval",
        subtitle=(
            f"{separated} of {len(rows)} intervals exclude zero; paired on shared tasks, "
            f"clustered by task; analysis plan {data['prereg']}."
        ),
        axis_label="difference in success rate, higher-ranked minus lower-ranked entry",
        domain=(-bound, bound),
        tick_format=lambda v: f"{v:+.2f}" if v else "0",
    )
    return svg, [path]


FIGURES_BUILDERS: dict[str, Callable[[], tuple[str, list[Path]] | None]] = {
    "e1-shares.svg": e1_shares,
    "e1-mde.svg": e1_mde,
    "e2-tb2-scores.svg": lambda: e2_scores("tb2"),
    "e2-epoch-swebench-scores.svg": lambda: e2_scores("epoch-swebench"),
    "e2-tb2-gaps.svg": lambda: e2_gaps("tb2"),
    "e2-epoch-swebench-gaps.svg": lambda: e2_gaps("epoch-swebench"),
}


def build() -> tuple[dict[str, str], dict[str, Any]]:
    """(figure name -> SVG text, manifest) for every figure whose artifacts exist."""
    figures: dict[str, str] = {}
    manifest: dict[str, Any] = {"generator": "python -m noisefloor.report", "figures": {}}
    skipped = []
    for name, builder in FIGURES_BUILDERS.items():
        made = builder()
        if made is None:
            skipped.append(name)
            continue
        svg, sources = made
        figures[name] = svg
        manifest["figures"][name] = {
            "sha256": hashlib.sha256(svg.encode()).hexdigest(),
            "sources": {p.relative_to(REPO).as_posix(): _sha(p) for p in sources},
        }
    manifest["skipped"] = skipped
    return figures, manifest


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m noisefloor.report [--check]``."""
    parser = argparse.ArgumentParser(description="figures from banked artifacts")
    parser.add_argument("--check", action="store_true", help="fail if a figure is stale")
    args = parser.parse_args(argv)
    figures, manifest = build()
    text = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    if args.check:
        stale = [
            name
            for name, svg in figures.items()
            if not (FIGURES / name).exists() or (FIGURES / name).read_text() != svg
        ]
        manifest_path = FIGURES / "manifest.json"
        if not manifest_path.exists() or manifest_path.read_text() != text:
            stale.append("manifest.json")
        print("report: figures current" if not stale else f"report: STALE {', '.join(stale)}")
        return 1 if stale else 0
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, svg in figures.items():
        (FIGURES / name).write_text(svg)
    (FIGURES / "manifest.json").write_text(text)
    print(f"report: wrote {len(figures)} figures; skipped {manifest['skipped'] or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
