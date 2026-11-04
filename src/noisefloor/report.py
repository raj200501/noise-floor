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
  --grid: #e1e0d9; --axis: #c3c2b7; --s1: #2a78d6; }
@media (prefers-color-scheme: dark) {
  svg { --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --s1: #3987e5; }
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


FIGURES_BUILDERS: dict[str, Callable[[], tuple[str, list[Path]] | None]] = {
    "e2-tb2-scores.svg": lambda: e2_scores("tb2"),
    "e2-epoch-swebench-scores.svg": lambda: e2_scores("epoch-swebench"),
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
