"""Exploratory: the noise floor of each reanalyzed leaderboard's own design (not preregistered).

For every adjacent pair of a board's primary analysis (``artifacts/e2/<board>/results.json``,
question 1), the banked paired, task-clustered t interval gives the standard error of the gap:
``se = (hi - lo) / (2 t)``, with ``t`` the two-sided 0.975 quantile of Student's t on the pair's
shared tasks minus one degrees of freedom. The smallest true gap that the board's own design (its
tasks and runs per task) detects with power 0.8 at two-sided alpha 0.05 is then
``MDE = (z_0.975 + z_0.8) * se``, the normal approximation of :func:`evalstats.mde`.

Writes ``artifacts/e2/noise_floor.json``: per board, each adjacent pair's gap and MDE, the median
MDE and the median absolute gap over adjacent pairs, and how many pairs have a gap at least as
large as their MDE. Derived from banked results only; no new data. Usage:
``python tools/noise_floor.py`` (``--check`` fails if the banked file differs).
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

from scipy import stats

REPO = Path(__file__).resolve().parent.parent
BOARDS = ("tb2", "epoch-swebench")
OUT = REPO / "artifacts" / "e2" / "noise_floor.json"
ALPHA, POWER = 0.05, 0.8


def board_floor(results: dict[str, Any]) -> dict[str, Any]:
    """The noise-floor summary of one board's primary analysis set."""
    level = float(results["sets"]["primary"]["level"])
    z = float(stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(POWER))
    rows = []
    for pair in results["sets"]["primary"]["q1"]["pairs"]:
        n = int(pair["n_tasks"])
        t = float(stats.t.ppf(1 - (1 - level) / 2, n - 1))
        se = (float(pair["hi"]) - float(pair["lo"])) / (2 * t)
        gap = float(pair["difference"])
        rows.append(
            {
                "higher": pair["higher"],
                "lower": pair["lower"],
                "n_tasks": n,
                "gap": gap,
                "se": se,
                "mde": z * se,
                "gap_at_least_mde": abs(gap) >= z * se,
            }
        )
    return {
        "n_pairs": len(rows),
        "median_mde": statistics.median(r["mde"] for r in rows),
        "median_abs_gap": statistics.median(abs(r["gap"]) for r in rows),
        "n_gap_at_least_mde": sum(1 for r in rows if r["gap_at_least_mde"]),
        "pairs": rows,
    }


def build() -> str:
    """The artifact's text, from the banked E2 results."""
    out: dict[str, Any] = {
        "label": (
            "exploratory, not preregistered: the gap each board's own design detects with power "
            "0.8 at two-sided alpha 0.05 (normal approximation), from the banked paired intervals"
        ),
        "generator": "tools/noise_floor.py",
        "alpha": ALPHA,
        "power": POWER,
        "boards": {},
    }
    for board in BOARDS:
        source = REPO / "artifacts" / "e2" / board / "results.json"
        out["boards"][board] = {
            "source": source.relative_to(REPO).as_posix(),
            **board_floor(json.loads(source.read_text())),
        }
    return json.dumps(out, indent=1, sort_keys=True) + "\n"


def main(argv: list[str]) -> int:
    """Write the artifact, or with ``--check`` compare it with a fresh build."""
    text = build()
    if "--check" in argv:
        same = OUT.exists() and OUT.read_text() == text
        print(f"NOISE-FLOOR: {'PASS' if same else 'FAIL'} {OUT.relative_to(REPO)}")
        return 0 if same else 1
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
