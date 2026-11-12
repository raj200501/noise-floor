"""The banked artifacts behind reported numbers equal what their committed code makes of them.

Run on the repository's own banked files (not a sandbox), so every `make test` and G1 compares
them (rule 1; DECISIONS.md D-040).
"""

import json
from pathlib import Path
from typing import Any

from noisefloor.reanalysis import e2, report

REPO = Path(__file__).resolve().parent.parent


def _load(path: str) -> Any:
    return json.loads((REPO / path).read_text())


def test_e3_bounds_are_each_boards_policy_question() -> None:
    """Every public-board entry of artifacts/e3/bounds.json is its board's results.json q3."""
    bounds = _load("artifacts/e3/bounds.json")
    entries = bounds["entries"]
    assert {e["source"] for e in entries} == set(e2.BOARDS)
    for entry in entries:
        results = _load(f"artifacts/e2/{entry['source']}/results.json")
        q3 = results["sets"][entry["analysis_set"]]["q3"]
        assert entry == {"source": entry["source"], "analysis_set": entry["analysis_set"], **q3}


def test_leaderboard_reports_render_from_their_artifacts() -> None:
    """docs/leaderboards/<board>.md is exactly what report.render makes of the banked files."""
    for board in e2.BOARDS:
        results = _load(f"artifacts/e2/{board}/results.json")
        manifest = _load(f"artifacts/e2/{board}/manifest.json")
        banked = (REPO / "docs" / "leaderboards" / f"{board}.md").read_text()
        assert report.render(board, results, manifest) == banked
