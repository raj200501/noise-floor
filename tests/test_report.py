"""Figures generated from banked artifacts (synthetic rows here)."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from noisefloor import report

NS = {"s": "http://www.w3.org/2000/svg"}


def test_nice_ticks() -> None:
    assert report.nice_ticks(0.0, 1.0) == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert report.nice_ticks(0.0, 0.07) == [0.0, 0.02, 0.04, 0.06]
    assert report.nice_ticks(3.0, 3.0) == [3.0]


def test_short_label() -> None:
    assert report.short_label("gpt-4o [PncXWJm7ZCY3]") == "gpt-4o [PncX]"
    assert report.short_label("vix__claude-opus-4-7") == "vix__claude-opus-4-7"
    assert report.short_label("a [b]") == "a [b]"


def test_interval_chart_is_valid_svg() -> None:
    rows = [
        report.Row("alpha", 0.5, 0.4, 0.6, note="0.500 [0.400, 0.600]"),
        report.Row("a" * 80, 0.2, None, None),
        report.Row("<b&c>", 1.5, -1.0, 2.0),
    ]
    svg = report.interval_chart(
        rows, title="T & t", subtitle="S", axis_label="x", domain=(0.0, 1.0)
    )
    root = ET.fromstring(svg)
    assert root.find("s:title", NS).text == "T & t"
    assert len(root.findall("s:circle", NS)) == 3
    assert len([e for e in root.findall("s:line", NS) if e.get("class") == "ci"]) == 2
    labels = [e.text for e in root.findall("s:text", NS) if e.get("class") == "label"]
    assert labels[0] == "alpha" and labels[1].endswith("…") and labels[2] == "<b&c>"
    xs = [float(c.get("cx")) for c in root.findall("s:circle", NS)]
    assert max(xs) <= 760 - 120 - 12  # values outside the domain are clamped to the plot
    assert "prefers-color-scheme: dark" in svg


def test_build_check_and_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    results = {
        "prereg": "prereg/0001-x.md",
        "sets": {
            "primary": {
                "level": 0.95,
                "entries": [
                    {"entry": "m1", "score": 0.6, "lo": 0.5, "hi": 0.7, "n_tasks": 10},
                    {"entry": "m2 [abcdefgh]", "score": 0.4, "lo": 0.3, "hi": 0.5, "n_tasks": 9},
                ],
            }
        },
    }
    board = tmp_path / "artifacts" / "e2" / "tb2"
    board.mkdir(parents=True)
    (board / "results.json").write_text(json.dumps(results))
    monkeypatch.setattr(report, "REPO", tmp_path)
    monkeypatch.setattr(report, "FIGURES", tmp_path / "docs" / "figures")
    assert report.main(["--check"]) == 1
    assert report.main([]) == 0
    skipped = "skipped ['e1-shares.svg', 'e1-mde.svg', 'e2-epoch-swebench-scores.svg']"
    assert skipped in capsys.readouterr().out
    assert report.main(["--check"]) == 0
    manifest = json.loads((tmp_path / "docs" / "figures" / "manifest.json").read_text())
    assert list(manifest["figures"]) == ["e2-tb2-scores.svg"]
    assert "artifacts/e2/tb2/results.json" in manifest["figures"]["e2-tb2-scores.svg"]["sources"]
    svg = (tmp_path / "docs" / "figures" / "e2-tb2-scores.svg").read_text()
    assert "2 entries, 9 to 10 tasks" in svg and "m2 [abcd]" in svg


def test_banked_figures_are_current() -> None:
    assert report.main(["--check"]) == 0


def write_e1(root: Path) -> None:
    e1 = root / "artifacts" / "e1"
    e1.mkdir(parents=True)
    comps = [
        {"name": "task", "share": 0.31, "share_lo": 0.22, "share_hi": 0.4},
        {"name": "residual", "share": 0.52, "share_lo": 0.45, "share_hi": 0.58},
        {"name": "model:task", "share": 0.08, "share_lo": 0.05, "share_hi": 0.12},
        {"name": "harness", "share": 0.01, "share_lo": None, "share_hi": None},
    ]
    (e1 / "variance_components.json").write_text(
        json.dumps(
            {
                "components": comps,
                "n_runs": 1440,
                "prereg": "prereg/0004-x.md",
                "settings": {"level": 0.95},
            }
        )
    )
    rows = [
        {
            "tasks": t,
            "runs": k,
            "mde": 0.5 / (t * k) ** 0.5,
            "lo": 0.4 / (t * k) ** 0.5,
            "hi": 0.6 / (t * k) ** 0.5,
        }
        for t in (20, 40, 89, 200, 500)
        for k in (1, 3, 5)
    ]
    (e1 / "mde.json").write_text(json.dumps({"rows": rows, "prereg": "prereg/0004-x.md"}))


def test_e1_figures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report, "REPO", tmp_path)
    assert report.e1_shares() is None and report.e1_mde() is None
    write_e1(tmp_path)
    shares, sources = report.e1_shares()
    root = ET.fromstring(shares)
    labels = [e.text for e in root.findall("s:text", NS) if e.get("class") == "label"]
    assert labels == ["residual (logistic)", "task", "model x task", "harness"]
    assert "52.0% [45.0%, 58.0%]" in shares and "1.0%" in shares
    assert sources == [tmp_path / "artifacts" / "e1" / "variance_components.json"]
    mde, _ = report.e1_mde()
    root = ET.fromstring(mde)
    assert len(root.findall("s:polyline", NS)) == 3
    assert len(root.findall("s:circle", NS)) == 15
    assert "5 runs per task" in mde and "1 run per task" in mde
    with pytest.raises(ValueError, match="one to three"):
        report.line_chart(
            [], title="", subtitle="", x_label="", y_label="", x_ticks=[1, 2], y_domain=(0, 1)
        )


def test_line_chart_end_labels() -> None:
    apart = [
        report.Series("a", [(1, 0.9, None, None), (10, 0.8, None, None)]),
        report.Series("b", [(1, 0.2, None, None), (10, 0.1, None, None)]),
    ]
    close = [
        report.Series("a", [(1, 0.9, None, None), (10, 0.11, None, None)]),
        report.Series("b", [(1, 0.2, None, None), (10, 0.1, None, None)]),
    ]
    kw = {
        "title": "t",
        "subtitle": "s",
        "x_label": "x",
        "y_label": "y",
        "x_ticks": [1, 10],
        "y_domain": (0.0, 1.0),
    }

    def labels(svg: str) -> list:
        return [
            e.text for e in ET.fromstring(svg).findall("s:text", NS) if e.get("class") == "label"
        ]

    assert labels(report.line_chart(apart, **kw)) == ["a", "b", "a", "b"]  # legend + ends
    assert labels(report.line_chart(close, **kw)) == ["a", "b"]  # legend only
