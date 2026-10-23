import json
from pathlib import Path

import pytest

from evalstats import cli, paired_difference
from synth import make_results


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "results.csv"
    make_results(n_tasks=40, runs=3).frame.to_csv(path, index=False)
    return path


def test_compare_text_and_json(csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["compare", str(csv_path), "--a", "a", "--b", "b"]) == 0
    text = capsys.readouterr().out
    assert "40 shared tasks" in text and "95% CI" in text and "cluster_t" in text
    assert cli.main(["compare", str(csv_path), "--a", "a", "--b", "b", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    expected = paired_difference(make_results(n_tasks=40, runs=3), "a", "b")
    assert payload["difference"]["estimate"] == pytest.approx(expected.difference.estimate)
    assert payload["supported"] == expected.supported


def test_compare_bootstrap_seeded(csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = [
        "compare",
        str(csv_path),
        "--a",
        "a",
        "--b",
        "b",
        "--method",
        "bca",
        "--n-boot",
        "500",
        "--seed",
        "3",
        "--json",
    ]
    assert cli.main(args) == 0
    first = capsys.readouterr().out
    assert cli.main(args) == 0
    assert capsys.readouterr().out == first and '"bca"' in first


def test_compare_errors(csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["compare", str(csv_path), "--a", "a", "--b", "zzz"]) == 1
    assert "unknown model" in capsys.readouterr().err
    assert cli.main(["compare", "missing.csv", "--a", "a", "--b", "b"]) == 1


def test_power_from_file_and_given(csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["power", str(csv_path), "--a", "a", "--b", "b", "--effect", "0.05"]) == 0
    text = capsys.readouterr().out
    assert "estimated from" in text and "tasks needed for effect 0.05" in text
    args = [
        "power",
        "--between-var",
        "0.02",
        "--within-var",
        "0.4",
        "--tasks",
        "100",
        "--runs",
        "1",
        "4",
        "--json",
    ]
    assert cli.main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [r["runs"] for r in payload["rows"]] == [1, 4]
    assert payload["rows"][1]["mde"] < payload["rows"][0]["mde"]


def test_power_argument_errors(csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["power"]) == 1
    assert cli.main(["power", str(csv_path)]) == 1
    assert "--a and --b" in capsys.readouterr().err


def test_run_column_optional(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    frame = make_results(n_tasks=10, runs=1).frame.drop(columns="run")
    path = tmp_path / "norun.csv"
    frame.to_csv(path, index=False)
    assert cli.main(["compare", str(path), "--a", "a", "--b", "b", "--run-col", ""]) == 0
    assert "10 shared tasks" in capsys.readouterr().out


def test_decompose_text_and_json(csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["decompose", str(csv_path), "--random", "task", "model", "--backend", "statsmodels"]
    assert cli.main(args) == 0
    text = capsys.readouterr().out
    assert "outcome ~ 1 + (1 | task) + (1 | model)" in text and "residual" in text
    assert cli.main([*args, "--json", "--no-intervals"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [r["component"] for r in payload["components"]] == ["task", "model", "residual"]
    assert payload["components"][0]["lo"] is None
    assert cli.main(["decompose", str(csv_path), "--random", "nope"]) == 1


def test_rank_text_and_json(csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["rank", str(csv_path), "--n-boot", "300", "--seed", "1"]) == 0
    text = capsys.readouterr().out
    assert "simultaneous rank intervals" in text and "adjacent pairs" in text
    assert cli.main(["rank", str(csv_path), "--marginal", "--n-boot", "300", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["simultaneous"] is False and len(payload["intervals"]) == 2
