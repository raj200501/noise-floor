import json
import shutil
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


ADAPTERS = Path(__file__).resolve().parent / "fixtures" / "adapters"
HARBOR_JOB = ADAPTERS / "harbor" / "jobs" / "2026-10-23__02-00-00"


def test_report_text_and_json(csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["report", str(csv_path), "--title", "synthetic", "--n-boot", "200"]) == 0
    text = capsys.readouterr().out
    assert text.startswith("# Confidence report: synthetic")
    assert "| Rank | Entry | Score | Rank interval | Could rank first |" in text
    assert "adjacent orderings are supported" in text and "## Ambiguous runs" not in text
    assert cli.main(["report", str(csv_path), "--n-boot", "200", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {i["model"] for i in payload["ranks"]["intervals"]} == {"a", "b"}
    assert payload["policy_bounds"] is None


def test_report_reads_harbor_jobs_and_bounds_ambiguous_runs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["report", str(HARBOR_JOB), "--from", "harbor"]) == 0
    text = capsys.readouterr().out
    assert "mock/model-a" in text and "mock/model-b" in text
    assert "## Ambiguous runs" in text
    assert "0 of 1 adjacent orderings hold under every way before sampling error" in text
    # Three tasks: the task bootstrap separates a pair that the paired interval does not.
    assert "Caution: the rank intervals order 1 adjacent pair whose paired interval" in text
    args = ["report", str(HARBOR_JOB), "--from", "harbor", "--level", "0.975"]
    assert cli.main(args) == 0
    assert "97.5% simultaneous rank intervals" in capsys.readouterr().out
    assert cli.main([*args, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ranks"]["level"] == payload["policy_bounds"]["level"] == 0.975
    assert len(payload["policy_bounds"]["models"]) == 2 and payload["synthetic"] is False


def _relabel_agent(job: Path, trials: tuple[str, ...], agent: str) -> None:
    for trial in trials:
        path = job / trial / "result.json"
        data = json.loads(path.read_text())
        data["config"]["agent"]["name"] = agent
        if isinstance(data.get("agent_info"), dict):
            data["agent_info"]["name"] = agent
        path.write_text(json.dumps(data))


def test_harbor_entries_that_pool_two_agents_warn(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = tmp_path / "job"
    shutil.copytree(HARBOR_JOB, job)
    # A second agent runs mock/model-a on every task: one entry now pools two agents.
    _relabel_agent(job, ("mock-build__mock005", "mock-hello__mock001", "mock-regex__mock003"), "b2")
    assert cli.main(["report", str(job), "--from", "harbor", "--n-boot", "200"]) == 0
    err = capsys.readouterr().err
    assert "warning: entry 'mock/model-a' pools the trials of agents b2, terminus-2" in err
    assert "pass --model-label agent_model" in err
    args = ["report", str(job), "--from", "harbor", "--model-label", "agent_model", "--json"]
    assert cli.main([*args, "--n-boot", "200"]) == 0
    out = capsys.readouterr()
    models = {i["model"] for i in json.loads(out.out)["ranks"]["intervals"]}
    assert models == {"b2__mock/model-a", "terminus-2__mock/model-a", "terminus-2__mock/model-b"}
    assert "pools" not in out.err


def test_compare_reads_inspect_logs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = next((ADAPTERS / "inspect" / "json").glob("*.json"))
    for i, model in enumerate(("mockllm/model-a", "mockllm/model-b")):
        log = json.loads(source.read_text())
        log["eval"]["model"] = model
        log["eval"]["eval_id"] = f"{log['eval']['eval_id']}-{i}"
        if i == 1:  # model-b answers the first sample correctly; model-a does not
            log["samples"][0]["scores"]["match"]["value"] = "C"
        (tmp_path / f"log-{i}.json").write_text(json.dumps(log))
    args = ["compare", str(tmp_path), "--from", "inspect"]
    assert cli.main([*args, "--a", "mockllm/model-a", "--b", "mockllm/model-b", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["difference"]["estimate"] < 0
    assert cli.main(["report", str(tmp_path / "log-0.json"), "--from", "inspect"]) == 1
    assert "need at least 2 models" in capsys.readouterr().err  # one entry: nothing to rank


def test_inspect_options_reach_the_reader(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = next((ADAPTERS / "inspect" / "json").glob("*.json"))
    # Two logs of model-a on one task (a repeated eval) and one of model-b.
    for i, model in enumerate(("mockllm/model-a", "mockllm/model-a", "mockllm/model-b")):
        log = json.loads(source.read_text())
        log["eval"]["model"] = model
        log["eval"]["eval_id"] = f"{log['eval']['eval_id']}-{i}"
        (tmp_path / f"log-{i}.json").write_text(json.dumps(log))
    assert cli.main(["report", str(tmp_path), "--from", "inspect"]) == 1
    assert "pass --run-key eval_epoch" in capsys.readouterr().err  # the CLI's flag, not a kwarg
    args = ["report", str(tmp_path), "--from", "inspect", "--run-key", "eval_epoch", "--json"]
    assert cli.main([*args, "--n-boot", "200"]) == 0
    assert len(json.loads(capsys.readouterr().out)["ranks"]["intervals"]) == 2
    assert cli.main([*args, "--scorer", "no-such-scorer"]) == 1
    assert "no-such-scorer" in capsys.readouterr().err
    # The fixture directory holds one eval in several formats: copies, not repeats.
    assert (
        cli.main(
            ["rank", str(ADAPTERS / "inspect"), "--from", "inspect", "--run-key", "eval_epoch"]
        )
        == 1
    )
    assert "copies of one eval" in capsys.readouterr().err


def test_report_renders_marginal_ranks_and_dropped_tasks() -> None:
    from evalstats import confidence_report, rank_confidence, render_report

    results = make_results(n_tasks=20, runs=2, models=("a", "b", "c"), p=(0.6, 0.5, 0.4))
    frame = results.frame
    keep = ~((frame["model"] == "c") & (frame["task"] == frame["task"].iloc[0]))
    partial = type(results).from_frame(frame[keep].reset_index(drop=True))
    text = confidence_report(partial, title="t", n_boot=200)
    assert "(1 other task left out)" in text
    marginal = rank_confidence(results, simultaneous=False, n_boot=200, seed=0)
    assert "95% marginal rank intervals" in render_report(marginal, title="m")


def test_report_sections_share_tasks_order_and_labels(tmp_path: Path) -> None:
    from evalstats import confidence_report, read_results
    from evalstats.report import report_inputs

    frame = make_results(n_tasks=12, runs=2, models=("a|x", "b", "c"), p=(0.7, 0.5, 0.3)).frame
    frame["status"] = "ok"
    frame.loc[(frame["model"] == "b") & (frame["task"] == "t001"), "status"] = "timeout"
    frame = frame[~((frame["model"] == "c") & frame["task"].isin(["t010", "t011"]))]
    frame["synthetic"] = True
    path = tmp_path / "runs.csv"
    frame.to_csv(path, index=False)
    results = read_results(path)
    ranks, bounds = report_inputs(results, n_boot=200)
    assert bounds is not None
    order = [r.model for r in sorted(ranks.intervals, key=lambda r: r.rank)]
    assert [m.model for m in bounds.models] == order  # the bounds follow the ranking
    for r, m in zip(sorted(ranks.intervals, key=lambda r: r.rank), bounds.models, strict=True):
        assert m.scores["as_scored"] == pytest.approx(r.score)  # on the same tasks
    text = confidence_report(results, n_boot=200)
    assert "(2 other tasks left out)" in text and "a\\|x" in text
    assert "**Synthetic data.**" in text and "also with it" in text
    frame["status"] = "ok"
    frame.to_csv(path, index=False)
    text = confidence_report(read_results(path), n_boot=200)
    assert "No run has an ambiguous status" in text and "hold under every way" not in text


def test_json_is_strict_and_flags_are_checked(
    tmp_path: Path, csv_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frame = make_results(n_tasks=10, runs=2, models=("a", "b", "c"), p=(0.6, 0.5, 0.4)).frame
    frame["status"] = "ok"
    frame.loc[frame["model"] == "c", "status"] = "infra_error"  # no unambiguous run of c
    path = tmp_path / "runs.csv"
    frame.to_csv(path, index=False)
    assert cli.main(["report", str(path), "--n-boot", "200", "--json"]) == 0
    text = capsys.readouterr().out
    assert "NaN" not in text  # undefined scores are null, not a bare NaN
    payload = json.loads(text)
    by_model = {m["model"]: m for m in payload["policy_bounds"]["models"]}
    assert by_model["c"]["scores"]["exclude"] is None
    assert cli.main(["rank", str(csv_path), "--level", "0.9", "--n-boot", "200"]) == 0
    assert "90% marginal interval" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["report", str(csv_path), "--n-boot", "0"])
    assert "must be at least 1" in capsys.readouterr().err
    args = ["power", "--from", "inspect", "--between-var", "0.01", "--within-var", "0.2"]
    assert cli.main(args) == 1
    assert "--from needs an input path" in capsys.readouterr().err
