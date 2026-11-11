import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from evalstats import EvalResults
from noisefloor.reanalysis import __main__ as cli
from noisefloor.reanalysis import e2, epoch, report, tb2


def synthetic_trials(
    n_subs: int = 4, n_tasks: int = 6, k: int = 3, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic TB2-shaped tables (mock data)."""
    rng = np.random.default_rng(seed)
    subs = [*["Alpha__m", "Beta__m", "Gamma__m", "OB-1_x", "Delta__m"][:n_subs], "Ledger__m"]
    rows, jobs = [], []
    for si, sub in enumerate(subs):
        job = f"job{si}"
        for t in range(n_tasks):
            for r in range(k):
                outcome = float(rng.random() < 0.3 + 0.1 * si)
                status = "ok" if rng.random() > 0.15 else "timeout"
                rows.append(
                    {
                        "submission": sub,
                        "job": job,
                        "trial": f"task{t}__{si}{r}",
                        "task": f"task{t}",
                        "reward": outcome,
                        "outcome": outcome,
                        "exception": "" if status == "ok" else "AgentTimeoutError",
                        "status": status,
                    }
                )
        datasets = (
            "terminal-bench/terminal-bench-2-1@None" if sub == "Ledger__m" else "terminal-bench@2.0"
        )
        jobs.append(
            {
                "submission": sub,
                "job": job,
                "n_total_trials": n_tasks * k + 1,
                "n_listed": n_tasks * k,
                "n_unlisted": 1,
                "datasets": datasets,
                "tb20": e2.is_tb20(datasets),
            }
        )
    return pd.DataFrame(rows, columns=tb2.COLUMNS), pd.DataFrame(jobs)


@pytest.fixture
def sandbox_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(e2, "REPO", tmp_path)
    monkeypatch.setattr(e2, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(e2, "TB2_SNAPSHOT", tmp_path / "data" / "tb2")
    monkeypatch.setattr(e2, "EPOCH_SNAPSHOT", tmp_path / "data" / "epoch")
    monkeypatch.setattr(e2, "N_BOOT", 300)
    trials, jobs = synthetic_trials()
    snap = tmp_path / "data" / "tb2"
    files = {}
    for name, frame in (("trials", trials), ("jobs", jobs)):
        path = snap / f"{name}.csv.gz"
        files[f"data/tb2/{name}.csv.gz"] = tb2.write_table(frame, path)
    man = {
        "board": "tb2",
        "files": files,
        "submissions": [*sorted(set(trials.submission)), "Crux__m"],
        "snapshot": "data/tb2/trials.csv.gz",
        "sha256": files["data/tb2/trials.csv.gz"],
    }
    (tmp_path / "artifacts" / "e2" / "tb2").mkdir(parents=True)
    (tmp_path / "artifacts" / "e2" / "tb2" / "manifest.json").write_text(json.dumps(man))
    rng = np.random.default_rng(1)
    rows = []
    for e in range(4):
        for t in range(30):
            status = "infra_error" if (e == 1 and t == 0) else ("timeout" if t == 3 else "ok")
            rows.append(
                {
                    "entry": f"model{e} [L{e}]",
                    "model": f"m{e}",
                    "log_id": f"L{e}",
                    "task": f"inst{t}",
                    "run": "1",
                    "outcome": float(rng.random() < 0.4 + 0.1 * e),
                    "status": status,
                    "status_reason": "" if status == "ok" else status,
                }
            )
    samples = pd.DataFrame(rows, columns=epoch.COLUMNS)
    esnap = tmp_path / "data" / "epoch"
    sha = epoch.write_table(samples, esnap / "samples.csv.gz")
    (tmp_path / "artifacts" / "e2" / "epoch-swebench").mkdir(parents=True)
    (tmp_path / "artifacts" / "e2" / "epoch-swebench" / "manifest.json").write_text(
        json.dumps(
            {
                "board": "epoch-swebench",
                "files": {"data/epoch/samples.csv.gz": sha},
                "snapshot": "data/epoch/samples.csv.gz",
                "sha256": sha,
            }
        )
    )
    return tmp_path


def test_dataset_helpers() -> None:
    cfg = {"datasets": [{"name": "terminal-bench", "version": "2.0"}, {"path": "/x/tb2"}, "junk"]}
    assert e2.job_datasets(cfg) == "terminal-bench@2.0;path:/x/tb2"
    assert e2.job_datasets({}) == ""
    assert e2.is_tb20("terminal-bench@2.0") and not e2.is_tb20(
        "terminal-bench/terminal-bench-2-1@None"
    )
    assert e2.is_tb20("path:/work/wecode-tb2-4-30-5")


def test_tb2_frame_exclusions() -> None:
    trials, jobs = synthetic_trials()
    frame, excluded = e2.tb2_frame(trials, jobs, [*sorted(set(trials.submission)), "Crux__m"])
    assert set(excluded) == {"OB-1_x", "Ledger__m", "Crux__m"}
    assert "maintainers" in excluded["OB-1_x"] and "2.0" in excluded["Ledger__m"]
    assert "result.json" in excluded["Crux__m"]
    assert set(frame["model"]) == {"Alpha__m", "Beta__m", "Gamma__m"}


def test_analyze_verify_and_report(sandbox_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results = e2.analyze()
    tb = results["tb2"]
    assert tb["excluded_entries"].keys() == {"OB-1_x", "Ledger__m", "Crux__m"}
    primary = tb["sets"]["primary"]
    assert primary["n_entries"] == 3 and primary["q1"]["n_pairs"] == 2
    assert 0 <= primary["q1"]["n_supported"] <= 2
    assert primary["q1"]["label"] in ("mostly resolved", "mostly unresolved")
    scores = [e["score"] for e in primary["entries"]]
    assert scores == sorted(scores, reverse=True)
    assert all(e["pooled_score"] < e["score"] + 1e-12 for e in primary["entries"])
    assert len(primary["q2"]["intervals"]) == 3 and primary["q3"]["n_pairs"] == 2
    ep = results["epoch-swebench"]["sets"]["primary"]
    assert ep["n_entries"] == 4 and ep["n_runs"] == 4 * 30 - 1
    bounds = json.loads((sandbox_repo / "artifacts" / "e3" / "bounds.json").read_text())
    assert [x["source"] for x in bounds["entries"]] == ["tb2", "epoch-swebench"]
    assert e2.verify() == []
    path = sandbox_repo / "artifacts" / "e2" / "tb2" / "results.json"
    path.write_text(path.read_text().replace('"n_entries": 3', '"n_entries": 4'))
    assert any("differ" in p for p in e2.verify(("tb2",)))
    snap = sandbox_repo / "data" / "tb2" / "jobs.csv.gz"
    snap.write_bytes(b"tampered")
    with pytest.raises(OSError):
        e2.verify(("tb2",))
    monkeypatch.setattr(report, "REPO", sandbox_repo)
    md = report.render(
        "tb2",
        results["tb2"],
        {
            "source_url": "u",
            "revision": "r",
            "license": "Apache-2.0",
            "snapshot": "s",
            "sha256": "h",
            "retrieved": "d",
        },
    )
    assert "## Adjacent pairs" in md and "Excluded entries" in md and "—" not in md
    (sandbox_repo / "artifacts" / "e2" / "epoch-swebench" / "results.json").write_text(
        json.dumps(results["epoch-swebench"])
    )
    paths = report.write_reports(("epoch-swebench",))
    assert paths[0].name == "epoch-swebench.md" and "Epoch" in paths[0].read_text()


def test_board_analysis_with_nan_free_json(sandbox_repo: Path) -> None:
    trials, jobs = synthetic_trials(n_subs=2)
    frame, _ = e2.tb2_frame(trials, jobs)
    res = EvalResults.from_frame(frame[["model", "task", "run", "outcome", "status"]])
    out = e2.board_analysis(res, res, pooled={}, analysis_set="x")
    json.dumps(out, allow_nan=False)
    assert out["entries"][0]["pooled_score"] is None


def test_clean_and_rel(sandbox_repo: Path) -> None:
    assert e2._clean({"a": [float("nan"), 1.0, float("inf")], "b": (2,)}) == {
        "a": [None, 1.0, None],
        "b": [2],
    }
    assert e2.rel(sandbox_repo / "x" / "y") == "x/y"


def test_cli(
    sandbox_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    built: list[Any] = []
    monkeypatch.setattr(
        e2, "build_tb2", lambda raw: built.append(raw) or {"snapshot": "s", "sha256": "0" * 64}
    )
    assert cli.main(["build", "tb2", "/raw"]) == 0 and built == [Path("/raw")]
    assert cli.main(["analyze"]) == 0 and "adjacent pairs supported" in capsys.readouterr().out
    assert cli.main(["verify"]) == 0 and "E2-VERIFY: PASS" in capsys.readouterr().out
    assert cli.main(["--verify"]) == 0
    monkeypatch.setattr(report, "REPO", sandbox_repo)
    assert cli.main(["report"]) == 0 and "wrote" in capsys.readouterr().out
    assert cli.main([]) == 2
    (sandbox_repo / "artifacts" / "e2" / "tb2" / "results.json").write_text("{}")
    assert cli.main(["verify", "tb2"]) == 1


def test_builds_from_raw_fetches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    monkeypatch.setattr(e2, "REPO", tmp_path)
    monkeypatch.setattr(e2, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(e2, "TB2_SNAPSHOT", tmp_path / "data" / "tb2")
    monkeypatch.setattr(e2, "EPOCH_SNAPSHOT", tmp_path / "data" / "epoch")
    raw = tmp_path / "raw-tb2"
    shutil.copytree(Path(__file__).parent / "fixtures" / "tb2" / "jobs", raw)
    for result in raw.glob("*/*.result.json"):
        cfg = result.with_name(result.name.replace(".result.json", ".config.json"))
        cfg.write_text(json.dumps({"datasets": [{"name": "terminal-bench", "version": "2.0"}]}))
    (raw / "fetch_manifest.json").write_text(
        json.dumps(
            {"revision": tb2.REVISION, "retrieved": "2026-11-04", "files": {}, "failures": []}
        )
    )
    monkeypatch.setattr(tb2, "list_submissions", lambda revision: ["A", "B"])
    man = e2.build_tb2(raw)
    assert man["submissions"] == ["A", "B"] and man["license"] == "Apache-2.0"
    assert (tmp_path / "data" / "tb2" / "trials.csv.gz").exists()
    trials, jobs = e2.tb2_tables()
    assert jobs["tb20"].all() and len(trials) > 0
    from test_epoch import fake_request_factory, make_archive, make_eval_zip

    rows = [
        {
            "Model version": "mx",
            "Logs": "https://logs.example/ok.eval",
            "id": "L1",
            "mean_score": "0.5",
            "Started at": "2026-06-01",
        }
    ]
    monkeypatch.setattr(
        epoch, "_request", fake_request_factory(make_eval_zip(), make_archive(rows))
    )
    eraw = tmp_path / "raw-epoch"
    manifest = epoch.fetch(eraw)
    (eraw / "fetch_manifest.json").write_text(json.dumps(manifest))
    eman = e2.build_epoch(eraw)
    assert eman["license"] == "CC-BY-4.0" and eman["snapshot"].endswith("samples.csv.gz")
    logs = e2.read_gz_csv(tmp_path / "data" / "epoch" / "logs.csv.gz")
    assert list(logs["entry"]) == ["mx [L1]"] and float(logs["published_mean_score"][0]) == 0.5
