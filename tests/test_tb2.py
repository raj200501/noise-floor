import json
from pathlib import Path
from typing import Any

import pytest

from noisefloor.reanalysis import tb2

FIXTURE = Path(__file__).parent / "fixtures" / "tb2" / "jobs"


def test_parse_job_status_mapping_and_aliases() -> None:
    result = {
        "stats": {
            "evals": {
                "agent__model__terminal-bench": {
                    "reward_stats": {
                        "reward": {
                            "1.0": ["a__1", "b__2"],
                            "0.0": ["c__3", "install-windows-3-11__4"],
                        }
                    },
                    "exception_stats": {
                        "AgentTimeoutError": ["b__2"],
                        "DaytonaError": ["d__5"],
                        "RuntimeError": ["c__3"],
                    },
                }
            }
        }
    }
    rows = {r["trial"]: r for r in tb2.parse_job(result, "sub", "job")}
    assert rows["a__1"]["status"] == "ok" and rows["a__1"]["outcome"] == 1.0
    assert rows["b__2"]["status"] == "timeout" and rows["b__2"]["outcome"] == 1.0
    assert rows["c__3"]["status"] == "infra_error" and rows["c__3"]["exception"] == "RuntimeError"
    assert rows["d__5"]["reward"] is None and rows["d__5"]["outcome"] == 0.0
    assert rows["install-windows-3-11__4"]["task"] == "install-windows-3.11"
    assert tb2.parse_job({}, "s", "j") == []


def test_load_fixture_trials_and_totals() -> None:
    trials = tb2.load_trials(FIXTURE)
    assert list(trials.columns) == tb2.COLUMNS
    assert trials["submission"].nunique() == 2
    assert set(trials["status"]) >= {"ok", "timeout"}
    assert not trials.duplicated(["submission", "trial"]).any()
    totals = tb2.load_job_totals(FIXTURE)
    assert (totals["n_unlisted"] == 2).all()
    assert (totals["n_listed"] == trials.groupby("submission").size().sort_index().values).all()


def test_table_round_trip_is_deterministic(tmp_path: Path) -> None:
    trials = tb2.load_trials(FIXTURE)
    h1 = tb2.write_table(trials, tmp_path / "a.csv.gz")
    h2 = tb2.write_table(trials, tmp_path / "b.csv.gz")
    assert h1 == h2
    back = tb2.read_table(tmp_path / "a.csv.gz")
    assert back["trial"].tolist() == trials["trial"].tolist()
    assert back["exception"].tolist() == trials["exception"].tolist()
    import gzip

    assert tb2.CANARY in gzip.decompress((tmp_path / "a.csv.gz").read_bytes()).decode()


def test_fetch_walks_tree_with_pinned_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tb2.ROOT
    trees: dict[str, list[dict[str, Any]]] = {
        root: [
            {"type": "directory", "path": f"{root}/S1"},
            {"type": "file", "path": f"{root}/x"},
            {"type": "directory", "path": f"{root}/S2"},
        ],
        f"{root}/S1": [{"type": "directory", "path": f"{root}/S1/group"}],
        f"{root}/S1/group": [{"type": "directory", "path": f"{root}/S1/group/job1"}],
    }
    result = json.dumps({"stats": {"evals": {}}, "n_total_trials": 0}).encode()
    seen: list[str] = []

    def fake_tree(path: str, revision: str, pause: float) -> list[dict[str, Any]]:
        assert revision == tb2.REVISION
        return trees.get(path, [])

    def fake_get(url: str, retries: int = 4) -> tuple[int, bytes]:
        seen.append(url)
        if url.endswith("job1/result.json"):
            return 200, result
        if url.endswith("job1/config.json"):
            return 200, b"{}"
        return 404, b""

    monkeypatch.setattr(tb2, "_tree", fake_tree)
    monkeypatch.setattr(tb2, "_get", fake_get)
    manifest = tb2.fetch(tmp_path, submissions=["S1"])
    assert manifest["revision"] == tb2.REVISION and manifest["license"] == "Apache-2.0"
    keys = sorted(manifest["files"])
    assert keys == [f"{root}/S1/group/job1/config.json", f"{root}/S1/group/job1/result.json"]
    assert (tmp_path / "S1" / "group__job1.result.json").read_bytes() == result
    assert all(f"/resolve/{tb2.REVISION}/" in u for u in seen)


def test_get_and_tree_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    calls = {"n": 0}

    def boom(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        raise urllib.error.HTTPError("u", 404, "nf", None, None)  # type: ignore[arg-type]  # minimal error

    monkeypatch.setattr(tb2.urllib.request, "urlopen", boom)
    assert tb2._get("https://example.invalid") == (404, b"")
    monkeypatch.setattr(tb2, "_get", lambda url, retries=4: (500, b""))
    with pytest.raises(RuntimeError, match="tree listing failed"):
        tb2._tree("x", tb2.REVISION, 0.0)
    monkeypatch.setattr(tb2, "_get", lambda url, retries=4: (200, b'{"not": "a list"}'))
    assert tb2._tree("x", tb2.REVISION, 0.0) == []


def test_get_success_and_rate_limit_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    class Resp:
        status = 200

        def __enter__(self) -> "Resp":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def read(self) -> bytes:
            return b"ok"

    replies: list[object] = [urllib.error.HTTPError("u", 429, "slow", None, None), Resp()]  # type: ignore[arg-type]  # minimal error

    def fake_urlopen(*args: object, **kwargs: object) -> object:
        item = replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(tb2.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(tb2.time, "sleep", lambda s: None)
    assert tb2._get("https://example.invalid") == (200, b"ok")
    monkeypatch.setattr(
        tb2.urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(urllib.error.HTTPError("u", 429, "x", None, None)),  # type: ignore[arg-type]  # minimal error
    )
    assert tb2._get("https://example.invalid", retries=2) == (429, b"")
