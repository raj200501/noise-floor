from pathlib import Path

import pytest

from noisefloor import demo
from noisefloor.synthetic import PlantedSpec, simulate

GOLDEN = Path(__file__).parent / "golden" / "demo.txt"


def test_demo_matches_golden(capsys: pytest.CaptureFixture[str]) -> None:
    assert demo.main([]) == 0
    assert capsys.readouterr().out == GOLDEN.read_text()


def test_fixture_is_reproducible_from_spec() -> None:
    regenerated = simulate(demo.SPEC, demo.SEED)
    committed = demo.load_fixture().frame
    assert len(regenerated) == len(committed)
    assert regenerated["outcome"].tolist() == committed["outcome"].tolist()
    assert regenerated["synthetic"].all()


def test_write_fixture_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(demo, "FIXTURE_DIR", tmp_path)
    monkeypatch.setattr(demo, "FIXTURE", tmp_path / "synthetic_results.csv")
    monkeypatch.setattr(demo, "MANIFEST", tmp_path / "MANIFEST.json")
    assert demo.main(["--write-fixture"]) == 0
    assert demo.load_fixture().models == list(demo.SPEC.models)
    (tmp_path / "synthetic_results.csv").write_text("tampered\n")
    with pytest.raises(SystemExit, match="hash mismatch"):
        demo.load_fixture()


def test_simulate_factors_and_validation() -> None:
    spec = PlantedSpec(
        models={"m": 0.0}, n_tasks=2, runs=2, factors={"timeout": {"short": -1.0, "long": 0.5}}
    )
    frame = simulate(spec, seed=1)
    assert len(frame) == 2 * 2 * 2 and set(frame["timeout"]) == {"short", "long"}
    assert frame["run"].is_unique is False  # run labels repeat across tasks
    assert frame.groupby(["task", "run"]).size().max() == 1
    for bad in (
        {"models": {}, "n_tasks": 1},
        {"models": {"m": 0.0}, "n_tasks": 0},
        {"models": {"m": 0.0}, "n_tasks": 1, "task_sd": -1.0},
        {"models": {"m": 0.0}, "n_tasks": 1, "factors": {"f": {}}},
    ):
        with pytest.raises(ValueError):
            PlantedSpec(**bad)  # type: ignore[arg-type]  # deliberately partial kwargs
