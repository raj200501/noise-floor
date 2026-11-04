import pytest

from evalstats import EvalResults
from synth import make_results


@pytest.fixture
def results() -> EvalResults:
    return make_results()


@pytest.fixture
def unfrozen_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let confirmatory runners use a test fixture preregistration that is not in the chain."""
    from noisefloor import preregs

    def fake(path: object, *args: object, **kwargs: object) -> dict[str, object]:
        return {"file": str(path), "sha256": "0" * 64, "frozen_at": "2000-01-01T00:00:00Z"}

    monkeypatch.setattr(preregs, "require_frozen", fake)
