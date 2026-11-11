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
    # The arms check their run code before each batch; a test runs from a clean,
    # unchanging checkout whatever the working tree holds.
    from noisefloor import e1

    clean = {"src_tree": "t" * 40, "uv_lock": "l" * 40, "dirty": False, "server_sha256": "s" * 64}
    monkeypatch.setattr(e1, "run_code", lambda local: dict(clean))
