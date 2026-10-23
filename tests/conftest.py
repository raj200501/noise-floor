import pytest

from evalstats import EvalResults
from synth import make_results


@pytest.fixture
def results() -> EvalResults:
    return make_results()
