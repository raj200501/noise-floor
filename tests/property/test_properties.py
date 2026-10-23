"""Property-based tests (hypothesis) for evalstats invariants."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evalstats import (
    EvalResults,
    PairedDesign,
    cluster_robust_mean,
    mde,
    paired_difference,
    t_interval,
    tasks_needed,
)
from evalstats.intervals import mean_se

pytestmark = pytest.mark.property

unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
vectors = st.lists(unit, min_size=2, max_size=40)


@st.composite
def paired_results(draw: st.DrawFn) -> EvalResults:
    n_tasks = draw(st.integers(min_value=2, max_value=15))
    runs = draw(st.integers(min_value=1, max_value=3))
    rows = []
    for model in ("a", "b"):
        for t in range(n_tasks):
            for r in range(runs):
                rows.append(
                    {
                        "model": model,
                        "task": f"t{t}",
                        "run": r,
                        "outcome": float(draw(st.booleans())),
                    }
                )
    return EvalResults.from_frame(pd.DataFrame(rows))


@settings(max_examples=60, deadline=None)
@given(paired_results())
def test_paired_difference_is_antisymmetric(res: EvalResults) -> None:
    ab = paired_difference(res, "a", "b")
    ba = paired_difference(res, "b", "a")
    assert ab.difference.estimate == pytest.approx(-ba.difference.estimate)
    assert ab.difference.lo == pytest.approx(-ba.difference.hi)
    assert ab.p_value == pytest.approx(ba.p_value, nan_ok=True)
    assert ab.supported == ba.supported


@settings(max_examples=60, deadline=None)
@given(paired_results())
def test_row_order_and_task_labels_do_not_matter(res: EvalResults) -> None:
    frame = res.frame.sample(frac=1.0, random_state=np.random.RandomState(0))
    relabeled = frame.assign(task="x" + frame["task"].str[::-1])
    other = EvalResults.from_frame(relabeled)
    a = paired_difference(res, "a", "b").difference
    b = paired_difference(other, "a", "b").difference
    assert (a.estimate, a.lo, a.hi) == pytest.approx((b.estimate, b.lo, b.hi), nan_ok=True)


@settings(max_examples=100, deadline=None)
@given(vectors, st.floats(min_value=0.5, max_value=0.98), st.floats(min_value=0.5, max_value=0.98))
def test_t_interval_contains_mean_and_is_monotone_in_level(
    values: list[float], l1: float, l2: float
) -> None:
    lo_level, hi_level = sorted((l1, l2))
    narrow, wide = t_interval(values, lo_level), t_interval(values, hi_level)
    assert narrow.contains(narrow.estimate)
    assert wide.lo <= narrow.lo + 1e-12 and wide.hi >= narrow.hi - 1e-12


@settings(max_examples=100, deadline=None)
@given(vectors)
def test_cluster_robust_singletons_equal_classic(values: list[float]) -> None:
    _, se = cluster_robust_mean(values, np.arange(len(values)))
    assert se == pytest.approx(mean_se(values), abs=1e-12)


@settings(max_examples=100, deadline=None)
@given(vectors, st.floats(min_value=-0.5, max_value=0.5))
def test_t_interval_is_shift_equivariant(values: list[float], shift: float) -> None:
    a = t_interval(values)
    b = t_interval(np.asarray(values) + shift)
    assert b.lo == pytest.approx(a.lo + shift, abs=1e-9)
    assert b.hi == pytest.approx(a.hi + shift, abs=1e-9)


@settings(max_examples=100, deadline=None)
@given(
    st.floats(min_value=0.0, max_value=0.25),
    st.floats(min_value=0.01, max_value=0.5),
    st.integers(min_value=1, max_value=5),
    st.floats(min_value=0.01, max_value=0.3),
)
def test_tasks_needed_is_minimal(between: float, within: float, runs: int, effect: float) -> None:
    design = PairedDesign(between, within)
    n = tasks_needed(effect, design, runs)
    assert mde(design, n, runs) <= effect * (1 + 1e-9)
    if n > 1:
        assert mde(design, n - 1, runs) > effect
