import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from evalstats import EvalResults, SchemaError
from evalstats.bounds import (
    POLICIES,
    imbens_manski_critical,
    imbens_manski_interval,
    policy_bounds,
)


def board(
    statuses: dict[str, list[str]], outcomes: dict[str, list[float]], runs: int = 1
) -> EvalResults:
    rows = []
    for model, stat in statuses.items():
        for i, (s, y) in enumerate(zip(stat, outcomes[model], strict=True)):
            rows.append(
                {
                    "model": model,
                    "task": f"t{i // runs}",
                    "run": i % runs,
                    "outcome": y,
                    "status": s,
                }
            )
    return EvalResults.from_frame(pd.DataFrame(rows))


def test_no_ambiguous_runs_gives_point_bounds() -> None:
    res = board(
        {"a": ["ok"] * 6, "b": ["ok"] * 6}, {"a": [1, 1, 0, 1, 1, 0], "b": [0, 1, 0, 0, 1, 0]}
    )
    rep = policy_bounds(res)
    a = rep.models[0]
    assert a.model == "a" and a.lower == a.upper == pytest.approx(4 / 6)
    assert len(set(a.scores.values())) == 1
    scores = res.task_scores("a").to_numpy()
    z = stats.norm.ppf(0.975)
    se = scores.std(ddof=1) / math.sqrt(6)
    assert a.ci_lo == pytest.approx(4 / 6 - z * se)
    assert rep.ranking_changes == 0 and a.ambiguous_runs == 0


def test_policies_inside_bounds_and_exclusion() -> None:
    res = board(
        {"a": ["ok", "infra_error", "ok", "timeout"], "b": ["ok", "ok", "flagged", "ok"]},
        {"a": [1, 0, 1, 0], "b": [1, 1, 1, 0]},
        runs=2,
    )
    rep = policy_bounds(res)
    for m in rep.models:
        for p in POLICIES:
            assert m.lower - 1e-12 <= m.scores[p] <= m.upper + 1e-12
        assert m.ci_lo <= m.lower and m.ci_hi >= m.upper
    a = next(m for m in rep.models if m.model == "a")
    assert a.ambiguous_runs == 2 and a.ambiguous_share == 0.5
    assert a.scores["exclude"] == 1.0 and a.scores["fail"] == 0.5 and a.scores["success"] == 1.0
    one_task = board({"a": ["timeout", "ok"]}, {"a": [0, 1]}, runs=1)
    assert policy_bounds(one_task).models[0].excluded_tasks == 1


def test_ambiguous_flips_ranking_and_pair_bounds() -> None:
    # a leads only if its infrastructure errors are failures of b ... construct so that
    # as_scored and success rankings differ.
    res = board(
        {
            "a": ["ok"] * 8,
            "b": ["ok", "ok", "ok", "ok", "infra_error", "infra_error", "infra_error", "ok"],
        },
        {"a": [1, 1, 1, 1, 1, 0, 0, 0], "b": [1, 1, 1, 0, 0, 0, 0, 0]},
    )
    rep = policy_bounds(res)
    assert rep.rankings["as_scored"] == ("a", "b")
    assert rep.rankings["success"] == ("b", "a")
    assert rep.ranking_changes >= 1
    pair = rep.adjacent[0]
    assert pair.bound_lo < 0 < pair.bound_hi and not pair.identified and not pair.robust
    assert pair.difference == pytest.approx(2 / 8)


def test_identified_and_robust_ordering() -> None:
    n = 60
    res = board(
        {"a": ["ok"] * n, "b": ["ok"] * (n - 2) + ["timeout"] * 2}, {"a": [1.0] * n, "b": [0.0] * n}
    )
    pair = policy_bounds(res).adjacent[0]
    assert pair.identified and pair.robust and pair.bound_lo == pytest.approx(1 - 2 / n)


def test_imbens_manski_critical_limits() -> None:
    assert imbens_manski_critical(0.0, 0.1, 0.95) == pytest.approx(stats.norm.ppf(0.975))
    assert imbens_manski_critical(100.0, 0.1, 0.95) == pytest.approx(stats.norm.ppf(0.95))
    mid = imbens_manski_critical(0.1, 0.1, 0.95)
    assert stats.norm.ppf(0.95) < mid < stats.norm.ppf(0.975)
    assert stats.norm.cdf(mid + 1.0) - stats.norm.cdf(-mid) == pytest.approx(0.95)
    assert imbens_manski_critical(0.1, 0.0, 0.9) == pytest.approx(stats.norm.ppf(0.95))
    lo, hi, ci_lo, ci_hi = imbens_manski_interval(np.array([0.5]), np.array([0.7]), 0.95)
    assert (lo, hi) == (0.5, 0.7) and math.isinf(ci_lo) and math.isinf(ci_hi)


def test_validation() -> None:
    res = board({"a": ["ok", "ok"]}, {"a": [1, 0]})
    with pytest.raises(ValueError, match="policy"):
        policy_bounds(res, policy="vibes")
    with pytest.raises(ValueError, match="level"):
        policy_bounds(res, level=2)
    plain = EvalResults.from_frame(res.frame.drop(columns="status"))
    with pytest.raises(SchemaError, match="status"):
        policy_bounds(plain)


@pytest.mark.property
@settings(max_examples=60, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.sampled_from(["ok", "ok", "infra_error", "timeout", "flagged"]), st.booleans()
        ),
        min_size=6,
        max_size=30,
    )
)
def test_every_policy_lies_within_bounds(cells: list[tuple[str, bool]]) -> None:
    statuses = [s for s, _ in cells]
    outcomes = [float(y) for _, y in cells]
    res = board({"a": statuses, "b": statuses[::-1]}, {"a": outcomes, "b": outcomes[::-1]})
    rep = policy_bounds(res)
    for m in rep.models:
        assert m.lower <= m.upper
        for p in POLICIES:
            if not math.isnan(m.scores[p]):
                assert m.lower - 1e-12 <= m.scores[p] <= m.upper + 1e-12
    for pair in rep.adjacent:
        assert pair.bound_lo <= pair.bound_hi
        assert pair.ci_lo <= pair.bound_lo + 1e-12


def test_reference_order_drives_pairs_and_changes() -> None:
    res = board(
        {"a": ["ok"] * 4, "b": ["ok", "ok", "infra_error", "ok"]},
        {"a": [1, 1, 0, 0], "b": [1, 1, 0, 1]},
    )
    default = policy_bounds(res)
    assert default.reference_order == default.rankings["as_scored"]
    custom = policy_bounds(res, reference_order=["a", "b"])
    assert custom.reference_order == ("a", "b")
    assert [m.model for m in custom.models] == ["a", "b"]
    assert (custom.adjacent[0].higher, custom.adjacent[0].lower) == ("a", "b")
    assert custom.ranking_changes == sum(o != ("a", "b") for o in custom.rankings.values())
    with pytest.raises(ValueError, match="every model"):
        policy_bounds(res, reference_order=["a"])
