import numpy as np
import pandas as pd
import pytest
from scipy import stats

from evalstats import EvalResults, SchemaError
from evalstats.rank import complete_score_matrix, rank_confidence
from synth import make_results


def leaderboard(
    gaps: tuple[float, ...], n_tasks: int = 60, runs: int = 2, seed: int = 0
) -> EvalResults:
    names = tuple(f"m{i}" for i in range(len(gaps)))
    probs = tuple(1 / (1 + np.exp(-g)) for g in gaps)
    return make_results(n_tasks=n_tasks, runs=runs, models=names, p=probs, seed=seed)


def test_well_separated_models_get_exact_ranks() -> None:
    res = leaderboard((3.0, 0.0, -3.0), n_tasks=150)
    rc = rank_confidence(res, seed=1)
    assert [r.model for r in rc.intervals] == ["m0", "m1", "m2"]
    assert all(r.best_rank == r.worst_rank == r.rank for r in rc.intervals)
    assert rc.top_set == ("m0",) and rc.share_adjacent_supported == 1.0
    assert all(p.supported and p.supported_simultaneous for p in rc.adjacent)


def test_identical_models_span_all_ranks() -> None:
    res = leaderboard((0.0, 0.0, 0.0, 0.0), n_tasks=40)
    rc = rank_confidence(res, seed=2)
    assert all(r.best_rank == 1 and r.worst_rank == 4 for r in rc.intervals)
    assert len(rc.top_set) == 4


def test_simultaneous_is_at_least_as_wide_as_marginal() -> None:
    res = leaderboard((0.6, 0.3, 0.0, -0.3, -0.6), n_tasks=80, seed=3)
    sim = rank_confidence(res, seed=4)
    marg = rank_confidence(res, simultaneous=False, seed=4)
    assert sim.critical_value >= marg.critical_value
    for s, m in zip(sim.intervals, marg.intervals, strict=True):
        assert s.best_rank <= m.best_rank and s.worst_rank >= m.worst_rank
        assert s.best_rank <= s.rank <= s.worst_rank


def test_adjacent_pairs_match_paired_t() -> None:
    res = leaderboard((0.5, 0.0, -0.2), n_tasks=50, seed=5)
    rc = rank_confidence(res, seed=0)
    first = rc.adjacent[0]
    a = res.task_scores(first.higher)
    b = res.task_scores(first.lower)
    ref = stats.ttest_rel(a, b)
    assert first.p_value == pytest.approx(ref.pvalue)
    assert first.difference == pytest.approx(float(a.mean() - b.mean()))
    assert first.supported == (first.lo > 0)


def test_seeded_and_level_monotone() -> None:
    res = leaderboard((0.4, 0.2, 0.0, -0.2), n_tasks=60, seed=6)
    assert rank_confidence(res, seed=9) == rank_confidence(res, seed=9)
    narrow = rank_confidence(res, level=0.5, seed=9)
    wide = rank_confidence(res, level=0.99, seed=9)
    assert wide.critical_value > narrow.critical_value


def test_incomplete_tasks_dropped_and_errors() -> None:
    res = leaderboard((0.0, 0.0), n_tasks=10)
    frame = res.frame
    frame = frame.loc[~((frame["model"] == "m1") & (frame["task"] == "t000"))]
    rc = rank_confidence(EvalResults.from_frame(frame), n_boot=200, seed=0)
    assert rc.dropped_tasks == 1 and rc.n_tasks == 9
    one = EvalResults.from_frame(
        pd.DataFrame({"model": ["a", "a"], "task": ["x", "y"], "run": [0, 0], "outcome": [1, 0]})
    )
    with pytest.raises(SchemaError, match="2 models"):
        rank_confidence(one)
    sparse = EvalResults.from_frame(
        pd.DataFrame({"model": ["a", "b"], "task": ["x", "y"], "run": [0, 0], "outcome": [1, 0]})
    )
    with pytest.raises(SchemaError, match="2 tasks"):
        complete_score_matrix(sparse)
    with pytest.raises(ValueError, match="level"):
        rank_confidence(res, level=0)


def test_deterministic_difference_counts_as_supported() -> None:
    rows = [
        {"model": m, "task": f"t{t}", "run": 0, "outcome": float(m == "a")}
        for m in "ab"
        for t in range(5)
    ]
    rc = rank_confidence(EvalResults.from_frame(pd.DataFrame(rows)), n_boot=200, seed=0)
    pair = rc.adjacent[0]
    assert pair.supported and pair.p_value == 0.0
    assert rc.intervals[0].worst_rank == 1
    ties = [{"model": m, "task": f"t{t}", "run": 0, "outcome": 1.0} for m in "ab" for t in range(5)]
    tied = rank_confidence(EvalResults.from_frame(pd.DataFrame(ties)), n_boot=200, seed=0)
    assert tied.adjacent[0].p_value == 1.0 and not tied.adjacent[0].supported


def test_empty_adjacent_share_is_nan() -> None:
    from evalstats.rank import RankConfidence

    rc = RankConfidence(0.95, True, 2.0, 10, 0, 100, (), ())
    assert np.isnan(rc.share_adjacent_supported)


def test_simultaneous_rank_coverage_smoke() -> None:
    # Small Monte Carlo smoke check (E0 runs the real coverage study).
    gaps = (0.5, 0.25, 0.0, -0.25)
    covered = 0
    reps = 40
    for rep in range(reps):
        res = leaderboard(gaps, n_tasks=60, runs=1, seed=100 + rep)
        rc = rank_confidence(res, n_boot=400, seed=rep)
        true_rank = {f"m{i}": i + 1 for i in range(len(gaps))}
        covered += all(r.best_rank <= true_rank[r.model] <= r.worst_rank for r in rc.intervals)
    assert covered / reps >= 0.8
