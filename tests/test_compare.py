import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from evalstats import EvalResults, paired_difference, score_interval
from synth import make_results


def test_paired_difference_matches_paired_t(results: EvalResults) -> None:
    res = paired_difference(results, "a", "b")
    sa, sb = results.task_scores("a"), results.task_scores("b")
    ref = stats.ttest_rel(sa, sb)
    assert res.t_stat == pytest.approx(ref.statistic)
    assert res.p_value == pytest.approx(ref.pvalue)
    assert res.difference.estimate == pytest.approx(sa.mean() - sb.mean())
    assert res.n_tasks == 30 and res.unpaired_tasks_a == res.unpaired_tasks_b == 0
    assert res.score_a == pytest.approx(results.score("a"))


def test_paired_bootstrap_methods(results: EvalResults) -> None:
    t = paired_difference(results, "a", "b")
    for method in ("bootstrap", "bca"):
        got = paired_difference(results, "a", "b", method=method, n_boot=4000, seed=1)
        assert got.difference.method == method
        assert got.difference.estimate == pytest.approx(t.difference.estimate)
        assert got.difference.width == pytest.approx(t.difference.width, rel=0.25)


def test_supported_flag_and_large_gap() -> None:
    res = make_results(n_tasks=80, p=(0.85, 0.2), seed=4)
    cmp = paired_difference(res, "a", "b")
    assert cmp.supported and cmp.p_value < 1e-6


def test_unpaired_tasks_counted_and_constant_difference() -> None:
    rows = [{"model": "a", "task": t, "run": 0, "outcome": 1.0} for t in ("x", "y", "z")] + [
        {"model": "b", "task": t, "run": 0, "outcome": 0.0} for t in ("x", "y")
    ]
    res = paired_difference(EvalResults.from_frame(pd.DataFrame(rows)), "a", "b")
    assert res.n_tasks == 2 and res.unpaired_tasks_a == 1 and res.unpaired_tasks_b == 0
    assert res.se == 0.0 and res.p_value == 0.0 and math.isinf(res.t_stat)
    assert res.difference.width == 0.0 and res.supported


def test_identical_models_and_single_task() -> None:
    rows = [{"model": m, "task": "x", "run": 0, "outcome": 1.0} for m in ("a", "b")]
    one = paired_difference(EvalResults.from_frame(pd.DataFrame(rows)), "a", "b")
    assert math.isnan(one.p_value) and not one.supported
    same = [{"model": m, "task": t, "run": 0, "outcome": 1.0} for m in "ab" for t in "xy"]
    zero = paired_difference(EvalResults.from_frame(pd.DataFrame(same)), "a", "b")
    assert zero.t_stat == 0.0 and zero.p_value == 1.0
    with pytest.raises(ValueError, match="different"):
        paired_difference(EvalResults.from_frame(pd.DataFrame(rows)), "a", "a")


def test_score_interval(results: EvalResults) -> None:
    got = score_interval(results, "a")
    scores = results.task_scores("a").to_numpy()
    lo, hi = stats.t.interval(0.95, df=29, loc=scores.mean(), scale=stats.sem(scores))
    assert got.interval.lo == pytest.approx(lo) and got.interval.hi == pytest.approx(hi)
    assert got.n_tasks == 30 and got.runs_per_task == 3.0
    assert got.se == pytest.approx(float(np.std(scores, ddof=1) / np.sqrt(30)))


def test_score_interval_cluster_wilson(results: EvalResults) -> None:
    from evalstats import cluster_wilson_interval

    got = score_interval(results, "a", method="cluster_wilson")
    direct = cluster_wilson_interval(
        results.task_scores("a").to_numpy(), results.run_counts("a").to_numpy(dtype=float)
    )
    assert got.interval == direct and got.interval.method == "cluster_wilson"
    assert got.interval.lo <= got.interval.estimate <= got.interval.hi


def test_paired_difference_mover_wilson(results: EvalResults) -> None:
    from evalstats.intervals import mover_wilson_interval, mover_wilson_p_value

    got = paired_difference(results, "a", "b", method="mover_wilson")
    pair = results.paired("a", "b")
    args = (pair.a, pair.runs_a.astype(float), pair.b, pair.runs_b.astype(float))
    assert got.difference == mover_wilson_interval(*args)
    assert got.p_value == pytest.approx(mover_wilson_p_value(*args))
    t_only = paired_difference(results, "a", "b")
    assert got.t_stat == t_only.t_stat and got.se == t_only.se
