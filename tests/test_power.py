import math

import numpy as np
import pytest
from scipy import stats

from evalstats import PairedDesign, design_from_results, mde, power_at, tasks_needed
from synth import make_results

DESIGN = PairedDesign(between_task_var=0.04, within_task_var=0.4)


def test_se_and_mde_closed_form() -> None:
    se = math.sqrt((0.04 + 0.4 / 4) / 100)
    assert DESIGN.se(100, 4) == pytest.approx(se)
    expected = (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * se
    assert mde(DESIGN, 100, 4) == pytest.approx(expected)
    one = (stats.norm.ppf(0.95) + stats.norm.ppf(0.8)) * se
    assert mde(DESIGN, 100, 4, two_sided=False) == pytest.approx(one)


def test_mde_decreases_with_tasks_and_runs() -> None:
    assert mde(DESIGN, 400) < mde(DESIGN, 100) < mde(DESIGN, 25)
    assert mde(DESIGN, 100, 5) < mde(DESIGN, 100, 1)


def test_power_at_mde_is_target_power() -> None:
    effect = mde(DESIGN, 150, 2, power=0.9)
    assert power_at(effect, DESIGN, 150, 2) == pytest.approx(0.9, abs=1e-3)
    assert power_at(0.0, DESIGN, 150, 2) == pytest.approx(0.05)
    assert power_at(effect, DESIGN, 150, 2, two_sided=False) > 0.9


def test_tasks_needed_inverts_mde() -> None:
    n = tasks_needed(0.05, DESIGN, 3)
    assert mde(DESIGN, n, 3) <= 0.05 < mde(DESIGN, n - 1, 3)
    with pytest.raises(ValueError, match="positive"):
        tasks_needed(0.0, DESIGN)


def test_zero_variance_design() -> None:
    zero = PairedDesign(0.0, 0.0)
    assert power_at(0.1, zero, 10) == 1.0 and power_at(0.0, zero, 10) == 0.05


@pytest.mark.parametrize("bad", [(-1.0, 0.1), (0.1, math.inf)])
def test_design_validation(bad: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        PairedDesign(*bad)


def test_argument_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        DESIGN.se(0)
    with pytest.raises(ValueError, match="alpha"):
        mde(DESIGN, 10, alpha=0.0)


def test_design_from_results_recovers_within_variance() -> None:
    res = make_results(n_tasks=200, runs=6, p=(0.5, 0.5), seed=7)
    design = design_from_results(res, "a", "b")
    frame = res.frame
    per_task = frame.groupby(["model", "task"])["outcome"].var(ddof=1)
    expected_within = float(per_task.groupby("task").sum().mean())
    assert design.within_task_var == pytest.approx(expected_within)
    assert design.between_task_var >= 0.0
    diffs = res.paired("a", "b").differences
    assert design.between_task_var <= float(np.var(diffs, ddof=1))


def test_design_from_results_needs_replicates() -> None:
    with pytest.raises(ValueError, match="2 runs"):
        design_from_results(make_results(runs=1), "a", "b")
    with pytest.raises(ValueError, match="2 shared tasks"):
        design_from_results(make_results(n_tasks=1), "a", "b")
