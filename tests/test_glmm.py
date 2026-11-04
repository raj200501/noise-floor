import math

import pandas as pd
import pytest

from evalstats import EvalResults, glmm
from evalstats.glmm import decompose, describe, formula_for, lme4_available
from noisefloor.synthetic import PlantedSpec, simulate

needs_r = pytest.mark.skipif(not lme4_available(), reason="R, lme4, or rpy2 not available")

SPEC = PlantedSpec(
    models={"m1": 0.6, "m2": 0.0, "m3": -0.6},
    n_tasks=40,
    runs=2,
    task_sd=1.0,
    model_task_sd=0.3,
    factors={"harness": {"h0": 0.0, "h1": 0.8}},
)


@pytest.fixture(scope="module")
def planted() -> EvalResults:
    return EvalResults.from_frame(simulate(SPEC, 11))


def check_structure(d: glmm.Decomposition) -> None:
    assert [c.name for c in d.components] == ["task", "model", "harness", "residual"]
    assert sum(c.share for c in d.components) == pytest.approx(1.0)
    assert d.total_variance == pytest.approx(sum(c.variance for c in d.components))
    assert d.component("residual").variance == pytest.approx(math.pi**2 / 3)
    for c in d.components[:-1]:
        assert c.lo is not None and c.hi is not None
        assert c.lo <= c.variance + 1e-9 and c.variance <= c.hi + 1e-9
    with pytest.raises(KeyError):
        d.component("nope")


@needs_r
def test_lme4_decompose_with_profile_intervals(planted: EvalResults) -> None:
    d = decompose(planted, random=("task", "model"), fixed=("harness",), backend="lme4")
    assert d.backend.startswith("lme4") and d.converged
    assert d.formula == "outcome ~ 1 + harness + (1 | task) + (1 | model)"
    check_structure(d)
    assert d.component("task").interval_method == "profile"
    assert d.fixed_effects["harnessh1"] == pytest.approx(0.8, abs=0.35)
    harness = d.component("harness")
    beta = d.fixed_effects["harnessh1"]
    assert harness.variance == pytest.approx(0.25 * beta**2)


@needs_r
def test_lme4_crossed_interaction_term(planted: EvalResults) -> None:
    d = decompose(planted, random=("task", "model:task"), intervals=False, backend="lme4")
    assert [c.name for c in d.components] == ["task", "model:task", "residual"]
    assert all(
        c.lo is None and c.interval_method in ("none", "fixed_logistic") for c in d.components
    )


def test_statsmodels_fallback(planted: EvalResults) -> None:
    d = decompose(planted, random=("task", "model"), fixed=("harness",), backend="statsmodels")
    assert "variational Bayes" in d.backend
    check_structure(d)
    assert d.component("model").interval_method == "vb_posterior"
    assert d.fixed_effects["harnessh1"] == pytest.approx(0.8, abs=0.35)


def test_three_level_factor_has_no_interval() -> None:
    spec = PlantedSpec(
        models={"m": 0.0}, n_tasks=30, runs=1, factors={"timeout": {"a": 0.0, "b": 0.5, "c": -0.5}}
    )
    res = EvalResults.from_frame(simulate(spec, 2))
    d = decompose(res, random=("task",), fixed=("timeout",), backend="statsmodels")
    comp = d.component("timeout")
    assert comp.lo is None and comp.hi is None and comp.variance > 0


def test_auto_backend_and_describe(planted: EvalResults, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(glmm, "lme4_available", lambda: False)
    d = decompose(planted, random=("task",), backend="auto")
    assert "statsmodels" in d.backend
    rows = describe(d)
    assert rows[0]["component"] == "task" and rows[-1]["kind"] == "residual"


def test_validation(planted: EvalResults) -> None:
    with pytest.raises(ValueError, match="unknown column"):
        decompose(planted, random=("nope",))
    with pytest.raises(ValueError, match="invalid column"):
        decompose(planted, random=("task; drop",))
    with pytest.raises(ValueError, match="backend"):
        decompose(planted, random=("task",), backend="spss")
    with pytest.raises(ValueError, match="level"):
        decompose(planted, random=("task",), level=1.5)
    graded = EvalResults.from_frame(
        pd.DataFrame(
            {"model": ["a", "a"], "task": ["x", "y"], "run": [0, 0], "outcome": [0.5, 1.0]}
        )
    )
    with pytest.raises(ValueError, match="binary"):
        decompose(graded)


def test_formula_and_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    assert formula_for([], []) == "outcome ~ 1"
    assert formula_for(["task"], ["h"]) == "outcome ~ 1 + h + (1 | task)"
    import builtins

    real_import = builtins.__import__

    def fail(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("rpy2"):
            raise ImportError("no rpy2")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]  # passthrough

    monkeypatch.setattr(builtins, "__import__", fail)
    assert lme4_available() is False


# ---------------------------------------------------------------------------
# parametric bootstrap


def test_boot_variance_interval_rules() -> None:
    import numpy as np

    draws = np.array([0.1, 0.2, 0.3, 0.4, 0.5, np.nan])
    lo, hi = glmm.boot_variance_interval(0.3, draws, 0.8, "percentile")
    assert lo == pytest.approx(np.quantile(draws[:5], 0.1)) and hi == pytest.approx(
        np.quantile(draws[:5], 0.9)
    )
    b_lo, b_hi = glmm.boot_variance_interval(0.3, draws, 0.8, "basic")
    assert b_lo == pytest.approx(0.6 - hi) and b_hi == pytest.approx(0.6 - lo)
    assert glmm.boot_variance_interval(0.01, draws, 0.8, "basic")[0] == 0.0  # cut at 0
    bc = glmm.boot_variance_interval(0.3, draws, 0.8, "bc")
    assert bc[0] <= 0.3 <= bc[1]
    shifted = glmm.boot_variance_interval(0.5, draws, 0.8, "bc")  # estimate above every draw
    assert shifted[1] >= hi
    assert glmm.boot_variance_interval(0.3, np.array([0.2]), 0.9) is None
    with pytest.raises(ValueError, match="unknown bootstrap interval"):
        glmm.boot_variance_interval(0.3, draws, 0.9, "student")


def test_box_interval_for_two_and_three_levels() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    weights = np.array([0.5, 0.5])
    spans_zero = rng.normal(0.0, 0.1, (200, 1))
    lo, hi = glmm.contribution_box(weights, np.array([0.02]), spans_zero, 0.95)
    assert lo == 0.0  # exactly: the box holds the origin
    assert hi == pytest.approx(0.25 * np.quantile(np.abs(spans_zero), 0.975) ** 2, rel=0.3)
    away = rng.normal(1.0, 0.1, (200, 1))
    lo, hi = glmm.contribution_box(weights, np.array([1.0]), away, 0.95)
    assert 0.0 < lo < 0.25 < hi
    three = np.full(3, 1 / 3)
    draws = rng.normal([0.5, -0.5], 0.05, (300, 2))
    lo, hi = glmm.contribution_box(three, np.array([0.5, -0.5]), draws, 0.95)
    est = glmm._contribution_variance(three, np.array([0.5, -0.5]))
    assert lo < est < hi
    assert glmm.contribution_box(three, np.zeros(2), np.full((5, 2), np.nan), 0.95) is None
    assert glmm.contribution_box(np.ones(1), np.zeros(0), np.empty((5, 0)), 0.95) is None


@needs_r
def test_lme4_parametric_bootstrap(planted: EvalResults) -> None:
    d = decompose(
        planted,
        random=("task", "model"),
        fixed=("harness",),
        method="parametric_bootstrap",
        n_boot=12,
        nagq=0,
        boot_interval="basic",
        seed=3,
    )
    check_structure(d)
    assert "nAGQ=0" in d.backend and "bootstrap: 12 replicates" in d.messages[-1]
    assert d.component("task").interval_method == "parametric_bootstrap_basic"
    assert d.component("harness").interval_method == "parametric_bootstrap_box_percentile"
    for comp in d.components:
        assert comp.share_lo is not None and comp.share_hi is not None
        assert 0.0 <= comp.share_lo <= comp.share_hi <= 1.0
    assert set(d.replicates) == {"task", "model", "harness"}
    assert all(len(v) == 12 for v in d.replicates.values())
    again = decompose(
        planted,
        random=("task", "model"),
        fixed=("harness",),
        method="parametric_bootstrap",
        n_boot=12,
        nagq=0,
        boot_interval="basic",
        seed=3,
    )
    assert again.replicates == d.replicates  # the seed fixes the replicates


def test_bootstrap_option_validation(planted: EvalResults) -> None:
    with pytest.raises(ValueError, match="unknown method"):
        decompose(planted, method="jackknife")
    with pytest.raises(ValueError, match="unknown bootstrap interval"):
        decompose(planted, method="parametric_bootstrap", boot_interval="t")
    with pytest.raises(ValueError, match="nagq"):
        decompose(planted, method="parametric_bootstrap", nagq=2)
    with pytest.raises(ValueError, match="nagq"):
        decompose(planted, method="parametric_bootstrap", n_boot=1)
    with pytest.raises(ValueError, match="needs the lme4 backend"):
        decompose(planted, method="parametric_bootstrap", backend="statsmodels")


def test_basic_log_interval() -> None:
    import numpy as np

    draws = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    lo, hi = glmm.boot_variance_interval(0.4, draws, 0.8, "basic_log")
    q_lo, q_hi = np.quantile(draws, [0.1, 0.9])
    assert lo == pytest.approx(0.16 / q_hi) and hi == pytest.approx(0.16 / q_lo)
    zero = np.array([0.0, 0.0, 0.1, 0.2])
    assert glmm.boot_variance_interval(0.1, zero, 0.8, "basic_log") == pytest.approx(
        tuple(np.quantile(zero, [0.1, 0.9]))
    )
    coef = glmm.boot_coef_interval(0.4, draws, 0.8, "basic_log")
    assert coef == pytest.approx(glmm.boot_coef_interval(0.4, draws, 0.8, "basic"))
