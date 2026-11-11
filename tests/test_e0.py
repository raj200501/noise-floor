import math

import numpy as np
import pytest

from evalstats.glmm import lme4_available
from noisefloor import e0
from noisefloor.e0 import (
    Cell,
    Replicate,
    cells_from_grid,
    expected_sigmoid,
    run_replicate,
    summarize,
)

BASE = {"n_tasks": 30, "runs": 2, "task_sd": 1.0, "intercept": 0.2}


def cell(method: str, interval: str = "cluster_t", reps: int = 4, **params: object) -> Cell:
    return Cell(f"test-{method}-{interval}", method, interval, 0.95, reps, {**BASE, **params})


def test_expected_sigmoid_matches_monte_carlo() -> None:
    z = np.random.default_rng(0).standard_normal(400_000)
    mc = float(np.mean(1 / (1 + np.exp(-(0.3 + 1.2 * z)))))
    assert expected_sigmoid(0.3, 1.2) == pytest.approx(mc, abs=2e-3)
    assert expected_sigmoid(0.0, 0.0) == pytest.approx(0.5)


def test_replicates_are_deterministic_and_distinct() -> None:
    c = cell("score_interval")
    assert run_replicate(c, 0) == run_replicate(c, 0)
    assert run_replicate(c, 0) != run_replicate(c, 1)
    assert c.seed_for(0) != c.seed_for(1)


@pytest.mark.parametrize(
    ("method", "interval", "params"),
    [
        ("score_interval", "cluster_t", {"effect": 0.5, "model_task_sd": 0.3}),
        ("score_interval", "bootstrap", {"n_boot": 200}),
        ("score_interval", "bca", {"n_boot": 200}),
        (
            "paired_difference",
            "cluster_t",
            {"effect_a": 0.4, "effect_b": 0.0, "model_task_sd": 0.4},
        ),
        ("paired_difference", "bootstrap", {"effect_a": 0.0, "effect_b": 0.0, "n_boot": 200}),
        ("cluster_robust_mean", "cluster_robust", {}),
        ("bounded_score", "imbens_manski", {"ambiguous_rate": 0.1}),
    ],
)
def test_interval_methods_produce_sane_replicates(
    method: str, interval: str, params: dict[str, object]
) -> None:
    rep = run_replicate(cell(method, interval, **params), 3)
    assert rep.lo <= rep.estimate <= rep.hi or method == "bounded_score"
    assert 0.0 <= abs(rep.truth) <= 1.0 and rep.hi - rep.lo < 1.0


def test_rank_methods() -> None:
    sim = cell("rank_simultaneous", "bootstrap_max_t", effects=[1.0, 0.0, -1.0], n_boot=200)
    rep = run_replicate(sim, 0)
    assert rep.hit is not None and isinstance(rep.covered, bool)
    marg = cell("rank_marginal", "bootstrap_max_t", effects=[1.0, 0.0, -1.0], n_boot=200, target=1)
    r2 = run_replicate(marg, 0)
    assert r2.truth == 2.0 and r2.lo <= r2.hi


def test_cluster_t_coverage_smoke() -> None:
    c = Cell("smoke", "score_interval", "cluster_t", 0.9, 150, {**BASE, "n_tasks": 40})
    reps = [run_replicate(c, i) for i in range(c.replicates)]
    out = summarize(c, reps)
    assert out["replicates"] == 150 and 0.8 <= out["coverage"] <= 0.98
    assert out["mcse"] == pytest.approx(math.sqrt(out["coverage"] * (1 - out["coverage"]) / 150))


@pytest.mark.skipif(not lme4_available(), reason="lme4 unavailable")
def test_glmm_component_replicate() -> None:
    c = Cell(
        "g",
        "glmm_component",
        "profile",
        0.95,
        1,
        {
            "n_models": 3,
            "model_sd": 0.5,
            "n_tasks": 25,
            "runs": 2,
            "task_sd": 1.0,
            "model_task_sd": 0.0,
            "factor_beta": 0.6,
            "component": "factor",
            "random": ["task", "model"],
        },
    )
    rep = run_replicate(c, 0)
    assert rep.truth == pytest.approx(0.25 * 0.36)
    assert rep.lo <= rep.estimate <= rep.hi


def test_glmm_failed_profile_never_covers(monkeypatch: pytest.MonkeyPatch) -> None:
    from evalstats.glmm import Decomposition, VarianceComponent

    def fake(*args: object, **kwargs: object) -> Decomposition:
        comp = VarianceComponent("task", "random", 1.0, 0.2, None, None, "none")
        return Decomposition("fake", "f", 10, 0.95, (comp,), {}, True)

    monkeypatch.setattr(e0, "decompose", fake)
    c = Cell(
        "g",
        "glmm_component",
        "profile",
        0.95,
        1,
        {"n_models": 2, "n_tasks": 5, "runs": 1, "task_sd": 1.0, "component": "task"},
    )
    rep = run_replicate(c, 0)
    assert math.isnan(rep.lo) and not rep.covered


def test_summary_and_grid_helpers() -> None:
    reps = [
        Replicate(0.5, 0.4, 0.6, 0.5),
        Replicate(0.5, 0.6, 0.7, 0.65),
        Replicate(math.nan, 0, 1, math.nan, hit=True),
    ]
    out = summarize(cell("score_interval"), reps)
    assert out["covered"] == 2 and out["replicates"] == 3
    empty = summarize(cell("score_interval"), [])
    assert math.isnan(empty["coverage"])
    grid = [
        {
            "cell": "a",
            "method": "score_interval",
            "interval_type": "cluster_t",
            "nominal": 0.95,
            "replicates": 10,
            "params": {"n_tasks": 5},
        }
    ]
    cells = cells_from_grid(grid)
    assert cells[0].cell == "a" and cells[0].params == {"n_tasks": 5}
    assert e0.canonical({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_paired_cells_with_unequal_runs() -> None:
    kw = {"n_tasks": 5, "runs": 2, "intercept": 0.0, "task_sd": 1.0, "model_task_sd": 0.5}
    same = e0._draw(np.random.default_rng(4), [0.1, -0.1], **kw)
    explicit = e0._draw(np.random.default_rng(4), [0.1, -0.1], runs_per_model=(2, 2), **kw)
    assert same.equals(explicit)  # equal runs draw exactly as before
    mixed = e0._draw(np.random.default_rng(4), [0.1, -0.1], runs_per_model=(1, 4), **kw)
    counts = mixed.groupby("model")["run"].nunique().to_dict()
    assert counts == {"m0": 1, "m1": 4} and len(mixed) == 5 * (1 + 4)
    c = cell("paired_difference", effect_a=0.4, effect_b=0.0, model_task_sd=0.7, runs=1, runs_b=4)
    rep = run_replicate(c, 0)
    assert rep.lo <= rep.estimate <= rep.hi and rep.truth > 0


def test_bounded_cells_with_truth_at_a_bound() -> None:
    base = {"runs": 4, "ambiguous_rate": 0.4, "model_task_sd": 0.7}
    fail = run_replicate(cell("bounded_score", "imbens_manski", ambiguous_among="fail", **base), 0)
    succ = run_replicate(
        cell("bounded_score", "imbens_manski", ambiguous_among="success", **base), 0
    )
    anyrun = run_replicate(cell("bounded_score", "imbens_manski", **base), 0)
    # Ambiguous true failures recorded as 0 leave the scored value at the truth's end.
    assert fail.lo <= fail.estimate <= fail.hi and succ.estimate < anyrun.estimate + 0.5
    with pytest.raises(ValueError, match="ambiguous_among"):
        run_replicate(cell("bounded_score", "imbens_manski", ambiguous_among="x", **base), 0)
    # The default ("any") draws exactly as before.
    plain = {"runs": 4, "ambiguous_rate": 0.4}
    assert run_replicate(cell("bounded_score", "imbens_manski", **plain), 2) == run_replicate(
        cell("bounded_score", "imbens_manski", ambiguous_among="any", **plain), 2
    )


def test_within_model_contrasts_share_task_effects_and_split_the_timeout() -> None:
    p = {"n_tasks": 3000, "runs": 1, "intercept": 0.0, "task_sd": 1.4, "model_task_sd": 0.7}
    frame = e0._draw_within(np.random.default_rng(9), {**p, "split_shift": 0.0}, [0.0, 0.0])
    wide = frame.pivot(index="task", columns="model", values="outcome")
    apart = e0._draw(np.random.default_rng(9), [0.0, 0.0], **p)  # independent model:task
    wide_apart = apart.pivot(index="task", columns="model", values="outcome")
    # Sharing the model-by-task effect raises the per-task correlation of the two sides.
    assert wide["m0"].corr(wide["m1"]) > wide_apart["m0"].corr(wide_apart["m1"]) + 0.05
    sd = math.sqrt(1.4**2 + 0.7**2)
    shifted = {**p, "split_shift": 0.6}
    expected = 0.5 * (
        (expected_sigmoid(0.4 + 0.6, sd) - expected_sigmoid(0.0, sd))
        + (expected_sigmoid(0.4, sd) - expected_sigmoid(0.6, sd))
    )
    assert e0._true_within(0.4, 0.0, shifted) == pytest.approx(expected)
    assert e0._true_within(0.4, 0.0, p) == pytest.approx(e0._true_difference(0.4, 0.0, p))
    c = cell(
        "paired_difference",
        "mover_wilson",
        effect_a=0.4,
        effect_b=0.0,
        runs=1,
        within=True,
        split_shift=0.4,
        model_task_sd=0.7,
    )
    rep = run_replicate(c, 0)
    assert rep.lo <= rep.estimate <= rep.hi and rep.truth == pytest.approx(
        e0._true_within(0.4, 0.0, c.params)
    )


def test_sampling_cell_simulates_discordant_pairs_of_independent_runs() -> None:
    import numpy as np

    from noisefloor import e0

    params = {
        "n_tasks": 80,
        "intercept": 0.0,
        "task_sd": 2.0,
        "model_task_sd": 0.7071,
        "effects": [0.6, 0.0, -0.6],
    }
    cell = e0.Cell("S-r-w-x", "sampling_discordance", "cluster_wilson", 0.95, 40, params)
    reps = [e0.run_replicate(cell, i) for i in range(40)]
    truth = reps[0].truth
    assert 0.08 < truth < 0.2 and all(r.truth == truth for r in reps)
    assert all(0 <= r.lo <= r.estimate <= r.hi <= 0.5 for r in reps)
    assert abs(float(np.mean([r.estimate for r in reps])) - truth) < 0.02
    assert sum(r.covered for r in reps) >= 32  # most intervals cover (a sanity check)
    low = e0._true_discordance({**params, "intercept": 6.0})
    assert low < 0.03
