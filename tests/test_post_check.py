"""The post-E1 check's mapping from an E1 fit to planted values (synthetic only)."""

import pandas as pd
import pytest

from noisefloor import post_check as pc
from noisefloor.e1 import E1Design

MODELS = ("qwen3-1.7b", "qwen3-4b", "granite-4.0-micro")


def test_planted_from_fit_maps_treatment_coding_to_simulation_codes() -> None:
    design = E1Design(models=MODELS, runs_per_cell=16)
    data = pd.DataFrame(
        {
            "model": ["qwen3-4b", "granite-4.0-micro", "qwen3-1.7b"] * 4,
            "task": [f"t{i}" for i in range(6)] * 2,
            "day": ["d1", "d2", "d3"] * 4,
            "block": ["b1", "b2"] * 6,
        }
    )
    fixed = {
        "(Intercept)": -0.5,
        "modelqwen3-1.7b": 0.3,  # granite-4.0-micro sorts first: the reference model
        "modelqwen3-4b": 0.8,
        "harnessnative": 0.4,
        "timeout300.0": 0.2,
        "steps8": -0.6,  # "15" sorts before "8": the fit's reference is 15 steps
        "temperature0.7": 0.1,
        "concurrency2": -0.05,
        "dayd2": 0.07,
        "dayd3": -0.02,
        "blockb2": 0.03,
    }
    sim, planted = pc.planted_from_fit(
        fixed,
        {"task": 4.0, "model:task": 0.5},
        batch_var=0.02,
        interactions={"harness": 0.1, "steps": 0.0, "timeout": 0.2},
        design=design,
        data=data,
    )
    assert (sim.n_models, sim.n_tasks, sim.days, sim.blocks_per_day) == (3, 6, 3, 2)
    assert planted.model == (0.0, 0.3, 0.8)  # granite, qwen3-1.7b, qwen3-4b
    assert planted.factors["harness"] == 0.4 and planted.factors["steps"] == pytest.approx(0.6)
    # Level 0 of steps is 8 steps, whose coefficient joins the simulation's intercept.
    assert planted.intercept == pytest.approx(-0.5 - 0.6)
    assert planted.day == (0.0, 0.07, -0.02) and planted.block == (0.0, 0.03)
    assert planted.task_var == 4.0 and planted.batch_var == 0.02
    assert planted.interactions == {"harness": 0.1, "steps": 0.0, "timeout": 0.2}


def test_summarize_applies_the_g5_rule_and_the_batch_rule() -> None:
    def row(hit: bool) -> dict:
        good = {"planted": 1.0, "lo": 0.5, "hi": 1.5}
        return {
            "components": {
                "task": good,
                "harness": good if hit else {"planted": 1.0, "lo": None, "hi": None},
            }
        }

    rows = [row(i % 5 != 0) for i in range(200)]  # harness covers 80 percent
    out = pc.summarize(rows, 0.95, batch_var=0.01)
    assert out["components"]["task"]["passes"] and not out["components"]["harness"]["passes"]
    assert out["failing"] == ["harness"] and out["not_validated"] == ["harness"]
    assert not out["batch_rule_exceeded"]
    noisy = pc.summarize(rows, 0.95, batch_var=0.2)
    assert noisy["batch_rule_exceeded"]
    assert noisy["not_validated"] == ["block", "concurrency", "day", "harness"]
    ok, coverage, threshold = pc.g5_rule(190, 200, 0.95)
    assert ok and coverage == 0.95 and threshold < 0.95


def test_seeds_and_settings() -> None:
    assert len(pc.POST_SEEDS) == pc.POST_DATASETS == 200
    assert pc.POST_SEEDS[0] == 980000
    s = pc.settings_from(
        {
            "n_boot": 200,
            "nagq": 1,
            "boot_interval": "basic_log",
            "level": 0.95,
            "random": ["task", "model:task"],
            "box_interval": "basic",
            "share_interval": "plugin",
            "fixed_interval": "jackknife",
        }
    )
    assert s.fixed_interval == "jackknife" and s.random == ("task", "model:task")
