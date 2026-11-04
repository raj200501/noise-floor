"""The E1, E0 part 2, and G6 preregistrations agree (reviews ops-8 and stats-6)."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from evalstats.glmm import lme4_available
from noisefloor import prereg_consistency as pc

REPO = Path(__file__).resolve().parent.parent


def blocks_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "prereg_blocks", REPO / "tools" / "prereg_blocks.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["prereg_blocks"] = module
    spec.loader.exec_module(module)
    return module


def write(path: Path, blocks: dict[str, object]) -> Path:
    text = "# test\n\n" + "".join(
        f"```json {name}\n{json.dumps(value)}\n```\n\n" for name, value in blocks.items()
    )
    path.write_text(text)
    return path


def test_generated_blocks_agree_and_mutations_are_caught(tmp_path: Path) -> None:
    tool = blocks_tool()
    rules = tool.pilot_rules()
    e1 = tool.e1_blocks(4, rules)
    config = tool.recovery_config(rules)
    p4 = write(tmp_path / "0004.md", e1)
    p5 = write(tmp_path / "0005.md", {"recovery-config": config})
    p6 = write(tmp_path / "0006.md", {"recovery-config": config})
    assert pc.problems(p4, p5, p6) == []
    wrong = json.loads(json.dumps(config))
    wrong["design"]["n_models"] = 4
    wrong["settings"]["boot_interval"] = "percentile"
    p6b = write(tmp_path / "0006b.md", {"recovery-config": wrong})
    found = pc.problems(p4, p5, p6b)
    assert "the recovery-config blocks of 0005 and 0006 differ" in found
    assert any(p.startswith("n_models") for p in found)
    assert any(p.startswith("boot_interval") for p in found)
    analysis = {**e1["e1-analysis"], "fixed_interval": "box"}
    p4b = write(tmp_path / "0004b.md", {**e1, "e1-analysis": analysis})
    assert any(p.startswith("fixed_interval") for p in pc.problems(p4b, p5, p6))


def test_frozen_preregistrations_agree() -> None:
    files = {n: sorted((REPO / "prereg").glob(f"{n}-*.md")) for n in ("0004", "0005", "0006")}
    if not all(files.values()):
        pytest.skip("preregistrations 0004 to 0006 are not in prereg/ yet")
    assert pc.problems(files["0004"][0], files["0005"][0], files["0006"][0]) == []


@pytest.mark.skipif(not lme4_available(), reason="R, lme4, or rpy2 not available")
def test_g6_component_list_matches_a_recovery_row() -> None:
    from noisefloor import recovery as rc

    tool = blocks_tool()
    design = rc.Design(n_models=3, n_tasks=6, runs_per_cell=16)
    planted = rc.Planted(model=(0.6, 0.0, -0.6), task_var=4.0, batch_var=0.0)
    settings = rc.AnalysisSettings(
        n_boot=3,
        nagq=0,
        boot_interval="basic_log",
        box_interval="basic",
        share_interval="plugin",
        fixed_interval="jackknife",
        random=("task", "model:task"),
        mde_designs=tuple(tuple(d) for d in rc.mde_lib.DESIGNS),
        mde_interval="bc",
    )
    row = rc.recover(design, planted, settings, 3)
    assert sorted(row["components"]) == tool.g6_components()
