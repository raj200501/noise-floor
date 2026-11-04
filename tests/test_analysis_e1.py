"""End-to-end E1 analysis on synthetic run records (no model was run)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from evalstats.glmm import lme4_available
from noisefloor.analysis import e1 as a
from noisefloor.e1 import E1Design, claude_plan, local_plan

needs_r = pytest.mark.skipif(not lme4_available(), reason="R, lme4, or rpy2 not available")
DESIGN = E1Design(
    models=("qwen3-4b", "granite-4.0-micro", "phi-4-mini"),
    families=("config", "pointer"),
    tasks_per_family=4,
    claude_tiers=("claude-haiku",),
)
SETTINGS = {
    "random": ["task", "model:task"],
    "fixed": ["model", "harness", "timeout", "steps", "temperature", "concurrency", "day", "block"],
    "level": 0.95,
    "n_boot": 6,
    "nagq": 0,
    "boot_interval": "basic_log",
    "box_interval": "basic",
    "share_interval": "basic_log",
    "score_interval": {"local": "cluster_t", "claude": "cluster_wilson"},
    "contrast_interval": {
        "harness-local": "cluster_t",
        "harness-claude": "mover_wilson",
        "models-local": "cluster_t",
        "models-claude": "mover_wilson",
    },
    "seed": 7,
    "exclude_status": ["infra_error"],
    "block_hours": [[0, 12], [12, 24]],
    "mde": {
        "designs": [[40, 1], [89, 5]],
        "reference": {
            "harness": "minimal",
            "timeout": "300.0",
            "steps": "15",
            "temperature": "0.0",
            "concurrency": "1",
        },
        "averaged": ["day", "block"],
        "interval": "basic",
    },
}


def write_prereg(path: Path) -> Path:
    blocks = {"e1-plan": DESIGN.to_json(), "e1-analysis": SETTINGS}
    text = ["# synthetic test prereg", ""]
    for name, value in blocks.items():
        text += [f"```json {name}", json.dumps(value), "```", ""]
    path.write_text("\n".join(text))
    return path


def fake_records() -> list[dict]:
    rng = np.random.default_rng(1)
    task_eff: dict[str, float] = {}
    out = []
    for i, spec in enumerate([*local_plan(DESIGN), *claude_plan(DESIGN)]):
        key = f"{spec.family}{spec.task_seed}"
        task_eff.setdefault(key, float(rng.normal(0, 1.2)))
        p = 1 / (1 + math.exp(-(task_eff[key] + (0.5 if spec.harness == "minimal" else 0.0))))
        day = 4 + (i % 3)
        hour = 3 if i % 2 else 15
        status = "infra_error" if i % 97 == 0 else "ok"
        out.append(
            {
                **spec.to_json(),
                "outcome": float(rng.random() < p),
                "outcome_strict": 0.0,
                "status": status,
                "end_reason": "submitted",
                "started_at": f"2026-11-0{day}T{hour:02d}:00:00+00:00",
                "batch_id": f"b{i // 10}",
            }
        )
    return out


def test_time_block_and_runs_table() -> None:
    assert a.time_block("2026-11-05T13:20:00+00:00", [[0, 12], [12, 24]]) == ("2026-11-05", "b2")
    with pytest.raises(ValueError, match="no time-of-day block"):
        a.time_block("2026-11-05T13:20:00", [[0, 12]])
    recs = fake_records()
    planned = {s.run_id: s.phase for s in local_plan(DESIGN)}
    rows = a.runs_table(recs, prereg="p.md", planned=planned, block_hours=[[0, 12], [12, 24]])
    assert len(rows) == len(recs)
    local = [r for r in rows if r["backend"] == "local"]
    assert all(r["confirmatory"] for r in local) and not any(
        r["confirmatory"] for r in rows if r["backend"] == "claude"
    )
    assert {r["family"] for r in local} == {"qwen3", "granite", "phi"}
    with pytest.raises(ValueError, match="no confirmatory"):
        a.analysis_frame([], ["infra_error"])


@needs_r
def test_build_and_verify(tmp_path: Path) -> None:
    prereg = write_prereg(tmp_path / "prereg.md")
    recs = fake_records()
    local = [r for r in recs if r["phase"] == "e1"]
    (tmp_path / "local.jsonl").write_text("".join(json.dumps(r) + "\n" for r in local))
    (tmp_path / "claude.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in recs if r["phase"] == "e1-claude")
    )
    manifest = a.build(prereg, tmp_path)
    assert manifest["prereg"] == str(prereg) and len(manifest["files"]) == 7
    vc = json.loads((tmp_path / "variance_components.json").read_text())
    names = [c["name"] for c in vc["components"]]
    assert names[:2] == ["task", "model:task"] and names[-1] == "residual"
    assert vc["excluded_runs"] == sum(1 for r in local if r["status"] == "infra_error")
    for c in vc["components"]:
        assert c["lo"] is not None and c["hi"] is not None
    residual = vc["components"][-1]
    assert residual["lo"] == residual["hi"] == residual["estimate"]
    mde = json.loads((tmp_path / "mde.json").read_text())
    assert [(r["tasks"], r["runs"]) for r in mde["rows"]] == [(40, 1), (89, 5)]
    assert mde["rows"][1]["mde"] < mde["rows"][0]["mde"]
    assert {n["gap"] for n in mde["tasks_needed"]} == set(a.GAPS)
    scores = json.loads((tmp_path / "scores.json").read_text())["rows"]
    assert {r["model"] for r in scores} == {*DESIGN.models, *DESIGN.claude_tiers}
    assert all(r["lo"] <= r["score"] <= r["hi"] for r in scores)
    methods = {r["model"]: r["interval_method"] for r in scores}
    assert methods["claude-haiku"] == "cluster_wilson" and methods["qwen3-4b"] == "cluster_t"
    contrasts = json.loads((tmp_path / "contrasts.json").read_text())
    assert {r["contrast"] for r in contrasts["harness"]} == {
        f"harness:{m}" for m in (*DESIGN.models, *DESIGN.claude_tiers)
    }
    labels = {r["contrast"] for r in contrasts["models"]}
    assert "model:granite-4.0-micro-phi-4-mini" in labels
    assert "model:claude-haiku-qwen3-4b" in labels
    claude_rows = [r for r in contrasts["models"] if r["a"] == "claude-haiku"]
    assert all(r["label"] == "temperature-confounded" for r in claude_rows)
    assert all(r["p_holm"] >= r["p_value"] for r in contrasts["models"])
    every = [*contrasts["harness"], *contrasts["models"]]
    by_family = {r["family"]: r["interval_method"] for r in every}
    assert by_family == {
        "harness-local": "cluster_t",
        "harness-claude": "mover_wilson",
        "models-local": "cluster_t",
        "models-claude": "mover_wilson",
    }
    assert {r["family"] for r in every} == {
        "harness-local",
        "harness-claude",
        "models-local",
        "models-claude",
    }
    runs = [json.loads(ln) for ln in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert sum(r["confirmatory"] for r in runs) == len(runs)  # both arms are planned
    monkey_dir = tmp_path
    assert a.verify(monkey_dir) == []
    (tmp_path / "mde.json").write_text("{}")
    problems = a.verify(monkey_dir)
    assert any("mde.json" in p for p in problems)


def test_holm() -> None:
    assert a.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert a.holm([0.5, float("nan")]) == pytest.approx([1.0, 1.0])
    assert a.holm([]) == []
