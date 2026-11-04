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
        "harness-diff-local": "cluster_t",
        "models-local": "cluster_t",
        "tiers-claude": "mover_wilson",
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


def fake_records(prereg: str = "p.md") -> list[dict]:
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
                "outcome_strict": float(rng.random() < p) if i % 3 else 0.0,
                "status": status,
                "prereg": prereg,
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
    with pytest.raises(ValueError, match="banked more than once"):
        a.runs_table(
            [*recs, recs[0]], prereg="p.md", planned=planned, block_hours=[[0, 12], [12, 24]]
        )


@needs_r
def test_build_and_verify(tmp_path: Path) -> None:
    prereg = write_prereg(tmp_path / "prereg.md")
    recs = fake_records(str(prereg))
    local = [r for r in recs if r["phase"] == "e1"]
    (tmp_path / "local.jsonl").write_text("".join(json.dumps(r) + "\n" for r in local))
    (tmp_path / "claude.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in recs if r["phase"] == "e1-claude")
    )
    manifest = a.build(prereg, tmp_path)
    assert manifest["prereg"] == str(prereg) and len(manifest["files"]) == 10
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
    for n in mde["tasks_needed"]:
        assert n["lo"] is not None and n["lo"] <= n["tasks"] <= n["hi"]
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
    for r in [*contrasts["harness"], *contrasts["models"]]:
        assert r["supported"] == (r["p_holm"] < 0.05)
        assert r["interval_excludes_zero"] == (r["lo"] > 0 or r["hi"] < 0)
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
    sens = json.loads((tmp_path / "sensitivity.json").read_text())
    assert "(1 | batch)" in sens["batch_random_effect"]["formula"]
    infra = sum(1 for r in local if r["status"] == "infra_error")
    assert sens["infra_error_as_failure"]["n_runs"] == sens["outcome_strict"]["n_runs"] + infra
    strict = {c["name"]: c for c in sens["outcome_strict"]["components"]}
    assert set(strict) == {c["name"] for c in vc["components"]}
    e3 = json.loads((tmp_path / "e3_bounds.json").read_text())
    assert e3["source"] == "e1" and e3["prereg"] == str(prereg)
    bounds = {b["model"]: b for b in e3["entries"]}
    assert set(bounds) == set(DESIGN.models)
    assert all(b["ci_lo"] <= b["lower"] <= b["upper"] <= b["ci_hi"] for b in bounds.values())
    runs = [json.loads(ln) for ln in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert sum(r["confirmatory"] for r in runs) == len(runs)  # both arms are planned
    # Mock-task labels on every output (non-negotiable 4).
    assert all(r["mock_tasks"] for r in runs)
    for name in ("variance_components.json", "scores.json", "e3_bounds.json", "sampling.json"):
        assert "mock task" in json.loads((tmp_path / name).read_text())["tasks"]
    monkey_dir = tmp_path
    assert a.verify(monkey_dir) == []
    (tmp_path / "mde.json").write_text("{}")
    problems = a.verify(monkey_dir)
    assert any("mde.json" in p for p in problems)


def test_holm() -> None:
    assert a.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert a.holm([0.5, float("nan")]) == pytest.approx([1.0, 1.0])
    assert a.holm([]) == []


def test_sensitivity_failure_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    planned = {s.run_id: s.phase for s in local_plan(DESIGN)}
    rows = a.runs_table(
        fake_records(), prereg="p.md", planned=planned, block_hours=[[0, 12], [12, 24]]
    )

    def fail(data: object, settings: object) -> None:
        raise RuntimeError("Response is constant")

    monkeypatch.setattr(a, "decomposition", fail)
    out = a.sensitivity_tables(rows, SETTINGS)
    assert set(out) == {"batch_random_effect", "outcome_strict", "infra_error_as_failure"}
    assert all(v["error"] == "RuntimeError: Response is constant" for v in out.values())


def test_runs_table_confirmatory_rules() -> None:
    spec = local_plan(DESIGN)[0]
    base = {
        **spec.to_json(),
        "outcome": 1.0,
        "status": "ok",
        "prereg": "p.md",
        "started_at": "2026-11-05T03:00:00+00:00",
    }
    planned = {spec.run_id: spec.phase}
    hours = [[0, 12], [12, 24]]

    def one(rec: dict, frozen_at: str | None = "2026-11-05T00:00:00Z") -> dict:
        return a.runs_table(
            [rec], prereg="p.md", planned=planned, block_hours=hours, frozen_at=frozen_at
        )[0]

    assert one(base)["confirmatory"] and not one(base)["start_from_batch"]
    assert not one({**base, "prereg": "other.md"})["confirmatory"]
    assert not one({**base, "started_at": "2026-11-04T23:00:00+00:00"})["confirmatory"]
    assert one({**base, "started_at": "2026-11-04T23:00:00+00:00"}, None)["confirmatory"]
    placeholder = {k: v for k, v in base.items() if k != "started_at"}
    row = one(
        {**placeholder, "batch_started_at": "2026-11-05T13:00:00+00:00", "status": "infra_error"}
    )
    assert row["start_from_batch"] and row["block"] == "b2" and row["confirmatory"]
    # Day and block follow the batch's start even when the run itself started later.
    late = one(
        {
            **base,
            "batch_started_at": "2026-11-05T11:58:00+00:00",
            "started_at": "2026-11-05T12:03:00+00:00",
        }
    )
    assert late["block"] == "b1" and late["day"] == "2026-11-05"


def test_claude_exclusions_and_runs_left_out() -> None:
    def claude(rid: str, start: str, **extra: object) -> dict:
        base = {"run_id": rid, "model": "claude-haiku", "started_at": start}
        return {**base, "factors": {"steps": "8"}, **extra}

    recs = [
        claude("b", "2026-11-06T01:00:00+00:00", claude={"model_id": "m1", "max_turns": 8}),
        claude("a", "2026-11-05T01:00:00+00:00", model_ids=["m1"], max_steps_used=8),
        claude("c", "2026-11-07T01:00:00+00:00", claude={"model_id": "m2", "max_turns": 8}),
        claude("d", "2026-11-08T01:00:00+00:00", model_ids=["m1"], max_steps_used=6),
        {"run_id": "l", "model": "qwen3-4b", "started_at": "2026-11-05T00:00:00+00:00"},
    ]
    assert a.claude_exclusions(recs) == {"c": "model_changed", "d": "step_budget"}
    rows = [
        {"confirmatory": True, "backend": "claude", "status": "ok", "excluded": "model_changed"},
        {"confirmatory": True, "backend": "local", "status": "infra_error", "excluded": None},
        {"confirmatory": True, "backend": "local", "status": "ok", "excluded": None},
        {"confirmatory": False, "backend": "local", "status": "infra_error", "excluded": None},
    ]
    assert a.exclusion_counts(rows, ["infra_error"]) == {
        "local": {"infra_error": 1},
        "claude": {"model_changed": 1},
    }
    assert a._usable(rows, ["infra_error"]) == [rows[2]]


def test_no_outcome_is_read_before_both_arms_end(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    planned = {"l1": "e1", "l2": "e1", "c1": "e1-claude", "c2": "e1-claude"}
    frozen = datetime(2026, 11, 5, tzinfo=UTC)
    soon = frozen + timedelta(days=10)
    recs = [{"run_id": "l1"}, {"run_id": "c1"}]
    out = a.arms_not_ended(recs, planned, tmp_path, frozen, soon)
    assert out == [
        "the local arm has not ended (1 of 2 banked)",
        "the Claude slice has not ended (1 of 2 banked)",
    ]
    (tmp_path / "stops-local.jsonl").write_text('{"kind": "day_limit", "at": "x"}\n')
    later = frozen + timedelta(days=45)
    assert a.arms_not_ended(recs, planned, tmp_path, frozen, later) == []
    (tmp_path / "stops-claude.jsonl").write_text('{"kind": "end_point", "at": "x"}\n')
    assert a.arms_not_ended(recs, planned, tmp_path, frozen, soon) == []
    everything = [{"run_id": r} for r in planned]
    assert a.arms_not_ended(everything, planned, tmp_path / "none", None, soon) == []


def test_compute_refuses_while_an_arm_runs(tmp_path: Path) -> None:
    prereg = write_prereg(tmp_path / "prereg.md")
    recs = [r for r in fake_records(str(prereg)) if r["phase"] == "e1"][:-1]
    (tmp_path / "local.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    with pytest.raises(ValueError, match="local arm has not ended"):
        a.compute(prereg, tmp_path)


def test_sampling_table_pairs_replicates_with_their_runs() -> None:
    def row(rid: str, model: str, task: str, outcome: int, temp: str, **extra: object) -> dict:
        factors = {"harness": "minimal", "timeout": "300.0", "steps": "15", "temperature": temp}
        base = {"run_id": rid, "model_id": model, "task": task, "outcome": outcome}
        return {
            **base,
            "factors": {**factors, "concurrency": "1" if temp == "0.0" else "2"},
            "confirmatory": True,
            "status": "ok",
            "excluded": None,
            "replicate": rid.endswith("-rep"),
            **extra,
        }

    rows = []
    for t in range(12):
        for m in ("a", "b", "c"):
            hot = int((t + len(m)) % 3 == 0)  # one discordant pair in three at 0.7
            rows += [
                row(f"{m}{t}-cold", m, f"t{t}", 1, "0.0"),
                row(f"{m}{t}-cold-rep", m, f"t{t}", 1, "0.0"),
                row(f"{m}{t}-hot", m, f"t{t}", 1, "0.7"),
                row(f"{m}{t}-hot-rep", m, f"t{t}", 1 - hot, "0.7"),
            ]
    rows.append(row("lost-rep", "a", "t0", 0, "0.7"))  # its planned run is not usable
    rows.append(row("x-hot", "a", "t1", 1, "0.7"))
    rows.append(row("x-hot-rep", "a", "t1", 0, "0.7", status="infra_error"))
    settings = {"exclude_status": ["infra_error"], "level": 0.95}
    reference = {"temperature": "0.0", "concurrency": "1"}
    out = a.sampling_table(rows, settings, reference=reference, assumed=0.2)
    cold, hot = sorted(out, key=lambda r: r["configuration"]["temperature"])
    assert cold["pairs"] == hot["pairs"] == 36 and cold["discordant"] == 0
    assert cold["within_var"] == 0.0 and cold["mde_reference"]
    assert cold["mde_assumed_within_var"] == 0.2 and hot["mde_assumed_within_var"] is None
    assert hot["discordant"] == 12 and hot["within_var"] == pytest.approx(12 / 36 / 2)
    assert hot["lo"] < hot["within_var"] < hot["hi"] <= 0.5
    assert hot["interval_method"].startswith("half of")
    # Replicates never reach the decomposition, scores, or contrasts.
    assert all(not r["replicate"] for r in a._usable(rows, ["infra_error"]))


def test_contrasts_include_harness_differences_and_tier_pairs() -> None:
    rng = np.random.default_rng(5)
    rows = []

    def add(model: str, backend: str, task: str, harness: str, k: int) -> None:
        rows.append(
            {
                "run_id": f"{model}-{task}-{harness}-{k}",
                "model_id": model,
                "backend": backend,
                "task": task,
                "factors": {"harness": harness, "steps": "8"},
                "outcome": int(rng.random() < 0.5),
                "confirmatory": True,
                "status": "ok",
                "excluded": None,
                "replicate": False,
            }
        )

    for t in range(8):
        for harness in ("minimal", "native"):
            for k in range(2):
                add("qwen3-4b", "local", f"t{t}", harness, k)
                add("granite-4.0-micro", "local", f"t{t}", harness, k)
            add("claude-haiku", "claude", f"t{t}", harness, 0)
            add("claude-sonnet", "claude", f"t{t}", harness, 0)
    settings = {
        "level": 0.95,
        "exclude_status": ["infra_error"],
        "contrast_interval": {
            "harness-local": "cluster_t",
            "harness-claude": "mover_wilson",
            "harness-diff-local": "cluster_t",
            "models-local": "cluster_t",
            "tiers-claude": "mover_wilson",
            "models-claude": "mover_wilson",
        },
    }
    out = a.contrasts_table(rows, settings)
    (diff,) = out["harness_differences"]
    assert diff["family"] == "harness-diff-local" and diff["n_tasks"] == 8
    assert {diff["a"], diff["b"]} == {"qwen3-4b", "granite-4.0-micro"}
    assert diff["p_holm"] == diff["p_value"] and "supported" in diff
    tiers = [r for r in out["models"] if r["family"] == "tiers-claude"]
    assert sorted(r["contrast"] for r in tiers) == [
        "tiers:minimal:claude-haiku-claude-sonnet",
        "tiers:native:claude-haiku-claude-sonnet",
    ]
    assert all(r["interval_method"].startswith("mover") for r in tiers)
    assert all(r["p_holm"] >= r["p_value"] for r in tiers)


@needs_r
@pytest.mark.usefixtures("unfrozen_ok")
def test_post_check_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import functools

    from noisefloor import post_check as pc
    from noisefloor import provenance

    lax = functools.partial(provenance.single_environment, allow_dirty=True)
    monkeypatch.setattr(pc, "single_environment", lax)
    prereg = write_prereg(tmp_path / "prereg.md")
    recs = fake_records(str(prereg))
    (tmp_path / "local.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in recs if r["phase"] == "e1")
    )
    (tmp_path / "claude.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in recs if r["phase"] == "e1-claude")
    )
    a.build(prereg, tmp_path)
    config = pc.prepare(prereg, tmp_path)
    assert config["design"]["n_models"] == len(DESIGN.models)
    assert set(config["planted"]["interactions"]) == set(pc.INTERACTION_FACTORS)
    assert pc.prepare(prereg, tmp_path) == config  # written once, then reused
    seeds = [980000, 980001]
    assert pc.run(prereg, tmp_path, log=lambda m: None, seeds=seeds) == 2
    assert pc.run(prereg, tmp_path, log=lambda m: None, seeds=seeds) == 0
    out = pc.bank(prereg, tmp_path, seeds=seeds)
    assert out["synthetic"] and set(out["components"]) >= {"task", "harness"}
    assert all(c["datasets"] == 2 for c in out["components"].values())
    assert json.loads((tmp_path / "post_check.json").read_text())["prereg"] == str(prereg)


def test_scores_count_timeout_runs_with_outcome_one() -> None:
    rows = []
    for t in range(4):
        for k, (status, outcome) in enumerate(
            [("ok", 1), ("timeout", 1), ("timeout", 0), ("infra_error", 0)]
        ):
            rows.append(
                {
                    "run_id": f"r{t}{k}",
                    "model_id": "qwen3-4b",
                    "backend": "local",
                    "task": f"t{t}",
                    "factors": {},
                    "outcome": outcome,
                    "status": status,
                    "confirmatory": True,
                    "excluded": None,
                    "replicate": False,
                }
            )
    settings = {"level": 0.95, "exclude_status": ["infra_error"], "score_interval": "cluster_t"}
    (row,) = a.scores_table(rows, settings)
    assert row["timeout_runs"] == 8 and row["timeout_runs_with_outcome_1"] == 4
    assert row["score"] == pytest.approx(2 / 3)


def test_schedule_check_flags_batches_off_their_slot() -> None:
    def row(batch: str, slot: str, day: str, block: str) -> dict:
        return {
            "confirmatory": True,
            "backend": "local",
            "batch_id": batch,
            "planned_slot": slot,
            "day": day,
            "block": block,
        }

    kept = [
        row("a", "day1-b1", "2026-11-06", "b1"),
        row("b", "day2-b2", "2026-11-07", "b2"),
        row("c", "day3-b1", "2026-11-08", "b1"),
    ]
    assert a.schedule_check(kept) == {
        "days": 3,
        "batches": 3,
        "batches_off_plan": 0,
        "simulated_design": True,
    }
    late = [*kept, row("d", "day2-b1", "2026-11-08", "b1")]  # overran into day 3
    out = a.schedule_check(late)
    assert out["batches_off_plan"] == 1 and not out["simulated_design"]
    longer = [*kept, row("e", "day3-b2", "2026-11-09", "b2")]
    assert a.schedule_check(longer)["days"] == 4
