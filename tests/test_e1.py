import collections

import pytest

from noisefloor import e1
from noisefloor.orchestrate import make_batches
from noisefloor.recovery import TWO_LEVEL
from noisefloor.runner import RunSpec

MODELS = ("qwen3-4b", "granite-4.0-micro", "phi-4-mini")


@pytest.fixture(scope="module")
def design() -> e1.E1Design:
    return e1.E1Design(models=MODELS, tasks_per_family=4)


def test_tasks_and_manifest(design: e1.E1Design) -> None:
    tasks = e1.tasks(design)
    assert len(tasks) == len(design.families) * 4
    assert all(t.seed in e1.CONFIRMATORY_SEEDS for t in tasks)
    assert collections.Counter(t.difficulty for t in tasks if t.family == "pointer") == {
        1: 2,
        2: 1,
        3: 1,
    }
    manifest = e1.manifest(design)
    assert manifest["n_tasks"] == len(tasks) and manifest == e1.manifest(design)


def test_local_plan_is_balanced(design: e1.E1Design) -> None:
    plan = e1.local_plan(design)
    assert len(plan) == len(MODELS) * len(e1.tasks(design)) * 8
    assert len({s.run_id for s in plan}) == len(plan)
    assert plan == e1.local_plan(design)  # deterministic
    cells = collections.defaultdict(list)
    for spec in plan:
        assert spec.phase == "e1" and spec.harness in ("minimal", "inspect")
        assert set(spec.factors) == set(TWO_LEVEL)
        for name, value in spec.factors.items():
            assert value in {str(v) for v in design.levels[name]}
        cells[(spec.model, spec.family, spec.task_seed, spec.difficulty)].append(spec)
    for runs in cells.values():
        assert len(runs) == 8
        for name in TWO_LEVEL:
            assert collections.Counter(r.factors[name] for r in runs).most_common(1)[0][1] == 4
    slots = [s.block.split("/")[0] for s in plan]
    assert slots == sorted(slots, key=design.slots.index)  # slot by slot
    assert set(slots) == set(design.slots)


def test_planned_batches_survive_make_batches(design: e1.E1Design) -> None:
    plan = e1.local_plan(design)
    batches = make_batches(plan, "e1", max_size=design.batch_size)
    planned = collections.Counter(s.block for s in plan)
    assert len(batches) == len(planned)
    for batch in batches:
        assert len({s.block for s in batch.runs}) == 1
        assert len({(s.model, s.concurrency) for s in batch.runs}) == 1


def test_make_batches_splits_on_block_change() -> None:
    base = dict(
        phase="t",
        family="config",
        task_seed=1,
        difficulty=1,
        model="qwen3-4b",
        harness="minimal",
        temperature=0.0,
        seed=0,
        timeout_s=10.0,
        max_steps=3,
        threads=1,
        concurrency=1,
    )
    specs = [RunSpec(run_id=f"r{i}", block=b, **base) for i, b in enumerate("aab")]
    assert [len(b.runs) for b in make_batches(specs, "p")] == [2, 1]


def test_claude_plan_is_balanced(design: e1.E1Design) -> None:
    plan = e1.claude_plan(design)
    assert len(plan) == len(design.claude_tiers) * len(e1.tasks(design)) * 2
    per_task = collections.defaultdict(list)
    for spec in plan:
        assert spec.phase == "e1-claude" and spec.max_steps == 8 and spec.concurrency == 1
        assert spec.factors["temperature"] == "cli_default"
        per_task[(spec.model, spec.family, spec.task_seed)].append(spec)
    for runs in per_task.values():
        assert {r.factors["harness"] for r in runs} == {"minimal", "native"}
        assert len({r.factors["timeout"] for r in runs}) == 2
    with pytest.raises(ValueError, match="even"):
        e1.claude_plan(e1.E1Design(models=MODELS, claude_runs_per_task=3))


def test_design_validation_and_json() -> None:
    with pytest.raises(ValueError, match="not pinned"):
        e1.E1Design(models=("gpt-x",))
    with pytest.raises(ValueError, match="not Claude tiers"):
        e1.E1Design(models=MODELS, claude_tiers=("claude-x",))
    with pytest.raises(ValueError, match="levels"):
        e1.E1Design(models=MODELS, levels={"harness": ("a", "b")})
    with pytest.raises(ValueError, match="tasks_per_family"):
        e1.E1Design(models=MODELS, tasks_per_family=0)
    data = e1.E1Design(models=MODELS).to_json()
    assert data["models"] == list(MODELS) and data["levels"]["steps"] == [8, 15]


def test_sixteen_runs_per_cell() -> None:
    full = e1.E1Design(models=MODELS[:2], tasks_per_family=2, runs_per_cell=16)
    plan = e1.local_plan(full)
    assert len(plan) == 2 * len(e1.tasks(full)) * 16 and len({s.run_id for s in plan}) == len(plan)
    cells = collections.Counter((s.model, s.family, s.task_seed) for s in plan)
    assert set(cells.values()) == {16}
    assert e1.E1Design.from_json(full.to_json()) == full
    with pytest.raises(ValueError, match="runs_per_cell"):
        e1.E1Design(models=MODELS, runs_per_cell=4)
