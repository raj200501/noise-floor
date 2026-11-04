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


def plan_prereg(tmp_path, design):
    import json as _json

    path = tmp_path / "prereg.md"
    path.write_text("# test\n\n```json e1-plan\n" + _json.dumps(design.to_json()) + "\n```\n")
    return path


def fake_execute(calls, clock=None):
    import json as _json

    def run(specs, *, artifact, ledger, prefix, stamp, log):
        calls.append((prefix, len(specs)))
        artifact.parent.mkdir(parents=True, exist_ok=True)
        with artifact.open("a") as fh:
            for s in specs:
                rec = {"run_id": s.run_id, **stamp}
                if clock is not None:
                    rec["started_at"] = clock[0].isoformat()
                fh.write(_json.dumps(rec) + "\n")
        return len(specs)

    return run


def test_blocks_and_waiting() -> None:
    from datetime import UTC, datetime

    morning = datetime(2026, 11, 5, 3, 30, tzinfo=UTC)
    evening = datetime(2026, 11, 5, 23, 59, 30, tzinfo=UTC)
    assert e1.current_block(morning) == 1 and e1.current_block(evening) == 2
    assert e1.seconds_to_next_block(morning) == pytest.approx(8.5 * 3600)
    assert e1.seconds_to_next_block(evening) == pytest.approx(30)
    with pytest.raises(ValueError, match="no block"):
        e1.current_block(morning, ((5, 6),))


def never_sleep(seconds):
    raise AssertionError(f"slept {seconds} s")


def slot_runs(design):
    out: dict[str, int] = {}
    for label, specs in e1.planned_batches(e1.local_plan(design)).items():
        slot = label.split("/")[0]
        out[slot] = out.get(slot, 0) + len(specs)
    return out


def test_slot_days_and_eligibility(tmp_path) -> None:
    import json

    assert e1.slot_day("day3-b2/b0009") == 3
    pending = ["day1-b1/b0001", "day2-b1/b0002", "day2-b2/b0003", "day3-b1/b0004"]
    assert e1.eligible_batches(pending, 1, 2) == ["day1-b1/b0001", "day2-b1/b0002"]
    assert e1.eligible_batches(pending, 2, 1) == []
    assert e1.eligible_batches(pending, 1, 9) == ["day1-b1/b0001", "day2-b1/b0002", "day3-b1/b0004"]
    art = tmp_path / "a.jsonl"
    assert e1.arm_start(art) is None
    art.write_text(json.dumps({"run_id": "x"}) + "\n")
    assert e1.arm_start(art) is None  # no start times recorded
    rows = [{"run_id": "a", "started_at": "2026-11-06T01:00:00+00:00"}]
    rows.append({"run_id": "b", "started_at": "2026-11-05T23:00:00+00:00"})
    art.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert str(e1.arm_start(art)) == "2026-11-05"


@pytest.mark.usefixtures("unfrozen_ok")
def test_run_local_follows_day_and_block(tmp_path) -> None:
    import json
    from datetime import UTC, datetime

    # Batches of 2 give every (model, concurrency) group one batch in each of the 6 slots.
    design = e1.E1Design(models=MODELS[:2], families=("config",), tasks_per_family=3, batch_size=2)
    per_slot = slot_runs(design)
    assert len(per_slot) == 6 and all(per_slot.values())
    prereg = plan_prereg(tmp_path, design)
    artifact = tmp_path / "local.jsonl"
    calls: list = []
    clock = [datetime(2026, 11, 5, 3, 0, tzinfo=UTC)]

    def go(sleep, wait=True):
        return e1.run_local(
            prereg,
            artifact=artifact,
            ledger=tmp_path / "l",
            now=lambda: clock[0],
            sleep=sleep,
            log=lambda m: None,
            wait=wait,
            execute=fake_execute(calls, clock),
        )

    assert go(never_sleep, wait=False) == per_slot["day1-b1"]
    assert all(prefix.startswith("e1-day1-b1-") for prefix, _ in calls)
    assert go(never_sleep, wait=False) == 0  # day 2 and 3 slots wait for their days
    clock[0] = datetime(2026, 11, 5, 13, 0, tzinfo=UTC)
    assert go(never_sleep, wait=False) == per_slot["day1-b2"]
    clock[0] = datetime(2026, 11, 6, 3, 0, tzinfo=UTC)  # a restart: day 2 comes from the artifact
    assert go(never_sleep, wait=False) == per_slot["day2-b1"]
    times = iter(
        [
            datetime(2026, 11, 6, 13, 0, tzinfo=UTC),
            datetime(2026, 11, 7, 3, 0, tzinfo=UTC),
            datetime(2026, 11, 7, 13, 0, tzinfo=UTC),
        ]
    )
    sleeps: list = []

    def tick(seconds):
        sleeps.append(seconds)
        clock[0] = next(times)

    rest = go(tick)
    assert rest == per_slot["day2-b2"] + per_slot["day3-b1"] + per_slot["day3-b2"]
    assert len(sleeps) == 3 and all(s <= 600.0 for s in sleeps)
    lines = [json.loads(ln) for ln in artifact.read_text().splitlines()]
    assert len(lines) == len(e1.local_plan(design)) and all(r["confirmatory"] for r in lines)
    assert all(r["prereg"] == str(prereg) for r in lines)
    assert len({r["started_at"][:10] for r in lines}) == 3  # three UTC days
    assert go(never_sleep) == 0


@pytest.mark.usefixtures("unfrozen_ok")
def test_run_local_stops_when_nothing_is_banked(tmp_path) -> None:
    from datetime import UTC, datetime

    design = e1.E1Design(models=MODELS[:2], families=("config",), tasks_per_family=3)
    prereg = plan_prereg(tmp_path, design)
    morning = datetime(2026, 11, 5, 3, 0, tzinfo=UTC)
    assert (
        e1.run_local(
            prereg,
            artifact=tmp_path / "a.jsonl",
            ledger=tmp_path / "l",
            now=lambda: morning,
            sleep=never_sleep,
            log=lambda m: None,
            execute=lambda *a, **k: 0,
        )
        == 0
    )


@pytest.mark.usefixtures("unfrozen_ok")
def test_run_claude_and_plan_command(tmp_path, capsys) -> None:
    design = e1.E1Design(models=MODELS[:1], families=("config",), tasks_per_family=2)
    prereg = plan_prereg(tmp_path, design)
    calls: list = []
    n = e1.run_claude(
        prereg,
        artifact=tmp_path / "c.jsonl",
        ledger=tmp_path / "l",
        log=lambda m: None,
        execute=fake_execute(calls),
    )
    assert n == len(e1.claude_plan(design)) and calls[0][0] == "e1c"
    assert e1.main(["plan", str(prereg)]) == 0
    assert "Claude runs" in capsys.readouterr().out


def test_frozen_plan_and_pins(tmp_path) -> None:
    import json as _json

    design = e1.E1Design(models=MODELS[:1], families=("config",), tasks_per_family=2)
    prereg = plan_prereg(tmp_path, design)
    e1.check_frozen_plan(prereg, design)  # no e1-design block: a fixture, not checked
    good = {"plan_sha256": e1.plan_sha256(design), "model_pins": e1.model_pins(design)}
    text = prereg.read_text()
    prereg.write_text(text + "\n```json e1-design\n" + _json.dumps(good) + "\n```\n")
    e1.check_frozen_plan(prereg, design)
    assert len(e1.plan_sha256(design)) == 64 and set(e1.model_pins(design)) == set(MODELS[:1])
    bad = {**good, "plan_sha256": "0" * 64}
    prereg.write_text(text + "\n```json e1-design\n" + _json.dumps(bad) + "\n```\n")
    with pytest.raises(ValueError, match="plan_sha256"):
        e1.check_frozen_plan(prereg, design)
    pins = {**good, "model_pins": {MODELS[0]: {"sha256": "x"}}}
    prereg.write_text(text + "\n```json e1-design\n" + _json.dumps(pins) + "\n```\n")
    with pytest.raises(ValueError, match="pins"):
        e1.check_frozen_plan(prereg, design)
    other = e1.E1Design(models=MODELS[:1], families=("config",), tasks_per_family=2, order_seed=9)
    assert e1.plan_sha256(other) != e1.plan_sha256(design)


def test_runners_refuse_an_unfrozen_prereg(tmp_path) -> None:
    design = e1.E1Design(models=MODELS[:1], families=("config",), tasks_per_family=2)
    prereg = plan_prereg(tmp_path, design)
    with pytest.raises(ValueError, match="not frozen"):
        e1.run_local(prereg, artifact=tmp_path / "a.jsonl", ledger=tmp_path / "l", wait=False)
    with pytest.raises(ValueError, match="not frozen"):
        e1.run_claude(prereg, artifact=tmp_path / "c.jsonl", ledger=tmp_path / "l")


@pytest.mark.usefixtures("unfrozen_ok")
def test_run_local_starts_in_the_first_block(tmp_path) -> None:
    from datetime import UTC, datetime

    design = e1.E1Design(models=MODELS[:2], families=("config",), tasks_per_family=3, batch_size=2)
    per_slot = slot_runs(design)
    prereg = plan_prereg(tmp_path, design)
    calls: list = []
    clock = [datetime(2026, 11, 5, 19, 0, tzinfo=UTC)]  # launched in block 2

    def go(sleep, wait=True):
        return e1.run_local(
            prereg,
            artifact=tmp_path / "local.jsonl",
            ledger=tmp_path / "l",
            now=lambda: clock[0],
            sleep=sleep,
            log=lambda m: None,
            wait=wait,
            execute=fake_execute(calls, clock),
        )

    assert go(never_sleep, wait=False) == 0  # nothing starts in block 2 of the first day
    clock[0] = datetime(2026, 11, 6, 0, 5, tzinfo=UTC)
    assert go(never_sleep, wait=False) == per_slot["day1-b1"]  # the arm's day 1 is Nov 6
    assert all(prefix.startswith("e1-day1-b1-") for prefix, _ in calls)
