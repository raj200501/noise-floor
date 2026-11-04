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
    calls: list = []

    def nothing(*a, **k):
        calls.append(1)
        return 0

    def go():
        return e1.run_local(
            prereg,
            artifact=tmp_path / "a.jsonl",
            ledger=tmp_path / "l",
            now=lambda: morning,
            sleep=never_sleep,
            log=lambda m: None,
            execute=nothing,
        )

    assert go() == 0 and len(calls) == 1
    stop = e1.open_stop(tmp_path / "stops-a.jsonl")
    assert stop is not None and stop["kind"] == "batch_failure"
    assert go() == 0 and len(calls) == 1  # a stopped arm starts nothing


def local_runner(tmp_path, prereg, clock, execute, messages=None):
    def go():
        return e1.run_local(
            prereg,
            artifact=tmp_path / "local.jsonl",
            ledger=tmp_path / "l",
            now=lambda: clock[0],
            sleep=never_sleep,
            log=(messages.append if messages is not None else lambda m: None),
            wait=False,
            execute=execute,
        )

    return go


@pytest.mark.usefixtures("unfrozen_ok")
def test_a_failed_batch_stops_the_arm_until_a_resume(tmp_path) -> None:
    from datetime import UTC, datetime

    from noisefloor.orchestrate import BatchFailedError, banked_ids

    design = e1.E1Design(models=MODELS[:2], families=("config",), tasks_per_family=3, batch_size=2)
    prereg = plan_prereg(tmp_path, design)
    clock = [datetime(2026, 11, 5, 3, 0, tzinfo=UTC)]
    calls: list = []
    healthy = fake_execute(calls, clock)
    fail = [True]

    def flaky(specs, **kw):
        done = banked_ids(kw["artifact"])  # like execute, never re-run a banked run
        specs = [s for s in specs if s.run_id not in done]
        if fail[0]:
            healthy(specs[:1], **kw)  # one run banked, then the server dies
            raise BatchFailedError("b-x", "3 infra_error runs in a row", 1)
        return healthy(specs, **kw)

    messages: list = []
    go = local_runner(tmp_path, prereg, clock, flaky, messages)
    assert go() == 1
    stops = tmp_path / "stops-local.jsonl"
    stop = e1.open_stop(stops)
    assert stop["kind"] == "batch_failure" and stop["reason"] == "3 infra_error runs in a row"
    assert stop["batch"].startswith("day1-b1/") and "stopped for repair" in messages[-1]
    assert go() == 0 and len(calls) == 1
    with pytest.raises(ValueError, match="not stopped"):
        e1.resume(tmp_path / "stops-none.jsonl", "x", clock[0])
    fail[0] = False
    e1.resume(stops, "restarted the sandbox", clock[0])
    assert e1.open_stop(stops) is None
    assert go() == slot_runs(design)["day1-b1"] - 1  # the unbanked runs ran again


@pytest.mark.usefixtures("unfrozen_ok")
def test_infra_error_share_stops_the_arm(tmp_path) -> None:
    import json
    from datetime import UTC, datetime

    design = e1.E1Design(models=MODELS[:2], families=("config",), tasks_per_family=3, batch_size=2)
    prereg = plan_prereg(tmp_path, design)
    artifact = tmp_path / "local.jsonl"
    start = "2026-11-05T01:00:00+00:00"
    rows = [
        {"run_id": f"old{i}", "started_at": start, "batch_started_at": start, "status": status}
        for i, status in enumerate(["infra_error"] * 5 + ["ok"] * 15)
    ]
    artifact.write_text("".join(json.dumps(r) + "\n" for r in rows))
    clock = [datetime(2026, 11, 5, 3, 0, tzinfo=UTC)]
    calls: list = []
    go = local_runner(tmp_path, prereg, clock, fake_execute(calls, clock))
    assert e1.infra_counts(artifact, tmp_path / "stops-local.jsonl") == (20, 5)
    assert go() == 0 and calls == []
    stop = e1.open_stop(tmp_path / "stops-local.jsonl")
    assert stop["kind"] == "infra_share" and stop["infra_error"] == 5 and stop["runs"] == 20
    # After a resume only runs banked since then count.
    e1.resume(tmp_path / "stops-local.jsonl", "fixed the proxy", clock[0])
    assert e1.infra_counts(artifact, tmp_path / "stops-local.jsonl") == (0, 0)
    assert go() > 0
    # Four infra_error runs, or five at no more than 5 percent, do not stop an arm.
    other = tmp_path / "other.jsonl"
    other.write_text("".join(json.dumps(r) + "\n" for r in rows[1:]))
    assert not e1._infra_stop(other, tmp_path / "s1.jsonl", clock[0], lambda m: None)
    many = rows + [{**rows[-1], "run_id": f"more{i}"} for i in range(80)]
    other.write_text("".join(json.dumps(r) + "\n" for r in many))
    assert not e1._infra_stop(other, tmp_path / "s2.jsonl", clock[0], lambda m: None)


@pytest.mark.usefixtures("unfrozen_ok")
def test_no_batch_starts_after_day_seven(tmp_path) -> None:
    import json
    from datetime import UTC, datetime

    design = e1.E1Design(models=MODELS[:2], families=("config",), tasks_per_family=3, batch_size=2)
    prereg = plan_prereg(tmp_path, design)
    first = "2026-11-01T01:00:00+00:00"
    row = {"run_id": "x", "started_at": first, "batch_started_at": first, "status": "ok"}
    (tmp_path / "local.jsonl").write_text(json.dumps(row) + "\n")
    clock = [datetime(2026, 11, 8, 3, 0, tzinfo=UTC)]  # day 8 of the arm
    calls: list = []
    assert local_runner(tmp_path, prereg, clock, fake_execute(calls, clock))() == 0
    stops = tmp_path / "stops-local.jsonl"
    stop = e1.open_stop(stops)
    assert calls == [] and stop["kind"] == "day_limit"
    assert stop["unbanked"] == len(e1.local_plan(design))
    with pytest.raises(ValueError, match="has ended"):
        e1.resume(stops, "more time", clock[0])


@pytest.mark.usefixtures("unfrozen_ok")
def test_one_runner_at_a_time(tmp_path) -> None:
    from datetime import UTC, datetime

    design = e1.E1Design(models=MODELS[:2], families=("config",), tasks_per_family=3, batch_size=2)
    prereg = plan_prereg(tmp_path, design)
    clock = [datetime(2026, 11, 5, 3, 0, tzinfo=UTC)]
    go = local_runner(tmp_path, prereg, clock, fake_execute([], clock))
    with e1.arm_lock(tmp_path / "local.jsonl"), pytest.raises(e1.ArmBusyError):
        go()
    assert go() > 0  # the lock is released with its holder


@pytest.mark.usefixtures("unfrozen_ok")
def test_stop_and_resume_commands(tmp_path, monkeypatch, capsys) -> None:
    from noisefloor import preregs

    design = e1.E1Design(models=MODELS[:1], families=("config",), tasks_per_family=2)
    prereg = plan_prereg(tmp_path, design)
    monkeypatch.setattr(e1, "ARTIFACTS", tmp_path)
    assert e1.main(["stop", "local", "--reason", "grader rejects valid answers"]) == 0
    stop = e1.open_stop(tmp_path / "stops-local.jsonl")
    assert stop["kind"] == "defect" and "grader" in stop["reason"]
    assert e1.main(["local", str(prereg), "--no-wait"]) == 3  # stopped
    with pytest.raises(ValueError, match="frozen amendment"):
        e1.main(["resume", "local", "--note", "fixed"])
    seen: list = []
    entry = {"sha256": "0" * 64}
    monkeypatch.setattr(preregs, "require_frozen", lambda p, *a, **k: seen.append(p) or entry)
    amendment = tmp_path / "0009-fix.md"
    assert e1.main(["resume", "local", "--note", "fixed", "--amendment", str(amendment)]) == 0
    resumed = e1.jsonl.read(tmp_path / "stops-local.jsonl")[-1]
    assert seen == [amendment] and resumed["kind"] == "resume" and resumed["after"] == stop["at"]
    assert resumed["amendment"] == str(amendment)
    with e1.arm_lock(tmp_path / "claude.jsonl"):
        assert e1.main(["claude", str(prereg)]) == 4
    assert "another runner" in capsys.readouterr().err
    with pytest.raises(ValueError, match="unknown stop kind"):
        e1.record_stop(tmp_path / "s.jsonl", "whim", "x", e1.datetime.now(e1.UTC))


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


def test_frozen_plan_and_pins(tmp_path, monkeypatch) -> None:
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
    # A frozen preregistration (in the chain) must carry both keys.
    chain = tmp_path / "CHAIN.jsonl"
    chain.write_text(_json.dumps({"file": str(prereg), "sha256": "x"}) + "\n")
    missing = {"plan_sha256": good["plan_sha256"]}
    prereg.write_text(text + "\n```json e1-design\n" + _json.dumps(missing) + "\n```\n")
    e1.check_frozen_plan(prereg, design)  # outside the project chain: keys are optional
    import functools

    from noisefloor import preregs

    monkeypatch.setattr(preregs, "chain_entry", functools.partial(preregs.chain_entry, chain=chain))
    with pytest.raises(ValueError, match="no model_pins"):
        e1.check_frozen_plan(prereg, design)
    prereg.write_text(text)
    with pytest.raises(ValueError, match="e1-design"):
        e1.check_frozen_plan(prereg, design)
    prereg.write_text(text + "\n```json e1-design\n" + _json.dumps(good) + "\n```\n")
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


def test_claude_plan_is_ordered_in_complete_groups(design: e1.E1Design) -> None:
    plan = e1.claude_plan(design)
    size = design.claude_runs_per_task
    for cut in range(len(plan) + 1):
        groups = collections.Counter((s.model, s.family, s.task_seed) for s in plan[:cut])
        assert sum(1 for n in groups.values() if n % size) <= 1
        full = collections.Counter(k[0] for k, n in groups.items() if n == size)
        counts = [full.get(t, 0) for t in design.claude_tiers]
        assert max(counts) - min(counts) <= 1
    assert e1.claude_plan(design) == plan  # fixed by the order seed


def write_rows(path, rows) -> None:
    import json

    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.mark.usefixtures("unfrozen_ok")
def test_claude_slice_end_point_and_changes(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime, timedelta

    design = e1.E1Design(models=MODELS[:1], families=("config",), tasks_per_family=2)
    prereg = plan_prereg(tmp_path, design)
    artifact = tmp_path / "claude.jsonl"
    stops = tmp_path / "stops-claude.jsonl"
    frozen = datetime(2026, 11, 5, tzinfo=UTC)
    clock = [frozen + timedelta(days=3)]
    monkeypatch.setattr(e1, "frozen_at", lambda path: frozen)
    calls: list = []

    def go():
        return e1.run_claude(
            prereg,
            artifact=artifact,
            ledger=tmp_path / "l",
            now=lambda: clock[0],
            log=lambda m: None,
            execute=fake_execute(calls),
        )

    def run(rid, model_id, cli=None, harness="claude_code"):
        if harness == "minimal":
            return {"run_id": rid, "model": "claude-haiku", "model_ids": [model_id]}
        return {
            "run_id": rid,
            "model": "claude-haiku",
            "claude": {"model_id": model_id, "cli_version": cli},
        }

    # A new Claude Code version stops the slice until a resume accepts it.
    write_rows(artifact, [run("a", "m1", "2.0.1"), run("b", "m1", "2.0.2")])
    assert go() == 0 and calls == []
    stop = e1.open_stop(stops)
    assert stop["kind"] == "cli_changed" and stop["to"] == "2.0.2"
    e1.resume(stops, "accepted the new CLI as a deviation", clock[0])
    assert e1.claude_changes(artifact, stops) == (None, None)
    assert go() == len(e1.claude_plan(design)) and calls
    # Another model ID for a tier ends the slice for good.
    write_rows(artifact, [run("a", "m1", "2.0.1"), run("c", "m2", harness="minimal")])
    stops.unlink()
    assert go() == 0
    stop = e1.open_stop(stops)
    assert stop["kind"] == "end_point" and "served by m2 after m1" in stop["reason"]
    with pytest.raises(ValueError, match="has ended"):
        e1.resume(stops, "x", clock[0])
    # The slice starts nothing 45 days after the freeze.
    write_rows(artifact, [run("a", "m1", "2.0.1")])
    stops.unlink()
    clock[0] = frozen + timedelta(days=e1.CLAUDE_END_DAYS)
    assert go() == 0 and e1.open_stop(stops)["reason"] == "45 days after the freeze"


def test_frozen_at_reads_the_chain_entry(tmp_path, monkeypatch) -> None:
    import functools
    import json
    from datetime import UTC, datetime

    from noisefloor import preregs

    prereg = tmp_path / "p.md"
    prereg.write_text("x")
    chain = tmp_path / "CHAIN.jsonl"
    chain.write_text(json.dumps({"file": str(prereg), "frozen_at": "2026-11-05T01:02:03Z"}) + "\n")
    assert e1.frozen_at(prereg) is None
    monkeypatch.setattr(preregs, "chain_entry", functools.partial(preregs.chain_entry, chain=chain))
    assert e1.frozen_at(prereg) == datetime(2026, 11, 5, 1, 2, 3, tzinfo=UTC)


@pytest.mark.usefixtures("unfrozen_ok")
def test_a_stopped_local_arm_still_ends_after_day_seven(tmp_path) -> None:
    from datetime import UTC, datetime

    design = e1.E1Design(models=MODELS[:2], families=("config",), tasks_per_family=3, batch_size=2)
    prereg = plan_prereg(tmp_path, design)
    first = "2026-11-01T01:00:00+00:00"
    write_rows(
        tmp_path / "local.jsonl",
        [{"run_id": "x", "started_at": first, "batch_started_at": first, "status": "ok"}],
    )
    stops = tmp_path / "stops-local.jsonl"
    e1.record_stop(
        stops, "batch_failure", "server did not start", datetime(2026, 11, 3, tzinfo=UTC)
    )
    clock = [datetime(2026, 11, 8, 3, 0, tzinfo=UTC)]
    go = local_runner(tmp_path, prereg, clock, fake_execute([], clock))
    assert go() == 0 and e1.open_stop(stops)["kind"] == "day_limit"
    assert go() == 0 and len(e1.jsonl.read(stops)) == 2  # recorded once
