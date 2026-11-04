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
    assert e1.seconds_since_midnight(morning) == pytest.approx(3.5 * 3600)
    assert e1.seconds_to_midnight(morning) == pytest.approx(20.5 * 3600)
    assert e1.seconds_to_midnight(evening) == pytest.approx(30)
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
    clock = [datetime(2026, 11, 5, 0, 5, tzinfo=UTC)]

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
    morning = datetime(2026, 11, 5, 0, 5, tzinfo=UTC)
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
    clock = [datetime(2026, 11, 5, 0, 5, tzinfo=UTC)]
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
    clock = [datetime(2026, 11, 5, 0, 5, tzinfo=UTC)]
    go = local_runner(tmp_path, prereg, clock, fake_execute([], clock))
    with e1.arm_lock(tmp_path / "local.jsonl"), pytest.raises(e1.ArmBusyError):
        go()
    assert go() > 0  # the lock is released with its holder


@pytest.mark.usefixtures("unfrozen_ok")
def test_stop_and_resume_commands(tmp_path, monkeypatch, capsys) -> None:
    import json

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
    amendment.write_text("---\namends: 0003\n---\n# a fix of another file\n")
    with pytest.raises(ValueError, match="does not amend prereg 0004"):
        e1.main(["resume", "local", "--note", "fixed", "--amendment", str(amendment)])
    answer = "\n```json defect-resume\n" + json.dumps({"at": stop["at"]}) + "\n```\n"
    amendment.write_text("---\namends: 0004\ndate: 2026-11-20\n---\n# fix\n" + answer)
    seen.clear()
    assert e1.main(["resume", "local", "--note", "fixed", "--amendment", str(amendment)]) == 0
    resumed = e1.jsonl.read(tmp_path / "stops-local.jsonl")[-1]
    assert seen == [amendment] and resumed["kind"] == "resume" and resumed["after"] == stop["at"]
    assert resumed["amendment"] == str(amendment)
    # A note (runs lost with a container, rerun under their ids) stops nothing.
    assert e1.main(["note", "local", "--reason", "runs r1, r2 lost with the container"]) == 0
    assert e1.jsonl.read(tmp_path / "stops-local.jsonl")[-1]["kind"] == "note"
    assert e1.open_stop(tmp_path / "stops-local.jsonl") is None
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
    tasks = e1.manifest(design)
    good = {
        "plan_sha256": e1.plan_sha256(design),
        "model_pins": e1.model_pins(design),
        "task_manifest_sha256": tasks["sha256"],
        "suite_version": tasks["version"],
    }
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
    tasks_changed = {**good, "task_manifest_sha256": "0" * 64}
    prereg.write_text(text + "\n```json e1-design\n" + _json.dumps(tasks_changed) + "\n```\n")
    with pytest.raises(ValueError, match="generated tasks differ"):
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
    clock[0] = datetime(2026, 11, 6, 3, 0, tzinfo=UTC)
    assert go(never_sleep, wait=False) == 0  # nor late in block 1: day 1 gets a full block
    sleeps: list = []

    class DoneError(Exception):
        pass

    def until_midnight(seconds):
        sleeps.append(seconds)
        if len(sleeps) > 1:
            raise DoneError  # day 1's first block is done; the arm would now wait for block 2
        clock[0] = datetime(2026, 11, 7, 0, 0, 30, tzinfo=UTC)

    with pytest.raises(DoneError):
        go(until_midnight)
    assert sleeps[0] == 600.0  # a capped sleep, then the arm starts at 00:00 UTC
    banked = e1.jsonl.read(tmp_path / "local.jsonl")
    assert len(banked) == per_slot["day1-b1"]
    assert all(prefix.startswith("e1-day1-b1-") for prefix, _ in calls)
    assert {r["started_at"][:10] for r in banked} == {"2026-11-07"}


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
    monkeypatch.setattr(e1, "smoke_passed", lambda version=None, artifact=None: True)
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
    with pytest.raises(ValueError, match=r"no SMOKE: PASS on Claude Code 2\.0\.2"):
        e1.resume(stops, "accepted the new CLI", clock[0], smoke=lambda version: False)
    e1.resume(stops, "accepted the new CLI as a deviation", clock[0], smoke=lambda v: True)
    assert e1.jsonl.read(stops)[-1]["smoke"] == "pass"
    assert e1.claude_changes(artifact, stops) == ({}, None)
    assert go() == len(e1.claude_plan(design)) and calls
    # Another model ID for a tier ends that tier only; the other tiers go on.
    write_rows(artifact, [run("a", "m1", "2.0.1"), run("c", "m2", harness="minimal")])
    stops.unlink()
    calls.clear()
    sonnet = [s for s in e1.claude_plan(design) if s.model == "claude-sonnet"]
    assert go() == len(sonnet) and calls
    assert e1.ended_tiers(stops) == {"claude-haiku"}
    note = next(r for r in e1.jsonl.read(stops) if r["kind"] == "tier_ended")
    assert "served by m2 after m1" in note["reason"]
    stop = e1.open_stop(stops)  # every planned run of the tiers still served is banked
    assert stop["kind"] == "end_point" and "still served" in stop["reason"]
    with pytest.raises(ValueError, match="has ended"):
        e1.resume(stops, "x", clock[0])
    # The slice starts nothing 45 days after the freeze.
    write_rows(artifact, [run("a", "m1", "2.0.1")])
    stops.unlink()
    clock[0] = frozen + timedelta(days=e1.CLAUDE_END_DAYS)
    assert go() == 0 and e1.open_stop(stops)["reason"] == "45 days after the freeze"
    # The end point is recorded at day 45 whatever stop is open (prereg 0004).
    stops.unlink()
    clock[0] = frozen + timedelta(days=40)
    e1.record_stop(stops, "defect", "grader bug", clock[0])
    assert go() == 0 and e1.open_stop(stops)["kind"] == "defect"
    clock[0] = frozen + timedelta(days=46)
    assert go() == 0 and e1.open_stop(stops)["kind"] == "end_point"
    assert [r["kind"] for r in e1.jsonl.read(stops)] == ["defect", "end_point"]
    assert go() == 0 and len(e1.jsonl.read(stops)) == 2  # recorded once


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


def test_dry_plan_balances_each_batch_and_pairs_thread_counts(design: e1.E1Design) -> None:
    plan = e1.dry_plan(design)
    batches = e1.planned_batches(plan)
    assert len(batches) == len(design.models) * len(e1.DRY_CONFIGS)
    assert all(s.phase == "e1-dry" and s.task_seed in e1.DRY_SEEDS for s in plan)
    assert not {s.task_seed for s in plan} & set(e1.CONFIRMATORY_SEEDS)
    for label, specs in batches.items():
        assert len(specs) == 4 and len({(s.threads, s.concurrency, s.model) for s in specs}) == 1
        for factor in ("harness", "timeout", "steps"):
            levels = collections.Counter(s.factors[factor] for s in specs)
            assert sorted(levels.values()) == [2, 2], (label, factor)
        assert len({s.factors["temperature"] for s in specs}) == 1
    by_key = collections.defaultdict(list)
    for s in plan:
        by_key[(s.model, s.concurrency, s.task_seed)].append(s)
    for runs in by_key.values():  # the two thread counts run the same configuration
        assert len(runs) == 2 and runs[0].seed == runs[1].seed
        assert runs[0].factors == runs[1].factors and {r.threads for r in runs} == {3, 4}
    assert len({s.run_id for s in plan}) == len(plan)


def test_run_dry_keeps_going_after_a_failed_batch(tmp_path) -> None:
    import json

    from noisefloor.orchestrate import BatchFailedError

    design = e1.E1Design(models=MODELS[:1], families=("config", "pointer"), tasks_per_family=2)
    plan_file = plan_prereg(tmp_path, design)
    artifact = tmp_path / "e1-dry.jsonl"
    seen: list = []

    def execute(specs, *, artifact, ledger, prefix, stamp, log):
        seen.append(specs[0].block)
        if specs[0].concurrency == 2 and specs[0].threads == 4:
            raise BatchFailedError("b", "server did not start", 0)
        rows = [
            {
                **s.to_json(),
                **stamp,
                "status": "timeout" if i == 0 else "ok",
                "wall_s": 10.0 * (i + 1),
                "host": {"loadavg": [3.5 + i, 1, 1], "mem_available_mb": 9000 - i},
            }
            for i, s in enumerate(specs)
        ]
        rows[0]["host"]["server_peak_rss_mb"] = 2400
        artifact.write_text(
            (artifact.read_text() if artifact.exists() else "")
            + "".join(json.dumps(r) + "\n" for r in rows)
        )
        return len(rows)

    fates = e1.run_dry(
        plan_file, artifact=artifact, ledger=tmp_path / "l", log=lambda m: None, execute=execute
    )
    assert len(fates) == len(seen) == len(e1.DRY_CONFIGS)
    failed = [f for f in fates if f["failed"]]
    assert len(failed) == 1 and failed[0]["block"].endswith("-t4-c2")
    banked = e1.jsonl.read(tmp_path / "e1-dry-batches.jsonl")
    assert banked == fates
    rows = e1.jsonl.read(artifact)
    assert all(r["exploratory"] and len(r["e1_plan_sha256"]) == 64 for r in rows)
    summary = e1.dry_summary(rows)
    assert [(s["threads"], s["concurrency"]) for s in summary] == [(3, 1), (3, 2), (4, 1)]
    first = summary[0]
    assert first["runs"] == 4 and first["timeout"] == 1 and first["infra_error"] == 0
    assert first["wall_median_s"] == 30.0 and first["wall_max_s"] == 40.0
    assert first["load_max"] == 6.5 and first["mem_available_min_mb"] == 8997
    assert first["server_peak_rss_mb"] == 2400


def test_replicate_runs_repeat_two_configurations_per_cell() -> None:
    reps = ((0, 1, 1, 0, 0), (0, 1, 1, 1, 1))
    full = e1.E1Design(models=MODELS[:2], tasks_per_family=1, runs_per_cell=16, replicates=reps)
    plan = e1.local_plan(full)
    replicates = [s for s in plan if s.phase == e1.REPLICATE_PHASE]
    assert len(replicates) == 2 * len(MODELS[:2]) * len(e1.tasks(full))
    by_id = {s.run_id: s for s in plan}
    for rep in replicates:
        original = by_id[rep.run_id.removesuffix("-rep")]
        assert original.factors == rep.factors and original.seed != rep.seed
        assert rep.block and rep.block.split("/")[0] in full.slots
    assert e1.E1Design.from_json(full.to_json()) == full
    with pytest.raises(ValueError, match="not a run of the half fraction"):
        e1.E1Design(models=MODELS[:1], runs_per_cell=16, replicates=((0, 1, 1, 0, 1),))
    with pytest.raises(ValueError, match="whole half fraction"):
        e1.E1Design(models=MODELS[:1], runs_per_cell=8, replicates=reps)


def test_claude_tasks_subset_and_equal_timeouts_across_tiers() -> None:
    design = e1.E1Design(
        models=MODELS,
        claude_tiers=("claude-haiku", "claude-sonnet", "claude-opus"),
        tasks_per_family=4,
        claude_tasks_per_family=2,
    )
    local, claude = e1.tasks(design), e1.claude_tasks(design)
    assert len(claude) == 2 * len(design.families) and {t.id for t in claude} <= {
        t.id for t in local
    }
    plan = e1.claude_plan(design)
    assert len(plan) == 3 * len(claude) * 2 and all(s.max_tokens == 0 for s in plan)
    timeouts = collections.defaultdict(set)
    for spec in plan:
        timeouts[(spec.family, spec.task_seed, spec.factors["harness"])].add(
            (spec.model, spec.factors["timeout"])
        )
    for pairs in timeouts.values():  # every tier ran the same timeout under one harness
        assert len({t for _, t in pairs}) == 1 and len(pairs) == 3
    assert e1.E1Design.from_json(design.to_json()) == design
    with pytest.raises(ValueError, match="claude_tasks_per_family"):
        e1.E1Design(models=MODELS, tasks_per_family=2, claude_tasks_per_family=3)
    groups = e1.claude_groups(plan)
    assert len(groups) == 3 * len(claude) and all(len(g) == 2 for g in groups)


@pytest.mark.usefixtures("unfrozen_ok")
def test_claude_slice_checks_for_changes_before_every_pair(tmp_path, monkeypatch) -> None:
    import json as _json
    from datetime import UTC, datetime, timedelta

    design = e1.E1Design(models=MODELS[:1], families=("config",), tasks_per_family=2)
    prereg = plan_prereg(tmp_path, design)
    artifact = tmp_path / "claude.jsonl"
    stops = tmp_path / "stops-claude.jsonl"
    frozen = datetime(2026, 11, 5, tzinfo=UTC)
    monkeypatch.setattr(e1, "frozen_at", lambda path: frozen)
    monkeypatch.setattr(e1, "smoke_passed", lambda version=None, artifact=None: True)
    calls: list = []

    def reporting(specs, *, artifact, ledger, prefix, stamp, log):
        # The second pair reports a new Claude Code version from its minimal-agent calls.
        calls.append(len(specs))
        version = "2.0.1" if len(calls) == 1 else "2.0.2"
        with artifact.open("a") as fh:
            for s in specs:
                rec = {"run_id": s.run_id, "model": s.model, "cli_versions": [version], **stamp}
                fh.write(_json.dumps(rec) + "\n")
        return len(specs)

    def go(execute):
        return e1.run_claude(
            prereg,
            artifact=artifact,
            ledger=tmp_path / "l",
            now=lambda: frozen + timedelta(days=1),
            log=lambda m: None,
            execute=execute,
        )

    assert go(reporting) == 4 and calls == [2, 2]  # stopped before the third pair
    stop = e1.open_stop(stops)
    assert stop["kind"] == "cli_changed" and stop["to"] == "2.0.2"
    # A refused call leaves a pair unfinished, and the call ends there.
    e1.resume(stops, "accepted 2.0.2", frozen + timedelta(days=1), smoke=lambda v: True)
    refused: list = []

    def one_then_refuse(specs, *, artifact, ledger, prefix, stamp, log):
        refused.append(len(specs))
        return reporting(
            specs[:1], artifact=artifact, ledger=ledger, prefix=prefix, stamp=stamp, log=log
        )

    assert go(one_then_refuse) == 1 and refused == [2]
    assert e1.cli_versions({"cli_versions": ["a"], "claude": {"cli_version": "b"}}) == ["a", "b"]


@pytest.mark.usefixtures("unfrozen_ok")
def test_a_batch_starts_only_from_the_arm_s_committed_run_code(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime

    design = e1.E1Design(models=MODELS[:2], families=("config",), tasks_per_family=3, batch_size=2)
    prereg = plan_prereg(tmp_path, design)
    clock = [datetime(2026, 11, 5, 0, 5, tzinfo=UTC)]
    code = {"src_tree": "a" * 40, "uv_lock": "l" * 40, "dirty": True, "server_sha256": "s"}
    monkeypatch.setattr(e1, "run_code", lambda local: dict(code))
    calls: list = []
    messages: list = []

    def with_code(specs, *, artifact, ledger, prefix, stamp, log):
        import json as _json

        calls.append(len(specs))
        with artifact.open("a") as fh:
            for s in specs:
                rec = {"run_id": s.run_id, "started_at": clock[0].isoformat(), **stamp}
                rec["provenance"] = {"code": dict(code)}
                fh.write(_json.dumps(rec) + "\n")
        return len(specs)

    go = local_runner(tmp_path, prereg, clock, with_code, messages)
    assert go() == 0 and calls == [] and "uncommitted changes" in messages[-1]
    code["dirty"] = False
    first = go()
    assert first > 0 and calls
    code["src_tree"] = "b" * 40  # a commit changed src/ while the arm ran
    clock[0] = datetime(2026, 11, 5, 13, 0, tzinfo=UTC)
    before = len(calls)
    assert go() == 0 and len(calls) == before
    stops = tmp_path / "stops-local.jsonl"
    stop = e1.open_stop(stops)
    assert stop["kind"] == "code_changed" and stop["to"].startswith("b" * 40)
    e1.resume(stops, "accepted the fix as a deviation", clock[0])
    assert go() > 0  # the resumed arm accepts the new code
    assert e1.code_key({"src_tree": "x", "uv_lock": "y", "server_sha256": "z"}, False) == "x:y"


def test_the_local_arm_starts_only_with_the_frozen_server(tmp_path, monkeypatch) -> None:
    import json as _json

    from noisefloor import preregs

    design = e1.E1Design(models=MODELS[:1], families=("config",), tasks_per_family=2)
    prereg = plan_prereg(tmp_path, design)
    block = {"llama_server": {"sha256": "a" * 64}}
    prereg.write_text(prereg.read_text() + "\n```json e1-design\n" + _json.dumps(block) + "\n```\n")
    artifact = tmp_path / "local.jsonl"
    e1.check_server(prereg, artifact)  # outside the chain: not checked
    monkeypatch.setattr(preregs, "chain_entry", lambda path: {"sha256": "0" * 64})
    monkeypatch.setattr(e1, "server_pin", lambda: {"sha256": "b" * 64})
    with pytest.raises(ValueError, match="not the frozen one"):
        e1.check_server(prereg, artifact)
    monkeypatch.setattr(e1, "server_pin", lambda: {"sha256": "a" * 64})
    assert e1.check_server(prereg, artifact) is None
    # A rebuilt server starts only under a frozen amendment naming its digest (a deviation).
    rebuilt = tmp_path / "0009-server.md"
    rebuilt.write_text(
        "---\namends: 0004\n---\n# rebuilt\n\n```json llama-server\n"
        + _json.dumps({"sha256": "c" * 64})
        + "\n```\n"
    )
    monkeypatch.setattr(preregs, "amendments_with", lambda number, block: [rebuilt])
    monkeypatch.setattr(preregs, "require_frozen", lambda path: {"sha256": "1" * 64})
    monkeypatch.setattr(e1, "server_pin", lambda: {"sha256": "c" * 64})
    assert e1.check_server(prereg, artifact) == str(rebuilt)
    monkeypatch.setattr(e1, "server_pin", lambda: {"sha256": None})  # no build at all
    with pytest.raises(ValueError, match="not the frozen one"):
        e1.check_server(prereg, artifact)
    pins = e1.model_pins(e1.E1Design(models=("qwen3-4b",)))
    assert pins["qwen3-4b"]["server_args"] and pins["qwen3-4b"]["ctx_per_slot"] == 32768


@pytest.mark.usefixtures("unfrozen_ok")
def test_claude_slice_stops_on_a_failure_streak_and_on_contamination(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime, timedelta

    design = e1.E1Design(models=MODELS[:1], families=("config",), tasks_per_family=2)
    prereg = plan_prereg(tmp_path, design)
    artifact = tmp_path / "claude.jsonl"
    stops = tmp_path / "stops-claude.jsonl"
    frozen = datetime(2026, 11, 5, tzinfo=UTC)
    monkeypatch.setattr(e1, "frozen_at", lambda path: frozen)
    at = frozen + timedelta(days=1)
    failed = {"model": "claude-haiku", "status": "infra_error"}
    write_rows(artifact, [{"run_id": f"f{i}", **failed} for i in range(3)])
    assert e1.claude_problem(prereg, artifact, stops, at)[:2] == (
        "batch_failure",
        "3 infra_error runs in a row",
    )
    write_rows(
        artifact,
        [{"run_id": "f0", **failed}, {"run_id": "ok", "status": "ok", "model": "claude-haiku"}],
    )
    assert e1.claude_problem(prereg, artifact, stops, at) is None
    # Pairs [infra, infra][infra, ok]: three in a row that end inside a pair still stop it.
    ok = {"run_id": "ok", "status": "ok", "model": "claude-haiku"}
    write_rows(artifact, [*({"run_id": f"f{i}", **failed} for i in range(3)), ok])
    assert e1.claude_problem(prereg, artifact, stops, at)[:2] == (
        "batch_failure",
        "3 infra_error runs in a row",
    )
    dirty = {"run_id": "d", "model": "claude-haiku", "status": "ok", "claude": {"clean": False}}
    write_rows(artifact, [dirty])
    kind, _reason, fields = e1.claude_problem(prereg, artifact, stops, at)
    assert kind == "contaminated" and fields["runs"] == ["d"]


def test_claude_stops_are_written_only_where_the_slice_runs(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(e1, "ARTIFACTS", tmp_path)  # never the repository's artifacts
    monkeypatch.setattr(e1, "billing_guard_findings", lambda: ["ANTHROPIC_API_KEY"])
    assert e1.main(["stop", "claude", "--reason", "x"]) == 5
    assert e1.main(["end", "claude", "--reason", "x"]) == 5
    assert "recorded where it runs" in capsys.readouterr().err
    assert not (tmp_path / "stops-claude.jsonl").exists()


def test_smoke_passed_reads_the_banked_smoke_runs(tmp_path) -> None:
    import json as _json

    smoke = tmp_path / "claude-smoke.jsonl"
    assert not e1.smoke_passed(artifact=smoke)
    child = {
        "api_key_source": "none",
        "tools": [],
        "mcp_servers": [],
        "memory_paths": {},
        "cli_version": "2.0.1",
        "model_id": "m",
    }
    smoke.write_text(_json.dumps({"run_id": "s", "status": "ok", "claude": child}) + "\n")
    assert e1.smoke_passed(artifact=smoke) and e1.smoke_passed("2.0.1", artifact=smoke)
    assert not e1.smoke_passed("2.0.2", artifact=smoke)


def test_arms_start_only_after_g6_and_e0_part_2(tmp_path, monkeypatch) -> None:
    from noisefloor import preregs
    from noisefloor.analysis import e1 as analysis

    design = e1.E1Design(models=MODELS[:1], families=("config",), tasks_per_family=2)
    prereg = plan_prereg(tmp_path, design)
    prereg.write_text(
        prereg.read_text() + '\n```json e1-analysis\n{"validated_range": "prereg/0005-x.md"}\n```\n'
    )
    e1.require_validation(prereg)  # outside the chain: not checked
    monkeypatch.setattr(preregs, "chain_entry", lambda path: {"sha256": "0" * 64})
    g6, e0 = tmp_path / "g6.json", tmp_path / "coverage-0005.json"
    monkeypatch.setattr(analysis, "validation_artifacts", lambda settings: {"g6": g6, "e0": e0})
    with pytest.raises(ValueError, match="G6 and E0 part 2 come first"):
        e1.require_validation(prereg)
    g6.write_text("{}")
    e0.write_text("{}")
    e1.require_validation(prereg)


def test_the_first_claude_code_version_needs_a_smoke_run(tmp_path) -> None:
    from datetime import UTC, datetime

    artifact = tmp_path / "claude.jsonl"
    stops = tmp_path / "stops-claude.jsonl"
    child = {"model_id": "m1", "cli_version": "2.0.1"}
    write_rows(artifact, [{"run_id": "a", "model": "claude-haiku", "claude": child}])
    # The smoke ran on 2.0.0 and the slice's first run reports 2.0.1: not accepted.
    assert e1.claude_changes(artifact, stops, smoke=lambda v: v == "2.0.0") == ({}, "2.0.1")
    assert e1.claude_changes(artifact, stops, smoke=lambda v: v == "2.0.1") == ({}, None)
    at = datetime(2026, 11, 6, tzinfo=UTC)
    e1.record_stop(stops, "cli_changed", "Claude Code 2.0.1 in use", at, to="2.0.1")
    assert e1.claude_changes(artifact, stops, smoke=lambda v: False) == ({}, None)


def test_a_defect_resume_needs_an_amendment_that_names_the_stop(tmp_path, monkeypatch) -> None:
    import json as _json
    from datetime import UTC, datetime

    from noisefloor import preregs

    stops = tmp_path / "stops-local.jsonl"
    at = datetime(2026, 11, 8, 3, tzinfo=UTC)
    e1.record_stop(stops, "defect", "grader bug", at)
    monkeypatch.setattr(preregs, "require_frozen", lambda path: {"sha256": "1" * 64})
    other = tmp_path / "0009-other.md"
    other.write_text(
        "---\namends: 0004\n---\n# x\n\n```json defect-resume\n"
        + _json.dumps({"at": "2026-11-07T00:00:00+00:00"})
        + "\n```\n"
    )
    with pytest.raises(ValueError, match="not the one at"):
        e1.resume(stops, "fixed", at, amendment=other)
    bare = tmp_path / "0009-bare.md"
    bare.write_text("---\namends: 0004\n---\n# x\n")
    with pytest.raises(ValueError, match="no ```json defect-resume``` block"):
        e1.resume(stops, "fixed", at, amendment=bare)
    fix = tmp_path / "0009-fix.md"
    fix.write_text(
        "---\namends: 0004\n---\n# fix\n\n```json defect-resume\n"
        + _json.dumps({"at": [at.isoformat(timespec="seconds")]})
        + "\n```\n"
    )
    e1.resume(stops, "fixed", at, amendment=fix)
    assert e1.open_stop(stops) is None and e1.jsonl.read(stops)[-1]["amendment"] == str(fix)
