import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from noisefloor import claude_arm, orchestrate
from noisefloor import client as cl
from noisefloor.claude_arm import (
    ANSWER_FILE,
    PREAMBLE,
    collect,
    end_reason,
    materialize,
    run_claude_code,
    smoke_report,
    smoke_spec,
    turn_cap,
)
from noisefloor.client import (
    ClaudeRun,
    ClaudeStream,
    Client,
    Ledger,
    ModelCallRefusedError,
    UsageLimitError,
    claude_child_env,
    claude_prompt,
    parse_stream_json,
    raise_for_stop,
)
from noisefloor.runner import RunSpec, run_batch, run_minimal
from noisefloor.suite import AnswerSpec, Snapshot, generate

FAKE = Path(__file__).with_name("fake_claude.py")
CAPS = {"claude_agent_runs_per_day": 6, "claude_agent_max_turns": 8}


def spec(**kw: Any) -> RunSpec:
    base: dict[str, Any] = dict(
        run_id="c1",
        phase="test",
        family="pointer",
        task_seed=900002,
        difficulty=1,
        model="claude-haiku",
        harness="claude_code",
        temperature=0.0,
        seed=3,
        timeout_s=30.0,
        max_steps=15,
        threads=1,
        concurrency=1,
    )
    return RunSpec(**{**base, **kw})


def expected(s: RunSpec) -> str:
    task = generate(s.family, s.task_seed, s.difficulty)
    assert isinstance(task.expected, AnswerSpec)
    return task.expected.expected


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake ``claude`` on PATH; returns the plan path (write a plan before running)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "claude"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
    wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_CLAUDE_PLAN", str(tmp_path / "plan.json"))
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(tmp_path / "calls.jsonl"))
    monkeypatch.delenv("NOISEFLOOR_NO_MODELS", raising=False)
    return tmp_path / "plan.json"


@pytest.fixture
def no_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith(("ANTHROPIC_", "CLAUDE_CODE_USE_")):
            monkeypatch.delenv(name)
    monkeypatch.setattr(cl, "billing_guard_findings", lambda: [])


def plan(path: Path, **kw: Any) -> None:
    path.write_text(json.dumps(kw))


def calls(tmp_path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in (tmp_path / "calls.jsonl").read_text().splitlines()]


def make_client(tmp_path: Path, **caps: int) -> Client:
    return Client(
        ledger=Ledger(tmp_path / "ledger.jsonl"),
        cache_dir=tmp_path / "cache",
        caps={**CAPS, **caps},
    )


def test_child_env_scrubs_credentials_and_keeps_the_login() -> None:
    env = claude_child_env(
        {
            "HOME": "/h",
            "PATH": "/bin",
            "HTTPS_PROXY": "http://proxy",
            "CLAUDE_CONFIG_DIR": "/c",
            "CLAUDE_CODE_OAUTH_TOKEN": "plan-token",
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "CLAUDECODE": "1",
            "CLAUDE_CODE_ENTRYPOINT": "cli",
            "CLAUDE_CODE_SSE_PORT": "1",
            "GH_TOKEN": "t",
            "AWS_SECRET_ACCESS_KEY": "s",
            "MY_SERVICE_API_KEY": "k",
            "DB_PASSWORD": "p",
        }
    )
    assert env == {
        "HOME": "/h",
        "PATH": "/bin",
        "HTTPS_PROXY": "http://proxy",
        "CLAUDE_CONFIG_DIR": "/c",
        "CLAUDE_CODE_OAUTH_TOKEN": "plan-token",
        "CLAUDE_CODE_USE_BEDROCK": "1",
        **cl.CHILD_SWITCHES,
    }
    assert isinstance(claude_child_env(), dict)


def test_claude_code_run_submits_and_is_graded(
    tmp_path: Path, fake: Path, no_guard: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = spec()
    plan(
        fake,
        init={"tools": ["Read", "Glob", "Grep", "Write", "Edit"], "claude_code_version": "9.9"},
        turns=[
            {"text": "reading", "tool_uses": 2},
            {"text": "answering", "write": {ANSWER_FILE: expected(s) + "\n"}, "tool_uses": 1},
            {"text": "done"},
        ],
    )
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("GH_TOKEN", "secret")
    c = make_client(tmp_path)
    out = run_claude_code(s, c)
    assert out["outcome"] == 1.0 and out["end_reason"] == "submitted" and out["status"] == "ok"
    assert out["steps"] == 3 and out["tool_calls"] == 3 and out["parse_errors"] == 0
    assert out["input_tokens"] == 37 and out["output_tokens"] == 9 and out["error"] == ""
    child = out["claude"]
    assert child["clean"] and child["model_id"] == "claude-fake-1" and child["cli_version"] == "9.9"
    assert child["max_turns"] == 8 and child["plan_equivalent_usd"] == 0.02
    kinds = [e["kind"] for e in c.ledger.entries()]
    assert kinds == ["agent_run", "agent_usage"]
    usage = c.ledger.entries()[-1]
    assert usage["cash_usd"] == 0 and usage["turns"] == 3 and usage["plan_equivalent_usd"] == 0.02
    (call,) = calls(tmp_path)
    argv = call["argv"]
    for flag in ("--restricted", "--strict-mcp-config", "--no-session-persistence", "--verbose"):
        assert flag in argv
    assert "--bare" not in argv
    assert argv[argv.index("--tools") + 1] == "Read,Glob,Grep,Write,Edit"
    assert argv[argv.index("--allowedTools") + 1] == "Read,Glob,Grep,Write,Edit"
    assert argv[argv.index("--permission-mode") + 1] == "dontAsk"
    assert argv[argv.index("--max-turns") + 1] == "8"
    assert argv[argv.index("--model") + 1] == "haiku"
    assert (
        call["present"] == [] and call["claude_mds_off"] == "1" and call["auto_memory_off"] == "1"
    )
    assert not Path(call["cwd"]).is_relative_to(cl.REPO)
    assert call["prompt"].startswith(PREAMBLE)
    assert not Path(call["cwd"]).exists()


def test_claude_code_end_reasons(tmp_path: Path, fake: Path, no_guard: None) -> None:
    c = make_client(tmp_path, claude_agent_max_turns=2, claude_agent_runs_per_day=20)
    plan(fake, turns=[{"text": "a"}])
    out = run_claude_code(spec(), c)
    assert out["end_reason"] == "stopped" and out["status"] == "ok" and out["outcome"] == 0.0
    plan(fake, turns=[{"text": str(i)} for i in range(5)])
    out = run_claude_code(spec(), c)
    assert (
        out["end_reason"] == "step_limit" and out["claude"]["result_subtype"] == "error_max_turns"
    )
    plan(fake, turns=[{"text": str(i)} for i in range(5)], ignore_max_turns=True, result=None)
    out = run_claude_code(spec(), c)
    assert out["end_reason"] == "step_limit" and out["claude"]["ended"] == "turn_cap"
    assert out["status"] == "timeout"
    plan(fake, turns=[{"text": "slow", "sleep": 20}])
    out = run_claude_code(spec(timeout_s=1.0), c)
    assert out["end_reason"] == "wall_clock" and out["wall_s"] < 15
    plan(fake, turns=[], result=None, exit=1)
    out = run_claude_code(spec(), c)
    assert out["end_reason"] == "backend_error" and out["status"] == "infra_error"
    assert out["error"]
    plan(
        fake, turns=[{"text": "x"}], result={"subtype": "error_during_execution", "is_error": True}
    )
    out = run_claude_code(spec(), c)
    assert out["end_reason"] == "backend_error" and out["error"] == "done"


def test_refusals_stop_the_run(tmp_path: Path, fake: Path, no_guard: None) -> None:
    c = make_client(tmp_path, claude_agent_runs_per_day=20)
    plan(fake, init={"apiKeySource": "/login managed key"}, turns=[{"text": "x"}])
    with pytest.raises(ModelCallRefusedError, match="API key source") as info:
        run_claude_code(spec(), c)
    assert not isinstance(info.value, UsageLimitError)
    assert [e["kind"] for e in c.ledger.entries()] == ["agent_run", "agent_usage"]
    plan(fake, turns=[{"text": "x", "rate_limit": {"status": "rejected", "resetsAt": 1999999999}}])
    with pytest.raises(UsageLimitError) as limit:
        run_claude_code(spec(), c)
    assert limit.value.resets_at == 1999999999
    plan(fake, turns=[{"text": "x", "rate_limit": {"status": "allowed", "isUsingOverage": True}}])
    with pytest.raises(UsageLimitError, match="extra usage"):
        run_claude_code(spec(), c)
    plan(fake, turns=[{"text": "x", "rate_limit": {"status": "allowed_warning"}}])
    assert run_claude_code(spec(), c)["claude"]["rate_limit_status"] == "allowed_warning"
    text = "Claude AI usage limit reached|1999999998"
    plan(fake, turns=[{"text": text}], result={"is_error": True, "result": text})
    with pytest.raises(UsageLimitError) as late:
        run_claude_code(spec(), c)
    assert late.value.resets_at == 1999999998


def test_daily_cap_and_guard_refuse_before_launch(
    tmp_path: Path, fake: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cl, "billing_guard_findings", lambda: [])
    c = make_client(tmp_path, claude_agent_runs_per_day=1)
    plan(fake, turns=[{"text": "x"}])
    run_claude_code(spec(), c)
    with pytest.raises(ModelCallRefusedError, match="daily cap"):
        run_claude_code(spec(), c)
    monkeypatch.setattr(cl, "billing_guard_findings", lambda: ["ANTHROPIC_BASE_URL"])
    with pytest.raises(ModelCallRefusedError, match="billing guard"):
        run_claude_code(spec(), make_client(tmp_path / "other"))
    assert len(calls(tmp_path)) == 1


def test_minimal_agent_with_claude(tmp_path: Path, fake: Path, no_guard: None) -> None:
    s = spec(harness="minimal")
    read = json.dumps({"tool": "read_file", "args": {"path": "/start.txt"}})
    submit = json.dumps({"tool": "submit", "args": {"answer": expected(s)}})
    plan(fake, replies=[read, f"Submitting now.\n```json\n{submit}\n```"])
    c = make_client(tmp_path)
    out = run_minimal(s, c)
    assert out["outcome"] == 1.0 and out["end_reason"] == "submitted" and out["steps"] == 2
    assert out["max_steps_used"] == 8 and out["model_ids"] == ["claude-fake-1"]
    kinds = [e["kind"] for e in c.ledger.entries()]
    assert kinds == ["agent_run", "agent_step", "agent_step"]
    first, second = calls(tmp_path)
    assert first["argv"][first["argv"].index("--tools") + 1] == ""
    assert "<assistant>" not in first["prompt"] and second["prompt"].count("<assistant>") == 1
    assert "Observation:" in second["prompt"]


def test_turn_cap_helper() -> None:
    assert turn_cap(15, {"claude_agent_max_turns": 8}) == 8
    assert turn_cap(5, {"claude_agent_max_turns": 8}) == 5
    assert turn_cap(5, {}) == 5
    assert turn_cap(5, {"claude_agent_max_turns": 0}) == 1


def test_materialize_and_collect(tmp_path: Path) -> None:
    files = Snapshot({"/a/b.txt": "x\n", "/c.txt": "y"})
    root = tmp_path / "ws"
    root.mkdir()
    materialize(files, root)
    assert collect(root) == (files, None)
    (root / ANSWER_FILE).write_text("42\n")
    (root / ".claude").mkdir()
    (root / ".claude" / "settings.local.json").write_text("{}")
    (root / "link.txt").symlink_to(root / "c.txt")
    (root / "big.txt").write_bytes(b"z" * (claude_arm.MAX_FILE_BYTES + 10))
    snap, answer = collect(root)
    assert (
        answer == "42\n"
        and "/link.txt" not in snap.files
        and "/.claude/settings.local.json" not in snap.files
    )
    assert len(snap.files["/big.txt"]) == claude_arm.MAX_FILE_BYTES
    with pytest.raises(ValueError, match="collides"):
        materialize(Snapshot({"/ANSWER.txt": "x"}), tmp_path / "ws2")


def test_end_reason_rules() -> None:
    def run(ended: str, result: dict[str, Any] | None) -> ClaudeRun:
        return ClaudeRun(ClaudeStream(result=result), ended, 0, "", 1.0, [])

    ok = {"subtype": "success"}
    assert end_reason(run("wall_clock", None), "7") == "submitted"
    assert end_reason(run("exit", ok), None) == "stopped"
    assert end_reason(run("turn_cap", None), None) == "step_limit"
    assert end_reason(run("exit", {"subtype": "error_max_turns"}), None) == "step_limit"
    assert end_reason(run("wall_clock", None), None) == "wall_clock"
    assert end_reason(run("exit", None), None) == "backend_error"
    assert end_reason(run("exit", {**ok, "is_error": True}), None) == "backend_error"


def test_stream_reader_details() -> None:
    s = ClaudeStream(max_turns=1)
    assert s.feed("not json") == "" and s.feed("[1]") == "" and s.feed('{"type": "user"}') == ""
    message = {
        "id": "m",
        "content": [{"type": "text", "text": "a"}, "junk"],
        "usage": {"output_tokens": "4"},
    }
    assert s.feed(json.dumps({"type": "assistant", "message": message})) == ""
    assert s.feed(json.dumps({"type": "assistant", "message": {**message, "content": []}})) == ""
    assert s.feed(json.dumps({"type": "assistant", "message": "junk"})) == ""
    assert s.turn_ids == ["m"] and s.usage() == (0, 4) and s.plan_equivalent_usd() == 0.0
    assert s.feed(json.dumps({"type": "assistant", "message": {"content": []}})) == "turn_cap"
    s.result = {"total_cost_usd": "n/a", "usage": {"input_tokens": None}}
    assert s.plan_equivalent_usd() == 0.0 and s.usage() == (0, 0)
    assert s.feed(json.dumps({"type": "rate_limit_event", "rate_limit_info": None})) == ""
    raise_for_stop("turn_cap", s)  # not a refusal
    with pytest.raises(UsageLimitError) as info:
        raise_for_stop("usage_limit", s)
    assert info.value.resets_at is None
    with pytest.raises(UsageLimitError, match="plan usage limit"):
        parse_stream_json(
            [json.dumps({"type": "result", "is_error": True, "result": "usage limit reached"})], 1.0
        )
    rejected = {"type": "rate_limit_event", "rate_limit_info": {"status": "rejected"}}
    with pytest.raises(UsageLimitError):
        parse_stream_json([json.dumps(rejected)], 1.0)


def test_claude_prompt_rendering() -> None:
    sys_msg = {"role": "system", "content": "S"}
    assert claude_prompt([sys_msg, {"role": "user", "content": "hi"}]) == ("S", "hi")
    system, prompt = claude_prompt(
        [
            sys_msg,
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "act"},
            {"role": "user", "content": "Observation:\nok"},
        ]
    )
    assert (
        system == "S"
        and prompt.count("<assistant>") == 1
        and prompt.index("task") < prompt.index("act")
    )


def test_agent_command_validation(tmp_path: Path) -> None:
    c = make_client(tmp_path)
    with pytest.raises(ValueError, match="unknown Claude model"):
        c.claude_agent_command("gpt", ["Read"], 3)
    with pytest.raises(ValueError, match="max_turns"):
        c.claude_agent_command("haiku", ["Read"], 0)


def test_single_call_timeout_and_missing_binary(tmp_path: Path, fake: Path, no_guard: None) -> None:
    plan(fake, replies=["x"])
    c = make_client(tmp_path)
    c.claude_bin = str(tmp_path / "nope")
    with pytest.raises(cl.BackendError, match="failed to run"):
        cl.run_claude_stream([c.claude_bin], "", cwd=tmp_path, timeout=5)
    slow = tmp_path / "slow"
    slow.write_text("#!/bin/sh\nsleep 30\n")
    slow.chmod(0o755)
    c.claude_bin = str(slow)
    request = cl.ChatRequest(
        backend="claude", model="haiku", messages=({"role": "user", "content": "q"},)
    )
    with pytest.raises(cl.BackendError, match="did not finish"):
        c.chat(request, kind="agent_step", timeout=1.0)


def test_runspec_rules_for_claude() -> None:
    with pytest.raises(ValueError, match="Claude tiers only"):
        spec(model="qwen2.5-1.5b")
    with pytest.raises(ValueError, match="local models only"):
        spec(harness="inspect")
    with pytest.raises(ValueError, match="unknown model"):
        spec(model="claude-mythical")


def test_run_batch_records_nothing_after_a_refusal(
    tmp_path: Path, fake: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cl, "billing_guard_findings", lambda: ["ANTHROPIC_BASE_URL"])
    budget = tmp_path / "budget.toml"
    budget.write_text(
        "claude_agent_runs_per_day = 30\nclaude_agent_max_turns = 40\ncap_share = 0.2\n"
    )
    batch = {
        "batch_id": "b1",
        "runs": [spec(run_id="x1").to_json(), spec(run_id="x2").to_json()],
        "ledger": str(tmp_path / "ledger.jsonl"),
        "budget": str(budget),
    }
    out = tmp_path / "out"
    assert run_batch(batch, out) == 0
    note = json.loads((out / "refused.json").read_text())
    assert "billing guard" in note["reason"] and note["run_id"] == "x1"
    assert not (out / "results.jsonl").exists()


def test_execute_claude_plan_on_the_host(
    tmp_path: Path, fake: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in list(os.environ):
        if name.startswith(("ANTHROPIC_", "CLAUDE_CODE_USE_")):
            monkeypatch.delenv(name)
    s = spec(run_id="e1")
    plan(fake, turns=[{"text": "a", "write": {ANSWER_FILE: expected(s)}}])
    artifact = tmp_path / "art.jsonl"
    ledger = tmp_path / "usage.jsonl"
    logs: list[str] = []
    banked = orchestrate.execute(
        [s],
        artifact=artifact,
        ledger=ledger,
        prefix="t",
        work_root=tmp_path / "work",
        log=logs.append,
    )
    assert banked == 1
    (row,) = [json.loads(ln) for ln in artifact.read_text().splitlines()]
    assert row["outcome"] == 1.0 and row["claude"]["clean"]
    assert row["provenance"]["model_alias"] == "haiku"
    assert row["provenance"]["sandbox"] == orchestrate.CLAUDE_SANDBOX
    assert "model_sha256" not in row["provenance"]
    kinds = [json.loads(ln)["kind"] for ln in ledger.read_text().splitlines()]
    assert kinds == ["agent_run", "agent_usage"]
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://elsewhere")
    later = [spec(run_id="e2"), spec(run_id="e3", model="claude-sonnet")]
    assert (
        orchestrate.execute(
            later,
            artifact=artifact,
            ledger=ledger,
            prefix="t",
            work_root=tmp_path / "work",
            log=logs.append,
        )
        == 0
    )
    assert any("refused" in m and "stopping" in m for m in logs)
    assert orchestrate.banked_ids(artifact) == {"e1"}


def test_prepare_keeps_old_output_of_a_banked_workspace(tmp_path: Path) -> None:
    batch = orchestrate.Batch("b", (spec(),))
    ws = tmp_path / "ws"
    (ws / "out").mkdir(parents=True)
    (ws / ".banked").write_text("x")
    orchestrate._prepare(batch, ws)
    assert (ws / "out.0").is_dir() and not (ws / ".banked").exists() and not (ws / "out").exists()
    assert json.loads((ws / "batch.json").read_text())["batch_id"] == "b"
    out = ws / "out"
    out.mkdir()
    (out / "refused.json").write_text("{}")
    assert orchestrate._rows(batch, out, 0) == []
    (out / "refused.json").unlink()
    # Without a refusal, a run without a record makes the batch a failed one; nothing is
    # made up for the missing run.
    assert orchestrate._rows(batch, out, 3) == []
    note = json.loads((out / "aborted.json").read_text())
    assert note["reason"] == "runner exited 3 before 1 of its runs were recorded"


def test_host_executor_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise orchestrate.subprocess.TimeoutExpired(
            cmd="runner", timeout=1, output=b"o", stderr=b"e"
        )

    monkeypatch.setattr(orchestrate.subprocess, "run", boom)
    rows = orchestrate.host_executor(tmp_path / "l.jsonl")(
        orchestrate.Batch("b", (spec(),)), tmp_path / "ws"
    )
    note = json.loads((tmp_path / "ws" / "out" / "aborted.json").read_text())
    assert rows == [] and "exited None" in note["reason"]


def test_smoke_report_and_main(tmp_path: Path) -> None:
    clean = {
        "status": "ok",
        "end_reason": "submitted",
        "outcome": 1.0,
        "steps": 2,
        "claude": {
            "api_key_source": "none",
            "tools": ["Read"],
            "mcp_servers": [],
            "memory_paths": {},
        },
    }
    assert smoke_report(clean)[-1] == "SMOKE: PASS"
    dirty = {
        **clean,
        "status": "infra_error",
        "claude": {"api_key_source": "x", "mcp_servers": ["m"]},
    }
    assert smoke_report(dirty)[-1] == "SMOKE: CHECK login, mcp servers, ended"
    s = smoke_spec("claude-opus")
    assert s.model == "claude-opus" and s.harness == "claude_code" and s.phase == "smoke"
    artifact = tmp_path / "smoke.jsonl"

    def fake_execute(specs: list[RunSpec], *, artifact: Path, **kw: Any) -> int:
        artifact.write_text(json.dumps({**clean, "run_id": specs[0].run_id}) + "\n")
        return 1

    assert claude_arm.main(["smoke"], execute=fake_execute, artifact=artifact) == 0
    assert (
        claude_arm.main(["smoke"], execute=lambda *a, **k: 0, artifact=tmp_path / "none.jsonl") == 1
    )
