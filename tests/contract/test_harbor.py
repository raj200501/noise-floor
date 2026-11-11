"""Contract tests for the Harbor results adapter (evalstats.adapters.harbor).

The recorded fixture under tests/fixtures/adapters/harbor/jobs/ is MOCK data in the real
Harbor 0.24.0 file format (see the README.md beside it): no agent, model, environment, or
verifier ran, and nothing in it is a model result or a Terminal-Bench result. Expected values
come from the raw JSON files, from Harbor's own job statistics in the job's result.json, and
from the designed cases listed in the fixture README, never from the adapter's output.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import pytest

from evalstats.adapters import read_harbor
from evalstats.adapters.harbor import REAL_OUTCOME_EXCEPTIONS, TIMEOUT_EXCEPTIONS
from evalstats.schema import EvalResults, SchemaError

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "adapters" / "harbor"
JOBS = FIXTURE / "jobs"
JOB = JOBS / "2026-10-23__02-00-00"


def load(path: Path) -> Any:
    return json.loads(path.read_text())


TRIAL_DIRS = sorted(p for p in JOB.iterdir() if p.is_dir())
RAW = {d.name: load(d / "result.json") for d in TRIAL_DIRS if (d / "result.json").is_file()}
PENDING = {d.name: load(d / "config.json") for d in TRIAL_DIRS if not (d / "result.json").exists()}
JOB_CONFIG = load(JOB / "config.json")
JOB_RESULT = load(JOB / "result.json")
STATS = JOB_RESULT["stats"]

# Designed case of every trial, from the table in the fixture README: (status, status_reason).
DESIGNED: dict[str, tuple[str, str | None]] = {
    "mock-hello__mock000": ("ok", None),
    "mock-hello__mock001": ("ok", None),
    "mock-regex__mock002": ("timeout", "AgentTimeoutError"),
    "mock-regex__mock003": ("ok", None),
    "mock-build__mock004": ("infra_error", "EnvironmentStartTimeoutError"),
    "mock-build__mock005": ("ok", None),
    "mock-hello__mock006": ("ok", None),
    "mock-hello__mock007": ("ok", "AgentSafetyRefusalError"),
    "mock-regex__mock008": ("timeout", "VerifierTimeoutError"),
    "mock-regex__mock009": ("flagged", "verifier_disabled"),
    "mock-build__mock010": ("timeout", "ContextWindowExceededError"),
    "mock-build__mock011": ("infra_error", "no_result"),
}

# Covariates in the order the module docstring lists them, and the numeric ones.
COVARIATES = (
    "status",
    "status_reason",
    "trial_name",
    "agent",
    "agent_version",
    "exception_type",
    "input_tokens",
    "cache_tokens",
    "output_tokens",
    "cost_usd",
    "agent_time",
    "total_time",
    "started_at",
    "dataset",
)
NUMERIC = ("input_tokens", "cache_tokens", "output_tokens", "cost_usd", "agent_time", "total_time")

# Harbor 0.24.0 exception type names (harbor/trial/errors.py, harbor/agents/installed/base.py,
# harbor/llms/base.py, harbor/environments/), grouped as the module docstring maps them.
LIMIT_TYPES = (
    "AgentTimeoutError",
    "VerifierTimeoutError",
    "ContextWindowExceededError",
    "ContextLengthExceededError",
    "OutputTokenExceededError",
    "OutputLengthExceededError",
)
INFRA_TYPES = (
    "EnvironmentStartTimeoutError",
    "AgentSetupTimeoutError",
    "NonZeroAgentExitCodeError",
    "ApiRateLimitError",
    "ApiConnectionError",
    "NetworkConnectionError",
    "AgentAuthenticationError",
    "ApiUsageLimitError",
    "ModelNotFoundError",
    "RewardFileNotFoundError",
    "RewardFileEmptyError",
    "VerifierOutputParseError",
    "HealthcheckError",
    "MemoryLimitExceededError",
    "SandboxLikelyOutOfMemoryError",
    "CancelledError",
)


# Expected values, derived from the raw files.


def total(values: Iterable[float | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return sum(present) if present else None


def seconds(start: str | None, end: str | None) -> float | None:
    if start is None or end is None:
        return None
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


def per_step(raw: Mapping[str, Any], key: str) -> list[Any]:
    """The trial's own ``key``, or each step's ``key`` when the trial has none (multi-step)."""
    if raw.get(key) is not None:
        return [raw[key]]
    return [step[key] for step in raw.get("step_results") or [] if step.get(key) is not None]


def expected_trial(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Row values of one finished trial, read straight from its result.json."""
    agent = raw["config"]["agent"]
    rewards = (raw["verifier_result"] or {}).get("rewards") or {}
    exception = raw["exception_info"]
    usage = per_step(raw, "agent_result")
    timing = per_step(raw, "agent_execution")
    return {
        "model": agent["model_name"],
        "task": raw["task_name"],
        "outcome": float(rewards.get("reward", 0.0)),
        "trial_name": raw["trial_name"],
        "agent": agent["name"],
        "agent_version": raw["agent_info"]["version"],
        "exception_type": exception["exception_type"] if exception else None,
        "input_tokens": total(u["n_input_tokens"] for u in usage),
        "cache_tokens": total(u["n_cache_tokens"] for u in usage),
        "output_tokens": total(u["n_output_tokens"] for u in usage),
        "cost_usd": total(u["cost_usd"] for u in usage),
        "agent_time": total(seconds(t["started_at"], t["finished_at"]) for t in timing),
        "total_time": seconds(raw["started_at"], raw["finished_at"]),
        "started_at": raw["started_at"],
        "dataset": raw["source"],
    }


def pending_task(config: Mapping[str, Any]) -> str:
    """Task of a trial without a result: that of the finished trials with the same task path."""
    (task,) = {
        raw["task_name"]
        for raw in RAW.values()
        if raw["config"]["task"]["path"] == config["task"]["path"]
    }
    return str(task)


def expected_runs() -> dict[str, str]:
    """Run index of each trial: start-time order within (model, task), unfinished trials last."""
    groups: defaultdict[tuple[str, str], list[tuple[int, float, str]]] = defaultdict(list)
    for name, raw in RAW.items():
        start = datetime.fromisoformat(raw["started_at"]).timestamp()
        groups[raw["config"]["agent"]["model_name"], raw["task_name"]].append((0, start, name))
    for name, config in PENDING.items():
        groups[config["agent"]["model_name"], pending_task(config)].append((1, 0.0, name))
    return {
        name: str(run)
        for members in groups.values()
        for run, (*_, name) in enumerate(sorted(members))
    }


def expected_rows() -> dict[str, dict[str, Any]]:
    """Every row of the fixture job, from the raw files and the designed cases."""
    runs = expected_runs()
    table = {}
    for name, raw in RAW.items():
        status, reason = DESIGNED[name]
        table[name] = {**expected_trial(raw), "status": status, "status_reason": reason}
    for name, config in PENDING.items():
        status, reason = DESIGNED[name]
        table[name] = {
            "model": config["agent"]["model_name"],
            "task": pending_task(config),
            "outcome": 0.0,
            "status": status,
            "status_reason": reason,
            "trial_name": config["trial_name"],
            "agent": config["agent"]["name"],
            "dataset": config["task"]["source"],
            **dict.fromkeys(("agent_version", "exception_type", *NUMERIC, "started_at")),
        }
    return {name: {**row, "run": runs[name]} for name, row in table.items()}


EXPECTED = expected_rows()


# Adapter output as plain values.


def clean(value: Any) -> Any:
    return None if pd.isna(value) else value


def rows_of(res: EvalResults) -> dict[str, dict[str, Any]]:
    records = res.frame.to_dict("records")
    table = {str(r["trial_name"]): {str(k): clean(v) for k, v in r.items()} for r in records}
    assert len(table) == len(records), "trial names repeat"
    return table


@pytest.fixture(scope="module")
def job() -> EvalResults:
    return read_harbor(JOB)


@pytest.fixture(scope="module")
def rows(job: EvalResults) -> dict[str, dict[str, Any]]:
    return rows_of(job)


# Synthetic trials in tmp_path, built from the recorded files (same layout, edited values).


def trial(name: str = "t1", base: str = "mock-hello__mock000", **fields: Any) -> dict[str, Any]:
    data = copy.deepcopy(RAW[base])
    data["trial_name"] = name
    data["config"]["trial_name"] = name
    data.update(fields)
    return data


def exception_info(kind: str) -> dict[str, Any]:
    info = copy.deepcopy(RAW["mock-regex__mock002"]["exception_info"])
    info["exception_type"] = kind
    return info


def write_trial(directory: Path, data: Mapping[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(json.dumps(data, indent=4))
    return directory


def write_pending(directory: Path, task: Mapping[str, Any]) -> Path:
    config = copy.deepcopy(PENDING["mock-build__mock011"])
    config["trial_name"] = directory.name
    config["task"] = dict(task)
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(config, indent=4))
    return directory


def read_one(tmp_path: Path, data: Mapping[str, Any], **options: Any) -> dict[str, Any]:
    res = read_harbor(write_trial(tmp_path / str(data["trial_name"]), data), **options)
    ((_, row),) = rows_of(res).items()
    return row


# The recorded job.


def test_designed_cases_match_the_fixture_files() -> None:
    assert DESIGNED.keys() == {d.name for d in TRIAL_DIRS}
    assert PENDING.keys() == {n for n, (_, reason) in DESIGNED.items() if reason == "no_result"}
    for name, raw in RAW.items():
        status, reason = DESIGNED[name]
        steps = raw["step_results"] or []
        assert all(step["exception_info"] is None for step in steps)
        if status == "flagged":
            assert raw["config"]["verifier"]["disable"] is True
            assert raw["exception_info"] is None
        elif reason is None:
            assert raw["exception_info"] is None
        else:
            assert raw["exception_info"]["exception_type"] == reason


def test_complete_job_reads_without_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = read_harbor(str(JOB))
    assert len(res) == JOB_RESULT["n_total_trials"]


def test_job_shape(job: EvalResults) -> None:
    assert len(job) == JOB_RESULT["n_total_trials"] == len(TRIAL_DIRS)
    assert sorted(job.models) == sorted(agent["model_name"] for agent in JOB_CONFIG["agents"])
    assert sorted(job.tasks) == sorted({raw["task_name"] for raw in RAW.values()})
    assert sorted(job.tasks) == sorted(PurePosixPath(t["path"]).name for t in JOB_CONFIG["tasks"])
    cells = Counter(zip(job.frame["model"], job.frame["task"], strict=True))
    assert set(cells.values()) == {JOB_CONFIG["n_attempts"]}
    assert set(job.frame["run"]) == {str(i) for i in range(JOB_CONFIG["n_attempts"])}
    assert list(job.frame.columns) == ["model", "task", "run", "outcome", *COVARIATES]
    assert job.covariates == list(COVARIATES)
    assert all(job.frame[column].dtype == "float64" for column in NUMERIC)
    rewards = [
        value
        for raw in RAW.values()
        for value in ((raw["verifier_result"] or {}).get("rewards") or {}).values()
    ]
    assert job.is_binary == all(value in (0, 1) for value in rewards)


def test_job_has_one_row_per_trial_directory(rows: dict[str, dict[str, Any]]) -> None:
    assert rows.keys() == EXPECTED.keys()


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_job_row(rows: dict[str, dict[str, Any]], name: str) -> None:
    assert rows[name] == EXPECTED[name]


def test_every_status_value_occurs(job: EvalResults) -> None:
    assert Counter(job.frame["status"]) == Counter(status for status, _ in DESIGNED.values())
    assert set(job.frame["status"]) == {"ok", "infra_error", "timeout", "flagged"}
    reasons = {reason for _, reason in DESIGNED.values() if reason is not None}
    assert set(job.frame["status_reason"].dropna()) == reasons


def test_rewards_match_harbor_job_stats(rows: dict[str, dict[str, Any]]) -> None:
    rewarded: set[str] = set()
    for key, stats in STATS["evals"].items():
        agent, model_name, dataset = key.split("__")
        for value, names in stats["reward_stats"]["reward"].items():
            for name in names:
                row = rows[name]
                assert row["outcome"] == float(value)
                assert row["agent"] == agent and row["dataset"] == dataset
                assert row["model"].split("/")[-1] == model_name
            rewarded.update(names)
    assert len(rewarded) == sum(stats["n_trials"] for stats in STATS["evals"].values())
    for name in rows.keys() - rewarded:
        assert rows[name]["outcome"] == 0.0
        assert rows[name]["status"] != "ok"


def test_exceptions_match_harbor_job_stats(rows: dict[str, dict[str, Any]]) -> None:
    errored: dict[str, str] = {}
    for stats in STATS["evals"].values():
        for exception_type, names in stats["exception_stats"].items():
            errored.update(dict.fromkeys(names, exception_type))
    assert len(errored) == STATS["n_errored_trials"]
    assert {n: r["exception_type"] for n, r in rows.items() if r["exception_type"]} == errored


def test_counts_and_tokens_match_harbor_job_stats(rows: dict[str, dict[str, Any]]) -> None:
    finished = [r for r in rows.values() if r["status_reason"] != "no_result"]
    assert len(finished) == STATS["n_completed_trials"]
    assert len(rows) - len(finished) == STATS["n_pending_trials"]
    for column in ("input_tokens", "cache_tokens", "output_tokens"):
        assert sum(r[column] for r in finished) == STATS[f"n_{column}"]
    assert STATS["cost_usd"] is None
    assert all(r["cost_usd"] is None for r in rows.values())


def test_multi_step_trial_sums_its_steps(rows: dict[str, dict[str, Any]]) -> None:
    raw = RAW["mock-build__mock005"]
    steps = raw["step_results"]
    # Harbor records usage and agent timing per step, not on a multi-step trial itself.
    assert len(steps) == 2
    assert raw["agent_result"] is None and raw["agent_execution"] is None
    row = rows["mock-build__mock005"]
    for column in ("input_tokens", "cache_tokens", "output_tokens"):
        assert row[column] == sum(step["agent_result"][f"n_{column}"] for step in steps)
    timings = [step["agent_execution"] for step in steps]
    assert row["agent_time"] == sum(seconds(t["started_at"], t["finished_at"]) for t in timings)
    # The outcome is Harbor's trial-level reward (its aggregate over steps), not a step reward.
    assert row["outcome"] == raw["verifier_result"]["rewards"]["reward"]
    assert row["outcome"] != steps[0]["verifier_result"]["rewards"]["reward"]


def test_agent_written_result_json_is_ignored(rows: dict[str, dict[str, Any]]) -> None:
    log = load(JOB / "mock-hello__mock000" / "agent" / "result.json")
    assert "task_name" not in log and "n_total_trials" not in log  # an agent log, not Harbor's
    assert rows["mock-hello__mock000"] == EXPECTED["mock-hello__mock000"]
    assert rows_of(read_harbor(JOB / "mock-hello__mock000")).keys() == {"mock-hello__mock000"}


def test_include_incomplete_false_drops_trials_without_a_result() -> None:
    kept = rows_of(read_harbor(JOB, include_incomplete=False))
    assert kept == {name: EXPECTED[name] for name in RAW}


def test_agent_model_label() -> None:
    res = read_harbor(JOB, model_label="agent_model")
    assert sorted(res.models) == sorted(
        f"{a['name']}__{a['model_name']}" for a in JOB_CONFIG["agents"]
    )
    for name, row in rows_of(res).items():
        agent = (RAW[name]["config"] if name in RAW else PENDING[name])["agent"]
        assert row["model"] == f"{agent['name']}__{agent['model_name']}"
        assert row["agent"] == agent["name"]


def test_unknown_model_label() -> None:
    label: Any = "agent"
    with pytest.raises(ValueError, match="model_label must be 'model' or 'agent_model'"):
        read_harbor(JOB, model_label=label)


def test_reward_key() -> None:
    assert rows_of(read_harbor(JOB, reward_key="reward")) == EXPECTED
    with pytest.raises(SchemaError, match=re.escape("no reward 'accuracy'; rewards are reward")):
        read_harbor(JOB, reward_key="accuracy")


# Inputs other than a job directory.


def test_directory_of_job_directories() -> None:
    assert rows_of(read_harbor(JOBS)) == EXPECTED


@pytest.mark.parametrize("name", sorted(RAW))
def test_single_trial_directory_and_result_file(name: str) -> None:
    expected = {name: {**EXPECTED[name], "run": "0"}}
    assert rows_of(read_harbor(JOB / name)) == expected
    assert rows_of(read_harbor(JOB / name / "result.json")) == expected


def test_trial_directory_without_a_result() -> None:
    ((name, config),) = PENDING.items()
    # Read alone, there is no finished trial to take the task name from: it comes from the path.
    task = PurePosixPath(config["task"]["path"]).name
    assert rows_of(read_harbor(JOB / name)) == {name: {**EXPECTED[name], "task": task, "run": "0"}}
    with pytest.raises(SchemaError, match="no Harbor trial results found"):
        read_harbor(JOB / name, include_incomplete=False)


def test_trial_directory_without_a_result_ignores_agent_logs(tmp_path: Path) -> None:
    name = next(iter(PENDING))
    copied = tmp_path / name
    shutil.copytree(JOB / name, copied)
    (copied / "agent").mkdir(exist_ok=True)
    (copied / "agent" / "result.json").write_text('{"note": "agent log"}\n')
    ((_, row),) = rows_of(read_harbor(copied)).items()
    assert (row["status"], row["status_reason"]) == DESIGNED[name]


def test_two_jobs_number_runs_across_jobs(tmp_path: Path) -> None:
    for job_name in ("job-a", "job-b"):
        shutil.copytree(JOB, tmp_path / job_name)
    res = read_harbor(tmp_path)
    assert len(res) == 2 * len(TRIAL_DIRS)
    runs = res.frame.groupby(["model", "task"])["run"].agg(set)
    assert all(r == {str(i) for i in range(2 * JOB_CONFIG["n_attempts"])} for r in runs)
    assert Counter(res.frame["status"]) == Counter(2 * [status for status, _ in DESIGNED.values()])


def test_regrade_source_trials_are_not_read(tmp_path: Path) -> None:
    # A regrade job links its hub source trials under .sources/<trial id>/<trial name>.
    copied = tmp_path / JOB.name
    shutil.copytree(JOB, copied)
    source = copied / ".sources" / RAW["mock-hello__mock000"]["id"] / "mock-hello__mock000"
    shutil.copytree(JOB / "mock-hello__mock000", source)
    assert rows_of(read_harbor(copied)) == EXPECTED
    assert rows_of(read_harbor(tmp_path)) == EXPECTED


def test_missing_trial_directories_warn(tmp_path: Path) -> None:
    copied = tmp_path / JOB.name
    shutil.copytree(JOB, copied)
    shutil.rmtree(copied / "mock-hello__mock001")
    declared = JOB_RESULT["n_total_trials"]
    message = f"job declares {declared} trials but {declared - 1} trial directories were found"
    with pytest.warns(UserWarning, match=message):
        res = read_harbor(copied)
    assert rows_of(res).keys() == EXPECTED.keys() - {"mock-hello__mock001"}


def test_job_without_trial_directories(tmp_path: Path) -> None:
    copied = tmp_path / JOB.name
    copied.mkdir()
    for name in ("config.json", "result.json"):
        shutil.copy(JOB / name, copied / name)
    with (
        pytest.warns(UserWarning, match="but 0 trial directories were found"),
        pytest.raises(SchemaError, match=re.escape(f"{copied}: no Harbor trial results found")),
    ):
        read_harbor(copied)


# Missing, empty, and malformed inputs.


def test_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(SchemaError, match=re.escape(f"{missing}: no such file or directory")):
        read_harbor(missing)


def test_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(SchemaError, match=re.escape(f"{tmp_path}: no Harbor trial results found")):
        read_harbor(tmp_path)


@pytest.mark.parametrize(
    "path",
    [
        JOB / "result.json",
        JOB / "config.json",
        JOB / "mock-hello__mock000" / "config.json",
        JOB / "mock-hello__mock000" / "agent" / "result.json",
    ],
)
def test_file_that_is_not_a_trial_result(path: Path) -> None:
    with pytest.raises(SchemaError, match=re.escape(f"{path}: not a Harbor trial result")):
        read_harbor(path)


TRUNCATED = (JOB / "mock-hello__mock000" / "result.json").read_text()[:300]
MALFORMED = [
    pytest.param("", "invalid JSON", id="empty"),
    pytest.param(TRUNCATED, "invalid JSON", id="truncated"),
    pytest.param("[1, 2]", "neither a Harbor trial result nor a job result", id="list"),
    pytest.param(
        '{"trial_name": "t1", "task_name": "x"}',
        "neither a Harbor trial result nor a job result",
        id="no-config",
    ),
    pytest.param(
        '{"trial_name": "t1", "task_name": "x", "config": []}',
        "neither a Harbor trial result nor a job result",
        id="config-not-an-object",
    ),
]


@pytest.mark.parametrize(("text", "message"), MALFORMED)
def test_malformed_result_json_in_a_directory(tmp_path: Path, text: str, message: str) -> None:
    result = tmp_path / "t1" / "result.json"
    result.parent.mkdir()
    result.write_text(text)
    with pytest.raises(SchemaError, match=re.escape(f"{result}: {message}")):
        read_harbor(tmp_path)


@pytest.mark.parametrize(("text", "message"), MALFORMED)
def test_malformed_result_file(tmp_path: Path, text: str, message: str) -> None:
    result = tmp_path / "result.json"
    result.write_text(text)
    expected = "invalid JSON" if message == "invalid JSON" else "not a Harbor trial result"
    with pytest.raises(SchemaError, match=re.escape(f"{result}: {expected}")):
        read_harbor(result)


def test_malformed_config_of_a_trial_without_a_result(tmp_path: Path) -> None:
    config = tmp_path / "t1" / "config.json"
    config.parent.mkdir()
    config.write_text("{")
    with pytest.raises(SchemaError, match=re.escape(f"{config}: invalid JSON")):
        read_harbor(tmp_path)


def test_directory_with_a_foreign_config_is_skipped(tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text('{"job_name": "x", "tasks": []}')
    write_trial(tmp_path / "t1", trial("t1"))
    assert rows_of(read_harbor(tmp_path)).keys() == {"t1"}


def test_only_unfinished_trials_and_include_incomplete_false(tmp_path: Path) -> None:
    write_pending(tmp_path / "late", {"path": "tasks/mock-hello"})
    assert len(read_harbor(tmp_path)) == 1
    with pytest.raises(SchemaError, match="no Harbor trial results found"):
        read_harbor(tmp_path, include_incomplete=False)


# Status mapping.


def test_exception_sets_are_the_documented_ones() -> None:
    assert frozenset(LIMIT_TYPES) == TIMEOUT_EXCEPTIONS
    assert frozenset({"AgentSafetyRefusalError"}) == REAL_OUTCOME_EXCEPTIONS
    assert not TIMEOUT_EXCEPTIONS & set(INFRA_TYPES)


@pytest.mark.parametrize("kind", LIMIT_TYPES)
def test_limit_exceptions_are_timeouts_and_keep_the_reward(tmp_path: Path, kind: str) -> None:
    data = trial(exception_info=exception_info(kind), verifier_result={"rewards": {"reward": 1.0}})
    row = read_one(tmp_path, data)
    assert (row["status"], row["status_reason"], row["exception_type"]) == ("timeout", kind, kind)
    assert row["outcome"] == 1.0


@pytest.mark.parametrize("kind", LIMIT_TYPES)
def test_limit_exceptions_without_a_reward(tmp_path: Path, kind: str) -> None:
    row = read_one(tmp_path, trial(exception_info=exception_info(kind), verifier_result=None))
    assert (row["status"], row["status_reason"], row["outcome"]) == ("timeout", kind, 0.0)


@pytest.mark.parametrize("kind", INFRA_TYPES)
def test_other_exceptions_are_infra_errors(tmp_path: Path, kind: str) -> None:
    row = read_one(tmp_path, trial(exception_info=exception_info(kind), verifier_result=None))
    assert (row["status"], row["status_reason"], row["outcome"]) == ("infra_error", kind, 0.0)
    with_reward = trial("t2", exception_info=exception_info(kind))
    row = read_one(tmp_path, with_reward)
    assert (row["status"], row["outcome"]) == ("infra_error", 1.0)


def test_safety_refusal_is_a_scored_outcome(tmp_path: Path) -> None:
    kind = "AgentSafetyRefusalError"
    row = read_one(tmp_path, trial(exception_info=exception_info(kind), verifier_result=None))
    assert (row["status"], row["status_reason"], row["outcome"]) == ("ok", kind, 0.0)


def test_exception_without_a_type(tmp_path: Path) -> None:
    info = exception_info("unused")
    del info["exception_type"]
    row = read_one(tmp_path, trial(exception_info=info))
    assert (row["status"], row["status_reason"], row["exception_type"]) == (
        "infra_error",
        "unknown",
        "unknown",
    )


@pytest.mark.parametrize(
    "verifier_result", [None, {"rewards": None}, {"rewards": {}}], ids=["none", "null", "empty"]
)
def test_missing_reward_without_an_exception(tmp_path: Path, verifier_result: Any) -> None:
    row = read_one(tmp_path, trial(verifier_result=verifier_result))
    assert (row["status"], row["status_reason"], row["outcome"]) == (
        "infra_error",
        "no_reward",
        0.0,
    )


def test_disabled_verifier_is_flagged_before_exceptions(tmp_path: Path) -> None:
    data = trial(exception_info=exception_info("AgentTimeoutError"), verifier_result=None)
    data["config"]["verifier"]["disable"] = True
    row = read_one(tmp_path, data)
    assert (row["status"], row["status_reason"], row["outcome"]) == (
        "flagged",
        "verifier_disabled",
        0.0,
    )


def test_install_only_trial_is_flagged(tmp_path: Path) -> None:
    # Harbor's TrialConfig validator also sets verifier.disable on install-only trials, and
    # the agent never runs.
    data = trial(verifier_result=None, agent_result=None, agent_execution=None)
    data["config"]["install_only"] = True
    data["config"]["verifier"]["disable"] = True
    row = read_one(tmp_path, data)
    assert (row["status"], row["status_reason"], row["outcome"]) == ("flagged", "install_only", 0.0)
    assert row["agent_time"] is None and row["input_tokens"] is None


def test_multi_step_exception_is_the_first_one_recorded(tmp_path: Path) -> None:
    data = copy.deepcopy(RAW["mock-build__mock005"])
    data["step_results"][1]["exception_info"] = exception_info("AgentTimeoutError")
    row = read_one(tmp_path / "a", data)
    assert (row["status"], row["status_reason"]) == ("timeout", "AgentTimeoutError")
    data["step_results"][0]["exception_info"] = exception_info("ApiRateLimitError")
    row = read_one(tmp_path / "b", data)
    assert (row["status"], row["status_reason"]) == ("infra_error", "ApiRateLimitError")
    data["exception_info"] = exception_info("CancelledError")
    row = read_one(tmp_path / "c", data)
    assert (row["status"], row["status_reason"]) == ("infra_error", "CancelledError")


# Rewards.


@pytest.mark.parametrize(
    ("rewards", "options", "outcome"),
    [
        ({"reward": 1}, {}, 1.0),
        ({"accuracy": 0.25}, {}, 0.25),
        ({"reward": 0.5, "accuracy": 1.0}, {}, 0.5),
        ({"reward": 0.5, "accuracy": 1.0}, {"reward_key": "accuracy"}, 1.0),
        ({"a": 0.2, "b": 0.4}, {"reward_key": "b"}, 0.4),
    ],
)
def test_reward_selection(
    tmp_path: Path, rewards: dict[str, Any], options: dict[str, Any], outcome: float
) -> None:
    row = read_one(tmp_path, trial(verifier_result={"rewards": rewards}), **options)
    assert (row["outcome"], row["status"]) == (outcome, "ok")


def test_partial_credit_is_not_binary(tmp_path: Path) -> None:
    write_trial(tmp_path / "t1", trial(verifier_result={"rewards": {"reward": 0.5}}))
    assert not read_harbor(tmp_path).is_binary


@pytest.mark.parametrize(
    ("rewards", "options", "message"),
    [
        (
            {"a": 0.2, "b": 0.4},
            {},
            "several rewards (a, b) and none named 'reward'; pass reward_key",
        ),
        ({"reward": 1.0}, {"reward_key": "accuracy"}, "no reward 'accuracy'; rewards are reward"),
        ({"reward": True}, {}, "reward 'reward' is not a number: True"),
        ({"reward": "1"}, {}, "reward 'reward' is not a number: '1'"),
        ({"reward": None}, {}, "reward 'reward' is not a number: None"),
        ({"reward": 1.5}, {}, "reward 'reward': outcome 1.5 is outside [0, 1]"),
        ({"reward": -0.25}, {}, "reward 'reward': outcome -0.25 is outside [0, 1]"),
    ],
)
def test_bad_rewards(
    tmp_path: Path, rewards: dict[str, Any], options: dict[str, Any], message: str
) -> None:
    trial_dir = write_trial(tmp_path / "t1", trial(verifier_result={"rewards": rewards}))
    with pytest.raises(SchemaError, match=f"^{re.escape(f't1: {message}')}$"):
        read_harbor(trial_dir, **options)


# Agent and model from the trial config and agent_info.


@pytest.mark.parametrize(
    ("agent_config", "agent_info", "agent", "model"),
    [
        pytest.param(
            {"name": "terminus-2", "model_name": "prov/m-x"},
            {"name": "terminus-2", "version": "1", "model_info": {"name": "m-y", "provider": "p"}},
            "terminus-2",
            "prov/m-x",
            id="config-model-first",
        ),
        pytest.param(
            {"name": "terminus-2", "model_name": None},
            {"name": "terminus-2", "version": "1", "model_info": {"name": "m-x", "provider": "p"}},
            "terminus-2",
            "p/m-x",
            id="model-info-with-provider",
        ),
        pytest.param(
            {"name": "terminus-2"},
            {"name": "terminus-2", "version": "1", "model_info": {"name": "m-x", "provider": None}},
            "terminus-2",
            "m-x",
            id="model-info-without-provider",
        ),
        pytest.param(
            {"name": "oracle", "model_name": None},
            {"name": "oracle", "version": "1", "model_info": None},
            "oracle",
            "oracle",
            id="no-model",
        ),
        pytest.param(
            {"name": None, "import_path": "pkg.mod:Agent", "model_name": "prov/m-x"},
            {"name": "my-agent", "version": "1", "model_info": None},
            "my-agent",
            "prov/m-x",
            id="agent-info-name",
        ),
        pytest.param(
            {"name": None, "import_path": "pkg.mod:Agent", "model_name": None},
            None,
            "pkg.mod:Agent",
            "pkg.mod:Agent",
            id="import-path",
        ),
        pytest.param(None, None, "oracle", "oracle", id="nothing"),
    ],
)
def test_agent_and_model(
    tmp_path: Path,
    agent_config: dict[str, Any] | None,
    agent_info: dict[str, Any] | None,
    agent: str,
    model: str,
) -> None:
    data = trial(agent_info=agent_info)
    if agent_config is None:
        del data["config"]["agent"]
    else:
        data["config"]["agent"] = agent_config
    row = read_one(tmp_path / "plain", data)
    assert (row["agent"], row["model"]) == (agent, model)
    assert row["agent_version"] == (agent_info or {}).get("version")
    row = read_one(tmp_path / "labeled", data, model_label="agent_model")
    assert (row["agent"], row["model"]) == (agent, f"{agent}__{model}")


# Trials without a result: task names.


def test_unfinished_trial_takes_the_task_name_of_a_finished_one(tmp_path: Path) -> None:
    finished = trial("done", task_name="hello-renamed")
    write_trial(tmp_path / "done", finished)
    write_pending(tmp_path / "late", {"path": finished["config"]["task"]["path"]})
    tasks = {name: row["task"] for name, row in rows_of(read_harbor(tmp_path)).items()}
    assert tasks == {"done": "hello-renamed", "late": "hello-renamed"}


@pytest.mark.parametrize(
    ("task", "name"),
    [
        ({"name": "org/registry-task", "source": "reg"}, "org/registry-task"),
        ({"path": "tasks/deep/thing"}, "thing"),
        ({"git_url": "https://example.invalid/tasks.git"}, "late"),
    ],
)
def test_unfinished_trial_task_name_fallbacks(
    tmp_path: Path, task: dict[str, Any], name: str
) -> None:
    write_pending(tmp_path / "late", task)
    ((_, row),) = rows_of(read_harbor(tmp_path)).items()
    assert (row["task"], row["dataset"]) == (name, task.get("source"))
    assert (row["status"], row["status_reason"], row["run"]) == ("infra_error", "no_result", "0")


# Usage, timing, and run order.


def test_missing_usage_and_timing(tmp_path: Path) -> None:
    data = trial(agent_result=None, agent_execution=None, started_at=None, finished_at=None)
    row = read_one(tmp_path, data)
    assert all(row[column] is None for column in NUMERIC)
    assert row["started_at"] is None


def test_unparsable_timestamps(tmp_path: Path) -> None:
    timing = {"started_at": "yesterday", "finished_at": RAW["mock-hello__mock000"]["finished_at"]}
    row = read_one(tmp_path, trial(agent_execution=timing, finished_at=7))
    assert (row["agent_time"], row["total_time"]) == (None, None)


def test_cost_and_partial_usage_are_summed_over_steps(tmp_path: Path) -> None:
    data = copy.deepcopy(RAW["mock-build__mock005"])
    first, second = (step["agent_result"] for step in data["step_results"])
    first["cost_usd"], second["cost_usd"] = 0.25, 0.5
    second["n_cache_tokens"] = None
    row = read_one(tmp_path, data)
    assert row["cost_usd"] == 0.75
    assert row["cache_tokens"] == first["n_cache_tokens"]
    assert row["input_tokens"] == first["n_input_tokens"] + second["n_input_tokens"]


def test_step_without_usage_or_timing_is_skipped(tmp_path: Path) -> None:
    data = copy.deepcopy(RAW["mock-build__mock005"])
    data["step_results"][1]["agent_result"] = None
    data["step_results"][1]["agent_execution"] = None
    row = read_one(tmp_path, data)
    first = data["step_results"][0]
    assert row["input_tokens"] == first["agent_result"]["n_input_tokens"]
    timing = first["agent_execution"]
    assert row["agent_time"] == seconds(timing["started_at"], timing["finished_at"])


def test_runs_follow_start_time_then_trial_name(tmp_path: Path) -> None:
    starts = {
        "t-c": "2026-10-23T02:00:00Z",
        "t-b": "2026-10-23T02:10:00Z",
        "t-a": "2026-10-23T02:10:00Z",
        "t-0": None,
    }
    for name, start in starts.items():
        write_trial(tmp_path / name, trial(name, started_at=start))
    runs = {name: row["run"] for name, row in rows_of(read_harbor(tmp_path)).items()}
    assert runs == {"t-c": "0", "t-a": "1", "t-b": "2", "t-0": "3"}
