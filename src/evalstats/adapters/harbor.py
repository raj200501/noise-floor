"""Reader for Harbor job outputs (the harness behind Terminal-Bench 2.0).

The files are parsed directly; the ``harbor`` package is not imported. Formats follow harbor
0.24.0 (``harbor/models/trial/result.py``, ``harbor/models/job/result.py``,
``harbor/trial/trial.py``, ``harbor/job.py``).

Files read. A job directory holds ``config.json`` (JobConfig), ``result.json`` (JobResult,
written without trial results), and one directory per trial holding ``config.json``
(TrialConfig) and ``result.json`` (TrialResult). The reader accepts a job directory, a
directory of job directories, a single trial directory, or a single trial ``result.json``.
It never reads below a trial directory, so agent logs (some agents write their own
``agent/result.json``) are ignored. Harbor deletes a trial directory before retrying it, so
each trial directory is one attempt.

Rows. One row per trial. ``model`` is ``config.agent.model_name``, else
``agent_info.model_info`` as ``<provider>/<name>``, else the agent name (for agents such as
``oracle`` that use no model). With ``model_label="agent_model"`` the model is
``<agent>__<model>``, as in Harbor's own evals keys, so two agents with one model stay apart.
``task`` is ``task_name``. ``run`` numbers the trials of each (model, task) from 0 in order of
``started_at`` and then trial name. ``outcome`` is ``verifier_result.rewards["reward"]``, or the
only reward when there is one key (the rule Harbor's uploader uses), or
``rewards[reward_key]``; a reward outside [0, 1] raises :class:`SchemaError`.

Status mapping, first match wins (``status_reason`` in brackets). The trial exception is
``exception_info``, or for multi-step trials the first step's ``exception_info``.

1. ``config.install_only`` or ``config.verifier.disable`` is true: ``flagged``
   [``install_only`` or ``verifier_disabled``]. Harbor also sets ``verifier.disable`` on
   install-only trials, so ``install_only`` is checked first.
2. The exception type is one of :data:`TIMEOUT_EXCEPTIONS` (agent or verifier time limit,
   context window, output token limit): ``timeout`` [the type]. Harbor still runs the verifier
   after an agent timeout, so such a trial can carry a reward.
3. The exception type is ``AgentSafetyRefusalError``: ``ok`` [the type]. Harbor documents a
   provider safety refusal as a real reward 0 outcome, not a flaky error.
4. Any other exception type (environment start or build, agent setup, API, network,
   authentication, cancellation, reward file or verifier output errors). This includes
   ``EnvironmentStartTimeoutError`` and ``AgentSetupTimeoutError``, whose time limits cover
   infrastructure rather than the agent: ``infra_error`` [the type].
5. No reward: ``infra_error`` [``no_reward``].
6. A trial directory with ``config.json`` but no ``result.json`` (the process died before
   Harbor wrote the result), including one passed as ``job_dir``: ``infra_error``
   [``no_result``]; ``include_incomplete=False`` drops these rows. Its task name comes from a
   finished trial with the same task config, else from the task config's ``name`` or ``path``.
7. Otherwise ``ok``.

Non-ok rows keep the trial's reward when it has one and 0 otherwise. A job whose
``result.json`` declares more trials (``n_total_trials``) than its directory holds triggers a
warning: trials that never started leave no files and cannot appear in the table.

Covariates: ``status``, ``status_reason``, ``trial_name``, ``agent``, ``agent_version``,
``exception_type``, ``input_tokens`` (including cache), ``cache_tokens``, ``output_tokens``,
``cost_usd``, ``agent_time`` and ``total_time`` (seconds), ``started_at``, and ``dataset``
(the trial's ``source``). Harbor records a multi-step trial's usage and agent timing on its
steps, so for those trials the token, cost, and ``agent_time`` columns sum over the steps.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from evalstats.adapters._common import (
    FLAGGED,
    INFRA_ERROR,
    OK,
    TIMEOUT,
    as_mapping,
    build_results,
    check_unit,
    load_json_file,
)
from evalstats.schema import EvalResults, SchemaError

TIMEOUT_EXCEPTIONS = frozenset(
    {
        "AgentTimeoutError",
        "VerifierTimeoutError",
        "ContextWindowExceededError",
        "ContextLengthExceededError",
        "OutputTokenExceededError",
        "OutputLengthExceededError",
    }
)
"""Harbor exception types that mean a time or token limit ended the trial."""

REAL_OUTCOME_EXCEPTIONS = frozenset({"AgentSafetyRefusalError"})
"""Harbor exception types that Harbor counts as a real (scored) outcome."""

_MAX_DEPTH = 2  # jobs directory, job directory, trial directory
_COLUMNS = (
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
_NUMERIC = ("input_tokens", "cache_tokens", "output_tokens", "cost_usd", "agent_time", "total_time")
_TOKEN_KEYS = {
    "input_tokens": "n_input_tokens",
    "cache_tokens": "n_cache_tokens",
    "output_tokens": "n_output_tokens",
    "cost_usd": "cost_usd",
}


@dataclass
class _Scan:
    trials: list[tuple[str, Mapping[str, Any]]] = field(default_factory=list)
    pending: list[tuple[str, Mapping[str, Any]]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def read_harbor(
    job_dir: str | Path,
    *,
    reward_key: str | None = None,
    model_label: Literal["model", "agent_model"] = "model",
    include_incomplete: bool = True,
) -> EvalResults:
    """Read Harbor trial results from a job directory (or a trial directory or file).

    ``job_dir`` is a job directory, a directory of job directories, a trial directory, or a
    trial ``result.json``. ``reward_key`` names the reward used as the outcome when the
    verifier reports several. ``model_label="agent_model"`` labels models
    ``<agent>__<model>``. ``include_incomplete=False`` drops the ``infra_error`` rows for trial
    directories without a result. The module docstring gives the row layout and the status
    mapping. Raises :class:`SchemaError` if a file is not a Harbor result, a reward is
    ambiguous, not a number, or outside [0, 1], or no trials are found.
    """
    if model_label not in ("model", "agent_model"):
        raise ValueError(f"model_label must be 'model' or 'agent_model', not {model_label!r}")
    path = Path(job_dir)
    scan = _Scan()
    if path.is_file():
        data = load_json_file(path)
        if not _is_trial(data):
            raise SchemaError(f"{path}: not a Harbor trial result (TrialResult)")
        scan.trials.append((path.parent.name or path.name, data))
    elif path.is_dir():
        _scan_dir(path, 0, scan)
    else:
        raise SchemaError(f"{path}: no such file or directory")
    for note in scan.notes:
        warnings.warn(note, stacklevel=2)
    rows = [_trial_row(label, data, reward_key, model_label) for label, data in scan.trials]
    if include_incomplete:
        names = {
            _task_key(as_mapping(d["config"], label).get("task")): str(d["task_name"])
            for label, d in scan.trials
        }
        rows += [_pending_row(label, cfg, names, model_label) for label, cfg in scan.pending]
    _number_runs(rows)
    if not rows:
        raise SchemaError(f"{path}: no Harbor trial results found")
    return build_results(rows, source=str(path), columns=_COLUMNS, numeric=_NUMERIC)


def _is_trial(data: Any) -> bool:
    return (
        isinstance(data, Mapping)
        and "trial_name" in data
        and "task_name" in data
        and isinstance(data.get("config"), Mapping)
    )


def _is_job(data: Any) -> bool:
    return isinstance(data, Mapping) and "n_total_trials" in data and "stats" in data


def _is_trial_config(data: Any) -> bool:
    return (
        isinstance(data, Mapping) and "trial_name" in data and isinstance(data.get("task"), Mapping)
    )


def _scan_dir(directory: Path, depth: int, scan: _Scan) -> None:
    result = directory / "result.json"
    if result.is_file():
        data = load_json_file(result)
        if _is_trial(data):
            scan.trials.append((directory.name, data))
            return
        if not _is_job(data):
            raise SchemaError(f"{result}: neither a Harbor trial result nor a job result")
        before = len(scan.trials) + len(scan.pending)
        _scan_children(directory, depth, scan)
        found = len(scan.trials) + len(scan.pending) - before
        declared = data.get("n_total_trials")
        if isinstance(declared, int) and declared > found:
            scan.notes.append(
                f"{directory}: job declares {declared} trials but {found} trial directories "
                "were found; trials that never started are not in the table"
            )
        return
    config = directory / "config.json"
    if config.is_file():
        data = load_json_file(config)
        if _is_trial_config(data):
            scan.pending.append((directory.name, data))
            return
    _scan_children(directory, depth, scan)


def _scan_children(directory: Path, depth: int, scan: _Scan) -> None:
    if depth < _MAX_DEPTH:
        for child in sorted(p for p in directory.iterdir() if p.is_dir()):
            _scan_dir(child, depth + 1, scan)


def _task_key(task: Any) -> tuple[str | None, ...]:
    task = task if isinstance(task, Mapping) else {}
    return tuple(task.get(k) for k in ("name", "path", "git_url"))


def _model(
    agent_config: Mapping[str, Any], agent_info: Mapping[str, Any], label: str
) -> tuple[str, str]:
    agent = str(
        agent_config.get("name")
        or agent_info.get("name")
        or agent_config.get("import_path")
        or "oracle"  # Harbor's default agent when none is configured
    )
    model = agent_config.get("model_name")
    info = agent_info.get("model_info")
    if not model and isinstance(info, Mapping) and info.get("name"):
        provider = info.get("provider")
        model = f"{provider}/{info['name']}" if provider else info["name"]
    model = str(model or agent)
    return agent, (f"{agent}__{model}" if label == "agent_model" else model)


def _trial_row(
    label: str, data: Mapping[str, Any], reward_key: str | None, model_label: str
) -> dict[str, Any]:
    config = as_mapping(data["config"], f"{label}: config")
    agent_info = data.get("agent_info")
    agent_info = agent_info if isinstance(agent_info, Mapping) else {}
    agent_config = config.get("agent")
    agent, model = _model(
        agent_config if isinstance(agent_config, Mapping) else {}, agent_info, model_label
    )
    exception = _exception_type(data)
    reward = _reward(data.get("verifier_result"), reward_key, label)
    status, reason = _status(config, exception, reward)
    return {
        "model": model,
        "task": str(data["task_name"]),
        "outcome": 0.0 if reward is None else reward,
        "status": status,
        "status_reason": reason,
        "trial_name": str(data["trial_name"]),
        "agent": agent,
        "agent_version": agent_info.get("version"),
        "exception_type": exception,
        **_tokens(data),
        "agent_time": _agent_time(data),
        "total_time": _seconds(data.get("started_at"), data.get("finished_at")),
        "started_at": data.get("started_at"),
        "dataset": data.get("source"),
    }


def _pending_row(
    label: str,
    config: Mapping[str, Any],
    names: Mapping[tuple[str | None, ...], str],
    model_label: str,
) -> dict[str, Any]:
    task = as_mapping(config["task"], f"{label}: task")
    path = task.get("path")
    task_name = (
        names.get(_task_key(task))
        or task.get("name")
        or (PurePosixPath(str(path)).name if path else None)
        or label
    )
    agent_config = config.get("agent")
    agent, model = _model(
        agent_config if isinstance(agent_config, Mapping) else {}, {}, model_label
    )
    return {
        "model": model,
        "task": str(task_name),
        "outcome": 0.0,
        "status": INFRA_ERROR,
        "status_reason": "no_result",
        "trial_name": str(config["trial_name"]),
        "agent": agent,
        "dataset": task.get("source"),
    }


def _exception_type(data: Mapping[str, Any]) -> str | None:
    infos = [data.get("exception_info")]
    steps = data.get("step_results")
    if isinstance(steps, list):
        infos += [s.get("exception_info") for s in steps if isinstance(s, Mapping)]
    for info in infos:
        if isinstance(info, Mapping):
            return str(info.get("exception_type") or "unknown")
    return None


def _reward(verifier: Any, reward_key: str | None, label: str) -> float | None:
    rewards = verifier.get("rewards") if isinstance(verifier, Mapping) else None
    if not isinstance(rewards, Mapping) or not rewards:
        return None
    if reward_key is not None:
        if reward_key not in rewards:
            raise SchemaError(
                f"{label}: no reward {reward_key!r}; rewards are {', '.join(map(str, rewards))}"
            )
        key = reward_key
    elif "reward" in rewards:
        key = "reward"
    elif len(rewards) == 1:
        key = str(next(iter(rewards)))
    else:
        raise SchemaError(
            f"{label}: several rewards ({', '.join(map(str, rewards))}) and none named "
            "'reward'; pass reward_key"
        )
    value = rewards[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SchemaError(f"{label}: reward {key!r} is not a number: {value!r}")
    return check_unit(float(value), f"{label}: reward {key!r}")


def _status(
    config: Mapping[str, Any], exception: str | None, reward: float | None
) -> tuple[str, str | None]:
    if config.get("install_only"):
        return FLAGGED, "install_only"
    verifier = config.get("verifier")
    if isinstance(verifier, Mapping) and verifier.get("disable"):
        return FLAGGED, "verifier_disabled"
    if exception is None:
        return (INFRA_ERROR, "no_reward") if reward is None else (OK, None)
    if exception in TIMEOUT_EXCEPTIONS:
        return TIMEOUT, exception
    return (OK if exception in REAL_OUTCOME_EXCEPTIONS else INFRA_ERROR), exception


def _per_step(data: Mapping[str, Any], key: str) -> list[Any]:
    """The trial's ``key``, or each step's ``key`` if the trial has none (multi-step trials)."""
    value = data.get(key)
    steps = data.get("step_results")
    if not isinstance(value, Mapping) and isinstance(steps, list):
        return [s.get(key) for s in steps if isinstance(s, Mapping)]
    return [value]


def _tokens(data: Mapping[str, Any]) -> dict[str, float | None]:
    totals: dict[str, float | None] = dict.fromkeys(_TOKEN_KEYS)
    for context in _per_step(data, "agent_result"):
        if not isinstance(context, Mapping):
            continue
        for column, key in _TOKEN_KEYS.items():
            value = context.get(key)
            if isinstance(value, int | float):
                totals[column] = (totals[column] or 0.0) + float(value)
    return totals


def _agent_time(data: Mapping[str, Any]) -> float | None:
    total: float | None = None
    for timing in _per_step(data, "agent_execution"):
        if isinstance(timing, Mapping):
            seconds = _seconds(timing.get("started_at"), timing.get("finished_at"))
            if seconds is not None:
                total = (total or 0.0) + seconds
    return total


def _seconds(start: Any, end: Any) -> float | None:
    try:
        delta = datetime.fromisoformat(str(end)) - datetime.fromisoformat(str(start))
    except (TypeError, ValueError):
        return None
    return delta.total_seconds()


def _number_runs(rows: list[dict[str, Any]]) -> None:
    def order(row: dict[str, Any]) -> tuple[bool, str, str]:
        started = row.get("started_at")
        return (started is None, str(started or ""), row["trial_name"])

    counts: dict[tuple[str, str], int] = {}
    for row in sorted(rows, key=order):
        key = (row["model"], row["task"])
        row["run"] = str(counts.get(key, 0))
        counts[key] = counts.get(key, 0) + 1
