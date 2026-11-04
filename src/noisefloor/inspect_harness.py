"""Inspect's basic agent on one nf-tools task: the second level of the harness factor.

The suite's tools (all but ``submit``) become Inspect tools bound to this episode's
mock file system; Inspect's ``basic_agent`` supplies its own ``submit`` tool. The
model is reached through the project proxy (``openai-api/nf/<run id>``), so every
call is ledgered and cached like minimal-agent's. Grading uses the suite's grader on
the submitted answer and the final mock file system, exactly as for minimal-agent.

End reasons: ``submitted`` (Inspect accepted a submit call), ``stopped`` (the agent
stopped without an accepted submit; with ``max_attempts=1``, a submit that fails
Inspect's argument validation also ends the sample, with no answer), ``step_limit`` or
``wall_clock`` (an Inspect message or time limit ended the sample), ``context_limit``
(the conversation outgrew the context window), ``backend_error`` (the model call
failed), ``harness_error`` (any other sample error).
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_ai import Task as InspectTask
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.solver import basic_agent
from inspect_ai.tool import ToolDef, ToolParam, ToolParams

from noisefloor.suite import Task, ToolEnv, ToolSpec, episode_status, generate

BACKEND_MARKERS = ("connection", "502", "503", "timed out", "server error", "apiconnection")


@dataclass(frozen=True)
class InspectSettings:
    """Inspect cell settings for one run."""

    run_id: str
    base_url: str
    temperature: float
    seed: int
    max_steps: int
    time_limit_s: float
    max_tokens: int
    log_dir: Path


def _executor(env: ToolEnv, name: str) -> Callable[..., Awaitable[str]]:
    async def execute(**kwargs: Any) -> str:
        return env.call(name, kwargs)

    return execute


def tool_defs(env: ToolEnv) -> list[Any]:
    """Inspect tools for every suite tool except ``submit``, bound to ``env``."""
    tools = []
    for spec in env.tool_specs():
        if spec.name == "submit":
            continue
        tools.append(
            ToolDef(
                tool=_executor(env, spec.name),
                name=spec.name,
                description=spec.description,
                parameters=_params(spec),
            ).as_tool()
        )
    return tools


def _params(spec: ToolSpec) -> ToolParams:
    schema = spec.parameters
    props = {name: ToolParam(**dict(value)) for name, value in schema.get("properties", {}).items()}
    return ToolParams(properties=props, required=list(schema.get("required", [])))


def _tool_results(messages: list[Any]) -> dict[str, Any]:
    return {
        str(m.tool_call_id): m
        for m in messages
        if getattr(m, "role", "") == "tool" and getattr(m, "tool_call_id", None)
    }


def submitted_answer(messages: list[Any]) -> tuple[str | None, bool]:
    """The answer of the last submit call Inspect accepted, and whether one was rejected.

    A submit call counts only if its tool result exists and carries no error (Inspect's
    argument validation rejects, for example, a number or an extra argument). The
    second value is True if the last submit call was rejected.
    """
    results = _tool_results(messages)
    answer: str | None = None
    rejected = False
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if getattr(call, "function", "") != "submit":
                continue
            result = results.get(str(getattr(call, "id", "")))
            if result is None or getattr(result, "error", None) is not None:
                answer, rejected = None, result is not None
                continue
            args = getattr(call, "arguments", {}) or {}
            answer = str(args.get("answer", "")) if isinstance(args, Mapping) else str(args)
            rejected = False
    return answer, rejected


def rejected_calls(messages: list[Any]) -> int:
    """Tool calls that Inspect rejected before they reached a tool (validation errors)."""
    return sum(
        1
        for m in messages
        if getattr(m, "role", "") == "tool" and getattr(m, "error", None) is not None
    )


def end_reason(sample: Any, answer: str | None) -> str:
    """How the Inspect sample ended, in the suite's end-reason vocabulary."""
    if getattr(sample, "error", None) is not None:
        text = str(getattr(sample.error, "message", sample.error)).lower()
        return "backend_error" if any(m in text for m in BACKEND_MARKERS) else "harness_error"
    limit = getattr(sample, "limit", None)
    if limit is not None:
        return "wall_clock" if getattr(limit, "type", "") in ("time", "working") else "step_limit"
    if getattr(getattr(sample, "output", None), "stop_reason", "") == "model_length":
        return "context_limit"
    return "submitted" if answer is not None else "stopped"


def run(task: Task, settings: InspectSettings) -> dict[str, Any]:
    """Run one Inspect basic-agent episode and return the run record fields."""
    os.environ.setdefault("NF_API_KEY", "local")
    env = task.start(max_calls=settings.max_steps)
    solver = basic_agent(
        tools=tool_defs(env),
        max_attempts=1,
        message_limit=2 + 2 * settings.max_steps,
        submit_description="Submit your final answer.",
    )
    itask = InspectTask(
        dataset=[Sample(input=task.instruction, id=task.id)],
        solver=solver,
        time_limit=int(settings.time_limit_s),
        name="nf_tools",
    )
    logs = inspect_eval(
        itask,
        model=f"openai-api/nf/{settings.run_id}",
        model_base_url=f"{settings.base_url}/v1",
        log_dir=str(settings.log_dir),
        log_format="json",
        display="none",
        temperature=settings.temperature,
        seed=settings.seed,
        max_tokens=settings.max_tokens,
        fail_on_error=False,
        max_retries=0,
    )
    sample = logs[0].samples[0] if logs and logs[0].samples else None
    if sample is None:
        return {
            "end_reason": "harness_error",
            "status": "infra_error",
            "outcome": 0.0,
            "error": str(getattr(logs[0], "error", "no sample")) if logs else "no log",
        }
    messages = list(sample.messages)
    answer, rejected = submitted_answer(messages)
    reason = end_reason(sample, answer)
    grade = task.grade(answer, env.files)
    usage = sample.model_usage or {}
    return {
        "end_reason": reason,
        "status": episode_status(reason),
        "outcome": 1.0 if grade.passed else 0.0,
        "outcome_strict": float(grade.score if grade.strict is None else grade.strict),
        "grade_reason": grade.reason,
        "answer": answer,
        "submit_rejected": rejected,
        "steps": sum(1 for m in messages if getattr(m, "role", "") == "assistant"),
        "tool_calls": sum(1 for entry in env.trace if entry.tool != "submit"),
        "rejected_calls": rejected_calls(messages),
        "input_tokens": sum(int(u.input_tokens or 0) for u in usage.values()),
        "output_tokens": sum(int(u.output_tokens or 0) for u in usage.values()),
        "wall_s": float(sample.total_time or 0.0),
        "error": str(getattr(sample.error, "message", "")) if sample.error else "",
    }


def main(argv: list[str] | None = None) -> int:
    """``python -m noisefloor.inspect_harness SPEC.json OUT.json`` (one run per process)."""
    args = sys.argv[1:] if argv is None else argv
    spec = json.loads(Path(args[0]).read_text())
    task = generate(spec["family"], int(spec["task_seed"]), int(spec["difficulty"]))
    settings = InspectSettings(
        run_id=spec["run_id"],
        base_url=spec["base_url"],
        temperature=float(spec["temperature"]),
        seed=int(spec["seed"]),
        max_steps=int(spec["max_steps"]),
        time_limit_s=float(spec["timeout_s"]),
        max_tokens=int(spec["max_tokens"]),
        log_dir=Path(spec["log_dir"]),
    )
    Path(args[1]).write_text(json.dumps(run(task, settings), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
