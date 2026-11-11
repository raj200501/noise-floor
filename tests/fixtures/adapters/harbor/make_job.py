"""Write a small mock Harbor job directory by serializing harbor's own models.

Usage: python make_job.py <jobs_dir>

Mock data in the real format: no agent, model, environment, or verifier ran. Every trial
result is constructed by hand with harbor.models.trial.result.TrialResult and written the
way harbor 0.24.0 writes it (trial/result.json and trial/config.json as in
harbor/trial/trial.py; job/result.json and job/config.json as in harbor/job.py).
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from harbor.models.agent.context import AgentContext
from harbor.models.job.config import JobConfig
from harbor.models.job.result import JobResult, JobStats
from harbor.models.task.id import LocalTaskId
from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig, VerifierConfig
from harbor.models.trial.result import (
    AgentInfo,
    ExceptionInfo,
    ModelInfo,
    StepResult,
    TimingInfo,
    TrialResult,
)
from harbor.models.verifier.result import VerifierResult

JOB_ID = UUID("00000000-0000-4000-8000-000000000001")
T0 = datetime(2026, 10, 23, 2, 0, 0, tzinfo=UTC)
JOB_NAME = "2026-10-23__02-00-00"
TASKS = ("mock-hello", "mock-regex", "mock-build")
MODELS = ("mock/model-a", "mock/model-b")

# (model, task, attempt) -> (reward or None, exception type or None, options)
PLAN: dict[tuple[str, str, int], tuple[float | None, str | None, dict[str, object]]] = {
    ("mock/model-a", "mock-hello", 0): (1.0, None, {}),
    ("mock/model-a", "mock-hello", 1): (1.0, None, {}),
    ("mock/model-a", "mock-regex", 0): (0.0, "AgentTimeoutError", {}),
    ("mock/model-a", "mock-regex", 1): (1.0, None, {}),
    ("mock/model-a", "mock-build", 0): (None, "EnvironmentStartTimeoutError", {}),
    ("mock/model-a", "mock-build", 1): (0.0, None, {"multi_step": True}),
    ("mock/model-b", "mock-hello", 0): (1.0, None, {}),
    ("mock/model-b", "mock-hello", 1): (0.0, "AgentSafetyRefusalError", {}),
    ("mock/model-b", "mock-regex", 0): (None, "VerifierTimeoutError", {}),
    ("mock/model-b", "mock-regex", 1): (None, None, {"verifier_disabled": True}),
    ("mock/model-b", "mock-build", 0): (0.0, "ContextWindowExceededError", {}),
    ("mock/model-b", "mock-build", 1): (None, None, {"incomplete": True}),
}


def exception(kind: str, at: datetime) -> ExceptionInfo:
    return ExceptionInfo(
        exception_type=kind,
        exception_message=f"mock {kind} (fixture)",
        exception_traceback=f"Traceback (most recent call last):\n{kind}: mock (fixture)\n",
        occurred_at=at,
    )


def main() -> None:
    job_dir = Path(sys.argv[1]) / JOB_NAME
    job_dir.mkdir(parents=True, exist_ok=False)
    results: list[TrialResult] = []
    for index, ((model, task, _attempt), (reward, exc, opts)) in enumerate(PLAN.items()):
        start = T0 + timedelta(minutes=5 * index)
        trial_name = f"{task}__mock{index:03d}"
        trial_dir = job_dir / trial_name
        (trial_dir / "agent").mkdir(parents=True)
        (trial_dir / "verifier").mkdir()
        config = TrialConfig(
            task=TaskConfig(path=Path("tasks") / task, source="mock-dataset"),
            trial_name=trial_name,
            trials_dir=job_dir,
            agent=AgentConfig(name="terminus-2", model_name=model),
            verifier=VerifierConfig(disable=bool(opts.get("verifier_disabled"))),
            job_id=JOB_ID,
        )
        trial_dir.joinpath("config.json").write_text(
            config.model_dump_json(indent=4, exclude_defaults=True)
        )
        if opts.get("incomplete"):
            # A trial whose process died before trial.py wrote result.json.
            trial_dir.joinpath("trial.log").write_text("mock trial log (fixture)\n")
            continue
        usage = AgentContext(
            n_input_tokens=1000 + 10 * index,
            n_cache_tokens=100,
            n_output_tokens=200 + index,
            cost_usd=None,
        )
        provider, name = model.split("/")
        multi_step = bool(opts.get("multi_step"))
        result = TrialResult(
            id=UUID(int=0x1000 + index),
            task_name=task,
            trial_name=trial_name,
            trial_uri=f"file:///jobs/{JOB_NAME}/{trial_name}",
            task_id=LocalTaskId(path=Path("tasks") / task),
            source="mock-dataset",
            task_checksum=f"mock-checksum-{task}",
            config=config,
            agent_info=AgentInfo(
                name="terminus-2",
                version="mock",
                model_info=ModelInfo(name=name, provider=provider),
            ),
            agent_result=None if multi_step else usage,
            verifier_result=None if reward is None else VerifierResult(rewards={"reward": reward}),
            exception_info=None if exc is None else exception(exc, start + timedelta(minutes=3)),
            started_at=start,
            finished_at=start + timedelta(minutes=4),
            environment_setup=TimingInfo(
                started_at=start, finished_at=start + timedelta(seconds=30)
            ),
            # A multi-step trial times the agent per step: harbor/trial/trial.py
            # _run_agent_phase records timing on its target, which multi_step.py sets to
            # each StepResult, so the trial's own agent_execution stays None.
            agent_execution=None
            if multi_step
            else TimingInfo(
                started_at=start + timedelta(minutes=1), finished_at=start + timedelta(minutes=3)
            ),
        )
        if multi_step:
            result.step_results = [
                StepResult(
                    step_name="step-1",
                    agent_result=usage,
                    verifier_result=VerifierResult(rewards={"reward": 1.0}),
                    agent_execution=TimingInfo(
                        started_at=start + timedelta(minutes=1),
                        finished_at=start + timedelta(minutes=2),
                    ),
                    verifier=TimingInfo(
                        started_at=start + timedelta(minutes=2),
                        finished_at=start + timedelta(minutes=2, seconds=15),
                    ),
                ),
                StepResult(
                    step_name="step-2",
                    agent_result=usage,
                    verifier_result=VerifierResult(rewards={"reward": 0.0}),
                    agent_execution=TimingInfo(
                        started_at=start + timedelta(minutes=2, seconds=15),
                        finished_at=start + timedelta(minutes=3),
                    ),
                    verifier=TimingInfo(
                        started_at=start + timedelta(minutes=3),
                        finished_at=start + timedelta(minutes=3, seconds=30),
                    ),
                ),
            ]
        if index == 0:
            # Agents may write their own result.json inside agent/ (for example langgraph).
            trial_dir.joinpath("agent", "result.json").write_text(
                '{"note": "agent log (fixture)"}\n'
            )
        trial_dir.joinpath("result.json").write_text(result.model_dump_json(indent=4))
        results.append(result)

    job_config = JobConfig(
        job_name=JOB_NAME,
        jobs_dir=Path(sys.argv[1]),
        n_attempts=2,
        agents=[AgentConfig(name="terminus-2", model_name=m) for m in MODELS],
        tasks=[TaskConfig(path=Path("tasks") / t, source="mock-dataset") for t in TASKS],
    )
    job_dir.joinpath("config.json").write_text(
        job_config.model_dump_json(indent=4, exclude_defaults=True)
    )
    job_result = JobResult(
        id=JOB_ID,
        started_at=T0,
        finished_at=T0 + timedelta(hours=1),
        n_total_trials=len(PLAN),
        stats=JobStats.from_trial_results(results, n_total_trials=len(PLAN)),
    )
    job_dir.joinpath("result.json").write_text(
        job_result.model_dump_json(indent=4, exclude={"trial_results"})
    )
    print(job_dir, len(results), "trial results")


if __name__ == "__main__":
    main()
