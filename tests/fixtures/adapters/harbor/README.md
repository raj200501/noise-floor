# Harbor job fixture (mock data in the real format)

`jobs/2026-10-23__02-00-00/` is a **mock** Harbor job directory: no agent, model, container,
or verifier ran. Harbor needs Docker, which this environment does not have, so the files were
built by serializing Harbor's own pydantic models with hand-chosen values. The model names
(`mock/model-a`, `mock/model-b`), task names (`mock-hello`, `mock-regex`, `mock-build`), and
dataset (`mock-dataset`) are invented; nothing here is a Terminal-Bench result.

Contract tests: `tests/contract/test_harbor.py`.

## How it was produced (2026-10-23)

1. `pip download harbor --no-deps` fetched `harbor-0.24.0-py3-none-any.whl` from PyPI (the
   package by Alex Shaw that describes itself as the harness for Terminal-Bench 2.0). The
   formats were read from its source: `harbor/models/trial/result.py` (TrialResult),
   `harbor/models/job/result.py` (JobResult), `harbor/models/trial/config.py`
   (TrialConfig), `harbor/models/trial/paths.py` (trial directory layout),
   `harbor/trial/trial.py` and `harbor/job.py` (how the files are written), and
   `harbor/trial/errors.py` plus `harbor/agents/installed/base.py` (exception types).
2. In a scratch virtual environment outside the repository:
   `uv pip install harbor-0.24.0-py3-none-any.whl`, then
   `python make_job.py <jobs_dir>` (the script in this directory). It writes each file the way
   Harbor 0.24.0 does: `TrialResult.model_dump_json(indent=4)` to `<trial>/result.json`,
   `TrialConfig.model_dump_json(indent=4, exclude_defaults=True)` to `<trial>/config.json`,
   `JobResult.model_dump_json(indent=4, exclude={"trial_results"})` to `result.json`, and
   `JobConfig.model_dump_json(indent=4, exclude_defaults=True)` to `config.json`. Re-running
   the script gives byte-identical files.

## What the 12 trials cover

Two models, three tasks, two attempts each (`n_attempts=2`):

| Trial | Model | Designed case |
| --- | --- | --- |
| `mock-hello__mock000`, `mock001` | a | reward 1; `mock000` also has an agent-written `agent/result.json` that the adapter must ignore |
| `mock-regex__mock002` | a | `AgentTimeoutError`, verifier still ran, reward 0 |
| `mock-regex__mock003` | a | reward 1 |
| `mock-build__mock004` | a | `EnvironmentStartTimeoutError`, no reward |
| `mock-build__mock005` | a | multi-step trial (`step_results`, tokens per step), reward 0 |
| `mock-hello__mock006` | b | reward 1 |
| `mock-hello__mock007` | b | `AgentSafetyRefusalError`, reward 0 |
| `mock-regex__mock008` | b | `VerifierTimeoutError`, no reward |
| `mock-regex__mock009` | b | `verifier.disable = true`, no reward |
| `mock-build__mock010` | b | `ContextWindowExceededError`, reward 0 |
| `mock-build__mock011` | b | `config.json` and `trial.log` only: the trial never wrote `result.json` |
