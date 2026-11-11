# SWE-bench result fixtures (mock data in the real format)

These are **mock** SWE-bench results: no model, container, or test suite ran (SWE-bench needs
Docker, which this environment does not have). The test logs were written by hand, and the
`swebench` package's own grading and reporting code turned them into the result files. The
instance ids `pytest-dev__pytest-90001` to `90008` are invented (the `pytest-dev__pytest`
prefix only selects swebench's pytest log parser); nothing here is a SWE-bench result.

Contract tests: `tests/contract/test_swebench.py`.

## How they were produced (2026-10-23)

`pip download swebench==<version> --no-deps` fetched the wheels from PyPI (the package named
"The official SWE-bench package"); each was installed with `uv pip install` into its own
scratch virtual environment outside the repository.

### `run-5.0.2/` (swebench 5.0.2)

`python make_run.py <work_dir>` (the script in this directory). It writes
`logs/run_evaluation/mock-run/mock__model-x/<instance_id>/` directories with hand-written
`test_output.txt`, `run_instance.log`, and `patch.diff`; grades each finished instance with
`swebench.harness.grading.get_eval_report` and writes `report.json` as
`swebench.harness.run_evaluation.run_instance` does (`json.dumps(report, indent=4)`); then
calls `swebench.harness.reporting.make_run_report`, which writes the run report
`mock__model-x.mock-run.json`. `instances.txt` is the full instance list. Re-running the
script gives byte-identical files.

| Instance | Designed case | Files |
| --- | --- | --- |
| 90001, 90008 | resolved | report.json |
| 90002 | unresolved (FAIL_TO_PASS test fails) | report.json |
| 90003 | empty patch (run_evaluation skips it) | none |
| 90004 | tests timed out (`Timeout error: 1800 seconds exceeded.`) | test_output.txt, no report.json |
| 90005 | environment failure (`Error response from daemon`), `infra_failure: true` | report.json |
| 90006 | errored before tests ran | run_instance.log only |
| 90007 | no prediction | none |

### `experiments-1.1.5/mock-submission-v1/results/results.json` (swebench 1.1.5)

`python make_v1.py <work_dir>` (the script in this directory) writes hand-made predictions,
a task file, and `<instance_id>.<model>.eval.log` files in the swebench 1.x layout, and calls
`swebench.metrics.report.get_model_report`, whose returned report map is the
experiments-style `results.json` (lists `no_generation`, `generated`, `with_logs`,
`install_fail`, `reset_failed`, `no_apply`, `applied`, `test_errored`, `test_timeout`,
`resolved`). Re-running the script gives a byte-identical file. Note that 90004 appears in
both `test_timeout` and `applied`: swebench 1.1.5 does that, because the `continue` after a
timeout only leaves the inner loop.

## Not covered by a recorded fixture

Later experiments-repository `results.json` files with lists such as `no_logs` and `error`
could not be checked against a published package (none on PyPI writes them, and GitHub is not
reachable from here), so the tests build such files inline instead of claiming a recorded
format. The swebench 2.1.8 run report (checked in its source) has the same keys as 5.0.2
without `infra_failure_ids`, `ambiguous_failure_ids`, and `failure_reasons`, and its
per-instance `report.json` lacks `infra_failure`; the tests cover that by removing those keys.
