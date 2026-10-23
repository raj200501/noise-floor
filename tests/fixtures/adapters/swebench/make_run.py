"""Write a mock SWE-bench evaluation log tree and run report with swebench 5.0.2's own code.

Usage: python make_run.py <work_dir>

Mock data in the real format: no model, container, or test suite ran. The test logs are
written by hand; swebench.harness.grading.get_eval_report grades them, report.json is written
the way swebench.harness.run_evaluation.run_instance writes it, and
swebench.harness.reporting.make_run_report writes the run report from the tree.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from swebench.harness.constants import (
    APPLY_PATCH_PASS,
    END_TEST_OUTPUT,
    LOG_INSTANCE,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
    START_TEST_OUTPUT,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.reporting import make_run_report
from swebench.types import TestSpec

RUN_ID = "mock-run"
MODEL = "mock/model-x"
REPO = "pytest-dev/pytest"
F2P = ["testing/test_mock.py::test_fixed"]
P2P = ["testing/test_mock.py::test_kept"]
IDS = [f"pytest-dev__pytest-9000{i}" for i in range(1, 9)]


def test_log(lines: list[str]) -> str:
    return "\n".join([f"{APPLY_PATCH_PASS} (pred)", START_TEST_OUTPUT, *lines, END_TEST_OUTPUT, ""])


PASS_ALL = test_log([f"PASSED {F2P[0]}", f"PASSED {P2P[0]}"])
FAIL_F2P = test_log([f"FAILED {F2P[0]} - AssertionError", f"PASSED {P2P[0]}"])
TIMEOUT = (
    f"{APPLY_PATCH_PASS} (pred)\n{START_TEST_OUTPUT}\n\n\nTimeout error: 1800 seconds exceeded."
)
DOCKER_DOWN = "Error response from daemon: mock container failure (fixture)\n"

# instance -> (model_patch, test output or None, write report.json)
PLAN: dict[str, tuple[str | None, str | None, bool]] = {
    IDS[0]: ("diff --git a/x b/x\n", PASS_ALL, True),  # resolved
    IDS[1]: ("diff --git a/x b/x\n", FAIL_F2P, True),  # unresolved
    IDS[2]: ("", None, False),  # empty patch: run_evaluation skips it
    IDS[3]: ("diff --git a/x b/x\n", TIMEOUT, False),  # EvaluationError on timeout
    IDS[4]: ("diff --git a/x b/x\n", DOCKER_DOWN, True),  # environment failure
    IDS[5]: ("diff --git a/x b/x\n", None, False),  # errored before tests ran
    IDS[7]: ("diff --git a/x b/x\n", PASS_ALL, True),  # resolved
}
# IDS[6] has no prediction at all (incomplete).


def main() -> None:
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=False)
    os.chdir(work)  # RUN_EVALUATION_LOG_DIR is relative to the working directory
    predictions = {}
    for instance_id, (patch, output, write_report) in PLAN.items():
        prediction = {"instance_id": instance_id, "model_name_or_path": MODEL, "model_patch": patch}
        predictions[instance_id] = prediction
        if patch == "":
            continue
        log_dir = RUN_EVALUATION_LOG_DIR / RUN_ID / MODEL.replace("/", "__") / instance_id
        log_dir.mkdir(parents=True)
        log_dir.joinpath("patch.diff").write_text(patch or "")
        log_dir.joinpath(LOG_INSTANCE).write_text(
            f"mock run_instance log for {instance_id} (fixture)\n"
        )
        if output is None:
            continue
        test_output = log_dir / LOG_TEST_OUTPUT
        test_output.write_text(output)
        if not write_report:
            continue
        spec = TestSpec(
            instance_id=instance_id,
            image="mock",
            eval_script_list=[],
            repo=REPO,
            version="mock",
            FAIL_TO_PASS=F2P,
            PASS_TO_PASS=P2P,
            log_parser="parse_log_pytest",
            eval_type="pass_and_fail",
        )
        report = get_eval_report(spec, prediction, str(test_output), include_tests_status=True)
        log_dir.joinpath(LOG_REPORT).write_text(json.dumps(report, indent=4))
    full_dataset = [{"instance_id": i} for i in IDS]
    path = make_run_report(predictions, full_dataset, RUN_ID, client=None, report_dir=".")
    Path("instances.txt").write_text("\n".join(IDS) + "\n")
    print(path)


if __name__ == "__main__":
    main()
