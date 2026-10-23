"""Write a mock SWE-bench experiments-style results.json with swebench 1.1.5's get_model_report.

Usage: python make_v1.py <work_dir>

Mock data in the real format: no model or test suite ran. The predictions, task file, and
evaluation logs are written by hand in the layout swebench 1.1.5 reads, and
swebench.metrics.report.get_model_report builds the report map that the SWE-bench
experiments repository stores as results/results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    INSTALL_FAIL,
    TESTS_TIMEOUT,
)
from swebench.metrics.report import get_model_report

MODEL = "mock-model-v1"
F2P = ["testing/test_mock.py::test_fixed"]
P2P = ["testing/test_mock.py::test_kept"]
IDS = [f"pytest-dev__pytest-9000{i}" for i in range(1, 9)]
APPLIED = f"{APPLY_PATCH_PASS} (test)\n{APPLY_PATCH_PASS} (pred)\n"

# instance -> (model_patch, eval log or None)
PLAN: dict[str, tuple[str | None, str | None]] = {
    IDS[0]: ("diff\n", APPLIED + f"PASSED {F2P[0]}\nPASSED {P2P[0]}\n"),  # resolved
    IDS[1]: ("diff\n", APPLIED + f"FAILED {F2P[0]}\nPASSED {P2P[0]}\n"),  # applied, unresolved
    IDS[2]: ("", None),  # no_generation
    IDS[3]: ("diff\n", APPLIED + f"{TESTS_TIMEOUT}\n"),  # test_timeout
    IDS[4]: ("diff\n", f"{INSTALL_FAIL}\n"),  # install_fail
    IDS[5]: ("diff\n", None),  # generated, no log file
    IDS[7]: ("diff\n", f"{APPLY_PATCH_FAIL}; (pred_try)\n"),  # no_apply
}
# IDS[6] has no prediction at all.


def main() -> None:
    work = Path(sys.argv[1])
    log_dir = work / "logs"
    log_dir.mkdir(parents=True, exist_ok=False)
    predictions = [
        {"instance_id": i, "model_name_or_path": MODEL, "model_patch": patch}
        for i, (patch, _) in PLAN.items()
    ]
    preds_path = work / "all_preds.jsonl"
    preds_path.write_text("".join(json.dumps(p) + "\n" for p in predictions))
    tasks = [{"instance_id": i, "FAIL_TO_PASS": F2P, "PASS_TO_PASS": P2P} for i in IDS]
    tasks_path = work / "tasks.json"
    tasks_path.write_text(json.dumps(tasks))
    for instance_id, (_, log) in PLAN.items():
        if log is not None:
            log_dir.joinpath(f"{instance_id}.{MODEL}.eval.log").write_text(log)
    report = get_model_report(MODEL, str(preds_path), str(tasks_path), str(log_dir))
    out = work / "results" / "results.json"
    out.parent.mkdir()
    out.write_text(json.dumps(report, indent=4))
    print(out)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
