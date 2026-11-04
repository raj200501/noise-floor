"""Contract tests for the SWE-bench results reader (evalstats.adapters.swebench).

The recorded fixtures under tests/fixtures/adapters/swebench/ are MOCK data in the real
SWE-bench file formats (see the README.md beside them): no model, container, or test suite
ran, the instance ids are invented, and nothing in them is a model result or a SWE-bench
result. run-5.0.2/ holds a log tree and the run report that swebench 5.0.2's own grading and
reporting code wrote from it; experiments-1.1.5/ holds the results.json that swebench 1.1.5's
get_model_report wrote. Expected values come from the raw JSON and log files (read here with
json and pathlib), from swebench's own counts in the run report, and from the designed cases
in the fixture README under the status mapping of the module docstring; none is copied from
the adapter's output. Malformed and edge-case inputs are built in tmp_path.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from evalstats import EvalResults, SchemaError
from evalstats.adapters import STATUSES, read_swebench
from evalstats.adapters.swebench import RESULTS_CATEGORIES, RUN_REPORT_LISTS, TIMEOUT_REASON

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "adapters" / "swebench"
RUN = FIXTURE / "run-5.0.2"
RUN_ID = "mock-run"  # make_run.py's RUN_ID
# make_run.py's model "mock/model-x", with "/" replaced by "__" as swebench does.
MODEL_DIR = "mock__model-x"
TREE = RUN / "logs" / "run_evaluation" / RUN_ID / MODEL_DIR
RUN_REPORT = RUN / f"{MODEL_DIR}.{RUN_ID}.json"  # make_run_report names it <model>.<run_id>.json
INSTANCE_LIST = RUN / "instances.txt"
SUBMISSION = FIXTURE / "experiments-1.1.5" / "mock-submission-v1"
RESULTS = SUBMISSION / "results" / "results.json"
TIMEOUT_LINE = "Timeout error: 1800 seconds exceeded."  # what run_instance appends (README)


def load(path: Path) -> Any:
    return json.loads(path.read_text())


REPORT = load(RUN_REPORT)
RESULTS_RAW = load(RESULTS)
IDS = tuple(INSTANCE_LIST.read_text().split())
INSTANCE_DIRS = sorted(p.name for p in TREE.iterdir() if p.is_dir())
INSTANCE_REPORTS = {
    name: load(TREE / name / "report.json")[name]
    for name in INSTANCE_DIRS
    if (TREE / name / "report.json").is_file()
}
COVARIATES = ["status", "status_reason", "failure_reason", "source_format"]  # docstring order
COLUMNS = ["model", "task", "run", "outcome", *COVARIATES]


def iid(n: int) -> str:
    return f"pytest-dev__pytest-9000{n}"


# Designed case of every instance (fixture README table, make_run.py, and make_v1.py), as the
# (status, status_reason) that the module docstring's mapping gives in each reading.
RUN_REPORT_CASES: dict[str, tuple[str, str]] = {
    iid(1): ("ok", "resolved_ids"),
    iid(2): ("ok", "unresolved_ids"),
    iid(3): ("ok", "empty_patch_ids"),  # SWE-bench scores an empty patch as unresolved
    iid(4): ("timeout", "tests_timed_out"),  # error_ids, classified tests_timed_out
    iid(5): ("infra_error", "infra_failure_ids"),  # also in unresolved_ids
    iid(6): ("infra_error", "error_ids"),
    iid(7): ("infra_error", "incomplete_ids"),  # no prediction
    iid(8): ("ok", "resolved_ids"),
}
TREE_CASES: dict[str, tuple[str, str]] = {
    iid(1): ("ok", "resolved"),
    iid(2): ("ok", "unresolved"),
    iid(3): ("infra_error", "missing"),  # swebench skips empty patches: no directory
    iid(4): ("timeout", "tests_timed_out"),
    iid(5): ("infra_error", "infra_failure"),
    iid(6): ("infra_error", "no_report"),
    iid(7): ("infra_error", "missing"),
    iid(8): ("ok", "resolved"),
}
RESULTS_CASES: dict[str, tuple[str, str]] = {
    iid(1): ("ok", "resolved"),
    iid(2): ("ok", "applied"),  # in generated, with_logs, and applied: the narrowest names it
    iid(3): ("ok", "no_generation"),
    iid(4): ("timeout", "test_timeout"),  # also in applied (README note on swebench 1.1.5)
    iid(5): ("infra_error", "install_fail"),
    iid(6): ("infra_error", "no_logs"),  # generated, but no evaluation log
    iid(7): ("infra_error", "missing"),  # no prediction: in no list of a file with generated
    iid(8): ("ok", "no_apply"),
}


# Expected rows, derived from the raw files and the designed cases.


def row(
    task: str,
    status: str,
    reason: str,
    *,
    model: str,
    source_format: str,
    outcome: float = 0.0,
    failure: str | None = None,
    run: str = "0",
) -> dict[str, Any]:
    return {
        "model": model,
        "task": task,
        "run": run,
        "outcome": outcome,
        "status": status,
        "status_reason": reason,
        "failure_reason": failure,
        "source_format": source_format,
    }


def expected_run_report(model: str = MODEL_DIR) -> dict[str, dict[str, Any]]:
    return {
        task: row(
            task,
            status,
            reason,
            model=model,
            source_format="run_report",
            outcome=float(task in REPORT["resolved_ids"]),
            failure=REPORT["failure_reasons"].get(task),
        )
        for task, (status, reason) in RUN_REPORT_CASES.items()
    }


def expected_tree(model: str = MODEL_DIR) -> dict[str, dict[str, Any]]:
    """Outcomes and failure reasons from each report.json, else from swebench's run report.

    swebench classified the logs of the instances without a report.json itself when it wrote
    the run report from this tree, so its failure_reasons are the reference for those.
    """
    expected = {}
    for task, (status, reason) in TREE_CASES.items():
        report = INSTANCE_REPORTS.get(task)
        if report is None:
            outcome, failure = 0.0, REPORT["failure_reasons"].get(task)
        else:
            outcome, failure = float(report["resolved"]), report.get("infra_failure_reason")
        expected[task] = row(
            task,
            status,
            reason,
            model=model,
            source_format="instance_reports",
            outcome=outcome,
            failure=failure,
        )
    return expected


def expected_results(model: str = SUBMISSION.name) -> dict[str, dict[str, Any]]:
    return {
        task: row(
            task,
            status,
            reason,
            model=model,
            source_format="results",
            outcome=float(task in RESULTS_RAW["resolved"]),
        )
        for task, (status, reason) in RESULTS_CASES.items()
    }


# Adapter output as plain values.


def clean(value: Any) -> Any:
    return None if pd.isna(value) else value


def rows_of(res: EvalResults) -> dict[str, dict[str, Any]]:
    records = res.frame.to_dict("records")
    table = {str(r["task"]): {str(k): clean(v) for k, v in r.items()} for r in records}
    assert len(table) == len(records), "instance ids repeat"
    return table


def read_rows(path: Path | str, instances: Any = INSTANCE_LIST, **options: Any) -> dict[str, Any]:
    return rows_of(read_swebench(path, instances, **options))


# Inputs built in tmp_path from the recorded files.


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4))
    return path


def edited_report(drop: tuple[str, ...] = (), **changes: Any) -> dict[str, Any]:
    data = copy.deepcopy(REPORT)
    for key in drop:
        del data[key]
    data.update(changes)
    return data


def copy_tree(tmp_path: Path) -> Path:
    return Path(shutil.copytree(TREE, tmp_path / RUN_ID / MODEL_DIR))


def write_results(tmp_path: Path, data: Mapping[str, Any]) -> Path:
    return write_json(tmp_path / SUBMISSION.name / "results" / "results.json", data)


@pytest.fixture(scope="module")
def report_rows() -> dict[str, dict[str, Any]]:
    return read_rows(RUN_REPORT)


@pytest.fixture(scope="module")
def tree_rows() -> dict[str, dict[str, Any]]:
    return read_rows(TREE)


@pytest.fixture(scope="module")
def results_rows() -> dict[str, dict[str, Any]]:
    return read_rows(RESULTS)


# The recorded fixtures.


def test_fixture_layout_matches_the_readme() -> None:
    assert tuple(iid(n) for n in range(1, 9)) == IDS
    assert set(IDS) == RUN_REPORT_CASES.keys() == TREE_CASES.keys() == RESULTS_CASES.keys()
    assert RUN_REPORT.is_file() and TREE.is_dir() and RESULTS.is_file()
    # One directory per evaluated prediction: not the empty patch, not the missing one.
    assert sorted(set(IDS) - {iid(3), iid(7)}) == INSTANCE_DIRS
    assert sorted(INSTANCE_REPORTS) == [iid(1), iid(2), iid(5), iid(8)]


def test_designed_cases_match_the_run_report() -> None:
    assert len(IDS) == REPORT["total_instances"]
    assert set(IDS) == set(REPORT["submitted_ids"]) | set(REPORT["incomplete_ids"])
    infra = (
        set(REPORT["error_ids"]) | set(REPORT["incomplete_ids"]) | set(REPORT["infra_failure_ids"])
    )
    for task, (status, reason) in RUN_REPORT_CASES.items():
        if reason == "tests_timed_out":
            assert REPORT["failure_reasons"][task] == reason
            assert task in REPORT["error_ids"] and task in REPORT["ambiguous_failure_ids"]
        else:
            assert task in REPORT[reason]
        assert (status == "ok") == (task not in infra)
    assert REPORT["failure_reasons"] == {iid(4): "tests_timed_out", iid(5): "container_unavailable"}


def test_designed_cases_match_the_log_tree() -> None:
    for task, (_, reason) in TREE_CASES.items():
        directory = TREE / task
        report = INSTANCE_REPORTS.get(task)
        if reason == "missing":
            assert not directory.exists()
            continue
        assert (directory / "run_instance.log").is_file() and (directory / "patch.diff").is_file()
        output = directory / "test_output.txt"
        if reason == "tests_timed_out":
            assert report is None and output.read_text().endswith(TIMEOUT_LINE)
        elif reason == "no_report":
            assert report is None and not output.exists()
        elif reason == "infra_failure":
            assert report is not None and report["infra_failure"] is True
            assert "Error response from daemon" in output.read_text()
        else:
            assert report is not None and report["infra_failure"] is False
            assert report["resolved"] is (reason == "resolved")
            assert report["patch_successfully_applied"] is True


def test_designed_cases_match_the_results_json() -> None:
    lists = {key: set(value) for key, value in RESULTS_RAW.items()}
    for task, (_, reason) in RESULTS_CASES.items():
        if reason == "no_logs":
            assert task in lists["generated"] - lists["with_logs"]
        elif reason == "missing":
            assert not any(task in members for members in lists.values())
        else:
            assert task in lists[reason]
    # swebench 1.1.5 lists the timed-out instance in applied too, and every prediction in
    # generated or no_generation.
    assert iid(4) in lists["applied"] & lists["test_timeout"]
    assert lists["generated"] | lists["no_generation"] == set(IDS) - {iid(7)}


def test_run_report_rows(report_rows: dict[str, dict[str, Any]]) -> None:
    assert report_rows == expected_run_report()


def test_log_tree_rows(tree_rows: dict[str, dict[str, Any]]) -> None:
    assert tree_rows == expected_tree()


def test_results_json_rows(results_rows: dict[str, dict[str, Any]]) -> None:
    assert results_rows == expected_results()


@pytest.mark.parametrize("path", [RUN_REPORT, TREE, RESULTS], ids=["report", "tree", "results"])
def test_table_shape_and_row_order(path: Path) -> None:
    res = read_swebench(path, INSTANCE_LIST)
    assert list(res.frame.columns) == COLUMNS
    assert res.covariates == COVARIATES
    assert list(res.frame["task"]) == list(IDS)  # one row per listed instance, in list order
    assert set(res.frame["run"]) == {"0"}
    assert len(res.models) == 1
    assert res.is_binary


@pytest.mark.parametrize(
    ("path", "cases"),
    [(RUN_REPORT, RUN_REPORT_CASES), (TREE, TREE_CASES), (RESULTS, RESULTS_CASES)],
    ids=["report", "tree", "results"],
)
def test_every_status_value_occurs(path: Path, cases: dict[str, tuple[str, str]]) -> None:
    res = read_swebench(path, INSTANCE_LIST)
    assert Counter(res.frame["status"]) == Counter(status for status, _ in cases.values())
    # SWE-bench marks no run as excluded: flagged comes only from categories.
    assert set(res.frame["status"]) == set(STATUSES) - {"flagged"}


def test_run_report_rows_match_swebench_counts(report_rows: dict[str, dict[str, Any]]) -> None:
    assert len(report_rows) == REPORT["total_instances"]
    resolved = {task for task, r in report_rows.items() if r["outcome"] == 1.0}
    assert resolved == set(REPORT["resolved_ids"])
    assert len(resolved) == REPORT["resolved_instances"]
    flagged = {task for task, r in report_rows.items() if r["failure_reason"] is not None}
    assert len(flagged) == REPORT["infra_failure_instances"] + REPORT["ambiguous_failure_instances"]
    not_ok = {task for task, r in report_rows.items() if r["status"] != "ok"}
    assert not_ok == (
        set(REPORT["error_ids"]) | set(REPORT["incomplete_ids"]) | set(REPORT["infra_failure_ids"])
    )
    scored = {task for task, r in report_rows.items() if r["status"] == "ok"}
    assert len(scored) == (
        REPORT["resolved_instances"]
        + REPORT["unresolved_instances"]
        + REPORT["empty_patch_instances"]
        - REPORT["infra_failure_instances"]
    )


REPORT_REASONS = {"resolved", "unresolved", "empty_patch", "patch_not_applied", "infra_failure"}


def test_log_tree_agrees_with_the_run_report_swebench_wrote_from_it(
    tree_rows: dict[str, dict[str, Any]],
) -> None:
    for task in IDS:
        tree = tree_rows[task]
        assert tree["outcome"] == float(task in REPORT["resolved_ids"])
        assert tree["failure_reason"] == REPORT["failure_reasons"].get(task)
        if task not in REPORT["empty_patch_ids"]:
            # The tree cannot tell an empty patch from a missing prediction; all else agrees.
            assert tree["status"] == RUN_REPORT_CASES[task][0]
    # The rows decided by a report.json are the instances swebench counted as completed.
    decided = {task for task, r in tree_rows.items() if r["status_reason"] in REPORT_REASONS}
    assert decided == set(REPORT["completed_ids"])


def test_results_json_rows_match_its_lists(results_rows: dict[str, dict[str, Any]]) -> None:
    assert sum(r["outcome"] for r in results_rows.values()) == len(RESULTS_RAW["resolved"])
    for key in ("install_fail", "reset_failed", "test_errored", "test_timeout", "no_apply"):
        for task in RESULTS_RAW[key]:
            assert results_rows[task]["status_reason"] == key


def test_documented_tables() -> None:
    assert dict(RUN_REPORT_LISTS) == {
        "resolved_ids": "ok",
        "unresolved_ids": "ok",
        "empty_patch_ids": "ok",
        "error_ids": "infra_error",
        "incomplete_ids": "infra_error",
        "infra_failure_ids": "infra_error",
    }
    assert dict(RESULTS_CATEGORIES) == {
        **dict.fromkeys(
            ("resolved", "unresolved", "applied", "generated", "with_logs", "no_generation"), "ok"
        ),
        "no_apply": "ok",
        "test_timeout": "timeout",
        **dict.fromkeys(
            ("test_errored", "install_fail", "reset_failed", "no_logs", "error"), "infra_error"
        ),
    }
    assert TIMEOUT_REASON == "tests_timed_out"
    # Every list that swebench writes is mapped, so the recorded files read without options.
    assert set(RESULTS_RAW) <= set(RESULTS_CATEGORIES)
    read = {*RUN_REPORT_LISTS, "failure_reasons"}
    informational = {"completed_ids", "submitted_ids", "ambiguous_failure_ids"}
    assert {k for k, v in REPORT.items() if not isinstance(v, int)} == read | informational


# Single files and directories.


def test_merged_per_instance_report_file(tmp_path: Path) -> None:
    merged = dict(INSTANCE_REPORTS)
    path = write_json(tmp_path / "all_reports.json", merged)
    expected = expected_tree(model="all_reports")  # a merged file is named by its stem
    for task, values in expected.items():
        if task not in merged:  # only report.json contents are merged: the rest are missing
            values.update(status="infra_error", status_reason="missing", failure_reason=None)
    assert read_rows(path) == expected


@pytest.mark.parametrize("task", sorted(INSTANCE_REPORTS))
def test_one_report_json_file(task: str) -> None:
    rows = read_rows(TREE / task / "report.json", [task])
    assert rows == {task: expected_tree(model="report")[task]}  # named by the file stem


def test_experiments_logs_directory(tmp_path: Path) -> None:
    logs = Path(shutil.copytree(TREE, tmp_path / "20261023_mock-agent" / "logs"))
    assert read_rows(logs) == expected_tree(model="20261023_mock-agent")


def test_log_tree_copy_reads_like_the_recorded_one(tmp_path: Path) -> None:
    tree = copy_tree(tmp_path)
    (tree / "notes").mkdir()  # a subdirectory without instance files is not an instance
    (tree / "notes" / "summary.md").write_text("mock notes\n")
    (tree / "README.md").write_text("a stray file beside the instance directories\n")
    assert read_rows(tree) == expected_tree()


@pytest.mark.parametrize(
    "path",
    [TREE.parent, TREE.parent.parent, SUBMISSION, SUBMISSION / "results"],
    ids=["run-id-dir", "run-evaluation-dir", "submission-dir", "results-dir"],
)
def test_directory_without_instance_directories(path: Path) -> None:
    message = f"{path}: no instance directories found; pass the directory that holds one"
    with pytest.raises(SchemaError, match=re.escape(message)):
        read_swebench(path, INSTANCE_LIST)


def test_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(SchemaError, match=re.escape(f"{tmp_path}: no instance directories")):
        read_swebench(tmp_path, INSTANCE_LIST)


def test_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(SchemaError, match=re.escape(f"{missing}: no such file or directory")):
        read_swebench(missing, INSTANCE_LIST)


# The instance list.


def instance_list_forms() -> Iterator[Any]:
    """The documented instance list files, with the ids of instances.txt."""
    objects = [{"instance_id": i, "repo": "pytest-dev/pytest"} for i in IDS]
    yield pytest.param("ids.json", json.dumps(list(IDS)), id="json-ids")
    yield pytest.param("objects.json", json.dumps(objects), id="json-objects")
    yield pytest.param("ids.jsonl", "".join(f"{json.dumps(i)}\n" for i in IDS), id="jsonl-ids")
    lines = "".join(f"{json.dumps(o)}\n" for o in objects) + "\n"
    yield pytest.param("objects.jsonl", lines, id="jsonl-objects")
    padded = "\r\n".join(f"  {i}\t" for i in IDS) + "\r\n\r\n"
    yield pytest.param("padded.txt", padded, id="text-crlf-padded")
    yield pytest.param("instances", "\n\n".join(IDS), id="text-no-suffix")


@pytest.mark.parametrize(("name", "text"), list(instance_list_forms()))
def test_instance_list_files(tmp_path: Path, name: str, text: str) -> None:
    path = tmp_path / name
    path.write_text(text)
    expected = expected_run_report()
    assert read_rows(RUN_REPORT, path) == expected
    assert read_rows(RUN_REPORT, str(path)) == expected
    assert list(read_swebench(RUN_REPORT, path).frame["task"]) == list(IDS)


@pytest.mark.parametrize(
    "instances",
    [
        pytest.param(INSTANCE_LIST, id="path"),
        pytest.param(str(INSTANCE_LIST), id="str-path"),
        pytest.param(list(IDS), id="list"),
        pytest.param(IDS, id="tuple"),
        pytest.param([{"instance_id": i} for i in IDS], id="mappings"),
    ],
)
def test_instance_list_kinds(instances: Any) -> None:
    assert read_rows(RUN_REPORT, instances) == expected_run_report()


def test_instance_list_from_a_generator() -> None:
    assert read_rows(RUN_REPORT, (i for i in IDS)) == expected_run_report()


def test_rows_follow_the_instance_list_and_repeats_are_dropped() -> None:
    order = [*reversed(IDS), *IDS, IDS[0]]
    res = read_swebench(RUN_REPORT, order)
    assert list(res.frame["task"]) == list(reversed(IDS))
    assert rows_of(res) == expected_run_report()


@pytest.mark.parametrize("path", [RUN_REPORT, TREE, RESULTS], ids=["report", "tree", "results"])
def test_listed_instances_without_results(path: Path) -> None:
    # The recorded results.json has a generated list, so it also treats these as missing.
    extra = "pytest-dev__pytest-99999"
    res = read_swebench(path, [*IDS, extra])
    assert list(res.frame["task"]) == [*IDS, extra]
    got = rows_of(res)[extra]
    assert (got["status"], got["status_reason"], got["outcome"]) == ("infra_error", "missing", 0.0)
    assert got["failure_reason"] is None
    assert rows_of(res.subset(tasks=list(IDS))) == read_rows(path)


def result_ids(path: Path) -> set[str]:
    """Ids that a file or tree reports on, by the lists the module docstring reads."""
    if path == RUN_REPORT:
        return {i for key in RUN_REPORT_LISTS for i in REPORT[key]} | set(REPORT["failure_reasons"])
    if path == TREE:
        return set(INSTANCE_DIRS)
    return {i for values in RESULTS_RAW.values() for i in values}


@pytest.mark.parametrize("path", [RUN_REPORT, TREE, RESULTS], ids=["report", "tree", "results"])
def test_result_ids_outside_the_instance_list(path: Path) -> None:
    listed = list(IDS[:3])
    unknown = sorted(result_ids(path) - set(listed))
    message = (
        f"{path}: {len(unknown)} result ids are not in the instance list (first: {unknown[0]}); "
        "check that the list matches the split, or pass on_unknown='drop'"
    )
    with pytest.raises(SchemaError, match=f"^{re.escape(message)}$"):
        read_swebench(path, listed)
    kept = read_rows(path, listed, on_unknown="drop")
    expected = {
        RUN_REPORT: expected_run_report,
        TREE: expected_tree,
        RESULTS: expected_results,
    }[path]()
    assert kept == {task: expected[task] for task in listed}


def test_unknown_on_unknown_value() -> None:
    option: Any = "ignore"
    with pytest.raises(ValueError, match="on_unknown must be 'error' or 'drop', not 'ignore'"):
        read_swebench(RUN_REPORT, INSTANCE_LIST, on_unknown=option)


def test_missing_instance_list(tmp_path: Path) -> None:
    missing = tmp_path / "instances.txt"
    with pytest.raises(SchemaError, match=re.escape(f"{missing}: instance list not found")):
        read_swebench(RUN_REPORT, missing)
    with pytest.raises(SchemaError, match=re.escape(f"{tmp_path}: instance list not found")):
        read_swebench(RUN_REPORT, tmp_path)


def test_a_str_instance_list_is_always_a_path() -> None:
    with pytest.raises(SchemaError, match=re.escape(f"{IDS[0]}: instance list not found")):
        read_swebench(RUN_REPORT, IDS[0])


@pytest.mark.parametrize(
    ("name", "text"),
    [("empty.txt", ""), ("blank.txt", " \n\t\n"), ("empty.json", "[]"), ("empty.jsonl", "\n")],
)
def test_empty_instance_list_files(tmp_path: Path, name: str, text: str) -> None:
    path = tmp_path / name
    path.write_text(text)
    with pytest.raises(SchemaError, match=f"^{re.escape(f'{path}: the instance list is empty')}$"):
        read_swebench(RUN_REPORT, path)


@pytest.mark.parametrize("instances", [[], (), iter([])], ids=["list", "tuple", "iterator"])
def test_empty_instance_list(instances: Any) -> None:
    with pytest.raises(SchemaError, match=r"^instances: the instance list is empty$"):
        read_swebench(RUN_REPORT, instances)


@pytest.mark.parametrize(
    ("name", "text", "message"),
    [
        ("ids.json", "[", "invalid JSON"),
        ("ids.json", '{"instance_id": "x"}', "expected a JSON list of instances"),
        ("ids.json", '"pytest-dev__pytest-90001"', "expected a JSON list of instances"),
        ("ids.json", "[1, 2]", "an instance is not an id string or has no instance_id"),
        ("ids.json", '[""]', "an instance is not an id string or has no instance_id"),
        ("ids.json", '[{"id": "x"}]', "an instance is not an id string or has no instance_id"),
        (
            "ids.json",
            '[{"instance_id": 7}]',
            "an instance is not an id string or has no instance_id",
        ),
        ("ids.jsonl", "[1, 2]\n", "an instance is not an id string or has no instance_id"),
        ("ids.jsonl", "null\n", "an instance is not an id string or has no instance_id"),
    ],
)
def test_malformed_instance_list_files(tmp_path: Path, name: str, text: str, message: str) -> None:
    path = tmp_path / name
    path.write_text(text)
    with pytest.raises(SchemaError, match=f"^{re.escape(f'{path}: {message}')}"):
        read_swebench(RUN_REPORT, path)


def test_malformed_json_lines_name_the_line(tmp_path: Path) -> None:
    path = tmp_path / "ids.jsonl"
    path.write_text(f'"{IDS[0]}"\n{{"instance_id": \n')
    with pytest.raises(SchemaError, match=re.escape(f"{path}:2: invalid JSON")):
        read_swebench(RUN_REPORT, path)


@pytest.mark.parametrize(
    "instances",
    [[IDS[0], 5], [IDS[0], None], [""], [{"instance_id": None}], [{"task": IDS[0]}]],
)
def test_malformed_instance_ids(instances: list[Any]) -> None:
    message = "instances: an instance is not an id string or has no instance_id"
    with pytest.raises(SchemaError, match=f"^{re.escape(message)}$"):
        read_swebench(RUN_REPORT, instances)


# Malformed and unknown result files.


TRUNCATED = RUN_REPORT.read_text()[:200]
NOT_A_FORMAT = (
    "not a SWE-bench run report (resolved_ids), results.json (resolved), or per-instance "
    "report ({instance_id: {resolved: ...}})"
)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        pytest.param("", "invalid JSON", id="empty"),
        pytest.param(" \n", "invalid JSON", id="blank"),
        pytest.param(TRUNCATED, "invalid JSON", id="truncated"),
        pytest.param("[]", "expected a JSON object, got list", id="list"),
        pytest.param('"resolved_ids"', "expected a JSON object, got str", id="string"),
        pytest.param("8", "expected a JSON object, got int", id="number"),
        pytest.param("{}", NOT_A_FORMAT, id="empty-object"),
        pytest.param('{"total_instances": 8}', NOT_A_FORMAT, id="counts-only"),
        pytest.param('{"resolved_ids": "x"}', NOT_A_FORMAT, id="resolved-ids-not-a-list"),
        pytest.param('{"resolved": 2}', NOT_A_FORMAT, id="resolved-not-a-list"),
        pytest.param('{"a": {"resolved": true}, "b": {}}', NOT_A_FORMAT, id="report-no-resolved"),
        pytest.param('{"a": {"resolved": true}, "b": 1}', NOT_A_FORMAT, id="report-not-object"),
    ],
)
def test_malformed_or_unknown_result_files(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "result.json"
    path.write_text(text)
    with pytest.raises(SchemaError, match=f"^{re.escape(f'{path}: {message}')}"):
        read_swebench(path, INSTANCE_LIST)


def test_undecodable_result_file(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_bytes(b"\xff\xfe\x00{")
    with pytest.raises(SchemaError, match=re.escape(f"{path}: invalid JSON")):
        read_swebench(path, INSTANCE_LIST)


@pytest.mark.parametrize("key", [*RUN_REPORT_LISTS])
@pytest.mark.parametrize(
    "value",
    [["ok", 3], [None], "pytest-dev__pytest-90001", {"a": 1}, {}, "", 0, False],
    ids=["int-id", "null-id", "string", "object", "empty-object", "empty-string", "zero", "false"],
)
def test_run_report_id_list_that_is_not_a_list_of_ids(tmp_path: Path, key: str, value: Any) -> None:
    path = write_json(tmp_path / "r.json", edited_report(**{key: value}))
    message = f"{path}: {key!r} is not a list of instance ids"
    if key == "resolved_ids" and not isinstance(value, list):
        message = f"{path}: {NOT_A_FORMAT}"  # a run report is recognized by this list
    with pytest.raises(SchemaError, match=f"^{re.escape(message)}$"):
        read_swebench(path, INSTANCE_LIST)


@pytest.mark.parametrize("value", [[], ["tests_timed_out"], "tests_timed_out", 0, False])
def test_failure_reasons_that_are_not_an_object(tmp_path: Path, value: Any) -> None:
    path = write_json(tmp_path / "r.json", edited_report(failure_reasons=value))
    with pytest.raises(SchemaError, match=re.escape(f"{path}: 'failure_reasons' is not an object")):
        read_swebench(path, INSTANCE_LIST)


def test_null_lists_and_failure_reasons_are_empty(tmp_path: Path) -> None:
    nulls = dict.fromkeys(("infra_failure_ids", "failure_reasons", "ambiguous_failure_ids"))
    path = write_json(tmp_path / f"{MODEL_DIR}.{RUN_ID}.json", edited_report(**nulls))
    assert read_rows(path) == expected_rows_without_classification()


def test_results_json_list_of_non_ids(tmp_path: Path) -> None:
    data = {**RESULTS_RAW, "no_apply": [iid(8), 8]}
    path = write_results(tmp_path, data)
    with pytest.raises(
        SchemaError, match=re.escape(f"{path}: 'no_apply' is not a list of instance")
    ):
        read_swebench(path, INSTANCE_LIST)


def test_unknown_results_json_categories(tmp_path: Path) -> None:
    path = write_results(tmp_path, {**RESULTS_RAW, "zz_flaky": [iid(2)], "aa_skipped": []})
    message = (
        f"{path}: unknown results.json categories ['aa_skipped', 'zz_flaky']; "
        "pass categories= to map them"
    )
    with pytest.raises(SchemaError, match=f"^{re.escape(message)}$"):
        read_swebench(path, INSTANCE_LIST)


def test_results_json_values_that_are_not_lists_are_not_categories(tmp_path: Path) -> None:
    data = {"note": "mock fixture", "total": 8, "meta": {"split": "test"}, **RESULTS_RAW}
    assert read_rows(write_results(tmp_path, data)) == expected_results()


# Run report status mapping.


def expected_rows_without_classification() -> dict[str, dict[str, Any]]:
    """Rows of the run report without swebench 5.0's log classification (as in 2.1.8)."""
    expected = expected_run_report()
    for task, values in expected.items():
        values["failure_reason"] = None
        if task in REPORT["error_ids"]:
            values.update(status="infra_error", status_reason="error_ids")
        elif task in REPORT["unresolved_ids"]:
            values.update(status="ok", status_reason="unresolved_ids")
    return expected


def test_swebench_2_run_report(tmp_path: Path) -> None:
    # swebench 2.1.8's make_run_report writes the same keys without the 5.0 classification.
    added = (
        "infra_failure_instances",
        "ambiguous_failure_instances",
        "infra_failure_ids",
        "ambiguous_failure_ids",
        "failure_reasons",
    )
    path = write_json(tmp_path / RUN_REPORT.name, edited_report(drop=added))
    rows = read_rows(path)
    assert rows == expected_rows_without_classification()
    assert rows[iid(4)]["status"] == "infra_error"  # an unclassified error is not a timeout
    assert rows[iid(5)]["status"] == "ok"  # swebench 2 scored it as a plain unresolved run


def test_informational_lists_are_not_read(tmp_path: Path) -> None:
    unknown = "pytest-dev__pytest-99999"
    data = edited_report(
        completed_ids=[*REPORT["completed_ids"], unknown],
        submitted_ids=[*REPORT["submitted_ids"], unknown],
        ambiguous_failure_ids=[iid(2)],
    )
    path = write_json(tmp_path / RUN_REPORT.name, data)
    assert read_rows(path) == expected_run_report()


@pytest.mark.parametrize(
    ("lists", "reasons", "expected"),
    [
        pytest.param(
            {"resolved_ids": ["a"], "error_ids": ["a"]},
            {},
            ("infra_error", "error_ids", 1.0, None),
            id="resolved-and-error-keeps-the-outcome",
        ),
        pytest.param(
            {"unresolved_ids": ["a"], "incomplete_ids": ["a"]},
            {},
            ("infra_error", "incomplete_ids", 0.0, None),
            id="infra-beats-ok",
        ),
        pytest.param(
            {"empty_patch_ids": ["a"]},
            {"a": "no_tests_collected"},
            ("ok", "empty_patch_ids", 0.0, "no_tests_collected"),
            id="other-reasons-are-recorded-only",
        ),
        pytest.param(
            {"unresolved_ids": ["a"], "infra_failure_ids": ["a"]},
            {"a": "out_of_memory"},
            ("infra_error", "infra_failure_ids", 0.0, "out_of_memory"),
            id="environment-failure",
        ),
        pytest.param(
            {"unresolved_ids": ["a"]},
            {"a": "tests_timed_out"},
            ("timeout", "tests_timed_out", 0.0, "tests_timed_out"),
            id="timeout-of-an-unresolved-run",
        ),
        pytest.param(
            {"resolved_ids": ["a"]},
            {"a": "tests_timed_out"},
            ("timeout", "tests_timed_out", 1.0, "tests_timed_out"),
            id="timeout-keeps-the-outcome",
        ),
        pytest.param(
            {"resolved_ids": []},
            {"a": "tests_timed_out"},
            ("timeout", "tests_timed_out", 0.0, "tests_timed_out"),
            id="timeout-in-no-list",
        ),
        pytest.param(
            {"resolved_ids": []},
            {"a": "container_unavailable"},
            ("infra_error", "missing", 0.0, "container_unavailable"),
            id="reason-in-no-list",
        ),
        pytest.param(
            # swebench 5.0.2: no report.json, and the instance log shows a Docker error
            {"infra_failure_ids": ["a"], "error_ids": ["a"]},
            {"a": "container_unavailable"},
            ("infra_error", "error_ids", 0.0, "container_unavailable"),
            id="equal-rank-goes-to-the-first-list-in-the-table",
        ),
        pytest.param(
            {"resolved_ids": [], "error_ids": None},
            {},
            ("infra_error", "missing", 0.0, None),
            id="in-no-list",
        ),
    ],
)
def test_run_report_status_precedence(
    tmp_path: Path,
    lists: dict[str, Any],
    reasons: dict[str, str],
    expected: tuple[str, str, float, str | None],
) -> None:
    report = {"resolved_ids": [], **lists, "failure_reasons": reasons}
    path = write_json(tmp_path / "m.r.json", report)
    got = read_rows(path, ["a"])["a"]
    assert (got["status"], got["status_reason"], got["outcome"], got["failure_reason"]) == expected
    assert (got["model"], got["source_format"]) == ("m", "run_report")


# results.json status mapping.


def test_results_json_with_later_experiments_lists(tmp_path: Path) -> None:
    # Later experiments-repository files list unresolved, no_logs, and error and have no
    # generated list; no published package writes them, so this one is built here.
    data = {
        "resolved": [iid(1)],
        "unresolved": [iid(2)],
        "no_generation": [iid(3)],
        "error": [iid(5)],
        "no_logs": [iid(6)],
    }
    cases = {
        iid(1): ("ok", "resolved"),
        iid(2): ("ok", "unresolved"),
        iid(3): ("ok", "no_generation"),
        iid(5): ("infra_error", "error"),
        iid(6): ("infra_error", "no_logs"),
    }
    # Without a generated list, an instance in no list is an unresolved, scored run.
    unlisted = {task: ("ok", "unresolved") for task in IDS if task not in cases}
    assert unlisted.keys() == {iid(4), iid(7), iid(8)}
    expected = {
        task: row(
            task,
            status,
            reason,
            model=SUBMISSION.name,
            source_format="results",
            outcome=float(task in data["resolved"]),
        )
        for task, (status, reason) in {**cases, **unlisted}.items()
    }
    assert read_rows(write_results(tmp_path, data)) == expected


@pytest.mark.parametrize(
    ("lists", "categories", "expected"),
    [
        pytest.param(
            {"resolved": ["a"], "test_timeout": ["a"]},
            None,
            ("timeout", "test_timeout", 1.0),
            id="timeout-keeps-the-outcome",
        ),
        pytest.param(
            {"resolved": [], "test_timeout": ["a"], "install_fail": ["a"]},
            None,
            ("infra_error", "install_fail", 0.0),
            id="infra-beats-timeout",
        ),
        pytest.param(
            {"resolved": [], "test_errored": ["a"]},
            None,
            ("infra_error", "test_errored", 0.0),
            id="test-errored",
        ),
        pytest.param(
            {"resolved": [], "reset_failed": ["a"], "generated": ["a"], "with_logs": ["a"]},
            None,
            ("infra_error", "reset_failed", 0.0),
            id="reset-failed",
        ),
        pytest.param(
            {"resolved": [], "excluded": ["a"], "install_fail": ["a"]},
            {"excluded": "flagged"},
            ("flagged", "excluded", 0.0),
            id="flagged-beats-infra",
        ),
        pytest.param(
            {"resolved": ["a"], "applied": ["a"], "generated": ["a"], "with_logs": ["a"]},
            None,
            ("ok", "resolved", 1.0),
            id="resolved-beats-broad-lists",
        ),
        pytest.param(
            {"resolved": [], "no_apply": ["a"], "generated": ["a"], "with_logs": ["a"]},
            None,
            ("ok", "no_apply", 0.0),
            id="no-apply-beats-broad-lists",
        ),
        pytest.param(
            {"generated": ["a"], "with_logs": ["a"], "resolved": []},
            None,
            ("ok", "with_logs", 0.0),
            id="with-logs-is-narrower-than-generated",
        ),
        pytest.param(
            {"with_logs": ["a"], "applied": ["a"], "generated": ["a"], "resolved": []},
            None,
            ("ok", "applied", 0.0),
            id="applied-is-the-narrowest-broad-list",
        ),
        pytest.param(
            {"generated": ["a"], "with_logs": [], "resolved": []},
            None,
            ("infra_error", "no_logs", 0.0),
            id="generated-without-logs",
        ),
        pytest.param(
            {"generated": ["a"], "resolved": []},
            None,
            ("ok", "generated", 0.0),
            id="generated-without-a-with-logs-list",
        ),
        pytest.param(
            {"generated": [], "resolved": []},
            None,
            ("infra_error", "missing", 0.0),
            id="in-no-list-with-generated",
        ),
        pytest.param(
            {"resolved": [], "no_logs": []},
            None,
            ("ok", "unresolved", 0.0),
            id="in-no-list-without-generated",
        ),
        pytest.param(
            {"resolved": [], "no_apply": ["a"]},
            {"no_apply": "infra_error"},
            ("infra_error", "no_apply", 0.0),
            id="categories-replace-a-status",
        ),
        pytest.param(
            {"broken_env": ["a"], "resolved": [], "install_fail": ["a"]},
            {"broken_env": "infra_error"},
            ("infra_error", "install_fail", 0.0),
            id="equal-rank-goes-to-the-table-before-categories",
        ),
    ],
)
def test_results_json_status_precedence(
    tmp_path: Path,
    lists: dict[str, list[str]],
    categories: dict[str, str] | None,
    expected: tuple[str, str, float],
) -> None:
    path = write_json(tmp_path / "sub" / "results.json", lists)
    got = read_rows(path, ["a"], categories=categories)["a"]
    assert (got["status"], got["status_reason"], got["outcome"]) == expected
    assert (got["model"], got["source_format"], got["failure_reason"]) == ("sub", "results", None)


@pytest.mark.parametrize("order", ["recorded", "reversed", "sorted"])
def test_results_json_key_order_does_not_matter(tmp_path: Path, order: str) -> None:
    # JSON objects are unordered: a file re-serialized with other key order reads the same.
    keys = {
        "recorded": list(RESULTS_RAW),
        "reversed": list(reversed(RESULTS_RAW)),
        "sorted": sorted(RESULTS_RAW),
    }[order]
    data = {key: RESULTS_RAW[key] for key in keys}
    assert read_rows(write_results(tmp_path, data)) == expected_results()


def test_categories_flag_a_list_and_only_affect_results_json(tmp_path: Path) -> None:
    data = {**RESULTS_RAW, "excluded": [iid(2), iid(7)]}
    rows = read_rows(write_results(tmp_path, data), categories={"excluded": "flagged"})
    for task in data["excluded"]:
        assert (rows[task]["status"], rows[task]["status_reason"]) == ("flagged", "excluded")
    assert {task for task, r in rows.items() if r["status"] == "flagged"} == set(data["excluded"])
    assert {r["status"] for r in rows.values()} == set(STATUSES)
    # A run report and a log tree have no categories: the option changes nothing there.
    options = {"categories": {"resolved_ids": "flagged", "resolved": "flagged"}}
    assert read_rows(RUN_REPORT, **options) == expected_run_report()
    assert read_rows(TREE, **options) == expected_tree()


def test_categories_must_map_to_a_status() -> None:
    message = f"category 'excluded': status must be one of {STATUSES}"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        read_swebench(RESULTS, INSTANCE_LIST, categories={"excluded": "skipped"})


# Per-instance reports.


BASE = INSTANCE_REPORTS[iid(2)]  # a complete, unresolved swebench 5.0.2 report.json entry


def report_entry(drop: tuple[str, ...] = (), **fields: Any) -> dict[str, Any]:
    entry = {key: value for key, value in copy.deepcopy(BASE).items() if key not in drop}
    entry.update(fields)
    return entry


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        pytest.param(INSTANCE_REPORTS[iid(1)], ("ok", "resolved", 1.0, None), id="resolved"),
        pytest.param(BASE, ("ok", "unresolved", 0.0, None), id="unresolved"),
        pytest.param(
            # what get_eval_report writes for a None patch
            report_entry(drop=("tests_status",), patch_is_None=True, patch_exists=False),
            ("ok", "empty_patch", 0.0, None),
            id="patch-is-none",
        ),
        pytest.param(
            report_entry(patch_exists=False), ("ok", "empty_patch", 0.0, None), id="no-patch"
        ),
        pytest.param(
            report_entry(drop=("tests_status",), patch_successfully_applied=False),
            ("ok", "patch_not_applied", 0.0, None),
            id="patch-not-applied",
        ),
        pytest.param(
            INSTANCE_REPORTS[iid(5)],
            ("infra_error", "infra_failure", 0.0, "container_unavailable"),
            id="environment-failure",
        ),
        pytest.param(
            report_entry(resolved=True, infra_failure=True, infra_failure_reason="out_of_memory"),
            ("infra_error", "infra_failure", 1.0, "out_of_memory"),
            id="environment-failure-keeps-the-outcome",
        ),
        pytest.param(
            report_entry(patch_successfully_applied=False, infra_failure_reason="tests_timed_out"),
            ("timeout", "tests_timed_out", 0.0, "tests_timed_out"),
            id="ambiguous-timeout",
        ),
        pytest.param(
            report_entry(patch_successfully_applied=False, infra_failure_reason="missing_module"),
            ("ok", "patch_not_applied", 0.0, "missing_module"),
            id="other-ambiguous-reason",
        ),
        pytest.param(
            report_entry(drop=("infra_failure",)), ("ok", "unresolved", 0.0, None), id="swebench-2"
        ),
    ],
)
def test_per_instance_report_status(
    tmp_path: Path, entry: dict[str, Any], expected: tuple[str, str, float, str | None]
) -> None:
    path = write_json(tmp_path / "merged.json", {"a": entry, "b": INSTANCE_REPORTS[iid(1)]})
    got = read_rows(path, ["a", "b"])
    assert (
        got["a"]["status"],
        got["a"]["status_reason"],
        got["a"]["outcome"],
        got["a"]["failure_reason"],
    ) == expected
    assert (got["a"]["model"], got["a"]["source_format"]) == ("merged", "instance_reports")
    assert (got["b"]["status"], got["b"]["outcome"]) == ("ok", 1.0)


def test_swebench_2_log_tree(tmp_path: Path) -> None:
    # swebench 2.1.8's report.json has no infra_failure fields.
    tree = copy_tree(tmp_path)
    for task, report in INSTANCE_REPORTS.items():
        entry = {
            k: v for k, v in report.items() if k not in ("infra_failure", "infra_failure_reason")
        }
        write_json(tree / task / "report.json", {task: entry})
    expected = expected_tree()
    expected[iid(5)].update(status="ok", status_reason="patch_not_applied", failure_reason=None)
    assert INSTANCE_REPORTS[iid(5)]["patch_successfully_applied"] is False
    assert read_rows(tree) == expected


# Log tree edits.


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param("  \n", id="blank"),
        pytest.param('{"pytest-dev__pytest-90002": {"resolved": fa', id="truncated"),
        pytest.param("[]", id="list"),
        pytest.param('{"pytest-dev__pytest-90001": {"resolved": true}}', id="other-instance"),
        pytest.param('{"pytest-dev__pytest-90002": {"patch_exists": true}}', id="no-resolved"),
        pytest.param('{"pytest-dev__pytest-90002": true}', id="entry-not-an-object"),
    ],
)
def test_unusable_report_json_is_an_error(tmp_path: Path, text: str) -> None:
    # swebench 5.0.2's make_run_report counts an empty, invalid, or incomplete report.json in
    # error_ids; the reader treats the other unusable shapes here the same way.
    tree = copy_tree(tmp_path)
    (tree / iid(2) / "report.json").write_text(text)
    expected = expected_tree()
    expected[iid(2)].update(status="infra_error", status_reason="invalid_report")
    assert read_rows(tree) == expected


def test_instance_without_report_and_without_timeout_line(tmp_path: Path) -> None:
    tree = copy_tree(tmp_path)
    (tree / iid(2) / "report.json").unlink()  # tests ran, grading did not finish
    expected = expected_tree()
    expected[iid(2)].update(status="infra_error", status_reason="no_report")
    assert "Timeout error" not in (tree / iid(2) / "test_output.txt").read_text()
    assert read_rows(tree) == expected


def test_timeout_line_needs_the_exact_swebench_wording(tmp_path: Path) -> None:
    tree = copy_tree(tmp_path)
    output = tree / iid(4) / "test_output.txt"
    output.write_text(output.read_text().replace("1800 seconds", "a long time"))
    expected = expected_tree()
    expected[iid(4)].update(status="infra_error", status_reason="no_report", failure_reason=None)
    assert read_rows(tree) == expected


def test_any_instance_file_makes_an_instance_directory(tmp_path: Path) -> None:
    tree = copy_tree(tmp_path)
    for task, name in ((iid(3), "patch.diff"), (iid(7), "run_instance.log")):
        (tree / task).mkdir()
        (tree / task / name).write_text("mock\n")
    (tree / "pytest-dev__pytest-99999").mkdir()  # no instance files: not an instance
    (tree / "pytest-dev__pytest-99999" / "eval.sh").write_text("mock\n")
    expected = expected_tree()
    for task in (iid(3), iid(7)):
        expected[task].update(status="infra_error", status_reason="no_report")
    assert read_rows(tree) == expected


def test_unknown_instance_directory(tmp_path: Path) -> None:
    tree = copy_tree(tmp_path)
    extra = tree / "pytest-dev__pytest-99999"
    shutil.copytree(tree / iid(1), extra)
    message = f"{tree}: 1 result ids are not in the instance list (first: {extra.name})"
    with pytest.raises(SchemaError, match=re.escape(message)):
        read_swebench(tree, INSTANCE_LIST)
    assert read_rows(tree, on_unknown="drop") == expected_tree()


# Model and run labels.


def test_model_labels_from_paths(tmp_path: Path) -> None:
    assert read_swebench(TREE, INSTANCE_LIST).models == [TREE.name]
    assert read_swebench(RESULTS, INSTANCE_LIST).models == [SUBMISSION.name]
    assert read_swebench(RUN_REPORT, INSTANCE_LIST).models == [MODEL_DIR]
    # The run report is named <model>.<run_id>.json: the last dot-separated part is dropped.
    dotted = write_json(tmp_path / "org__model-1.5.run-7.json", REPORT)
    assert read_swebench(dotted, INSTANCE_LIST).models == ["org__model-1.5"]
    flat = write_json(tmp_path / "flat-submission" / "results.json", RESULTS_RAW)
    assert read_swebench(flat, INSTANCE_LIST).models == ["flat-submission"]


def test_model_labels_from_relative_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(TREE)
    assert rows_of(read_swebench(".", IDS)) == expected_tree()
    monkeypatch.chdir(TREE / iid(1))
    assert read_swebench("..", IDS).models == [MODEL_DIR]
    monkeypatch.chdir(RESULTS.parent)
    assert rows_of(read_swebench("results.json", IDS)) == expected_results()
    monkeypatch.chdir(SUBMISSION)
    assert read_swebench(Path("results") / "results.json", IDS).models == [SUBMISSION.name]
    monkeypatch.chdir(RUN)
    assert rows_of(read_swebench(RUN_REPORT.name, INSTANCE_LIST.name)) == expected_run_report()


def test_experiments_logs_directory_by_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logs = Path(shutil.copytree(TREE, tmp_path / "20261023_mock-agent" / "logs"))
    monkeypatch.chdir(logs)
    assert read_swebench(".", IDS).models == ["20261023_mock-agent"]


@pytest.mark.parametrize("path", [RUN_REPORT, TREE, RESULTS], ids=["report", "tree", "results"])
def test_model_and_run_options(path: Path) -> None:
    res = read_swebench(path, INSTANCE_LIST, model="mock-label", run=3)
    assert res.models == ["mock-label"]
    assert set(res.frame["run"]) == {"3"}
    named = read_swebench(path, INSTANCE_LIST, run="seed-1")
    assert set(named.frame["run"]) == {"seed-1"}
    assert len(named) == len(IDS)
