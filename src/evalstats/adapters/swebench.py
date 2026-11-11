"""Reader for SWE-bench evaluation results of one submission.

The files are parsed directly; the ``swebench`` package is not imported. Every instance in the
given instance list gets exactly one row, so unresolved and missing instances are explicit.

Formats read, detected from the content:

1. Run report from ``swebench.harness.reporting.make_run_report`` (swebench 2.0 and later,
   file ``<model>.<run_id>.json``, ``schema_version`` 2; ``sb-cli get-report`` returns the same
   object): lists ``resolved_ids``, ``unresolved_ids``, ``empty_patch_ids``, ``error_ids``,
   ``incomplete_ids``, and from swebench 5.0 ``infra_failure_ids``, ``ambiguous_failure_ids``,
   and ``failure_reasons``.
2. Experiments-style ``results.json``: lists of instance ids by category, as built by
   swebench 1.x ``swebench.metrics.report.get_model_report`` (``no_generation``,
   ``generated``, ``with_logs``, ``install_fail``, ``reset_failed``, ``no_apply``, ``applied``,
   ``test_errored``, ``test_timeout``, ``resolved``) and stored per submission in the
   SWE-bench experiments repository. Lists named ``unresolved``, ``no_logs``, and ``error``
   are read with the same table; other list names raise :class:`SchemaError` unless
   ``categories`` maps them to a status.
3. Per-instance ``report.json`` (``{instance_id: {"resolved": bool, ...}}``, written by
   ``run_instance`` in swebench 2.0 and later): a directory with one subdirectory per instance
   (``logs/run_evaluation/<run_id>/<model>/`` or an experiments ``<submission>/logs/``), or one
   JSON object that merges several reports.

Rows. ``model`` is ``model`` if given, else taken from the path: the run report's file name
without its last dot-separated part (the run id), the submission directory for
``<submission>/results/results.json`` or ``<submission>/logs/``, the model directory for
``logs/run_evaluation/<run_id>/<model>/``, and the file stem for a merged report file.
``task`` is the instance id, ``run`` is ``run`` (default ``"0"``), and ``outcome`` is 1 for a
resolved instance and 0 otherwise.

Status mapping (``status_reason`` names the deciding list or field). When an id appears in
several lists, the most severe status wins (``flagged``, then ``infra_error``, then
``timeout``, then ``ok``). Among equals, a specific list beats the broad ``applied``, which
beats ``with_logs``, which beats ``generated``; other ties go to the list that comes first in
:data:`RUN_REPORT_LISTS` or :data:`RESULTS_CATEGORIES` (then ``categories``), whatever the key
order of the file. The outcome is still 1 if a resolved list holds the id.

- Run report: ``resolved_ids``, ``unresolved_ids``, and ``empty_patch_ids`` give ``ok`` (an
  empty patch counts as unresolved in SWE-bench); ``error_ids`` (no usable report.json),
  ``incomplete_ids`` (no prediction), and ``infra_failure_ids`` (swebench classifies the logs as
  an environment fault) give ``infra_error``; a ``failure_reasons`` entry of
  ``tests_timed_out`` gives ``timeout``; an instance in no list gives ``infra_error``
  [``missing``].
- Experiments ``results.json``: ``resolved``, ``unresolved``, ``applied``, ``generated``,
  ``with_logs``, ``no_generation`` (empty patch), and ``no_apply`` (the patch did not apply)
  give ``ok``; ``test_timeout`` gives ``timeout``; ``test_errored``, ``install_fail``,
  ``reset_failed``, ``no_logs``, and ``error`` give ``infra_error``; ids in ``generated`` but
  not ``with_logs`` give ``infra_error`` [``no_logs``]. An instance in no list gives
  ``infra_error`` [``missing``] when the file has a ``generated`` list (swebench 1.x lists every
  prediction in ``generated`` or ``no_generation``), and ``ok`` [``unresolved``] otherwise
  (files that list only resolved instances and failures).
- Per-instance reports: ``infra_failure`` true gives ``infra_error``; an
  ``infra_failure_reason`` of ``tests_timed_out`` gives ``timeout``; otherwise ``ok``
  [``resolved``, ``empty_patch``, ``patch_not_applied``, or ``unresolved``]. In a directory,
  an instance directory whose ``report.json`` is empty, invalid, or lacks the instance gives
  ``infra_error`` [``invalid_report``], as ``make_run_report`` counts it as an error; one
  without ``report.json`` gives ``timeout`` [``tests_timed_out``] when ``test_output.txt``
  has swebench's timeout line (``Timeout error: N seconds exceeded.``) and ``infra_error``
  [``no_report``] otherwise; an instance without a directory gives ``infra_error``
  [``missing``]. The tree does not record whether a prediction existed, and swebench skips
  empty patches before evaluation, so empty patches also appear as ``missing`` here; the run
  report tells them apart.

SWE-bench marks no run as excluded, so ``flagged`` comes only from ``categories``.

Covariates: ``status``, ``status_reason``, ``failure_reason`` (swebench 5.0 log
classification, when present), and ``source_format`` (``run_report``, ``results``, or
``instance_reports``).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from evalstats.adapters._common import (
    FLAGGED,
    INFRA_ERROR,
    OK,
    STATUSES,
    TIMEOUT,
    as_mapping,
    build_results,
    load_json,
    load_json_file,
)
from evalstats.schema import EvalResults, SchemaError

RUN_REPORT_LISTS: Mapping[str, str] = {
    "resolved_ids": OK,
    "unresolved_ids": OK,
    "empty_patch_ids": OK,
    "error_ids": INFRA_ERROR,
    "incomplete_ids": INFRA_ERROR,
    "infra_failure_ids": INFRA_ERROR,
}
"""Status of each id list in a swebench run report (other lists are informational)."""

RESULTS_CATEGORIES: Mapping[str, str] = {
    "resolved": OK,
    "unresolved": OK,
    "applied": OK,
    "generated": OK,
    "with_logs": OK,
    "no_generation": OK,
    "no_apply": OK,
    "test_timeout": TIMEOUT,
    "test_errored": INFRA_ERROR,
    "install_fail": INFRA_ERROR,
    "reset_failed": INFRA_ERROR,
    "no_logs": INFRA_ERROR,
    "error": INFRA_ERROR,
}
"""Status of each category list in an experiments-style ``results.json``."""

TIMEOUT_REASON = "tests_timed_out"
_TIMEOUT_LINE = re.compile(r"Timeout error: \d+ seconds exceeded")
_SEVERITY = {OK: 0, TIMEOUT: 1, INFRA_ERROR: 2, FLAGGED: 3}
_BROAD_LISTS = ("generated", "with_logs", "applied")  # supersets of other lists, broadest first
_RESOLVED_LISTS = ("resolved_ids", "resolved")
_INSTANCE_FILES = ("report.json", "test_output.txt", "run_instance.log", "patch.diff")
_COLUMNS = ("status_reason", "failure_reason", "source_format")

_Verdict = tuple[str, float, str, str | None]  # status, outcome, status_reason, failure_reason


def read_swebench(
    path: str | Path,
    instances: Iterable[str] | str | Path,
    *,
    model: str | None = None,
    run: str | int = 0,
    categories: Mapping[str, str] | None = None,
    on_unknown: Literal["error", "drop"] = "error",
) -> EvalResults:
    """Read one submission's SWE-bench results, one row per instance in ``instances``.

    ``path`` is a run report, an experiments ``results.json``, a merged per-instance report
    file, or a directory with one subdirectory per instance. ``instances`` is the full instance
    list: ids, or a path (a ``str`` is always taken as a path) to a text file with one id per
    line, a JSON list of ids or of objects with ``instance_id``, or JSON Lines of either.
    ``model`` labels every row (default: inferred from ``path``) and ``run`` sets the run label.
    ``categories`` adds or replaces ``results.json`` categories, mapped to a status.
    ``on_unknown="drop"`` ignores result ids outside ``instances`` instead of raising. The
    module docstring gives the formats and the status mapping. Raises :class:`SchemaError` if a
    file is malformed or of an unknown format or the instance list is empty.
    """
    if on_unknown not in ("error", "drop"):
        raise ValueError(f"on_unknown must be 'error' or 'drop', not {on_unknown!r}")
    table = dict(RESULTS_CATEGORIES)
    for name, status in (categories or {}).items():
        if status not in STATUSES:
            raise ValueError(f"category {name!r}: status must be one of {STATUSES}")
        table[name] = status
    ids = _load_instances(instances)
    source = Path(path)
    if source.is_dir():
        fmt, verdicts = "instance_reports", _read_tree(source)
        default = _verdict(INFRA_ERROR, False, "missing")
    elif source.is_file():
        fmt, verdicts, default = _read_file(source, table)
    else:
        raise SchemaError(f"{source}: no such file or directory")
    known = set(ids)
    unknown = sorted(set(verdicts) - known)
    if unknown and on_unknown == "error":
        raise SchemaError(
            f"{source}: {len(unknown)} result ids are not in the instance list (first: "
            f"{unknown[0]}); check that the list matches the split, or pass on_unknown='drop'"
        )
    label = model or _infer_model(source, fmt)
    rows = []
    for instance_id in ids:
        status, outcome, reason, failure = verdicts.get(instance_id, default)
        rows.append(
            {
                "model": label,
                "task": instance_id,
                "run": str(run),
                "outcome": outcome,
                "status": status,
                "status_reason": reason,
                "failure_reason": failure,
                "source_format": fmt,
            }
        )
    return build_results(rows, source=str(source), columns=_COLUMNS)


def _verdict(status: str, resolved: bool, reason: str, failure: str | None = None) -> _Verdict:
    return status, 1.0 if resolved else 0.0, reason, failure


def _load_instances(instances: Iterable[str] | str | Path) -> list[str]:
    if isinstance(instances, str | Path):
        file = Path(instances)
        if not file.is_file():
            raise SchemaError(f"{file}: instance list not found")
        if file.suffix == ".json":
            items = load_json_file(file)
            if not isinstance(items, list):
                raise SchemaError(f"{file}: expected a JSON list of instances")
        elif file.suffix == ".jsonl":
            lines = file.read_text().splitlines()
            items = [load_json(line, f"{file}:{n}") for n, line in enumerate(lines, 1) if line]
        else:
            items = [line.strip() for line in file.read_text().splitlines() if line.strip()]
        where = str(file)
    else:
        items, where = list(instances), "instances"
    ids = [_instance_id(item, where) for item in items]
    if not ids:
        raise SchemaError(f"{where}: the instance list is empty")
    return list(dict.fromkeys(ids))


def _instance_id(item: Any, where: str) -> str:
    if isinstance(item, Mapping):
        item = item.get("instance_id")
    if not isinstance(item, str) or not item:
        raise SchemaError(f"{where}: an instance is not an id string or has no instance_id")
    return item


def _read_file(path: Path, table: Mapping[str, str]) -> tuple[str, dict[str, _Verdict], _Verdict]:
    data = as_mapping(load_json_file(path), str(path))
    if isinstance(data.get("resolved_ids"), list):
        return "run_report", _read_run_report(data, path), _verdict(INFRA_ERROR, False, "missing")
    if isinstance(data.get("resolved"), list):
        return "results", *_read_results(data, path, table)
    if data and all(isinstance(v, Mapping) and "resolved" in v for v in data.values()):
        verdicts = {str(k): _report_verdict(v) for k, v in data.items()}
        return "instance_reports", verdicts, _verdict(INFRA_ERROR, False, "missing")
    raise SchemaError(
        f"{path}: not a SWE-bench run report (resolved_ids), results.json (resolved), or "
        "per-instance report ({instance_id: {resolved: ...}})"
    )


def _id_list(data: Mapping[str, Any], key: str, path: Path) -> list[str]:
    values = data.get(key)
    if values is None:  # absent or null; any other value that is not a list is malformed
        return []
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise SchemaError(f"{path}: {key!r} is not a list of instance ids")
    return values


def _rank(verdict: _Verdict) -> tuple[int, int]:
    """Severity, then specificity: a broad list ranks below every other list, broadest lowest."""
    reason = verdict[2]
    narrowness = _BROAD_LISTS.index(reason) if reason in _BROAD_LISTS else len(_BROAD_LISTS)
    return _SEVERITY[verdict[0]], narrowness


def _merge(verdicts: dict[str, _Verdict], ids: Iterable[str], new: _Verdict) -> None:
    """Record ``new`` per id unless a verdict of equal or higher rank is already set."""
    rank = _rank(new)
    for instance_id in ids:
        old = verdicts.get(instance_id)
        if old is None or rank > _rank(old):
            verdicts[instance_id] = new


def _resolve(verdicts: dict[str, _Verdict], resolved: Iterable[str]) -> None:
    for instance_id in resolved:
        status, _, reason, failure = verdicts[instance_id]
        verdicts[instance_id] = (status, 1.0, reason, failure)


def _read_run_report(data: Mapping[str, Any], path: Path) -> dict[str, _Verdict]:
    verdicts: dict[str, _Verdict] = {}
    for key, status in RUN_REPORT_LISTS.items():
        _merge(verdicts, _id_list(data, key, path), _verdict(status, False, key))
    _resolve(verdicts, _id_list(data, "resolved_ids", path))
    reasons = data.get("failure_reasons")
    if reasons is None:  # absent before swebench 5.0
        reasons = {}
    elif not isinstance(reasons, Mapping):
        raise SchemaError(f"{path}: 'failure_reasons' is not an object")
    for instance_id, reason in reasons.items():
        status, outcome, deciding, _ = verdicts.get(
            instance_id, _verdict(INFRA_ERROR, False, "missing")
        )
        if reason == TIMEOUT_REASON:
            status, deciding = TIMEOUT, TIMEOUT_REASON
        verdicts[instance_id] = (status, outcome, deciding, str(reason))
    return verdicts


def _read_results(
    data: Mapping[str, Any], path: Path, table: Mapping[str, str]
) -> tuple[dict[str, _Verdict], _Verdict]:
    unknown = sorted(k for k, v in data.items() if isinstance(v, list) and k not in table)
    if unknown:
        raise SchemaError(
            f"{path}: unknown results.json categories {unknown}; pass categories= to map them"
        )
    lists = [key for key in table if isinstance(data.get(key), list)]  # table order breaks ties
    verdicts: dict[str, _Verdict] = {}
    for key in lists:
        _merge(verdicts, _id_list(data, key, path), _verdict(table[key], False, key))
    if "generated" in lists and "with_logs" in lists:
        logged = set(_id_list(data, "with_logs", path))
        no_logs = [i for i in _id_list(data, "generated", path) if i not in logged]
        _merge(verdicts, no_logs, _verdict(INFRA_ERROR, False, "no_logs"))
    _resolve(verdicts, _id_list(data, "resolved", path))
    if "generated" in lists:
        return verdicts, _verdict(INFRA_ERROR, False, "missing")
    return verdicts, _verdict(OK, False, "unresolved")


def _report_verdict(report: Mapping[str, Any]) -> _Verdict:
    resolved = report.get("resolved") is True
    failure = report.get("infra_failure_reason")
    failure = None if failure is None else str(failure)
    if report.get("infra_failure") is True:
        return _verdict(INFRA_ERROR, resolved, "infra_failure", failure)
    if failure == TIMEOUT_REASON:
        return _verdict(TIMEOUT, resolved, TIMEOUT_REASON, failure)
    if resolved:
        return _verdict(OK, True, "resolved", failure)
    if report.get("patch_is_None") is True or report.get("patch_exists") is False:
        return _verdict(OK, False, "empty_patch", failure)
    if report.get("patch_successfully_applied") is False:
        return _verdict(OK, False, "patch_not_applied", failure)
    return _verdict(OK, False, "unresolved", failure)


def _read_tree(directory: Path) -> dict[str, _Verdict]:
    verdicts: dict[str, _Verdict] = {}
    for child in sorted(p for p in directory.iterdir() if p.is_dir()):
        if not any((child / name).is_file() for name in _INSTANCE_FILES):
            continue
        verdicts[child.name] = _tree_verdict(child)
    if not verdicts:
        raise SchemaError(
            f"{directory}: no instance directories found; pass the directory that holds one "
            "subdirectory per instance (logs/run_evaluation/<run_id>/<model>)"
        )
    return verdicts


def _tree_verdict(instance_dir: Path) -> _Verdict:
    report_file = instance_dir / "report.json"
    if report_file.is_file():
        try:
            data = load_json(report_file.read_bytes(), str(report_file))
        except SchemaError:
            data = None
        report = data.get(instance_dir.name) if isinstance(data, Mapping) else None
        if not isinstance(report, Mapping) or "resolved" not in report:
            return _verdict(INFRA_ERROR, False, "invalid_report")
        return _report_verdict(report)
    output = instance_dir / "test_output.txt"
    if output.is_file() and _TIMEOUT_LINE.search(output.read_text(errors="replace")):
        return _verdict(TIMEOUT, False, TIMEOUT_REASON, TIMEOUT_REASON)
    return _verdict(INFRA_ERROR, False, "no_report")


def _infer_model(given: Path, fmt: str) -> str:
    path = Path(os.path.abspath(given))  # "." or a bare file name names no directory itself
    if fmt == "instance_reports" and path.is_dir():
        return path.parent.name if path.name == "logs" else path.name
    if fmt == "results":
        parent = path.parent
        return parent.parent.name if parent.name == "results" else parent.name
    if fmt == "run_report":
        return path.stem.rsplit(".", 1)[0]
    return path.stem
