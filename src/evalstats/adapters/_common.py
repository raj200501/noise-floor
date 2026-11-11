"""Shared pieces of the results adapters: status values, JSON loading, and table building."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from evalstats.schema import KEY_COLUMNS, EvalResults, SchemaError

STATUS = "status"
OK = "ok"
INFRA_ERROR = "infra_error"
TIMEOUT = "timeout"
FLAGGED = "flagged"
STATUSES = (OK, INFRA_ERROR, TIMEOUT, FLAGGED)
"""Values of the ``status`` covariate that every adapter adds.

- ``ok``: the run finished and was scored normally.
- ``infra_error``: the harness, environment, or grader failed, the run errored, or it left no
  logs or results.
- ``timeout``: a time, token, or step limit ended the run.
- ``flagged``: the source marks the run as invalid or excluded.

Non-ok rows keep the source's own outcome (0 when the source has none).
"""


def load_json(data: str | bytes, where: str) -> Any:
    """Parse JSON text, raising :class:`SchemaError` that names ``where`` on failure."""
    try:
        return json.loads(data)
    except (ValueError, UnicodeDecodeError) as exc:
        raise SchemaError(f"{where}: invalid JSON ({exc})") from exc


def load_json_file(path: Path) -> Any:
    """Read and parse a JSON file, raising :class:`SchemaError` on failure."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SchemaError(f"{path}: cannot read ({exc})") from exc
    return load_json(data, str(path))


def as_mapping(value: Any, where: str) -> Mapping[str, Any]:
    """Return ``value`` if it is a JSON object, else raise :class:`SchemaError`."""
    if not isinstance(value, Mapping):
        raise SchemaError(f"{where}: expected a JSON object, got {type(value).__name__}")
    return value


def check_unit(value: float, where: str) -> float:
    """Return ``value`` if it lies in [0, 1], else raise :class:`SchemaError`."""
    if not 0.0 <= value <= 1.0:
        raise SchemaError(f"{where}: outcome {value!r} is outside [0, 1]")
    return value


def build_results(
    records: Sequence[Mapping[str, Any]],
    *,
    source: str,
    columns: Iterable[str],
    numeric: Iterable[str] = (),
) -> EvalResults:
    """Turn adapter rows into :class:`EvalResults` with a fixed covariate column order.

    ``columns`` lists the covariates in output order; ``numeric`` names covariates that are
    coerced to float (missing values become NaN). Duplicate (model, task, run) keys raise
    :class:`SchemaError` naming the first clash.
    """
    if not records:
        raise SchemaError(f"{source}: no runs found")
    frame = pd.DataFrame.from_records(list(records))
    ordered = [*KEY_COLUMNS, "outcome", STATUS, *columns]
    frame = frame.reindex(columns=ordered)
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    return EvalResults.from_frame(frame)
