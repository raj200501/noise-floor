"""Readers for generic long-format results files (CSV, TSV, and JSON Lines)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from evalstats.schema import EvalResults, SchemaError


def read_csv(
    path: str | Path,
    *,
    model: str = "model",
    task: str = "task",
    run: str | None = "run",
    outcome: str = "outcome",
) -> EvalResults:
    """Read a CSV file with one row per task run."""
    frame = pd.read_csv(path, dtype={model: str, task: str})
    return EvalResults.from_frame(frame, model=model, task=task, run=run, outcome=outcome)


def read_jsonl(
    path: str | Path,
    *,
    model: str = "model",
    task: str = "task",
    run: str | None = "run",
    outcome: str = "outcome",
) -> EvalResults:
    """Read a JSON Lines file with one object per task run (blank lines skipped)."""
    records = []
    with Path(path).open() as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except ValueError as exc:
                raise SchemaError(f"{path}:{lineno}: invalid JSON") from exc
    return EvalResults.from_records(records, model=model, task=task, run=run, outcome=outcome)


def read_results(
    path: str | Path,
    *,
    model: str = "model",
    task: str = "task",
    run: str | None = "run",
    outcome: str = "outcome",
) -> EvalResults:
    """Read CSV or JSON Lines, chosen by file extension."""
    suffix = Path(path).suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        return read_jsonl(path, model=model, task=task, run=run, outcome=outcome)
    if suffix in (".csv", ".tsv"):
        if suffix == ".tsv":
            frame = pd.read_csv(path, sep="\t", dtype={model: str, task: str})
            return EvalResults.from_frame(frame, model=model, task=task, run=run, outcome=outcome)
        return read_csv(path, model=model, task=task, run=run, outcome=outcome)
    raise SchemaError(f"unsupported file type {suffix!r}; use .csv, .tsv, or .jsonl")
