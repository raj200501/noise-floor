"""Long-format eval results: one row per (model, task, run).

The benchmark score of a model is defined throughout evalstats as the mean over
tasks of the per-task mean outcome (every task weighs the same, however many runs
it has). Outcomes are success indicators or partial-credit scores in [0, 1].
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

KEY_COLUMNS = ("model", "task", "run")
OUTCOME = "outcome"
REQUIRED = (*KEY_COLUMNS, OUTCOME)

FloatArray = npt.NDArray[np.float64]


class SchemaError(ValueError):
    """Raised when results do not fit the long-format schema."""


@dataclass(frozen=True)
class PairedScores:
    """Per-task mean outcomes of two models on the tasks both attempted."""

    tasks: tuple[str, ...]
    a: FloatArray
    b: FloatArray
    runs_a: npt.NDArray[np.int64]
    runs_b: npt.NDArray[np.int64]

    @property
    def differences(self) -> FloatArray:
        """Per-task difference ``a - b``."""
        return self.a - self.b


@dataclass(frozen=True)
class EvalResults:
    """Validated long-format results table.

    Columns: ``model``, ``task``, ``run`` (all stored as strings), ``outcome``
    (float in [0, 1]) and any number of covariate columns. Each (model, task, run)
    appears once. Build with :meth:`from_frame` or :meth:`from_records`.
    """

    frame: pd.DataFrame

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        *,
        model: str = "model",
        task: str = "task",
        run: str | None = "run",
        outcome: str = "outcome",
    ) -> EvalResults:
        """Validate a DataFrame and rename its columns to the schema names.

        If ``run`` is None, runs are numbered within each (model, task) in row order.
        """
        mapping = {model: "model", task: "task", outcome: OUTCOME}
        if run is not None:
            mapping[run] = "run"
        missing = [src for src in mapping if src not in df.columns]
        if missing:
            raise SchemaError(f"missing columns: {', '.join(missing)}")
        clashes = [dst for src, dst in mapping.items() if dst in df.columns and dst != src]
        clashes = [c for c in clashes if c not in mapping]
        if clashes:
            raise SchemaError(f"columns would be overwritten: {', '.join(clashes)}")
        frame = df.rename(columns=mapping).copy()
        if run is None:
            frame["run"] = frame.groupby(["model", "task"], sort=False).cumcount()
        return cls(_validate(frame))

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        model: str = "model",
        task: str = "task",
        run: str | None = "run",
        outcome: str = "outcome",
    ) -> EvalResults:
        """Build from an iterable of row mappings (for example parsed JSONL)."""
        frame = pd.DataFrame.from_records(list(records))
        return cls.from_frame(frame, model=model, task=task, run=run, outcome=outcome)

    @property
    def models(self) -> list[str]:
        """Model identifiers in first-appearance order."""
        return list(pd.unique(self.frame["model"]))

    @property
    def tasks(self) -> list[str]:
        """Task identifiers in first-appearance order."""
        return list(pd.unique(self.frame["task"]))

    @property
    def covariates(self) -> list[str]:
        """Columns other than the schema columns."""
        return [c for c in self.frame.columns if c not in REQUIRED]

    @property
    def is_binary(self) -> bool:
        """True if every outcome is 0 or 1."""
        return bool(self.frame[OUTCOME].isin([0.0, 1.0]).all())

    def __len__(self) -> int:
        """Number of rows (task runs)."""
        return len(self.frame)

    def subset(
        self, models: Sequence[str] | None = None, tasks: Sequence[str] | None = None
    ) -> EvalResults:
        """Rows for the given models and tasks only."""
        mask = pd.Series(True, index=self.frame.index)
        if models is not None:
            mask &= self.frame["model"].isin(list(models))
        if tasks is not None:
            mask &= self.frame["task"].isin(list(tasks))
        sub = self.frame.loc[mask]
        if sub.empty:
            raise SchemaError("subset is empty")
        return EvalResults(sub.reset_index(drop=True))

    def _model_rows(self, model: str) -> pd.DataFrame:
        rows = self.frame.loc[self.frame["model"] == model]
        if rows.empty:
            raise SchemaError(f"unknown model: {model!r}")
        return rows

    def task_scores(self, model: str) -> pd.Series:
        """Mean outcome per task for one model, indexed by task."""
        rows = self._model_rows(model)
        return rows.groupby("task", sort=True)[OUTCOME].mean()

    def run_counts(self, model: str) -> pd.Series:
        """Number of runs per task for one model, indexed by task."""
        rows = self._model_rows(model)
        return rows.groupby("task", sort=True)[OUTCOME].size()

    def score(self, model: str) -> float:
        """Benchmark score: mean over tasks of the per-task mean outcome."""
        return float(self.task_scores(model).mean())

    def paired(self, model_a: str, model_b: str) -> PairedScores:
        """Per-task scores of two models on the tasks both attempted (sorted by task)."""
        sa, sb = self.task_scores(model_a), self.task_scores(model_b)
        shared = sa.index.intersection(sb.index).sort_values()
        if len(shared) == 0:
            raise SchemaError(f"{model_a!r} and {model_b!r} share no tasks")
        ra, rb = self.run_counts(model_a), self.run_counts(model_b)
        return PairedScores(
            tasks=tuple(str(t) for t in shared),
            a=sa.loc[shared].to_numpy(dtype=np.float64),
            b=sb.loc[shared].to_numpy(dtype=np.float64),
            runs_a=ra.loc[shared].to_numpy(dtype=np.int64),
            runs_b=rb.loc[shared].to_numpy(dtype=np.int64),
        )

    def score_matrix(self) -> pd.DataFrame:
        """Tasks by models table of per-task mean outcomes (NaN where a model has no runs)."""
        table = self.frame.pivot_table(
            index="task", columns="model", values=OUTCOME, aggfunc="mean", sort=True
        )
        return table.reindex(columns=self.models)


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    for col in REQUIRED:
        if col not in frame.columns:
            raise SchemaError(f"missing column: {col}")
    if frame.empty:
        raise SchemaError("no rows")
    for col in KEY_COLUMNS:
        if frame[col].isna().any():
            raise SchemaError(f"missing values in {col}")
        frame[col] = frame[col].astype(str)
    try:
        values = pd.to_numeric(frame[OUTCOME], errors="raise").astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"non-numeric outcome: {exc}") from exc
    if not np.isfinite(values.to_numpy()).all():
        raise SchemaError("outcome has missing or non-finite values")
    if ((values < 0) | (values > 1)).any():
        raise SchemaError("outcome outside [0, 1]")
    frame[OUTCOME] = values
    dup = frame.duplicated(subset=list(KEY_COLUMNS))
    if dup.any():
        first = frame.loc[dup, list(KEY_COLUMNS)].iloc[0].tolist()
        raise SchemaError(f"duplicate (model, task, run): {tuple(first)}")
    ordered = [*REQUIRED, *[c for c in frame.columns if c not in REQUIRED]]
    return frame[ordered].reset_index(drop=True)
