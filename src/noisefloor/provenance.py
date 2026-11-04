"""Provenance stamps for simulation results (the E0 cells of G5 and the G6 datasets).

Every result row carries a compact stamp (:func:`stamp`): the hash of its process's
environment, the commit it ran from, whether ``src/`` (or the lock or project file) had
uncommitted changes, and the time. The environment is everything the numbers depend on
besides the seed: the tree hash of ``src/`` and the blob hash of ``uv.lock`` at that
commit, and the Python, NumPy, SciPy, pandas, R, lme4, Matrix, BLAS, and LAPACK versions
(:func:`environment`). Each process reads its environment once, at its first stamp, so the
stamp names the code it loaded. Banking accepts the rows of one study only when they share
a single environment from a clean ``src/`` (:func:`single_environment`), and a run command
refuses to add rows to a study whose rows ran in another environment
(:func:`check_same_environment`).

The stamp also names the analysis code (:func:`analysis_code`): the files that compute
the E1 intervals and their G6 and E0 part 2 validation, the lock, and the libraries. The
E1 analysis and the post-E1 check refuse to run under analysis code other than the code
G6 and E0 part 2 ran (prereg 0004). E1 run records carry :func:`code_identity`, which the
arms compare before every batch.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

REPO = Path(__file__).resolve().parents[2]
_PROCESS: dict[str, Any] | None = None
# The files that compute E1's intervals, the G6 and E0 part 2 analyses, and the post-E1
# check (prereg 0004 pins them from G6 to the E1 analysis).
ANALYSIS_CODE = (
    "src/evalstats/bounds.py",
    "src/evalstats/compare.py",
    "src/evalstats/glmm.py",
    "src/evalstats/intervals.py",
    "src/evalstats/power.py",
    "src/evalstats/rank.py",
    "src/evalstats/schema.py",
    "src/noisefloor/analysis/e1.py",
    "src/noisefloor/analysis/mde.py",
    "src/noisefloor/post_check.py",
    "src/noisefloor/recovery.py",
)


def _git(*args: str, repo: Path = REPO) -> str:
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def library_versions() -> dict[str, str | None]:
    """Python and numerical library versions; R entries are None without rpy2 or R."""
    out: dict[str, str | None] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
    }
    try:
        from evalstats.glmm import r_versions  # noqa: PLC0415 - rpy2 loads R lazily

        out.update(r_versions())
    except Exception:  # no rpy2, no R, or no lme4: the R entries stay empty
        out.update(dict.fromkeys(("r", "lme4", "matrix", "blas", "lapack")))
    return out


def _hash(key: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def analysis_code(repo: Path = REPO, versions: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """A 16-hex hash of :data:`ANALYSIS_CODE` at HEAD, ``uv.lock``, and the libraries.

    ``dirty`` says whether any of those files has uncommitted changes.
    """
    files = {path: _git("rev-parse", f"HEAD:{path}", repo=repo) for path in ANALYSIS_CODE}
    key = {
        "files": files,
        "uv_lock": _git("rev-parse", "HEAD:uv.lock", repo=repo),
        **(library_versions() if versions is None else versions),
    }
    dirty = _git("status", "--porcelain", "--", *ANALYSIS_CODE, "uv.lock", repo=repo)
    return {"hash": _hash(key), "dirty": bool(dirty)}


def code_identity(repo: Path = REPO) -> dict[str, Any]:
    """The run code of an E1 batch: the ``src/`` tree and ``uv.lock`` at HEAD, and ``dirty``."""
    dirty = _git("status", "--porcelain", "--", "src", "uv.lock", "pyproject.toml", repo=repo)
    return {
        "src_tree": _git("rev-parse", "HEAD:src", repo=repo),
        "uv_lock": _git("rev-parse", "HEAD:uv.lock", repo=repo),
        "dirty": bool(dirty),
    }


def environment(repo: Path = REPO) -> dict[str, Any]:
    """The code and libraries this process runs, with ``env``, a 16-hex hash of them."""
    code = {"src_tree": _git("rev-parse", "HEAD:src", repo=repo)}
    code["uv_lock"] = _git("rev-parse", "HEAD:uv.lock", repo=repo)
    versions = library_versions()
    key = {**code, **versions}
    dirty = _git("status", "--porcelain", "--", "src", "uv.lock", "pyproject.toml", repo=repo)
    return {
        "env": _hash(key),
        "key": key,
        "analysis": analysis_code(repo, versions)["hash"],
        "commit": _git("rev-parse", "HEAD", repo=repo),
        "src_dirty": bool(dirty),
    }


def process_environment() -> dict[str, Any]:
    """The :func:`environment` of the calling process, read once and then reused."""
    global _PROCESS  # noqa: PLW0603 - one reading per process, by design
    if _PROCESS is None:
        _PROCESS = environment()
    return _PROCESS


def stamp() -> dict[str, Any]:
    """The compact stamp for one result row."""
    found = process_environment()
    return {
        "env": found["env"],
        "analysis": found["analysis"],
        "commit": str(found["commit"])[:12],
        "src_dirty": found["src_dirty"],
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def check_same_environment(stamps: Iterable[Mapping[str, Any] | None], what: str) -> None:
    """Refuse to add rows to ``what`` when rows from another environment are present.

    Prereg 0005 and 0006: no commit changes ``src/`` or ``uv.lock`` while a study runs;
    if one did, the study's rows are set aside and the study runs again in one
    environment, reported as a deviation.
    """
    mine = process_environment()["env"]
    others = sorted({str(s.get("env")) for s in stamps if s and s.get("env") != mine})
    if others:
        raise ValueError(
            f"{what}: rows from environment {', '.join(others)} exist and this process runs "
            f"{mine}; a study runs in one environment (prereg 0005 and 0006)"
        )


def single_analysis_code(stamps: Iterable[Mapping[str, Any] | None]) -> str | None:
    """The one analysis-code hash of a study's rows (None for rows without one)."""
    found = {s.get("analysis") for s in stamps if s}
    if len(found) > 1:
        raise ValueError(f"rows ran {len(found)} versions of the analysis code")
    value = next(iter(found)) if found else None
    return None if value is None else str(value)


def single_environment(
    stamps: Iterable[Mapping[str, Any] | None], *, allow_dirty: bool = False
) -> str | None:
    """The one environment hash of a study's rows; raises if they do not share one.

    Rows without a stamp (banked before stamps existed) are accepted only when no row has
    one, and then None is returned. A row from a ``src/`` with uncommitted changes is
    refused, because no commit reproduces it (``allow_dirty`` is for tests that run in a
    working tree under edit; banking never sets it).
    """
    seen = list(stamps)
    found = {s.get("env") if s else None for s in seen}
    if not allow_dirty and any(s and s.get("src_dirty") for s in seen):
        raise ValueError("rows ran from uncommitted changes under src/")
    if len(found) > 1:
        raise ValueError(f"rows ran in {len(found)} environments: {sorted(map(str, found))}")
    return next(iter(found)) if found else None
