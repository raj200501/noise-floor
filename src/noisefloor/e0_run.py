"""Run E0 cells from frozen grids and bank ``artifacts/e0/coverage.json`` (gate G5).

``python -m noisefloor.e0_run run PREREG [--cells ID ...] [--workers N]`` runs every
missing replicate of the grid's cells in worker processes and appends one JSON line per
replicate to ``artifacts/e0/raw/<cell>.jsonl`` (so an interrupted run resumes).
``python -m noisefloor.e0_run bank PREREG [AMENDMENT ...]`` summarizes the raw files
of every cell in the base grid and its amendments into ``coverage.json``. Grids come
from the ```json e0-grid``` blocks of the preregistration files.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import multiprocessing as mp
import os
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from noisefloor.e0 import Cell, Replicate, cells_from_grid, run_replicate, summarize

REPO = Path(__file__).resolve().parents[2]
E0_DIR = REPO / "artifacts" / "e0"
FENCE = re.compile(r"```json[ \t]+e0-grid[ \t]*\n(.*?)\n```", re.S)


def grid_from_prereg(path: Path) -> list[Cell]:
    """The cells of the ```json e0-grid``` block in a preregistration file."""
    match = FENCE.search(path.read_text())
    if match is None:
        raise ValueError(f"{path}: no ```json e0-grid``` block")
    return cells_from_grid(json.loads(match.group(1)))


def raw_path(cell: Cell, root: Path = E0_DIR) -> Path:
    """Where a cell's replicate lines are banked."""
    return root / "raw" / f"{cell.cell}.jsonl"


def read_raw(cell: Cell, root: Path = E0_DIR) -> dict[int, Replicate]:
    """Banked replicates of a cell, by index (later lines win for a repeated index)."""
    path = raw_path(cell, root)
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[int(row["i"])] = Replicate(
                truth=_num(row["truth"]),
                lo=_num(row["lo"]),
                hi=_num(row["hi"]),
                estimate=_num(row["estimate"]),
                hit=row.get("hit"),
            )
    return out


def _num(value: Any) -> float:
    return float("nan") if value is None else float(value)


def _json_num(value: float) -> float | None:
    return value if math.isfinite(value) else None  # NaN and infinities become null


def _work(args: tuple[dict[str, Any], int]) -> tuple[int, dict[str, Any]]:
    spec, i = args
    cell = Cell(**spec)
    rep = run_replicate(cell, i)
    return i, {
        "i": i,
        "truth": _json_num(rep.truth),
        "lo": _json_num(rep.lo),
        "hi": _json_num(rep.hi),
        "estimate": _json_num(rep.estimate),
        "hit": rep.hit,
    }


def run_cell(cell: Cell, workers: int = 1, root: Path = E0_DIR, log: Any = print) -> int:
    """Run the cell's missing replicates; returns how many were added."""
    done = read_raw(cell, root)
    todo = [i for i in range(cell.replicates) if i not in done]
    if not todo:
        return 0
    path = raw_path(cell, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = {
        "cell": cell.cell,
        "method": cell.method,
        "interval_type": cell.interval_type,
        "nominal": cell.nominal,
        "replicates": cell.replicates,
        "params": dict(cell.params),
    }
    jobs = [(spec, i) for i in todo]
    added = 0
    with path.open("a") as fh:
        if workers <= 1:
            results: Iterable[tuple[int, dict[str, Any]]] = map(_work, jobs)
            for _, row in results:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                added += 1
        else:
            with mp.get_context("spawn").Pool(workers, initializer=_lower_priority) as pool:
                for _, row in pool.imap_unordered(_work, jobs, chunksize=4):
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                    fh.flush()
                    added += 1
    log(f"{cell.cell}: {added} replicates added ({len(done) + added}/{cell.replicates})")
    return added


def lower_priority(autogroup: Path = Path("/proc/self/autogroup")) -> None:
    """Lowest CPU priority for this process.

    Sets nice 10 and, where the kernel groups jobs into scheduler autogroups (in which
    ``nice`` only orders processes within a group, D-024), the group's own nice to 19. A
    failure to set the group is ignored.
    """
    os.nice(10)
    with contextlib.suppress(OSError):
        autogroup.write_text("19\n")


_lower_priority = lower_priority


def bank(preregs: Sequence[Path], root: Path = E0_DIR) -> dict[str, Any]:
    """Summarize every cell of the base grid (first path) and its amendments."""
    cells: list[Cell] = []
    for path in preregs:
        cells.extend(grid_from_prereg(path))
    ids = [c.cell for c in cells]
    if len(set(ids)) != len(ids):
        raise ValueError("cell ids repeat across grids")
    summaries = []
    for cell in cells:
        reps = read_raw(cell, root)
        ordered = [reps[i] for i in sorted(reps) if i < cell.replicates]
        summaries.append(summarize(cell, ordered))
    artifact = {
        "prereg": str(preregs[0].resolve().relative_to(REPO)),
        "amendments": [str(p.resolve().relative_to(REPO)) for p in preregs[1:]],
        "confirmatory": True,
        "generator": "python -m noisefloor.e0_run bank "
        + " ".join(str(p.resolve().relative_to(REPO)) for p in preregs),
        "raw": "artifacts/e0/raw/<cell>.jsonl",
        "cells": summaries,
    }
    (root / "coverage.json").write_text(
        json.dumps(_clean(artifact), indent=1, sort_keys=True, allow_nan=False) + "\n"
    )
    return artifact


def _clean(value: Any) -> Any:
    """NaN and infinities become None, recursively (strict JSON)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="python -m noisefloor.e0_run")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("prereg", type=Path)
    r.add_argument("--cells", nargs="*")
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    b = sub.add_parser("bank")
    b.add_argument("preregs", type=Path, nargs="+")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "run":
        for cell in grid_from_prereg(args.prereg):
            if args.cells and cell.cell not in args.cells:
                continue
            if "group" in cell.params:
                # E1-shaped cells share one simulated dataset (and one fit) per replicate.
                print(f"{cell.cell}: skipped; run it with python -m noisefloor.recovery e0-run")
                continue
            run_cell(cell, args.workers, log=lambda m: print(m, flush=True))
        return 0
    artifact = bank(args.preregs)
    print(f"banked {len(artifact['cells'])} cells in artifacts/e0/coverage.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
