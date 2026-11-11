"""Designs where an evalstats interval failed its coverage check (experiment E0).

E0 (``prereg/0003``) simulates every interval method in a grid of designs. A cell
fails when its coverage falls below ``nominal - 0.015 - 2 * MCSE``. The failing cells
are listed in ``evalstats/data/validity.json`` (generated from
``artifacts/e0/coverage.json`` by ``python -m noisefloor.e0_validity``). When a call
uses a method in a design at least as small as a failing cell's, evalstats emits an
:class:`UnsupportedDesignWarning` that names the cell and a supported alternative.
"""

from __future__ import annotations

import json
import warnings
from importlib import resources
from typing import Any


class UnsupportedDesignWarning(UserWarning):
    """The method under-covered in a simulated design like this one."""


def _load() -> dict[str, Any]:
    text = resources.files("evalstats").joinpath("data/validity.json").read_text()
    data: dict[str, Any] = json.loads(text)
    return data


_DATA = _load()
UNSUPPORTED_CELLS: frozenset[str] = frozenset(_DATA.get("unsupported", {}))


def unsupported_rules() -> dict[str, dict[str, Any]]:
    """The failing cells and the designs they cover."""
    return {k: dict(v) for k, v in _DATA.get("unsupported", {}).items()}


def check(function: str, method: str, n_tasks: int) -> list[str]:
    """Warn if ``function`` with ``method`` on ``n_tasks`` tasks matches a failing cell.

    Returns the ids of the matching cells.
    """
    hits = []
    for cell, rule in _DATA.get("unsupported", {}).items():
        if (
            rule["function"] == function
            and rule["method"] == method
            and n_tasks <= int(rule["max_tasks"])
        ):
            hits.append(cell)
            warnings.warn(
                f"{function}(method={method!r}) under-covered in simulation with "
                f"{rule['max_tasks']} tasks (E0 cell {cell}: coverage {rule['coverage']:.3f} at "
                f"nominal {rule['nominal']}); prefer {rule['alternative']}",
                UnsupportedDesignWarning,
                stacklevel=3,
            )
    return hits
