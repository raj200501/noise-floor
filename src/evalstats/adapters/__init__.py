"""Readers that turn benchmark harness outputs into :class:`~evalstats.schema.EvalResults`.

Each reader parses the harness's files directly (standard library and pandas only; the harness
packages are not imported) and adds a ``status`` covariate with one of :data:`STATUSES`, which
the policy-sensitivity bounds use. Each module docstring documents its status mapping.
"""

from __future__ import annotations

from evalstats.adapters._common import STATUSES
from evalstats.adapters.harbor import read_harbor
from evalstats.adapters.inspect import read_inspect
from evalstats.adapters.swebench import read_swebench

__all__ = ["STATUSES", "read_harbor", "read_inspect", "read_swebench"]
