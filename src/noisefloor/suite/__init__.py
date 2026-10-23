"""nf-tools: procedurally generated, deterministic tool-use tasks over a mock file system.

Every task is mock data generated from a (family, seed, difficulty) triple. The
tools are in-process Python functions over an in-memory file system; nothing
executes model-generated code and nothing touches the real file system or the
network. Results from these tasks are model results on synthetic tasks and are
labeled as such.

Modules: :mod:`~noisefloor.suite.fs` (mock file system),
:mod:`~noisefloor.suite.tools` (tools and the episode environment),
:mod:`~noisefloor.suite.tasks` (task families),
:mod:`~noisefloor.suite.grading` (graders and episode status), and
:mod:`~noisefloor.suite.suite` (suites, seed splits, manifests).
"""

from __future__ import annotations

from noisefloor.suite.fs import FsError, Snapshot, WorkingCopy, normalize_path
from noisefloor.suite.grading import (
    AnswerSpec,
    FileSpec,
    Grade,
    episode_status,
    grade,
    normalize_text,
    parse_number,
)
from noisefloor.suite.suite import (
    SUITE_NAME,
    SUITE_VERSION,
    Split,
    build_suite,
    canonical_json,
    check_disjoint,
    split,
    suite_manifest,
    task_sha256,
)
from noisefloor.suite.tasks import DIFFICULTIES, FAMILIES, FAMILY_NAMES, Family, Task, generate
from noisefloor.suite.tools import TOOLS, CalcError, ToolEnv, ToolSpec, evaluate_arithmetic

__all__ = [
    "DIFFICULTIES",
    "FAMILIES",
    "FAMILY_NAMES",
    "SUITE_NAME",
    "SUITE_VERSION",
    "TOOLS",
    "AnswerSpec",
    "CalcError",
    "Family",
    "FileSpec",
    "FsError",
    "Grade",
    "Snapshot",
    "Split",
    "Task",
    "ToolEnv",
    "ToolSpec",
    "WorkingCopy",
    "build_suite",
    "canonical_json",
    "check_disjoint",
    "episode_status",
    "evaluate_arithmetic",
    "generate",
    "grade",
    "normalize_path",
    "normalize_text",
    "parse_number",
    "split",
    "suite_manifest",
    "task_sha256",
]
