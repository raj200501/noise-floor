"""Reading the machine-readable blocks of preregistration files.

A block is a fenced code block whose info string is ``json <name>``; its content is JSON.
Tools that run confirmatory work take their settings only from these blocks, so what runs
is what was frozen.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]


def prereg_block(path: Path, name: str) -> Any:
    """The JSON of the ```json <name>``` block of a preregistration file."""
    pattern = rf"```json[ \t]+{re.escape(name)}[ \t]*\n(.*?)\n```"
    match = re.search(pattern, path.read_text(), re.S)
    if match is None:
        raise ValueError(f"{path}: no ```json {name}``` block")
    return json.loads(match.group(1))


def repo_path(path: Path) -> str:
    """``path`` relative to the repository (POSIX), or as given if it lies outside."""
    absolute = path.resolve()
    return absolute.relative_to(REPO).as_posix() if absolute.is_relative_to(REPO) else str(path)
