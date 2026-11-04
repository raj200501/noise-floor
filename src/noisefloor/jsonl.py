"""JSON Lines files that survive a hard kill.

Every record is appended with one ``write`` call on a file opened with ``O_APPEND``, so a
kill can cut at most the last record. Readers either reject any line that does not parse
(:func:`read`) or skip a cut final line (``tail_ok``), and :func:`repair_tail` removes a
cut final line before more records are appended after it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def append(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Append ``rows`` to ``path``, one ``write`` per record; returns the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    count = 0
    try:
        for row in rows:
            data = (json.dumps(row, sort_keys=True) + "\n").encode()
            while data:
                data = data[os.write(fd, data) :]
            count += 1
    finally:
        os.close(fd)
    return count


def read(path: Path, *, tail_ok: bool = False) -> list[dict[str, Any]]:
    """The records of ``path`` (an empty list if it does not exist).

    A line that does not parse raises ``ValueError`` naming the file and line, except,
    with ``tail_ok``, a final line without its newline (a record cut by a kill), which is
    left out.
    """
    if not path.exists():
        return []
    text = path.read_text()
    lines = text.split("\n")
    rows = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            if tail_ok and number == len(lines):
                break
            raise ValueError(f"{path}:{number}: not a JSON record") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: not a JSON object")
        rows.append(row)
    return rows


def repair_tail(path: Path) -> str | None:
    """Remove a cut final line (no newline, does not parse); returns the text removed.

    A final line that parses but lacks its newline gets the newline instead, and None is
    returned, as it is when the file is missing or already ends cleanly.
    """
    if not path.exists():
        return None
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return None
    cut = data.rfind(b"\n") + 1
    tail = data[cut:]
    try:
        json.loads(tail)
    except ValueError:
        with path.open("r+b") as fh:
            fh.truncate(cut)
        return tail.decode(errors="replace")
    with path.open("ab") as fh:
        fh.write(b"\n")
    return None
