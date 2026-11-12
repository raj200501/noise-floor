#!/usr/bin/env python3
"""Stage banked row files for a commit, refusing a cut last line (DECISIONS.md D-043).

``python3 tools/stage_rows.py PATH...`` (or ``--changed``, for every modified or new file under
``artifacts/e0/raw``, ``artifacts/g6``, ``artifacts/e1``, and ``artifacts/usage``) checks each
file and then runs ``git add`` on all of them at once, or on none:

* E0 and G6 raw rows (``artifacts/e0/raw/*.jsonl``, ``artifacts/g6/raw.jsonl``): every line must
  be a JSON object and the file must end in a newline. A study may be appending; retry later.
* The E1 arms' records (``artifacts/e1/local.jsonl``, ``artifacts/e1/claude.jsonl``) and the
  usage ledger (``artifacts/usage/ledger.jsonl``): only the last byte is read, and it must be a
  newline. Prereg 0004 lets only the runners read the arms' records before both arms end, so no
  line of them is parsed (D-037 item 1, D-043 item 11).

Any other path (an ``environments.json`` that changed, a summary, a document) is refused: those
are staged by hand, by path. Standard library only. Exit 0 when staged (or nothing to stage),
1 when anything was refused.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BYTES_ONLY = (
    "artifacts/e1/local.jsonl",
    "artifacts/e1/claude.jsonl",
    "artifacts/usage/ledger.jsonl",
)
WATCHED = ("artifacts/e0/raw", "artifacts/g6", "artifacts/e1", "artifacts/usage")


def _parsed_kind(rel: str) -> bool:
    return (rel.startswith("artifacts/e0/raw/") and rel.endswith(".jsonl")) or (
        rel == "artifacts/g6/raw.jsonl"
    )


def check(rel: str, repo: Path = REPO) -> tuple[str | None, str]:
    """(why the file cannot be staged now, or None; what was checked)."""
    path = repo / rel
    if rel not in BYTES_ONLY and not _parsed_kind(rel):
        return f"{rel}: not a row file this tool stages (stage it by hand, by path)", ""
    if not path.is_file():
        return f"{rel}: no such file", ""
    if rel in BYTES_ONLY:
        size = path.stat().st_size
        if size == 0:
            return None, f"{rel}: empty"
        with path.open("rb") as fh:
            fh.seek(-1, os.SEEK_END)
            last = fh.read(1)
        if last != b"\n":
            return f"{rel}: last line incomplete (a writer may be appending; retry)", ""
        return None, f"{rel}: {size} bytes, ends in a newline (records not read)"
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        return f"{rel}: last line incomplete (a study may be appending; retry)", ""
    rows = 0
    for number, line in enumerate(data.decode().split("\n")[:-1], 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            row = None
        if not isinstance(row, dict):
            return f"{rel}: line {number} is not a JSON object", ""
        rows += 1
    return None, f"{rel}: {rows} rows"


def changed(repo: Path = REPO) -> list[str]:
    """Modified or new files under the watched directories (``git status``)."""
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *WATCHED],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(line[3:] for line in proc.stdout.splitlines() if line[3:])


def main(argv: Sequence[str], repo: Path = REPO) -> int:
    paths = changed(repo) if list(argv) == ["--changed"] else list(argv)
    if not paths:
        print("STAGE-ROWS: nothing to stage")
        return 0
    problems, notes = [], []
    for rel in paths:
        problem, note = check(rel, repo)
        if problem:
            problems.append(problem)
        else:
            notes.append(note)
    for line in problems:
        print(f"refused {line}")
    if problems:
        print(f"STAGE-ROWS: FAIL ({len(problems)} refused; nothing staged)")
        return 1
    subprocess.run(["git", "add", "--", *paths], cwd=repo, check=True)
    for line in notes:
        print(f"staged {line}")
    print(f"STAGE-ROWS: staged {len(paths)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
