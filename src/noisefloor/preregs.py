"""Reading the machine-readable blocks of preregistration files.

A block is a fenced code block whose info string is ``json <name>``; its content is JSON.
Tools that run confirmatory work take their settings only from these blocks, so what runs
is what was frozen.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
CHAIN = REPO / "prereg" / "CHAIN.jsonl"


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


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def amends(path: Path) -> str | None:
    """The four-digit number in the file's ``amends:`` front matter, if it has one."""
    match = re.match(r"---\n(.*?)\n---\n", path.read_text(), re.S)
    if match is None:
        return None
    found = re.search(r"^amends:\s*(\d{4})\s*$", match.group(1), re.M)
    return found.group(1) if found else None


def chain_entry(path: Path, chain: Path = CHAIN) -> dict[str, Any] | None:
    """The prereg/CHAIN.jsonl entry of a preregistration file, if it is frozen."""
    if not chain.exists():
        return None
    rel = repo_path(path)
    for line in chain.read_text().splitlines():
        if line.strip():
            entry: dict[str, Any] = json.loads(line)
            if entry.get("file") == rel:
                return entry
    return None


def _committed(rel: str, repo: Path) -> bytes | None:
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"HEAD:{rel}"], capture_output=True, check=False
    )
    return proc.stdout if proc.returncode == 0 else None


def require_frozen(path: Path, chain: Path = CHAIN, repo: Path = REPO) -> dict[str, Any]:
    """The chain entry of a preregistration that is frozen, unchanged, and committed.

    Every confirmatory runner calls this first, so nothing confirmatory runs on a draft:
    the file must have a chain entry, its bytes must match that entry's SHA-256, and both
    the file and the entry must be in the HEAD commit. Raises ``ValueError`` otherwise.
    """
    entry = chain_entry(path, chain)
    rel = repo_path(path)
    if entry is None:
        raise ValueError(f"{rel} is not frozen in prereg/CHAIN.jsonl")
    if file_sha256(path) != entry["sha256"]:
        raise ValueError(f"{rel} changed after it was frozen")
    if _committed(rel, repo) != path.read_bytes():
        raise ValueError(f"{rel} is not committed as frozen")
    head_chain = _committed(repo_path(chain), repo) or b""
    if json.dumps(entry, sort_keys=True, separators=(",", ":")).encode() not in head_chain:
        raise ValueError(f"the chain entry of {rel} is not committed")
    return entry
