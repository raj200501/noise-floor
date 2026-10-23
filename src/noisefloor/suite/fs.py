"""Mock in-memory file system for nf-tools episodes.

A file system is a mapping from absolute POSIX paths to text. Directories are
implied by the paths: ``/a/b.txt`` implies the directory ``/a``. There are no
empty directories other than the root. A :class:`Snapshot` is immutable and is
what a task ships; each episode works on its own :class:`WorkingCopy`, so runs
never share state. Nothing here touches the real file system.
"""

from __future__ import annotations

import posixpath
from collections.abc import Mapping
from types import MappingProxyType

ROOT = "/"


class FsError(ValueError):
    """A file system operation failed. The message is meant to be shown to the agent."""


def normalize_path(path: str) -> str:
    """Return the absolute, normalized form of ``path``.

    Relative paths resolve against the root, ``.`` and ``..`` are collapsed (``..``
    stops at the root), and repeated or trailing slashes are removed.
    """
    if "\x00" in path:
        raise FsError("path contains a NUL character")
    return posixpath.normpath(ROOT + path.lstrip("/"))


def _ancestors(path: str) -> list[str]:
    """Directories that contain ``path``, nearest first, excluding the root."""
    found = []
    parent = posixpath.dirname(path)
    while parent != ROOT:
        found.append(parent)
        parent = posixpath.dirname(parent)
    return found


def _validated(files: Mapping[str, str]) -> dict[str, str]:
    for path, content in files.items():
        if not isinstance(path, str) or not isinstance(content, str):
            raise TypeError("file system paths and contents must be strings")
        if path == ROOT or normalize_path(path) != path:
            raise FsError(f"not a normalized absolute file path: {path!r}")
    for path in files:
        for parent in _ancestors(path):
            if parent in files:
                raise FsError(f"{parent} is both a file and a directory")
    return {path: files[path] for path in sorted(files)}


class FileView:
    """Read-only operations shared by :class:`Snapshot` and :class:`WorkingCopy`."""

    _files: dict[str, str]

    def paths(self) -> tuple[str, ...]:
        """All file paths, sorted."""
        return tuple(sorted(self._files))

    def to_dict(self) -> dict[str, str]:
        """A sorted copy of the path to content mapping."""
        return {path: self._files[path] for path in sorted(self._files)}

    def get(self, path: str) -> str | None:
        """Content of the file at ``path``, or None if there is no such file."""
        return self._files.get(normalize_path(path))

    def is_file(self, path: str) -> bool:
        """True if ``path`` is a file."""
        return normalize_path(path) in self._files

    def is_dir(self, path: str) -> bool:
        """True if ``path`` is the root or a directory implied by some file path."""
        norm = normalize_path(path)
        if norm == ROOT:
            return True
        prefix = norm + "/"
        return any(key.startswith(prefix) for key in self._files)

    def read(self, path: str) -> str:
        """Content of the file at ``path``; raises :class:`FsError` if there is none."""
        norm = normalize_path(path)
        if norm in self._files:
            return self._files[norm]
        if self.is_dir(norm):
            raise FsError(f"is a directory: {norm}")
        raise FsError(f"no such file: {norm}")

    def list_dir(self, path: str) -> list[str]:
        """Sorted names of the entries of a directory; subdirectory names end with ``/``."""
        norm = normalize_path(path)
        if norm in self._files:
            raise FsError(f"not a directory: {norm}")
        if not self.is_dir(norm):
            raise FsError(f"no such directory: {norm}")
        prefix = ROOT if norm == ROOT else norm + "/"
        entries: set[str] = set()
        for key in self._files:
            if key.startswith(prefix):
                head, sep, _ = key[len(prefix) :].partition("/")
                entries.add(head + ("/" if sep else ""))
        return sorted(entries)

    def files_under(self, path: str) -> list[str]:
        """Sorted paths of every file below the directory ``path``, at any depth."""
        norm = normalize_path(path)
        prefix = ROOT if norm == ROOT else norm + "/"
        return sorted(key for key in self._files if key.startswith(prefix))

    def __len__(self) -> int:
        """Number of files."""
        return len(self._files)


class Snapshot(FileView):
    """Immutable file system state, as shipped with a task."""

    def __init__(self, files: Mapping[str, str] | None = None) -> None:
        """Validate and freeze ``files`` (paths must already be normalized and absolute)."""
        self._files = _validated(files or {})

    @property
    def files(self) -> Mapping[str, str]:
        """Read-only view of the path to content mapping, sorted by path."""
        return MappingProxyType(self._files)

    def working_copy(self) -> WorkingCopy:
        """A fresh mutable copy for one episode."""
        return WorkingCopy(self)

    def __eq__(self, other: object) -> bool:
        """Snapshots are equal when they hold the same files with the same contents."""
        if not isinstance(other, Snapshot):
            return NotImplemented
        return self._files == other._files

    def __hash__(self) -> int:
        """Hash of the sorted (path, content) pairs."""
        return hash(tuple(self._files.items()))

    def __repr__(self) -> str:
        """Short description with the file count."""
        return f"Snapshot({len(self._files)} files)"


class WorkingCopy(FileView):
    """Mutable copy of a snapshot, private to one episode."""

    def __init__(self, base: Snapshot) -> None:
        """Start from the files of ``base``; ``base`` itself never changes."""
        self._base = base
        self._files = dict(base.files)

    @property
    def base(self) -> Snapshot:
        """The snapshot this copy started from."""
        return self._base

    def write(self, path: str, content: str) -> str:
        """Create or replace a file and return its normalized path.

        Parent directories are implied, so they never need creating. Raises
        :class:`FsError` if ``path`` is the root or a directory, or if one of its
        parents is a file.
        """
        norm = normalize_path(path)
        if norm == ROOT:
            raise FsError("cannot write to the root directory")
        if self.is_dir(norm):
            raise FsError(f"is a directory: {norm}")
        for parent in _ancestors(norm):
            if parent in self._files:
                raise FsError(f"not a directory: {parent}")
        self._files[norm] = content
        return norm

    def snapshot(self) -> Snapshot:
        """Freeze the current state as a new snapshot."""
        return Snapshot(self._files)

    def changed_paths(self) -> list[str]:
        """Sorted paths of files that are new or differ from the base snapshot."""
        base = self._base.files
        return sorted(path for path, text in self._files.items() if base.get(path) != text)
