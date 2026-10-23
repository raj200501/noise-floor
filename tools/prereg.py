#!/usr/bin/env python3
# Adapted from raj200501/raise-v1-public tools/prereg.py (Apache-2.0); changes: preregistrations
# are Markdown files (prereg/NNNN-slug.md) hashed whole instead of JSON documents with sealed
# fields; the chain file is prereg/CHAIN.jsonl with the fields seq, file, sha256, prev, frozen_at,
# entry_hash; the beacon anchors, reader-script hashing, and status artifact were dropped; verify
# adds git-history checks (one commit per prereg, append-only chain), confirmatory-artifact
# links, amendment citations, fenced JSON blocks, and the brief hash; merged chainlib helpers in.
"""prereg: hash-chained preregistrations (docs/BUILD_BRIEF.md section 9 rule 2, gate G11).

Commands::

    freeze prereg/NNNN-slug.md   append the file to prereg/CHAIN.jsonl
    verify                       run checks (a) to (g) below; what `make prereg-verify` runs
    show                         print the chain and the next unused number

Chain entries are one canonical JSON object per line (sorted keys, no spaces, UTF-8)::

    {"entry_hash": ..., "file": "prereg/0001-slug.md", "frozen_at": "2026-10-23T01:51:00Z",
     "prev": <entry_hash of the previous entry, or 64 zeros>, "seq": 1, "sha256": <file hash>}

entry_hash is the SHA-256 of the canonical JSON of the entry without entry_hash. The number in
the file name equals seq, so numbers are gapless and in freeze order.

verify checks:

(a) chain: every line is a canonical entry, seq runs 1..n, prev links and entry hashes hold.
(b) files: every prereg/NNNN-*.md has exactly one entry with a matching sha256, and every
    entry's file exists.
(c) history: every prereg file ever committed is touched by exactly one commit, its committed
    content matches the chain, and each committed version of prereg/CHAIN.jsonl (then the
    working copy) is a prefix of the next. A commit touches a path when the path's content
    differs from every parent, so an ordinary merge does not count as a second touch.
(d) artifacts: every JSON file under artifacts/ with a top-level "prereg" field, and every row of
    an artifacts/**/*.jsonl file with a "prereg" field, cites a file in the chain. Both that
    file's commit and the commit that added its chain entry are strict ancestors of every commit
    that introduced the artifact file. An artifact that is not committed yet passes when the
    prereg is committed, because its first commit will descend from HEAD. A record with
    "confirmatory": true must cite a prereg.
(e) amendments: a prereg whose front matter has `amends: NNNN` cites an earlier prereg in the
    chain and has a `date: YYYY-MM-DD` line. Front matter is the block between a first line
    `---` and the next `---`.
(f) json blocks: every fenced ```json``` block in a prereg parses, and tags are unique per file.
(g) brief: docs/BUILD_BRIEF.md matches docs/BUILD_BRIEF.sha256, and neither has more than one
    committed version.

Output is one line per check, problems indented below a failing check, then
`PREREG-VERIFY: PASS` or `PREREG-VERIFY: FAIL <n> problems`. Exit 0 only on PASS.
Standard library only (Python 3.11+).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

REPO = Path(__file__).resolve().parent.parent
PREREG_DIR = "prereg"
CHAIN = "prereg/CHAIN.jsonl"
BRIEF = "docs/BUILD_BRIEF.md"
BRIEF_HASH = "docs/BUILD_BRIEF.sha256"
ARTIFACTS = "artifacts"
GENESIS = "0" * 64
ENTRY_KEYS = frozenset({"seq", "file", "sha256", "prev", "frozen_at", "entry_hash"})
# What counts as a preregistration (the same glob ship_check uses: [0-9][0-9][0-9][0-9]-*.md).
PREREG_NAME = re.compile(r"^([0-9]{4})-(.+)\.md$")
# What freeze accepts as a slug.
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FROZEN_AT = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# The exact pattern tools/ship_check.py uses to read tagged blocks.
FENCE_RE = re.compile(r"```json[ \t]+([\w-]+)[ \t]*\n(.*?)\n```", re.S)
FENCE_LINE = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*)$")
FRONT_KEY = re.compile(r"^([A-Za-z_][\w-]*)[ \t]*:[ \t]*(.*?)[ \t]*$")
AMENDS_LINE = re.compile(r"^[ \t]*amends[ \t]*:", re.I)


class PreregError(Exception):
    """A command cannot proceed; the message says why."""


# --------------------------------------------------------------------------- hashing


def canonical(obj: Any) -> str:
    """Deterministic JSON text: sorted keys, no whitespace, no NaN, UTF-8 characters kept."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def entry_hash(entry: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON of the entry without its entry_hash field."""
    body = {k: v for k, v in entry.items() if k != "entry_hash"}
    return sha256_bytes(canonical(body).encode("utf-8"))


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- chain


@dataclass
class Chain:
    entries: list[dict[str, Any]]
    lines: list[str]
    problems: list[str]


def read_chain(root: Path) -> Chain:
    """Parse prereg/CHAIN.jsonl. A missing file is an empty chain."""
    path = root / CHAIN
    if not path.exists():
        return Chain([], [], [])
    problems: list[str] = []
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return Chain([], [], [f"{CHAIN} is not UTF-8"])
    if text and not text.endswith("\n"):
        problems.append(f"{CHAIN} does not end with a newline")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    entries: list[dict[str, Any]] = []
    for n, line in enumerate(lines, 1):
        try:
            obj = json.loads(line)
        except ValueError:
            problems.append(f"{CHAIN} line {n} is not JSON")
            obj = {}
        if not isinstance(obj, dict):
            problems.append(f"{CHAIN} line {n} is not a JSON object")
            obj = {}
        entries.append(obj)
    return Chain(entries, lines, problems)


def check_chain(chain: Chain) -> list[str]:
    """Check (a): canonical lines, seq, prev links, entry hashes, file names, timestamps."""
    problems = list(chain.problems)
    prev = GENESIS
    last_time = ""
    seen: set[str] = set()
    for i, (entry, line) in enumerate(zip(chain.entries, chain.lines, strict=True), 1):
        if not entry:
            prev = ""
            continue
        label = f"entry {i}"
        keys = set(entry)
        if keys != ENTRY_KEYS:
            missing = sorted(ENTRY_KEYS - keys)
            extra = sorted(keys - ENTRY_KEYS)
            problems.append(f"{label}: wrong fields (missing {missing}, extra {extra})")
        seq = entry.get("seq")
        if type(seq) is not int or seq != i:
            problems.append(f"{label}: seq is {seq!r}, expected {i}")
        fname = entry.get("file")
        if not isinstance(fname, str):
            problems.append(f"{label}: file is not a string")
        else:
            label = f"entry {i} ({fname})"
            p = PurePosixPath(fname)
            m = PREREG_NAME.match(p.name)
            if str(p.parent) != PREREG_DIR or m is None:
                problems.append(f"{label}: file is not prereg/NNNN-<slug>.md")
            elif int(m.group(1)) != i:
                problems.append(f"{label}: file number {m.group(1)} does not equal seq {i}")
            if fname in seen:
                problems.append(f"{label}: file appears twice in the chain")
            seen.add(fname)
        if not isinstance(entry.get("sha256"), str) or not HEX64.match(entry["sha256"]):
            problems.append(f"{label}: sha256 is not 64 lowercase hex characters")
        if entry.get("prev") != prev:
            problems.append(f"{label}: prev does not equal the previous entry_hash (chain broken)")
        stamp = entry.get("frozen_at")
        if not isinstance(stamp, str) or not FROZEN_AT.match(stamp):
            problems.append(f"{label}: frozen_at is not UTC ISO (YYYY-MM-DDTHH:MM:SSZ)")
        else:
            try:
                dt.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                problems.append(f"{label}: frozen_at is not a valid time")
            if stamp < last_time:
                problems.append(f"{label}: frozen_at is earlier than the previous entry")
            last_time = max(last_time, stamp)
        recorded = entry.get("entry_hash")
        try:
            computed = entry_hash(entry)
        except ValueError:
            computed = ""
        if recorded != computed:
            problems.append(f"{label}: entry_hash does not match the entry (entry was edited)")
        elif line != canonical(entry):
            problems.append(f"{label}: line is not in canonical JSON form")
        prev = recorded if isinstance(recorded, str) else ""
    return problems


# --------------------------------------------------------------------------- prereg text


@dataclass
class FrontMatter:
    fields: dict[str, str]
    end_line: int  # index of the first line after the front matter (0 if none)
    problems: list[str]


def front_matter(text: str) -> FrontMatter:
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return FrontMatter({}, 0, [])
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            fields: dict[str, str] = {}
            problems: list[str] = []
            for ln in lines[1:i]:
                m = FRONT_KEY.match(ln)
                if m is None:
                    if ln.strip():
                        problems.append(f"front matter line is not 'key: value': {ln.strip()!r}")
                    continue
                key = m.group(1).lower()
                if key in fields:
                    problems.append(f"front matter key {key!r} appears twice")
                fields[key] = m.group(2)
            return FrontMatter(fields, i + 1, problems)
    return FrontMatter({}, 0, ["front matter starts with --- but is never closed"])


def prereg_number(file: str) -> int | None:
    m = PREREG_NAME.match(PurePosixPath(file).name)
    return int(m.group(1)) if m else None


def json_block_problems(text: str) -> tuple[list[str], int]:
    """Check (f) for one file: every ```json block parses, tags unique. Returns (problems, n)."""
    problems: list[str] = []
    blocks = 0
    tags: dict[str, int] = {}
    seen_bodies: set[str] = set()
    for m in FENCE_RE.finditer(text):
        tags[m.group(1)] = tags.get(m.group(1), 0) + 1
        seen_bodies.add(m.group(2))
        try:
            json.loads(m.group(2))
        except ValueError as exc:
            problems.append(f"```json {m.group(1)}``` block is not valid JSON ({exc.msg})")
    for tag, count in sorted(tags.items()):
        if count > 1:
            problems.append(f"```json {tag}``` block appears {count} times (tags must be unique)")
    # A line-based pass also catches blocks the regex above skips (untagged ```json, ~~~json,
    # longer fences). Blocks the regex already parsed are not reported twice.
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        m = FENCE_LINE.match(lines[i])
        if m is None:
            i += 1
            continue
        fence, info = m.group(1), m.group(2).strip()
        j = i + 1
        while j < len(lines):
            close = FENCE_LINE.match(lines[j])
            if (
                close
                and close.group(1)[0] == fence[0]
                and len(close.group(1)) >= len(fence)
                and not close.group(2).strip()
            ):
                break
            j += 1
        words = info.split()
        if words and words[0].lower() == "json":
            blocks += 1
            body = "\n".join(lines[i + 1 : j])
            if j >= len(lines):
                problems.append(f"```json block at line {i + 1} is never closed")
            elif body not in seen_bodies:
                try:
                    json.loads(body)
                except ValueError as exc:
                    problems.append(f"```json block at line {i + 1} is not valid JSON ({exc.msg})")
        i = j + 1
    return problems, blocks


def amendment_problems(
    rel: str, text: str, frozen_numbers: Iterable[int]
) -> tuple[list[str], bool]:
    """Check (e) for one file. Returns (problems, is_amendment)."""
    fm = front_matter(text)
    problems = list(fm.problems)
    lines = text.split("\n")
    for n, ln in enumerate(lines[fm.end_line :], fm.end_line + 1):
        if AMENDS_LINE.match(ln):
            problems.append(f"line {n}: 'amends:' outside the front matter is ambiguous")
    if "amends" not in fm.fields:
        return problems, False
    own = prereg_number(rel) or 0
    value = fm.fields["amends"].strip()
    if not re.fullmatch(r"[0-9]{4}", value):
        problems.append(f"amends: {value!r} is not a four-digit prereg number")
    else:
        target = int(value)
        if target >= own:
            problems.append(f"amends: {value} is not an earlier preregistration than {own:04d}")
        elif target not in set(frozen_numbers):
            problems.append(f"amends: {value} names no preregistration in the chain")
    date = fm.fields.get("date", "").strip()
    if not ISO_DATE.match(date):
        problems.append("an amendment needs a 'date: YYYY-MM-DD' front matter line")
    else:
        try:
            dt.date.fromisoformat(date)
        except ValueError:
            problems.append(f"date: {date} is not a valid date")
    return problems, True


# --------------------------------------------------------------------------- git


def git(root: Path, *args: str, stdin: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=root, input=stdin, capture_output=True, check=False)


@dataclass
class History:
    """The commit graph reachable from HEAD and per-commit blob ids of selected paths."""

    head: str
    commits: list[str]
    parents: dict[str, list[str]]
    blobs: dict[str, dict[str, str | None]] = field(default_factory=dict)
    contents: dict[str, bytes] = field(default_factory=dict)

    def touches(self, path: str) -> list[str]:
        """Commits whose version of path differs from every parent's version."""
        bl = self.blobs[path]
        out = []
        for c in self.commits:
            b = bl.get(c)
            ps = self.parents[c]
            if (not ps and b is not None) or (ps and all(bl.get(p) != b for p in ps)):
                out.append(c)
        return out

    def introductions(self, path: str) -> list[str]:
        """Commits that contain path while none of their parents do."""
        bl = self.blobs[path]
        return [
            c
            for c in self.commits
            if bl.get(c) is not None and all(bl.get(p) is None for p in self.parents[c])
        ]

    def versions(self, path: str) -> set[str]:
        return {b for b in self.blobs[path].values() if b is not None}

    def at_head(self, path: str) -> str | None:
        return self.blobs[path].get(self.head) if self.head else None

    def strictly_before(self, earlier: str, later: str) -> bool:
        if earlier == later:
            return False
        stack = list(self.parents.get(later, []))
        seen: set[str] = set()
        while stack:
            c = stack.pop()
            if c == earlier:
                return True
            if c not in seen:
                seen.add(c)
                stack.extend(self.parents.get(c, []))
        return False

    def content(self, blob: str | None) -> bytes:
        """Bytes of a blob loaded with load_contents; a missing path or a non-blob is empty."""
        if blob is None or ":" in blob:
            return b""
        return self.contents[blob]


def load_history(root: Path) -> tuple[History | None, list[str]]:
    inside = git(root, "rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != b"true":
        return None, [f"{root} is not a git repository"]
    top = git(root, "rev-parse", "--show-toplevel").stdout.decode().strip()
    if not top or Path(top).resolve() != root.resolve():
        return None, [f"{root} is not the top of its git work tree ({top})"]
    problems: list[str] = []
    shallow = git(root, "rev-parse", "--is-shallow-repository").stdout.strip()
    if shallow == b"true":
        problems.append(
            "shallow clone: history checks need the full history (git fetch --unshallow)"
        )
    head = git(root, "rev-parse", "--verify", "-q", "HEAD")
    if head.returncode != 0:
        return History("", [], {}), problems
    out = git(root, "rev-list", "--parents", "HEAD")
    if out.returncode != 0:
        return None, [*problems, "git rev-list failed"]
    commits: list[str] = []
    parents: dict[str, list[str]] = {}
    for line in out.stdout.decode().splitlines():
        parts = line.split()
        commits.append(parts[0])
        parents[parts[0]] = parts[1:]
    return History(head.stdout.decode().strip(), commits, parents), problems


def lookup_blobs(root: Path, hist: History, paths: Sequence[str]) -> None:
    """Fill hist.blobs for paths with one git cat-file call."""
    todo = [p for p in dict.fromkeys(paths) if p not in hist.blobs]
    for p in todo:
        hist.blobs[p] = {}
    if not todo or not hist.commits:
        return
    names = [f"{c}:{p}" for p in todo for c in hist.commits]
    proc = git(
        root,
        "cat-file",
        "--batch-check=%(objectname) %(objecttype)",
        stdin=("\n".join(names) + "\n").encode(),
    )
    rows = proc.stdout.decode().splitlines()
    if proc.returncode != 0 or len(rows) != len(names):
        raise PreregError("git cat-file --batch-check failed")
    k = 0
    for p in todo:
        for c in hist.commits:
            row = rows[k]
            k += 1
            if row.endswith(" missing") or " " not in row:
                continue
            sha, kind = row.split(" ", 1)
            hist.blobs[p][c] = sha if kind == "blob" else f"{kind}:{sha}"


def load_contents(root: Path, hist: History, blobs: Iterable[str | None]) -> None:
    todo = sorted({b for b in blobs if b is not None and b not in hist.contents})
    todo = [b for b in todo if ":" not in b]
    if not todo:
        return
    proc = git(root, "cat-file", "--batch", stdin=("\n".join(todo) + "\n").encode())
    if proc.returncode != 0:
        raise PreregError("git cat-file --batch failed")
    data = proc.stdout
    pos = 0
    for sha in todo:
        nl = data.index(b"\n", pos)
        header = data[pos:nl].decode().split()
        size = int(header[2])
        hist.contents[sha] = data[nl + 1 : nl + 1 + size]
        pos = nl + 1 + size + 1


def ever_committed_preregs(root: Path) -> list[str]:
    """Every path prereg/NNNN-*.md that appears in any commit reachable from HEAD."""
    out = git(
        root,
        "log",
        "--format=",
        "--name-only",
        "--no-renames",
        "-m",
        "--full-history",
        "HEAD",
        "--",
        PREREG_DIR,
    )
    if out.returncode != 0:
        return []
    paths = set()
    for line in out.stdout.decode().splitlines():
        p = PurePosixPath(line.strip())
        if str(p.parent) == PREREG_DIR and PREREG_NAME.match(p.name):
            paths.add(str(p))
    return sorted(paths)


# --------------------------------------------------------------------------- verify


@dataclass
class Check:
    key: str
    name: str
    summary: str = ""
    problems: list[str] = field(default_factory=list)


def short(sha: str) -> str:
    return sha[:12]


def disk_preregs(root: Path) -> tuple[list[str], list[str]]:
    """(prereg files on disk, misnamed files that look like preregistrations)."""
    d = root / PREREG_DIR
    if not d.is_dir():
        return [], []
    files, bad = [], []
    for p in sorted(d.iterdir()):
        if not p.is_file():
            continue
        if PREREG_NAME.match(p.name):
            files.append(f"{PREREG_DIR}/{p.name}")
        elif p.suffix.lower() == ".md" and p.name[:1].isdigit():
            bad.append(f"{PREREG_DIR}/{p.name}")
    return files, bad


@dataclass
class Citation:
    artifact: str
    prereg: Any
    where: str


def artifact_citations(root: Path) -> tuple[list[Citation], int, list[str]]:
    """Every prereg citation under artifacts/. Returns (citations, files scanned, problems)."""
    base = root / ARTIFACTS
    if not base.is_dir():
        return [], 0, []
    cites: list[Citation] = []
    problems: list[str] = []
    scanned = 0
    paths = sorted(p for p in base.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl"))
    for path in paths:
        rel = path.relative_to(root).as_posix()
        scanned += 1
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            problems.append(f"{rel}: cannot read as UTF-8, so its prereg citations are unknown")
            continue
        records: list[tuple[str, Any]] = []
        if path.suffix == ".json":
            try:
                records.append(("", json.loads(text)))
            except ValueError:
                problems.append(f"{rel}: not valid JSON, so its prereg citations are unknown")
                continue
        else:
            for n, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    records.append((f" row {n}", json.loads(line)))
                except ValueError:
                    problems.append(f"{rel} line {n}: not valid JSON")
        for where, rec in records:
            if not isinstance(rec, dict):
                continue
            if rec.get("prereg") is not None:
                cites.append(Citation(rel, rec["prereg"], where))
            elif rec.get("confirmatory") is True:
                problems.append(f"{rel}{where}: confirmatory record cites no preregistration")
    return cites, scanned, problems


def verify(root: Path) -> tuple[list[str], int]:
    """Run checks (a) to (g). Returns (output lines, number of problems)."""
    chain = read_chain(root)
    entries = chain.entries
    by_file = {e.get("file"): e for e in entries if isinstance(e.get("file"), str)}
    frozen_numbers = [n for f in by_file if (n := prereg_number(str(f))) is not None]
    on_disk, misnamed = disk_preregs(root)
    checks = {
        k: Check(k, n)
        for k, n in (
            ("a", "chain"),
            ("b", "files"),
            ("c", "history"),
            ("d", "artifacts"),
            ("e", "amendments"),
            ("f", "json blocks"),
            ("g", "brief"),
        )
    }

    # (a) chain
    ca = checks["a"]
    ca.problems = check_chain(chain)
    head_hash = entries[-1].get("entry_hash", "") if entries else GENESIS
    ca.summary = f"{len(entries)} entries, head {str(head_hash)[:12]}"

    # (b) files
    cb = checks["b"]
    for f in misnamed:
        cb.problems.append(f"{f}: looks like a preregistration but is not named NNNN-<slug>.md")
    for f in on_disk:
        n = sum(1 for e in entries if e.get("file") == f)
        if n != 1:
            cb.problems.append(f"{f}: has {n} chain entries (expected exactly 1; run freeze)")
        elif sha256_file(root / f) != by_file[f].get("sha256"):
            cb.problems.append(f"{f}: content differs from its chain sha256 (edited after freeze)")
    for f in by_file:
        if not (root / f).is_file():
            cb.problems.append(f"{f}: in the chain but missing on disk")
    cb.summary = f"{len(on_disk)} prereg files, each with one matching chain entry"

    # (c) history
    cc = checks["c"]
    hist, hp = load_history(root)
    cc.problems.extend(hp)
    cites, scanned, cite_problems = artifact_citations(root)
    preregs = sorted(set(on_disk) | set(by_file))
    if hist is not None:
        try:
            ever = ever_committed_preregs(root) if hist.commits else []
            artifact_paths = sorted({c.artifact for c in cites})
            lookup_blobs(root, hist, [*preregs, *ever, CHAIN, BRIEF, BRIEF_HASH, *artifact_paths])
            history_problems(root, hist, chain, preregs, ever, cc)
        except PreregError as exc:
            cc.problems.append(str(exc))
            hist = None
    elif not cc.problems:
        cc.problems.append("no git history")

    # (d) artifacts
    cd = checks["d"]
    cd.problems.extend(cite_problems)
    linked = check_citations(hist, chain, cites, cd.problems)
    cd.summary = (
        f"{linked} confirmatory citations in {len({c.artifact for c in cites})} of {scanned} "
        "artifact files, each predated by its prereg"
    )

    # (e) amendments and (f) json blocks
    ce, cf = checks["e"], checks["f"]
    amendments = 0
    blocks = 0
    for f in preregs:
        path = root / f
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            cf.problems.append(f"{f}: not UTF-8")
            continue
        probs, is_amendment = amendment_problems(f, text, frozen_numbers)
        amendments += is_amendment
        ce.problems.extend(f"{f}: {p}" for p in probs)
        probs, n = json_block_problems(text)
        blocks += n
        cf.problems.extend(f"{f}: {p}" for p in probs)
    ce.summary = f"{amendments} amendments, each citing an earlier preregistration"
    cf.summary = f"{blocks} json blocks parse"

    # (g) brief
    cg = checks["g"]
    brief_problems(root, hist, cg)

    lines = [f"prereg-verify: {len(entries)} preregistrations in {CHAIN} ({amendments} amendments)"]
    total = 0
    for c in checks.values():
        if c.problems:
            lines.append(f"[{c.key}] {c.name}: FAIL {len(c.problems)} problems")
            lines.extend(f"    - {p}" for p in c.problems)
        else:
            lines.append(f"[{c.key}] {c.name}: PASS {c.summary}")
        total += len(c.problems)
    lines.append("PREREG-VERIFY: PASS" if total == 0 else f"PREREG-VERIFY: FAIL {total} problems")
    return lines, total


def history_problems(
    root: Path, hist: History, chain: Chain, preregs: list[str], ever: list[str], cc: Check
) -> None:
    by_file = {e.get("file"): e for e in chain.entries}
    for f in sorted(set(preregs) | set(ever)):
        touches = hist.touches(f)
        blobs = hist.versions(f)
        if f not in by_file:
            if f in ever and not (root / f).exists():
                cc.problems.append(f"{f}: was committed, then removed (preregistrations are kept)")
            continue
        if not touches:
            cc.problems.append(f"{f}: not committed yet (commit it with {CHAIN})")
            continue
        if len(touches) != 1 or len(blobs) != 1:
            cc.problems.append(
                f"{f}: touched by {len(touches)} commits with {len(blobs)} versions "
                "(a frozen preregistration is never edited)"
            )
            continue
        blob = hist.at_head(f)
        if blob is None:
            cc.problems.append(f"{f}: not present at HEAD")
            continue
        load_contents(root, hist, [blob])
        if sha256_bytes(hist.content(blob)) != by_file[f].get("sha256"):
            cc.problems.append(f"{f}: committed content does not match its chain sha256")

    # CHAIN.jsonl only grows by appending, across every commit and its parents.
    bl = hist.blobs[CHAIN]
    load_contents(root, hist, bl.values())
    versions = 0
    for c in hist.commits:
        mine = bl.get(c)
        if mine is not None:
            versions += 1
        data = hist.content(mine)
        if data and not data.endswith(b"\n"):
            cc.problems.append(f"{CHAIN} at {short(c)} does not end with a newline")
        for p in hist.parents[c]:
            theirs = bl.get(p)
            if theirs != mine and not data.startswith(hist.content(theirs)):
                cc.problems.append(
                    f"{CHAIN} at {short(c)} is not an append to its parent {short(p)} "
                    "(entries were changed, removed, or reordered)"
                )
    disk = (root / CHAIN).read_bytes() if (root / CHAIN).exists() else b""
    if not disk.startswith(hist.content(hist.at_head(CHAIN))):
        cc.problems.append(f"{CHAIN} in the working tree is not an append to the HEAD version")
    head_lines = hist.content(hist.at_head(CHAIN)).count(b"\n")
    if len(chain.entries) > head_lines:
        cc.problems.append(
            f"{CHAIN}: {len(chain.entries) - head_lines} entries not committed yet "
            "(commit them with their prereg files)"
        )
    cc.summary = (
        f"each prereg touched by one commit; {CHAIN} append-only over {versions} committed versions"
    )


def entry_introductions(hist: History, seq: int) -> list[str]:
    """Commits whose CHAIN.jsonl first holds at least seq lines."""
    bl = hist.blobs[CHAIN]

    def count(c: str) -> int:
        return hist.content(bl.get(c)).count(b"\n")

    return [
        c for c in hist.commits if count(c) >= seq and all(count(p) < seq for p in hist.parents[c])
    ]


def check_citations(
    hist: History | None, chain: Chain, cites: list[Citation], problems: list[str]
) -> int:
    """Check (d). Appends problems; returns the number of citing records that passed.

    Records are grouped by (artifact file, cited prereg), so a JSONL file with many rows citing
    the same preregistration is checked once and reported once.
    """
    seq_of = {e.get("file"): i for i, e in enumerate(chain.entries, 1)}
    groups: dict[tuple[str, str], list[Citation]] = {}
    for cite in cites:
        key = cite.prereg if isinstance(cite.prereg, str) else repr(cite.prereg)
        groups.setdefault((cite.artifact, key), []).append(cite)
    ok = 0
    for (artifact, prereg), group in groups.items():
        rows = f" ({len(group)} records)" if len(group) > 1 else ""
        where = f"{artifact}{group[0].where}{rows}"
        if not isinstance(group[0].prereg, str) or prereg not in seq_of:
            problems.append(f"{where}: cites {group[0].prereg!r}, which is not in {CHAIN}")
            continue
        if hist is None:
            problems.append(f"{where}: cannot check commit order without git history")
            continue
        p_intro = hist.introductions(prereg)
        e_intro = entry_introductions(hist, seq_of[prereg])
        if len(p_intro) != 1 or not e_intro:
            problems.append(f"{where}: {prereg} or its chain entry is not committed")
            continue
        late = []
        for a in hist.introductions(artifact):
            if not hist.strictly_before(p_intro[0], a):
                late.append(f"{prereg} (committed in {short(p_intro[0])})")
            for e in e_intro:
                if not hist.strictly_before(e, a):
                    late.append(f"its chain entry (committed in {short(e)})")
            if late:
                problems.append(
                    f"{where}: first committed in {short(a)}, which {' and '.join(late)} "
                    "does not strictly predate"
                )
                break
        if late:
            continue
        ok += len(group)
    return ok


def brief_problems(root: Path, hist: History | None, cg: Check) -> None:
    brief, digest = root / BRIEF, root / BRIEF_HASH
    if not brief.is_file() or not digest.is_file():
        cg.problems.append(f"missing {BRIEF} or {BRIEF_HASH}")
        return
    words = digest.read_text(encoding="utf-8").split()
    recorded = words[0] if words else ""
    if not HEX64.match(recorded):
        cg.problems.append(f"{BRIEF_HASH} does not start with a SHA-256")
    elif sha256_file(brief) != recorded:
        cg.problems.append(f"{BRIEF} does not match {BRIEF_HASH} (the brief is immutable)")
    if hist is not None and hist.commits:
        for path in (BRIEF, BRIEF_HASH):
            n = len(hist.versions(path))
            if n > 1:
                cg.problems.append(f"{path} has {n} committed versions (the brief is immutable)")
            blob = hist.at_head(path)
            if blob is not None:
                load_contents(root, hist, [blob])
                if hist.content(blob) != (root / path).read_bytes():
                    cg.problems.append(f"{path} in the working tree differs from HEAD")
    cg.summary = f"{BRIEF} matches {BRIEF_HASH}"


# --------------------------------------------------------------------------- freeze


def resolve(root: Path, arg: str) -> Path:
    p = Path(arg)
    return p if p.is_absolute() else root / p


def freeze(root: Path, arg: str, now: str | None = None) -> str:
    """Append a chain entry for a preregistration file. Returns the entry line."""
    path = resolve(root, arg).resolve()
    try:
        rel = path.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise PreregError(f"{arg} is not inside {root}") from exc
    if not path.is_file():
        raise PreregError(f"{rel} does not exist")
    m = PREREG_NAME.match(path.name)
    if str(PurePosixPath(rel).parent) != PREREG_DIR or m is None:
        raise PreregError(f"{rel} is not named prereg/NNNN-<slug>.md")
    if not SLUG.match(m.group(2)):
        raise PreregError(f"slug {m.group(2)!r} must be lowercase letters, digits, and hyphens")
    chain = read_chain(root)
    problems = check_chain(chain)
    if problems:
        raise PreregError(f"{CHAIN} does not verify, fix it first: {problems[0]}")
    if any(e.get("file") == rel for e in chain.entries):
        raise PreregError(f"{rel} is already frozen; preregistrations are append-only")
    seq = len(chain.entries) + 1
    if int(m.group(1)) != seq:
        raise PreregError(f"{rel}: the next unused number is {seq:04d}")
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PreregError(f"{rel} is not UTF-8") from exc
    if not text.strip():
        raise PreregError(f"{rel} is empty")
    digest = sha256_bytes(data)
    if any(e.get("sha256") == digest for e in chain.entries):
        raise PreregError(f"{rel} has the same content as a frozen preregistration")
    numbers = [n for e in chain.entries if (n := prereg_number(str(e.get("file")))) is not None]
    probs, _ = amendment_problems(rel, text, numbers)
    probs += json_block_problems(text)[0]
    if probs:
        raise PreregError(f"{rel}: {probs[0]}")
    committed = git(root, "show", f"HEAD:{rel}")
    if committed.returncode == 0 and committed.stdout != data:
        raise PreregError(
            f"{rel} is committed with different content; freezing it now would give it two commits"
        )
    entry: dict[str, Any] = {
        "seq": seq,
        "file": rel,
        "sha256": digest,
        "prev": chain.entries[-1]["entry_hash"] if chain.entries else GENESIS,
        "frozen_at": now or utc_now(),
    }
    entry["entry_hash"] = entry_hash(entry)
    line = canonical(entry)
    chain_path = root / CHAIN
    with chain_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return line


def show(root: Path) -> list[str]:
    chain = read_chain(root)
    lines = [
        f"{e.get('seq'):>4}  {e.get('file')}  {e.get('frozen_at')}  {str(e.get('entry_hash'))[:12]}"
        for e in chain.entries
    ]
    lines.append(f"next unused number: {len(chain.entries) + 1:04d}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=REPO, help="repository root (default: here)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("freeze", help="append a preregistration to the chain")
    p.add_argument("path")
    sub.add_parser("verify", help="check the chain, files, history, and artifact links")
    sub.add_parser("show", help="print the chain")
    args = parser.parse_args(list(argv) if argv is not None else None)
    root: Path = args.root
    if args.cmd == "freeze":
        try:
            line = freeze(root, args.path)
        except PreregError as exc:
            print(f"prereg freeze: refused: {exc}", file=sys.stderr)
            return 2
        entry = json.loads(line)
        print(
            f"FROZEN {entry['file']} seq={entry['seq']} sha256={entry['sha256'][:12]} "
            f"entry_hash={entry['entry_hash'][:12]}"
        )
        print(f"Next: commit {entry['file']} and {CHAIN} together, before any confirmatory data.")
        return 0
    if args.cmd == "show":
        print("\n".join(show(root)))
        return 0
    lines, problems = verify(root)
    print("\n".join(lines))
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
