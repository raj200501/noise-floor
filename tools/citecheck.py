#!/usr/bin/env python3
"""Citation check (`make citecheck`, gate G13). Stdlib only, never touches the network.

Validates the banked citation record artifacts/citecheck.json written by the
citation-checker subagent (docs/BUILD_BRIEF.md section 9 rule 11):

- schema: required fields and types, lowercase sha256 hex, YYYY-MM-DD dates, a
  boolean `verified`, no duplicate JSON object keys, reference keys, or URLs;
- every reference is verified;
- every external http(s) URL in README.md and docs/REPORT.md appears in the file
  as a `url` or `final_url`, except links into this repository.

A missing README.md or docs/REPORT.md is skipped with a note, and so is README.md
while it is still the build brief. Prints one line per problem, then
`CITECHECK: PASS <n> references verified` or `CITECHECK: FAIL <n> problems`.
Exit 0 only on PASS.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parent.parent
CITECHECK = Path("artifacts/citecheck.json")
SCANNED = (Path("README.md"), Path("docs/REPORT.md"))
BRIEF_MARKER = "This README is a build brief"
OWN_OWNER, OWN_REPO = "raj200501", "noise-floor"
OWN_HOSTS = ("github.com", "www.github.com", "raw.githubusercontent.com")
# Same URL pattern and trailing-punctuation rule as G13 in tools/ship_check.py.
URL_RE = re.compile(r"https?://[^\s)>\]\"'`]+")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
SCHEMA_VERSION = 1
REQUIRED = (
    "key",
    "url",
    "final_url",
    "http_status",
    "title",
    "authors",
    "year",
    "retrieved",
    "content_sha256",
    "verified",
    "note",
    "summary",
)


class DuplicateKeyError(ValueError):
    """A JSON object repeats a key (json.loads would silently keep the last one)."""


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise DuplicateKeyError(f"duplicate JSON key {key!r}")
        seen[key] = value
    return seen


def load_record(path: Path) -> tuple[Any, list[str]]:
    """Parse the citation record, rejecting duplicate JSON object keys."""
    if not path.exists():
        return None, [f"{path.as_posix()}: missing"]
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates), []
    except DuplicateKeyError as exc:
        return None, [f"{path.as_posix()}: {exc}"]
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, [f"{path.as_posix()}: not valid JSON ({exc})"]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_http_url(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith(("http://", "https://")):
        return False
    return bool(urlsplit(value).netloc) and not any(ch.isspace() for ch in value)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str) or DATE_RE.fullmatch(value) is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def check_reference(ref: Any, where: str, today: date) -> list[str]:
    """Schema problems of one reference object (verification is checked separately)."""
    if not isinstance(ref, dict):
        return [f"{where}: not an object"]
    problems = [f"{where}: missing field '{name}'" for name in REQUIRED if name not in ref]
    verified = ref.get("verified")
    if "verified" in ref and not isinstance(verified, bool):
        problems.append(f"{where}: 'verified' must be true or false")
    if "key" in ref and not _nonempty_str(ref["key"]):
        problems.append(f"{where}: 'key' must be a non-empty string")
    for name in ("url", "final_url"):
        if name in ref and not _is_http_url(ref[name]):
            problems.append(f"{where}: '{name}' is not an http(s) URL")
    if "http_status" in ref:
        status = ref["http_status"]
        if not _is_int(status) or not 100 <= status <= 599:
            problems.append(f"{where}: 'http_status' must be an HTTP status code")
        elif verified is True and not 200 <= status <= 299:
            problems.append(f"{where}: verified but 'http_status' is {status}, not 2xx")
    if "content_sha256" in ref and not _is_sha256(ref["content_sha256"]):
        problems.append(f"{where}: 'content_sha256' is not 64 lowercase hex characters")
    retrieved = _parse_date(ref.get("retrieved"))
    if "retrieved" in ref:
        if retrieved is None:
            problems.append(f"{where}: 'retrieved' is not a YYYY-MM-DD date")
        elif retrieved > today:
            problems.append(f"{where}: 'retrieved' {retrieved.isoformat()} is in the future")
    if "note" in ref and not isinstance(ref["note"], str):
        problems.append(f"{where}: 'note' must be a string")
    problems += _check_bibliographic(ref, where, verified is True, retrieved)
    problems += _check_summary(ref.get("summary"), where) if "summary" in ref else []
    problems += _check_also_read(ref.get("also_read"), where) if "also_read" in ref else []
    return problems


def _check_bibliographic(
    ref: dict[str, Any], where: str, verified: bool, retrieved: date | None
) -> list[str]:
    """Title, authors, and year: required in full when verified, may be empty otherwise."""
    problems = []
    if "title" in ref:
        title = ref["title"]
        if verified and not _nonempty_str(title):
            problems.append(f"{where}: verified but 'title' is empty")
        elif not verified and title is not None and not isinstance(title, str):
            problems.append(f"{where}: 'title' must be a string or null")
    if "authors" in ref:
        authors = ref["authors"]
        if not isinstance(authors, list) or not all(_nonempty_str(a) for a in authors):
            problems.append(f"{where}: 'authors' must be a list of non-empty strings")
        elif verified and not authors:
            problems.append(f"{where}: verified but 'authors' is empty")
    if "year" in ref:
        year = ref["year"]
        if verified or year is not None:
            latest = retrieved.year if retrieved else 2100
            if not _is_int(year) or not 1900 <= year <= latest:
                problems.append(f"{where}: 'year' must be an integer from 1900 to {latest}")
    return problems


def _check_summary(summary: Any, where: str) -> list[str]:
    if not isinstance(summary, dict):
        return [f"{where}: 'summary' must be an object"]
    problems = []
    for name in ("text", "based_on"):
        if not _nonempty_str(summary.get(name)):
            problems.append(f"{where}: 'summary.{name}' must be a non-empty string")
    unconfirmed = summary.get("unconfirmed")
    if not isinstance(unconfirmed, list) or not all(_nonempty_str(u) for u in unconfirmed):
        problems.append(f"{where}: 'summary.unconfirmed' must be a list of strings")
    return problems


def _check_also_read(also: Any, where: str) -> list[str]:
    if not isinstance(also, list):
        return [f"{where}: 'also_read' must be a list"]
    problems = []
    for i, item in enumerate(also):
        at = f"{where}.also_read[{i}]"
        if not isinstance(item, dict):
            problems.append(f"{at}: not an object")
            continue
        if not _is_http_url(item.get("url")):
            problems.append(f"{at}: 'url' is not an http(s) URL")
        if not _is_int(item.get("http_status")):
            problems.append(f"{at}: 'http_status' must be an integer")
        if not _is_sha256(item.get("content_sha256")):
            problems.append(f"{at}: 'content_sha256' is not 64 lowercase hex characters")
    return problems


def check_record(doc: Any, today: date | None = None) -> tuple[list[str], list[dict[str, Any]]]:
    """Schema, uniqueness, and verification problems of the whole record."""
    today = today or date.today()
    if not isinstance(doc, dict):
        return ["citecheck.json: top level must be an object"], []
    problems = []
    if doc.get("schema") != SCHEMA_VERSION:
        problems.append(f"citecheck.json: 'schema' must be {SCHEMA_VERSION}")
    if not _nonempty_str(doc.get("checked_by")):
        problems.append("citecheck.json: 'checked_by' must be a non-empty string")
    refs = doc.get("references")
    if not isinstance(refs, list) or not refs:
        problems.append("citecheck.json: 'references' must be a non-empty list")
        return problems, []
    good: list[dict[str, Any]] = []
    seen: dict[str, dict[str, str]] = {"key": {}, "url": {}, "final_url": {}}
    for i, ref in enumerate(refs):
        label = ref.get("key") if isinstance(ref, dict) else None
        where = f"references[{i}]" + (f" ({label})" if isinstance(label, str) else "")
        problems += check_reference(ref, where, today)
        if not isinstance(ref, dict):
            continue
        good.append(ref)
        for name, values in seen.items():
            value = ref.get(name)
            if not isinstance(value, str):
                continue
            if value in values:
                problems.append(f"{where}: duplicate {name} {value!r} (also {values[value]})")
            else:
                values[value] = where
        if ref.get("verified") is False:
            note = str(ref.get("note", ""))[:80]
            problems.append(f"{where}: not verified ({note})")
    return problems, good


def is_own_repo(url: str) -> bool:
    """True for links into this repository (github.com/raj200501/noise-floor/...)."""
    parts = urlsplit(url)
    if parts.netloc.lower() not in OWN_HOSTS:
        return False
    segments = [s for s in parts.path.split("/") if s]
    if len(segments) < 2:
        return False
    owner, repo = segments[0].lower(), segments[1].lower()
    return owner == OWN_OWNER and repo in (OWN_REPO, f"{OWN_REPO}.git")


def cited_urls(text: str) -> list[str]:
    """External http(s) URLs in `text`, in order of first appearance."""
    found = (u.rstrip(".,") for u in URL_RE.findall(text))
    return list(dict.fromkeys(u for u in found if not is_own_repo(u)))


def check_documents(root: Path, refs: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Problems for cited URLs not in the record, and notes for skipped files."""
    known = {str(r.get(k)) for r in refs for k in ("url", "final_url") if r.get(k)}
    problems: list[str] = []
    notes: list[str] = []
    for rel_path in SCANNED:
        path = root / rel_path
        name = rel_path.as_posix()
        if not path.exists():
            notes.append(f"note: {name} does not exist, skipped")
            continue
        text = path.read_text(encoding="utf-8")
        if rel_path == Path("README.md") and BRIEF_MARKER in text:
            notes.append("note: README.md is still the build brief, skipped")
            continue
        problems += [
            f"{name}: URL not in citecheck.json: {u}" for u in cited_urls(text) if u not in known
        ]
    return problems, notes


def run(root: Path, today: date | None = None) -> tuple[list[str], list[str], int]:
    """All problems, notes, and the number of references in the record."""
    doc, problems = load_record(root / CITECHECK)
    refs: list[dict[str, Any]] = []
    if doc is not None:
        record_problems, refs = check_record(doc, today)
        problems += record_problems
    doc_problems, notes = check_documents(root, refs)
    return problems + doc_problems, notes, len(refs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO, help="repository root to check")
    args = parser.parse_args(argv)
    problems, notes, count = run(args.root)
    for line in notes:
        print(line)
    for line in problems:
        print(line)
    if problems:
        print(f"CITECHECK: FAIL {len(problems)} problems")
        return 1
    print(f"CITECHECK: PASS {count} references verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
