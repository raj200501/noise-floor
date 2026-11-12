#!/usr/bin/env python3
"""Independent review (`make review`), docs/BUILD_BRIEF.md section 10.1 step 8.

Two paths:

* Headless: when the Claude backend is usable (billing guard clean, `claude` CLI
  present, review cap not used up today), run the reviewer prompt from
  .claude/agents/reviewer.md through `claude -p --model opus` via the client
  wrapper, which writes artifacts/review/latest.json.
* Subagent fallback: otherwise the `reviewer` subagent writes
  artifacts/review/latest.json itself (the main session never edits it), and this
  script validates and prints it.

`--validate` only validates and prints. The last stdout line is always one of
``REVIEW: SHIP``, ``REVIEW: BLOCK``, ``REVIEW: STALE`` or ``REVIEW: MISSING``.

latest.json schema::

    {"reviewed_commit": "<sha>", "verdict": "SHIP" | "BLOCK", "date": "<iso>",
     "reviewer": "<how it ran>",
     "blocking": [{"id": "B1", "title": "...", "where": "file:line or artifact",
                   "evidence": "..."}],
     "non_blocking": [{"id": "N1", "title": "..."}]}

The reviewed commit may differ from HEAD only in artifacts/review/, STATUS.md and
docs/LOG.md. Every non-blocking issue id must appear in docs/REVIEW_LOG.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
LATEST = REPO / "artifacts" / "review" / "latest.json"
REVIEW_LOG = REPO / "docs" / "REVIEW_LOG.md"
ALLOWED_DRIFT = ("artifacts/review/", "STATUS.md", "docs/LOG.md")


def _git(*args: str) -> tuple[int, str]:
    proc = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stdout.strip()


def id_listed(issue_id: str, text: str) -> bool:
    """True if ``issue_id`` appears in ``text`` as a whole id (N1 is not listed by N10)."""
    pattern = rf"(?<![\w-]){re.escape(issue_id)}(?![\w-])"
    return bool(issue_id) and re.search(pattern, text) is not None


def validate(data: dict[str, Any]) -> tuple[str, list[str]]:
    """Return (final verdict line, problems). Problems make the review not current."""
    problems = []
    verdict = data.get("verdict")
    blocking = data.get("blocking")
    if verdict not in ("SHIP", "BLOCK"):
        problems.append(f"verdict is {verdict!r}")
    if not isinstance(blocking, list):
        problems.append("blocking is not a list")
        blocking = []
    if verdict == "SHIP" and blocking:
        problems.append("verdict SHIP with blocking issues")
    for issue in blocking:
        for key in ("id", "title", "where", "evidence"):
            if not issue.get(key):
                problems.append(f"blocking issue lacks {key}")
    reviewed = str(data.get("reviewed_commit", ""))
    code, head = _git("rev-parse", "HEAD")
    if not reviewed:
        problems.append("no reviewed_commit")
    elif reviewed != head:
        code, _ = _git("merge-base", "--is-ancestor", reviewed, head)
        if code != 0:
            problems.append(f"reviewed commit {reviewed[:12]} is not an ancestor of HEAD")
        else:
            _, diff = _git("diff", "--name-only", reviewed, head)
            stray = [
                p
                for p in diff.splitlines()
                if not (p.startswith(ALLOWED_DRIFT[0]) or p in ALLOWED_DRIFT[1:])
            ]
            if stray:
                problems.append(f"code changed since review: {', '.join(stray[:5])}")
    log_text = REVIEW_LOG.read_text() if REVIEW_LOG.exists() else ""
    for issue in data.get("non_blocking", []):
        if not id_listed(str(issue.get("id")), log_text):
            problems.append(f"non-blocking {issue.get('id')} not in docs/REVIEW_LOG.md")
    if problems:
        return "REVIEW: STALE", problems
    return f"REVIEW: {verdict}", []


def _doctor() -> Any:
    spec = importlib.util.spec_from_file_location("doctor", REPO / "tools" / "doctor.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def headless_available() -> tuple[bool, str]:
    """Whether the headless reviewer path can run here, and why not."""
    doctor = _doctor()
    guard = doctor.billing_guard_findings()
    if guard:
        return False, "billing guard tripped by " + ", ".join(guard)
    if importlib.util.find_spec("noisefloor.client") is None:
        return False, "client wrapper (noisefloor.client) not built yet"
    return True, ""


def print_review(data: dict[str, Any], verdict_line: str, problems: list[str]) -> None:
    print(f"reviewed_commit: {data.get('reviewed_commit')}")
    print(f"reviewer: {data.get('reviewer')}  date: {data.get('date')}")
    print(f"non-blocking issues: {len(data.get('non_blocking', []))} (docs/REVIEW_LOG.md)")
    for problem in problems:
        print(f"problem: {problem}")
    for i, issue in enumerate(data.get("blocking", []), 1):
        print(f"{i}. [{issue.get('id')}] {issue.get('title')} ({issue.get('where')})")
        print(f"   evidence: {issue.get('evidence')}")
    print(verdict_line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="validate and print only")
    args = parser.parse_args(argv)
    if not args.validate:
        ok, why = headless_available()
        if ok:
            print("headless review: not wired yet; use the reviewer subagent", file=sys.stderr)
        else:
            print(
                f"headless review unavailable ({why}); validating the subagent review",
                file=sys.stderr,
            )
    if not LATEST.exists():
        print(
            "no artifacts/review/latest.json: run the `reviewer` subagent "
            "(.claude/agents/reviewer.md) to write it"
        )
        print("REVIEW: MISSING")
        return 1
    data = json.loads(LATEST.read_text())
    verdict_line, problems = validate(data)
    print_review(data, verdict_line, problems)
    return 0 if verdict_line == "REVIEW: SHIP" else 1


if __name__ == "__main__":
    sys.exit(main())
