#!/usr/bin/env python3
"""Secrets scan (`make secrets-scan`).

Scans every tracked file in the working tree and every blob reachable from any
ref in git history for credential patterns, and for the literal values of
sensitive environment variables set in this shell. Matches are reported by file
and pattern name only; a matched value is never printed. Exit 1 on any finding.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Built by concatenation so this file never matches its own patterns.
PATTERNS: dict[str, re.Pattern[bytes]] = {
    "aws-access-key": re.compile(b"AKIA" + rb"[0-9A-Z]{16}"),
    "github-token": re.compile(b"gh" + rb"[pousr]_[A-Za-z0-9]{36,}"),
    "github-pat": re.compile(b"github" + rb"_pat_[A-Za-z0-9_]{60,}"),
    "anthropic-key": re.compile(b"sk-" + rb"ant-[A-Za-z0-9_-]{20,}"),
    "openrouter-key": re.compile(b"sk-" + rb"or-v1-[a-f0-9]{40,}"),
    "openai-key": re.compile(b"sk-" + rb"(?:proj-)?[A-Za-z0-9]{40,}"),
    "google-api-key": re.compile(b"AI" + rb"za[0-9A-Za-z_-]{35}"),
    "slack-token": re.compile(b"xox" + rb"[abposr]-[A-Za-z0-9-]{10,}"),
    "huggingface-token": re.compile(b"hf" + rb"_[A-Za-z]{34,}"),
    "private-key": re.compile(b"-----BEGIN " + rb"(?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY"),
    "generic-assignment": re.compile(
        rb"(?i)(?:api[_-]?key|secret|access[_-]?token|auth[_-]?token|password)['\"]?\s*[:=]\s*"
        rb"['\"][A-Za-z0-9/+_-]{24,}['\"]"
    ),
}
SENSITIVE_ENV = (
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "GITHUB_MODELS_TOKEN",
    "OPENAI_API_KEY",
    "HF_TOKEN",
    "AWS_SECRET_ACCESS_KEY",
    "CLAUDE_CODE_MESSAGING_TOKEN",
)
MIN_ENV_SECRET_LEN = 16


def scan_bytes(data: bytes, env_values: list[bytes]) -> list[str]:
    """Names of the patterns (or env-value hits) found in `data`."""
    hits = [name for name, pat in PATTERNS.items() if pat.search(data)]
    hits += ["env-value" for value in env_values if value in data]
    return hits


def env_secret_values() -> list[bytes]:
    """Values of sensitive environment variables long enough to be meaningful."""
    values = []
    for name in SENSITIVE_ENV:
        value = os.environ.get(name, "")
        if len(value) >= MIN_ENV_SECRET_LEN:
            values.append(value.encode())
    return values


def _git(*args: str, data: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=REPO, input=data, capture_output=True, check=True
    ).stdout


def history_blobs() -> list[tuple[str, bytes]]:
    """(label, content) for every blob reachable from any ref."""
    listing = _git("rev-list", "--objects", "--all").decode().splitlines()
    paths = {}
    for line in listing:
        oid, _, path = line.partition(" ")
        paths.setdefault(oid, path)
    if not paths:
        return []
    check = _git("cat-file", "--batch-check", data="\n".join(paths).encode()).decode()
    blobs = [ln.split()[0] for ln in check.splitlines() if ln.split()[1:2] == ["blob"]]
    raw = _git("cat-file", "--batch", data="\n".join(blobs).encode())
    out = []
    pos = 0
    while pos < len(raw):
        header_end = raw.index(b"\n", pos)
        oid, _, size = raw[pos:header_end].decode().split()
        start = header_end + 1
        end = start + int(size)
        out.append((f"history:{paths.get(oid, oid)}@{oid[:10]}", raw[start:end]))
        pos = end + 1
    return out


def main() -> int:
    env_values = env_secret_values()
    findings = []
    tracked = _git("ls-files", "-z").decode().split("\0")
    for rel in filter(None, tracked):
        path = REPO / rel
        if path.is_file():
            findings += [(rel, h) for h in scan_bytes(path.read_bytes(), env_values)]
    blobs = history_blobs()
    for label, content in blobs:
        findings += [(label, h) for h in scan_bytes(content, env_values)]
    for where, what in sorted(set(findings)):
        print(f"FINDING {what}: {where}")
    n_tracked = len([t for t in tracked if t])
    print(
        f"scanned {n_tracked} tracked files and {len(blobs)} history blobs; "
        f"{len(env_values)} env values checked"
    )
    print("SECRETS-SCAN: " + ("CLEAN" if not findings else f"{len(set(findings))} FINDINGS"))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
