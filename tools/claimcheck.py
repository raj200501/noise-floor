#!/usr/bin/env python3
# Adapted from raj200501/raise-v1-public tools/claimcheck.py (Apache-2.0); changes: a number is
# backed only by a registered claim (artifacts/claims.json) marked at its position in the
# document, not by any banked value that rounds to it; the TSV allowlist of bare values became a
# TOML file of exact strings in context, each with a category from the brief and a reason; the
# scrubber is Markdown-aware (fenced code by language, inline code, HTML comments and tags, link
# targets, reference definitions, list context, character references) and keeps signs with
# numbers; the registry loader, JSON pointer
# resolver, derivations, and formatter used by tools/freshness.py live here; stdlib only.
"""claimcheck: every result number in a results document traces to a banked artifact.

Usage (what `make claimcheck` runs)::

    python tools/claimcheck.py README.md docs/REPORT.md

This enforces docs/BUILD_BRIEF.md section 9 rule 1 (no hand-typed result numbers) for gate G10.

Claim registry (artifacts/claims.json, written by a committed script, see docs/INTEGRITY.md)::

    {"schema": "noise-floor/claims/1", "generator": "<committed script>",
     "claims": [{"id": "e1.vc.task", "value": "0.31",
                 "artifact": "artifacts/e1/variance_components.json",
                 "artifact_sha256": "<sha256 of the artifact file>",
                 "pointer": "/components/task/estimate", "format": ".2f"}]}

Optional claim fields: "derive" (one of len, count, sum, mean, min, max, applied to the value at
the pointer), "where" (row filter for derive), "field" (row field for sum, mean, min, max),
"description".

Marking a number: put `<!-- claim:<id> -->` immediately before it. The rendered text that follows
the marker must start with the claim's value exactly, and every number token inside that value is
then covered. A number token that is only partly inside a claim value fails.

A number token is a run of digits with optional `.` or `,` groups (0.31, 1,042, 3.11) or a leading
decimal point (.31), with its sign when the sign is not attached to a word. Digits that follow a
letter, digit, or period (G10, E1, sha256, v0.1) are part of an identifier, not a token. An
underscore does not shield digits, so _0.31_ (emphasis) is a token.

Not checked (they do not render as text, or they are code, addresses, or digests):

- fenced code blocks whose language is a programming or config language (CODE_LANGS); other
  fenced blocks (no language, text, console, output) are checked, and a block whose info string
  has `from=artifacts/<file>` must equal that file and is then covered;
- inline code that contains a letter (inline code without letters, such as `0.31`, is checked);
- HTML comments, HTML tags, link and image targets, reference labels and definitions, footnote
  markers, bare URLs, arXiv ids, DOIs, and hex digests of 7 or more characters;
- ordered-list markers where CommonMark starts or continues a list (a line "31. ..." inside a
  paragraph is text and is checked).

A character reference that renders a digit or a sign (&#48;, &minus;) fails, since the scan reads
the source text.

Exemptions (docs/claims_exempt.toml) admit the number tokens inside one exact string of the
Markdown source, in the listed documents only::

    [[exempt]]
    text = "Python 3.11 and 3.12"
    docs = ["README.md"]
    category = "version"
    reason = "supported interpreter versions, not a measurement"

Categories are the non-result kinds that section 9 rule 1 and section 6 name: design-parameter,
threshold, cap, date, version, gate-number, section-number (also figure, table, and step
numbering), and brief-quote (the text must occur in docs/BUILD_BRIEF.md). The text must contain a
letter (a bare ISO date is allowed for the date category), every field is required, and an
exemption that covers nothing in its documents fails, so the file stays narrow.

Last line: `CLAIMCHECK: PASS <n> claims` or `CLAIMCHECK: FAIL <n> problems`; exit 0 only on PASS.
Standard library only (Python 3.11+).
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import html
import json
import math
import re
import string
import sys
import tomllib
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

REPO = Path(__file__).resolve().parent.parent
REGISTRY = "artifacts/claims.json"
EXEMPTIONS = "docs/claims_exempt.toml"
BRIEF = "docs/BUILD_BRIEF.md"
ARTIFACTS = "artifacts"
SCHEMA = "noise-floor/claims/1"

CLAIM_ID = r"[A-Za-z0-9][A-Za-z0-9_.:-]*"
CLAIM_ID_RE = re.compile(rf"^{CLAIM_ID}$")
MARKER_BODY = re.compile(rf"^\s*claim:\s*({CLAIM_ID})\s*$")
MINUS = chr(0x2212)  # the Unicode minus sign, written as a code point
# A token is not preceded by a letter, digit, or period. An underscore does not block, so
# _0.31_ (emphasis) is still a token.
TOKEN_RE = re.compile(rf"(?<![^\W_])(?<!\.)(?:[-+{MINUS}](?=\.?\d))?(?:\d+(?:[.,]\d+)*|\.\d+)")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
LETTER = re.compile(r"[A-Za-z]")

CATEGORIES = (
    "design-parameter",
    "threshold",
    "cap",
    "date",
    "version",
    "gate-number",
    "section-number",
    "brief-quote",
)
CODE_LANGS = frozenset(
    {
        "python", "py", "bash", "sh", "shell", "zsh", "toml", "json", "jsonc", "yaml", "yml",
        "ini", "cfg", "make", "makefile", "dockerfile", "diff", "sql", "r", "js", "javascript",
        "ts", "typescript", "html", "xml", "css", "mermaid", "c", "cpp", "go", "rust", "java",
    }
)  # fmt: skip
DERIVATIONS = ("len", "count", "sum", "mean", "min", "max")
CLAIM_KEYS = frozenset(
    {"id", "value", "artifact", "artifact_sha256", "pointer", "derive", "where", "field"}
    | {"format", "description"}
)
CLAIM_REQUIRED = ("id", "value", "artifact", "artifact_sha256", "pointer")
EXEMPT_KEYS = frozenset({"text", "docs", "category", "reason"})

# Private-use code points stand in for claim markers and inline-code boundaries while the text
# is rewritten, so their positions survive deletions. Plane 15 for markers, plane 16 for code.
MARK_BASE = 0xF0000
CODE_BASE = 0x100000
SENTINEL_MIN = MARK_BASE

OPEN_FENCE = re.compile(r"^(?P<prefix>[ \t]*(?:>[ \t]?)*[ \t]*)(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
CLOSE_FENCE = re.compile(r"^[ \t]*(?:>[ \t]?)*[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*$")
BLANK_LINE = re.compile(r"\n[ \t]*\n")
REF_DEF = re.compile(
    r"(?m)^[ \t]{0,3}\[(?P<label>[^\]\n^][^\]\n]*)\]:[ \t]*(?:<[^>\n]*>|\S+)"
    r"(?:[ \t]+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^)\n]*\)))?[ \t]*$"
)
FOOTNOTE = re.compile(r"\[\^[^\]\s]+\]:?")
LINK_TARGET = re.compile(
    r"(?<=\])\((?:<[^>\n]*>|(?:[^()\s]|\([^()\s]*\))*)"
    r"(?:[ \t]+(?:\"[^\"\n]*\"|'[^'\n]*'))?[ \t]*\)"
)
REF_LABEL = re.compile(r"(?<=\])\[(?P<label>[^\]\n]*)\]")
HTML_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>")
URL_RE = re.compile(r"(?i)(?:https?|ftp)://[^\s<>\"'`)\]]+|www\.[^\s<>\"'`)\]]+")
HEX_RE = re.compile(
    r"(?<![\w])(?=[0-9a-fA-F]*[a-fA-F])(?=[0-9a-fA-F]*[0-9])[0-9a-fA-F]{7,}(?![\w])"
)
# Identifiers, not claims: arXiv ids and DOIs (verified by citecheck instead).
ARXIV_RE = re.compile(r"(?i)\barxiv:\s?\d{4}\.\d{4,5}(?:v\d+)?")
DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>)\]]+")
ID_RES = (URL_RE, ARXIV_RE, DOI_RE, HEX_RE)
LIST_MARKER = re.compile(r"^[ \t]*(?:>[ \t]?)*[ \t]*(\d{1,9})[.)](?=[ \t]|$)")
BULLET = re.compile(r"^[ \t]*(?:>[ \t]?)*[ \t]*[-*+](?=[ \t]|$)")
HEADING = re.compile(r"^[ \t]{0,3}#{1,6}(?=[ \t]|$)")
ENTITY = re.compile(r"&(?:#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")
FROM_ATTR = re.compile(r"(?:^|\s)from=(\S+)")


class ClaimError(Exception):
    """A claim cannot be resolved, derived, or formatted; the message says why."""


# --------------------------------------------------------------------------- paths


def display(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def resolve(root: Path, arg: str) -> Path:
    p = Path(arg)
    return p if p.is_absolute() else root / p


def artifact_path(root: Path, rel: Any) -> tuple[Path | None, str]:
    """Validate a repository-relative path under artifacts/. Returns (path, problem)."""
    if not isinstance(rel, str) or not rel:
        return None, "artifact path is not a non-empty string"
    p = PurePosixPath(rel)
    if p.is_absolute() or "\\" in rel or ".." in p.parts or p.parts[:1] != (ARTIFACTS,):
        return None, f"{rel} is not a relative path under {ARTIFACTS}/"
    full = root / p
    if not full.is_file():
        return None, f"{rel} does not exist"
    return full, ""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- registry


def load_registry(
    root: Path, rel: str, definitions_only: bool = False
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Load and validate the claim registry. A missing registry is an empty one.

    With definitions_only (used by freshness --regenerate), schema, generator, value, and
    artifact_sha256 may be absent or out of date, since regeneration writes them.
    """
    path = resolve(root, rel)
    shown = display(root, path)
    if not path.exists():
        return {}, []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        return {}, [f"{shown}: not valid JSON ({exc})"]
    if not isinstance(data, dict):
        return {}, [f"{shown}: top level is not an object"]
    problems: list[str] = []
    extra = sorted(set(data) - {"schema", "generator", "claims", "description"})
    if extra:
        problems.append(f"{shown}: unknown top-level keys {extra}")
    if data.get("schema") != SCHEMA and not definitions_only:
        problems.append(f"{shown}: schema is {data.get('schema')!r}, expected {SCHEMA!r}")
    gen = data.get("generator")
    if definitions_only:
        pass
    elif not isinstance(gen, str) or not gen:
        problems.append(f"{shown}: generator must name the committed script that writes it")
    else:
        g = PurePosixPath(gen)
        if g.is_absolute() or ".." in g.parts or not g.parts or g.parts[0] == ARTIFACTS:
            problems.append(
                f"{shown}: generator {gen} must be a repository script outside artifacts/"
            )
        elif not (root / g).is_file():
            problems.append(f"{shown}: generator {gen} does not exist")
    claims = data.get("claims")
    if not isinstance(claims, list):
        problems.append(f"{shown}: claims is not a list")
        return {}, problems
    out: dict[str, dict[str, Any]] = {}
    for n, claim in enumerate(claims, 1):
        if not isinstance(claim, dict):
            problems.append(f"{shown}: claim {n} is not an object")
            continue
        cid = claim.get("id")
        label = f"{shown}: claim {cid if isinstance(cid, str) else n}"
        probs = claim_problems(root, claim, path, definitions_only)
        problems.extend(f"{label}: {p}" for p in probs)
        if isinstance(cid, str) and CLAIM_ID_RE.match(cid):
            if cid in out:
                problems.append(f"{label}: id appears twice")
            # Kept even when invalid, so a marker that uses it is not also reported as unknown.
            out[cid] = claim
    return out, problems


def claim_problems(
    root: Path, claim: dict[str, Any], registry: Path, definitions_only: bool = False
) -> list[str]:
    written = ("value", "artifact_sha256")  # what a generator writes
    required = [k for k in CLAIM_REQUIRED if not (definitions_only and k in written)]
    probs = [f"missing field {k!r}" for k in required if k not in claim]
    extra = sorted(set(claim) - CLAIM_KEYS)
    if extra:
        probs.append(f"unknown fields {extra}")
    cid = claim.get("id")
    if not isinstance(cid, str) or not CLAIM_ID_RE.match(cid):
        probs.append(f"id {cid!r} is not [A-Za-z0-9][A-Za-z0-9_.:-]*")
    value = claim.get("value")
    if definitions_only:
        pass
    elif not isinstance(value, str) or not re.search(r"\d", value) or value != value.strip():
        probs.append("value must be the exact document string, with a digit, no outer spaces")
    if "artifact" in claim:
        full, problem = artifact_path(root, claim["artifact"])
        if problem:
            probs.append(problem)
        elif full is not None and full.resolve() == registry.resolve():
            probs.append("a claim cannot point at the registry itself")
    digest = claim.get("artifact_sha256")
    if definitions_only:
        pass
    elif "artifact_sha256" in claim and (not isinstance(digest, str) or not HEX64.match(digest)):
        probs.append("artifact_sha256 is not 64 lowercase hex characters")
    pointer = claim.get("pointer")
    if "pointer" in claim and (
        not isinstance(pointer, str)
        or (pointer and not pointer.startswith("/"))
        or re.search(r"~(?![01])", pointer)
    ):
        probs.append(f"pointer {pointer!r} is not a JSON pointer ('' or /a/0/b)")
    derive = claim.get("derive")
    if derive is not None and derive not in DERIVATIONS:
        probs.append(f"derive {derive!r} is not one of {', '.join(DERIVATIONS)}")
    if derive is None and ("where" in claim or "field" in claim):
        probs.append("where and field need a derive")
    if "where" in claim and not isinstance(claim["where"], dict):
        probs.append("where must be an object of field: value")
    if "field" in claim and not isinstance(claim["field"], str):
        probs.append("field must be a string")
    if "format" in claim and not isinstance(claim["format"], str):
        probs.append("format must be a Python format spec string such as .2f")
    return probs


# --------------------------------------------------------------------------- derivation


def load_artifact(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = []
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as exc:
                    raise ClaimError(f"{path.name} line {n} is not JSON") from exc
        return rows
    if path.suffix == ".json":
        try:
            return json.loads(text)
        except ValueError as exc:
            raise ClaimError(f"{path.name} is not valid JSON") from exc
    raise ClaimError(f"{path.name}: only .json and .jsonl artifacts can back a claim")


def resolve_pointer(doc: Any, pointer: str) -> Any:
    """RFC 6901 JSON pointer. Every failure is an error, never a default."""
    if pointer == "":
        return doc
    if not pointer.startswith("/"):
        raise ClaimError(f"pointer {pointer!r} does not start with /")
    cur = doc
    for raw in pointer[1:].split("/"):
        part = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", part) or int(part) >= len(cur):
                raise ClaimError(f"pointer {pointer!r}: no list index {part!r}")
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            if part not in cur:
                raise ClaimError(f"pointer {pointer!r}: no key {part!r}")
            cur = cur[part]
        else:
            raise ClaimError(f"pointer {pointer!r}: cannot index into {type(cur).__name__}")
    return cur


_MISSING = object()


def _get(row: Any, key: str) -> Any:
    if key.startswith("/"):
        try:
            return resolve_pointer(row, key)
        except ClaimError:
            return _MISSING
    return row.get(key, _MISSING) if isinstance(row, dict) else _MISSING


def _same(a: Any, b: Any) -> bool:
    """JSON equality that keeps true apart from 1 and 1 apart from 1.0."""
    if a is _MISSING:
        return False
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def derive(value: Any, name: str, where: dict[str, Any] | None, fld: str | None) -> Any:
    """The named derivations, kept small so they can be audited by reading them."""
    if name == "len":
        if where or fld:
            raise ClaimError("len takes no where or field")
        if not isinstance(value, list | dict | str):
            raise ClaimError(f"len of a {type(value).__name__}")
        return len(value)
    if not isinstance(value, list):
        raise ClaimError(f"{name} needs a list at the pointer, found {type(value).__name__}")
    rows = [r for r in value if all(_same(_get(r, k), v) for k, v in (where or {}).items())]
    if name == "count":
        if fld:
            raise ClaimError("count takes no field; filter with where")
        return len(rows)
    nums = []
    for r in rows:
        x = r if fld is None else _get(r, fld)
        if x is _MISSING:
            raise ClaimError(f"a row has no field {fld!r}")
        if not isinstance(x, int | float) or (isinstance(x, float) and not math.isfinite(x)):
            raise ClaimError(f"{name} over a non-number {x!r}")
        nums.append(x)
    if not nums:
        raise ClaimError(f"{name} over no rows")
    if name == "sum":
        return sum(nums)
    if name == "mean":
        return sum(nums) / len(nums)
    if name == "min":
        return min(nums)
    if name == "max":
        return max(nums)
    raise ClaimError(f"unknown derivation {name!r}")


def format_value(value: Any, spec: str | None) -> str:
    """The document string for a value: format(value, spec), or str() for int and str."""
    if isinstance(value, bool) or value is None or isinstance(value, list | dict):
        raise ClaimError(f"{type(value).__name__} is not a number claim")
    if isinstance(value, float) and not math.isfinite(value):
        raise ClaimError(f"non-finite value {value!r}")
    if spec is None:
        if isinstance(value, float):
            raise ClaimError("a float claim needs a format such as .2f")
        return str(value)
    if isinstance(value, str):
        raise ClaimError("format applies to numbers, but the artifact holds a string")
    try:
        return format(value, spec)
    except (ValueError, TypeError) as exc:
        raise ClaimError(f"format {spec!r} failed: {exc}") from exc


def derive_claim(root: Path, claim: dict[str, Any]) -> tuple[str, str]:
    """Re-derive a claim from its artifact. Returns (document string, artifact sha256)."""
    full, problem = artifact_path(root, claim.get("artifact"))
    if full is None:
        raise ClaimError(problem)
    where, fld, spec = claim.get("where"), claim.get("field"), claim.get("format")
    if where is not None and not isinstance(where, dict):
        raise ClaimError("where must be an object")
    if fld is not None and not isinstance(fld, str):
        raise ClaimError("field must be a string")
    if spec is not None and not isinstance(spec, str):
        raise ClaimError("format must be a string")
    pointer = claim.get("pointer", "")
    if not isinstance(pointer, str):
        raise ClaimError("pointer must be a string")
    digest = sha256_file(full)
    value = resolve_pointer(load_artifact(full), pointer)
    if claim.get("derive") is not None:
        value = derive(value, str(claim["derive"]), where, fld)
    return format_value(value, spec), digest


# --------------------------------------------------------------------------- exemptions


@dataclass
class Exemption:
    index: int
    text: str
    docs: tuple[str, ...]
    category: str
    reason: str
    used: bool = False


def _squash(text: str) -> str:
    """Whitespace-insensitive form for brief quotes; drops blockquote markers."""
    return " ".join(re.sub(r"(?m)^[ \t]*(?:>[ \t]?)+", " ", text).split())


def load_exemptions(root: Path, rel: str) -> tuple[list[Exemption], list[str]]:
    path = resolve(root, rel)
    shown = display(root, path)
    if not path.exists():
        return [], []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        return [], [f"{shown}: not valid TOML ({exc})"]
    problems: list[str] = []
    extra = sorted(set(data) - {"exempt"})
    if extra:
        problems.append(f"{shown}: unknown top-level keys {extra} (use [[exempt]] tables)")
    rows = data.get("exempt", [])
    if not isinstance(rows, list):
        return [], [*problems, f"{shown}: exempt must be an array of tables"]
    brief = root / BRIEF
    brief_text = _squash(brief.read_text(encoding="utf-8")) if brief.is_file() else ""
    out: list[Exemption] = []
    for n, row in enumerate(rows, 1):
        label = f"{shown}: exemption {n}"
        if not isinstance(row, dict):
            problems.append(f"{label}: not a table")
            continue
        probs = [f"missing {k!r}" for k in sorted(EXEMPT_KEYS) if k not in row]
        unknown = sorted(set(row) - EXEMPT_KEYS)
        if unknown:
            probs.append(f"unknown keys {unknown}")
        text, docs = row.get("text"), row.get("docs")
        category, reason = row.get("category"), row.get("reason")
        if not isinstance(text, str) or not text.strip():
            probs.append("text must be a non-empty string")
            text = ""
        elif not TOKEN_RE.search(text):
            probs.append("text contains no number token, so it exempts nothing")
        elif not LETTER.search(text) and not (category == "date" and ISO_DATE.match(text)):
            probs.append("text is a bare number; quote the surrounding words so it stays narrow")
        if not isinstance(docs, list) or not docs or not all(isinstance(d, str) for d in docs):
            probs.append("docs must be a non-empty list of document paths")
            docs = []
        if category not in CATEGORIES:
            probs.append(f"category {category!r} is not one of {', '.join(CATEGORIES)}")
        if not isinstance(reason, str) or not reason.strip():
            probs.append("reason must be a non-empty string")
        if category == "brief-quote" and text and _squash(text) not in brief_text:
            probs.append(f"brief-quote text does not occur in {BRIEF}")
        if probs:
            problems.extend(f"{label} ({text[:40]!r}): {p}" for p in probs)
            continue
        norm = tuple(PurePosixPath(d).as_posix() for d in docs)
        out.append(Exemption(n, text, norm, str(category), str(reason)))
    return out, problems


# --------------------------------------------------------------------------- scanning


@dataclass
class Fence:
    line: int
    info: str
    lang: str
    source: str | None
    content: str
    start: int
    end: int
    closed: bool


@dataclass
class Marker:
    claim_id: str
    offset: int  # offset of the comment in the source
    pos: int = -1  # position in the rendered text, -1 if the marker was swallowed


@dataclass
class Doc:
    path: str
    raw: str
    text: str = ""
    src: list[int] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)
    code_spans: list[tuple[int, int]] = field(default_factory=list)
    tokens: list[tuple[int, int]] = field(default_factory=list)
    fences: list[Fence] = field(default_factory=list)
    fence_tokens: list[tuple[int, int]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    line_starts: list[int] = field(default_factory=list)

    def line_of(self, offset: int) -> int:
        return bisect.bisect_right(self.line_starts, offset)

    def where(self, offset: int) -> str:
        return f"{self.path}:{self.line_of(offset)}"

    def context(self, offset: int) -> str:
        n = self.line_of(offset)
        start = self.line_starts[n - 1]
        end = self.raw.find("\n", start)
        line = self.raw[start : end if end >= 0 else len(self.raw)].strip()
        return line if len(line) <= 100 else line[:97] + "..."


def _split_fences(doc: Doc, raw: str) -> str:
    """Record fenced blocks and return the source with their lines blanked (offsets kept)."""
    lines = raw.split("\n")
    starts = doc.line_starts
    prose = list(raw)
    i = 0
    while i < len(lines):
        m = OPEN_FENCE.match(lines[i])
        if m is None or (m["fence"][0] == "`" and "`" in m["info"]):
            i += 1
            continue
        fch, flen = m["fence"][0], len(m["fence"])
        j = i + 1
        while j < len(lines):
            c = CLOSE_FENCE.match(lines[j])
            if c and c["fence"][0] == fch and len(c["fence"]) >= flen:
                break
            j += 1
        closed = j < len(lines)
        body_start = starts[i + 1] if i + 1 < len(lines) else len(raw)
        body_end = starts[j] - 1 if closed else len(raw)
        body_end = max(body_end, body_start)
        prefix = m["prefix"]
        body_lines = raw[body_start:body_end].split("\n") if body_end > body_start else []
        content = "\n".join(ln[len(prefix) :] if ln.startswith(prefix) else ln for ln in body_lines)
        info = m["info"].strip()
        src = FROM_ATTR.search(info)
        doc.fences.append(
            Fence(
                line=i + 1,
                info=info,
                lang=info.split()[0].lower() if info else "",
                source=src.group(1) if src else None,
                content=content,
                start=body_start,
                end=body_end,
                closed=closed,
            )
        )
        if not closed:
            doc.problems.append(f"{doc.where(starts[i])}: code fence is never closed")
        end = starts[j] + len(lines[j]) if closed else len(raw)
        for k in range(starts[i], end):
            if prose[k] != "\n":
                prose[k] = " "
        i = j + 1
    return "".join(prose)


def _inline(doc: Doc, prose: str) -> tuple[str, list[int]]:
    """Left-to-right pass: escapes, inline code, and HTML comments (claim markers)."""
    out: list[str] = []
    src: list[int] = []
    blanks = [m.start() for m in BLANK_LINE.finditer(prose)]
    i, n = 0, len(prose)
    while i < n:
        ch = prose[i]
        if ch == "\\" and i + 1 < n and prose[i + 1] in string.punctuation:
            out.append(prose[i + 1])
            src.append(i + 1)
            i += 2
            continue
        if prose.startswith("<!--", i):
            j = prose.find("-->", i + 4)
            if j < 0:
                doc.problems.append(f"{doc.where(i)}: HTML comment is never closed")
                out.append(ch)
                src.append(i)
                i += 1
                continue
            body = prose[i + 4 : j]
            m = MARKER_BODY.match(body)
            if m:
                out.append(chr(MARK_BASE + len(doc.markers)))
                src.append(i)
                doc.markers.append(Marker(m.group(1), i))
            elif body.strip().lower().startswith("claim"):
                doc.problems.append(
                    f"{doc.where(i)}: malformed claim marker <!--{body}--> "
                    "(use <!-- claim:<id> -->)"
                )
            i = j + 3
            continue
        if ch == "`":
            run = 0
            while i + run < n and prose[i + run] == "`":
                run += 1
            close = re.compile(rf"(?<!`)`{{{run}}}(?!`)").search(prose, i + run)
            k = bisect.bisect_left(blanks, i)
            para_end = blanks[k] if k < len(blanks) else n
            if close and close.start() < para_end:
                k = len(doc.code_spans)
                doc.code_spans.append((-1, -1))
                out.append(chr(CODE_BASE + 2 * k))
                src.append(i)
                out.extend(prose[i + run : close.start()])
                src.extend(range(i + run, close.start()))
                out.append(chr(CODE_BASE + 2 * k + 1))
                src.append(close.start())
                i = close.end()
                continue
            out.extend(prose[i : i + run])
            src.extend(range(i, i + run))
            i += run
            continue
        out.append(ch)
        src.append(i)
        i += 1
    return "".join(out), src


def _code_ranges(text: str) -> list[tuple[int, int]]:
    ranges = []
    start = -1
    for i, ch in enumerate(text):
        o = ord(ch)
        if o >= CODE_BASE:
            if (o - CODE_BASE) % 2 == 0:
                start = i
            else:
                ranges.append((start, i))
    return ranges


def _delete(
    text: str,
    src: list[int],
    pattern: re.Pattern[str],
    keep: Callable[[re.Match[str]], bool] | None = None,
) -> tuple[str, list[int]]:
    """Delete every match of pattern that does not overlap inline code (and that keep rejects)."""
    codes = _code_ranges(text)
    pieces: list[str] = []
    out_src: list[int] = []
    pos = 0
    for m in pattern.finditer(text):
        if any(m.start() <= b and a <= m.end() - 1 for a, b in codes):
            continue
        if keep is not None and keep(m):
            continue
        pieces.append(text[pos : m.start()])
        out_src.extend(src[pos : m.start()])
        pos = m.end()
    pieces.append(text[pos:])
    out_src.extend(src[pos:])
    return "".join(pieces), out_src


def _finish(doc: Doc, text: str, src: list[int]) -> None:
    """Remove the sentinels, recording marker positions and inline-code spans."""
    out: list[str] = []
    out_src: list[int] = []
    for ch, s in zip(text, src, strict=True):
        o = ord(ch)
        if MARK_BASE <= o < CODE_BASE:
            doc.markers[o - MARK_BASE].pos = len(out)
            continue
        if o >= CODE_BASE:
            k, closing = divmod(o - CODE_BASE, 2)
            a, b = doc.code_spans[k]
            doc.code_spans[k] = (a, len(out)) if closing else (len(out), b)
            continue
        out.append(ch)
        out_src.append(s)
    doc.text = "".join(out)
    doc.src = out_src
    for mk in doc.markers:
        if mk.pos < 0:
            doc.problems.append(
                f"{doc.where(mk.offset)}: claim marker {mk.claim_id} sits inside a link target "
                "or HTML tag, where it does not mark rendered text"
            )


def _blank(patterns: Sequence[re.Pattern[str]], text: str) -> str:
    """Replace every match with spaces, keeping offsets."""
    for pattern in patterns:
        text = pattern.sub(lambda m: " " * len(m.group(0)), text)
    return text


def _list_markers(text: str) -> list[tuple[int, int]]:
    """Spans of ordered-list numbers that CommonMark renders as list markers.

    A line such as "31. The rest" inside a paragraph is paragraph text, not a list item: only a
    list that starts with 1 may interrupt a paragraph. So a marker counts when it follows a blank
    line, a heading, the start of the document, or another list line, or when it is 1.
    """
    spans = []
    pos = 0
    in_list = False
    prev = "start"  # start, blank, heading, list, text
    for line in text.split("\n"):
        m = LIST_MARKER.match(line)
        if not line.strip():
            prev = "blank"
        elif m:
            if in_list or prev in ("start", "blank", "heading", "list") or int(m.group(1)) == 1:
                spans.append((pos + m.start(1), pos + m.end(1)))
                in_list = True
                prev = "list"
            else:
                prev = "text"
        elif BULLET.match(line):
            in_list = True
            prev = "list"
        else:
            if prev == "blank" and line[:1] not in (" ", "\t"):
                in_list = False
            prev = "heading" if HEADING.match(line) else ("list" if in_list else "text")
        pos += len(line) + 1
    return spans


def _mask(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(_blank(ID_RES, text))
    for a, b in spans:
        if a >= 0 and LETTER.search(text[a:b]):
            chars[a:b] = " " * (b - a)
    for a, b in _list_markers(text):
        chars[a:b] = " " * (b - a)
    return "".join(chars)


def _entity_problems(doc: Doc) -> None:
    """A character reference that renders a digit or a sign would hide a number from the scan."""
    for m in ENTITY.finditer(doc.text):
        if any(a <= m.start() < b for a, b in doc.code_spans):
            continue
        ch = html.unescape(m.group(0))
        if len(ch) == 1 and (unicodedata.category(ch) == "Nd" or ch in "+-.,%" or ch == MINUS):
            offset = doc.src[m.start()]
            doc.problems.append(
                f"{doc.where(offset)}: character reference {m.group(0)} renders {ch!r}; "
                "write the character itself"
            )


def scan_document(raw: str, path: str) -> Doc:
    """Render a Markdown document far enough to find claim markers and number tokens."""
    doc = Doc(path=path, raw=raw)
    doc.line_starts = [0] + [m.end() for m in re.finditer("\n", raw)]
    if any(ord(c) >= SENTINEL_MIN for c in raw):
        doc.problems.append(f"{path}: contains private-use characters this checker reserves")
        raw = "".join("?" if ord(c) >= SENTINEL_MIN else c for c in raw)
    prose = _split_fences(doc, raw)
    text, src = _inline(doc, prose)
    labels: set[str] = set()

    def note_label(m: re.Match[str]) -> bool:
        labels.add(m.group("label").strip().lower())
        return False

    text, src = _delete(text, src, REF_DEF, note_label)
    text, src = _delete(text, src, FOOTNOTE)
    text, src = _delete(text, src, LINK_TARGET)
    text, src = _delete(
        text,
        src,
        REF_LABEL,
        lambda m: bool(m.group("label").strip()) and m.group("label").strip().lower() not in labels,
    )
    text, src = _delete(text, src, HTML_TAG)
    _finish(doc, text, src)
    _entity_problems(doc)
    masked = _mask(doc.text, doc.code_spans)
    doc.tokens = [(m.start(), m.end()) for m in TOKEN_RE.finditer(masked)]
    for fence in doc.fences:
        if fence.source is not None or fence.lang in CODE_LANGS:
            continue
        body_masked = _blank(ID_RES, raw[fence.start : fence.end])
        doc.fence_tokens.extend(
            (fence.start + m.start(), fence.start + m.end()) for m in TOKEN_RE.finditer(body_masked)
        )
    return doc


def marker_value(doc: Doc, mk: Marker, value: str) -> str | None:
    """The rendered text after a marker if it equals value, else what is there instead."""
    got = doc.text[mk.pos : mk.pos + len(value)]
    if got == value:
        return None
    end = doc.text.find("\n", mk.pos)
    shown = doc.text[mk.pos : end if end >= 0 else len(doc.text)]
    return shown[: max(len(value), 12)]


def fence_sources(root: Path, doc: Doc) -> tuple[list[str], int]:
    """Blocks with from=artifacts/<file> must equal that file. Returns (problems, tokens)."""
    problems: list[str] = []
    covered = 0
    for fence in doc.fences:
        if fence.source is None:
            continue
        loc = f"{doc.path}:{fence.line}"
        full, problem = artifact_path(root, fence.source)
        if full is None:
            problems.append(f"{loc}: code block from={fence.source}: {problem}")
            continue
        try:
            want = full.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"{loc}: code block from={fence.source}: not UTF-8 text")
            continue
        if fence.content.rstrip("\n") != want.rstrip("\n"):
            problems.append(f"{loc}: code block differs from {fence.source} (re-render the doc)")
            continue
        covered += len(TOKEN_RE.findall(fence.content))
    return problems, covered


@dataclass
class DocResult:
    claims: int = 0
    exempt: int = 0
    generated: int = 0
    problems: list[str] = field(default_factory=list)


def check_doc(
    root: Path, doc: Doc, values: dict[str, str], exemptions: list[Exemption]
) -> DocResult:
    """Every number token must sit inside a matching claim value or an exemption."""
    res = DocResult(problems=list(doc.problems))
    spans: list[tuple[int, int, str]] = []
    for mk in doc.markers:
        if mk.pos < 0:
            continue
        if mk.claim_id not in values:
            res.problems.append(
                f"{doc.where(mk.offset)}: claim marker {mk.claim_id} is not in the registry"
            )
            continue
        value = values[mk.claim_id]
        wrong = marker_value(doc, mk, value)
        if wrong is not None:
            res.problems.append(
                f"{doc.where(mk.offset)}: claim {mk.claim_id} is {value!r} in the registry, but "
                f"the document shows {wrong!r}"
            )
            continue
        spans.append((mk.pos, mk.pos + len(value), mk.claim_id))
        res.claims += 1
    occurrences: list[tuple[int, int, Exemption]] = []
    for ex in exemptions:
        if doc.path not in ex.docs:
            continue
        start = doc.raw.find(ex.text)
        while start >= 0:
            occurrences.append((start, start + len(ex.text), ex))
            start = doc.raw.find(ex.text, start + 1)

    def exempt(offsets: list[int]) -> bool:
        lo, hi = min(offsets), max(offsets)
        for a, b, ex in occurrences:
            if a <= lo and hi < b:
                ex.used = True
                return True
        return False

    for s, e in doc.tokens:
        token = doc.text[s:e]
        if any(a <= s and e <= b for a, b, _ in spans):
            continue
        partial = [cid for a, b, cid in spans if s < b and a < e]
        offset = doc.src[s]
        if partial:
            res.problems.append(
                f"{doc.where(offset)}: number {token} is only partly inside claim {partial[0]} "
                f"| {doc.context(offset)}"
            )
            continue
        if exempt(doc.src[s:e]):
            res.exempt += 1
            continue
        res.problems.append(
            f"{doc.where(offset)}: number {token} is not a registered claim or an exemption "
            f"| {doc.context(offset)}"
        )
    for s, e in doc.fence_tokens:
        if exempt(list(range(s, e))):
            res.exempt += 1
            continue
        res.problems.append(
            f"{doc.where(s)}: number {doc.raw[s:e]} in a code block that is not marked as code "
            f"(give it a language or from=artifacts/...) | {doc.context(s)}"
        )
    probs, covered = fence_sources(root, doc)
    res.problems.extend(probs)
    res.generated = covered
    res.problems.sort(key=lambda p: _line_key(doc.path, p))
    return res


def _line_key(path: str, problem: str) -> int:
    m = re.match(rf"{re.escape(path)}:(\d+):", problem)
    return int(m.group(1)) if m else 0


# --------------------------------------------------------------------------- main


def run(
    root: Path, docs: Sequence[str], registry: str = REGISTRY, exempt_file: str = EXEMPTIONS
) -> tuple[list[str], int, int]:
    """Check documents. Returns (output lines, problems, claims)."""
    lines: list[str] = []
    problems: list[str] = []
    claims, reg_problems = load_registry(root, registry)
    problems.extend(reg_problems)
    values = {cid: c["value"] for cid, c in claims.items() if isinstance(c.get("value"), str)}
    exemptions, ex_problems = load_exemptions(root, exempt_file)
    problems.extend(ex_problems)
    lines.append(
        f"claimcheck: {len(claims)} registered claims in {registry}, "
        f"{len(exemptions)} exemptions in {exempt_file}"
    )
    checked: set[str] = set()
    n_claims = 0
    for arg in dict.fromkeys(docs):
        path = resolve(root, arg)
        shown = display(root, path)
        if not path.is_file():
            problems.append(f"{shown}: document missing")
            lines.append(f"{shown}: MISSING")
            continue
        checked.add(shown)
        doc = scan_document(path.read_text(encoding="utf-8"), shown)
        res = check_doc(root, doc, values, exemptions)
        n_claims += res.claims
        problems.extend(res.problems)
        lines.append(
            f"{shown}: {res.claims} claims, {res.exempt} exempt numbers, "
            f"{res.generated} numbers in generated blocks, {len(res.problems)} problems"
        )
    for ex in exemptions:
        if not ex.used and set(ex.docs) <= checked:
            problems.append(
                f"{exempt_file}: exemption {ex.index} ({ex.text[:40]!r}) covers no number in "
                f"{', '.join(ex.docs)}; remove it"
            )
    lines.extend(f"  - {p}" for p in problems)
    if problems:
        lines.append(f"CLAIMCHECK: FAIL {len(problems)} problems")
    else:
        lines.append(f"CLAIMCHECK: PASS {n_claims} claims")
    return lines, len(problems), n_claims


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("docs", nargs="+", help="documents to check, relative to the root")
    parser.add_argument("--root", type=Path, default=REPO, help="repository root (default: here)")
    parser.add_argument("--claims", default=REGISTRY, help=f"claim registry (default {REGISTRY})")
    parser.add_argument("--exempt", default=EXEMPTIONS, help=f"exemptions (default {EXEMPTIONS})")
    args = parser.parse_args(list(argv) if argv is not None else None)
    lines, problems, _ = run(args.root, args.docs, args.claims, args.exempt)
    print("\n".join(lines))
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
