"""Deterministic graders for nf-tools tasks.

A task's expectation is data, not code, so it serializes into the suite manifest:

* :class:`AnswerSpec` grades the submitted answer. The answer is first taken out of
  common wrappers by :func:`extract_answer` (the same rule in every harness), and the
  grade records both the result (``score``) and the strict comparison without that
  step (``strict``), for a sensitivity analysis. Mode ``text`` compares after
  normalization (surrounding whitespace, wrapping quotes or backticks, trailing
  periods, and whitespace runs); ``casefold`` also ignores case; ``numeric``
  parses both sides as numbers (``"1,234.50"`` is 1234.5) and accepts a
  difference of at most ``max(NUMERIC_ABS_TOL, NUMERIC_REL_TOL * |expected|)``,
  capped at half a unit in the last decimal place written in the expected answer.
  The tolerance absorbs floating-point noise from ``calc`` (``15.750000000000002``
  for ``15.75``); the cap means an answer that is wrong at the precision of the
  expected answer (one cent off a large total) never passes.
* :class:`FileSpec` grades the content of one file in the final file system,
  line by line after stripping each line, dropping blank lines, collapsing
  whitespace runs, and removing spaces around commas.

Every grade is 1 or 0 with a short reason. :func:`episode_status` maps how an
episode ended to the ``status`` covariate that evalstats uses for policy bounds.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Literal

from noisefloor.suite.fs import FileView, normalize_path

AnswerMode = Literal["text", "casefold", "numeric"]
ANSWER_MODES: tuple[str, ...] = ("text", "casefold", "numeric")
NUMERIC_REL_TOL = 1e-9
NUMERIC_ABS_TOL = 1e-6

Status = Literal["ok", "timeout", "infra_error"]
EndReason = Literal[
    "submitted",
    "stopped",
    "step_limit",
    "wall_clock",
    "context_limit",
    "backend_error",
    "harness_error",
]
_STATUS: dict[str, Status] = {
    "submitted": "ok",
    "stopped": "ok",
    "step_limit": "timeout",
    "wall_clock": "timeout",
    "context_limit": "timeout",
    "backend_error": "infra_error",
    "harness_error": "infra_error",
}
END_REASONS: tuple[str, ...] = tuple(_STATUS)

_QUOTES = "\"'`"
_WRAP_QUOTES = (("\u201c", "\u201d"), ("\u2018", "\u2019"))  # curly double and single quotes
_EMPHASIS = ("**", "__", "*", "_")
_FENCE = re.compile(r"^```[A-Za-z0-9_+-]*\s*(.*?)\s*```$", re.DOTALL)
_LEAD = re.compile(
    r"^(?:the\s+)?(?:final\s+)?(?:answer|value|total|sum|result|code|id)(?:\s+is)?\s*[:=]?\s+",
    re.IGNORECASE,
)
_LABEL = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,30}\s*[:=]\s*")  # one word, no spaces
_NOTE = re.compile(r"\s+\([^()]*\)$")
_TRAILING = ".,;:!"
_CURRENCY = "$\u20ac\u00a3\u00a5"
_UNIT = re.compile(r"^(\S+)\s+[A-Za-z%]+\.?$")
_GROUPED = re.compile(r"[+-]?[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?")
_PLAIN = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_COMMA = re.compile(r"\s*,\s*")


@dataclass(frozen=True)
class Grade:
    """A 0 or 1 score and a short reason.

    ``strict`` is the score of the strict comparison, without :func:`extract_answer`
    (None for file grades, where both are the same).
    """

    score: int
    reason: str
    strict: int | None = None

    @property
    def passed(self) -> bool:
        """True if the score is 1."""
        return self.score == 1


@dataclass(frozen=True)
class AnswerSpec:
    """The expected submitted answer and how to compare it."""

    expected: str
    mode: AnswerMode = "text"

    def __post_init__(self) -> None:
        """Reject unknown modes and numeric expectations that do not parse."""
        if self.mode not in ANSWER_MODES:
            raise ValueError(f"unknown answer mode: {self.mode!r}")
        if self.mode == "numeric" and parse_number(self.expected) is None:
            raise ValueError(f"expected answer is not a number: {self.expected!r}")

    def canonical(self) -> dict[str, str]:
        """Plain-data form for serialization."""
        return {"kind": "answer", "expected": self.expected, "mode": self.mode}


@dataclass(frozen=True)
class FileSpec:
    """The expected content of one file in the final file system."""

    path: str
    expected: str

    def __post_init__(self) -> None:
        """Require a normalized absolute path."""
        if normalize_path(self.path) != self.path:
            raise ValueError(f"not a normalized absolute path: {self.path!r}")

    def canonical(self) -> dict[str, str]:
        """Plain-data form for serialization."""
        return {"kind": "file", "path": self.path, "expected": self.expected}


Expectation = AnswerSpec | FileSpec


def _clean_once(text: str) -> str:
    text = " ".join(text.split())
    if len(text) >= 2 and text[0] == text[-1] and text[0] in _QUOTES:
        text = text[1:-1]
    return text.rstrip(".")


def normalize_text(text: str, *, casefold: bool = False) -> str:
    """Normalize an answer for comparison.

    Collapses whitespace runs to one space, strips the ends, removes wrapping
    quotes or backticks and trailing periods (repeatedly, until nothing changes),
    and case-folds if ``casefold`` is set. The result is a fixed point: normalizing
    it again changes nothing.
    """
    previous = None
    while text != previous:
        previous, text = text, _clean_once(text)
    return text.casefold() if casefold else text


def _unwrap_once(text: str, numeric: bool) -> str:
    text = " ".join(text.split())
    fence = _FENCE.match(text)
    if fence:
        return fence.group(1)
    if text.startswith("{") and text.endswith("}"):
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if isinstance(data, dict) and len(data) == 1:
            (value,) = data.values()
            if isinstance(value, str | int | float) and not isinstance(value, bool):
                return str(value)
    for left, right in (*((q, q) for q in _QUOTES), *_WRAP_QUOTES, *((e, e) for e in _EMPHASIS)):
        if len(text) > len(left) + len(right) and text.startswith(left) and text.endswith(right):
            return text[len(left) : -len(right)]
    for pattern in (_LEAD, _LABEL):
        stripped = pattern.sub("", text, count=1)
        if stripped and stripped != text:
            return stripped
    text = _NOTE.sub("", text).rstrip(_TRAILING)
    if numeric:
        unit = _UNIT.match(text)
        text = unit.group(1) if unit else text
        sign = text[:1] if text[:1] in "+-" else ""
        text = sign + text[len(sign) :].lstrip(_CURRENCY)
    return text


def extract_answer(text: str, *, numeric: bool = False) -> str:
    """Take an answer out of common wrappers, applied until nothing changes.

    Removes a code fence (with or without a language tag), a one-field JSON object
    (``{"answer": "X"}``), wrapping straight or curly quotes, backticks, or markdown
    emphasis, a leading phrase such as ``The answer is`` or ``Answer:``, a leading
    label with ``=`` or ``:`` (``port = X``, ``id=X``), a trailing note in
    parentheses, and trailing ``. , ; : !``. With ``numeric`` it also drops a
    currency sign and a trailing unit word. No expected answer in the suite contains
    any of these, so only a correct value can match after extraction.
    """
    previous = None
    while text != previous:
        previous, text = text, _unwrap_once(text, numeric)
    return text


def parse_number(text: str) -> float | None:
    """Parse a number given as text, or return None.

    Accepts the :func:`normalize_text` cleanup, an optional sign, ASCII digits,
    comma thousands separators in groups of three (``"1,234.50"``), a decimal
    point, and an exponent. Rejects anything else, including ``"1,2"`` (an
    ambiguous decimal comma), currency signs, and non-finite values.
    """
    cleaned = normalize_text(text)
    if "," in cleaned:
        if not _GROUPED.fullmatch(cleaned):
            return None
        cleaned = cleaned.replace(",", "")
    if not _PLAIN.fullmatch(cleaned):
        return None
    value = float(cleaned)
    return value if math.isfinite(value) else None


def normalize_lines(text: str) -> tuple[str, ...]:
    """Lines of a file for comparison (see the module docstring for the rules)."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned = (_COMMA.sub(",", " ".join(line.split())) for line in lines)
    return tuple(line for line in cleaned if line)


def _half_unit(number: str) -> float:
    """Half a unit in the last decimal place written in ``number`` (a valid number)."""
    mantissa, _, exponent = normalize_text(number).replace(",", "").lower().partition("e")
    places = len(mantissa.partition(".")[2]) - int(exponent or "0")
    return 0.5 * 10.0**-places


def numeric_tolerance(expected: str) -> float:
    """Largest accepted difference from the expected number (see the module docstring)."""
    want = parse_number(expected)
    if want is None:
        raise ValueError(f"not a number: {expected!r}")
    return min(_half_unit(expected), max(NUMERIC_ABS_TOL, NUMERIC_REL_TOL * abs(want)))


def _short(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def grade_answer(spec: AnswerSpec, answer: str | None) -> Grade:
    """Grade a submitted answer (None means nothing was submitted).

    The score uses :func:`extract_answer`; ``strict`` records the comparison of the
    answer as submitted.
    """
    if answer is None:
        return Grade(0, "no answer submitted", strict=0)
    strict = _compare(spec, answer)
    lenient = _compare(spec, extract_answer(answer, numeric=spec.mode == "numeric"))
    if lenient.score and not strict.score:
        lenient = Grade(1, lenient.reason + " after extraction")
    return Grade(lenient.score, lenient.reason, strict=strict.score)


def _compare(spec: AnswerSpec, answer: str) -> Grade:
    if spec.mode == "numeric":
        got = parse_number(answer)
        want = parse_number(spec.expected)
        if got is None or want is None:
            return Grade(0, f"answer is not a number: {_short(answer)!r}")
        if abs(got - want) <= numeric_tolerance(spec.expected):
            return Grade(1, "number matches")
        return Grade(0, f"expected {spec.expected}, got {_short(answer)!r}")
    casefold = spec.mode == "casefold"
    if normalize_text(answer, casefold=casefold) == normalize_text(
        spec.expected, casefold=casefold
    ):
        return Grade(1, "answer matches")
    return Grade(0, f"expected {spec.expected!r}, got {_short(answer)!r}")


def grade_file(spec: FileSpec, files: FileView) -> Grade:
    """Grade the content of ``spec.path`` in the final file system."""
    content = files.get(spec.path)
    if content is None:
        what = "a directory" if files.is_dir(spec.path) else "missing"
        return Grade(0, f"{spec.path} is {what}")
    got = normalize_lines(content)
    want = normalize_lines(spec.expected)
    if got == want:
        return Grade(1, "file matches")
    for index, (expected_line, found_line) in enumerate(zip(want, got, strict=False)):
        if expected_line != found_line:
            return Grade(
                0,
                f"line {index + 1} of {spec.path}: expected {_short(expected_line)!r}, "
                f"found {_short(found_line)!r}",
            )
    return Grade(0, f"expected {len(want)} lines in {spec.path}, found {len(got)}")


def grade(spec: Expectation, answer: str | None, files: FileView) -> Grade:
    """Grade an episode outcome: the submitted answer and the final file system."""
    if isinstance(spec, AnswerSpec):
        return grade_answer(spec, answer)
    return grade_file(spec, files)


def episode_status(reason: str) -> Status:
    """Map how an episode ended to the evalstats ``status`` covariate.

    * ``ok``: the episode ended on the agent's terms, by ``submitted`` or
      ``stopped`` (the agent stopped calling tools without submitting, which is a
      genuine failure on answer tasks, not an ambiguous run);
    * ``timeout``: the ``step_limit`` or ``wall_clock`` limit was reached first;
    * ``infra_error``: a ``backend_error`` (model server or provider failure) or a
      ``harness_error`` (a failure in this project's code) ended the episode.
    """
    try:
        return _STATUS[reason]
    except KeyError:
        raise ValueError(
            f"unknown end reason {reason!r}; expected one of {', '.join(END_REASONS)}"
        ) from None
