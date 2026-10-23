"""Tests for nf-tools graders, answer normalization, and episode status."""

import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evalstats.bounds import AMBIGUOUS
from noisefloor.suite.fs import Snapshot
from noisefloor.suite.grading import (
    END_REASONS,
    AnswerSpec,
    FileSpec,
    Grade,
    episode_status,
    grade,
    grade_answer,
    grade_file,
    normalize_lines,
    normalize_text,
    numeric_tolerance,
    parse_number,
)

# ---------------------------------------------------------------------------
# normalization


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("  Foo  ", "Foo"),
        ('"Foo"', "Foo"),
        ("`x`.", "x"),
        ("a \t  b", "a b"),
        ("'\" x \"'", "x"),
        ("x...", "x"),
        ("'", "'"),
        ("\"a'", "\"a'"),
        ("", ""),
    ],
)
def test_normalize_text(raw, want):
    assert normalize_text(raw) == want


def test_normalize_text_casefold():
    assert normalize_text(" Tk-AbC ", casefold=True) == "tk-abc"
    assert normalize_text("STRASSE", casefold=True) == normalize_text("strasse", casefold=True)
    assert normalize_text("Tk-AbC") == "Tk-AbC"


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("1,234.50", 1234.5),
        ("-1,234,567", -1234567.0),
        ("1234.5", 1234.5),
        ("-12", -12.0),
        ("+3", 3.0),
        (".5", 0.5),
        ("5.", 5.0),
        ("1e3", 1000.0),
        ("2.5E-2", 0.025),
        ("  42 ", 42.0),
        ('"42"', 42.0),
        ("42.", 42.0),
        ("`7`", 7.0),
    ],
)
def test_parse_number_accepts(raw, want):
    assert parse_number(raw) == want


@pytest.mark.parametrize(
    "raw",
    [
        "1,2",
        "12,34.5",
        "1,2345",
        "$5",
        "5 USD",
        "five",
        "nan",
        "inf",
        "1e999",
        "",
        "1 234",
        chr(0x661) + chr(0x662),
        "0x10",
        "1_000",
        "--1",
        "1.2.3",
    ],
)
def test_parse_number_rejects(raw):
    assert parse_number(raw) is None


def test_normalize_lines():
    text = "  a ,b \r\n\r\n c\td  \rlast\n\n"
    assert normalize_lines(text) == ("a,b", "c d", "last")
    assert normalize_lines("") == ()


# ---------------------------------------------------------------------------
# specs


def test_answer_spec_validation_and_canonical():
    with pytest.raises(ValueError, match="unknown answer mode"):
        AnswerSpec("x", "fuzzy")
    with pytest.raises(ValueError, match="not a number"):
        AnswerSpec("abc", "numeric")
    assert AnswerSpec("12", "numeric").canonical() == {
        "kind": "answer",
        "expected": "12",
        "mode": "numeric",
    }
    assert AnswerSpec("x").mode == "text"


def test_file_spec_validation_and_canonical():
    with pytest.raises(ValueError, match="normalized"):
        FileSpec("reports/x.txt", "a\n")
    assert FileSpec("/r/x.txt", "a\n").canonical() == {
        "kind": "file",
        "path": "/r/x.txt",
        "expected": "a\n",
    }


# ---------------------------------------------------------------------------
# answer grading


def test_grade_answer_text_modes():
    text = AnswerSpec("Tk-AbC", "text")
    assert grade_answer(text, " `Tk-AbC`. ") == Grade(1, "answer matches")
    assert grade_answer(text, "tk-abc").score == 0
    folded = AnswerSpec("QX-1234", "casefold")
    assert grade_answer(folded, "qx-1234").passed
    assert not grade_answer(folded, "QX-1235").passed
    assert grade_answer(folded, None) == Grade(0, "no answer submitted")
    assert grade_answer(folded, "").score == 0


def test_grade_answer_numeric():
    spec = AnswerSpec("1234.50", "numeric")
    for answer in ("1,234.50", "1234.5", " 1234.500 ", '"1,234.5"', "1.2345e3", "1234.500000001"):
        assert grade_answer(spec, answer).passed, answer
    assert grade_answer(AnswerSpec("15.75", "numeric"), repr(12.5 + 3.25)).passed
    assert grade_answer(AnswerSpec("0.3", "numeric"), repr(0.1 + 0.2)).passed
    wrong = grade_answer(spec, "1234.51")
    assert wrong == Grade(0, "expected 1234.50, got '1234.51'")
    assert grade_answer(spec, "about 1234").reason == "answer is not a number: 'about 1234'"
    long = grade_answer(spec, "z" * 200)
    assert long.score == 0 and long.reason.endswith("...'")
    assert grade_answer(spec, None).score == 0


def test_numeric_tolerance_is_capped_by_expected_precision():
    assert numeric_tolerance("1234.50") == pytest.approx(1.2345e-6)
    assert numeric_tolerance("9999999.77") == pytest.approx(0.005)
    assert numeric_tolerance("3") == pytest.approx(1e-6)
    assert numeric_tolerance("1.5e3") == pytest.approx(1.5e-6)
    with pytest.raises(ValueError, match="not a number"):
        numeric_tolerance("abc")
    big = AnswerSpec("9999999.77", "numeric")
    assert not grade_answer(big, "9999999.78").passed
    assert grade_answer(big, repr(9999999.77 + 1e-9)).passed
    assert not grade_answer(AnswerSpec("499", "numeric"), "499.4").passed


# ---------------------------------------------------------------------------
# file grading


def test_grade_file():
    spec = FileSpec("/out/r.txt", "SKU-1,4\nSKU-2,9\n")
    good = Snapshot({"/out/r.txt": "SKU-1, 4\r\n\r\nSKU-2 ,9  \n\n"})
    assert grade_file(spec, good) == Grade(1, "file matches")
    assert grade_file(spec, good.working_copy()).passed
    assert grade_file(spec, Snapshot()) == Grade(0, "/out/r.txt is missing")
    as_dir = Snapshot({"/out/r.txt/inner": "x"})
    assert grade_file(spec, as_dir) == Grade(0, "/out/r.txt is a directory")
    swapped = Snapshot({"/out/r.txt": "SKU-2,9\nSKU-1,4\n"})
    assert grade_file(spec, swapped).reason == (
        "line 1 of /out/r.txt: expected 'SKU-1,4', found 'SKU-2,9'"
    )
    short = Snapshot({"/out/r.txt": "SKU-1,4\n"})
    assert grade_file(spec, short).reason == "expected 2 lines in /out/r.txt, found 1"
    extra = Snapshot({"/out/r.txt": "SKU-1,4\nSKU-2,9\nSKU-3,1\n"})
    assert grade_file(spec, extra).reason == "expected 2 lines in /out/r.txt, found 3"
    case = Snapshot({"/out/r.txt": "sku-1,4\nSKU-2,9\n"})
    assert grade_file(spec, case).score == 0


def test_grade_dispatches_on_spec_kind():
    files = Snapshot({"/a.txt": "x\n"})
    assert grade(AnswerSpec("7", "numeric"), "7.0", files).passed
    assert grade(FileSpec("/a.txt", "x"), None, files).passed
    assert not grade(FileSpec("/b.txt", "x"), "x", files).passed


# ---------------------------------------------------------------------------
# episode status


@pytest.mark.parametrize(
    ("reason", "status"),
    [
        ("submitted", "ok"),
        ("stopped", "ok"),
        ("step_limit", "timeout"),
        ("wall_clock", "timeout"),
        ("backend_error", "infra_error"),
        ("harness_error", "infra_error"),
    ],
)
def test_episode_status(reason, status):
    assert episode_status(reason) == status


def test_episode_status_rejects_unknown_reasons():
    with pytest.raises(ValueError, match="unknown end reason 'crashed'"):
        episode_status("crashed")


def test_ambiguous_statuses_match_evalstats():
    statuses = {episode_status(reason) for reason in END_REASONS}
    assert statuses == {"ok", "timeout", "infra_error"}
    assert {"timeout", "infra_error"} <= set(AMBIGUOUS)
    assert "ok" not in AMBIGUOUS


# ---------------------------------------------------------------------------
# properties

TOKENS = st.text(alphabet=string.ascii_letters + string.digits + "-_", min_size=1, max_size=20)
PAD = st.text(alphabet=" \t\n", max_size=3)


@pytest.mark.property
@settings(max_examples=200)
@given(st.integers(min_value=-(10**13), max_value=10**13), PAD, PAD)
def test_property_numeric_formats_grade_the_same(cents, left, right):
    sign = "-" if cents < 0 else ""
    whole, frac = divmod(abs(cents), 100)
    expected = f"{sign}{whole}.{frac:02d}"
    spec = AnswerSpec(expected, "numeric")
    grouped = f"{sign}{whole:,}.{frac:02d}"
    for answer in (expected, grouped, f"{left}{grouped}{right}", f'"{grouped}"', f"{grouped}."):
        assert grade_answer(spec, answer).passed, answer
    assert parse_number(grouped) == pytest.approx(float(expected), rel=1e-12)
    nearby = cents + 1 if cents < 10**13 else cents - 1
    sign2 = "-" if nearby < 0 else ""
    whole2, frac2 = divmod(abs(nearby), 100)
    assert not grade_answer(spec, f"{sign2}{whole2}.{frac2:02d}").passed


@pytest.mark.property
@given(TOKENS, PAD, PAD)
def test_property_casefold_ignores_case_and_padding(token, left, right):
    folded = AnswerSpec(token, "casefold")
    assert grade_answer(folded, f"{left}{token.upper()}{right}").passed
    assert grade_answer(folded, f"{left}`{token.lower()}`{right}").passed
    text = AnswerSpec(token, "text")
    assert grade_answer(text, f"{left}{token}{right}").passed
    assert grade_answer(text, token.swapcase()).passed == (token.swapcase() == token)


@pytest.mark.property
@given(st.text(max_size=40), st.booleans())
def test_property_normalize_text_is_idempotent(text, casefold):
    once = normalize_text(text, casefold=casefold)
    assert normalize_text(once, casefold=casefold) == once
    assert once == once.strip()


@pytest.mark.property
@given(st.text(max_size=40))
def test_property_parse_number_never_raises(text):
    value = parse_number(text)
    assert value is None or isinstance(value, float)


@pytest.mark.property
@given(st.lists(TOKENS, min_size=1, max_size=8), st.sampled_from(["\n", "\r\n", "\r"]), PAD)
def test_property_file_grade_ignores_line_endings_and_padding(lines, newline, pad):
    spec = FileSpec("/out.txt", "\n".join(lines) + "\n")
    written = newline.join(f"{pad}{line}{pad.replace(chr(10), '')}" for line in lines)
    assert grade_file(spec, Snapshot({"/out.txt": written})).passed
    if len(lines) > 1 and lines != lines[::-1]:
        flipped = Snapshot({"/out.txt": "\n".join(lines[::-1])})
        assert not grade_file(spec, flipped).passed
