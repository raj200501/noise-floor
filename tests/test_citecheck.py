"""Tests for tools/citecheck.py, the offline check behind `make citecheck` (gate G13)."""

import copy
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
TODAY = date(2026, 10, 23)


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"tools_{name}", TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


citecheck = load("citecheck")


def reference(key: str, url: str) -> dict[str, Any]:
    return {
        "key": key,
        "url": url,
        "final_url": url,
        "http_status": 200,
        "title": f"A paper called {key}",
        "authors": ["Doe, Jane"],
        "year": 2025,
        "retrieved": "2026-10-23",
        "content_sha256": "ab" * 32,
        "verified": True,
        "note": "",
        "summary": {"text": "Reports something.", "based_on": "abstract", "unconfirmed": []},
    }


VALID: dict[str, Any] = {
    "schema": 1,
    "checked_by": "citation-checker subagent",
    "references": [
        reference("alpha", "https://arxiv.org/abs/0000.00001"),
        reference("beta", "https://example.org/notes/beta/"),
    ],
}


def write_repo(root: Path, doc: Any, readme: str | None = None, report: str | None = None) -> Path:
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    (root / "docs").mkdir(exist_ok=True)
    text = doc if isinstance(doc, str) else json.dumps(doc)
    (root / "artifacts" / "citecheck.json").write_text(text, encoding="utf-8")
    if readme is not None:
        (root / "README.md").write_text(readme, encoding="utf-8")
    if report is not None:
        (root / "docs" / "REPORT.md").write_text(report, encoding="utf-8")
    return root


def problems_for(root: Path) -> list[str]:
    return citecheck.run(root, TODAY)[0]


@pytest.fixture
def doc() -> dict[str, Any]:
    return copy.deepcopy(VALID)


def test_valid_file_passes(tmp_path: Path, doc: dict[str, Any], capsys: Any) -> None:
    readme = "# noise-floor\n\nSee [alpha](https://arxiv.org/abs/0000.00001).\n"
    write_repo(tmp_path, doc, readme=readme, report="Also https://example.org/notes/beta/.\n")
    assert citecheck.main(["--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == "CITECHECK: PASS 2 references verified"


@pytest.mark.parametrize("field", ["content_sha256", "final_url", "verified", "summary", "year"])
def test_missing_field_fails(tmp_path: Path, doc: dict[str, Any], field: str) -> None:
    del doc["references"][0][field]
    problems = problems_for(write_repo(tmp_path, doc))
    assert any(f"missing field '{field}'" in p for p in problems)


def test_unverified_fails(tmp_path: Path, doc: dict[str, Any], capsys: Any) -> None:
    ref = doc["references"][1]
    ref.update(verified=False, title=None, authors=[], year=None, http_status=403)
    ref["note"] = "HTTP 403, not opened"
    write_repo(tmp_path, doc)
    problems = problems_for(tmp_path)
    assert problems == ["references[1] (beta): not verified (HTTP 403, not opened)"]
    assert citecheck.main(["--root", str(tmp_path)]) == 1
    assert capsys.readouterr().out.splitlines()[-1] == "CITECHECK: FAIL 1 problems"


def test_verified_must_be_boolean(tmp_path: Path, doc: dict[str, Any]) -> None:
    doc["references"][0]["verified"] = "true"
    problems = problems_for(write_repo(tmp_path, doc))
    assert any("'verified' must be true or false" in p for p in problems)


@pytest.mark.parametrize("bad", ["ab" * 31, "AB" * 32, "zz" * 32, "", 123])
def test_bad_sha_fails(tmp_path: Path, doc: dict[str, Any], bad: Any) -> None:
    doc["references"][0]["content_sha256"] = bad
    problems = problems_for(write_repo(tmp_path, doc))
    assert any("'content_sha256' is not 64 lowercase hex" in p for p in problems)


@pytest.mark.parametrize("bad", ["23-10-2026", "2026-13-01", "20261023", "2026-12-01"])
def test_bad_or_future_date_fails(tmp_path: Path, doc: dict[str, Any], bad: str) -> None:
    doc["references"][0]["retrieved"] = bad
    problems = problems_for(write_repo(tmp_path, doc))
    assert any("'retrieved'" in p for p in problems)


def test_duplicate_keys_and_urls_fail(tmp_path: Path, doc: dict[str, Any]) -> None:
    doc["references"].append(reference("alpha", "https://arxiv.org/abs/0000.00001"))
    problems = problems_for(write_repo(tmp_path, doc))
    assert any("duplicate key 'alpha'" in p for p in problems)
    assert any("duplicate url" in p for p in problems)
    assert any("duplicate final_url" in p for p in problems)


def test_duplicate_json_object_key_fails(tmp_path: Path, doc: dict[str, Any]) -> None:
    text = json.dumps(doc).replace('"schema": 1,', '"schema": 1, "schema": 1,', 1)
    problems = problems_for(write_repo(tmp_path, text))
    assert any("duplicate JSON key 'schema'" in p for p in problems)


def test_missing_or_broken_record_fails(tmp_path: Path) -> None:
    assert any("missing" in p for p in problems_for(tmp_path))
    assert any("not valid JSON" in p for p in problems_for(write_repo(tmp_path, "{")))


def test_verified_needs_bibliographic_fields(tmp_path: Path, doc: dict[str, Any]) -> None:
    doc["references"][0].update(title="", authors=[], year=None, http_status=404)
    problems = problems_for(write_repo(tmp_path, doc))
    for expected in ("'title' is empty", "'authors' is empty", "'year'", "not 2xx"):
        assert any(expected in p for p in problems), expected


def test_unlisted_url_in_readme_fails(tmp_path: Path, doc: dict[str, Any]) -> None:
    readme = "# noise-floor\n\nCites [a source](https://unlisted.example.com/paper).\n"
    problems = problems_for(write_repo(tmp_path, doc, readme=readme))
    assert problems == ["README.md: URL not in citecheck.json: https://unlisted.example.com/paper"]


def test_unlisted_url_in_report_fails(tmp_path: Path, doc: dict[str, Any]) -> None:
    report = "Related work: http://unlisted.example.com/x, and more.\n"
    problems = problems_for(write_repo(tmp_path, doc, report=report))
    assert problems == ["docs/REPORT.md: URL not in citecheck.json: http://unlisted.example.com/x"]


def test_own_repo_links_are_allowed(tmp_path: Path, doc: dict[str, Any]) -> None:
    readme = (
        "# noise-floor\n\n"
        "git clone https://github.com/raj200501/noise-floor.git\n"
        "See [issues](https://github.com/raj200501/noise-floor/issues) and\n"
        "<https://raw.githubusercontent.com/raj200501/noise-floor/main/README.md>.\n"
    )
    assert problems_for(write_repo(tmp_path, doc, readme=readme)) == []


def test_lookalike_repo_is_not_own(tmp_path: Path, doc: dict[str, Any]) -> None:
    readme = "# noise-floor\n\nFork: https://github.com/raj200501/noise-floor-extra\n"
    problems = problems_for(write_repo(tmp_path, doc, readme=readme))
    assert problems == [
        "README.md: URL not in citecheck.json: https://github.com/raj200501/noise-floor-extra"
    ]


def test_final_url_counts_as_listed(tmp_path: Path, doc: dict[str, Any]) -> None:
    doc["references"][1]["final_url"] = "https://example.org/redirected/"
    report = "Old link https://example.org/notes/beta/ and new https://example.org/redirected/\n"
    assert problems_for(write_repo(tmp_path, doc, report=report)) == []


def test_brief_readme_and_missing_report_are_skipped(
    tmp_path: Path, doc: dict[str, Any], capsys: Any
) -> None:
    readme = "# noise-floor\n\n> **This README is a build brief, not a finished project.**\n"
    readme += "https://unlisted.example.com/in-the-brief\n"
    write_repo(tmp_path, doc, readme=readme)
    assert citecheck.main(["--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "README.md is still the build brief, skipped" in out
    assert "docs/REPORT.md does not exist, skipped" in out


def test_banked_record_is_well_formed() -> None:
    doc, problems = citecheck.load_record(ROOT / "artifacts" / "citecheck.json")
    assert problems == []
    record_problems, refs = citecheck.check_record(doc)
    assert refs
    assert [p for p in record_problems if "not verified" not in p] == []
