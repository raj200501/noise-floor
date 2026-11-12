"""Tests for tools/stage_rows.py: staging banked rows without parsing the E1 records (D-043)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tools_stage_rows", TOOLS / "stage_rows.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sr = _load()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    for sub in ("artifacts/e0/raw", "artifacts/g6", "artifacts/e1", "artifacts/usage"):
        (repo / sub).mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    return repo


def _staged(repo: Path) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(out.stdout.split())


def test_raw_rows_are_parsed_and_a_cut_line_is_refused(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    raw = repo / "artifacts/e0/raw/G-x.jsonl"
    raw.write_text(json.dumps({"i": 0}) + "\n" + json.dumps({"i": 1}) + "\n")
    assert sr.check("artifacts/e0/raw/G-x.jsonl", repo) == (
        None,
        "artifacts/e0/raw/G-x.jsonl: 2 rows",
    )
    raw.write_text(raw.read_text() + '{"i": 2, "tr')
    assert "incomplete" in str(sr.check("artifacts/e0/raw/G-x.jsonl", repo)[0])
    raw.write_text('{"i": 0}\n[1, 2]\n')
    assert "not a JSON object" in str(sr.check("artifacts/e0/raw/G-x.jsonl", repo)[0])
    (repo / "artifacts/e0/raw/environments.json").write_text("{}")
    assert "by hand" in str(sr.check("artifacts/e0/raw/environments.json", repo)[0])


def test_e1_records_and_the_ledger_are_checked_by_their_last_byte_only(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    local = repo / "artifacts/e1/local.jsonl"
    local.write_text("not json at all, and never parsed\n")  # only the last byte is read
    problem, note = sr.check("artifacts/e1/local.jsonl", repo)
    assert problem is None and "records not read" in note
    local.write_text("cut mid-record")
    assert "incomplete" in str(sr.check("artifacts/e1/local.jsonl", repo)[0])
    (repo / "artifacts/usage/ledger.jsonl").write_text("")
    assert sr.check("artifacts/usage/ledger.jsonl", repo)[0] is None


def test_changed_stages_all_or_nothing(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "artifacts/e0/raw/A.jsonl").write_text('{"i": 0}\n')
    (repo / "artifacts/g6/raw.jsonl").write_text('{"seed": 1}\n')
    (repo / "artifacts/e1/local.jsonl").write_text('{"x": 1}')  # cut: nothing is staged
    assert sr.main(["--changed"], repo) == 1
    assert _staged(repo) == []
    (repo / "artifacts/e1/local.jsonl").write_text('{"x": 1}\n')
    assert sr.main(["--changed"], repo) == 0
    assert _staged(repo) == [
        "artifacts/e0/raw/A.jsonl",
        "artifacts/e1/local.jsonl",
        "artifacts/g6/raw.jsonl",
    ]
    assert sr.main(["--changed"], repo) == 0  # staged files are still listed as changed
    assert sr.main([], repo) == 0  # nothing named, nothing staged
