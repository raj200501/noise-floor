# Adapted from raj200501/raise-v1-public tests/mutation_test.py (Apache-2.0); changes: pytest
# cases instead of a standalone runner; each case builds a temporary git repository with real
# commits; mutations target the noise-floor chain (Markdown preregistrations, history checks,
# artifact links, amendments, the brief hash) rather than raise-v1's sealed JSON fields.
"""Tests for tools/prereg.py.

The mutation table follows raise-v1: a seeded repository verifies, and each mutation breaks one
property the gate guards. A mutation that still verifies is a hole in the gate.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"tools_{name}", TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prereg = load("prereg")

BRIEF = "# brief\n\nImmutable text with numbers 14 and 0.95.\n"
PREREG_1 = """# 0001 E0 grid

Hypotheses, metrics, and the analysis plan.

```json e0-grid
{"cells": [{"cell": "a", "method": "wald", "nominal": 0.95, "replicates": 2000}]}
```
"""
PREREG_2 = '# 0002 E1 design\n\n```json e1-design\n{"factors": {}}\n```\n'
AMENDMENT = "---\namends: 0001\ndate: 2026-10-23\n---\n# 0002 amendment\n\nDeviation.\n"


class Repo:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.n = 0

    def git(self, *args: str) -> str:
        proc = subprocess.run(
            ["git", *args], cwd=self.root, capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.strip()

    def write(self, rel: str, text: str) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def commit(self, msg: str = "") -> str:
        self.n += 1
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", msg or f"commit {self.n}")
        return self.git("rev-parse", "HEAD")

    def freeze(self, rel: str, text: str) -> None:
        self.write(rel, text)
        prereg.freeze(self.root, rel, now=f"2026-10-23T00:00:{self.n:02d}Z")

    def verify(self) -> tuple[list[str], int]:
        lines, problems = prereg.verify(self.root)
        return lines, problems

    def chain_lines(self) -> list[str]:
        return (self.root / prereg.CHAIN).read_text(encoding="utf-8").splitlines()

    def write_chain(self, entries: list[dict[str, object]]) -> None:
        text = "".join(prereg.canonical(e) + "\n" for e in entries)
        self.write(prereg.CHAIN, text)


def rehash(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    """Recompute prev links and entry hashes, as a careful tamperer would."""
    prev = prereg.GENESIS
    out = []
    for i, e in enumerate(entries, 1):
        body = {k: v for k, v in e.items() if k != "entry_hash"}
        body["seq"] = i
        body["prev"] = prev
        body["entry_hash"] = prereg.entry_hash(body)
        prev = str(body["entry_hash"])
        out.append(body)
    return out


def force_entry(repo: Repo, rel: str) -> None:
    """Append a chain entry without freeze's content checks (to plant a bad prereg)."""
    chain = prereg.read_chain(repo.root)
    entry: dict[str, object] = {
        "seq": len(chain.entries) + 1,
        "file": rel,
        "sha256": hashlib.sha256((repo.root / rel).read_bytes()).hexdigest(),
        "prev": chain.entries[-1]["entry_hash"] if chain.entries else prereg.GENESIS,
        "frozen_at": "2026-10-23T00:00:00Z",
    }
    entry["entry_hash"] = prereg.entry_hash(entry)
    with (repo.root / prereg.CHAIN).open("a", encoding="utf-8") as fh:
        fh.write(prereg.canonical(entry) + "\n")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Repo:
    for key, value in {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }.items():
        monkeypatch.setenv(key, value)
    r = Repo(tmp_path / "repo")
    r.root.mkdir()
    r.git("init", "-q", "-b", "main")
    r.write("docs/BUILD_BRIEF.md", BRIEF)
    digest = hashlib.sha256(BRIEF.encode()).hexdigest()
    r.write("docs/BUILD_BRIEF.sha256", f"{digest}  docs/BUILD_BRIEF.md\n")
    r.write("prereg/README.md", "# Preregistrations\n")
    r.commit("bootstrap")
    return r


@pytest.fixture
def seeded(repo: Repo) -> Repo:
    """A frozen, committed prereg and a confirmatory artifact committed after it."""
    repo.freeze("prereg/0001-e0-grid.md", PREREG_1)
    repo.commit("prereg 0001")
    repo.write(
        "artifacts/e0/coverage.json",
        json.dumps({"prereg": "prereg/0001-e0-grid.md", "cells": []}),
    )
    repo.write(
        "artifacts/e1/runs.jsonl",
        "".join(
            json.dumps({"run_id": i, "prereg": "prereg/0001-e0-grid.md", "confirmatory": True})
            + "\n"
            for i in range(3)
        ),
    )
    repo.commit("bank artifacts")
    return repo


def failed(lines: list[str], key: str) -> bool:
    return any(line.startswith(f"[{key}]") and ": FAIL" in line for line in lines)


# --------------------------------------------------------------------------- controls


def test_empty_repository_passes_and_reports_the_count(repo: Repo) -> None:
    lines, problems = repo.verify()
    assert problems == 0
    assert lines[0].startswith("prereg-verify: 0 preregistrations")
    assert lines[-1] == "PREREG-VERIFY: PASS"
    assert [line[:3] for line in lines[1:-1]] == [f"[{k}]" for k in "abcdefg"]


def test_seeded_repository_passes(seeded: Repo) -> None:
    lines, problems = seeded.verify()
    assert problems == 0, lines
    assert lines[-1] == "PREREG-VERIFY: PASS"
    assert "4 confirmatory citations in 2 of 2 artifact files" in "\n".join(lines)


def test_chain_entry_fields_and_hash(seeded: Repo) -> None:
    (line,) = seeded.chain_lines()
    entry = json.loads(line)
    assert set(entry) == {"seq", "file", "sha256", "prev", "frozen_at", "entry_hash"}
    assert entry["seq"] == 1 and entry["prev"] == "0" * 64
    assert entry["file"] == "prereg/0001-e0-grid.md"
    assert entry["sha256"] == hashlib.sha256(PREREG_1.encode()).hexdigest()
    body = {k: v for k, v in entry.items() if k != "entry_hash"}
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert entry["entry_hash"] == hashlib.sha256(canon.encode()).hexdigest()
    assert line == prereg.canonical(entry)


def test_second_prereg_links_to_first(seeded: Repo) -> None:
    seeded.freeze("prereg/0002-e1-design.md", PREREG_2)
    seeded.commit()
    first, second = (json.loads(x) for x in seeded.chain_lines())
    assert second["prev"] == first["entry_hash"] and second["seq"] == 2
    assert seeded.verify()[1] == 0


def test_uncommitted_artifact_passes_when_prereg_is_committed(seeded: Repo) -> None:
    seeded.write("artifacts/e1/mde.json", json.dumps({"prereg": "prereg/0001-e0-grid.md"}))
    lines, problems = seeded.verify()
    assert problems == 0, lines


def test_amendment_citing_earlier_prereg_passes(seeded: Repo) -> None:
    seeded.freeze("prereg/0002-amend-grid.md", AMENDMENT)
    seeded.commit()
    lines, problems = seeded.verify()
    assert problems == 0, lines
    assert "1 amendments" in lines[0]


def test_feature_branch_merge_counts_as_one_touch(repo: Repo) -> None:
    repo.git("checkout", "-q", "-b", "feature")
    repo.freeze("prereg/0001-e0-grid.md", PREREG_1)
    repo.commit("prereg on a branch")
    repo.git("checkout", "-q", "main")
    repo.write("notes.txt", "other work\n")
    repo.commit("main moves on")
    repo.git("merge", "-q", "--no-ff", "feature", "-m", "merge feature")
    repo.write("artifacts/x.json", json.dumps({"prereg": "prereg/0001-e0-grid.md"}))
    repo.commit("artifact after merge")
    lines, problems = repo.verify()
    assert problems == 0, lines


def test_cli_verify_and_exit_codes(seeded: Repo) -> None:
    cmd = [sys.executable, str(TOOLS / "prereg.py"), "--root", str(seeded.root), "verify"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stdout
    assert proc.stdout.strip().splitlines()[-1] == "PREREG-VERIFY: PASS"
    (seeded.root / "prereg/0001-e0-grid.md").write_text("edited\n")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 1
    assert proc.stdout.strip().splitlines()[-1].startswith("PREREG-VERIFY: FAIL")


def test_cli_freeze_and_show(repo: Repo) -> None:
    repo.write("prereg/0001-e0-grid.md", PREREG_1)
    base = [sys.executable, str(TOOLS / "prereg.py"), "--root", str(repo.root)]
    proc = subprocess.run(
        [*base, "freeze", "prereg/0001-e0-grid.md"], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0 and proc.stdout.startswith("FROZEN prereg/0001-e0-grid.md")
    again = subprocess.run(
        [*base, "freeze", "prereg/0001-e0-grid.md"], capture_output=True, text=True, check=False
    )
    assert again.returncode == 2 and "already frozen" in again.stderr
    shown = subprocess.run([*base, "show"], capture_output=True, text=True, check=False)
    assert "next unused number: 0002" in shown.stdout


# --------------------------------------------------------------------------- freeze refusals


@pytest.mark.parametrize(
    ("rel", "text", "message"),
    [
        ("prereg/0002-skip.md", PREREG_1, "next unused number is 0001"),
        ("prereg/0001_bad.md", PREREG_1, "not named"),
        ("prereg/0001-Bad_Slug.md", PREREG_1, "slug"),
        ("docs/0001-elsewhere.md", PREREG_1, "not named"),
        ("prereg/0001-empty.md", "  \n", "empty"),
        ("prereg/0001-badjson.md", "```json grid\n{not json}\n```\n", "not valid JSON"),
        ("prereg/0001-dup.md", "```json g\n{}\n```\n```json g\n{}\n```\n", "unique"),
        ("prereg/0001-amend.md", "---\namends: 0001\ndate: 2026-10-23\n---\n", "not an earlier"),
    ],
)
def test_freeze_refuses(repo: Repo, rel: str, text: str, message: str) -> None:
    repo.write(rel, text)
    with pytest.raises(prereg.PreregError, match=message):
        prereg.freeze(repo.root, rel)
    assert not (repo.root / prereg.CHAIN).exists()


def test_freeze_refuses_double_freeze_and_duplicate_content(seeded: Repo) -> None:
    with pytest.raises(prereg.PreregError, match="already frozen"):
        prereg.freeze(seeded.root, "prereg/0001-e0-grid.md")
    seeded.write("prereg/0002-copy.md", PREREG_1)
    with pytest.raises(prereg.PreregError, match="same content"):
        prereg.freeze(seeded.root, "prereg/0002-copy.md")


def test_freeze_refuses_amendment_without_date_or_target(seeded: Repo) -> None:
    seeded.write("prereg/0002-amend.md", "---\namends: 0001\n---\nchange\n")
    with pytest.raises(prereg.PreregError, match="date"):
        prereg.freeze(seeded.root, "prereg/0002-amend.md")
    seeded.write("prereg/0002-amend.md", "---\namends: 0000\ndate: 2026-10-23\n---\nchange\n")
    with pytest.raises(prereg.PreregError, match="names no preregistration"):
        prereg.freeze(seeded.root, "prereg/0002-amend.md")


def test_freeze_refuses_committed_then_edited_file(repo: Repo) -> None:
    repo.write("prereg/0001-e0-grid.md", PREREG_1)
    repo.commit("draft committed")
    repo.write("prereg/0001-e0-grid.md", PREREG_1 + "edited after commit\n")
    with pytest.raises(prereg.PreregError, match="two commits"):
        prereg.freeze(repo.root, "prereg/0001-e0-grid.md")


def test_freeze_refuses_on_broken_chain(seeded: Repo) -> None:
    entry = json.loads(seeded.chain_lines()[0])
    entry["sha256"] = "0" * 64
    seeded.write_chain([entry])
    seeded.write("prereg/0002-e1-design.md", PREREG_2)
    with pytest.raises(prereg.PreregError, match="does not verify"):
        prereg.freeze(seeded.root, "prereg/0002-e1-design.md")


# --------------------------------------------------------------------------- mutations


def edit_frozen_and_commit(r: Repo) -> None:
    r.write("prereg/0001-e0-grid.md", PREREG_1.replace("0.95", "0.90"))
    r.commit("tweak threshold")


def edit_frozen_in_working_tree(r: Repo) -> None:
    r.write("prereg/0001-e0-grid.md", PREREG_1.replace("2000", "200"))


def edit_frozen_and_rehash_chain(r: Repo) -> None:
    text = PREREG_1.replace("0.95", "0.80")
    r.write("prereg/0001-e0-grid.md", text)
    entry = json.loads(r.chain_lines()[0])
    entry["sha256"] = hashlib.sha256(text.encode()).hexdigest()
    r.write_chain(rehash([entry]))
    r.commit("rewrite history the careful way")


def tamper_chain_sha(r: Repo) -> None:
    entry = json.loads(r.chain_lines()[0])
    entry["sha256"] = "f" * 64
    r.write_chain([entry])


def tamper_chain_frozen_at(r: Repo) -> None:
    entry = json.loads(r.chain_lines()[0])
    entry["frozen_at"] = "2020-01-01T00:00:00Z"
    r.write_chain([entry])


def reorder_chain(r: Repo) -> None:
    r.freeze("prereg/0002-e1-design.md", PREREG_2)
    r.commit()
    first, second = (json.loads(x) for x in r.chain_lines())
    r.write_chain([second, first])


def reorder_chain_rehashed_and_committed(r: Repo) -> None:
    r.freeze("prereg/0002-e1-design.md", PREREG_2)
    r.commit()
    first, second = (json.loads(x) for x in r.chain_lines())
    r.write_chain(rehash([second, first]))
    r.commit("reorder")


def truncate_chain(r: Repo) -> None:
    r.write(prereg.CHAIN, "")
    r.commit("drop the chain")


def noncanonical_chain_line(r: Repo) -> None:
    entry = json.loads(r.chain_lines()[0])
    r.write(prereg.CHAIN, json.dumps(entry, sort_keys=True) + "\n")


def backdate_artifact(r: Repo) -> None:
    r.write("artifacts/e1/early.json", json.dumps({"prereg": "prereg/0002-e1-design.md"}))
    r.commit("artifact first")
    r.freeze("prereg/0002-e1-design.md", PREREG_2)
    r.commit("prereg after the data")


def artifact_in_same_commit_as_prereg(r: Repo) -> None:
    r.freeze("prereg/0002-e1-design.md", PREREG_2)
    r.write("artifacts/e1/same.json", json.dumps({"prereg": "prereg/0002-e1-design.md"}))
    r.commit("prereg and data together")


def backdate_jsonl_rows(r: Repo) -> None:
    r.write("artifacts/e1/early.jsonl", json.dumps({"prereg": "prereg/0002-e1-design.md"}) + "\n")
    r.commit("rows first")
    r.freeze("prereg/0002-e1-design.md", PREREG_2)
    r.commit("prereg after")


def chain_entry_committed_after_artifact(r: Repo) -> None:
    r.write("prereg/0002-e1-design.md", PREREG_2)
    r.commit("prereg file without its chain entry")
    r.write("artifacts/e1/mid.json", json.dumps({"prereg": "prereg/0002-e1-design.md"}))
    r.commit("data")
    prereg.freeze(r.root, "prereg/0002-e1-design.md")
    r.commit("freeze later")


def cite_unknown_prereg(r: Repo) -> None:
    r.write("artifacts/e1/x.json", json.dumps({"prereg": "prereg/0009-missing.md"}))


def cite_non_string(r: Repo) -> None:
    r.write("artifacts/e1/x.json", json.dumps({"prereg": ["prereg/0001-e0-grid.md"]}))


def confirmatory_row_without_prereg(r: Repo) -> None:
    r.write("artifacts/e1/x.jsonl", json.dumps({"run_id": 9, "confirmatory": True}) + "\n")


def invalid_artifact_json(r: Repo) -> None:
    r.write("artifacts/e1/broken.json", '{"prereg": "prereg/0001-e0-grid.md",')


def draft_not_in_chain(r: Repo) -> None:
    r.write("prereg/0002-draft.md", "# draft\n")


def misnamed_prereg(r: Repo) -> None:
    r.write("prereg/0002_e1.md", "# misnamed\n")


def frozen_file_deleted(r: Repo) -> None:
    (r.root / "prereg/0001-e0-grid.md").unlink()
    r.commit("remove prereg")


def committed_prereg_removed_from_history(r: Repo) -> None:
    r.write("prereg/0002-withdrawn.md", "# withdrawn\n")
    r.commit("commit a prereg without freezing it")
    (r.root / "prereg/0002-withdrawn.md").unlink()
    r.commit("quietly remove it")


def prereg_not_committed(r: Repo) -> None:
    r.freeze("prereg/0002-e1-design.md", PREREG_2)


def side_branch_edit_discarded_by_merge(r: Repo) -> None:
    r.git("checkout", "-q", "-b", "side")
    r.write("prereg/0001-e0-grid.md", PREREG_1 + "side edit\n")
    r.commit("edit on a side branch")
    r.git("checkout", "-q", "main")
    r.git("merge", "-q", "-s", "ours", "side", "-m", "merge keeping main")


def invalid_json_block_frozen(r: Repo) -> None:
    r.write("prereg/0002-bad.md", '```json e1-design\n{"factors": }\n```\n')
    force_entry(r, "prereg/0002-bad.md")
    r.commit()


def untagged_invalid_json_block(r: Repo) -> None:
    r.write("prereg/0002-bad.md", "~~~json\n[1, 2,\n~~~\n")
    force_entry(r, "prereg/0002-bad.md")
    r.commit()


def amendment_citing_later(r: Repo) -> None:
    r.write("prereg/0002-amend.md", "---\namends: 0003\ndate: 2026-10-23\n---\nx\n")
    force_entry(r, "prereg/0002-amend.md")
    r.commit()


def amendment_line_outside_front_matter(r: Repo) -> None:
    r.write("prereg/0002-amend.md", "# amendment\n\namends: 0001\n")
    force_entry(r, "prereg/0002-amend.md")
    r.commit()


def brief_edited(r: Repo) -> None:
    r.write("docs/BUILD_BRIEF.md", BRIEF + "one more line\n")


def brief_edited_and_rehashed(r: Repo) -> None:
    text = BRIEF + "a quiet change\n"
    r.write("docs/BUILD_BRIEF.md", text)
    r.write("docs/BUILD_BRIEF.sha256", hashlib.sha256(text.encode()).hexdigest() + "\n")
    r.commit("edit the brief")


MUTATIONS: list[tuple[str, Callable[[Repo], None], str]] = [
    ("edit-frozen-prereg-and-commit", edit_frozen_and_commit, "c"),
    ("edit-frozen-prereg-in-working-tree", edit_frozen_in_working_tree, "b"),
    ("edit-frozen-prereg-and-rehash-chain", edit_frozen_and_rehash_chain, "c"),
    ("chain-sha256-tampered", tamper_chain_sha, "a"),
    ("chain-frozen-at-backdated", tamper_chain_frozen_at, "a"),
    ("chain-reordered", reorder_chain, "a"),
    ("chain-reordered-rehashed-and-committed", reorder_chain_rehashed_and_committed, "c"),
    ("chain-truncated-and-committed", truncate_chain, "c"),
    ("chain-line-not-canonical", noncanonical_chain_line, "a"),
    ("artifact-backdated-before-prereg", backdate_artifact, "d"),
    ("artifact-in-same-commit-as-prereg", artifact_in_same_commit_as_prereg, "d"),
    ("jsonl-rows-backdated-before-prereg", backdate_jsonl_rows, "d"),
    ("chain-entry-committed-after-artifact", chain_entry_committed_after_artifact, "d"),
    ("artifact-cites-unknown-prereg", cite_unknown_prereg, "d"),
    ("artifact-cites-non-string", cite_non_string, "d"),
    ("confirmatory-row-without-prereg", confirmatory_row_without_prereg, "d"),
    ("artifact-json-unparseable", invalid_artifact_json, "d"),
    ("draft-prereg-not-in-chain", draft_not_in_chain, "b"),
    ("misnamed-prereg-file", misnamed_prereg, "b"),
    ("frozen-prereg-deleted", frozen_file_deleted, "b"),
    ("committed-prereg-removed-from-history", committed_prereg_removed_from_history, "c"),
    ("frozen-prereg-not-committed", prereg_not_committed, "c"),
    ("side-branch-edit-discarded-by-merge", side_branch_edit_discarded_by_merge, "c"),
    ("frozen-prereg-with-invalid-json-block", invalid_json_block_frozen, "f"),
    ("untagged-invalid-json-block", untagged_invalid_json_block, "f"),
    ("amendment-cites-later-prereg", amendment_citing_later, "e"),
    ("amends-line-outside-front-matter", amendment_line_outside_front_matter, "e"),
    ("brief-edited", brief_edited, "g"),
    ("brief-edited-rehashed-and-committed", brief_edited_and_rehashed, "g"),
]


@pytest.mark.parametrize(("name", "mutate", "check"), MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_mutation_is_detected(
    seeded: Repo, name: str, mutate: Callable[[Repo], None], check: str
) -> None:
    assert seeded.verify()[1] == 0, "the unmutated seed must verify"
    mutate(seeded)
    lines, problems = seeded.verify()
    assert problems > 0, f"mutation {name} survived:\n" + "\n".join(lines)
    assert lines[-1] == f"PREREG-VERIFY: FAIL {problems} problems"
    assert failed(lines, check), f"expected check [{check}] to fail:\n" + "\n".join(lines)


def test_not_a_git_repository_fails(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    lines, problems = prereg.verify(tmp_path)
    assert problems > 0 and failed(lines, "c") and failed(lines, "g")


def test_root_must_be_the_top_of_the_work_tree(seeded: Repo) -> None:
    lines, problems = prereg.verify(seeded.root / "docs")
    assert problems > 0
    assert any("is not the top of its git work tree" in line for line in lines)
