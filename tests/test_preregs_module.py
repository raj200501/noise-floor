"""Frozen-preregistration checks for confirmatory runners (noisefloor.preregs)."""

import json
import subprocess
from pathlib import Path

import pytest

from noisefloor import preregs


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_require_frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preregs, "REPO", tmp_path)
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@example.com")
    git(tmp_path, "config", "user.name", "t")
    (tmp_path / "prereg").mkdir()
    doc = tmp_path / "prereg" / "0001-x.md"
    doc.write_text("# x\n")
    chain = tmp_path / "prereg" / "CHAIN.jsonl"
    with pytest.raises(ValueError, match="not frozen"):
        preregs.require_frozen(doc, chain, tmp_path)
    entry = {"file": "prereg/0001-x.md", "seq": 1, "sha256": preregs.file_sha256(doc)}
    chain.write_text(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="not committed as frozen"):
        preregs.require_frozen(doc, chain, tmp_path)
    git(tmp_path, "add", "prereg/0001-x.md")
    git(tmp_path, "commit", "-q", "-m", "doc only")
    with pytest.raises(ValueError, match="chain entry"):
        preregs.require_frozen(doc, chain, tmp_path)
    git(tmp_path, "add", "prereg/CHAIN.jsonl")
    git(tmp_path, "commit", "-q", "-m", "chain")
    assert preregs.require_frozen(doc, chain, tmp_path)["seq"] == 1
    assert preregs.chain_entry(doc, chain)["sha256"] == entry["sha256"]
    doc.write_text("# edited\n")
    with pytest.raises(ValueError, match="changed after it was frozen"):
        preregs.require_frozen(doc, chain, tmp_path)
    assert preregs.chain_entry(doc, tmp_path / "missing.jsonl") is None


def test_repo_prereg_is_frozen() -> None:
    entry = preregs.require_frozen(preregs.REPO / "prereg" / "0003-e0-coverage-grid-part1.md")
    assert entry["file"] == "prereg/0003-e0-coverage-grid-part1.md"


def test_amends_reads_the_front_matter(tmp_path) -> None:
    from noisefloor import preregs

    a = tmp_path / "0007-a.md"
    a.write_text("---\namends: 0004\ndate: 2026-11-20\n---\n# a\n")
    b = tmp_path / "0008-b.md"
    b.write_text("# no front matter\namends: 0004\n")
    assert preregs.amends(a) == "0004" and preregs.amends(b) is None


def test_amendments_with_reads_the_chain_in_order(tmp_path) -> None:
    import json

    from noisefloor import preregs

    one = tmp_path / "0007-a.md"
    one.write_text("---\namends: 0004\n---\n\n```json llama-server\n{}\n```\n")
    two = tmp_path / "0008-b.md"
    two.write_text("---\namends: 0004\n---\n# no block\n")
    three = tmp_path / "0009-c.md"
    three.write_text("---\namends: 0003\n---\n\n```json llama-server\n{}\n```\n")
    chain = tmp_path / "CHAIN.jsonl"
    rows = [{"file": p.name, "seq": i} for i, p in enumerate((three, two, one), start=1)]
    chain.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert preregs.chain_files(chain, repo=tmp_path) == [three, two, one]
    assert preregs.amendments_with("0004", "llama-server", chain, repo=tmp_path) == [one]
    assert preregs.amendments_with("0003", "llama-server", chain, repo=tmp_path) == [three]
    assert preregs.chain_files(tmp_path / "missing.jsonl") == []
