"""Provenance stamps of simulation rows (no model, no network)."""

import json
import subprocess
from pathlib import Path

import pytest

from noisefloor import provenance


def git_repo(root: Path) -> Path:
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "PATH": "/usr/bin:/bin",
    }
    (root / "src").mkdir(parents=True)
    (root / "src" / "a.py").write_text("x = 1\n")
    (root / "uv.lock").write_text("lock\n")
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "c"]):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, env=env)
    return root


def test_environment_hashes_code_and_libraries(tmp_path: Path) -> None:
    repo = git_repo(tmp_path / "r")
    first = provenance.environment(repo)
    assert len(first["env"]) == 16 and len(first["commit"]) == 40 and not first["src_dirty"]
    assert first["key"]["src_tree"] and first["key"]["uv_lock"] and first["key"]["numpy"]
    (repo / "notes.txt").write_text("unrelated\n")
    assert provenance.environment(repo)["env"] == first["env"]  # only src/ and the lock count
    assert not provenance.environment(repo)["src_dirty"]
    (repo / "src" / "a.py").write_text("x = 2\n")
    edited = provenance.environment(repo)
    assert edited["src_dirty"] and edited["env"] == first["env"]  # HEAD's tree, flagged dirty


def test_stamp_reads_the_process_environment_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    fake = {"env": "e" * 16, "key": {}, "commit": "c" * 40, "src_dirty": False}

    def environment() -> dict[str, object]:
        calls.append(1)
        return fake

    monkeypatch.setattr(provenance, "_PROCESS", None)
    monkeypatch.setattr(provenance, "environment", environment)
    a, b = provenance.stamp(), provenance.stamp()
    assert calls == [1] and a["env"] == b["env"] == "e" * 16 and a["commit"] == "c" * 12
    assert a["src_dirty"] is False and a["at"].endswith("+00:00")


def test_single_environment() -> None:
    one = {"env": "a" * 16, "src_dirty": False}
    assert provenance.single_environment([one, dict(one)]) == "a" * 16
    assert provenance.single_environment([None, None]) is None
    assert provenance.single_environment([]) is None
    with pytest.raises(ValueError, match="2 environments"):
        provenance.single_environment([one, {"env": "b" * 16, "src_dirty": False}])
    with pytest.raises(ValueError, match="2 environments"):
        provenance.single_environment([one, None])
    dirty = {**one, "src_dirty": True}
    with pytest.raises(ValueError, match="uncommitted changes under src/"):
        provenance.single_environment([one, dirty])
    assert provenance.single_environment([one, dirty], allow_dirty=True) == "a" * 16


def test_library_versions_without_r(monkeypatch: pytest.MonkeyPatch) -> None:
    import evalstats.glmm

    def broken() -> dict[str, str]:
        raise RuntimeError("no R here")

    monkeypatch.setattr(evalstats.glmm, "r_versions", broken)
    out = provenance.library_versions()
    assert out["r"] is None and out["lme4"] is None and out["python"]
    json.dumps(out)
