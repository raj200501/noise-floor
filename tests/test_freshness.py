# Adapted from raj200501/raise-v1-public tests/mutation_test.py (Apache-2.0); changes: pytest
# cases instead of a standalone runner; the freshness cases (stale count that claimcheck still
# passes, missing artifact is a failure not a skip, unresolvable path, unknown operator) were
# rewritten for the claim registry, JSON pointers, and named derivations, and cases were added
# for artifact hashes, re-rendered documents, generated blocks, and --regenerate.
"""Tests for tools/freshness.py (gate G10: every live number is re-derived from its artifact)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"tools_{name}", TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fresh = load("freshness")
cc = fresh.cc

VC = {"components": {"task": {"estimate": 0.3114, "lo": 0.1203, "hi": 0.4719}}}
DEFS: list[dict[str, Any]] = [
    {
        "id": "e1.vc.task",
        "artifact": "artifacts/e1/vc.json",
        "pointer": "/components/task/estimate",
        "format": ".2f",
    },
    {
        "id": "e1.vc.task.ci",
        "artifact": "artifacts/e1/vc.json",
        "pointer": "/components/task",
        "derive": "len",
    },
    {
        "id": "e1.runs",
        "artifact": "artifacts/e1/runs.jsonl",
        "pointer": "",
        "derive": "count",
        "where": {"confirmatory": True},
        "format": ",d",
    },
    {
        "id": "e1.rate",
        "artifact": "artifacts/e1/runs.jsonl",
        "pointer": "",
        "derive": "mean",
        "where": {"confirmatory": True},
        "field": "outcome",
        "format": ".1%",
    },
]
README = """# noise-floor

Task share <!-- claim:e1.vc.task -->0.31 from <!-- claim:e1.vc.task.ci -->3 fields, over
<!-- claim:e1.runs -->1,200 runs with success rate <!-- claim:e1.rate -->25.0%.
"""
REPORT = """# Report

```text from=artifacts/demo/out.txt
estimate 0.31
```
"""


class Site:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, rel: str, text: str) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def registry(self) -> dict[str, Any]:
        data: dict[str, Any] = json.loads((self.root / cc.REGISTRY).read_text())
        return data

    def save_registry(self, data: dict[str, Any]) -> None:
        self.write(cc.REGISTRY, json.dumps(data, indent=2))

    def claim(self, cid: str) -> dict[str, Any]:
        return next(c for c in self.registry()["claims"] if c["id"] == cid)

    def set_claim(self, cid: str, **fields: Any) -> None:
        data = self.registry()
        for c in data["claims"]:
            if c["id"] == cid:
                c.update(fields)
        self.save_registry(data)

    def runs(self, n_conf: int, n_success: int, n_other: int = 4) -> None:
        rows = [
            {"run_id": i, "confirmatory": True, "outcome": i < n_success} for i in range(n_conf)
        ]
        rows += [
            {"run_id": 10_000 + i, "confirmatory": False, "outcome": True} for i in range(n_other)
        ]
        self.write("artifacts/e1/runs.jsonl", "".join(json.dumps(r) + "\n" for r in rows))

    def regenerate(self) -> list[str]:
        lines, problems = fresh.regenerate(self.root)
        assert problems == 0, lines
        return lines

    def check(self, docs: tuple[str, ...] = ("README.md", "docs/REPORT.md")) -> list[str]:
        lines, problems, _ = fresh.run(self.root, list(docs))
        assert (problems == 0) == lines[-1].startswith("FRESHNESS: PASS")
        return lines


@pytest.fixture
def site(tmp_path: Path) -> Site:
    s = Site(tmp_path)
    s.write("tools/freshness.py", "# stands in for the committed generator\n")
    s.write("artifacts/e1/vc.json", json.dumps(VC))
    s.runs(1200, 300)
    s.write("artifacts/demo/out.txt", "estimate 0.31\n")
    s.save_registry({"schema": cc.SCHEMA, "claims": DEFS})
    s.regenerate()
    s.write("README.md", README)
    s.write("docs/REPORT.md", REPORT)
    return s


def problems_of(lines: list[str]) -> list[str]:
    return [line[4:] for line in lines if line.startswith("  - ")]


# --------------------------------------------------------------------------- controls


def test_seeded_site_passes(site: Site) -> None:
    lines = site.check()
    assert lines[-1] == "FRESHNESS: PASS 5 live numbers", lines
    assert "README.md: 4 of 4 claim markers current" in "\n".join(lines)


def test_regenerate_fills_values_and_hashes(site: Site) -> None:
    data = site.registry()
    assert data["generator"] == "tools/freshness.py"
    values = {c["id"]: c["value"] for c in data["claims"]}
    assert values == {
        "e1.vc.task": "0.31",
        "e1.vc.task.ci": "3",
        "e1.runs": "1,200",
        "e1.rate": "25.0%",
    }
    digest = cc.sha256_file(site.root / "artifacts/e1/vc.json")
    assert site.claim("e1.vc.task")["artifact_sha256"] == digest


def test_claimcheck_and_freshness_agree_on_the_seed(site: Site) -> None:
    lines, problems, claims = cc.run(site.root, ["README.md", "docs/REPORT.md"])
    assert problems == 0 and claims == 4, lines


def test_regenerate_refuses_an_underivable_claim(site: Site) -> None:
    site.set_claim("e1.vc.task", pointer="/components/missing")
    before = (site.root / cc.REGISTRY).read_text()
    lines, problems = fresh.regenerate(site.root)
    assert problems == 1 and "no key 'missing'" in "\n".join(lines)
    assert (site.root / cc.REGISTRY).read_text() == before


def test_regenerate_needs_a_registry(tmp_path: Path) -> None:
    lines, problems = fresh.regenerate(tmp_path)
    assert problems == 1 and "does not exist" in lines[0]


def test_cli_exit_codes(site: Site) -> None:
    cmd = [sys.executable, str(TOOLS / "freshness.py"), "--root", str(site.root)]
    ok = subprocess.run(
        [*cmd, "README.md", "docs/REPORT.md"], capture_output=True, text=True, check=False
    )
    assert ok.returncode == 0, ok.stdout
    assert ok.stdout.strip().splitlines()[-1] == "FRESHNESS: PASS 5 live numbers"
    site.write("artifacts/e1/vc.json", json.dumps({"components": {"task": {"estimate": 0.2}}}))
    bad = subprocess.run([*cmd, "README.md"], capture_output=True, text=True, check=False)
    assert bad.returncode == 1 and bad.stdout.strip().splitlines()[-1].startswith("FRESHNESS: FAIL")


# --------------------------------------------------------------------------- derivations


ROWS = [
    {"ok": True, "x": 1, "g": {"arm": "a"}},
    {"ok": 1, "x": 2, "g": {"arm": "b"}},
    {"ok": False, "x": 4.5, "g": {"arm": "a"}},
]


@pytest.mark.parametrize(
    ("name", "where", "fld", "expected"),
    [
        ("len", None, None, 3),
        ("count", None, None, 3),
        ("count", {"ok": True}, None, 1),
        ("count", {"/g/arm": "a"}, None, 2),
        ("sum", None, "x", 7.5),
        ("mean", {"/g/arm": "a"}, "x", 2.75),
        ("min", None, "x", 1),
        ("max", None, "/x", 4.5),
        ("mean", None, "ok", 2 / 3),
    ],
)
def test_derivations(
    name: str, where: dict[str, Any] | None, fld: str | None, expected: float
) -> None:
    assert cc.derive(ROWS, name, where, fld) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("value", "name", "where", "fld", "message"),
    [
        ({"a": 1}, "count", None, None, "needs a list"),
        (ROWS, "count", None, "x", "takes no field"),
        (ROWS, "sum", None, "missing", "no field"),
        (ROWS, "mean", {"ok": "yes"}, "x", "over no rows"),
        ([{"x": "1"}], "sum", None, "x", "non-number"),
        (ROWS, "len", {"ok": True}, None, "takes no where"),
        (ROWS, "median", None, "x", "unknown derivation"),
    ],
)
def test_derivation_errors(
    value: Any, name: str, where: dict[str, Any] | None, fld: str | None, message: str
) -> None:
    with pytest.raises(cc.ClaimError, match=message):
        cc.derive(value, name, where, fld)


def test_json_pointer() -> None:
    doc = {"a/b": {"m~n": [10, 20]}, "": 5}
    assert cc.resolve_pointer(doc, "/a~1b/m~0n/1") == 20
    assert cc.resolve_pointer(doc, "/") == 5
    assert cc.resolve_pointer(doc, "") is doc
    for bad in ("/a~1b/m~0n/2", "/a~1b/m~0n/01", "/nope", "a", "/a~1b/m~0n/1/x"):
        with pytest.raises(cc.ClaimError):
            cc.resolve_pointer(doc, bad)


@pytest.mark.parametrize(
    ("value", "spec", "expected"),
    [(0.3114, ".2f", "0.31"), (1200, ",d", "1,200"), (0.25, ".1%", "25.0%"), (7, None, "7")],
)
def test_format_value(value: Any, spec: str | None, expected: str) -> None:
    assert cc.format_value(value, spec) == expected


@pytest.mark.parametrize(
    ("value", "spec"),
    [(0.5, None), (True, None), (None, ".2f"), ([1], None), (float("nan"), ".2f"), ("a", ".2f")],
)
def test_format_value_errors(value: Any, spec: str | None) -> None:
    with pytest.raises(cc.ClaimError):
        cc.format_value(value, spec)


# --------------------------------------------------------------------------- mutations


def artifact_value_changed_after_render(s: Site) -> None:
    vc = json.loads(json.dumps(VC))
    vc["components"]["task"]["estimate"] = 0.2875
    s.write("artifacts/e1/vc.json", json.dumps(vc))


def registry_regenerated_but_doc_not_rendered(s: Site) -> None:
    artifact_value_changed_after_render(s)
    s.regenerate()


def rows_appended_after_render(s: Site) -> None:
    s.runs(1250, 300)


def registry_value_and_doc_hand_edited(s: Site) -> None:
    s.set_claim("e1.vc.task", value="0.35")
    s.write("README.md", README.replace("-->0.31", "-->0.35"))


def artifact_hash_hand_edited(s: Site) -> None:
    s.set_claim("e1.vc.task", artifact_sha256="0" * 64)


def artifact_deleted(s: Site) -> None:
    (s.root / "artifacts/e1/vc.json").unlink()


def artifact_not_json(s: Site) -> None:
    s.write("artifacts/e1/vc.json", "{")


def pointer_no_longer_resolves(s: Site) -> None:
    s.write("artifacts/e1/vc.json", json.dumps({"components": {"tasks": VC["components"]["task"]}}))


def derivation_over_non_list(s: Site) -> None:
    s.set_claim("e1.runs", artifact="artifacts/e1/vc.json")


def float_claim_without_format(s: Site) -> None:
    data = s.registry()
    for c in data["claims"]:
        if c["id"] == "e1.vc.task":
            del c["format"]
    s.save_registry(data)


def unknown_claim_in_doc(s: Site) -> None:
    s.write("README.md", README + "\nAlso <!-- claim:e9.ghost -->0.5.\n")


def generated_block_artifact_changed(s: Site) -> None:
    s.write("artifacts/demo/out.txt", "estimate 0.29\n")


def missing_document(s: Site) -> None:
    (s.root / "docs/REPORT.md").unlink()


def unknown_derivation(s: Site) -> None:
    s.set_claim("e1.runs", derive="median")


def registry_claim_points_outside_artifacts(s: Site) -> None:
    s.write("other/vc.json", json.dumps(VC))
    s.set_claim("e1.vc.task", artifact="other/vc.json")


MUTATIONS: list[tuple[str, Callable[[Site], None], str]] = [
    ("artifact-value-changed-after-render", artifact_value_changed_after_render, "changed since"),
    (
        "artifact-changed-registry-regenerated",
        registry_regenerated_but_doc_not_rendered,
        "re-render",
    ),
    ("rows-appended-after-render", rows_appended_after_render, "e1.runs is stale"),
    ("registry-and-doc-hand-edited", registry_value_and_doc_hand_edited, "e1.vc.task is stale"),
    ("artifact-hash-hand-edited", artifact_hash_hand_edited, "changed since"),
    ("artifact-deleted", artifact_deleted, "does not exist"),
    ("artifact-not-json", artifact_not_json, "cannot re-derive"),
    ("pointer-no-longer-resolves", pointer_no_longer_resolves, "no key 'task'"),
    ("derivation-over-non-list", derivation_over_non_list, "needs a list"),
    ("float-claim-without-format", float_claim_without_format, "needs a format"),
    ("unknown-claim-in-doc", unknown_claim_in_doc, "e9.ghost is not in the registry"),
    ("generated-block-artifact-changed", generated_block_artifact_changed, "code block differs"),
    ("missing-document", missing_document, "docs/REPORT.md: document missing"),
    ("unknown-derivation", unknown_derivation, "derive 'median'"),
    ("claim-points-outside-artifacts", registry_claim_points_outside_artifacts, "under artifacts/"),
]


@pytest.mark.parametrize(("name", "mutate", "expect"), MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_mutation_is_detected(
    site: Site, name: str, mutate: Callable[[Site], None], expect: str
) -> None:
    assert site.check()[-1].startswith("FRESHNESS: PASS"), "the unmutated seed must pass"
    mutate(site)
    lines = site.check()
    probs = problems_of(lines)
    assert lines[-1] == f"FRESHNESS: FAIL {len(probs)} problems", f"{name} survived: {lines}"
    assert any(expect in p for p in probs), f"{name}: no problem mentions {expect!r}: {probs}"


def test_stale_value_passes_claimcheck_but_fails_freshness(site: Site) -> None:
    # The registry and the document agree, so claimcheck has no grounds to object. The artifact
    # moved on. That gap is why freshness is a separate gate.
    rows_appended_after_render(site)
    _, cc_problems, _ = cc.run(site.root, ["README.md", "docs/REPORT.md"])
    assert cc_problems == 0
    assert site.check()[-1].startswith("FRESHNESS: FAIL")


def test_malformed_claim_fields_fail_without_crashing(site: Site) -> None:
    site.set_claim("e1.runs", where=["confirmatory"])
    site.set_claim("e1.vc.task", pointer=5)
    probs = problems_of(site.check())
    assert any("where must be an object" in p for p in probs)
    assert any("pointer must be a string" in p for p in probs)
