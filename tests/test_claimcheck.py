# Adapted from raj200501/raise-v1-public tests/mutation_test.py (Apache-2.0); changes: pytest
# cases instead of a standalone runner; the claimcheck cases (fabricated number, number in a code
# block, number next to a URL, exemption without a reason, exemption admits exactly one string)
# were rewritten for the registry-and-marker design, and cases were added for Markdown hiding
# places (emphasis, tables, headings, inline code, HTML comments, link targets, signs).
"""Tests for tools/claimcheck.py (gate G10, section 9 rule 1).

A seeded site passes; each mutation adds one way a hand-typed or fabricated number could reach a
document, and the check must fail naming the file, line, and token.
"""

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


cc = load("claimcheck")

BRIEF = "`make ship-check` must print PASS for all 14 gates, checked on a fresh clone.\n"
README = """# noise-floor

Task variance share was <!-- claim:e1.vc.task -->0.31 across <!-- claim:e1.runs -->1,042 runs.

Requires Python 3.11 or later. See [the report](docs/REPORT.md) and https://example.org/r/42.
"""
REPORT = """## 6. Results

The interval was [<!-- claim:e1.vc.task.lo -->0.12, <!-- claim:e1.vc.task.hi -->0.47].

1. Runs are listed in `artifacts/e1/runs.jsonl` at commit 4598e76c.

```python
seed = 7
```
"""
EXEMPT = """
[[exempt]]
text = "Python 3.11 or later"
docs = ["README.md"]
category = "version"
reason = "minimum supported interpreter, not a measurement"

[[exempt]]
text = "## 6. Results"
docs = ["docs/REPORT.md"]
category = "section-number"
reason = "section heading number"
"""
CLAIMS = [
    {"id": "e1.vc.task", "pointer": "/components/task/estimate", "format": ".2f"},
    {"id": "e1.vc.task.lo", "pointer": "/components/task/lo", "format": ".2f"},
    {"id": "e1.vc.task.hi", "pointer": "/components/task/hi", "format": ".2f"},
]


class Site:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, rel: str, text: str) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def append(self, rel: str, text: str) -> None:
        with (self.root / rel).open("a", encoding="utf-8") as fh:
            fh.write(text)

    def registry(self) -> dict[str, Any]:
        data: dict[str, Any] = json.loads((self.root / cc.REGISTRY).read_text())
        return data

    def save_registry(self, data: dict[str, Any]) -> None:
        self.write(cc.REGISTRY, json.dumps(data, indent=2))

    def check(self, docs: tuple[str, ...] = ("README.md", "docs/REPORT.md")) -> list[str]:
        lines, problems, _ = cc.run(self.root, list(docs))
        assert (problems == 0) == lines[-1].startswith("CLAIMCHECK: PASS")
        return lines


def build_registry(root: Path, defs: list[dict[str, object]]) -> dict[str, object]:
    claims = []
    for d in defs:
        claim = {"artifact": "artifacts/e1/variance_components.json", **d}
        value, digest = cc.derive_claim(root, claim)
        claims.append({**claim, "value": value, "artifact_sha256": digest})
    return {"schema": cc.SCHEMA, "generator": "tools/make_claims.py", "claims": claims}


@pytest.fixture
def site(tmp_path: Path) -> Site:
    s = Site(tmp_path)
    s.write("docs/BUILD_BRIEF.md", BRIEF)
    s.write("tools/make_claims.py", "# writes artifacts/claims.json\n")
    vc = {"components": {"task": {"estimate": 0.3114, "lo": 0.1203, "hi": 0.4719}}}
    s.write("artifacts/e1/variance_components.json", json.dumps(vc))
    rows = [{"run_id": i, "confirmatory": True} for i in range(1042)]
    rows += [{"run_id": 2000 + i, "confirmatory": False} for i in range(5)]
    s.write("artifacts/e1/runs.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    runs = {
        "id": "e1.runs",
        "artifact": "artifacts/e1/runs.jsonl",
        "pointer": "",
        "derive": "count",
        "where": {"confirmatory": True},
        "format": ",d",
    }
    s.save_registry(build_registry(tmp_path, [*CLAIMS, runs]))
    s.write("README.md", README)
    s.write("docs/REPORT.md", REPORT)
    s.write(cc.EXEMPTIONS, EXEMPT)
    return s


def problems_of(lines: list[str]) -> list[str]:
    return [line[4:] for line in lines if line.startswith("  - ")]


# --------------------------------------------------------------------------- controls


def test_seeded_site_passes(site: Site) -> None:
    lines = site.check()
    assert lines[-1] == "CLAIMCHECK: PASS 4 claims", lines
    assert "README.md: 2 claims, 1 exempt numbers" in "\n".join(lines)


def test_registry_values_follow_the_artifact(site: Site) -> None:
    values = {c["id"]: c["value"] for c in site.registry()["claims"]}
    assert values == {
        "e1.vc.task": "0.31",
        "e1.vc.task.lo": "0.12",
        "e1.vc.task.hi": "0.47",
        "e1.runs": "1,042",
    }


def test_cli_exit_codes(site: Site) -> None:
    cmd = [sys.executable, str(TOOLS / "claimcheck.py"), "--root", str(site.root)]
    ok = subprocess.run(
        [*cmd, "README.md", "docs/REPORT.md"], capture_output=True, text=True, check=False
    )
    assert ok.returncode == 0 and ok.stdout.strip().endswith("CLAIMCHECK: PASS 4 claims")
    site.append("README.md", "We reached 0.981.\n")
    bad = subprocess.run([*cmd, "README.md"], capture_output=True, text=True, check=False)
    assert bad.returncode == 1 and "CLAIMCHECK: FAIL" in bad.stdout.splitlines()[-1]


@pytest.mark.parametrize(
    "line",
    [
        "Code `seed=0`, `n_boot=2000`, and `--max-turns 40` is not prose.",
        "A link [to the plan](docs/PLAN.md#section-10) and an image ![chart](figs/f1.svg).",
        'An HTML tag <img src="docs/fig.svg" width="720"> carries no rendered number.',
        "A hidden note <!-- reviewer: check 0.5 later --> is not rendered.",
        "Digests 9f2c4e1ab3 and 954acc7fc19fb4fc3dcc3c347ad52478bd22fa10 are identifiers.",
        "Preprint arXiv:2401.12345v2 and DOI 10.48550/arXiv.2401.12345 are identifiers.",
        "Identifiers G10, E1, sha256, v0.1, and py311 carry no number token.",
        "Escaped \\<!-- claim is text --> stays text without digits.",
        "3. A list marker is markup.",
        "A bare URL https://example.org/runs/12345?page=2 is an address.",
        "Reference [link][ref] with a label.\n\n[ref]: https://example.org/v/3",
        "Steps:\n\n1. one\n2. two\n   continued\n3. three",
        "## Steps\n2. starts after a heading\n3. and continues",
        "Text before a list\n1. a list may interrupt a paragraph\n2. the next item",
        "- a bullet\n- another\n\n10. a new list after a blank line",
    ],
)
def test_non_claim_digits_are_not_tokens(site: Site, line: str) -> None:
    site.append("README.md", "\n" + line + "\n")
    lines = site.check()
    assert lines[-1] == "CLAIMCHECK: PASS 4 claims", problems_of(lines)


def test_code_language_block_is_skipped_but_output_block_is_not(site: Site) -> None:
    site.append("README.md", "\n```bash\nmake -j4 demo\n```\n")
    assert site.check()[-1].startswith("CLAIMCHECK: PASS")
    site.append("README.md", "\n```text\nestimate 0.31 (0.12 to 0.47)\n```\n")
    probs = problems_of(site.check())
    assert len(probs) == 3 and all("in a code block" in p for p in probs)


def test_generated_block_must_equal_its_artifact(site: Site) -> None:
    site.write("artifacts/demo/quickstart.txt", "estimate 0.31 (0.12 to 0.47)\n")
    site.append(
        "README.md",
        "\n```text from=artifacts/demo/quickstart.txt\nestimate 0.31 (0.12 to 0.47)\n```\n",
    )
    lines = site.check()
    assert lines[-1] == "CLAIMCHECK: PASS 4 claims"
    assert "3 numbers in generated blocks" in "\n".join(lines)
    site.write("artifacts/demo/quickstart.txt", "estimate 0.29 (0.10 to 0.45)\n")
    assert any("differs from artifacts/demo/quickstart.txt" in p for p in problems_of(site.check()))


def test_brief_quote_and_date_exemptions(site: Site) -> None:
    site.append(
        cc.EXEMPTIONS,
        '\n[[exempt]]\ntext = "must print PASS for all 14 gates"\ndocs = ["README.md"]\n'
        'category = "brief-quote"\nreason = "quoted from section 6"\n'
        '\n[[exempt]]\ntext = "2026-10-23"\ndocs = ["README.md"]\ncategory = "date"\n'
        'reason = "release date"\n',
    )
    quote = "> `make ship-check` must print PASS for all 14 gates."
    site.append("README.md", f"\n{quote}\n\n2026-10-23\n")
    lines = site.check()
    assert lines[-1] == "CLAIMCHECK: PASS 4 claims", problems_of(lines)


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("a -0.12 b", ["-0.12"]),
        ("(MINUS3)", ["MINUS3"]),
        ("1,042 runs and .5 of them", ["1,042", ".5"]),
        ("on 2026-10-23", ["2026", "10", "23"]),
        ("Python 3.11.", ["3.11"]),
        ("12pp and 3x", ["12", "3"]),
        ("_7_ and **8**", ["7", "8"]),
        ("1e-3", ["1", "3"]),
        ("x86 G10 v0.1 sha256 a.31", []),
        ("x-0.5 and 5-3", ["0.5", "5", "3"]),
    ],
)
def test_number_tokens(text: str, tokens: list[str]) -> None:
    text = text.replace("MINUS", cc.MINUS)
    found = [m.group(0) for m in cc.TOKEN_RE.finditer(text)]
    assert found == [t.replace("MINUS", cc.MINUS) for t in tokens]


# --------------------------------------------------------------------------- mutations


def fabricated_number(s: Site) -> None:
    s.append("README.md", "\nWe also reached 0.981 on the held-out slice.\n")


def fabricated_number_in_table(s: Site) -> None:
    s.append("docs/REPORT.md", "\n| model | rate |\n|---|---|\n| m1 | 0.64 |\n")


def fabricated_number_in_heading(s: Site) -> None:
    s.append("docs/REPORT.md", "\n## 31 of 40 tasks passed\n")


def number_in_underscore_emphasis(s: Site) -> None:
    s.append("README.md", "\nThe rate was _0.73_ overall.\n")


def number_in_bold(s: Site) -> None:
    s.append("README.md", "\nThe rate was **0.73** overall.\n")


def leading_dot_decimal(s: Site) -> None:
    s.append("README.md", "\nAgreement was .87 on the slice.\n")


def number_with_unit_letters(s: Site) -> None:
    s.append("README.md", "\nThe gap closed by 12pp, a 3x change.\n")


def sign_outside_the_marker(s: Site) -> None:
    s.append("README.md", "\nThe change was -<!-- claim:e1.vc.task -->0.31.\n")


def marker_value_mismatch(s: Site) -> None:
    text = (s.root / "README.md").read_text().replace("-->0.31", "-->0.32")
    s.write("README.md", text)


def marker_covers_part_of_token(s: Site) -> None:
    s.append("README.md", "\nPrecision overclaim <!-- claim:e1.vc.task -->0.3114.\n")


def unknown_claim_id(s: Site) -> None:
    s.append("README.md", "\nSee <!-- claim:e9.made.up -->0.55.\n")


def malformed_marker(s: Site) -> None:
    s.append("README.md", "\nSee <!-- claim e1.vc.task -->0.31.\n")


def ignore_line_escape_hatch(s: Site) -> None:
    s.append("README.md", "\nWe reached 0.981. <!-- claimcheck:ignore-line trust me -->\n")


def number_in_untagged_code_block(s: Site) -> None:
    s.append("README.md", "\n```\nfinal_score = 0.777\n```\n")


def numeric_inline_code(s: Site) -> None:
    s.append("README.md", "\nThe score was `0.77`.\n")


def number_next_to_url(s: Site) -> None:
    s.append("README.md", "\nSee https://example.com/run/12345 which reports 0.633 accuracy.\n")


def number_in_link_text(s: Site) -> None:
    s.append("README.md", "\nSee [0.66 accuracy](docs/REPORT.md).\n")


def unclosed_comment_hides_text(s: Site) -> None:
    s.append("README.md", "\n<!-- the rest is hidden\n")


def unclosed_code_fence(s: Site) -> None:
    s.append("README.md", "\n```python\nprint(1)\n")


def marker_inside_link_target(s: Site) -> None:
    s.append("README.md", "\nSee [x](docs/<!-- claim:e1.vc.task -->a.md).\n")


def exemption_without_reason(s: Site) -> None:
    s.append(cc.EXEMPTIONS, '\n[[exempt]]\ntext = "n = 9"\ndocs = ["README.md"]\n')
    s.append(cc.EXEMPTIONS, 'category = "cap"\nreason = "  "\n')
    s.append("README.md", "\nn = 9\n")


def exemption_with_unknown_category(s: Site) -> None:
    s.append(cc.EXEMPTIONS, '\n[[exempt]]\ntext = "rate 0.9"\ndocs = ["README.md"]\n')
    s.append(cc.EXEMPTIONS, 'category = "result"\nreason = "looks fine"\n')
    s.append("README.md", "\nrate 0.9\n")


def exemption_for_a_bare_number(s: Site) -> None:
    s.append(cc.EXEMPTIONS, '\n[[exempt]]\ntext = "0.9"\ndocs = ["README.md"]\n')
    s.append(cc.EXEMPTIONS, 'category = "threshold"\nreason = "a threshold"\n')
    s.append("README.md", "\nthe threshold 0.9\n")


def exemption_with_unknown_key(s: Site) -> None:
    s.append(cc.EXEMPTIONS, '\n[[exempt]]\ntext = "cap 30"\ndocs = ["README.md"]\n')
    s.append(cc.EXEMPTIONS, 'category = "cap"\nreason = "daily cap"\nscope = "all"\n')
    s.append("README.md", "\ncap 30\n")


def brief_quote_not_in_brief(s: Site) -> None:
    s.append(cc.EXEMPTIONS, '\n[[exempt]]\ntext = "all 15 gates"\ndocs = ["README.md"]\n')
    s.append(cc.EXEMPTIONS, 'category = "brief-quote"\nreason = "quoted"\n')
    s.append("README.md", "\nall 15 gates\n")


def exemption_used_in_other_doc(s: Site) -> None:
    s.append("docs/REPORT.md", "\nRequires Python 3.11 or later.\n")


def exempt_number_reused_in_new_context(s: Site) -> None:
    s.append("README.md", "\nThe pass rate rose by 3.11 points.\n")


def stale_exemption(s: Site) -> None:
    text = (s.root / "README.md").read_text().replace("Requires Python 3.11 or later. ", "")
    s.write("README.md", text)


def missing_document(s: Site) -> None:
    (s.root / "docs/REPORT.md").unlink()


def registry_duplicate_id(s: Site) -> None:
    data = s.registry()
    data["claims"] = [*data["claims"], data["claims"][0]]
    s.save_registry(data)


def registry_artifact_outside_artifacts(s: Site) -> None:
    s.write("results.json", json.dumps({"x": 1}))
    data = s.registry()
    data["claims"][0]["artifact"] = "results.json"
    s.save_registry(data)


def registry_artifact_missing(s: Site) -> None:
    (s.root / "artifacts/e1/variance_components.json").unlink()


def registry_without_generator(s: Site) -> None:
    data = s.registry()
    data["generator"] = "scripts/does_not_exist.py"
    s.save_registry(data)


def registry_bad_schema(s: Site) -> None:
    data = s.registry()
    data["schema"] = "something/else"
    s.save_registry(data)


def registry_unknown_claim_field(s: Site) -> None:
    data = s.registry()
    data["claims"][0]["value_override"] = "0.99"
    s.save_registry(data)


def registry_not_json(s: Site) -> None:
    s.write(cc.REGISTRY, "{")


def number_at_start_of_wrapped_line(s: Site) -> None:
    s.append("README.md", "\nThe number of failing tasks was\n31. The rest passed.\n")


def fake_reference_definition(s: Site) -> None:
    s.append("README.md", "\n[Note]: the accuracy was 0.93 overall\n")


def digit_as_character_reference(s: Site) -> None:
    s.append("README.md", "\nAccuracy was &#48;.93 overall.\n")


def sign_as_named_entity(s: Site) -> None:
    s.append("README.md", "\nThe change was &minus;<!-- claim:e1.vc.task -->0.31.\n")


MUTATIONS: list[tuple[str, Callable[[Site], None], str]] = [
    ("fabricated-number-in-prose", fabricated_number, "README.md:7: number 0.981"),
    ("fabricated-number-in-table", fabricated_number_in_table, "docs/REPORT.md:13: number 0.64"),
    ("fabricated-number-in-heading", fabricated_number_in_heading, "number 31"),
    ("number-in-underscore-emphasis", number_in_underscore_emphasis, "number 0.73"),
    ("number-in-bold", number_in_bold, "number 0.73"),
    ("leading-dot-decimal", leading_dot_decimal, "number .87"),
    ("number-with-unit-letters", number_with_unit_letters, "number 12"),
    ("sign-outside-the-marker", sign_outside_the_marker, "-0.31 is only partly inside"),
    ("marker-value-mismatch", marker_value_mismatch, "the document shows '0.32"),
    ("marker-covers-part-of-token", marker_covers_part_of_token, "0.3114 is only partly"),
    ("unknown-claim-id", unknown_claim_id, "e9.made.up is not in the registry"),
    ("malformed-marker", malformed_marker, "malformed claim marker"),
    ("ignore-line-escape-hatch", ignore_line_escape_hatch, "number 0.981"),
    ("number-in-untagged-code-block", number_in_untagged_code_block, "0.777 in a code block"),
    ("numeric-inline-code", numeric_inline_code, "number 0.77"),
    ("number-next-to-url", number_next_to_url, "number 0.633"),
    ("number-in-link-text", number_in_link_text, "number 0.66"),
    ("unclosed-comment", unclosed_comment_hides_text, "never closed"),
    ("unclosed-code-fence", unclosed_code_fence, "code fence is never closed"),
    ("marker-inside-link-target", marker_inside_link_target, "inside a link target"),
    ("exemption-without-reason", exemption_without_reason, "reason must be"),
    ("exemption-unknown-category", exemption_with_unknown_category, "category 'result'"),
    ("exemption-for-a-bare-number", exemption_for_a_bare_number, "bare number"),
    ("exemption-unknown-key", exemption_with_unknown_key, "unknown keys ['scope']"),
    ("brief-quote-not-in-brief", brief_quote_not_in_brief, "does not occur in"),
    ("exemption-applies-only-to-its-docs", exemption_used_in_other_doc, "number 3.11"),
    ("exempt-number-in-new-context", exempt_number_reused_in_new_context, "number 3.11"),
    ("stale-exemption", stale_exemption, "covers no number"),
    ("missing-document", missing_document, "docs/REPORT.md: document missing"),
    ("registry-duplicate-id", registry_duplicate_id, "id appears twice"),
    ("registry-artifact-outside", registry_artifact_outside_artifacts, "not a relative path"),
    ("registry-artifact-missing", registry_artifact_missing, "does not exist"),
    ("registry-generator-missing", registry_without_generator, "does not exist"),
    ("registry-bad-schema", registry_bad_schema, "schema is 'something/else'"),
    ("registry-unknown-field", registry_unknown_claim_field, "unknown fields"),
    ("registry-not-json", registry_not_json, "not valid JSON"),
    ("number-at-start-of-wrapped-line", number_at_start_of_wrapped_line, "README.md:8: number 31"),
    ("fake-reference-definition", fake_reference_definition, "number 0.93"),
    ("digit-as-character-reference", digit_as_character_reference, "renders '0'"),
    ("sign-as-named-entity", sign_as_named_entity, "renders"),
]


@pytest.mark.parametrize(("name", "mutate", "expect"), MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_mutation_is_detected(
    site: Site, name: str, mutate: Callable[[Site], None], expect: str
) -> None:
    assert site.check()[-1].startswith("CLAIMCHECK: PASS"), "the unmutated seed must pass"
    mutate(site)
    lines = site.check()
    probs = problems_of(lines)
    assert lines[-1] == f"CLAIMCHECK: FAIL {len(probs)} problems", f"{name} survived: {lines}"
    assert any(expect in p for p in probs), f"{name}: no problem mentions {expect!r}: {probs}"
