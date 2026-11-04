# ruff: noqa
"""Verification follow-ups (workflow wf_42976fca-8a6), applied after its verifiers finish.

Run from the repository root: python3 <this file>
"""

from pathlib import Path

S = Path("/tmp/claude-0/-home-user-noise-floor/b5fd6b69-b3b0-56ef-9f17-e15c370adddf/scratchpad")


def patch(path: Path, pairs: list[tuple[str, str]]) -> None:
    text = path.read_text()
    for old, new in pairs:
        assert text.count(old) == 1, (str(path), old[:120])
        text = text.replace(old, new)
    path.write_text(text)
    print("patched", path)


# ---------------------------------------------------------------- tools/ship_check.py
patch(
    Path("tools/ship_check.py"),
    [
        (
            """E1_NUMBER = "0004"  # the E1 preregistration, whose analysis names each kind's method
# The recovery-config setting that is a GLMM kind's method (its cells' interval_type label
# does not change with it): a replacement of the kind must change one of these.
GLMM_RULE_KEYS = {
    "random": ("boot_interval",),
    "fixed": ("fixed_interval", "box_interval"),
    "share": ("share_interval",),
    "mde": ("mde_interval",),
}
""",
            '''E1_NUMBER = "0004"  # the E1 preregistration, whose analysis names each kind's method
GLMM_KINDS = ("random", "fixed", "share", "mde")


def glmm_rule(kind: str, settings: dict[str, Any]) -> tuple[Any, ...]:
    """The recovery-config setting that is a GLMM kind's method.

    Its cells' interval_type label does not change with it, so a replacement of the kind
    must change this. Under the task jackknife the bootstrap box rule (``box_interval``)
    has no effect, so a fixed contribution's rule is the jackknife, or the box with its
    ``box_interval``.
    """
    if kind == "fixed":
        rule = settings.get("fixed_interval", "box")
        return ("jackknife",) if rule == "jackknife" else ("box", settings.get("box_interval"))
    key = {"random": "boot_interval", "share": "share_interval", "mde": "mde_interval"}[kind]
    return (settings.get(key),)
''',
        ),
        (
            '''    ``grid`` maps every frozen cell to (its preregistration, its cell entry). A GLMM kind
    must change its rule in ``prereg``'s own recovery-config block; another kind's new
    cells must name another (method, interval_type) than every retired cell.
    """
    if kind in GLMM_RULE_KEYS:
        mine = _settings(ctx, prereg)
        if not mine or not new:
            return False
        sources = {grid[c][0] for c in gone if c in grid}
        return all(
            any(mine.get(k) != _settings(ctx, src).get(k) for k in GLMM_RULE_KEYS[kind])
            for src in sources
        )''',
            '''    ``grid`` maps every frozen cell to (its preregistration, its cell entry). A GLMM kind
    must change its rule (:func:`glmm_rule`) in ``prereg``'s own recovery-config block;
    another kind's new cells must name another (method, interval_type) than every retired
    cell.
    """
    if kind in GLMM_KINDS:
        mine = _settings(ctx, prereg)
        if not mine or not new:
            return False
        sources = {grid[c][0] for c in gone if c in grid}
        return all(
            glmm_rule(kind, mine) != glmm_rule(kind, _settings(ctx, src)) for src in sources
        )''',
        ),
        (
            """    before the first confirmatory E1 run, beside new headline cells of the kind that test
    another method (:func:`method_replaced`) and a frozen amendment of the E1
    preregistration from before that run (which allows the new analysis code).""",
            """    before the first confirmatory E1 run, beside new headline cells of the kind that test
    another method (:func:`method_replaced`) and a frozen amendment of the E1
    preregistration from before that run whose ```json analysis-code``` block allows the
    new analysis code (an amendment of 0004 for another purpose, such as a rebuilt
    llama-server, does not count).""",
        ),
        (
            """            e1_amendments = [
                f
                for f in chain_files(ctx)
                if amends(ctx, f) == E1_NUMBER
                and (first is None or chain_frozen_at(ctx, f) < first)
            ]
            need(
                bool(e1_amendments),
                f"{prereg} retires headline cells, but no amendment of prereg {E1_NUMBER} "
                "frozen before E1 data allows the new method",
            )""",
            """            e1_amendments = [
                f
                for f in chain_files(ctx)
                if amends(ctx, f) == E1_NUMBER
                and has_block(ctx, f, "analysis-code")
                and (first is None or chain_frozen_at(ctx, f) < first)
            ]
            need(
                bool(e1_amendments),
                f"{prereg} retires headline cells, but no amendment of prereg {E1_NUMBER} "
                "frozen before E1 data allows the new method",
            )""",
        ),
        (
            """    # The one analysis-code hash the E1 analysis requires (prereg 0004), and the same one
    # E0 part 2 ran once it is banked.
    code = data.get("analysis_code")
    need(
        isinstance(code, str) and re.fullmatch(r"[0-9a-f]{16}", code) is not None,
        "no 16-hex analysis_code",
    )
    e0 = rel(ctx, E0_PART2_ARTIFACT)
    if e0.exists():
        other = load_json(e0).get("analysis_code")
        need(other == code, f"G6 ran analysis code {code}, E0 part 2 {other}")""",
            """    # The one analysis-code hash the E1 analysis requires (prereg 0004): the one E0 part 2
    # ran once it is banked, or, for a rerun of G6 under an amendment of its preregistration
    # (a replaced GLMM method), one that a frozen amendment of 0004 allows.
    code = data.get("analysis_code")
    need(
        isinstance(code, str) and re.fullmatch(r"[0-9a-f]{16}", code) is not None,
        "no 16-hex analysis_code",
    )
    if amends(ctx, prereg) is None:
        e0 = rel(ctx, E0_PART2_ARTIFACT)
        if e0.exists():
            other = load_json(e0).get("analysis_code")
            need(other == code, f"G6 ran analysis code {code}, E0 part 2 {other}")
    else:
        allowed = {
            str(prereg_block(ctx, f, "analysis-code").get("allows"))
            for f in chain_files(ctx)
            if amends(ctx, f) == E1_NUMBER and has_block(ctx, f, "analysis-code")
        }
        need(code in allowed, f"no frozen amendment of {E1_NUMBER} allows G6's code {code}")""",
        ),
    ],
)

# ---------------------------------------------------------------- tests/test_tools.py
patch(
    Path("tests/test_tools.py"),
    [
        (
            """    (pre / "0008-e1.md").write_text("---\\namends: 0004\\n---\\n# allows the new code\\n")""",
            """    allows = '```json analysis-code\\n{"allows": "' + "e" * 16 + '"}\\n```\\n'
    (pre / "0008-e1.md").write_text("---\\namends: 0004\\n---\\n# allows the new code\\n\\n" + allows)""",
        ),
        (
            """    # Without an amendment of the E1 preregistration, nothing allows the new method.
    (pre / "CHAIN.jsonl").write_text("".join(json.dumps(e) + "\\n" for e in chain[:3]))
    with pytest.raises(ship_check.GateError, match="no amendment of prereg 0004"):
        ship_check.frozen_headline_cells(ctx, grids)
    (pre / "CHAIN.jsonl").write_text("".join(json.dumps(e) + "\\n" for e in chain))""",
            """    # Without an amendment of the E1 preregistration, nothing allows the new method.
    (pre / "CHAIN.jsonl").write_text("".join(json.dumps(e) + "\\n" for e in chain[:3]))
    with pytest.raises(ship_check.GateError, match="no amendment of prereg 0004"):
        ship_check.frozen_headline_cells(ctx, grids)
    # An amendment of 0004 for another purpose (a rebuilt server) allows no new method.
    server = '```json llama-server\\n{"sha256": "' + "f" * 64 + '"}\\n```\\n'
    (pre / "0008-e1.md").write_text("---\\namends: 0004\\n---\\n# rebuilt server\\n\\n" + server)
    (pre / "CHAIN.jsonl").write_text("".join(json.dumps(e) + "\\n" for e in chain))
    with pytest.raises(ship_check.GateError, match="no amendment of prereg 0004"):
        ship_check.frozen_headline_cells(ctx, grids)
    (pre / "0008-e1.md").write_text("---\\namends: 0004\\n---\\n# allows the new code\\n\\n" + allows)
    # Under the task jackknife the box rule has no effect: changing it alone is no new method.
    jack = {"fixed_interval": "jackknife"}
    assert ship_check.glmm_rule("fixed", {**jack, "box_interval": "basic"}) == (
        ship_check.glmm_rule("fixed", {**jack, "box_interval": "percentile"})
    )
    box = {"fixed_interval": "box", "box_interval": "basic"}
    assert ship_check.glmm_rule("fixed", box) != ship_check.glmm_rule("fixed", jack)
    assert ship_check.glmm_rule("fixed", box) != (
        ship_check.glmm_rule("fixed", {**box, "box_interval": "student"})
    )""",
        ),
    ],
)

# ---------------------------------------------------------------- docs/GATES.md
patch(
    Path("docs/GATES.md"),
    [
        (
            """  test another method (another `method` or `interval_type`; for a GLMM kind, another rule in the
  amendment's own ` ```json recovery-config ` settings: `boot_interval`, `fixed_interval` or
  `box_interval`, `share_interval`, `mde_interval`), and only when a frozen amendment of the E1
  preregistration (0004) from before that run exists (D-034 item 13). All""",
            """  test another method (another `method` or `interval_type`; for a GLMM kind, another rule in the
  amendment's own ` ```json recovery-config ` settings: `boot_interval`; the fixed rule, the task
  jackknife or the bootstrap box with its `box_interval`, which has no effect under the
  jackknife; `share_interval`; `mde_interval`), and only when a frozen amendment of the E1
  preregistration (0004) from before that run has a ` ```json analysis-code ` block (D-034
  items 13 and 16). All""",
        ),
        (
            """analysis-code hash, which prereg 0004's analysis requires and which must equal the
`analysis_code` of `artifacts/e0/coverage-0005.json` once E0 part 2 is banked; reported bias""",
            """analysis-code hash, which prereg 0004's analysis requires and which must equal the
`analysis_code` of `artifacts/e0/coverage-0005.json` once E0 part 2 is banked (for a rerun of G6
under an amendment of its preregistration, a replaced GLMM method, the hash that a frozen
amendment of 0004 allows in its ` ```json analysis-code ` block); reported bias""",
        ),
    ],
)

# ---------------------------------------------------------------- tools/prereg_blocks.py
patch(
    Path("tools/prereg_blocks.py"),
    [
        (
            """    Families: the normal-approximation power of :func:`evalstats.power_at` for every
    confirmatory contrast family at stated, assumed effects and the planted variances,
    with the tasks each comparison needs for power 0.8 and a frozen verdict per family
    (underpowered when its least-powered row is below 0.8).""",
            """    Families: the power of one comparison and of the Holm reading for every confirmatory
    contrast family at stated, assumed effects and the planted variances, with each
    family's test statistic (t on tasks - 1 degrees of freedom for the cluster_t families,
    normal for the Wilson families, :func:`evalstats.power_at`), the tasks one comparison
    needs for power 0.8, and a frozen verdict per family (underpowered when its smallest
    ``power_holm`` is below 0.8).""",
        ),
    ],
)

# ---------------------------------------------------------------- src/noisefloor/e0_validity.py
patch(
    Path("src/noisefloor/e0_validity.py"),
    [
        (
            '''def main() -> int:
    """Write both files from the banked coverage artifacts."""
    coverage = load_coverage()
    validity, headline = build(coverage, amendment_headlines(coverage))
    VALIDITY.parent.mkdir(parents=True, exist_ok=True)
    VALIDITY.write_text(json.dumps(validity, indent=1, sort_keys=True) + "\\n")
    HEADLINE.write_text(json.dumps(headline, indent=1, sort_keys=True) + "\\n")
    print(
        f"unsupported cells: {sorted(validity['unsupported'])}; "
        f"headline cells: {len(headline['cells'])}"
    )
    return 0''',
            '''def outputs(directory: Path = COVERAGE.parent) -> dict[Path, str]:
    """The text of each output file, built from the banked coverage artifacts."""
    coverage = load_coverage(directory)
    validity, headline = build(coverage, amendment_headlines(coverage))
    return {
        VALIDITY: json.dumps(validity, indent=1, sort_keys=True) + "\\n",
        HEADLINE: json.dumps(headline, indent=1, sort_keys=True) + "\\n",
    }


def stale(directory: Path = COVERAGE.parent) -> list[Path]:
    """Output files whose text differs from what the banked coverage gives (run main)."""
    return [p for p, text in outputs(directory).items() if not p.exists() or p.read_text() != text]


def main() -> int:
    """Write both files from the banked coverage artifacts."""
    texts = outputs()
    VALIDITY.parent.mkdir(parents=True, exist_ok=True)
    for path, text in texts.items():
        path.write_text(text)
    validity, headline = (json.loads(texts[p]) for p in (VALIDITY, HEADLINE))
    print(
        f"unsupported cells: {sorted(validity['unsupported'])}; "
        f"headline cells: {len(headline['cells'])}"
    )
    return 0''',
        ),
    ],
)

# ---------------------------------------------------------------- src/noisefloor/e1.py
patch(
    Path("src/noisefloor/e1.py"),
    [
        (
            '''def require_validation(prereg: Path) -> None:
    """Refuse to start an arm before G6 and E0 part 2 are banked (prereg 0004).

    Then a kind of interval that fails its cells can still change method by a dated
    amendment before any E1 run exists. A file outside the chain (a fixture) is not checked.
    """
    if preregs.chain_entry(prereg) is None:
        return
    from noisefloor.analysis.e1 import validation_artifacts  # noqa: PLC0415 - imports e1

    for name, path in validation_artifacts(prereg_block(prereg, "e1-analysis")).items():
        if not path.exists():
            raise ValueError(f"{name}: {path.name} is not banked; G6 and E0 part 2 come first")''',
            '''def require_validation(prereg: Path) -> None:
    """Refuse to start an arm before G6 and E0 part 2 are banked (prereg 0004).

    Then a kind of interval that fails its cells can still change method by a dated
    amendment before any E1 run exists. The arm also refuses analysis code at HEAD other
    than the code each artifact recorded, recomputed with that artifact's own library
    versions (so the check holds on any machine; a frozen amendment of this preregistration
    may allow other code in its ```json analysis-code``` block), and a list of failing cells
    not rebuilt from the banked coverage (``python -m noisefloor.e0_validity``, committed),
    so that neither mistake is found only after E1 data exist. A file outside the chain (a
    fixture) is not checked.
    """
    if preregs.chain_entry(prereg) is None:
        return
    from noisefloor import e0_validity, provenance  # noqa: PLC0415 - provenance loads R lazily
    from noisefloor.analysis.e1 import validation_artifacts  # noqa: PLC0415 - imports e1

    artifacts = validation_artifacts(prereg_block(prereg, "e1-analysis"))
    for name, path in artifacts.items():
        if not path.exists():
            raise ValueError(f"{name}: {path.name} is not banked; G6 and E0 part 2 come first")
    allowed = {
        str(prereg_block(a, "analysis-code").get("allows"))
        for a in preregs.amendments_with(E1_PREREG, "analysis-code")
    }
    for name, path in artifacts.items():
        data = json.loads(path.read_text())
        key = dict((data.get("environment") or {}).get("key") or {})
        versions = {k: v for k, v in key.items() if k not in ("src_tree", "uv_lock")}
        here = provenance.analysis_code(versions=versions)["hash"] if versions else None
        if here != data.get("analysis_code") and here not in allowed:
            raise ValueError(
                f"{name}: the analysis code at HEAD ({here}) is not the code {path.name} "
                f"recorded ({data.get('analysis_code')}); a frozen amendment must allow it"
            )
    stale = e0_validity.stale()
    if stale:
        names = ", ".join(p.name for p in stale)
        raise ValueError(
            f"{names} not rebuilt from the banked coverage: run python -m "
            "noisefloor.e0_validity and commit its outputs first"
        )''',
        ),
    ],
)

# ---------------------------------------------------------------- tests/test_e1.py
patch(
    Path("tests/test_e1.py"),
    [
        (
            """    with pytest.raises(ValueError, match="G6 and E0 part 2 come first"):
        e1.require_validation(prereg)
    g6.write_text("{}")
    e0.write_text("{}")
    e1.require_validation(prereg)
""",
            """    with pytest.raises(ValueError, match="G6 and E0 part 2 come first"):
        e1.require_validation(prereg)
    import json as _json

    from noisefloor import e0_validity, provenance

    versions = {"python": "3.12.0", "numpy": "2.0", "r": "4.3"}  # each artifact's own
    key = {"src_tree": "t" * 40, "uv_lock": "l" * 40, **versions}
    ran = provenance.analysis_code(versions=versions)["hash"]
    for path in (g6, e0):
        path.write_text(_json.dumps({"environment": {"key": key}, "analysis_code": ran}))
    monkeypatch.setattr(e0_validity, "stale", lambda: [])
    monkeypatch.setattr(preregs, "amendments_with", lambda number, block: [])
    e1.require_validation(prereg)
    # Other analysis code at HEAD (here: another recorded hash) refuses the arm.
    e0.write_text(_json.dumps({"environment": {"key": key}, "analysis_code": "0" * 16}))
    with pytest.raises(ValueError, match="is not the code coverage-0005.json recorded"):
        e1.require_validation(prereg)
    allows = tmp_path / "0009-code.md"
    allows.write_text(
        "---\\namends: 0004\\n---\\n\\n```json analysis-code\\n"
        + _json.dumps({"allows": ran})
        + "\\n```\\n"
    )
    monkeypatch.setattr(preregs, "amendments_with", lambda number, block: [allows])
    e1.require_validation(prereg)  # a frozen amendment allows the code in use
    monkeypatch.setattr(e0_validity, "stale", lambda: [e0_validity.VALIDITY])
    with pytest.raises(ValueError, match="validity.json not rebuilt from the banked coverage"):
        e1.require_validation(prereg)
""",
        ),
    ],
)

# ---------------------------------------------------------------- tests/test_validity.py
text = Path("tests/test_validity.py").read_text()
text += """

def test_the_validity_outputs_are_rebuilt_from_the_banked_coverage(tmp_path: Path, monkeypatch) -> None:
    import json

    cell = {
        "cell": "X",
        "method": "score_interval",
        "interval_type": "cluster_t",
        "nominal": 0.95,
        "replicates": 100,
        "covered": 80,
        "params": {"n_tasks": 10},
    }
    (tmp_path / "coverage.json").write_text(json.dumps({"prereg": "p", "cells": [cell]}))
    monkeypatch.setattr(e0_validity, "VALIDITY", tmp_path / "validity.json")
    monkeypatch.setattr(e0_validity, "HEADLINE", tmp_path / "headline_cells.json")
    assert e0_validity.stale(tmp_path) == [tmp_path / "validity.json", tmp_path / "headline_cells.json"]
    for path, content in e0_validity.outputs(tmp_path).items():
        path.write_text(content)
    assert e0_validity.stale(tmp_path) == []
    assert "X" in json.loads((tmp_path / "validity.json").read_text())["unsupported"]
    # The committed file in the repository is what the banked coverage gives.
    monkeypatch.undo()
    assert e0_validity.stale() == []
"""
Path("tests/test_validity.py").write_text(text)
print("patched tests/test_validity.py")
