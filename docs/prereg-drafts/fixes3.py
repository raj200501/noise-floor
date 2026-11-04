# ruff: noqa
"""Text follow-ups of the fix verification: drafts, PLAN. Run from the repository root."""

from pathlib import Path

S = Path("/tmp/claude-0/-home-user-noise-floor/b5fd6b69-b3b0-56ef-9f17-e15c370adddf/scratchpad")


def patch(path: Path, pairs: list[tuple[str, str]]) -> None:
    text = path.read_text()
    for old, new in pairs:
        assert text.count(old) == 1, (str(path), old[:120])
        text = text.replace(old, new)
    path.write_text(text)
    print("patched", path)


patch(
    S / "0004-draft3.md",
    [
        (
            """suite version. The runners refuse to start unless this file is frozen and committed, the code
computes the same plan hash, pins, task manifest, and suite version, and the G6 and E0 part 2
artifacts are banked; the local arm's first batch also needs the pinned llama-server digest, or
one that a ```json llama-server``` block of a frozen amendment of this file names (a rebuilt
server, a deviation; D-034).""",
            """suite version. The runners refuse to start unless this file is frozen and committed, the code
computes the same plan hash, pins, task manifest, and suite version, the G6 and E0 part 2
artifacts are banked, the analysis code at HEAD is the code they recorded (or one a frozen
amendment of this file allows), and the library's list of failing cells is rebuilt from the
banked coverage; the local arm's first batch also needs the pinned llama-server digest, or one
that a ```json llama-server``` block of a frozen amendment of this file names (a rebuilt server,
a deviation; D-034).""",
        ),
        (
            """block ```json retired-headline``` (G5 then requires the new cells to pass, prereg 0005); one of this
file that allows the new analysis code; and, for a GLMM kind, one of prereg 0006 that reruns G6
with the new method on new seeds. The labels that rest on other banked artifacts (underpowered
families from `family_verdicts`, the fixed factors' power from G6's `above_zero`, and a kind
resting on an unsupported method) are applied by the report from those artifacts (D-034). Each interval's point estimate is compared with the span its kind's cells""",
            """block ```json retired-headline``` (G5 then requires the new cells to pass, prereg 0005); one
of this file whose ```json analysis-code``` block allows the new analysis code; and, for a GLMM
kind, one of prereg 0006 that reruns G6 with the new method on new seeds. The labels that rest
on other banked artifacts (underpowered families from `family_verdicts`, the fixed factors'
power from G6's `above_zero`, and a kind resting on an unsupported method) are applied by the
report from those artifacts (D-034). Each interval's point estimate is compared with the span
its kind's cells""",
        ),
        (
            """code's hash (`{"allows": "<hash>"}`), which is a deviation; the outputs, the post-E1 check's
included, name the analysis code they ran.""",
            """code's hash (`{"allows": "<hash>"}`), which is a deviation; the outputs, the post-E1 check's
included, name the analysis code they ran. The pinned files include the plan and runner module
(`noisefloor.e1`) and the model families (`noisefloor.models`), so a run-code change that a
resume accepts during an arm and that touches one of them also needs such an amendment before
the analysis runs, and the pinned files stay unchanged through G7's recompute in the final
ship-check.""",
        ),
    ],
)

patch(
    S / "0005-draft3.md",
    [
        (
            """method (another method or interval type; for a GLMM kind, another rule in the amendment's own
block ```json recovery-config```) and a frozen amendment of 0004 from before the first
confirmatory E1 run. The spans""",
            """method (another method or interval type; for a GLMM kind, another rule in the amendment's own
block ```json recovery-config```, where the box rule does not count for fixed contributions under
the task jackknife) and a frozen amendment of 0004 from before the first confirmatory E1 run
whose ```json analysis-code``` block allows the new analysis code (an amendment of 0004 for
another purpose does not count). The spans""",
        ),
    ],
)

patch(
    S / "0006-draft3.md",
    [
        (
            """  environment and the analysis code, which gate G6 requires to be the one E0 part 2 ran;
  prereg 0004's analysis runs only under that analysis code.""",
            """  environment and the analysis code, which gate G6 requires to be the one E0 part 2 ran (for
  a rerun of G6 under an amendment of this file, which a replaced GLMM method needs, the one a
  frozen amendment of prereg 0004 allows); prereg 0004's analysis runs only under that analysis
  code.""",
        ),
    ],
)

patch(
    Path("docs/PLAN.md"),
    [
        (
            """G6 and E0 part 2 ran (D-033 item 10, D-034 item 8). Raw rows are committed and pushed at""",
            """G6 and E0 part 2 ran (D-033 item 10, D-034 item 8). The pinned files include `noisefloor.e1`
and `noisefloor.models`, so they stay unchanged through G7's recompute in the final ship-check;
a repair to one of them during an arm, which a resume accepts, also needs a frozen amendment of
0004 with a ```json analysis-code``` block before the analysis runs. Both arms check before they
start that the analysis code at HEAD is the code G6 and E0 part 2 recorded and that the list of
failing cells is rebuilt from the banked coverage (D-034 item 16). Raw rows are committed and pushed at""",
        ),
    ],
)

for name in ("0004-draft3.md", "0005-draft3.md", "0006-draft3.md"):
    for i, line in enumerate((S / name).read_text().splitlines(), 1):
        assert not line.lstrip().startswith("```"), (name, i, line)
print("no prose line starts with a fence")
