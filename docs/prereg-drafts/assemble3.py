# ruff: noqa
"""Assemble preregistrations 0004 to 0006 from the third drafts and generated blocks.

Usage: uv run python assemble2.py BLOCKS_DIR OUT_DIR FREEZE_DATE

Placeholders are filled from the generated blocks (rules, cell counts) and from the pinned
model metadata; no value is typed here.
"""

import json
import re
import sys
from pathlib import Path

from noisefloor.e1 import CONFIRMATORY_SEEDS
from noisefloor.models import MODELS

HERE = Path(__file__).parent
DRAFTS = {
    "0004-draft3.md": "0004-e1-factorial.md",
    "0005-draft3.md": "0005-e0-glmm-cells.md",
    "0006-draft3.md": "0006-g6-recovery.md",
}
blocks_dir, out_dir, date = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
rules = json.loads((blocks_dir / "rules.json").read_text())
plan = json.loads((blocks_dir / "e1-plan.json").read_text())
grid = json.loads((blocks_dir / "e0-grid.json").read_text())
e1_design = json.loads((blocks_dir / "e1-design.json").read_text())


def describe(model: str) -> str:
    m = MODELS[model]
    return f"`{model}` ({m.repo}, file {m.filename}, {m.license})"


names = [describe(m) for m in plan["models"]]
subs = {
    "FREEZE_DATE": date,
    "LOCAL_MODELS_SENTENCE": ", ".join(names[:-1]) + ", and " + names[-1],
    "RANDOM_RULE": f"`{rules['random']}`",
    "MDE_RULE": f"`{rules['mde']}`",
    "SHARE_RULE": f"`{rules['share']}`",
    "CELL_COUNT": str(len(grid)),
    # Design values of the generated blocks (the local sample follows the sizing rule).
    "LOCAL_TASKS_N": str(e1_design["tasks"]),
    "JACKKNIFE_DF_N": str(e1_design["tasks"] - 1),
    "TASKS_PER_FAMILY_N": str(plan["tasks_per_family"]),
    "TASK_SEED_LAST": str(CONFIRMATORY_SEEDS[0] + plan["tasks_per_family"] - 1),
    "LOCAL_RUNS_N": f"{e1_design['local_runs']:,}",
    "REPLICATE_RUNS_N": f"{e1_design['replicate_runs']:,}",
}
out_dir.mkdir(parents=True, exist_ok=True)
for draft, name in DRAFTS.items():
    text = (HERE / draft).read_text()
    for key, value in subs.items():
        text = text.replace(key, value)
    match = re.search(r"## Blocks\n\n\(([^)]*)\)\n", text)
    assert match, draft
    parts = []
    for block in [b.strip() for b in match.group(1).split(",")]:
        value = json.loads((blocks_dir / f"{block}.json").read_text())
        parts.append(f"```json {block}\n{json.dumps(value, indent=1)}\n```\n")
    text = text[: match.start()] + "## Blocks\n\n" + "\n".join(parts)
    left = [
        w
        for w in re.findall(r"\b[A-Z][A-Z_]{4,}\b", text)
        if w.endswith(("_RULE", "_SENTENCE", "_COUNT", "_DATE"))
    ]
    left += [k for k in subs if k in text]
    assert not left, (draft, left)
    assert "\u2014" not in text, f"{draft}: em dash"
    (out_dir / name).write_text(text)
    print(out_dir / name, len(text.splitlines()), "lines")
