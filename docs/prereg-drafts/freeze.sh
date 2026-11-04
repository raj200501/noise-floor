#!/bin/bash
# Freeze 0004 to 0006 in one commit (run only after the final review's fixes are in).
set -euo pipefail
S=/tmp/claude-0/-home-user-noise-floor/b5fd6b69-b3b0-56ef-9f17-e15c370adddf/scratchpad
cd /home/user/noise-floor
test -z "$(git status --porcelain)" || { echo "tree not clean"; exit 1; }
rm -rf $S/blocks-final $S/assembled-final
uv run python tools/prereg_blocks.py $S/blocks-final > $S/blocks-final.txt
uv run python $S/assemble3.py $S/blocks-final $S/assembled-final "$(date -u +%F)"
uv run python - <<PY
from pathlib import Path
from noisefloor import prereg_consistency as pc
d = Path("$S/assembled-final")
found = pc.problems(d / "0004-e1-factorial.md", d / "0005-e0-glmm-cells.md", d / "0006-g6-recovery.md")
assert found == [], found
print("consistency: ok")
# The frozen llama-server pin is this machine's build (review final-stats-1, res-1, ops-1).
import json, re
from noisefloor import e1, preregs
pin = preregs.prereg_block(d / "0004-e1-factorial.md", "e1-design")["llama_server"]["sha256"]
assert isinstance(pin, str) and re.fullmatch(r"[0-9a-f]{64}", pin), pin
assert pin == e1.server_pin()["sha256"], (pin, e1.server_pin())
print("llama-server pin: ok", pin[:12])
PY
for f in 0004-e1-factorial.md 0005-e0-glmm-cells.md 0006-g6-recovery.md; do
  cp $S/assembled-final/$f prereg/$f
done
for f in 0004-e1-factorial.md 0005-e0-glmm-cells.md 0006-g6-recovery.md; do
  uv run python tools/prereg.py freeze prereg/$f
done
uv run pytest -q -p no:cacheprovider tests/test_prereg_consistency.py tests/test_prereg.py
make -s prereg-verify
git status --short
