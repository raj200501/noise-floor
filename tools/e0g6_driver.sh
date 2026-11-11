#!/bin/bash
# G6 (prereg 0006), then E0 part 2 (prereg 0005): the driver for every launch and relaunch from
# now on (DECISIONS.md D-036). Each step is resumable and skips the work already done. It first
# runs `tools/runcheck.py e0g6-preflight` (no other driver or study process, a clean src/, raw
# row files that end cleanly, the environment the rows recorded) and stops at the first step
# that fails, so no step starts after an earlier one refused.
#
# Usage, from the repository root, in a fresh shell (never inside an E1 runner's session):
#   WORKERS=4 setsid nohup bash tools/e0g6_driver.sh >> runs/e0g6/nohup.out 2>&1 < /dev/null &
# then start the harness-tracked waiter `bash tools/wait_e0g6.sh` as a background task.
# Never edit this file while it runs (bash reads a running script as it goes).
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/e0g6
LOG=runs/e0g6/log.txt
P5=prereg/0005-e0-glmm-cells.md
P6=prereg/0006-g6-recovery.md
W=${WORKERS:-4}
exec 9> runs/e0g6/.driver.lock
if ! flock -n 9; then
  echo "another tools/e0g6_driver.sh holds runs/e0g6/.driver.lock" >&2
  exit 4
fi
echo "start $(date -u +%FT%TZ) workers $W head $(git rev-parse --short HEAD) (tools/e0g6_driver.sh)" >> "$LOG"
.venv/bin/python tools/runcheck.py e0g6-preflight >> "$LOG" 2>&1
code=$?
if [ "$code" -ne 0 ]; then
  echo "preflight exit $code $(date -u +%FT%TZ); nothing started" >> "$LOG"
  exit 1
fi

step() {
  local name=$1
  shift
  .venv/bin/python -m "$@" --workers "$W" >> "$LOG" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "$name exit $rc $(date -u +%FT%TZ); later steps not started" >> "$LOG"
    exit 1
  fi
  echo "$name done $(date -u +%FT%TZ)" >> "$LOG"
}

step g6-run noisefloor.recovery g6-run "$P6"
step e0-run noisefloor.recovery e0-run "$P5"
step e0_run noisefloor.e0_run run "$P5"
echo "all done $(date -u +%FT%TZ)" >> "$LOG"
