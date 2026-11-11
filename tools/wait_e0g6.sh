#!/bin/bash
# Harness-tracked waiter for the G6 and E0 part 2 driver (DECISIONS.md D-036). Run it as a
# background task with the longest timeout, and start it again first on every wake-up: a
# session with no tracked work is stopped when idle, which kills the detached driver. It logs a
# line a minute to runs/e0g6/alive.txt and exits when the driver ends, or when no raw row file
# has grown for STALL_MIN minutes (a hung pool), so that the session wakes up to look.
# Never edit this file while it runs.
cd "$(dirname "$0")/.." || exit 1
STALL_MIN=${STALL_MIN:-45}
alive() { pgrep -f '^bash (runs/e0g6/driver\.sh|tools/e0g6_driver\.sh)' > /dev/null; }
while alive; do
  newest=$(find artifacts/g6/raw.jsonl artifacts/e0/raw -name '*.jsonl' -printf '%T@\n' 2> /dev/null | sort -n | tail -1)
  [ -n "$newest" ] || newest=$(date +%s)
  age=$(($(date +%s) - ${newest%.*}))
  echo "$(date -u +%FT%TZ) newest-row-age ${age}s" >> runs/e0g6/alive.txt
  if [ "$age" -gt $((STALL_MIN * 60)) ]; then
    echo "STALL: no raw row written for ${age} s while the driver is alive"
    tail -4 runs/e0g6/log.txt
    exit 2
  fi
  sleep 60
done
echo "driver ended $(date -u +%FT%TZ)"
grep -E ' exit | done |^start|preflight' runs/e0g6/log.txt | tail -8
