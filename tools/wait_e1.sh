#!/bin/bash
# Harness-tracked waiter for the E1 local arm (DECISIONS.md D-036). Run it as a background task
# with the longest timeout, and start it again first on every wake-up; on launch night its
# two-hour limit must not fall between 23:45 and 00:30 UTC. It writes a timestamped line a
# minute to runs/e1/alive.txt (model servers alive and the arm's last log line, which names
# counts only, never outcomes) and exits when the arm's process (runs/e1/pid) ends, so that the
# session wakes up to relaunch it or to record why it stopped. Never edit this file while it runs.
cd "$(dirname "$0")/.." || exit 1
pid=$(cat runs/e1/pid 2> /dev/null)
if [ -z "$pid" ]; then
  echo "no runs/e1/pid"
  exit 1
fi
while grep -qa 'noisefloor.e1' "/proc/$pid/cmdline" 2> /dev/null; do
  echo "$(date -u +%FT%TZ) servers=$(pgrep -c -x llama-server) $(tail -n 1 runs/e1/log.txt)" >> runs/e1/alive.txt
  sleep 60
done
echo "arm $pid ended $(date -u +%FT%TZ)"
tail -n 3 runs/e1/log.txt
