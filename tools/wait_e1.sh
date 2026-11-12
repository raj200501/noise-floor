#!/bin/bash
# Harness-tracked waiter for the E1 local arm (DECISIONS.md D-036, D-043). Run it as a background
# task with the longest timeout, and start it again first on a wake-up that finds it not running;
# on launch night its two-hour limit must not fall between 23:45 and 00:30 UTC. It writes a timestamped line a
# minute to runs/e1/alive.txt (model servers alive and the arm's last log line, which names
# counts only, never outcomes) and exits when the arm's process ends, so that the session wakes
# up to relaunch it or to record why it stopped. The arm is found by its own argv
# (`python3 tools/runcheck.py e1-alive`), never by the `$!` of the launch line, which names a
# bash subshell. One copy runs at a time (a lock in runs/e1/); a second exits with code 3. Never
# edit this file while it runs.
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/e1
exec 9> runs/e1/.wait.lock
if ! flock -n 9; then
  echo "another tools/wait_e1.sh runs; not starting a second copy"
  exit 3
fi
if ! pid=$(python3 tools/runcheck.py e1-alive); then
  echo "no E1 local arm"
  exit 1
fi
echo "$pid" > runs/e1/pid
while python3 tools/runcheck.py e1-alive > /dev/null; do
  echo "$(date -u +%FT%TZ) arm=$pid servers=$(pgrep -c -x llama-server) $(tail -n 1 runs/e1/log.txt)" >> runs/e1/alive.txt
  sleep 60 9>&-
done
echo "arm $pid ended $(date -u +%FT%TZ)"
tail -n 3 runs/e1/log.txt
