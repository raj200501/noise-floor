#!/bin/bash
# Harness-tracked keep-alive until a UTC clock time (DECISIONS.md D-043). Between the end of the
# G6 and E0 part 2 driver and the E1 launch nothing else keeps this session's container up, so
# run it as a background task with the longest timeout, start it again at every limit and on
# every wake-up, and let its exit at the next HH:MM UTC wake the session for the launch. It
# writes a timestamped line a minute to runs/alive.txt and reads nothing else.
# Usage: bash tools/wait_until.sh HH:MM
cd "$(dirname "$0")/.." || exit 1
if ! [[ "$1" =~ ^([01][0-9]|2[0-3]):[0-5][0-9]$ ]]; then
  echo "usage: bash tools/wait_until.sh HH:MM (a UTC time)"
  exit 2
fi
now=$(date -u +%s)
target=$(date -u -d "$(date -u +%F) $1" +%s) || exit 2
if [ "$target" -le "$now" ]; then
  target=$((target + 86400))
fi
mkdir -p runs
echo "$(date -u +%FT%TZ) waiting until $(date -u -d "@$target" +%FT%TZ)" >> runs/alive.txt
while [ "$(date -u +%s)" -lt "$target" ]; do
  left=$((target - $(date -u +%s)))
  sleep $((left < 60 ? left : 60))
  echo "$(date -u +%FT%TZ) waiting until $1 UTC" >> runs/alive.txt
done
echo "reached $1 UTC at $(date -u +%FT%TZ)"
