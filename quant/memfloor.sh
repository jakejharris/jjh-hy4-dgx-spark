#!/usr/bin/env bash
# Run on a DGX Spark beside any job that loads weights.
# DGX Sparks have 121.7 GiB unified memory and zero swap, so this guard stops the target
# when MemAvailable falls below FLOOR_GIB and logs the lowest availability seen.
#
# Usage: memfloor.sh FLOOR_GIB pid PID [LOG]        stop = TERM to process group, KILL after 10 s
#        memfloor.sh FLOOR_GIB docker NAME [LOG]    stop = docker stop -t 10 NAME
# Recommended floors: 18 GiB on node3 while source weights are still downloading, 8 GiB otherwise.
# Exit 0 when the target ends by itself, 9 when this guard stopped it.
set -uo pipefail
floor="${1:?floor in GiB}"; kind="${2:?pid or docker}"; target="${3:?pid or container name}"
log="${4:-$HOME/hy4/logs/memfloor-$kind-$target.log}"
mkdir -p "$(dirname "$log")"
floor_kb=$(( floor * 1024 * 1024 ))
min_kb=0
alive() {
  if [ "$kind" = docker ]; then
    [ "$(docker inspect -f '{{.State.Running}}' "$target" 2>/dev/null)" = true ]
  else
    kill -0 "$target" 2>/dev/null
  fi
}
stop_job() {
  if [ "$kind" = docker ]; then
    docker stop -t 10 "$target" >/dev/null 2>&1
  else
    pgid=$(ps -o pgid= -p "$target" | tr -d ' ')
    kill -TERM -- "-$pgid" 2>/dev/null
    sleep 10
    kill -KILL -- "-$pgid" 2>/dev/null
  fi
}
gib() { awk -v kb="$1" 'BEGIN { printf "%.1f", kb / 1048576 }'; }
echo "$(date -u +%FT%TZ) watching $kind $target, floor $floor GiB" >> "$log"
n=0
while alive; do
  kb=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  if [ "$min_kb" -eq 0 ] || [ "$kb" -lt "$min_kb" ]; then min_kb=$kb; fi
  if [ "$kb" -lt "$floor_kb" ]; then
    echo "$(date -u +%FT%TZ) BREACH avail=$(gib "$kb") GiB under floor $floor: stopping $kind $target" >> "$log"
    stop_job
    echo "min_avail_gib=$(gib "$min_kb") stopped=1" >> "$log"
    exit 9
  fi
  n=$(( n + 1 ))
  if [ $(( n % 30 )) -eq 0 ]; then echo "$(date -u +%FT%TZ) avail=$(gib "$kb") min=$(gib "$min_kb")" >> "$log"; fi
  sleep 2
done
echo "min_avail_gib=$(gib "$min_kb") stopped=0" >> "$log"
exit 0
