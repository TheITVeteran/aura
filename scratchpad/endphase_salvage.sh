#!/bin/zsh
# endphase_salvage.sh — attach to the ALREADY-RUNNING healthy :8001 instance
# (booted 07:33 from the worktree at c8d4ef5c; the original driver's stale
# readiness grep false-aborted while conversation_ready=true) and run the
# remaining phases: 50-min idle window → 200-turn probe → DONE marker.
set -u
WT=/Users/bryan/.aura/live-source/.claude/worktrees/fable-improvement-pass
SC=$WT/scratchpad
TS=$(date +%Y%m%d_%H%M%S)
RSS_CSV=$SC/rss_trend_$TS.csv
GROWTH_DIR=$SC/growth_$TS
PROBE_LOG=$SC/probe_$TS.log
DONE=$SC/ENDPHASE_DONE_$TS
mkdir -p "$GROWTH_DIR"
cd "$WT" || exit 1

echo "phase=attach t=$(date +%H:%M:%S)"
BODY=$(curl -s -m 5 http://localhost:8001/api/health/boot 2>/dev/null || true)
if ! echo "$BODY" | grep -q '"conversation_ready": *true\|"conversation_operational": *true'; then
  echo "phase=abort reason=instance_not_conversation_ready"
  echo "verdict=aborted reason=instance_not_conversation_ready" > "$DONE"; exit 1
fi
echo "phase=idle_window minutes=50 t=$(date +%H:%M:%S)"
echo "epoch,phase,rss_mb_total,n_procs" > "$RSS_CSV"
curl -s -m 10 http://localhost:8001/api/system/memory/growth > "$GROWTH_DIR/growth_start.json" 2>/dev/null || true
DEAD_COUNT=0
for i in $(seq 1 100); do
  EPOCH=$(date +%s)
  STATS=$(ps -axo pid=,ppid=,rss=,command= | awk '/aura_main/ && !/awk/ {s+=$3; n+=1} END {printf "%d,%d", s/1024, n}')
  echo "$EPOCH,idle,$STATS" >> "$RSS_CSV"
  if ! curl -s -m 5 http://localhost:8001/api/health/boot >/dev/null 2>&1; then
    DEAD_COUNT=$((DEAD_COUNT + 1))
    if [ "$DEAD_COUNT" -ge 4 ]; then
      echo "phase=abort reason=instance_died_during_idle t=$(date +%H:%M:%S)"
      tail -40 "$SC/launch_20260717_073318.log" > "$SC/death_tail_$TS.log" 2>/dev/null
      echo "verdict=aborted reason=instance_died_during_idle" > "$DONE"; exit 1
    fi
  else
    DEAD_COUNT=0
  fi
  if [ "$i" = "50" ]; then
    curl -s -m 10 http://localhost:8001/api/system/memory/growth > "$GROWTH_DIR/growth_mid.json" 2>/dev/null || true
  fi
  sleep 30
done
curl -s -m 10 http://localhost:8001/api/system/memory/growth > "$GROWTH_DIR/growth_end.json" 2>/dev/null || true
echo "phase=idle_done t=$(date +%H:%M:%S)"

echo "phase=endurance_probe t=$(date +%H:%M:%S)"
( while true; do
    EPOCH=$(date +%s)
    STATS=$(ps -axo pid=,ppid=,rss=,command= | awk '/aura_main/ && !/awk/ {s+=$3; n+=1} END {printf "%d,%d", s/1024, n}')
    echo "$EPOCH,load,$STATS" >> "$RSS_CSV"
    sleep 30
  done ) &
SAMPLER_PID=$!
caffeinate -dims nice -n 10 .venv/bin/python tools/conversation_endurance_probe.py --base http://127.0.0.1:8001 \
  --turns 200 --deadline-min 180 --session "endurance-20260717-endphase" \
  > "$PROBE_LOG" 2>&1
PROBE_RC=$?
kill $SAMPLER_PID 2>/dev/null
echo "phase=done probe_rc=$PROBE_RC t=$(date +%H:%M:%S)"
{
  echo "probe_rc=$PROBE_RC"
  echo "rss_csv=$RSS_CSV"
  echo "growth_dir=$GROWTH_DIR"
  echo "probe_log=$PROBE_LOG"
} > "$DONE"
