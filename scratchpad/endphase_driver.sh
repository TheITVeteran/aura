#!/bin/zsh
# endphase_driver.sh — the mandate's last act, zero-babysitting.
#
# Phase A (attribution): boot the worktree instance with tracemalloc on,
#   wait conversation_ready, let it IDLE 50 min while sampling RSS +
#   /api/system/memory/growth snapshots (start/mid/end) — the leak
#   attribution the 05:50 run never delivered.
# Phase B (endurance): 200-turn probe, deadline 180 min.
# One DONE marker; all artifacts under this scratchpad dir.
#
# Run detached: nohup scratchpad/endphase_driver.sh > scratchpad/endphase.log 2>&1 &
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

echo "phase=preflight t=$(date +%H:%M:%S)"
if curl -s -m 3 http://localhost:8000/api/health >/dev/null 2>&1; then
  echo "phase=abort reason=port_8000_already_serving"
  echo "verdict=aborted reason=port_busy" > "$DONE"; exit 1
fi

# Boot from the worktree with allocation attribution armed.
export AURA_RUNTIME_HYGIENE_TRACEMALLOC=1
export AURA_RUNTIME_HYGIENE_TRACEMALLOC_FRAMES=10
nohup ./launch_aura.sh > "$SC/launch_$TS.log" 2>&1 &
LAUNCH_PID=$!
echo "phase=booting launcher_pid=$LAUNCH_PID t=$(date +%H:%M:%S)"

# Wait ready (≤20 min).
READY=0
for i in $(seq 1 120); do
  sleep 10
  BODY=$(curl -s -m 5 http://localhost:8000/api/health/boot 2>/dev/null || true)
  if echo "$BODY" | grep -q '"conversation_operational": *true\|"status": *"ready"'; then READY=1; break; fi
done
if [ "$READY" != "1" ]; then
  echo "phase=abort reason=never_ready t=$(date +%H:%M:%S)"
  echo "verdict=aborted reason=never_ready" > "$DONE"; exit 1
fi
echo "phase=ready t=$(date +%H:%M:%S)"

# Cortex warmup settle (≤20 min, RSS of tree > 15GB or timeout — soft gate).
sleep 300
echo "phase=idle_window minutes=50 t=$(date +%H:%M:%S)"
echo "epoch,phase,rss_mb_total,n_procs" > "$RSS_CSV"
curl -s -m 10 http://localhost:8000/api/system/memory/growth > "$GROWTH_DIR/growth_start.json" 2>/dev/null || true
for i in $(seq 1 100); do  # 100 × 30s = 50 min
  EPOCH=$(date +%s)
  STATS=$(ps -axo pid=,ppid=,rss=,command= | awk '/aura_main/ && !/awk/ {s+=$3; n+=1} END {printf "%d,%d", s/1024, n}')
  echo "$EPOCH,idle,$STATS" >> "$RSS_CSV"
  if [ "$i" = "50" ]; then
    curl -s -m 10 http://localhost:8000/api/system/memory/growth > "$GROWTH_DIR/growth_mid.json" 2>/dev/null || true
  fi
  sleep 30
done
curl -s -m 10 http://localhost:8000/api/system/memory/growth > "$GROWTH_DIR/growth_end.json" 2>/dev/null || true
echo "phase=idle_done t=$(date +%H:%M:%S)"

# Phase B: the 200-turn endurance probe (RSS sampler keeps running inline).
echo "phase=endurance_probe t=$(date +%H:%M:%S)"
( while true; do
    EPOCH=$(date +%s)
    STATS=$(ps -axo pid=,ppid=,rss=,command= | awk '/aura_main/ && !/awk/ {s+=$3; n+=1} END {printf "%d,%d", s/1024, n}')
    echo "$EPOCH,load,$STATS" >> "$RSS_CSV"
    sleep 30
  done ) &
SAMPLER_PID=$!
caffeinate -dims nice -n 10 .venv/bin/python tools/conversation_endurance_probe.py \
  --turns 200 --deadline-min 180 --session "endurance-20260713-endphase" \
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
