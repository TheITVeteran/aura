#!/bin/zsh
# verify_driver.sh — does the fix set actually change the live behaviour?
#
# The 02:38 soak ran BEFORE eleven fixes landed. Its verdict measured the old
# code. This run measures the new one, and it only needs to answer three
# questions, so it is short:
#
#   1. Does the welfare defer storm clear?  (fatigue/recovery-debt ratchets)
#      -> 25 min idle is enough: fatigue decays from saturation in ~2 min and
#         recovery debt in ~11, so a quiet 25 covers both with margin.
#   2. Does the cortex now finish a load?   (recovery handoff no longer counted
#      as a stuck load -> no 240s warmup backoff)
#   3. Do turns get answered?               (protected turns fall to Brainstem
#      instead of returning nothing) -> 60 turns is enough to move 86.5%.
set -u
WT=/Users/bryan/.aura/live-source/.claude/worktrees/fable-improvement-pass
SC=$WT/scratchpad
TS=$(date +%Y%m%d_%H%M%S)
PROBE_LOG=$SC/cap_probe_$TS.log
LAUNCH=$SC/cap_launch_$TS.log
DONE=$SC/CAP_DONE_$TS
cd "$WT" || exit 1

echo "phase=preflight t=$(date +%H:%M:%S)"
if curl -s -m 3 http://localhost:8001/api/health >/dev/null 2>&1; then
  echo "phase=abort reason=port_8001_already_serving"
  echo "verdict=aborted reason=port_busy" > "$DONE"; exit 1
fi

nohup .venv/bin/python -u aura_main.py --headless --port 8001 > "$LAUNCH" 2>&1 &
echo "phase=booting t=$(date +%H:%M:%S)"

READY=0
for i in $(seq 1 120); do
  sleep 10
  BODY=$(curl -s -m 5 http://localhost:8001/api/health/boot 2>/dev/null || true)
  if echo "$BODY" | grep -q '"conversation_operational": *true\|"status": *"ready"'; then READY=1; break; fi
done
if [ "$READY" != "1" ]; then
  echo "phase=abort reason=never_ready t=$(date +%H:%M:%S)"
  echo "verdict=aborted reason=never_ready" > "$DONE"; exit 1
fi
echo "phase=ready t=$(date +%H:%M:%S)"

sleep 150                       # warmup settle
echo "phase=idle_window minutes=8 t=$(date +%H:%M:%S)"
for i in $(seq 1 16); do        # 50 x 30s = 25 min
  if ! curl -s -m 5 http://localhost:8001/api/health/boot >/dev/null 2>&1; then
    DEAD_COUNT=$((${DEAD_COUNT:-0} + 1))
    if [ "$DEAD_COUNT" -ge 4 ]; then
      echo "phase=abort reason=died_during_idle t=$(date +%H:%M:%S)"
      echo "verdict=aborted reason=died_during_idle" > "$DONE"; exit 1
    fi
  else
    DEAD_COUNT=0
  fi
  sleep 30
done
echo "phase=idle_done t=$(date +%H:%M:%S)"

echo "phase=probe turns=60 t=$(date +%H:%M:%S)"
caffeinate -dims nice -n 10 .venv/bin/python tools/conversation_endurance_probe.py \
  --base http://127.0.0.1:8001 --turns 80 --deadline-min 150 \
  --session "capability-$TS" > "$PROBE_LOG" 2>&1
PROBE_RC=$?
echo "phase=done probe_rc=$PROBE_RC t=$(date +%H:%M:%S)"
{
  echo "probe_rc=$PROBE_RC"
  echo "launch_log=$LAUNCH"
  echo "probe_log=$PROBE_LOG"
} > "$DONE"
