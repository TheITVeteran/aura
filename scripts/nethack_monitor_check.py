#!/usr/bin/env python3
# scripts/nethack_monitor_check.py
import json
import os
import sys
import time
from pathlib import Path

TRACE_FILE = Path(os.environ.get("AURA_NETHACK_LOG", "~/.aura/logs/nethack/kernel_trace.jsonl")).expanduser()
RUNNER_LOG = Path("~/.aura/logs/nethack/runner.log").expanduser()

def check_status():
    if not TRACE_FILE.exists():
        return {
            "status": "not_started",
            "message": f"Trace file {TRACE_FILE} does not exist yet."
        }

    records = []
    try:
        with open(TRACE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        return {
            "status": "error",
            "message": f"Failed to read trace file: {e}"
        }

    if not records:
        return {
            "status": "empty",
            "message": "Trace file is empty."
        }

    latest = records[-1]
    latest_timestamp = latest.get("timestamp", 0.0)
    current_time = time.time()
    elapsed_since_last_step = current_time - latest_timestamp

    # Detect Stall
    stalled = elapsed_since_last_step > 300  # 5 minutes without writing a step

    # Count recent identical intents to detect loop
    recent_intents = [
        r.get("action_intent", {}).get("name") 
        for r in records 
        if r.get("action_intent") and r.get("action_intent", {}).get("name") is not None
    ]
    looping = len(recent_intents) >= 10 and len(set(recent_intents[-10:])) == 1

    # Check for deaths or terminal state
    died = False
    for r in reversed(records[-10:]):
        events = r.get("outcome_assessment", {}).get("observed_events", [])
        if "death" in events or "dywypy" in events or "You die" in str(r.get("execution_result", {}).get("observation_after", {}).get("raw", "")):
            died = True
            break

    # Level steps and stall detection
    latest_context = latest.get("context_id")
    consecutive_count = 0
    for r in reversed(records):
        if r.get("context_id") == latest_context:
            consecutive_count += 1
        else:
            break

    # Every actual action / step corresponds to 2 trace log rows (observe and step)
    actual_steps_on_level = consecutive_count // 2

    level_stall_threshold = None
    level_stall = False
    if latest_context and latest_context.startswith("dlvl_"):
        try:
            level_num = int(latest_context.split("_")[1])
        except (IndexError, ValueError):
            level_num = 1

        level_stall_threshold = 400 if level_num == 1 else 800
        if actual_steps_on_level > level_stall_threshold:
            level_stall = True

    # Analyze runner log for python exception / crash
    runner_crashed = False
    last_error = ""
    if RUNNER_LOG.exists():
        try:
            content = RUNNER_LOG.read_text(encoding="utf-8")
            if "Traceback" in content or "Error:" in content:
                runner_crashed = True
                lines = content.splitlines()
                last_error = "\n".join(lines[-10:])
        except (OSError, UnicodeError) as exc:
            runner_crashed = True
            last_error = f"Failed to read runner log: {exc}"

    return {
        "status": "running" if not (stalled or level_stall) else "stalled",
        "looping": looping,
        "died": died,
        "runner_crashed": runner_crashed,
        "level_stall": level_stall,
        "actual_steps_on_level": actual_steps_on_level,
        "level_stall_threshold": level_stall_threshold,
        "last_error": last_error,
        "elapsed_since_last_step_seconds": round(elapsed_since_last_step, 1),
        "total_steps": len(records),
        "latest_step": {
            "sequence_id": latest.get("sequence_id"),
            "action": latest.get("action_intent", {}).get("name"),
            "events": latest.get("outcome_assessment", {}).get("observed_events", []),
            "success_score": latest.get("outcome_assessment", {}).get("success_score"),
            "context": latest.get("context_id"),
        }
    }

if __name__ == "__main__":
    status = check_status()
    print(json.dumps(status, indent=2))
