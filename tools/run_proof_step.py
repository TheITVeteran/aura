#!/usr/bin/env python3
"""Run one proof step with a hard timeout and an evidence artifact.

The final-proof chain must never hang: a proof step that cannot finish
is a FAILED step with a written artifact, not an overnight mystery.
Wraps any command:

    python tools/run_proof_step.py --name dnu_battery --timeout 5400 \
        --artifact artifacts/current/proof_steps/dnu_battery.json -- \
        python tools/agi/run_dnu_agi_proof_battery.py --full ...

Writes {name, command, returncode, timed_out, duration_s, started_at,
finished_at, passed} and mirrors the child's exit code (124 on
timeout, after SIGKILLing the child's process group).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_MONO_START = 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    global _MONO_START
    _MONO_START = time.monotonic()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("run_proof_step: no command given", file=sys.stderr)
        return 2

    started = time.time()
    timed_out = False
    returncode: int | None = None
    # Sleep-proofing (the 9-hour 'hang' was a lid-closed MacBook):
    # caffeinate -i holds off idle sleep for exactly this step's
    # lifetime, and the wait loop below checks WALL clock too —
    # monotonic deadlines freeze during sleep, wall time does not, so
    # a sleep-spanning step is killed on wake instead of resuming as
    # a zombie run.
    if sys.platform == "darwin":
        command = ["/usr/bin/caffeinate", "-i", *command]
    proc = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    wall_deadline = started + args.timeout * 1.25 + 60.0
    try:
        # Bounded poll: the slice count is derived from the wall deadline,
        # so this loop terminates even if monotonic time froze in sleep.
        max_slices = int((args.timeout * 1.25 + 120.0) / 30.0) + 2
        returncode = None
        for _ in range(max_slices):
            remaining = args.timeout - (time.monotonic() - _MONO_START)
            try:
                returncode = proc.wait(timeout=min(30.0, max(1.0, remaining)))
                break
            except subprocess.TimeoutExpired:
                if remaining <= 0 or time.time() > wall_deadline:
                    raise
        if returncode is None:
            raise subprocess.TimeoutExpired(command, args.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        returncode = 124

    finished = time.time()
    artifact = {
        "schema": "aura.proof_step.v1",
        "name": args.name,
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "timeout_s": args.timeout,
        "duration_s": round(finished - started, 2),
        "started_at": started,
        "finished_at": finished,
        "passed": (returncode == 0) and not timed_out,
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact, indent=2) + "\n")

    marker = "✅" if artifact["passed"] else ("⏱️ TIMEOUT" if timed_out else "❌")
    print(
        f"{marker} proof step '{args.name}': rc={returncode} "
        f"in {artifact['duration_s']}s → {args.artifact}",
        flush=True,
    )
    return int(returncode if returncode is not None else 1)


if __name__ == "__main__":
    sys.exit(main())
