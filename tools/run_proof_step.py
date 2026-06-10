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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("run_proof_step: no command given", file=sys.stderr)
        return 2

    started = time.time()
    timed_out = False
    returncode: int | None = None
    proc = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    try:
        returncode = proc.wait(timeout=args.timeout)
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
