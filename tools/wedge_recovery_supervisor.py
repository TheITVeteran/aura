"""#50 phase 2 — out-of-process wedge-recovery supervisor.

In-process recovery of a Metal-GPU command-buffer deadlock is impossible: the
deadlock parks every thread of the host process (event loop, in-process
watchdog, signal handlers) behind the GPU semaphore, so nothing inside the
process can act — `loop_wedge_stacks.log` never even gets written. The ONLY
recovery is process-level restart.

Phase 1 shipped the EXTERNAL liveness sentinel (`tools/liveness_sentinel.py`),
which SIGKILLs a wedged runtime tree and relies on a launchd supervisor to
restart it — covering LIVE use. This module is the phase-2 piece for contexts
WITHOUT launchd (a proof battery, an unattended long run): it runs the heavy
runtime as a SUBPROCESS, monitors the loop-liveness beacon out-of-process (the
supervisor itself does NO GPU work, so it can never wedge), and on a wedge it
SIGKILLs the tree, RESTARTS the subprocess, and invokes a resume hook — so a
Metal deadlock kills only the killable subprocess and the run survives.

It reuses the phase-1 staleness logic verbatim (`read_heartbeat`,
`is_stale_sample`, `kill_tree`, `read_heartbeat_state`) so the wedge-detection
contract is identical to the sentinel that protects live use.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Runtime.WedgeRecoverySupervisor")


@dataclass
class SupervisorOutcome:
    completed: bool
    restarts: int
    reason: str
    final_pid: Optional[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed": self.completed,
            "restarts": self.restarts,
            "reason": self.reason,
            "final_pid": self.final_pid,
        }


class WedgeRecoverySupervisor:
    """Spawn a target subprocess; detect a loop wedge via the liveness beacon;
    SIGKILL + restart + resume. The out-of-process recovery a Metal-GPU deadlock
    needs and in-process recovery cannot provide.

    ``spawn`` returns a fresh process handle for the target each call.
    ``on_restart(restart_index)`` (optional) is the resume hook — e.g. a proof
    battery records "resume from the next task" before the runtime comes back.
    """

    def __init__(
        self,
        *,
        spawn: Callable[[], Any],
        heartbeat_path: str | Path,
        stale_ceiling_s: float = 180.0,
        grace_s: float = 300.0,
        poll_interval_s: float = 5.0,
        max_restarts: int = 3,
        on_restart: Optional[Callable[[int], None]] = None,
        kill: Optional[Callable[[int], Any]] = None,
        clock: Optional[Callable[[], float]] = None,
        sleep: Optional[Callable[[float], None]] = None,
    ) -> None:
        self._spawn = spawn
        self._hb = Path(heartbeat_path)
        self._ceiling = float(stale_ceiling_s)
        self._grace = float(grace_s)
        self._poll = float(poll_interval_s)
        self._max_restarts = int(max_restarts)
        self._on_restart = on_restart
        # Wall-clock by default: beacon timestamps (last_loop_run/written_at) are
        # time.time()-based, so staleness must be measured against the same clock.
        self._clock = clock or time.time
        self._sleep = sleep or time.sleep
        if kill is not None:
            self._kill = kill
        else:
            from tools.liveness_sentinel import kill_tree

            self._kill = kill_tree

    def run(self, *, done: Optional[Callable[[], bool]] = None, max_polls: Optional[int] = None) -> SupervisorOutcome:
        """Supervise until ``done()`` is True (the run finished), the target exits
        cleanly, or the restart budget is exhausted."""
        from tools.liveness_sentinel import is_stale_sample, read_heartbeat, read_heartbeat_state

        done = done or (lambda: False)
        proc = self._spawn()
        started = self._clock()
        consecutive_stale = 0
        restarts = 0
        polls = 0

        def _restart(reason: str) -> Optional[SupervisorOutcome]:
            nonlocal proc, started, consecutive_stale, restarts
            restarts += 1
            if restarts > self._max_restarts:
                return SupervisorOutcome(False, restarts, "max_restarts_exceeded", None)
            logger.warning("Restarting supervised runtime (%s) — restart %d/%d.", reason, restarts, self._max_restarts)
            if self._on_restart is not None:
                self._on_restart(restarts)
            proc = self._spawn()
            started = self._clock()
            consecutive_stale = 0
            return None

        while max_polls is None or polls < max_polls:
            if done():
                return SupervisorOutcome(True, restarts, "completed", getattr(proc, "pid", None))
            self._sleep(self._poll)
            polls += 1
            now = self._clock()

            rc = proc.poll()
            if rc is not None:
                if done():
                    return SupervisorOutcome(True, restarts, "completed", getattr(proc, "pid", None))
                state = read_heartbeat_state(self._hb)
                if rc == 0 or state == "retired":
                    return SupervisorOutcome(True, restarts, "clean_exit", getattr(proc, "pid", None))
                outcome = _restart(f"exited rc={rc}")
                if outcome is not None:
                    return outcome
                continue

            last_loop_run, written_at, mtime = read_heartbeat(self._hb)
            stale = is_stale_sample(
                now=now, last_loop_run=last_loop_run, written_at=written_at, file_mtime=mtime,
                started_at=started, grace_s=self._grace, stale_ceiling_s=self._ceiling,
            )
            consecutive_stale = consecutive_stale + 1 if stale else 0
            if stale and consecutive_stale >= 2:
                # WEDGE: the event loop's last callback is older than the ceiling
                # yet the process is still alive — a Metal-GPU deadlock (or a GIL
                # hold). In-process recovery is impossible; kill out-of-process.
                logger.error(
                    "Supervised runtime WEDGED (loop beacon stale > %.0fs, pid=%s) — SIGKILL + restart.",
                    self._ceiling, getattr(proc, "pid", None),
                )
                try:
                    self._kill(proc.pid)
                except (OSError, ProcessLookupError) as exc:  # pragma: no cover
                    logger.warning("kill of wedged tree failed (already gone?): %s", exc)
                # Reap the SIGKILLed tree so it does not linger as a zombie.
                try:
                    proc.wait(timeout=3.0)
                except (subprocess.TimeoutExpired, OSError, ValueError):  # pragma: no cover
                    pass
                outcome = _restart("loop_wedge")
                if outcome is not None:
                    return outcome
        return SupervisorOutcome(False, restarts, "max_polls", getattr(proc, "pid", None))


def main(argv: list[str] | None = None) -> int:
    """CLI: supervise a runtime command out-of-process, restarting it on a wedge.

    Example (survive a Metal-GPU deadlock during an unattended run without
    launchd, restarting up to 3x):

        python tools/wedge_recovery_supervisor.py \\
            --spawn-cmd ".venv/bin/python aura_main.py --headless --port 8000" \\
            --heartbeat data/runtime/liveness_heartbeat.json
    """
    import argparse
    import json
    import shlex

    parser = argparse.ArgumentParser(description="Out-of-process wedge-recovery supervisor (#50 phase 2).")
    parser.add_argument("--spawn-cmd", required=True, help="command to (re)spawn the supervised runtime")
    parser.add_argument("--heartbeat", required=True, help="path to the loop-liveness beacon JSON")
    parser.add_argument("--stale-ceiling", type=float, default=180.0)
    parser.add_argument("--grace", type=float, default=300.0)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--max-restarts", type=int, default=3)
    args = parser.parse_args(argv)

    def _spawn() -> Any:
        return get_subprocess_gateway().spawn(
            shlex.split(args.spawn_cmd),
            start_new_session=True,
            offline_tooling=True,
            source="maintenance_tooling:wedge_recovery_supervisor",
        )

    supervisor = WedgeRecoverySupervisor(
        spawn=_spawn,
        heartbeat_path=args.heartbeat,
        stale_ceiling_s=args.stale_ceiling,
        grace_s=args.grace,
        poll_interval_s=args.interval,
        max_restarts=args.max_restarts,
        on_restart=lambda n: logger.warning("wedge-recovery restart %d issued", n),
    )
    outcome = supervisor.run()
    print(json.dumps(outcome.to_dict()))
    return 0 if outcome.completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
