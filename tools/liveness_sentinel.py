#!/usr/bin/env python3
"""External event-loop liveness sentinel.

The in-process StallWatchdog hard-exit (core/resilience/stall_watchdog.py) can
itself be defeated: a Metal GPU command-buffer deadlock during a steered/strict
generation can hold the GIL (so no Python thread — including the watchdog daemon
— can run) or the OS GPU-watchdog can SIGKILL the worker while the main loop
stays wedged. Observed live (DNU round 34): the runtime loop went silent ~18min
with NO recovery firing, because every in-process mechanism lives inside the
wedged process.

This sentinel is a standalone process (no heavy Aura imports) spawned at boot.
It watches a HEARTBEAT FILE that the event loop refreshes ~1x/s via the
StallWatchdog. The file carries the loop's last-callback timestamp
(`last_loop_run`) and the writer's wall-clock (`written_at`). Two staleness
views, EITHER of which means the loop is dead:

  - last_loop_run stale  → the event loop stopped running callbacks (wedge).
  - written_at / mtime stale → the watchdog daemon itself can't write
    (GIL-locked deadlock) — the case the in-process hard-exit cannot catch.

Past the ceiling, it SIGKILLs the process tree so the launchd supervisor
(tools/install_supervisor.sh) restarts the runtime with state-vault continuity.
Out-of-process by design: it is unaffected by the GIL or the wedged loop.

Usage:
    python tools/liveness_sentinel.py --pid 12345 --heartbeat path/to/hb.json \
        [--stale-ceiling 180] [--interval 5] [--grace 240]
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.resource_observation import (  # noqa: E402
    HostResourceObserver,
    ObservationSource,
)

_OBSERVER = HostResourceObserver(
    source=ObservationSource.HOST,
    scenario_id="external-liveness-sentinel",
)

RING_MAX_LINES = 500


def _read_heartbeat_payload(path: Path) -> tuple[dict, float | None]:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}, None
    try:
        data = json.loads(path.read_text(errors="replace") or "{}")
    except (OSError, ValueError):
        return {}, mtime
    return data if isinstance(data, dict) else {}, mtime


def read_heartbeat(path: Path) -> tuple[float | None, float | None, float | None]:
    """Return (last_loop_run, written_at, file_mtime) or Nones if unreadable."""
    data, mtime = _read_heartbeat_payload(path)
    if mtime is None:
        return None, None, None
    if not data:
        return None, None, mtime
    try:
        last_loop_run = float(data.get("last_loop_run") or 0.0)
    except (TypeError, ValueError):
        last_loop_run = 0.0
    try:
        written_at = float(data.get("written_at") or 0.0)
    except (TypeError, ValueError):
        written_at = 0.0
    return last_loop_run, written_at, mtime


def read_runtime_service_progress(path: Path) -> float | None:
    data, mtime = _read_heartbeat_payload(path)
    if mtime is None or not data:
        return None
    try:
        value = float(data.get("last_runtime_service_progress") or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    return value if value > 0.0 else None


def read_heartbeat_state(path: Path) -> str:
    """Return the optional loop_state field from a heartbeat file.

    ``retired`` means the in-process watchdog intentionally stopped monitoring
    because the loop it owned stopped running (for example, a boot-loop to
    server-loop handoff). That is not a wedge, so the external sentinel should
    exit instead of killing the runtime later from a stale retired heartbeat.
    """
    try:
        data = json.loads(path.read_text(errors="replace") or "{}")
    except (OSError, ValueError):
        return ""
    return str(data.get("loop_state") or data.get("status") or "").strip().lower()


def child_pids(root_pid: int, *, max_children: int = 256) -> list[int]:
    table = _OBSERVER.process_table()
    if not table.available:
        return []
    descendants = [
        process
        for process in table.processes
        if root_pid in process.ancestor_pids
    ]
    descendants.sort(
        key=lambda process: (process.ancestor_pids.index(root_pid), process.pid),
        reverse=True,
    )
    return [process.pid for process in descendants[:max_children]]


def kill_tree(root_pid: int) -> list[int]:
    """SIGKILL the whole tree — a wedged/suspended process cannot handle softer."""
    killed: list[int] = []
    pids = child_pids(root_pid) + [root_pid]
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except (ProcessLookupError, PermissionError):
            continue
    return killed


def _shutdown_in_progress(shutdown_flag: Path | None) -> bool:
    """True when an intentional shutdown is underway (don't kill then)."""
    if shutdown_flag is None:
        return False
    try:
        return shutdown_flag.exists()
    except OSError:
        return False


def is_stale_sample(
    *,
    now: float,
    last_loop_run: float | None,
    written_at: float | None,
    file_mtime: float | None,
    started_at: float,
    grace_s: float,
    stale_ceiling_s: float,
    last_runtime_service_progress: float | None = None,
    service_progress_grace_s: float = 240.0,
) -> bool:
    """Per-sample staleness: is the loop-liveness beacon older than the ceiling
    (or never written) past the boot grace? This is the time-window half of the
    kill decision; ``should_kill`` adds the two-consecutive-samples requirement.
    """
    if now - started_at < grace_s:
        return False
    if (
        service_progress_grace_s > 0
        and (last_loop_run or 0.0) <= 0.0
        and (last_runtime_service_progress or 0.0) > 0.0
        and (now - float(last_runtime_service_progress)) < service_progress_grace_s
    ):
        return False
    live_ts = _liveness_timestamp(last_loop_run, written_at, file_mtime)
    if live_ts <= 0.0:
        # No heartbeat file at all, and we are past grace → loop never lived.
        return True
    return (now - live_ts) >= stale_ceiling_s


def should_kill(
    *,
    now: float,
    last_loop_run: float | None,
    written_at: float | None,
    file_mtime: float | None,
    started_at: float,
    grace_s: float,
    stale_ceiling_s: float,
    consecutive_stale: int,
    last_runtime_service_progress: float | None = None,
    service_progress_grace_s: float = 240.0,
) -> bool:
    """Kill only after the boot grace AND two consecutive stale samples.

    Staleness = the loop's last callback OR the heartbeat writer is older than
    the ceiling. Missing file (never written) past grace also counts — the loop
    never came up.
    """
    return is_stale_sample(
        now=now,
        last_loop_run=last_loop_run,
        written_at=written_at,
        file_mtime=file_mtime,
        started_at=started_at,
        grace_s=grace_s,
        stale_ceiling_s=stale_ceiling_s,
        last_runtime_service_progress=last_runtime_service_progress,
        service_progress_grace_s=service_progress_grace_s,
    ) and consecutive_stale >= 2


def _liveness_timestamp(
    last_loop_run: float | None,
    written_at: float | None,
    file_mtime: float | None,
) -> float:
    """The authoritative 'loop is alive' timestamp.

    last_loop_run is the ONLY true loop-liveness signal (advanced solely when
    the event loop runs the heartbeat callback). file_mtime must NOT be mixed in
    via max(): the StallWatchdog daemon keeps rewriting the file ~1x/s even when
    the loop is wedged, so a fresh mtime would mask a stale last_loop_run and
    hide the wedge. Fall back to written_at, then file_mtime, only when
    last_loop_run is absent (older/partial heartbeat formats).
    """
    if (last_loop_run or 0.0) > 0.0:
        return float(last_loop_run)
    if (written_at or 0.0) > 0.0:
        return float(written_at)
    return float(file_mtime or 0.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--stale-ceiling", type=float, default=180.0)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument(
        "--grace",
        type=float,
        default=240.0,
        help="Seconds after sentinel start before kills are allowed (cold boot/model load).",
    )
    parser.add_argument(
        "--service-progress-grace",
        type=float,
        default=240.0,
        help=(
            "Seconds a fresh API/UI service-progress proof suppresses stale-loop kills. "
            "This prevents false kills during runtime loop handoff while the desktop API "
            "is actively responding."
        ),
    )
    parser.add_argument(
        "--shutdown-flag",
        type=Path,
        default=None,
        help="If this file exists, an intentional shutdown is underway; never kill.",
    )
    parser.add_argument(
        "--tombstone-dir",
        type=Path,
        default=Path("data/error_logs/liveness"),
    )
    args = parser.parse_args(argv)

    if _OBSERVER.process(args.pid) is None:
        return 0  # Already gone.

    signal.signal(signal.SIGHUP, signal.SIG_IGN)  # survive parent re-exec

    started_at = time.monotonic()
    started_wall = time.time()
    consecutive_stale = 0
    interval = max(0.5, float(args.interval))
    # Sanity floor only (guards against 0/negative); the operator/default sets
    # the real policy (default 180s). NOT floored to 30s — that would silently
    # override an intentionally low ceiling.
    ceiling = max(1.0, float(args.stale_ceiling))
    grace = max(0.0, float(args.grace))
    service_grace = max(0.0, float(args.service_progress_grace))

    # started_at uses monotonic for the grace window; staleness uses wall clock
    # because the heartbeat timestamps are wall clock.
    while _OBSERVER.process(args.pid) is not None:
        now = time.time()
        last_loop_run, written_at, file_mtime = read_heartbeat(args.heartbeat)
        last_service_progress = read_runtime_service_progress(args.heartbeat)
        heartbeat_state = read_heartbeat_state(args.heartbeat)
        if heartbeat_state in {"retired", "stopped", "disabled"}:
            return 0
        live_ts = _liveness_timestamp(last_loop_run, written_at, file_mtime)
        if (
            service_grace > 0
            and (last_loop_run or 0.0) <= 0.0
            and (last_service_progress or 0.0) > 0.0
            and (now - float(last_service_progress)) < service_grace
        ):
            consecutive_stale = 0
            time.sleep(interval)
            continue
        stale_age = (now - live_ts) if live_ts > 0.0 else (now - started_wall)
        is_stale = stale_age >= ceiling
        consecutive_stale = consecutive_stale + 1 if is_stale else 0

        if _shutdown_in_progress(args.shutdown_flag):
            consecutive_stale = 0
            time.sleep(interval)
            continue

        if should_kill(
            now=now,
            last_loop_run=last_loop_run,
            written_at=written_at,
            file_mtime=file_mtime,
            started_at=time.monotonic(),  # compared against started_at below
            grace_s=0.0,  # grace handled explicitly via monotonic window
            stale_ceiling_s=ceiling,
            consecutive_stale=consecutive_stale,
            last_runtime_service_progress=last_service_progress,
            service_progress_grace_s=service_grace,
        ) and (time.monotonic() - started_at) >= grace:
            killed = kill_tree(args.pid)
            tombstone = {
                "schema": "aura.liveness_sentinel.tombstone.v1",
                "reason": "external liveness sentinel killed wedged runtime tree",
                "stale_age_s": round(stale_age, 1),
                "stale_ceiling_s": ceiling,
                "last_loop_run": last_loop_run,
                "written_at": written_at,
                "file_mtime": file_mtime,
                "last_runtime_service_progress": last_service_progress,
                "killed_pids": killed,
                "written_at_kill": now,
            }
            try:
                args.tombstone_dir.mkdir(parents=True, exist_ok=True)
                (
                    args.tombstone_dir / f"liveness_tombstone_{int(now)}.json"
                ).write_text(json.dumps(tombstone, indent=2))
            except OSError:
                pass
            return 0

        time.sleep(interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
