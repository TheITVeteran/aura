#!/usr/bin/env python3
"""External memory sentinel: the killer that cannot be suspended with you.

The in-process MemoryWatchdog thread was alive during the 115GB host
crash and still couldn't save the machine: when the hog is the main
process itself, macOS suspends/thrashes the whole process — including
every watchdog thread inside it. Protection must live OUTSIDE the
process being protected.

This sentinel is a tiny standalone process with only the lightweight canonical
resource observer and process-action dependencies. It samples the target process tree's
RSS on a tight interval and, past the lethal ceiling, SIGKILLs the
entire tree — no cooperation from the dying process required. Every
sample lands in a ring file so the next post-mortem has data even if
everything else is lost.

It exits on its own when the target process disappears.

Usage:
    python tools/memory_sentinel.py --pid 12345 \
        [--lethal-mb 57344] [--interval 2.0] [--ring data/.../ring.jsonl]
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

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.process_footprint import (  # noqa: E402
    DarwinRUsageInfoV4,
    current_darwin_footprint_bytes,
    darwin_phys_footprint_bytes,
)
from core.runtime.resource_observation import (  # noqa: E402
    HostResourceObserver,
    ObservationSource,
)

_RUsageV4 = DarwinRUsageInfoV4
current_phys_footprint_bytes = current_darwin_footprint_bytes

_OBSERVER = HostResourceObserver(
    source=ObservationSource.HOST,
    scenario_id="external-memory-sentinel",
)

RING_MAX_LINES = 600  # ~20 minutes at 2s
IMMEDIATE_KILL_OVERSHOOT = 1.15

def phys_footprint_mb(pid: int) -> float:
    """Current RSS + compressed + IOKit-mapped footprint."""
    return darwin_phys_footprint_bytes(pid) / (1024 * 1024)


def child_pids(root_pid: int, *, recursive: bool = True, max_children: int = 128) -> list[int]:
    """Return observed descendants of one protected root."""

    table = _OBSERVER.process_table()
    if not table.available:
        return []
    direct = [process.pid for process in table.processes if process.ppid == root_pid]
    if not recursive:
        return direct[:max_children]
    return [
        process.pid
        for process in table.processes
        if root_pid in process.ancestor_pids
    ][:max_children]


def tree_rss_mb(root_pid: int) -> tuple[float, float, int, float]:
    table = _OBSERVER.process_table()
    if not table.available:
        return 0.0, 0.0, 0, 0.0
    root = next((process for process in table.processes if process.pid == root_pid), None)
    if root is None:
        return 0.0, 0.0, 0, 0.0
    descendants = [
        process for process in table.processes if root_pid in process.ancestor_pids
    ]
    core = 0.0
    children = 0.0
    count = 1
    footprint = phys_footprint_mb(root.pid)
    core = root.rss_bytes / (1024 * 1024)
    count += len(descendants)
    for child in descendants:
        children += child.rss_bytes / (1024 * 1024)
        footprint += phys_footprint_mb(child.pid)
    return core, children, count, footprint


def write_ring(path: Path, entry: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        if path.exists():
            lines = path.read_text(errors="replace").splitlines()[-(RING_MAX_LINES - 1):]
        lines.append(json.dumps(entry))
        tmp = path.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + "\n")
        os.replace(tmp, path)
    except OSError:
        pass  # The sentinel must never die from bookkeeping.


def kill_tree(root_pid: int) -> list[int]:
    killed: list[int] = []
    procs: list[psutil.Process] = []
    for pid in child_pids(root_pid, recursive=True):
        try:
            procs.append(psutil.Process(pid))
        except psutil.Error:
            continue
    try:
        procs.append(psutil.Process(root_pid))
    except psutil.Error:
        pass
    for proc in procs:
        try:
            proc.kill()  # SIGKILL: a suspended process cannot handle anything gentler.
            killed.append(proc.pid)
        except psutil.Error:
            continue
    return killed


def should_kill_for_memory(
    *,
    managed_mb: float,
    lethal_mb: float,
    consecutive_over: int,
    overshoot_factor: float = IMMEDIATE_KILL_OVERSHOOT,
) -> bool:
    """Return true when the sentinel must kill the protected process tree."""

    if managed_mb >= lethal_mb * max(1.0, float(overshoot_factor)):
        return True
    return consecutive_over >= 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--lethal-mb", type=float, default=46080.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--ring",
        type=Path,
        default=Path("data/error_logs/memory/sentinel_ring.jsonl"),
    )
    parser.add_argument(
        "--tombstone-dir",
        type=Path,
        default=Path("data/error_logs/memory"),
    )
    args = parser.parse_args(argv)

    if _OBSERVER.process(args.pid) is None:
        return 0  # Already gone; nothing to guard.

    # Die quietly if our own parent re-execs; we re-attach by pid anyway.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    # Death evidence: every start and exit leaves one line in sentinel.log.
    # A live incident once left a sentinel dead within 85s of arming with a
    # zero-byte log — undiagnosable. Never again: announce arming, announce
    # every exit path, and turn SIGTERM into a logged, orderly exit.
    def _evidence(line: str) -> None:
        try:
            print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] pid={os.getpid()} {line}", flush=True)
        except OSError:
            pass  # Evidence must never kill the guard.

    def _on_sigterm(signum, frame):  # noqa: ARG001 — signal handler signature
        _evidence(f"exiting: SIGTERM received while guarding target pid={args.pid}")
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)
    _evidence(
        f"armed: target pid={args.pid} lethal_mb={args.lethal_mb:.0f} "
        f"interval_s={args.interval:.1f} ring={args.ring}"
    )

    consecutive_over = 0
    # Bounded by the target's own lifetime: the sentinel exists exactly
    # as long as the process it guards.
    while (target := _OBSERVER.process(args.pid)) is not None:
        if target.status.lower() in {"dead", "zombie"}:
            break
        core_mb, child_mb, proc_count, footprint_mb = tree_rss_mb(args.pid)
        # memorystatus kills on footprint (RSS + compressed); guard on
        # whichever view is larger so compression cannot hide a runaway.
        managed = max(core_mb + child_mb, footprint_mb)
        entry = {
            "at": time.time(),
            "core_mb": round(core_mb, 1),
            "child_mb": round(child_mb, 1),
            "footprint_mb": round(footprint_mb, 1),
            "managed_mb": round(managed, 1),
            "procs": proc_count,
            "observation_source": _OBSERVER.provenance.source.value,
            "observation_scenario_id": _OBSERVER.provenance.scenario_id,
        }
        write_ring(args.ring, entry)

        if managed >= args.lethal_mb:
            consecutive_over += 1
        else:
            consecutive_over = 0

        # Two consecutive samples over lethal: kill. A single large overshoot
        # also kills immediately; a runaway can add tens of GB between 1s
        # samples, and protecting the host beats waiting for confirmation.
        # No reclaim attempts here — the in-process watchdog owns graceful
        # reclaim at lower ceilings.
        if should_kill_for_memory(
            managed_mb=managed,
            lethal_mb=args.lethal_mb,
            consecutive_over=consecutive_over,
        ):
            killed = kill_tree(args.pid)
            tombstone = {
                "schema": "aura.memory_sentinel.tombstone.v1",
                "reason": "external sentinel killed process tree at lethal ceiling",
                "lethal_mb": args.lethal_mb,
                "final_sample": entry,
                "killed_pids": killed,
                "written_at": time.time(),
            }
            try:
                args.tombstone_dir.mkdir(parents=True, exist_ok=True)
                (
                    args.tombstone_dir / f"sentinel_tombstone_{int(time.time())}.json"
                ).write_text(json.dumps(tombstone, indent=2))
            except OSError:
                pass
            _evidence(
                f"exiting: killed target tree at lethal ceiling "
                f"(managed_mb={managed:.0f} >= lethal_mb={args.lethal_mb:.0f}, killed={killed})"
            )
            return 0

        time.sleep(max(0.5, args.interval))

    # Target vanished without OUR kill: capture the unified log around
    # the death immediately — silent SIGKILLs leave their only evidence
    # in the kernel namespace, and it ages out fast.
    _evidence(f"exiting: target pid={args.pid} vanished; capturing death syslog")
    try:
        capture = subprocess.run(
            [
                "log", "show", "--last", "3m",
                "--predicate",
                f'eventMessage CONTAINS "{args.pid}" OR '
                'eventMessage CONTAINS "memorystatus" OR '
                'eventMessage CONTAINS "Python" OR '
                'eventMessage CONTAINS "SIGKILL"',
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        out_path = args.tombstone_dir / f"death_syslog_{int(time.time())}.log"
        args.tombstone_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            f"target pid {args.pid} vanished at {time.time()}\n"
            + (capture.stdout or "")[-200_000:]
        )
    except (OSError, subprocess.SubprocessError):
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
