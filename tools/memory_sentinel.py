#!/usr/bin/env python3
"""External memory sentinel: the killer that cannot be suspended with you.

The in-process MemoryWatchdog thread was alive during the 115GB host
crash and still couldn't save the machine: when the hog is the main
process itself, macOS suspends/thrashes the whole process — including
every watchdog thread inside it. Protection must live OUTSIDE the
process being protected.

This sentinel is a tiny standalone process (stdlib + psutil only, no
Aura imports) spawned at boot. It samples the target process tree's
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

RING_MAX_LINES = 600  # ~20 minutes at 2s
IMMEDIATE_KILL_OVERSHOOT = 1.15

# macOS killed Aura as 'largest compressed process Python 78557 MB'
# while RSS read 20GB: compressed pages leave RSS but live in
# phys_footprint. Every guard must watch footprint, not RSS alone.
try:
    import ctypes

    class _RUsageV4(ctypes.Structure):
        _fields_ = [
            ("ri_uuid", ctypes.c_uint8 * 16),
            ("ri_user_time", ctypes.c_uint64),
            ("ri_system_time", ctypes.c_uint64),
            ("ri_pkg_idle_wkups", ctypes.c_uint64),
            ("ri_interrupt_wkups", ctypes.c_uint64),
            ("ri_pageins", ctypes.c_uint64),
            ("ri_wired_size", ctypes.c_uint64),
            ("ri_resident_size", ctypes.c_uint64),
            ("ri_phys_footprint", ctypes.c_uint64),
            ("ri_proc_start_abstime", ctypes.c_uint64),
            ("ri_proc_exit_abstime", ctypes.c_uint64),
            ("ri_child_user_time", ctypes.c_uint64),
            ("ri_child_system_time", ctypes.c_uint64),
            ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
            ("ri_child_interrupt_wkups", ctypes.c_uint64),
            ("ri_child_pageins", ctypes.c_uint64),
            ("ri_child_elapsed_abstime", ctypes.c_uint64),
            ("ri_diskio_bytesread", ctypes.c_uint64),
            ("ri_diskio_byteswritten", ctypes.c_uint64),
            ("ri_cpu_time_qos_default", ctypes.c_uint64),
            ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
            ("ri_cpu_time_qos_background", ctypes.c_uint64),
            ("ri_cpu_time_qos_utility", ctypes.c_uint64),
            ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
            ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
            ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
            ("ri_billed_system_time", ctypes.c_uint64),
            ("ri_serviced_system_time", ctypes.c_uint64),
            ("ri_logical_writes", ctypes.c_uint64),
            ("ri_lifetime_max_phys_footprint", ctypes.c_uint64),
            ("ri_instructions", ctypes.c_uint64),
            ("ri_cycles", ctypes.c_uint64),
            ("ri_billed_energy", ctypes.c_uint64),
            ("ri_serviced_energy", ctypes.c_uint64),
            # Full rusage_info_v4 is 304 bytes; truncating here at 280 let
            # the kernel write 24 bytes past the buffer on every snapshot.
            ("ri_interval_max_phys_footprint", ctypes.c_uint64),
            ("ri_runnable_time", ctypes.c_uint64),
            ("ri_flags", ctypes.c_uint64),
            # Spare capacity so a future flavor bump can never overrun.
            ("_ri_spare", ctypes.c_uint64 * 16),
        ]

    _LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib")
    _LIBPROC.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    _LIBPROC.proc_pid_rusage.restype = ctypes.c_int
    _LIBPROC.proc_listchildpids.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    _LIBPROC.proc_listchildpids.restype = ctypes.c_int
except (OSError, AttributeError):  # non-macOS or restricted
    _LIBPROC = None


def current_phys_footprint_bytes(usage: _RUsageV4) -> int:
    """Return current footprint; the lifetime maximum is telemetry, not usage."""
    current = int(getattr(usage, "ri_phys_footprint", 0) or 0)
    if current > 0:
        return current
    return int(getattr(usage, "ri_resident_size", 0) or 0)


def phys_footprint_mb(pid: int) -> float:
    """Current RSS + compressed + IOKit-mapped footprint."""
    if _LIBPROC is None:
        return 0.0
    ru = _RUsageV4()
    if _LIBPROC.proc_pid_rusage(int(pid), 4, ctypes.byref(ru)) != 0:
        return 0.0
    return current_phys_footprint_bytes(ru) / (1024 * 1024)


def child_pids(root_pid: int, *, recursive: bool = True, max_children: int = 128) -> list[int]:
    """Return child pids without relying on psutil's recursive ppid map."""

    if sys.platform == "darwin" and _LIBPROC is not None:
        seen: set[int] = set()
        frontier = [int(root_pid)]
        deadline = time.monotonic() + 1.0
        while frontier and len(seen) < max_children and time.monotonic() < deadline:
            parent = frontier.pop(0)
            try:
                buffer = (ctypes.c_int * max_children)()
                count = int(
                    _LIBPROC.proc_listchildpids(
                        int(parent),
                        ctypes.byref(buffer),
                        ctypes.sizeof(buffer),
                    )
                )
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                break
            if count <= 0:
                break
            for raw_pid in list(buffer)[: min(count, max_children)]:
                pid = int(raw_pid)
                if pid <= 0:
                    continue
                if pid in seen:
                    continue
                seen.add(pid)
                if recursive and len(seen) < max_children:
                    frontier.append(pid)
        return list(seen)

    try:
        return [child.pid for child in psutil.Process(root_pid).children(recursive=recursive)]
    except psutil.Error:
        return []


def tree_rss_mb(root: psutil.Process) -> tuple[float, float, int, float]:
    core = 0.0
    children = 0.0
    count = 1
    footprint = phys_footprint_mb(root.pid)
    try:
        core = root.memory_info().rss / (1024 * 1024)
    except psutil.Error:
        return 0.0, 0.0, 0, footprint
    try:
        kids = child_pids(root.pid, recursive=True)
        count += len(kids)
        for child_pid in kids:
            try:
                child = psutil.Process(child_pid)
                children += child.memory_info().rss / (1024 * 1024)
                footprint += phys_footprint_mb(child.pid)
            except psutil.Error:
                continue
    except psutil.Error:
        pass
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


def kill_tree(root: psutil.Process) -> list[int]:
    killed: list[int] = []
    procs: list[psutil.Process] = []
    for pid in child_pids(root.pid, recursive=True):
        try:
            procs.append(psutil.Process(pid))
        except psutil.Error:
            continue
    procs.append(root)
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

    try:
        target = psutil.Process(args.pid)
    except psutil.Error:
        return 0  # Already gone; nothing to guard.

    # Die quietly if our own parent re-execs; we re-attach by pid anyway.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    consecutive_over = 0
    # Bounded by the target's own lifetime: the sentinel exists exactly
    # as long as the process it guards.
    while target.is_running():
        core_mb, child_mb, proc_count, footprint_mb = tree_rss_mb(target)
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
            killed = kill_tree(target)
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
            return 0

        time.sleep(max(0.5, args.interval))

    # Target vanished without OUR kill: capture the unified log around
    # the death immediately — silent SIGKILLs leave their only evidence
    # in the kernel namespace, and it ages out fast.
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
