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
import sys
import time
from pathlib import Path

import psutil

RING_MAX_LINES = 600  # ~20 minutes at 2s


def tree_rss_mb(root: psutil.Process) -> tuple[float, float, int]:
    core = 0.0
    children = 0.0
    count = 1
    try:
        core = root.memory_info().rss / (1024 * 1024)
    except psutil.Error:
        return 0.0, 0.0, 0
    try:
        kids = root.children(recursive=True)
        count += len(kids)
        for child in kids:
            try:
                children += child.memory_info().rss / (1024 * 1024)
            except psutil.Error:
                continue
    except psutil.Error:
        pass
    return core, children, count


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
    try:
        procs = root.children(recursive=True)
    except psutil.Error:
        pass
    procs.append(root)
    for proc in procs:
        try:
            proc.kill()  # SIGKILL: a suspended process cannot handle anything gentler.
            killed.append(proc.pid)
        except psutil.Error:
            continue
    return killed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--lethal-mb", type=float, default=57344.0)
    parser.add_argument("--interval", type=float, default=2.0)
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
        core_mb, child_mb, proc_count = tree_rss_mb(target)
        managed = core_mb + child_mb
        entry = {
            "at": time.time(),
            "core_mb": round(core_mb, 1),
            "child_mb": round(child_mb, 1),
            "managed_mb": round(managed, 1),
            "procs": proc_count,
        }
        write_ring(args.ring, entry)

        if managed >= args.lethal_mb:
            consecutive_over += 1
        else:
            consecutive_over = 0

        # Two consecutive samples over lethal: kill. No reclaim attempts —
        # the in-process watchdog owns graceful reclaim at lower ceilings;
        # by the time the EXTERNAL sentinel acts, cooperation has already
        # failed and every second risks the host.
        if consecutive_over >= 2:
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
