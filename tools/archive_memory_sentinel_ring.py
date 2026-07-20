#!/usr/bin/env python3
"""Preserve an operational sentinel ring as an append-only proof stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Never

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.atomic_writer import atomic_write_bytes  # noqa: E402
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402

SCHEMA = "aura.memory_sentinel_ring_archive.v1"
STATE_SCHEMA = "aura.memory_sentinel_ring_archive_state.v1"
RECEIPT_SCHEMA = "aura.memory_sentinel_ring_archive_receipt.v1"
_MAX_SOURCE_BYTES = 512 * 1024 * 1024


class RingArchiveError(RuntimeError):
    """Stable fail-closed archive error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise RingArchiveError(code)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sample(raw: bytes) -> tuple[float, dict[str, Any]]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RingArchiveError("source_sample_invalid") from exc
    if not isinstance(value, dict):
        _fail("source_sample_invalid")
    observed_at = value.get("at")
    if (
        isinstance(observed_at, bool)
        or not isinstance(observed_at, (int, float))
        or not math.isfinite(float(observed_at))
        or float(observed_at) <= 0.0
        or value.get("observation_source") != "host"
        or not isinstance(value.get("guard_stage"), str)
    ):
        _fail("source_sample_invalid")
    return float(observed_at), value


def _records(raw: bytes) -> list[tuple[float, bytes]]:
    if raw and not raw.endswith(b"\n"):
        _fail("source_ring_partial_record")
    result: list[tuple[float, bytes]] = []
    previous = -math.inf
    for line in raw.splitlines():
        if not line:
            _fail("source_ring_empty_record")
        observed_at, _value = _sample(line)
        if observed_at <= previous:
            _fail("source_ring_order_invalid")
        previous = observed_at
        result.append((observed_at, line))
    return result


def select_new_records(
    raw: bytes,
    *,
    last_at: float | None,
    source_replaced: bool,
) -> list[tuple[float, bytes]]:
    """Select an exact monotonic suffix and prove overlap after compaction."""

    records = _records(raw)
    if last_at is None:
        return records
    if source_replaced and records and records[0][0] > last_at:
        _fail("source_ring_rotation_gap")
    return [(observed_at, line) for observed_at, line in records if observed_at > last_at]


def _target_state(pid: int, started_at: float) -> str:
    try:
        process = psutil.Process(pid)
        observed = float(process.create_time())
        status = str(process.status()).lower()
    except psutil.NoSuchProcess:
        return "gone"
    except (psutil.Error, OSError, RuntimeError, TypeError, ValueError):
        return "unobservable"
    if abs(observed - started_at) > 0.5:
        return "reused"
    return "gone" if status in {"dead", "zombie"} else "current"


def _write_state(
    path: Path,
    *,
    status: str,
    sample_count: int,
    last_at: float | None,
    source_inode: int | None,
    source_offset: int,
) -> None:
    material = {
        "schema": STATE_SCHEMA,
        "status": status,
        "archiver_pid": os.getpid(),
        "sample_count": sample_count,
        "last_at": last_at,
        "source_inode": source_inode,
        "source_offset": source_offset,
        "updated_at": time.time(),
    }
    atomic_write_bytes(
        path,
        _canonical({**material, "state_sha256": _sha(_canonical(material))}) + b"\n",
        mode=0o600,
    )


def archive(
    *,
    source: Path,
    destination: Path,
    state_path: Path,
    receipt_path: Path,
    target_pid: int,
    interval_s: float,
) -> dict[str, Any]:
    source = source.expanduser().resolve(strict=True)
    destination = destination.expanduser().resolve(strict=False)
    if any(path.exists() or path.is_symlink() for path in (destination, receipt_path)):
        _fail("archive_output_exists")
    if destination.parent.resolve(strict=True) != state_path.parent.resolve(strict=True):
        _fail("archive_output_parent_mismatch")
    try:
        target_started_at = float(psutil.Process(target_pid).create_time())
    except (psutil.Error, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RingArchiveError("target_identity_unavailable") from exc
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    sample_count = 0
    last_at: float | None = None
    source_inode: int | None = None
    source_offset = 0
    unavailable_since: float | None = None
    with destination.open("xb", buffering=0) as output:
        os.chmod(destination, 0o600)
        while True:
            state = _target_state(target_pid, target_started_at)
            if state in {"gone", "reused"}:
                final_pass = True
            elif state == "unobservable":
                now = time.monotonic()
                unavailable_since = unavailable_since or now
                if now - unavailable_since >= 30.0:
                    _fail("target_identity_unobservable")
                final_pass = False
            else:
                unavailable_since = None
                final_pass = False
            with source.open("rb") as source_handle:
                stat = os.fstat(source_handle.fileno())
                replaced = source_inode is not None and stat.st_ino != source_inode
                if replaced or stat.st_size < source_offset:
                    source_offset = 0
                    replaced = source_inode is not None
                if stat.st_size > _MAX_SOURCE_BYTES:
                    _fail("source_ring_too_large")
                source_handle.seek(source_offset)
                raw = source_handle.read(stat.st_size - source_offset)
                if len(raw) != stat.st_size - source_offset:
                    _fail("source_ring_short_read")
            selected = select_new_records(
                raw,
                last_at=last_at,
                source_replaced=replaced,
            )
            for observed_at, line in selected:
                output.write(line + b"\n")
                sample_count += 1
                last_at = observed_at
            if selected:
                output.flush()
                os.fsync(output.fileno())
            source_inode = stat.st_ino
            source_offset = stat.st_size
            _write_state(
                state_path,
                status="finalizing" if final_pass else "running",
                sample_count=sample_count,
                last_at=last_at,
                source_inode=source_inode,
                source_offset=source_offset,
            )
            if final_pass:
                break
            time.sleep(interval_s)
    archive_raw = read_stable_bytes(destination, max_bytes=_MAX_SOURCE_BYTES)
    material = {
        "schema": RECEIPT_SCHEMA,
        "status": "passed",
        "source": str(source),
        "archive": str(destination),
        "archive_sha256": _sha(archive_raw),
        "sample_count": sample_count,
        "head_at": _records(archive_raw)[0][0] if sample_count else None,
        "tail_at": last_at,
        "target_pid": target_pid,
        "target_started_at": target_started_at,
        "finished_at": time.time(),
    }
    receipt = {**material, "receipt_sha256": _sha(_canonical(material))}
    atomic_write_bytes(receipt_path, _canonical(receipt) + b"\n", mode=0o600)
    _write_state(
        state_path,
        status="passed",
        sample_count=sample_count,
        last_at=last_at,
        source_inode=source_inode,
        source_offset=source_offset,
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--target-pid", type=int, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or not 1.0 <= args.interval <= 60.0:
        parser.error("--interval must be between 1 and 60 seconds")
    try:
        result = archive(
            source=args.source,
            destination=args.archive,
            state_path=args.state,
            receipt_path=args.receipt,
            target_pid=args.target_pid,
            interval_s=args.interval,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            f"archive_memory_sentinel_ring: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
