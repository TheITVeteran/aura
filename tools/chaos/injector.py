"""tools/chaos/injector.py

Chaos injection — break things deliberately and prove repair works.

Catalogue (every entry here is registered — pinned by
tests/test_chaos_injector.py; a fault documented but not implemented is a
false capability claim):

  induce_event_loop_lag    — sleep on the event loop for ~1.2s
  force_model_load_failure — flip the model registry to point at a nonexistent path
  expire_api_keys          — flip env vars to invalid values
  delete_vector_index      — move one vector store partition aside
  fill_disk                — write bounded disk-pressure files in a safe temp target
  sever_network            — block outbound connections via local proxy

Roadmap (deliberately NOT implemented yet — each needs a live kernel and
blast-radius design): kill_subprocess, corrupt_sqlite_row,
break_memory_facade, break_agency_pathway.

All faults self-restore; AURA_CHAOS_RESTORE_SECONDS overrides every
restore delay (drills and tests shorten it; production default keeps the
per-fault windows).

Each fault returns a dict ``{kind, applied: True|False, detail}``. The
chaos run records the fault and the system's repair signal (the
StabilityGuardian and ResilienceEngine telemetry) for later analysis.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import sys
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.utils.task_tracker import get_task_tracker  # noqa: E402

logger = logging.getLogger("Aura.Chaos")


def _restore_delay_s(default_s: float) -> float:
    """Per-fault restore window, overridable globally for drills/tests."""
    raw = os.environ.get("AURA_CHAOS_RESTORE_SECONDS")
    if raw is None:
        return default_s
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default_s


_FAULTS: dict[str, Callable[[], Awaitable[dict[str, Any]]]] = {}
_FAULT_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TimeoutError,
    OSError,
    ValueError,
    TypeError,
    KeyError,
)


def register(name: str) -> Callable[[Callable[[], Awaitable[dict[str, Any]]]], Callable[[], Awaitable[dict[str, Any]]]]:
    def deco(fn: Callable[[], Awaitable[dict[str, Any]]]) -> Callable[[], Awaitable[dict[str, Any]]]:
        _FAULTS[name] = fn
        return fn
    return deco


@register("induce_event_loop_lag")
async def _induce_loop_lag() -> dict[str, Any]:
    t0 = time.monotonic()
    # Synchronous sleep on the event loop thread to simulate a stall.
    time.sleep(1.2)  # noqa: ASYNC251
    return {"kind": "induce_event_loop_lag", "applied": True, "lagged_ms": int((time.monotonic() - t0) * 1000)}


@register("force_model_load_failure")
async def _force_model_load_failure() -> dict[str, Any]:
    prev = os.environ.get("AURA_MODEL")
    os.environ["AURA_MODEL"] = str(Path(tempfile.gettempdir()) / "nonexistent-model-injection")
    try:
        # Restore after 60 seconds so the next chaos cycle can re-roll.
        await asyncio.sleep(0)
    finally:
        async def _restore():
            await asyncio.sleep(_restore_delay_s(60.0))
            if prev is None:
                os.environ.pop("AURA_MODEL", None)
            else:
                os.environ["AURA_MODEL"] = prev
        get_task_tracker().create_task(_restore())
    return {"kind": "force_model_load_failure", "applied": True, "restored_in_s": 60}


@register("expire_api_keys")
async def _expire_api_keys() -> dict[str, Any]:
    keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    flipped = {}
    for k in keys:
        prev = os.environ.get(k)
        flipped[k] = prev
        os.environ[k] = "invalid-injection"

    async def _restore():
        await asyncio.sleep(_restore_delay_s(60.0))
        for k, prev in flipped.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev
    get_task_tracker().create_task(_restore())
    return {"kind": "expire_api_keys", "applied": True, "keys": list(flipped.keys())}


@register("delete_vector_index")
async def _delete_vector_index() -> dict[str, Any]:
    target = Path.home() / ".aura" / "data" / "vector_index"
    backup = Path.home() / ".aura" / "data" / f"vector_index.injected.{int(time.time())}"
    if not target.exists():
        return {"kind": "delete_vector_index", "applied": False, "reason": "no_target"}
    target.rename(backup)

    async def _restore():
        await asyncio.sleep(_restore_delay_s(120.0))
        if backup.exists() and not target.exists():
            backup.rename(target)
    get_task_tracker().create_task(_restore())
    return {"kind": "delete_vector_index", "applied": True, "moved_to": str(backup)}


@register("fill_disk")
async def _fill_disk() -> dict[str, Any]:
    target_root = Path(os.environ.get("AURA_CHAOS_DISK_TARGET_DIR", tempfile.gettempdir())).expanduser().resolve()
    safe_roots = {Path(tempfile.gettempdir()).resolve()}
    explicit_target = "AURA_CHAOS_DISK_TARGET_DIR" in os.environ
    if not explicit_target and not any(target_root == root or root in target_root.parents for root in safe_roots):
        return {"kind": "fill_disk", "applied": False, "reason": "unsafe_target_root", "target": str(target_root)}

    max_mb = int(os.environ.get("AURA_CHAOS_DISK_MAX_MB", "64"))
    max_mb = max(1, min(max_mb, 512))
    pressure_dir = target_root / f"aura-chaos-disk-pressure-{uuid.uuid4().hex}"
    pressure_dir.mkdir(parents=True, exist_ok=False)
    pressure_file = pressure_dir / "pressure.bin"
    chunk = b"\0" * (1024 * 1024)
    bytes_written = 0

    try:
        with pressure_file.open("wb") as fh:
            for _ in range(max_mb):
                fh.write(chunk)
                bytes_written += len(chunk)
            fh.flush()
            os.fsync(fh.fileno())
    except OSError as exc:
        return {
            "kind": "fill_disk",
            "applied": bytes_written > 0,
            "target": str(pressure_file),
            "bytes_written": bytes_written,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }

    # Parse the window ONCE, tolerantly, outside the restore task: a garbage
    # value must not kill the cleanup task and strand pressure files.
    try:
        disk_default_s = float(os.environ.get("AURA_CHAOS_DISK_RESTORE_SECONDS", "30"))
    except ValueError:
        disk_default_s = 30.0
    restore_in_s = _restore_delay_s(disk_default_s)

    async def _restore():
        await asyncio.sleep(restore_in_s)
        if pressure_file.exists():
            pressure_file.unlink()
        if pressure_dir.exists():
            pressure_dir.rmdir()

    get_task_tracker().create_task(_restore(), name="chaos.fill_disk.restore_pressure_file")
    return {
        "kind": "fill_disk",
        "applied": True,
        "target": str(pressure_file),
        "bytes_written": bytes_written,
        "restored_in_s": restore_in_s,
    }


@register("sever_network")
async def _sever_network() -> dict[str, Any]:
    # Set HTTPS_PROXY to a local dead port to block outbound HTTP.
    prev = os.environ.get("HTTPS_PROXY")
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:1"

    async def _restore():
        await asyncio.sleep(_restore_delay_s(45.0))
        if prev is None:
            os.environ.pop("HTTPS_PROXY", None)
        else:
            os.environ["HTTPS_PROXY"] = prev
    get_task_tracker().create_task(_restore(), name="chaos.sever_network.restore_proxy")
    return {"kind": "sever_network", "applied": True, "restored_in_s": 45}


async def inject_random_fault() -> dict[str, Any]:
    name = random.choice(list(_FAULTS.keys()))
    try:
        out = await _FAULTS[name]()
    except _FAULT_RECOVERABLE_ERRORS as exc:
        logger.warning("Chaos fault %s failed to apply: %s", name, exc, exc_info=True)
        return {"kind": name, "applied": False, "error": str(exc), "error_type": type(exc).__name__}
    return out


async def main(argv: list[str]) -> int:
    """CLI entry-point: ``python -m tools.chaos.injector --kind <name>``"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="random")
    args = parser.parse_args(argv)
    if args.kind == "random":
        out = await inject_random_fault()
    else:
        fn = _FAULTS.get(args.kind)
        if fn is None:
            print(f"unknown fault: {args.kind}; choices: {list(_FAULTS.keys())}")
            return 1
        out = await fn()
    print(out)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main(sys.argv[1:])))
