"""Register the reasoning background loops on the autonomy conductor.

Two bounded, internal, governed loops complete the flywheel:

* ``reasoning_idle_precompute`` — drains the precompute queue during idle, solving
  verifier-dirty hard problems off the foreground critical path; wins land in the
  solved-cache for instant re-use.
* ``reasoning_self_improve`` — when enough verifier-clean traces accumulate, feeds
  them to the existing governed/validated fine-tune pipe (STaR bootstrap).

Wiring (one line, where the conductor's defaults are registered)::

    from core.brain.reasoning_background import register_reasoning_jobs
    register_reasoning_jobs(conductor)

Idempotent: registering twice is a no-op. Kept out of the conductor module itself so
this lands without entangling concurrent edits there.
"""
from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import os
import threading
from collections.abc import AsyncIterator, Mapping
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.ReasoningBackground")

# Conservative intervals: idle pre-compute is cheap-ish but uses the cortex, and the
# self-improve feed can trigger governed training, so it runs rarely.
_PRECOMPUTE_INTERVAL_S = 180.0
_SELF_IMPROVE_INTERVAL_S = 3600.0
_NONPARAM_INGEST_INTERVAL_S = 1800.0

_registered: set[int] = set()
_encoder_cache: dict[str, Any] = {}
_encoder_lease: Any = None
_encoder_users = 0
_encoder_lifecycle_gate = threading.Lock()


def _flag_on(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "on", "yes", "enabled"}


async def _job_idle_precompute() -> dict[str, Any]:
    from core.brain.reasoning_precompute import idle_precompute_tick

    solved = await idle_precompute_tick(max_items=1, per_item_timeout=90.0)
    return {"ok": True, "solved": solved}


async def _job_self_improve() -> dict[str, Any]:
    from core.brain.reasoning_self_improvement import get_reasoning_self_improvement

    result = await get_reasoning_self_improvement().maybe_improve()
    if not isinstance(result, Mapping):
        raise TypeError("reasoning self-improvement returned a non-mapping result")
    return dict(result)


def _resolve_active_model_path() -> str | None:
    """The cortex model path (so datastore keys come from the SAME model that generates)."""
    try:
        import json
        from pathlib import Path

        active = Path(__file__).resolve().parents[2] / "training" / "fused-model" / "active.json"
        if active.exists():
            data = json.loads(active.read_text(encoding="utf-8"))
            p = str(data.get("active_model_path") or "").strip()
            if p and Path(p).exists():
                return p
    except (OSError, ValueError, TypeError) as exc:
        record_degradation("reasoning_background_active_model", exc)
    return os.getenv("AURA_NONPARAMETRIC_ENCODER_MODEL") or None


@contextlib.asynccontextmanager
async def _encoder_lifecycle_context() -> AsyncIterator[None]:
    # This state is shared across event loops, so an asyncio.Event is not a
    # valid replacement for the loop-agnostic lock.
    while not _encoder_lifecycle_gate.acquire(blocking=False):  # noqa: ASYNC110
        await asyncio.sleep(0.01)
    try:
        yield
    finally:
        _encoder_lifecycle_gate.release()


async def _release_encoder_locked(*, reason: str) -> None:
    global _encoder_lease, _encoder_users
    _encoder_users = max(0, _encoder_users - 1)
    if _encoder_users > 0:
        return
    _encoder_cache.clear()
    lease, _encoder_lease = _encoder_lease, None
    await asyncio.to_thread(gc.collect)
    if lease is not None:
        await lease.release(reason=reason)


async def _release_encoder(*, reason: str) -> None:
    async with _encoder_lifecycle_context():
        await _release_encoder_locked(reason=reason)


async def _get_encoder() -> Any | None:
    """Lazily build (and cache) an MLXEncoder over the cortex. Heavy — guarded by flags."""
    global _encoder_lease, _encoder_users
    path = _resolve_active_model_path()
    if not path:
        return None
    async with _encoder_lifecycle_context():
        if path in _encoder_cache and _encoder_lease is not None:
            _encoder_users += 1
            return _encoder_cache[path]
        if _encoder_users > 0:
            logger.warning(
                "Reasoning encoder path changed while %d user(s) remain active; deferring.",
                _encoder_users,
            )
            return None
        try:
            from mlx_lm import load

            from core.brain.nonparametric_generation import MLXEncoder
            from core.runtime.model_lane_control import (
                acquire_in_process_model_lane,
                run_owned_model_thread_call,
            )

            if _encoder_cache or _encoder_lease is not None:
                _encoder_users = 1
                await _release_encoder_locked(reason="reasoning_encoder_path_changed")
            lease = await acquire_in_process_model_lane(
                owner_id="reasoning-background-encoder",
                model_path=path,
                purpose="benchmark",
                priority=80,
                preemptible=False,
                metadata={"job": "nonparametric_ingest"},
            )
            try:
                model, tok = await run_owned_model_thread_call(
                    lambda: load(path),
                    operation_name="reasoning-background-model-load",
                )
                enc = await run_owned_model_thread_call(
                    lambda: MLXEncoder(model, tok),
                    operation_name="reasoning-background-encoder-init",
                )
            except asyncio.CancelledError:
                await lease.release(reason="reasoning_encoder_load_cancelled")
                raise
            except (ImportError, RuntimeError, OSError, ValueError, TypeError, AttributeError):
                await lease.release(reason="reasoning_encoder_load_failed")
                raise
            _encoder_cache.clear()
            _encoder_cache[path] = enc
            _encoder_lease = lease
            _encoder_users = 1
            return enc
        except asyncio.CancelledError:
            raise
        except (ImportError, RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
            record_degradation("reasoning_background_encoder", exc)
            return None


async def _job_nonparametric_ingest() -> dict[str, Any]:
    """Ingest trusted knowledge into the non-parametric datastore (background-first).

    OFF by default (AURA_NONPARAMETRIC_INGEST). Skips under memory pressure — loading the
    cortex encoder is heavy and must never compete with a resident foreground model on a
    64GB host. Bounded batch per run.
    """
    if not _flag_on("AURA_NONPARAMETRIC_INGEST"):
        return {"status": "disabled"}
    try:
        from core.utils.memory_monitor import get_memory_pressure_snapshot

        if get_memory_pressure_snapshot().refuse_heavy_local_generation:
            return {"status": "skipped_memory_pressure"}
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        pass
    encoder = await _get_encoder()
    if encoder is None:
        return {"status": "no_encoder"}
    try:
        from core.brain.nonparametric_ingest import NonParametricIngestor, collect_trusted_pairs
        from core.brain.nonparametric_memory import get_nonparametric_memory

        mem = get_nonparametric_memory(int(encoder.dim))
        if mem is None:
            return {"status": "no_memory"}
        ing = NonParametricIngestor(mem)
        pairs = collect_trusted_pairs(limit=50)
        from core.runtime.model_lane_control import run_owned_model_thread_call

        def _ingest_and_persist() -> int:
            positions = sum(ing.ingest_sequence(c, a, encoder) for c, a in pairs)
            mem.persist()
            return int(positions)

        positions = await run_owned_model_thread_call(
            _ingest_and_persist,
            operation_name="reasoning-background-nonparametric-ingest",
        )
        return {"ok": True, "ingested_positions": positions, "pairs": len(pairs)}
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("reasoning_background_ingest_job", exc)
        return {"status": "error", "error": f"{type(exc).__name__}"}
    finally:
        await _release_encoder(reason="reasoning_ingest_finished")


def register_reasoning_jobs(conductor: Any) -> bool:
    """Register the two reasoning background loops on an autonomy conductor.

    Returns True if registration happened. Safe to call multiple times.
    """
    if conductor is None:
        return False
    if id(conductor) in _registered:
        return False
    register = getattr(conductor, "register", None)
    if not callable(register):
        return False
    try:
        register(
            "reasoning_idle_precompute",
            _PRECOMPUTE_INTERVAL_S,
            _job_idle_precompute,
            run_immediately=False,
            policy="maintenance",
        )
        register(
            "reasoning_self_improve",
            _SELF_IMPROVE_INTERVAL_S,
            _job_self_improve,
            run_immediately=False,
            policy="research",
        )
        register(
            "reasoning_nonparametric_ingest",
            _NONPARAM_INGEST_INTERVAL_S,
            _job_nonparametric_ingest,
            run_immediately=False,
            policy="research",
        )
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("reasoning_background_register", exc)
        return False
    _registered.add(id(conductor))
    logger.info(
        "🧠 Reasoning background loops registered (idle pre-compute + self-improve + non-parametric ingest)."
    )
    return True
