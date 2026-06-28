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

import logging
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.ReasoningBackground")

import os

# Conservative intervals: idle pre-compute is cheap-ish but uses the cortex, and the
# self-improve feed can trigger governed training, so it runs rarely.
_PRECOMPUTE_INTERVAL_S = 180.0
_SELF_IMPROVE_INTERVAL_S = 3600.0
_NONPARAM_INGEST_INTERVAL_S = 1800.0

_registered: set[int] = set()
_encoder_cache: dict[str, Any] = {}


def _flag_on(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "on", "yes", "enabled"}


async def _job_idle_precompute() -> dict[str, Any]:
    from core.brain.reasoning_precompute import idle_precompute_tick

    solved = await idle_precompute_tick(max_items=1, per_item_timeout=90.0)
    return {"ok": True, "solved": solved}


async def _job_self_improve() -> dict[str, Any]:
    from core.brain.reasoning_self_improvement import get_reasoning_self_improvement

    return await get_reasoning_self_improvement().maybe_improve()


def _resolve_active_model_path() -> str | None:
    """The cortex model path (so datastore keys come from the SAME model that generates)."""
    try:
        import json
        from pathlib import Path

        active = Path.home().parent  # placeholder; prefer training/fused-model/active.json
        active = Path(__file__).resolve().parents[2] / "training" / "fused-model" / "active.json"
        if active.exists():
            data = json.loads(active.read_text(encoding="utf-8"))
            p = str(data.get("active_model_path") or "").strip()
            if p and Path(p).exists():
                return p
    except (OSError, ValueError, TypeError) as exc:
        record_degradation("reasoning_background_active_model", exc)
    return os.getenv("AURA_NONPARAMETRIC_ENCODER_MODEL") or None


def _get_encoder() -> Any | None:
    """Lazily build (and cache) an MLXEncoder over the cortex. Heavy — guarded by flags."""
    path = _resolve_active_model_path()
    if not path:
        return None
    if path in _encoder_cache:
        return _encoder_cache[path]
    try:
        from mlx_lm import load

        from core.brain.nonparametric_generation import MLXEncoder

        model, tok = load(path)
        enc = MLXEncoder(model, tok)
        _encoder_cache.clear()  # only ever keep one cortex resident
        _encoder_cache[path] = enc
        return enc
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
    encoder = _get_encoder()
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
        positions = sum(ing.ingest_sequence(c, a, encoder) for c, a in pairs)
        mem.persist()
        return {"ok": True, "ingested_positions": positions, "pairs": len(pairs)}
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("reasoning_background_ingest_job", exc)
        return {"status": "error", "error": f"{type(exc).__name__}"}


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
