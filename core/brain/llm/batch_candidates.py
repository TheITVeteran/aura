"""Batched candidate generation for verifier-selection reasoning.

Resolves the primary (heaviest live) MLX client and decodes N sampled
candidates in one batched worker pass. Every failure degrades to None/[] so
callers keep their serial-sampling fallback — this module only ever makes
best-of-N cheaper, never a new failure mode.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("Aura.BatchCandidates")


def _batching_enabled() -> bool:
    return str(os.environ.get("AURA_BATCHED_CANDIDATES", "1")).strip().lower() in {
        "1", "true", "yes", "on",
    }


def _resolve_primary_client() -> Any | None:
    try:
        from core.brain.llm.mlx_client import _CLIENTS
    except ImportError:
        return None
    best = None
    for client in _CLIENTS.values():
        if not getattr(client, "is_alive", lambda: False)():
            continue
        path = str(getattr(client, "model_path", "") or "").lower()
        if any(k in path for k in ("32b", "72b", "zenith", "cortex", "solver")):
            return client
        best = best or client
    return best


async def generate_candidates_batched(
    prompt: str,
    n: int,
    *,
    max_tokens: int = 512,
    temperature: float = 0.8,
    timeout_s: float = 180.0,
) -> list[str] | None:
    """Return N raw candidates from one batched pass, or None to signal
    'use the serial path' (disabled, no live client, or failure)."""
    if not _batching_enabled() or n < 2:
        return None
    client = _resolve_primary_client()
    if client is None or not hasattr(client, "generate_batch_async"):
        return None
    try:
        texts = await client.generate_batch_async(
            prompt,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )
    except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
        logger.debug("Batched candidate generation unavailable: %s", exc)
        return None
    return texts or None
