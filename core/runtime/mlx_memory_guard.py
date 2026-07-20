"""Host-safe MLX memory envelope for ANY tool that loads a model.

An unguarded evaluation tool drove this host to 103 GB and forced a
shutdown. Training already ran inside a resource envelope
(``tools/run_recurrence_training_envelope.py``); evaluation, probes, and
one-off scripts did not, so a single unbounded decode loop could exhaust
the machine. Memory safety cannot be a property of one lane — it has to be
a property of loading a model at all.

This module makes the envelope a two-line call any script can make, with
limits derived from the ACTUAL host rather than hardcoded, and with a
periodic reclaim hook for long generation loops.

    from core.runtime.mlx_memory_guard import mlx_memory_envelope

    with mlx_memory_envelope(fraction=0.5) as envelope:
        model, tokenizer = load(...)
        ...
        envelope.reclaim(step)   # inside any long loop

Exceeding ``set_memory_limit`` makes MLX raise instead of swapping the
machine to death: a failed run is recoverable, a wedged host is not.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

logger = logging.getLogger("Aura.MLXMemoryGuard")

MLX_MEMORY_GUARD_SCHEMA = "aura.mlx_memory_guard.v1"

# Never hand MLX more than this share of physical RAM by default. The host
# still needs room for the OS, the window server, and whatever else the
# operator is running; jetsam kills the largest process, which would be us.
DEFAULT_FRACTION = 0.5
MIN_LIMIT_BYTES = 2 * 1024**3
# Reclaim cadence for generation loops. Cheap relative to a forward pass.
DEFAULT_RECLAIM_EVERY = 16


def host_memory_bytes() -> int:
    """Physical RAM, or a conservative floor when it cannot be read."""
    try:
        size = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        if size > 0:
            return int(size)
    except (AttributeError, ValueError, OSError):
        pass
    return 8 * 1024**3


@dataclass
class MemoryEnvelope:
    """Applied limits plus the reclaim hook for long loops."""

    memory_bytes: int
    cache_bytes: int
    wired_bytes: int
    reclaim_every: int = DEFAULT_RECLAIM_EVERY

    def reclaim(self, step: int | None = None, *, force: bool = False) -> bool:
        """Release MLX's buffer cache. Call inside generation/eval loops.

        Returns True when a reclaim actually ran, so callers can receipt it.
        """
        if not force and step is not None and self.reclaim_every > 0:
            if step % self.reclaim_every != 0:
                return False
        try:
            import mlx.core as mx

            mx.clear_cache()
            return True
        except (ImportError, RuntimeError):
            return False

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema": MLX_MEMORY_GUARD_SCHEMA,
            "memory_limit_gb": round(self.memory_bytes / 1024**3, 3),
            "cache_limit_gb": round(self.cache_bytes / 1024**3, 3),
            "wired_limit_gb": round(self.wired_bytes / 1024**3, 3),
            "reclaim_every": self.reclaim_every,
            "host_memory_gb": round(host_memory_bytes() / 1024**3, 3),
        }


def _resolve_bytes(
    value: float | None,
    *,
    host: int,
    fraction: float,
    floor: int = 0,
) -> int:
    """Resolve one limit in bytes, validated against the real host.

    ``floor`` applies only to the working-memory limit: a small CACHE
    limit is a legitimate choice (it just means more frequent reclaim),
    whereas a tiny working limit cannot load a model at all.
    """
    if value is None:
        return max(floor, int(host * fraction))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("memory limits must be numeric gigabytes or None")
    resolved = int(float(value) * 1024**3)
    if resolved <= 0:
        raise ValueError("memory limits must be positive")
    if floor and resolved < floor:
        raise ValueError("memory limit below the 2 GiB floor is unusable")
    if resolved > host:
        raise ValueError(
            "memory limit exceeds physical RAM; the host would be swapped "
            "to death rather than the run failing"
        )
    return resolved


@contextmanager
def mlx_memory_envelope(
    *,
    fraction: float = DEFAULT_FRACTION,
    memory_gb: float | None = None,
    cache_gb: float | None = 2.0,
    wired_gb: float | None = None,
    reclaim_every: int = DEFAULT_RECLAIM_EVERY,
) -> Iterator[MemoryEnvelope]:
    """Bound MLX memory for the duration of the block.

    ``fraction`` sizes the default limit from real host RAM. Explicit
    gigabyte values override it and are validated against the host, so a
    typo cannot silently authorize an unbounded run.
    """
    if not 0.05 <= float(fraction) <= 0.9:
        raise ValueError("fraction must be inside [0.05, 0.9]")
    host = host_memory_bytes()
    memory_bytes = _resolve_bytes(
        memory_gb, host=host, fraction=fraction, floor=MIN_LIMIT_BYTES
    )
    cache_bytes = _resolve_bytes(
        cache_gb, host=host, fraction=min(fraction, 0.05)
    )
    wired_bytes = _resolve_bytes(
        wired_gb, host=host, fraction=min(fraction + 0.15, 0.85)
    )
    if cache_bytes > memory_bytes:
        raise ValueError("cache limit cannot exceed the memory limit")

    import mlx.core as mx

    previous = {
        "memory": mx.set_memory_limit(memory_bytes),
        "cache": mx.set_cache_limit(cache_bytes),
        "wired": mx.set_wired_limit(wired_bytes),
    }
    envelope = MemoryEnvelope(
        memory_bytes=memory_bytes,
        cache_bytes=cache_bytes,
        wired_bytes=wired_bytes,
        reclaim_every=reclaim_every,
    )
    logger.info("MLX memory envelope applied: %s", envelope.to_receipt())
    try:
        yield envelope
    finally:
        try:
            mx.clear_cache()
            mx.set_memory_limit(previous["memory"])
            mx.set_cache_limit(previous["cache"])
            mx.set_wired_limit(previous["wired"])
        except (RuntimeError, ValueError) as exc:  # pragma: no cover
            logger.warning("Could not restore MLX memory limits: %s", exc)


__all__ = [
    "DEFAULT_FRACTION",
    "MLX_MEMORY_GUARD_SCHEMA",
    "MemoryEnvelope",
    "host_memory_bytes",
    "mlx_memory_envelope",
]
