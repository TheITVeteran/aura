"""KV-cached foreground wiring for non-parametric memory — the latency-correct form.

The validation-grade processor in ``nonparametric_generation`` recomputes a full forward
over the running tokens every step just to recover the hidden-state query key. That is
O(n²) and would make the 32B foreground take minutes — which is exactly why it was kept out
of the live response path.

This module closes that gap. The model's *normal* generation forward (the one
``stream_generate`` already runs with a KV cache, O(1) per token) computes the hidden state
we need. We capture it as a side effect of that forward instead of recomputing:

  * ``HiddenStateTap`` wraps ``model.model`` (the inner transformer that returns hidden
    states) with a transparent proxy that records the last-token hidden each call. Calling
    the model during generation therefore fills ``tap.last_key`` for free.
  * ``make_tapped_nonparametric_processor`` is a standard ``(tokens, logits) -> logits``
    mlx_lm logits-processor that reads ``tap.last_key`` — no extra forward — and interpolates
    the non-parametric recall into the logits. If the tap has nothing (structure mismatch,
    first call), it returns the logits unchanged: fail-open, and crucially **never** falls
    back to the O(n²) recompute on the foreground path.

``cached_generate_with_memory`` is the standalone O(n) reference loop (real KV cache, one
forward per token) used to validate the mechanism end-to-end without the worker.

Everything is fail-open: a tap that can't install, a model whose head can't be found, or any
per-token error leaves generation exactly as it would have been without memory.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional, Tuple

import numpy as np

from core.brain.nonparametric_generation import cosine_from_l2, normalize
from core.runtime.errors import record_degradation

logger = logging.getLogger("Brain.NonParametricWorker")


def foreground_enabled() -> bool:
    """Whether the foreground non-parametric memory path is switched on (default OFF)."""
    return os.getenv("AURA_NONPARAMETRIC_FOREGROUND", "0").strip().lower() in {"1", "true", "on", "yes"}


def maybe_build_foreground(model: Any) -> Optional[Tuple["HiddenStateTap", Callable[[Any, Any], Any]]]:
    """Build (tap, processor) for the live worker iff foreground memory is on and non-empty.

    Returns None when disabled, when there is no datastore, or when the datastore is empty —
    so the live path pays nothing (no tap, no processor) unless there is genuinely something
    to recall. Fully fail-open: any error returns None and the worker generates normally.
    """
    if not foreground_enabled():
        return None
    try:
        dim = int(getattr(getattr(model, "args", None), "hidden_size", 0) or 0)
        if dim <= 0:
            return None
        from core.brain.nonparametric_memory import get_nonparametric_memory
        memory = get_nonparametric_memory(dim)
        if memory is None or len(memory) == 0:
            return None
        tap = HiddenStateTap(model)
        proc = make_tapped_nonparametric_processor(tap, memory)
        logger.info("🧠 [WORKER] Foreground non-parametric memory ACTIVE (%d entries, dim=%d).",
                    len(memory), dim)
        return tap, proc
    except Exception as exc:  # noqa: BLE001 - never block generation on memory wiring
        record_degradation("nonparametric_foreground_build", exc, severity="debug",
                           action="foreground non-parametric memory not installed; normal generation")
        return None


# ── head detection: hidden states → logits, model-agnostic ──────────────────

def _logits_from_hidden(model: Any, hidden: Any) -> Any:
    """Project hidden states to vocab logits using the model's own (possibly tied) head."""
    args = getattr(model, "args", None)
    if getattr(args, "tie_word_embeddings", False):
        inner = getattr(model, "model", None)
        embed = getattr(inner, "embed_tokens", None)
        if embed is not None and hasattr(embed, "as_linear"):
            return embed.as_linear(hidden)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None:
        return lm_head(hidden)
    # Last resort: a tied embedding without the flag set.
    inner = getattr(model, "model", None)
    embed = getattr(inner, "embed_tokens", None)
    if embed is not None and hasattr(embed, "as_linear"):
        return embed.as_linear(hidden)
    raise AttributeError("could not locate the model's output head")


# ── the tap: capture the hidden the generation forward already computes ──────

class _TappedInner:
    """Transparent proxy around model.model that records the last-token hidden per call."""

    def __init__(self, inner: Any, tap: "HiddenStateTap") -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_tap", tap)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self._inner(*args, **kwargs)
        try:
            # out is hidden states [B, T, H]; record the last position of the last call.
            self._tap.last_key = normalize(np.array(out[0, -1], dtype=np.float32))
        except (IndexError, ValueError, TypeError):
            self._tap.last_key = None
        return out

    # Delegate everything else so the proxy is indistinguishable from the real module.
    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), name)


class HiddenStateTap:
    """Installs a recording proxy around ``model.model`` for the span of a generation.

    Use as a context manager. If the swap can't be done (unexpected structure, mlx setattr
    quirk), ``active`` stays False and the tap simply records nothing — the caller's
    processor then no-ops, so generation is unaffected.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self._real_inner: Any = None
        self.active = False
        self.last_key: Optional[np.ndarray] = None

    def __enter__(self) -> "HiddenStateTap":
        try:
            inner = getattr(self._model, "model", None)
            if inner is None or not callable(inner):
                return self
            self._real_inner = inner
            self._model.model = _TappedInner(inner, self)
            self.active = True
        except Exception as exc:  # noqa: BLE001 - never let the tap break generation
            record_degradation("nonparametric_worker_tap", exc, severity="debug",
                               action="hidden-state tap disabled; foreground memory inert")
            self.active = False
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._real_inner is not None:
            try:
                self._model.model = self._real_inner
            except Exception as e:  # noqa: BLE001
                record_degradation("nonparametric_worker_tap", e, severity="warning",
                                   action="failed to restore model.model after tap")
        self._real_inner = None
        self.active = False


# ── the foreground processor: reads the tap, no recompute ───────────────────

def make_tapped_nonparametric_processor(
    tap: HiddenStateTap,
    memory: Any,
    *,
    k: int = 4,
    temperature: float = 2.0,
    phi: float | None = 0.5,
    free_energy: float | None = 0.7,
    min_cos: float = 0.55,
    base_lam: float = 0.75,
) -> Callable[[Any, Any], Any]:
    """O(1)-per-token non-parametric logits-processor driven by the hidden-state tap."""
    import mlx.core as mx

    def _proc(tokens: Any, logits: Any) -> Any:
        key = tap.last_key
        if key is None:
            return logits  # tap inactive / no hidden yet → leave logits untouched (fail-open)
        try:
            neighbors = memory.query(key, k=k)
            if not neighbors:
                return logits
            cos = cosine_from_l2(neighbors[0].distance)
            if cos < min_cos:
                return logits
            fe = 0.5 if free_energy is None else float(free_energy)
            lam = base_lam * ((cos - min_cos) / (1.0 - min_cos)) * (0.6 + 0.8 * fe)
            lg = np.array(logits, dtype=np.float32).reshape(-1)
            ktop = min(64, lg.shape[0])
            idx = np.argpartition(lg, -ktop)[-ktop:]
            sub = lg[idx] - lg[idx].max()
            ex = np.exp(sub)
            ex /= ex.sum()
            lm_probs = {int(t): float(p) for t, p in zip(idx, ex)}
            blended = memory.interpolate(
                lm_probs, key, k=k, temperature=temperature, phi=phi,
                free_energy=free_energy, lam_override=min(lam, 0.9),
            )
            out = lg.copy()
            import math as _m

            for t, p in blended.items():
                out[int(t)] = _m.log(max(p, 1e-12))
            return mx.array(out).reshape(logits.shape)
        except (RuntimeError, ValueError, TypeError, AttributeError, IndexError) as exc:
            record_degradation("nonparametric_tapped_processor", exc)
            return logits

    return _proc


# ── standalone O(n) reference loop (validation without the worker) ──────────

def cached_generate_with_memory(
    model: Any,
    tokenizer: Any,
    prompt: str,
    memory: Any,
    *,
    max_tokens: int = 40,
    k: int = 4,
    temperature: float = 2.0,
    phi: float | None = 0.5,
    free_energy: float | None = 0.7,
    use_memory: bool = True,
    min_cos: float = 0.55,
    base_lam: float = 0.75,
) -> str:
    """Greedy generation with a real KV cache: one incremental forward per token.

    The hidden key for step t is read from the *same* cached forward that produces the
    step-t logits, so adding non-parametric recall costs a datastore query, not a forward.
    This is the latency-correct shape the foreground tap mirrors. Fail-open per token.
    """
    import mlx.core as mx

    try:
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(model)
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("nonparametric_cached_generate", exc,
                           action="KV cache unavailable; aborting cached generation")
        return ""

    ids = list(tokenizer.encode(prompt))
    start = len(ids)
    eos = getattr(tokenizer, "eos_token_id", None)
    out_ids: list[int] = []

    # Prefill the cache with the full prompt, then decode one token at a time.
    cursor = mx.array([ids])
    for step in range(max(1, int(max_tokens))):
        try:
            hidden = model.model(cursor, cache=cache)           # cached forward (incremental)
            logits = _logits_from_hidden(model, hidden)
            key = normalize(np.array(hidden[0, -1], dtype=np.float32))
            lg = np.array(logits[0, -1], dtype=np.float32)
        except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
            record_degradation("nonparametric_cached_generate", exc)
            break

        next_id = int(np.argmax(lg))
        if use_memory:
            try:
                neighbors = memory.query(key, k=k)
                if neighbors:
                    cos = cosine_from_l2(neighbors[0].distance)
                    if cos >= min_cos:
                        fe = 0.5 if free_energy is None else float(free_energy)
                        lam = base_lam * max(0.0, (cos - min_cos) / (1.0 - min_cos)) * (0.6 + 0.8 * fe)
                        from core.brain.nonparametric_generation import _topk_probs
                        blended = memory.interpolate(
                            _topk_probs(lg), key, k=k, temperature=temperature, phi=phi,
                            free_energy=free_energy, lam_override=min(lam, 0.9),
                        )
                        next_id = int(max(blended, key=blended.get))
            except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
                record_degradation("nonparametric_cached_generate_interp", exc)

        out_ids.append(next_id)
        if eos is not None and next_id == eos:
            break
        cursor = mx.array([[next_id]])   # only the new token next step — O(1) forward

    return tokenizer.decode(out_ids).strip()
