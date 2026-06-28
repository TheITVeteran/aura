"""Contrastive decoding + plausibility-constrained reasoning steering.

Two Tier-1 "free / forward-pass" reasoning levers that live where the MLX worker
already has a hook — the ``logits_processors`` list applied each decode step:

1. **Contrastive decoding** (O'Brien & Lewis 2023): subtract a weak "amateur"
   distribution from the strong model's, ``L = L_smart + α·(L_smart − L_amateur)``,
   *but only among tokens the strong model already finds plausible* (the adaptive
   plausibility constraint: keep tokens with ``p_smart ≥ β·max p_smart``, mask the
   rest). This suppresses lazy/generic continuations the amateur also likes without
   ever promoting an implausible token. The amateur is a small model or the same
   model under a deliberately dull prompt.

2. **Plausibility-constrained steering**: a bounded per-token logit bias (suppress
   degenerate filler / hedging-collapse tokens, nudge toward informative ones)
   applied *only* to plausible tokens, so it re-ranks within the safe set and can
   never force the model off a cliff. This is the cheap, near-zero-cost reasoning
   nudge — the SEAL/thinking-speed family expressed in vocab space.

The decision math is written backend-agnostically and unit-tested on numpy; the MLX
processor mirrors it with ``mx`` ops. Both are opt-in (off by default) and gated so
they only run when explicitly enabled — correctness here is load-bearing, so it
fails open to the unmodified logits on any error.
"""
from __future__ import annotations

import logging
import math
import os
import threading
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

logger = logging.getLogger("Aura.ContrastiveDecoding")


def _log_softmax_np(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits)
    shifted = logits - m
    return shifted - np.log(np.sum(np.exp(shifted)))


def plausible_mask_np(logits: np.ndarray, beta: float) -> np.ndarray:
    """Adaptive plausibility set: tokens with prob ≥ beta * max prob.

    Computed in log space: ``log p ≥ log beta + max log p``. Returns a boolean mask.
    """
    logp = _log_softmax_np(logits)
    threshold = math.log(max(beta, 1e-9)) + float(np.max(logp))
    return logp >= threshold


def contrastive_combine_np(
    smart: np.ndarray,
    amateur: np.ndarray,
    *,
    alpha: float = 0.5,
    beta: float = 0.1,
) -> np.ndarray:
    """Return contrastive-decoding logits with the adaptive plausibility constraint.

    Implausible tokens (outside the strong model's beta-top set) are forced to
    ``-inf`` so they can never be sampled; among plausible tokens the amateur's
    preference is subtracted. ``alpha`` is the contrast strength.
    """
    smart = np.asarray(smart, dtype=np.float64).reshape(-1)
    amateur = np.asarray(amateur, dtype=np.float64).reshape(-1)
    if smart.shape != amateur.shape:
        # Shape mismatch ⇒ cannot contrast; fail open to the strong logits.
        return smart
    mask = plausible_mask_np(smart, beta)
    smart_lp = _log_softmax_np(smart)
    amateur_lp = _log_softmax_np(amateur)
    combined = (1.0 + alpha) * smart_lp - alpha * amateur_lp
    out = np.where(mask, combined, -np.inf)
    if not np.any(np.isfinite(out)):  # safety: never return all -inf
        return smart
    return out


def steering_combine_np(
    logits: np.ndarray,
    bias: dict[int, float],
    *,
    beta: float = 0.1,
    scale: float = 1.0,
) -> np.ndarray:
    """Apply a bounded logit bias only to plausible tokens (re-rank within the safe set)."""
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if not bias:
        return logits
    mask = plausible_mask_np(logits, beta)
    out = logits.copy()
    for token_id, delta in bias.items():
        if 0 <= token_id < out.shape[0] and mask[token_id]:
            out[token_id] += scale * float(delta)
    return out


class ContrastiveLogitsProcessor:
    """MLX ``logits_processors`` entry: contrast against an amateur logits source.

    ``amateur_logits_fn(tokens) -> mx.array`` supplies the weak distribution for the
    current prefix (a small model, or the same model under a dull prompt). When the
    amateur source is unavailable it is a transparent no-op.
    """

    def __init__(
        self,
        amateur_logits_fn: Callable[[Sequence[int]], Any] | None,
        *,
        alpha: float = 0.5,
        beta: float = 0.1,
    ) -> None:
        self._amateur_fn = amateur_logits_fn
        self.alpha = float(alpha)
        self.beta = float(beta)

    def __call__(self, tokens: Any, logits: Any) -> Any:
        if self._amateur_fn is None:
            return logits
        try:
            import mlx.core as mx

            amateur = self._amateur_fn(tokens)
            if amateur is None:
                return logits
            smart = logits
            mask = self._plausible_mask_mx(mx, smart, self.beta)
            smart_lp = smart - mx.logsumexp(smart, axis=-1, keepdims=True)
            amateur_lp = amateur - mx.logsumexp(amateur, axis=-1, keepdims=True)
            combined = (1.0 + self.alpha) * smart_lp - self.alpha * amateur_lp
            neg_inf = mx.full(smart.shape, -float("inf"))
            return mx.where(mask, combined, neg_inf)
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            logger.debug("contrastive processor fell open: %s", exc)
            return logits

    @staticmethod
    def _plausible_mask_mx(mx: Any, logits: Any, beta: float) -> Any:
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        threshold = math.log(max(beta, 1e-9)) + mx.max(logp, axis=-1, keepdims=True)
        return logp >= threshold


class ReasoningSteeringProcessor:
    """MLX ``logits_processors`` entry: plausibility-gated reasoning bias.

    ``bias`` maps token-id → additive logit delta (negative suppresses, positive
    promotes), applied only to tokens in the model's plausible set this step.
    Standalone — needs no second model.
    """

    def __init__(self, bias: dict[int, float], *, beta: float = 0.1, scale: float = 1.0) -> None:
        self._bias = {int(k): float(v) for k, v in (bias or {}).items()}
        self.beta = float(beta)
        self.scale = float(scale)
        self._bias_vec: Any | None = None

    def __call__(self, tokens: Any, logits: Any) -> Any:
        if not self._bias:
            return logits
        try:
            import mlx.core as mx

            if self._bias_vec is None or self._bias_vec.shape[-1] != logits.shape[-1]:
                vec = np.zeros((logits.shape[-1],), dtype=np.float32)
                for tid, delta in self._bias.items():
                    if 0 <= tid < vec.shape[0]:
                        vec[tid] = self.scale * delta
                self._bias_vec = mx.array(vec)
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            threshold = math.log(max(self.beta, 1e-9)) + mx.max(logp, axis=-1, keepdims=True)
            mask = logp >= threshold
            return mx.where(mask, logits + self._bias_vec, logits)
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            logger.debug("steering processor fell open: %s", exc)
            return logits


# Degenerate "mode-collapse" filler the worker already fights with repetition
# penalties — the reasoning steering layer suppresses these tokens directly when
# they are merely plausible, freeing probability mass for informative continuations.
_FILLER_WORDS = (
    " something", " shifting", " moving", " somehow", " just", " really",
    " basically", " actually", " maybe", " kind", " sort",
)


def build_reasoning_bias(tokenizer: Any, *, suppress: float = -2.0) -> dict[int, float]:
    """Build a small token-bias map that suppresses low-information filler.

    Best-effort: encodes a handful of filler tokens with the tokenizer and assigns
    a mild negative bias. Returns ``{}`` if the tokenizer cannot encode them.
    """
    bias: dict[int, float] = {}
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return bias
    for word in _FILLER_WORDS:
        try:
            ids = encode(word, add_special_tokens=False)
        except (TypeError, ValueError, RuntimeError):
            continue
        if isinstance(ids, (list, tuple)) and len(ids) == 1:
            bias[int(ids[0])] = suppress
    return bias


class MLXAmateurModel:
    """A small MLX model that supplies the amateur distribution for contrastive decoding.

    The amateur MUST share the strong model's tokenizer/vocab for the logit
    subtraction to be meaningful — use a same-family small model (e.g. Qwen2.5-1.5B
    as the amateur for a Qwen2.5-32B cortex). The callable keeps a bounded MLX
    prompt/KV cache when the installed ``mlx_lm`` build supports it, and falls
    open to full-prefix forward passes when it does not.
    """

    def __init__(self, model_path: str) -> None:
        from mlx_lm import load

        self._model, self._tokenizer = load(model_path)
        self.model_path = model_path
        self._lock = threading.RLock()
        self._cached_tokens: list[int] = []
        self._prompt_cache: Any | None = None
        self._last_logits: Any | None = None
        self._cache_disabled = False
        self._cache_announced = False
        self._max_cache_tokens = max(
            0,
            int(os.getenv("AURA_CONTRASTIVE_AMATEUR_CACHE_TOKENS", "4096") or "0"),
        )
        self._make_prompt_cache: Callable[..., Any] | None = None
        self._trim_prompt_cache: Callable[..., Any] | None = None
        if self._max_cache_tokens > 0:
            try:
                from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

                self._make_prompt_cache = make_prompt_cache
                self._trim_prompt_cache = trim_prompt_cache
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                logger.debug("amateur KV cache unavailable for %s: %s", model_path, exc)
                self._cache_disabled = True

    @staticmethod
    def _last_position_logits(output: Any) -> Any:
        logits = output[0] if isinstance(output, (list, tuple)) else output
        return logits[:, -1, :]

    def _new_prompt_cache(self) -> Any | None:
        if self._cache_disabled or self._make_prompt_cache is None:
            return None
        try:
            cache = self._make_prompt_cache(self._model, max_kv_size=self._max_cache_tokens)
            if not getattr(self, "_cache_announced", False):
                logger.info(
                    "amateur KV cache active for %s (max_tokens=%d)",
                    self.model_path,
                    self._max_cache_tokens,
                )
                self._cache_announced = True
            return cache
        except TypeError:
            try:
                cache = self._make_prompt_cache(self._model)
                if not getattr(self, "_cache_announced", False):
                    logger.info(
                        "amateur KV cache active for %s (unbounded backend cache)",
                        self.model_path,
                    )
                    self._cache_announced = True
                return cache
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                logger.debug("amateur KV cache creation failed for %s: %s", self.model_path, exc)
                self._cache_disabled = True
                return None
        except (RuntimeError, AttributeError, ValueError) as exc:
            logger.debug("amateur KV cache creation failed for %s: %s", self.model_path, exc)
            self._cache_disabled = True
            return None

    def _reset_cache(self) -> None:
        self._cached_tokens = []
        self._last_logits = None
        self._prompt_cache = self._new_prompt_cache()

    def _full_forward(self, mx: Any, ids: list[int]) -> Any:
        out = self._model(mx.array([ids]))
        return self._last_position_logits(out)

    def _cached_forward(self, mx: Any, ids: list[int]) -> Any:
        if (
            self._cache_disabled
            or self._max_cache_tokens <= 0
            or self._make_prompt_cache is None
        ):
            return self._full_forward(mx, ids)

        if len(ids) > self._max_cache_tokens:
            ids = ids[-self._max_cache_tokens:]
            self._reset_cache()

        if self._prompt_cache is None:
            self._reset_cache()

        if self._cached_tokens == ids and self._last_logits is not None:
            return self._last_logits

        if self._cached_tokens and ids[: len(self._cached_tokens)] == self._cached_tokens:
            suffix = ids[len(self._cached_tokens):]
        else:
            self._reset_cache()
            suffix = ids

        if not suffix:
            return self._full_forward(mx, ids)

        try:
            out = self._model(mx.array([suffix]), cache=self._prompt_cache)
            logits = self._last_position_logits(out)
        except TypeError:
            # Older/variant MLX models may not accept ``cache=``. Disable once
            # and keep contrastive decoding correct via full-prefix forwards.
            self._cache_disabled = True
            self._reset_cache()
            return self._full_forward(mx, ids)

        self._cached_tokens = list(ids)
        self._last_logits = logits
        if (
            self._trim_prompt_cache is not None
            and self._max_cache_tokens > 0
            and len(self._cached_tokens) > self._max_cache_tokens
        ):
            overflow = len(self._cached_tokens) - self._max_cache_tokens
            try:
                self._prompt_cache = self._trim_prompt_cache(self._prompt_cache, overflow)
                self._cached_tokens = self._cached_tokens[-self._max_cache_tokens:]
            except (RuntimeError, AttributeError, TypeError, ValueError):
                self._reset_cache()
        return logits

    def logits_fn(self) -> Callable[[Sequence[int]], Any]:
        import mlx.core as mx

        def amateur(tokens: Any) -> Any:
            try:
                ids = [int(t) for t in (tokens.tolist() if hasattr(tokens, "tolist") else tokens)]
                if not ids:
                    return None
                with self._lock:
                    return self._cached_forward(mx, ids)
            except (RuntimeError, ValueError, TypeError, AttributeError):
                return None

        return amateur


_amateur_cache: dict[str, MLXAmateurModel] = {}


def get_amateur_logits_fn(model_path: str) -> Callable[[Sequence[int]], Any] | None:
    """Load (and cache) a small MLX amateur model and return its per-step logits fn."""
    try:
        if model_path not in _amateur_cache:
            _amateur_cache[model_path] = MLXAmateurModel(model_path)
        return _amateur_cache[model_path].logits_fn()
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        logger.warning("could not load amateur model %s: %s", model_path, exc)
        return None


def build_reasoning_logits_processors(
    tokenizer: Any,
    *,
    enable_steering: bool = False,
    amateur_logits_fn: Callable[[Sequence[int]], Any] | None = None,
    amateur_model_path: str | None = None,
    alpha: float = 0.5,
    beta: float = 0.1,
    steering_scale: float = 1.0,
) -> list[Callable[[Any, Any], Any]]:
    """Factory the MLX worker uses to assemble reasoning logits processors.

    Returns an (possibly empty) list ready to extend the worker's
    ``logits_processors``. Contrastive decoding is included only when an amateur
    source is supplied; steering only when ``enable_steering``.
    """
    procs: list[Callable[[Any, Any], Any]] = []
    if amateur_logits_fn is None and amateur_model_path:
        amateur_logits_fn = get_amateur_logits_fn(amateur_model_path)
    if amateur_logits_fn is not None:
        procs.append(ContrastiveLogitsProcessor(amateur_logits_fn, alpha=alpha, beta=beta))
        logger.info("contrastive decoding active (alpha=%.2f beta=%.2f)", alpha, beta)
    if enable_steering:
        bias = build_reasoning_bias(tokenizer)
        if bias:
            procs.append(ReasoningSteeringProcessor(bias, beta=beta, scale=steering_scale))
    return procs
