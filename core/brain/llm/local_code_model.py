"""Un-steered local code generation path.

Aura's MLX persona cortex enforces an "always-steered" invariant — the worker
deliberately crashes rather than run unsteered inference (see mlx_worker.py
"crashed worker to prevent unsteered inference") — and persona/substrate
steering corrupts symbolic code tokens (collapsed newlines, garbage that the
hallucination sanitizer then rejects). Code generation therefore must NOT use
that cortex.

This module provides a separate, un-steered local generation surface: raw
``mlx_lm`` generation against the same on-device fine-tuned weights, with no
affective-steering hooks installed. Conversation stays steered (the real mind);
coding / RSI / self-modification get clean, structurally-valid code — all local,
no cloud. It exposes ``think``/``generate`` so it drops into ``LLMCodeGenerator``
as a router, bypassing the steered worker entirely.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import os
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Same on-device weights the cortex uses — proven to produce clean multi-line
# code when generated WITHOUT steering. Overridable for a dedicated coder model.
_DEFAULT_MODEL_REL = "training/fused-model/Aura-32B-20260510-151144"
_MAX_CODE_TOKENS = 2048

_load_lock = threading.RLock()
_lifecycle_gate = threading.Lock()
_generation_state_lock = threading.Lock()
_model: Any = None
_tokenizer: Any = None
_loaded_path: str | None = None
_lane_lease: Any = None
_active_generations = 0
_eviction_in_progress = False


def _resolve_model_path() -> str:
    for env in ("AURA_CODE_MODEL_PATH", "AURA_MODEL_PATH"):
        val = os.environ.get(env)
        if val and Path(val).expanduser().exists():
            return str(Path(val).expanduser())
    root = Path(__file__).resolve().parents[3]
    return str(root / _DEFAULT_MODEL_REL)


def _ensure_loaded(model_path: str) -> None:
    global _model, _tokenizer, _loaded_path
    if _model is not None and _loaded_path == model_path:
        return
    with _load_lock:
        if _model is not None and _loaded_path == model_path:
            return
        from mlx_lm import load

        logger.info(
            "🧰 [CODE] Loading un-steered local code model: %s",
            os.path.basename(model_path),
        )
        _model, _tokenizer = load(model_path)
        _loaded_path = model_path
        logger.info("🧰 [CODE] Un-steered local code model ready (no steering hooks).")


def _clear_loaded_model() -> None:
    global _model, _tokenizer, _loaded_path
    with _load_lock:
        _model = None
        _tokenizer = None
        _loaded_path = None
        gc.collect()


@contextlib.asynccontextmanager
async def _lifecycle_context() -> AsyncIterator[None]:
    # A threading lock is required across event loops; nonblocking polling is
    # cancellation-safe and cannot strand a background acquire thread.
    while not _lifecycle_gate.acquire(blocking=False):  # noqa: ASYNC110
        await asyncio.sleep(0.01)
    try:
        yield
    finally:
        _lifecycle_gate.release()


async def unload_local_code_model(*, reason: str = "local_code_model_unloaded") -> bool:
    global _lane_lease
    async with _lifecycle_context():
        lease, _lane_lease = _lane_lease, None
        await asyncio.to_thread(_clear_loaded_model)
        if lease is None:
            return False
        await lease.release(reason=reason)
        return True


async def _evict_local_code_model(_owner: Any, reason: str) -> bool:
    global _eviction_in_progress
    with _generation_state_lock:
        if _active_generations > 0 or _eviction_in_progress:
            return False
        _eviction_in_progress = True
    try:
        return await unload_local_code_model(reason=f"lane_eviction:{reason}")
    finally:
        with _generation_state_lock:
            _eviction_in_progress = False


async def _compensate_local_code_model(owner: Any, _reason: str) -> bool:
    path = str(getattr(owner, "model_path", "") or _resolve_model_path())
    await _ensure_loaded_with_lane(path)
    return bool(_model is not None and _loaded_path == path and _lane_lease is not None)


async def _ensure_loaded_with_lane(model_path: str) -> None:
    global _lane_lease
    async with _lifecycle_context():
        if _model is not None and _loaded_path == model_path and _lane_lease is not None:
            return
        prior_lease, _lane_lease = _lane_lease, None
        if _model is not None:
            await asyncio.to_thread(_clear_loaded_model)
        if prior_lease is not None:
            await prior_lease.release(reason="local_code_model_path_replaced")

        from core.runtime.model_lane_control import (
            acquire_in_process_model_lane,
            run_owned_model_thread_call,
        )

        lease = await acquire_in_process_model_lane(
            owner_id="local-code-model",
            model_path=model_path,
            purpose="serve",
            priority=50,
            preemptible=False,
            evict=_evict_local_code_model,
            compensate=_compensate_local_code_model,
            metadata={
                "provider": "local_code_model",
                "unsteered": True,
                "activation_state": "loading",
            },
        )
        try:
            await run_owned_model_thread_call(
                lambda: _ensure_loaded(model_path),
                operation_name="local-code-model-load",
            )
        except asyncio.CancelledError:
            await lease.release(reason="local_code_model_load_cancelled")
            raise
        except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError):
            await lease.release(reason="local_code_model_load_failed")
            raise
        if not await lease.set_preemptible(True):
            await asyncio.to_thread(_clear_loaded_model)
            await lease.release(reason="local_code_model_activation_fence_lost")
            raise RuntimeError("local_code_model_activation_fence_lost")
        _lane_lease = lease


class LocalCodeModel:
    """Raw, un-steered local generation surface for code synthesis.

    Drop-in ``router`` for ``LLMCodeGenerator``: it calls ``think(prompt,
    system_prompt=..., max_tokens=..., temperature=...)`` and extracts the code
    from the returned text. No steering hooks, no cortex worker, no cloud.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or _resolve_model_path()

    def is_available(self) -> bool:
        try:
            return Path(self.model_path).exists()
        except OSError:
            return False

    def _generate_sync(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        with _load_lock:
            if _model is None or _tokenizer is None or _loaded_path != self.model_path:
                raise RuntimeError("local_code_model_not_loaded")
            from mlx_lm import generate as mlx_generate
            from mlx_lm.sample_utils import make_sampler

            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": str(system_prompt)})
            messages.append({"role": "user", "content": str(prompt or "")})
            full_prompt = _tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            capped = max(64, min(int(max_tokens or _MAX_CODE_TOKENS), _MAX_CODE_TOKENS))
            sampler = make_sampler(temp=max(0.0, float(temperature)))
            return str(mlx_generate(
                _model,
                _tokenizer,
                prompt=full_prompt,
                max_tokens=capped,
                sampler=sampler,
                verbose=False,
            ))

    async def think(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> str:
        global _active_generations
        max_tokens = kwargs.get("max_tokens", _MAX_CODE_TOKENS)
        temperature = kwargs.get("temperature", 0.0)
        with _generation_state_lock:
            if _eviction_in_progress:
                raise RuntimeError("local_code_model_eviction_in_progress")
            _active_generations += 1
        try:
            await _ensure_loaded_with_lane(self.model_path)
            from core.runtime.model_lane_control import run_owned_model_thread_call

            return await run_owned_model_thread_call(
                lambda: self._generate_sync(
                    prompt,
                    system_prompt,
                    max_tokens,
                    temperature,
                ),
                operation_name="local-code-model-generate",
            )
        finally:
            with _generation_state_lock:
                _active_generations = max(0, _active_generations - 1)

    async def generate(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> str:
        return await self.think(prompt, system_prompt=system_prompt, **kwargs)

    async def close(self) -> None:
        await unload_local_code_model(reason="local_code_model_closed")


_singleton: LocalCodeModel | None = None
_singleton_lock = threading.Lock()


def get_local_code_model() -> LocalCodeModel | None:
    """Return the shared un-steered local code model, or None if weights absent."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            candidate = LocalCodeModel()
            if not candidate.is_available():
                logger.warning(
                    "LocalCodeModel weights not found at %s; code-gen will fall back.",
                    candidate.model_path,
                )
                return None
            _singleton = candidate
        return _singleton
