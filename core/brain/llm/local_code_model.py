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
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Same on-device weights the cortex uses — proven to produce clean multi-line
# code when generated WITHOUT steering. Overridable for a dedicated coder model.
_DEFAULT_MODEL_REL = "training/fused-model/Aura-32B-20260510-151144"
_MAX_CODE_TOKENS = 2048

_load_lock = threading.Lock()
_model: Any = None
_tokenizer: Any = None
_loaded_path: str | None = None


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
        _ensure_loaded(self.model_path)
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
        return mlx_generate(
            _model,
            _tokenizer,
            prompt=full_prompt,
            max_tokens=capped,
            sampler=sampler,
            verbose=False,
        )

    async def think(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> str:
        max_tokens = kwargs.get("max_tokens", _MAX_CODE_TOKENS)
        temperature = kwargs.get("temperature", 0.0)
        return await asyncio.to_thread(
            self._generate_sync, prompt, system_prompt, max_tokens, temperature
        )

    async def generate(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> str:
        return await self.think(prompt, system_prompt=system_prompt, **kwargs)


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
