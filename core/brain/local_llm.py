"""Compatibility bridge for Aura's internal MLX inference lane.

Older callers still import ``LocalBrain`` for summarization, actuator synthesis,
or agent loops.  This module deliberately contains no HTTP model-server path:
all generation delegates to :class:`core.brain.unified_inference.UnifiedInferenceEngine`.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from core.config import config
from core.runtime.errors import FallbackClassification, record_degradation

logger = logging.getLogger("Aura.LocalBrain")

_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    ConnectionError,
    TimeoutError,
)

_CODING_KEYWORDS = {
    "api",
    "bug",
    "build",
    "class",
    "code",
    "compile",
    "database",
    "debug",
    "deploy",
    "docker",
    "error",
    "exception",
    "fix",
    "function",
    "git",
    "implement",
    "javascript",
    "python",
    "query",
    "refactor",
    "script",
    "sql",
    "test",
    "traceback",
    "typescript",
}


def _record_llm_degradation(
    subsystem: str,
    error: BaseException,
    *,
    action: str,
    severity: str = "degraded",
    extra: dict[str, Any] | None = None,
):
    return record_degradation(
        subsystem,
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )


def detect_task_tier(prompt: str, system_prompt: str = "") -> str:
    """Detect a lightweight generation tier for callers that still ask."""
    combined = f"{prompt} {system_prompt or ''}".lower()
    if any(keyword in combined for keyword in _CODING_KEYWORDS):
        return "coding"
    if any(keyword in combined for keyword in ("summarize", "compress", "distill")):
        return "summary"
    return "chat"


class LocalBrain:
    """MLX-only compatibility facade.

    This class intentionally refuses to fall through to raw external endpoints.
    If unified inference fails, the caller receives an empty structured failure
    and a degradation receipt instead of a generic assistant reply.
    """

    THOUGHT_STREAM_PREFIX = "__THOUGHT__:"
    THOUGHT_STREAM_SUFFIX = ":__ENDTHOUGHT__"

    def __init__(self, model_name: str | None = None):
        self.model = model_name or str(getattr(config.llm, "model", None) or "Aura-MLX")
        self.timeout = getattr(config.llm, "timeout", config.llm_request_timeout_s)
        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_open_until = 0.0

    async def close(self) -> None:
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    def check_health(self) -> bool:
        try:
            from core.brain.llm.mlx_client import get_mlx_client

            client = get_mlx_client()
            status = client.get_lane_status() if hasattr(client, "get_lane_status") else {}
            state = str(status.get("state") or "").lower()
            return bool(client) and state not in {"failed", "retired"}
        except _RECOVERABLE_ERRORS:
            return False

    async def check_health_async(self) -> bool:
        return await asyncio.to_thread(self.check_health)

    async def warmup(self) -> bool:
        try:
            from core.brain.llm.mlx_client import get_mlx_client

            client = get_mlx_client()
            if hasattr(client, "warmup"):
                result = await client.warmup()
                return bool(result)
            return bool(client)
        except _RECOVERABLE_ERRORS as exc:
            _record_llm_degradation(
                "local_brain_mlx_warmup",
                exc,
                action="left internal MLX lane cold after compatibility warmup failed",
                severity="warning",
                extra={"model": self.model},
            )
            return False

    def _check_circuit(self) -> bool:
        if self._circuit_open and time.time() < self._circuit_open_until:
            logger.warning("Circuit breaker OPEN; skipping compatibility generation")
            return False
        if self._circuit_open:
            self._circuit_open = False
        return True

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_open_until = 0.0

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= 5:
            self._circuit_open = True
            self._circuit_open_until = time.time() + 30.0

    @staticmethod
    def _extract_think_segments(text: str) -> tuple[str, str]:
        thoughts: list[str] = []
        for match in re.finditer(r"<think>(.*?)</think>", text or "", flags=re.DOTALL):
            thought_text = match.group(1).strip()
            if thought_text:
                thoughts.append(thought_text)
        cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
        cleaned = cleaned.replace("</think>", "").replace("<think>", "")
        return cleaned.strip(), "\n\n".join(thoughts)

    def _strip_think_tags(self, text: str) -> str:
        return self._extract_think_segments(text)[0]

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, str]:
        if not self._check_circuit():
            return {
                "response": "",
                "thought": "",
                "error": "internal_mlx_circuit_open",
            }

        try:
            from core.brain.unified_inference import UnifiedInferenceEngine

            result = await UnifiedInferenceEngine().generate_unified(
                prompt=prompt,
                system_prompt=system_prompt,
                options=options,
                endpoint_name=kwargs.get("endpoint_name"),
                **kwargs,
            )
            self._record_success()
            return result
        except _RECOVERABLE_ERRORS as exc:
            _record_llm_degradation(
                "local_llm_unified_fallback",
                exc,
                action="refused raw external generation fallback after unified MLX inference failed",
                severity="error",
                extra={"model": self.model, "endpoint": kwargs.get("endpoint_name")},
            )
            self._record_failure()
            return {
                "response": "",
                "thought": "",
                "error": "internal_mlx_unified_inference_failed",
            }

    async def chat(
        self,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, str]:
        if not self._check_circuit():
            return {
                "response": "",
                "thought": "",
                "error": "internal_mlx_circuit_open",
            }

        try:
            from core.brain.unified_inference import UnifiedInferenceEngine

            result = await UnifiedInferenceEngine().generate_unified(
                prompt="",
                messages=messages,
                options=options,
                endpoint_name=kwargs.get("endpoint_name"),
                **kwargs,
            )
            self._record_success()
            return result
        except _RECOVERABLE_ERRORS as exc:
            _record_llm_degradation(
                "local_llm_unified_fallback",
                exc,
                action="refused raw external chat fallback after unified MLX inference failed",
                severity="error",
                extra={"model": self.model, "endpoint": kwargs.get("endpoint_name")},
            )
            self._record_failure()
            return {
                "response": "",
                "thought": "",
                "error": "internal_mlx_unified_inference_failed",
            }

    async def generate_text_stream_async(
        self,
        prompt: str,
        system_prompt: str | None = None,
        cancel_event=None,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        if cancel_event and cancel_event.is_set():
            return
        result = await self.generate(
            prompt,
            system_prompt=system_prompt,
            options=options,
            **kwargs,
        )
        response = result.get("response", "")
        thought = result.get("thought", "")
        if response:
            yield response
        if thought:
            yield f"{self.THOUGHT_STREAM_PREFIX}{thought}{self.THOUGHT_STREAM_SUFFIX}"
        if result.get("error") and not response:
            yield f" [Error: Sovereign stream interrupted: {result['error']}]"

    async def chat_stream_async(
        self,
        messages: list[dict[str, str]],
        cancel_event=None,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        if cancel_event and cancel_event.is_set():
            return
        result = await self.chat(messages, options=options, **kwargs)
        response = result.get("response", "")
        thought = result.get("thought", "")
        if response:
            yield response
        if thought:
            yield f"{self.THOUGHT_STREAM_PREFIX}{thought}{self.THOUGHT_STREAM_SUFFIX}"
        if result.get("error") and not response:
            yield f" [Error: Sovereign chat stream interrupted: {result['error']}]"
