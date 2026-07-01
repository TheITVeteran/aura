"""Retired external runtime compatibility boundary.

Aura's live desktop Cortex is intentionally MLX-only.  This module remains as
an explicit non-spawning tombstone for retirement tests; it contains no spawn,
HTTP, or process-adoption path.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from core.brain.llm.model_registry import PRIMARY_ENDPOINT
from core.runtime.errors import FallbackClassification, record_degradation

logger = logging.getLogger("Aura.RetiredExternalRuntime")

_RETIREMENT_REASON = (
    "external_local_runtime_retired: live Aura uses the in-process MLX Cortex lane"
)


class ExternalLocalRuntimeRetiredError(RuntimeError):
    """Raised when retired external runtime code is invoked."""


class RetiredExternalRuntimeClient:
    """Non-spawning sentinel kept for retirement tests."""

    def __init__(self, model_path: str | None = None, **_kwargs: Any) -> None:
        self.model_path = str(model_path or "")
        self._lane_name = PRIMARY_ENDPOINT
        self._lane_state = "retired"
        self._last_error = _RETIREMENT_REASON
        now = time.time()
        self._last_ready_at = 0.0
        self._last_progress_at = now
        self._last_generation_completed_at = 0.0
        self._runtime_identity_ok = False

    def _retired(self) -> ExternalLocalRuntimeRetiredError:
        return ExternalLocalRuntimeRetiredError(_RETIREMENT_REASON)

    def _record_blocked_invocation(self, operation: str) -> None:
        error = self._retired()
        self._last_error = str(error)
        record_degradation(
            "retired_external_runtime",
            error,
            severity="critical",
            action="blocked retired external model runtime invocation",
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra={"operation": operation, "model_path": self.model_path},
        )

    def _spawn_server_blocking(self) -> None:
        self._record_blocked_invocation("spawn_server")
        raise self._retired()

    async def _ensure_runtime_ready(self, *_args: Any, **_kwargs: Any) -> bool:
        return False

    async def warmup(self, *_args: Any, **_kwargs: Any) -> bool:
        self._last_error = _RETIREMENT_REASON
        return False

    async def generate_text_async(self, *_args: Any, **_kwargs: Any) -> None:
        self._record_blocked_invocation("generate_text_async")
        raise self._retired()

    async def generate(self, *_args: Any, **_kwargs: Any) -> None:
        self._record_blocked_invocation("generate")
        raise self._retired()

    async def stream_generate(self, *_args: Any, **_kwargs: Any) -> AsyncGenerator[str, None]:
        raise self._retired()
        yield ""

    async def reboot_worker(self, *_args: Any, **_kwargs: Any) -> bool:
        self._lane_state = "retired"
        self._last_error = _RETIREMENT_REASON
        return False

    def force_abort_active_generation(self, *_args: Any, **_kwargs: Any) -> bool:
        return False

    def is_alive(self) -> bool:
        return False

    def get_lane_status(self) -> dict[str, Any]:
        return {
            "lane": self._lane_name,
            "state": "retired",
            "conversation_ready": False,
            "readiness_blockers": ["external_local_runtime_retired"],
            "last_error": self._last_error,
            "runtime_identity_ok": False,
            "last_ready_at": 0.0,
            "last_progress_at": self._last_progress_at,
            "last_generation_completed_at": self._last_generation_completed_at,
        }

    def get_supervision_status(self) -> dict[str, Any]:
        return {
            "alive": False,
            "state": "retired",
            "last_error": self._last_error,
            "managed": False,
        }

    def should_recycle_for_fragmentation(self, *_args: Any, **_kwargs: Any) -> bool:
        return False


def get_retired_external_runtime_client(*_args: Any, **_kwargs: Any) -> RetiredExternalRuntimeClient:
    logger.error("Blocked request for retired external model runtime client")
    raise ExternalLocalRuntimeRetiredError(_RETIREMENT_REASON)
