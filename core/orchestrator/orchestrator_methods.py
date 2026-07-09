"""CNS-based orchestrator message processing compatibility mixin."""

from __future__ import annotations

import inspect
import logging
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger(__name__)

_CNS_RECOVERABLE_ERRORS = (
    AttributeError,
    KeyError,
    LookupError,
    OSError,
    ConnectionError,
    TimeoutError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _record_cns_degradation(error: BaseException, *, action: str) -> None:
    record_degradation("orchestrator_methods", error, severity="error", action=action)


def _emit_safe(emitter: Any, event: str, text: str, **kwargs: Any) -> None:
    emit = getattr(emitter, "emit", None)
    if not callable(emit):
        return
    try:
        emit(event, text, **kwargs)
    except _CNS_RECOVERABLE_ERRORS as exc:
        _record_cns_degradation(
            exc,
            action="continued CNS processing after diagnostic emitter failed",
        )


class OrchestratorCNSMixin:
    """Mixin for CNS-based orchestrator processing."""

    cns: Any
    emitter: Any
    cognitive_engine: Any
    _execute_task: Any

    async def process_user_input_cns(self, message: str) -> dict[str, Any]:
        """Public alias for the CNS processing path."""
        return await self._process_message_cns(message)

    async def _process_message_cns(self, message: str) -> dict[str, Any]:
        """Process a user message through CNS with explicit failure semantics."""
        message_text = str(message or "").strip()
        if not message_text:
            return {"ok": False, "status": "rejected", "error": "empty_message"}

        try:
            cns = getattr(self, "cns", None)
            process_stimulus = getattr(cns, "process_stimulus", None)
            if not callable(process_stimulus):
                return await self._process_cns_fallback(message_text, reason="cns_unavailable")

            cns_response = process_stimulus(message_text)
            if inspect.isawaitable(cns_response):
                cns_response = await cns_response
            if not isinstance(cns_response, dict):
                raise TypeError("CNS response must be a mapping")

            if cns_response.get("status") == "inhibited":
                reason = str(cns_response.get("reason", "unknown"))
                _emit_safe(
                    getattr(self, "emitter", None),
                    "thought",
                    f"Inhibited: {reason}",
                    level="info",
                )
                return {"ok": False, "status": "inhibited", "reason": reason}

            execution_plan = cns_response.get("execution")
            if not execution_plan:
                return await self._process_cns_fallback(message_text, reason="no_neural_path")

            neuron = execution_plan["neuron"]
            synapse = execution_plan["synapse"]
            neuron_id = str(getattr(neuron, "id", ""))
            skill_name = neuron_id.removeprefix("skill:").strip()
            if not skill_name:
                raise ValueError("CNS execution plan did not provide a skill id")

            _emit_safe(
                getattr(self, "emitter", None),
                "thought",
                f"Synapse Fired: {getattr(synapse, 'intent_pattern', 'unknown')} -> {getattr(neuron, 'name', skill_name)}",
                level="success",
            )

            execute_task = getattr(self, "_execute_task", None)
            if not callable(execute_task):
                raise AttributeError("orchestrator has no _execute_task callable")
            result = execute_task({"skill": skill_name, "params": {"query": message_text}})
            if inspect.isawaitable(result):
                result = await result
            return {"ok": True, "status": "executed", "skill": skill_name, "result": result}

        except _CNS_RECOVERABLE_ERRORS as exc:
            _record_cns_degradation(
                exc,
                action="returned explicit CNS processing failure without executing a partial task",
            )
            logger.error("CNS processing failed: %s", exc)
            _emit_safe(getattr(self, "emitter", None), "error", f"Neural Error: {exc}")
            return {
                "ok": False,
                "status": "failed",
                "error": "cns_processing_failed",
                "detail": str(exc)[:240],
            }

    async def _process_cns_fallback(self, message: str, *, reason: str) -> dict[str, Any]:
        _emit_safe(
            getattr(self, "emitter", None),
            "thought",
            "No neural path found. Engaging cognitive engine...",
            level="info",
        )
        engine = getattr(self, "cognitive_engine", None)
        process = getattr(engine, "process", None)
        if not callable(process):
            return {
                "ok": False,
                "status": "failed",
                "error": "cns_fallback_unavailable",
                "reason": reason,
            }
        result = process(message, self)
        if inspect.isawaitable(result):
            result = await result
        return {"ok": True, "status": "fallback", "reason": reason, "result": result}
