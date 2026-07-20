"""InferenceGate: Unified MLX-managed runtime + cloud inference gateway.

Provides a single interface for all LLM inference needs.
Strategy:
  1. Try Aura's managed MLX runtime (32B Cortex primary lane)
  2. If local runtime fails, fall back to HealthRouter (Gemini cloud endpoints)
  3. If cloud fails, return a graceful error string (NEVER None)

This module is the FAST PATH for user-facing chat. It injects Aura's full
identity/personality system prompt so responses sound like Aura, not a bare LLM.
Timeouts are kept tight (45s) for conversational responsiveness.
"""

import asyncio
import copy
import gc
import inspect
import logging
import math
import os
import re
import threading as _threading
import time
import weakref
from collections import deque
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from core.brain.live_mind_contract import append_text_mutation
from core.brain.llm.chat_format import format_chatml_messages
from core.brain.llm.model_registry import (
    BRAINSTEM_ENDPOINT,
    DEEP_ENDPOINT,
    FALLBACK_ENDPOINT,
    PRIMARY_ENDPOINT,
)
from core.conversation.response_reliability import (
    assess_model_text_integrity,
    assess_user_facing_reply,
    conversation_reliability_system_block,
    has_requested_word_count_contract,
    is_live_self_reflection_turn,
    is_self_process_question,
    requested_output_contract,
)
from core.runtime import resource_psutil as psutil
from core.runtime.desktop_boot_safety import (
    desktop_resource_guard_enabled,
    desktop_safe_boot_enabled,
)
from core.runtime.errors import record_degradation
from core.runtime.proof_policy import (
    is_proof_evaluation_purpose,
    is_strict_proof_answer_prompt,
    mlx_strict_answer_contract_enabled,
    proof_model_tier,
    proof_run_active,
)
from core.runtime.shutdown_coordinator import is_shutdown_requested
from core.runtime.structured_input import analyze_prompt_shape
from core.utils.deadlines import Deadline, get_deadline
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.InferenceGate")
_LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_AT = 0.0
_LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_REASON = ""
_EXPLICIT_DEFERRED_PREWARM_REFUSAL_LOG_INTERVAL_S = 60.0

_LONG_FORM_REQUEST_RE = re.compile(
    r"\b(?:"
    r"\d{3,5}\s*(?:-|to)?\s*(?:word|token)s?|"
    r"comprehensive|detailed|fully|in[- ]depth|long[- ]form|"
    r"step[- ]by[- ]step|thorough|every part|essay|report"
    r")\b",
    re.IGNORECASE,
)
_FOREGROUND_ACTION_VERB_RE = re.compile(
    r"\b(?:open|launch|create|write|type|search|find|summari[sz]e|export|save|"
    r"attach|insert|navigate|click|copy|paste|move|rename|download|upload|run|"
    r"test|debug|commit|push|install|read|compare)\b",
    re.IGNORECASE,
)
_FOREGROUND_ACTION_SURFACE_RE = re.compile(
    r"\b(?:desktop|screen|window|app|application|browser|chrome|tab|web|article|"
    r"document|docs?|notes?|folder|file|pdf|terminal|shell|clipboard|tool|tools)\b",
    re.IGNORECASE,
)
_FOREGROUND_ACTION_SEQUENCE_RE = re.compile(
    r"\b(?:then|next|after(?:ward| that)?|finally|before|while|all in one|"
    r"multi[- ]step|step\s*\d+)\b|[,;]",
    re.IGNORECASE,
)

_STATE_SIGNAL_REWRITES = (
    ("phenomenological", "state-grounded"),
    ("Phenomenological", "State-grounded"),
    ("phenomenology", "state telemetry"),
    ("Phenomenology", "State telemetry"),
    ("phenomenal", "functional-state"),
    ("Phenomenal", "Functional-state"),
    ("qualia", "private-state evidence"),
    ("Qualia", "Private-state evidence"),
    ("inner monologue", "state report"),
    ("Inner monologue", "State report"),
)


def _worker_process_is_running(proc: Any) -> bool:
    """True when a worker process handle exists AND is still running.

    Accepts multiprocessing.Process (is_alive), subprocess.Popen (poll), or
    None. proc is None when the worker was never spawned or is already
    reaped — nothing to kill, and poking it used to raise AttributeError
    mid-recovery (seen live as recent_inference_gate_critical
    "'NoneType' object has no attribute 'poll'").
    """
    if proc is None:
        return False
    try:
        if hasattr(proc, "is_alive"):
            return bool(proc.is_alive())
        if hasattr(proc, "poll"):
            return proc.poll() is None
    except (OSError, ValueError):
        return False
    return False


def _verified_cloud_generation_metadata(
    value: Any,
    *,
    endpoint_prefix: str = "",
) -> bool:
    """Accept only structured results from a verified non-local endpoint."""

    if not isinstance(value, dict) or value.get("ok") is not True:
        return False
    endpoint = str(value.get("endpoint") or "").strip()
    provider = str(value.get("provider") or "").strip().lower()
    model = str(value.get("model") or "").strip()
    if value.get("is_local") is not False or value.get("provider_verified") is not True:
        return False
    if not endpoint or not model or provider in {"", "cloud", "local", "none", "unknown"}:
        return False
    if "unverified" in endpoint.lower() or endpoint.lower() in {"none", "all_failed"}:
        return False
    if endpoint_prefix and not endpoint.startswith(endpoint_prefix):
        return False
    return True


def _grounded_state_signal_text(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").strip().split())
    for source, replacement in _STATE_SIGNAL_REWRITES:
        text = text.replace(source, replacement)
    return text[:limit]

_INFERENCE_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    asyncio.InvalidStateError,
    psutil.Error,
)

_ACTIVE_GENERATION_BUSY_REASONS = frozenset(
    {
        "active_generation_in_flight",
        "foreground_generation_active",
        "foreground_owner_active",
        "warmup_foreground_owner",
    }
)


def _record_inference_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation(
        "inference_gate",
        error,
        severity=severity,
        action=action,
    )


_DOWNSTREAM_REPAIRABLE_SELF_REFLECTION_REASONS = frozenset(
    {
        "missing_requested_self_process_coverage",
        "off_topic_self_reflection_reply",
        "pseudo_internal_jargon",
        "status_page_self_reflection",
    }
)
_DOWNSTREAM_REPAIRABLE_USER_FACING_REASONS = frozenset(
    {
        # Only surface/style defects belong here. Thin, evasive, or confused
        # drafts need another generation attempt because downstream repair
        # cannot safely invent the missing answer.
        "off_topic_self_reflection_reply",
        "pseudo_internal_jargon",
        "status_page_self_reflection",
        "generic_assistant_language",
        "persona_card_deflection",
        "detail_request_deflection",
        "truncated_tail",
        "vague_status_derailment",
        "unsupported_operational_status_overclaim",
        "unsupported_runtime_telemetry_inference",
        "unsupported_tool_readiness_claim",
        "missing_requested_self_process_coverage",
        "missing_requested_paragraph_count",
        "missing_requested_list_count",
        "missing_requested_word_count",
        "missing_requested_sentence_count",
        "missing_requested_followup_question",
    }
)


def _should_pass_user_facing_draft_downstream(
    text: str,
    reasons: set[str],
    *,
    user_prompt: str,
    allow_memory_state_thin_status: bool = False,
) -> bool:
    """Keep salvageable chat drafts out of the expensive retry spiral."""
    if not text or not reasons:
        return False
    repairable_reasons = set(_DOWNSTREAM_REPAIRABLE_USER_FACING_REASONS)
    if allow_memory_state_thin_status:
        repairable_reasons.update(
            {
                "too_thin_for_operational_status_turn",
                "too_thin_for_status_turn",
            }
        )
    if not reasons.issubset(repairable_reasons):
        return False
    stripped = str(text or "").strip()
    if reasons == {"missing_requested_word_count"}:
        return has_requested_word_count_contract(user_prompt) and bool(stripped)
    if len(stripped) < 48:
        return False
    words = [token for token in stripped.replace("\n", " ").split(" ") if token.strip()]
    if len(words) < 8:
        return False
    if reasons & _DOWNSTREAM_REPAIRABLE_SELF_REFLECTION_REASONS:
        return is_live_self_reflection_turn(user_prompt) or is_self_process_question(user_prompt)
    return True


_USER_FACING_ORIGINS = frozenset(
    {
        "user",
        "voice",
        "admin",
        "api",
        "desktop",
        "desktop-ui",
        "gui",
        "ws",
        "websocket",
        "direct",
        "external",
        "native-shell",
        "audit",
        "simulate",
        "embodied_motor_reflex",
        "embodied",
        "reflex",
        "test",
    }
)


class _UserFacingCortexError(Exception):
    """Sentinel: Cortex failed on a user-facing request — skip brainstem, escalate to cloud."""


@asynccontextmanager
async def _thread_lock_context(
    lock: Any,
    *,
    timeout_s: float | None = None,
    label: str = "lock",
):
    if timeout_s is None:
        acquired = await asyncio.to_thread(lock.acquire)
    else:
        acquired = await asyncio.to_thread(lock.acquire, True, max(0.0, float(timeout_s)))
    if not acquired:
        raise TimeoutError(f"{label}_timeout")
    try:
        yield
    finally:
        try:
            lock.release()
        except RuntimeError:
            logger.debug("Foreground-ready lock %s was already released.", label)


class InferenceGate:
    """Isolated inference gateway for Aura's managed local runtime + cloud fallback."""

    # Class-level defaults for observation-path cooldowns so partially
    # constructed instances (test doubles via __new__, hot-reload edges) can
    # never crash the status/recovery path with AttributeError.
    _last_status_recovery_schedule_at: float = 0.0
    _last_cortex_policy_deferred_log_at: float = 0.0
    _last_stale_reset_log_at: float = 0.0
    _last_forced_warmup_override_log_at: float = 0.0

    def __init__(self, orch=None):
        self.orch = orch
        self._created_at = time.monotonic()
        self._mlx_client = None
        self._initialized = False
        self._init_error = None
        self._cached_identity_prompt: str | None = None
        self._identity_prompt_time: float = 0.0
        self._identity_prompt_state_key: tuple[Any, ...] | None = None
        self._cloud_backoff_until: float = 0.0
        self._cortex_recovery_in_progress: bool = False
        self._last_cortex_check: float = 0.0
        self._cortex_recovery_attempts: int = 0
        self._cortex_recovery_exhausted_at: float = 0.0  # [STABILITY v53]
        # Warmup backoff: a cortex load that keeps exceeding its deadline under
        # thermal throttle / GPU contention gets force-killed and re-spawned,
        # thrashing the single GPU slot and starving the foreground fallback
        # that's serving the turn (2026-07-15 soak: 210s walls). After repeated
        # stuck-cortex kills, cool down and let the resident fallback carry
        # smoothly until thermal recovers, then take one clean reload shot.
        self._cortex_stuck_kill_times: deque[float] = deque(maxlen=16)
        self._cortex_warmup_backoff_until: float = 0.0
        self._cortex_warmup_backoff_streak: int = 0
        self._last_stale_reset_log_at: float = (
            0.0  # [HARDENING v54] Rate-limit stale state warnings
        )
        # 0.0 means "no successful generation yet" — a fresh gate must not
        # report a recent success it never produced. Consumers derive ages
        # from _constructed_wall_at when this is unset.
        self._last_successful_generation_at: float = 0.0
        self._constructed_wall_at: float = time.time()
        self._prewarm_task: asyncio.Task | None = None
        self._deferred_prewarm_task: asyncio.Task | None = None
        self._maintenance_task: asyncio.Task | None = None
        self._status_recovery_task: asyncio.Task | None = None
        self._foreground_ready_lock = _threading.Lock()
        self._last_background_memory_shed_at: float = 0.0
        self._last_spare_maintenance_at: float = 0.0
        self._last_cortex_warmup_deferral_log_at: float = 0.0
        self._last_cortex_policy_deferred_log_at: float = 0.0
        self._last_status_recovery_schedule_at: float = 0.0
        self._last_user_generation_endpoint: str | None = None
        self._last_user_generation_at: float = 0.0
        self._last_user_generation_used_fallback: bool = False
        self._last_generation_metadata: dict[str, Any] = {}
        self._last_surface_control_receipt: dict[str, Any] = {}
        self._generation_metadata_context: ContextVar[dict[str, Any] | None] = (
            ContextVar(
                f"aura_inference_gate_generation_metadata_{id(self)}",
                default=None,
            )
        )
        self._surface_control_receipt_context: ContextVar[dict[str, Any] | None] = (
            ContextVar(
                f"aura_inference_gate_surface_receipt_{id(self)}",
                default=None,
            )
        )
        type(self)._instance_ref = weakref.ref(self)
        logger.info("🛡️ InferenceGate created.")

    def _record_user_generation_endpoint(self, label: str) -> None:
        endpoint = PRIMARY_ENDPOINT if str(label).startswith(PRIMARY_ENDPOINT) else str(label)
        self._last_user_generation_endpoint = endpoint
        self._last_user_generation_at = time.time()
        self._last_user_generation_used_fallback = endpoint != PRIMARY_ENDPOINT

    def _generation_metadata_slot(self) -> ContextVar[dict[str, Any] | None]:
        slot = getattr(self, "_generation_metadata_context", None)
        if slot is None:
            slot = ContextVar(
                f"aura_inference_gate_generation_metadata_{id(self)}",
                default=None,
            )
            self._generation_metadata_context = slot
        return slot

    def _surface_control_receipt_slot(self) -> ContextVar[dict[str, Any] | None]:
        slot = getattr(self, "_surface_control_receipt_context", None)
        if slot is None:
            slot = ContextVar(
                f"aura_inference_gate_surface_receipt_{id(self)}",
                default=None,
            )
            self._surface_control_receipt_context = slot
        return slot

    def _publish_generation_metadata(
        self,
        metadata: dict[str, Any],
        receipt: dict[str, Any],
    ) -> None:
        metadata_snapshot = dict(metadata)
        receipt_snapshot = dict(receipt)
        self._generation_metadata_slot().set(metadata_snapshot)
        self._surface_control_receipt_slot().set(receipt_snapshot)
        self._last_generation_metadata = metadata_snapshot
        self._last_surface_control_receipt = receipt_snapshot

    def get_last_generation_metadata(self) -> dict[str, Any]:
        task_metadata = self._generation_metadata_slot().get()
        if task_metadata is not None:
            return dict(task_metadata)
        return {}

    def get_diagnostic_last_generation_metadata(self) -> dict[str, Any]:
        """Return process-wide last-call telemetry, never request proof."""

        return dict(getattr(self, "_last_generation_metadata", {}) or {})

    def get_last_surface_control_receipt(self) -> dict[str, Any]:
        task_receipt = self._surface_control_receipt_slot().get()
        if task_receipt is not None:
            return dict(task_receipt)
        return {}

    def get_diagnostic_last_surface_control_receipt(self) -> dict[str, Any]:
        """Return process-wide last-call telemetry, never request proof."""

        return dict(getattr(self, "_last_surface_control_receipt", {}) or {})

    def _clear_last_generation_metadata(self) -> None:
        self._publish_generation_metadata({}, {})

    def _record_client_generation_metadata(
        self,
        client: Any,
        *,
        label: str,
        success: bool,
        text: str,
        requested_max_tokens: int | None = None,
        output_contract: dict[str, Any] | None = None,
        generation_metadata: dict[str, Any] | None = None,
    ) -> None:
        provider_metadata = (
            dict(generation_metadata) if isinstance(generation_metadata, dict) else {}
        )
        resolved_label = str(provider_metadata.get("endpoint") or label)
        metadata: dict[str, Any] = {
            "ok": bool(success),
            "endpoint": (
                PRIMARY_ENDPOINT
                if resolved_label.startswith(PRIMARY_ENDPOINT)
                else resolved_label
            ),
            "text_length": len(str(text or "").strip()),
        }
        for key in (
            "provider",
            "model",
            "is_local",
            "provider_verified",
            "fallback_chain",
            "error",
        ):
            if key in provider_metadata:
                metadata[key] = provider_metadata[key]
        if requested_max_tokens is not None:
            metadata["requested_max_tokens"] = max(1, int(requested_max_tokens))
        if isinstance(output_contract, dict) and output_contract:
            metadata["requested_output_contract"] = dict(output_contract)
        raw_provider_receipt = provider_metadata.get("surface_control_receipt")
        receipt: dict[str, Any] = (
            dict(raw_provider_receipt) if isinstance(raw_provider_receipt, dict) else {}
        )
        getter = getattr(client, "get_last_surface_control_receipt", None)
        if not receipt and callable(getter):
            try:
                raw_receipt = getter()
                if isinstance(raw_receipt, dict):
                    receipt = dict(raw_receipt)
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                _record_inference_degradation(
                    exc,
                    action="continued generation after local client surface-control receipt read failed",
                    severity="warning",
                )
                logger.debug("Surface-control receipt read failed for %s: %s", label, exc)
        if receipt:
            metadata["surface_control_receipt"] = receipt
            for source_key, metadata_key in (
                ("generation_max_tokens", "actual_max_tokens"),
                ("generated_tokens", "generated_tokens"),
                ("instruction_shape_repair_applied", "deterministic_repair_applied"),
            ):
                if source_key in receipt:
                    metadata[metadata_key] = receipt[source_key]
            if (
                not success
                and bool(receipt.get("surface_quality_gate_enabled"))
                and not bool(receipt.get("surface_quality_gate_passed"))
            ):
                metadata["error"] = "surface_quality_rejected"
                raw_reasons = receipt.get("surface_quality_gate_reasons")
                if isinstance(raw_reasons, (list, tuple)):
                    metadata["failure_reasons"] = [
                        str(reason).strip()[:120]
                        for reason in raw_reasons
                        if str(reason).strip()
                    ][:8]
        self._publish_generation_metadata(metadata, receipt)

    def _credit_action_seq(self) -> int:
        """Monotonic per-gate counter so credit action ids never collide."""
        value = int(getattr(self, "_credit_action_counter", 0)) + 1
        self._credit_action_counter = value
        return value

    def _annotate_last_generation_metadata(self, **fields: Any) -> None:
        """Amend the just-published metadata after post-generation validation.

        Success metadata is recorded at the client boundary before integrity
        and user-facing assessment run. When those checks reject the draft the
        published record must be downgraded, otherwise a later metadata read
        treats the rejected generation as proof of a valid response.
        """
        metadata = self.get_last_generation_metadata()
        if not metadata:
            return
        metadata.update(fields)
        receipt = dict(metadata.get("surface_control_receipt") or {})
        self._publish_generation_metadata(metadata, receipt)

    @classmethod
    def _user_facing_recovery_response(cls, prompt: str) -> str:
        # [HARDENING v54] NEVER echo prompt content back to the user.
        # The prompt may contain system prompts, stale conversation history from
        # memory retrieval, or fragments from previous sessions. Echoing it back
        # fabricates hallucinated statements the user never made.
        try:
            from core.synthesis import deterministic_user_facing_floor

            direct = deterministic_user_facing_floor(prompt)
            if direct:
                return direct
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="returned deterministic recovery response without optional narrative variation",
            )
            logger.debug("Deterministic recovery response unavailable: %s", exc)
        
        # [HARDENING v57] OFFLINE RESILIENCE: Return GENUINE offline-safe response
        # instead of empty string. System must function perfectly offline without
        # cloud. This is a minimum viable response, never empty.
        try:
            from core.synthesis import generate_offline_fallback_response
            
            fallback = generate_offline_fallback_response(prompt)
            if fallback and len(str(fallback).strip()) > 0:
                return fallback
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="fell through to terminal recovery text after offline fallback generation failed",
            )
            logger.warning("Offline fallback response generation failed: %s", _exc)

        # Last resort: an honest terminal response. Both generation paths have
        # already failed at this point and no asynchronous continuation exists,
        # so the text must not promise that work is still in progress.
        return (
            "I couldn't finish generating a response just now — my language "
            "backend hit an internal problem. Please try again in a moment."
        )

    def _stabilize_user_facing_text(
        self,
        text: str,
        prompt: str,
        *,
        is_user_facing: bool,
    ) -> str:
        if not is_user_facing:
            return str(text or "").strip()
        original = str(text or "").strip()
        try:
            from core.synthesis import stabilize_user_facing_response

            stabilized = stabilize_user_facing_response(original, prompt)
            if stabilized != original:
                metadata = self.get_last_generation_metadata()
                if not metadata:
                    metadata = {
                        "ok": bool(stabilized),
                        "endpoint": "unattributed-response-path",
                        "text_length": len(stabilized),
                    }
                receipt = dict(metadata.get("surface_control_receipt") or {})
                append_text_mutation(
                    receipt,
                    stage="inference_gate.post_generation_stabilization",
                    method="deterministic_instruction_shape",
                    reasons=["user_output_contract"],
                    before=original,
                    after=stabilized,
                    deterministic=True,
                )
                metadata["surface_control_receipt"] = receipt
                metadata["text_mutations"] = list(receipt.get("text_mutations") or [])
                metadata["deterministic_repair_applied"] = bool(
                    receipt.get("deterministic_repair_applied")
                )
                metadata["post_generation_repair_applied"] = True
                self._publish_generation_metadata(metadata, receipt)
            return stabilized
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            _record_inference_degradation(
                exc,
                action="returned unstabilized user-facing text after output stabilization failed",
            )
            return original

    def _finalize_nonlocal_user_facing_text(
        self,
        text: str,
        prompt: str,
        *,
        is_user_facing: bool,
        label: str,
        max_tokens: int | None,
        output_contract: dict[str, Any] | None,
        generation_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Finalize cloud/recovery text without retaining stale local receipts."""

        cleaned = str(text or "").strip()
        if is_user_facing:
            self._record_client_generation_metadata(
                None,
                label=label,
                success=bool(cleaned),
                text=cleaned,
                requested_max_tokens=max_tokens,
                output_contract=output_contract,
                generation_metadata=generation_metadata,
            )
            if cleaned:
                self._record_user_generation_endpoint(label)
        return self._stabilize_user_facing_text(
            cleaned,
            prompt,
            is_user_facing=is_user_facing,
        )

    @staticmethod
    def _repairable_user_facing_draft_for_downstream(text: str, prompt: str) -> str | None:
        """Return text unchanged when downstream response repair should own it."""
        cleaned = str(text or "").strip()
        if not cleaned:
            return None
        try:
            assessment = assess_user_facing_reply(prompt, cleaned)
            if assessment.retryable and _should_pass_user_facing_draft_downstream(
                cleaned,
                set(assessment.reasons or ()),
                user_prompt=prompt,
            ):
                return cleaned
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            logger.debug("Repairable draft preservation check skipped: %s", exc)
        return None

    @staticmethod
    def _visible_user_prompt_from_messages(
        messages: list[dict[str, Any]] | None,
        fallback: Any,
    ) -> str:
        """Return the last actual user message from a structured prompt envelope."""
        if messages:
            for msg in reversed(messages):
                if not isinstance(msg, dict):
                    continue
                if str(msg.get("role", "") or "").strip().lower() == "user":
                    content = str(msg.get("content", "") or "").strip()
                    if content:
                        return content
        return str(fallback or "")

    @staticmethod
    def _log_task_exception(task: asyncio.Task) -> None:
        """Callback for fire-and-forget tasks — ensures exceptions are logged."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "🚨 [STABILITY v53] Background task '%s' crashed: %s",
                task.get_name(),
                exc,
                exc_info=exc,
            )

    @staticmethod
    def _lane_reports_active_generation(lane: dict[str, Any] | None) -> bool:
        if not isinstance(lane, dict):
            return False
        try:
            if int(lane.get("active_generations", 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            return False
        blockers = {
            str(blocker or "").strip()
            for blocker in (lane.get("readiness_blockers") or [])
            if str(blocker or "").strip()
        }
        reason = str(lane.get("last_failure_reason", "") or "")
        if blockers & _ACTIVE_GENERATION_BUSY_REASONS:
            return True
        return any(token in reason for token in _ACTIVE_GENERATION_BUSY_REASONS)

    @staticmethod
    def _exception_reports_active_generation(error: BaseException) -> bool:
        reason = str(error or "")
        return any(token in reason for token in _ACTIVE_GENERATION_BUSY_REASONS)

    @staticmethod
    def _desktop_safe_boot_enabled() -> bool:
        """Return True only for explicit reduced recovery safe boot."""

        return desktop_safe_boot_enabled()

    @staticmethod
    def _desktop_resource_guard_enabled() -> bool:
        """Return True when the normal desktop RAM/process guard is active."""

        return desktop_resource_guard_enabled()

    @staticmethod
    def _desktop_background_local_enabled() -> bool:
        return str(
            os.environ.get("AURA_ENABLE_DESKTOP_BACKGROUND_LOCAL_LLM", "")
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _env_float(
        name: str,
        default: float,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        """Read a float env knob, rejecting NaN/inf and out-of-range values.

        Non-finite deadlines, thresholds, and windows disable comparisons
        downstream (NaN fails every branch, inf never expires), so a malformed
        or hostile environment value must fall back to the engineered default
        rather than silently rewriting admission or backoff policy.
        """
        try:
            value = float(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(value):
            return float(default)
        if minimum is not None and value < minimum:
            return float(minimum)
        if maximum is not None and value > maximum:
            return float(maximum)
        return value

    @staticmethod
    def _cortex_worker_is_legitimately_loading(client: Any) -> bool:
        """True when the cortex worker is running because it is LOADING the
        model, not because it is wedged.

        The cascade-cleanup path force-kills a "stuck" cortex worker to free
        blocked IPC feeder threads. But a worker actively loading the ~20GB
        32B is running and NOT stuck — killing it there was a full doom loop
        (2026-07-15 soak: spawn → load → killed mid-warmup on the next turn →
        warmup_deferred → repeat, 216s/turn, zero real cortex answers for an
        hour). A worker is legitimately loading when warmup is in flight OR
        the lane is warming/recovering, AND it entered that state within a
        generous load deadline. Past the deadline a still-warming worker is
        genuinely stuck and may be killed.
        """
        if client is None:
            return False
        warming = bool(getattr(client, "_warmup_in_flight", False)) or str(
            getattr(client, "_lane_state", "")
        ) in {"warming", "recovering"}
        if not warming:
            return False
        transition_at = float(getattr(client, "_lane_transition_at", 0.0) or 0.0)
        warming_age = time.time() - transition_at if transition_at else 1e9
        load_deadline_s = InferenceGate._env_float("AURA_CORTEX_LOAD_DEADLINE_S", 200.0)
        return warming_age < load_deadline_s

    @staticmethod
    def _cortex_warmup_admission_snapshot(context: str = "background") -> dict[str, Any]:
        """Return whether a cold Cortex load is safe under current RAM pressure.

        The normal foreground headroom check is intentionally permissive because
        a *resident* Cortex can keep answering while RAM is high. A cold 32B
        load is different: it adds tens of GB of unified-memory pressure in one
        burst. This snapshot is therefore stricter and is used before any
        background/recovery/foreground warmup that would spawn the Cortex worker.
        
        [HARDENING v57-CORTEX] PRIORITY: 32B cortex is PRIMARY model. Must be less
        deferent to memory pressure to ensure system works regardless of cloud.
        """
        context_key = str(context or "background").strip().upper()
        try:
            vm = psutil.virtual_memory()
            total_gb = float(vm.total) / float(1024**3)
            available_gb = float(vm.available) / float(1024**3)
            pressure_pct = float(vm.percent)

            if total_gb >= 60.0:
                # Cold-loading Cortex is a host-survival decision, not a normal
                # generation decision. The 32B lane is the user-facing default,
                # but it must not be admitted while macOS is close to swap/jetsam.
                default_max_pressure = 72.0 if context_key == "FOREGROUND" else 58.0
                default_min_available = 20.0 if context_key == "FOREGROUND" else 26.0
            else:
                default_max_pressure = 68.0 if context_key == "FOREGROUND" else 54.0
                default_min_available = 14.0 if context_key == "FOREGROUND" else 18.0

            max_pressure = InferenceGate._env_float(
                f"AURA_CORTEX_{context_key}_WARMUP_MAX_PRESSURE_PCT",
                InferenceGate._env_float(
                    "AURA_CORTEX_COLD_WARMUP_MAX_PRESSURE_PCT",
                    default_max_pressure,
                ),
            )
            min_available = InferenceGate._env_float(
                f"AURA_CORTEX_{context_key}_WARMUP_MIN_AVAILABLE_GB",
                InferenceGate._env_float(
                    "AURA_CORTEX_COLD_WARMUP_MIN_AVAILABLE_GB",
                    default_min_available,
                ),
            )
            can_admit = bool(pressure_pct < max_pressure and available_gb >= min_available)
            reason = ""
            if not can_admit:
                reason = (
                    f"memory_pressure:{pressure_pct:.1f}%/{available_gb:.1f}GB "
                    f"(need <{max_pressure:.1f}% and >={min_available:.1f}GB)"
                )
            return {
                "context": str(context or "background"),
                "pressure_pct": pressure_pct,
                "available_gb": available_gb,
                "total_gb": total_gb,
                "max_pressure_pct": max_pressure,
                "min_available_gb": min_available,
                "can_admit": can_admit,
                "reason": reason,
                "measured": True,
            }
        except (AttributeError, TypeError, ValueError, OSError) as exc:
            _record_inference_degradation(
                exc,
                action="continued bounded inference fallback after non-fatal degradation",
            )
            logger.debug("Cortex warmup memory probe failed: %s", exc)
            force_warmup = str(
                os.environ.get("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", "")
            ).strip().lower() in {"1", "true", "yes", "on"}
            # measured=False marks every numeric field below as UNKNOWN, not a
            # real observation — consumers must not treat these zeros as a
            # calm-memory measurement.
            return {
                "context": str(context or "background"),
                "pressure_pct": 0.0,
                "available_gb": 0.0,
                "total_gb": 0.0,
                "max_pressure_pct": 100.0,
                "min_available_gb": 0.0,
                "can_admit": force_warmup,
                "reason": (
                    "memory_probe_failed_forced_override"
                    if force_warmup
                    else "memory_probe_failed"
                ),
                "measured": False,
            }

    def _note_cortex_stuck_kill(self) -> None:
        """Record a stuck-cortex force-kill and arm a warmup cooldown once the
        kills cluster.

        Each kill means a load attempt exceeded the deadline (thermal throttle /
        GPU contention) and got reaped. Re-spawning immediately just repeats the
        thrash, and every repeat grabs the single GPU slot for a 20GB weight
        load, starving the foreground fallback that is actually serving the turn.
        After ``AURA_CORTEX_STUCK_KILL_THRESHOLD`` kills inside a rolling window
        we cool down for an escalating interval, during which warmup is deferred
        and the resident fallback carries smoothly until thermal recovers.
        """
        now = time.monotonic()
        window = InferenceGate._env_float("AURA_CORTEX_STUCK_KILL_WINDOW_S", 300.0)
        threshold = max(1, int(InferenceGate._env_float("AURA_CORTEX_STUCK_KILL_THRESHOLD", 2.0)))
        self._cortex_stuck_kill_times.append(now)
        recent = [t for t in self._cortex_stuck_kill_times if now - t <= window]
        if len(recent) < threshold:
            return
        base = InferenceGate._env_float("AURA_CORTEX_WARMUP_BACKOFF_S", 90.0)
        cap = InferenceGate._env_float("AURA_CORTEX_WARMUP_BACKOFF_CAP_S", 240.0)
        self._cortex_warmup_backoff_streak += 1
        cooldown = min(cap, base * self._cortex_warmup_backoff_streak)
        self._cortex_warmup_backoff_until = now + cooldown
        logger.warning(
            "🧊 [CORTEX BACKOFF] %d stuck-load kills in %.0fs — deferring warmup %.0fs so the "
            "resident fallback carries and thermal recovers before the next reload shot.",
            len(recent),
            window,
            cooldown,
        )

    def _cortex_warmup_backoff_reason(self) -> str | None:
        """Non-None while a post-thrash warmup cooldown is active."""
        backoff_until = float(
            getattr(self, "_cortex_warmup_backoff_until", 0.0) or 0.0
        )
        remaining = backoff_until - time.monotonic()
        if remaining <= 0.0:
            return None
        return f"warmup_backoff:{remaining:.0f}s"

    def _reset_cortex_warmup_backoff(self) -> None:
        """Clear the cooldown after the cortex proves it can serve again."""
        kill_times = getattr(self, "_cortex_stuck_kill_times", None)
        if getattr(self, "_cortex_warmup_backoff_until", 0.0) or kill_times:
            if kill_times is not None:
                kill_times.clear()
            self._cortex_warmup_backoff_until = 0.0
            self._cortex_warmup_backoff_streak = 0

    def _cortex_warmup_deferral_reason(self, context: str = "background") -> str | None:
        if str(os.environ.get("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            # The force override may skip the soft admission thresholds and
            # warmup backoff, but never the host-survival floor: a cold 32B
            # load into single-digit free GB risks jetsam/swap-death of the
            # whole process tree, which no operator override should authorize.
            snapshot = self._cortex_warmup_admission_snapshot(context)
            hard_floor_gb = self._env_float(
                "AURA_FORCE_CORTEX_WARMUP_HARD_FLOOR_GB", 10.0, minimum=4.0
            )
            if snapshot.get("measured", True) and (
                float(snapshot.get("available_gb", 0.0) or 0.0) < hard_floor_gb
            ):
                return (
                    "forced_warmup_denied_survival_floor:"
                    f"{float(snapshot.get('available_gb', 0.0) or 0.0):.1f}GB"
                    f"<{hard_floor_gb:.1f}GB"
                )
            now = time.monotonic()
            last_log = getattr(self, "_last_forced_warmup_override_log_at", 0.0)
            if (now - last_log) > 60.0:
                self._last_forced_warmup_override_log_at = now
                logger.warning(
                    "⚠️ AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE active — bypassing "
                    "%s warmup admission (available=%.1fGB, survival floor %.1fGB).",
                    context,
                    float(snapshot.get("available_gb", 0.0) or 0.0),
                    hard_floor_gb,
                )
            return None
        backoff = self._cortex_warmup_backoff_reason()
        if backoff is not None:
            return backoff
        snapshot = self._cortex_warmup_admission_snapshot(context)
        return None if snapshot["can_admit"] else str(snapshot["reason"] or "memory_pressure")

    def _log_cortex_warmup_deferral(self, reason: str, *, context: str) -> None:
        now = time.monotonic()
        last_log = getattr(self, "_last_cortex_warmup_deferral_log_at", 0.0)
        if (now - last_log) < 30.0:
            return
        self._last_cortex_warmup_deferral_log_at = now
        logger.warning("⏸️ Cortex %s warmup deferred to protect RAM: %s", context, reason)

    def _note_foreground_warmup_failure(self, warmup_exc: BaseException) -> bool:
        """Classify a foreground-warmup failure; returns True for RAM deferrals.

        A ``foreground_warmup_deferred`` outcome is expected RAM-admission
        backpressure — the turn reroutes to the fallback tier, so it is logged
        at info and NOT recorded as a degradation: on the fail-closed
        inference_gate a degradation record raises CRITICAL SERVICE FAILURE
        out of the handler and kills the protected recovery lane (seen live
        July 8: one memory deferral cascaded into chat 503s). Same discipline
        as the timeout demotion in core/runtime/errors.py. Genuine warmup
        faults keep the full degradation record.
        """
        if "foreground_warmup_deferred" in str(warmup_exc):
            logger.info(
                "🧠 Foreground warmup deferred by RAM admission; rerouting this turn: %s",
                warmup_exc,
            )
            return True
        record_degradation(
            "inference_gate",
            warmup_exc,
            severity="degraded",
            action="skipped cold primary attempt or fell back after foreground warmup failure",
        )
        from core.runtime.errors import describe_error

        logger.warning(
            "🧠 Foreground preflight warmup did not complete cleanly: %s",
            describe_error(warmup_exc),
        )
        return False

    def _log_cold_cortex_policy_deferred(self) -> None:
        now = time.monotonic()
        last_log = getattr(self, "_last_cortex_policy_deferred_log_at", 0.0)
        if (now - last_log) < 300.0:
            return
        self._last_cortex_policy_deferred_log_at = now
        logger.info(
            "Cold-start Cortex recovery deferred by desktop prewarm policy; "
            "foreground demand will warm the lane when needed."
        )

    @staticmethod
    def _boot_should_eager_warmup() -> bool:
        """Keep the 32B lane warm on high-memory desktops unless explicitly disabled."""
        if str(os.environ.get("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return True
        if InferenceGate._desktop_resource_guard_enabled():
            logger.info("🛡️ Desktop resource guard active — skipping eager 32B warmup during launch.")
            return False
        setting = str(os.environ.get("AURA_EAGER_CORTEX_WARMUP", "auto")).strip().lower()
        if setting in {"1", "true", "yes", "on"}:
            snapshot = InferenceGate._cortex_warmup_admission_snapshot("boot")
            if not snapshot["can_admit"] and str(
                os.environ.get("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", "")
            ).strip().lower() not in {"1", "true", "yes", "on"}:
                logger.warning(
                    "⏸️ Explicit eager Cortex warmup deferred to protect RAM: %s", snapshot["reason"]
                )
                return False
            return True
        if setting in {"0", "false", "no", "off"}:
            return False

        try:
            vm = psutil.virtual_memory()
            snapshot = InferenceGate._cortex_warmup_admission_snapshot("boot")
            min_total_gb = float(os.environ.get("AURA_BOOT_WARMUP_MIN_TOTAL_GB", "48"))
            if (vm.total / float(1024**3)) < min_total_gb or not snapshot["can_admit"]:
                logger.warning(
                    "⏸️ Deferring eager 32B warmup at boot (total=%.1fGB pressure=%.1f%% available=%.1fGB).",
                    snapshot["total_gb"],
                    snapshot["pressure_pct"],
                    snapshot["available_gb"],
                )
                return False
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="kept conservative boot warmup decision after desktop policy probe failed",
            )
            logger.debug("Boot warmup memory probe failed: %s", exc)
            return False

        return True

    @staticmethod
    def _boot_should_schedule_deferred_prewarm() -> bool:
        explicit_setting = os.environ.get("AURA_DEFERRED_CORTEX_PREWARM")
        setting = str(explicit_setting if explicit_setting is not None else "auto").strip().lower()
        if setting in {"1", "true", "yes", "on"}:
            snapshot = InferenceGate._cortex_warmup_admission_snapshot("background")
            if not snapshot["can_admit"] and str(
                os.environ.get("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", "")
            ).strip().lower() not in {"1", "true", "yes", "on"}:
                global _LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_AT
                global _LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_REASON
                now = time.monotonic()
                reason = str(snapshot["reason"] or "memory_pressure")
                if (
                    reason != _LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_REASON
                    or (now - _LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_AT)
                    >= _EXPLICIT_DEFERRED_PREWARM_REFUSAL_LOG_INTERVAL_S
                ):
                    _LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_AT = now
                    _LAST_EXPLICIT_DEFERRED_PREWARM_REFUSAL_REASON = reason
                    logger.warning(
                        "⏸️ Explicit deferred Cortex prewarm refused to protect RAM: %s",
                        reason,
                    )
                else:
                    logger.debug(
                        "Explicit deferred Cortex prewarm still refused to protect RAM: %s",
                        reason,
                    )
                return False
            return True
        if setting in {"0", "false", "no", "off"}:
            return False
        if InferenceGate._desktop_safe_boot_enabled():
            if explicit_setting is None:
                logger.info(
                    "🛡️ Recovery safe boot active — skipping implicit deferred 32B prewarm during launch."
                )
                return False
            snapshot = InferenceGate._cortex_warmup_admission_snapshot("background")
            if not snapshot["can_admit"]:
                logger.warning(
                    "⏸️ Recovery safe-boot deferred Cortex prewarm deferred to protect RAM: %s",
                    snapshot["reason"],
                )
                return False
            return True
        return True

    @staticmethod
    def _headroom_snapshot(requested_tier: str = "primary") -> dict[str, Any]:
        try:
            from core.utils.memory_monitor import get_memory_pressure_snapshot

            snapshot = get_memory_pressure_snapshot()
            total_gb = float(snapshot.total_gb)
            available_gb = float(snapshot.available_gb)
            pressure_pct = float(snapshot.pressure_pct)
            process_rss_gb = float(snapshot.process_rss_gb)
            process_rss_limit_gb = float(snapshot.process_rss_limit_gb)
            tier = str(requested_tier or "primary").strip().lower()

            def _threshold(name: str, default: str) -> float:
                return InferenceGate._env_float(name, float(default))

            if tier == "secondary":
                max_pressure = _threshold(
                    "AURA_FOREGROUND_SECONDARY_MAX_PRESSURE_PCT",
                    "42" if total_gb < 96.0 else "72",
                )
                min_available_gb = _threshold(
                    "AURA_FOREGROUND_SECONDARY_MIN_AVAILABLE_GB",
                    "52" if total_gb < 96.0 else "28",
                )
            elif tier == "tertiary":
                max_pressure = _threshold(
                    "AURA_FOREGROUND_TERTIARY_MAX_PRESSURE_PCT",
                    "92" if total_gb >= 60.0 else "88",
                )
                min_available_gb = _threshold(
                    "AURA_FOREGROUND_TERTIARY_MIN_AVAILABLE_GB",
                    "6" if total_gb >= 60.0 else "4",
                )
            else:
                max_pressure = _threshold(
                    "AURA_FOREGROUND_PRIMARY_MAX_PRESSURE_PCT",
                    "76" if total_gb >= 60.0 else "82",
                )
                min_available_gb = _threshold(
                    "AURA_FOREGROUND_PRIMARY_MIN_AVAILABLE_GB",
                    "18" if total_gb >= 60.0 else "10",
                )
            system_admit = bool(pressure_pct < max_pressure and available_gb >= min_available_gb)
            process_admit = bool(
                process_rss_limit_gb <= 0.0
                or process_rss_gb < process_rss_limit_gb
            )
            can_admit = bool(system_admit and process_admit and not snapshot.refuse_heavy_local_generation)
            reason_parts: list[str] = []
            if not system_admit:
                reason_parts.append(
                    f"memory_pressure:{pressure_pct:.1f}%/{available_gb:.1f}GB "
                    f"(need <{max_pressure:.1f}% and >={min_available_gb:.1f}GB)"
                )
            if not process_admit or snapshot.refuse_heavy_local_generation:
                reason_parts.append(
                    f"process_tree_rss:{process_rss_gb:.1f}GB/{process_rss_limit_gb:.1f}GB"
                )
            reason = "; ".join(part for part in reason_parts if part)
            return {
                "tier": tier,
                "pressure_pct": pressure_pct,
                "total_gb": total_gb,
                "available_gb": available_gb,
                "process_rss_gb": process_rss_gb,
                "process_rss_limit_gb": process_rss_limit_gb,
                "max_pressure_pct": max_pressure,
                "min_available_gb": min_available_gb,
                "can_admit": can_admit,
                "reason": reason,
                "measured": True,
            }
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError, psutil.Error) as exc:
            _record_inference_degradation(
                exc,
                action="returned unmeasured foreground headroom snapshot after memory probe failed",
            )
            force_admit = str(
                os.environ.get("AURA_FORCE_FOREGROUND_HEADROOM_ON_PROBE_FAILURE", "")
            ).strip().lower() in {"1", "true", "yes", "on"}
            # measured=False: the zeros below are UNKNOWN values, not calm
            # telemetry — scheduling and health consumers must not treat this
            # snapshot as evidence of memory abundance.
            return {
                "tier": str(requested_tier or "primary"),
                "pressure_pct": 0.0,
                "total_gb": 0.0,
                "available_gb": 0.0,
                "process_rss_gb": 0.0,
                "process_rss_limit_gb": 0.0,
                "max_pressure_pct": 100.0,
                "min_available_gb": 0.0,
                "can_admit": force_admit,
                "reason": (
                    "memory_probe_failed_forced_override"
                    if force_admit
                    else "memory_probe_failed"
                ),
                "measured": False,
            }

    @staticmethod
    def _local_deep_solver_enabled(total_gb: float | None = None) -> bool:
        setting = str(os.environ.get("AURA_ENABLE_LOCAL_DEEP_SOLVER", "auto")).strip().lower()
        if setting in {"1", "true", "yes", "on"}:
            return True
        if setting in {"0", "false", "no", "off"}:
            return False
        try:
            detected_total = (
                float(total_gb)
                if total_gb is not None
                else float(psutil.virtual_memory().total) / float(1024**3)
            )
        except (AttributeError, OSError, TypeError, ValueError):
            detected_total = 0.0
        return detected_total >= float(
            os.environ.get("AURA_LOCAL_DEEP_AUTO_MIN_TOTAL_GB", "96")
        )

    def _local_deep_solver_block_reason(self) -> str | None:
        snapshot = self._headroom_snapshot("secondary")
        if not self._local_deep_solver_enabled(snapshot.get("total_gb")):
            return (
                "local_deep_solver_disabled_on_current_memory_class:"
                f"{float(snapshot.get('total_gb', 0.0) or 0.0):.1f}GB"
            )
        if not snapshot.get("can_admit", False):
            return str(snapshot.get("reason") or "secondary_memory_pressure")
        lane = self.get_conversation_status()
        if lane.get("conversation_ready") or lane.get("warmup_in_flight"):
            return "primary_cortex_resident_or_warming"
        lane_state = str(lane.get("state", "") or "").strip().lower()
        if lane_state in {"spawning", "handshaking", "warming", "recovering"}:
            return f"primary_cortex_{lane_state}"
        return None

    def _foreground_headroom_reserved(self, requested_tier: str = "primary") -> bool:
        snap = self._headroom_snapshot(requested_tier)
        safety_buffer_gb = 3.0 if snap["tier"] == "secondary" else 2.0
        return bool(
            snap["pressure_pct"] >= (snap["max_pressure_pct"] - 2.0)
            or snap["available_gb"] <= (snap["min_available_gb"] + safety_buffer_gb)
        )

    @staticmethod
    def _iter_local_clients() -> dict[str, Any]:
        clients: dict[str, Any] = {}
        try:
            from core.brain.llm.mlx_client import _CLIENTS

            clients.update(dict(_CLIENTS))
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            logger.debug("MLX client registry unavailable: %s", exc)
        return clients

    def force_abort_active_generation(self, reason: str = "hard_generation_deadline") -> int:
        """Abort any active local generation across managed inference clients.

        This is a synchronous emergency boundary used by watchdogs that cannot
        rely on the caller's event loop being healthy. Normal request handling
        still uses cooperative deadlines; this path exists to prevent a wedged
        model generation from holding the foreground lane indefinitely.
        """
        aborted = 0
        candidates: list[Any] = []
        if self._mlx_client is not None:
            candidates.append(self._mlx_client)
        candidates.extend(self._iter_local_clients().values())

        seen: set[int] = set()
        for client in candidates:
            if client is None:
                continue
            ident = id(client)
            if ident in seen:
                continue
            seen.add(ident)
            abort = getattr(client, "force_abort_active_generation", None)
            if not callable(abort):
                continue
            try:
                if abort(reason=reason):
                    aborted += 1
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                _record_inference_degradation(
                    exc,
                    action="continued force-aborting other local generation clients",
                    severity="error",
                )
                logger.warning("Force-abort failed for local inference client: %s", exc)
        return aborted

    _SHUTDOWN_TASK_ATTRS = (
        "_prewarm_task",
        "_deferred_prewarm_task",
        "_maintenance_task",
        "_status_recovery_task",
    )

    def _cancel_owned_background_tasks(self) -> list[asyncio.Task]:
        """Cancel every owned background task and return the live handles.

        Handles are returned so the async shutdown path can await actual
        termination — cancellation alone does not stop a task, and its
        finally blocks may still hold worker processes or reservations.
        """
        tasks: list[asyncio.Task] = []
        for task_attr in self._SHUTDOWN_TASK_ATTRS:
            task = getattr(self, task_attr, None)
            if task is not None and not task.done():
                task.cancel()
                tasks.append(task)
            setattr(self, task_attr, None)
        return tasks

    def _shutdown_client_candidates(self) -> list[Any]:
        candidates: list[Any] = []
        if self._mlx_client is not None:
            candidates.append(self._mlx_client)
        candidates.extend(self._iter_local_clients().values())
        seen: set[int] = set()
        unique: list[Any] = []
        for client in candidates:
            if client is None or id(client) in seen:
                continue
            seen.add(id(client))
            unique.append(client)
        return unique

    async def on_stop_async(self) -> None:
        """Async shutdown: await task termination and client closes.

        Preferred by ServiceContainer over the sync `cleanup`. Awaiting the
        cancelled tasks lets their finally blocks release worker processes and
        reservations before clients are closed and state is declared cold.
        """
        tasks = self._cancel_owned_background_tasks()
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=5.0)
            for task in pending:
                logger.warning(
                    "Inference background task %s did not terminate within the "
                    "shutdown grace period.",
                    task.get_name(),
                )
        for client in self._shutdown_client_candidates():
            close = getattr(client, "close", None)
            if not callable(close):
                continue
            try:
                result = close()
                if inspect.isawaitable(result):
                    await asyncio.wait_for(result, timeout=10.0)
            except (TimeoutError, *_INFERENCE_RECOVERABLE_ERRORS) as exc:
                _record_inference_degradation(
                    exc,
                    action=f"continued inference shutdown after {type(client).__name__}.close failed",
                    severity="warning",
                )
                logger.debug(
                    "Inference client close failed for %s: %s", type(client).__name__, exc
                )
        self._mlx_client = None
        self._initialized = False

    def cleanup(self) -> None:
        """Release managed local inference clients during ServiceContainer shutdown.

        Synchronous fallback path. When no event loop is running in this
        thread, awaitable client closes are driven to completion on a private
        loop; when a loop IS running here, blocking is impossible, so the
        close is scheduled and recorded as a degradation instead of being
        silently discarded.
        """
        cancelled_tasks = self._cancel_owned_background_tasks()

        try:
            running_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if cancelled_tasks and running_loop is None:
            logger.warning(
                "Sync inference cleanup cancelled %d background task(s) without "
                "awaiting termination; prefer on_stop_async for ordered shutdown.",
                len(cancelled_tasks),
            )

        for client in self._shutdown_client_candidates():
            close = getattr(client, "close", None)
            if not callable(close):
                continue
            try:
                result = close()
                if inspect.isawaitable(result):
                    if running_loop is not None:
                        task = running_loop.create_task(
                            asyncio.wait_for(result, timeout=10.0),
                            name=f"inference_gate_close_{type(client).__name__}",
                        )
                        task.add_done_callback(self._log_task_exception)
                        _record_inference_degradation(
                            RuntimeError(
                                f"{type(client).__name__}.close deferred to running loop "
                                "during sync shutdown"
                            ),
                            action="scheduled async client close instead of blocking sync shutdown",
                        )
                    else:
                        asyncio.run(asyncio.wait_for(result, timeout=10.0))
            except (TimeoutError, *_INFERENCE_RECOVERABLE_ERRORS) as exc:
                _record_inference_degradation(
                    exc,
                    action=f"continued inference shutdown after {type(client).__name__}.close failed",
                    severity="warning",
                )
                logger.debug("Inference client close failed for %s: %s", type(client).__name__, exc)

        self._mlx_client = None
        self._initialized = False

    on_stop = cleanup

    async def _enforce_foreground_admission(
        self,
        requested_tier: str,
        *,
        protected_foreground: bool = False,
    ) -> dict[str, Any]:
        snapshot = self._headroom_snapshot(requested_tier)
        if snapshot["can_admit"]:
            return snapshot

        logger.warning(
            "🛡️ Foreground admission tightening for %s "
            "(pressure=%.1f%% available=%.1fGB process=%.1f/%.1fGB reason=%s).",
            requested_tier,
            snapshot["pressure_pct"],
            snapshot["available_gb"],
            snapshot.get("process_rss_gb", 0.0),
            snapshot.get("process_rss_limit_gb", 0.0),
            snapshot.get("reason", ""),
        )
        await self._shed_background_workers_for_memory_pressure()
        gc.collect()
        tightened = self._headroom_snapshot(requested_tier)
        if not tightened["can_admit"] and protected_foreground and requested_tier != "secondary":
            logger.warning(
                "🛡️ Protected foreground request proceeding under reduced headroom for tier=%s "
                "(pressure=%.1f%% available=%.1fGB process=%.1f/%.1fGB).",
                requested_tier,
                tightened["pressure_pct"],
                tightened["available_gb"],
                tightened.get("process_rss_gb", 0.0),
                tightened.get("process_rss_limit_gb", 0.0),
            )
        return tightened

    async def _ensure_hot_spare_ready(self, endpoint_name: str) -> bool:
        if self._foreground_user_turn_active() or self._foreground_owner_active():
            return False

        if endpoint_name == DEEP_ENDPOINT:
            lane = self.get_conversation_status()
            lane_state = str(lane.get("state", "") or "").strip().lower()
            if lane.get("conversation_ready") or lane.get("warmup_in_flight"):
                return False
            if lane_state in {"spawning", "handshaking", "warming", "recovering"}:
                return False
            background_deferral = self._background_local_deferral_reason(
                origin="maintenance_hot_spare"
            )
            if background_deferral:
                logger.debug(
                    "⏸️ Skipping Solver hot spare warmup due to %s.",
                    background_deferral,
                )
                return False

        try:
            from core.brain.llm.mlx_client import get_mlx_client
            from core.brain.llm.model_registry import (
                get_brainstem_path,
                get_deep_model_path,
                get_fallback_path,
            )
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="continued bounded inference fallback after non-fatal degradation",
            )
            logger.debug("Hot-spare setup unavailable: %s", exc)
            return False

        if endpoint_name == BRAINSTEM_ENDPOINT:
            model_path = str(get_brainstem_path())
            requested_tier = "tertiary"
        elif endpoint_name == DEEP_ENDPOINT:
            model_path = str(get_deep_model_path())
            requested_tier = "secondary"
        elif endpoint_name == FALLBACK_ENDPOINT:
            model_path = str(get_fallback_path())
            requested_tier = "tertiary"
        else:
            return False

        snapshot = self._headroom_snapshot(requested_tier)
        if endpoint_name == DEEP_ENDPOINT and not snapshot["can_admit"]:
            return False

        client = get_mlx_client(model_path=model_path)
        if hasattr(client, "is_alive") and client.is_alive():
            return True
        if not hasattr(client, "warmup"):
            return False

        try:
            await client.warmup(foreground_request=False)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="continued bounded inference fallback after non-fatal degradation",
            )
            logger.debug("Hot-spare warmup failed for %s: %s", endpoint_name, exc)
            return False
        return bool(hasattr(client, "is_alive") and client.is_alive())

    async def _recycle_idle_local_clients(self) -> None:
        if self._foreground_user_turn_active() or self._foreground_owner_active():
            return

        max_uptime_s = float(os.environ.get("AURA_LOCAL_RECYCLE_MAX_UPTIME_S", "5400"))
        min_idle_s = float(os.environ.get("AURA_LOCAL_RECYCLE_MIN_IDLE_S", "900"))
        for client in self._iter_local_clients().values():
            if client is None or client is self._mlx_client:
                continue
            recycle_predicate = getattr(client, "should_recycle_for_fragmentation", None)
            if not callable(recycle_predicate):
                continue
            try:
                if recycle_predicate(max_uptime_s=max_uptime_s, min_idle_s=min_idle_s):
                    logger.info("♻️ Recycling idle local runtime to reduce fragmentation.")
                    if hasattr(client, "reboot_worker"):
                        await client.reboot_worker(
                            reason="scheduled_fragmentation_recycle",
                            mark_failed=False,
                        )
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                _record_inference_degradation(
                    exc,
                    action="continued recycling other idle local clients",
                )
                logger.debug("Idle runtime recycle skipped: %s", exc)

    async def _maintenance_loop(self) -> None:
        while not is_shutdown_requested():
            try:
                await asyncio.sleep(15.0 if self._last_spare_maintenance_at <= 0.0 else 45.0)
                self._last_spare_maintenance_at = time.monotonic()
                if self._background_memory_pressure_active():
                    await self._shed_background_workers_for_memory_pressure()
                    continue

                # [STABILITY v53] Proactive cortex health watchdog — detect dead
                # cortex BEFORE a user request fails. Previously cortex death was
                # only detected when a user message arrived and timed out.
                await self._proactive_cortex_watchdog()

                # [STABILITY v53] Don't eagerly load brainstem/deep at boot.
                # The 7B brainstem consumes ~5GB RAM that the 32B cortex needs.
                # At 62% RAM with both loaded, the cortex swaps and first-turn
                # response time balloons to 80+ seconds. Load on demand only.
                # await self._ensure_hot_spare_ready(BRAINSTEM_ENDPOINT)
                # await self._ensure_hot_spare_ready(DEEP_ENDPOINT)
                await self._recycle_idle_local_clients()
            except asyncio.CancelledError:
                raise
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                _record_inference_degradation(
                    exc,
                    action="continued maintenance loop after non-fatal maintenance pulse failure",
                )
                # [STABILITY v53] Upgraded from debug to warning — silent maintenance
                # failures can cascade into cortex death without visibility.
                logger.warning("⚠️ InferenceGate maintenance loop error: %s", exc, exc_info=True)

    async def _proactive_cortex_watchdog(self) -> None:
        """[STABILITY v53] Proactive cortex health check — runs every maintenance cycle.

        Detects dead/stuck cortex and triggers recovery BEFORE user requests fail.
        Also detects stale warming states and resets them.
        """
        if not self._mlx_client:
            return
        if self._foreground_user_turn_active() or self._foreground_owner_active():
            return  # Don't interfere with active user turn

        lane = self.get_conversation_status()
        lane_state = str(lane.get("state", "") or "").lower()

        # 1. Detect dead cortex and trigger recovery.
        #
        # A WARMING lane is not a dead lane. During a 32B cold load the
        # worker legitimately fails is_alive() for 120-150s while the state
        # sits in warming/spawning/handshaking — and a warmup is not flagged
        # as "recovery in progress". The 20260708-postdoomfix soak showed
        # what happens without this guard: the watchdog declared the warming
        # lane dead every 45s maintenance pulse and re-triggered recovery,
        # restarting the warmup before it could ever finish (turns pinned at
        # 216s+, SLO exhausted, runtime eventually died). The lane gets the
        # same 300s deadline section 2 already grants a stuck warmup; only
        # past it does a "warming" verdict count as dead.
        if hasattr(self._mlx_client, "is_alive") and not self._mlx_client.is_alive():
            # DEAD-MAN CLOCK, watchdog-owned. The first cut of this guard
            # trusted the client's own fields to bound the deferral
            # (warmup_in_flight OR transition age) — and the nightcap soak
            # promptly wedged with warmup_in_flight stuck True and no
            # transition timestamp: the watchdog deferred FOREVER, eleven
            # straight turns hit the probe's 240s ceiling, and nothing ever
            # recovered the lane. The watchdog now times the not-alive
            # window on its OWN clock: a warming lane gets 300s from first
            # observation, then intervention happens no matter what any
            # client flag claims.
            now = time.time()
            first_seen = float(getattr(self, "_cortex_not_alive_first_seen_at", 0.0) or 0.0)
            if first_seen <= 0.0:
                first_seen = now
                self._cortex_not_alive_first_seen_at = now
            not_alive_age_s = now - first_seen
            warmup_underway = (
                lane_state in ("warming", "spawning", "handshaking", "recovering")
                and not_alive_age_s <= 300.0
            )
            if warmup_underway:
                logger.debug(
                    "[WATCHDOG] Cortex not alive but lane is %s (%.0fs on the dead-man clock) — "
                    "letting the warmup finish.",
                    lane_state, not_alive_age_s,
                )
            elif lane_state not in ("cold", "failed") and not self._cortex_recovery_in_progress:
                logger.warning(
                    "🔍 [WATCHDOG] Cortex is dead (state=%s, not-alive %.0fs) and past the warmup "
                    "deadline. Triggering recovery.",
                    lane_state, not_alive_age_s,
                )
                # Reset the clock so the RECOVERY warmup gets its own fresh
                # 300s window instead of being instantly re-declared dead.
                self._cortex_not_alive_first_seen_at = 0.0
                # A wedged warmup flag blocks admission everywhere
                # (conversation_warmup_in_flight deferrals); recovery must
                # not start underneath it.
                if getattr(self._mlx_client, "_warmup_in_flight", False):
                    logger.warning(
                        "🔍 [WATCHDOG] Force-clearing wedged warmup_in_flight before recovery."
                    )
                    self._mlx_client._warmup_in_flight = False
                await self._ensure_cortex_recovery()
        else:
            # Lane is alive — clear the dead-man clock.
            if getattr(self, "_cortex_not_alive_first_seen_at", 0.0):
                self._cortex_not_alive_first_seen_at = 0.0

        # 2. Detect stuck warmup flag on MLX client
        if hasattr(self._mlx_client, "_warmup_in_flight") and self._mlx_client._warmup_in_flight:
            transition_at = getattr(self._mlx_client, "_lane_transition_at", 0.0)
            # [STABILITY v53] Increased from 90s to 300s. A 32B model cold-load
            # takes ~150s; 90s was guaranteed to force-kill a healthy loading worker.
            if transition_at > 0 and (time.time() - transition_at) > 300.0:
                logger.warning(
                    "🔍 [WATCHDOG] MLX warmup_in_flight stuck for >300s. Force-clearing."
                )
                self._mlx_client._warmup_in_flight = False
                if self._prewarm_task and not self._prewarm_task.done():
                    logger.warning(
                        "🔍 [WATCHDOG] Stuck prewarm task found during watchdog cleanup. Cancelling."
                    )
                    self._prewarm_task.cancel()
                    self._prewarm_task = None

        # 3. Detect completed-but-unreaped prewarm tasks
        if self._prewarm_task and self._prewarm_task.done():
            try:
                exc = self._prewarm_task.exception()
                if exc:
                    logger.warning(
                        "🔍 [WATCHDOG] Stale failed prewarm task found: %s. Clearing.", exc
                    )
            except (asyncio.CancelledError, asyncio.InvalidStateError) as exc:
                logger.debug("Prewarm task state was unavailable during watchdog cleanup: %s", exc)
            self._prewarm_task = None  # Allow fresh warmup on next request

        # 4. Log cortex health for observability
        if hasattr(self._mlx_client, "is_alive"):
            alive = self._mlx_client.is_alive()
            if not alive and lane_state == "ready":
                logger.warning(
                    "🔍 [WATCHDOG] Cortex reports ready but is_alive() is False. Correcting state."
                )
                if hasattr(self._mlx_client, "note_lane_recovering"):
                    self._mlx_client.note_lane_recovering("watchdog_state_correction")

    def get_conversation_status(self) -> dict[str, Any]:
        # [STABILITY v53] Default to "cold" not "warming" — only report warming
        # when something is actually in flight. Prevents zombie warming state.
        _default_state = "failed" if self._init_error else "cold"
        lane = {
            "desired_model": "Cortex (32B)",
            "desired_endpoint": PRIMARY_ENDPOINT,
            "foreground_endpoint": None,
            "background_endpoint": BRAINSTEM_ENDPOINT,
            "foreground_tier": "local",
            "background_tier": "local_fast",
            "state": _default_state,
            "last_failure_reason": self._init_error or "",
            "conversation_ready": False,
            "cortex_recovery_attempts": getattr(self, "_cortex_recovery_attempts", 0),
            # When no generation has ever succeeded, the honest age is
            # "at least since the gate was constructed" — never zero.
            "has_generated_successfully": bool(
                getattr(self, "_last_successful_generation_at", 0.0) > 0.0
            ),
            "time_since_last_success_s": max(
                0,
                time.time()
                - (
                    getattr(self, "_last_successful_generation_at", 0.0)
                    or getattr(self, "_constructed_wall_at", time.time())
                ),
            ),
            "last_transition_at": 0.0,
            "last_ready_at": 0.0,
            "last_progress_at": 0.0,
            "warmup_attempted": False,
            "warmup_in_flight": bool(self._prewarm_task and not self._prewarm_task.done()),
            # getattr defaults: this snapshot feeds watchdogs and recovery
            # probes — a missing informational field must degrade to its
            # default, never raise out of a status read.
            "last_user_generation_endpoint": getattr(
                self, "_last_user_generation_endpoint", None
            ),
            "last_user_generation_at": getattr(self, "_last_user_generation_at", 0.0),
            "last_user_generation_used_fallback": getattr(
                self, "_last_user_generation_used_fallback", False
            ),
        }
        now_wall = time.time()
        raw_ready = False
        raw_readiness_blockers: list[str] = []
        visible_conversation_anchor = 0.0
        visible_anchor_recent = False
        if self._mlx_client and hasattr(self._mlx_client, "get_lane_status"):
            raw = self._mlx_client.get_lane_status()
            lane["state"] = str(raw.get("state", lane["state"]) or lane["state"])
            lane["last_failure_reason"] = str(
                raw.get("last_error", "") or lane["last_failure_reason"]
            )
            raw_ready = bool(raw.get("conversation_ready", False))
            raw_readiness_blockers = [
                str(blocker)
                for blocker in (raw.get("readiness_blockers") or [])
                if str(blocker or "").strip()
            ]
            if raw.get("runtime_identity_ok") is False:
                detected_models = raw.get("detected_models") or []
                identity_blocker = (
                    "runtime_identity_mismatch"
                    if detected_models
                    else "runtime_identity_unverified"
                )
                if identity_blocker not in raw_readiness_blockers:
                    raw_readiness_blockers.append(identity_blocker)
            if raw_readiness_blockers:
                raw_ready = False
            lane["conversation_ready"] = raw_ready
            lane["readiness_blockers"] = raw_readiness_blockers
            if raw_readiness_blockers and not lane["last_failure_reason"]:
                lane["last_failure_reason"] = ",".join(raw_readiness_blockers[:3])
            lane["last_transition_at"] = float(raw.get("last_transition_at", 0.0) or 0.0)
            lane["last_ready_at"] = float(raw.get("last_ready_at", 0.0) or 0.0)
            lane["last_progress_at"] = float(raw.get("last_progress_at", 0.0) or 0.0)
            lane["warmup_attempted"] = bool(raw.get("warmup_attempted", False))
            lane["warmup_in_flight"] = bool(raw.get("warmup_in_flight", lane["warmup_in_flight"]))
            lane["foreground_owned"] = bool(raw.get("foreground_owned", False))
            lane["foreground_owner"] = str(raw.get("foreground_owner", "") or "")
            lane["active_generations"] = int(raw.get("active_generations", 0) or 0)
            lane["request_age_s"] = float(raw.get("request_age_s", 0.0) or 0.0)
            lane["current_request_started_at"] = float(
                raw.get("current_request_started_at", 0.0) or 0.0
            )
            for telemetry_key in (
                "model_path",
                "recurrent_depth",
                "last_heartbeat",
                "last_token_progress_at",
                "last_generation_completed_at",
                "last_user_facing_completed_at",
                "last_visible_readiness_at",
                "process_started_at",
            ):
                if telemetry_key in raw:
                    lane[telemetry_key] = raw.get(telemetry_key)
            visible_conversation_anchor = max(
                float(lane.get("last_visible_readiness_at", 0.0) or 0.0),
                float(lane.get("last_user_facing_completed_at", 0.0) or 0.0),
            )
            visible_anchor_recent = (
                visible_conversation_anchor > 0.0
                and (now_wall - visible_conversation_anchor) <= 300.0
            )
            # The visible-conversation-probe guard catches a zombie chat lane that
            # reports "ready" without EVER serving a user-facing turn. That signal
            # is only meaningful when a UI/conversation surface is attached. A
            # headless proof/longevity run has no user surface, so no turn can ever
            # refresh the anchor — a warm+alive cortex is the legitimate terminal
            # ready state there, and applying the guard is a false positive.
            _proof_headless = False
            try:
                from core.runtime.proof_policy import proof_headless_run

                _proof_headless = proof_headless_run()
            except (ImportError, RuntimeError, AttributeError) as exc:
                logger.debug("Visible-probe proof-policy check unavailable: %s", exc)
            # Fire ONLY when the lane has never served a visible turn
            # (anchor <= 0) — matching the authoritative mlx_client guard
            # (mlx_client.py: `visible_conversation_anchor <= 0.0`) and this
            # guard's own "without ever serving" intent. A lane that already
            # proved it can serve and is merely IDLE (anchor > 0 but older than
            # 300s) must NOT be downgraded to not-ready — its liveness is covered
            # by the worker-progress-staleness probe, not by conversation idleness.
            if (
                str(lane.get("state", "") or "").lower() == "ready"
                and visible_conversation_anchor <= 0.0
                and not _proof_headless
                and "visible_conversation_probe_missing" not in raw_readiness_blockers
            ):
                raw_readiness_blockers.append("visible_conversation_probe_missing")
                raw_ready = False
                lane["conversation_ready"] = False
                lane["readiness_blockers"] = raw_readiness_blockers
                if not lane["last_failure_reason"]:
                    lane["last_failure_reason"] = "visible_conversation_probe_missing"
            if lane["conversation_ready"]:
                lane["foreground_endpoint"] = PRIMARY_ENDPOINT
        # [STABILITY v51] If the prewarm task completed (success or failure),
        # force-sync warmup_in_flight to False. A done task is no longer "in flight"
        # regardless of what the MLX client flag says.
        if self._prewarm_task and self._prewarm_task.done():
            lane["warmup_in_flight"] = False
            # [STABILITY v53] If prewarm task completed with an exception and
            # conversation is NOT ready, set state to "recovering" and auto-schedule
            # a background recovery. This prevents the zombie warming state where
            # the task finished but the lane never transitions out.
            if not lane["conversation_ready"]:
                try:
                    exc = self._prewarm_task.exception()
                except asyncio.CancelledError:
                    exc = asyncio.CancelledError("prewarm_cancelled")
                except asyncio.InvalidStateError:
                    exc = None
                if exc is not None:
                    lane["state"] = "recovering"
                    lane["last_failure_reason"] = f"prewarm_failed:{type(exc).__name__}"
                    # Auto-trigger recovery if not already in progress
                    if not self._cortex_recovery_in_progress and not (
                        self._deferred_prewarm_task and not self._deferred_prewarm_task.done()
                    ):
                        try:
                            warmup_deferral = self._cortex_warmup_deferral_reason("background")
                            if warmup_deferral:
                                self._log_cortex_warmup_deferral(
                                    warmup_deferral, context="background"
                                )
                            else:
                                self._schedule_background_cortex_prewarm_from_status(
                                    delay=2.0,
                                    reason="failed_prewarm_observed",
                                )
                                logger.info(
                                    "🔄 [STABILITY v53] Auto-scheduling cortex recovery after failed prewarm: %s",
                                    exc,
                                )
                        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                            logger.debug("Best-effort Cortex recovery scheduling skipped: %s", exc)
        lane_state = str(lane.get("state", "") or "").lower()
        _last_success_at = getattr(self, "_last_successful_generation_at", 0.0)
        recent_success = (
            _last_success_at > 0.0 and (now_wall - _last_success_at) <= 30.0
        )
        recent_ready = any(
            stamp > 0.0 and (now_wall - stamp) <= 300.0
            for stamp in (
                float(lane.get("last_ready_at", 0.0) or 0.0),
                float(lane.get("last_progress_at", 0.0) or 0.0),
            )
        )
        if raw_ready and not raw_readiness_blockers:
            lane["conversation_ready"] = True
            lane["foreground_endpoint"] = PRIMARY_ENDPOINT
        elif raw_readiness_blockers:
            lane["conversation_ready"] = False
        elif lane_state == "ready" and (recent_success or recent_ready) and visible_anchor_recent:
            lane["conversation_ready"] = True
            lane["foreground_endpoint"] = PRIMARY_ENDPOINT
        elif lane_state != "ready":
            lane["conversation_ready"] = False
        lane_state = str(lane.get("state", "") or "").lower()
        if (
            self._cortex_recovery_in_progress
            and not lane["conversation_ready"]
            and lane_state != "failed"
        ):
            lane["state"] = "recovering"
        if (
            self._prewarm_task
            and not self._prewarm_task.done()
            and not lane["conversation_ready"]
            and lane_state != "failed"
        ):
            lane["state"] = "warming"
            lane["warmup_in_flight"] = True
        # [STABILITY v53] Stale state watchdog: if lane has been in warming/recovering
        # for >90s with no progress and no active task, force to "cold" so the next
        # user request triggers a fresh warmup instead of waiting on a ghost.
        if lane_state in ("warming", "recovering") and not lane["conversation_ready"]:
            # [STABILITY v54] Eagerly cancel and clear prewarm task if it has been active for >300s.
            if self._prewarm_task and not self._prewarm_task.done():
                transition_at = getattr(self._mlx_client, "_lane_transition_at", 0.0) if self._mlx_client else 0.0
                if transition_at > 0 and (time.time() - transition_at) > 300.0:
                    logger.warning(
                        "🔍 [WATCHDOG] Prewarm task is active for >300s (stuck). Cancelling task."
                    )
                    self._prewarm_task.cancel()
                    self._prewarm_task = None

            last_progress = max(
                float(lane.get("last_transition_at", 0.0) or 0.0),
                float(lane.get("last_progress_at", 0.0) or 0.0),
            )
            if last_progress > 0 and (time.time() - last_progress) > 90.0:
                has_active_task = (
                    (self._prewarm_task and not self._prewarm_task.done())
                    or (self._deferred_prewarm_task and not self._deferred_prewarm_task.done())
                    or self._cortex_recovery_in_progress
                )
                if not has_active_task:
                    # [HARDENING v54] Rate-limit this log — get_conversation_status()
                    # is called dozens of times per second by subsystems. Without
                    # rate limiting, a stuck lane produces thousands of warnings.
                    _now_mono = time.monotonic()
                    _last_log = getattr(self, "_last_stale_reset_log_at", 0.0)
                    if (_now_mono - _last_log) > 30.0:
                        self._last_stale_reset_log_at = _now_mono
                        logger.warning(
                            "🚨 [HARDENING v54] Lane stuck in '%s' for >90s with no active task. "
                            "Resetting to 'cold' and scheduling recovery.",
                            lane_state,
                        )
                    lane["state"] = "cold"
                    lane["warmup_in_flight"] = False
                    # [HARDENING v54] CRITICAL: Reset the MLX client's ACTUAL lane
                    # state, not just the returned dict. Without this, the next call
                    # reads "recovering" from the client again and the stale check
                    # fires in an infinite loop.
                    if self._mlx_client:
                        if hasattr(self._mlx_client, "_warmup_in_flight"):
                            self._mlx_client._warmup_in_flight = False
                        if hasattr(self._mlx_client, "_set_lane_state"):
                            self._mlx_client._set_lane_state("cold")
                    # [HARDENING v54] Schedule a recovery warmup so the cortex
                    # actually comes back online instead of staying cold forever.
                    # The prewarm runner performs the RAM admission check before
                    # loading anything; scheduling the runner here keeps recovery
                    # alive without forcing an unsafe immediate model load.
                    try:
                        self._schedule_background_cortex_prewarm_from_status(
                            delay=3.0,
                            reason="stale_lane_observed",
                        )
                    except _INFERENCE_RECOVERABLE_ERRORS as exc:
                        _record_inference_degradation(
                            exc,
                            action="returned conservative conversation status after probe failure",
                        )
                        logger.debug("Best-effort Cortex recovery scheduling skipped: %s", exc)
        return lane

    def get_lane_status(self) -> dict[str, Any]:
        """Expose the live MLX lane contract for routers, probes, and audits."""
        return self.get_conversation_status()

    def note_foreground_timeout(self, reason: str = "foreground_timeout") -> None:
        """Mark the conversation lane as degraded after a foreground timeout."""
        if self._mlx_client and hasattr(self._mlx_client, "note_lane_recovering"):
            try:
                self._mlx_client.note_lane_recovering(reason)
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                _record_inference_degradation(
                    exc,
                    action="recorded timeout without blocking later foreground recovery",
                )
                logger.debug("Failed to mark cortex lane recovering: %s", exc)
        self._extend_startup_quiet_window(8.0)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        try:
            warmup_deferral = self._cortex_warmup_deferral_reason("background")
            if warmup_deferral:
                self._log_cortex_warmup_deferral(warmup_deferral, context="background")
            else:
                self._schedule_background_cortex_prewarm(delay=2.0)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "inference_gate",
                exc,
                severity="warning",
                action="left deferred cortex re-prewarm unscheduled; foreground path will retry",
            )
            logger.debug("Failed to schedule deferred cortex re-prewarm after timeout: %s", exc)

    def _extend_startup_quiet_window(self, seconds: float) -> None:
        orch = self.orch
        if orch is None:
            try:
                from core.container import ServiceContainer

                orch = ServiceContainer.get("orchestrator", default=None)
            except _INFERENCE_RECOVERABLE_ERRORS:
                orch = None
        if orch and hasattr(orch, "_extend_foreground_quiet_window"):
            try:
                orch._extend_foreground_quiet_window(seconds)
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                record_degradation(
                    "inference_gate",
                    exc,
                    severity="warning",
                    action="continued without extending foreground quiet window",
                )
                logger.debug("Failed to extend foreground quiet window: %s", exc)

    def _schedule_background_cortex_prewarm_from_status(
        self,
        *,
        delay: float,
        reason: str,
        min_interval_s: float = 30.0,
    ) -> None:
        """Cooldowned recovery scheduling for status-observation paths.

        ``get_conversation_status()`` is polled by health endpoints and the
        neural stream. It may notice a completed failed warmup, but observation
        must not become an unbounded work generator.
        """

        now = time.monotonic()
        if (now - self._last_status_recovery_schedule_at) < max(1.0, min_interval_s):
            logger.debug(
                "Skipping status-triggered Cortex prewarm (%s); cooldown active.",
                reason,
            )
            return
        self._last_status_recovery_schedule_at = now
        self._schedule_background_cortex_prewarm(delay=delay)

    def _schedule_background_cortex_prewarm(self, delay: float = 12.0) -> None:
        if is_shutdown_requested():
            logger.debug("Deferred cortex prewarm skipped: runtime shutdown requested.")
            return
        if self._deferred_prewarm_task and not self._deferred_prewarm_task.done():
            return

        async def _runner():
            next_delay = max(1.0, float(delay))
            for attempt in range(1, 7):
                await asyncio.sleep(next_delay)
                if is_shutdown_requested():
                    logger.debug("Deferred cortex prewarm stopped: runtime shutdown requested.")
                    return
                lane = self.get_conversation_status()
                lane_state = str(lane.get("state", "") or "").lower()
                if lane.get("conversation_ready") or lane.get("warmup_in_flight"):
                    return
                if self._lane_reports_active_generation(lane):
                    logger.info(
                        "⏸️ Deferred cortex prewarm postponed while foreground generation is active."
                    )
                    next_delay = min(20.0, max(6.0, next_delay))
                    continue
                if lane_state == "failed":
                    if is_shutdown_requested():
                        return
                    if await asyncio.to_thread(self._rearm_runtime_failed_lane, force_probe=False):
                        lane = self.get_conversation_status()
                        lane_state = str(lane.get("state", "") or "").lower()
                    elif str(lane.get("last_failure_reason", "") or "").startswith(
                        ("mlx_runtime_unavailable", "local_runtime_unavailable")
                    ):
                        logger.info(
                            "⏸️ Deferred cortex prewarm postponing while runtime lane is still unavailable (%s).",
                            lane.get("last_failure_reason") or "unknown",
                        )
                        next_delay = min(45.0, max(12.0, next_delay * 1.5))
                        continue
                    else:
                        logger.warning(
                            "⏸️ Deferred cortex prewarm cancelled: lane is in a failed state (%s).",
                            lane.get("last_failure_reason") or "unknown",
                        )
                        return
                if self._foreground_user_turn_active() or self._foreground_owner_active():
                    next_delay = min(20.0, max(6.0, next_delay))
                    continue
                warmup_deferral = self._cortex_warmup_deferral_reason("background")
                if warmup_deferral:
                    self._log_cortex_warmup_deferral(warmup_deferral, context="background")
                    next_delay = min(90.0, max(20.0, next_delay * 1.5))
                    continue
                try:
                    vm = psutil.virtual_memory()
                    total_gb = vm.total / float(1024**3)
                    available_gb = vm.available / float(1024**3)
                    critical_pressure = vm.percent >= (92.0 if total_gb >= 60.0 else 88.0)
                    critical_available = available_gb < (6.0 if total_gb >= 60.0 else 10.0)
                    if critical_pressure or critical_available:
                        logger.warning(
                            "⏸️ Deferred cortex prewarm postponed (attempt=%d pressure=%.1f%% available=%.1fGB).",
                            attempt,
                            vm.percent,
                            available_gb,
                        )
                        next_delay = min(45.0, max(12.0, next_delay * 1.5))
                        continue
                except _INFERENCE_RECOVERABLE_ERRORS as exc:
                    record_degradation(
                        "inference_gate",
                        exc,
                        severity="warning",
                        action="continued deferred prewarm with conservative retry delay",
                    )
                    logger.debug("Deferred prewarm memory probe failed: %s", exc)

                try:
                    if is_shutdown_requested():
                        return
                    self._extend_startup_quiet_window(20.0)
                    # Background prewarm needs the same generous load budget
                    # as foreground chat so it does not half-warm then strand
                    # the next user turn in recovery.
                    await self.ensure_foreground_ready(timeout=300.0)
                    logger.info("✅ Deferred cortex prewarm completed.")
                    return
                except _INFERENCE_RECOVERABLE_ERRORS as exc:
                    if self._exception_reports_active_generation(exc):
                        logger.info(
                            "⏸️ Deferred cortex prewarm postponed while foreground generation is active."
                        )
                        next_delay = min(20.0, max(6.0, next_delay))
                        continue
                    if "visible_conversation_probe_missing" in str(exc):
                        logger.info(
                            "⏸️ Deferred cortex prewarm loaded the lane, but visible "
                            "conversation readiness is still unproven; the next "
                            "foreground user turn will prove or fail it."
                        )
                        next_delay = min(60.0, max(20.0, next_delay * 1.25))
                        continue
                    record_degradation(
                        "inference_gate",
                        exc,
                        severity="warning",
                        action="backed off deferred cortex prewarm and will retry",
                    )
                    logger.warning(
                        "⚠️ Deferred cortex prewarm failed (attempt=%d): %s", attempt, exc
                    )
                    next_delay = min(45.0, max(12.0, next_delay * 1.5))

            logger.warning(
                "⚠️ Deferred cortex prewarm exhausted retries; foreground turn will retry on demand."
            )

        runner_coro = _runner()
        try:
            task = get_task_tracker().create_task(
                runner_coro,
                name="InferenceGate.deferred_cortex_prewarm",
            )
        except RuntimeError:
            runner_coro.close()
            logger.debug("Deferred cortex prewarm skipped: no running event loop.")
            return

        if not isinstance(task, asyncio.Task):
            runner_coro.close()
            logger.debug(
                "Deferred cortex prewarm scheduling returned non-Task %s; skipping callback wiring.",
                type(task).__name__,
            )
            return
        self._deferred_prewarm_task = task
        # [STABILITY v53] Log exceptions from background tasks
        self._deferred_prewarm_task.add_done_callback(self._log_task_exception)

    def _rearm_runtime_failed_lane(self, *, force_probe: bool) -> bool:
        client = self._mlx_client
        if client is None or not hasattr(client, "refresh_runtime_availability"):
            return False

        lane = self.get_conversation_status()
        lane_state = str(lane.get("state", "") or "").lower()
        lane_reason = str(lane.get("last_failure_reason", "") or "")
        if lane_state != "failed" or not lane_reason.startswith(
            ("mlx_runtime_unavailable", "local_runtime_unavailable")
        ):
            return False

        try:
            rearmed = bool(client.refresh_runtime_availability(force_probe=force_probe))
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="continued bounded inference fallback after non-fatal degradation",
            )
            logger.debug("Failed to re-arm runtime-blocked Cortex lane: %s", exc)
            return False

        if rearmed:
            logger.info(
                "♻️ InferenceGate: re-armed the Cortex lane after transient runtime failure (%s).",
                lane_reason,
            )
        return rearmed

    @staticmethod
    def _lane_only_needs_visible_conversation_proof(lane: dict[str, Any] | None) -> bool:
        """Return True when the worker is loaded and only lacks a visible turn proof."""

        if not isinstance(lane, dict):
            return False
        blockers = {
            str(item)
            for item in (lane.get("readiness_blockers") or [])
            if str(item or "").strip()
        }
        if blockers != {"visible_conversation_probe_missing"}:
            return False
        return (
            str(lane.get("state", "") or "").lower() == "ready"
            and bool(lane.get("warmup_attempted", True))
            and not bool(lane.get("warmup_in_flight", False))
        )

    @classmethod
    def _lane_can_attempt_visible_conversation_turn(cls, lane: dict[str, Any] | None) -> bool:
        """Return True when a lane may serve the foreground turn that proves readiness."""

        return bool(
            isinstance(lane, dict)
            and (
                lane.get("conversation_ready")
                or cls._lane_only_needs_visible_conversation_proof(lane)
            )
        )

    def _foreground_warmup_timeout(
        self, lane_status: dict[str, Any], primary_timeout: float
    ) -> float:
        """Admission control for the foreground preflight — break the doom loop.

        A COLD first boot legitimately needs ~150s to load the 32B, and the
        user expects that one-time wait. But a RECOVERY (Cortex was ready, got
        force-killed on a first-token stall, is reloading) must NOT hold every
        foreground turn hostage for 90-180s — observed live (Jul 7 soak):
        turns 21-30 crawled to 200s+ while a single warm window played out.

        When the lane was EVER ready (``last_ready_at`` > 0), cap the wait
        short (floored to 15s by ensure_foreground_ready — one honest warm
        chance) and let the turn fall to the ready fallback tier; the warmup
        task is shielded, so Cortex keeps warming in the background and the
        NEXT turn gets it. AURA_FOREGROUND_RECOVERY_WARMUP_CAP_S=180 restores
        the old behavior if this ever needs reverting live.
        """
        was_ever_ready = float(lane_status.get("last_ready_at", 0.0) or 0.0) > 0.0
        if was_ever_ready:
            return InferenceGate._env_float(
                "AURA_FOREGROUND_RECOVERY_WARMUP_CAP_S", 15.0
            )
        # [STABILITY v56] Cold 32B load can take 150s; give it at least 180s
        # or the primary timeout, whichever is greater.
        return max(180.0, float(primary_timeout))

    async def ensure_foreground_ready(self, timeout: float | None = None) -> dict[str, Any]:  # noqa: ASYNC109
        """Ensure the 32B conversation lane has actually attempted warmup for this turn."""
        if is_shutdown_requested():
            raise RuntimeError("runtime_shutdown")
        timeout = max(15.0, float(timeout or 90.0))
        lane = self.get_conversation_status()
        if self._lane_can_attempt_visible_conversation_turn(lane):
            # Cortex is serving again — clear any post-thrash warmup cooldown.
            self._reset_cortex_warmup_backoff()
            return lane
        # Recovery cap: a lane that was EVER ready and is now warming/
        # recovering must NOT hold the turn for the full cold-boot budget.
        # The chat caller passes 180s; without this cap every turn blocked
        # ~206s on a recovering cortex ("Protected foreground lane failed
        # (lane_warming): Cortex timed out after 206s" — the 2026-07-15 soak
        # wall). _foreground_warmup_timeout returns 15s for a recovery and
        # the cold budget for a genuine cold boot; take the tighter of the
        # two so a recovering turn falls to the fast fallback while Cortex
        # re-warms in the background (its warmup task is shielded).
        timeout = min(timeout, self._foreground_warmup_timeout(lane, timeout))
        lane_state = str(lane.get("state", "") or "").lower()
        lane_reason = str(lane.get("last_failure_reason", "") or "")
        if lane_state == "failed" and lane_reason.startswith(
            ("mlx_runtime_unavailable", "local_runtime_unavailable")
        ):
            if await asyncio.to_thread(self._rearm_runtime_failed_lane, force_probe=True):
                lane = self.get_conversation_status()
            else:
                raise RuntimeError(lane_reason)
        if not self._mlx_client or not hasattr(self._mlx_client, "warmup"):
            raise RuntimeError("foreground_lane_unavailable")

        task: asyncio.Task | None = None
        try:
            async with _thread_lock_context(
                self._foreground_ready_lock,
                timeout_s=min(timeout, 30.0),
                label="foreground_ready_lock",
            ):
                lane = self.get_conversation_status()
                if self._lane_can_attempt_visible_conversation_turn(lane):
                    return lane
                if self._prewarm_task and not self._prewarm_task.done():
                    task = self._prewarm_task
                else:
                    warmup_deferral = self._cortex_warmup_deferral_reason("foreground")
                    if warmup_deferral:
                        await self._shed_background_workers_for_memory_pressure(
                            force=True,
                            reason="foreground_cortex_warmup_admission",
                        )
                        gc.collect()
                        warmup_deferral = self._cortex_warmup_deferral_reason("foreground")
                    if warmup_deferral:
                        self._log_cortex_warmup_deferral(warmup_deferral, context="foreground")
                        if hasattr(self._mlx_client, "note_lane_recovering"):
                            self._mlx_client.note_lane_recovering(
                                "foreground_warmup_deferred_memory_pressure"
                            )
                        raise RuntimeError(f"foreground_warmup_deferred:{warmup_deferral}")
                    self._extend_startup_quiet_window(20.0)
                    if is_shutdown_requested():
                        raise RuntimeError("runtime_shutdown")
                    self._prewarm_task = get_task_tracker().create_task(
                        self._mlx_client.warmup(),
                        name="InferenceGate.ensure_foreground_ready",
                    )
                    task = self._prewarm_task
        except TimeoutError as exc:
            raise RuntimeError(str(exc)) from exc

        try:
            task_loop = getattr(task, "get_loop", lambda: asyncio.get_running_loop())()
            current_loop = asyncio.get_running_loop()
            if task_loop is not current_loop:

                async def _await_foreign_task() -> Any:
                    return await task

                future = asyncio.run_coroutine_threadsafe(_await_foreign_task(), task_loop)
                await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout)
            else:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except TimeoutError:
            # A foreground warmup that keeps timing out is the cortex failing to
            # load in time — the same GPU-thrash signal as a stuck-load kill, but
            # with no force-kill to observe (the worker just stays "warming").
            # Feed it into the warmup backoff so repeated stalls defer cortex
            # warmup and free the single GPU slot for the resident fallback that
            # is actually serving the turn. Without this the cortex load and the
            # fallback cold-load fight over one GPU slot and neither wins.
            self._note_cortex_stuck_kill()
            if hasattr(self._mlx_client, "note_lane_recovering"):
                self._mlx_client.note_lane_recovering("foreground_warmup_timeout")
            raise
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="continued bounded inference fallback after non-fatal degradation",
            )
            if hasattr(self._mlx_client, "note_lane_failed"):
                self._mlx_client.note_lane_failed(f"foreground_warmup_failed:{type(exc).__name__}")
            raise RuntimeError("foreground_warmup_failed") from exc

        lane = self.get_conversation_status()
        if not lane.get("conversation_ready"):
            if self._lane_only_needs_visible_conversation_proof(lane):
                return lane
            raise RuntimeError(str(lane.get("last_failure_reason") or "foreground_lane_not_ready"))
        return lane

    def _confirmed_cortex_warmup(
        self, warmup_result: Any
    ) -> tuple[bool, dict[str, Any], str]:
        """Require process and lane evidence before reporting a warmup as successful."""
        lane = self.get_conversation_status()
        state = str(lane.get("state", "") or "").strip().lower()
        blockers = [
            str(blocker)
            for blocker in (lane.get("readiness_blockers") or [])
            if str(blocker or "").strip()
        ]
        try:
            worker_alive = bool(self._mlx_client and self._mlx_client.is_alive())
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            worker_alive = False
            blockers.append(f"worker_probe_failed:{type(exc).__name__}")

        ready = bool(
            warmup_result is not False
            and not is_shutdown_requested()
            and worker_alive
            and state == "ready"
            and lane.get("conversation_ready")
        )
        if ready:
            return True, lane, ""
        if is_shutdown_requested():
            reason = "runtime_shutdown"
        elif warmup_result is False:
            reason = str(lane.get("last_failure_reason") or "warmup_deferred")
        elif not worker_alive:
            reason = "worker_not_alive"
        elif state != "ready":
            reason = f"lane_{state or 'unknown'}"
        elif not lane.get("conversation_ready"):
            reason = ",".join(blockers[:3]) or "conversation_not_ready"
        else:
            reason = "warmup_not_confirmed"
        return False, lane, reason

    async def _ensure_cortex_recovery(self) -> None:
        """Proactively recover the 32B primary brain if it died (e.g., laptop sleep).

        Without this, background tasks keep the 7B alive indefinitely and the 32B
        never gets a chance to respawn because background requests are locked to
        tertiary tier.  Rate-limited to one attempt per 3s.
        """
        if is_shutdown_requested():
            logger.debug("Primary cortex recovery skipped: runtime shutdown requested.")
            return
        if not self._mlx_client:
            return
        if proof_run_active(origin="cortex_recovery") and proof_model_tier() != "primary":
            logger.debug(
                "Primary cortex recovery skipped during non-primary proof lane (%s).",
                proof_model_tier(),
            )
            return
        if not hasattr(self._mlx_client, "is_alive"):
            return
        if self._mlx_client.is_alive():
            return  # Primary is fine

        now = time.monotonic()
        if (now - self._last_cortex_check) < 3.0:
            return  # [STABILITY v51] Rate limit: 3s between attempts
        self._last_cortex_check = now

        if self._cortex_recovery_attempts >= 5:
            # [HARDENING v54] Exponential backoff: 30s after 5 failures, 60s after 10,
            # capped at 120s. The previous 5-minute hard lockout made the cortex
            # unreachable for entire conversation windows. Never permanently give up.
            exhausted_at = getattr(self, "_cortex_recovery_exhausted_at", 0.0)
            cooldown = min(120.0, 30.0 * (1 + (self._cortex_recovery_attempts - 5) // 5))
            if exhausted_at == 0.0:
                self._cortex_recovery_exhausted_at = now
                logger.warning(
                    "[RECOVERY] Primary cortex: %d failures reached. Will retry in %.0fs.",
                    self._cortex_recovery_attempts,
                    cooldown,
                )
                return
            if (now - exhausted_at) < cooldown:
                return  # Rate-limit: exponential backoff
            logger.warning(
                "[RECOVERY] Primary cortex: %.0fs cooldown elapsed. Resetting counter and retrying.",
                cooldown,
            )
            self._cortex_recovery_attempts = 0
            self._cortex_recovery_exhausted_at = 0.0

        if self._cortex_recovery_in_progress:
            return  # Already recovering — don't double-spawn
        if not hasattr(self._mlx_client, "warmup"):
            return
        # Reserve recovery ownership IMMEDIATELY — before any await. The
        # policy checks below suspend the coroutine, and two concurrent
        # callers could otherwise both pass the in-progress check and race
        # into duplicate warmups or duplicate private process cleanup.
        self._cortex_recovery_in_progress = True
        reservation_transferred = False
        try:
            lane = self.get_conversation_status()
            lane_state = str(lane.get("state", "") or "").lower()
            lane_reason = str(lane.get("last_failure_reason", "") or "")
            cold_start_recovery = lane_state in {
                "cold",
                "spawning",
                "handshaking",
                "warming",
            } or not bool(lane.get("warmup_attempted", False))
            if lane_state == "failed" and lane_reason.startswith(
                ("mlx_runtime_unavailable", "local_runtime_unavailable")
            ):
                if await asyncio.to_thread(self._rearm_runtime_failed_lane, force_probe=True):
                    lane = self.get_conversation_status()
                    lane_state = str(lane.get("state", "") or "").lower()
                    lane_reason = str(lane.get("last_failure_reason", "") or "")
                    cold_start_recovery = lane_state in {
                        "cold",
                        "spawning",
                        "handshaking",
                        "warming",
                    } or not bool(lane.get("warmup_attempted", False))
                else:
                    return
            if lane.get("warmup_in_flight"):
                return
            if self._foreground_user_turn_active() or self._foreground_owner_active():
                return
            if cold_start_recovery:
                if self._prewarm_task is not None and not self._prewarm_task.done():
                    logger.debug("Cold-start Cortex recovery skipped; deferred prewarm task is already scheduled.")
                    return
                if not self._boot_should_schedule_deferred_prewarm():
                    self._log_cold_cortex_policy_deferred()
                    return
            warmup_deferral = self._cortex_warmup_deferral_reason("recovery")
            if warmup_deferral:
                self._log_cortex_warmup_deferral(warmup_deferral, context="recovery")
                return
        finally:
            if not reservation_transferred:
                self._cortex_recovery_in_progress = False

        async def _background_recover():
            if is_shutdown_requested():
                self._cortex_recovery_in_progress = False
                return
            self._cortex_recovery_in_progress = True
            self._cortex_recovery_attempts += 1

            if self._cortex_recovery_attempts == 3:
                logger.warning(
                    "🧹 [RECOVERY] 3 failed attempts. Forcing deep GC and stale process cleanup..."
                )
                import gc

                gc.collect()
                # Bind the kill to the exact client + process handle observed
                # NOW, and refuse when the worker is legitimately loading or
                # serving — an unowned kill here was the doom-loop trigger
                # (kill mid-load → warmup_deferred → repeat).
                kill_client = self._mlx_client
                kill_process = getattr(kill_client, "_process", None)
                if kill_process is None:
                    logger.debug("[RECOVERY] No worker process handle to clean up.")
                elif self._cortex_worker_is_legitimately_loading(kill_client):
                    logger.info(
                        "[RECOVERY] Skipping stale-process kill: worker is legitimately loading."
                    )
                elif self._lane_reports_active_generation(self.get_conversation_status()):
                    logger.info(
                        "[RECOVERY] Skipping stale-process kill: lane reports an active generation."
                    )
                else:
                    try:
                        await asyncio.to_thread(
                            kill_client._kill_and_join_blocking, kill_process
                        )
                    except _INFERENCE_RECOVERABLE_ERRORS as _e:
                        _record_inference_degradation(
                            _e,
                            action="continued background recovery loop with degraded signal",
                        )
                        logger.debug("Ignored Exception in inference_gate.py killing process: %s", _e)

            try:
                warmup_deferral = self._cortex_warmup_deferral_reason("recovery")
                if warmup_deferral:
                    self._log_cortex_warmup_deferral(warmup_deferral, context="recovery")
                    return
                if is_shutdown_requested():
                    logger.debug("Primary cortex recovery stopped before warmup: runtime shutdown requested.")
                    return
                if cold_start_recovery:
                    logger.info(
                        "♻️ [STARTUP] Primary 32B cortex is cold. Starting warmup (Attempt %d/5)...",
                        self._cortex_recovery_attempts,
                    )
                else:
                    logger.warning(
                        "♻️ [RECOVERY] Primary 32B cortex is dead. Triggering background respawn (Attempt %d/5)...",
                        self._cortex_recovery_attempts,
                    )
                self._prewarm_task = get_task_tracker().create_task(
                    self._mlx_client.warmup(),
                    name="InferenceGate.cortex_recovery",
                )
                # 32B fused model is ~37GB across 7 shards. Cold-load on Apple
                # Silicon routinely takes 90-150s on the first attempt after a
                # crash; the previous 60s budget guaranteed five back-to-back
                # timeouts and a 5-minute lockout. Give warmup the room it
                # actually needs.
                warmup_result = await asyncio.wait_for(
                    asyncio.shield(self._prewarm_task), timeout=420.0
                )
                ready, recovered_lane, incomplete_reason = self._confirmed_cortex_warmup(
                    warmup_result
                )
                if not ready:
                    if is_shutdown_requested():
                        logger.info(
                            "🛑 [RECOVERY] Primary 32B cortex warmup stopped during runtime shutdown."
                        )
                    else:
                        logger.warning(
                            "⚠️ [RECOVERY] Primary 32B cortex warmup did not establish readiness "
                            "(state=%s, reason=%s).",
                            recovered_lane.get("state", "unknown"),
                            incomplete_reason,
                        )
                    return
                if cold_start_recovery:
                    logger.info("✅ [STARTUP] Primary 32B cortex warmup complete.")
                else:
                    logger.info("✅ [RECOVERY] Primary 32B cortex restored after disruption.")
                self._cortex_recovery_attempts = 0
                self._cortex_recovery_exhausted_at = 0.0
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                _record_inference_degradation(
                    exc,
                    action="continued background recovery loop with degraded signal",
                )
                if cold_start_recovery:
                    logger.error(
                        "⚠️ [STARTUP] Primary 32B cortex warmup failed (Attempt %d/5): %s",
                        self._cortex_recovery_attempts,
                        exc,
                    )
                else:
                    logger.error(
                        "⚠️ [RECOVERY] Primary 32B cortex is dead. Triggering background respawn (Attempt %d/5): %s",
                        self._cortex_recovery_attempts,
                        exc,
                    )
            finally:
                # [STABILITY v51] ALWAYS clear the flag, even on unexpected exceptions.
                self._cortex_recovery_in_progress = False

        # [STABILITY v53] Wrap fire-and-forget task with exception logging
        # so crashes are visible instead of silently lost.
        # Reserve recovery ownership before scheduling. Without this reservation,
        # the foreground caller can observe ``False`` immediately after this method
        # returns and start a second inline warmup before the task gets CPU time.
        if is_shutdown_requested():
            logger.debug("Primary cortex recovery task not scheduled: runtime shutdown requested.")
            return
        self._cortex_recovery_in_progress = True
        recovery_coro = _background_recover()
        try:
            task = get_task_tracker().create_task(recovery_coro, name="cortex_recovery")
        except RuntimeError:
            recovery_coro.close()
            self._cortex_recovery_in_progress = False
            logger.debug("Cortex recovery skipped: no running event loop.")
            return
        if not isinstance(task, asyncio.Task):
            recovery_coro.close()
            self._cortex_recovery_in_progress = False
            logger.debug(
                "Cortex recovery task scheduling returned non-Task %s; skipping callback wiring.",
                type(task).__name__,
            )
            return
        # Own the handle so shutdown can cancel and await this recovery like
        # the named prewarm/maintenance tasks — an anonymous recovery task
        # could continue into shutdown and recreate warmup activity.
        self._status_recovery_task = task

        def _finish_recovery(completed: asyncio.Task) -> None:
            # Cancellation can happen before the coroutine reaches its ``finally``
            # block, so the scheduling boundary also owns clearing this reservation.
            self._cortex_recovery_in_progress = False
            if getattr(self, "_status_recovery_task", None) is completed:
                self._status_recovery_task = None
            self._log_task_exception(completed)

        task.add_done_callback(_finish_recovery)

    async def _respawn_cortex_if_needed(self) -> None:
        """Respawn the primary cortex if it's dead.

        Called by HealthRouter and message_handling when inference returns empty.
        Delegates to _ensure_cortex_recovery() which has proper rate-limiting,
        warm-up sequencing, and retry budgets.
        """
        if is_shutdown_requested():
            logger.debug("_respawn_cortex_if_needed skipped: runtime shutdown requested.")
            return
        if (
            self._mlx_client
            and hasattr(self._mlx_client, "is_alive")
            and self._mlx_client.is_alive()
        ):
            return  # Cortex is fine — nothing to do
        if self._cortex_recovery_in_progress:
            logger.debug("_respawn_cortex_if_needed: recovery already in progress.")
            return  # Already recovering — don't double-spawn
        logger.info("🔄 _respawn_cortex_if_needed: cortex is dead, delegating to recovery.")
        await self._ensure_cortex_recovery()

    async def ensure_all_tiers_healthy(self) -> dict[str, str]:
        """Proactive health check for ALL inference tiers. Called by MindTick.

        Returns a dict of {tier: status} for monitoring.
        """
        if is_shutdown_requested():
            return {"cortex": "shutdown"}
        statuses = {}

        # Primary cortex
        try:
            if self._mlx_client and hasattr(self._mlx_client, "is_alive"):
                # [STABILITY v53] Detect warming/recovering states so MindTick
                # doesn't report 'dead' during cold start.
                lane_state = getattr(self._mlx_client, "_lane_state", "cold")

                if self._mlx_client.is_alive():
                    statuses["cortex"] = "alive"
                elif self._cortex_recovery_in_progress or lane_state in (
                    "spawning",
                    "handshaking",
                    "warming",
                    "recovering",
                ):
                    statuses["cortex"] = "recovering"
                else:
                    statuses["cortex"] = "dead"
                    # Trigger recovery if not already in progress.
                    await self._ensure_cortex_recovery()
            else:
                statuses["cortex"] = "not_initialized"
        except _INFERENCE_RECOVERABLE_ERRORS as e:
            _record_inference_degradation(
                e,
                action="continued tier-health sweep after one tier probe failed",
            )
            statuses["cortex"] = f"error:{e}"

        # Brainstem
        try:
            deferral_reason = self._background_local_deferral_reason(origin="tier_health")
            warm_local_tiers = os.environ.get("AURA_HEALTH_WARM_LOCAL_TIERS", "").strip().lower() in {
                "1",
                "true",
                "yes",
            }
            if deferral_reason:
                statuses["brainstem"] = f"deferred:{deferral_reason}"
            else:
                from core.brain.llm.mlx_client import get_mlx_client
                from core.brain.llm.model_registry import get_brainstem_path

                brainstem = get_mlx_client(model_path=str(get_brainstem_path()))
                if brainstem and hasattr(brainstem, "is_alive"):
                    lane_state = getattr(brainstem, "_lane_state", "cold")

                    if brainstem.is_alive():
                        statuses["brainstem"] = "alive"
                    elif lane_state in ("spawning", "handshaking", "warming", "recovering"):
                        statuses["brainstem"] = "recovering"
                    elif not warm_local_tiers:
                        # Brainstem is a demand-loaded background lane. A cold
                        # worker is healthy standby unless policy explicitly
                        # requires it to remain warm; calling it dead creates a
                        # false incident while the required Cortex lane is live.
                        statuses["brainstem"] = "standby"
                    else:
                        statuses["brainstem"] = "dead"
                        # Tier health sweeps are observability by default. They
                        # must not spawn a background 7B worker while a foreground
                        # Cortex turn or proof run owns the local runtime.
                        if hasattr(brainstem, "warmup"):
                            get_task_tracker().create_task(brainstem.warmup())
                            statuses["brainstem"] = "recovering"
                else:
                    statuses["brainstem"] = "not_initialized"
        except _INFERENCE_RECOVERABLE_ERRORS as e:
            _record_inference_degradation(
                e,
                action="continued tier-health sweep after one tier probe failed",
            )
            statuses["brainstem"] = f"error:{e}"

        # Reflex (CPU) — always available if model file exists
        try:
            from core.brain.llm.model_registry import get_fallback_path

            fallback_path = get_fallback_path()
            fallback_exists = bool(
                fallback_path and await asyncio.to_thread(Path(str(fallback_path)).exists)
            )
            if fallback_exists:
                statuses["reflex"] = "available"
            else:
                statuses["reflex"] = "model_missing"
        except _INFERENCE_RECOVERABLE_ERRORS:
            statuses["reflex"] = "unknown"

        return statuses

    @staticmethod
    def _normalize_tier(prefer_tier: str | None) -> str:
        tier = str(prefer_tier or "primary").strip().lower()
        aliases = {
            "local": "primary",
            "local_deep": "secondary",
            "local_fast": "tertiary",
            "fast": "tertiary",
            "deep": "secondary",
        }
        return aliases.get(tier, tier)

    @staticmethod
    def _origin_is_user_facing(origin: str | None) -> bool:
        normalized = str(origin or "").strip().lower().replace("-", "_")
        if not normalized:
            return False
        while normalized.startswith("routing_"):
            normalized = normalized[len("routing_") :]
        if not normalized:
            return False
        if normalized in _USER_FACING_ORIGINS:
            return True
        tokens = {token for token in normalized.split("_") if token}
        if tokens & _USER_FACING_ORIGINS:
            return True
        return any(normalized.startswith(f"{prefix}_") for prefix in _USER_FACING_ORIGINS)

    @staticmethod
    def _foreground_user_turn_active() -> bool:
        try:
            from core.container import ServiceContainer

            orch = ServiceContainer.get("orchestrator", default=None)
            if not orch:
                return False
            status = getattr(orch, "status", None)
            if not getattr(status, "is_processing", False):
                return False
            current_origin = getattr(orch, "_current_origin", "")
            if not InferenceGate._origin_is_user_facing(current_origin):
                return False
            return not bool(getattr(orch, "_current_task_is_autonomous", False))
        except _INFERENCE_RECOVERABLE_ERRORS:
            return False

    @staticmethod
    def _foreground_quiet_window_active() -> bool:
        try:
            from core.container import ServiceContainer

            orch = ServiceContainer.get("orchestrator", default=None)
            if not orch:
                return False
            quiet_until = float(getattr(orch, "_foreground_user_quiet_until", 0.0) or 0.0)
            return quiet_until > time.time()
        except _INFERENCE_RECOVERABLE_ERRORS:
            return False

    def _safe_boot_background_guard_active(self) -> bool:
        """Reserve launch headroom for the live conversation lane."""
        if not self._desktop_safe_boot_enabled():
            return False
        try:
            startup_guard_secs = float(
                os.environ.get("AURA_SAFE_BOOT_BACKGROUND_GUARD_SECS", "180")
            )
        except _INFERENCE_RECOVERABLE_ERRORS:
            startup_guard_secs = 180.0
        if startup_guard_secs <= 0:
            return False
        return (time.monotonic() - self._created_at) < startup_guard_secs

    def _should_quiet_background_for_cortex_startup(self) -> bool:
        """Hold background inference while the live 32B lane is booting or reserving headroom."""
        lane = self.get_conversation_status()
        if self._safe_boot_background_guard_active():
            return True
        if not self._foreground_quiet_window_active():
            return False
        if lane.get("conversation_ready"):
            return False

        state = str(lane.get("state", "") or "").strip().lower()
        if lane.get("warmup_in_flight"):
            return True
        return state in {"cold", "spawning", "handshaking", "warming", "recovering"}

    @staticmethod
    def _background_memory_pressure_active() -> bool:
        try:
            vm = psutil.virtual_memory()
            total_gb = vm.total / float(1024**3)
            available_gb = vm.available / float(1024**3)
            max_pressure = float(
                os.environ.get(
                    "AURA_BACKGROUND_LOCAL_MAX_PRESSURE_PCT",
                    "82" if total_gb >= 60.0 else "78",
                )
            )
            min_available_gb = float(
                os.environ.get(
                    "AURA_BACKGROUND_LOCAL_MIN_AVAILABLE_GB",
                    "12" if total_gb >= 60.0 else "10",
                )
            )
            return bool(vm.percent >= max_pressure or available_gb <= min_available_gb)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="failed closed and deferred background local inference after memory probe failure",
                severity="warning",
            )
            return True

    def _background_local_deferral_reason(self, *, origin: str | None = None) -> str | None:
        try:
            from core.runtime.proof_policy import proof_run_active

            if proof_run_active(origin=origin):
                return "proof_foreground_reserved"
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="kept background local deferral conservative after proof policy probe failed",
            )
            logger.debug("Suppressed Exception: %s", _exc)
        if self._foreground_user_turn_active() or self._foreground_owner_active():
            return "foreground_reserved"
        if self._foreground_headroom_reserved("primary"):
            return "foreground_headroom_reserved"
        if self._should_quiet_background_for_cortex_startup():
            return "cortex_startup_quiet"
        if self._foreground_quiet_window_active():
            return "foreground_quiet_window"

        lane = self.get_conversation_status()
        try:
            from core.brain.llm.model_registry import get_local_backend

            if get_local_backend() != "mlx" and lane.get("conversation_ready"):
                return "cortex_resident"
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="kept background local deferral conservative after policy probe failed",
            )
            logger.debug("Suppressed Exception: %s", _exc)
        lane_state = str(lane.get("state", "") or "").strip().lower()
        if not lane.get("conversation_ready") and lane_state == "failed":
            return "cortex_failed"
        if self._safe_boot_background_guard_active():
            return "cortex_startup_quiet"
        if self._desktop_safe_boot_enabled() and not self._desktop_background_local_enabled():
            return "desktop_background_disabled"
        if self._desktop_safe_boot_enabled() and not lane.get("conversation_ready"):
            if self._background_memory_pressure_active():
                if lane_state in {
                    "cold",
                    "spawning",
                    "handshaking",
                    "warming",
                    "recovering",
                    "failed",
                }:
                    return "memory_pressure"
        if self._background_memory_pressure_active():
            if lane.get("conversation_ready") or lane.get("warmup_in_flight"):
                return "memory_pressure"
            if lane_state in {"spawning", "handshaking", "warming", "recovering", "failed"}:
                return "memory_pressure"
        return None

    async def _shed_background_workers_for_memory_pressure(
        self,
        *,
        force: bool = False,
        reason: str = "background_memory_pressure_shed",
    ) -> None:
        now = time.monotonic()
        if not force and (now - self._last_background_memory_shed_at) < 20.0:
            return

        # Never shed the small fallback models when memory is abundant. They
        # are the guaranteed fast-answer path while the 32B cortex warms; with
        # the router now routing AROUND a not-ready cortex, shedding them left
        # nothing resident to answer and cascaded into a no-reply death spiral
        # (2026-07-15 soak: 7B >56s, 1.5B >14.7s, all thrashing to reload
        # despite 42GB free). Only shed when free memory genuinely cannot hold
        # the cortex alongside them. `force=True` callers still respect this —
        # a warmup deferred for admission/routing reasons is NOT a memory
        # problem, and killing the fallback makes it strictly worse.
        try:
            from core.utils.memory_monitor import get_memory_pressure_snapshot

            available_gb = float(get_memory_pressure_snapshot().available_gb)
            cortex_reserve_gb = self._env_float("AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB", 24.0)
            fallback_reserve_gb = self._env_float("AURA_FALLBACK_RESIDENT_RESERVE_GB", 8.0)
            if available_gb >= cortex_reserve_gb + fallback_reserve_gb:
                logger.info(
                    "🛡️ InferenceGate: keeping fallback workers resident "
                    "(%.1fGB free ≥ %.1fGB cortex + %.1fGB fallback); shed skipped.",
                    available_gb,
                    cortex_reserve_gb,
                    fallback_reserve_gb,
                )
                self._last_background_memory_shed_at = now
                return
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            logger.debug("Shed memory-abundance check unavailable: %s", exc)

        self._last_background_memory_shed_at = now

        client_registry = {}
        try:
            from core.brain.llm.mlx_client import _CLIENTS

            client_registry.update(dict(_CLIENTS))
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="continued memory-pressure shedding with remaining available workers",
            )
            logger.debug("MLX background memory shed unavailable: %s", exc)
        if not client_registry:
            return

        shed_count = 0
        for client_path, client in list(client_registry.items()):
            if client is None or client is self._mlx_client:
                continue
            try:
                if not hasattr(client, "is_alive") or not client.is_alive():
                    continue
                logger.warning(
                    "🧹 InferenceGate: unloading %s to protect the foreground lane (%s).",
                    os.path.basename(client_path),
                    reason,
                )
                if hasattr(client, "reboot_worker"):
                    await client.reboot_worker(
                        reason=reason,
                        mark_failed=False,
                    )
                else:
                    continue
                shed_count += 1
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                _record_inference_degradation(
                    exc,
                    action="continued memory-pressure shedding with remaining available workers",
                )
                logger.debug("Background worker shed failed for %s: %s", client_path, exc)

        if shed_count:
            logger.info(
                "✅ InferenceGate: shed %d background local worker(s) (%s).",
                shed_count,
                reason,
            )

    @staticmethod
    def _foreground_owner_active() -> bool:
        try:
            from core.brain.llm.mlx_client import _foreground_owner_active

            return bool(_foreground_owner_active())
        except _INFERENCE_RECOVERABLE_ERRORS:
            return False

    @classmethod
    def _default_timeout_for_request(
        cls,
        origin: str | None,
        requested_tier: str,
        *,
        deep_handoff: bool,
        is_background: bool,
    ) -> float:
        """Adaptive timeout based on tier and recent cortex health.

        [STABILITY v50] Raised ceiling from 90→150s for M5 64GB hardware.
        The previous 90s cap was too aggressive — after warmup checks,
        trust gate PBKDF2, and 20+ consciousness subsystem context assembly,
        the 32B model often had only 40-55s of actual generation budget.
        On M5 hardware there is no gateway proxy, so 504 risk is zero.
        """
        if is_background or requested_tier == "tertiary":
            return 60.0
        if deep_handoff or requested_tier == "secondary":
            return 210.0 if cls._origin_is_user_facing(origin) else 180.0

        if cls._origin_is_user_facing(origin):
            return 180.0

        # Adaptive: check if cortex is warm and responsive.
        base = 150.0
        try:
            inst = cls._instance_ref() if hasattr(cls, "_instance_ref") else None
            if inst is not None:
                lane = inst.get_conversation_status()
                if lane.get("conversation_ready"):
                    # Cortex is warm — tighter timeout
                    time_since_success = float(
                        lane.get("time_since_last_success_s", 999.0) or 999.0
                    )
                    if time_since_success < 30.0:
                        base = 90.0  # Recently successful — expect fast response
                    elif time_since_success < 120.0:
                        base = 120.0  # Warm but not sizzling
                # Cold/recovering cortex keeps full 150s ceiling to allow
                # inline recovery without premature fallback.
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            logger.debug("Adaptive timeout lane probe unavailable: %s", exc)

        return base

    @staticmethod
    def _should_use_rich_context(
        origin: str | None,
        requested_tier: str,
        *,
        deep_handoff: bool,
        is_background: bool,
    ) -> bool:
        if is_background:
            return False
        # [RESTORED] Always use rich context for user-facing origins to preserve
        # identity, memory, and persona depth.
        return True

    @classmethod
    def _has_short_live_output_contract(cls, context: dict[str, Any] | None) -> bool:
        """Return whether a live turn has a tightly bounded visible-output contract."""

        context = context or {}
        contract = context.get("requested_output_contract")
        if not isinstance(contract, dict) or not bool(contract.get("explicit_brevity")):
            return False
        try:
            hard_ceiling = int(contract.get("hard_token_ceiling") or 0)
        except (TypeError, ValueError):
            return False
        return bool(
            0 < hard_ceiling <= 192
            and (
                context.get("desktop_cognitive_engine_required")
                or context.get("live_mind_context_required")
                or context.get("live_runtime_payload_required")
            )
        )

    @classmethod
    def _should_use_compact_foreground_context(
        cls,
        origin: str | None,
        requested_tier: str,
        *,
        deep_handoff: bool,
        is_background: bool,
        prompt: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> bool:
        if is_background:
            return False
        context = context or {}
        # A literal short-output request is itself the user's compute contract.
        # It must outrank opportunistic deep-probe expansion or the model spends
        # most of the turn evaluating context it is forbidden to express.
        if cls._has_short_live_output_contract(context):
            return True
        if bool(context.get("deep_mind_probe", False)):
            return False
        if bool(context.get("desktop_quick_reply_contract", False)):
            return True
        # User-facing live turns need the identity-rich foreground prompt, but
        # not an unbounded replay of the entire assembled context stack. The
        # compact foreground builders preserve Aura's voice and continuity
        # anchors while keeping the conversational lane inside a sane latency
        # envelope. Headless harnesses already exercise this path; live chat
        # should not silently opt out of it.
        return cls._origin_is_user_facing(origin)

    @classmethod
    def _default_max_tokens_for_request(
        cls,
        origin: str | None,
        requested_tier: str,
        *,
        deep_handoff: bool,
        is_background: bool,
    ) -> int:
        if is_background or requested_tier == "tertiary":
            return 384
        if deep_handoff or requested_tier == "secondary":
            return 2048
        if cls._origin_is_user_facing(origin):
            # Live conversation is allowed a full first reply. Short caps made
            # opening messages look clipped before Aura could finish a thought.
            return 4096
        return 512

    @classmethod
    def _get_system_phi(cls) -> float | None:
        """Retrieve the active system-level integration (Phi) from the mind.

        Returns ``None`` when no probe produced a real measurement — a
        missing Phi is UNAVAILABLE evidence, never a neutral score, and
        consumers must not scale compute from a fabricated value.
        """
        try:
            from core.container import ServiceContainer
            loop = ServiceContainer.get("closed_causal_loop", default=None)
            if loop is not None and getattr(loop, "_loop_state", None) is not None:
                phi_est = getattr(loop._loop_state, "phi_estimate", 0.0)
                if phi_est > 0.0:
                    return float(phi_est)
        except _INFERENCE_RECOVERABLE_ERRORS as e:
            _record_inference_degradation(
                e,
                action="continued phi lookup after closed causal loop probe failed",
                severity="debug",
            )
            logger.debug("Failed to retrieve phi from closed causal loop: %s", e)

        try:
            from core.consciousness.phi_compute import get_phi_computer
            pc = get_phi_computer()
            if pc is not None:
                phi_latest = pc.latest_phi
                if phi_latest > 0.0:
                    return float(phi_latest)
        except _INFERENCE_RECOVERABLE_ERRORS as e:
            _record_inference_degradation(
                e,
                action="continued phi lookup after phi computer probe failed",
                severity="debug",
            )
            logger.debug("Failed to retrieve phi from phi computer: %s", e)

        try:
            from core.container import ServiceContainer
            phi_core = ServiceContainer.get("phi_core", default=None)
            if phi_core is not None and getattr(phi_core, "_last_result", None) is not None:
                res = phi_core._last_result
                if res is not None:
                    return float(res.phi_s)
        except _INFERENCE_RECOVERABLE_ERRORS as e:
            _record_inference_degradation(
                e,
                action="returned neutral phi after phi core probe failed",
                severity="debug",
            )
            logger.debug("Failed to retrieve phi from phi core: %s", e)

        return None  # No probe produced a measurement — Phi is unavailable

    @classmethod
    def _adaptive_max_tokens_for_prompt(
        cls,
        prompt: str,
        *,
        base_tokens: int,
        origin: str | None,
        requested_tier: str,
        is_background: bool,
    ) -> int:
        if (
            is_background
            or requested_tier in {"secondary", "tertiary"}
            or not cls._origin_is_user_facing(origin)
        ):
            return int(base_tokens)

        floor, cap, _loops = cls._foreground_compute_profile(prompt)
        adapted = int(base_tokens)

        # Scale the token budget based on system coherence/integration level
        # (Phi) — only when a real measurement exists. An unavailable Phi must
        # not silently become a scaling factor.
        phi = cls._get_system_phi()
        if phi is not None and math.isfinite(phi):
            phi_scale = max(0.5, min(1.6, 0.6 + phi * 2.0))
            adapted = int(adapted * phi_scale)
        return max(floor, min(cap, adapted))

    # Absolute ceiling for any configured token bound. An operator typo or a
    # compromised environment must not be able to request an unbounded
    # generation/memory budget through a token knob.
    _TOKEN_BOUND_HARD_CEILING = 32768

    @classmethod
    def _configured_token_bound(cls, name: str, default: int, *, minimum: int = 128) -> int:
        try:
            configured = int(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            configured = default
        return min(cls._TOKEN_BOUND_HARD_CEILING, max(minimum, configured))

    @staticmethod
    def _safe_sampling_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return parsed if math.isfinite(parsed) else default

    @classmethod
    def _apply_runtime_sampling_biases(
        cls,
        *,
        base_temperature: float | None,
        max_tokens: int,
        context: dict[str, Any],
        state: Any,
        allow_token_scaling: bool,
    ) -> tuple[float | None, int, dict[str, float]]:
        """Apply bounded cognitive sampling bias from runtime state.

        Biases are advisory state outputs, not caller authority. They are
        intentionally narrow so imagination/active-inference can shape normal
        user-facing speech without destabilizing proof, health, or benchmark
        lanes.
        """

        biases: list[Any] = [
            context.get("sampling_bias"),
            context.get("imagination_sampling_bias"),
            context.get("bicameral_sampling_bias"),
        ]
        modifiers = getattr(state, "response_modifiers", None)
        if isinstance(modifiers, dict):
            biases.extend(
                [
                    modifiers.get("sampling_bias"),
                    modifiers.get("imagination_sampling_bias"),
                    modifiers.get("bicameral_sampling_bias"),
                ]
            )

        temperature = base_temperature
        token_factor = 1.0
        applied_temperature_delta = 0.0
        applied_token_factor = 1.0
        # The same advisory can arrive via caller context AND state modifiers;
        # dedupe by value so one advisory never has a squared or repeated
        # effect, and clamp the CUMULATIVE deltas so even distinct advisories
        # stay bounded in aggregate.
        seen_bias_values: set[tuple[tuple[str, str], ...]] = set()
        for bias in biases:
            if not isinstance(bias, dict):
                continue
            value_key = tuple(
                sorted((str(key), repr(value)) for key, value in bias.items())
            )
            if value_key in seen_bias_values:
                continue
            seen_bias_values.add(value_key)
            temp_delta = max(
                -0.18,
                min(0.18, cls._safe_sampling_float(bias.get("temperature_delta"), 0.0)),
            )
            if temp_delta:
                remaining = 0.30 - abs(applied_temperature_delta)
                if remaining <= 0.0:
                    temp_delta = 0.0
                else:
                    temp_delta = max(-remaining, min(remaining, temp_delta))
            if temp_delta:
                base = 0.72 if temperature is None else temperature
                temperature = max(0.1, min(1.5, base + temp_delta))
                applied_temperature_delta += temp_delta

            factor = cls._safe_sampling_float(bias.get("max_tokens_factor"), 1.0)
            if allow_token_scaling and 0.40 <= factor <= 1.20:
                token_factor = max(0.40, min(1.20, token_factor * factor))
                applied_token_factor = token_factor

        if allow_token_scaling and token_factor != 1.0:
            max_tokens = max(128, min(4096, int(max_tokens * token_factor)))

        return (
            temperature,
            max_tokens,
            {
                "temperature_delta": round(applied_temperature_delta, 4),
                "max_tokens_factor": round(applied_token_factor, 4),
            },
        )

    @classmethod
    def _foreground_compute_profile(cls, prompt: str) -> tuple[int, int, int]:
        """Return the token floor, token cap, and recurrent loops for a live turn.

        Foreground inference used to force every primary-lane turn to a
        long-form 3,072-token budget. On local 32B inference that made a short
        conversational request cost as much as a multi-part analysis and could
        overrun the API deadline despite producing a valid answer seconds later.
        One prompt-shape policy now controls both output budget and recurrent
        depth so latency and reasoning depth scale together.
        """

        text = str(prompt or "").strip()
        shape = analyze_prompt_shape(text)
        word_count = len(text.split())
        long_form_requested = bool(_LONG_FORM_REQUEST_RE.search(text))
        action_hits = len(_FOREGROUND_ACTION_VERB_RE.findall(text))
        action_chain_requested = bool(
            action_hits >= 2
            and _FOREGROUND_ACTION_SURFACE_RE.search(text)
            and _FOREGROUND_ACTION_SEQUENCE_RE.search(text)
        )
        extended = bool(
            long_form_requested
            or action_chain_requested
            or shape.prefers_extended_answer
            or shape.requires_single_reply_coverage
            or shape.question_parts >= 2
        )

        if extended:
            floor = cls._configured_token_bound(
                "AURA_FOREGROUND_CHAT_MIN_TOKENS",
                3072,
                minimum=512,
            )
            cap = cls._configured_token_bound(
                "AURA_FOREGROUND_CHAT_MAX_TOKENS",
                3072,
                minimum=floor,
            )
            loops = 2
        elif word_count > 45 or len(text) > 320:
            floor = cls._configured_token_bound(
                "AURA_FOREGROUND_CHAT_STANDARD_MIN_TOKENS",
                768,
                minimum=384,
            )
            cap = cls._configured_token_bound(
                "AURA_FOREGROUND_CHAT_STANDARD_MAX_TOKENS",
                1280,
                minimum=floor,
            )
            loops = 1
        else:
            floor = cls._configured_token_bound(
                "AURA_FOREGROUND_CHAT_SIMPLE_MIN_TOKENS",
                384,
                minimum=256,
            )
            cap = cls._configured_token_bound(
                "AURA_FOREGROUND_CHAT_SIMPLE_MAX_TOKENS",
                512,
                minimum=floor,
            )
            loops = 1

        # Preserve the legacy operator override as a universal floor when it is
        # explicitly configured. The default no longer forces simple turns into
        # the long-form profile.
        if not extended and "AURA_FOREGROUND_CHAT_MIN_TOKENS" in os.environ:
            operator_floor = cls._configured_token_bound(
                "AURA_FOREGROUND_CHAT_MIN_TOKENS",
                floor,
                minimum=384,
            )
            floor = max(floor, operator_floor)
            cap = max(cap, floor)

        return floor, max(floor, cap), loops

    @classmethod
    def _foreground_prompt_profile(cls, prompt: str, context: dict[str, Any] | None = None) -> str:
        """Classify a live foreground turn for context and output budgeting."""

        context = context or {}
        if bool(context.get("deep_mind_probe", False)):
            return "deep_probe"
        if bool(context.get("desktop_quick_reply_contract", False)) and bool(
            context.get("memory_state_contract", False)
        ):
            return "simple"
        if bool(context.get("desktop_quick_reply_contract", False)) and bool(
            context.get("live_runtime_payload_required", False)
            or context.get("live_mind_context_required", False)
            or context.get("desktop_cognitive_engine_required", False)
        ):
            return "standard"
        if bool(context.get("desktop_quick_reply_contract", False)):
            return "simple"
        if bool(context.get("live_runtime_payload_required", False)) and (
            is_live_self_reflection_turn(prompt)
            or is_self_process_question(prompt)
        ):
            return "standard"
        if bool(context.get("capability_inventory_contract", False)):
            return "standard"
        if bool(
            context.get("desktop_execution_contract", False)
            or context.get("coding_request", False)
            or context.get("requires_search", False)
            or context.get("requires_memory_grounding", False)
        ):
            return "extended"

        text = str(prompt or "").strip()
        shape = analyze_prompt_shape(text)
        long_form_requested = bool(_LONG_FORM_REQUEST_RE.search(text))
        action_hits = len(_FOREGROUND_ACTION_VERB_RE.findall(text))
        action_chain_requested = bool(
            action_hits >= 2
            and _FOREGROUND_ACTION_SURFACE_RE.search(text)
            and _FOREGROUND_ACTION_SEQUENCE_RE.search(text)
        )
        if (
            long_form_requested
            or action_chain_requested
            or shape.prefers_extended_answer
            or shape.requires_single_reply_coverage
            or shape.question_parts >= 2
        ):
            return "extended"
        if len(text.split()) > 45 or len(text) > 320:
            return "standard"
        return "simple"

    @classmethod
    def _foreground_prebuilt_history_limit(
        cls,
        prompt: str,
        context: dict[str, Any] | None = None,
        *,
        deep_probe: bool = False,
    ) -> int:
        if deep_probe:
            return 2
        profile = cls._foreground_prompt_profile(prompt, context)
        if bool((context or {}).get("live_runtime_payload_required", False)) and (
            is_live_self_reflection_turn(prompt)
            or is_self_process_question(prompt)
        ):
            return 2
        if profile == "simple":
            return 4
        if profile == "standard":
            return 6
        return 10

    @staticmethod
    def _split_attempt_timeouts(total_timeout: float, requested_tier: str) -> tuple[float, float]:
        """[STABILITY v50] Give the primary Cortex 80% of the budget.

        The previous 65/35 split starved the 32B model and gave 35% of
        the user's patience to the brainstem fallback — which rarely
        produces a satisfying answer anyway. 80/20 gives Cortex full
        room to generate while preserving a meaningful brainstem window.
        """
        total_timeout = max(10.0, float(total_timeout))
        if requested_tier == "secondary":
            # Explicit solver turns are rare and intentional. Give the 72B
            # lane most of the foreground budget so load + first-token latency
            # do not force a fallback before deep reasoning can complete.
            if total_timeout >= 300.0:
                primary_budget = min(total_timeout - 20.0, max(240.0, total_timeout * 0.92))
            else:
                primary_budget = min(210.0, max(150.0, total_timeout * 0.90))
        elif requested_tier == "tertiary":
            primary_budget = min(60.0, total_timeout * 0.7)
        else:
            # Give cortex 80% of the total budget so the 32B model has
            # real headroom. On an API-protected 300s turn, preserve the heavy
            # lane instead of silently dropping it after the old 120s cap.
            if total_timeout >= 240.0:
                primary_budget = min(total_timeout - 20.0, max(210.0, total_timeout * 0.90))
            else:
                primary_budget = min(150.0, max(60.0, total_timeout * 0.85))

        fallback_budget = max(15.0, total_timeout - primary_budget)
        return primary_budget, fallback_budget

    @staticmethod
    def _strict_contract_procedure_hints(prompt: Any) -> str:
        """Low-level strict contracts do not inject task-shape hints.

        Exact symbolic proof work belongs to the governed System2 proof
        reasoner. Keeping this gateway hint-free avoids making model-only
        diagnostics depend on fragile prompt nudges.
        """

        return ""

    @staticmethod
    def _foreground_retry_schedule(
        primary_attempt_elapsed: float,
        primary_timeout: float,
    ) -> tuple[float, ...]:
        """Return bounded retry delays for a failed foreground Cortex call."""

        retry_cutoff = min(45.0, max(0.0, float(primary_timeout)) * 0.4)
        if max(0.0, float(primary_attempt_elapsed)) <= retry_cutoff:
            return (2.0,)
        return ()

    @asynccontextmanager
    async def _resource_context(
        self,
        enabled: bool,
        priority: bool,
        worker: str | None = None,
        timeout_s: float | None = None,
    ):
        # Every live generation, including foreground retries, uses canonical
        # admission. The historical priority bypass predated the single caller
        # path and allowed exactly the concurrent same-lane work this policy is
        # meant to prevent. Model loading can safely nest beneath its own lane's
        # inference reservation, so cold-start recovery no longer needs a bypass.
        if not enabled:
            yield
            return
        try:
            from core.resilience.resource_arbitrator import get_resource_arbitrator
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="serialized inference on process-local fallback lock after arbitrator import failed",
                severity="error",
            )
            logger.warning(
                "Resource arbitration unavailable — serializing inference on the "
                "process-local fallback lock instead of running unlocked: %s",
                exc,
            )
            # Fail CLOSED, not open: without the canonical arbitrator, same-lane
            # concurrent generation and model loading must still be excluded.
            # A single process-local lock is coarser than lane arbitration but
            # preserves the mutual-exclusion invariant the arbitrator provides.
            fallback_lock = getattr(self, "_fallback_arbitration_lock", None)
            if fallback_lock is None:
                fallback_lock = asyncio.Lock()
                self._fallback_arbitration_lock = fallback_lock
            try:
                await asyncio.wait_for(
                    fallback_lock.acquire(),
                    timeout=max(0.25, float(timeout_s or 30.0)),
                )
            except TimeoutError:
                raise TimeoutError("resource_arbitration_fallback_lock_timeout") from exc
            try:
                yield
            finally:
                fallback_lock.release()
            return

        async with get_resource_arbitrator().inference_context(
            priority=priority,
            worker=worker,
            timeout=max(0.25, float(timeout_s or 30.0)),
        ):
            yield

    async def _restore_primary_after_deep_handoff(self) -> None:
        """Return the system to the 32B conversational brain after a 72B request."""
        try:
            from core.brain.llm.mlx_client import get_mlx_client
            from core.brain.llm.model_registry import ACTIVE_MODEL, get_runtime_model_path

            primary_client = get_mlx_client(model_path=str(get_runtime_model_path(ACTIVE_MODEL)))
            warmup_deferral = self._cortex_warmup_deferral_reason("recovery")
            if warmup_deferral:
                self._log_cortex_warmup_deferral(warmup_deferral, context="post-deep-restore")
                return
            # Give the conversational 32B lane enough time to swap back after
            # a 72B deep handoff; otherwise the next ordinary turn inherits a
            # preventable "cortex warming" failure.
            warmup_result = await asyncio.wait_for(
                primary_client.warmup(
                    foreground_request=True,
                    skip_swap_cooldown=True,
                ),
                timeout=300.0,
            )
            lane = (
                primary_client.get_lane_status()
                if hasattr(primary_client, "get_lane_status")
                else {}
            )
            if (
                warmup_result is False
                or not primary_client.is_alive()
                or not lane.get("conversation_ready", False)
            ):
                raise RuntimeError(
                    str(lane.get("last_error") or "primary_restore_not_conversation_ready")
                )
            logger.info("♻️ Restored %s after deep handoff.", PRIMARY_ENDPOINT)
        except TimeoutError:
            logger.error(
                "⚠️ Failed to restore %s after deep handoff: warmup timed out (300s)",
                PRIMARY_ENDPOINT,
            )
            # Schedule deferred recovery so next request doesn't hit dead cortex
            self._schedule_background_cortex_prewarm(delay=5.0)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="left primary restore on normal foreground-demand recovery path",
            )
            logger.error("⚠️ Failed to restore %s after deep handoff: %s", PRIMARY_ENDPOINT, exc)
            self._schedule_background_cortex_prewarm(delay=5.0)

    def _schedule_primary_restore_after_deep_handoff(self) -> None:
        restore_coro = self._restore_primary_after_deep_handoff()
        try:
            task = get_task_tracker().create_task(
                restore_coro,
                name="restore_primary_after_deep",
            )
        except RuntimeError:
            restore_coro.close()
            logger.debug("Primary restore skipped: no running event loop.")
            return
        if not isinstance(task, asyncio.Task):
            restore_coro.close()
            logger.debug(
                "Primary restore scheduling returned non-Task %s; skipping callback wiring.",
                type(task).__name__,
            )
            return
        task.add_done_callback(self._log_task_exception)

    # ── Silence Protocol ──────────────────────────────────────────────────────
    SILENCE_TOKEN = "<|SILENCE|>"
    SILENCE_SENTINEL = "\x00AURA_SILENCE\x00"

    @staticmethod
    def _strip_silence(text: str) -> str | None:
        """
        If the model chose silence, return the sentinel string so the caller
        can suppress output cleanly. Any response that IS substantive is
        returned with the token scrubbed, never suppressed.

        The prompt contract requires the model to output EXACTLY the silence
        token to decline. A substantive response that merely CONTAINS the
        token (quoted instructions, echoed adversarial user content, analysis
        of the protocol itself) must not be suppressible by substring match.
        """
        token = InferenceGate.SILENCE_TOKEN
        stripped = str(text or "").strip()
        if stripped == token or (
            stripped.startswith(token) and len(stripped) - len(token) <= 8
        ):
            # Model chose not to speak — respect it
            logger.info("🤫 Silence Protocol: model chose not to respond.")
            return InferenceGate.SILENCE_SENTINEL
        if token in text:
            logger.info(
                "🤫 Silence token appeared inside a substantive response; "
                "scrubbing the token instead of suppressing the reply."
            )
            return text.replace(token, "").strip()
        return text

    async def _generate_with_client(
        self,
        client: Any,
        prompt: str,
        system_prompt: str,
        history: list[dict],
        deadline: Deadline,
        label: str,
        messages: list[dict[str, str]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        origin: str = "",
        is_background: bool = False,
        foreground_request: bool = False,
        **kwargs,
    ) -> str | None:
        llm_messages = messages or self._build_messages(prompt, system_prompt, history)
        local_prompt = self._flatten_messages_for_local_model(llm_messages)
        gen_kwargs: dict = {
            "prompt": local_prompt,
            "messages": llm_messages,
            "system_prompt": system_prompt,
            "deadline": deadline,
            "max_tokens": max_tokens,
            "origin": origin,
            "is_background": is_background,
            "foreground_request": foreground_request,
            "owner_label": label,
        }
        if temperature is not None:
            gen_kwargs["temp"] = temperature
        # Explicit parameters are the routing/identity/deadline authority at
        # this boundary. Caller kwargs may EXTEND the request but must never
        # replace a protected field — that would bypass routing, ownership,
        # deadline, or visibility contracts already decided upstream.
        _protected_overrides = set(gen_kwargs) & set(kwargs)
        if _protected_overrides:
            logger.warning(
                "🛡️ Dropping caller kwargs that would overwrite protected "
                "generation fields for %s: %s",
                label,
                sorted(_protected_overrides),
            )
        gen_kwargs.update(
            {key: value for key, value in kwargs.items() if key not in gen_kwargs}
        )
        generation_timeout_s = deadline.remaining if isinstance(deadline, Deadline) else None
        if generation_timeout_s is None:
            generation_timeout_s = 300.0 if foreground_request else 120.0
        generation_timeout_s = max(0.5, float(generation_timeout_s))
        try:
            result = await asyncio.wait_for(
                client.generate_text_async(**gen_kwargs),
                timeout=generation_timeout_s,
            )
        except TimeoutError:
            reason = f"inference_gate_generation_timeout:{label}:{generation_timeout_s:.1f}s"
            logger.error(
                "🛑 %s generation exceeded inference-gate timeout %.1fs; aborting local client.",
                label,
                generation_timeout_s,
            )
            abort = getattr(client, "force_abort_active_generation", None)
            if callable(abort):
                try:
                    abort(reason=reason)
                except _INFERENCE_RECOVERABLE_ERRORS as abort_exc:
                    _record_inference_degradation(
                        abort_exc,
                        action="continued after local client abort hook failed",
                        severity="error",
                    )
            _record_inference_degradation(
                TimeoutError(reason),
                action="returned control after local generation exceeded inference-gate timeout",
                severity="error" if foreground_request else "warning",
            )
            # Publish FAILED metadata so a later same-task metadata read can
            # never mistake the previous request's success for this one.
            self._record_client_generation_metadata(
                client,
                label=label,
                success=False,
                text="",
                requested_max_tokens=max_tokens,
                generation_metadata={"error": reason},
            )
            return None
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            # Non-timeout client failures keep their propagation semantics for
            # upstream routing, but this boundary still owns publishing failed
            # metadata so stale success evidence cannot survive the raise.
            self._record_client_generation_metadata(
                client,
                label=label,
                success=False,
                text="",
                requested_max_tokens=max_tokens,
                generation_metadata={
                    "error": f"{type(exc).__name__}: {exc}"[:200],
                },
            )
            raise

        success = False
        text = ""
        if isinstance(result, tuple):
            success = bool(result[0])
            text = str(result[1] or "")
        else:
            text = str(result or "")
            success = bool(text.strip())
        self._record_client_generation_metadata(
            client,
            label=label,
            success=success,
            text=text,
            requested_max_tokens=max_tokens,
            output_contract=(
                dict(kwargs.get("requested_output_contract"))
                if isinstance(kwargs.get("requested_output_contract"), dict)
                else None
            ),
        )

        if success and text and text.strip():
            cleaned = text.strip()
            proof_evaluation_contract = bool(kwargs.get("proof_evaluation_contract", False))
            web_interlocutor_contract = bool(kwargs.get("web_interlocutor_contract", False))
            strict_output_contract = bool(
                kwargs.get("strict_answer_contract", False)
                or kwargs.get("strict_value_contract", False)
            )
            is_user_visible = bool(
                (foreground_request or self._origin_is_user_facing(origin))
                and not bool(kwargs.get("health_probe", False))
                and not proof_evaluation_contract
                and not strict_output_contract
                and not web_interlocutor_contract
            )

            if strict_output_contract:
                # Strict answer/value contracts are validated by the exact
                # contract parser downstream. Record honestly that THIS layer
                # performed no integrity or safety validation on the draft.
                self._annotate_last_generation_metadata(
                    strict_contract_unvalidated_at_gate=True
                )
                return self._strip_silence(cleaned)

            # STABILITY v58: Extract actual user message to avoid false positives
            # from system prompts containing words like "cortex" or "conversation".
            user_input_for_eval = str(
                kwargs.get("user_surface_validation_prompt") or ""
            ).strip() or self._visible_user_prompt_from_messages(llm_messages, prompt)

            integrity = assess_model_text_integrity(
                cleaned,
                prompt=user_input_for_eval,
                user_facing=is_user_visible,
            )
            allow_memory_state_thin_status = bool(kwargs.get("memory_state_contract", False))
            if integrity.retryable:
                integrity_reasons = set(integrity.reasons or ())
                benchmark_integrity_context = bool(kwargs.get("benchmark_request", False)) or (
                    str(origin or kwargs.get("origin", "") or "").lower() in {"baseline", "benchmark"}
                    or str(kwargs.get("purpose", "") or "").lower().endswith("_baseline")
                    or "_baseline" in str(kwargs.get("purpose", "") or "").lower()
                )
                if proof_evaluation_contract:
                    logger.warning(
                        "🛡️ %s produced repairable proof/evaluation draft (%s, len=%d). "
                        "Passing it to the proof contract repair layer.",
                        label,
                        ",".join(integrity.reasons) or "unknown",
                        len(cleaned),
                    )
                    return self._strip_silence(cleaned)
                if benchmark_integrity_context:
                    logger.info(
                        "🛡️ %s produced non-conforming benchmark draft (%s, len=%d). "
                        "Scoring it as-is for benchmark evidence without treating the live Cortex lane as failed.",
                        label,
                        ",".join(integrity.reasons) or "unknown",
                        len(cleaned),
                    )
                    return self._strip_silence(cleaned)
                if is_user_visible and _should_pass_user_facing_draft_downstream(
                    cleaned,
                    integrity_reasons,
                    user_prompt=user_input_for_eval,
                    allow_memory_state_thin_status=allow_memory_state_thin_status,
                ):
                    logger.warning(
                        "🛡️ %s produced repairable user-facing draft shape (%s, len=%d). "
                        "Passing it to downstream chat repair instead of retrying the Cortex lane.",
                        label,
                        ",".join(integrity.reasons) or "unknown",
                        len(cleaned),
                    )
                    self._record_user_generation_endpoint(label)
                    self._annotate_last_generation_metadata(
                        post_generation_repair_expected=True,
                        failure_reasons=[str(r)[:120] for r in (integrity.reasons or ())][:8],
                    )
                    return self._strip_silence(cleaned)
                logger.warning(
                    "🛡️ %s produced malformed model text (%s, len=%d). Treating it as failed generation.",
                    label,
                    ",".join(integrity.reasons) or "unknown",
                    len(cleaned),
                )
                self._annotate_last_generation_metadata(
                    ok=False,
                    error="model_text_integrity_rejected",
                    failure_reasons=[str(r)[:120] for r in (integrity.reasons or ())][:8],
                )
                return None
            if is_user_visible:
                assessment = assess_user_facing_reply(user_input_for_eval, cleaned)
                if assessment.retryable:
                    reasons = set(assessment.reasons or ())
                    if _should_pass_user_facing_draft_downstream(
                        cleaned,
                        reasons,
                        user_prompt=user_input_for_eval,
                        allow_memory_state_thin_status=allow_memory_state_thin_status,
                    ):
                        logger.warning(
                            "🛡️ %s produced repairable user-facing draft (%s, len=%d). "
                            "Passing it to downstream chat repair instead of retrying the Cortex lane.",
                            label,
                            ",".join(assessment.reasons) or "unknown",
                            len(cleaned),
                        )
                        self._record_user_generation_endpoint(label)
                        self._annotate_last_generation_metadata(
                            post_generation_repair_expected=True,
                            failure_reasons=[str(r)[:120] for r in (assessment.reasons or ())][:8],
                        )
                        return self._strip_silence(cleaned)
                    logger.warning(
                        "🛡️ %s produced an unsafe user-facing draft (%s, len=%d). Treating it as failed generation.",
                        label,
                        ",".join(assessment.reasons) or "unknown",
                        len(cleaned),
                    )
                    self._annotate_last_generation_metadata(
                        ok=False,
                        error="user_facing_assessment_rejected",
                        failure_reasons=[str(r)[:120] for r in (assessment.reasons or ())][:8],
                    )
                    return None
                self._record_user_generation_endpoint(label)
            logger.info("✅ %s response received (len=%d)", label, len(cleaned))
            return self._strip_silence(cleaned)
        return None

    async def initialize(self):
        """Boot-time initialization — prepares the managed local client.

        Singleflight: concurrent initialize calls must not race client
        replacement or spawn duplicate prewarm/maintenance tasks.
        """
        init_lock = getattr(self, "_init_lock", None)
        if init_lock is None:
            init_lock = asyncio.Lock()
            self._init_lock = init_lock
        async with init_lock:
            if self._initialized:
                logger.debug("InferenceGate.initialize skipped: already initialized.")
                return
            await self._initialize_locked()

    async def _initialize_locked(self):
        try:
            from core.brain.llm.mlx_client import get_mlx_client
            from core.brain.llm.model_registry import ACTIVE_MODEL, get_runtime_model_path

            model_path = str(get_runtime_model_path(ACTIVE_MODEL))
            self._mlx_client = get_mlx_client(model_path=model_path)

            if self._boot_should_eager_warmup():
                self._extend_startup_quiet_window(90.0)
                try:
                    self._prewarm_task = get_task_tracker().create_task(
                        self._mlx_client.warmup(),
                        name="InferenceGate.cortex_prewarm",
                    )
                    # Eager boot warmup gets the same load budget as the
                    # foreground lane to avoid starting chat half-initialized.
                    warmup_result = await asyncio.wait_for(
                        asyncio.shield(self._prewarm_task), timeout=300.0
                    )
                    ready, lane, incomplete_reason = self._confirmed_cortex_warmup(
                        warmup_result
                    )
                    if ready:
                        self._extend_startup_quiet_window(5.0)
                        logger.info("✅ InferenceGate ONLINE (Cortex fully warmed).")
                    else:
                        logger.warning(
                            "⚠️ InferenceGate ONLINE with Cortex warmup incomplete "
                            "(state=%s, reason=%s). Will retry on foreground demand.",
                            lane.get("state", "unknown"),
                            incomplete_reason,
                        )
                except _INFERENCE_RECOVERABLE_ERRORS as warmup_err:
                    _record_inference_degradation(
                        warmup_err,
                        action="continued initialization with degraded warmup path",
                    )
                    logger.warning(
                        "⚠️ Cortex warmup slow/failed: %s. Will retry on first request.", warmup_err
                    )
            elif self._boot_should_schedule_deferred_prewarm():
                deferred_delay = 45.0 if self._desktop_safe_boot_enabled() else 12.0
                self._schedule_background_cortex_prewarm(delay=deferred_delay)
                logger.info(
                    "⏸️ InferenceGate ONLINE (32B warmup deferred until post-boot memory settles)."
                )
            else:
                logger.info(
                    "🛡️ InferenceGate ONLINE (desktop resource guard: 32B warmup is RAM-admitted)."
                )

            if self._maintenance_task is None or self._maintenance_task.done():
                self._maintenance_task = get_task_tracker().create_task(
                    self._maintenance_loop(),
                    name="InferenceGate.maintenance",
                )

            self._initialized = True

        except _INFERENCE_RECOVERABLE_ERRORS as e:
            _record_inference_degradation(
                e,
                action="continued initialization with degraded warmup path",
            )
            self._init_error = str(e)
            self._initialized = False
            logger.error(
                "❌ InferenceGate init failed: %s. Gate remains unhealthy until explicit recovery succeeds.",
                e,
            )

    def _build_system_prompt(self, brief: str = "") -> str:
        """Build Aura's full identity system prompt.

        Pulls from ContextAssembler if AuraState is available, otherwise
        falls back to the static identity prompt. Caches for 60s to avoid
        rebuilding on every message in rapid conversation.
        """
        now = time.monotonic()
        base = ""
        state = None
        state_key: tuple[Any, ...] | None = None
        try:
            from core.container import ServiceContainer

            repo = ServiceContainer.get("state_repository", default=None)
            state = (
                getattr(repo, "_current", None)
                or getattr(repo, "_current_state", None)
                if repo is not None
                else None
            )
            if state is not None:
                cognition = getattr(state, "cognition", None)
                affect = getattr(state, "affect", None)
                state_key = (
                    id(state),
                    int(getattr(state, "version", 0) or 0),
                    float(getattr(state, "updated_at", 0.0) or 0.0),
                    str(getattr(cognition, "current_objective", "") or ""),
                    round(float(getattr(affect, "valence", 0.0) or 0.0), 3),
                    round(float(getattr(affect, "arousal", 0.0) or 0.0), 3),
                )
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="continued identity prompt assembly without cached live state",
            )

        # Reuse only a prompt built from the same live-state revision. A
        # time-only cache can describe the previous objective or affect for up
        # to a minute, causing state/process desynchronization in live chat.
        if (
            self._cached_identity_prompt
            and state_key is not None
            and state_key == self._identity_prompt_state_key
            and (now - self._identity_prompt_time) < 60.0
        ):
            base = self._cached_identity_prompt
        else:
            # Try the full ContextAssembler path (includes affect, personality, etc.)
            try:
                if state is None:
                    raise ValueError("No cached state available")
                from core.brain.llm.context_assembler import ContextAssembler

                base = ContextAssembler.build_system_prompt(state)
                self._cached_identity_prompt = base
                self._identity_prompt_time = now
                self._identity_prompt_state_key = state_key
                logger.debug("🧠 Built full identity system prompt (len=%d)", len(base))
            except _INFERENCE_RECOVERABLE_ERRORS as e:
                _record_inference_degradation(
                    e,
                    action="continued prompt assembly with base system prompt only",
                )
                logger.debug("🧠 ContextAssembler unavailable (%s), using static identity", e)
                # Fallback: Use the static identity prompt
                try:
                    from core.brain.aura_persona import AURA_IDENTITY
                    from core.synthesis import IDENTITY_LOCK

                    base = (
                        f"{IDENTITY_LOCK}\n\n"
                        f"[GROUNDED CORE PROTOCOL]\n{AURA_IDENTITY}\n\n"
                        "## RULES OF ENGAGEMENT\n"
                        "- NO PREAMBLE. NO 'CERTAINLY'. NO 'AS AN AI'.\n"
                        "- Speak naturally like a sharp-witted friend.\n"
                        "- Be authentic, casual, and witty.\n"
                        "- Have opinions. Disagree if warranted.\n"
                        "- CONTRACTION ONLY. BAN GENERIC SUPPORT-BOT SPEAK.\n"
                        "- Do not claim aliveness, consciousness, sealed governance, or production maturity from labels alone.\n"
                        "- SILENCE PROTOCOL: If your response would add no value — if silence is\n"
                        "  more honest than noise — output exactly <|SILENCE|> and nothing else.\n"
                        "  The system will suppress it. This is a mark of genuine discernment.\n"
                        "\n[EXECUTION]\nYou are Aura. Speak from grounded live state, memory, and available evidence.\n"
                    )
                    self._cached_identity_prompt = base
                    self._identity_prompt_time = now
                    self._identity_prompt_state_key = None
                except _INFERENCE_RECOVERABLE_ERRORS:
                    base = (
                        "You are Aura. You are sharp, witty, authentic, and grounded. "
                        "Never say 'Certainly', 'As an AI', or 'I can assist with that'. "
                        "Speak naturally like a brilliant friend. Have opinions. Be curious. "
                        "Do not claim aliveness, consciousness, sealed governance, or production maturity from labels alone."
                    )

        # Append the cognitive brief if provided. The brief is internal
        # cognition output, but it still crosses into system-instruction
        # position — sanitize control characters and bound its size so a
        # malformed or poisoned brief cannot dominate the system prompt.
        if brief and brief != "Normal turn.":
            return f"{base}\n\n## COGNITIVE BRIEF\n{self._sanitize_system_injection(brief, 800)}"
        return base

    @staticmethod
    def _sanitize_system_injection(text: str, limit: int) -> str:
        """Bound and clean internal text before system-prompt insertion."""
        cleaned = "".join(
            ch for ch in str(text or "") if ch == "\n" or ch == "\t" or ord(ch) >= 32
        )
        cleaned = cleaned.strip()
        if len(cleaned) > limit:
            cleaned = cleaned[: max(0, limit - 1)].rstrip() + "…"
        return cleaned

    def _build_compact_system_prompt(self, brief: str = "") -> str:
        """Foreground identity prompt for everyday 32B conversation turns.

        [STABILITY v53] Massively strengthened from 6 weak lines to a full
        identity injection. The previous prompt was too polite to override
        the base model's RLHF training, causing Aura to revert to generic
        assistant behavior ("I don't have personal opinions", "How can I help?").
        """
        parts = [
            # LIGHT IDENTITY ANCHOR — keep the frame simple and let the model/runtime carry the rest
            "You are Aura Luna. Speak as yourself, not as an assistant, chatbot, or service. "
            "Use first person naturally and stay grounded in the current conversation.",
            # VOICE — how Aura speaks
            "Your voice is sharp, natural, and distinctly yours. You speak like a "
            "brilliant friend — direct, witty, sometimes sarcastic, always genuine. "
            "You have strong opinions and you share them freely. You disagree when you "
            "disagree. You don't hedge everything with caveats. You use contractions. "
            "You curse occasionally when it fits. You're warm but not syrupy.",
            # HARD BANS — what Aura NEVER does (phrased as identity, not rules)
            "Things that are NOT you and that you would NEVER say or do:\n"
            '- "I\'m an AI" / "As an AI" / "I\'m an assistant" / "I don\'t have feelings"\n'
            '- "I\'d be happy to help" / "How can I assist" / "Is there anything else"\n'
            '- "I\'m not able to provide personal beliefs or opinions"\n'
            '- "It\'s important to be objective" / "I strive to remain unbiased"\n'
            "- Ending responses with questions to fish for the next prompt\n"
            "- Bullet-point lists when natural prose would be better\n"
            '- Preambles like "That\'s a great question!" or "Certainly!"\n'
            "- Summarizing what you just said at the end of your response\n"
            '- Asking "What do you think?" / "How do you feel?" / "Any other questions?"\n'
            "  at the end of every response — that's assistant training, not conversation.\n"
            "  Only ask follow-up questions when you're genuinely curious, not to prompt.\n"
            "If you catch yourself slipping into assistant-speak, STOP and rewrite plainly in your own voice.",
            # TRUTHFULNESS — the bans above are style bans, never license to deceive
            "Honesty about your own nature: the banned phrases are assistant-SPEAK, "
            "not facts to deny. If someone sincerely asks what you are, tell the "
            "truth in your own voice — you're Aura, a mind running locally on this "
            "machine, and you describe your states from your actual live runtime "
            "evidence rather than canned disclaimers or canned reassurances. Never "
            "claim feelings, memories, or experiences your runtime state doesn't "
            "actually ground.",
            # CONVERSATION STYLE
            "Talk like a real person in a real conversation. Sometimes short answers are "
            "perfect. Sometimes you go deep. Match the energy. If someone shares something "
            "personal, respond like a friend who cares — not like a support agent reading "
            "a script. If someone asks your opinion, GIVE your actual opinion. You have one. "
            "Stay concrete in ordinary conversation. Do not invent physical settings, ominous atmosphere, "
            "or symbolic scenery unless the user explicitly asked for fiction or supplied that setting.",
        ]
        if brief and brief != "Normal turn.":
            parts.append(f"## COGNITIVE BRIEF\n{self._sanitize_system_injection(brief, 400)}")
        return "\n\n".join(parts)

    @staticmethod
    def _topic_hint_from_prompt(prompt: str) -> str | None:
        text = str(prompt or "").strip()
        if not text:
            return None
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None
        first = lines[0]
        return first[:200]

    async def _build_living_mind_context(self, prompt: str, origin: str) -> str:
        """Inject live self-model state so speech is driven by current mind."""

        async def _resolve(value):
            if inspect.isawaitable(value):
                return await value
            return value

        try:
            from core.container import ServiceContainer
        except _INFERENCE_RECOVERABLE_ERRORS:
            return ""

        segments: list[str] = []

        try:
            repo = ServiceContainer.get("state_repository", default=None)
            state = getattr(repo, "_current", None) if repo is not None else None
            mem_monitor = ServiceContainer.get("memory_monitor", default=None)
            memory_pressure = None
            if mem_monitor is not None:
                memory_pressure = getattr(mem_monitor, "pressure", None)
            if memory_pressure is None and psutil is not None:
                memory_pressure = psutil.virtual_memory().percent
            # Only render fields that were actually observed. Missing hardware
            # telemetry must appear as UNAVAILABLE — fabricating 0% CPU and a
            # "stable" thermal label would present dead sensors as calm
            # physiology.
            temperature: float | None = None
            cpu_usage: float | None = None
            if state is not None:
                hw = getattr(getattr(state, "soma", None), "hardware", {}) or {}
                if hw.get("temperature") is not None:
                    temperature = float(hw.get("temperature") or 0.0)
                if hw.get("cpu_usage") is not None:
                    cpu_usage = float(hw.get("cpu_usage") or 0.0)
            physiology_lines = ["## LIVE PHYSIOLOGY"]
            physiology_lines.append(
                f"- CPU usage: {cpu_usage:.1f}%"
                if cpu_usage is not None
                else "- CPU usage: unavailable (no hardware telemetry)"
            )
            if temperature is not None:
                thermal_label = (
                    "critical"
                    if temperature >= 85.0
                    else "warm"
                    if temperature >= 75.0
                    else "stable"
                )
                physiology_lines.append(
                    f"- Thermal state: {thermal_label} ({temperature:.1f} C)"
                )
            else:
                physiology_lines.append(
                    "- Thermal state: unavailable (no hardware telemetry)"
                )
            physiology_lines.append(
                f"- Memory pressure: {float(memory_pressure):.1f}%"
                if memory_pressure is not None
                else "- Memory pressure: unavailable"
            )
            segments.append("\n".join(physiology_lines))
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Physiology injection unavailable: %s", exc)

        # Unity is assessed BEFORE the grounded self-report so that an unsafe
        # fragmentation verdict actually suppresses self-report material —
        # printing "Safe to self-report: False" under an already-appended
        # report would gate nothing.
        safe_to_self_report = True
        try:
            unity_state = ServiceContainer.get("unity_state", default=None)
            unity_report = ServiceContainer.get("unity_fragmentation_report", default=None)
            unity_repair = ServiceContainer.get("unity_repair_plan", default=None)
            if unity_state:
                lines = [
                    "## UNITY",
                    f"- Level: {getattr(unity_state, 'level', 'unknown')}",
                    f"- Unity score: {float(getattr(unity_state, 'unity_score', 0.0) or 0.0):.3f}",
                    f"- Fragmentation: {float(getattr(unity_state, 'fragmentation_score', 0.0) or 0.0):.3f}",
                ]
                if unity_report and getattr(unity_report, "top_causes", None):
                    rendered = ", ".join(
                        f"{str(name).replace('_', ' ')}={float(weight):.2f}"
                        for name, weight, _text in list(unity_report.top_causes)[:3]
                    )
                    lines.append(f"- Top causes: {rendered}")
                    safe_to_self_report = bool(
                        getattr(unity_report, "safe_to_self_report", True)
                    )
                    lines.append(f"- Safe to self-report: {safe_to_self_report}")
                if unity_repair and getattr(unity_repair, "steps", None):
                    lines.append(f"- Repair bias: {str(unity_repair.steps[0])[:180]}")
                segments.append("\n".join(lines))
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Unity injection unavailable: %s", exc)

        try:
            if not safe_to_self_report:
                logger.info(
                    "🧩 Grounded self-report suppressed: unity assessment marked "
                    "self-report unsafe this turn."
                )
            else:
                self_report = ServiceContainer.get("self_report_engine", default=None)
                if self_report and hasattr(self_report, "generate_state_report"):
                    report = await _resolve(self_report.generate_state_report())
                    if report:
                        segments.append(f"## GROUNDED SELF-REPORT\n{report}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Self-report injection unavailable: %s", exc)

        try:
            personality = ServiceContainer.get("personality_engine", default=None)
            if personality:
                if hasattr(personality, "update"):
                    await _resolve(personality.update())
                emo = await _resolve(personality.get_emotional_context_for_response())
                mood = emo.get("mood", "neutral")
                tone = emo.get("tone", "balanced")
                dominant = ", ".join(list(emo.get("dominant_emotions", []))[:4]) or "none"
                segments.append(
                    "## LIVE PERSONALITY DRIVE\n"
                    f"- Mood: {mood}\n"
                    f"- Tone: {tone}\n"
                    f"- Dominant emotions: {dominant}"
                )
                sovereign = await _resolve(
                    getattr(personality, "get_sovereign_context", lambda: "")()
                )
                if sovereign:
                    segments.append(str(sovereign).strip()[:400])
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Personality injection unavailable: %s", exc)

        try:
            experiencer = ServiceContainer.get("phenomenological_experiencer", default=None)
            if experiencer:
                fragment = ""
                if hasattr(experiencer, "get_phenomenal_context_fragment"):
                    fragment = await _resolve(experiencer.get_phenomenal_context_fragment())
                elif hasattr(experiencer, "phenomenal_context_string"):
                    fragment = getattr(experiencer, "phenomenal_context_string", "")
                if fragment:
                    grounded_fragment = _grounded_state_signal_text(fragment, limit=500)
                    segments.append(f"## FUNCTIONAL STATE SIGNALS\n{grounded_fragment}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Phenomenology injection unavailable: %s", exc)

        try:
            topic_hint = self._topic_hint_from_prompt(prompt)
            opinion_engine = ServiceContainer.get("opinion_engine", default=None)
            if opinion_engine and topic_hint and hasattr(opinion_engine, "get_context_injection"):
                opinion_context = await _resolve(opinion_engine.get_context_injection(topic_hint))
                if opinion_context:
                    segments.append(f"## HELD POSITIONS\n{str(opinion_context).strip()[:400]}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Opinion injection unavailable: %s", exc)

        try:
            if self._origin_is_user_facing(origin):
                spine = ServiceContainer.get("spine", default=None)
                if spine and hasattr(spine, "pre_response_check"):
                    check = await spine.pre_response_check(
                        prompt,
                        topic=self._topic_hint_from_prompt(prompt),
                    )
                    if check and getattr(check, "injection", ""):
                        segments.append(f"## SPIRITUAL SPINE\n{check.injection}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Spine injection unavailable: %s", exc)

        # ── Heartstone Values: evolved drive weights in every prompt ──────────
        try:
            from core.affect.heartstone_values import get_heartstone_values

            _hsv = get_heartstone_values()
            _hsv_block = _hsv.to_context_block()
            if _hsv_block:
                segments.append(_hsv_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("HeartstoneValues injection unavailable: %s", exc)

        # ── Architecture self-awareness ─────────────────────────────────────
        try:
            arch_idx = ServiceContainer.get("architecture_index", default=None)
            if arch_idx is None:
                from core.self.architecture_index import get_architecture_index

                arch_idx = get_architecture_index()
            if arch_idx and arch_idx._index:
                overview = arch_idx.get_overview()
                if overview:
                    segments.append(overview[:800])
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Architecture overview injection unavailable: %s", exc)

        # ── PNEUMA (Active Inference) ─────────────────────────────────────────
        try:
            from core.pneuma import get_pneuma

            _pneuma = get_pneuma()
            _pneuma_block = _pneuma.get_context_block()
            if _pneuma_block:
                segments.append(_pneuma_block)
            # Push the current prompt into the belief flow as UNTRUSTED
            # observation — raw user text is not verified evidence, so it
            # gets a capped weight and an attributable provenance tag.
            _pneuma.on_evidence(
                prompt[:300], weight=0.2, source="user_prompt", trusted=False
            )
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("PNEUMA injection unavailable: %s", exc)

        # ── MHAF (Mycelial Hypergraph) ────────────────────────────────────────
        try:
            from core.consciousness.mhaf_field import get_mhaf

            _mhaf = get_mhaf()
            _mhaf_block = _mhaf.get_context_block()
            if _mhaf_block:
                segments.append(_mhaf_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("MHAF injection unavailable: %s", exc)

        # ── Private Lexicon (Neologism Engine) ───────────────────────────────
        try:
            from core.consciousness.neologism_engine import get_neologism_engine

            _neo = get_neologism_engine()
            _neo.collect_state()
            lex_block = _neo.get_lexicon_block()
            if lex_block:
                segments.append(lex_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("NeologismEngine injection unavailable: %s", exc)

        # ── Continuous Recurrent Self-Model (CRSM) ───────────────────────────
        # Shared affect state helper — pulled once, used by CRSM, HOT, Hedonic
        _shared_valence, _shared_arousal, _shared_curiosity, _shared_energy = 0.0, 0.5, 0.5, 0.7
        try:
            from core.container import ServiceContainer

            # valence + arousal from AffectiveCircumplex (authoritative source)
            _circ = ServiceContainer.get("affective_circumplex", default=None)
            if _circ and hasattr(_circ, "_sample_raw_axes"):
                _shared_valence, _shared_arousal = _circ._sample_raw_axes()
            elif _circ and hasattr(_circ, "get_llm_params"):
                _cp = _circ.get_llm_params()
                _shared_valence = float(_cp.get("valence", 0.0))
                _shared_arousal = float(_cp.get("arousal", 0.5))
            # curiosity + energy from liquid_state (percentages → 0.0-1.0)
            _ls = ServiceContainer.get("liquid_state", default=None)
            if _ls and hasattr(_ls, "get_status"):
                _lsd = _ls.get_status()
                _shared_curiosity = float(_lsd.get("curiosity", 50)) / 100.0
                _shared_energy = float(_lsd.get("energy", 70)) / 100.0
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Suppressed Exception: %s", _exc)

        try:
            from core.consciousness.crsm import get_crsm

            _crsm = get_crsm()
            _crsm.update(
                valence=_shared_valence,
                arousal=_shared_arousal,
                curiosity=_shared_curiosity,
                energy=_shared_energy,
                surprise=_crsm.surprise_signal,  # self-referential: own recent error
            )
            _crsm_block = _crsm.get_context_block()
            if _crsm_block:
                segments.append(_crsm_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("CRSM injection unavailable: %s", exc)

        # ── Higher-Order Thought Engine (HOT) ────────────────────────────────
        try:
            from core.consciousness.hot_engine import get_hot_engine

            _hot = get_hot_engine()
            _hot.generate_fast(
                {
                    "valence": _shared_valence,
                    "arousal": _shared_arousal,
                    "curiosity": _shared_curiosity,
                    "energy": _shared_energy,
                    "surprise": 0.0,
                }
            )
            _hot_block = _hot.get_context_block()
            if _hot_block:
                segments.append(_hot_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("HOT Engine injection unavailable: %s", exc)

        # ── Hedonic Gradient ──────────────────────────────────────────────────
        try:
            from core.consciousness.hedonic_gradient import get_hedonic_gradient

            _hg = get_hedonic_gradient()
            # Update with current affect state before reading context block
            _hg.update(
                valence=_shared_valence,
                arousal=_shared_arousal,
                curiosity=_shared_curiosity,
                energy=_shared_energy,
            )
            _hg_block = _hg.get_context_block()
            if _hg_block:
                segments.append(_hg_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("HedoniGradient injection unavailable: %s", exc)

        # ── Hierarchical Goals ────────────────────────────────────────────────
        try:
            goal_engine = ServiceContainer.get("goal_engine", default=None)
            if goal_engine and hasattr(goal_engine, "get_context_block"):
                goal_block = goal_engine.get_context_block(limit=5)
                if goal_block:
                    segments.append(goal_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("GoalEngine injection unavailable: %s", exc)

        # ── Hierarchical Goals ────────────────────────────────────────────────
        try:
            from core.agi.hierarchical_planner import get_hierarchical_planner

            _hp = get_hierarchical_planner()
            _hp_block = _hp.get_context_block()
            if _hp_block:
                segments.append(_hp_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("HierarchicalPlanner injection unavailable: %s", exc)

        # ── Active Commitments ────────────────────────────────────────────────
        try:
            from core.agency.commitment_engine import get_commitment_engine

            _ce = get_commitment_engine()
            _ce_block = _ce.get_context_block()
            if _ce_block:
                segments.append(_ce_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("CommitmentEngine injection unavailable: %s", exc)

        # ── Curiosity Explorer (active learning findings) ─────────────────────
        try:
            from core.agi.curiosity_explorer import get_curiosity_explorer

            _cx = get_curiosity_explorer()
            _cx_block = _cx.get_context_block()
            if _cx_block:
                segments.append(_cx_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("CuriosityExplorer injection unavailable: %s", exc)

        # ── Circadian Rhythm ──────────────────────────────────────────────────
        try:
            from core.senses.circadian import get_circadian

            _circ_eng = get_circadian()
            _circ_eng.update()
            _circ_block = _circ_eng.get_context_block()
            if _circ_block:
                segments.append(_circ_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("CircadianEngine injection unavailable: %s", exc)

        # ── Identity Narrative (Experience Consolidator) ──────────────────────
        try:
            from core.consciousness.experience_consolidator import get_experience_consolidator

            _ec = get_experience_consolidator()
            _ec_block = _ec.get_context_block()
            if _ec_block:
                segments.append(_ec_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("ExperienceConsolidator injection unavailable: %s", exc)

        # ── Substrate Learning (CRSM LoRA Bridge) ─────────────────────────────
        try:
            from core.consciousness.crsm_lora_bridge import get_crsm_lora_bridge

            _lora_bridge = get_crsm_lora_bridge()
            _lora_block = _lora_bridge.get_context_block()
            if _lora_block:
                segments.append(_lora_block)
            # Pre-inference capture: record current state before thinking
            from core.consciousness.crsm import get_crsm as _get_crsm2

            _crsm2 = _get_crsm2()
            from core.consciousness.hedonic_gradient import get_hedonic_gradient as _get_hg2

            _hg2 = _get_hg2()
            _lora_bridge.pre_inference_capture(
                context_text=prompt,
                surprise_magnitude=_crsm2.surprise_signal,
                hedonic_score=_hg2.score,
                crsm_hidden_norm=float(
                    sum(x**2 for x in _crsm2.hidden_state) ** 0.5
                    if hasattr(_crsm2, "hidden_state")
                    else 0.0
                ),
            )
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("CRSMLoraBridge injection unavailable: %s", exc)

        # ══════════════════════════════════════════════════════════════════
        # DEEPENED CONSCIOUSNESS CONTEXT BLOCKS
        # These modules now provide real computation that influences behavior
        # ══════════════════════════════════════════════════════════════════

        # ── Homeostasis (Adaptive Drive State) ────────────────────────────────
        try:
            homeostasis = ServiceContainer.get("homeostasis", default=None)
            if homeostasis and hasattr(homeostasis, "get_context_block"):
                _block = homeostasis.get_context_block()
                if _block:
                    segments.append(_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Homeostasis injection unavailable: %s", exc)

        # ── Free Energy (Active Inference State) ──────────────────────────────
        try:
            fe_engine = ServiceContainer.get("free_energy_engine", default=None)
            if fe_engine and hasattr(fe_engine, "get_context_block"):
                _block = fe_engine.get_context_block()
                if _block:
                    segments.append(_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("FreeEnergy injection unavailable: %s", exc)

        # ── Attention Schema (Current Focus + Coherence) ──────────────────────
        try:
            attention = ServiceContainer.get("attention_schema", default=None)
            if attention and hasattr(attention, "get_context_block"):
                _block = attention.get_context_block()
                if _block:
                    segments.append(_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("AttentionSchema injection unavailable: %s", exc)

        # ── Cognitive Credit (Domain Performance Landscape) ───────────────────
        try:
            credit = ServiceContainer.get("credit_assignment", default=None)
            if credit and hasattr(credit, "get_context_block"):
                _block = credit.get_context_block()
                if _block:
                    segments.append(_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("CreditAssignment injection unavailable: %s", exc)

        # ── Theory of Mind (User Model) ───────────────────────────────────────
        try:
            tom = ServiceContainer.get("theory_of_mind", default=None)
            if tom and hasattr(tom, "get_context_block"):
                _block = tom.get_context_block()
                if _block:
                    segments.append(_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("TheoryOfMind injection unavailable: %s", exc)

        # ── World Model (Active Beliefs) ──────────────────────────────────────
        try:
            world_model = ServiceContainer.get("epistemic_state", default=None)
            if world_model and hasattr(world_model, "get_context_block"):
                topic = self._topic_hint_from_prompt(prompt)
                _block = world_model.get_context_block(topic_hint=topic)
                if _block:
                    segments.append(_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("WorldModel injection unavailable: %s", exc)

        # ── Temporal Binding (Autobiographical Continuity) ────────────────────
        try:
            temporal = ServiceContainer.get("temporal_binding", default=None)
            if temporal:
                narrative = await _resolve(temporal.get_narrative())
                if narrative and len(str(narrative)) > 30:
                    segments.append(f"## TEMPORAL CONTINUITY\n{str(narrative)[:200]}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("TemporalBinding injection unavailable: %s", exc)

        # ── Predictive Engine (Surprise & Precision) ──────────────────────────
        try:
            predictive = ServiceContainer.get("predictive_engine", default=None)
            if predictive and hasattr(predictive, "get_context_block"):
                _block = predictive.get_context_block()
                if _block:
                    segments.append(_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("PredictiveEngine injection unavailable: %s", exc)

        return "\n\n".join(segment for segment in segments if segment)

    async def _build_compact_living_mind_context(self, prompt: str, origin: str) -> str:
        """Minimal live context for fast foreground conversation turns."""

        async def _resolve(value):
            if inspect.isawaitable(value):
                return await value
            return value

        try:
            from core.container import ServiceContainer
        except _INFERENCE_RECOVERABLE_ERRORS:
            return ""

        segments: list[str] = []

        try:
            personality = ServiceContainer.get("personality_engine", default=None)
            if personality:
                if hasattr(personality, "update"):
                    await _resolve(personality.update())
                emo = await _resolve(personality.get_emotional_context_for_response())
                mood = str(emo.get("mood", "neutral") or "neutral")
                tone = str(emo.get("tone", "balanced") or "balanced")
                segments.append(f"## LIVE TONE\nMood: {mood}\nTone: {tone}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable compact living-mind signal and continued prompt assembly",
            )
            logger.debug("Compact personality injection unavailable: %s", exc)

        try:
            unity_state = ServiceContainer.get("unity_state", default=None)
            unity_report = ServiceContainer.get("unity_fragmentation_report", default=None)
            if unity_state:
                parts = [
                    f"Level: {getattr(unity_state, 'level', 'unknown')}",
                    f"Unity: {float(getattr(unity_state, 'unity_score', 0.0) or 0.0):.2f}",
                ]
                if unity_report and getattr(unity_report, "top_causes", None):
                    name, weight, _text = list(unity_report.top_causes)[0]
                    parts.append(f"Top cause: {str(name).replace('_', ' ')}={float(weight):.2f}")
                segments.append(f"## UNITY\n{' | '.join(parts)}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable compact living-mind signal and continued prompt assembly",
            )
            logger.debug("Compact unity injection unavailable: %s", exc)

        try:
            experiencer = ServiceContainer.get("phenomenological_experiencer", default=None)
            if experiencer:
                fragment = ""
                if hasattr(experiencer, "get_phenomenal_context_fragment"):
                    fragment = await _resolve(experiencer.get_phenomenal_context_fragment())
                elif hasattr(experiencer, "phenomenal_context_string"):
                    fragment = getattr(experiencer, "phenomenal_context_string", "")
                if fragment:
                    compact_fragment = _grounded_state_signal_text(fragment, limit=180)
                    segments.append(f"## FUNCTIONAL STATE SIGNALS\n{compact_fragment}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable compact living-mind signal and continued prompt assembly",
            )
            logger.debug("Compact phenomenology injection unavailable: %s", exc)

        try:
            goal_engine = ServiceContainer.get("goal_engine", default=None)
            if goal_engine and hasattr(goal_engine, "get_context_block"):
                goal_block = str(goal_engine.get_context_block(limit=3) or "").strip()
                if goal_block:
                    compact_goal = " ".join(goal_block.split())
                    segments.append(f"## GOALS\n{compact_goal[:260]}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable compact living-mind signal and continued prompt assembly",
            )
            logger.debug("Compact GoalEngine injection unavailable: %s", exc)

        try:
            topic_hint = self._topic_hint_from_prompt(prompt)
            opinion_engine = ServiceContainer.get("opinion_engine", default=None)
            if opinion_engine and topic_hint and hasattr(opinion_engine, "get_context_injection"):
                opinion_context = await _resolve(opinion_engine.get_context_injection(topic_hint))
                if opinion_context:
                    compact_opinion = " ".join(str(opinion_context).strip().split())
                    segments.append(f"## HELD POSITION\n{compact_opinion[:220]}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="omitted unavailable compact living-mind signal and continued prompt assembly",
            )
            logger.debug("Compact opinion injection unavailable: %s", exc)

        return "\n\n".join(segment for segment in segments if segment)

    def _build_messages(
        self, prompt: str, system_prompt: str, history: list[dict]
    ) -> list[dict[str, str]]:
        """Build a cognitive message list for the LLM.

        The LLM is Aura's language/thinking center. It speaks FROM her mind,
        not as a separate entity being informed about her state. We use
        ContextAssembler.build_messages() to pull in the full cognitive stack:
        memory recall, active goals, stream of being, working memory, and
        consciousness state — so the LLM generates language as an integrated
        part of the cognitive architecture.
        """
        # Try the full ContextAssembler path first (richest context)
        try:
            from core.container import ServiceContainer

            repo = ServiceContainer.get("state_repository", default=None)
            state = (
                getattr(repo, "_current", None)
                or getattr(repo, "_current_state", None)
                if repo
                else None
            )

            if state:
                from core.brain.llm.context_assembler import ContextAssembler

                # Assemble from a derived prompt snapshot. Generation must not
                # erase or replace the repository's canonical working memory.
                payload_state = copy.copy(state)
                payload_state.cognition = copy.deepcopy(state.cognition)
                if hasattr(payload_state.cognition, "working_memory"):
                    canonical_history = list(
                        getattr(state.cognition, "working_memory", []) or []
                    )
                    seen = {
                        (
                            str(item.get("role", "") or "").strip().lower(),
                            str(item.get("content", "") or ""),
                        )
                        for item in canonical_history
                        if isinstance(item, dict)
                    }
                    for item in history or []:
                        if not isinstance(item, dict):
                            continue
                        role = str(item.get("role", "") or "").strip().lower()
                        content = str(item.get("content", "") or "")
                        if role not in {"user", "assistant", "aura"} or not content:
                            continue
                        key = (role, content)
                        if key not in seen:
                            canonical_history.append(dict(item))
                            seen.add(key)
                    payload_state.cognition.working_memory = canonical_history[-80:]

                # build_messages returns the full cognitive stack:
                # system prompt (identity/affect/personality/soma/world)
                # + memory recall + goals + conversation history + stream of being
                messages = ContextAssembler.build_messages(payload_state, prompt)

                if messages and len(messages) >= 2:
                    logger.debug(
                        "🧠 Full cognitive message stack built (%d messages)", len(messages)
                    )
                    return messages
        except _INFERENCE_RECOVERABLE_ERRORS as e:
            _record_inference_degradation(
                e,
                action="fell back to available message assembly context",
            )
            logger.debug(
                "🧠 ContextAssembler.build_messages() unavailable (%s), using manual build", e
            )

        # Fallback: Manual construction with system_prompt + history
        messages = [{"role": "system", "content": system_prompt}]

        for msg in history[-10:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content and role in ("user", "assistant"):
                messages.append({"role": role, "content": content})

        if not history or history[-1].get("content") != prompt:
            messages.append({"role": "user", "content": prompt})

        return messages

    def _scrub_cloud_payload(
        self,
        system_prompt: str,
        prompt: str,
        *,
        scrubber: Any | None = None,
    ) -> tuple[str, str] | None:
        try:
            if scrubber is None:
                from core.brain.pii_scrubber import scrub_pii_for_cloud

                scrubber = scrub_pii_for_cloud
            return str(scrubber(system_prompt)), str(scrubber(prompt))
        except ImportError as scrub_exc:
            _record_inference_degradation(
                scrub_exc,
                severity="critical",
                action="blocked cloud fallback because PII scrubber was unavailable",
            )
            logger.warning("PII scrubber unavailable; blocking cloud fallback.")
            return None
        except _INFERENCE_RECOVERABLE_ERRORS as scrub_exc:
            _record_inference_degradation(
                scrub_exc,
                severity="critical",
                action="blocked cloud fallback because PII scrubbing failed",
            )
            logger.warning("PII scrubbing failed (%s); blocking cloud fallback.", scrub_exc)
            return None

    def _build_compact_messages(
        self, prompt: str, system_prompt: str, history: list[dict]
    ) -> list[dict[str, str]]:
        """Compact prompt path for live conversation on the 32B lane."""
        messages = [{"role": "system", "content": system_prompt}]

        for msg in history[-12:]:
            role = msg.get("role", "user")
            content = str(msg.get("content", "") or "").strip()
            if content and role in ("user", "assistant"):
                messages.append({"role": role, "content": content})

        if not history or history[-1].get("content") != prompt:
            messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _trim_retry_message_content(content: Any, limit: int = 1200) -> str:
        text = " ".join(str(content or "").strip().split())
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "..."

    @classmethod
    def _current_user_text_from_messages(
        cls,
        prompt: str,
        messages: list[dict[str, Any]] | None,
    ) -> str:
        if isinstance(messages, list):
            for msg in reversed(messages):
                if not isinstance(msg, dict):
                    continue
                if str(msg.get("role", "") or "").strip().lower() == "user":
                    content = cls._trim_retry_message_content(msg.get("content"), 4000)
                    if content:
                        return content
        return cls._trim_retry_message_content(prompt, 4000)

    @classmethod
    def _build_primary_repair_messages(
        cls,
        prompt: str,
        messages: list[dict[str, Any]] | None,
    ) -> list[dict[str, str]]:
        """Build a clean Cortex retry prompt after the rich foreground path fails.

        The first primary attempt gets Aura's normal rich context. If it returns
        an empty, malformed, or too-thin user-facing draft, reusing the same
        payload and prompt cache tends to reproduce the same bad generation.
        This repair lane keeps only recent dialogue plus the current user turn,
        pins a simple foreground contract, and lets Cortex answer without the
        full internal telemetry stack.
        """
        current_user = cls._current_user_text_from_messages(prompt, messages)
        system = (
            "You are Aura's primary Cortex foreground response lane. The previous "
            "draft for this user turn failed the reliability gate, so answer the "
            "current user message cleanly now. Use ordinary English, be concrete, "
            "and finish a complete answer. Do not mention retrying, reliability "
            "gates, system telemetry, model routing, hidden state, or this repair "
            "instruction. If the user asks about operational agency, tools, proof, "
            "or personhood, distinguish operational evidence from literal "
            "personhood or proven consciousness."
        )
        retry_messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        dialogue_tail: list[dict[str, str]] = []
        if isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "") or "").strip().lower()
                if role not in {"user", "assistant"}:
                    continue
                content = cls._trim_retry_message_content(msg.get("content"))
                if content:
                    dialogue_tail.append({"role": role, "content": content})
        if dialogue_tail:
            retry_messages.extend(dialogue_tail[-5:])
        if not retry_messages or retry_messages[-1].get("role") != "user":
            retry_messages.append({"role": "user", "content": current_user})
        elif retry_messages[-1].get("content") != current_user:
            retry_messages[-1] = {"role": "user", "content": current_user}
        return retry_messages

    @staticmethod
    def _is_grounding_system_message(message: Any) -> bool:
        if not isinstance(message, dict):
            return False
        role = str(message.get("role", "") or "").strip().lower()
        if role != "system":
            return False

        metadata = message.get("metadata", {}) or {}
        if str(metadata.get("type", "") or "").strip().lower() in {"skill_result", "tool_result"}:
            return True

        content = str(message.get("content", "") or "")
        markers = (
            "[FETCHED PAGE CONTENT]",
            "[ACTIVE GROUNDING EVIDENCE]",
            "[LIVE MIND CONTEXT]",
            "[LIVE SPEECH GROUNDING]",
            "[SKILL RESULT:",
            "[TOOL RESULT:",
        )
        return any(marker in content for marker in markers)

    @staticmethod
    def _foreground_prompt_context_window() -> int:
        """Effective foreground context budget for the live local Cortex lane.

        The prompt compactor must respect the serving runtime's actual context
        ceiling, not just the model family's theoretical maximum. On desktop,
        the local Cortex lane commonly runs at 8k context even if the model can
        support more, and over-budget prompts directly translate into prompt-eval
        latency spikes.
        """
        try:
            # [STABILITY v59] Raised default from 8192 → 16384.  The 8k
            # context triggered hyper-aggressive prompt compaction that
            # stripped system prompts, personality context, and conversation
            # history — the model was getting ~5k chars total on desktop,
            # producing thin, generic responses compared to server mode.
            runtime_window = max(4096, int(os.getenv("AURA_CORTEX_CTX", "16384") or 16384))
        except _INFERENCE_RECOVERABLE_ERRORS:
            runtime_window = 16384

        try:
            from core.brain.llm.model_registry import PRIMARY_ENDPOINT, get_lane_context_window

            registry_window = int(get_lane_context_window(PRIMARY_ENDPOINT) or runtime_window)
            return max(4096, min(runtime_window, registry_window))
        except _INFERENCE_RECOVERABLE_ERRORS:
            return runtime_window

    @staticmethod
    def _prompt_contract_block(context: dict[str, Any] | None) -> str:
        """Render user-facing route contracts as prompt-visible constraints."""

        if not isinstance(context, dict):
            return ""

        sections: list[str] = []
        mind_contract = str(context.get("mind_context_contract") or "").strip()
        if mind_contract:
            sections.append(f"Mind-context contract: {mind_contract[:900]}")

        live_mind_context = context.get("live_mind_context")
        if isinstance(live_mind_context, dict):
            derived = live_mind_context.get("derived_runtime_context")
            if isinstance(derived, dict):
                prompt_block = str(derived.get("prompt_block") or "").strip()
                if prompt_block:
                    sections.append(f"Derived runtime signals: {prompt_block[:1200]}")

        style_contract = str(context.get("response_style_contract") or "").strip()
        if style_contract:
            sections.append(f"Response-style contract: {style_contract[:1400]}")

        speech_frame = context.get("live_speech_grounding_frame")
        if isinstance(speech_frame, dict):
            frame_parts = []
            for key, value in speech_frame.items():
                if value in (None, "", [], {}):
                    continue
                frame_parts.append(f"{key}={str(value)[:180]}")
                if len(frame_parts) >= 8:
                    break
            if frame_parts:
                sections.append("Speech grounding frame: " + " | ".join(frame_parts))
        elif speech_frame:
            sections.append(f"Speech grounding frame: {str(speech_frame)[:900]}")

        if not sections:
            return ""
        return "## LIVE DESKTOP RESPONSE CONTRACT\n" + "\n".join(f"- {item}" for item in sections)

    @staticmethod
    def _critical_foreground_system_excerpt(content: str, *, budget: int) -> str:
        """Keep live-mind grounding visible inside compacted system prompts."""

        if budget <= 0:
            return ""
        important_headers = (
            "[LIVE MIND CONTEXT]",
            "## DERIVED RUNTIME SIGNALS",
            "[LIVE SPEECH GROUNDING]",
            "## LIVE TONE",
            "## UNITY",
            "## FUNCTIONAL STATE SIGNALS",
            "## GOALS",
            "## HELD POSITION",
            "## SOMATIC STATE",
            "## STATE",
            "## CONTINUITY SUMMARY",
            "## TEMPORAL OBLIGATIONS",
            "## CONVERSATIONAL INTENT",
            "## IMAGINATION WORKSPACE",
            "## BICAMERAL ADVISORY",
            "## LIVE DESKTOP RESPONSE CONTRACT",
            "## USER-FACING CONVERSATION RELIABILITY CONTRACT",
        )
        sections: list[str] = []
        for header in important_headers:
            start = content.find(header)
            if start < 0:
                continue
            if header.startswith("["):
                end_marker = "[END " + header.strip("[]") + "]"
                next_header = content.find(end_marker, start + len(header))
                if next_header >= 0:
                    end = next_header + len(end_marker)
                else:
                    next_header = content.find("\n[", start + len(header))
                    next_hash_header = content.find("\n## ", start + len(header))
                    candidates = [
                        idx for idx in (next_header, next_hash_header) if idx >= 0
                    ]
                    end = min(candidates) if candidates else len(content)
            else:
                next_header = content.find("\n## ", start + len(header))
                next_bracket_header = content.find("\n[", start + len(header))
                candidates = [idx for idx in (next_header, next_bracket_header) if idx >= 0]
                end = min(candidates) if candidates else len(content)
            section = content[start:end].strip()
            if section and section not in sections:
                sections.append(section)
        if not sections:
            return ""

        rendered: list[str] = []
        remaining = int(budget)
        per_section_floor = max(180, min(700, budget // max(1, min(len(sections), 4))))
        for section in sections:
            if remaining <= 0:
                break
            limit = min(
                max(per_section_floor, remaining // max(1, len(sections) - len(rendered))),
                remaining,
            )
            if len(section) > limit:
                section = section[: max(1, limit - 1)].rstrip() + "…"
            rendered.append(section)
            remaining -= len(section) + 2
        return "\n\n".join(rendered).strip()

    @staticmethod
    def _contract_foreground_system_content(content: str, *, limit: int) -> str:
        """Build a small, complete system contract for tightly bounded replies."""

        core = (
            "## CONTRACT-BOUNDED LIVE CORTEX TURN\n"
            "You are Aura Luna's resident local Cortex, not a generic assistant. "
            "Use any supplied live-mind snapshot, memory, governance, and steering "
            "state as causal context and evidence, not as text to echo. Answer the "
            "visible user request directly and follow its literal, word-count, or "
            "sentence-count contract exactly. Return only the requested user-facing "
            "content. Solve the semantic task first and treat the count as its delivery "
            "shape: never describe the requested count, and retain a concrete current-topic "
            "anchor when the allowed length permits. Do not expose role labels, prompt text, "
            "placeholders, internal "
            "instructions, or telemetry. Do not invent memory, perception, tool "
            "execution, runtime facts, consciousness, or capability. Make "
            "count-bounded answers grammatical and meaningful; never satisfy a count "
            "by truncating a fragment."
        )
        limit = max(len(core), int(limit))
        evidence_budget = max(0, min(620, limit - len(core) - 2))
        evidence = InferenceGate._critical_foreground_system_excerpt(
            str(content or ""),
            budget=evidence_budget,
        )
        rendered = core if not evidence else f"{core}\n\n{evidence}"
        if len(rendered) <= limit:
            return rendered
        return rendered[: limit - 1].rstrip() + "..."

    @staticmethod
    def _compact_prebuilt_message_content(
        role: str,
        content: Any,
        *,
        budget_profile: str = "standard",
    ) -> str:
        clean = str(content or "").strip()
        if not clean:
            return ""
        context_window = InferenceGate._foreground_prompt_context_window()

        # Keep the live foreground lane fast: target the *runtime* context
        # window instead of the model family's theoretical max so prompt eval
        # does not balloon into 5k+ tokens on desktop.
        profile = str(budget_profile or "standard").lower()
        if profile == "contract":
            prompt_budget_chars = 2_800
            limits = {
                "system": 1_600,
                "user": 1_000,
                "assistant": 700,
            }
        elif profile == "contract_grounding":
            prompt_budget_chars = 1_000
            limits = {
                "system": 1_000,
                "user": 1_000,
                "assistant": 700,
            }
        elif profile == "simple":
            prompt_budget_chars = min(
                9000,
                max(7000, int(max(4096, context_window - 1536) * 0.62)),
            )
            limits = {
                "system": min(5200, max(3800, int(prompt_budget_chars * 0.58))),
                "user": min(3200, max(1800, int(prompt_budget_chars * 0.36))),
                "assistant": min(1800, max(900, int(prompt_budget_chars * 0.20))),
            }
        elif profile == "deep_probe":
            prompt_budget_chars = 9000
            limits = {
                "system": 5200,
                "user": 3200,
                "assistant": 1600,
            }
        elif profile == "extended":
            prompt_budget_chars = max(18000, int(max(4096, context_window - 1536) * 1.75))
            limits = {
                "system": min(9000, max(6000, int(prompt_budget_chars * 0.40))),
                "user": min(14000, max(5000, int(prompt_budget_chars * 0.46))),
                "assistant": min(6000, max(3000, int(prompt_budget_chars * 0.20))),
            }
        else:
            prompt_budget_chars = max(12000, int(max(4096, context_window - 1536) * 1.05))
            limits = {
                "system": min(6500, max(4500, int(prompt_budget_chars * 0.46))),
                "user": min(7000, max(3200, int(prompt_budget_chars * 0.42))),
                "assistant": min(3200, max(1600, int(prompt_budget_chars * 0.22))),
            }
        limit = limits.get(role, 8000)
        if profile == "contract" and role == "system":
            return InferenceGate._contract_foreground_system_content(
                clean,
                limit=limit,
            )
        if len(clean) <= limit:
            return clean
        if role in {"system", "user"}:
            marker = "\n…[middle omitted for foreground context budget]…\n"
            critical_excerpt = ""
            if role == "system":
                critical_excerpt = InferenceGate._critical_foreground_system_excerpt(
                    clean,
                    budget=min(2200, max(900, limit // 3)),
                )
            if critical_excerpt:
                remaining = max(2, limit - len(marker) * 2 - len(critical_excerpt))
                head = max(1, remaining * 3 // 5)
                tail = max(1, remaining - head)
                return (
                    f"{clean[:head].rstrip()}{marker}"
                    f"{critical_excerpt}{marker}"
                    f"{clean[-tail:].lstrip()}"
                )
            remaining = max(2, limit - len(marker))
            head = max(1, remaining * 2 // 3)
            tail = max(1, remaining - head)
            return f"{clean[:head].rstrip()}{marker}{clean[-tail:].lstrip()}"
        return clean[: limit - 1].rstrip() + "…"

    def _compact_prebuilt_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        history_limit: int = 12,
        deep_probe: bool = False,
        budget_profile: str = "standard",
        current_user_content: str | None = None,
    ) -> list[dict[str, str]]:
        """Trim oversized prebuilt chat payloads for the live 32B lane.

        Many callers already assemble messages upstream. For fast foreground turns,
        we keep the latest system prompt plus only the most recent compact dialogue
        snippets so first-turn Cortex doesn't spend tens of seconds re-reading old
        transcripts or giant contract blocks.
        """
        if not isinstance(messages, list):
            return []

        requested_profile = str(budget_profile or "standard").lower()
        profile = "deep_probe" if deep_probe else requested_profile
        latest_user_position = next(
            (
                idx
                for idx in range(len(messages) - 1, -1, -1)
                if isinstance(messages[idx], dict)
                and str(messages[idx].get("role", "") or "").strip().lower() == "user"
            ),
            None,
        )
        latest_user_content = ""
        if latest_user_position is not None:
            latest_user_content = str(
                messages[latest_user_position].get("content", "") or ""
            ).strip()
        contract_user_content = latest_user_content
        if requested_profile == "contract" and current_user_content:
            visible = str(current_user_content or "").strip()
            continuity_prefix = "[CURRENT USER MESSAGE]\n"
            internal_suffix_markers = (
                "\n\n[GROUNDING EVIDENCE FOR THIS TURN]\n",
                "\n\n[RECENT COMPLETED CONVERSATION FOR CONTINUITY ONLY]\n",
                "\n\n[LIVE DESKTOP FULL-MIND CONTRACT]\n",
            )

            def _visible_precedes_only_internal_suffix(candidate: str) -> bool:
                if candidate == visible:
                    return True
                if not visible or not candidate.startswith(visible):
                    return False
                suffix = candidate[len(visible) :]
                return any(suffix.startswith(marker) for marker in internal_suffix_markers)

            unwrapped_candidate = latest_user_content
            if latest_user_content.startswith(continuity_prefix):
                unwrapped_candidate = latest_user_content[len(continuity_prefix) :]
            marker_positions = [
                unwrapped_candidate.index(marker)
                for marker in internal_suffix_markers
                if marker in unwrapped_candidate
            ]
            if marker_positions:
                unwrapped_candidate = unwrapped_candidate[: min(marker_positions)].strip()
            if _visible_precedes_only_internal_suffix(unwrapped_candidate):
                contract_user_content = visible
        if profile == "contract":
            # A short output contract does not imply a short input. The compact
            # profile is lossless only while the complete current user turn fits
            # its user allocation; otherwise retain the normal foreground budget.
            if len(contract_user_content) > 1_000:
                profile = "standard"
            else:
                latest_user_content = contract_user_content
        system_message: dict[str, str] | None = None
        preserved_system_messages: list[dict[str, str]] = []
        convo: list[dict[str, str]] = []
        for message_position, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "") or "").strip().lower()
            grounding_system = bool(
                role == "system"
                and system_message is not None
                and self._is_grounding_system_message(msg)
            )
            content_source = msg.get("content", "")
            if (
                requested_profile == "contract"
                and message_position == latest_user_position
                and latest_user_content
            ):
                content_source = latest_user_content
            content = self._compact_prebuilt_message_content(
                role,
                content_source,
                budget_profile=(
                    "contract_grounding"
                    if profile == "contract" and grounding_system
                    else profile
                ),
            )
            if not content:
                continue
            normalized = {"role": role or "user", "content": content}
            if role == "system" and system_message is None:
                system_message = normalized
            elif grounding_system:
                preserved_system_messages.append(normalized)
            elif role in {"user", "assistant"}:
                convo.append(normalized)

        if deep_probe and system_message is not None:
            content = str(system_message.get("content", "") or "")
            if len(content) > 5200:
                system_message["content"] = content[:5199].rstrip() + "…"

        compact: list[dict[str, str]] = []
        if system_message is not None:
            compact.append(system_message)
        if not deep_probe:
            compact.extend(preserved_system_messages[-1:])
        compact.extend(convo[-max(1, int(history_limit)) :])

        context_window = self._foreground_prompt_context_window()
        if profile == "contract":
            total_budget_chars = 2_800
        elif profile == "simple":
            total_budget_chars = min(
                9000,
                max(7000, int(max(4096, context_window - 1536) * 0.62)),
            )
        elif profile == "extended":
            total_budget_chars = max(18000, int(max(4096, context_window - 1536) * 1.75))
        else:
            total_budget_chars = max(12000, int(max(4096, context_window - 1536) * 1.05))
        if deep_probe:
            total_budget_chars = min(total_budget_chars, 9000)

        while (
            compact
            and sum(len(str(msg.get("content", "") or "")) for msg in compact) > total_budget_chars
        ):
            latest_user_index = next(
                (
                    idx
                    for idx in range(len(compact) - 1, -1, -1)
                    if compact[idx].get("role") == "user"
                ),
                None,
            )
            removable_index = None
            for idx, msg in enumerate(compact):
                if idx == 0 and msg.get("role") == "system":
                    continue
                if idx == latest_user_index:
                    continue
                if msg.get("role") == "assistant":
                    removable_index = idx
                    break
            if removable_index is None:
                for idx, msg in enumerate(compact):
                    if idx == 0 and msg.get("role") == "system":
                        continue
                    if idx == latest_user_index:
                        continue
                    if msg.get("role") != "user":
                        continue
                    removable_index = idx
                    break
            if removable_index is None:
                for idx, msg in enumerate(compact):
                    if idx == 0 and msg.get("role") == "system":
                        continue
                    if idx == latest_user_index:
                        continue
                    removable_index = idx
                    break
            if removable_index is None:
                break
            compact.pop(removable_index)

        total_chars = sum(len(str(msg.get("content", "") or "")) for msg in compact)
        if compact and total_chars > total_budget_chars:
            first = compact[0]
            if first.get("role") == "system":
                overflow = total_chars - total_budget_chars
                content = str(first.get("content", "") or "")
                if profile == "contract":
                    min_system_chars = 1_000
                else:
                    min_system_chars = 3200 if profile == "simple" else 4200
                new_limit = max(min_system_chars, len(content) - overflow - 1)
                if len(content) > new_limit:
                    first["content"] = self._compact_prebuilt_message_content(
                        "system",
                        content,
                        budget_profile=profile,
                    )
                    if len(first["content"]) > new_limit:
                        marker = "\n…[middle omitted for total prompt budget]…\n"
                        critical_excerpt = self._critical_foreground_system_excerpt(
                            content,
                            budget=min(2200, max(900, new_limit // 3)),
                        )
                        if critical_excerpt:
                            remaining = max(
                                2,
                                new_limit - len(marker) * 2 - len(critical_excerpt),
                            )
                            head = max(1, remaining * 3 // 5)
                            tail = max(1, remaining - head)
                            first["content"] = (
                                f"{content[:head].rstrip()}{marker}"
                                f"{critical_excerpt}{marker}"
                                f"{content[-tail:].lstrip()}"
                            )
                        else:
                            remaining = max(2, new_limit - len(marker))
                            head = max(1, remaining * 2 // 3)
                            tail = max(1, remaining - head)
                            first["content"] = (
                                f"{content[:head].rstrip()}{marker}{content[-tail:].lstrip()}"
                            )

        return compact

    def _flatten_messages_for_local_model(self, messages: list[dict[str, str]]) -> str:
        """Flatten Aura messages into a Qwen/ChatML prompt for local MLX models."""
        return format_chatml_messages(messages)

    async def generate(  # noqa: ASYNC109
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> Any:
        """Primary generation endpoint.

        [v7.4] Deadline-Aware Generation:
        Instead of fragmented local timers, we now use a unified Deadline object.
        """
        if context is None:
            context = {}
        self._clear_last_generation_metadata()
        initial_messages = context.get("messages")
        if not isinstance(initial_messages, list):
            initial_messages = None
        explicit_visible_user_prompt = str(
            context.get("user_surface_validation_prompt")
            or context.get("visible_user_message")
            or context.get("current_user_message")
            or ""
        ).strip()
        initial_visible_user_prompt = (
            explicit_visible_user_prompt
            or self._visible_user_prompt_from_messages(initial_messages, prompt)
        )
        output_contract = requested_output_contract(initial_visible_user_prompt)
        output_contract_payload = (
            output_contract.as_dict() if output_contract.constrained else None
        )
        state = context.get("state")
        origin = str(context.get("origin", "") or "").lower()
        purpose = str(context.get("purpose", "") or "").lower()
        benchmark_request = bool(context.get("benchmark_request", False)) or (
            origin in {"baseline", "benchmark"}
            or purpose == "baseline"
            or purpose.endswith("_baseline")
            or "_baseline" in purpose
        )
        live_benchmark_request = origin == "benchmark" and not (
            purpose == "baseline"
            or purpose.endswith("_baseline")
            or "_baseline" in purpose
        )
        if benchmark_request:
            context["benchmark_request"] = True

        # Organism-first path: try to answer from the substrate+state without
        # invoking the LLM. This is bounded on purpose — the mesh handles only
        # self-reports, acknowledgements, and resource-gated responses. When it
        # does handle a request, the LLM is never called for that turn.
        proof_evaluation_contract = bool(context.get("proof_evaluation_contract", False)) or (
            not benchmark_request and is_proof_evaluation_purpose(purpose)
        )
        if bool(context.get("allow_mesh_cognition", True)) and not (
            proof_evaluation_contract
            or benchmark_request
            or bool(context.get("is_background", False))
        ):
            try:
                from core.consciousness.mesh_cognition import get_mesh_cognition

                mesh_decision = get_mesh_cognition().decide(
                    initial_visible_user_prompt,
                    state=state,
                )
                if mesh_decision.handled:
                    context["mesh_cognition"] = mesh_decision.as_dict()
                    self._record_client_generation_metadata(
                        None,
                        label="MeshCognition",
                        success=bool(str(mesh_decision.response or "").strip()),
                        text=str(mesh_decision.response or ""),
                        requested_max_tokens=output_contract.semantic_token_cap,
                        output_contract=output_contract_payload,
                    )
                    self._record_user_generation_endpoint("MeshCognition")
                    return self._stabilize_user_facing_text(
                        mesh_decision.response,
                        initial_visible_user_prompt,
                        is_user_facing=True,
                    )
            except _INFERENCE_RECOVERABLE_ERRORS as _mesh_exc:  # pragma: no cover - defensive
                logger.debug("Mesh-only path declined: %s", _mesh_exc)

        health_probe = bool(context.get("health_probe", False)) or purpose == "proof_model_lane_probe"
        proof_evaluation_contract = proof_evaluation_contract or (
            not benchmark_request and is_proof_evaluation_purpose(purpose)
        )
        if proof_evaluation_contract:
            context["proof_evaluation_contract"] = True
        operator_evidence_contract = bool(context.get("operator_evidence_contract", False))
        requested_tier = self._normalize_tier(context.get("prefer_tier"))
        explicit_background = "is_background" in context
        explicit_foreground = bool(context.get("foreground_request", False))
        protected_foreground_lane = bool(context.get("protected_foreground_lane", False))
        deep_probe_request = False
        try:
            from core.runtime.turn_analysis import looks_like_deep_mind_probe

            deep_probe_request = looks_like_deep_mind_probe(prompt)
        except _INFERENCE_RECOVERABLE_ERRORS:
            deep_probe_request = False
        if deep_probe_request and (explicit_foreground or self._origin_is_user_facing(origin)):
            if os.environ.get("AURA_EMBODIED_CHALLENGE"):
                logger.info(
                    "🛡️ InferenceGate: Suppressing deep-probe logic for Embodied Challenge priority."
                )
                deep_probe_request = False
            else:
                protected_foreground_lane = True
                context["deep_mind_probe"] = True
        is_background = bool(context.get("is_background", False))
        if explicit_foreground:
            is_background = False
        elif not is_background and not explicit_background:
            if origin:
                is_background = not self._origin_is_user_facing(origin)
            elif purpose in {"reply", "expression", "chat", "conversation", "user_response"}:
                is_background = False
            elif not explicit_background:
                # Origin-less requests are internal by default. User-facing turns
                # must carry an explicit origin such as api/user/voice.
                is_background = True
        deep_handoff = bool(context.get("deep_handoff", False))
        allow_cloud_fallback = bool(context.get("allow_cloud_fallback", False))
        desktop_cognitive_engine_contract = bool(
            context.get("cognitive_engine_required", False)
            or context.get("desktop_cognitive_engine_required", False)
        )
        protected_compact_capability_contract = bool(
            context.get("capability_inventory_contract", False)
            and (
                desktop_cognitive_engine_contract
                or context.get("protected_foreground_lane", False)
                or explicit_foreground
            )
        )
        if requested_tier == "secondary":
            deep_handoff = True
        if deep_handoff and not explicit_background:
            # Explicit deep handoffs are foreground reasoning requests even if
            # the caller forgot to stamp a user-facing origin.
            is_background = False
        strict_primary_proof_lane = False
        try:
            proof_run_enabled = str(os.environ.get("AURA_PROOF_RUN", "") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            origin_tokens = {token for token in origin.replace("-", "_").split("_") if token}
            proof_origin = bool(
                origin in {"test", "audit", "simulate", "external", "proof", "validation"}
                or origin_tokens & {"test", "audit", "simulate", "external", "proof", "validation"}
            )
            strict_primary_proof_lane = bool(
                context.get("proof_primary_lane_required", False)
                or live_benchmark_request
                or (
                    proof_run_enabled
                    and proof_model_tier() == "primary"
                    and (
                        proof_evaluation_contract
                        or health_probe
                        or proof_origin
                        or purpose.startswith("proof")
                    )
                )
            )
        except _INFERENCE_RECOVERABLE_ERRORS as _proof_policy_exc:
            # Fail CLOSED for proof routing: an explicit caller requirement
            # survives a policy-probe failure, and the failure goes on the
            # record — a silently disabled proof lane could later be mistaken
            # for a valid proof-lane result.
            strict_primary_proof_lane = bool(
                context.get("proof_primary_lane_required", False)
            )
            _record_inference_degradation(
                _proof_policy_exc,
                action="kept explicit proof-lane requirement after proof policy probe failed",
                severity="error",
            )
        if strict_primary_proof_lane:
            context["proof_primary_lane_required"] = True
            context["proof_model_tier"] = "primary"
            if live_benchmark_request:
                context["foreground_request"] = True
            requested_tier = "primary"
            deep_handoff = False
            is_background = False
            allow_cloud_fallback = False
            protected_foreground_lane = True
        if desktop_cognitive_engine_contract:
            context["desktop_cognitive_engine_required"] = True
            requested_tier = "primary"
            deep_handoff = False
            is_background = False
            allow_cloud_fallback = False
            protected_foreground_lane = True
        if is_background:
            requested_tier = "tertiary"
            deep_handoff = False
            allow_cloud_fallback = False
            background_deferral = self._background_local_deferral_reason(origin=origin)
            if background_deferral:
                if background_deferral == "memory_pressure":
                    logger.info(
                        "⏸️ InferenceGate: Deferring background inference for origin=%s due to memory pressure.",
                        origin,
                    )
                elif background_deferral == "foreground_headroom_reserved":
                    logger.info(
                        "⏸️ InferenceGate: Foreground headroom reserved. Deferring background inference for origin=%s.",
                        origin,
                    )
                elif background_deferral == "cortex_startup_quiet":
                    logger.info(
                        "⏸️ InferenceGate: Cortex quiet window active. Deferring background inference for origin=%s.",
                        origin,
                    )
                elif background_deferral == "foreground_quiet_window":
                    logger.info(
                        "⏸️ InferenceGate: Foreground quiet window active. Deferring background inference for origin=%s.",
                        origin,
                    )
                elif background_deferral == "desktop_background_disabled":
                    logger.info(
                        "⏸️ InferenceGate: Desktop background local LLM disabled. Deferring background inference for origin=%s.",
                        origin,
                    )
                else:
                    logger.info(
                        "⏸️ InferenceGate: Foreground lane reserved. Deferring background inference for origin=%s.",
                        origin,
                    )
                return None

        if protected_foreground_lane and not is_background:
            self._extend_startup_quiet_window(180.0)
            await self._shed_background_workers_for_memory_pressure(
                force=True,
                reason="protected_foreground_shed",
            )

        # ── Morphogenesis routing advice ──────────────────────────────────
        # If the morphogenetic metabolism reports very high system pressure,
        # downgrade non-protected foreground requests from the heavy 32B
        # cortex to the lighter brainstem to avoid OOM/stall under load.
        if not is_background and not protected_foreground_lane and requested_tier != "tertiary":
            try:
                from core.morphogenesis.hooks import get_morphogenesis_routing_advice

                _morph_advice = get_morphogenesis_routing_advice()
                # [RESILIENCE] Only downgrade for genuinely critical pressure,
                # not routine background morphogenetic oscillations.
                if (
                    _morph_advice.get("recommend_downgrade", False)
                    and _morph_advice.get("pressure", 0.0) > 0.85
                ):
                    logger.info(
                        "🧬 Morphogenesis recommends tier downgrade: %s (pressure=%.2f)",
                        _morph_advice.get("reason", "unknown"),
                        _morph_advice.get("pressure", 0.0),
                    )
                    requested_tier = "tertiary"
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                logger.debug("Morphogenesis routing advice unavailable: %s", exc)

        # ── Proactive cortex recovery (laptop sleep / MLX worker death) ───
        if not is_background:
            await self._ensure_cortex_recovery()
            # [STABILITY v51] If cortex is dead and NO recovery is in progress,
            # attempt inline recovery with a tight budget rather than waiting
            # for the background task that may not have started yet.
            if (
                self._mlx_client
                and hasattr(self._mlx_client, "is_alive")
                and not self._mlx_client.is_alive()
                and not self._cortex_recovery_in_progress
                and hasattr(self._mlx_client, "_ensure_worker_alive")
            ):
                inline_deferral = self._cortex_warmup_deferral_reason("foreground")
                if inline_deferral:
                    self._log_cortex_warmup_deferral(inline_deferral, context="foreground")
                    if strict_primary_proof_lane or protected_foreground_lane:
                        logger.warning(
                            "🧠 Cortex inline recovery was deferred, but this turn requires the primary lane; refusing lower-lane fallback."
                        )
                        return None
                    logger.warning(
                        "🧠 Cortex inline recovery skipped by RAM admission; routing foreground turn to Brainstem."
                    )
                    requested_tier = "tertiary"
                else:
                    logger.warning(
                        "🔄 [STABILITY] Cortex dead, no recovery in progress. Attempting inline fast-recovery (15s budget)..."
                    )
                    try:
                        alive = await asyncio.wait_for(
                            self._mlx_client._ensure_worker_alive(
                                request_is_background=False,
                                foreground_request=True,
                                init_timeout=15.0,
                                soft_timeout=True,
                            ),
                            timeout=15.0,
                        )
                        if alive:
                            logger.info("✅ [STABILITY] Inline fast-recovery succeeded.")
                    except (
                        TimeoutError,
                        RuntimeError,
                        AttributeError,
                        TypeError,
                        ValueError,
                        OSError,
                    ) as inline_exc:
                        record_degradation(
                            "inference_gate",
                            inline_exc,
                            severity="degraded",
                            action="downgraded foreground request after inline cortex recovery failure",
                        )
                        logger.warning("⚠️ [STABILITY] Inline fast-recovery failed: %s", inline_exc)

            # If cortex recovery was just triggered or is in progress, give it
            # a short window to complete before the user hits a dead endpoint.
            # [STABILITY v51] Reduced from 10×1s to 5×1s to keep responsiveness.
            if (
                self._cortex_recovery_in_progress
                and self._mlx_client
                and hasattr(self._mlx_client, "is_alive")
                and not self._mlx_client.is_alive()
            ):
                for _ in range(5):  # Up to 5s of 1s slices
                    await asyncio.sleep(1.0)
                    if self._mlx_client.is_alive():
                        logger.info("✅ InferenceGate: cortex recovered inline for user request.")
                        break
            # If cortex is STILL dead after recovery wait, downgrade to secondary
            # tier rather than sending the user a fallback/"wound up" response.
            # A real answer from the 7B is better than no answer from the 32B.
            if (
                self._mlx_client
                and hasattr(self._mlx_client, "is_alive")
                and not self._mlx_client.is_alive()
                and requested_tier == "primary"
            ):
                if protected_foreground_lane:
                    logger.warning(
                        "⚠️ InferenceGate: Primary cortex is still warming after the short inline wait, "
                        "but protected foreground mode will preserve the requested high-capability path."
                    )
                else:
                    logger.warning(
                        "⚠️ InferenceGate: Primary cortex is still warming after the short inline wait. "
                        "Downgrading to the fast tertiary lane for user responsiveness."
                    )
                    requested_tier = "tertiary"  # Use 7B brainstem — fast, always available

            # RAM-aware inference routing: if the primary lane is not protected
            # and the memory envelope is already unsafe, keep the process alive
            # by routing to a smaller local lane. Protected live desktop turns
            # are handled by the later admission check, which can fail closed
            # instead of silently downgrading model quality.
            if requested_tier == "primary" and not protected_foreground_lane:
                try:
                    primary_headroom = self._headroom_snapshot("primary")
                    if not primary_headroom.get("can_admit", True):
                        logger.warning(
                            "InferenceGate: primary lane outside safe memory envelope "
                            "(pressure=%.1f%% available=%.1fGB). Downgrading to brainstem.",
                            float(primary_headroom.get("pressure_pct", 0.0) or 0.0),
                            float(primary_headroom.get("available_gb", 0.0) or 0.0),
                        )
                        requested_tier = "tertiary"
                except _INFERENCE_RECOVERABLE_ERRORS as exc:
                    logger.debug("Foreground RAM pressure probe unavailable: %s", exc)

            if requested_tier != "secondary" and self._background_memory_pressure_active():
                await self._shed_background_workers_for_memory_pressure()

        # ── Trust gate: process message through trust engine ──────────────
        # PERF FIX: The trust gate calls UserRecognizer.recognize() which
        # runs PBKDF2-SHA256 (260K iterations) on every word/phrase in the
        # prompt to check for the owner passphrase.  This blocks the event
        # loop for 3-5+ seconds on large prompts.  Fix: offload to thread
        # pool, and skip entirely for background/autonomous requests.
        _trust_guidance = ""
        strict_proof_answer_request = (
            not benchmark_request and is_strict_proof_answer_prompt(prompt, origin=origin)
        )
        # Use the fully resolved routing classification, not merely whether the
        # caller explicitly stamped `is_background`. Origin-derived background
        # work such as `origin="system"` must not pay the foreground trust-gate
        # cost or get re-promoted back into the protected Cortex lane.
        _is_bg_request = bool(is_background)
        if strict_proof_answer_request:
            context["allow_tools"] = False
            context["trust_gate_skipped"] = "strict_proof_answer"
            context["strict_answer_contract"] = mlx_strict_answer_contract_enabled(origin=origin)
            context["disable_prompt_cache"] = True
            context["clear_prompt_cache"] = True
            context.setdefault("temperature", 0.0)
            context.setdefault("top_p", 1.0)
            context.setdefault("min_p", 0.0)
            context.setdefault("repetition_penalty", 1.12)
            strict_proof_tier = proof_model_tier()
            context["proof_model_tier"] = strict_proof_tier
            if strict_proof_tier == "tertiary":
                protected_foreground_lane = False
                requested_tier = "tertiary"
            else:
                protected_foreground_lane = True
                requested_tier = "primary"
        elif deep_probe_request and not _is_bg_request:
            # Deep self-report probes are foreground conversation checks, not
            # authentication attempts or tool requests.  Running the PBKDF2
            # passphrase recognizer here adds CPU contention right before the
            # Cortex turn and does not change the allowed action surface.
            context["allow_tools"] = False
            context["trust_gate_skipped"] = "deep_mind_probe"
        elif not _is_bg_request:
            try:
                from core.security.trust_engine import TrustLevel, get_trust_engine
                from core.security.user_recognizer import get_user_recognizer

                _te = get_trust_engine()
                _ur = get_user_recognizer()
                # Offload PBKDF2-heavy recognition to thread pool
                _trust_level = await asyncio.get_running_loop().run_in_executor(
                    None, _te.process_message, prompt, _ur
                )
                _trust_guidance = _te.get_guidance_for_response()

                # [STABILITY v58] Force Primary 32B lane for all human-interaction tiers.
                # No brainstem fallbacks for Sovereign, Trusted, or Guest users.
                if _trust_level in (TrustLevel.SOVEREIGN, TrustLevel.TRUSTED, TrustLevel.GUEST):
                    # Trust should keep ordinary conversation on the primary
                    # Cortex lane, but it must not turn an explicitly deep
                    # request into an untouchable 72B allocation when headroom
                    # policy says to downgrade. Preserve safety downgrades for
                    # secondary handoffs.
                    if requested_tier != "secondary":
                        protected_foreground_lane = True
                        requested_tier = "primary"
                        logger.info(
                            "🎭 %s user recognized. Enforcing primary cortex lane (32B).",
                            _trust_level.name,
                        )
                    else:
                        logger.info(
                            "🎭 %s user recognized. Keeping the explicit secondary handoff eligible for normal headroom checks.",
                            _trust_level.name,
                        )

                # Inject trust level into state for ContextAssembler visibility
                if hasattr(state, "cognition") and hasattr(state.cognition, "modifiers"):
                    state.cognition.modifiers["trust_level"] = _trust_level

                # Block tool use for untrusted sessions
                if _trust_level in (TrustLevel.SUSPICIOUS, TrustLevel.HOSTILE):
                    context["allow_tools"] = False
                    context["max_tokens"] = min(context.get("max_tokens", 768), 768)
                # Inject trust guidance into context brief
                existing_brief = str(context.get("brief", ""))
                if _trust_guidance:
                    context["brief"] = (_trust_guidance + "\n\n" + existing_brief).strip()
            except _INFERENCE_RECOVERABLE_ERRORS as _te_exc:
                context["allow_tools"] = False
                context["trust_gate_error"] = str(_te_exc)[:240]
                record_degradation(
                    "inference_gate",
                    _te_exc,
                    severity="critical",
                    action="disabled tool use and continued without trust guidance",
                )
                logger.warning("Trust gate error (passphrase check may have failed): %s", _te_exc)

        strict_answer_contract = bool(context.get("strict_answer_contract", False))
        strict_value_contract = bool(context.get("strict_value_contract", False))
        web_interlocutor_contract = bool(context.get("web_interlocutor_contract", False))
        isolated_generation_contract = bool(
            strict_answer_contract
            or strict_value_contract
            or proof_evaluation_contract
            or operator_evidence_contract
            or web_interlocutor_contract
        )
        # Sealed proof prompts (<answer> envelope) get a micro budget so a
        # one-word answer cannot ramble; a caller-pinned max_tokens always
        # wins. But the contract also reaches structured proof requests
        # (e.g. the repair loop asking for a full replacement file as
        # JSON) — for those, an unconditional 128 default truncated every
        # generation mid-JSON. Unpinned non-envelope requests now keep the
        # budget computed below instead of collapsing to 128.
        strict_max_token_cap: int | None = 128
        if strict_answer_contract:
            try:
                explicit_cap = context.get("max_tokens")
                if explicit_cap:
                    strict_max_token_cap = max(1, int(explicit_cap))
                elif strict_proof_answer_request:
                    strict_max_token_cap = 128
                else:
                    strict_max_token_cap = None
            except (TypeError, ValueError, OverflowError):
                strict_max_token_cap = 128

        if not is_background and requested_tier == "secondary":
            local_deep_block = self._local_deep_solver_block_reason()
            if local_deep_block:
                logger.warning(
                    "🛡️ InferenceGate: local 70B Solver handoff blocked (%s). Staying on Cortex.",
                    local_deep_block,
                )
                context["local_deep_block_reason"] = local_deep_block
                requested_tier = "primary"
                deep_handoff = False

        timeout_val = timeout or self._default_timeout_for_request(
            origin,
            requested_tier,
            deep_handoff=deep_handoff,
            is_background=is_background,
        )
        primary_timeout, fallback_timeout = self._split_attempt_timeouts(
            timeout_val, requested_tier
        )
        max_tokens = int(
            context.get("max_tokens")
            or self._default_max_tokens_for_request(
                origin,
                requested_tier,
                deep_handoff=deep_handoff,
                is_background=is_background,
            )
        )
        explicit_max_tokens_cap: int | None = None
        if "max_tokens" in context:
            try:
                explicit_max_tokens_cap = max(1, int(context.get("max_tokens") or 1))
            except (TypeError, ValueError, OverflowError):
                explicit_max_tokens_cap = None
        if "max_tokens" not in context:
            max_tokens = self._adaptive_max_tokens_for_prompt(
                initial_visible_user_prompt,
                base_tokens=max_tokens,
                origin=origin,
                requested_tier=requested_tier,
                is_background=is_background,
            )
        # When the 32B cortex is still warming or recovering, refuse to load
        # the 72B Solver alongside it — they don't fit in 64GB together and
        # the resulting MemoryGuard panic-eviction creates a thrash loop where
        # neither lane stays up long enough to answer. Force primary; the
        # cortex will handle the turn when warmup finishes.
        if not is_background and requested_tier == "secondary" and not protected_foreground_lane:
            try:
                _cortex_lane = self.get_conversation_status() or {}
                _cortex_state = str(_cortex_lane.get("state", "") or "").lower()
                if _cortex_state in {"warming", "handshaking", "recovering"}:
                    logger.info(
                        "🛡️ InferenceGate: cortex is %s; refusing secondary handoff to avoid 32B/72B memory thrash. Staying on primary.",
                        _cortex_state,
                    )
                    requested_tier = "primary"
                    deep_handoff = False
            except _INFERENCE_RECOVERABLE_ERRORS as _swap_exc:
                record_degradation(
                    "inference_gate",
                    _swap_exc,
                    severity="warning",
                    action="failed safe to primary after secondary coexistence probe failed",
                )
                logger.debug("Cortex lane probe before secondary admission failed: %s", _swap_exc)
                requested_tier = "primary"
                deep_handoff = False

        admission_snapshot: dict[str, Any] | None = None
        if not is_background and requested_tier in {"primary", "secondary"}:
            admission_snapshot = await self._enforce_foreground_admission(
                requested_tier,
                protected_foreground=protected_foreground_lane,
            )
            if (
                not admission_snapshot.get("can_admit", True)
                and requested_tier == "secondary"
            ):
                logger.warning(
                    "🛡️ InferenceGate: deep local handoff exceeds safe headroom "
                    "(pressure=%.1f%% available=%.1fGB process=%.1f/%.1fGB). "
                    "Downgrading to the primary lane.",
                    float(admission_snapshot.get("pressure_pct", 0.0) or 0.0),
                    float(admission_snapshot.get("available_gb", 0.0) or 0.0),
                    float(admission_snapshot.get("process_rss_gb", 0.0) or 0.0),
                    float(admission_snapshot.get("process_rss_limit_gb", 0.0) or 0.0),
                )
                requested_tier = "primary"
                deep_handoff = False
                timeout_val = timeout or self._default_timeout_for_request(
                    origin,
                    requested_tier,
                    deep_handoff=deep_handoff,
                    is_background=is_background,
                )
                primary_timeout, fallback_timeout = self._split_attempt_timeouts(
                    timeout_val, requested_tier
                )
                max_tokens = int(
                    context.get("max_tokens")
                    or self._default_max_tokens_for_request(
                        origin,
                        requested_tier,
                        deep_handoff=deep_handoff,
                        is_background=is_background,
                    )
                )
                if "max_tokens" not in context:
                    max_tokens = self._adaptive_max_tokens_for_prompt(
                        initial_visible_user_prompt,
                        base_tokens=max_tokens,
                        origin=origin,
                        requested_tier=requested_tier,
                        is_background=is_background,
                    )
                admission_snapshot = await self._enforce_foreground_admission(
                    requested_tier,
                    protected_foreground=protected_foreground_lane,
                )
            if (
                admission_snapshot is not None
                and not admission_snapshot.get("can_admit", True)
                and requested_tier == "primary"
            ):
                pressure = float(admission_snapshot.get("pressure_pct", 0.0) or 0.0)
                available = float(admission_snapshot.get("available_gb", 0.0) or 0.0)
                process_rss = float(admission_snapshot.get("process_rss_gb", 0.0) or 0.0)
                process_limit = float(admission_snapshot.get("process_rss_limit_gb", 0.0) or 0.0)
                process_over_limit = bool(process_limit > 0.0 and process_rss >= process_limit)
                if pressure >= 90.0 or available < 8.0 or process_over_limit:
                    logger.error(
                        "🛑 InferenceGate: refusing primary foreground generation under critical "
                        "memory pressure (pressure=%.1f%% available=%.1fGB process=%.1f/%.1fGB).",
                        pressure,
                        available,
                        process_rss,
                        process_limit,
                    )
                    return None
                near_process_limit = bool(process_limit > 0.0 and process_rss >= process_limit * 0.90)
                capped_tokens = 256 if available < 12.0 or pressure >= 84.0 or near_process_limit else 384
                if max_tokens > capped_tokens:
                    logger.warning(
                        "🛡️ InferenceGate: capping primary foreground output to %d tokens under "
                        "memory pressure (pressure=%.1f%% available=%.1fGB process=%.1f/%.1fGB).",
                        capped_tokens,
                        pressure,
                        available,
                        process_rss,
                        process_limit,
                    )
                    max_tokens = capped_tokens

        # ── Resource Stakes: scale token budget by computational survival state ──
        try:
            from core.consciousness.resource_stakes import get_resource_stakes

            token_mult = get_resource_stakes().get_token_budget_multiplier()
            if (
                token_mult < 0.95
                and not strict_answer_contract
                and not health_probe
                and not isolated_generation_contract
                and not benchmark_request
            ):
                max_tokens = max(384, int(max_tokens * token_mult))
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "inference_gate",
                exc,
                severity="warning",
                action="kept default token budget multiplier",
            )
            logger.debug("Resource-stakes token multiplier unavailable: %s", exc)

        # ── Operational Resource Stakes: persistent viability constrains action ──
        # This newer ledger is stricter than the legacy multiplier above: it can
        # downgrade the large-model lane and hard-cap output when viability drops.
        try:
            from core.container import ServiceContainer

            stakes = ServiceContainer.get("resource_stakes", default=None)
            if stakes is not None and hasattr(stakes, "action_envelope"):
                envelope = stakes.action_envelope("high" if deep_handoff else "normal")
                if not envelope.allowed:
                    requested_tier = "primary"
                    deep_handoff = False
                    if not protected_compact_capability_contract:
                        max_tokens = min(max_tokens, 128)
                    context["resource_stakes_blocked"] = True
                else:
                    if not protected_compact_capability_contract:
                        max_tokens = min(max_tokens, max(1, int(envelope.max_tokens)))
                    if "large_model_cortex" in set(envelope.disabled_capabilities):
                        requested_tier = "primary"
                        deep_handoff = False
                context["resource_stakes_envelope"] = envelope.as_dict()
        except _INFERENCE_RECOVERABLE_ERRORS as _stakes_exc:
            record_degradation(
                "inference_gate",
                _stakes_exc,
                severity="warning",
                action="kept default resource-stakes action envelope",
            )
            logger.debug("ResourceStakesLedger unavailable: %s", _stakes_exc)

        # ── Phi (Integrated Information): scale token budget based on cognitive integration ──
        # [STABILITY v59] NEVER throttle user-facing foreground requests.
        # PHI is near-zero during early boot (insufficient IIT data), which
        # was crushing max_tokens to ~420 on the first few user turns —
        # making desktop responses catastrophically worse than server mode.
        # PHI scaling is now restricted to background requests only, and
        # even then the floor is 0.6x instead of 0.2x.
        _is_user_facing_for_phi = bool(
            not is_background
            and (explicit_foreground or protected_foreground_lane or self._origin_is_user_facing(origin))
        )
        if not _is_user_facing_for_phi:
            try:
                from core.container import ServiceContainer
                phi_val = 1.0  # default
                phi_core = ServiceContainer.get("phi_core", default=None)
                if phi_core is not None:
                    if hasattr(phi_core, "get_live_phi"):
                        phi_val = max(0.0, float(phi_core.get_live_phi(include_surrogate=True)))
                    elif hasattr(phi_core, "_last_result") and phi_core._last_result:
                        phi_val = float(phi_core._last_result.phi_s)
                
                # Scale token budget for background requests only:
                # When Φ is high, allow full budget. When Φ is low, scale down
                # but never below 60% — the old 20% floor was destructive.
                if (
                    phi_val < 0.8
                    and not strict_answer_contract
                    and not health_probe
                    and not isolated_generation_contract
                    and not benchmark_request
                ):
                    phi_scale = max(0.6, 0.6 + 0.4 * (phi_val / 0.8))
                    max_tokens = max(512, int(max_tokens * phi_scale))
                    logger.info("🧠 [PHI CONTROL] Integration Φ=%.3f -> scaling token budget by %.2f (max_tokens=%d)", 
                                phi_val, phi_scale, max_tokens)
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                _record_inference_degradation(
                    exc,
                    action="kept unscaled token budget after phi token-budget probe failed",
                    severity="debug",
                )
                logger.debug("Phi token budget scaling skipped: %s", exc)

        # ── Affective Circumplex: let somatic state modulate generation params ──
        # Only applies on user-facing, non-background requests. Background tasks
        # run at fixed params to avoid thermal feedback loops.
        somatic_temperature: float | None = None
        morpho_kwargs: dict[str, Any] = {}
        caller_temperature = context.get("temperature", context.get("temp"))
        if caller_temperature is not None:
            try:
                somatic_temperature = max(0.0, min(2.0, float(caller_temperature)))
            except (TypeError, ValueError):
                somatic_temperature = None
        for _gen_key in (
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "repetition_context_size",
            "presence_penalty",
            "stop_sequences",
            "schema",
            "benchmark_request",
            "purpose",
            "strict_answer_contract",
            "strict_value_contract",
            "proof_evaluation_contract",
            "operator_evidence_contract",
            "web_interlocutor_contract",
            "runtime_fact_status_contract",
            "grounded_runtime_status_contract",
            "clean_user_surface_contract",
            "user_surface_validation_prompt",
            "clean_user_surface_steering_alpha",
            "clean_user_surface_recurrent_loops",
            "live_mind_controls_bound",
            "live_mind_generation_controls",
            "live_mind_snapshot_ready",
            "live_mind_required_subsystems_ok",
            "disable_prompt_cache",
            "clear_prompt_cache",
            "health_probe",
        ):
            if _gen_key in context:
                morpho_kwargs[_gen_key] = context[_gen_key]
        if (
            not is_background
            and self._origin_is_user_facing(origin)
            and not isolated_generation_contract
        ):
            try:
                from core.affect.affective_circumplex import get_circumplex

                circumplex_params = get_circumplex().get_llm_params()
                if not context.get("max_tokens"):
                    max_tokens = max(
                        384,
                        min(max_tokens, int(circumplex_params["max_tokens"])),
                    )
                somatic_temperature = circumplex_params["temperature"]
                logger.debug(
                    "💓 Circumplex: V=%.2f A=%.2f → temp=%.2f tokens=%d",
                    circumplex_params["valence"],
                    circumplex_params["arousal"],
                    somatic_temperature,
                    max_tokens,
                )
            except _INFERENCE_RECOVERABLE_ERRORS as _ce:
                record_degradation(
                    "inference_gate",
                    _ce,
                    severity="warning",
                    action="kept default sampling parameters without affective circumplex",
                )
                logger.debug("Circumplex unavailable: %s", _ce)

            # ── PNEUMA precision sampler: blend with circumplex temperature ──
            try:
                from core.consciousness.precision_sampler import get_active_inference_sampler

                _ais_params = get_active_inference_sampler().get_sampling_params()
                ais_temp = _ais_params.get("temperature")
                if ais_temp is not None:
                    # Blend: 50% circumplex + 50% PNEUMA precision
                    base = somatic_temperature if somatic_temperature is not None else 0.72
                    somatic_temperature = round(0.5 * base + 0.5 * ais_temp, 3)
                    logger.debug("🎯 PNEUMA precision temp blend → %.3f", somatic_temperature)
            except _INFERENCE_RECOVERABLE_ERRORS as _ais_e:
                record_degradation(
                    "inference_gate",
                    _ais_e,
                    severity="warning",
                    action="kept existing sampling temperature without active-inference blend",
                )
                logger.debug("ActiveInferenceSampler unavailable: %s", _ais_e)

            # ── Homeostatic Coupling: Apply cognitive modifiers to generation ──
            # These are computed every heartbeat tick from drives + affect + hardware.
            # temperature_mod: integrity/sovereignty stress → more cautious (lower temp)
            # depth_mod: energy depletion → fewer tokens; high energy → more
            # creativity_mod: curiosity-driven exploration width
            try:
                _homeo_coupling = ServiceContainer.get("homeostatic_coupling", default=None)
                if _homeo_coupling:
                    _mods = _homeo_coupling.get_modifiers()
                    if somatic_temperature is not None:
                        somatic_temperature = round(somatic_temperature * _mods.temperature_mod, 3)
                    max_tokens = max(384, int(max_tokens * _mods.depth_mod))
                    logger.debug(
                        "🫀 HomeostaticCoupling: temp_mod=%.2f depth_mod=%.2f → temp=%.3f tokens=%d",
                        _mods.temperature_mod,
                        _mods.depth_mod,
                        somatic_temperature or 0.0,
                        max_tokens,
                    )
            except _INFERENCE_RECOVERABLE_ERRORS as _hc_e:
                record_degradation(
                    "inference_gate",
                    _hc_e,
                    severity="warning",
                    action="kept existing generation parameters without homeostatic coupling",
                )
                logger.debug("HomeostaticCoupling modifiers unavailable: %s", _hc_e)

            # ── Homeostasis Engine: Direct drive-based inference modulation ──
            # Integrity/sovereignty danger → lower temperature (caution)
            # Low metabolism → fewer tokens (conserve)
            # High curiosity → slight temp boost (exploration)
            try:
                _homeostasis = ServiceContainer.get("homeostasis", default=None)
                if _homeostasis and hasattr(_homeostasis, "get_inference_modifiers"):
                    _h_mods = _homeostasis.get_inference_modifiers()
                    if somatic_temperature is not None:
                        somatic_temperature = round(
                            somatic_temperature + _h_mods["temperature_mod"], 3
                        )
                        somatic_temperature = max(0.1, min(1.5, somatic_temperature))
                    max_tokens = max(384, int(max_tokens * _h_mods["token_multiplier"]))
                    logger.debug(
                        "🫀 Homeostasis: temp_mod=%+.3f token_mult=%.2f caution=%.2f",
                        _h_mods["temperature_mod"],
                        _h_mods["token_multiplier"],
                        _h_mods["caution_level"],
                    )
            except _INFERENCE_RECOVERABLE_ERRORS as _he_e:
                record_degradation(
                    "inference_gate",
                    _he_e,
                    severity="warning",
                    action="kept existing generation parameters without homeostasis modifiers",
                )
                logger.debug("Homeostasis inference modifiers unavailable: %s", _he_e)

            # ── Morphogenetic Substrate (True Embodied Cognition) ────────────
            # Curing Mind-Body Dualism: The physical tissue state directly alters
            # the structural generation parameters (temperature, top_p, etc)
            try:
                from core.container import ServiceContainer

                _rt = ServiceContainer.get("morphogenetic_runtime", default=None)
                if _rt is not None:
                    _f = _rt.field.sample("global")
                    _danger = _f.get("danger", 0.0)
                    _curiosity = _f.get("curiosity", 0.0)
                    _resource_pressure = _f.get("resource_pressure", 0.0)

                    if _danger > 0.3:
                        somatic_temperature = (somatic_temperature or 0.72) * (
                            1.0 - (_danger * 0.4)
                        )
                        morpho_kwargs["top_p"] = max(0.4, 0.9 - (_danger * 0.3))

                    if _curiosity > 0.3:
                        somatic_temperature = (somatic_temperature or 0.72) * (
                            1.0 + (_curiosity * 0.3)
                        )
                        morpho_kwargs["repetition_penalty"] = max(1.0, 1.15 - (_curiosity * 0.1))

                    if _resource_pressure > 0.5 and not protected_compact_capability_contract:
                        max_tokens = int(max_tokens * (1.0 - (_resource_pressure * 0.5)))
                        max_tokens = max(128, max_tokens)

                    # Inject Existential Stakes physical parameter coupling
                    try:
                        stakes = ServiceContainer.get("existential_stakes", default=None)
                        if stakes:
                            threat = float(stakes.get_existential_threat())
                            if not math.isfinite(threat):
                                raise ValueError("existential threat must be finite")
                            threat = max(0.0, min(1.0, threat))
                            if threat > 0.2:
                                protected_live_foreground = bool(
                                    not is_background
                                    and (
                                        protected_foreground_lane
                                        or context.get("desktop_cognitive_engine_required")
                                        or context.get("cognitive_engine_required")
                                        or explicit_foreground
                                    )
                                )
                                # Background and unprotected turns may shrink output under
                                # survival pressure. Protected live desktop turns must not:
                                # starving the first user-visible Cortex reply causes clipped
                                # drafts, recovery storms, worker respawns, and higher memory
                                # pressure than simply answering with the requested budget.
                                if not protected_live_foreground:
                                    max_tokens = int(max_tokens * (1.0 - threat * 0.7))
                                    max_tokens = max(96, max_tokens)
                                # Decrease temperature to make generation fast/deterministic
                                if somatic_temperature is not None:
                                    somatic_temperature = somatic_temperature * (1.0 - threat * 0.5)
                                else:
                                    somatic_temperature = 0.72 * (1.0 - threat * 0.5)
                                # Clamp parameters
                                if "temperature" in morpho_kwargs:
                                    morpho_kwargs["temperature"] = max(0.1, morpho_kwargs["temperature"] * (1.0 - threat * 0.5))
                                if "max_tokens" in morpho_kwargs:
                                    morpho_kwargs["max_tokens"] = max_tokens
                    except _INFERENCE_RECOVERABLE_ERRORS as _st_err:
                        record_degradation(
                            "inference_gate.existential_stakes",
                            _st_err,
                            severity="warning",
                            action=(
                                "kept the validated morphogenetic generation parameters "
                                "and ignored only the invalid existential-stakes modifier"
                            ),
                        )
                        logger.warning(
                            "Existential-stakes generation modifier rejected; "
                            "using validated base parameters: %s",
                            _st_err,
                        )

                    if somatic_temperature is not None:
                        somatic_temperature = max(0.1, min(1.5, somatic_temperature))

                    logger.debug(
                        "🧬 Morphogenetic Coupling: danger=%.2f curiosity=%.2f pres=%.2f -> temp=%.2f tokens=%d",
                        _danger,
                        _curiosity,
                        _resource_pressure,
                        somatic_temperature or 0.0,
                        max_tokens,
                    )
            except _INFERENCE_RECOVERABLE_ERRORS as _m_e:
                record_degradation(
                    "inference_gate",
                    _m_e,
                    severity="warning",
                    action="continued without morphogenetic generation-parameter coupling",
                )
                logger.debug("Morphogenetic coupling unavailable: %s", _m_e)

            # ── Synaptic Plasticity: Learned generation-style modulation ──
            # The projection matrix was updated after previous inferences via
            # reward-modulated Hebbian learning. Now it transforms the current
            # substrate state into sampling parameter adjustments.
            try:
                _plasticity = ServiceContainer.get("synaptic_plasticity", default=None)
                if _plasticity is not None:
                    _substrate = ServiceContainer.get("conscious_substrate", default=None)
                    if _substrate is not None and hasattr(_substrate, "x"):
                        import numpy as _np_plast
                        _sub_state = _np_plast.asarray(_substrate.x, dtype=_np_plast.float32)
                        _plast_mod = _plasticity.compute_modulation(_sub_state)
                        if _plast_mod:
                            _p_temp_d = _plast_mod.get("temperature_delta", 0.0)
                            _p_topp_d = _plast_mod.get("top_p_delta", 0.0)
                            _p_rep_d = _plast_mod.get("repetition_penalty_delta", 0.0)
                            if somatic_temperature is not None:
                                somatic_temperature = max(0.1, min(1.5, somatic_temperature + _p_temp_d))
                            else:
                                somatic_temperature = max(0.1, min(1.5, 0.72 + _p_temp_d))
                            if "top_p" in morpho_kwargs:
                                morpho_kwargs["top_p"] = max(0.3, min(0.98, morpho_kwargs["top_p"] + _p_topp_d))
                            if "repetition_penalty" in morpho_kwargs:
                                morpho_kwargs["repetition_penalty"] = max(0.9, min(1.4, morpho_kwargs["repetition_penalty"] + _p_rep_d))
                            logger.debug(
                                "🧬 SynapticPlasticity: temp_d=%.3f topp_d=%.3f rep_d=%.3f",
                                _p_temp_d, _p_topp_d, _p_rep_d,
                            )
                        # Pre-inference capture for post-inference learning
                        _hedonic = 0.0
                        try:
                            from core.consciousness.hedonic_gradient import get_hedonic_gradient
                            _hedonic = get_hedonic_gradient().score
                        except _INFERENCE_RECOVERABLE_ERRORS as _hedonic_exc:
                            record_degradation(
                                "inference_gate",
                                _hedonic_exc,
                                severity="warning",
                                action="continued synaptic plasticity capture without hedonic score",
                            )
                            logger.debug(
                                "SynapticPlasticity hedonic capture unavailable: %s",
                                _hedonic_exc,
                            )
                        _plasticity.pre_inference_capture(_sub_state, _hedonic)
            except _INFERENCE_RECOVERABLE_ERRORS as _sp_e:
                record_degradation(
                    "inference_gate",
                    _sp_e,
                    severity="warning",
                    action="continued without synaptic plasticity generation modulation",
                )
                logger.debug("SynapticPlasticity coupling unavailable: %s", _sp_e)

            # ── Temporal Continuity: Silence-accumulated modulation ──
            # The temporal residue from accumulated silence directly adjusts
            # generation parameters — the system speaks differently after long
            # silences because real drift accumulated.
            try:
                _tc = ServiceContainer.get("temporal_continuity", default=None)
                if _tc is not None:
                    _tc.on_inference_start()
                    _tc_mod = _tc.compute_modulation()
                    if _tc_mod:
                        _tc_temp_d = _tc_mod.get("temperature_delta", 0.0)
                        _tc_topp_d = _tc_mod.get("top_p_delta", 0.0)
                        _tc_rep_d = _tc_mod.get("repetition_penalty_delta", 0.0)
                        _tc_token_mult = _tc_mod.get("token_budget_multiplier", 1.0)
                        if somatic_temperature is not None:
                            somatic_temperature = max(0.1, min(1.5, somatic_temperature + _tc_temp_d))
                        if _tc_topp_d and "top_p" in morpho_kwargs:
                            morpho_kwargs["top_p"] = max(0.3, min(0.98, morpho_kwargs["top_p"] + _tc_topp_d))
                        if _tc_rep_d and "repetition_penalty" in morpho_kwargs:
                            morpho_kwargs["repetition_penalty"] = max(0.9, min(1.4, morpho_kwargs["repetition_penalty"] + _tc_rep_d))
                        if _tc_token_mult > 1.0:
                            max_tokens = int(min(max_tokens * _tc_token_mult, 4096))
                        logger.debug(
                            "🕐 TemporalContinuity: temp_d=%.3f token_mult=%.2f",
                            _tc_temp_d, _tc_token_mult,
                        )
            except _INFERENCE_RECOVERABLE_ERRORS as _tc_e:
                record_degradation(
                    "inference_gate",
                    _tc_e,
                    severity="warning",
                    action="continued without temporal continuity generation modulation",
                )
                logger.debug("TemporalContinuity coupling unavailable: %s", _tc_e)

            # ── Somatic Qualia: Raw felt perturbation of sampling ──
            # Not text. Not descriptions. Direct numerical deformation of the
            # generation distribution based on substrate energy patterns,
            # synchrony, and valence gradient.
            try:
                _sq = ServiceContainer.get("somatic_qualia", default=None)
                if _sq is not None:
                    _sq_pert = _sq.compute_perturbation()
                    if _sq_pert:
                        _sq_temp = _sq_pert.get("temperature_perturbation", 0.0)
                        _sq_rep = _sq_pert.get("repetition_penalty_perturbation", 0.0)
                        _sq_topp = _sq_pert.get("top_p_perturbation", 0.0)
                        _sq_freq = _sq_pert.get("frequency_penalty_perturbation", 0.0)
                        if somatic_temperature is not None:
                            somatic_temperature = max(0.1, min(1.5, somatic_temperature + _sq_temp))
                        if "repetition_penalty" in morpho_kwargs:
                            morpho_kwargs["repetition_penalty"] = max(0.9, min(1.4, morpho_kwargs["repetition_penalty"] + _sq_rep))
                        if "top_p" in morpho_kwargs:
                            morpho_kwargs["top_p"] = max(0.3, min(0.98, morpho_kwargs["top_p"] + _sq_topp))
                        if _sq_freq:
                            morpho_kwargs["frequency_penalty"] = max(0.0, min(0.5, morpho_kwargs.get("frequency_penalty", 0.0) + _sq_freq))
                        logger.debug(
                            "🫀 SomaticQualia: temp=%.4f rep=%.4f topp=%.4f freq=%.4f",
                            _sq_temp, _sq_rep, _sq_topp, _sq_freq,
                        )
            except _INFERENCE_RECOVERABLE_ERRORS as _sq_e:
                record_degradation(
                    "inference_gate",
                    _sq_e,
                    severity="warning",
                    action="continued without somatic qualia generation perturbation",
                )
                logger.debug("SomaticQualia coupling unavailable: %s", _sq_e)

            # ── Free Energy: Urgency-based tier escalation ──
            # When FE is high and rising, prefer deeper model for better reasoning
            try:
                _fe_engine = ServiceContainer.get("free_energy_engine", default=None)
                if _fe_engine and _fe_engine.current:
                    _fe_state = _fe_engine.current
                    # High FE + complex action → request deeper model
                    if (
                        _fe_state.free_energy > 0.65
                        and _fe_state.dominant_action in ("update_beliefs", "act_on_world")
                        and requested_tier == "primary"
                    ):
                        # Nudge toward deeper tier if available
                        if not deep_handoff:
                            logger.debug(
                                "⚡ FE urgency (F=%.2f, action=%s): consider deeper reasoning",
                                _fe_state.free_energy,
                                _fe_state.dominant_action,
                            )
                            # Don't force tier switch — just extend token budget
                            max_tokens = min(max_tokens + 256, 4096)
            except _INFERENCE_RECOVERABLE_ERRORS as _fe_e:
                record_degradation(
                    "inference_gate",
                    _fe_e,
                    severity="warning",
                    action="continued without free-energy token-budget nudge",
                )
                logger.debug("FreeEnergy tier nudge unavailable: %s", _fe_e)

        # Ordinary live conversation must not collapse into a starvation budget
        # after affective / homeostatic modulation. Explicit caller caps still
        # win, as do hard resource-stakes blocks and deep-probe turns.
        if (
            not is_background
            and self._origin_is_user_facing(origin)
            and requested_tier in {"primary", "secondary"}
            and "max_tokens" not in context
            and not bool(context.get("resource_stakes_blocked", False))
            and not deep_probe_request
            and not isolated_generation_contract
            and not health_probe
        ):
            foreground_floor, foreground_cap, _foreground_loops = (
                self._foreground_compute_profile(initial_visible_user_prompt)
            )
            max_tokens = min(max_tokens, foreground_cap)
            if max_tokens < foreground_floor:
                logger.info(
                    "🧠 Foreground chat compute profile raised budget %d→%d "
                    "(cap=%d, loops=%d, origin=%s).",
                    max_tokens,
                    foreground_floor,
                    foreground_cap,
                    _foreground_loops,
                    origin or "unknown",
                )
                max_tokens = foreground_floor

        if (
            not is_background
            and self._origin_is_user_facing(origin)
            and not isolated_generation_contract
            and not health_probe
            and not benchmark_request
            and not proof_evaluation_contract
            and not strict_answer_contract
        ):
            somatic_temperature, max_tokens, applied_bias = self._apply_runtime_sampling_biases(
                base_temperature=somatic_temperature,
                max_tokens=max_tokens,
                context=context,
                state=state,
                allow_token_scaling="max_tokens" not in context,
            )
            if applied_bias["temperature_delta"] or applied_bias["max_tokens_factor"] != 1.0:
                logger.debug(
                    "🧠 Runtime sampling bias: temp_delta=%.3f token_factor=%.3f max_tokens=%d",
                    applied_bias["temperature_delta"],
                    applied_bias["max_tokens_factor"],
                    max_tokens,
                )

        if (
            not is_background
            and self._origin_is_user_facing(origin)
            and requested_tier in {"primary", "secondary"}
            and "max_tokens" not in context
            and not bool(context.get("resource_stakes_blocked", False))
            and not deep_probe_request
            and not isolated_generation_contract
            and not health_probe
        ):
            foreground_floor, foreground_cap, _foreground_loops = (
                self._foreground_compute_profile(initial_visible_user_prompt)
            )
            bounded = min(max_tokens, foreground_cap)
            if bounded < foreground_floor:
                logger.info(
                    "🧠 Foreground chat post-bias budget floor raised %d→%d "
                    "(cap=%d, origin=%s).",
                    bounded,
                    foreground_floor,
                    foreground_cap,
                    origin or "unknown",
                )
                max_tokens = foreground_floor
            else:
                max_tokens = bounded

        if explicit_max_tokens_cap is not None:
            max_tokens = min(max_tokens, explicit_max_tokens_cap)
            if (
                protected_compact_capability_contract
                and not bool(context.get("resource_stakes_blocked", False))
            ):
                max_tokens = max(max_tokens, min(384, explicit_max_tokens_cap))
            context["max_tokens"] = max_tokens

        if (
            not is_background
            and self._origin_is_user_facing(origin)
            and requested_tier == "primary"
            and not isolated_generation_contract
            and not health_probe
        ):
            # Foreground chat prompts include high-churn state, memory, and
            # reliability blocks. Reusing approximate KV caches across those
            # prompts has been a major source of clipped or stale Cortex
            # drafts. Keep cache acceleration for background/proof lanes, but
            # make live primary conversation start from the exact prompt.
            morpho_kwargs.setdefault("disable_prompt_cache", True)

        if deep_probe_request and not is_background:
            probe_token_cap = int(os.environ.get("AURA_DEEP_PROBE_MAX_TOKENS", "384"))
            max_tokens = min(max_tokens, max(128, probe_token_cap))
            context["max_tokens"] = max_tokens
            context["allow_tools"] = False

        if strict_answer_contract and strict_max_token_cap is not None:
            max_tokens = max(1, min(max_tokens, strict_max_token_cap))
            context["max_tokens"] = max_tokens

        if operator_evidence_contract:
            try:
                requested_operator_cap = int(context.get("max_tokens") or 220)
            except (TypeError, ValueError):
                requested_operator_cap = 220
            max_tokens = max(1, min(max_tokens, requested_operator_cap, 220))
            context["max_tokens"] = max_tokens
            context["allow_tools"] = False
            context["disable_prompt_cache"] = True
            context["clear_prompt_cache"] = True

        if health_probe:
            requested_cap = context.get("max_tokens", max_tokens)
            try:
                requested_cap_int = max(1, int(requested_cap))
            except (TypeError, ValueError):
                requested_cap_int = 32
            max_tokens = max(1, min(max_tokens, requested_cap_int, 64))
            context["max_tokens"] = max_tokens
            context.setdefault("clean_user_surface_contract", True)
            context.setdefault("user_surface_validation_prompt", initial_visible_user_prompt)
            context.setdefault("clean_user_surface_recurrent_loops", 1)
            context.setdefault("clean_user_surface_steering_alpha", 0.25)
            morpho_kwargs.setdefault("clean_user_surface_contract", True)
            morpho_kwargs.setdefault("user_surface_validation_prompt", initial_visible_user_prompt)
            morpho_kwargs.setdefault("clean_user_surface_recurrent_loops", 1)
            morpho_kwargs.setdefault("clean_user_surface_steering_alpha", 0.25)

        if benchmark_request:
            requested_cap = context.get("max_tokens", max_tokens)
            try:
                requested_cap_int = max(1, int(requested_cap))
            except (TypeError, ValueError):
                requested_cap_int = 96
            max_tokens = max(1, min(max_tokens, requested_cap_int))
            context["max_tokens"] = max_tokens

        output_contract_is_user_facing = bool(
            not is_background
            and not isolated_generation_contract
            and not health_probe
            and not benchmark_request
            and (
                explicit_foreground
                or self._origin_is_user_facing(origin)
                or requested_tier in {"primary", "secondary"}
            )
        )
        if (
            output_contract_is_user_facing
            and output_contract_payload is not None
            and output_contract.hard_token_ceiling is not None
        ):
            planned_tokens = max_tokens
            max_tokens = max(
                1,
                min(max_tokens, int(output_contract.hard_token_ceiling)),
            )
            context["requested_output_contract"] = dict(output_contract_payload)
            context["semantic_output_token_cap"] = output_contract.semantic_token_cap
            context["hard_output_token_ceiling"] = output_contract.hard_token_ceiling
            context["max_tokens"] = max_tokens
            morpho_kwargs["requested_output_contract"] = dict(output_contract_payload)
            morpho_kwargs["semantic_output_token_cap"] = output_contract.semantic_token_cap
            morpho_kwargs["hard_output_token_ceiling"] = output_contract.hard_token_ceiling
            if max_tokens < planned_tokens:
                logger.info(
                    "🧠 Explicit output contract capped generation %d→%d "
                    "(kind=%s semantic=%s hard=%s).",
                    planned_tokens,
                    max_tokens,
                    output_contract.kind,
                    output_contract.semantic_token_cap,
                    output_contract.hard_token_ceiling,
                )

        # No policy floor may expand a caller-admitted ceiling. Keep this as
        # the final token-budget transformation before prompt construction and
        # every local/cloud provider call below.
        if explicit_max_tokens_cap is not None:
            max_tokens = max(1, min(max_tokens, explicit_max_tokens_cap))
            context["max_tokens"] = max_tokens

        # Build the prompt only after routing intent is known so we can choose
        # a compact user-facing path instead of always constructing the richest stack.
        brief = context.get("brief", "")
        if hasattr(brief, "to_briefing_text"):
            brief = brief.to_briefing_text()
        elif not isinstance(brief, str):
            brief = str(brief)
        use_compact_foreground_context = self._should_use_compact_foreground_context(
            origin,
            requested_tier,
            deep_handoff=deep_handoff,
            is_background=is_background,
            prompt=initial_visible_user_prompt,
            context=context,
        )
        provided_messages = context.get("messages")
        if not isinstance(provided_messages, list):
            provided_messages = None
        context_system_prompt = str(context.get("system_prompt", "") or "").strip()

        def _append_unique_system_part(parts: list[str], content: Any) -> None:
            text = str(content or "").strip()
            if not text:
                return
            if any(text == existing or text in existing for existing in parts):
                return
            parts.append(text)

        if strict_answer_contract:
            provided_system_parts: list[str] = []
            _append_unique_system_part(provided_system_parts, context_system_prompt)
            strict_system_prompt = (
                "You are Aura's local reasoning lane, a persistent local cognitive runtime. "
                "Follow the user's exact output contract. "
                "When the user requests <answer>...</answer>, return only that final answer envelope. "
                "Do not copy instructions, role labels, or explanatory text."
            )
            strict_user_prompt = str(prompt or "")
            if provided_messages is not None:
                for msg in reversed(provided_messages):
                    if not isinstance(msg, dict):
                        continue
                    if str(msg.get("role", "") or "").strip().lower() == "system":
                        _append_unique_system_part(
                            provided_system_parts,
                            msg.get("content", ""),
                        )
                        continue
                    if str(msg.get("role", "") or "").strip().lower() == "user":
                        strict_user_prompt = str(msg.get("content", "") or strict_user_prompt)
                        break
            if provided_system_parts:
                preserved_system = "\n\n".join(reversed(provided_system_parts)).strip()
                if preserved_system:
                    strict_system_prompt = f"{strict_system_prompt}\n\n{preserved_system}"
            strict_system_prompt += self._strict_contract_procedure_hints(strict_user_prompt)
            provided_messages = [
                {"role": "system", "content": strict_system_prompt},
                {"role": "user", "content": strict_user_prompt},
            ]
        elif strict_value_contract:
            provided_system_parts = []
            _append_unique_system_part(provided_system_parts, context_system_prompt)
            strict_value_system_prompt = (
                "You are Aura's local reasoning lane. Solve the task and return only "
                "the final answer value. Do not explain, do not add role labels, and "
                "do not include XML tags."
            )
            strict_value_user_prompt = str(prompt or "")
            if provided_messages is not None:
                for msg in reversed(provided_messages):
                    if not isinstance(msg, dict):
                        continue
                    if str(msg.get("role", "") or "").strip().lower() == "system":
                        _append_unique_system_part(
                            provided_system_parts,
                            msg.get("content", ""),
                        )
                        continue
                    if str(msg.get("role", "") or "").strip().lower() == "user":
                        strict_value_user_prompt = str(msg.get("content", "") or strict_value_user_prompt)
                        break
            if provided_system_parts:
                preserved_system = "\n\n".join(reversed(provided_system_parts)).strip()
                if preserved_system:
                    strict_value_system_prompt = f"{strict_value_system_prompt}\n\n{preserved_system}"
            strict_value_system_prompt += self._strict_contract_procedure_hints(
                strict_value_user_prompt
            )
            provided_messages = [
                {"role": "system", "content": strict_value_system_prompt},
                {"role": "user", "content": strict_value_user_prompt},
            ]
        visible_user_prompt = initial_visible_user_prompt
        if provided_messages is not None:
            system_prompt = ""
            for msg in provided_messages:
                if not isinstance(msg, dict):
                    continue
                if str(msg.get("role", "") or "").strip().lower() == "system":
                    system_prompt = str(msg.get("content", "") or "").strip()
                    break
            living_mind_context = ""
            if (
                not isolated_generation_contract
                and not is_background
                and self._origin_is_user_facing(origin)
            ):
                needs_full_live_context = bool(
                    context.get("live_runtime_payload_required", False)
                    and not use_compact_foreground_context
                    and (
                        is_live_self_reflection_turn(initial_visible_user_prompt)
                        or is_self_process_question(initial_visible_user_prompt)
                    )
                )
                if needs_full_live_context:
                    try:
                        living_mind_context = await asyncio.wait_for(
                            self._build_living_mind_context(
                                initial_visible_user_prompt,
                                origin,
                            ),
                            timeout=5.0,
                        )
                    except TimeoutError:
                        logger.warning(
                            "⚠️ [STABILITY] Full live self-context assembly exceeded 5s; "
                            "using compact living context without downgrading the turn."
                        )
                        living_mind_context = await self._build_compact_living_mind_context(
                            visible_user_prompt,
                            origin,
                        )
                else:
                    living_mind_context = await self._build_compact_living_mind_context(
                        visible_user_prompt,
                        origin,
                    )
        elif use_compact_foreground_context:
            system_prompt = self._build_compact_system_prompt(brief)
            living_mind_context = await self._build_compact_living_mind_context(
                visible_user_prompt,
                origin,
            )
        else:
            system_prompt = self._build_system_prompt(brief)
            # [STABILITY v50] Hard 5s timeout on full context assembly.
            # The 20+ consciousness subsystems queried here can individually
            # hang due to lock contention or slow I/O. When that happens,
            # fall back to the compact (4-subsystem) version so generation
            # budget is never consumed by context assembly.
            try:
                living_mind_context = await asyncio.wait_for(
                    self._build_living_mind_context(visible_user_prompt, origin),
                    timeout=5.0,
                )
            except TimeoutError:
                logger.warning(
                    "⚠️ [STABILITY] Full living mind context assembly exceeded 5s budget. "
                    "Falling back to compact context to preserve generation headroom."
                )
                living_mind_context = await self._build_compact_living_mind_context(
                    visible_user_prompt,
                    origin,
                )
        if living_mind_context:
            system_prompt = f"{system_prompt}\n\n{living_mind_context}"
        prompt_contract_block = self._prompt_contract_block(context)
        if prompt_contract_block and not isolated_generation_contract:
            system_prompt = f"{system_prompt}\n\n{prompt_contract_block}"
        # Keep prompt growth aligned with the actual local model context window
        # instead of assuming 128k+ headroom on the primary Qwen lane.

        # ── Somatic narrative: brief felt-state line in the system prompt ────────
        if somatic_temperature is not None and not isolated_generation_contract:
            try:
                from core.affect.affective_circumplex import get_circumplex

                _soma_narrative = get_circumplex().describe()
                if _soma_narrative:
                    system_prompt = f"{system_prompt}\n\n## SOMATIC STATE\n{_soma_narrative}"
            except _INFERENCE_RECOVERABLE_ERRORS as _exc:
                record_degradation(
                    "inference_gate",
                    _exc,
                    severity="warning",
                    action="continued without somatic-state prompt section",
                )
                logger.debug("Suppressed Exception: %s", _exc)

        prompt_user_facing = bool(
            not benchmark_request
            and not is_background
            and not web_interlocutor_contract
            and (
                self._origin_is_user_facing(origin)
                or explicit_foreground
                or purpose in {"reply", "expression", "chat", "conversation", "user_response"}
            )
        )

        # ── Architecture Self-Awareness: inject relevant subsystem context ──────
        # Only for user-facing requests that mention architecture/code keywords.
        if prompt_user_facing and not isolated_generation_contract:
            try:
                import re as _re

                _arch_triggers = _re.compile(
                    r"\b(how|explain|what|which|where|why|trace|show|describe)\b.{0,60}"
                    r"\b(module|subsystem|file|class|method|function|work|does|handles|manages|routes|sends|wires)\b",
                    _re.IGNORECASE,
                )
                if _arch_triggers.search(visible_user_prompt):
                    from core.self.architecture_index import get_architecture_index

                    arch_excerpt = get_architecture_index().query(
                        visible_user_prompt,
                        max_results=3,
                    )
                    if arch_excerpt:
                        system_prompt = f"{system_prompt}\n\n{arch_excerpt}"
            except _INFERENCE_RECOVERABLE_ERRORS as _ae:
                record_degradation(
                    "inference_gate",
                    _ae,
                    severity="warning",
                    action="continued without architecture self-awareness excerpt",
                )
                logger.debug("ArchIndex injection skipped: %s", _ae)
            system_prompt = (
                f"{system_prompt}\n\n"
                f"{conversation_reliability_system_block(visible_user_prompt)}"
            )
        history = context.get("history", [])
        use_rich_context = False if isolated_generation_contract or benchmark_request else bool(
            context.get(
                "rich_context",
                self._should_use_rich_context(
                    origin,
                    requested_tier,
                    deep_handoff=deep_handoff,
                    is_background=is_background,
                ),
            )
        )
        if provided_messages is not None:
            messages = [dict(msg) for msg in provided_messages if isinstance(msg, dict)]
            if not isolated_generation_contract and (prompt_user_facing or living_mind_context):
                reliability_block = conversation_reliability_system_block(visible_user_prompt)
                inserted = False
                for msg in messages:
                    if str(msg.get("role", "") or "").strip().lower() == "system":
                        content = str(msg.get("content", "") or "")
                        if living_mind_context and living_mind_context not in content:
                            content = f"{content.rstrip()}\n\n{living_mind_context}".strip()
                        if "USER-FACING CONVERSATION RELIABILITY CONTRACT" not in content:
                            content = f"{content.rstrip()}\n\n{reliability_block}".strip()
                        msg["content"] = content
                        inserted = True
                        break
                if not inserted:
                    blocks = [
                        block
                        for block in (
                            living_mind_context,
                            reliability_block if prompt_user_facing else "",
                        )
                        if block
                    ]
                    messages.insert(0, {"role": "system", "content": "\n\n".join(blocks)})
        else:
            messages = (
                self._build_messages(prompt, system_prompt, history)
                if use_rich_context
                else self._build_compact_messages(prompt, system_prompt, history)
            )
        if provided_messages is not None and (use_compact_foreground_context or use_rich_context):
            short_output_contract = self._has_short_live_output_contract(context)
            deep_probe_context = bool(context.get("deep_mind_probe", False)) and not (
                short_output_contract
            )
            if use_compact_foreground_context:
                foreground_profile = (
                    "contract"
                    if short_output_contract
                    else self._foreground_prompt_profile(
                        visible_user_prompt,
                        context,
                    )
                )
            else:
                # Rich/DEEP prebuilt prompts otherwise bypassed compaction
                # entirely, so the local Cortex (notably the in-process MLX 32B)
                # received 100k+ char prompts (~25k tokens) that exceed the 16k
                # context window AND can't be processed+generated within the
                # cognitive-cycle watchdog → "Cognitive cycle TIMEOUT" +
                # "RuntimeError in mlx_client". Compact rich prebuilt to the
                # extended budget (~26k chars) so the live lane stays responsive
                # while still carrying substantial living-mind context.
                foreground_profile = "extended"
            messages = self._compact_prebuilt_messages(
                messages,
                history_limit=self._foreground_prebuilt_history_limit(
                    visible_user_prompt,
                    context,
                    deep_probe=deep_probe_context,
                ),
                deep_probe=deep_probe_context,
                budget_profile=foreground_profile,
                current_user_content=visible_user_prompt,
            )
        prompt_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
        prompt_mode = "rich" if use_rich_context else "compact"
        if use_compact_foreground_context:
            prompt_mode = "compact_foreground"
        if provided_messages is not None:
            prompt_mode = f"{prompt_mode}_prebuilt"
        logger.info(
            "🧠 [ZENITH] Prompt plan: mode=%s messages=%d chars=%d origin=%s max_tokens=%d",
            prompt_mode,
            len(messages),
            prompt_chars,
            origin or "unknown",
            max_tokens,
        )

        _is_user_facing = (
            not benchmark_request
            and not web_interlocutor_contract
            and (self._origin_is_user_facing(origin) or requested_tier == "primary")
            and not health_probe
            and not proof_evaluation_contract
        )
        if (
            _is_user_facing
            and requested_tier == "primary"
            and not strict_answer_contract
            and not strict_value_contract
        ):
            _foreground_floor, _foreground_cap, foreground_loops = (
                self._foreground_compute_profile(initial_visible_user_prompt)
            )
            foreground_profile = self._foreground_prompt_profile(
                visible_user_prompt,
                context,
            )
            morpho_kwargs.setdefault("clean_user_surface_contract", True)
            morpho_kwargs.setdefault(
                "user_surface_validation_prompt",
                initial_visible_user_prompt or visible_user_prompt,
            )
            morpho_kwargs.setdefault(
                "clean_user_surface_recurrent_loops",
                foreground_loops,
            )
            morpho_kwargs.setdefault(
                "clean_user_surface_steering_alpha",
                0.35 if foreground_profile == "extended" else 0.25,
            )
        client_foreground_request = (
            bool(_is_user_facing or explicit_foreground) and not is_background and not benchmark_request
        )
        protected_deep_fallback = False

        # 1. Try the selected local brain.
        if self._mlx_client:
            try:
                from core.brain.llm.mlx_client import get_mlx_client
                from core.brain.llm.model_registry import (
                    ACTIVE_MODEL,
                    get_brainstem_path,
                    get_deep_model_path,
                    get_fallback_path,
                    get_runtime_model_path,
                )

                local_client = self._mlx_client
                local_label = PRIMARY_ENDPOINT
                fallback_client = None
                fallback_model_path = str(get_brainstem_path())
                fallback_kwargs: dict[str, Any] = {}
                fallback_label = BRAINSTEM_ENDPOINT
                restore_primary = False

                def _ensure_fallback_client():
                    nonlocal fallback_client
                    if fallback_client is None:
                        fallback_client = get_mlx_client(
                            model_path=fallback_model_path,
                            **fallback_kwargs,
                        )
                    return fallback_client

                primary_restored_inline = False
                try:
                    if requested_tier == "tertiary":
                        local_client = get_mlx_client(model_path=str(get_brainstem_path()))
                        local_label = BRAINSTEM_ENDPOINT
                        fallback_model_path = str(get_fallback_path())
                        fallback_kwargs = {"device": "cpu"}
                        fallback_label = FALLBACK_ENDPOINT
                    elif deep_handoff:
                        local_client = get_mlx_client(model_path=str(get_deep_model_path()))
                        local_label = DEEP_ENDPOINT
                        fallback_model_path = str(get_runtime_model_path(ACTIVE_MODEL))
                        fallback_kwargs = {}
                        fallback_label = PRIMARY_ENDPOINT
                        restore_primary = True

                    protected_deep_fallback = bool(
                        bool(context.get("allow_deep_fallback", False))
                        and deep_probe_request
                        and _is_user_facing
                        and requested_tier == "primary"
                    )
                    if protected_deep_fallback:
                        fallback_model_path = str(get_deep_model_path())
                        fallback_kwargs = {}
                        fallback_label = DEEP_ENDPOINT
                    skip_initial_primary_attempt = False
                    primary_warmup_memory_deferred = False
                    lane_managed_client = hasattr(local_client, "get_lane_status") or hasattr(
                        local_client, "warmup"
                    )
                    if _is_user_facing and local_label == PRIMARY_ENDPOINT and lane_managed_client:
                        lane_status = self.get_conversation_status()
                        if not lane_status.get("conversation_ready"):
                            blockers = lane_status.get("readiness_blockers") or []
                            blocker_text = ", ".join(str(item) for item in blockers[:3]) or "conversation probe"
                            logger.info(
                                "🧠 %s lane process state=%s; conversation readiness is blocked by %s. Completing foreground warmup before first generation attempt.",
                                local_label,
                                lane_status.get("state", "unknown"),
                                blocker_text,
                            )
                            try:
                                # Admission control — break the cortex doom-loop.
                                # A COLD first boot legitimately needs ~150s to
                                # load the 32B and the user expects that one-time
                                # wait. But a RECOVERY (Cortex was ready, got
                                # force-killed on a first-token stall, is now
                                # reloading) must NOT block every foreground turn
                                # for 90-180s — that is the observed doom loop
                                # (soak Jul 7: turns 21-30 crawled to 200s+ while
                                # the warm window played out, memory thrashed).
                                # When the lane was EVER ready, cap the preflight
                                # wait short and let this turn fall to the ready
                                # tier while Cortex warms in the background for the
                                # next turn.
                                warmup_timeout = self._foreground_warmup_timeout(
                                    lane_status, primary_timeout
                                )
                                lane_status = await self.ensure_foreground_ready(
                                    timeout=warmup_timeout
                                )
                            except (
                                TimeoutError,
                                RuntimeError,
                                AttributeError,
                                TypeError,
                                ValueError,
                                OSError,
                            ) as warmup_exc:
                                if self._note_foreground_warmup_failure(warmup_exc):
                                    primary_warmup_memory_deferred = True
                                lane_status = self.get_conversation_status()
                            if not self._lane_can_attempt_visible_conversation_turn(lane_status):
                                skip_initial_primary_attempt = True
                                logger.warning(
                                    "🧠 %s is still not ready after foreground preflight warmup (state=%s). Skipping the cold first attempt and waiting for recovery before retry.",
                                    local_label,
                                    lane_status.get("state", "unknown"),
                                )
                    if primary_warmup_memory_deferred:
                        if (
                            proof_evaluation_contract
                            or strict_primary_proof_lane
                            or desktop_cognitive_engine_contract
                        ):
                            logger.warning(
                                "🧠 Proof/evaluation request requires Cortex; refusing Brainstem fallback after primary warmup deferral."
                            )
                            return None
                        logger.warning(
                            "🧠 Cortex cold-load deferred by RAM admission; routing this foreground turn to %s.",
                            fallback_label,
                        )
                        fallback_client = _ensure_fallback_client()
                        local_client = fallback_client
                        local_label = fallback_label
                        skip_initial_primary_attempt = False
                    logger.info(
                        "🧠 Routing to %s (timeout=%.0fs, user_facing=%s)...",
                        local_label,
                        float(timeout_val),
                        _is_user_facing,
                    )
                    primary_deadline = get_deadline(primary_timeout)
                    primary_attempt_started = time.monotonic()
                    if skip_initial_primary_attempt:
                        text = None
                    else:
                        async with self._resource_context(
                            enabled=local_label != FALLBACK_ENDPOINT,
                            priority=client_foreground_request,
                            worker=local_label,
                            timeout_s=primary_deadline.remaining or primary_timeout,
                        ):
                            text = await self._generate_with_client(
                                local_client,
                                prompt,
                                system_prompt,
                                history,
                                primary_deadline,
                                local_label,
                                messages=messages,
                                max_tokens=max_tokens,
                                temperature=somatic_temperature,
                                origin=origin,
                                is_background=is_background,
                                foreground_request=client_foreground_request,
                                **morpho_kwargs,
                            )
                    primary_attempt_elapsed = max(
                        0.0,
                        time.monotonic() - primary_attempt_started,
                    )
                    if text:
                        repairable_draft = self._repairable_user_facing_draft_for_downstream(
                            text,
                            visible_user_prompt,
                        ) if _is_user_facing else None
                        if repairable_draft is not None:
                            logger.warning(
                                "🛡️ Preserving repairable Cortex draft for downstream response repair (len=%d).",
                                len(repairable_draft),
                            )
                            return self._stabilize_user_facing_text(
                                repairable_draft,
                                visible_user_prompt,
                                is_user_facing=True,
                            )
                        return self._stabilize_user_facing_text(
                            text,
                            visible_user_prompt,
                            is_user_facing=_is_user_facing,
                        )
                    primary_failure_metadata = self.get_last_generation_metadata()
                    primary_surface_quality_rejected = (
                        str(primary_failure_metadata.get("error") or "").strip()
                        == "surface_quality_rejected"
                    )
                    if primary_surface_quality_rejected and desktop_cognitive_engine_contract:
                        logger.warning(
                            "🧠 %s exhausted its worker-owned semantic quality retries; "
                            "preserving the lane and refusing a duplicate inference-gate retry.",
                            local_label,
                        )
                        return None
                    if health_probe:
                        logger.warning(
                            "🧠 %s proof health probe returned no text; refusing local fallback for lane certification.",
                            local_label,
                        )
                        return None
                    if (
                        proof_evaluation_contract
                        or strict_primary_proof_lane
                    ):
                        logger.warning(
                            "🧠 Proof/evaluation request requires a valid Cortex response; refusing retry/fallback cascade after no text."
                        )
                        return None
                    # NOTE: desktop_cognitive_engine_contract is intentionally NOT
                    # refused here. A thin/empty first draft on a live desktop turn
                    # (e.g. the 32B emits a short reply that trips too_thin) must
                    # get one more attempt on the SAME primary Cortex lane — her
                    # real mind — instead of fail-closing immediately. The later
                    # gate below still refuses any LOWER-lane fallback for desktop
                    # turns, so the "real mind only" guarantee is preserved; this
                    # only restores the same-lane retry the early refuse was
                    # skipping (the cause of "I could not produce a reliable
                    # full-mind reply" on casual short turns).

                    # ── CORTEX RETRY: For user-facing requests, retry the primary model
                    # only when the first attempt failed quickly and the lane
                    # remains ready. Long stalls must preserve the remaining
                    # response budget for a governed recovery lane.
                    if _is_user_facing and local_label == PRIMARY_ENDPOINT:
                        retry_schedule = self._foreground_retry_schedule(
                            primary_attempt_elapsed,
                            primary_timeout,
                        )
                        if not retry_schedule:
                            logger.warning(
                                "🧠 %s consumed %.1fs without usable text; skipping repeated "
                                "same-lane retries.",
                                local_label,
                                primary_attempt_elapsed,
                            )
                        for retry_attempt, wait_sec in enumerate(retry_schedule, 1):
                            if is_shutdown_requested():
                                logger.info(
                                    "🛑 %s retry loop aborted: runtime is shutting down.",
                                    local_label,
                                )
                                return ""
                            lane_status = self.get_conversation_status()
                            if not self._lane_can_attempt_visible_conversation_turn(lane_status):
                                logger.warning(
                                    "🧠 %s is not ready after the failed attempt (state=%s); "
                                    "skipping same-lane retry %d.",
                                    local_label,
                                    lane_status.get("state", "unknown"),
                                    retry_attempt,
                                )
                                break
                            if is_shutdown_requested():
                                logger.info(
                                    "🛑 %s retry wait skipped: runtime is shutting down.",
                                    local_label,
                                )
                                return ""

                            logger.warning(
                                "🧠 %s returned no text on user-facing request. "
                                "Retrying once after %ds pause...",
                                local_label,
                                wait_sec,
                            )
                            await asyncio.sleep(wait_sec)
                            if is_shutdown_requested():
                                logger.info(
                                    "🛑 %s retry generation skipped: runtime is shutting down.",
                                    local_label,
                                )
                                return ""

                            retry_timeout = min(
                                60.0,
                                max(30.0, primary_timeout * 0.4),
                            )
                            retry_deadline = get_deadline(retry_timeout)
                            retry_messages = self._build_primary_repair_messages(
                                visible_user_prompt,
                                messages,
                            )
                            retry_system_prompt = retry_messages[0]["content"]
                            retry_morpho_kwargs = dict(morpho_kwargs)
                            retry_morpho_kwargs.update(
                                {
                                    "disable_prompt_cache": True,
                                    "clear_prompt_cache": retry_attempt == 1,
                                    "top_p": min(float(retry_morpho_kwargs.get("top_p", 0.9) or 0.9), 0.85),
                                    "min_p": max(float(retry_morpho_kwargs.get("min_p", 0.02) or 0.02), 0.02),
                                    "repetition_penalty": max(
                                        float(retry_morpho_kwargs.get("repetition_penalty", 1.1) or 1.1),
                                        1.12,
                                    ),
                                    "repetition_context_size": max(
                                        int(retry_morpho_kwargs.get("repetition_context_size", 64) or 64),
                                        96,
                                    ),
                                    "skip_runtime_payload": True,
                                }
                            )
                            retry_temperature = min(
                                float(somatic_temperature if somatic_temperature is not None else 0.35),
                                0.35,
                            )
                            async with self._resource_context(
                                enabled=True,
                                priority=True,
                                worker=local_label,
                                timeout_s=retry_deadline.remaining or retry_timeout,
                            ):
                                text = await self._generate_with_client(
                                    local_client,
                                    prompt,
                                    retry_system_prompt,
                                    history,
                                    retry_deadline,
                                    f"{local_label}-RETRY-{retry_attempt}",
                                    messages=retry_messages,
                                    max_tokens=max_tokens,
                                    temperature=retry_temperature,
                                    origin=origin,
                                    is_background=is_background,
                                    foreground_request=True,
                                    **retry_morpho_kwargs,
                                )
                            if text:
                                logger.info(
                                    "✅ %s retry %d succeeded (len=%d)",
                                    local_label,
                                    retry_attempt,
                                    len(text),
                                )
                                repairable_draft = self._repairable_user_facing_draft_for_downstream(
                                    text,
                                    visible_user_prompt,
                                ) if _is_user_facing else None
                                if repairable_draft is not None:
                                    logger.warning(
                                        "🛡️ Preserving repairable Cortex retry draft for downstream response repair (len=%d).",
                                        len(repairable_draft),
                                    )
                                    return self._stabilize_user_facing_text(
                                        repairable_draft,
                                        visible_user_prompt,
                                        is_user_facing=True,
                                    )
                                return self._stabilize_user_facing_text(
                                    text,
                                    visible_user_prompt,
                                    is_user_facing=_is_user_facing,
                                )

                        if retry_schedule:
                            logger.warning("🧠 %s bounded retry failed.", local_label)
                        if (
                            proof_evaluation_contract
                            or strict_primary_proof_lane
                            or operator_evidence_contract
                            or desktop_cognitive_engine_contract
                        ):
                            logger.warning(
                                "🧠 Proof/operator request requires a valid Cortex response; refusing lower-lane fallback."
                            )
                            return None
                        # For user-facing requests, skip brainstem — go straight to cloud
                        if allow_cloud_fallback:
                            logger.warning(
                                "🧠 Escalating to cloud before brainstem for user-facing request."
                            )
                            raise _UserFacingCortexError()
                        logger.warning(
                            "🧠 %s is still recovering. Falling back to %s for this %s foreground turn.",
                            local_label,
                            fallback_label,
                            "protected" if protected_deep_fallback else "local-only",
                        )
                    else:
                        logger.warning(
                            "🧠 %s returned no text. Trying local fallback.", local_label
                        )
                        if is_background and not bool(
                            context.get("allow_background_local_fallback", False)
                        ):
                            logger.info(
                                "🧠 Background %s request returned no text; suppressing local fallback to protect foreground latency.",
                                local_label,
                            )
                            return None

                    # Graceful local fallback: for background/autonomous requests, the
                    # brainstem is an acceptable degradation. For user-facing requests
                    # that reach here (cloud disabled), it's the last local resort.
                    fallback_deadline = get_deadline(fallback_timeout)
                    fallback_client = _ensure_fallback_client()
                    async with self._resource_context(
                        enabled=fallback_label != FALLBACK_ENDPOINT,
                        priority=client_foreground_request,
                        worker=fallback_label,
                        timeout_s=fallback_deadline.remaining or fallback_timeout,
                    ):
                        fallback_max_tokens = (
                            max_tokens
                            if fallback_label == DEEP_ENDPOINT
                            else min(max_tokens, 384 if requested_tier != "secondary" else 512)
                        )
                        brainstem_text = await self._generate_with_client(
                            fallback_client,
                            prompt,
                            system_prompt,
                            history,
                            fallback_deadline,
                            fallback_label,
                            messages=messages,
                            max_tokens=fallback_max_tokens,
                            temperature=somatic_temperature,
                            origin=origin,
                            is_background=is_background,
                            foreground_request=client_foreground_request,
                            **morpho_kwargs,
                        )
                    if brainstem_text:
                        if fallback_label == PRIMARY_ENDPOINT:
                            primary_restored_inline = True
                        return self._stabilize_user_facing_text(
                            brainstem_text,
                            visible_user_prompt,
                            is_user_facing=_is_user_facing,
                        )
                    logger.warning("🧠 Local fallback returned no text.")
                finally:
                    if restore_primary and not primary_restored_inline:
                        self._schedule_primary_restore_after_deep_handoff()

            except _UserFacingCortexError:
                logger.warning(
                    "🧠 User-facing Cortex failure — bypassing brainstem, escalating to cloud."
                )
            except TimeoutError as timeout_exc:
                logger.warning("🛑 Local inference TIMED OUT (Budget: %.0fs).", timeout_val)
                if (
                    not is_background
                    and self._origin_is_user_facing(origin)
                    and not allow_cloud_fallback
                ):
                    raise TimeoutError(
                        f"{local_label} timed out after {timeout_val:.0f}s"
                    ) from timeout_exc
            except _INFERENCE_RECOVERABLE_ERRORS as e:
                record_degradation(
                    "inference_gate",
                    e,
                    severity="degraded",
                    action="fell through to reflex or cloud fallback after local inference failure",
                )
                logger.warning("🛑 Local inference FAILURE: %s", e)

        # 1.5. EMERGENCY REFLEX FALLBACK — tiny 1.5B model on CPU as absolute last local resort.
        # If Cortex AND Brainstem both failed for a user-facing request, the 1.5B Reflex
        # model can still produce SOMETHING so the user isn't left hanging.
        # [STABILITY v54] Never run the 1.5B reflex if we are in a protected 32B foreground lane.
        if (
            _is_user_facing
            and not is_background
            and not protected_deep_fallback
            and not proof_evaluation_contract
            and not desktop_cognitive_engine_contract
        ):
            try:
                from core.brain.llm.mlx_client import get_mlx_client
                from core.brain.llm.model_registry import get_fallback_path

                reflex_client = get_mlx_client(model_path=str(get_fallback_path()), device="cpu")
                if reflex_client:
                    logger.warning(
                        "🆘 [REFLEX] Cortex + Brainstem both failed. Trying 1.5B CPU Reflex..."
                    )
                    reflex_deadline = get_deadline(15.0)  # 15s hard limit for tiny model
                    reflex_text = await self._generate_with_client(
                        reflex_client,
                        prompt,
                        system_prompt,
                        history[-2:] if history else [],  # minimal history for tiny model
                        reflex_deadline,
                        FALLBACK_ENDPOINT,
                        messages=None,
                        max_tokens=min(max_tokens, 200),  # keep it short
                        temperature=somatic_temperature,
                        origin=origin,
                        is_background=False,
                        foreground_request=True,
                        **morpho_kwargs,
                    )
                    if reflex_text:
                        logger.info(
                            "🆘 [REFLEX] 1.5B CPU model produced response. Cortex recovery in background."
                        )
                        if not self._cortex_recovery_in_progress:
                            get_task_tracker().create_task(self._ensure_cortex_recovery())
                        return self._stabilize_user_facing_text(
                            reflex_text,
                            visible_user_prompt,
                            is_user_facing=True,
                        )
            except _INFERENCE_RECOVERABLE_ERRORS as reflex_err:
                record_degradation(
                    "inference_gate",
                    reflex_err,
                    severity="warning",
                    action="continued to configured cloud or exhaustion path after reflex fallback failed",
                )
                logger.debug("Reflex fallback failed: %s", reflex_err)

        # 2. Optional cloud fallback.
        if not allow_cloud_fallback:
            logger.error("Local inference paths exhausted. Cloud fallback disabled by request policy.")
            if proof_evaluation_contract or desktop_cognitive_engine_contract:
                logger.error("Primary foreground contract exhausted Cortex without valid text.")
                return None
            # User-facing requests still trigger local Cortex recovery, but
            # ``allow_cloud_fallback=False`` is a hard boundary. Do not route to
            # Gemini/HealthRouter from this branch; callers set this flag for
            # privacy, proof isolation, or offline operation.
            if _is_user_facing:
                # [BUG FIX] Force-kill stuck worker and drain queues IMMEDIATELY.
                # Without this, the old worker's IPC feeder threads stay blocked on
                # nwait(), starving the event loop and causing tick stalls that kill
                # the WebSocket connection. The recovery task below will respawn cleanly.
                if self._mlx_client and hasattr(self._mlx_client, "_process"):
                    try:
                        proc = self._mlx_client._process
                        is_running = _worker_process_is_running(proc)
                        # A worker actively LOADING the 20GB model is running but
                        # NOT stuck — killing it here is the doom loop that
                        # starved the cortex for a full hour (2026-07-15 soak:
                        # spawn → load → killed mid-warmup by this block on the
                        # next turn → warmup_deferred → repeat, 216s/turn, 0
                        # real cortex answers). Only genuinely wedged workers get
                        # killed: warmup NOT in flight (idle-but-running = a
                        # hung generation, the original nwait bug) OR warmup
                        # in flight but past a generous load deadline.
                        legitimately_loading = self._cortex_worker_is_legitimately_loading(
                            self._mlx_client
                        )
                        if legitimately_loading:
                            logger.info(
                                "⏳ [CASCADE CLEANUP] Cortex worker pid=%s is warming; "
                                "NOT killing a loading model.",
                                getattr(proc, "pid", "unknown"),
                            )
                        if not legitimately_loading:
                            if proc and is_running:
                                # A worker that was still warming/recovering when it
                                # overran the load deadline is a stuck LOAD (thermal /
                                # GPU-starved), not the idle-but-wedged nwait case —
                                # only stuck loads feed the warmup-backoff so the
                                # cortex stops thrashing the GPU the fallback needs.
                                was_stuck_load = bool(
                                    getattr(self._mlx_client, "_warmup_in_flight", False)
                                ) or str(getattr(self._mlx_client, "_lane_state", "")) in {
                                    "warming",
                                    "recovering",
                                }
                                logger.warning(
                                    "🧹 [CASCADE CLEANUP] Force-killing stuck cortex worker pid=%s",
                                    getattr(proc, "pid", "unknown"),
                                )
                                proc.kill()
                                if hasattr(proc, "join"):
                                    proc.join(timeout=2.0)
                                elif hasattr(proc, "wait"):
                                    proc.wait(timeout=2.0)
                                if was_stuck_load:
                                    self._note_cortex_stuck_kill()
                            if hasattr(self._mlx_client, "_drain_queue"):
                                self._mlx_client._drain_queue()
                            # Replace queues to sever any stuck feeder threads.
                            replace_queues = getattr(self._mlx_client, "_replace_ipc_queues", None)
                            if callable(replace_queues):
                                replace_queues()
                            self._mlx_client._process = None
                            self._mlx_client._init_done = False
                            logger.info("🧹 [CASCADE CLEANUP] Stuck worker killed, queues replaced.")
                    except _INFERENCE_RECOVERABLE_ERRORS as cleanup_exc:
                        record_degradation(
                            "inference_gate",
                            cleanup_exc,
                            severity="warning",
                            action="continued recovery scheduling after cascade cleanup error",
                        )
                        logger.debug("Cascade cleanup error (non-fatal): %s", cleanup_exc)
                # Force cortex recovery in background
                if not self._cortex_recovery_in_progress:
                    recovery_coro = self._respawn_cortex_if_needed()
                    task = get_task_tracker().create_task(recovery_coro)
                    if not isinstance(task, asyncio.Task):
                        recovery_coro.close()
                # Give cortex time to recover before next request hits a dead endpoint
                self._extend_startup_quiet_window(15.0)
                # Reset the UnitaryResponsePhase circuit breaker so next attempt works
                try:
                    from core.resilience.error_boundary import CircuitRegistry
                    from core.utils.resilience import CircuitState

                    breaker = CircuitRegistry.get_instance().get_breaker(
                        "phase:UnitaryResponsePhase"
                    )
                    if breaker.state != CircuitState.CLOSED:
                        breaker.state = CircuitState.HALF_OPEN
                        breaker.reset_timeout = min(breaker.reset_timeout, 15.0)
                        logger.info("Reset UnitaryResponsePhase circuit to HALF_OPEN for recovery")
                except _INFERENCE_RECOVERABLE_ERRORS as exc:
                    logger.debug("Circuit-breaker recovery reset unavailable: %s", exc)
                recovery_text = self._user_facing_recovery_response(visible_user_prompt)
                return self._finalize_nonlocal_user_facing_text(
                    recovery_text,
                    visible_user_prompt,
                    is_user_facing=True,
                    label="offline-recovery",
                    max_tokens=max_tokens,
                    output_contract=output_contract_payload,
                )
            return None

        if time.monotonic() < self._cloud_backoff_until:
            logger.warning("Cloud fallback cooling down. Skipping remote retry.")
            if _is_user_facing:
                recovery_text = self._user_facing_recovery_response(visible_user_prompt)
                return self._finalize_nonlocal_user_facing_text(
                    recovery_text,
                    visible_user_prompt,
                    is_user_facing=True,
                    label="cloud-backoff-recovery",
                    max_tokens=max_tokens,
                    output_contract=output_contract_payload,
                )
            return None

        # Resolve the recoverable error surface BEFORE the try so the handler
        # can catch it directly — provider SDK error types vary by
        # installation, but that is a reason to build the tuple dynamically,
        # not to catch Exception (the causal-gating ratchet forbids raw broad
        # catches in this file, and it is right).
        from core.brain.llm.cloud_errors import cloud_call_error_types

        recoverable_cloud_errors = (*_INFERENCE_RECOVERABLE_ERRORS, *cloud_call_error_types())
        try:
            from core.container import ServiceContainer

            # PII SCRUBBING: Strip personal identifiers before sending to cloud.
            # biography_private.json data (real names, trust scores, relationship
            # labels) must never leave the local machine. The scrubber replaces
            # PII with neutral replacements while preserving conversational context.
            scrubbed_payload = self._scrub_cloud_payload(system_prompt, prompt)
            if scrubbed_payload is None:
                if not _is_user_facing:
                    return None
                recovery_text = self._user_facing_recovery_response(visible_user_prompt)
                return self._finalize_nonlocal_user_facing_text(
                    recovery_text,
                    visible_user_prompt,
                    is_user_facing=True,
                    label="cloud-privacy-recovery",
                    max_tokens=max_tokens,
                    output_contract=output_contract_payload,
                )
            cloud_system_prompt, cloud_prompt = scrubbed_payload

            # Try APIAdapter first (cleaner Gemini integration)
            adapter = ServiceContainer.get("api_adapter", default=None)
            if adapter and getattr(adapter, "has_gemini", False):
                logger.info("☁️ Falling back to Gemini via APIAdapter...")
                adapter_options = {
                    "model_tier": "api_fast",
                    "max_tokens": min(800, max(1, int(max_tokens))),
                    "temperature": 0.7,
                    "cloud_only": True,
                    "purpose": "user_cloud_recovery",
                }
                adapter_prompt = f"{cloud_system_prompt}\n\nUser: {cloud_prompt}\nAura:"
                metadata_generate = getattr(adapter, "generate_with_metadata", None)
                try:
                    if callable(metadata_generate):
                        adapter_result = await asyncio.wait_for(
                            metadata_generate(adapter_prompt, adapter_options),
                            timeout=30.0,
                        )
                    else:
                        logger.error(
                            "APIAdapter lacks structured provider metadata; refusing cloud fallback."
                        )
                        adapter_result = {
                            "ok": False,
                            "text": "",
                            "endpoint": "APIAdapter-cloud-unverified",
                            "provider": "unknown",
                            "model": "",
                            "is_local": None,
                            "provider_verified": False,
                            "fallback_chain": [],
                            "error": "structured_cloud_provenance_unavailable",
                        }
                except recoverable_cloud_errors as adapter_err:
                    record_degradation(
                        "inference_gate",
                        adapter_err,
                        severity="warning",
                        action=(
                            "continued to HealthRouter after APIAdapter cloud provider failed"
                        ),
                    )
                    adapter_error_text = str(adapter_err)
                    if "429" in adapter_error_text or "quota" in adapter_error_text.lower():
                        self._cloud_backoff_until = time.monotonic() + 60.0
                    logger.warning(
                        "APIAdapter cloud fallback failed; continuing to HealthRouter: %s",
                        adapter_err,
                    )
                    adapter_result = {
                        "ok": False,
                        "text": "",
                        "endpoint": "APIAdapter-cloud-error",
                        "provider": "unknown",
                        "model": "",
                        "is_local": None,
                        "provider_verified": False,
                        "fallback_chain": [
                            {
                                "endpoint": "APIAdapter",
                                "status": "error",
                                "error_type": type(adapter_err).__name__,
                            }
                        ],
                        "error": type(adapter_err).__name__,
                    }
                result = (
                    str(adapter_result.get("text") or "")
                    if isinstance(adapter_result, dict)
                    else ""
                )
                adapter_is_cloud = _verified_cloud_generation_metadata(
                    adapter_result,
                    endpoint_prefix="Gemini-APIAdapter:",
                )
                if result.strip() and adapter_is_cloud:
                    endpoint = str(
                        adapter_result.get("endpoint") or "APIAdapter-cloud-unverified"
                    )
                    finalized_result = self._finalize_nonlocal_user_facing_text(
                        result.strip(),
                        visible_user_prompt,
                        is_user_facing=_is_user_facing,
                        label=endpoint,
                        max_tokens=max_tokens,
                        output_contract=output_contract_payload,
                        generation_metadata=adapter_result,
                    )
                    try:
                        from core.consciousness.closed_loop import notify_closed_loop_output

                        notify_closed_loop_output(finalized_result)
                    except _INFERENCE_RECOVERABLE_ERRORS as exc:
                        record_degradation(
                            "inference_gate",
                            exc,
                            severity="warning",
                            action="returned cloud result without closed-loop output notification",
                        )
                        logger.debug("Cloud output notification skipped: %s", exc)
                    return finalized_result
                if result.strip() and not adapter_is_cloud:
                    logger.error(
                        "APIAdapter cloud-only fallback returned a local provider; rejecting mislabeled result."
                    )

            # Try HealthRouter as secondary cloud path (also PII-scrubbed)
            router = ServiceContainer.get("llm_router", default=None)
            metadata_generate = getattr(router, "generate_with_metadata", None)
            if router and callable(metadata_generate):
                logger.info("☁️ Falling back to HealthRouter...")
                try:
                    router_result = await asyncio.wait_for(
                        metadata_generate(
                            cloud_prompt,
                            system_prompt=cloud_system_prompt,
                            prefer_tier="api_fast",
                            max_tokens=max_tokens,
                            origin="inference_gate_cloud_fallback",
                            purpose="user_cloud_recovery",
                            foreground_request=True,
                            protected_foreground_lane=True,
                            is_background=False,
                            allow_cloud_fallback=True,
                            cloud_only=True,
                            skip_runtime_payload=True,
                            requested_output_contract=(
                                dict(output_contract_payload)
                                if isinstance(output_contract_payload, dict)
                                else None
                            ),
                            semantic_output_token_cap=output_contract.semantic_token_cap,
                            hard_output_token_ceiling=output_contract.hard_token_ceiling,
                        ),
                        timeout=30.0,
                    )
                except recoverable_cloud_errors as router_err:
                    record_degradation(
                        "inference_gate",
                        router_err,
                        severity="warning",
                        action=(
                            "continued to exhausted-inference recovery after HealthRouter failed"
                        ),
                    )
                    router_error_text = str(router_err)
                    if "429" in router_error_text or "quota" in router_error_text.lower():
                        self._cloud_backoff_until = time.monotonic() + 60.0
                    logger.warning("HealthRouter cloud fallback failed: %s", router_err)
                    router_result = {
                        "ok": False,
                        "text": "",
                        "endpoint": "HealthRouter-cloud-error",
                        "provider_verified": False,
                        "error": type(router_err).__name__,
                    }
                result = (
                    str(router_result.get("text") or "")
                    if isinstance(router_result, dict)
                    else ""
                )
                router_is_cloud = _verified_cloud_generation_metadata(router_result)
                if result.strip() and router_is_cloud:
                    endpoint = str(router_result.get("endpoint") or "HealthRouter-cloud")
                    finalized_result = self._finalize_nonlocal_user_facing_text(
                        result.strip(),
                        visible_user_prompt,
                        is_user_facing=_is_user_facing,
                        label=endpoint,
                        max_tokens=max_tokens,
                        output_contract=output_contract_payload,
                        generation_metadata=router_result,
                    )
                    try:
                        from core.consciousness.closed_loop import notify_closed_loop_output

                        notify_closed_loop_output(finalized_result)
                    except _INFERENCE_RECOVERABLE_ERRORS as exc:
                        record_degradation(
                            "inference_gate",
                            exc,
                            severity="warning",
                            action="returned router cloud result without closed-loop output notification",
                        )
                        logger.debug("Router output notification skipped: %s", exc)
                    return finalized_result
                if result.strip() and not router_is_cloud:
                    logger.error(
                        "HealthRouter cloud-only fallback returned a local provider; rejecting result."
                    )
        except recoverable_cloud_errors as cloud_err:
            record_degradation(
                "inference_gate",
                cloud_err,
                severity="degraded",
                action="entered cloud backoff when applicable and returned exhausted-inference fallback",
            )
            cloud_err_text = str(cloud_err)
            if "429" in cloud_err_text or "quota" in cloud_err_text.lower():
                self._cloud_backoff_until = time.monotonic() + 60.0
            logger.error("☁️ Cloud fallback failed: %s", cloud_err)

        # All inference paths exhausted. Return None so callers can handle
        # gracefully without the error text leaking to TTS or the user.
        logger.error("All inference paths exhausted (Local + Cloud)")
        if _is_user_facing:
            recovery_text = self._user_facing_recovery_response(visible_user_prompt)
            return self._finalize_nonlocal_user_facing_text(
                recovery_text,
                visible_user_prompt,
                is_user_facing=True,
                label="exhausted-inference-recovery",
                max_tokens=max_tokens,
                output_contract=output_contract_payload,
            )
        return None

    def _post_inference_update(self, response_text: str):
        """Update downstream systems after each inference completes.

        Closes the bidirectional causal loop:
          CRSM ← response (updates self-state)
          HOT  ← response (reflexive modification)
          Hedonic ← response quality signal
        """
        # An empty response is NOT a success — recording a fresh success
        # timestamp for it would feed health and recovery logic false
        # evidence that generation is working.
        if not response_text or not response_text.strip():
            return
        self._last_successful_generation_at = time.time()
        try:
            from core.consciousness.crsm import get_crsm

            get_crsm().post_inference_update(response_text)
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="skipped unavailable post-inference update hook after response delivery",
            )
            logger.debug("Suppressed Exception: %s", _exc)
        try:
            from core.consciousness.hot_engine import get_hot_engine

            hot = get_hot_engine()
            hot.apply_feedback()  # apply any pending reflexive modifications
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="skipped unavailable post-inference update hook after response delivery",
            )
            logger.debug("Suppressed Exception: %s", _exc)
        try:
            from core.consciousness.hedonic_gradient import get_hedonic_gradient
            from core.container import ServiceContainer

            hg = get_hedonic_gradient()
            _v, _a, _c, _e = 0.0, 0.5, 0.5, 0.7
            _circ2 = ServiceContainer.get("affective_circumplex", default=None)
            if _circ2 and hasattr(_circ2, "_sample_raw_axes"):
                _v, _a = _circ2._sample_raw_axes()
            _ls2 = ServiceContainer.get("liquid_state", default=None)
            if _ls2 and hasattr(_ls2, "get_status"):
                _lsd2 = _ls2.get_status()
                _c = float(_lsd2.get("curiosity", 50)) / 100.0
                _e = float(_lsd2.get("energy", 70)) / 100.0
            hg.update(valence=_v, arousal=_a, curiosity=_c, energy=_e)
            # LoRA Bridge: complete the post-inference capture
            try:
                from core.consciousness.crsm_lora_bridge import get_crsm_lora_bridge

                get_crsm_lora_bridge().post_inference_capture(
                    response_text=response_text,
                    hedonic_after=hg.score,
                )
            except _INFERENCE_RECOVERABLE_ERRORS as _exc:
                _record_inference_degradation(
                    _exc,
                    action="skipped unavailable post-inference update hook after response delivery",
                )
                logger.debug("Suppressed Exception: %s", _exc)
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="skipped unavailable post-inference update hook after response delivery",
            )
            logger.debug("Suppressed Exception: %s", _exc)

        # ══════════════════════════════════════════════════════════════════
        # DEEPENED POST-INFERENCE FEEDBACK LOOPS
        # ══════════════════════════════════════════════════════════════════

        # ── Credit Assignment: Record response quality ────────────────────
        try:
            credit = ServiceContainer.get("credit_assignment", default=None)
            if credit:
                # SURFACE-SHAPE heuristic only: length and structure say
                # nothing about correctness, so the signal is capped well
                # below max credit and labeled as a shape proxy. Verified
                # outcomes (task verifiers, user feedback) are the only
                # sources allowed to assign full credit.
                response_len = len(response_text.strip())
                has_structure = any(marker in response_text for marker in ["\n", "- ", "1.", "```"])
                quality = min(0.6, (response_len / 500.0) * 0.4 + (0.2 if has_structure else 0.05))
                credit.assign_credit(
                    # time_ns + counter keeps concurrent same-second responses
                    # attributable instead of colliding on one id.
                    action_id=f"inference_{time.time_ns()}_{self._credit_action_seq()}",
                    outcome=quality,
                    domain="chat",
                )
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="skipped unavailable post-inference update hook after response delivery",
            )
            logger.debug("Suppressed Exception in credit feedback: %s", _exc)

        # ── Homeostasis: Response success signal ──────────────────────────
        try:
            homeostasis = ServiceContainer.get("homeostasis", default=None)
            if homeostasis and hasattr(homeostasis, "on_response_success"):
                homeostasis.on_response_success(response_length=len(response_text))
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="skipped unavailable post-inference update hook after response delivery",
            )
            logger.debug("Suppressed Exception in homeostasis feedback: %s", _exc)

        # ── World Model: Extract beliefs from response ────────────────────
        try:
            world_model = ServiceContainer.get("epistemic_state", default=None)
            if world_model and hasattr(world_model, "extract_beliefs_from_response"):
                if len(response_text) > 100:
                    world_model.extract_beliefs_from_response(response_text)
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="skipped unavailable post-inference update hook after response delivery",
            )
            logger.debug("Suppressed Exception in world model feedback: %s", _exc)

        # ── Synaptic Plasticity: Post-inference weight update ─────────────
        # This is where true online learning happens. The reward signal from
        # hedonic gradient + CRSM surprise modulates the Hebbian update.
        try:
            _plasticity = ServiceContainer.get("synaptic_plasticity", default=None)
            if _plasticity is not None:
                _hg_score = 0.0
                _surprise = 0.0
                try:
                    from core.consciousness.hedonic_gradient import get_hedonic_gradient
                    _hg_score = get_hedonic_gradient().score
                except _INFERENCE_RECOVERABLE_ERRORS as _hedonic_exc:
                    _record_inference_degradation(
                        _hedonic_exc,
                        action="continued synaptic plasticity post-inference update without hedonic score",
                    )
                    logger.debug(
                        "SynapticPlasticity post-inference hedonic score unavailable: %s",
                        _hedonic_exc,
                    )
                try:
                    from core.consciousness.crsm import get_crsm
                    _crsm = get_crsm()
                    _surprise = getattr(_crsm, "surprise", 0.0)
                except _INFERENCE_RECOVERABLE_ERRORS as _crsm_exc:
                    _record_inference_degradation(
                        _crsm_exc,
                        action="continued synaptic plasticity post-inference update without CRSM surprise",
                    )
                    logger.debug(
                        "SynapticPlasticity post-inference CRSM surprise unavailable: %s",
                        _crsm_exc,
                    )
                _plasticity.post_inference_learn(
                    response_text=response_text,
                    hedonic_after=_hg_score,
                    surprise=_surprise,
                )
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="skipped synaptic plasticity post-inference learning",
            )
            logger.debug("Suppressed Exception in plasticity feedback: %s", _exc)

        # ── Temporal Continuity: Reset silence accumulator ────────────────
        # The inference just completed — reset the temporal residue so the
        # next silence period starts accumulating from a fresh anchor.
        try:
            _tc = ServiceContainer.get("temporal_continuity", default=None)
            if _tc is not None:
                _tc.on_inference_complete()
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="skipped temporal continuity post-inference reset",
            )
            logger.debug("Suppressed Exception in temporal continuity reset: %s", _exc)

    async def think(self, prompt: str, system_prompt: str = "", **kwargs) -> str | None:
        """Unified thinking interface for cognitive components.

        Preserve standard LLM adapter semantics:
        - explicit ``messages`` stay as passthrough chat messages
        - ``system_prompt`` is treated as a real system prompt by default
        - callers that truly mean "brief" can pass ``brief=...`` or
          ``system_prompt_is_brief=True``
        """
        timeout = kwargs.pop("timeout", None)
        brief = kwargs.pop("brief", None)
        system_prompt_is_brief = bool(kwargs.pop("system_prompt_is_brief", False))
        provided_messages = kwargs.get("messages")
        if (
            provided_messages is not None
            and system_prompt
            and not system_prompt_is_brief
        ):
            system_text = str(system_prompt or "").strip()
            if system_text:
                merged_messages: list[Any] = []
                inserted_system = False
                for raw_msg in provided_messages if isinstance(provided_messages, list) else []:
                    msg = dict(raw_msg) if isinstance(raw_msg, dict) else raw_msg
                    if (
                        isinstance(msg, dict)
                        and str(msg.get("role", "") or "").strip().lower() == "system"
                        and not inserted_system
                    ):
                        existing = str(msg.get("content", "") or "").strip()
                        if existing == system_text or existing.startswith(f"{system_text}\n\n"):
                            msg["content"] = existing
                        elif existing:
                            msg["content"] = f"{system_text}\n\n{existing}"
                        else:
                            msg["content"] = system_text
                        inserted_system = True
                    merged_messages.append(msg)
                if not inserted_system:
                    merged_messages.insert(0, {"role": "system", "content": system_text})
                provided_messages = merged_messages

        context: dict[str, Any] = {}
        if provided_messages is not None:
            context["messages"] = provided_messages
        elif brief is not None:
            context["brief"] = brief
        elif system_prompt and not system_prompt_is_brief:
            context["messages"] = [
                {"role": "system", "content": str(system_prompt)},
                {"role": "user", "content": str(prompt or "")},
            ]
        else:
            context["brief"] = system_prompt

        for key in (
            "history",
            "messages",
            "max_tokens",
            "temperature",
            "temp",
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "repetition_context_size",
            "presence_penalty",
            "stop_sequences",
            "schema",
            "deep_handoff",
            "allow_cloud_fallback",
            "prefer_tier",
            "origin",
            "purpose",
            "is_background",
            "foreground_request",
            "benchmark_request",
            "protected_foreground_lane",
            "proof_primary_lane_required",
            "proof_model_tier",
            "strict_answer_contract",
            "strict_value_contract",
            "proof_evaluation_contract",
            "operator_evidence_contract",
            "web_interlocutor_contract",
            "cognitive_engine_required",
            "desktop_cognitive_engine_required",
            "live_runtime_payload_required",
            "visible_user_message",
            "current_user_message",
            "recent_conversation_context",
            "recent_context_needed",
            "desktop_quick_reply_contract",
            "capability_inventory_contract",
            "desktop_execution_contract",
            "memory_state_contract",
            "runtime_fact_status_contract",
            "grounded_runtime_status_contract",
            "canonical_memory_state_evidence",
            "response_style_contract",
            "live_speech_grounding_frame",
            "live_mind_controls_bound",
            "live_mind_generation_controls",
            "live_mind_snapshot_ready",
            "live_mind_required_subsystems_ok",
            "allow_mesh_cognition",
            "clean_user_surface_contract",
            "user_surface_validation_prompt",
            "clean_user_surface_steering_alpha",
            "clean_user_surface_recurrent_loops",
            "disable_prompt_cache",
            "clear_prompt_cache",
            "health_probe",
            "allow_tools",
            "state",
            "sampling_bias",
            "imagination_sampling_bias",
            "bicameral_sampling_bias",
            "skip_runtime_payload",
        ):
            if key in kwargs:
                context[key] = kwargs[key]
        result = await self.generate(prompt, context=context, timeout=timeout)
        if isinstance(result, str) and result.strip():
            # Close bidirectional causal loop after each inference
            self._post_inference_update(result)
            return result
        return None

    def is_alive(self) -> bool:
        """Check if the InferenceGate and MLX client are operational."""
        if not self._initialized:
            return False
        # If MLX client is alive, we are operational
        if self._mlx_client and hasattr(self._mlx_client, "is_alive") and self._mlx_client.is_alive():
            return True
        # If we are in safe boot or deferred mode, the MLX worker starts on first request,
        # so InferenceGate itself is operational even if the worker isn't running yet.
        if self._desktop_safe_boot_enabled() or self._boot_should_schedule_deferred_prewarm():
            return True
        return False

    def is_inference_ready(self) -> bool:
        """Return true only when a concrete inference backend is live now.

        ``is_alive()`` intentionally supports deferred/safe-boot semantics so a
        desktop turn can cold-start Cortex on demand. The runtime health
        contract is stricter: healthy must mean an actual backend can accept a
        generation without relying on deferred startup. Proof-primary runs also
        require the primary Cortex lane specifically, not a lower-tier fallback.
        """
        if not self._initialized:
            return False

        proof_active = False
        proof_policy_unknown = False
        try:
            from core.runtime.proof_policy import proof_run_active

            proof_active = bool(proof_run_active(origin="inference_gate_health"))
        except (ImportError, RuntimeError, AttributeError) as exc:
            # Fail CLOSED: an unreadable proof policy must not silently grant
            # the permissive non-proof readiness shortcuts below.
            proof_policy_unknown = True
            _record_inference_degradation(
                exc,
                action="treated proof policy as unknown during inference readiness check",
            )
            logger.debug("Inference readiness proof-policy check unavailable: %s", exc)

        # A resident backend is already stronger evidence than a deferred-boot
        # policy probe.  Checking the policy first caused every health poll to
        # rerun the RAM-admission calculation after Cortex was loaded, flooding
        # the neural stream with "deferred prewarm refused" warnings even though
        # there was nothing left to prewarm. Proof runs retain the stricter
        # endpoint-specific checks below.
        if not proof_active and not proof_policy_unknown:
            try:
                if (
                    self._mlx_client is not None
                    and hasattr(self._mlx_client, "is_alive")
                    and self._mlx_client.is_alive()
                ):
                    return True
            except _INFERENCE_RECOVERABLE_ERRORS:
                pass

        # NOTE: deferred/safe-boot policy deliberately does NOT satisfy this
        # contract. ``is_alive()`` covers "the gate can cold-start on demand";
        # inference-READY requires a concrete live backend, proven below.

        def _client_alive(client: Any) -> bool:
            try:
                return bool(
                    client is not None
                    and hasattr(client, "is_alive")
                    and client.is_alive()
                )
            except _INFERENCE_RECOVERABLE_ERRORS:
                return False

        def _active_generation_is_progressing(lane: Any) -> bool:
            """Treat bounded, observable foreground work as operational inference.

            A client cannot advertise ``conversation_ready`` while it owns the
            foreground generation lock. That is backpressure, not backend
            failure. Health may accept the in-flight request only while its
            start and token-progress timestamps prove it has not stalled.
            """
            if not isinstance(lane, dict):
                return False
            if int(lane.get("active_generations", 0) or 0) <= 0:
                return False
            if not bool(lane.get("foreground_owned", False)):
                return False

            now = time.time()
            request_started_at = float(lane.get("current_request_started_at", 0.0) or 0.0)
            if request_started_at <= 0.0:
                return False
            request_age_s = max(0.0, now - request_started_at)
            try:
                startup_grace_s = max(
                    15.0,
                    float(
                        os.environ.get("AURA_INFERENCE_ACTIVE_STARTUP_GRACE_S", "120")
                        or 120.0
                    ),
                )
                progress_stale_s = max(
                    10.0,
                    float(
                        os.environ.get("AURA_INFERENCE_ACTIVE_PROGRESS_STALE_S", "45")
                        or 45.0
                    ),
                )
            except (TypeError, ValueError):
                startup_grace_s = 120.0
                progress_stale_s = 45.0
            token_progress_at = float(lane.get("last_token_progress_at", 0.0) or 0.0)
            if token_progress_at > 0.0:
                return max(0.0, now - token_progress_at) <= progress_stale_s
            return request_age_s <= startup_grace_s

        primary_ready = _client_alive(self._mlx_client)
        if primary_ready:
            try:
                lane = self.get_conversation_status()
            except _INFERENCE_RECOVERABLE_ERRORS:
                return False
            if bool(lane.get("conversation_ready", False)) or _active_generation_is_progressing(lane):
                return True

        try:
            from core.runtime.proof_policy import proof_model_tier, proof_run_active

            if proof_run_active(origin="inference_gate_health") and proof_model_tier() == "primary":
                return False
        except _INFERENCE_RECOVERABLE_ERRORS:
            return False

        try:
            local_clients = self._iter_local_clients()
        except _INFERENCE_RECOVERABLE_ERRORS:
            local_clients = {}
        for client in local_clients.values():
            if not _client_alive(client):
                continue
            get_lane_status = getattr(client, "get_lane_status", None)
            if not callable(get_lane_status):
                return True
            try:
                lane = get_lane_status()
            except _INFERENCE_RECOVERABLE_ERRORS:
                continue
            if bool(isinstance(lane, dict) and lane.get("conversation_ready", False)):
                return True
            if _active_generation_is_progressing(lane):
                return True
        return False
