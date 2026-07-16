from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import math
import os
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, cast

from core.container import ServiceContainer
from core.memory.retention_policy import working_history_retention_policy
from core.runtime.errors import Severity, record_degradation
from core.runtime.flags import FlagKind, declare
from core.runtime.receipts import WorkspaceGateReceipt, get_receipt_store
from core.utils.task_tracker import get_task_tracker

if TYPE_CHECKING:
    from core.resilience.inhibition_manager import InhibitionManager

logger = logging.getLogger("Consciousness.GlobalWorkspace")

_WORKSPACE_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    OSError,
    ConnectionError,
    TimeoutError,
    TypeError,
    ValueError,
)
_INHIBITION_GATE_DEGRADATION_PHASES = frozenset(
    {
        "global_inhibition_lookup",
        "global_inhibition_check",
        "global_inhibition_check_cancelled",
        "global_inhibition_revalidation_lookup",
        "global_inhibition_revalidation",
        "global_inhibition_revalidation_cancelled",
    }
)
_INHIBITION_GATE_TIMEOUT_FLAG = declare(
    "AURA_WORKSPACE_INHIBITION_GATE_TIMEOUT_S",
    kind=FlagKind.FLOAT,
    default=0.5,
    description="Maximum time allowed for the workspace global-inhibition safety gate",
    owner="core.consciousness.global_workspace",
)
_MAX_STRUCTURED_SIGNAL_BYTES = 64 * 1024
_MAX_STRUCTURED_SIGNAL_PREVIEW_CHARS = 4096


def _record_workspace_degradation(
    error: BaseException,
    *,
    phase: str,
    action: str,
    severity: Severity = "warning",
) -> None:
    record_degradation(
        "global_workspace",
        error,
        severity=severity,
        action=action,
        extra={"phase": phase},
        enforce_failure_policy=False,
    )


def _error_summary(error: BaseException) -> str:
    return f"{type(error).__qualname__}: {error}"[:240]


def _emit_workspace_gate_receipt(receipt: WorkspaceGateReceipt) -> WorkspaceGateReceipt:
    emitted = get_receipt_store().emit(receipt)
    if not isinstance(emitted, WorkspaceGateReceipt):
        raise TypeError("workspace gate receipt store returned the wrong receipt type")
    return emitted



# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ContentType(Enum):
    """Types of cognitive content for workspace processing."""
    UNKNOWN = auto()
    PERCEPTUAL = auto()
    AFFECTIVE = auto()
    MEMORIAL = auto()
    INTENTIONAL = auto()
    LINGUISTIC = auto()
    SOMATIC = auto()
    SOCIAL = auto()
    META = auto()


@dataclass(order=True)
class WorkItem:
    """Backward compatibility for legacy AttentionSummarizer."""
    priority: float
    ts: float = field(compare=False)
    id: str = field(compare=False)
    source: str = field(compare=False)
    payload: dict[str, Any] = field(compare=False)
    reason: str | None = field(compare=False)


class HistoryBuffer:
    """Fixed-size history with deque performance and list-like slices.

    Ported from the retired core/global_workspace.py, whose design was better
    than the canonical's here: the canonical kept a plain list and truncated it
    inside ``publish``, so the bound held only for the one path that remembered
    to enforce it. A buffer that cannot exceed its own limit is the difference
    between a bound and a convention — and an unbounded workspace history on a
    long autonomous run is a slow leak, which is the failure mode Aura is least
    able to notice about itself.
    """

    def __init__(self, maxlen: int, items: Iterable[Any] | None = None):
        self.maxlen = maxlen
        self._items: deque[Any] = deque(items or [], maxlen=maxlen)

    def append(self, item: Any) -> None:
        self._items.append(item)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)

    def __getitem__(self, index):
        return list(self._items)[index]

    def __bool__(self) -> bool:
        return bool(self._items)


@dataclass
class CognitiveCandidate:
    """A bid for the global workspace broadcast slot.
    Any subsystem can submit one each tick.
    """

    content: str                       # What wants to be broadcast
    source: str                        # e.g. "drive_curiosity", "affect_distress", "memory"
    priority: float                    # 0.0–1.0 base weight
    content_type: ContentType = ContentType.UNKNOWN
    affect_weight: float = 0.0        # Emotional urgency boost (from AffectEngine)
    focus_bias: float = 0.0           # Priority boost for focused attention (from AttentionSchema)
    submitted_at: float = field(default_factory=time.time)
    gate_instance_id: str = field(default="", repr=False)
    gate_checked_at: float = field(default=0.0, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def salience(self) -> float:
        """Alias for effective_priority for downstream compatibility."""
        return self.effective_priority

    @property
    def effective_priority(self) -> float:
        """Priority decays slightly with age (recent events are more salient)."""
        age = time.time() - self.submitted_at
        recency = max(0.0, 1.0 - (age / 10.0))  # Full weight within 10s, then decays
        
        # Free Energy dynamic gating
        fe_bias = 0.0
        try:
            from core.consciousness.free_energy import get_free_energy_engine
            fe_engine = get_free_energy_engine()
            if fe_engine and fe_engine.current:
                fe_state = fe_engine.current
                dom_action = fe_state.dominant_action
                fe_val = fe_state.free_energy
                
                # High free energy makes the gate much more selective (higher boost for aligned action)
                boost_magnitude = 0.25 * fe_val
                
                aligned = False
                src = self.source.lower()
                ct = self.content_type
                
                if dom_action == "update_beliefs":
                    if ct == ContentType.MEMORIAL or any(x in src for x in ("belief", "memory", "epistemic", "prediction")):
                        aligned = True
                elif dom_action == "act_on_world":
                    if ct == ContentType.INTENTIONAL or any(x in src for x in ("motivation", "action", "goal", "agency")):
                        aligned = True
                elif dom_action == "explore":
                    if ct == ContentType.PERCEPTUAL or any(x in src for x in ("curiosity", "exploration", "perceptual", "search")):
                        aligned = True
                elif dom_action == "reflect":
                    if ct == ContentType.META or any(x in src for x in ("hot", "reflection", "self", "identity")):
                        aligned = True
                elif dom_action == "engage":
                    if ct in (ContentType.LINGUISTIC, ContentType.SOCIAL) or any(x in src for x in ("chat", "user", "linguistic", "social")):
                        aligned = True
                elif dom_action == "rest":
                    if ct == ContentType.SOMATIC or any(x in src for x in ("soma", "sleep", "rest")):
                        aligned = True
                        
                if aligned:
                    fe_bias = boost_magnitude
        except _WORKSPACE_RECOVERABLE_ERRORS as exc:
            _record_workspace_degradation(
                exc,
                phase="free_energy_priority",
                action="Skipped free-energy priority bias and used base salience only",
                severity="debug",
            )

        return min(1.0, (self.priority + self.affect_weight * 0.3 + self.focus_bias + fe_bias) * (0.7 + 0.3 * recency))



@dataclass
class BroadcastEvent:
    """The formal event emitted on a workspace competition win.
    Compatible with PhenomenologicalExperiencer.
    """
    winners: list[CognitiveCandidate]
    timestamp: float = field(default_factory=time.time)


@dataclass
class BroadcastRecord:
    winner: CognitiveCandidate
    losers: list[str]          # source names of losers
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class SomaticImpulse:
    """A bounded non-optimal workspace bid produced by bodily noise.

    This is not a bypass around governance. It only creates a candidate for the
    same workspace competition as every other subsystem. Downstream action still
    has to pass Will/Authority/tool gates.
    """

    content: str
    priority: float
    reason: str
    content_type: ContentType = ContentType.SOMATIC


class SomaticNoiseInjector:
    """Injects rare, bounded, non-goal-maximizing candidates into workspace."""

    DEFAULT_IMPULSES: tuple[tuple[str, str, ContentType], ...] = (
        ("look again at a recent percept before assuming the world is stable", "perceptual_recheck", ContentType.PERCEPTUAL),
        ("write a brief private reflection about the current internal texture", "reflection_whim", ContentType.META),
        ("inspect one recent file or log because it feels slightly salient", "environmental_curiosity", ContentType.INTENTIONAL),
        ("hold a strange analogy and see whether it connects two unrelated ideas", "creative_association", ContentType.META),
        ("notice whether the current plan is becoming too optimized and brittle", "anti_brittleness_impulse", ContentType.META),
    )

    def __init__(
        self,
        *,
        rng: random.Random | None = None,
        rate: float | None = None,
        max_priority: float | None = None,
        min_ticks_between: int | None = None,
    ) -> None:
        self.rng = rng or random.Random()
        self.rate = self._bounded_float(
            os.environ.get("AURA_SOMATIC_NOISE_RATE"),
            0.035 if rate is None else rate,
            minimum=0.0,
            maximum=0.35,
        )
        self.max_priority = self._bounded_float(
            os.environ.get("AURA_SOMATIC_NOISE_MAX_PRIORITY"),
            0.72 if max_priority is None else max_priority,
            minimum=0.05,
            maximum=0.9,
        )
        self.min_ticks_between = int(
            self._bounded_float(
                os.environ.get("AURA_SOMATIC_NOISE_MIN_TICKS"),
                30 if min_ticks_between is None else min_ticks_between,
                minimum=1,
                maximum=10_000,
            )
        )
        self.enabled = os.environ.get("AURA_SOMATIC_NOISE", "1").strip().lower() not in {"0", "false", "off", "no"}
        self.last_impulse: SomaticImpulse | None = None
        self.injected_count = 0
        self._last_injected_tick = 0

    @staticmethod
    def _bounded_float(value: Any, default: float, *, minimum: float, maximum: float) -> float:
        try:
            parsed = float(value if value is not None else default)
        except (TypeError, ValueError, OverflowError):
            parsed = default
        return max(minimum, min(maximum, parsed))

    def maybe_generate(self, *, tick: int, candidate_count: int, inhibited_sources: set[str]) -> SomaticImpulse | None:
        if not self.enabled or candidate_count >= GlobalWorkspace._MAX_CANDIDATES:
            return None
        if "somatic_noise" in inhibited_sources:
            return None
        force = os.environ.get("AURA_SOMATIC_NOISE_FORCE", "0").strip().lower() in {"1", "true", "on", "yes"}
        if not force and tick - self._last_injected_tick < self.min_ticks_between:
            return None
        if not force and self.rng.random() > self.rate:
            return None
        content, reason, content_type = self.rng.choice(self.DEFAULT_IMPULSES)
        jitter = self.rng.uniform(-0.08, 0.08)
        priority = max(0.18, min(self.max_priority, 0.48 + jitter))
        impulse = SomaticImpulse(
            content=f"somatic impulse t{tick}: {content}",
            priority=round(priority, 4),
            reason=reason,
            content_type=content_type,
        )
        self.last_impulse = impulse
        self.injected_count += 1
        self._last_injected_tick = tick
        return impulse


# ---------------------------------------------------------------------------
# Processor registration type
# ---------------------------------------------------------------------------

ProcessorFn = Callable[
    [BroadcastEvent | CognitiveCandidate],
    Awaitable[Any] | Any,
]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GlobalWorkspace:
    """The competitive bottleneck. One winner per cognitive tick.

    Inhibition model:
      - Losing subsystems are placed in a cooldown dict.
      - They cannot re-submit for _INHIBIT_TICKS ticks.
      - This prevents the same subsystem from dominating every cycle
        and forces genuine competition.
    """

    _INHIBIT_TICKS: int = 1       # One refractory tick keeps competition moving without winner lock-in
    _MAX_CANDIDATES: int = 20     # Hard cap — prevents memory leak if submissions pile up
    _IGNITION_THRESHOLD: float = 0.6  # Priority above which workspace "ignites"
    _PHI_PRIORITY_BOOST: float = 0.15  # Max priority bonus for high-Φ sources
    _SEIZURE_REPLACEMENT_MARGIN: float = 0.05  # Avoid recency-only churn under floods.

    def __init__(self, attention_schema: Any = None):
        self._lock: asyncio.Lock | None = None
        self._candidates: list[CognitiveCandidate] = []
        self._inhibited: dict[str, int] = {}   # source -> ticks_remaining
        self._processors: list[ProcessorFn] = []
        # Retention comes from the shared working-history policy rather than a
        # hardcoded 100, and lives in a self-bounding buffer. Both were the
        # superseded core/global_workspace.py's design; absorbing them here is
        # what makes retiring that module a merge rather than a loss.
        self.max_history: int = working_history_retention_policy(
            "AURA_GLOBAL_WORKSPACE_HISTORY_MAX",
        ).max_items
        self._history: HistoryBuffer = HistoryBuffer(self.max_history)
        self._tick: int = 0
        self.attention_schema: Any = attention_schema
        self.last_winner: CognitiveCandidate | None = None
        
        # [UNITY] Global Inhibition Link
        self._global_inhibition: InhibitionManager | None = None
        
        # --- Ignition Detection (GWT) ---
        self.ignition_level: float = 0.0    # 0.0-1.0 current ignition intensity
        self.ignited: bool = False          # True when ignition_level >= threshold
        self._ignition_count: int = 0       # Total ignition events
        self._current_phi: float = 0.0      # Φ from substrate (updated externally)
        self._degraded_channels: dict[str, str] = {}
        self._degradation_events: list[dict[str, Any]] = []
        self._processor_failures: dict[str, int] = {}
        self._somatic_noise = SomaticNoiseInjector()
        self._inhibition_gate_ready = False
        self._last_inhibition_gate_reason = "not_checked"
        self._gate_rejections: list[dict[str, Any]] = []
        
        logger.info("GlobalWorkspace initialized (ignition_threshold=%.2f).", self._IGNITION_THRESHOLD)

    def _record_degradation(
        self,
        error: BaseException,
        *,
        phase: str,
        action: str,
        severity: Severity = "warning",
    ) -> None:
        summary = _error_summary(error)
        self._degraded_channels[phase] = summary
        self._degradation_events.append(
            {
                "tick": self._tick,
                "phase": phase,
                "severity": severity,
                "error": summary,
                "action": action,
            }
        )
        if len(self._degradation_events) > 50:
            self._degradation_events = self._degradation_events[-50:]
        _record_workspace_degradation(error, phase=phase, action=action, severity=severity)

    @property
    def history(self) -> HistoryBuffer:
        """Broadcast history. Bounded by construction — see HistoryBuffer.

        Iterable, sliceable and len()-able like the list it replaced, so
        AttentionSummarizer and other readers are unaffected.
        """
        return self._history

    @history.setter
    def history(self, value: Iterable[Any]) -> None:
        # Assigning a plain list must not silently unbound the buffer.
        self._history = (
            value
            if isinstance(value, HistoryBuffer)
            else HistoryBuffer(self.max_history, value)
        )

    # ------------------------------------------------------------------
    # Submission API — called by subsystems every heartbeat tick
    # ------------------------------------------------------------------

    async def publish(
        self,
        *,
        priority: float,
        source: str,
        payload: dict[str, Any],
        reason: str = "",
        content_type: ContentType = ContentType.UNKNOWN,
    ) -> bool:
        """Admit a structured signal through the canonical workspace gate.

        Older infrastructure producers publish structured work items. This
        adapter preserves that payload while routing the signal through the
        same inhibition and competition path as every native candidate.
        """
        normalized_source = " ".join(str(source or "").strip().split())[:160]
        if not normalized_source:
            raise ValueError("workspace signal source must be non-empty")
        if not isinstance(payload, dict):
            raise TypeError("workspace signal payload must be a dictionary")
        try:
            normalized_priority = float(priority)
        except (TypeError, ValueError) as exc:
            raise ValueError("workspace signal priority must be numeric") from exc
        if not math.isfinite(normalized_priority):
            raise ValueError("workspace signal priority must be finite")
        normalized_priority = max(0.0, min(1.0, normalized_priority))
        normalized_reason = " ".join(str(reason or "").strip().split())[:500]
        content = normalized_reason or f"Structured workspace signal from {normalized_source}"
        payload_json = json.dumps(payload, ensure_ascii=True, default=str, sort_keys=True)
        payload_bytes = payload_json.encode("utf-8")
        if len(payload_bytes) <= _MAX_STRUCTURED_SIGNAL_BYTES:
            preserved_payload: dict[str, Any] = json.loads(payload_json)
        else:
            preserved_payload = {
                "truncated": True,
                "original_bytes": len(payload_bytes),
                "sha256": hashlib.sha256(payload_bytes).hexdigest(),
                "preview": payload_json[:_MAX_STRUCTURED_SIGNAL_PREVIEW_CHARS],
            }
        metadata = {
            "schema": "aura.workspace.signal.v1",
            "reason": normalized_reason,
            "payload": preserved_payload,
        }
        return await self.submit(
            CognitiveCandidate(
                content=content,
                source=normalized_source,
                priority=normalized_priority,
                content_type=content_type,
                metadata=metadata,
            )
        )

    def _resolve_global_inhibition(self) -> InhibitionManager:
        manager = ServiceContainer.get("inhibition_manager", default=None)
        if manager is None:
            from core.resilience.inhibition_manager import get_inhibition_manager

            manager = get_inhibition_manager()
            ServiceContainer.register_instance(
                "inhibition_manager",
                manager,
                required=True,
                owner="core.resilience.inhibition_manager",
                registered_by="core.consciousness.global_workspace",
                required_for="workspace_candidate_admission",
                failure_policy="fail_closed",
            )
        check = getattr(manager, "is_inhibited", None)
        if not callable(check):
            raise TypeError("inhibition manager lacks callable is_inhibited")
        return cast("InhibitionManager", manager)

    async def _reject_for_inhibition_gate(
        self,
        candidate: CognitiveCandidate,
        *,
        phase: str,
        reason: str,
        error: BaseException | None = None,
        gate: str = "global_inhibition",
        retryable: bool = True,
        gate_instance_id: str | None = None,
    ) -> bool:
        if gate == "global_inhibition" and error is not None:
            self._inhibition_gate_ready = False
            self._last_inhibition_gate_reason = reason[:240]
        if error is not None:
            self._record_degradation(
                error,
                phase=phase,
                action="Rejected workspace candidate because global inhibition authority was unavailable",
                severity="degraded",
            )
        event = {
            "tick": self._tick,
            "candidate_source": candidate.source[:160],
            "gate": gate,
            "phase": phase,
            "reason": reason[:240],
            "retryable": retryable,
        }
        receipt = WorkspaceGateReceipt(
            cause="workspace_candidate_submission",
            candidate_source=candidate.source[:160],
            gate=gate,
            decision="rejected",
            reason=reason[:240],
            retryable=retryable,
            gate_instance_id=str(
                gate_instance_id
                if gate_instance_id is not None
                else getattr(self._global_inhibition, "instance_id", "") or ""
            )[:160],
            metadata={
                "tick": self._tick,
                "phase": phase,
                "lane": "workspace_candidate_admission",
                "content_type": candidate.content_type.name,
                "candidate_age_s": round(max(0.0, time.time() - candidate.submitted_at), 6),
            },
        )
        try:
            emitted = await asyncio.to_thread(_emit_workspace_gate_receipt, receipt)
            event["receipt_id"] = emitted.receipt_id
        except _WORKSPACE_RECOVERABLE_ERRORS as receipt_error:
            self._record_degradation(
                receipt_error,
                phase="global_inhibition_receipt",
                action="Rejected workspace candidate but gate receipt persistence failed",
                severity="critical",
            )
            event["receipt_error"] = _error_summary(receipt_error)
        self._gate_rejections.append(event)
        self._gate_rejections = self._gate_rejections[-50:]
        return False

    @staticmethod
    def _manager_instance_id(manager: Any) -> str:
        return str(getattr(manager, "instance_id", "") or "")[:160]

    async def _check_candidates_for_competition(
        self,
        candidates: list[CognitiveCandidate],
    ) -> list[CognitiveCandidate]:
        """Revalidate every bid immediately before selection.

        Submission approval is deliberately not a bearer token. Inhibition can
        change or the canonical manager can be replaced between submission and
        broadcast, so every pending candidate must pass the current instance.
        """

        if not candidates:
            return []
        try:
            manager = self._resolve_global_inhibition()
            check = getattr(manager, "is_inhibited", None)
            if not callable(check):
                raise TypeError("inhibition manager lacks callable is_inhibited")
            self._global_inhibition = manager
        except _WORKSPACE_RECOVERABLE_ERRORS as exc:
            for candidate in candidates:
                await self._reject_for_inhibition_gate(
                    candidate,
                    phase="global_inhibition_revalidation_lookup",
                    reason=f"gate_revalidation_lookup_failed:{type(exc).__qualname__}",
                    error=exc,
                )
            self._global_inhibition = None
            return []

        timeout = max(0.01, float(_INHIBITION_GATE_TIMEOUT_FLAG.value()))
        checks = [
            asyncio.wait_for(manager.is_inhibited(candidate.source), timeout=timeout)
            for candidate in candidates
        ]
        try:
            results = await asyncio.gather(*checks, return_exceptions=True)
        except asyncio.CancelledError as exc:
            for candidate in candidates:
                await asyncio.shield(
                    self._reject_for_inhibition_gate(
                        candidate,
                        phase="global_inhibition_revalidation_cancelled",
                        reason="gate_revalidation_cancelled:CancelledError",
                        error=exc,
                    )
                )
            self._candidates = []
            self._global_inhibition = None
            raise

        accepted: list[CognitiveCandidate] = []
        gate_fault = False
        current_instance_id = self._manager_instance_id(manager)
        for candidate, result in zip(candidates, results, strict=True):
            if isinstance(result, BaseException):
                gate_fault = True
                await self._reject_for_inhibition_gate(
                    candidate,
                    phase="global_inhibition_revalidation",
                    reason=f"gate_revalidation_failed:{type(result).__qualname__}",
                    error=result,
                    gate_instance_id=current_instance_id,
                )
                continue
            if not isinstance(result, bool):
                gate_fault = True
                error = TypeError("inhibition manager returned a non-boolean decision")
                await self._reject_for_inhibition_gate(
                    candidate,
                    phase="global_inhibition_revalidation",
                    reason="gate_revalidation_failed:TypeError",
                    error=error,
                    gate_instance_id=current_instance_id,
                )
                continue
            if result:
                await self._reject_for_inhibition_gate(
                    candidate,
                    phase="global_inhibition_revalidation_policy",
                    reason="source_inhibited_before_competition",
                    gate_instance_id=current_instance_id,
                )
                continue
            candidate.gate_instance_id = current_instance_id
            candidate.gate_checked_at = time.time()
            accepted.append(candidate)

        if len(accepted) == len(candidates):
            self._inhibition_gate_ready = True
            self._last_inhibition_gate_reason = "healthy"
            for phase in _INHIBITION_GATE_DEGRADATION_PHASES:
                self._degraded_channels.pop(phase, None)
        elif gate_fault:
            self._global_inhibition = None
        return accepted

    async def submit(self, candidate: CognitiveCandidate) -> bool:
        """Submit a candidate for the next broadcast competition.
        Returns False if the source is currently inhibited.
        """
        if self._lock is None:
            self._lock = asyncio.Lock()
            
        async with self._lock:
            # Check internal inhibition
            if candidate.source in self._inhibited and self._inhibited[candidate.source] > 0:
                logger.debug("GW: %s is internal-inhibited (%d ticks)", candidate.source, self._inhibited[candidate.source])
                return await self._reject_for_inhibition_gate(
                    candidate,
                    phase="workspace_refractory_policy",
                    reason="source_in_refractory_period",
                    gate="workspace_refractory",
                    gate_instance_id="global_workspace",
                )
                
            # Check global inhibition
            if self._global_inhibition is None:
                try:
                    self._global_inhibition = self._resolve_global_inhibition()
                except _WORKSPACE_RECOVERABLE_ERRORS as exc:
                    return await self._reject_for_inhibition_gate(
                        candidate,
                        phase="global_inhibition_lookup",
                        reason=f"gate_lookup_failed:{type(exc).__qualname__}",
                        error=exc,
                    )
            try:
                gate_timeout = max(0.01, float(_INHIBITION_GATE_TIMEOUT_FLAG.value()))
                inhibited = await asyncio.wait_for(
                    self._global_inhibition.is_inhibited(candidate.source),
                    timeout=gate_timeout,
                )
            except asyncio.CancelledError as exc:
                await asyncio.shield(
                    self._reject_for_inhibition_gate(
                        candidate,
                        phase="global_inhibition_check_cancelled",
                        reason="gate_check_cancelled:CancelledError",
                        error=exc,
                    )
                )
                self._global_inhibition = None
                raise
            except _WORKSPACE_RECOVERABLE_ERRORS as exc:
                rejected = await self._reject_for_inhibition_gate(
                    candidate,
                    phase="global_inhibition_check",
                    reason=f"gate_check_failed:{type(exc).__qualname__}",
                    error=exc,
                )
                self._global_inhibition = None
                return rejected
            if not isinstance(inhibited, bool):
                decision_error = TypeError(
                    "inhibition manager returned a non-boolean decision"
                )
                rejected = await self._reject_for_inhibition_gate(
                    candidate,
                    phase="global_inhibition_check",
                    reason="gate_check_failed:TypeError",
                    error=decision_error,
                )
                self._global_inhibition = None
                return rejected
            if inhibited:
                logger.debug("GW: %s is GLOBAL-inhibited", candidate.source)
                return await self._reject_for_inhibition_gate(
                    candidate,
                    phase="global_inhibition_policy",
                    reason="source_inhibited",
                )
            self._inhibition_gate_ready = True
            self._last_inhibition_gate_reason = "healthy"
            for phase in _INHIBITION_GATE_DEGRADATION_PHASES:
                self._degraded_channels.pop(phase, None)
            candidate.gate_instance_id = self._manager_instance_id(self._global_inhibition)
            candidate.gate_checked_at = time.time()
            
            # Φ-aware priority boost: high integration → higher salience
            if self._current_phi > 0.1:
                phi_boost = min(self._PHI_PRIORITY_BOOST, self._current_phi * 0.1)
                # Fix Issue 68: Don't mutate candidate.priority; use focus_bias instead
                candidate.focus_bias = min(1.0, candidate.focus_bias + phi_boost)
            
            # Replace any existing candidate from same source (only one bid per source).
            # Done BEFORE the flood check so a source updating its own bid never counts
            # as new pressure and never gets spuriously dropped.
            self._candidates = [c for c in self._candidates if c.source != candidate.source]

            # --- Seizure Guard (Phase 23.5) + salience-ranked backpressure ---
            if len(self._candidates) >= self._MAX_CANDIDATES:
                # The workspace is full. Rather than blanket-drop every new bid (which
                # let a flood of low-salience submissions lock out a genuinely urgent
                # one that arrives later), keep the N *most salient* bids: evict the
                # weakest queued candidate iff the incoming one outranks it. A valid,
                # high-priority candidate is never dropped just for arriving late.
                weakest = min(self._candidates, key=lambda c: c.effective_priority)
                incoming = candidate.effective_priority

                replacement_threshold = (
                    weakest.effective_priority + self._SEIZURE_REPLACEMENT_MARGIN
                )
                if incoming <= replacement_threshold:
                    # Incoming really is the least important — drop it and signal flood.
                    logger.warning(
                        "🧠 [SEIZURE GUARD] GlobalWorkspace FLOODED (%d); dropping lowest bid %s "
                        "(%.3f ≤ replacement threshold %.3f).",
                        len(self._candidates), candidate.source, incoming, replacement_threshold,
                    )
                    self._signal_neural_flood(candidate.source)
                    return await self._reject_for_inhibition_gate(
                        candidate,
                        phase="workspace_capacity_policy",
                        reason="workspace_capacity_rejected",
                        gate="workspace_capacity",
                        gate_instance_id="global_workspace",
                    )

                # Incoming outranks the weakest → evict the weakest, admit the incoming.
                logger.warning(
                    "🧠 [SEIZURE GUARD] GlobalWorkspace FLOODED (%d); evicting weakest %s (%.3f) for %s (%.3f).",
                    len(self._candidates), weakest.source, weakest.effective_priority,
                    candidate.source, incoming,
                )
                self._candidates = [c for c in self._candidates if c is not weakest]
                self._signal_neural_flood(weakest.source)
                await self._reject_for_inhibition_gate(
                    weakest,
                    phase="workspace_capacity_policy",
                    reason="workspace_capacity_evicted",
                    gate="workspace_capacity",
                    gate_instance_id="global_workspace",
                )

            self._candidates.append(candidate)
            return True

    def _signal_neural_flood(self, dropped_source: str) -> None:
        """Broadcast a workspace-flood tension reflex via the mycelial network.

        Fired whenever backpressure has to drop a bid (the incoming one or an evicted
        weakest one). Best-effort: a missing/erroring mycelium never blocks competition.
        """
        try:
            mycelium = ServiceContainer.get("mycelial_network", default=None)
            if mycelium:
                h = mycelium.get_hypha("consciousness", "workspace")
                if h:
                    h.strength = 10.0  # Thicken the visual noise — the system *feels* flooded.
                get_task_tracker().create_task(
                    mycelium.emit_reflex("NEURAL_FLOOD", {"source": dropped_source})
                )
        except _WORKSPACE_RECOVERABLE_ERRORS as _e:
            self._record_degradation(
                _e,
                phase="seizure_guard_reflex",
                action="Dropped flooded workspace bid and skipped mycelial flood reflex",
                severity="warning",
            )
            logger.debug("GW seizure guard reflex failed after dropping bid: %s", _e)

    # ------------------------------------------------------------------
    # Processor registration — subsystems register to receive broadcasts
    # ------------------------------------------------------------------

    def register_processor(self, fn: ProcessorFn) -> None:
        """Register a coroutine function to be called when a winner is broadcast."""
        self._processors.append(fn)

    def subscribe(self, fn: ProcessorFn) -> None:
        """Alias for register_processor to support AgencyCore subscriptions."""
        self.register_processor(fn)

    # ------------------------------------------------------------------
    # Competition — called once per heartbeat tick
    # ------------------------------------------------------------------

    async def run_competition(self) -> CognitiveCandidate | None:
        """Run the competitive selection. Returns the winner (or None if no candidates).
        Inhibits losers and broadcasts winner to all registered processors.
        """
        self._tick += 1

        if self._lock is None:
            self._lock = asyncio.Lock()

        # Mycelial Pulse (Proof of Life for Workspace)
        try:
            mycelium = ServiceContainer.get("mycelial_network", default=None)
            if mycelium:
                hypha = mycelium.get_hypha("consciousness", "workspace")
                if hypha:
                    hypha.pulse(success=True)
        except _WORKSPACE_RECOVERABLE_ERRORS as _e:
            self._record_degradation(
                _e,
                phase="workspace_pulse",
                action="Skipped mycelial proof-of-life pulse and continued workspace competition",
                severity="debug",
            )
            logger.debug("GW mycelial proof-of-life pulse skipped: %s", _e)

        async with self._lock:
            # Decay inhibition counters before a possible somatic submission so
            # the synthetic source follows the same refractory policy as every
            # other producer.
            self._inhibited = {
                src: count - 1
                for src, count in self._inhibited.items()
                if count > 1
            }
            pending_count = len(self._candidates)
            inhibited_sources = set(self._inhibited)

        try:
            impulse = self._somatic_noise.maybe_generate(
                tick=self._tick,
                candidate_count=pending_count,
                inhibited_sources=inhibited_sources,
            )
            if impulse is not None:
                await self.submit(
                    CognitiveCandidate(
                        content=impulse.content,
                        source="somatic_noise",
                        priority=impulse.priority,
                        content_type=impulse.content_type,
                        affect_weight=0.05,
                        focus_bias=0.0,
                    )
                )
        except asyncio.CancelledError:
            raise
        except _WORKSPACE_RECOVERABLE_ERRORS as exc:
            self._record_degradation(
                exc,
                phase="somatic_noise",
                action="Skipped stochastic somatic impulse and continued workspace competition",
                severity="warning",
            )

        async with self._lock:
            if not self._candidates:
                return None

            self._candidates = await self._check_candidates_for_competition(
                list(self._candidates)
            )
            if not self._candidates:
                return None

            # Sort by effective priority (highest wins)
            self._candidates.sort(key=lambda c: c.effective_priority, reverse=True)
            winner = self._candidates[0]
            losers = self._candidates[1:]

            # Inhibit all losers
            for loser in losers:
                self._inhibited[loser.source] = self._INHIBIT_TICKS

            # Clear candidate pool
            self._candidates = []

            # Record
            record = BroadcastRecord(
                winner=winner,
                losers=[loser.source for loser in losers]
            )
            # No manual truncation: the buffer enforces its own bound, so it
            # holds for every writer rather than only for the ones that
            # remember to check.
            self._history.append(record)

            self.last_winner = winner

        # --- Peripheral Awareness (Attention/Consciousness Dissociation) ---
        # Feed losers into the peripheral field so content that didn't win
        # broadcast can still be phenomenally present at low intensity.
        try:
            from core.consciousness.peripheral_awareness import get_peripheral_awareness_engine
            all_candidates_data = [
                {"source": winner.source, "priority": winner.effective_priority, "content": str(winner.content)[:200]}
            ] + [
                {"source": loser.source, "priority": loser.effective_priority, "content": str(loser.content)[:200]}
                for loser in losers
            ]
            get_peripheral_awareness_engine().process_workspace_results(
                winner_source=winner.source,
                all_candidates=all_candidates_data,
            )
        except _WORKSPACE_RECOVERABLE_ERRORS as _pa_exc:
            self._record_degradation(
                _pa_exc,
                phase="peripheral_awareness",
                action="Retained broadcast winner and skipped peripheral awareness side-feed",
                severity="warning",
            )
            logger.debug("GW peripheral awareness feed skipped: %s", _pa_exc)

        try:
            from core.unity import get_unity_runtime

            get_unity_runtime().record_workspace_competition(winner, losers)
        except _WORKSPACE_RECOVERABLE_ERRORS as exc:
            self._record_degradation(
                exc,
                phase="unity_runtime",
                action="Retained broadcast record and skipped unity workspace frame",
                severity="warning",
            )
            logger.debug("GW unity workspace frame skipped: %s", exc)

        # --- Ignition Detection ---
        winner_priority = winner.effective_priority
        self.ignition_level = min(1.0, winner_priority / self._IGNITION_THRESHOLD)
        was_ignited = self.ignited
        self.ignited = winner_priority >= self._IGNITION_THRESHOLD
        
        if self.ignited and not was_ignited:
            self._ignition_count += 1
            logger.info(
                "⚡ GW IGNITION #%d: source=%s, priority=%.3f, phi=%.4f",
                self._ignition_count, winner.source, winner_priority, self._current_phi,
            )

            # ── Theory Arbitration: GWT predicts broadcast improves accessibility ──
            try:
                from core.consciousness.theory_arbitration import get_theory_arbitration
                arb = get_theory_arbitration()
                event_id = f"gw_ignition_{self._ignition_count}"
                arb.log_prediction(
                    theory="gwt",
                    event_id=event_id,
                    prediction="broadcast_improves_coherence",
                    confidence=min(1.0, winner_priority),
                )
                # IIT counter-prediction: integration matters more than broadcast
                arb.log_prediction(
                    theory="iit_4_0",
                    event_id=event_id,
                    prediction="phi_determines_coherence_not_broadcast",
                    confidence=0.6,
                )
            except _WORKSPACE_RECOVERABLE_ERRORS as exc:
                self._record_degradation(
                    exc,
                    phase="theory_arbitration",
                    action="Recorded ignition locally and skipped theory arbitration prediction feed",
                    severity="warning",
                )
                logger.debug("GW theory arbitration feed skipped: %s", exc)

        # 4. Neural Feed Transparency (Phase 13)
        try:
            from core.thought_stream import get_emitter
            emitter = get_emitter()
            if emitter:
                emitter.emit(
                    title="Neural Competition",
                    content=f"Winner: {winner.source} | Content: {winner.content[:100]}",
                    level="info",
                    metadata={
                        "tick": self._tick,
                        "winner_priority": round(winner.effective_priority, 3),
                        "losers": [loser.source for loser in losers[:3]]
                    }
                )
        except _WORKSPACE_RECOVERABLE_ERRORS as e:
            self._record_degradation(
                e,
                phase="thought_stream",
                action="Retained winner and skipped Neural Feed transparency event",
                severity="warning",
            )
            logger.debug("Failed to emit Neural Feed match: %s", e)

        # Update attention schema with winner (outside lock)
        if self.attention_schema:
            try:
                await self.attention_schema.set_focus(
                    content=winner.content,
                    source=winner.source,
                    priority=winner.effective_priority,
                )
            except _WORKSPACE_RECOVERABLE_ERRORS as exc:
                self._record_degradation(
                    exc,
                    phase="attention_schema",
                    action="Retained broadcast winner and skipped attention-schema focus update",
                    severity="warning",
                )

        # Broadcast to all registered processors (outside lock, concurrent)
        if self._processors:
            event = BroadcastEvent(winners=[winner], timestamp=time.time())
            await asyncio.gather(
                *[self._safe_call(proc, event) for proc in self._processors],
                return_exceptions=True
            )

        logger.debug(
            "GW tick %d: winner='%s' (pri=%.2f), inhibited=%s",
            self._tick, winner.source, winner.effective_priority, list(self._inhibited.keys())
        )
        return winner

    async def _safe_call(
        self,
        fn: ProcessorFn,
        event: BroadcastEvent | CognitiveCandidate,
    ) -> None:
        try:
            # Handle both legacy single-candidate and new broadcast-event formats
            res = fn(event)
            if res is not None and inspect.isawaitable(res):
                await res
        except _WORKSPACE_RECOVERABLE_ERRORS as e:
            processor_name = getattr(fn, "__qualname__", getattr(fn, "__name__", fn.__class__.__name__))
            self._processor_failures[processor_name] = self._processor_failures.get(processor_name, 0) + 1
            self._record_degradation(
                e,
                phase="processor_broadcast",
                action=f"Isolated processor {processor_name} failure and continued remaining broadcasts",
                severity="warning",
            )
            logger.error("GW processor error: %s", e)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def get_snapshot(self) -> dict[str, Any]:
        last = self.last_winner
        return {
            "tick": self._tick,
            "last_winner": last.source if last else None,
            "last_content": last.content[:80] if last else None,
            "last_priority": round(last.effective_priority, 3) if last else 0.0,
            "pending_candidates": len(self._candidates),
            "inhibited_sources": list(self._inhibited.keys()),
            "broadcast_history_len": len(self._history),
            "ignition_level": round(self.ignition_level, 3),
            "ignited": self.ignited,
            "ignition_count": self._ignition_count,
            "phi": round(self._current_phi, 4),
            "degraded_channels": dict(self._degraded_channels),
            "recent_degradations": list(self._degradation_events[-5:]),
            "processor_failures": dict(self._processor_failures),
            "inhibition_gate": {
                "ready": self._inhibition_gate_ready,
                "reason": self._last_inhibition_gate_reason,
                "instance_id": str(
                    getattr(self._global_inhibition, "instance_id", "") or ""
                ),
                "rejection_count": len(self._gate_rejections),
                "recent_rejections": list(self._gate_rejections[-5:]),
            },
            "somatic_noise": {
                "enabled": self._somatic_noise.enabled,
                "rate": self._somatic_noise.rate,
                "injected_count": self._somatic_noise.injected_count,
                "last_reason": self._somatic_noise.last_impulse.reason if self._somatic_noise.last_impulse else None,
            },
        }

    def is_alive(self) -> bool:
        return True

    def is_ready(self) -> bool:
        from core.runtime.service_access import optional_service

        manager = self._global_inhibition or optional_service(
            "inhibition_manager", default=None
        )
        if manager is None or not callable(getattr(manager, "is_inhibited", None)):
            return False
        manager_ready = getattr(manager, "is_ready", None)
        if callable(manager_ready):
            try:
                if not bool(manager_ready()):
                    return False
            except _WORKSPACE_RECOVERABLE_ERRORS:
                return False
        return bool(
            self._last_inhibition_gate_reason in {"not_checked", "healthy"}
            and not (_INHIBITION_GATE_DEGRADATION_PHASES & self._degraded_channels.keys())
            and "global_inhibition_receipt" not in self._degraded_channels
        )

    def get_status(self) -> dict[str, Any]:
        snapshot = self.get_snapshot()
        snapshot["alive"] = self.is_alive()
        snapshot["ready"] = self.is_ready()
        snapshot["lane"] = "workspace_candidate_admission"
        return snapshot

    def update_phi(self, phi: float) -> None:
        """Update the current Φ value from the LiquidSubstrate.
        Called by the heartbeat or consciousness system each tick.
        """
        self._current_phi = max(0.0, float(phi))

    def is_ignited(self) -> bool:
        """Whether the workspace is currently in an ignited state."""
        return self.ignited

    def get_ignition_level(self) -> float:
        """Current ignition intensity (0.0-1.0)."""
        return self.ignition_level

    def get_last_n_winners(self, n: int = 5) -> list[dict[str, Any]]:
        return [
            {
                "winner": r.winner.source,
                "content": r.winner.content[:60],
                "losers": r.losers,
                "timestamp": r.timestamp,
            }
            for r in self._history[-n:]
        ]

    def get_context_stream(self, n: int = 5) -> str:
        """Return a formatted string of the last N winners for prompt injection."""
        winners = self.get_last_n_winners(n)
        if not winners:
            return ""
        
        lines = []
        for w in winners:
            lines.append(f"- [{w['winner']}] {w['content']}")
        return "\n".join(lines)
