from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import deque
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.runtime.errors import FallbackClassification, Severity, record_degradation

if TYPE_CHECKING:
    from core.brain.inference_gate import InferenceGate
    from core.bus.actor_bus import ActorBus

logger = logging.getLogger(__name__)

_ORCHESTRATOR_TYPE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_BACKGROUND_TASK_FAILURES: deque[dict[str, Any]] = deque(maxlen=128)


def _record_orchestrator_types_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    try:
        record_degradation(
            "orchestrator_types",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError as signature_exc:
        try:
            record_degradation("orchestrator_types", error, severity=severity, action=action)
        except TypeError:
            logger.debug("Orchestrator type degradation could not be recorded: %s", signature_exc)


def _task_name(task: asyncio.Task) -> str:
    try:
        return task.get_name()
    except _ORCHESTRATOR_TYPE_ERRORS:
        return f"task:{id(task)}"


def _remember_background_failure(task_name: str, exc: BaseException) -> None:
    _BACKGROUND_TASK_FAILURES.append(
        {
            "task": task_name,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "at": time.time(),
        }
    )


def get_background_task_failure_status() -> dict[str, Any]:
    return {
        "recent_failures": list(_BACKGROUND_TASK_FAILURES),
        "recent_failure_count": len(_BACKGROUND_TASK_FAILURES),
    }


def _dispose_unawaited(result: Awaitable[Any]) -> None:
    close = getattr(result, "close", None)
    if callable(close):
        close()


def _notify_immune_system(task_name: str, exc: BaseException) -> None:
    try:
        from ..container import ServiceContainer

        immune = ServiceContainer.get("immune_system", default=None)
        on_error = getattr(immune, "on_error", None) if immune is not None else None
        if not callable(on_error):
            return
        result = on_error(exc, {"task": task_name, "source": "orchestrator_background_task"})
        if inspect.isawaitable(result):
            _dispose_unawaited(result)
            _record_orchestrator_types_degradation(
                RuntimeError("immune on_error returned awaitable in sync task callback"),
                action="closed unawaited immune callback result after task failure was already recorded",
                severity="warning",
                extra={"task": task_name},
            )
    except _ORCHESTRATOR_TYPE_ERRORS as exc2:
        _record_orchestrator_types_degradation(
            exc2,
            action="kept task exception handler alive after immune routing failed",
            severity="warning",
            extra={"task": task_name},
        )
        logger.debug("Immune system unavailable for background task error logging: %s", exc2)


def _notify_morphogenesis(task_name: str, exc: BaseException) -> None:
    try:
        from core.morphogenesis.hooks import observe_orchestrator_exception

        observe_orchestrator_exception(
            subsystem="orchestrator_types",
            exc=exc,
            metadata={"task": task_name},
        )
    except _ORCHESTRATOR_TYPE_ERRORS as exc2:
        _record_orchestrator_types_degradation(
            exc2,
            action="kept task exception handler alive after morphogenesis routing failed",
            severity="warning",
            extra={"task": task_name},
        )
        logger.debug("Morphogenesis task failure observer unavailable: %s", exc2)


class SystemStatus(BaseModel):
    """System status tracking"""
    model_config = ConfigDict(validate_assignment=True)
    
    initialized: bool = False
    running: bool = False
    healthy: bool = False
    start_time: float | None = None
    uptime: float = 0.0
    cycle_count: int = 0
    last_error: str | None = None
    skills_loaded: int = 0
    dependencies_ok: bool = False
    is_processing: bool = False
    is_throttled: bool = False
    agency: float = 0.8
    curiosity: float = 0.5
    last_active: float | None = None
    acceleration_factor: float = 1.0 # Phase 21: Cognitive Acceleration
    singularity_threshold: bool = False # Phase 21: Convergence State
    temporal_drift_s: float = 0.0 # Phase 22: Temporal Synchronization
    is_idle: bool = False
    message: str = "Standby"
    last_heartbeat: float | None = None
    
    # Subsystem Aggregation (v5.0 Hardening)
    memory_status: dict[str, Any] | None = None
    agency_status: dict[str, Any] | None = None
    cognition_status: dict[str, Any] | None = None
    liquid_state_status: dict[str, Any] | None = None
    health_metrics: dict[str, Any] = Field(default_factory=dict)
    
    def add_error(self, error: str):
        self.last_error = error
        self.healthy = False

    @field_validator("singularity_threshold", mode="before")
    @classmethod
    def _coerce_singularity_threshold(cls, value: Any) -> bool:
        return bool(value)

class OrchestratorState(BaseModel):
    """Unified state model for the Orchestrator."""
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    boredom: int = 0
    stealth_mode: bool = False
    cycle_count: int = 0
    history_length: int = 0
    thoughts_snapshot: list[dict[str, Any]] = Field(default_factory=list)
    active_objectives: list[str] = Field(default_factory=list)
    mycelial_cohesion: float = 1.0
    health_score: float = 1.0


@dataclass
class OrchestratorComponents:
    """Typed container for all subsystem references that the orchestrator
    lazy-loads during boot.  Replaces the ~30 ``Optional[Any]`` class
    attributes scattered across ``main.py``.

    **Usage** – instantiate once in ``__init__``, then access via
    ``self.components.<name>`` instead of ``self._<name>``.
    """

    # ── Core cognitive pipeline ────────────────────────────
    inference_gate: InferenceGate | None = None
    capability_engine: Any | None = None
    cognitive_engine: Any | None = None

    # ── Coordination ───────────────────────────────────────
    actor_bus: ActorBus | None = None
    supervisor_tree: Any | None = None
    kernel_interface: Any | None = None

    # ── Subsystems (alphabetical) ──────────────────────────
    agency_core: Any | None = None
    autonomic_core: Any | None = None
    ears: Any | None = None
    global_workspace: Any | None = None
    goal_hierarchy: Any | None = None
    healing_service: Any | None = None
    identity: Any | None = None
    intent_router: Any | None = None
    knowledge_graph: Any | None = None
    liquid_state: Any | None = None
    memory_manager: Any | None = None
    memory_optimizer: Any | None = None
    meta_cognition: Any | None = None
    meta_learning: Any | None = None
    metabolic_monitor: Any | None = None
    personality_engine: Any | None = None
    project_store: Any | None = None
    scratchpad_engine: Any | None = None
    self_healer: Any | None = None
    self_model: Any | None = None
    singularity_monitor: Any | None = None
    state_machine: Any | None = None
    strategic_planner: Any | None = None
    subsystem_audit: Any | None = None
    world_state: Any | None = None

    # ── Telemetry / Monitoring ─────────────────────────────
    event_loop_monitor: Any | None = None
    integrity_monitor: Any | None = None

    # ── Sensory ────────────────────────────────────────────
    sensory_actor: Any | None = None
    last_sensory_heartbeat: float = 0.0


def _bg_task_exception_handler(task: asyncio.Task):
    """Callback for background tasks to record, route, and retain failures."""
    try:
        if task.cancelled():
            return
        exc = task.exception()
    except asyncio.CancelledError as exc:
        logger.debug("Task was cancelled: %s", exc)
        return
    except asyncio.InvalidStateError as exc:
        logger.debug("Task in invalid state: %s", exc)
        return
    except _ORCHESTRATOR_TYPE_ERRORS as exc:
        _record_orchestrator_types_degradation(
            exc,
            action="kept event loop alive after background task exception handler failed",
            severity="degraded",
        )
        return

    if exc is None:
        return

    name = _task_name(task)
    logger.warning("Background task %s failed: %s", name, exc)
    _remember_background_failure(name, exc)
    _record_orchestrator_types_degradation(
        exc,
        action="captured background task failure and routed it to recovery observers",
        severity="degraded",
        extra={"task": name},
    )
    _notify_immune_system(name, exc)
    _notify_morphogenesis(name, exc)
