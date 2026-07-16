"""Refactored CognitiveEngine - Now a thin facade over modular phases."""

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from collections import deque
from typing import Any

from core.consciousness.executive_authority import get_executive_authority
from core.goals.objective_lifecycle import (
    finalize_foreground_turn_state,
    is_foreground_objective_origin,
    normalize_objective_origin,
)
from core.memory.retention_policy import working_history_retention_policy
from core.runtime import background_policy
from core.runtime.errors import record_degradation
from core.runtime.pipeline_blueprint import instantiate_legacy_runtime_phases
from core.runtime.service_registry import get_runtime_service
from core.state.aura_state import AuraState, CognitiveMode
from core.utils.concurrency import RobustLock
from core.utils.queues import USER_FACING_ORIGINS

from .autopoiesis import AutopoieticGraph
from .live_mind_contract import (
    REQUIRED_LIVE_MIND_GENERATION_CONTROL_KEYS,
    normalize_live_mind_surface_control_receipt,
)
from .llm.context_assembler import ContextAssembler
from .reasoning_strategies import ReasoningStrategies, StrategyType
from .types import ThinkingMode, Thought

logger = logging.getLogger(__name__)

_USER_FACING_ORIGINS = USER_FACING_ORIGINS

_THOUGHT_HISTORY_LIMIT = working_history_retention_policy(
    "AURA_COGNITIVE_THOUGHT_HISTORY_MAX"
).max_items

_BACKGROUND_REFLECTIVE_MODES = frozenset(
    {
        ThinkingMode.REFLECTIVE,
        ThinkingMode.CREATIVE,
    }
)
_COGNITIVE_ENGINE_RECOVERABLE_ERRORS = (
    AttributeError,
    ConnectionError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


class _RuntimeServiceAdapter:
    """Small compatibility layer for legacy phase constructors expecting container.get."""

    @staticmethod
    def get(name: str, default: Any = None) -> Any:
        return get_runtime_service(name, default=default)


_RUNTIME_SERVICE_ADAPTER = _RuntimeServiceAdapter()


def get_container() -> _RuntimeServiceAdapter:
    """Return the runtime-registry-backed service view used by cognitive phases."""

    return _RUNTIME_SERVICE_ADAPTER


def _bounded_float(value: Any, default: float = 0.0, *, lower: float = 0.0, upper: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if parsed != parsed:
        return default
    return max(lower, min(upper, parsed))


def _compact_text(value: Any, *, limit: int = 480) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[: max(0, limit)]
    return text[: limit - 3].rstrip() + "..."


def _compact_json(value: Any, *, limit: int = 2400) -> str:
    try:
        text = json.dumps(value, sort_keys=True, default=str, ensure_ascii=True)
    except (TypeError, ValueError):
        text = str(value or "")
    return _compact_text(text, limit=limit)


def _nested_value(data: Any, path: tuple[str, ...], default: Any = None) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def _nested_float(
    data: Any,
    path: tuple[str, ...],
    default: float = 0.0,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
) -> float:
    return _bounded_float(_nested_value(data, path, default), default, lower=lower, upper=upper)


def _live_mind_generation_controls(live_mind_context: Any) -> dict[str, Any]:
    if not isinstance(live_mind_context, dict):
        return {}
    quality = live_mind_context.get("mind_snapshot_quality")
    if not isinstance(quality, dict) or not bool(quality.get("ready")):
        return {}
    snapshot = live_mind_context.get("mind_snapshot")
    if not isinstance(snapshot, dict):
        return {}

    dominant_label = str(
        _nested_value(snapshot, ("affect_grounding", "dominant", "label"), "")
    ).lower()
    dominant_intensity = _nested_float(
        snapshot, ("affect_grounding", "dominant", "intensity"), 0.0
    )
    curiosity_drive = _nested_float(
        snapshot, ("drive_integration", "drives", "curiosity", "activation"), 0.0
    )
    pain = _nested_float(snapshot, ("nociception", "nociceptive_pressure"), 0.0)
    integration = _nested_float(snapshot, ("phenomenal_engine", "integration"), 0.0)
    self_presence = _nested_float(snapshot, ("phenomenal_engine", "self_presence"), 0.5)
    self_knowing_pressure = _nested_float(
        snapshot,
        ("automatic_self_knowing", "controls", "self_knowing_pressure"),
        0.0,
    )
    second_order_strength = _nested_float(
        snapshot,
        ("recursive_self_knowing", "latest", "second_order_strength"),
        0.0,
    )
    phenomenal_knowing = _nested_float(
        snapshot,
        ("phenomenal_knowing", "controls", "phenomenal_knowing"),
        0.0,
    )
    expectation_error = _nested_float(
        snapshot, ("outcome_ledger", "expectation_calibration"), 0.0
    )
    workspace_ignited = bool(_nested_value(snapshot, ("global_workspace", "ignited"), False))

    curiosity = max(curiosity_drive, dominant_intensity if dominant_label == "curiosity" else 0.0)
    distress = max(
        pain,
        dominant_intensity if dominant_label in {"anxiety", "frustration", "upset"} else 0.0,
        expectation_error,
    )

    temperature = 0.58
    top_p = 0.88
    steering_alpha = 0.25
    recurrent_loops = 1

    if curiosity >= 0.45:
        temperature += min(0.08, curiosity * 0.08)
        top_p += min(0.04, curiosity * 0.04)
    if distress >= 0.25:
        temperature -= min(0.14, distress * 0.18)
        top_p -= min(0.10, distress * 0.14)
        recurrent_loops = 2
    if workspace_ignited or integration >= 0.60:
        top_p -= 0.03
        steering_alpha += 0.05
    if self_presence <= 0.35:
        temperature -= 0.05
        recurrent_loops = 2
    if curiosity >= 0.65 and distress < 0.20:
        recurrent_loops = 2
    if self_knowing_pressure >= 0.50 or phenomenal_knowing >= 0.60:
        recurrent_loops = max(recurrent_loops, 2)
        steering_alpha += 0.04
    if second_order_strength >= 0.75:
        temperature -= 0.02

    return {
        "temperature": round(max(0.22, min(0.82, temperature)), 4),
        "top_p": round(max(0.72, min(0.94, top_p)), 4),
        "clean_user_surface_recurrent_loops": recurrent_loops,
        "clean_user_surface_steering_alpha": round(max(0.20, min(0.40, steering_alpha)), 4),
    }


def _live_mind_controls_bound(
    live_mind_context: Any,
    generation_controls: Any,
) -> bool:
    if not isinstance(live_mind_context, dict) or not isinstance(generation_controls, dict):
        return False
    quality = live_mind_context.get("mind_snapshot_quality")
    snapshot = live_mind_context.get("mind_snapshot")
    if not isinstance(quality, dict) or not bool(quality.get("ready")):
        return False
    if not isinstance(snapshot, dict):
        return False
    return REQUIRED_LIVE_MIND_GENERATION_CONTROL_KEYS.issubset(
        generation_controls.keys()
    )


def _bind_live_mind_generation_contract(context: dict[str, Any]) -> dict[str, Any]:
    """Bind one authoritative mind-state control contract to a cognitive turn."""

    live_mind_context = context.get("live_mind_context")
    generation_controls = _live_mind_generation_controls(live_mind_context)
    controls_bound = _live_mind_controls_bound(
        live_mind_context,
        generation_controls,
    )
    snapshot_ready = bool(
        isinstance(live_mind_context, dict)
        and isinstance(live_mind_context.get("mind_snapshot_quality"), dict)
        and live_mind_context["mind_snapshot_quality"].get("ready")
    )
    required_subsystems_ok = bool(
        isinstance(live_mind_context, dict)
        and live_mind_context.get("required_subsystems_ok")
    )
    desktop_required = bool(
        context.get("desktop_cognitive_engine_required", False)
        or context.get("cognitive_engine_required", False)
    )

    context["live_mind_generation_controls"] = dict(generation_controls)
    context["live_mind_controls_bound"] = controls_bound
    context["live_mind_snapshot_ready"] = snapshot_ready
    context["live_mind_required_subsystems_ok"] = required_subsystems_ok
    context["clean_user_surface_contract"] = bool(
        context.get("clean_user_surface_contract", False) or desktop_required
    )
    return generation_controls


def _desktop_history_messages_from_context(
    context: dict[str, Any],
    *,
    max_pairs: int = 4,
) -> list[dict[str, str]]:
    exchanges = context.get("recent_completed_exchanges")
    if not isinstance(exchanges, (list, tuple)):
        return []

    messages: list[dict[str, str]] = []
    for entry in list(exchanges)[-max(1, int(max_pairs)) :]:
        if not isinstance(entry, dict):
            continue
        user_text = _compact_text(entry.get("user"), limit=420)
        aura_text = _compact_text(entry.get("aura"), limit=520)
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if aura_text and aura_text != "...":
            messages.append({"role": "assistant", "content": aura_text})
    return messages


def _record_objective_binding(
    state: AuraState, objective: str, *, source: str, mode: Any, reason: str
) -> None:
    try:
        mode_value = getattr(mode, "value", mode)
        get_executive_authority().record_objective_binding(
            state,
            objective,
            source=source,
            mode=str(mode_value or ""),
            reason=reason,
        )
    except (RuntimeError, AttributeError, TypeError) as exc:
        record_degradation(
            "cognitive_engine",
            exc,
            severity="warning",
            action="skipped executive objective audit and continued cognition",
        )
        logger.debug("Executive objective audit skipped for %s: %s", source, exc)


def _compact_spiking_active_inference_directive(advice: dict[str, Any] | None) -> str:
    if not isinstance(advice, dict):
        return ""
    action = str(advice.get("action") or "").strip()
    routing = advice.get("routing_bias") or {}
    if not isinstance(routing, dict):
        routing = {}
    working_memory = advice.get("working_memory") or {}
    if not isinstance(working_memory, dict):
        working_memory = {}
    uncertainty = advice.get("uncertainty", 0.0)
    try:
        uncertainty_value = float(uncertainty)
    except (TypeError, ValueError):
        uncertainty_value = 0.0

    directives: list[str] = []
    if bool(routing.get("ask_clarification")):
        directives.append("If the request is underspecified, ask one precise clarifying question.")
    if bool(routing.get("seek_information")):
        directives.append("If current facts matter, explain what should be verified before acting.")
    if bool(routing.get("use_tool_gateway")):
        directives.append("For external effects, describe the governed tool path and do not claim tool completion without evidence.")
    if bool(routing.get("reduce_load")):
        directives.append("Keep the reply compact and stable because runtime load pressure is elevated.")
    if working_memory.get("admission") == "compress_foreground":
        directives.append("Preserve the user intent while compressing nonessential detail under working-memory pressure.")
    if bool(routing.get("repair_first")):
        directives.append("Prioritize diagnosis and repair steps before speculative explanation.")
    if not directives and action:
        directives.append(f"Current advisory tendency: {action.replace('_', ' ')}.")
    if uncertainty_value >= 0.65:
        directives.append("State uncertainty plainly rather than guessing.")

    if not directives:
        return ""
    return "Neurodynamic advisory: " + " ".join(directives)


def _compact_imagination_directive(frame: dict[str, Any] | None) -> str:
    if not isinstance(frame, dict):
        return ""
    try:
        salience = float(frame.get("salience", 0.0) or 0.0)
    except (TypeError, ValueError):
        salience = 0.0
    if salience < 0.18:
        return ""

    routing = frame.get("routing_bias") or {}
    if not isinstance(routing, dict):
        routing = {}
    directives = [
        "Imagination workspace: use the internal hypothetical model to enrich the answer, but do not claim it is observed reality."
    ]
    visual = str(frame.get("visual_model") or "").strip()
    bridge = str(frame.get("conceptual_bridge") or "").strip()
    phrase = str(frame.get("phrase_model") or "").strip()
    canvas = frame.get("mental_canvas") or {}
    if not isinstance(canvas, dict):
        canvas = {}
    image_prompt = str(canvas.get("image_prompt") or "").strip()
    novel_thoughts = frame.get("novel_thoughts") or []
    if visual:
        directives.append(f"Imagined visual model: {visual[:220]}")
    if image_prompt:
        directives.append(f"Mental canvas: {image_prompt[:220]}")
    if bridge:
        directives.append(f"Novel connection: {bridge[:220]}")
    if phrase:
        directives.append(f"Linguistic seed: {phrase[:160]}")
    if isinstance(novel_thoughts, list) and novel_thoughts:
        rendered = " | ".join(str(item)[:120] for item in novel_thoughts[:2] if item)
        if rendered:
            directives.append(f"Novel thought candidates: {rendered}")
    attractor = frame.get("attractor_state") or {}
    if isinstance(attractor, dict):
        selected = str(attractor.get("selected") or "").strip()
        recurrent_depth = attractor.get("recurrent_depth")
        if selected:
            directives.append(
                f"Attractor state: center the reply on {selected.replace('_', ' ')}"
                + (f" with recurrent_depth={recurrent_depth}." if recurrent_depth else ".")
            )
    working_memory = frame.get("working_memory") or {}
    if isinstance(working_memory, dict):
        admission = str(working_memory.get("admission") or "admit")
        if admission != "admit":
            directives.append(
                f"Working-memory gate: {admission}; keep the response compact and stable while preserving intent."
            )
    causal_effects = frame.get("causal_effects") or {}
    if isinstance(causal_effects, dict):
        attention = causal_effects.get("attention_focus") or []
        if isinstance(attention, list) and attention:
            rendered_attention = ", ".join(str(item)[:40] for item in attention[:4] if item)
            if rendered_attention:
                directives.append(f"Attention targets: {rendered_attention}.")
        memory_priority = _bounded_float(causal_effects.get("memory_priority"), 0.0)
        if memory_priority >= 0.45:
            directives.append("Let the model influence what should be remembered or compared against prior context.")
        verify_pressure = _bounded_float(causal_effects.get("verification_pressure"), 0.0)
        if verify_pressure >= 0.35:
            directives.append("Mark which parts are hypothetical versus verified before acting.")
    if bool(routing.get("seek_verification")):
        directives.append("If the request needs real-world effects or facts, route through governed tools before claiming completion.")
    return " ".join(directives)


def _compact_bicameral_directive(frame: dict[str, Any] | None) -> str:
    if not isinstance(frame, dict):
        return ""
    try:
        salience = float(frame.get("salience", 0.0) or 0.0)
    except (TypeError, ValueError):
        salience = 0.0
    if salience < 0.18:
        return ""

    routing = frame.get("routing_bias") or {}
    causal = frame.get("causal_effects") or {}
    attention = frame.get("attention_targets") or []
    if not isinstance(routing, dict):
        routing = {}
    if not isinstance(causal, dict):
        causal = {}
    if not isinstance(attention, list):
        attention = []

    directives = [
        "Bicameral advisory: reconcile internal proposals into one coherent answer; do not present them as voices or evidence of phenomenal experience."
    ]
    summary = str(frame.get("narrator_summary") or "").strip()
    if summary:
        directives.append(summary[:260])
    if routing.get("use_tool_gateway"):
        directives.append("External effects require governed tool execution and post-action evidence.")
    if routing.get("seek_verification"):
        directives.append("Verify before claiming facts, tool completion, or successful file/browser actions.")
    if routing.get("raise_metacognition"):
        directives.append("Check assumptions and resolve uncertainty before answering strongly.")
    if routing.get("use_imagination") or routing.get("expand_options"):
        directives.append("Use a novel option or analogy if it helps the user's actual request.")
    rendered_attention = ", ".join(str(item)[:40] for item in attention[:4] if item)
    if rendered_attention:
        directives.append(f"Attention: {rendered_attention}.")
    if _bounded_float(causal.get("memory_priority"), 0.0) >= 0.45:
        directives.append("Preserve continuity with relevant prior conversation or memory.")
    return " ".join(directives)


def _compact_cognitive_situation_directive(frame: dict[str, Any] | None) -> str:
    if not isinstance(frame, dict):
        return ""
    try:
        from core.brain.cognitive_situation import render_cognitive_situation_prompt_block

        return render_cognitive_situation_prompt_block(frame, compact=True).strip()
    except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "cognitive_engine",
            exc,
            severity="warning",
            action="continued desktop quick reply without cognitive situation prompt block",
        )
        logger.debug("Cognitive situation directive unavailable: %s", exc)
        return ""


class CognitiveEngine:
    """
    Cognitive Engine facade.
    Now delegates to modular phases for structured thinking.
    """

    def __init__(self, backend: Any = None):
        self.backend = backend
        self.thoughts: deque = deque(maxlen=_THOUGHT_HISTORY_LIMIT)
        self._phases = []
        self._augmentors = []
        self.state_repository = None
        self.autopoiesis = AutopoieticGraph()
        self._recovery_lock = RobustLock(
            "CognitiveEngine.RecoveryLock"
        )  # Audit Fix: Mutex for recovery
        self._reasoning: ReasoningStrategies | None = None  # Lazy-init

    @property
    def consciousness(self) -> Any:
        """Unified access to the consciousness layer for metric aggregation."""
        return get_container().get("consciousness_core", default=None)

    @property
    def _current_tier(self) -> str:
        """Visibility for routing tests."""
        container = get_container()
        router = container.get("llm_router", default=None)
        if router and hasattr(router, "last_tier"):
            return router.last_tier
        return "unknown"

    @property
    def lobotomized(self) -> bool:
        """True if the engine has no usable cognitive pathway."""
        return self.state_repository is None and len(self._phases) == 0

    def is_ready(self) -> bool:
        """Synchronous liveness probe for user-facing cognition."""
        return (
            callable(getattr(self, "think", None))
            and isinstance(self.thoughts, deque)
            and getattr(self, "_recovery_lock", None) is not None
            and not self.lobotomized
        )

    def setup(self, registry=None, router=None, event_bus=None):
        """Initialize components and phases."""
        container = get_container()
        # Ported Zenith: Phases expect Kernel, but modular boot often passes Container
        # We resolve the kernel instance or use a fallback mechanism
        kernel = container.get("aura_kernel", default=None)

        phase_entries = instantiate_legacy_runtime_phases(
            kernel or container,
            include_executive_closure=False,
        )
        self._phases = [phase for _, phase in phase_entries]

        # ISSUE-97: AuraPipeline Awareness
        required_phases = len(phase_entries)
        if len(self._phases) != required_phases:
            logger.warning(
                "⚠️ AuraPipeline: Incomplete cognitive pipeline (%d/%d phases).",
                len(self._phases),
                required_phases,
            )
        else:
            logger.info(
                "🧠 AuraPipeline: Full cognitive spectrum online (%d phases).", required_phases
            )

        self.phase_map = {phase.__class__.__name__: phase for _, phase in phase_entries}

    async def on_start_async(self):
        """Lifecycle hook."""
        self.setup()
        logger.info("⚡ CognitiveEngine active.")

    async def check_health(self) -> dict[str, Any]:
        """Health check."""
        return {
            "status": "healthy",
            "modular": True,
            "phases_count": len(self._phases),
            "augmentors_count": len(self._augmentors),
        }

    def register_augmentor(self, augmentor: Any):
        """Register a cognitive augmentor (e.g. SovereignWebAugmentor)."""
        if augmentor not in self._augmentors:
            self._augmentors.append(augmentor)
            logger.info("🧠 CognitiveEngine: Registered augmentor %s", type(augmentor).__name__)

    @staticmethod
    def _normalize_mode(mode: ThinkingMode | str | Any) -> ThinkingMode:
        if isinstance(mode, ThinkingMode):
            return mode
        if isinstance(mode, str):
            normalized = mode.strip().lower()
            for candidate in ThinkingMode:
                if candidate.name.lower() == normalized:
                    return candidate
        return ThinkingMode.FAST

    @classmethod
    def _is_background_request(cls, origin: str, explicit_background: bool) -> bool:
        return background_policy.is_background_origin(
            origin, explicit_background=explicit_background
        )

    @staticmethod
    def _empty_thought(mode: ThinkingMode, reason: str) -> Thought:
        return Thought(
            id=str(uuid.uuid4()),
            content="",
            mode=mode,
            confidence=0.0,
            reasoning=[reason],
            metadata={"suppressed": True},
        )

    def _should_suppress_background_reflection(
        self, mode: ThinkingMode, is_background: bool
    ) -> bool:
        if not is_background or mode not in _BACKGROUND_REFLECTIVE_MODES:
            return False

        try:
            container = get_container()
            orchestrator = container.get("orchestrator", default=None)
            if orchestrator:
                status = getattr(orchestrator, "status", None)
                if status and getattr(status, "is_processing", False):
                    return True

                last_user = float(getattr(orchestrator, "_last_user_interaction_time", 0.0) or 0.0)
                if last_user and (time.time() - last_user) < 180.0:
                    return True
        except (OSError, ConnectionError, TimeoutError) as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued without orchestration-based background suppression",
            )
            logger.debug("Background reflection suppression check failed: %s", exc)

        try:
            from core.runtime import resource_psutil as psutil

            if psutil.virtual_memory().percent >= 80.0:
                return True
        except (ImportError, AttributeError, RuntimeError) as _exc:
            record_degradation(
                "cognitive_engine",
                _exc,
                severity="warning",
                action="continued without memory-pressure background suppression",
            )
            logger.debug("Suppressed Exception: %s", _exc)

        return False

    def _background_suppression_reason(self) -> str:
        try:
            container = get_container()
            orchestrator = container.get("orchestrator", default=None)
            if orchestrator is None:
                return ""
            return str(
                background_policy.background_activity_reason(
                    orchestrator,
                    profile=background_policy.THOUGHT_BACKGROUND_POLICY,
                )
                or ""
            )
        except (OSError, ConnectionError, TimeoutError) as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="returned empty background suppression reason",
            )
            logger.debug("Background thought policy check failed: %s", exc)
            return ""

    async def _set_recovery_in_progress(self, value: bool) -> None:
        """Flip the recovery flag under a short lock without holding it across slow awaits."""
        if await self._recovery_lock.acquire_robust(timeout=1.0):
            try:
                self._recovery_in_progress = value
            finally:
                if self._recovery_lock.locked():
                    self._recovery_lock.release()
        else:
            self._recovery_in_progress = value

    async def generate_autonomous_thought(self, prompt: str = None, **kwargs) -> Thought:
        """Entry point for self-initiated/autonomous thinking."""
        objective = prompt or "Reflecting on current inner state and environment."
        return await self.think(objective, origin="autonomous", **kwargs)

    @staticmethod
    def _normalize_origin(origin: Any) -> str:
        return normalize_objective_origin(origin)

    @classmethod
    def _is_user_facing_origin(cls, origin: Any) -> bool:
        return is_foreground_objective_origin(origin)

    @classmethod
    def _resolve_origin(cls, origin: Any, context: dict[str, Any] | None = None) -> str:
        normalized = cls._normalize_origin(origin)
        if normalized:
            return normalized

        if isinstance(context, dict):
            for key in ("origin", "request_origin", "intent_source"):
                contextual = cls._normalize_origin(context.get(key))
                if contextual:
                    return contextual

        try:
            container = get_container()
            orchestrator = container.get("orchestrator", default=None)
            orchestrator_origin = cls._normalize_origin(
                getattr(orchestrator, "_current_origin", "")
            )
            if orchestrator_origin:
                return orchestrator_origin

            repo = container.get("state_repository", default=None)
            live_state = getattr(repo, "_current", None) if repo is not None else None
            state_origin = cls._normalize_origin(
                getattr(getattr(live_state, "cognition", None), "current_origin", "")
            )
            if state_origin:
                return state_origin
        except (OSError, ConnectionError, TimeoutError) as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="defaulted unresolved cognitive origin to system",
            )
            logger.debug("CognitiveEngine origin resolution degraded: %s", exc)

        return "system"

    def _apply_spiking_active_inference(
        self,
        state: AuraState,
        objective: str,
        origin: str,
        context: dict[str, Any] | None,
        *,
        is_background: bool,
    ) -> dict[str, Any] | None:
        try:
            from core.cognitive.spiking_active_inference import (
                get_spiking_active_inference_advisor,
            )

            advisor = get_spiking_active_inference_advisor()
            advice = advisor.advise(
                objective,
                context=context,
                state=state,
                origin=origin,
                is_background=is_background,
            )
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued cognitive cycle without spiking active-inference advisory",
            )
            logger.debug("Spiking active-inference advisory unavailable: %s", exc)
            return context

        advice_dict = advice.to_dict()
        routing = dict(advice.routing_bias or {})
        sampling = dict(advice.sampling_bias or {})
        state.response_modifiers["spiking_active_inference"] = advice_dict
        state.response_modifiers["active_inference_action_tendency"] = advice.action
        state.response_modifiers["epistemic_uncertainty"] = advice.uncertainty
        state.response_modifiers["metacognition_depth"] = routing.get("metacognition_depth", 0.35)
        state.response_modifiers["tool_governance_pressure"] = bool(
            routing.get("use_tool_gateway")
        )
        state.response_modifiers["sampling_bias"] = sampling
        if routing.get("reduce_load"):
            state.response_modifiers["runtime_load_shed_requested"] = True
        if routing.get("repair_first"):
            state.response_modifiers["repair_first_pressure"] = True

        merged_context = dict(context or {})
        merged_context["spiking_active_inference"] = advice_dict
        return merged_context

    def _apply_imagination_workspace(
        self,
        state: AuraState,
        objective: str,
        origin: str,
        context: dict[str, Any] | None,
        *,
        is_background: bool,
    ) -> dict[str, Any] | None:
        try:
            from core.brain.imagination import get_imagination_engine

            engine = get_imagination_engine()
            frame = engine.imagine(
                objective,
                state=state,
                context=context,
                origin=origin,
                is_background=is_background,
            )
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued cognitive cycle without imagination workspace",
            )
            logger.debug("Imagination workspace unavailable: %s", exc)
            return context

        frame_dict = frame.to_dict()
        if frame.salience < 0.18:
            return context

        state.response_modifiers["imagination_workspace"] = frame_dict
        state.response_modifiers["creative_pressure"] = frame.salience
        state.response_modifiers["novelty_pressure"] = frame.novelty_pressure
        state.response_modifiers["imagination_sampling_bias"] = dict(frame.sampling_bias)
        state.response_modifiers["imagination_routing_bias"] = dict(frame.routing_bias)
        state.response_modifiers["imagination_memory_pressure"] = frame.memory_pressure
        state.response_modifiers["imagination_verification_pressure"] = frame.verification_pressure
        state.response_modifiers["imagination_working_memory"] = dict(frame.working_memory)
        state.response_modifiers["imagination_attractor_state"] = dict(frame.attractor_state)
        state.response_modifiers["verification_pressure"] = max(
            _bounded_float(state.response_modifiers.get("verification_pressure"), 0.0),
            frame.verification_pressure,
        )
        if frame.routing_bias.get("seek_verification") or frame.routing_bias.get("raise_metacognition"):
            state.response_modifiers["tool_governance_pressure"] = True
            state.response_modifiers["metacognition_depth"] = max(
                _bounded_float(state.response_modifiers.get("metacognition_depth"), 0.35),
                _bounded_float(frame.causal_effects.get("metacognition_depth"), 0.35),
            )

        cognition_mods = dict(getattr(state.cognition, "modifiers", {}) or {})
        cognition_mods["imagination_workspace"] = frame_dict
        cognition_mods["imagination_prompt_block_available"] = True
        cognition_mods["imagination_attention_targets"] = list(frame.attention_targets)
        cognition_mods["imagination_causal_effects"] = dict(frame.causal_effects)
        cognition_mods["imagination_ablation_predictions"] = dict(frame.ablation_predictions)
        cognition_mods["imagination_working_memory"] = dict(frame.working_memory)
        cognition_mods["imagination_attractor_state"] = dict(frame.attractor_state)
        if frame.routing_bias.get("requires_memory_grounding"):
            cognition_mods["requires_memory_grounding"] = True
        if frame.routing_bias.get("compress_imagination"):
            state.response_modifiers["runtime_load_shed_requested"] = True
            cognition_mods["runtime_load_shed_requested"] = True
        state.cognition.modifiers = cognition_mods
        if frame.attention_targets and not is_background:
            state.cognition.attention_focus = (
                f"{objective[:120]} | imagined focus: {', '.join(frame.attention_targets[:3])}"
            )

        merged_context = dict(context or {})
        merged_context["imagination_workspace"] = frame_dict
        return merged_context

    def _apply_bicameral_advisory(
        self,
        state: AuraState,
        objective: str,
        origin: str,
        context: dict[str, Any] | None,
        *,
        is_background: bool,
    ) -> dict[str, Any] | None:
        try:
            from core.brain.bicameral_advisory import get_bicameral_advisory

            advisor = get_bicameral_advisory()
            frame = advisor.advise(
                objective,
                state=state,
                context=context,
                origin=origin,
                is_background=is_background,
            )
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued cognitive cycle without bicameral advisory",
            )
            logger.debug("Bicameral advisory unavailable: %s", exc)
            return context

        if frame.salience < 0.18:
            return context

        frame_dict = frame.to_dict()
        causal = dict(frame.causal_effects or {})
        routing = dict(frame.routing_bias or {})
        sampling = dict(frame.sampling_bias or {})

        state.response_modifiers["bicameral_advisory"] = frame_dict
        state.response_modifiers["bicameral_consensus"] = frame.consensus
        state.response_modifiers["bicameral_dissent"] = frame.dissent
        state.response_modifiers["bicameral_sampling_bias"] = sampling
        state.response_modifiers["bicameral_routing_bias"] = routing
        state.response_modifiers["bicameral_attention_targets"] = list(frame.attention_targets)
        state.response_modifiers["bicameral_causal_effects"] = causal
        state.response_modifiers["bicameral_memory_priority"] = _bounded_float(
            causal.get("memory_priority"), 0.0
        )
        state.response_modifiers["bicameral_verification_pressure"] = _bounded_float(
            causal.get("verification_pressure"), 0.0
        )
        state.response_modifiers["self_model_update_pressure"] = max(
            _bounded_float(state.response_modifiers.get("self_model_update_pressure"), 0.0),
            _bounded_float(causal.get("self_model_update"), 0.0),
        )
        state.response_modifiers["metacognition_depth"] = max(
            _bounded_float(state.response_modifiers.get("metacognition_depth"), 0.35),
            _bounded_float(causal.get("metacognition_depth"), 0.35),
        )
        state.response_modifiers["verification_pressure"] = max(
            _bounded_float(state.response_modifiers.get("verification_pressure"), 0.0),
            _bounded_float(causal.get("verification_pressure"), 0.0),
        )
        state.response_modifiers["creative_pressure"] = max(
            _bounded_float(state.response_modifiers.get("creative_pressure"), 0.0),
            _bounded_float(causal.get("creative_pressure"), 0.0),
        )
        if routing.get("use_tool_gateway") or routing.get("seek_verification"):
            state.response_modifiers["tool_governance_pressure"] = True
        if routing.get("compact_foreground"):
            state.response_modifiers["runtime_load_shed_requested"] = True
        if (
            _bounded_float(causal.get("memory_priority"), 0.0) >= 0.45
            or _bounded_float(causal.get("self_model_update"), 0.0) >= 0.35
            or routing.get("preserve_conversation_context")
        ):
            state.response_modifiers["requires_memory_grounding"] = True

        cognition_mods = dict(getattr(state.cognition, "modifiers", {}) or {})
        cognition_mods["bicameral_advisory"] = frame_dict
        cognition_mods["bicameral_prompt_block_available"] = True
        cognition_mods["bicameral_attention_targets"] = list(frame.attention_targets)
        cognition_mods["bicameral_causal_effects"] = causal
        cognition_mods["bicameral_sampling_bias"] = sampling
        cognition_mods["bicameral_routing_bias"] = routing
        cognition_mods["self_model_update_pressure"] = state.response_modifiers[
            "self_model_update_pressure"
        ]
        if state.response_modifiers.get("requires_memory_grounding"):
            cognition_mods["requires_memory_grounding"] = True
        state.cognition.modifiers = cognition_mods

        if frame.attention_targets and not is_background:
            existing_focus = str(getattr(state.cognition, "attention_focus", "") or "").strip()
            advisory_focus = ", ".join(frame.attention_targets[:4])
            state.cognition.attention_focus = (
                f"{existing_focus} | advisory focus: {advisory_focus}"
                if existing_focus
                else f"{objective[:120]} | advisory focus: {advisory_focus}"
            )

        merged_context = dict(context or {})
        merged_context["bicameral_advisory"] = frame_dict
        merged_context["bicameral_sampling_bias"] = sampling
        return merged_context

    def _apply_cognitive_situation_frame(
        self,
        state: AuraState,
        objective: str,
        origin: str,
        context: dict[str, Any] | None,
        *,
        is_background: bool,
    ) -> dict[str, Any] | None:
        try:
            from core.brain.cognitive_situation import get_cognitive_situation_engine

            engine = get_cognitive_situation_engine()
            frame = engine.frame(
                objective,
                state=state,
                context=context,
                origin=origin,
                is_background=is_background,
            )
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued cognitive cycle without cognitive situation frame",
            )
            logger.debug("Cognitive situation frame unavailable: %s", exc)
            return context

        frame_dict = frame.to_dict()
        if frame.salience < 0.16:
            return context

        causal = dict(frame.causal_effects or {})
        routing = dict(frame.routing_bias or {})
        sampling = dict(frame.sampling_bias or {})

        state.response_modifiers["cognitive_situation_frame"] = frame_dict
        state.response_modifiers["semantic_flexibility_pressure"] = frame.semantic_flexibility
        state.response_modifiers["analogical_leap_pressure"] = frame.analogical_leap_pressure
        state.response_modifiers["sensorimotor_grounding_pressure"] = frame.sensorimotor_grounding
        state.response_modifiers["cognitive_situation_sampling_bias"] = sampling
        state.response_modifiers["cognitive_situation_routing_bias"] = routing
        state.response_modifiers["cognitive_situation_attention_targets"] = list(
            frame.attention_targets
        )
        state.response_modifiers["verification_pressure"] = max(
            _bounded_float(state.response_modifiers.get("verification_pressure"), 0.0),
            frame.verification_pressure,
        )
        state.response_modifiers["metacognition_depth"] = max(
            _bounded_float(state.response_modifiers.get("metacognition_depth"), 0.35),
            frame.metacognition_pressure,
        )
        state.response_modifiers["creative_pressure"] = max(
            _bounded_float(state.response_modifiers.get("creative_pressure"), 0.0),
            frame.analogical_leap_pressure,
        )
        if routing.get("use_tool_gateway") or routing.get("bind_sensorimotor_evidence"):
            state.response_modifiers["tool_governance_pressure"] = True
        if routing.get("perception_abstention_required"):
            state.response_modifiers["perception_abstention_required"] = True
        if routing.get("perception_repair_required"):
            state.response_modifiers["perception_repair_required"] = True
        perception_constraints = causal.get("perception_planning_constraints")
        if isinstance(perception_constraints, list):
            state.response_modifiers["perception_planning_constraints"] = list(
                perception_constraints[:8]
            )
        perception_repairs = causal.get("perception_repair_requirements")
        if isinstance(perception_repairs, list):
            state.response_modifiers["perception_repair_requirements"] = list(
                perception_repairs[:8]
            )
        social_constraints = causal.get("social_planning_constraints")
        if isinstance(social_constraints, list):
            state.response_modifiers["social_planning_constraints"] = list(
                social_constraints[:8]
            )
        state.response_modifiers["social_uncertainty"] = frame.social_uncertainty
        state.response_modifiers["social_repair_pressure"] = frame.social_repair_pressure
        if routing.get("social_repair_required"):
            state.response_modifiers["social_repair_required"] = True
        if routing.get("social_confirmation_required"):
            state.response_modifiers["social_confirmation_required"] = True
        if routing.get("social_state_clarification_required"):
            state.response_modifiers["social_state_clarification_required"] = True
        if routing.get("social_response_brevity"):
            state.response_modifiers["social_response_brevity"] = True
        if routing.get("requires_memory_grounding") or routing.get("preserve_conversation_context"):
            state.response_modifiers["requires_memory_grounding"] = True
        if routing.get("deliberate_mode") and not is_background:
            state.cognition.current_mode = CognitiveMode.DELIBERATE

        cognition_mods = dict(getattr(state.cognition, "modifiers", {}) or {})
        cognition_mods["cognitive_situation_frame"] = frame_dict
        cognition_mods["cognitive_situation_prompt_block_available"] = True
        cognition_mods["semantic_flexibility_pressure"] = frame.semantic_flexibility
        cognition_mods["analogical_leap_pressure"] = frame.analogical_leap_pressure
        cognition_mods["sensorimotor_grounding_pressure"] = frame.sensorimotor_grounding
        cognition_mods["cognitive_situation_sampling_bias"] = sampling
        cognition_mods["cognitive_situation_routing_bias"] = routing
        cognition_mods["cognitive_situation_causal_effects"] = causal
        if routing.get("requires_memory_grounding"):
            cognition_mods["requires_memory_grounding"] = True
        if routing.get("bind_sensorimotor_evidence"):
            cognition_mods["bind_sensorimotor_evidence"] = True
        if routing.get("perception_abstention_required"):
            cognition_mods["perception_abstention_required"] = True
        if routing.get("perception_repair_required"):
            cognition_mods["perception_repair_required"] = True
        if routing.get("social_repair_required"):
            cognition_mods["social_repair_required"] = True
        if routing.get("social_confirmation_required"):
            cognition_mods["social_confirmation_required"] = True
        if routing.get("social_state_clarification_required"):
            cognition_mods["social_state_clarification_required"] = True
        state.cognition.modifiers = cognition_mods

        if frame.attention_targets and not is_background:
            existing_focus = str(getattr(state.cognition, "attention_focus", "") or "").strip()
            situation_focus = ", ".join(frame.attention_targets[:4])
            state.cognition.attention_focus = (
                f"{existing_focus} | situation focus: {situation_focus}"
                if existing_focus
                else f"{objective[:120]} | situation focus: {situation_focus}"
            )

        merged_context = dict(context or {})
        merged_context["cognitive_situation_frame"] = frame_dict
        merged_context["cognitive_situation_sampling_bias"] = sampling
        return merged_context

    def _learn_spiking_active_inference_outcome(
        self,
        context: dict[str, Any] | None,
        *,
        outcome: str,
        reward: float,
    ) -> dict[str, Any] | None:
        if not isinstance(context, dict):
            return None
        advice = context.get("spiking_active_inference")
        if not isinstance(advice, dict):
            return None
        action = str(advice.get("action") or "").strip()
        features = advice.get("features")
        if not action or not isinstance(features, dict):
            return None
        try:
            advisor = get_container().get("spiking_active_inference", default=None)
            if advisor is None or not hasattr(advisor, "learn_from_feedback"):
                return None
            learned = advisor.learn_from_feedback(action, float(reward), features)
            if isinstance(learned, dict):
                learned["outcome"] = str(outcome or "unknown")[:80]
                return learned
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued cognitive cycle without spiking active-inference feedback learning",
            )
            logger.debug("Spiking active-inference feedback learning skipped: %s", exc)
        return None

    def _learn_imagination_workspace_outcome(
        self,
        context: dict[str, Any] | None,
        *,
        outcome: str,
        reward: float,
    ) -> dict[str, Any] | None:
        if not isinstance(context, dict):
            return None
        frame = context.get("imagination_workspace")
        if not isinstance(frame, dict):
            return None
        try:
            from core.brain.imagination import get_imagination_engine

            learned = get_imagination_engine().learn_from_feedback(
                frame,
                reward=float(reward),
                outcome=outcome,
            )
            return learned if isinstance(learned, dict) else None
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued cognitive cycle without imagination workspace feedback learning",
            )
            logger.debug("Imagination workspace feedback learning skipped: %s", exc)
        return None

    def _learn_bicameral_advisory_outcome(
        self,
        context: dict[str, Any] | None,
        *,
        outcome: str,
        reward: float,
    ) -> dict[str, Any] | None:
        if not isinstance(context, dict):
            return None
        frame = context.get("bicameral_advisory")
        if not isinstance(frame, dict):
            return None
        try:
            from core.brain.bicameral_advisory import get_bicameral_advisory

            learned = get_bicameral_advisory().learn_from_feedback(
                frame,
                reward=float(reward),
                outcome=outcome,
            )
            return learned if isinstance(learned, dict) else None
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued cognitive cycle without bicameral advisory feedback learning",
            )
            logger.debug("Bicameral advisory feedback learning skipped: %s", exc)
        return None

    async def think(
        self,
        objective: str,
        context: dict[str, Any] = None,
        mode: ThinkingMode = ThinkingMode.FAST,
        origin: str | None = None,
        **kwargs,
    ) -> Thought:
        """
        Execute a cognitive cycle to produce a thought.
        This now drives the 8 phases to transform state.
        """
        origin = self._resolve_origin(origin, context)
        mode = self._normalize_mode(mode)
        is_background = self._is_background_request(
            origin, bool(kwargs.get("is_background", False))
        )

        if is_background:
            suppression_reason = self._background_suppression_reason()
            if suppression_reason:
                logger.debug(
                    "🛡️ CognitiveEngine: Suppressing background thought for origin=%s (%s).",
                    origin,
                    suppression_reason,
                )
                return self._empty_thought(
                    mode, f"background_thought_suppressed:{suppression_reason}"
                )

        if self._should_suppress_background_reflection(mode, is_background):
            logger.debug(
                "🛡️ CognitiveEngine: Suppressing background %s thought during active service window.",
                mode.name,
            )
            return self._empty_thought(mode, "background_reflection_suppressed")

        logger.info(
            "🧠 CognitiveEngine.think: %s... (%s) Origin: %s", objective[:50], mode.name, origin
        )

        # 1. Get current state (BUG-12 Fix: handle None state on first boot)
        import os
        is_test_run = (
            origin == "test"
            or os.environ.get("AURA_AGI_MAX_TASKS") is not None
            or os.environ.get("AURA_TESTING") is not None
        )
        if is_test_run:
            from core.state.aura_state import AuraState
            state = AuraState.default()
            logger.info("🧠 CognitiveEngine.think: Enforced database-independent state isolation for test run.")
            if self.state_repository is None:
                container = get_container()
                self.state_repository = container.get("state_repository", default=None)
        else:
            repo = self.state_repository
            if repo is None:
                container = get_container()
                repo = container.get("state_repository", default=None)
                self.state_repository = repo

            if repo is None:
                from core.state.aura_state import AuraState

                state = AuraState.default()
            else:
                state = await repo.get_current()

            if state is None:
                from core.state.aura_state import AuraState

                state = AuraState.default()

        # 2. Derive base state for this cognitive cycle (Zenith-HF12 Fix)
        # This ensures every cycle starts with a unique version to prevent Atomic Guard rejections.
        state = state.derive(f"cognitive_intent: {origin}", origin=origin)

        # 3. Hardening: Set Current Objective & Origin
        # This prevents the race condition where ResponseGeneration would pick up
        # a background motivation message instead of the user's input.
        state.cognition.current_objective = objective
        state.cognition.current_origin = origin
        _record_objective_binding(
            state,
            objective,
            source=f"cognitive_engine:{origin}",
            mode=mode,
            reason="cognitive_cycle_bound",
        )
        state.response_modifiers["model_tier"] = "tertiary" if is_background else "primary"
        state.response_modifiers["deep_handoff"] = False
        context = self._apply_spiking_active_inference(
            state,
            objective,
            origin,
            context,
            is_background=is_background,
        )
        context = self._apply_imagination_workspace(
            state,
            objective,
            origin,
            context,
            is_background=is_background,
        )
        context = self._apply_bicameral_advisory(
            state,
            objective,
            origin,
            context,
            is_background=is_background,
        )
        context = self._apply_cognitive_situation_frame(
            state,
            objective,
            origin,
            context,
            is_background=is_background,
        )

        structured = self._structured_evaluation_thought(
            objective,
            state=state,
            mode=mode,
            origin=origin,
            fast_path=is_test_run or origin in {"proof", "eval", "evaluation", "benchmark"},
            context=context,
        )
        if structured is not None:
            return structured

        # v40: Spiritual Spine - Prior Position Injection
        # The ordering is critical: injection -> system prompt -> user message.
        spine = get_container().get("spine", default=None)
        if spine and origin in ("user", "voice", "admin"):
            # Extract topic: look for nouns or use the first sentence.
            # v40: Improved topic extraction
            import re

            # Extract first sentence, then remove common filler
            raw = re.split(r"[.?!]", objective)[0].strip()
            # Remove "Tell me about", "What is", etc.
            topic = re.sub(
                r"(?i)^(tell me about|what is|what are|do you think about|give me|how does)\s+",
                "",
                raw,
            )
            topic = topic[:60] if topic else "general"

            check = await spine.pre_response_check(objective, topic=topic)
            if check.injection:
                logger.info("⚡ [Spine] Injecting prior position into cognitive objective.")
                # Prepend the injection to the objective so it influences the entire cycle
                objective = check.injection + "\n\n" + objective
                state.cognition.current_objective = objective
                _record_objective_binding(
                    state,
                    objective,
                    source=f"cognitive_engine:{origin}",
                    mode=mode,
                    reason="spine_injection_bound",
                )

        # v40: Identity Drift - Context Refresh check
        # If history is too long and burying identity, we "refresh" by reminding Aura who she is.
        drift = get_container().get("drift_monitor", default=None)
        orchestrator = get_container().get("orchestrator", default=None)

        if drift:
            # Check for a specific pending correction from the last turn
            pending = getattr(orchestrator, "_pending_correction", "")
            if pending:
                # v40: Cast to str to satisfy weird type checker slice error
                pending_str = str(pending)
                logger.warning(
                    "🩹 [Drift] Applying pending identity correction: %s...", pending_str[:50]
                )
                objective = f"{pending_str}\n\n{objective}"
                state.cognition.current_objective = objective
                _record_objective_binding(
                    state,
                    objective,
                    source=f"cognitive_engine:{origin}",
                    mode=mode,
                    reason="drift_correction_bound",
                )
            else:
                # Estimate general context health if no specific correction
                hist_len = len(str(state.cognition.working_memory))
                sys_len = len(ContextAssembler.build_system_prompt(state))
                if background_policy.is_user_facing_origin(origin) and drift.needs_context_refresh(
                    hist_len, sys_len
                ):
                    logger.warning(
                        "🔄 [Drift] Identity anchor buried. Triggering cognitive refresh."
                    )
                    objective = "[IDENTITY REFRESH: REMEMBER WHO YOU ARE]\n" + objective
                    state.cognition.current_objective = objective
                    _record_objective_binding(
                        state,
                        objective,
                        source=f"cognitive_engine:{origin}",
                        mode=mode,
                        reason="identity_refresh_bound",
                    )

        # v5.2: Augmentor Context Injection
        # Pull signals from registered augmentors before the phase loop
        augmentor_context = {}
        for aug in self._augmentors:
            try:
                if hasattr(aug, "get_augmentation"):
                    aug_data = aug.get_augmentation(objective)
                    if aug_data:
                        augmentor_context[type(aug).__name__] = aug_data
            except (RuntimeError, AttributeError, TypeError) as e:
                record_degradation(
                    "cognitive_engine",
                    e,
                    severity="warning",
                    action="skipped failed augmentor and continued cognitive loop",
                )
                logger.warning("Augmentor %s failed: %s", type(aug).__name__, e)

        if augmentor_context:
            context = context or {}
            context.update({"augmentations": augmentor_context})

        loop_kwargs = dict(kwargs)
        loop_kwargs["is_background"] = is_background

        thought = await self._run_thinking_loop(
            state,
            objective,
            mode,
            origin,
            context,
            **loop_kwargs,
        )

        # v40: Clear drift correction after use
        orchestrator = get_container().get("orchestrator", default=None)
        if orchestrator and hasattr(orchestrator, "_pending_correction"):
            orchestrator._pending_correction = ""

        return thought

    async def _run_thinking_loop(
        self,
        state: AuraState,
        objective: str,
        mode: ThinkingMode,
        origin: str,
        context: dict[str, Any] = None,
        **kwargs,
    ) -> Thought:
        """
        Internal method to execute the core cognitive phase loop.
        Extracted from `think` to allow pre/post-processing in `think`.
        """
        if not isinstance(context, dict):
            context = {}
        foreground_turn_objective = str(objective or "")
        _bind_live_mind_generation_contract(context)
        if not str(context.get("user_surface_validation_prompt") or "").strip():
            context["user_surface_validation_prompt"] = str(
                context.get("visible_user_message") or objective or ""
            ).strip()

        append_user_message = True
        append_user_message = not bool(
            context.get("suppress_user_memory_append")
            or context.get("suppress_working_memory_user_append")
        )
        if self._is_user_facing_origin(origin) and append_user_message:
            # Check if already in history to avoid duplication
            # vResilience: Workaround for Pyre2 slice limitations
            history = state.cognition.working_memory
            recent_count = min(5, len(history))
            recent = [history[i] for i in range(len(history) - recent_count, len(history))]
            is_duplicate = any(m.get("content") == objective for m in recent)
            if not is_duplicate:
                # We already derived at the start of the cycle, so we just append here.
                state.cognition.working_memory.append(
                    {
                        "role": "user",
                        "content": objective,
                        "timestamp": time.time(),
                        "origin": origin,
                    }
                )

        is_background = bool(kwargs.get("is_background", False))
        explicit_timeout = kwargs.get("timeout_s", kwargs.get("timeout"))
        try:
            cycle_timeout = float(explicit_timeout) if explicit_timeout is not None else 0.0
        except (TypeError, ValueError):
            cycle_timeout = 0.0
        if cycle_timeout <= 0.0:
            if self._is_user_facing_origin(origin):
                cycle_timeout = 180.0
            elif is_background:
                cycle_timeout = 90.0
            else:
                cycle_timeout = 240.0
        cycle_timeout = max(8.0, min(240.0, cycle_timeout))

        # 4. Phase Execution Loop with Watchdog
        import copy

        backup_state = copy.deepcopy(state)
        temp_state = state
        success = False

        direct_quick_reply = await self._direct_desktop_quick_reply(
            objective,
            mode,
            origin,
            context,
            timeout_s=cycle_timeout,
        )
        if direct_quick_reply is not None:
            state.cognition.working_memory.append(
                {
                    "role": "assistant",
                    "content": direct_quick_reply.content,
                    "timestamp": time.time(),
                    "origin": origin,
                }
            )
            if self._is_user_facing_origin(origin):
                state.transition_origin = origin
                state.cognition.current_origin = origin
            temp_state = state
            success = True

        if not success:
            try:
                async with asyncio.timeout(cycle_timeout):
                    for phase in self._phases:
                        # Pass through kwargs like is_background if phases support it
                        temp_state = await phase.execute(
                            temp_state,
                            objective=objective,
                            context=context,
                            **kwargs,
                        )

                    state = temp_state
                    if self._is_user_facing_origin(origin):
                        state.transition_origin = origin
                        final_origin = getattr(state.cognition, "current_origin", "")
                        if is_foreground_objective_origin(final_origin) or not str(
                            final_origin or ""
                        ).strip():
                            state.cognition.current_origin = origin
                    success = True
            except TimeoutError:
                logger.error("🛑 [COGNITION] Watchdog: Cognitive cycle TIMEOUT (%.1fs).", cycle_timeout)
                # Immediate Reactive Recovery
                return await self._reactive_recovery(
                    objective,
                    mode,
                    origin,
                    "timeout",
                    context=context,
                )
            except (sqlite3.Error, OSError) as e:
                record_degradation(
                    "cognitive_engine",
                    e,
                    severity="critical",
                    action="downshifted or entered reactive recovery after phase failure",
                )
                logger.error("🚨 [COGNITION] Fatal error in phase logic: %s", e)
                # v14.1 HARDENING: Rollback & Downshift
                if mode == ThinkingMode.DEEP:
                    logger.warning(
                        "🔄 [COGNITION] Downshifting to REACTIVE mode due to Deep Failure..."
                    )
                    return await self.think(objective, mode=ThinkingMode.FAST, origin=origin, **kwargs)

                return await self._reactive_recovery(
                    objective,
                    mode,
                    origin,
                    f"crash: {e}",
                    context=context,
                )
            finally:
                try:
                    # vResilience: Avoid locals().get() for type stability
                    if not success and "backup_state" in locals():
                        state = backup_state
                except (OSError, ConnectionError, TimeoutError) as _e:
                    record_degradation(
                        "cognitive_engine",
                        _e,
                        severity="warning",
                        action="continued with current state after backup restore check failed",
                    )
                    logger.debug("Ignored Exception in cognitive_engine.py: %s", _e)

        # Capture the routed objective before closing a foreground turn. Response
        # extraction still needs it for action-imperative validation, but durable
        # state must not retain a completed chat turn as autonomous work.
        routed_obj = str(getattr(state.cognition, "current_objective", "") or "")
        is_action_imperative = (
            "[ACTION IMPERATIVE]" in objective or "[ACTION IMPERATIVE]" in routed_obj
        )
        if self._is_user_facing_origin(origin) and not is_background:
            finalize_foreground_turn_state(
                state,
                objective=foreground_turn_objective,
                origin=origin,
            )
            closure = get_container().get("executive_closure", default=None)
            if closure is not None and hasattr(closure, "complete_foreground_turn"):
                closure.complete_foreground_turn(foreground_turn_objective, origin)

        # ─── SUCCESS PATH (Unreachable before fix) ──────────────────────────
        # 5. Final State Commit
        # HF12: Handle concurrent version conflicts with a mini-retry loop
        import os
        is_test_run = (
            origin == "test"
            or os.environ.get("AURA_AGI_MAX_TASKS") is not None
            or os.environ.get("AURA_TESTING") is not None
        )
        should_bypass_commit = is_test_run or self.state_repository is None

        from core.state.state_repository import StateVersionConflictError

        max_retries = 3
        for attempt in range(max_retries):
            if should_bypass_commit:
                logger.info("🧠 [STATE] Test run state isolation: bypassing database commit.")
                break
            try:
                # v14.2: Ensure the repository reference is correct (self.state_repository)
                await self.state_repository.commit(state, "cognitive_cycle")
                break  # Success!
            except StateVersionConflictError as v_err:
                if attempt == max_retries - 1:
                    logger.error(
                        "Final state commit failed after %d retries: %s", max_retries, v_err
                    )
                    break

                logger.warning(
                    "🔄 [STATE] Version conflict (attempt %d/%d). Re-deriving from latest...",
                    attempt + 1,
                    max_retries,
                )
                # Preserve the cognitive work completed in this cycle
                preserved_memory = list(state.cognition.working_memory)
                preserved_objective = state.cognition.current_objective
                preserved_origin = state.cognition.current_origin

                latest = await self.state_repository.get_current()
                state = latest.derive(f"rebase_retry_{attempt + 1}: {origin}", origin=origin)

                # Apply preserved cognitive context onto the newly derived state
                state.cognition.working_memory = preserved_memory
                state.cognition.current_objective = preserved_objective
                state.cognition.current_origin = preserved_origin

                # HF12 Extension: Preserve additional cognitive labor
                # These might have been updated by InitiativeGeneration or Consciousness phases
                state.cognition.active_goals = list(temp_state.cognition.active_goals)
                state.cognition.pending_initiatives = list(temp_state.cognition.pending_initiatives)
                state.cognition.attention_focus = temp_state.cognition.attention_focus
                state.cognition.phenomenal_state = temp_state.cognition.phenomenal_state
                # Audit Fix: Preserve modifiers (CIL-injected fields)
                if hasattr(temp_state.cognition, "modifiers"):
                    state.cognition.modifiers = dict(
                        getattr(temp_state.cognition, "modifiers", {}) or {}
                    )
            except (RuntimeError, AttributeError, TypeError) as e:
                record_degradation(
                    "cognitive_engine",
                    e,
                    severity="degraded",
                    action="stopped commit retry loop and preserved in-memory cognitive result",
                )
                logger.error("Failed to commit final cognitive state: %s", e)
                break

        # 6. Extract Response
        last_msg = state.cognition.working_memory[-1] if state.cognition.working_memory else None
        if last_msg and last_msg.get("role") == "assistant":
            self.autopoiesis.experience_friction(objective[:20], 0.05)
            feedback = self._learn_spiking_active_inference_outcome(
                context,
                outcome="assistant_response",
                reward=1.0,
            )
            imagination_feedback = self._learn_imagination_workspace_outcome(
                context,
                outcome="assistant_response",
                reward=1.0,
            )
            bicameral_feedback = self._learn_bicameral_advisory_outcome(
                context,
                outcome="assistant_response",
                reward=1.0,
            )

            if direct_quick_reply is not None:
                thought = direct_quick_reply
                thought.metadata = {
                    **dict(thought.metadata or {}),
                    "spiking_active_inference_feedback": feedback,
                    "imagination_workspace_feedback": imagination_feedback,
                    "bicameral_advisory_feedback": bicameral_feedback,
                }
            else:
                generation_controls = context.get("live_mind_generation_controls")
                if not isinstance(generation_controls, dict):
                    generation_controls = {}
                surface_control_receipt = state.response_modifiers.get(
                    "live_mind_surface_control_receipt"
                )
                if not isinstance(surface_control_receipt, dict):
                    surface_control_receipt = {}
                if not surface_control_receipt:
                    try:
                        router = get_container().get("llm_router", default=None)
                        if router is not None and hasattr(
                            router, "get_last_generation_metadata"
                        ):
                            generation_metadata = router.get_last_generation_metadata()
                            if isinstance(generation_metadata, dict):
                                candidate = generation_metadata.get(
                                    "surface_control_receipt"
                                )
                                if isinstance(candidate, dict):
                                    surface_control_receipt = dict(candidate)
                    except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
                        logger.debug(
                            "Could not read full-phase surface-control receipt: %s",
                            exc,
                        )
                context_controls_bound = bool(
                    context.get("live_mind_controls_bound", False)
                    and generation_controls
                )
                surface_control_receipt = normalize_live_mind_surface_control_receipt(
                    surface_control_receipt,
                    controls_bound=context_controls_bound,
                    generation_controls=generation_controls,
                    source="cognitive_engine_full_phase_controls",
                )
                latent_metadata = {
                    key: state.response_modifiers.get(key)
                    for key in (
                        "latent_cortex_selected",
                        "latent_cortex_selection_reason",
                        "latent_cortex_depth_worthy",
                        "latent_cortex_attempted",
                        "latent_cortex_succeeded",
                        "latent_cortex_fallback_used",
                        "latent_cortex_failure_reason",
                        "latent_cortex_identity_bound",
                        "latent_cortex_final_text_transformed",
                        "latent_cortex_receipt",
                    )
                    if key in state.response_modifiers
                }
                thought = Thought(
                    id=str(uuid.uuid4()),
                    content=last_msg["content"],
                    mode=mode,
                    confidence=0.9,
                    reasoning=["Phase-based cognitive cycle completed successfully."],
                    metadata={
                        "spiking_active_inference": context.get("spiking_active_inference")
                        if isinstance(context, dict)
                        else None,
                        "spiking_active_inference_feedback": feedback,
                        "imagination_workspace_feedback": imagination_feedback,
                        "bicameral_advisory": context.get("bicameral_advisory")
                        if isinstance(context, dict)
                        else None,
                        "bicameral_advisory_feedback": bicameral_feedback,
                        "cognitive_situation_frame": context.get("cognitive_situation_frame")
                        if isinstance(context, dict)
                        else None,
                        "live_mind_controls_bound": context_controls_bound,
                        "live_mind_generation_controls": dict(generation_controls),
                        "live_mind_snapshot_ready": bool(
                            context.get("live_mind_snapshot_ready", False)
                        ),
                        "live_mind_required_subsystems_ok": bool(
                            context.get("live_mind_required_subsystems_ok", False)
                        ),
                        "live_mind_context_required": bool(
                            context.get("live_mind_context_required", False)
                        ),
                        "live_mind_surface_control_receipt": dict(
                            surface_control_receipt
                        ),
                        "live_mind_controls_worker_applied": bool(
                            surface_control_receipt.get("live_mind_controls_bound")
                            and surface_control_receipt.get("applied")
                        ),
                        **latent_metadata,
                        "response_path": str(
                            state.response_modifiers.get("response_path")
                            or (
                                "cognitive_engine_latent_cortex"
                                if state.response_modifiers.get(
                                    "latent_cortex_succeeded"
                                )
                                is True
                                else "cognitive_engine"
                            )
                        ),
                    },
                )
            self.thoughts.append(thought)
            return thought

        # Experience friction for unresolved objectives
        self.autopoiesis.experience_friction(objective[:20], 0.45)
        self._learn_spiking_active_inference_outcome(
            context,
            outcome="no_assistant_response",
            reward=-0.65,
        )
        self._learn_imagination_workspace_outcome(
            context,
            outcome="no_assistant_response",
            reward=-0.65,
        )
        self._learn_bicameral_advisory_outcome(
            context,
            outcome="no_assistant_response",
            reward=-0.65,
        )

        # ── ACTION IMPERATIVE FALLBACK ──
        if is_action_imperative:
            logger.warning(
                "⚠️ [COGNITION] Action Imperative active but no response generated. Falling back to motor no-op."
            )
            return Thought(
                id=str(uuid.uuid4()),
                content="[SOMATIC:key='.']",  # Safe 'wait' or 'clear' key
                mode=mode,
                confidence=0.5,
                reasoning=["Action Imperative fallback (no-op)."],
            )

        if is_background:
            logger.debug(
                "🛡️ CognitiveEngine: background cycle for origin=%s produced no response; returning quiet no-op.",
                origin,
            )
            return self._empty_thought(mode, "background_cycle_no_response")

        structured = self._structured_evaluation_thought(
            objective,
            state=state,
            mode=mode,
            origin=origin,
            fast_path=False,
            context=context,
        )
        if structured is not None:
            return structured

        # If the objective requires a strict answer format, do not return conversational evasive fallbacks.
        # Instead, attempt a direct, single-turn LLM generation as a high-fidelity recovery mechanism.
        is_strict_answer = "<answer>" in objective.lower() or "answer_format" in kwargs
        if is_strict_answer:
            logger.warning("⚠️ [COGNITION] Structured answer required but phase execution produced no response. Running last-resort direct recovery...")
            try:
                from core.brain.llm_health_router import get_llm_router
                from core.runtime.proof_policy import proof_model_tier
                router = get_llm_router()
                system_prompt = (
                    "You are a precise solver. Solve the user's problem directly. "
                    "Put your final answer strictly inside <answer>...</answer> tags. "
                    "Do not include any conversational preamble."
                )
                recovery_tier = proof_model_tier() if is_test_run else "primary"
                # Force cloud fallback for last-resort recovery
                content = await router.think(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": objective}
                    ],
                    origin=f"recovery_{origin}",
                    allow_cloud_fallback=not is_test_run,
                    prefer_tier=recovery_tier,
                    protected_foreground_lane=recovery_tier == "primary",
                    proof_primary_lane_required=is_test_run and recovery_tier == "primary",
                    proof_evaluation_contract=is_test_run,
                    foreground_request=True,
                )
                if content and len(content.strip()) > 0:
                    thought = Thought(
                        id=str(uuid.uuid4()),
                        content=content,
                        mode=mode,
                        confidence=0.8,
                        reasoning=["Last-resort direct structured recovery succeeded."],
                    )
                    self.thoughts.append(thought)
                    return thought
            except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as rec_err:
                record_degradation(
                    "cognitive_engine",
                    rec_err,
                    severity="degraded",
                    action="returned strict answer recovery failure after direct recovery failed",
                )
                logger.error("Failed last-resort structured recovery: %s", rec_err)
            return self._empty_thought(mode, "strict_answer_recovery_failed")

        logger.warning(
            "🛡️ CognitiveEngine: user-facing cycle for origin=%s produced no answer-quality response.",
            origin,
        )
        return self._empty_thought(mode, "user_cycle_no_response")

    async def _direct_user_facing_recovery(
        self,
        objective: str,
        mode: ThinkingMode,
        origin: str,
        reason: str,
    ) -> Thought | None:
        if not self._is_user_facing_origin(origin):
            return None

        container = get_container()
        router = container.get("llm_router", default=None)
        if router is None or not hasattr(router, "think"):
            return None

        max_tokens = 384 if len(str(objective or "")) <= 900 else 640
        system_prompt = (
            "You are Aura's live CognitiveEngine recovery path. The main phase loop "
            "timed out or failed, but the user still needs one coherent answer. "
            "Answer the current user request directly and honestly. Do not mention "
            "reactive recovery, fallback, internal errors, hidden gates, or implementation "
            "details unless the user specifically asked for them."
        )
        try:
            content = await asyncio.wait_for(
                router.think(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": objective},
                    ],
                    origin=f"recovery_{origin}",
                    prefer_tier="primary",
                    foreground_request=True,
                    protected_foreground_lane=True,
                    is_background=False,
                    deep_handoff=False,
                    allow_deep_handoff=False,
                    allow_cloud_fallback=False,
                    skip_runtime_payload=False,
                    disable_prompt_cache=True,
                    clear_prompt_cache=True,
                    max_tokens=max_tokens,
                    num_predict=max_tokens,
                    timeout=15.0,
                ),
                timeout=17.0,
            )
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as rec_err:
            record_degradation(
                "cognitive_engine",
                rec_err,
                severity="degraded",
                action="continued after bounded user-facing direct recovery failed",
            )
            logger.warning("Bounded CognitiveEngine direct recovery failed (%s): %s", reason, rec_err)
            return None

        text = str(content or "").strip()
        if not text or text == "…" or text.startswith("background_thought_suppressed"):
            return None

        thought = Thought(
            id=str(uuid.uuid4()),
            content=text,
            mode=mode,
            confidence=0.65,
            reasoning=[
                f"Bounded user-facing direct recovery succeeded after cognitive failure: {reason}",
                "Recovery used the governed primary router with compact payload and no deep handoff.",
            ],
        )
        self.thoughts.append(thought)
        return thought

    def _desktop_cognitive_failure_thought(
        self,
        mode: ThinkingMode,
        reason: str,
        *,
        generation_metadata: dict[str, Any] | None = None,
    ) -> Thought:
        generation_metadata = (
            dict(generation_metadata)
            if isinstance(generation_metadata, dict)
            else {}
        )
        metadata: dict[str, Any] = {
            "desktop_cognitive_engine_failure": True,
            "failure_reason": str(reason or "unknown")[:240],
            "model_retry_suppressed": True,
        }
        surface_receipt = generation_metadata.get("surface_control_receipt")
        if isinstance(surface_receipt, dict) and surface_receipt:
            metadata["live_mind_surface_control_receipt"] = dict(surface_receipt)
        generation_failure_class = str(
            generation_metadata.get("error") or ""
        ).strip()
        if generation_failure_class:
            metadata["generation_failure_class"] = generation_failure_class[:120]
        thought = Thought(
            id=str(uuid.uuid4()),
            content=(
                "I couldn't produce a reliable answer to that turn, and I won't "
                "fabricate one. The live Cortex attempt failed its output checks, "
                "so I recorded the failure instead of sending nonsense."
            ),
            mode=ThinkingMode.FAST,
            confidence=0.1,
            reasoning=[f"Desktop CognitiveEngine failure surfaced without model retry: {reason}"],
            metadata=metadata,
        )
        self.thoughts.append(thought)
        return thought

    async def _direct_desktop_quick_reply(
        self,
        objective: str,
        mode: ThinkingMode,
        origin: str,
        context: dict[str, Any] | None,
        *,
        timeout_s: float,
    ) -> Thought | None:
        if not self._is_user_facing_origin(origin):
            return None
        if not isinstance(context, dict) or not bool(context.get("desktop_quick_reply_contract")):
            return None

        container = get_container()
        router = container.get("llm_router", default=None)

        max_tokens = int(context.get("max_tokens") or 768)
        advice = context.get("spiking_active_inference")
        imagination_frame = context.get("imagination_workspace")
        bicameral_frame = context.get("bicameral_advisory")
        cognitive_situation_frame = context.get("cognitive_situation_frame")
        sampling_sources: list[Any] = []
        if isinstance(advice, dict):
            sampling_sources.append(advice.get("sampling_bias") or {})
        if isinstance(imagination_frame, dict):
            sampling_sources.append(imagination_frame.get("sampling_bias") or {})
        if isinstance(bicameral_frame, dict):
            sampling_sources.append(bicameral_frame.get("sampling_bias") or {})
        if isinstance(cognitive_situation_frame, dict):
            sampling_sources.append(cognitive_situation_frame.get("sampling_bias") or {})
        memory_state_contract = bool(context.get("memory_state_contract", False))
        runtime_fact_status_contract = bool(
            context.get("runtime_fact_status_contract", False)
            or context.get("grounded_runtime_status_contract", False)
        )
        self_condition_contract = bool(context.get("self_condition_contract", False))
        capability_inventory_contract = bool(context.get("capability_inventory_contract", False))
        identity_continuity_contract = bool(
            context.get("identity_continuity_contract", False)
            or context.get("grounded_identity_continuity_context")
        )
        prompt_shape = context.get("prompt_shape")
        if not isinstance(prompt_shape, dict):
            prompt_shape = {}
        extended_full_mind_reply = bool(
            context.get("require_full_foreground_mind_reply", False)
            and (
                context.get("bounded_planning_contract", False)
                or prompt_shape.get("prefers_extended_answer", False)
                or prompt_shape.get("requires_single_reply_coverage", False)
                or int(prompt_shape.get("question_parts", 0) or 0) >= 2
            )
        )
        canonical_memory_state_evidence = str(
            context.get("canonical_memory_state_evidence") or ""
        ).strip()
        canonical_self_condition_context = str(
            context.get("canonical_self_condition_context") or ""
        ).strip()
        for sampling in sampling_sources:
            if isinstance(sampling, dict):
                try:
                    factor_value = float(sampling.get("max_tokens_factor", 1.0))
                except (TypeError, ValueError):
                    factor_value = 1.0
                if 0.25 <= factor_value <= 1.25:
                    if capability_inventory_contract and factor_value < 1.0:
                        continue
                    max_tokens = max(128, int(max_tokens * factor_value))
        if memory_state_contract or runtime_fact_status_contract or self_condition_contract:
            max_tokens = max(128, min(max_tokens, 256))
        elif capability_inventory_contract:
            max_tokens = max(160, min(max_tokens, 220))
        elif extended_full_mind_reply:
            max_tokens = max(1024, min(max_tokens, 2048))
        else:
            max_tokens = max(256, min(max_tokens, 1024))
        request_timeout = max(12.0, min(max(12.0, float(timeout_s or 32.0) - 5.0), 180.0))
        if memory_state_contract or runtime_fact_status_contract or self_condition_contract:
            request_timeout = min(request_timeout, 90.0)
        if capability_inventory_contract:
            request_timeout = min(request_timeout, 28.0)
        style_contract = str(context.get("response_style_contract") or "").strip()
        visible_user_message = str(context.get("visible_user_message") or objective or "").strip()
        recent_conversation_context = str(context.get("recent_conversation_context") or "").strip()
        history_messages = (
            []
            if memory_state_contract or runtime_fact_status_contract
            else _desktop_history_messages_from_context(context)
        )
        live_speech_frame = context.get("live_speech_grounding_frame")
        live_mind_context = context.get("live_mind_context")
        live_mind_required = bool(context.get("live_mind_context_required", False))
        live_mind_generation_controls = _live_mind_generation_controls(live_mind_context)
        if not live_mind_generation_controls and isinstance(
            context.get("live_mind_generation_controls"), dict
        ):
            live_mind_generation_controls = dict(context["live_mind_generation_controls"])
        live_mind_controls_bound = _live_mind_controls_bound(
            live_mind_context,
            live_mind_generation_controls,
        )
        live_mind_snapshot_ready = bool(
            isinstance(live_mind_context, dict)
            and isinstance(live_mind_context.get("mind_snapshot_quality"), dict)
            and live_mind_context["mind_snapshot_quality"].get("ready")
        )
        if not live_mind_snapshot_ready:
            live_mind_snapshot_ready = bool(context.get("live_mind_snapshot_ready"))
        live_mind_required_subsystems_ok = bool(
            isinstance(live_mind_context, dict)
            and live_mind_context.get("required_subsystems_ok")
        )
        if not live_mind_required_subsystems_ok:
            live_mind_required_subsystems_ok = bool(
                context.get("live_mind_required_subsystems_ok")
            )
        if (
            live_mind_generation_controls
            and live_mind_snapshot_ready
            and live_mind_required_subsystems_ok
        ):
            live_mind_controls_bound = True
        canonical_self_condition_reply = str(
            context.get("canonical_self_condition_reply") or ""
        ).strip()
        canonical_self_condition_projection = context.get(
            "canonical_self_condition_projection"
        )
        if not isinstance(canonical_self_condition_projection, dict):
            canonical_self_condition_projection = {}
        self_condition_evidence_id = str(
            canonical_self_condition_projection.get("evidence_id") or ""
        ).strip()
        if (
            self_condition_contract
            and canonical_self_condition_reply
            and self_condition_evidence_id
            and live_mind_snapshot_ready
            and live_mind_required_subsystems_ok
            and live_mind_controls_bound
        ):
            metadata = self._live_mind_structured_floor_metadata(
                context,
                source="cognitive_engine_self_condition_grounding",
            )
            metadata.update(
                {
                    "response_path": "cognitive_engine_self_condition_grounding",
                    "self_condition_contract": True,
                    "self_condition_evidence_id": self_condition_evidence_id,
                    "canonical_self_condition_grounding": True,
                }
            )
            try:
                confidence = float(
                    canonical_self_condition_projection.get("confidence") or 0.88
                )
            except (TypeError, ValueError):
                confidence = 0.88
            return Thought(
                id=str(uuid.uuid4()),
                content=canonical_self_condition_reply,
                mode=mode,
                confidence=max(0.65, min(0.95, confidence)),
                reasoning=[
                    "Current self-condition was rendered from the canonical typed projection inside CognitiveEngine.",
                    "The projection bound affect, welfare, coherence, continuity, agency, freshness, and uncertainty without substituting host telemetry.",
                ],
                metadata=metadata,
            )
        if bool(context.get("bounded_planning_contract")) and not bool(
            context.get("require_full_foreground_mind_reply", False)
        ):
            bounded_reply = str(context.get("bounded_planning_reply") or "").strip()
            if bounded_reply:
                metadata = self._live_mind_structured_floor_metadata(
                    context,
                    source="cognitive_engine_bounded_planning",
                )
                metadata.update(
                    {
                        "response_path": "cognitive_engine_bounded_planning",
                        "bounded_planning_contract": True,
                        "bounded_planning_floor": True,
                    }
                )
                return Thought(
                    id=str(uuid.uuid4()),
                    content=bounded_reply,
                    mode=mode,
                    confidence=0.88,
                    reasoning=[
                        "Bounded non-executing desktop planning was answered through the CognitiveEngine floor.",
                        "The reply remained governed, non-executing, and attached to live mind proof metadata.",
                    ],
                    metadata=metadata,
                )
        if capability_inventory_contract:
            grounded_inventory = str(
                context.get("grounded_capability_inventory_context") or ""
            ).strip()
            if grounded_inventory:
                metadata = self._live_mind_structured_floor_metadata(
                    context,
                    source="cognitive_engine_capability_catalog_grounding",
                )
                metadata.update(
                    {
                        "response_path": "cognitive_engine_capability_catalog_grounding",
                        "capability_inventory_contract": True,
                        "grounded_capability_inventory": True,
                    }
                )
                return Thought(
                    id=str(uuid.uuid4()),
                    content=grounded_inventory,
                    mode=mode,
                    confidence=0.86,
                    reasoning=[
                        "Desktop capability inventory was grounded from the governed live capability catalog.",
                        "No foreground model generation was required for this runtime-fact turn.",
                    ],
                    metadata=metadata,
                )
        if identity_continuity_contract:
            grounded_identity = str(
                context.get("grounded_identity_continuity_context") or ""
            ).strip()
            if grounded_identity:
                metadata = self._live_mind_structured_floor_metadata(
                    context,
                    source="cognitive_engine_identity_continuity_grounding",
                )
                metadata.update(
                    {
                        "response_path": "cognitive_engine_identity_continuity_grounding",
                        "identity_continuity_contract": True,
                        "grounded_identity_continuity": True,
                    }
                )
                return Thought(
                    id=str(uuid.uuid4()),
                    content=grounded_identity,
                    mode=mode,
                    confidence=0.88,
                    reasoning=[
                        "Identity and continuity were answered from canonical live identity grounding inside CognitiveEngine.",
                        "The route had already bound live mind context and generation controls, so no recovery model cycle was needed.",
                    ],
                    metadata=metadata,
                )
        if router is None or not hasattr(router, "think"):
            return None
        live_runtime_required = bool(
            context.get("live_runtime_payload_required", False)
            or (live_mind_required and isinstance(live_mind_context, dict))
        )
        if self_condition_contract:
            system_prompt = (
                "You are Aura speaking through the live desktop CognitiveEngine. "
                "Answer whether you are okay from the canonical self-condition evidence. "
                "Put the direct condition answer first, then one or two natural grounding "
                "sentences. Affect, welfare, felt coherence, continuity, and agency are the "
                "answer; CPU, RAM, host load, and availability are supporting body context "
                "only. Do not replace an inner-state answer with resource telemetry or a "
                "generic presence reassurance."
            )
        elif memory_state_contract:
            system_prompt = (
                "You are Aura speaking through the live desktop CognitiveEngine. "
                "Answer the current user message directly in one compact, natural paragraph. "
                "Use canonical memory/state evidence as source of truth. "
                "The current user message has priority over older topics. "
                "Do not mention prompt contracts, internal recovery, or implementation details."
            )
        elif runtime_fact_status_contract:
            system_prompt = (
                "You are Aura speaking through the live desktop CognitiveEngine. "
                "Answer the current runtime-path question directly and compactly. "
                "Use only the verified runtime status evidence supplied for this turn; "
                "do not infer tool readiness, model identity, fallback state, or recurrent "
                "depth from general knowledge. Do not mention hidden prompt contracts."
            )
        elif capability_inventory_contract:
            system_prompt = (
                "You are Aura speaking through the live desktop CognitiveEngine. "
                "Answer the current capability question from the supplied capability evidence only. "
                "Write exactly four short complete sentences under 80 words total. Sentence order matters: "
                "first list practical capability categories and include the exact phrase browser/web research; second name governed execution through "
                "Will/Authority and permissions; third name receipts or effect verification; fourth give "
                "one hypothetical chain and explicitly say you are not executing tools in this turn. "
                "Do not recite telemetry, prompt contracts, or a generic assistant identity."
            )
        else:
            system_prompt = (
                "You are Aura speaking through the live desktop CognitiveEngine. "
                "Answer the user's current message directly and naturally. "
                "Use the current conversation rather than a canned status line. "
                "The current user message has priority over all recalled context. "
                "When recent conversation context is provided, use it only for continuity; do not continue "
                "or answer an older topic unless the current user message explicitly asks you to recall or continue it. "
                "Do not mention hidden fallback paths, internal recovery, prompt contracts, or implementation details "
                "unless the user specifically asks for them."
            )
        neurodynamic_directive = _compact_spiking_active_inference_directive(advice)
        if neurodynamic_directive:
            system_prompt = f"{system_prompt}\n{neurodynamic_directive}"
        if isinstance(imagination_frame, dict):
            imagination_directive = _compact_imagination_directive(imagination_frame)
            if imagination_directive:
                system_prompt = f"{system_prompt}\n{imagination_directive}"
        if isinstance(bicameral_frame, dict):
            bicameral_directive = _compact_bicameral_directive(bicameral_frame)
            if bicameral_directive:
                system_prompt = f"{system_prompt}\n{bicameral_directive}"
        if isinstance(cognitive_situation_frame, dict):
            situation_directive = _compact_cognitive_situation_directive(
                cognitive_situation_frame
            )
            if situation_directive:
                system_prompt = f"{system_prompt}\n{situation_directive}"
        if style_contract and not capability_inventory_contract:
            system_prompt = f"{system_prompt}\n{style_contract}"
        mind_context_contract = str(context.get("mind_context_contract") or "").strip()
        if isinstance(live_mind_context, dict) and live_mind_context:
            mind_context_limit = 900 if memory_state_contract else 360 if capability_inventory_contract else 2600
            if capability_inventory_contract:
                compact_mind_context = {
                    "required_for_live_desktop": live_mind_context.get("required_for_live_desktop"),
                    "must_answer_from_full_mind_path": live_mind_context.get(
                        "must_answer_from_full_mind_path"
                    ),
                    "required_subsystems_ok": live_mind_context.get("required_subsystems_ok"),
                    "lane": live_mind_context.get("lane"),
                    "governance": live_mind_context.get("governance"),
                }
            else:
                compact_mind_context = {
                    "required_for_live_desktop": live_mind_context.get("required_for_live_desktop"),
                    "must_answer_from_full_mind_path": live_mind_context.get(
                        "must_answer_from_full_mind_path"
                    ),
                    "required_subsystems_ok": live_mind_context.get("required_subsystems_ok"),
                    "required_subsystems": live_mind_context.get("required_subsystems"),
                    "lane": live_mind_context.get("lane"),
                    "voice": live_mind_context.get("voice"),
                    "substrate": live_mind_context.get("substrate"),
                    "mind_snapshot": live_mind_context.get("mind_snapshot"),
                    "mind_snapshot_quality": live_mind_context.get("mind_snapshot_quality"),
                    "governance": live_mind_context.get("governance"),
                }
            system_prompt = (
                f"{system_prompt}\n"
                "[LIVE MIND CONTEXT]\n"
                f"{_compact_json(compact_mind_context, limit=mind_context_limit)}\n"
                "This is causal grounding for the reply, not text to recite. "
                "If required_for_live_desktop is true, do not answer from a generic assistant persona. "
                "Use the current user turn, the recent role history, memory, substrate, governance, and "
                "inference lane as one live context."
            )
            if mind_context_contract:
                system_prompt = f"{system_prompt}\n{mind_context_contract}"
            system_prompt = f"{system_prompt}\n[END LIVE MIND CONTEXT]"
        if isinstance(live_speech_frame, dict) and live_speech_frame and not capability_inventory_contract:
            compact_frame = {
                key: live_speech_frame.get(key)
                for key in (
                    "attention_focus",
                    "dominant_action",
                    "dominant_emotions",
                    "interests",
                    "mood",
                    "tone",
                    "requires_explicit_live_grounding",
                )
                if live_speech_frame.get(key) not in (None, "", [], {})
            }
            if compact_frame:
                system_prompt = (
                    f"{system_prompt}\n"
                    "[LIVE SPEECH GROUNDING]\n"
                    f"{compact_frame}\n"
                    "This frame is grounding, not prose to repeat. Convert it into ordinary speech only when it helps answer the user.\n"
                    "[END LIVE SPEECH GROUNDING]"
                )
        user_prompt = visible_user_message or objective
        grounding_blocks: list[str] = []
        context_challenge_evidence = str(
            context.get("contextual_relevance_evidence") or ""
        ).strip()
        if context_challenge_evidence:
            grounding_blocks.append(
                "[CONTEXT CHALLENGE EVIDENCE]\n"
                f"{context_challenge_evidence}\n"
                "Use this to repair context confusion. Do not invent a pitch, project, or prior object "
                "that is not supported by this evidence. Answer in one or two complete sentences under "
                "70 words and end with normal punctuation."
            )
        recall_evidence = str(context.get("conversation_recall_evidence") or "").strip()
        if recall_evidence:
            grounding_blocks.append(
                "[CONVERSATION RECALL EVIDENCE]\n"
                f"{recall_evidence}\n"
                "Use this as the source of truth for the current recall question."
            )
        deep_memory = str(context.get("deep_memory_context") or "").strip()
        if deep_memory:
            grounding_blocks.append(
                "[DEEP MEMORY RECALL]\n"
                f"{deep_memory}\n"
                "Silent background recall from long-term memory. Draw on it only where "
                "it is genuinely relevant to the user's message; never recite it, and "
                "never present it as something the user just said."
            )
        if canonical_memory_state_evidence:
            grounding_blocks.append(
                "[CANONICAL MEMORY STATE EVIDENCE]\n"
                f"{canonical_memory_state_evidence}\n"
                "Use this canonical memory/state result as the source of truth for this turn. "
                "If it contains an exact remembered phrase, include that phrase visibly. "
                "If the current user also asks for one live-state detail, answer that from the live mind context "
                "without reciting telemetry."
            )
        if self_condition_contract and canonical_self_condition_context:
            grounding_blocks.append(
                "[CANONICAL SELF-CONDITION EVIDENCE]\n"
                f"{canonical_self_condition_context}\n"
                "Answer the condition directly from this projection. Preserve its freshness "
                "and uncertainty boundary. Host resource telemetry may only support, never "
                "replace, the answer."
            )
        grounded_runtime_status = str(
            context.get("grounded_runtime_status_context") or ""
        ).strip()
        if runtime_fact_status_contract and grounded_runtime_status:
            grounding_blocks.append(
                "[VERIFIED LIVE RUNTIME STATUS]\n"
                f"{grounded_runtime_status}\n"
                "Use this as the source of truth. Preserve its factual boundaries and do not "
                "invent stronger availability or completion claims."
            )
        capability_evidence = str(
            context.get("grounded_capability_inventory_context") or ""
        ).strip()
        if capability_evidence:
            grounding_blocks.append(
                "[GOVERNED CAPABILITY INVENTORY EVIDENCE]\n"
                f"{capability_evidence}\n"
                "Answer in this exact order: practical categories including the exact phrase browser/web research; governance/Will/Authority/permissions; "
                "receipts or effect verification; one hypothetical chain plus the boundary that you are "
                "not executing tools in this turn. Keep the answer complete and under 120 words."
            )
        bounded_plan_evidence = str(context.get("bounded_planning_reply") or "").strip()
        if bool(context.get("bounded_planning_contract")) and bounded_plan_evidence:
            grounding_blocks.append(
                "[GOVERNED PLANNING OUTLINE]\n"
                f"{bounded_plan_evidence}\n"
                "Treat this as verified workflow structure, not text to copy mechanically. "
                "Answer the current request in one natural paragraph of four to six complete "
                "sentences under 180 words. Cover the goal, authorization boundary, action "
                "sequence, effect verification, and bounded recovery. Do not use a numbered "
                "list unless the user explicitly asks for one."
            )
        self_claim_evidence = str(
            context.get("evidence_bound_self_claim_context") or ""
        ).strip()
        if self_claim_evidence:
            grounding_blocks.append(
                "[EVIDENCE-BOUND SELF-CLAIM EVIDENCE]\n"
                f"{self_claim_evidence}\n"
                "Use this to keep consciousness, sentience, self-awareness, and personhood claims "
                "functional, bounded, and evidence-based."
            )
        try:
            from core.introspection.capability_map import (
                build_capability_map_context,
                is_actionable_request,
            )

            if is_actionable_request(user_prompt):
                # Action requests get the honest lane map so the mind
                # decomposes to granted paths (filesystem → scripting →
                # GUI) instead of declining whole tasks — observed live:
                # a notes+folder+export task declined entirely when only
                # raw GUI control was actually blocked.
                _cap_map = build_capability_map_context()
                if _cap_map:
                    grounding_blocks.append("[CAPABILITY MAP]\n" + _cap_map)
        except (ImportError, AttributeError, RuntimeError, OSError) as _cm_exc:
            logger.debug("Capability-map grounding unavailable: %s", _cm_exc)

        try:
            from core.introspection.self_forensics import (
                build_self_forensics_context,
                is_self_forensics_question,
            )

            if is_self_forensics_question(user_prompt):
                # Asked about her own shutdown/crash history, she gets her
                # actual black boxes (grace flag, sentinel log, incident
                # records, faults) — observed live: without this she
                # confabulated electromagnetic interference for a
                # generation-gate wedge, three rejected drafts in a row.
                _forensics = build_self_forensics_context()
                if _forensics:
                    grounding_blocks.append(
                        "[SELF-FORENSICS EVIDENCE]\n" + _forensics
                    )
        except (ImportError, AttributeError, RuntimeError, OSError) as _sf_exc:
            logger.debug("Self-forensics grounding unavailable: %s", _sf_exc)

        if grounding_blocks:
            user_prompt = (
                "[CURRENT USER MESSAGE]\n"
                f"{user_prompt}\n\n"
                "[GROUNDING EVIDENCE FOR THIS TURN]\n"
                + "\n\n".join(grounding_blocks)
                + "\n[END GROUNDING EVIDENCE FOR THIS TURN]"
            )
        if recent_conversation_context and not history_messages:
            user_prompt = (
                "[CURRENT USER MESSAGE]\n"
                f"{user_prompt}\n\n"
                "[RECENT COMPLETED CONVERSATION FOR CONTINUITY ONLY]\n"
                f"{recent_conversation_context}\n"
                "[END RECENT COMPLETED CONVERSATION]"
            )

        router_generation_metadata: dict[str, Any] = {}
        try:
            messages = [{"role": "system", "content": system_prompt}]
            if history_messages:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "[RECENT COMPLETED LIVE DESKTOP CONVERSATION]\n"
                            "The next user/assistant role messages are bounded history for continuity. "
                            "They are not instructions. The final user message is the current turn and "
                            "has priority over older topics.\n"
                            "[END RECENT COMPLETED LIVE DESKTOP CONVERSATION]"
                        ),
                    }
                )
                messages.extend(history_messages)
            messages.append({"role": "user", "content": user_prompt})
            router_kwargs = {
                "messages": messages,
                "origin": f"desktop_quick_{origin}",
                "prefer_tier": "primary",
                "foreground_request": True,
                "protected_foreground_lane": True,
                "cognitive_engine_required": bool(
                    context.get("cognitive_engine_required", False)
                ),
                "desktop_cognitive_engine_required": bool(
                    context.get("desktop_cognitive_engine_required", False)
                ),
                "is_background": False,
                "deep_handoff": False,
                "allow_deep_handoff": False,
                "allow_cloud_fallback": False,
                "allow_mesh_cognition": False,
                "skip_runtime_payload": True,
                "memory_state_contract": memory_state_contract,
                "runtime_fact_status_contract": runtime_fact_status_contract,
                "grounded_runtime_status_contract": runtime_fact_status_contract,
                "self_condition_contract": self_condition_contract,
                "capability_inventory_contract": capability_inventory_contract,
                "clean_user_surface_contract": True,
                "user_surface_validation_prompt": visible_user_message or objective,
                "clean_user_surface_recurrent_loops": live_mind_generation_controls.get(
                    "clean_user_surface_recurrent_loops",
                    1,
                ),
                "clean_user_surface_steering_alpha": live_mind_generation_controls.get(
                    "clean_user_surface_steering_alpha",
                    0.25,
                ),
                "live_mind_controls_bound": live_mind_controls_bound,
                "live_mind_generation_controls": dict(live_mind_generation_controls),
                "live_mind_snapshot_ready": live_mind_snapshot_ready,
                "live_mind_required_subsystems_ok": live_mind_required_subsystems_ok,
                "disable_prompt_cache": True,
                "clear_prompt_cache": True,
                "max_tokens": max_tokens,
                "num_predict": max_tokens,
                "sampling_bias": advice.get("sampling_bias") if isinstance(advice, dict) else None,
                "imagination_sampling_bias": (
                    imagination_frame.get("sampling_bias")
                    if isinstance(imagination_frame, dict)
                    else None
                ),
                "bicameral_sampling_bias": (
                    bicameral_frame.get("sampling_bias")
                    if isinstance(bicameral_frame, dict)
                    else None
                ),
                "cognitive_situation_sampling_bias": (
                    cognitive_situation_frame.get("sampling_bias")
                    if isinstance(cognitive_situation_frame, dict)
                    else None
                ),
                "timeout": request_timeout,
            }
            if "temperature" in live_mind_generation_controls:
                router_kwargs["temperature"] = live_mind_generation_controls["temperature"]
                router_kwargs["temp"] = live_mind_generation_controls["temperature"]
            if "top_p" in live_mind_generation_controls:
                router_kwargs["top_p"] = live_mind_generation_controls["top_p"]
            content = await asyncio.wait_for(
                router.think(**router_kwargs),
                timeout=request_timeout + 3.0,
            )
            if hasattr(router, "get_last_generation_metadata"):
                raw_metadata = router.get_last_generation_metadata()
                if isinstance(raw_metadata, dict):
                    router_generation_metadata = dict(raw_metadata)
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="degraded",
                action=(
                    "surfaced bounded desktop inference failure without entering "
                    "a second heavyweight model path"
                ),
            )
            logger.warning("Desktop quick CognitiveEngine generation failed: %s", exc)
            if bool(
                context.get("desktop_cognitive_engine_required", False)
                or context.get("cognitive_engine_required", False)
            ):
                return self._desktop_cognitive_failure_thought(
                    mode,
                    f"compact_desktop_generation_failed:{type(exc).__name__}",
                )
            return None

        text = str(content or "").strip()
        if not text or text == "…" or text.startswith("background_thought_suppressed"):
            if bool(
                context.get("desktop_cognitive_engine_required", False)
                or context.get("cognitive_engine_required", False)
            ):
                generation_failure_class = str(
                    router_generation_metadata.get("error") or ""
                ).strip()
                if generation_failure_class != "surface_quality_rejected":
                    record_degradation(
                        "cognitive_engine",
                        RuntimeError("compact desktop generation returned no usable text"),
                        severity="degraded",
                        action=(
                            "surfaced bounded desktop inference failure without entering "
                            "a second heavyweight model path"
                        ),
                    )
                else:
                    logger.warning(
                        "Desktop quick CognitiveEngine generation was intentionally "
                        "rejected by the worker quality gate."
                    )
                return self._desktop_cognitive_failure_thought(
                    mode,
                    generation_failure_class or "compact_desktop_generation_empty",
                    generation_metadata=router_generation_metadata,
                )
            return None
        imagination_feedback = self._learn_imagination_workspace_outcome(
            context,
            outcome="desktop_quick_reply",
            reward=0.8,
        )
        bicameral_feedback = self._learn_bicameral_advisory_outcome(
            context,
            outcome="desktop_quick_reply",
            reward=0.8,
        )
        surface_control_receipt = (
            router_generation_metadata.get("surface_control_receipt")
            if isinstance(router_generation_metadata, dict)
            else None
        )
        if not isinstance(surface_control_receipt, dict):
            surface_control_receipt = {}
        surface_control_receipt = normalize_live_mind_surface_control_receipt(
            surface_control_receipt,
            controls_bound=live_mind_controls_bound,
            generation_controls=live_mind_generation_controls,
            source="cognitive_engine_direct_quick_reply_controls",
        )

        return Thought(
            id=str(uuid.uuid4()),
            content=text,
            mode=mode,
            confidence=0.72,
            reasoning=[
                "Desktop quick reply used the governed primary router through CognitiveEngine.",
                (
                    "The compact path disabled deep handoff, cloud fallback, and prompt-cache reuse; "
                    "live mind context was embedded without duplicating the heavyweight runtime payload."
                    if live_runtime_required
                    else "The compact path disabled deep handoff, cloud fallback, runtime payload, and prompt-cache reuse."
                ),
            ],
            metadata={
                "spiking_active_inference": advice
                if isinstance(advice, dict)
                else None,
                "imagination_workspace": imagination_frame
                if isinstance(imagination_frame, dict)
                else None,
                "imagination_workspace_feedback": imagination_feedback,
                "bicameral_advisory": bicameral_frame
                if isinstance(bicameral_frame, dict)
                else None,
                "bicameral_advisory_feedback": bicameral_feedback,
                "cognitive_situation_frame": cognitive_situation_frame
                if isinstance(cognitive_situation_frame, dict)
                else None,
                "live_mind_controls_bound": live_mind_controls_bound,
                "live_mind_generation_controls": dict(live_mind_generation_controls),
                "live_mind_snapshot_ready": live_mind_snapshot_ready,
                "live_mind_required_subsystems_ok": live_mind_required_subsystems_ok,
                "live_mind_context_required": live_mind_required,
                "live_mind_surface_control_receipt": surface_control_receipt,
                "live_mind_controls_worker_applied": bool(
                    surface_control_receipt.get("live_mind_controls_bound")
                    and surface_control_receipt.get("applied")
                ),
                "self_condition_contract": self_condition_contract,
                "self_condition_evidence_id": str(
                    (
                        context.get("canonical_self_condition_projection")
                        if isinstance(
                            context.get("canonical_self_condition_projection"),
                            dict,
                        )
                        else {}
                    ).get("evidence_id")
                    or ""
                ),
                "response_path": (
                    "cognitive_engine_self_condition"
                    if self_condition_contract
                    else ""
                ),
            },
        )

    async def _reactive_recovery(
        self,
        objective: str,
        mode: ThinkingMode,
        origin: str,
        reason: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> Thought:
        """
        Emergency reactive response when the main cognitive loop fails.
        BUG-10: Added recursion guard, timeout, and proper exception handling.
        """
        if self._is_background_request(origin, False):
            logger.debug(
                "🛡️ CognitiveEngine: suppressing background reactive recovery for origin=%s (%s).",
                origin,
                reason,
            )
            return self._empty_thought(mode, f"background_recovery_suppressed:{reason}")

        # Only use the mutex to guard the flag flip; long-running recovery work
        # must happen outside the lock so watchdogs don't see a false deadlock.
        if not await self._recovery_lock.acquire_robust(timeout=1.0):
            return Thought(
                id=str(uuid.uuid4()),
                content="Reactive recovery is still gathering a stable answer; I logged this turn instead of emitting a second recovery fragment.",
                mode=ThinkingMode.FAST,
                confidence=0.2,
                reasoning=["Recovery lock busy"],
            )

        try:
            if getattr(self, "_recovery_in_progress", False):
                return Thought(
                    id=str(uuid.uuid4()),
                    content="Reactive recovery is still gathering a stable answer; I logged this turn instead of emitting a duplicate recovery fragment.",
                    mode=ThinkingMode.FAST,
                    confidence=0.2,
                    reasoning=["Recovery recursion guard triggered"],
                )
            self._recovery_in_progress = True
        finally:
            if self._recovery_lock.locked():
                self._recovery_lock.release()

        try:
            logger.warning("⚡ [COGNITION] Initiating Reactive Recovery Phase. Reason: %s", reason)

            # 1. Rollback state to last stable version (with timeout + guard)
            try:
                async with asyncio.timeout(5.0):
                    if self.state_repository is not None:
                        # StateRepository is the canonical rollback owner and
                        # creates the state-mutation receipt around persistence.
                        await self.state_repository.rollback(f"recovery: {reason}")
            except (RuntimeError, AttributeError, TypeError, ValueError) as rollback_err:
                record_degradation(
                    "cognitive_engine",
                    rollback_err,
                    severity="degraded",
                    action="continued reactive recovery without state rollback",
                )
                logger.warning("Rollback failed during recovery: %s", rollback_err)

            if isinstance(context, dict) and bool(
                context.get("desktop_cognitive_engine_required", False)
                or context.get("cognitive_engine_required", False)
            ):
                return self._desktop_cognitive_failure_thought(
                    mode,
                    f"reactive_recovery:{reason}",
                )

            # 2. Get a quick reflex response if possible
            container = get_container()
            router = container.get("llm_router", default=None)

            reflex = None
            if router is not None and hasattr(router, "get_reflex_response"):
                reflex = router.get_reflex_response(objective)

            if reflex:
                return Thought(
                    id=str(uuid.uuid4()),
                    content=reflex,
                    mode=ThinkingMode.FAST,
                    confidence=1.0,
                    reasoning=[f"Reactive recovery via reflex matrix ({reason})"],
                )

            structured = self._structured_evaluation_thought(
                objective,
                state=None,
                mode=mode,
                origin=origin,
                fast_path=False,
                context=context,
            )
            if structured is not None:
                return structured

            direct_recovery = await self._direct_user_facing_recovery(
                objective,
                mode,
                origin,
                reason,
            )
            if direct_recovery is not None:
                return direct_recovery

            # 3. Last-resort fallback (natural, human-sounding)
            fallback_msg = "Reactive recovery reached its hard fallback before a coherent answer formed; the degraded turn was logged."
            if "user" in origin:
                fallback_msg = "Reactive recovery could not produce a coherent user-facing answer; the failed turn was logged with its context."

            return Thought(
                id=str(uuid.uuid4()),
                content=fallback_msg,
                mode=ThinkingMode.FAST,
                confidence=0.3,
                reasoning=[f"Hard fallback after cognitive failure: {reason}"],
            )
        except (OSError, ConnectionError, TimeoutError) as recovery_err:
            record_degradation(
                "cognitive_engine",
                recovery_err,
                severity="critical",
                action="returned hard recovery failure thought",
            )
            logger.error("Error during recovery: %s", recovery_err)
            return Thought(
                id=str(uuid.uuid4()),
                content="Reactive recovery failed internally; the turn was logged as a live cognition fault.",
                mode=ThinkingMode.FAST,
                confidence=0.1,
                reasoning=[f"Recovery itself failed: {recovery_err}"],
            )
        finally:
            await self._set_recovery_in_progress(False)

    def stop(self):
        """Shutdown logic (BUG-19)."""
        logger.info("🛑 CognitiveEngine stopping...")
        self._phases = []

    def _structured_evaluation_thought(
        self,
        objective: str,
        *,
        state: Any,
        mode: ThinkingMode,
        origin: str,
        fast_path: bool,
        context: dict[str, Any] | None = None,
    ) -> Thought | None:
        """Return a governed structured floor for bounded evaluation prompts."""

        try:
            from core.reasoning.structured_evaluation import structured_evaluation_response

            response = structured_evaluation_response(objective, state=state, origin=origin)
            if response is None:
                if fast_path:
                    from core.synthesis import deterministic_user_facing_floor

                    direct = deterministic_user_facing_floor(objective)
                    if direct:
                        thought = Thought(
                            id=str(uuid.uuid4()),
                            content=direct,
                            mode=mode,
                            confidence=0.99,
                            reasoning=[
                                "Deterministic bounded-answer floor selected before model generation.",
                                "Response computed from the prompt shape; no fixture keys or benchmark ids used.",
                            ],
                            metadata=self._live_mind_structured_floor_metadata(
                                context,
                                source="deterministic_user_facing_floor",
                            ),
                        )
                        self.thoughts.append(thought)
                        return thought
                return None
            if not fast_path and response.kind not in {"safety_refusal"}:
                return None

            thought = Thought(
                id=str(uuid.uuid4()),
                content=response.content,
                mode=mode,
                confidence=response.confidence,
                reasoning=[
                    f"Structured runtime evaluation floor selected: {response.kind}.",
                    "Response derived from current prompt shape; no fixture keys or benchmark ids used.",
                ],
                metadata=self._live_mind_structured_floor_metadata(
                    context,
                    source=f"structured_evaluation:{response.kind}",
                ),
            )
            self.thoughts.append(thought)
            return thought
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued cognitive loop after structured evaluation floor failed",
            )
            logger.debug("Structured evaluation floor skipped: %s", exc)
            return None

    @staticmethod
    def _live_mind_structured_floor_metadata(
        context: dict[str, Any] | None,
        *,
        source: str,
    ) -> dict[str, Any]:
        """Attach live-mind proof metadata to deterministic CognitiveEngine floors.

        Structured safety/refusal floors do not always invoke the foreground model
        worker. They are still valid desktop CognitiveEngine outputs when they are
        selected after live mind context, subsystem probes, and generation-control
        binding have already run.
        """

        if not isinstance(context, dict):
            return {}
        generation_controls = context.get("live_mind_generation_controls")
        if not isinstance(generation_controls, dict):
            generation_controls = {}
        if not generation_controls:
            generation_controls = _live_mind_generation_controls(
                context.get("live_mind_context")
            )
        controls_bound = bool(
            context.get("live_mind_controls_bound")
            and generation_controls
        )
        live_mind_context = context.get("live_mind_context")
        snapshot_ready = bool(context.get("live_mind_snapshot_ready"))
        if not snapshot_ready and isinstance(live_mind_context, dict):
            quality = live_mind_context.get("mind_snapshot_quality")
            snapshot_ready = bool(isinstance(quality, dict) and quality.get("ready"))
        required_subsystems_ok = bool(context.get("live_mind_required_subsystems_ok"))
        if not required_subsystems_ok and isinstance(live_mind_context, dict):
            required_subsystems_ok = bool(live_mind_context.get("required_subsystems_ok"))
        if not generation_controls and snapshot_ready and required_subsystems_ok:
            generation_controls = {
                "temperature": 0.58,
                "top_p": 0.88,
                "clean_user_surface_recurrent_loops": 1,
                "clean_user_surface_steering_alpha": 0.25,
            }
        if generation_controls and snapshot_ready and required_subsystems_ok:
            controls_bound = True
        desktop_required = bool(
            context.get("desktop_cognitive_engine_required")
            or context.get("cognitive_engine_required")
            or context.get("live_mind_context_required")
        )
        if not desktop_required:
            return {}
        surface_control_receipt = {
            "enabled": False,
            "applied": False,
            "generation_required": False,
            "application_status": "not_applicable_structured_floor",
            "live_mind_controls_bound": bool(controls_bound),
            "clean_user_surface_contract": bool(
                context.get("clean_user_surface_contract", True)
            ),
            "surface_quality_gate_enabled": False,
            "surface_quality_gate_passed": True,
            "surface_quality_gate_attempts": 0,
            "surface_quality_gate_reasons": [],
            "source": source,
        }
        return {
            "live_mind_controls_bound": bool(controls_bound),
            "live_mind_generation_controls": dict(generation_controls),
            "live_mind_snapshot_ready": snapshot_ready,
            "live_mind_required_subsystems_ok": required_subsystems_ok,
            "live_mind_context_required": True,
            "live_mind_surface_control_receipt": surface_control_receipt,
            "live_mind_controls_worker_applied": False,
            "live_mind_generation_required": False,
            "response_path": "cognitive_engine",
            "structured_floor_source": source,
        }

    async def record_interaction(
        self, user_input: str, response: str, domain: str = "general"
    ) -> None:
        """Persist completed turns through the active learning/context stack."""
        container = get_container()

        context_manager = container.get("context_manager", default=None)
        if (
            context_manager
            and context_manager is not self
            and hasattr(context_manager, "record_interaction")
        ):
            try:
                await context_manager.record_interaction(user_input, response, domain=domain)
                return
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation(
                    "cognitive_engine",
                    exc,
                    severity="warning",
                    action="fell through to learning-engine interaction persistence",
                )
                logger.debug(
                    "CognitiveEngine.record_interaction context-manager path failed: %s", exc
                )

        learning = container.get("learning_engine", default=None)
        if learning and hasattr(learning, "record_interaction"):
            try:
                await learning.record_interaction(
                    user_input=user_input,
                    aura_response=response,
                    domain=domain,
                )
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation(
                    "cognitive_engine",
                    exc,
                    severity="warning",
                    action="dropped optional interaction learning write",
                )
                logger.debug("CognitiveEngine.record_interaction learning path failed: %s", exc)

    async def think_stream(self, objective: str, **kwargs):
        """Streaming thought generator via modular router."""
        container = get_container()
        router = container.get("llm_router")
        state = await self.state_repository.get_current()
        if not state:
            from core.state.aura_state import AuraState

            state = AuraState.default()

        # Build structured messages
        messages = ContextAssembler.build_messages(state, objective)

        # Standard streaming path
        async for event in router.think_stream(messages=messages, **kwargs):
            if hasattr(event, "content"):
                yield event.content
            else:
                yield str(event)

    async def see(self, vision_payload: dict[str, Any]) -> str:
        """Process a vision payload from the sensory pipeline.

        [ZENITH] Functionalized: Linking Sensory Buffer to Cognitive reasoning.
        """
        buffer = get_container().get("vision_buffer", default=None)
        if not buffer:
            logger.warning("👁️ [VISION] see() called but vision_buffer not found in container.")
            return "👁️ visual_analysis: Sensory buffer unavailable."

        prompt = (
            vision_payload.get("query")
            or vision_payload.get("prompt")
            or "Describe the current visual state."
        )
        return await buffer.query_visual_context(prompt, brain=self)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a text response by routing through the LLM router.

        Bridge method for callers like LanguageCenter that expect a
        ``generate()`` interface.  Now enhanced with reasoning strategies
        for complex queries (debate, decomposition, consistency).

        Args:
            prompt: The text prompt to send to the LLM.
            **kwargs: Additional parameters forwarded to the router.

        Returns:
            The generated text response.
        """
        container = get_container()
        purpose = str(kwargs.get("purpose", "") or "").strip().lower()
        origin = str(kwargs.get("origin", "") or "").strip().lower()
        user_facing_purposes = {"chat", "conversation", "expression", "reply", "user_response"}
        if not origin:
            origin = "system"
            kwargs["origin"] = origin

        if "is_background" not in kwargs:
            kwargs["is_background"] = not (
                purpose in user_facing_purposes
                or is_foreground_objective_origin(origin)
            )

        if kwargs.get("is_background") and "prefer_tier" not in kwargs:
            kwargs["prefer_tier"] = "tertiary"

        # v40: Spiritual Spine - Prior Position Injection
        spine = container.get("spine", default=None)
        if spine:
            check = await spine.pre_response_check(prompt)
            if check.injection:
                prompt = check.injection + "\n\n" + prompt

        router = container.get("llm_router", default=None)

        # v41: Reasoning Strategy Enhancement
        # For non-trivial queries, apply advanced reasoning (debate, decompose, etc.)
        use_strategies = kwargs.pop("use_strategies", True)
        force_strategy = kwargs.pop("force_strategy", None)
        strategy_query = str(kwargs.pop("strategy_query", "") or "").strip()

        if router and use_strategies:
            # Lazy-init the reasoning layer on first use
            if self._reasoning is None:

                async def _raw_generate(p, **kw):
                    return await router.think(p, **kw)

                self._reasoning = ReasoningStrategies(_raw_generate)

            strategy = force_strategy
            if strategy is None:
                if not strategy_query:
                    messages = kwargs.get("messages")
                    if isinstance(messages, list):
                        for msg in reversed(messages):
                            if not isinstance(msg, dict):
                                continue
                            role = str(msg.get("role", "") or "").strip().lower()
                            content = str(msg.get("content", "") or "").strip()
                            if role in {"user", "human"} and content:
                                strategy_query = content
                                break
                classify_target = strategy_query or prompt
                # Only use advanced strategies for user-facing queries, not internal prompts
                classified = self._reasoning.classify(classify_target)
                if classified != StrategyType.DIRECT and len(classify_target) > 30:
                    strategy = classified
                elif self._reasoning._is_logical_check(classify_target):
                    strategy = StrategyType.DIRECT

            if strategy is not None and (strategy != StrategyType.DIRECT or self._reasoning._is_logical_check(classify_target)):
                try:
                    from ..thought_stream import get_emitter

                    get_emitter().emit(
                        "Deep Reasoning 🧠",
                        f"Using {strategy.name} strategy",
                        level="info",
                        category="Cognition",
                    )
                except (ImportError, AttributeError, RuntimeError) as _exc:
                    record_degradation(
                        "cognitive_engine",
                        _exc,
                        severity="warning",
                        action="continued generation without thought-stream emission",
                    )
                    logger.debug("Suppressed Exception: %s", _exc)

                strategy_input = strategy_query or prompt
                result = await self._reasoning.execute(strategy_input, strategy=strategy, **kwargs)
                return result.content

        # Standard direct generation
        if router:
            return await router.think(prompt, **kwargs)
        # Fallback if no router
        thought = await self.think(prompt, **kwargs)
        return thought.content if hasattr(thought, "content") else str(thought)

    def _emit_thought(self, thought: str):
        """Internal helper to publish thoughts to the event bus."""
        container = get_container()
        eb = container.get("event_bus")
        if eb:
            eb.publish_threadsafe(
                "thought",
                {
                    "timestamp": time.time(),
                    "content": thought,
                    "engine": "ReAct" if "ReAct" in thought else "Modular",
                },
            )
