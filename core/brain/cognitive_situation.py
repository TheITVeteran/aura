"""Unified semantic, analogical, and sensorimotor situation frame.

This module is a side-effect-free cognitive organ. It does not capture the
screen, click, type, write files, or call models. It reads the objective,
current runtime state, and already-owned perception/embodiment telemetry, then
emits a compact causal frame consumed by CognitiveEngine and response
generation. The point is to make semantic flexibility, analogical leaps, and
embodied grounding affect routing, sampling, attention, verification pressure,
and tool-governance posture through the same live path.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import record_degradation

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}")
_SEMANTIC_RE = re.compile(
    r"\b(mean|meaning|interpret|understand|explain|define|why|how|what is|"
    r"what are|what would|could|should|ambiguous|nuance|frame|concept|abstract|"
    r"metaphor|symbol|implication|criteria|label|ontology|paradigm)\b",
    re.IGNORECASE,
)
_ANALOGY_RE = re.compile(
    r"\b(analogy|analogical|metaphor|like|compare|pattern|connection|bridge|"
    r"leap|novel|creative|imagine|what would .* look like|model out|synthesize|"
    r"cross-domain|transfer|fictional|inspiration)\b",
    re.IGNORECASE,
)
_SENSORIMOTOR_RE = re.compile(
    r"\b(screen|see|look|visible|desktop|window|cursor|click|type|keyboard|mouse|"
    r"open|close|scroll|drag|drop|select|copy|paste|write|save|export|download|"
    r"upload|folder|file|pdf|notes|docs|chrome|browser|app|application|wallpaper|"
    r"image|camera|microphone|voice|hear|speak|tool|tools|external|real[- ]world)\b",
    re.IGNORECASE,
)
_ACTION_RE = re.compile(
    r"\b(open|click|type|write|save|export|download|upload|create|delete|move|"
    r"rename|install|run|execute|search|browse|change|set|send|commit|push)\b",
    re.IGNORECASE,
)
_UNCERTAINTY_RE = re.compile(
    r"\b(confus|unsure|uncertain|maybe|probably|hypothetical|prove|verify|test|"
    r"check|investigate|review|audit|does this work|is this true)\b",
    re.IGNORECASE,
)

_STOPWORDS = {
    "about",
    "again",
    "also",
    "and",
    "are",
    "because",
    "been",
    "being",
    "but",
    "can",
    "could",
    "does",
    "doing",
    "for",
    "from",
    "have",
    "how",
    "into",
    "just",
    "like",
    "make",
    "more",
    "need",
    "not",
    "now",
    "out",
    "that",
    "the",
    "then",
    "there",
    "this",
    "through",
    "want",
    "what",
    "when",
    "where",
    "with",
    "would",
    "you",
    "your",
}


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    if not math.isfinite(value):
        return lower
    return max(lower, min(upper, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _compact(value: Any, *, limit: int = 180) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _keywords(text: str, *, limit: int = 10) -> list[str]:
    seen: set[str] = set()
    words: list[str] = []
    for match in _WORD_RE.finditer(text.lower()):
        token = match.group(0).strip("'_-")
        if len(token) < 3 or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        words.append(token)
        if len(words) >= limit:
            break
    return words


def _stable_id(*parts: Any) -> str:
    payload = "\n".join(str(part or "") for part in parts)
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:16]


def _affect_value(state: Any, *names: str, default: float = 0.0) -> float:
    affect = getattr(state, "affect", None)
    emotions = getattr(affect, "emotions", None)
    values: list[float] = []
    for name in names:
        values.append(_safe_float(getattr(affect, name, default), default))
        if isinstance(emotions, dict):
            values.append(_safe_float(emotions.get(name), default))
    return max(values) if values else default


def _read_status(service: Any) -> dict[str, Any]:
    if service is None:
        return {}
    for attr in ("get_status", "status", "snapshot", "to_dict"):
        fn = getattr(service, attr, None)
        if not callable(fn):
            continue
        try:
            value = fn()
        except (OSError, ConnectionError, TimeoutError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "cognitive_situation",
                exc,
                severity="warning",
                action=f"skipped unreadable {type(service).__name__}.{attr} status",
            )
            continue
        if isinstance(value, dict):
            return dict(value)
    return {}


def _service_state(name: str) -> tuple[bool, dict[str, Any]]:
    try:
        service = ServiceContainer.get(name, default=None)
    except (OSError, ConnectionError, TimeoutError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation(
            "cognitive_situation",
            exc,
            severity="warning",
            action=f"treated {name} as unavailable while building situation frame",
        )
        return False, {}
    return service is not None, _read_status(service)


@dataclass(frozen=True)
class CognitiveSituationFrame:
    frame_id: str
    objective: str
    semantic_flexibility: float
    analogical_leap_pressure: float
    sensorimotor_grounding: float
    abstraction_level: float
    ambiguity: float
    verification_pressure: float
    metacognition_pressure: float
    keywords: list[str] = field(default_factory=list)
    semantic_interpretations: list[dict[str, Any]] = field(default_factory=list)
    analogy_bridges: list[dict[str, str]] = field(default_factory=list)
    embodied_affordances: list[str] = field(default_factory=list)
    perception_summary: dict[str, Any] = field(default_factory=dict)
    attention_targets: list[str] = field(default_factory=list)
    routing_bias: dict[str, bool] = field(default_factory=dict)
    sampling_bias: dict[str, float] = field(default_factory=dict)
    causal_effects: dict[str, Any] = field(default_factory=dict)
    governance: dict[str, Any] = field(
        default_factory=lambda: {
            "side_effect_free": True,
            "reads_existing_perception_only": True,
            "screen_capture_owner": "core/perception/screen_perception.py",
            "external_effects_require_authority_gateway": True,
            "claims_require_receipts": True,
        }
    )
    created_at: float = field(default_factory=time.time)

    @property
    def salience(self) -> float:
        return _clamp(
            max(
                self.semantic_flexibility,
                self.analogical_leap_pressure,
                self.sensorimotor_grounding,
                self.ambiguity,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["salience"] = self.salience
        return payload

    def prompt_block(self, *, compact: bool = False) -> str:
        if self.salience < 0.16:
            return ""
        if compact:
            directives = [
                "COGNITIVE SITUATION FRAME: use semantic alternatives, analogies, and embodiment only as causal grounding.",
            ]
            if self.semantic_flexibility >= 0.35:
                directives.append("Disambiguate the current user intent before carrying older topics forward.")
            if self.analogical_leap_pressure >= 0.35:
                directives.append("Use a relevant analogy or cross-domain bridge when it improves the answer.")
            if self.sensorimotor_grounding >= 0.30:
                directives.append("Ground screen/tool claims in observed state or governed receipts before claiming completion.")
            return " ".join(directives)

        lines = [
            "## COGNITIVE SITUATION FRAME",
            (
                f"Semantic flexibility={self.semantic_flexibility:.2f}; "
                f"analogical pressure={self.analogical_leap_pressure:.2f}; "
                f"sensorimotor grounding={self.sensorimotor_grounding:.2f}; "
                f"ambiguity={self.ambiguity:.2f}."
            ),
        ]
        if self.semantic_interpretations:
            rendered = "; ".join(
                f"{item.get('label')}: {item.get('focus')}"
                for item in self.semantic_interpretations[:3]
            )
            lines.append(f"Candidate interpretations: {rendered}.")
        if self.analogy_bridges:
            rendered = "; ".join(
                f"{item.get('source')} -> {item.get('target')}: {item.get('relation')}"
                for item in self.analogy_bridges[:3]
            )
            lines.append(f"Analogy bridges: {rendered}.")
        if self.embodied_affordances:
            lines.append(
                "Embodied affordances: " + ", ".join(self.embodied_affordances[:5]) + "."
            )
        if self.attention_targets:
            lines.append("Attention targets: " + ", ".join(self.attention_targets[:5]) + ".")
        lines.append(
            "This frame is causal grounding, not prose to recite. It may change routing, "
            "sampling, verification, and attention. It does not prove perception or tool completion."
        )
        return "\n".join(lines) + "\n\n"


class CognitiveSituationEngine:
    """Build a unified live-turn situation frame without side effects."""

    def __init__(self) -> None:
        self.frames_built = 0
        self.last_frame: CognitiveSituationFrame | None = None

    def get_status(self) -> dict[str, Any]:
        latest = self.last_frame.to_dict() if self.last_frame is not None else None
        return {
            "running": True,
            "frames_built": self.frames_built,
            "latest": latest,
            "governance": {
                "side_effect_free": True,
                "external_effects_require_authority_gateway": True,
                "claims_require_receipts": True,
            },
        }

    status = get_status

    def frame(
        self,
        objective: str,
        *,
        state: Any = None,
        context: dict[str, Any] | None = None,
        origin: str = "system",
        is_background: bool = False,
    ) -> CognitiveSituationFrame:
        text = " ".join(str(objective or "").split())
        lower = text.lower()
        words = _keywords(text)
        context = context if isinstance(context, dict) else {}

        semantic_hits = len(_SEMANTIC_RE.findall(lower))
        analogy_hits = len(_ANALOGY_RE.findall(lower))
        sensor_hits = len(_SENSORIMOTOR_RE.findall(lower))
        action_hits = len(_ACTION_RE.findall(lower))
        uncertainty_hits = len(_UNCERTAINTY_RE.findall(lower))

        curiosity = _affect_value(state, "curiosity", "wonder", "interest", default=0.0)
        confusion = _affect_value(state, "confused", "uncertainty", default=0.0)
        frustration = _affect_value(state, "frustration", "upset", default=0.0)

        live_desktop = bool(
            context.get("desktop_cognitive_engine_required")
            or context.get("desktop_quick_reply_contract")
            or str(origin or "").startswith(("desktop", "voice", "user"))
        )
        live_mind_required = bool(context.get("live_mind_context_required"))
        has_screen_context = bool(
            context.get("screen_context")
            or context.get("desktop_task_contract")
            or context.get("desktop_execution_contract")
        )

        semantic = _clamp(
            0.10
            + 0.10 * min(4, semantic_hits)
            + 0.10 * min(2, uncertainty_hits)
            + 0.15 * curiosity
            + 0.20 * confusion
            + (0.10 if len(words) >= 6 else 0.0)
        )
        analogy = _clamp(
            0.04
            + 0.14 * min(4, analogy_hits)
            + 0.18 * curiosity
            + (0.12 if semantic >= 0.42 else 0.0)
            + (0.08 if "fictional" in lower or "inspiration" in lower else 0.0)
        )
        sensorimotor = _clamp(
            0.04
            + 0.12 * min(5, sensor_hits)
            + 0.10 * min(3, action_hits)
            + (0.14 if live_desktop else 0.0)
            + (0.12 if has_screen_context else 0.0)
        )
        ambiguity = _clamp(
            0.08
            + 0.12 * min(3, uncertainty_hits)
            + 0.16 * confusion
            + (0.10 if len(words) >= 8 else 0.0)
            + (0.08 if "or" in lower or "maybe" in lower else 0.0)
        )
        abstraction = _clamp(0.15 + 0.16 * min(4, semantic_hits) + 0.12 * min(3, analogy_hits))
        verification = _clamp(
            max(
                0.10 + 0.15 * min(3, uncertainty_hits) + 0.20 * sensorimotor,
                0.35 if action_hits else 0.0,
                0.42 if live_mind_required and sensorimotor >= 0.25 else 0.0,
            )
        )
        metacognition = _clamp(0.20 + 0.35 * ambiguity + 0.15 * semantic + 0.10 * frustration)

        perception_summary = self._perception_summary()
        embodied_affordances = self._embodied_affordances(lower, sensorimotor, perception_summary)
        interpretations = self._semantic_interpretations(text, words, semantic, sensorimotor)
        bridges = self._analogy_bridges(words, analogy, sensorimotor)
        attention_targets = self._attention_targets(
            words,
            semantic=semantic,
            analogy=analogy,
            sensorimotor=sensorimotor,
            ambiguity=ambiguity,
            embodied_affordances=embodied_affordances,
        )

        routing_bias = {
            "raise_metacognition": metacognition >= 0.35,
            "seek_verification": verification >= 0.35,
            "use_tool_gateway": sensorimotor >= 0.35 or action_hits > 0,
            "preserve_conversation_context": semantic >= 0.35 or ambiguity >= 0.35,
            "use_imagination": analogy >= 0.30,
            "use_analogy": analogy >= 0.35,
            "bind_sensorimotor_evidence": sensorimotor >= 0.30,
            "requires_memory_grounding": semantic >= 0.45 or ambiguity >= 0.42,
            "deliberate_mode": not is_background and (semantic >= 0.52 or ambiguity >= 0.48 or sensorimotor >= 0.62),
        }
        if is_background:
            routing_bias["deliberate_mode"] = False

        sampling_bias = {
            "temperature_delta": _clamp(0.05 * analogy + 0.03 * semantic - 0.06 * sensorimotor, -0.12, 0.12),
            "max_tokens_factor": _clamp(
                1.0 + 0.08 * semantic + 0.06 * analogy + 0.05 * ambiguity - 0.04 * sensorimotor,
                0.80,
                1.18,
            ),
            "top_p_delta": _clamp(0.03 * analogy - 0.02 * sensorimotor, -0.05, 0.05),
        }

        causal_effects = {
            "semantic_flexibility_pressure": round(semantic, 4),
            "analogical_leap_pressure": round(analogy, 4),
            "sensorimotor_grounding_pressure": round(sensorimotor, 4),
            "verification_pressure": round(verification, 4),
            "metacognition_depth": round(metacognition, 4),
            "attention_focus": attention_targets,
            "tool_governance_pressure": bool(routing_bias["use_tool_gateway"]),
            "screen_or_body_evidence_available": bool(
                perception_summary.get("screen_perception_available")
                or perception_summary.get("perception_daemon_available")
                or perception_summary.get("embodiment_available")
            ),
        }

        frame = CognitiveSituationFrame(
            frame_id=_stable_id(text, origin, time.time() // 60),
            objective=_compact(text, limit=240),
            semantic_flexibility=semantic,
            analogical_leap_pressure=analogy,
            sensorimotor_grounding=sensorimotor,
            abstraction_level=abstraction,
            ambiguity=ambiguity,
            verification_pressure=verification,
            metacognition_pressure=metacognition,
            keywords=words,
            semantic_interpretations=interpretations,
            analogy_bridges=bridges,
            embodied_affordances=embodied_affordances,
            perception_summary=perception_summary,
            attention_targets=attention_targets,
            routing_bias=routing_bias,
            sampling_bias=sampling_bias,
            causal_effects=causal_effects,
        )
        self.frames_built += 1
        self.last_frame = frame
        return frame

    def _perception_summary(self) -> dict[str, Any]:
        screen_available, screen_status = _service_state("screen_perception")
        daemon_available, daemon_status = _service_state("perception_daemon")
        runtime_available, runtime_status = _service_state("perception_runtime")
        embodiment_available, embodiment_status = _service_state("embodiment")
        if not embodiment_available:
            embodiment_available, embodiment_status = _service_state("embodiment_system")
        world_available, world_status = _service_state("world_bridge")

        def compact_status(status: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
            return {
                key: status.get(key)
                for key in keys
                if status.get(key) not in (None, "", [], {})
            }

        return {
            "screen_perception_available": screen_available,
            "screen": compact_status(
                screen_status,
                ("active_app", "frontmost_app", "focused_control", "last_capture_at", "permissions_ok"),
            ),
            "perception_daemon_available": daemon_available,
            "perception_daemon": compact_status(
                daemon_status,
                ("running", "healthy", "last_observation_at", "active_app", "last_error"),
            ),
            "perception_runtime_available": runtime_available,
            "perception_runtime": compact_status(
                runtime_status,
                ("running", "healthy", "last_frame_at", "last_error"),
            ),
            "embodiment_available": embodiment_available,
            "embodiment": compact_status(
                embodiment_status,
                ("running", "healthy", "permissions_ok", "last_action_at", "last_error"),
            ),
            "world_bridge_available": world_available,
            "world_bridge": compact_status(
                world_status,
                ("running", "healthy", "last_observation_at", "last_error"),
            ),
        }

    @staticmethod
    def _semantic_interpretations(
        text: str,
        words: list[str],
        semantic: float,
        sensorimotor: float,
    ) -> list[dict[str, Any]]:
        focus = ", ".join(words[:4]) if words else _compact(text, limit=80)
        interpretations = [
            {
                "label": "literal_request",
                "focus": focus or "current user objective",
                "weight": round(_clamp(0.45 + 0.25 * (1.0 - semantic)), 3),
            },
            {
                "label": "conceptual_frame",
                "focus": "meaning, criteria, and implications",
                "weight": round(_clamp(0.20 + 0.50 * semantic), 3),
            },
        ]
        if sensorimotor >= 0.30:
            interpretations.append(
                {
                    "label": "operational_frame",
                    "focus": "visible environment, tool state, and effect evidence",
                    "weight": round(_clamp(0.25 + 0.50 * sensorimotor), 3),
                }
            )
        return interpretations

    @staticmethod
    def _analogy_bridges(
        words: list[str],
        analogy: float,
        sensorimotor: float,
    ) -> list[dict[str, str]]:
        if analogy < 0.22:
            return []
        seed = words[0] if words else "objective"
        bridges = [
            {
                "source": seed,
                "target": "navigation",
                "relation": "treat ambiguous intent like a route with landmarks and checkpoints",
            },
            {
                "source": seed,
                "target": "engineering",
                "relation": "turn claims into interfaces, tests, receipts, and rollback paths",
            },
        ]
        if sensorimotor >= 0.30:
            bridges.append(
                {
                    "source": "screen",
                    "target": "body",
                    "relation": "bind perception to action only through observable affordances",
                }
            )
        return bridges

    @staticmethod
    def _embodied_affordances(
        lower: str,
        sensorimotor: float,
        perception_summary: dict[str, Any],
    ) -> list[str]:
        if sensorimotor < 0.22:
            return []
        affordances: list[str] = []
        if any(word in lower for word in ("screen", "see", "visible", "look")):
            affordances.append("inspect screen state before claiming what is visible")
        if any(word in lower for word in ("open", "click", "type", "write", "save", "export")):
            affordances.append("route external actions through governed desktop/tool execution")
        if any(word in lower for word in ("browser", "chrome", "docs", "notes", "pdf", "folder")):
            affordances.append("verify target app, focus, and artifact path after each step")
        if perception_summary.get("screen_perception_available"):
            affordances.append("use existing screen perception telemetry when available")
        if perception_summary.get("embodiment_available"):
            affordances.append("bind action planning to embodiment permission state")
        return affordances[:6]

    @staticmethod
    def _attention_targets(
        words: list[str],
        *,
        semantic: float,
        analogy: float,
        sensorimotor: float,
        ambiguity: float,
        embodied_affordances: list[str],
    ) -> list[str]:
        targets: list[str] = []
        if semantic >= 0.30:
            targets.append("current-intent-semantics")
        if ambiguity >= 0.30:
            targets.append("ambiguity-resolution")
        if analogy >= 0.30:
            targets.append("cross-domain-bridge")
        if sensorimotor >= 0.30:
            targets.append("sensorimotor-evidence")
        if embodied_affordances:
            targets.append("tool-effect-verification")
        targets.extend(words[:3])
        seen: set[str] = set()
        ordered: list[str] = []
        for target in targets:
            if target and target not in seen:
                seen.add(target)
                ordered.append(target)
        return ordered[:8]


_ENGINE: CognitiveSituationEngine | None = None


def get_cognitive_situation_engine() -> CognitiveSituationEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = CognitiveSituationEngine()
    try:
        current = ServiceContainer.get("cognitive_situation", default=None)
        if current is not _ENGINE:
            ServiceContainer.register_instance(
                "cognitive_situation",
                _ENGINE,
                required=False,
                owner="core/brain/cognitive_situation.py",
                registered_by="core.brain.cognitive_situation.get_cognitive_situation_engine",
                required_for="semantic flexibility, analogical routing, and sensorimotor grounding",
                failure_policy="degrade_with_receipt",
            )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "cognitive_situation",
            exc,
            severity="warning",
            action="continued without ServiceContainer registration for cognitive situation engine",
        )
    return _ENGINE


def render_cognitive_situation_prompt_block(frame: dict[str, Any], *, compact: bool = False) -> str:
    if not isinstance(frame, dict):
        return ""
    try:
        salience = _safe_float(frame.get("salience"), 0.0)
        if salience < 0.16:
            return ""
        semantic = _safe_float(frame.get("semantic_flexibility"), 0.0)
        analogy = _safe_float(frame.get("analogical_leap_pressure"), 0.0)
        sensorimotor = _safe_float(frame.get("sensorimotor_grounding"), 0.0)
        ambiguity = _safe_float(frame.get("ambiguity"), 0.0)
        if compact:
            directives = [
                "COGNITIVE SITUATION FRAME: use semantic alternatives, analogies, and embodiment as causal grounding only.",
            ]
            if semantic >= 0.35 or ambiguity >= 0.35:
                directives.append("Resolve current-turn meaning before continuing older topics.")
            if analogy >= 0.35:
                directives.append("Use a relevant analogy when it helps.")
            if sensorimotor >= 0.30:
                directives.append("Ground screen/tool claims in observed state or receipts.")
            return " ".join(directives) + "\n\n"

        lines = [
            "## COGNITIVE SITUATION FRAME",
            (
                f"Semantic flexibility={semantic:.2f}; analogical pressure={analogy:.2f}; "
                f"sensorimotor grounding={sensorimotor:.2f}; ambiguity={ambiguity:.2f}."
            ),
        ]
        interpretations = frame.get("semantic_interpretations") or []
        if isinstance(interpretations, list) and interpretations:
            rendered = "; ".join(
                f"{item.get('label')}: {item.get('focus')}"
                for item in interpretations[:3]
                if isinstance(item, dict)
            )
            if rendered:
                lines.append(f"Candidate interpretations: {rendered}.")
        affordances = frame.get("embodied_affordances") or []
        if isinstance(affordances, list) and affordances:
            lines.append("Embodied affordances: " + ", ".join(map(str, affordances[:5])) + ".")
        lines.append(
            "This frame changes routing, sampling, attention, and verification; do not recite it."
        )
        return "\n".join(lines) + "\n\n"
    except (TypeError, ValueError, AttributeError) as exc:
        record_degradation(
            "cognitive_situation",
            exc,
            severity="warning",
            action="skipped malformed cognitive situation prompt block",
        )
        return ""
