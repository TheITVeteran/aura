from __future__ import annotations

import hashlib
import re
from enum import Enum
from typing import Any

from core.goals.goal_text import normalize_goal_text
from core.runtime.turn_analysis import analyze_turn

_FOREGROUND_ORIGINS = frozenset(
    {
        "admin",
        "api",
        "chat",
        "chat_api",
        "desktop",
        "desktop_task",
        "desktop_ui",
        "direct",
        "external",
        "frontend",
        "gui",
        "http",
        "interface",
        "live",
        "live_chat",
        "native",
        "native_shell",
        "owner",
        "routing_user",
        "routing_voice_command",
        "tauri",
        "test",
        "ui",
        "user",
        "voice",
        "voice_bridge",
        "voice_input",
        "web_ui",
        "websocket",
        "ws",
    }
)

_ACTION_MARKERS = (
    "analyze",
    "apply",
    "archive",
    "audit",
    "backup",
    "browse",
    "build",
    "check",
    "clean",
    "close",
    "collect",
    "compare",
    "complete",
    "configure",
    "continue",
    "convert",
    "copy",
    "create",
    "debug",
    "delete",
    "deploy",
    "design",
    "develop",
    "diagnose",
    "discover",
    "document",
    "download",
    "edit",
    "ensure",
    "evaluate",
    "execute",
    "explore",
    "export",
    "extract",
    "find",
    "finish",
    "fix",
    "generate",
    "help",
    "implement",
    "import",
    "improve",
    "inspect",
    "install",
    "investigate",
    "learn",
    "launch",
    "maintain",
    "make",
    "measure",
    "migrate",
    "monitor",
    "move",
    "open",
    "optimize",
    "organize",
    "patch",
    "plan",
    "protect",
    "publish",
    "read",
    "reconcile",
    "refactor",
    "remove",
    "repair",
    "replace",
    "reproduce",
    "reset",
    "resolve",
    "restore",
    "retrieve",
    "rollback",
    "remember",
    "research",
    "resume",
    "review",
    "run",
    "save",
    "schedule",
    "search",
    "send",
    "set",
    "ship",
    "stabilize",
    "stop",
    "synthesize",
    "sync",
    "test",
    "trace",
    "triage",
    "uninstall",
    "update",
    "upgrade",
    "upload",
    "validate",
    "verify",
    "write",
)

_ACTION_MARKER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(marker) for marker in _ACTION_MARKERS) + r")\b",
    re.IGNORECASE,
)
_ACTION_START_RE = re.compile(
    r"^(?:please\s+)?(?:"
    + "|".join(re.escape(marker) for marker in _ACTION_MARKERS)
    + r")\b",
    re.IGNORECASE,
)

_EXPLICIT_TASK_BINDING_RE = re.compile(
    r"^(?:"
    r"i\s+(?:need|want|expect|require|would\s+like)\s+you\s+to|"
    r"we\s+(?:need|must|should|have\s+to)\s+|"
    r"you\s+(?:need|must|should|have\s+to)\s+|"
    r"(?:the|your)\s+(?:task|mission|project|goal)\s+is\s+to\s+|"
    r"let(?:'s|\s+us)\s+|keep\s+|make\s+sure\s+|"
    r"do\s+not\s+|don't\s+|never\s+"
    r")",
    re.IGNORECASE,
)

_MODAL_REQUEST_RE = re.compile(
    r"^(?:please\s+)?(?:can|could|would|will)\s+you\s+(?P<body>.+)$",
    re.IGNORECASE,
)

_INTERROGATIVE_RE = re.compile(
    r"^(?:what|when|where|which|who|whose|why|how|"
    r"am|are|is|was|were|do|does|did|have|has|had)\b",
    re.IGNORECASE,
)

_ONE_TURN_RESPONSE_RE = re.compile(
    r"^(?:(?:please|kindly)\s+|(?:can|could|would|will)\s+you\s+)?"
    r"(?:answer|calculate|compose\s+(?:a\s+)?(?:poem|joke|haiku)|"
    r"describe|explain|give\s+me|reply|respond|share|summarize|tell\s+me|"
    r"translate|write\s+(?:a\s+)?(?:short\s+)?(?:poem|joke|haiku))\b",
    re.IGNORECASE,
)

_TURN_SCOPED_RESPONSE_RE = re.compile(
    r"^(?:latency|quality|response|conversation)\s+sample\s+\d+\s*:",
    re.IGNORECASE,
)

_CONVERSATIONAL_STATEMENT_RE = re.compile(
    r"^(?:hey|hi|hello|thanks|thank\s+you|sorry|"
    r"i\b|i'm\b|im\b|we\b|we're\b|you\b|you're\b|"
    r"it\b|it's\b|this\b|that\b|there\b|my\b|your\b)",
    re.IGNORECASE,
)

_DIRECT_CHAT_MARKERS = (
    "are you ok",
    "are you okay",
    "anything you want to talk about",
    "bear with me",
    "quick check in",
    "how are you",
    "how can i help",
    "how do you feel",
    "how you feel",
    "feeling fine",
    "sorry,",
    "tell me more",
    "what do you think",
    "what parts did you find",
    "what were we talking about",
    "you with me",
    "your thoughts",
)

_CLOSURE_SOURCES = frozenset(
    {
        "executive_closure",
    }
)

_EXPLICIT_BINDING_KEYS = (
    "commitment_id",
    "intention_id",
    "mission_id",
    "plan_id",
    "project_id",
    "task_id",
)


class ForegroundObjectiveDisposition(str, Enum):
    CHAT = "chat"
    TASK = "task"
    UNKNOWN = "unknown"


def normalize_objective_origin(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(":", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


def is_foreground_objective_origin(value: Any) -> bool:
    normalized = normalize_objective_origin(value)
    if not normalized:
        return False
    if normalized in _FOREGROUND_ORIGINS:
        return True
    return normalized.startswith(
        (
            "api_",
            "desktop_",
            "native_",
            "owner_",
            "user_",
            "voice_",
            "websocket_",
            "ws_",
        )
    )


def classify_foreground_objective(value: Any) -> ForegroundObjectiveDisposition:
    """Classify only high-confidence foreground lifecycle semantics.

    Unknown language is deliberately preserved at restore boundaries and is not
    promoted into autonomous work. Normal successful turn finalization closes it
    regardless, so uncertainty cannot either erase unfinished work or leak a
    completed prompt into durable goals.
    """
    text = normalize_goal_text(value)
    lowered = text.casefold()
    if not lowered:
        return ForegroundObjectiveDisposition.UNKNOWN
    if _TURN_SCOPED_RESPONSE_RE.search(text) or _ONE_TURN_RESPONSE_RE.search(text):
        return ForegroundObjectiveDisposition.CHAT
    if _EXPLICIT_TASK_BINDING_RE.search(text):
        return ForegroundObjectiveDisposition.TASK

    modal_request = _MODAL_REQUEST_RE.search(text)
    if modal_request and _ACTION_MARKER_RE.search(modal_request.group("body")):
        return ForegroundObjectiveDisposition.TASK
    if _ACTION_START_RE.search(text):
        return ForegroundObjectiveDisposition.TASK
    if any(marker in lowered for marker in _DIRECT_CHAT_MARKERS):
        return ForegroundObjectiveDisposition.CHAT
    if text.rstrip().endswith("?") or _INTERROGATIVE_RE.search(text):
        return ForegroundObjectiveDisposition.CHAT

    analysis = analyze_turn(text)
    if analysis.intent_type in {"SKILL", "SYSTEM", "TASK"}:
        return ForegroundObjectiveDisposition.TASK
    if (
        analysis.is_execution_report
        or analysis.requires_live_aura_voice
        or _CONVERSATIONAL_STATEMENT_RE.search(text)
    ):
        return ForegroundObjectiveDisposition.CHAT
    return ForegroundObjectiveDisposition.UNKNOWN


def is_ephemeral_conversation_turn(value: Any) -> bool:
    """Return true only for a high-confidence non-durable dialogue turn."""

    return classify_foreground_objective(value) is ForegroundObjectiveDisposition.CHAT


def is_actionable_foreground_objective(value: Any) -> bool:
    """Return true only for foreground work safe to bind during the turn."""

    return classify_foreground_objective(value) is ForegroundObjectiveDisposition.TASK


def _metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    metadata = value.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def has_explicit_durable_binding(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    metadata = _metadata(value)
    if bool(
        value.get("explicit_goal")
        or value.get("continuity_obligation")
        or metadata.get("explicit_goal")
        or metadata.get("continuity_obligation")
    ):
        return True
    return any(
        bool(value.get(key) or metadata.get(key))
        for key in _EXPLICIT_BINDING_KEYS
    )


def is_executive_closure_projection(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    metadata = _metadata(value)
    labels = {
        str(value.get("source") or "").strip().lower(),
        str(value.get("type") or "").strip().lower(),
        str(metadata.get("source") or "").strip().lower(),
        str(metadata.get("kind") or "").strip().lower(),
        str(metadata.get("initiative_kind") or "").strip().lower(),
        str(metadata.get("initiative_source") or "").strip().lower(),
    }
    return bool(labels & _CLOSURE_SOURCES)


def _has_definitive_legacy_closure_provenance(value: Any) -> bool:
    """Recognize pre-foreground-flag closure rows without guessing by text."""

    if not isinstance(value, dict):
        return False
    metadata = _metadata(value)
    initiative_source = str(metadata.get("initiative_source") or "").strip().lower()
    initiative_kind = str(metadata.get("initiative_kind") or "").strip().lower()
    if initiative_source in _CLOSURE_SOURCES and initiative_kind in _CLOSURE_SOURCES:
        return True
    row_type = str(value.get("type") or "").strip().lower()
    metadata_source = str(metadata.get("source") or "").strip().lower()
    return row_type in _CLOSURE_SOURCES and metadata_source in _CLOSURE_SOURCES


def is_transient_foreground_projection(value: Any) -> bool:
    """Return true only for a provenance-confirmed foreground projection."""

    if not isinstance(value, dict) or has_explicit_durable_binding(value):
        return False
    metadata = _metadata(value)
    if bool(value.get("foreground_turn") or metadata.get("foreground_turn")):
        return True
    if (
        _has_definitive_legacy_closure_provenance(value)
        and is_ephemeral_conversation_turn(value)
    ):
        return True
    return bool(
        (value.get("continuity_restored") or metadata.get("continuity_restored"))
        and is_ephemeral_conversation_turn(value)
    )


def is_projection_of_completed_turn(value: Any, objective: Any) -> bool:
    objective_text = normalize_goal_text(objective)
    if not objective_text or normalize_goal_text(value) != objective_text:
        return False
    if not isinstance(value, dict) or has_explicit_durable_binding(value):
        return False
    return is_executive_closure_projection(value) or is_transient_foreground_projection(value)


def finalize_foreground_turn_state(
    state: Any,
    *,
    objective: Any,
    origin: Any,
) -> dict[str, Any]:
    """Close one foreground objective before state persistence.

    A newly selected background objective is preserved. Derived projections of
    the completed turn are removed, while explicit task/mission bindings remain.
    """

    receipt: dict[str, Any] = {
        "completed": False,
        "origin": normalize_objective_origin(origin),
        "objective_digest": "",
        "cleared_current": False,
        "preserved_background": False,
        "removed_pending": 0,
        "removed_active_goals": 0,
    }
    if state is None or not is_foreground_objective_origin(origin):
        return receipt
    cognition = getattr(state, "cognition", None)
    if cognition is None:
        return receipt

    objective_text = normalize_goal_text(objective)
    if objective_text:
        receipt["objective_digest"] = hashlib.sha256(
            objective_text.encode("utf-8")
        ).hexdigest()[:16]

    final_objective = normalize_goal_text(getattr(cognition, "current_objective", ""))
    final_origin = getattr(cognition, "current_origin", "")
    final_origin_normalized = normalize_objective_origin(final_origin)
    has_explicit_background_origin = bool(
        final_origin_normalized
        and final_origin_normalized not in {"default", "system", "unknown", "unresolved"}
        and not is_foreground_objective_origin(final_origin_normalized)
    )
    preserve_background = bool(
        final_objective
        and has_explicit_background_origin
    )

    pending = list(getattr(cognition, "pending_initiatives", []) or [])
    kept_pending = [
        item
        for item in pending
        if not is_projection_of_completed_turn(item, objective_text)
        and not is_transient_foreground_projection(item)
    ]
    active = list(getattr(cognition, "active_goals", []) or [])
    kept_active = [
        item
        for item in active
        if not is_projection_of_completed_turn(item, objective_text)
        and not is_transient_foreground_projection(item)
    ]
    cognition.pending_initiatives = kept_pending
    cognition.active_goals = kept_active
    receipt["removed_pending"] = len(pending) - len(kept_pending)
    receipt["removed_active_goals"] = len(active) - len(kept_active)

    modifiers = dict(getattr(cognition, "modifiers", {}) or {})
    if normalize_goal_text(modifiers.get("executive_objective")) == objective_text:
        modifiers.pop("executive_objective", None)
    if normalize_goal_text(modifiers.get("executive_background_commitment")) == objective_text:
        modifiers.pop("executive_background_commitment", None)
    hysteresis = modifiers.get("executive_hysteresis")
    if isinstance(hysteresis, dict) and normalize_goal_text(
        hysteresis.get("committed_objective")
    ) == objective_text:
        modifiers.pop("executive_hysteresis", None)
    binding = modifiers.get("current_objective_binding")
    if is_projection_of_completed_turn(binding, objective_text):
        modifiers.pop("current_objective_binding", None)
        modifiers.pop("current_goal_id", None)
    cognition.modifiers = modifiers

    response_modifiers = dict(getattr(state, "response_modifiers", {}) or {})
    closure = response_modifiers.get("executive_closure")
    if isinstance(closure, dict):
        closure = dict(closure)
        for key in ("selected_objective", "committed_objective", "background_commitment"):
            if normalize_goal_text(closure.get(key)) == objective_text:
                closure[key] = ""
        if not closure.get("committed_objective"):
            closure["hysteresis_active"] = False
            closure["commitment_age_s"] = 0.0
        response_modifiers["executive_closure"] = closure

    if preserve_background:
        receipt["preserved_background"] = True
    else:
        cognition.current_objective = None
        cognition.current_origin = None
        receipt["cleared_current"] = True
    receipt["completed"] = True
    response_modifiers["foreground_turn_completion"] = dict(receipt)
    state.response_modifiers = response_modifiers
    return receipt


__all__ = [
    "ForegroundObjectiveDisposition",
    "classify_foreground_objective",
    "finalize_foreground_turn_state",
    "has_explicit_durable_binding",
    "is_actionable_foreground_objective",
    "is_ephemeral_conversation_turn",
    "is_executive_closure_projection",
    "is_foreground_objective_origin",
    "is_projection_of_completed_turn",
    "is_transient_foreground_projection",
    "normalize_objective_origin",
]
