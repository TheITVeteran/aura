"""Runtime policy for proof/evaluation turns.

Proof runs must use the same live runtime path as normal Aura launches while
remaining auditable and isolated. This module centralizes the small set of
proof-specific knobs so individual phases do not quietly diverge.
"""

from __future__ import annotations

import os
import re
from typing import Any

_PROOF_ACTIVE_ENV = ("AURA_PROOF_RUN", "AURA_AGI_MAX_TASKS", "AURA_TESTING")
_PROOF_REPAIR_PREFIX = "Your previous proof/evaluation answer failed validation."
_PROOF_REPAIR_ORIGINAL_TASK_RE = re.compile(
    r"(?:^|\n)Original task:\s*\n(?P<task>.*?)(?:\n\s*\nValidation status:|\Z)",
    re.DOTALL,
)

TRANSIENT_RESPONSE_MODIFIER_KEYS = frozenset(
    {
        "adaptive_effector",
        "adaptive_immune_coverage",
        "adaptive_immune_verdict",
        "adaptive_immune_verification",
        "adaptive_immunity",
        "affective_reasoning_pressure",
        "agency_comparator",
        "anomaly_score",
        "anomaly_threat_level",
        "auto_browse_urls",
        "autonomous_resilience",
        "autonomous_resilience_immune",
        "conv_dynamics_state",
        "conversation_intelligence",
        "conversational_dynamics",
        "credit_assignment",
        "deep_handoff",
        "dialogue_validation",
        "execution_report",
        "executive_closure",
        "executive_dominant_need",
        "executive_hysteresis",
        "executive_need_pressure",
        "executive_objective",
        "grounded_actions",
        "higher_order_thought",
        "humor_guidance",
        "intent_type",
        "interaction_signals",
        "last_skill_ok",
        "last_skill_result_payload",
        "last_skill_run",
        "last_task_id",
        "last_task_outcome",
        "last_task_result_payload",
        "matched_skills",
        "metacognitive_strategy",
        "model_tier",
        "multiple_drafts",
        "narrative_context",
        "narrative_gravity",
        "natural_followup",
        "pending_followup",
        "pre_linguistic_decision",
        "precomputed_grounded_reply",
        "prediction_error",
        "proof_model_tier",
        "proof_repair_turn",
        "proof_turn_objective",
        "queued_messages",
        "relational_intelligence",
        "response_contract",
        "semantic_intent",
        "strict_proof_answer_request",
        "structured_proof_solver",
        "system_failure_state",
        "thermal_guard",
        "user_sentiment",
    }
)

_PRIMARY_ALIASES = frozenset({"", "primary", "cortex", "32b", "live", "production"})
_TERTIARY_ALIASES = frozenset({"tertiary", "brainstem", "7b", "fast", "diagnostic"})


def proof_run_active(origin: Any = None) -> bool:
    """Return True when the current turn is part of a proof/eval run."""

    normalized = str(origin or "").strip().lower()
    if normalized in {"test", "proof", "eval", "evaluation", "benchmark"}:
        return True
    return any(os.environ.get(name) for name in _PROOF_ACTIVE_ENV)


def is_strict_proof_answer_prompt(prompt: Any, *, origin: Any = None) -> bool:
    """Detect sealed proof tasks that require a strict ``<answer>`` envelope."""

    return "<answer>" in str(prompt or "").lower() and proof_run_active(origin=origin)


def is_proof_evaluation_purpose(purpose: Any) -> bool:
    """Return True for non-atomic proof/eval generation lanes."""

    normalized = str(purpose or "").strip().lower()
    return normalized in {"proof_evaluation", "proof_evaluation_repair"}


def is_proof_repair_prompt(prompt: Any, *, origin: Any = None) -> bool:
    """Detect internal proof repair prompts that must not become durable goals."""

    text = str(prompt or "").strip()
    return (
        bool(text)
        and proof_run_active(origin=origin)
        and text.startswith(_PROOF_REPAIR_PREFIX)
        and "Original task:" in text
        and "Validation status:" in text
    )


def extract_original_task_from_proof_repair_prompt(prompt: Any) -> str:
    """Extract the user-facing task from an internal proof repair prompt."""

    match = _PROOF_REPAIR_ORIGINAL_TASK_RE.search(str(prompt or ""))
    if not match:
        return ""
    return " ".join(match.group("task").strip().split())


def proof_persistent_objective(prompt: Any, *, origin: Any = None) -> str:
    """Return the durable objective represented by a proof/evaluation prompt.

    Repair prompts are runtime scaffolding. They should steer one generation
    attempt, but they must not be committed as Aura's active goal, memory focus,
    or future continuity anchor.
    """

    text = str(prompt or "")
    if is_proof_repair_prompt(text, origin=origin):
        original = extract_original_task_from_proof_repair_prompt(text)
        if original:
            return original
    return text


def proof_model_tier(default: str = "primary") -> str:
    """Resolve the LLM lane for proof tasks.

    The default is the production 32B Cortex lane. A fast 7B lane is still
    available for diagnostic isolation, but it must be requested explicitly via
    ``AURA_PROOF_MODEL_TIER=tertiary`` or the DNU runner's ``--model-tier`` flag.
    """

    raw = str(os.environ.get("AURA_PROOF_MODEL_TIER", default) or default).strip().lower()
    if raw in _TERTIARY_ALIASES:
        return "tertiary"
    if raw in _PRIMARY_ALIASES:
        return "primary"
    return "primary"


def clear_transient_response_modifiers(modifiers: Any, *, strict: bool = False) -> None:
    """Remove per-turn prompt/runtime modifiers before a new turn starts.

    AuraState derives by copying ``response_modifiers`` so downstream phases can
    see the previous transition. That is correct for durable state summaries but
    unsafe for turn-local prompt directives: an old open conversational thread,
    tool result, or recovery flag can become an instruction on the next task.
    Proof/eval turns use ``strict=True`` to start from an empty transient surface;
    normal live turns clear only known per-turn keys while preserving any future
    durable modifiers that are intentionally added.
    """

    if not isinstance(modifiers, dict):
        return
    if strict:
        modifiers.clear()
        return
    for key in TRANSIENT_RESPONSE_MODIFIER_KEYS:
        modifiers.pop(key, None)
