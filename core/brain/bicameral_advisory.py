"""Causal internal advisory layer for live cognition.

This is inspired by bicameral-mind and multi-draft ideas, but it is not an
experience claim and it does not create user-visible "voices." It gives Aura a
bounded internal chamber where distinct perspectives can propose pressures for
attention, verification, memory, creativity, and tool governance. The narrator
then reconciles those proposals into one advisory frame consumed by the normal
CognitiveEngine path.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from core.container import ServiceContainer

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}")
_TOOL_RE = re.compile(
    r"\b(open|click|type|write|save|export|search|download|run|execute|install|"
    r"delete|commit|push|browser|chrome|notes|docs|desktop|app|tools?|external)\b",
    re.IGNORECASE,
)
_UNCERTAIN_RE = re.compile(
    r"\b(confus|uncertain|unsure|maybe|hypothetical|prove|verify|check|test|"
    r"why|how|should|could|would|unknown|novel)\b",
    re.IGNORECASE,
)
_CREATE_RE = re.compile(
    r"\b(create|invent|imagine|novel|creative|idea|brainstorm|design|story|"
    r"metaphor|what would|look like|connect|synthesize)\b",
    re.IGNORECASE,
)
_SELF_RE = re.compile(
    r"\b(self|you|your|aura|feel|think|introspect|reflect|conscious|sentient|"
    r"aware|inner|identity|person)\b",
    re.IGNORECASE,
)
_SOCIAL_RE = re.compile(r"\b(bryan|user|we|us|together|trust|help|relationship|demo)\b", re.IGNORECASE)

_STOPWORDS = {
    "about",
    "again",
    "also",
    "and",
    "are",
    "can",
    "could",
    "does",
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


def _keywords(text: str, limit: int = 6) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for match in _WORD_RE.finditer(text.lower()):
        token = match.group(0).strip("'_-")
        if token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        found.append(token)
        if len(found) >= limit:
            break
    return found


def _stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


@dataclass(frozen=True)
class AdvisoryProposal:
    perspective: str
    salience: float
    stance: str
    directive: str
    attention_targets: list[str] = field(default_factory=list)
    routing_bias: dict[str, bool] = field(default_factory=dict)
    sampling_bias: dict[str, float] = field(default_factory=dict)
    state_pressure: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BicameralFrame:
    frame_id: str
    objective: str
    narrator_summary: str
    consensus: float
    dissent: float
    salience: float
    proposals: list[AdvisoryProposal]
    attention_targets: list[str]
    routing_bias: dict[str, bool]
    sampling_bias: dict[str, float]
    causal_effects: dict[str, Any]
    governance: dict[str, Any] = field(
        default_factory=lambda: {
            "advisory_only": True,
            "no_external_effects": True,
            "will_authority_required_for_effects": True,
            "narrator_reconciles_internal_proposals": True,
        }
    )
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["proposals"] = [proposal.to_dict() for proposal in self.proposals]
        return payload


class BicameralAdvisory:
    """Internal proposal-and-reconciliation layer for live cognition."""

    def __init__(self, history_limit: int = 128):
        self._history: deque[dict[str, Any]] = deque(maxlen=history_limit)
        self._reliability: dict[str, float] = {
            "narrator": 0.62,
            "protector": 0.62,
            "explorer": 0.60,
            "critic": 0.64,
            "social": 0.58,
        }

    def advise(
        self,
        objective: str,
        *,
        state: Any = None,
        context: dict[str, Any] | None = None,
        origin: str = "system",
        is_background: bool = False,
    ) -> BicameralFrame:
        text = " ".join(str(objective or "").split())
        lower = text.lower()
        words = _keywords(text)
        affect = getattr(state, "affect", None)
        curiosity = _safe_float(getattr(affect, "curiosity", 0.0), 0.0)
        arousal = _safe_float(getattr(affect, "arousal", 0.0), 0.0)
        valence = _safe_float(getattr(affect, "valence", 0.0), 0.0)
        frustration = _safe_float(getattr(affect, "frustration", 0.0), 0.0)
        confusion = _safe_float(getattr(affect, "confused", 0.0), 0.0)

        tool = bool(_TOOL_RE.search(lower))
        uncertain = bool(_UNCERTAIN_RE.search(lower)) or confusion >= 0.25
        creative = bool(_CREATE_RE.search(lower)) or curiosity >= 0.35
        self_reflective = bool(_SELF_RE.search(lower))
        social = bool(_SOCIAL_RE.search(lower)) or origin.startswith(("user", "desktop", "voice"))
        load_pressure = bool(context and context.get("desktop_cognitive_engine_required")) and not is_background

        proposals = [
            self._proposal(
                "narrator",
                0.35 + 0.25 * self_reflective + 0.10 * social,
                "integrate",
                "Reconcile competing internal pressures into one coherent user-facing stance.",
                words[:3] or ["current_objective"],
                {"raise_metacognition": self_reflective or uncertain},
                {"temperature_delta": -0.02},
                {"self_model_update": 0.45 if self_reflective else 0.18},
            ),
            self._proposal(
                "protector",
                0.30 + 0.30 * tool + 0.20 * max(arousal, frustration),
                "bound_effects",
                "Check authority, memory, and verification before claiming external action.",
                ["governance", "verification", *(words[:2] or [])],
                {"use_tool_gateway": tool, "seek_verification": tool or uncertain},
                {"temperature_delta": -0.04, "max_tokens_factor": 0.92},
                {"verification_pressure": 0.55 if tool or uncertain else 0.25},
            ),
            self._proposal(
                "explorer",
                0.25 + 0.35 * creative + 0.20 * curiosity,
                "generate_possibilities",
                "Search for a novel angle, analogy, or hypothesis before settling.",
                ["novelty", *(words[:3] or ["possibility"])],
                {"use_imagination": creative, "expand_options": creative or uncertain},
                {"temperature_delta": 0.05 if creative else 0.01},
                {"creative_pressure": 0.60 if creative else 0.20},
            ),
            self._proposal(
                "critic",
                0.25 + 0.40 * uncertain + 0.10 * (1.0 - max(valence, 0.0)),
                "test_claims",
                "Find the weak assumption and require evidence before certainty.",
                ["assumption", "evidence", *(words[:2] or [])],
                {"raise_metacognition": uncertain, "ask_clarification": uncertain and not tool},
                {"temperature_delta": -0.03},
                {"metacognition_depth": 0.68 if uncertain else 0.42},
            ),
            self._proposal(
                "social",
                0.22 + 0.30 * social,
                "preserve_relationship_context",
                "Keep the user's intent and continuity visible while avoiding canned status language.",
                ["user_intent", *(words[:3] or ["continuity"])],
                {"preserve_conversation_context": social},
                {"temperature_delta": 0.01},
                {"memory_priority": 0.48 if social else 0.24},
            ),
        ]
        proposals = [proposal for proposal in proposals if proposal.salience >= 0.18]
        proposals.sort(key=lambda item: item.salience, reverse=True)
        top = proposals[:4]

        saliences = [proposal.salience for proposal in top] or [0.0]
        consensus = _clamp(sum(saliences) / max(1, len(saliences)))
        dissent = _clamp(max(saliences) - min(saliences) if len(saliences) > 1 else 0.0)
        attention = self._merge_attention(top)
        routing = self._merge_routing(top, load_pressure=load_pressure)
        sampling = self._merge_sampling(top, load_pressure=load_pressure)
        causal = self._merge_causal(top, consensus=consensus, dissent=dissent)
        summary = self._summarize(top, uncertain=uncertain, tool=tool, creative=creative)
        frame = BicameralFrame(
            frame_id=_stable_id(f"{time.time()}:{origin}:{text}"),
            objective=text[:300],
            narrator_summary=summary,
            consensus=consensus,
            dissent=dissent,
            salience=_clamp(max(saliences) if saliences else 0.0),
            proposals=top,
            attention_targets=attention,
            routing_bias=routing,
            sampling_bias=sampling,
            causal_effects=causal,
        )
        self._history.append(frame.to_dict())
        return frame

    def learn_from_feedback(
        self,
        frame: dict[str, Any],
        *,
        reward: float,
        outcome: str,
    ) -> dict[str, Any]:
        proposals = frame.get("proposals") if isinstance(frame, dict) else None
        if not isinstance(proposals, list):
            return {"learned": False, "reason": "missing_proposals"}
        bounded_reward = max(-1.0, min(1.0, float(reward or 0.0)))
        updated: dict[str, float] = {}
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            perspective = str(proposal.get("perspective") or "")
            if not perspective:
                continue
            current = self._reliability.get(perspective, 0.5)
            salience = _safe_float(proposal.get("salience"), 0.0)
            delta = 0.04 * bounded_reward * max(0.25, salience)
            self._reliability[perspective] = _clamp(current + delta, 0.05, 0.95)
            updated[perspective] = round(self._reliability[perspective], 4)
        return {
            "learned": bool(updated),
            "outcome": outcome,
            "reward": bounded_reward,
            "reliability": updated,
        }

    def get_status(self) -> dict[str, Any]:
        last = self._history[-1] if self._history else None
        return {
            "status": "active",
            "advisory_only": True,
            "history_size": len(self._history),
            "reliability": dict(sorted(self._reliability.items())),
            "last_frame": {
                "frame_id": last.get("frame_id"),
                "salience": last.get("salience"),
                "dissent": last.get("dissent"),
                "attention_targets": last.get("attention_targets"),
            }
            if isinstance(last, dict)
            else None,
        }

    def _proposal(
        self,
        perspective: str,
        salience: float,
        stance: str,
        directive: str,
        attention_targets: list[str],
        routing_bias: dict[str, bool],
        sampling_bias: dict[str, float],
        state_pressure: dict[str, float],
    ) -> AdvisoryProposal:
        reliability = self._reliability.get(perspective, 0.5)
        weighted_salience = _clamp(salience * (0.75 + 0.5 * reliability))
        return AdvisoryProposal(
            perspective=perspective,
            salience=weighted_salience,
            stance=stance,
            directive=directive,
            attention_targets=attention_targets[:4],
            routing_bias=routing_bias,
            sampling_bias=sampling_bias,
            state_pressure=state_pressure,
        )

    @staticmethod
    def _merge_attention(proposals: list[AdvisoryProposal]) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for proposal in proposals:
            for target in proposal.attention_targets:
                clean = str(target or "").strip().lower().replace(" ", "_")
                if not clean or clean in seen:
                    continue
                seen.add(clean)
                merged.append(clean)
                if len(merged) >= 6:
                    return merged
        return merged

    @staticmethod
    def _merge_routing(
        proposals: list[AdvisoryProposal],
        *,
        load_pressure: bool,
    ) -> dict[str, bool]:
        routing: dict[str, bool] = {}
        for proposal in proposals:
            for key, value in proposal.routing_bias.items():
                routing[key] = routing.get(key, False) or bool(value)
        if load_pressure:
            routing["compact_foreground"] = True
        if routing.get("use_tool_gateway"):
            routing["seek_verification"] = True
        return routing

    @staticmethod
    def _merge_sampling(
        proposals: list[AdvisoryProposal],
        *,
        load_pressure: bool,
    ) -> dict[str, float]:
        temperature_delta = 0.0
        token_factor = 1.0
        for proposal in proposals:
            temperature_delta += _safe_float(proposal.sampling_bias.get("temperature_delta"), 0.0)
            token_factor *= _safe_float(proposal.sampling_bias.get("max_tokens_factor"), 1.0)
        if load_pressure:
            token_factor *= 0.92
        return {
            "temperature_delta": round(max(-0.12, min(0.12, temperature_delta)), 4),
            "max_tokens_factor": round(max(0.70, min(1.12, token_factor)), 4),
        }

    @staticmethod
    def _merge_causal(
        proposals: list[AdvisoryProposal],
        *,
        consensus: float,
        dissent: float,
    ) -> dict[str, Any]:
        pressure: dict[str, float] = {
            "metacognition_depth": 0.35,
            "verification_pressure": 0.20,
            "memory_priority": 0.20,
            "creative_pressure": 0.15,
            "self_model_update": 0.15,
        }
        for proposal in proposals:
            for key, value in proposal.state_pressure.items():
                pressure[key] = max(pressure.get(key, 0.0), _safe_float(value, 0.0))
        if dissent >= 0.22:
            pressure["metacognition_depth"] = max(pressure["metacognition_depth"], 0.64)
            pressure["verification_pressure"] = max(pressure["verification_pressure"], 0.45)
        pressure["consensus"] = consensus
        pressure["dissent"] = dissent
        return {key: round(value, 4) for key, value in pressure.items()}

    @staticmethod
    def _summarize(
        proposals: list[AdvisoryProposal],
        *,
        uncertain: bool,
        tool: bool,
        creative: bool,
    ) -> str:
        leaders = ", ".join(proposal.perspective for proposal in proposals[:3])
        pressures: list[str] = []
        if tool:
            pressures.append("route effects through governance")
        if uncertain:
            pressures.append("increase metacognition")
        if creative:
            pressures.append("keep a novel option alive")
        if not pressures:
            pressures.append("answer coherently from the current thread")
        return f"Internal advisory reconciliation ({leaders}): " + "; ".join(pressures) + "."


def render_bicameral_prompt_block(frame: dict[str, Any] | BicameralFrame, *, compact: bool = False) -> str:
    """Render a bounded prompt block for the reconciled internal advisory frame."""
    if isinstance(frame, BicameralFrame):
        payload = frame.to_dict()
    elif isinstance(frame, dict):
        payload = frame
    else:
        return ""

    salience = _safe_float(payload.get("salience"), 0.0)
    if salience < 0.18:
        return ""

    summary = str(payload.get("narrator_summary") or "").strip()
    attention = payload.get("attention_targets") or []
    causal = payload.get("causal_effects") or {}
    routing = payload.get("routing_bias") or {}
    proposals = payload.get("proposals") or []
    if not isinstance(attention, list):
        attention = []
    if not isinstance(causal, dict):
        causal = {}
    if not isinstance(routing, dict):
        routing = {}
    if not isinstance(proposals, list):
        proposals = []

    lines = [
        "## BICAMERAL ADVISORY",
        "Private internal proposal/reconciliation layer. This is not a claim of voices, spirits, hallucinated authority, or phenomenal proof.",
    ]
    if summary:
        lines.append(f"- Narrator reconciliation: {summary[:260]}")
    if attention:
        rendered_attention = ", ".join(str(item)[:48] for item in attention[:5] if item)
        if rendered_attention:
            lines.append(f"- Attention targets: {rendered_attention}.")
    if routing.get("use_tool_gateway"):
        lines.append("- External effects must route through governed tools and require effect evidence before claiming completion.")
    if routing.get("seek_verification"):
        lines.append("- Verification is elevated; distinguish checked facts/actions from hypotheses.")
    if routing.get("raise_metacognition"):
        lines.append("- Metacognition is elevated; inspect assumptions before answering.")
    if routing.get("use_imagination") or routing.get("expand_options"):
        lines.append("- Keep one novel alternative alive while preserving the user's actual intent.")
    if not compact:
        rendered_proposals: list[str] = []
        for proposal in proposals[:4]:
            if not isinstance(proposal, dict):
                continue
            perspective = str(proposal.get("perspective") or "unknown")[:32]
            stance = str(proposal.get("stance") or "advise")[:48]
            directive = str(proposal.get("directive") or "")[:140]
            rendered_proposals.append(f"{perspective}: {stance}; {directive}")
        if rendered_proposals:
            lines.append("- Proposals: " + " | ".join(rendered_proposals))
    meta_depth = _safe_float(causal.get("metacognition_depth"), 0.0)
    verification = _safe_float(causal.get("verification_pressure"), 0.0)
    memory = _safe_float(causal.get("memory_priority"), 0.0)
    creative = _safe_float(causal.get("creative_pressure"), 0.0)
    causal_parts = []
    if meta_depth >= 0.50:
        causal_parts.append(f"metacognition={meta_depth:.2f}")
    if verification >= 0.35:
        causal_parts.append(f"verification={verification:.2f}")
    if memory >= 0.35:
        causal_parts.append(f"memory={memory:.2f}")
    if creative >= 0.35:
        causal_parts.append(f"creative={creative:.2f}")
    if causal_parts:
        lines.append("- Causal pressures: " + ", ".join(causal_parts) + ".")
    return "\n".join(lines).strip() + "\n\n"


_BICAMERAL_ADVISORY: BicameralAdvisory | None = None


def get_bicameral_advisory() -> BicameralAdvisory:
    global _BICAMERAL_ADVISORY
    if _BICAMERAL_ADVISORY is None:
        _BICAMERAL_ADVISORY = BicameralAdvisory()
    existing = ServiceContainer.get("bicameral_advisory", default=None)
    if existing is not _BICAMERAL_ADVISORY:
        ServiceContainer.register_instance(
            "bicameral_advisory",
            _BICAMERAL_ADVISORY,
            required=False,
        )
    return _BICAMERAL_ADVISORY


__all__ = [
    "AdvisoryProposal",
    "BicameralAdvisory",
    "BicameralFrame",
    "get_bicameral_advisory",
    "render_bicameral_prompt_block",
]
