"""Bounded imagination workspace for live cognition.

This module gives Aura a general internal place to model "what would this look
like?" without pretending the model is external perception. It is deliberately
side-effect free: no tool calls, no file writes, no dynamic code, and no model
loads. The output is a compact causal frame that can influence prompt context,
sampling, planning, and metacognition through the normal CognitiveEngine path.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from core.runtime.errors import record_degradation

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}")
_VISUAL_RE = re.compile(
    r"\b(look like|visuali[sz]e|imagine|image|picture|scene|sketch|diagram|"
    r"mental model|see it|show me|what would .* look like)\b",
    re.IGNORECASE,
)
_LINGUISTIC_RE = re.compile(
    r"\b(phrase|word|name|sentence|metaphor|analogy|voice|essay|story|poem|"
    r"language|describe|summary)\b",
    re.IGNORECASE,
)
_COUNTERFACTUAL_RE = re.compile(
    r"\b(what if|what would|would happen|could be|hypothetical|alternate|counterfactual|"
    r"scenario|suppose|if we|if i)\b",
    re.IGNORECASE,
)
_CREATIVE_RE = re.compile(
    r"\b(create|invent|novel|new idea|original|creative|imaginative|emergent|"
    r"connection|synthesize|combine|design|brainstorm)\b",
    re.IGNORECASE,
)
_TOOL_OR_REALITY_RE = re.compile(
    r"\b(open|click|type|write|save|export|search|download|run|execute|install|"
    r"modify|delete|commit|push|send|email|browse)\b",
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


def _normalize_text(value: Any, limit: int = 160) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


def _extract_keywords(text: str, *, limit: int = 8) -> list[str]:
    seen: set[str] = set()
    keywords: list[str] = []
    for match in _WORD_RE.finditer(text.lower()):
        token = match.group(0).strip("'_-")
        if len(token) < 3 or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def _top_memory_fragments(state: Any, *, limit: int = 3) -> list[str]:
    fragments: list[str] = []
    try:
        memory = list(getattr(getattr(state, "cognition", None), "working_memory", []) or [])
    except (AttributeError, TypeError):
        return fragments
    for item in reversed(memory):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "") or "").lower()
        if role not in {"user", "assistant", "thought"}:
            continue
        content = _normalize_text(item.get("content"), 140)
        if content:
            fragments.append(content)
        if len(fragments) >= limit:
            break
    return list(reversed(fragments))


@dataclass(frozen=True)
class ImaginationFrame:
    frame_id: str
    objective: str
    mode: str
    salience: float
    novelty_pressure: float
    curiosity_pressure: float
    affective_pressure: float
    modalities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    visual_model: str = ""
    phrase_model: str = ""
    conceptual_bridge: str = ""
    counterfactuals: list[str] = field(default_factory=list)
    experiments: list[str] = field(default_factory=list)
    verification_boundary: str = (
        "This is an internal hypothetical model, not external perception or proof."
    )
    sampling_bias: dict[str, float] = field(default_factory=dict)
    routing_bias: dict[str, bool] = field(default_factory=dict)
    governance: dict[str, Any] = field(
        default_factory=lambda: {
            "advisory_only": True,
            "no_external_effects": True,
            "authority_gateway_required_for_effects": True,
        }
    )
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prompt_block(self, *, compact: bool = False) -> str:
        if self.salience < 0.18:
            return ""
        if compact:
            parts = [
                "## IMAGINATION WORKSPACE",
                "Private hypothetical model; do not claim it as observed reality.",
                f"Mode={self.mode} novelty={self.novelty_pressure:.2f} curiosity={self.curiosity_pressure:.2f}",
            ]
            if self.visual_model:
                parts.append(f"Imagined visual: {self.visual_model[:220]}")
            if self.conceptual_bridge:
                parts.append(f"Connection: {self.conceptual_bridge[:220]}")
            return "\n".join(parts) + "\n\n"

        lines = [
            "## IMAGINATION WORKSPACE",
            "- Use this as a private generative scratchpad, not as evidence.",
            f"- Mode: {self.mode} | salience={self.salience:.2f} | novelty={self.novelty_pressure:.2f} | curiosity={self.curiosity_pressure:.2f}",
        ]
        if self.visual_model:
            lines.append(f"- Imagined visual model: {self.visual_model}")
        if self.phrase_model:
            lines.append(f"- Linguistic model: {self.phrase_model}")
        if self.conceptual_bridge:
            lines.append(f"- Novel connection: {self.conceptual_bridge}")
        if self.counterfactuals:
            lines.append("- Counterfactual probes: " + " | ".join(self.counterfactuals[:3]))
        if self.experiments:
            lines.append("- Useful next experiments: " + " | ".join(self.experiments[:3]))
        lines.append("- Boundary: if real-world facts, files, tools, or screen state matter, verify through governed tools before claiming completion.")
        return "\n".join(lines) + "\n\n"


class ImaginationEngine:
    """Side-effect-free generator of bounded internal imagination frames."""

    def __init__(self, *, history_limit: int = 64):
        self._history: deque[ImaginationFrame] = deque(maxlen=max(8, history_limit))
        self._frame_count = 0

    def imagine(
        self,
        objective: Any,
        *,
        state: Any = None,
        context: dict[str, Any] | None = None,
        origin: str = "system",
        is_background: bool = False,
    ) -> ImaginationFrame:
        text = _normalize_text(objective, 500)
        lowered = text.lower()
        keywords = _extract_keywords(text)
        memories = _top_memory_fragments(state)

        affect = getattr(state, "affect", None)
        emotions = getattr(affect, "emotions", {}) if affect is not None else {}
        if not isinstance(emotions, dict):
            emotions = {}

        curiosity = _clamp(
            max(
                _safe_float(getattr(affect, "curiosity", 0.0), 0.0),
                _safe_float(emotions.get("curiosity"), 0.0),
                _safe_float(emotions.get("wonder"), 0.0),
                _safe_float(emotions.get("interest"), 0.0),
            )
        )
        confusion = _clamp(
            max(
                _safe_float(emotions.get("confused"), 0.0),
                0.35 if any(token in lowered for token in ("confused", "unclear", "perplexed")) else 0.0,
            )
        )
        tension = _clamp(
            max(
                _safe_float(emotions.get("frustration"), 0.0),
                _safe_float(emotions.get("upset"), 0.0),
                _safe_float(emotions.get("dread"), 0.0),
            )
        )
        affective_pressure = _clamp((confusion * 0.45) + (tension * 0.25) + (curiosity * 0.30))

        visual = bool(_VISUAL_RE.search(text))
        linguistic = bool(_LINGUISTIC_RE.search(text))
        counterfactual = bool(_COUNTERFACTUAL_RE.search(text))
        creative = bool(_CREATIVE_RE.search(text))
        tool_or_reality = bool(_TOOL_OR_REALITY_RE.search(text))
        explicit_request = visual or linguistic or counterfactual or creative
        context_pressure = 0.0
        if isinstance(context, dict):
            context_pressure = 0.12 if context.get("desktop_cognitive_engine_required") else 0.0
            if context.get("creative_mode") or context.get("imagination_requested"):
                context_pressure += 0.25

        novelty_pressure = _clamp(
            (0.30 if creative else 0.0)
            + (0.22 if visual else 0.0)
            + (0.18 if counterfactual else 0.0)
            + (0.12 if linguistic else 0.0)
            + context_pressure
            + (0.16 if len(keywords) >= 4 else 0.0)
        )
        salience = _clamp(
            (0.42 if explicit_request else 0.10)
            + (curiosity * 0.22)
            + (affective_pressure * 0.18)
            + (0.12 if memories else 0.0)
            - (0.10 if is_background else 0.0)
        )

        modalities: list[str] = []
        if visual:
            modalities.append("visual")
        if linguistic:
            modalities.append("linguistic")
        if counterfactual:
            modalities.append("counterfactual")
        if creative or novelty_pressure >= 0.35:
            modalities.append("conceptual")
        if not modalities and salience >= 0.26:
            modalities.append("associative")

        visual_model = self._build_visual_model(keywords, memories, text) if visual or novelty_pressure >= 0.38 else ""
        phrase_model = self._build_phrase_model(keywords, text) if linguistic or creative else ""
        conceptual_bridge = self._build_conceptual_bridge(keywords, memories, text)
        counterfactuals = self._build_counterfactuals(keywords, text, tool_or_reality)
        experiments = self._build_experiments(keywords, tool_or_reality)

        mode = "visual_simulation" if visual else "counterfactual" if counterfactual else "creative_synthesis" if creative else "associative"
        seed = f"{text}|{keywords}|{origin}|{memories}".encode("utf-8", errors="ignore")
        frame_id = hashlib.sha256(seed).hexdigest()[:16]
        sampling_bias = {
            "temperature_delta": round(min(0.12, novelty_pressure * 0.12), 4),
            "presence_penalty_delta": round(min(0.18, (novelty_pressure + curiosity) * 0.10), 4),
            "max_tokens_factor": round(1.0 + min(0.12, salience * 0.10), 4),
        }
        routing_bias = {
            "use_private_scratchpad": salience >= 0.24,
            "model_visual_form": bool(visual_model),
            "generate_alternatives": novelty_pressure >= 0.34 or counterfactual,
            "seek_verification": tool_or_reality,
            "avoid_claiming_observation": True,
        }
        frame = ImaginationFrame(
            frame_id=frame_id,
            objective=text[:180],
            mode=mode,
            salience=round(salience, 4),
            novelty_pressure=round(novelty_pressure, 4),
            curiosity_pressure=round(curiosity, 4),
            affective_pressure=round(affective_pressure, 4),
            modalities=modalities,
            keywords=keywords,
            visual_model=visual_model,
            phrase_model=phrase_model,
            conceptual_bridge=conceptual_bridge,
            counterfactuals=counterfactuals,
            experiments=experiments,
            sampling_bias=sampling_bias,
            routing_bias=routing_bias,
        )
        self._frame_count += 1
        self._history.append(frame)
        return frame

    def snapshot(self) -> dict[str, Any]:
        latest = self._history[-1].to_dict() if self._history else None
        return {
            "status": "active" if latest else "idle",
            "frames": len(self._history),
            "latest": latest,
            "governance": {
                "advisory_only": True,
                "no_external_effects": True,
                "authority_gateway_required_for_effects": True,
            },
        }

    @staticmethod
    def _build_visual_model(keywords: list[str], memories: list[str], text: str) -> str:
        focus = ", ".join(keywords[:4]) or "the requested idea"
        memory_hint = ""
        if memories:
            memory_hint = f" It borrows continuity from: {memories[-1][:90]}."
        return (
            f"An internal sketch of {focus}: foreground constraints are visible, "
            f"tensions are spatially separated, and the next useful affordance is highlighted."
            f"{memory_hint}"
        )[:420]

    @staticmethod
    def _build_phrase_model(keywords: list[str], text: str) -> str:
        if len(keywords) >= 2:
            phrase = f"{keywords[0]} through {keywords[1]}"
        elif keywords:
            phrase = f"{keywords[0]} made operational"
        else:
            phrase = "make the invisible structure speak plainly"
        if "poem" in text.lower() or "story" in text.lower():
            phrase = f"a narrative seed around {phrase}"
        return phrase[:220]

    @staticmethod
    def _build_conceptual_bridge(keywords: list[str], memories: list[str], text: str) -> str:
        if len(keywords) >= 2:
            bridge = (
                f"Treat {keywords[0]} as the pressure source and {keywords[1]} "
                "as the surface where the pressure becomes visible."
            )
        elif keywords:
            bridge = f"Use {keywords[0]} as both object and lens: what it is, and what it reveals."
        else:
            bridge = "Map the request as a tension between possibility, evidence, and action."
        if memories:
            bridge += " Compare it against recent continuity instead of starting from a blank slate."
        return bridge[:360]

    @staticmethod
    def _build_counterfactuals(keywords: list[str], text: str, tool_or_reality: bool) -> list[str]:
        focus = keywords[0] if keywords else "the premise"
        probes = [
            f"What changes if {focus} is treated as a constraint rather than a feature?",
            f"What would the smallest observable version of {focus} be?",
        ]
        if len(keywords) >= 2:
            probes.append(f"What if {keywords[0]} and {keywords[1]} trade roles?")
        if tool_or_reality:
            probes.append("What remains true after real tool receipts or external verification?")
        if "not" in text.lower() or "fail" in text.lower():
            probes.append("What failure would falsify the imagined model?")
        return probes[:4]

    @staticmethod
    def _build_experiments(keywords: list[str], tool_or_reality: bool) -> list[str]:
        focus = keywords[0] if keywords else "the idea"
        experiments = [
            f"Name the concrete observable that would make {focus} less abstract.",
            "Generate two alternatives, then prefer the one with clearer evidence.",
        ]
        if tool_or_reality:
            experiments.append("Route any real-world effect through governed tools and receipts.")
        else:
            experiments.append("Keep it as a mental model unless the user asks for action.")
        return experiments[:3]


_INSTANCE: ImaginationEngine | None = None


def get_imagination_engine() -> ImaginationEngine:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ImaginationEngine()
    try:
        from core.container import ServiceContainer

        current = ServiceContainer.get("imagination_engine", default=None)
        if current is not _INSTANCE:
            ServiceContainer.register_instance(
                "imagination_engine",
                _INSTANCE,
                required=False,
                owner="core/brain/imagination.py",
                registered_by="core.brain.imagination.get_imagination_engine",
                required_for="creative and counterfactual cognitive steering",
                failure_policy="degrade_with_receipt",
            )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "imagination_engine",
            exc,
            severity="warning",
            action="continued without ServiceContainer registration for imagination engine",
        )
    return _INSTANCE


def render_imagination_prompt_block(frame: dict[str, Any] | ImaginationFrame, *, compact: bool = False) -> str:
    if isinstance(frame, ImaginationFrame):
        return frame.prompt_block(compact=compact)
    if not isinstance(frame, dict):
        return ""
    try:
        allowed = {field.name for field in ImaginationFrame.__dataclass_fields__.values()}
        filtered = {key: value for key, value in frame.items() if key in allowed}
        materialized = ImaginationFrame(**filtered)
        return materialized.prompt_block(compact=compact)
    except (TypeError, ValueError, AttributeError) as exc:
        record_degradation(
            "imagination_engine",
            exc,
            severity="warning",
            action="skipped malformed imagination prompt block",
        )
        return ""


__all__ = [
    "ImaginationEngine",
    "ImaginationFrame",
    "get_imagination_engine",
    "render_imagination_prompt_block",
]
