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
    r"modify|delete|commit|push|send|email|browse|tool|tools|workflow|desktop|"
    r"browser|app|application|external|real-world|real world|visible action)\b",
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


def _stable_softmax(scores: dict[str, float], *, temperature: float = 1.0) -> dict[str, float]:
    if not scores:
        return {}
    temp = max(0.05, min(5.0, float(temperature or 1.0)))
    finite_scores = {
        key: (_safe_float(value, 0.0) / temp)
        for key, value in scores.items()
    }
    highest = max(finite_scores.values())
    exps = {
        key: math.exp(max(-60.0, min(60.0, value - highest)))
        for key, value in finite_scores.items()
    }
    total = sum(exps.values()) or 1.0
    return {key: value / total for key, value in exps.items()}


def _entropy01(probabilities: dict[str, float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    entropy = 0.0
    for probability in probabilities.values():
        p = max(1e-12, min(1.0, float(probability or 0.0)))
        entropy -= p * math.log(p)
    return _clamp(entropy / math.log(len(probabilities)))


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
    memory_pressure: float
    verification_pressure: float
    working_memory: dict[str, Any] = field(default_factory=dict)
    attractor_state: dict[str, Any] = field(default_factory=dict)
    eligibility_trace: dict[str, float] = field(default_factory=dict)
    modalities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    attention_targets: list[str] = field(default_factory=list)
    visual_model: str = ""
    phrase_model: str = ""
    conceptual_bridge: str = ""
    mental_canvas: dict[str, Any] = field(default_factory=dict)
    associative_links: list[dict[str, str]] = field(default_factory=list)
    novel_thoughts: list[str] = field(default_factory=list)
    simulation_steps: list[str] = field(default_factory=list)
    counterfactuals: list[str] = field(default_factory=list)
    experiments: list[str] = field(default_factory=list)
    action_affordances: list[str] = field(default_factory=list)
    ablation_predictions: dict[str, str] = field(default_factory=dict)
    causal_effects: dict[str, Any] = field(default_factory=dict)
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
                f"Mode={self.mode} novelty={self.novelty_pressure:.2f} curiosity={self.curiosity_pressure:.2f} "
                f"memory={self.memory_pressure:.2f} verify={self.verification_pressure:.2f}",
            ]
            if self.visual_model:
                parts.append(f"Imagined visual: {self.visual_model[:220]}")
            if self.conceptual_bridge:
                parts.append(f"Connection: {self.conceptual_bridge[:220]}")
            if self.attention_targets:
                parts.append("Attention targets: " + ", ".join(self.attention_targets[:4]))
            return "\n".join(parts) + "\n\n"

        lines = [
            "## IMAGINATION WORKSPACE",
            "- Use this as a private generative scratchpad, not as evidence.",
            f"- Mode: {self.mode} | salience={self.salience:.2f} | novelty={self.novelty_pressure:.2f} | curiosity={self.curiosity_pressure:.2f} | memory={self.memory_pressure:.2f} | verify={self.verification_pressure:.2f}",
        ]
        if self.attention_targets:
            lines.append("- Attention targets: " + ", ".join(self.attention_targets[:5]))
        if self.visual_model:
            lines.append(f"- Imagined visual model: {self.visual_model}")
        canvas = self.mental_canvas if isinstance(self.mental_canvas, dict) else {}
        image_prompt = _normalize_text(canvas.get("image_prompt"), 260) if canvas else ""
        if image_prompt:
            lines.append(f"- Mental canvas: {image_prompt}")
        if self.phrase_model:
            lines.append(f"- Linguistic model: {self.phrase_model}")
        if self.conceptual_bridge:
            lines.append(f"- Novel connection: {self.conceptual_bridge}")
        if self.novel_thoughts:
            lines.append("- Novel thought candidates: " + " | ".join(self.novel_thoughts[:3]))
        if self.associative_links:
            rendered_links = [
                f"{link.get('source')} -> {link.get('relation')} -> {link.get('target')}"
                for link in self.associative_links[:3]
                if isinstance(link, dict)
            ]
            if rendered_links:
                lines.append("- Association map: " + " | ".join(rendered_links))
        if self.counterfactuals:
            lines.append("- Counterfactual probes: " + " | ".join(self.counterfactuals[:3]))
        if self.simulation_steps:
            lines.append("- Internal simulation steps: " + " | ".join(self.simulation_steps[:3]))
        if self.attractor_state:
            selected = str(self.attractor_state.get("selected") or "").strip()
            entropy = _safe_float(self.attractor_state.get("entropy"), 0.0)
            margin = _safe_float(self.attractor_state.get("stability_margin"), 0.0)
            if selected:
                lines.append(
                    f"- Attractor: {selected} | entropy={entropy:.2f} | stability_margin={margin:.2f}"
                )
        if self.working_memory:
            admission = str(self.working_memory.get("admission") or "admit")
            queue_load = _safe_float(self.working_memory.get("queue_load"), 0.0)
            overload = _safe_float(self.working_memory.get("overload_pressure"), 0.0)
            if admission != "admit" or overload >= 0.20:
                lines.append(
                    f"- Working-memory gate: {admission} | queue_load={queue_load:.2f} | overload={overload:.2f}"
                )
        if self.experiments:
            lines.append("- Useful next experiments: " + " | ".join(self.experiments[:3]))
        if self.action_affordances:
            lines.append("- Action affordances: " + " | ".join(self.action_affordances[:3]))
        if self.causal_effects:
            effects = []
            for key in (
                "attention_focus",
                "memory_priority",
                "verification_pressure",
                "metacognition_depth",
                "tool_governance",
            ):
                if key in self.causal_effects:
                    effects.append(f"{key}={self.causal_effects.get(key)}")
            if effects:
                lines.append("- Causal effects: " + " | ".join(effects))
        lines.append("- Boundary: if real-world facts, files, tools, or screen state matter, verify through governed tools before claiming completion.")
        return "\n".join(lines) + "\n\n"


class ImaginationEngine:
    """Side-effect-free generator of bounded internal imagination frames."""

    def __init__(self, *, history_limit: int = 64):
        self._history: deque[ImaginationFrame] = deque(maxlen=max(8, history_limit))
        self._frame_index: dict[str, ImaginationFrame] = {}
        self._outcomes: deque[dict[str, Any]] = deque(maxlen=max(8, history_limit))
        self._frame_count = 0
        self._queue_load = 0.0
        self._arrival_rate_ema = 0.0
        self._service_rate_ema = 1.0
        self._last_observed_at = time.monotonic()
        self._attractor_bias: dict[str, float] = {}
        self._eligibility_trace: dict[str, float] = {}

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
        memory_pressure = _clamp(
            (salience * 0.32)
            + (novelty_pressure * 0.26)
            + (curiosity * 0.18)
            + (affective_pressure * 0.14)
            + (0.10 if memories else 0.0)
        )
        verification_pressure = _clamp(
            (0.46 if tool_or_reality else 0.0)
            + (0.20 if counterfactual else 0.0)
            + (0.16 if confusion >= 0.25 else 0.0)
            + (novelty_pressure * 0.12)
            + (0.06 if visual else 0.0)
        )
        working_memory = self._observe_working_memory_gate(
            salience=salience,
            novelty_pressure=novelty_pressure,
            verification_pressure=verification_pressure,
            memory_pressure=memory_pressure,
            context=context,
            is_background=is_background,
        )
        admission = str(working_memory.get("admission") or "admit")
        if admission == "defer_background":
            salience = _clamp(salience * 0.55)
            novelty_pressure = _clamp(novelty_pressure * 0.70)
            memory_pressure = _clamp(memory_pressure * 0.70)
        elif admission in {"compress_foreground", "thin_frame"}:
            salience = _clamp(salience * 0.86)
            novelty_pressure = _clamp(novelty_pressure * 0.82)
            memory_pressure = _clamp(memory_pressure * 0.82)
            verification_pressure = max(verification_pressure, 0.35)

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
        mental_canvas = self._build_mental_canvas(
            keywords,
            memories,
            text,
            visual=visual,
            linguistic=linguistic,
            counterfactual=counterfactual,
            creative=creative,
        )
        associative_links = self._build_associative_links(keywords, memories, text)
        novel_thoughts = self._build_novel_thoughts(
            keywords,
            memories,
            text,
            creative=creative,
            counterfactual=counterfactual,
        )
        simulation_steps = self._build_simulation_steps(
            keywords,
            visual=visual,
            counterfactual=counterfactual,
            tool_or_reality=tool_or_reality,
        )
        counterfactuals = self._build_counterfactuals(keywords, text, tool_or_reality)
        experiments = self._build_experiments(keywords, tool_or_reality)
        attention_targets = self._build_attention_targets(keywords, text)
        action_affordances = self._build_action_affordances(
            keywords,
            tool_or_reality=tool_or_reality,
            visual=visual,
            linguistic=linguistic,
            counterfactual=counterfactual,
        )
        ablation_predictions = self._build_ablation_predictions(
            memory_pressure=memory_pressure,
            verification_pressure=verification_pressure,
            novelty_pressure=novelty_pressure,
            tool_or_reality=tool_or_reality,
        )
        causal_effects = self._build_causal_effects(
            attention_targets=attention_targets,
            memory_pressure=memory_pressure,
            verification_pressure=verification_pressure,
            curiosity=curiosity,
            confusion=confusion,
            tool_or_reality=tool_or_reality,
            working_memory=working_memory,
        )

        mode = "visual_simulation" if visual else "counterfactual" if counterfactual else "creative_synthesis" if creative else "associative"
        attractor_state = self._select_attractor_state(
            mode=mode,
            salience=salience,
            novelty_pressure=novelty_pressure,
            curiosity=curiosity,
            affective_pressure=affective_pressure,
            memory_pressure=memory_pressure,
            verification_pressure=verification_pressure,
            working_memory=working_memory,
            visual=visual,
            linguistic=linguistic,
            counterfactual=counterfactual,
            creative=creative,
            tool_or_reality=tool_or_reality,
        )
        eligibility_trace = self._update_eligibility_trace(
            keywords,
            selected_attractor=str(attractor_state.get("selected") or mode),
            salience=salience,
            novelty_pressure=novelty_pressure,
            memory_pressure=memory_pressure,
        )
        seed = f"{text}|{keywords}|{origin}|{memories}".encode("utf-8", errors="ignore")
        frame_id = hashlib.sha256(seed).hexdigest()[:16]
        token_factor = 1.0 + min(0.12, salience * 0.10)
        if admission in {"compress_foreground", "thin_frame"}:
            token_factor = min(token_factor, 0.92)
        if admission == "defer_background":
            token_factor = min(token_factor, 0.70)
        sampling_bias = {
            "temperature_delta": round(min(0.12, novelty_pressure * 0.12), 4),
            "presence_penalty_delta": round(min(0.18, (novelty_pressure + curiosity) * 0.10), 4),
            "max_tokens_factor": round(token_factor, 4),
        }
        routing_bias = {
            "use_private_scratchpad": salience >= 0.24,
            "model_visual_form": bool(visual_model),
            "generate_alternatives": novelty_pressure >= 0.34 or counterfactual,
            "seek_verification": tool_or_reality,
            "requires_memory_grounding": memory_pressure >= 0.55,
            "raise_metacognition": verification_pressure >= 0.45 or confusion >= 0.25,
            "consolidate_if_success": memory_pressure >= 0.50,
            "avoid_claiming_observation": True,
            "compress_imagination": admission in {"compress_foreground", "thin_frame", "defer_background"},
        }
        frame = ImaginationFrame(
            frame_id=frame_id,
            objective=text[:180],
            mode=mode,
            salience=round(salience, 4),
            novelty_pressure=round(novelty_pressure, 4),
            curiosity_pressure=round(curiosity, 4),
            affective_pressure=round(affective_pressure, 4),
            memory_pressure=round(memory_pressure, 4),
            verification_pressure=round(verification_pressure, 4),
            working_memory=working_memory,
            attractor_state=attractor_state,
            eligibility_trace=eligibility_trace,
            modalities=modalities,
            keywords=keywords,
            attention_targets=attention_targets,
            visual_model=visual_model,
            phrase_model=phrase_model,
            conceptual_bridge=conceptual_bridge,
            mental_canvas=mental_canvas,
            associative_links=associative_links,
            novel_thoughts=novel_thoughts,
            simulation_steps=simulation_steps,
            counterfactuals=counterfactuals,
            experiments=experiments,
            action_affordances=action_affordances,
            ablation_predictions=ablation_predictions,
            causal_effects=causal_effects,
            sampling_bias=sampling_bias,
            routing_bias=routing_bias,
        )
        self._frame_count += 1
        self._history.append(frame)
        self._frame_index[frame.frame_id] = frame
        while len(self._frame_index) > self._history.maxlen:
            live_ids = {item.frame_id for item in self._history}
            for stale_id in list(self._frame_index):
                if stale_id not in live_ids:
                    self._frame_index.pop(stale_id, None)
        return frame

    def snapshot(self) -> dict[str, Any]:
        latest = self._history[-1].to_dict() if self._history else None
        return {
            "status": "active" if latest else "idle",
            "frames": len(self._history),
            "latest": latest,
            "working_memory": self._working_memory_snapshot(),
            "attractor_bias": {
                key: round(value, 4)
                for key, value in sorted(self._attractor_bias.items())
            },
            "eligibility_trace": {
                key: round(value, 4)
                for key, value in sorted(
                    self._eligibility_trace.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:12]
            },
            "recent_outcomes": list(self._outcomes)[-5:],
            "governance": {
                "advisory_only": True,
                "no_external_effects": True,
                "authority_gateway_required_for_effects": True,
            },
        }

    def learn_from_feedback(
        self,
        frame: str | dict[str, Any] | ImaginationFrame | None,
        *,
        reward: float,
        outcome: str = "unknown",
    ) -> dict[str, Any] | None:
        materialized = self._coerce_frame(frame)
        if materialized is None:
            return None
        selected = str(
            (materialized.attractor_state or {}).get("selected")
            or materialized.mode
        )
        reward_value = max(-1.0, min(1.0, _safe_float(reward, 0.0)))
        current_bias = _safe_float(self._attractor_bias.get(selected), 0.0)
        rpe = reward_value - current_bias
        self._attractor_bias[selected] = max(-0.45, min(0.45, current_bias + 0.12 * rpe))
        for key, value in list(materialized.eligibility_trace.items())[:16]:
            previous = _safe_float(self._eligibility_trace.get(key), 0.0)
            self._eligibility_trace[key] = _clamp(previous + reward_value * value * 0.04)
        record = {
            "frame_id": materialized.frame_id,
            "outcome": str(outcome or "unknown")[:80],
            "reward": round(reward_value, 4),
            "selected_attractor": selected,
            "reward_prediction_error": round(rpe, 4),
            "updated_bias": round(self._attractor_bias[selected], 4),
        }
        self._outcomes.append(record)
        return record

    def _coerce_frame(
        self, frame: str | dict[str, Any] | ImaginationFrame | None
    ) -> ImaginationFrame | None:
        if isinstance(frame, ImaginationFrame):
            return frame
        if isinstance(frame, str):
            return self._frame_index.get(frame)
        if isinstance(frame, dict):
            frame_id = str(frame.get("frame_id") or "")
            if frame_id and frame_id in self._frame_index:
                return self._frame_index[frame_id]
            try:
                allowed = {field.name for field in ImaginationFrame.__dataclass_fields__.values()}
                filtered = {key: value for key, value in frame.items() if key in allowed}
                return ImaginationFrame(**filtered)
            except (TypeError, ValueError, AttributeError):
                return None
        return None

    def _observe_working_memory_gate(
        self,
        *,
        salience: float,
        novelty_pressure: float,
        verification_pressure: float,
        memory_pressure: float,
        context: dict[str, Any] | None,
        is_background: bool,
    ) -> dict[str, Any]:
        now = time.monotonic()
        elapsed = max(0.05, min(60.0, now - self._last_observed_at))
        self._last_observed_at = now
        runtime_pressure = self._runtime_memory_pressure(context)
        pressure_level = str(runtime_pressure.get("level") or "normal")
        pressure_rank = {
            "normal": 0.0,
            "warning": 0.18,
            "high": 0.34,
            "critical": 0.62,
            "emergency": 0.90,
        }.get(pressure_level, 0.0)
        arrival_load = _clamp(
            0.12
            + salience * 0.36
            + novelty_pressure * 0.20
            + verification_pressure * 0.16
            + memory_pressure * 0.18
            + pressure_rank * 0.30
        )
        service_rate = _clamp(
            1.05
            - pressure_rank * 0.55
            - (0.20 if is_background else 0.0),
            lower=0.18,
            upper=1.20,
        )
        decay = min(self._queue_load, (elapsed / 6.0) * service_rate)
        self._queue_load = _clamp(self._queue_load - decay + arrival_load * 0.28)
        instantaneous_rate = min(12.0, 1.0 / elapsed)
        self._arrival_rate_ema = (0.82 * self._arrival_rate_ema) + (0.18 * instantaneous_rate)
        self._service_rate_ema = (0.86 * self._service_rate_ema) + (0.14 * service_rate)
        overload = _clamp(max(0.0, self._queue_load - 0.68) / 0.32)
        if pressure_level in {"critical", "emergency"}:
            admission = "thin_frame"
        elif is_background and (overload >= 0.35 or pressure_level in {"warning", "high"}):
            admission = "defer_background"
        elif overload >= 0.45 or pressure_level == "high":
            admission = "compress_foreground"
        else:
            admission = "admit"
        expected_wait = self._queue_load / max(0.05, self._service_rate_ema)
        return {
            "admission": admission,
            "admitted": admission != "defer_background",
            "queue_load": round(self._queue_load, 4),
            "overload_pressure": round(overload, 4),
            "arrival_rate_hz": round(self._arrival_rate_ema, 4),
            "service_rate_hz": round(self._service_rate_ema, 4),
            "utilization": round(_clamp(self._arrival_rate_ema / max(0.05, self._service_rate_ema) / 8.0), 4),
            "expected_wait_s": round(expected_wait, 4),
            "runtime_memory_level": pressure_level,
            "runtime_memory_pressure_pct": round(_safe_float(runtime_pressure.get("pressure_pct"), 0.0), 4),
            "reason": str(runtime_pressure.get("reason") or "")[:220],
        }

    @staticmethod
    def _runtime_memory_pressure(context: dict[str, Any] | None) -> dict[str, Any]:
        if isinstance(context, dict):
            raw = context.get("memory_pressure_snapshot") or context.get("memory_pressure")
            if isinstance(raw, dict):
                return {
                    "level": str(raw.get("level") or "normal"),
                    "pressure_pct": _safe_float(raw.get("pressure_pct"), 0.0),
                    "reason": str(raw.get("reason") or ""),
                }
            if isinstance(raw, (int, float)):
                pct = _safe_float(raw, 0.0)
                if pct >= 94.0:
                    level = "emergency"
                elif pct >= 90.0:
                    level = "critical"
                elif pct >= 84.0:
                    level = "high"
                elif pct >= 78.0:
                    level = "warning"
                else:
                    level = "normal"
                return {"level": level, "pressure_pct": pct, "reason": f"context_memory_pressure:{pct:.1f}%"}
        try:
            from core.utils.memory_monitor import get_memory_pressure_snapshot

            snapshot = get_memory_pressure_snapshot()
            return {
                "level": str(getattr(snapshot, "level", "normal") or "normal"),
                "pressure_pct": _safe_float(getattr(snapshot, "pressure_pct", 0.0), 0.0),
                "reason": str(getattr(snapshot, "reason", "") or ""),
            }
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
            record_degradation(
                "imagination_engine",
                exc,
                severity="warning",
                action="used neutral memory-pressure signal for imagination admission",
            )
            return {"level": "normal", "pressure_pct": 0.0, "reason": "memory_pressure_probe_failed"}

    def _select_attractor_state(
        self,
        *,
        mode: str,
        salience: float,
        novelty_pressure: float,
        curiosity: float,
        affective_pressure: float,
        memory_pressure: float,
        verification_pressure: float,
        working_memory: dict[str, Any],
        visual: bool,
        linguistic: bool,
        counterfactual: bool,
        creative: bool,
        tool_or_reality: bool,
    ) -> dict[str, Any]:
        overload = _safe_float(working_memory.get("overload_pressure"), 0.0)
        scores = {
            "direct_answer": 0.30 + (1.0 - salience) * 0.20 - novelty_pressure * 0.08,
            "mental_canvas": (0.55 if visual else 0.05) + novelty_pressure * 0.34 + curiosity * 0.12,
            "linguistic_surface": (0.50 if linguistic else 0.08) + novelty_pressure * 0.12,
            "counterfactual_probe": (0.55 if counterfactual else 0.06) + verification_pressure * 0.22,
            "memory_bridge": 0.10 + memory_pressure * 0.58,
            "governed_action_boundary": (0.62 if tool_or_reality else 0.02) + verification_pressure * 0.40,
            "creative_synthesis": (0.50 if creative else 0.08) + novelty_pressure * 0.36 + affective_pressure * 0.14,
            "load_stabilization": overload * 0.72,
        }
        for key in list(scores):
            scores[key] = _safe_float(scores[key], 0.0) + _safe_float(self._attractor_bias.get(key), 0.0)
        probabilities = _stable_softmax(scores, temperature=max(0.30, 0.72 + overload * 0.35))
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        selected = ranked[0][0] if ranked else mode
        margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 1.0
        recurrent_depth = 1 + int(round(_clamp(novelty_pressure + verification_pressure + curiosity * 0.5) * 4))
        if overload >= 0.50:
            recurrent_depth = min(recurrent_depth, 2)
        return {
            "selected": selected,
            "probabilities": {key: round(value, 4) for key, value in ranked[:6]},
            "entropy": round(_entropy01(probabilities), 4),
            "stability_margin": round(_clamp(margin), 4),
            "recurrent_depth": recurrent_depth,
            "bias": round(_safe_float(self._attractor_bias.get(selected), 0.0), 4),
            "load_stabilized": selected == "load_stabilization" or overload >= 0.50,
        }

    def _update_eligibility_trace(
        self,
        keywords: list[str],
        *,
        selected_attractor: str,
        salience: float,
        novelty_pressure: float,
        memory_pressure: float,
    ) -> dict[str, float]:
        decayed: dict[str, float] = {}
        for key, value in self._eligibility_trace.items():
            next_value = _safe_float(value, 0.0) * 0.82
            if next_value >= 0.01:
                decayed[key] = next_value
        self._eligibility_trace = decayed
        self._eligibility_trace[f"attractor:{selected_attractor}"] = _clamp(
            self._eligibility_trace.get(f"attractor:{selected_attractor}", 0.0)
            + salience * 0.26
            + novelty_pressure * 0.12
        )
        for token in keywords[:6]:
            trace_key = f"keyword:{token}"
            self._eligibility_trace[trace_key] = _clamp(
                self._eligibility_trace.get(trace_key, 0.0)
                + 0.05
                + memory_pressure * 0.05
            )
        return {
            key: round(value, 4)
            for key, value in sorted(
                self._eligibility_trace.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:12]
        }

    def _working_memory_snapshot(self) -> dict[str, Any]:
        return {
            "queue_load": round(self._queue_load, 4),
            "arrival_rate_hz": round(self._arrival_rate_ema, 4),
            "service_rate_hz": round(self._service_rate_ema, 4),
            "utilization": round(_clamp(self._arrival_rate_ema / max(0.05, self._service_rate_ema) / 8.0), 4),
            "history_limit": self._history.maxlen,
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
    def _build_mental_canvas(
        keywords: list[str],
        memories: list[str],
        text: str,
        *,
        visual: bool,
        linguistic: bool,
        counterfactual: bool,
        creative: bool,
    ) -> dict[str, Any]:
        focus = keywords[:5] or ["request"]
        primary = focus[0]
        secondary = focus[1] if len(focus) > 1 else "context"
        modality = (
            "visual"
            if visual
            else "linguistic"
            if linguistic
            else "counterfactual"
            if counterfactual
            else "conceptual"
            if creative
            else "associative"
        )
        objects = [
            {"id": token, "role": "focus" if index == 0 else "support"}
            for index, token in enumerate(focus[:4])
        ]
        relations = [
            {
                "source": primary,
                "target": secondary,
                "relation": "pressures" if creative or counterfactual else "clarifies",
            }
        ]
        if len(focus) >= 3:
            relations.append(
                {
                    "source": focus[2],
                    "target": primary,
                    "relation": "reframes",
                }
            )
        memory_anchor = _normalize_text(memories[-1], 120) if memories else ""
        image_prompt = (
            f"Internal {modality} canvas: {primary} in the foreground, {secondary} as "
            "the shaping constraint, with tensions, missing evidence, and next affordances "
            "made spatially visible."
        )
        if memory_anchor:
            image_prompt += f" Continuity anchor: {memory_anchor}."
        thought_form = (
            f"Ask what {primary} becomes when {secondary} is treated as a live constraint, "
            "then compare that model against evidence before acting."
        )
        linguistic_surface = (
            f"{primary} under {secondary}"
            if len(focus) >= 2
            else f"{primary} made inspectable"
        )
        return {
            "modality": modality,
            "image_prompt": image_prompt[:500],
            "objects": objects,
            "relations": relations,
            "sensory_style": "clear edges, low ornament, constraints visible as structure",
            "linguistic_surface": linguistic_surface[:160],
            "thought_form": thought_form[:280],
            "memory_anchor": memory_anchor,
            "externalization_path": (
                "If the user asks to see or generate this, request governed image/tool execution; "
                "otherwise keep it private."
            ),
        }

    @staticmethod
    def _build_associative_links(
        keywords: list[str], memories: list[str], text: str
    ) -> list[dict[str, str]]:
        links: list[dict[str, str]] = []
        if len(keywords) >= 2:
            links.append(
                {
                    "source": keywords[0],
                    "relation": "constrains",
                    "target": keywords[1],
                }
            )
        if len(keywords) >= 3:
            links.append(
                {
                    "source": keywords[2],
                    "relation": "reframes",
                    "target": keywords[0],
                }
            )
        if memories and keywords:
            links.append(
                {
                    "source": "recent_memory",
                    "relation": "anchors",
                    "target": keywords[0],
                }
            )
        if "tool" in text.lower() and keywords:
            links.append(
                {
                    "source": keywords[0],
                    "relation": "requires_verification_through",
                    "target": "governed_tools",
                }
            )
        return links[:4]

    @staticmethod
    def _build_novel_thoughts(
        keywords: list[str],
        memories: list[str],
        text: str,
        *,
        creative: bool,
        counterfactual: bool,
    ) -> list[str]:
        focus = keywords[0] if keywords else "the idea"
        secondary = keywords[1] if len(keywords) > 1 else "its constraint"
        candidates = [
            f"What if {focus} is not the object, but the lens for seeing {secondary}?",
            f"The useful novelty may be the smallest testable form of {focus}, not the largest imagined one.",
        ]
        if creative:
            candidates.append(
                f"Combine {focus} with an opposing pressure and look for the behavior neither has alone."
            )
        if counterfactual:
            candidates.append(f"Invert the premise: what would make {focus} fail gracefully?")
        if memories:
            candidates.append(
                "Use recent continuity as material, then deliberately rotate it into a new frame."
            )
        return candidates[:4]

    @staticmethod
    def _build_simulation_steps(
        keywords: list[str],
        *,
        visual: bool,
        counterfactual: bool,
        tool_or_reality: bool,
    ) -> list[str]:
        focus = keywords[0] if keywords else "the premise"
        steps = [
            f"Render {focus} as concrete constraints instead of a label.",
            "Generate at least two alternate forms before settling on one.",
        ]
        if visual:
            steps.append("Place actors, constraints, and missing evidence in the imagined scene.")
        if counterfactual:
            steps.append("Run the premise forward, then invert it and compare behavior.")
        if tool_or_reality:
            steps.append("Stop at the boundary where real-world verification or authority is required.")
        return steps[:4]

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

    @staticmethod
    def _build_attention_targets(keywords: list[str], text: str) -> list[str]:
        targets = list(keywords[:5])
        lowered = text.lower()
        if "tool" in lowered and "governed_tools" not in targets:
            targets.append("governed_tools")
        if ("remember" in lowered or "memory" in lowered) and "memory_continuity" not in targets:
            targets.append("memory_continuity")
        if any(token in lowered for token in ("verify", "proof", "evidence")) and "verification" not in targets:
            targets.append("verification")
        return targets[:6]

    @staticmethod
    def _build_action_affordances(
        keywords: list[str],
        *,
        tool_or_reality: bool,
        visual: bool,
        linguistic: bool,
        counterfactual: bool,
    ) -> list[str]:
        focus = keywords[0] if keywords else "the model"
        affordances = [f"hold {focus} as an internal model before answering"]
        if counterfactual:
            affordances.append("compare at least two possible futures")
        if visual:
            affordances.append("model spatial/visual structure privately before describing it")
        if linguistic:
            affordances.append("search for a concise surface phrase after the model is stable")
        if tool_or_reality:
            affordances.append("route any external effect through governed tools and receipts")
        return affordances[:5]

    @staticmethod
    def _build_ablation_predictions(
        *,
        memory_pressure: float,
        verification_pressure: float,
        novelty_pressure: float,
        tool_or_reality: bool,
    ) -> dict[str, str]:
        predictions: dict[str, str] = {
            "no_imagination": "fewer alternatives and weaker counterfactual framing",
        }
        if memory_pressure >= 0.50:
            predictions["no_memory_continuity"] = "recent context should anchor less of the response"
        if verification_pressure >= 0.45:
            predictions["no_governance_or_tools"] = "real-world claims should lose verification pressure"
        if novelty_pressure >= 0.35:
            predictions["no_novelty_drive"] = "creative synthesis should collapse toward a safer default framing"
        if tool_or_reality:
            predictions["no_authority_gateway"] = "external action must block rather than proceed directly"
        return predictions

    @staticmethod
    def _build_causal_effects(
        *,
        attention_targets: list[str],
        memory_pressure: float,
        verification_pressure: float,
        curiosity: float,
        confusion: float,
        tool_or_reality: bool,
        working_memory: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metacognition_depth = _clamp(0.30 + verification_pressure * 0.45 + confusion * 0.35 + curiosity * 0.20)
        working_memory = working_memory if isinstance(working_memory, dict) else {}
        admission = str(working_memory.get("admission") or "admit")
        overload = _safe_float(working_memory.get("overload_pressure"), 0.0)
        return {
            "attention_focus": attention_targets[:4],
            "memory_priority": round(memory_pressure, 4),
            "verification_pressure": round(verification_pressure, 4),
            "metacognition_depth": round(metacognition_depth, 4),
            "tool_governance": bool(tool_or_reality or verification_pressure >= 0.45),
            "external_effects_allowed": False,
            "working_memory_admission": admission,
            "working_memory_overload": round(overload, 4),
            "load_shed_requested": admission in {"compress_foreground", "thin_frame", "defer_background"},
            "expected_downstream": [
                effect
                for effect, active in (
                    ("attention_bias", bool(attention_targets)),
                    ("memory_retrieval_bias", memory_pressure >= 0.45),
                    ("memory_consolidation_bias", memory_pressure >= 0.50),
                    ("verification_bias", verification_pressure >= 0.35),
                    ("governed_tool_boundary", tool_or_reality),
                    ("runtime_load_shed", admission in {"compress_foreground", "thin_frame", "defer_background"}),
                )
                if active
            ],
        }


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
