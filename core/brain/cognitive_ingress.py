"""core/brain/cognitive_ingress.py

Typed ingress for latent-reasoning allocation: stakes and uncertainty come
from the MIND, not the prompt's shape.

`select_foreground_episode` keeps deciding WHETHER a turn is depth-worthy
(a multipart question is a legitimate trigger). This module decides HOW MUCH
thought the episode deserves, by reading the organs that actually know:

    memory      — recall familiarity lowers uncertainty; total blankness raises it
    body        — real + anticipatory pressure raises stakes (errors cost more
                  when the body is strained; the economy separately damps spend)
    goals       — overlap with active goals raises stakes
    will        — volitional preference for deliberate work raises stakes
    affect      — felt uncertainty/doubt raises uncertainty
    self_model  — identity-relevant subjects raise stakes
    world_model — a live world-model context lowers uncertainty slightly

Every signal is defensive (an absent organ contributes nothing and says so),
bounded, and RECEIPTED: the allocation receipt lists each source with
present/value/contribution, so "allocation came from memory hits, body,
goals, Will, uncertainty, self-model" is provable per turn — never inferred
from prompt punctuation.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.CognitiveIngress")

COGNITIVE_INGRESS_SCHEMA = "aura.cognitive_ingress.v1"

_WORD_RE = re.compile(r"[^\W\d_]{4,}", re.UNICODE)

# Baselines when no organ speaks: match the previous hardcoded allocation so
# behavior is stable on a cold registry.
_BASE_STAKES = 0.60
_BASE_UNCERTAINTY = 0.70

_SELF_TERMS = {
    "aura", "yourself", "your", "identity", "memory", "memories", "weights",
    "cortex", "consciousness", "governance", "constitution", "will",
}


@dataclass
class IngressSignal:
    source: str
    present: bool
    value: float | None = None
    stakes_delta: float = 0.0
    uncertainty_delta: float = 0.0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "present": self.present,
            "value": None if self.value is None else round(float(self.value), 4),
            "stakes_delta": round(self.stakes_delta, 4),
            "uncertainty_delta": round(self.uncertainty_delta, 4),
            "detail": self.detail[:160],
        }


@dataclass
class CognitiveIngress:
    stakes: float
    uncertainty: float
    signals: list[IngressSignal] = field(default_factory=list)

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema": COGNITIVE_INGRESS_SCHEMA,
            "stakes": round(self.stakes, 4),
            "uncertainty": round(self.uncertainty, 4),
            "base_stakes": _BASE_STAKES,
            "base_uncertainty": _BASE_UNCERTAINTY,
            "present_sources": [s.source for s in self.signals if s.present],
            "absent_sources": [s.source for s in self.signals if not s.present],
            "signals": [s.to_dict() for s in self.signals],
        }


def _objective_terms(objective: str) -> set[str]:
    return {word.lower() for word in _WORD_RE.findall(objective or "")}


def _get_service(name: str):
    try:
        from core.runtime.service_registry import get_runtime_service

        return get_runtime_service(name, default=None)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _signal_memory(objective: str) -> IngressSignal:
    """Recall familiarity: strong hits ⇒ less uncertainty; blankness ⇒ more."""
    for name in ("memory_facade", "episodic_memory"):
        service = _get_service(name)
        if service is None:
            continue
        for method in ("recall", "search", "retrieve"):
            fn = getattr(service, method, None)
            if not callable(fn):
                continue
            try:
                hits = fn(objective, limit=4) if method != "recall" else fn(objective)
            except TypeError:
                try:
                    hits = fn(objective)
                except Exception:  # noqa: BLE001 - organ contract unknown; absent
                    continue
            except Exception:  # noqa: BLE001 - organ contract unknown; absent
                continue
            count = len(hits) if isinstance(hits, (list, tuple)) else 0
            familiarity = min(1.0, count / 4.0)
            return IngressSignal(
                source="memory",
                present=True,
                value=familiarity,
                uncertainty_delta=(0.10 if count == 0 else -0.10 * familiarity),
                detail=f"{name}.{method}: {count} hits",
            )
    return IngressSignal(source="memory", present=False)


def _signal_body(orchestrator: Any) -> IngressSignal:
    try:
        from core.being.aura_now import BodyState

        state = getattr(orchestrator, "state", None)
        pressure = float(BodyState.from_aura_state(state).total_pressure())
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return IngressSignal(source="body", present=False)
    return IngressSignal(
        source="body",
        present=True,
        value=pressure,
        stakes_delta=0.10 * min(1.0, max(0.0, pressure)),
        detail=f"total_pressure={pressure:.3f}",
    )


def _signal_goals(objective: str) -> IngressSignal:
    service = _get_service("goal_engine") or _get_service("goals")
    if service is None:
        return IngressSignal(source="goals", present=False)
    goals_text: list[str] = []
    for accessor in ("active_goals", "get_active_goals", "goals"):
        candidate = getattr(service, accessor, None)
        try:
            items = candidate() if callable(candidate) else candidate
        except Exception:  # noqa: BLE001 - organ contract unknown; absent
            continue
        if isinstance(items, (list, tuple)):
            for item in items[:16]:
                # NB: attribute probes must type-check the VALUE — a plain
                # string goal has a truthy built-in .title METHOD, which is
                # not a title.
                text = None
                for attr in ("description", "title", "text"):
                    value = getattr(item, attr, None)
                    if isinstance(value, str) and value.strip():
                        text = value
                        break
                if text is None and isinstance(item, str):
                    text = item
                if text:
                    goals_text.append(text)
            break
    if not goals_text:
        return IngressSignal(source="goals", present=False)
    terms = _objective_terms(objective)
    overlap = 0.0
    matched = ""
    for goal in goals_text:
        goal_terms = _objective_terms(goal)
        if not goal_terms:
            continue
        score = len(terms & goal_terms) / max(4, min(len(goal_terms), 12))
        if score > overlap:
            overlap, matched = min(1.0, score), goal[:80]
    return IngressSignal(
        source="goals",
        present=True,
        value=overlap,
        stakes_delta=0.15 * overlap,
        detail=f"best_overlap={overlap:.2f} goal={matched!r}",
    )


def _signal_will(orchestrator: Any) -> IngressSignal:
    service = _get_service("volition") or getattr(orchestrator, "volition", None)
    if service is None:
        return IngressSignal(source="will", present=False)
    for accessor in ("deliberation_preference", "preference_for_deliberation"):
        candidate = getattr(service, accessor, None)
        try:
            value = candidate() if callable(candidate) else candidate
        except Exception:  # noqa: BLE001 - organ contract unknown; absent
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = min(1.0, max(0.0, float(value)))
            return IngressSignal(
                source="will",
                present=True,
                value=value,
                stakes_delta=0.10 * value,
                detail=f"{accessor}={value:.2f}",
            )
    return IngressSignal(source="will", present=False)


def _signal_affect(orchestrator: Any) -> IngressSignal:
    """Felt uncertainty/doubt from the affect organ raises uncertainty."""
    service = _get_service("affect_engine") or getattr(orchestrator, "affect", None)
    if service is None:
        return IngressSignal(source="affect", present=False)
    for accessor in ("felt_uncertainty", "uncertainty", "doubt"):
        candidate = getattr(service, accessor, None)
        try:
            value = candidate() if callable(candidate) else candidate
        except Exception:  # noqa: BLE001 - organ contract unknown; absent
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = min(1.0, max(0.0, float(value)))
            return IngressSignal(
                source="affect",
                present=True,
                value=value,
                uncertainty_delta=0.15 * value,
                detail=f"{accessor}={value:.2f}",
            )
    return IngressSignal(source="affect", present=False)


def _signal_self_model(objective: str) -> IngressSignal:
    terms = _objective_terms(objective)
    matched = sorted(terms & _SELF_TERMS)
    relevance = min(1.0, len(matched) / 2.0)
    return IngressSignal(
        source="self_model",
        present=bool(matched),
        value=relevance if matched else None,
        stakes_delta=0.10 * relevance,
        detail=f"identity_terms={matched[:4]}" if matched else "",
    )


def _signal_world_model(orchestrator: Any) -> IngressSignal:
    service = _get_service("world_model") or _get_service("unified_world_model")
    if service is None:
        return IngressSignal(source="world_model", present=False)
    return IngressSignal(
        source="world_model",
        present=True,
        value=1.0,
        uncertainty_delta=-0.05,
        detail="world model resident",
    )


def assemble_cognitive_ingress(
    orchestrator: Any,
    objective: str,
) -> CognitiveIngress:
    """Typed allocation inputs for one latent episode, with receipts."""
    signals = [
        _signal_memory(objective),
        _signal_body(orchestrator),
        _signal_goals(objective),
        _signal_will(orchestrator),
        _signal_affect(orchestrator),
        _signal_self_model(objective),
        _signal_world_model(orchestrator),
    ]
    stakes = _BASE_STAKES + sum(s.stakes_delta for s in signals if s.present)
    uncertainty = _BASE_UNCERTAINTY + sum(
        s.uncertainty_delta for s in signals if s.present
    )
    return CognitiveIngress(
        stakes=min(1.0, max(0.0, stakes)),
        uncertainty=min(1.0, max(0.0, uncertainty)),
        signals=signals,
    )


__all__ = [
    "COGNITIVE_INGRESS_SCHEMA",
    "CognitiveIngress",
    "IngressSignal",
    "assemble_cognitive_ingress",
]
