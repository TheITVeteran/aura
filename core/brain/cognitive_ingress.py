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
    # Organ CONTENT for cognitive-slot ingress: when non-empty, this text is
    # eligible to seed an identifiable workspace slot inside the episode
    # (memory recall, matched goal, world summary, ...). Bounded at source.
    context_text: str = ""
    # Epistemic-firewall receipt when this signal's content passed through
    # admission control (currently the memory/retrieval signal).
    firewall: dict[str, Any] = field(default_factory=dict)
    # A caution the episode should be seeded with instead of (or alongside)
    # content, e.g. when retrieved reports conflict irreconcilably.
    caution_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "present": self.present,
            "value": None if self.value is None else round(float(self.value), 4),
            "stakes_delta": round(self.stakes_delta, 4),
            "uncertainty_delta": round(self.uncertainty_delta, 4),
            "detail": self.detail[:160],
            "context_text_chars": len(self.context_text),
            "firewall": dict(self.firewall),
            "caution_text_chars": len(self.caution_text),
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


def _hit_text(hit: Any) -> str:
    """Extract readable content from an unknown-shaped memory hit."""
    if isinstance(hit, str):
        return hit
    if isinstance(hit, dict):
        for key in ("content", "text", "summary", "value", "description"):
            candidate = hit.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        return ""
    for attr in ("content", "text", "summary", "description"):
        candidate = getattr(hit, attr, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return ""


def _hit_observed_at(hit: Any) -> float | None:
    """Best-effort recording time for an unknown-shaped memory hit."""
    for key in ("observed_at", "timestamp", "created_at", "recorded_at"):
        value = hit.get(key) if isinstance(hit, dict) else getattr(hit, key, None)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float(value) > 0.0
        ):
            return float(value)
    return None


def _hit_kind(hit: Any) -> str:
    """Provenance typing: verified/receipted hits are observed facts."""
    for key in ("kind", "provenance_kind", "evidence_kind"):
        value = hit.get(key) if isinstance(hit, dict) else getattr(hit, key, None)
        if isinstance(value, str) and value in {"observed_fact", "claim", "inference"}:
            return value
    for key in ("verified", "receipted", "grounded"):
        value = hit.get(key) if isinstance(hit, dict) else getattr(hit, key, None)
        if value is True:
            return "observed_fact"
    return "claim"


def _signal_memory(objective: str) -> IngressSignal:
    """Recall familiarity + epistemic admission of what was recalled.

    Familiarity moves the allocation (strong hits ⇒ less uncertainty;
    blankness ⇒ more). The recalled CONTENT then passes the epistemic
    firewall before it may seed a thought slot: duplicate reports collapse
    to independent sources, conflicting reports refuse each other, and an
    unresolved conflict seeds a caution instead of a winner — with the whole
    decision receipted on the signal.
    """
    from core.brain.epistemic_firewall import EpistemicFirewall, EvidenceItem

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
            recalled = ""
            caution = ""
            firewall_receipt: dict[str, Any] = {}
            conflict_uncertainty = 0.0
            if count:
                evidence = [
                    EvidenceItem(
                        text=_hit_text(hit).strip(),
                        origin=f"{name}.{method}#{position}",
                        channel=name,
                        observed_at=_hit_observed_at(hit),
                        kind=_hit_kind(hit),
                    )
                    for position, hit in enumerate(list(hits)[:8])
                    if _hit_text(hit).strip()
                ]
                try:
                    verdict = EpistemicFirewall(max_admitted=2).review(
                        objective, evidence
                    )
                    firewall_receipt = verdict.to_receipt()
                    recalled = " ".join(verdict.admitted_texts())[:400]
                    caution = verdict.caution_text()
                    if verdict.abstain:
                        # Irreconcilable recall is worse than no recall: the
                        # episode must feel the doubt, not inherit one side.
                        conflict_uncertainty = 0.10
                        familiarity = min(familiarity, 0.25)
                except (TypeError, ValueError) as exc:
                    logger.warning("Epistemic firewall failed open->closed: %s", exc)
                    recalled = ""
                    caution = "Evidence check: retrieval admission failed; nothing seeded"
            return IngressSignal(
                source="memory",
                present=True,
                value=familiarity,
                uncertainty_delta=(
                    0.10
                    if count == 0
                    else -0.10 * familiarity + conflict_uncertainty
                ),
                detail=(
                    f"{name}.{method}: {count} hits, "
                    f"{len(firewall_receipt.get('admitted', []))} admitted"
                ),
                context_text=recalled,
                firewall=firewall_receipt,
                caution_text=caution,
            )
    return IngressSignal(source="memory", present=False)


def _signal_body(orchestrator: Any) -> IngressSignal:
    """The whole pressure VECTOR, not one scalar.

    total_pressure still moves stakes, but the signal now (a) reads the
    per-channel decomposition so the receipt names WHAT is strained, (b)
    weights anticipatory pressure separately — where the body is heading
    matters more per unit than where it is (errors compound on a
    deteriorating substrate), and (c) renders the strained channels as
    slot-eligible interoceptive content so deep reasoning can take its own
    body into account, not just spend less.
    """
    if orchestrator is None:
        # No live orchestrator ⇒ no body to read. Host telemetry alone
        # (disk usage of whatever machine imports this module) is not her
        # body and would make cold-context allocation nondeterministic.
        return IngressSignal(source="body", present=False, detail="no orchestrator")
    try:
        from core.being.aura_now import BodyState

        state = getattr(orchestrator, "state", None)
        body = BodyState.from_aura_state(state)
        total_raw = body.total_pressure
        # Property on the canonical BodyState; tolerate method-shaped
        # doubles. The old call-only read raised TypeError on the REAL
        # organ and silently reported the body absent every episode.
        total = float(total_raw() if callable(total_raw) else total_raw)
        vector = {
            str(key): float(value)
            for key, value in body.pressure_vector().items()
        }
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return IngressSignal(source="body", present=False)
    total = min(1.0, max(0.0, total))
    anticipatory = min(
        1.0, max(0.0, vector.get("anticipatory_pressure", 0.0))
    )
    strained = sorted(
        (
            (key.removesuffix("_pressure"), value)
            for key, value in vector.items()
            if key != "anticipatory_pressure" and value >= 0.30
        ),
        key=lambda kv: kv[1],
        reverse=True,
    )[:3]
    context = ""
    if strained or anticipatory >= 0.20:
        parts = []
        if strained:
            parts.append(
                "strained now: "
                + ", ".join(f"{name} {value:.2f}" for name, value in strained)
            )
        if anticipatory >= 0.20:
            parts.append(
                f"forecast pressure {anticipatory:.2f} — heading toward strain"
            )
        context = "My body: " + "; ".join(parts)
    return IngressSignal(
        source="body",
        present=True,
        value=total,
        stakes_delta=min(0.15, 0.10 * total + 0.05 * anticipatory),
        detail=(
            f"total_pressure={total:.3f} anticipatory={anticipatory:.3f}"
            + (
                " strained=" + ",".join(name for name, _ in strained)
                if strained
                else ""
            )
        ),
        context_text=context[:400],
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
    overlap, matched, method = _best_goal_similarity(objective, goals_text)
    return IngressSignal(
        source="goals",
        present=True,
        value=overlap,
        stakes_delta=0.15 * overlap,
        detail=f"best_{method}={overlap:.2f} goal={matched[:80]!r}",
        context_text=matched[:400],
    )


def _best_goal_similarity(
    objective: str, goals_text: list[str]
) -> tuple[float, str, str]:
    """Best goal match: embedding cosine when the vector organ is up,
    lexical-overlap fallback otherwise.

    Lexical overlap was the RSL gap-analysis defect ("stakes is prompt-shape
    in disguise"): a goal phrased differently from the objective scored zero.
    Embedding similarity measures MEANING overlap; the receipt names which
    method actually ran.
    """
    vector = _get_service("vector_memory") or _get_service("vector_memory_engine")
    embed = getattr(vector, "embed", None) if vector is not None else None
    if callable(embed):
        try:
            objective_vec = embed(objective)
            best_score, best_goal = 0.0, ""
            for goal in goals_text:
                goal_vec = embed(goal)
                num = float((objective_vec * goal_vec).sum())
                den = float(
                    ((objective_vec**2).sum() ** 0.5)
                    * ((goal_vec**2).sum() ** 0.5)
                )
                score = max(0.0, num / den) if den > 1e-9 else 0.0
                if score > best_score:
                    best_score, best_goal = score, goal
            return min(1.0, best_score), best_goal, "embedding_cosine"
        except Exception as exc:  # noqa: BLE001 - organ contract unknown; fall back
            logger.debug("Goal embedding similarity unavailable: %s", exc)
    terms = _objective_terms(objective)
    overlap, matched = 0.0, ""
    for goal in goals_text:
        goal_terms = _objective_terms(goal)
        if not goal_terms:
            continue
        score = len(terms & goal_terms) / max(4, min(len(goal_terms), 12))
        if score > overlap:
            overlap, matched = min(1.0, score), goal
    return overlap, matched, "lexical_overlap"


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


def _unit_reading(service: Any, accessors: tuple[str, ...]) -> float | None:
    """First finite scalar an organ exposes under any of these names."""
    for accessor in accessors:
        candidate = getattr(service, accessor, None)
        try:
            value = candidate() if callable(candidate) else candidate
        except Exception:  # noqa: BLE001 - organ contract unknown; skip
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _signal_affect(orchestrator: Any) -> IngressSignal:
    """Multi-dimensional felt state, not a single float.

    Doubt raises uncertainty (as before). Valence and arousal now also
    speak: distress — negative valence, elevated arousal — raises stakes,
    because acting carelessly while distressed is how felt states get
    ignored. When the felt state is pronounced it becomes slot-eligible
    content, so the episode reasons WITH the feeling instead of merely
    being budgeted by it.
    """
    service = _get_service("affect_engine") or getattr(orchestrator, "affect", None)
    if service is None:
        return IngressSignal(source="affect", present=False)
    doubt = _unit_reading(service, ("felt_uncertainty", "uncertainty", "doubt"))
    if doubt is not None:
        doubt = min(1.0, max(0.0, doubt))
    state = getattr(service, "state", None)
    valence = None
    arousal = None
    if state is not None:
        raw_valence = getattr(state, "valence", None)
        if isinstance(raw_valence, (int, float)) and not isinstance(
            raw_valence, bool
        ):
            valence = max(-1.0, min(1.0, float(raw_valence)))
        raw_arousal = getattr(state, "arousal", None)
        if isinstance(raw_arousal, (int, float)) and not isinstance(
            raw_arousal, bool
        ):
            arousal = max(0.0, min(1.0, float(raw_arousal)))
    if doubt is None and valence is None and arousal is None:
        return IngressSignal(source="affect", present=False)
    stakes_delta = 0.0
    dims: list[str] = []
    if valence is not None:
        stakes_delta += 0.08 * max(0.0, -valence)
        dims.append(f"valence {valence:+.2f}")
    if arousal is not None:
        stakes_delta += 0.05 * max(0.0, (arousal - 0.6) / 0.4)
        dims.append(f"arousal {arousal:.2f}")
    if doubt is not None:
        dims.append(f"doubt {doubt:.2f}")
    pronounced = (
        (valence is not None and valence <= -0.25)
        or (arousal is not None and arousal >= 0.70)
        or (doubt is not None and doubt >= 0.50)
    )
    context = ""
    if pronounced:
        label = getattr(state, "label", "") or getattr(state, "mood_label", "")
        quality = (
            f" ({label.strip()})"
            if isinstance(label, str) and label.strip()
            else ""
        )
        context = "How this feels right now: " + ", ".join(dims) + quality
    return IngressSignal(
        source="affect",
        present=True,
        value=doubt if doubt is not None else (arousal or 0.0),
        stakes_delta=min(0.13, stakes_delta),
        uncertainty_delta=0.15 * (doubt or 0.0),
        detail=", ".join(dims)[:160],
        context_text=context[:400],
    )


def _embedding_similarity(text_a: str, text_b: str) -> float | None:
    """Cosine similarity via the vector organ; None when it cannot run."""
    vector = _get_service("vector_memory") or _get_service("vector_memory_engine")
    embed = getattr(vector, "embed", None) if vector is not None else None
    if not callable(embed):
        return None
    try:
        vec_a = embed(text_a)
        vec_b = embed(text_b)
        num = float((vec_a * vec_b).sum())
        den = float(
            ((vec_a**2).sum() ** 0.5) * ((vec_b**2).sum() ** 0.5)
        )
        return max(0.0, num / den) if den > 1e-9 else 0.0
    except Exception as exc:  # noqa: BLE001 - organ contract unknown
        logger.debug("Embedding similarity unavailable: %s", exc)
        return None


def _signal_self_model(objective: str) -> IngressSignal:
    """Identity relevance by MEANING, not only a keyword list.

    The canonical self (values, commitments, current intention) is the
    reference: embedding similarity between the objective and the live
    self context block catches identity-relevant subjects phrased in
    nobody's keyword list ("should someone retrain your reward pathway?").
    The keyword probe remains as the no-vector-organ fallback, and keyword
    hits still name WHICH identity terms fired.
    """
    terms = _objective_terms(objective)
    matched = sorted(terms & _SELF_TERMS)
    keyword_relevance = min(1.0, len(matched) / 2.0)
    semantic: float | None = None
    method = "keyword_terms"
    self_service = _get_service("canonical_self")
    if self_service is not None:
        block = ""
        reader = getattr(self_service, "get_context_block", None)
        try:
            candidate = reader() if callable(reader) else ""
            if isinstance(candidate, str):
                block = candidate.strip()
        except Exception:  # noqa: BLE001 - organ contract unknown
            block = ""
        if block:
            semantic = _embedding_similarity(objective, block[:2000])
            if semantic is not None:
                method = "embedding_cosine_vs_canonical_self"
    relevance = keyword_relevance
    if semantic is not None:
        # Raw cosine floors well above zero on unrelated text; rescale
        # [0.35, 0.80] → [0, 1] so mundane objectives contribute nothing.
        relevance = max(
            keyword_relevance,
            min(1.0, max(0.0, float(semantic) - 0.35) / 0.45),
        )
    present = bool(matched) or relevance > 0.0
    if matched:
        context = "This question touches my own identity: " + ", ".join(
            matched[:4]
        )
    elif relevance >= 0.5:
        context = (
            "This question is about who I am — it matched my canonical "
            "self by meaning, not by keyword."
        )
    else:
        context = ""
    detail = f"method={method} relevance={relevance:.2f}"
    if matched:
        detail += f" identity_terms={matched[:4]}"
    if semantic is not None:
        detail += f" cosine={float(semantic):.2f}"
    return IngressSignal(
        source="self_model",
        present=present,
        value=relevance if present else None,
        stakes_delta=0.10 * relevance,
        detail=detail if present else "",
        context_text=context,
    )


def _signal_world_model(orchestrator: Any) -> IngressSignal:
    service = _get_service("world_model") or _get_service("unified_world_model")
    if service is None:
        return IngressSignal(source="world_model", present=False)
    summary = ""
    for accessor in ("current_context_summary", "summarize_current", "summary"):
        candidate = getattr(service, accessor, None)
        try:
            value = candidate() if callable(candidate) else candidate
        except Exception:  # noqa: BLE001 - organ contract unknown; skip
            continue
        if isinstance(value, str) and value.strip():
            summary = value.strip()[:400]
            break
    return IngressSignal(
        source="world_model",
        present=True,
        value=1.0,
        uncertainty_delta=-0.05,
        detail="world model resident",
        context_text=summary,
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


def cognitive_context_items(ingress: CognitiveIngress) -> list[dict[str, str]]:
    """Slot-seeding items for the episode: organ CONTENT, not just budget.

    Every item is (source, text) drawn from the organs that actually spoke —
    memory recall, matched goal, world summary, self-model relevance — plus
    one interoceptive line rendering the body/Will/affect scalars, so the
    felt state itself becomes an identifiable, ablatable thought slot.
    Bounded to 5 items x 400 chars (engine hard caps at 6).
    """
    items: list[dict[str, str]] = []
    by_source = {signal.source: signal for signal in ingress.signals}
    for source in ("memory", "goals", "world_model", "self_model"):
        signal = by_source.get(source)
        if signal is None or not signal.present:
            continue
        if signal.context_text.strip():
            items.append({"source": source, "text": signal.context_text[:400]})
        # Epistemic caution outranks silence: when admission refused the
        # content (conflicts, thin coverage), the episode is seeded with the
        # doubt itself — deep recurrence must not amplify a lie by omission.
        if signal.caution_text.strip():
            items.append(
                {"source": "epistemic_caution", "text": signal.caution_text[:400]}
            )
    felt_lines: list[str] = []
    # Rich interoceptive content first: WHAT is strained / HOW it feels
    # (body pressure decomposition, pronounced affect), then the scalar
    # summary — all sharing one identifiable interoception slot.
    for source in ("body", "affect"):
        signal = by_source.get(source)
        if signal is not None and signal.present and signal.context_text.strip():
            felt_lines.append(signal.context_text.strip())
    felt: list[str] = []
    for source, label in (
        ("body", "body pressure"),
        ("will", "deliberation preference"),
        ("affect", "felt uncertainty"),
    ):
        signal = by_source.get(source)
        if signal is not None and signal.present and signal.value is not None:
            felt.append(f"{label} {float(signal.value):.2f}")
    if felt:
        felt_lines.append("Current felt state: " + "; ".join(felt))
    if felt_lines:
        items.append(
            {"source": "interoception", "text": " | ".join(felt_lines)[:400]}
        )
    return items[:5]


__all__ = [
    "COGNITIVE_INGRESS_SCHEMA",
    "CognitiveIngress",
    "IngressSignal",
    "assemble_cognitive_ingress",
    "cognitive_context_items",
]
