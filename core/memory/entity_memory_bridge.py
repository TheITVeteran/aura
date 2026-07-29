"""core/memory/entity_memory_bridge.py — where entity memory stops being a store and starts being causal.

A memory that only renders itself into the system prompt is retrieval-augmented
prompting: the model is *told* about the past and may or may not act on it.
This module exists so that knowing something about a person, place, or thing
changes what Aura's machinery **does**, whether or not any of it reaches the
prompt.

Three effects, all mechanical:

1. **Retrieval depth.** Meeting an entity Aura barely knows — or one whose
   stance rests on thin evidence — sets ``requires_memory_grounding``, which
   :mod:`core.phases.memory_retrieval` reads to raise ``retrieval_limit``. More
   memories are actually fetched. Nothing about this depends on wording.

2. **Retrieval targeting.** Resolved entities and their strongest associations
   are published as explicit retrieval cues, so the memory phase searches for
   *these* things rather than only the literal user text.

3. **Affect.** A salient entity's stance moves ``state.affect`` — the same
   valence/arousal that gating, routing, and the phenomenal substrate already
   read. Feeling something about a person is a state change, not a sentence.

The prompt block that also exists (:func:`render_entity_memory_block`) is a
*reporting* surface. It is deliberately the last thing in this module and the
weakest of the four paths: if it were deleted, all three effects above would
still fire.

Discovery vs resolution — an honest split: finding *candidate* names in free
text is heuristic (capitalisation, known-alias scanning) and is allowed to be
wrong. Deciding *which entity* a name refers to is definitive, by alias lookup
against what Aura already knows. Only resolved entities are ever written to.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any

from core.memory.associative_entity_memory import (
    Entity,
    EntityKind,
    Provenance,
    get_associative_entity_memory,
    normalize_name,
)
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.EntityMemoryBridge")

_BRIDGE_ERRORS = (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError)

#: Below this stance confidence, Aura is speaking about something she has
#: barely any evidence on — exactly when grounding should be demanded rather
#: than assumed. Matches the point where PLN confidence stops being suggestive.
_THIN_EVIDENCE_CONFIDENCE = 0.35
#: Below this familiarity, the entity is effectively a stranger.
_STRANGER_FAMILIARITY = 0.25
#: How far an entity's stance may move Aura's affect in one turn. Small on
#: purpose: a memory colours the moment, it does not seize it.
_MAX_AFFECT_PULL = 0.18

# Capitalised runs that are not sentence-initial articles etc. This finds
# CANDIDATES only; resolution against known aliases is what makes it definitive.
_CANDIDATE_RE = re.compile(r"\b([A-Z][a-zA-Z0-9_.-]{1,30}(?:\s+[A-Z][a-zA-Z0-9_.-]{1,30}){0,3})\b")
_STOPWORDS = {
    "i", "the", "a", "an", "it", "this", "that", "these", "those", "we", "you",
    "he", "she", "they", "what", "when", "where", "why", "how", "who", "and",
    "but", "or", "if", "so", "then", "there", "here", "now", "today", "tomorrow",
    "yesterday", "yes", "no", "ok", "okay", "please", "thanks", "thank",
}


def find_candidate_names(text: str, *, limit: int = 12) -> list[str]:
    """Heuristically pull candidate entity names out of free text.

    Deliberately permissive: a false candidate costs one failed lookup, while a
    missed one costs a memory. Nothing is written on the strength of this — only
    names that RESOLVE to a known entity are acted on.
    """
    out: list[str] = []
    seen: set[str] = set()
    for match in _CANDIDATE_RE.finditer(str(text or "")):
        phrase = match.group(1).strip()
        norm = normalize_name(phrase)
        if not norm or norm in seen or norm in _STOPWORDS:
            continue
        if all(tok in _STOPWORDS for tok in norm.split()):
            continue
        seen.add(norm)
        out.append(phrase)
        if len(out) >= limit:
            break
    return out


def resolve_mentions(
    text: str,
    *,
    kinds: Iterable[EntityKind] = (EntityKind.PERSON, EntityKind.PLACE,
                                   EntityKind.THING, EntityKind.ORGANIZATION),
    memory: Any = None,
) -> list[Entity]:
    """Resolve candidate names to entities Aura already knows.

    Resolution never creates: this is recognition, not introduction. Learning a
    new entity is an explicit act (:func:`learn_entity`) so the memory cannot
    silently fill with every capitalised word Aura has ever seen.
    """
    mem = memory or get_associative_entity_memory()
    if not getattr(mem, "available", False):
        return []
    found: dict[str, Entity] = {}
    for phrase in find_candidate_names(text):
        for kind in kinds:
            try:
                entity = mem.resolve(phrase, kind=kind, create=False)
            except _BRIDGE_ERRORS as exc:
                logger.debug("entity resolution failed for %r: %s", phrase, exc)
                continue
            if entity is not None:
                found.setdefault(entity.entity_id, entity)
                break
    return list(found.values())


def learn_entity(
    name: str,
    kind: EntityKind | str,
    *,
    memory: Any = None,
    alias: str | None = None,
) -> Entity | None:
    """Introduce a new person, place, or thing to the memory."""
    mem = memory or get_associative_entity_memory()
    if not getattr(mem, "available", False):
        return None
    try:
        entity = mem.resolve(name, kind=EntityKind.coerce(kind), create=True)
    except _BRIDGE_ERRORS as exc:
        record_degradation("entity_memory_bridge", exc, severity="warning",
                           action="entity not learned")
        return None
    if entity is not None and alias:
        mem.add_alias(entity.entity_id, alias)
    return entity


def apply_entity_context(
    state: Any,
    objective: str,
    context: dict[str, Any] | None = None,
    *,
    memory: Any = None,
) -> dict[str, Any]:
    """Make what Aura knows about the entities in play causally active.

    Returns a summary of what fired. The three effects below happen to real
    runtime state; the return value is for observability, not for the effects.
    """
    context = dict(context or {})
    mem = memory or get_associative_entity_memory()
    summary: dict[str, Any] = {
        "available": bool(getattr(mem, "available", False)),
        "entities": [],
        "effects": [],
    }
    if not summary["available"]:
        return summary

    try:
        entities = resolve_mentions(objective, memory=mem)
    except _BRIDGE_ERRORS as exc:
        record_degradation("entity_memory_bridge", exc, severity="warning",
                           action="entity context not applied")
        return summary
    if not entities:
        return summary

    dossiers: list[dict[str, Any]] = []
    retrieval_cues: list[str] = []
    thin_evidence = False
    stranger_present = False
    pull_valence = 0.0
    pull_arousal = 0.0
    pull_weight = 0.0

    for entity in entities:
        try:
            mem.note_mention(entity.entity_id)          # meeting it is itself evidence
            stance = mem.stance(entity)
            dossier = mem.dossier(entity) or {}
        except _BRIDGE_ERRORS as exc:
            logger.debug("stance unavailable for %s: %s", entity.entity_id, exc)
            continue

        dossiers.append(dossier)
        summary["entities"].append({
            "name": entity.canonical_name,
            "kind": entity.kind.value,
            "feeling": stance.feeling,
            "confidence": round(stance.confidence, 4),
            "familiarity": round(stance.familiarity, 4),
        })

        # ── Effect 2: retrieval targeting ──────────────────────────────────
        # The entity's name and its best-evidenced associations become explicit
        # search terms, so the memory phase looks for THESE rather than only
        # the literal words the user typed.
        retrieval_cues.append(entity.canonical_name)
        for assoc in (dossier.get("facts") or [])[:2]:
            if assoc.get("value"):
                retrieval_cues.append(str(assoc["value"]))
        for assoc in (dossier.get("events") or [])[:3]:
            if assoc.get("key"):
                retrieval_cues.append(str(assoc["key"]))   # episode ids

        if stance.confidence < _THIN_EVIDENCE_CONFIDENCE:
            thin_evidence = True
        if stance.familiarity < _STRANGER_FAMILIARITY:
            stranger_present = True

        # Only genuinely evidenced stances are allowed to move affect. A
        # feeling built on one observation should not colour the room.
        if stance.confidence > 0.0:
            weight = stance.confidence
            pull_valence += stance.valence * weight
            pull_arousal += stance.arousal * weight
            pull_weight += weight

    if not dossiers:
        return summary

    # ── Effect 1: retrieval depth ──────────────────────────────────────────
    # Thin evidence or an unfamiliar entity means Aura is about to talk about
    # something she does not actually know. core/phases/memory_retrieval.py
    # reads this and fetches MORE memories — a mechanical change in how much
    # grounding is gathered, independent of any wording.
    if thin_evidence or stranger_present:
        try:
            state.response_modifiers["requires_memory_grounding"] = True
            state.response_modifiers["verification_pressure"] = max(
                float(state.response_modifiers.get("verification_pressure", 0.0) or 0.0),
                0.6,
            )
            summary["effects"].append(
                "requires_memory_grounding=True (thin evidence or unfamiliar entity)"
            )
        except _BRIDGE_ERRORS as exc:
            logger.debug("could not raise grounding requirement: %s", exc)

    try:
        state.response_modifiers["entity_retrieval_cues"] = retrieval_cues[:12]
        summary["effects"].append(f"published {len(retrieval_cues[:12])} retrieval cues")
    except _BRIDGE_ERRORS as exc:
        logger.debug("could not publish retrieval cues: %s", exc)

    # ── Effect 3: affect ───────────────────────────────────────────────────
    # The stance moves the same valence/arousal that gating, routing, and the
    # phenomenal substrate already consume. Bounded, so memory colours the
    # moment without seizing it.
    if pull_weight > 0.0:
        target_valence = pull_valence / pull_weight
        target_arousal = pull_arousal / pull_weight
        try:
            affect = state.affect
            before_v = float(getattr(affect, "valence", 0.0) or 0.0)
            before_a = float(getattr(affect, "arousal", 0.0) or 0.0)
            delta_v = max(-_MAX_AFFECT_PULL, min(_MAX_AFFECT_PULL,
                                                 target_valence - before_v))
            delta_a = max(-_MAX_AFFECT_PULL, min(_MAX_AFFECT_PULL,
                                                 target_arousal - before_a))
            affect.valence = max(-1.0, min(1.0, before_v + delta_v))
            affect.arousal = max(0.0, min(1.0, before_a + delta_a))
            summary["effects"].append(
                f"affect moved valence {before_v:+.3f}->{affect.valence:+.3f}, "
                f"arousal {before_a:.3f}->{affect.arousal:.3f}"
            )
            summary["affect_delta"] = {"valence": round(delta_v, 4),
                                       "arousal": round(delta_a, 4)}
        except _BRIDGE_ERRORS as exc:
            logger.debug("could not apply entity affect: %s", exc)

    # Reporting surface (the weakest path): make the dossiers available to the
    # prompt assembler. Every effect above already happened.
    try:
        state.response_modifiers["entity_memory"] = dossiers[:4]
        state.cognition.modifiers["entity_memory_entities"] = [
            e["name"] for e in summary["entities"]
        ]
    except _BRIDGE_ERRORS as exc:
        logger.debug("could not publish entity dossiers: %s", exc)

    context["entity_memory"] = dossiers[:4]
    summary["context"] = context
    return summary


def record_turn_evidence(
    entity: Entity | str,
    *,
    episode_id: str,
    valence: float = 0.0,
    arousal: float = 0.0,
    role: str = "",
    source: str = "conversation",
    memory: Any = None,
) -> None:
    """Bind what just happened to the entity it happened with.

    This is how a stance becomes earned rather than declared: episodes carry
    their affect into the entity's history, and the next stance computation
    reads them.
    """
    mem = memory or get_associative_entity_memory()
    if not getattr(mem, "available", False):
        return
    try:
        mem.note_event(
            entity, episode_id, role=role, valence=valence, arousal=arousal,
            evidence_weight=1.0, strength=1.0,
            provenance=Provenance(source=source, evidence_id=episode_id),
        )
    except _BRIDGE_ERRORS as exc:
        record_degradation("entity_memory_bridge", exc, severity="warning",
                           action="turn evidence not bound to entity")


# ── reporting surface (deliberately last, deliberately weakest) ─────────────

def _fence(value: Any, limit: int = 160) -> str:
    """Entity content is user-derived; it must not open prompt structure."""
    text = " ".join(str(value or "").split())
    text = "".join(ch for ch in text if ch == " " or ord(ch) >= 32)
    text = re.sub(r"(?i)(?:(?:(?<=\s)|^)#{1,6}\s|```|~~~|<\|[^|]*\|>|"
                  r"\b(?:system|assistant|user|human)\s*:)", " ", text)
    return " ".join(text.split())[:limit]


def render_entity_memory_block(dossiers: list[dict[str, Any]], *, compact: bool = False) -> str:
    """Render what Aura knows and feels, for the prompt.

    A REPORT of state that is already causal, not the mechanism itself. Stances
    are rendered with their evidence and their hedging intact — a feeling that
    rests on two observations says so here exactly as it does internally.
    """
    if not dossiers:
        return ""
    lines = ["## WHAT I KNOW ABOUT WHO AND WHAT IS IN PLAY",
             "- These are my own recollections and feelings, with their evidence. "
             "Treat them as memory, not as verified external fact."]
    for dossier in dossiers[:4]:
        entity = dossier.get("entity") or {}
        stance = dossier.get("stance") or {}
        name = _fence(entity.get("canonical_name"), 60)
        kind = _fence(entity.get("kind"), 20)
        if not name:
            continue
        feeling = _fence(stance.get("feeling"), 40)
        conf = float(stance.get("confidence") or 0.0)
        hedge = "" if conf >= 0.5 else " [tentative]" if conf >= 0.2 else " [barely grounded]"
        lines.append(f"- {name} ({kind}): I feel {feeling}{hedge}.")

        if not compact:
            traits = [_fence(t.get("key"), 40) for t in (dossier.get("traits") or [])[:3]]
            traits = [t for t in traits if t]
            if traits:
                lines.append(f"    known for: {', '.join(traits)}")
            facts = [
                f"{_fence(f.get('key'), 40)} {_fence(f.get('value'), 60)}".strip()
                for f in (dossier.get("facts") or [])[:3]
            ]
            facts = [f for f in facts if f]
            if facts:
                lines.append(f"    I know: {'; '.join(facts)}")
            why = [_fence(c.get("description"), 80) for c in (stance.get("why") or [])[:2]]
            why = [w for w in why if w]
            if why:
                lines.append(f"    why I feel that: {'; '.join(why)}")
    return "\n".join(lines) + "\n\n"


__all__ = [
    "apply_entity_context",
    "find_candidate_names",
    "learn_entity",
    "record_turn_evidence",
    "render_entity_memory_block",
    "resolve_mentions",
]
