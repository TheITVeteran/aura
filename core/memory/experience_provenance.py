"""Did this happen, or did she write it?

Bryan, 2026-07-28: *"i think something makes her create false memories too.
she's referenced a lot of stuff in the past that she has literally never
experienced or heard a specific person say/do."*

He is right, and there is a factory for it. ``NarrativeMemory`` runs on a
loop and does this:

1. takes the last N real episodes,
2. asks the model to write a journal entry about them — the prompt says
   **"Keep it evocative"**,
3. stores that prose with ``type="narrative_journal"`` and nothing else,
4. **deletes the source episodes**,
5. later consolidates journals into a "narrative arc", and arcs into an
   "eternal record" — each one a generation over generations.

So a lived afternoon becomes evocative prose, the lived record is deleted,
and the prose is then summarised into more prose. At no point does anything
mark the result as written rather than experienced, and at recall it renders
with no type attribute at all — indistinguishable from a fact.

That is how "the moon was full and I got to thinking about things" ends up in
her mouth as a memory. It is exactly the register step 2 asks for.

This module is the provenance boundary. Memory is either:

``LIVED``
    It happened to her. A conversation turn, a tool result, an observation.

``GENERATED``
    She wrote it — a journal, a narrative arc, a dream, an imagined scenario,
    a hypothesis. Real content, real value, and **not** a record of events.

``UNKNOWN``
    Nothing said. Treated as generated at the render boundary, because the
    failure of calling a journal a fact is far worse than the failure of
    hedging a fact.

Generated memory is not suppressed. It is *labelled*, so that a mind reading
its own journal knows it is reading its own journal.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "GENERATED",
    "LIVED",
    "UNKNOWN",
    "GENERATED_MEMORY_TYPES",
    "LIVED_MEMORY_TYPES",
    "is_generated",
    "provenance_of",
    "provenance_label",
]

LIVED = "lived"
GENERATED = "generated"
UNKNOWN = "unknown"

#: Memory whose content the model composed. Every one of these is worth
#: keeping and none of them is evidence that anything happened.
GENERATED_MEMORY_TYPES: frozenset[str] = frozenset(
    {
        "narrative_journal",
        "narrative_arc",
        "eternal_record",
        "journal",
        "journal_entry",
        "dream",
        "dream_fragment",
        "dream_journal",
        "reverie",
        "imagination",
        "imagined",
        "simulation",
        "simulated",
        "hypothesis",
        "hypothetical",
        "counterfactual",
        "speculation",
        "reflection",
        "introspection",
        "synthesis",
        "consolidation",
        "abstraction",
        "prediction",
        "forecast",
        "self_narrative",
        "story",
        "fiction",
    }
)

#: Memory that records something that actually occurred.
LIVED_MEMORY_TYPES: frozenset[str] = frozenset(
    {
        "recent_episode",
        "episode",
        "episodic",
        "conversation",
        "conversation_turn",
        "user_message",
        "aura_response",
        "fact",
        "preference",
        "shared_ground",
        "observation",
        "tool_result",
        "skill_result",
        "receipt",
        "measurement",
        "web_page",
        "document",
    }
)

#: Metadata keys that have carried a type or a provenance in some store.
_TYPE_KEYS = ("provenance", "type", "memory_type", "kind", "category", "source")

#: Values of an explicit provenance field.
_EXPLICIT = {
    LIVED: LIVED,
    "experienced": LIVED,
    "observed": LIVED,
    "measured": LIVED,
    "real": LIVED,
    GENERATED: GENERATED,
    "composed": GENERATED,
    "written": GENERATED,
    "authored": GENERATED,
    "imagined": GENERATED,
    "synthetic": GENERATED,
}


def _normalise(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def provenance_of(metadata: Mapping[str, Any] | None) -> str:
    """Classify a memory record as :data:`LIVED`, :data:`GENERATED`, or
    :data:`UNKNOWN`.

    An explicit ``provenance`` wins. Otherwise the record's type decides, and
    a type nobody recognises is UNKNOWN — which callers must not treat as
    lived.
    """
    if not isinstance(metadata, Mapping):
        return UNKNOWN
    explicit = _EXPLICIT.get(_normalise(metadata.get("provenance")))
    if explicit:
        return explicit
    for key in _TYPE_KEYS:
        token = _normalise(metadata.get(key))
        if not token:
            continue
        if token in GENERATED_MEMORY_TYPES:
            return GENERATED
        if token in LIVED_MEMORY_TYPES:
            return LIVED
    return UNKNOWN


def is_generated(metadata: Mapping[str, Any] | None) -> bool:
    """True when this memory is something she wrote, not something that
    happened.

    UNKNOWN counts as generated here on purpose: presenting a journal as a
    fact is a much worse error than hedging a fact.
    """
    return provenance_of(metadata) != LIVED


def provenance_label(metadata: Mapping[str, Any] | None) -> str:
    """The word to render at the recall boundary, or ``""`` for lived memory.

    Lived memory gets no label — it is the ordinary case and the prompt should
    not be cluttered by it. Anything else says what it is.
    """
    verdict = provenance_of(metadata)
    if verdict == LIVED:
        return ""
    if verdict == GENERATED:
        return "written-by-me-not-witnessed"
    return "provenance-unknown"
