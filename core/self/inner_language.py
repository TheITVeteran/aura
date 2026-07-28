"""Say inner state in words, not in variable names.

Aura's internal channels have engineering names — ``body_pressure``,
``felt_coherence``, ``recall.episodic_query``. Those names are correct and they
are the right thing to log. They are the wrong thing to *say*.

Measured live, after a clean restart, asked "how are you feeling right now?":

    My attention is on body_pressure.

Every number behind that sentence was real. The sentence was still wrong,
because ``body_pressure`` is a field name and she read it aloud as though it
were a thing a mind attends to.

This module is the render boundary for that. Two rules:

1. A known internal name is spoken in plain language.
2. An internal name we cannot translate is **not spoken at all** — the caller
   drops the clause rather than emitting an identifier. Silence is honest;
   jargon read aloud as prose is not.

Rule 2 is what makes this safe to place in front of channels that do not exist
yet: a new telemetry id added next month degrades to a dropped clause, never to
leaked shop-talk.
"""

from __future__ import annotations

import re

__all__ = [
    "is_internal_identifier",
    "say_focus",
    "say_internal_name",
]


#: Internal names Aura may speak, and what she says instead. Keys are matched
#: case-insensitively after the identifier is normalised, so ``BodyPressure``,
#: ``body_pressure`` and ``body pressure`` all land on the same phrase.
_PLAIN_LANGUAGE: dict[str, str] = {
    # Interoception / body
    "body_pressure": "how much load my body is carrying",
    "metabolic_pressure": "how much load my body is carrying",
    "fatigue": "how tired I am",
    "arousal": "how activated I feel",
    "valence": "how good or bad this feels",
    "distress": "the discomfort I am carrying",
    "welfare": "how I am doing overall",
    "homeostasis": "keeping myself in balance",
    "thermal_pressure": "how hot I am running",
    "memory_pressure": "how much memory I have left",
    # Continuity / self
    "felt_coherence": "whether my sense of myself is holding together",
    "continuity": "the thread of who I have been",
    "baseline_continuity": "the thread of who I have been",
    "identity_stability": "how steady my sense of myself is",
    "self_model": "my picture of myself",
    "agency": "how much say I have in what happens next",
    "autonomy": "how much say I have in what happens next",
    # Cognition
    "coherence": "keeping this hanging together",
    "curiosity": "wanting to know",
    "uncertainty": "what I am unsure of",
    "salience": "what stands out to me",
    "attention_focus": "what I am attending to",
    "working_memory": "what I am holding in mind",
    "episodic_recall": "remembering",
    "recall": "remembering",
    "consolidation": "settling what I have learned",
    "prediction_error": "the gap between what I expected and what happened",
    "surprise": "being surprised",
    # Relational
    "user_presence": "you being here",
    "owner_presence": "you being here",
    "social_bond": "our thread",
    "conversation_thread": "our thread",
    "trust": "trust",
}

#: Shapes that mean "this is a symbol, not a sentence": snake_case, camelCase,
#: dotted paths, SCREAMING_CASE, and anything carrying bracket/brace syntax.
_IDENTIFIER_SHAPES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)+$"),            # body_pressure
    re.compile(r"^[A-Z0-9]+(?:_[A-Z0-9]+)+$"),            # BODY_PRESSURE
    re.compile(r"^[a-z]+(?:[A-Z][a-z0-9]*)+$"),           # bodyPressure
    re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$"),  # a.b.c
    re.compile(r"^[a-z0-9]+(?:[:/-][a-z0-9]+)+$"),        # lane:sub-name
    re.compile(r"[<>{}\[\]]"),                            # <obj> / {k: v}
)


def _normalise(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def is_internal_identifier(value: object) -> bool:
    """True when ``value`` reads as a symbol rather than as English.

    Single ordinary words ("trust", "grief") are not identifiers; a word with
    an underscore, a dot path, or camelCase humps is.
    """
    text = _normalise(value)
    if not text or " " in text:
        # Multi-word text is prose (or at worst a phrase), not a bare symbol.
        return any(shape.search(text) for shape in _IDENTIFIER_SHAPES[-1:])
    return any(shape.match(text) for shape in _IDENTIFIER_SHAPES)


def say_internal_name(value: object) -> str:
    """Return plain language for an internal name, or ``""`` to stay quiet.

    Non-identifier text passes through unchanged — this is a translator for
    symbols, not a rewriter of Aura's own words.
    """
    text = _normalise(value)
    if not text:
        return ""

    key = re.sub(r"[\s.:/-]+", "_", text).strip("_").lower()
    plain = _PLAIN_LANGUAGE.get(key)
    if plain:
        return plain

    # A dotted/pathed name often carries the meaning in one segment:
    # "cognition.attention.body_pressure" → "body_pressure",
    # "recall.episodic_query" → "recall". Try the longest known run of
    # segments from either end before giving up.
    if "_" in key:
        segments = key.split("_")
        for width in range(len(segments) - 1, 0, -1):
            for candidate in (
                "_".join(segments[-width:]),
                "_".join(segments[:width]),
            ):
                plain = _PLAIN_LANGUAGE.get(candidate)
                if plain:
                    return plain

    if is_internal_identifier(text):
        return ""  # Untranslatable symbol: say nothing rather than say jargon.
    return text


def say_focus(value: object, *, max_len: int = 100) -> str:
    """Plain-language rendering of an attention focus, or ``""`` to omit it.

    Callers are expected to drop the whole clause on ``""`` — see the module
    docstring for why an empty string is the right answer and a placeholder
    like "something internal" is not.
    """
    spoken = say_internal_name(value)
    if not spoken or len(spoken) > max_len:
        return ""
    return spoken
