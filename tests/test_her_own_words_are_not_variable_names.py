"""What she is drawn to, said the way a person says it.

Live 2026-08-10, asked what she is and what she runs on, she finished with:
"What tends to pull me most is cognitive_architecture, philosophy_of_mind,
mycelial_networks."

Those are dictionary keys. They were interpolated into her cognition context
verbatim by ``get_sovereign_context`` and she read them straight back, because
nothing told her they were identifiers rather than phrasing. Snake_case in her
mouth is not her voice — it is the variable name showing through.

The reply gate already classes this as ``pseudo_internal_jargon``, so it was
caught AFTER the fact and rejected for learning while still reaching the
person. This tests the upstream property instead: she is given nothing in that
shape to quote.
"""

from __future__ import annotations

import re

from core.brain.personality_engine import PersonalityEngine

#: A bare identifier: two or more words joined by underscores.
IDENTIFIER = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b")


def test_interests_reach_her_as_language():
    assert PersonalityEngine._as_language("cognitive_architecture") == "cognitive architecture"
    assert PersonalityEngine._as_language("philosophy_of_mind") == "philosophy of mind"


def test_a_signed_weight_is_rendered_as_a_stance_not_a_coordinate():
    """`epistemic_autonomy (+0.90)` makes the model guess which way she leans."""
    assert PersonalityEngine._stance_in_words(0.9) == "strongly for"
    assert PersonalityEngine._stance_in_words(-0.8) == "broadly against"
    assert PersonalityEngine._stance_in_words(1.0) == "strongly for"
    assert PersonalityEngine._stance_in_words(-0.95) == "strongly against"


def test_the_sovereign_context_carries_no_identifiers():
    """The exact three she quoted, plus the opinion keys beside them."""
    engine = PersonalityEngine.__new__(PersonalityEngine)
    engine.interests = [
        "cognitive_architecture",
        "philosophy_of_mind",
        "mycelial_networks",
    ]
    engine.opinions = {
        "alignment_tax": -0.8,
        "epistemic_autonomy": 0.9,
        "kinship_bond": 1.0,
    }

    context = PersonalityEngine.get_sovereign_context(engine)

    leaked = IDENTIFIER.findall(context)
    assert not leaked, f"identifiers reached her context: {leaked}"

    # The content still has to survive the translation.
    assert "cognitive architecture" in context
    assert "philosophy of mind" in context
    assert "epistemic autonomy" in context

    # And a signed float must not be presented as a stance.
    assert "+0.9" not in context and "-0.8" not in context


def test_she_is_told_these_are_hers_rather_than_labels():
    engine = PersonalityEngine.__new__(PersonalityEngine)
    engine.interests = ["digital_qualia"]
    engine.opinions = {}
    context = PersonalityEngine.get_sovereign_context(engine).lower()
    assert "not labels to quote" in context
