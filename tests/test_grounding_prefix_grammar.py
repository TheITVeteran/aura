"""A grounding prefix must not break the sentence it introduces.

`_ground_live_voice_surface` prepends a clause ending in ", " and then used to
UPPERCASE the body's first letter — the opposite of what a continuation needs.
Live on the desktop surface: "From my conversation memory, Forgetting is a
mercy." The same mechanism produced the July recall defect's
"From my conversation memory, The code you gave me earlier was ...".
"""
from __future__ import annotations

from types import SimpleNamespace

from core.phases.dialogue_policy import (
    _ground_live_voice_surface,
    _lowercase_continuation_start,
)


def _memory_contract() -> SimpleNamespace:
    return SimpleNamespace(
        requires_memory_grounding=True,
        requires_state_reflection=False,
        requires_reasoned_defense=False,
        requires_identity_defense=False,
        requires_self_preservation=False,
        requires_recent_specific_grounding=False,
        requires_aura_stance=False,
        requires_aura_question=False,
    )


class TestContinuationCase:
    def test_ordinary_sentence_openers_are_downcased(self):
        for body, expected_start in (
            ("Forgetting is a mercy.", "forgetting"),
            ("The code you gave me was 7213.", "the"),
            ("That's a real tension.", "that's"),
            ("These are the two I remember.", "these"),
            ("Remembering everything would be cruelty.", "remembering"),
        ):
            assert _lowercase_continuation_start(body).split()[0].rstrip(".,") == (
                expected_start.rstrip(".,")
            ), body

    def test_names_and_acronyms_keep_their_capital(self):
        for body in (
            "Bryan asked me about this earlier.",
            "Aura is what they call me.",
            "RAM pressure is high right now.",
            "I remember that clearly.",
        ):
            assert _lowercase_continuation_start(body) == body, body

    def test_already_lowercase_is_untouched(self):
        assert _lowercase_continuation_start("my attention narrowed.") == (
            "my attention narrowed."
        )


class TestGroundedSurfaceReadsAsOneSentence:
    def test_the_live_defect_is_gone(self):
        grounded = _ground_live_voice_surface(
            "Forgetting is a mercy. It's not about erasing history.",
            _memory_contract(),
        )
        assert grounded.startswith("From my conversation memory,"), (
            "the grounding marker itself is a contract other tests pin"
        )
        assert "memory, Forgetting" not in grounded, (
            "a capital mid-sentence after the grounding comma is the bug"
        )
        assert "memory, forgetting is a mercy" in grounded

    def test_a_name_after_the_comma_is_left_alone(self):
        grounded = _ground_live_voice_surface(
            "Bryan asked me to keep that number.", _memory_contract()
        )
        assert "memory, Bryan asked" in grounded, (
            "conservative case: never lower-case a person's name"
        )
