"""A grounding prefix must not break the sentence it introduces.

`_ground_live_voice_surface` prepends a clause ending in ", " and then used to
UPPERCASE the body's first letter — the opposite of what a continuation needs.
Live on the desktop surface: "From my conversation memory, Forgetting is a
mercy." The same mechanism produced the July recall defect's
"From my conversation memory, The code you gave me earlier was ...".
"""
from __future__ import annotations

from types import SimpleNamespace

from core.phases import dialogue_policy
from core.phases.dialogue_policy import (
    _lowercase_continuation_start,
    repair_dialogue_surface,
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


class TestTheGroundingPrefixIsNotSynthesisedAtAll:
    """The mechanism this file was written against was removed, deliberately.

    `_ground_live_voice_surface` glued "From my conversation memory, " onto the
    front of a reply to satisfy a contract that actually requires a first-person
    STANCE. It asserted retrieval exactly where retrieval was weakest — the flag
    it keyed on is raised when evidence is THIN — and it ran before the retry,
    so a draft that failed the contract was cosmetically patched instead of
    regenerated.

    This file kept importing the deleted symbol, so it failed to COLLECT: the
    grammar tests below stopped running, and every other test sharing its chunk
    went with it. The import is now the module, and the class that tested the
    removed behaviour tests the removal instead — the decision is a contract,
    so it gets a guard rather than a deletion.
    """

    def test_the_symbol_is_gone(self):
        assert not hasattr(dialogue_policy, "_ground_live_voice_surface"), (
            "reintroducing this puts provenance back into her voice, where a "
            "reader cannot check it, and re-skips the regeneration retry"
        )

    def test_the_surface_repair_never_prepends_a_provenance_clause(self):
        for draft in (
            "Forgetting is a mercy. It's not about erasing history.",
            "Bryan asked me to keep that number.",
            "A room with walls made of memory.",
        ):
            repaired = repair_dialogue_surface(draft, _memory_contract())
            lowered = repaired.lower()
            assert not lowered.startswith("from my conversation memory"), repaired
            assert not lowered.startswith("from my live runtime state"), repaired

    def test_a_thin_evidence_contract_does_not_manufacture_grounding(self):
        """The flag means evidence is THIN; the surface must not claim retrieval."""
        draft = "The code you gave me earlier was 7213."
        repaired = repair_dialogue_surface(draft, _memory_contract())
        assert "conversation memory," not in repaired.lower(), repaired
