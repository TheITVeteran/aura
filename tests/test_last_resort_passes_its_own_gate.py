"""The same sentence is a defect from one author and the truth from another.

Live 2026-08-04, two turns shipped with ``assessment=runtime_boilerplate``
recorded against text the runtime had written itself:

    "I couldn't get a clear enough answer together, and I'd rather say that
     than hand you something thin. I understood you to be asking about physics
     and teach. Ask me again and I should have it."

``_BROKEN_LANE_BOILERPLATE_RE`` matched "ask me again". That detector is right
about what it was built for — the model narrating the runtime *instead of*
answering, while an answer was there to give. It was wrong here, and the wrong
part was never the words.

``_build_degraded_live_reply`` runs only after generation, every recovery, and
the verified-floor lookup have all come back empty. When it says no answer
exists, it is the one author in the system that has already proven it. The gate
had no way to know that, because it never asked where the text came from.

So the fix is provenance, not vocabulary. Nothing Aura can say is restricted;
what changed is that a reason which is an inference about a CHOICE is not drawn
when there was no choice. These tests hold that line in both directions: the
exemption must apply to the composer, and must NOT apply to a model draft
saying the identical thing.
"""

from __future__ import annotations

import pytest

from core.conversation.reply_provenance import (
    ReplyProvenance,
    admission_defects,
    excused_reasons,
    forget_declared_provenance,
)

# The exact text that shipped.
SHIPPED_LIVE = (
    "I couldn't get a clear enough answer together, and I'd rather say that "
    "than hand you something thin. I understood you to be asking about "
    "physics and teach. Ask me again and I should have it."
)
TURN = "I've always wanted to teach myself physics and get really good at it."

REASONS = (
    "degraded_turn",
    "repeated_reflex",
    "desktop_cognitive_engine_repair_failed",
)
FRAMES = (
    {"attention_focus": "", "mood": "steady"},
    {"attention_focus": "the live thread", "mood": "pressed"},
)
TURNS = (
    TURN,
    "Do you ever get tired of being asked how you are?",
    "What did you make of that argument?",
    "",
)


@pytest.fixture(autouse=True)
def _clean_provenance():
    forget_declared_provenance()
    yield
    forget_declared_provenance()


class TestTheSameWordsFromDifferentAuthors:
    def test_a_model_draft_saying_it_is_still_a_defect(self):
        """The detector keeps its teeth. Nothing here softens it."""
        from core.conversation.response_reliability import assess_user_facing_reply

        assessment = assess_user_facing_reply(TURN, SHIPPED_LIVE)
        reasons = [str(item) for item in (assessment.reasons or ())]
        assert "runtime_boilerplate" in reasons

    def test_the_composer_saying_it_is_not(self):
        from core.conversation.response_reliability import assess_user_facing_reply

        assessment = assess_user_facing_reply(
            TURN, SHIPPED_LIVE, provenance=ReplyProvenance.HONEST_FAILURE.value
        )
        reasons = [str(item) for item in (assessment.reasons or ())]
        assert "runtime_boilerplate" not in reasons, (
            "an admission that no answer exists, from the author that proved it, "
            "is not the model narrating the runtime in place of an answer"
        )


class TestProvenanceExcusesOneThingOnly:
    def test_leaks_are_never_excused(self):
        excused = excused_reasons(ReplyProvenance.HONEST_FAILURE)
        for never in (
            "internal_task_prompt_leak",
            "raw_lane_telemetry",
            "prompt_artifact",
            "corrupted_language",
            "raw_model_identity_leak",
            "unsupported_embodiment_claim",
        ):
            assert never not in excused

    def test_an_unknown_provenance_excuses_nothing(self):
        assert excused_reasons("something_made_up") == frozenset()
        assert excused_reasons(None) == frozenset()
        assert excused_reasons(ReplyProvenance.MODEL_DRAFT) == frozenset()

    def test_an_admission_may_not_claim_an_answer_it_does_not_have(self):
        check = admission_defects(
            "what is the 5000th prime",
            "I couldn't get there. I understood you to be asking about primes. "
            "The answer is 48611.",
        )
        assert check.ok is False
        assert "admission_claims_an_answer" in check.defects

    def test_an_admission_has_to_show_what_it_understood(self):
        check = admission_defects("what is the 5000th prime", "I couldn't get there.")
        assert check.ok is False
        assert "admission_names_nothing_understood" in check.defects

    def test_the_live_admission_is_a_real_admission(self):
        assert admission_defects(TURN, SHIPPED_LIVE).ok is True


class TestTheDeclarationTravelsWithTheText:
    """Gates a dozen frames away take a string and nothing else."""

    def test_a_declared_reply_is_judged_by_what_it_is(self):
        from core.conversation.reply_provenance import declare_provenance
        from core.conversation.response_reliability import assess_user_facing_reply

        declare_provenance(SHIPPED_LIVE, ReplyProvenance.HONEST_FAILURE)
        assessment = assess_user_facing_reply(TURN, SHIPPED_LIVE)
        reasons = [str(item) for item in (assessment.reasons or ())]
        assert "runtime_boilerplate" not in reasons

    def test_the_live_gate_that_recorded_the_defect_now_agrees(self):
        """``_looks_semantically_glitched`` is the frame that produced the
        recorded ``assessment=runtime_boilerplate`` on both live turns."""
        from core.conversation.reply_provenance import declare_provenance
        from interface.routes.chat import _looks_semantically_glitched

        glitched, reason = _looks_semantically_glitched(TURN, SHIPPED_LIVE)
        assert (glitched, reason) == (True, "runtime_boilerplate")

        declare_provenance(SHIPPED_LIVE, ReplyProvenance.HONEST_FAILURE)
        glitched, reason = _looks_semantically_glitched(TURN, SHIPPED_LIVE)
        assert glitched is False, f"still rejected as {reason!r}"

    def test_undeclared_text_is_unaffected(self):
        from core.conversation.response_reliability import assess_user_facing_reply

        assessment = assess_user_facing_reply(TURN, SHIPPED_LIVE)
        assert "runtime_boilerplate" in [str(r) for r in (assessment.reasons or ())]


@pytest.mark.parametrize("reason", REASONS)
@pytest.mark.parametrize("frame", FRAMES)
@pytest.mark.parametrize("turn", TURNS)
def test_the_composer_declares_every_branch_it_writes(reason, frame, turn):
    """Whatever wording a branch chooses, the gate knows what it is."""
    from core.conversation.response_reliability import assess_user_facing_reply
    from interface.routes.chat import _build_degraded_live_reply

    composed = _build_degraded_live_reply(dict(frame), turn, reason=reason)
    assert composed.strip(), "the last resort must still say something"

    assessment = assess_user_facing_reply(turn, composed)
    reasons = [str(item) for item in (assessment.reasons or ())]
    assert not reasons, (
        f"the degraded-turn composer emitted text its own gate still rejects "
        f"({reasons}); there is no further fallback, so this ships as-is"
    )
