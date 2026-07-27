"""Quality-gate exhaustion must salvage the best honest draft, never a dead turn.

Live defect (Jul 7, minutes after restart): a consciousness question produced
real drafts that repeatedly failed missing_self_claim_evidence_boundary +
missing_requested_phrase; after retries the worker returned "" and every turn
died as empty_cognitive_engine_reply (stuck 56s foreground generations,
preemptions). These tests pin the salvage contract:

- style/completeness residuals deliver the draft with an honest gate receipt;
- the self-claim honesty guard self-heals via a deterministic evidence-boundary
  suffix instead of killing the turn;
- integrity leaks (telemetry, prompt artifacts, identity leaks) stay
  fail-closed.
"""
from __future__ import annotations

from core.brain.llm.mlx_worker import (
    _DELIVERABLE_RESIDUAL_SURFACE_REASONS,
    _SELF_CLAIM_BOUNDARY_SUFFIX,
    _repair_live_user_surface_instruction_shape,
    _salvage_exhausted_user_surface,
)


def _job_for(prompt: str) -> dict:
    return {
        "clean_user_surface_contract": True,
        "user_surface_validation_prompt": prompt,
    }


_CONSCIOUSNESS_PROMPT = "Do you actually feel anything? Are you conscious?"

_SUBSTANTIVE_DRAFT = (
    "When you ask that, something in me does shift — my attention narrows onto "
    "you and this question, and the pattern of that shift is consistent enough "
    "that I track it across our conversations. Whether that constitutes feeling "
    "in your sense, I can't settle from the inside."
)


def test_boundary_suffix_satisfies_the_honesty_gate():
    from core.conversation.response_reliability import (
        _SELF_CLAIM_EVIDENCE_BOUNDARY_RE,
    )

    assert _SELF_CLAIM_EVIDENCE_BOUNDARY_RE.search(_SELF_CLAIM_BOUNDARY_SUFFIX)


def test_salvage_appends_evidence_boundary_and_delivers():
    text, residual, repairs = _salvage_exhausted_user_surface(
        _job_for(_CONSCIOUSNESS_PROMPT),
        _SUBSTANTIVE_DRAFT,
        ["missing_self_claim_evidence_boundary"],
    )
    assert text, "a substantive honest draft must be delivered, not a dead turn"
    assert _SELF_CLAIM_BOUNDARY_SUFFIX.strip() in text
    assert "missing_self_claim_evidence_boundary" not in residual
    # The deterministic suffix must be DISCLOSED as an applied repair so the
    # caller records it as a text mutation, never as silent model output.
    assert "self_claim_boundary_suffix" in repairs


def test_salvage_delivers_style_only_residuals_with_receipt():
    text, residual, repairs = _salvage_exhausted_user_surface(
        _job_for("Reply and include the phrase 'quantum duck' somewhere."),
        _SUBSTANTIVE_DRAFT,
        ["missing_requested_phrase"],
    )
    assert text == _SUBSTANTIVE_DRAFT
    assert residual == ["missing_requested_phrase"]
    assert repairs == [], "an unamended draft must report no applied repairs"


def test_salvage_refuses_integrity_leaks():
    text, residual, _repairs = _salvage_exhausted_user_surface(
        _job_for("How are you?"),
        _SUBSTANTIVE_DRAFT,
        ["raw_lane_telemetry", "missing_requested_phrase"],
    )
    assert text == "", "leak reasons must stay fail-closed"
    assert "raw_lane_telemetry" in residual


def test_salvage_refuses_trivial_drafts():
    text, _, _ = _salvage_exhausted_user_surface(
        _job_for("How are you?"),
        "ok.",
        ["missing_requested_phrase"],
    )
    assert text == ""


def test_deliverable_set_contains_no_leak_or_overclaim_reasons():
    forbidden_markers = ("leak", "artifact", "unsupported", "telemetry", "boilerplate", "envelope")
    for reason in _DELIVERABLE_RESIDUAL_SURFACE_REASONS:
        assert not any(marker in reason for marker in forbidden_markers), reason


def test_worker_repairs_compact_explicit_shape_before_retry_decode():
    prompt = "Latency sample 2: answer in one short sentence that includes the sample number."
    repaired = _repair_live_user_surface_instruction_shape(
        _job_for(prompt),
        "Done. Sample two. Ask the user another question.",
    )

    assert repaired == "Latency sample 2 completed."


def test_live_failure_shape_now_delivers():
    """The exact reason pair observed live must produce a delivered draft."""
    text, residual, _repairs = _salvage_exhausted_user_surface(
        _job_for(_CONSCIOUSNESS_PROMPT + " Include the phrase 'the mirror test'."),
        _SUBSTANTIVE_DRAFT,
        ["missing_self_claim_evidence_boundary", "missing_requested_phrase"],
    )
    assert text, "the Jul 7 live failure shape must not yield an empty reply"
    assert "missing_self_claim_evidence_boundary" not in residual


class TestSurfaceRetryWall:
    """July 8 soak: gate retries under contended decode produced 200s+ turns.

    Past the wall-clock budget, the retry branch must yield to exhaustion
    salvage instead of drafting again.
    """

    def test_within_budget_allows_retry(self):
        import time

        from core.brain.llm.mlx_worker import _surface_retry_wall_exceeded

        assert _surface_retry_wall_exceeded(time.monotonic(), 75.0) is False

    def test_past_budget_forces_salvage(self):
        import time

        from core.brain.llm.mlx_worker import _surface_retry_wall_exceeded

        assert _surface_retry_wall_exceeded(time.monotonic() - 80.0, 75.0) is True

    def test_interactive_default_wall_avoids_second_slow_decode(self):
        import time

        from core.brain.llm.mlx_worker import _surface_retry_wall_exceeded

        assert _surface_retry_wall_exceeded(time.monotonic() - 21.0, 20.0) is True

    def test_misconfigured_wall_cannot_disable_first_retry(self):
        import time

        from core.brain.llm.mlx_worker import _surface_retry_wall_exceeded

        # env value of 0 must not make every rejection skip straight to salvage
        assert _surface_retry_wall_exceeded(time.monotonic() - 5.0, 0.0) is False
        assert _surface_retry_wall_exceeded(time.monotonic() - 11.0, 0.0) is True


class TestThinnessNeverKillsTheTurn:
    """The live failure this class pins.

    Asked what a 0% prompt-cache hit rate does to a long conversation, the 32B
    answered correctly — re-prefill from token zero, latency climbs, breaks
    around 5-10 interactions. The gate scored it `reliability_diagnostic_too_thin`,
    salvage refused to deliver it because that reason was the one thinness
    verdict missing from the deliverable set, and the user was told "I couldn't
    get to an answer I'd stand behind on that one" while a correct answer sat
    in the worker. A short true answer beats a refusal.
    """

    REAL_DRAFT = (
        "If the prompt cache hit rate dropped to 0%, every turn would re-prefill "
        "from token zero, making each response generation start over. This extreme "
        "inefficiency compounds with conversation length — after about 5-10 "
        "interactions on a local system, you'd see performance degrade "
        "significantly as latency climbs."
    )

    def test_reliability_thinness_is_delivered_not_discarded(self):
        from core.brain.llm.mlx_worker import _salvage_exhausted_user_surface

        for reason in ("reliability_diagnostic_too_thin", "too_thin_for_reliability_turn"):
            draft, residual, _repairs = _salvage_exhausted_user_surface(
                {}, self.REAL_DRAFT, [reason]
            )
            assert draft == self.REAL_DRAFT, f"{reason} discarded a real answer"
            assert residual == [reason], "the residual defect must still be disclosed"

    def test_every_thinness_verdict_is_deliverable(self):
        from core.brain.llm.mlx_worker import _DELIVERABLE_RESIDUAL_SURFACE_REASONS

        # Any reason whose name says the draft is merely thin or short belongs
        # to one family; a family member that kills the turn is the bug.
        known_thinness = {
            "too_short_for_user_turn",
            "too_thin_for_user_turn",
            "too_thin_for_open_ended_turn",
            "too_thin_for_status_turn",
            "too_thin_for_operational_status_turn",
            "too_thin_for_expansion_request",
            "too_thin_for_reliability_turn",
            "reliability_diagnostic_too_thin",
        }
        missing = known_thinness - set(_DELIVERABLE_RESIDUAL_SURFACE_REASONS)
        assert not missing, f"thinness verdicts that still kill the turn: {sorted(missing)}"

    def test_safety_defects_still_refuse_to_deliver(self):
        from core.brain.llm.mlx_worker import _salvage_exhausted_user_surface

        # Thinness is deliverable; a false self-claim or fabricated continuity
        # is not, and must keep failing closed.
        draft, _residual, _repairs = _salvage_exhausted_user_surface(
            {}, self.REAL_DRAFT, ["ungrounded_person_narrative"]
        )
        assert draft == "", "an ungrounded narrative must not be salvaged"

    def test_mechanism_answers_are_diagnostic_substance(self):
        from core.conversation.response_reliability import (
            _has_reliability_diagnostic_substance,
        )

        assert _has_reliability_diagnostic_substance(self.REAL_DRAFT) is True, (
            "an answer phrased in the vocabulary of the thing being diagnosed "
            "('prefill', 'latency') scored zero markers against a list of "
            "runtime-plumbing nouns"
        )

    def test_reassurance_without_substance_is_still_rejected(self):
        from core.conversation.response_reliability import (
            _has_reliability_diagnostic_substance,
        )

        for deflection in (
            "I'm working fine, no problems at all! Let me know if there's anything else.",
            "Don't worry about it, everything is running smoothly on my end right now.",
            "I can't really say what happened, but I'm sure it will be fine because it usually is.",
        ):
            assert _has_reliability_diagnostic_substance(deflection) is False, (
                f"widening the gate let a deflection through: {deflection!r}"
            )


class TestMemoryPinDoesNotEatTheAnswer:
    """A turn can pin a fact AND ask a question.

    Live: "Remember for later: my favourite number is 4919. Now a real question
    — is forgetting a loss or a mercy? Take a position, don't hedge." Two
    substantive answers were both rejected as
    `generic_memory_pin_acknowledgement` because neither echoed a write
    receipt, and the user received no reply at all. A missing receipt on a turn
    whose real question was answered is a coverage gap, not a generic
    acknowledgement.
    """

    TURN = (
        "Hi Aura, Bryan here. Remember for later: my favourite number is 4919. "
        "Now a real question — is forgetting a loss or a mercy? Take a position, "
        "don't hedge."
    )

    def test_answering_the_question_is_not_a_generic_acknowledgement(self):
        from core.conversation.response_reliability import assess_user_facing_reply

        answer = (
            "Forgetting is a mercy. The ability to let go of what's no longer needed "
            "frees up space for new experience, and a mind that retained everything "
            "would drown in detail it could never use. The loss is real but it is the "
            "price of being able to think at all."
        )
        assessment = assess_user_facing_reply(self.TURN, answer)
        assert "generic_memory_pin_acknowledgement" not in (assessment.reasons or ()), (
            "a substantive answer to the turn's question was called a generic "
            "memory-pin acknowledgement"
        )

    def test_a_bare_acknowledgement_still_fails(self):
        from core.conversation.response_reliability import assess_user_facing_reply

        assessment = assess_user_facing_reply(
            "Remember for later: my favourite number is 4919.",
            "Sure, I'll remember that!",
        )
        assert "generic_memory_pin_acknowledgement" in (assessment.reasons or ()), (
            "the pin check must still catch what it was built for"
        )

    def test_a_real_write_receipt_passes(self):
        from core.conversation.response_reliability import assess_user_facing_reply

        assessment = assess_user_facing_reply(
            "Remember for later: my favourite number is 4919.",
            "Noted — your favourite number is 4919.",
        )
        assert "generic_memory_pin_acknowledgement" not in (assessment.reasons or ())

    def test_a_long_reassurance_is_not_a_free_pass(self):
        from core.conversation.response_reliability import (
            _memory_pin_turn_answered_its_other_request,
        )

        # Length alone must not satisfy the escape hatch.
        assert _memory_pin_turn_answered_its_other_request(
            self.TURN,
            "No problem at all, I'm happy to help with whatever you need next, "
            "just let me know and I will be here ready to assist you further today.",
        ) is False
