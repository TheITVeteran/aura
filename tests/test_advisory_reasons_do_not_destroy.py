"""An advisory objection describes a reply. It must never delete one.

ADVISORY_REASONS has said so in its docstring since it was introduced —
"describe a turn but never destroy it" — but disposition_for did not know
the set existed, so every advisory reason came back REPAIR. At the
conversation-learning gate REPAIR means "do not remember this exchange",
and the exchange was dropped.

The reply that triggered it was correct. "My laptop crashed when Aura used
over 100GB of RAM." answered with "I will treat that as a live desktop
reliability fault and preserve the context." shares no content word with
the question, because it paraphrases instead of echoing — which is what a
good reply does. Overlap zero, verdict reply_abandons_thread, memory gone.
"""

from __future__ import annotations

from core.conversation.surface_disposition import (
    ADVISORY_ONLY_REASONS,
    SurfaceDisposition,
    disposition_for,
    draft_is_servable,
)
from core.conversation.thread_continuity import assess_thread_continuity


class TestAdvisoryReasonsServe:
    def test_an_advisory_reason_alone_serves(self) -> None:
        assert disposition_for(("reply_abandons_thread",)) is SurfaceDisposition.SERVE

    def test_every_declared_advisory_reason_serves_alone(self) -> None:
        for reason in ADVISORY_ONLY_REASONS:
            assert disposition_for((reason,)) is SurfaceDisposition.SERVE, reason

    def test_an_advisory_reason_beside_a_real_one_still_repairs(self) -> None:
        """Advisory does not launder a genuine defect sitting next to it."""
        assert (
            disposition_for(("reply_abandons_thread", "truncated_tail"))
            is SurfaceDisposition.REPAIR
        )

    def test_unspeakable_still_discards(self) -> None:
        assert disposition_for(("empty_reply",)) is SurfaceDisposition.DISCARD

    def test_advisory_drafts_remain_servable(self) -> None:
        assert draft_is_servable(("reply_abandons_thread",))

    def test_the_two_declarations_are_one(self) -> None:
        """response_reliability must not carry a second, drifting copy."""
        from core.conversation.response_reliability import ADVISORY_REASONS

        assert ADVISORY_REASONS is ADVISORY_ONLY_REASONS


class TestTheParaphraseThatTriggeredIt:
    USER = "My laptop crashed when Aura used over 100GB of RAM."
    REPLY = (
        "I will treat that as a live desktop reliability fault and "
        "preserve the context."
    )

    def test_the_overlap_heuristic_still_flags_it(self) -> None:
        """The heuristic is not wrong to notice — it is wrong to be obeyed."""
        verdict = assess_thread_continuity(self.REPLY, self.USER)
        assert verdict.abandoned
        assert verdict.overlap_with_turn == 0.0

    def test_but_the_exchange_is_still_recorded(self) -> None:
        verdict = assess_thread_continuity(self.REPLY, self.USER)
        assert disposition_for((verdict.reason,)) is SurfaceDisposition.SERVE
