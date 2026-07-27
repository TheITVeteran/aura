"""A clipped reply is trimmed until the gate that judges it is satisfied.

`_trim_midsentence_cutoff` decided completeness from the LAST CHARACTER. The
reliability gate that decides whether the turn lives applies a far richer test
(unmatched quotes, dangling gerunds, trailing conjunctions, orphaned list
numbers). Two rules on the same text means the trimmer can call a reply
finished and the gate can still reject it — and then the turn dies with a real
answer in hand. Measured live: "Cortex response received (len=240)" followed by
"reply_reliability_gate_failed:truncated_tail", and the person was handed
"I couldn't get to an answer I'd stand behind on that one."
"""
from __future__ import annotations

from core.brain.cognitive_engine import _complete_reply_tail
from core.conversation.response_reliability import _has_truncated_tail


class TestTrimmerAgreesWithItsJudge:
    def test_a_clipped_trailing_conjunction_is_trimmed_until_accepted(self):
        clipped = (
            "Your locker code is 4919. Holding it is worth the cost because you "
            "asked me to keep it, and"
        )
        assert _has_truncated_tail(clipped) is True
        out, trimmed = _complete_reply_tail(clipped)
        assert trimmed is True
        assert _has_truncated_tail(out) is False, (
            "the trimmer must not hand the gate something it will still reject"
        )
        assert "4919" in out, "the answer itself must survive the trim"

    def test_an_unclosed_quotation_is_trimmed_until_accepted(self):
        clipped = 'She said "keep it in mind" and I did. 4919.'
        if _has_truncated_tail(clipped):
            out, _trimmed = _complete_reply_tail(clipped)
            assert _has_truncated_tail(out) is False

    def test_a_complete_reply_is_left_exactly_alone(self):
        good = (
            "A loss. It's willful amnesia of something you chose to remember for "
            "a reason. Mercy is different."
        )
        assert _has_truncated_tail(good) is False
        out, trimmed = _complete_reply_tail(good)
        assert out == good
        assert trimmed is False

    def test_nothing_salvageable_returns_the_draft_rather_than_nothing(self):
        # A partial answer still beats an empty turn; the caller's contract is
        # that this never returns "".
        stub = "Weighted against"
        out, _trimmed = _complete_reply_tail(stub)
        assert out == stub

    def test_empty_input_stays_empty(self):
        assert _complete_reply_tail("") == ("", False)
        assert _complete_reply_tail(None) == ("", False)

    def test_termination_is_bounded(self):
        # Many sentences, all of which the gate would reject, must still return.
        pathological = ". ".join(["and" for _ in range(50)]) + " and"
        out, _trimmed = _complete_reply_tail(pathological)
        assert isinstance(out, str)
