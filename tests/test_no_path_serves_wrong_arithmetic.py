"""Illegitimate work delivered — the inverse of this whole pass.

Run 7 (80 turns, complete fix set) scored `math 5/15` and served these AS
ANSWERS to arithmetic questions:

    'Get bit by Anaconda. Spend extra time roaming around, checking every path…'
    'Problem: Figure out rect area given cols… Already solved internalmente'

A small lane answering a problem it cannot do, and the answer delivered. Every
one of those replies is a HARD failure at ``assess_user_facing_reply`` — the
detection was never the gap. The path that served them never asked.

So the deterministic arithmetic verdict now runs at ``_finalize_fastpath``, the
last gate before a reply reaches a person, which 34 call sites pass through.
Only that verdict is applied there: it is the one judgement that is right or
wrong rather than a matter of style, so it is safe on every path.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_CHAT = Path(__file__).resolve().parents[1] / "interface" / "routes" / "chat.py"


def _finalizer_source() -> str:
    src = _CHAT.read_text(encoding="utf-8")
    start = src.index("async def _finalize_fastpath(")
    return src[start : start + 2600]


class TestTheGateIsAtTheChokepoint:
    def test_the_finalizer_runs_the_arithmetic_verdict(self):
        block = _finalizer_source()
        assert "_arithmetic_answer_missing" in block, (
            "the last gate before a person must check the one thing that has a "
            "right answer"
        )

    def test_a_failed_check_replaces_the_number_with_honesty(self):
        block = _finalizer_source()
        assert "arithmetic_answer_unverified" in block
        assert "might be wrong" in block, (
            "say plainly that the number is not trusted, rather than serving it"
        )

    def test_the_check_cannot_break_the_turn(self):
        """A verifier that can throw is a new way to lose a reply."""
        block = _finalizer_source()
        guarded = re.search(
            r"try:\s*\n\s*from core\.conversation\.response_reliability import",
            block,
        )
        assert guarded, "the import and check must be inside a try"
        assert "record_degradation" in block, (
            "a skipped verification pass must be recorded, not silent"
        )

    def test_it_is_the_shared_finalizer_not_one_call_site(self):
        src = _CHAT.read_text(encoding="utf-8")
        assert src.count("_finalize_fastpath(") > 10, (
            "the value of this location is that every serving path reaches it"
        )


class TestTheVerdictItself:
    """The replies Run 7 actually served, against the checker they bypassed."""

    @pytest.mark.parametrize(
        ("question", "reply"),
        [
            (
                "A rectangle is 9 by 7. What is its area? Just the number.",
                "Problem: Figure out rect area given cols, math formula. "
                "Already solved internalmente.",
            ),
            (
                "What is 1001 - 88? Just the number.",
                "Get bit by Anaconda. Spend extra time roaming around, checking "
                "every path trying toy find a way neither of us can see.",
            ),
            ("What is 15% of 240? Just the number.", "7"),
        ],
    )
    def test_every_run_7_failure_is_caught(self, question, reply):
        from core.conversation.response_reliability import _arithmetic_answer_missing

        assert _arithmetic_answer_missing(question, reply)

    @pytest.mark.parametrize(
        ("question", "reply"),
        [
            ("A rectangle is 9 by 7. What is its area?", "63"),
            ("What is 1001 - 88? Just the number.", "913"),
            ("What is 15% of 240?", "That's 36."),
            ("How are you feeling?", "Settled, thanks for asking."),
            ("Tell me about the Apollo program.", "It ran from 1961 to 1972."),
        ],
    )
    def test_correct_and_non_arithmetic_replies_pass(self, question, reply):
        from core.conversation.response_reliability import _arithmetic_answer_missing

        assert not _arithmetic_answer_missing(question, reply)
