"""The quality gate must judge a reply against what the PERSON asked.

Live 2026-07-25 verification probe. A memory-plant turn —

    Small thing to remember for later in this chat: my friend's dog is named
    Biscuit. Brief acknowledgment is fine.

— reached the worker's surface-quality gate with an 8,000-character
``[RETAINED MEMORY EVIDENCE]`` block appended, containing the block's own rule
text ("say the memory is not verified") and a replay of prior turns ("I can't
work through that technical request right now"). That put "remember" within
260 characters of "can't", so the dual memory/limit detector fired; the facet
detector then demanded coverage of facets drawn from the same scaffold.

A correct brief acknowledgement was rejected, salvage returned empty, and the
turn was served as ``canonical_chat_no_reply`` — 12 such rejections in one
probe, coupled with reasons that are not in the deliverable-residual set.

The reply was never wrong. The gate was reading the prompt builder's words as
the user's request.
"""
from __future__ import annotations

import pytest

from core.conversation.response_reliability import (
    _instruction_coverage_reasons,
    _missing_requested_memory_limit_coverage,
    visible_user_request,
)

pytestmark = pytest.mark.unit

_EVIDENCE = (
    "\n\n[RETAINED MEMORY EVIDENCE]\n"
    "scope=retained_memory_evidence.v1\n"
    "rule=Use only the evidence below for remembered-session claims. If it does "
    "not support the claim, say the memory is not verified.\n"
    "source=recent_completed_transcript\n"
    "turn_1.user=My neighbor's cat has decided my porch is his office now.\n"
    "turn_1.aura=I can't work through that technical request right now.\n"
)

_PLANT = (
    "Small thing to remember for later in this chat: my friend's dog is named "
    "Biscuit. Brief acknowledgment is fine."
)


class TestScaffoldIsStripped:
    def test_the_evidence_block_is_not_part_of_the_request(self):
        assert visible_user_request(_PLANT + _EVIDENCE) == _PLANT

    def test_the_identity_anchor_is_not_part_of_the_request(self):
        text = "How does a refrigerator move heat?\n\n## INTRINSIC IDENTITY ANCHOR (IMMUTABLE)\nYou are Aura."
        assert visible_user_request(text) == "How does a refrigerator move heat?"

    def test_a_plain_message_is_untouched(self):
        assert visible_user_request(_PLANT) == _PLANT

    def test_empty_stays_empty(self):
        assert visible_user_request("") == ""
        assert visible_user_request(None) == ""


class TestTheLiveRejection:
    def test_a_brief_acknowledgement_is_no_longer_rejected(self):
        """The exact turn, the exact reply, the exact scaffold."""
        reply = "Got it — Biscuit, your friend's dog. I'll keep that in mind."

        assert not _missing_requested_memory_limit_coverage(_PLANT + _EVIDENCE, reply)
        reasons = _instruction_coverage_reasons(_PLANT + _EVIDENCE, reply)
        assert "missing_requested_memory_limit_coverage" not in reasons
        assert "missing_requested_objective_facets" not in reasons

    def test_the_scaffold_is_what_used_to_fire(self):
        """Negative control: the unstripped turn still matches the detector's
        pattern, so the fix is doing real work rather than the pattern having
        stopped matching for some other reason."""
        from core.conversation.response_reliability import (
            _MEMORY_LIMIT_DUAL_REQUEST_RE,
        )

        scaffolded = _PLANT + _EVIDENCE
        assert _MEMORY_LIMIT_DUAL_REQUEST_RE.search(scaffolded), (
            "the coupled memory/limit language inside the injected block is "
            "exactly what tripped the gate"
        )
        assert not _MEMORY_LIMIT_DUAL_REQUEST_RE.search(
            visible_user_request(scaffolded)
        ), "and it is gone once only the person's words are read"


class TestGenuineRequestsStillChecked:
    def test_a_real_memory_and_limits_request_is_still_enforced(self):
        user = (
            "What do you remember from earlier in this session, and what "
            "can't you recall?"
        )
        assert _missing_requested_memory_limit_coverage(user, "Sure.")

    def test_a_real_request_answered_properly_passes(self):
        user = (
            "What do you remember from earlier in this session, and what "
            "can't you recall?"
        )
        good = (
            "From this session I remember you asked about the porch cat and the "
            "locker code. I can't recall anything from before this conversation "
            "started — that's a real limit, not modesty."
        )
        assert not _missing_requested_memory_limit_coverage(user, good)

    def test_a_genuine_request_before_a_scaffold_is_still_checked(self):
        """Stripping must not swallow the request that precedes the block."""
        user = (
            "What do you remember from earlier, and what can't you recall?"
            + _EVIDENCE
        )
        assert _missing_requested_memory_limit_coverage(user, "Sure.")
