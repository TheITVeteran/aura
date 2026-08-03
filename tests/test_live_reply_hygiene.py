"""Contracts for the live-conversation reply hygiene fixes.

Root incident (2026-07-19 live chat): user-facing desktop replies were cut
off mid-sentence ("... Weighted against") because four advisory sampling
frames compounded their max_tokens factors multiplicatively, and stored
endurance-probe turns ("(turn 169) In a few sentences, how does a
refrigerator move heat?") resurfaced from long-term memory into live chats,
producing visible thread-jumping drift.
"""
from __future__ import annotations

import pytest

from core.brain.cognitive_engine import (
    _combine_advisory_token_factors,
    _trim_midsentence_cutoff,
)
from core.memory.memory_facade import _probe_harness_reason


class TestAdvisoryTokenFactorCombination:
    def test_no_factors_is_identity(self):
        assert _combine_advisory_token_factors([]) == 1.0

    def test_single_reduction_applies(self):
        assert _combine_advisory_token_factors([0.8]) == 0.8

    def test_multiple_reductions_do_not_compound(self):
        # 0.85 * 0.8 * 0.75 * 0.8 ≈ 0.41 was the incident behavior; the
        # combined factor must be the strongest single reduction instead.
        assert _combine_advisory_token_factors([0.85, 0.8, 0.75, 0.8]) == 0.75

    def test_boost_applies_only_without_reductions(self):
        assert _combine_advisory_token_factors([1.1, 1.25]) == 1.25
        assert _combine_advisory_token_factors([1.25, 0.9]) == 0.9

    def test_incident_budget_stays_above_floor(self):
        # 768 base with the incident's factor set must stay a full-sentence
        # budget, not collapse toward ~250.
        combined = _combine_advisory_token_factors([0.85, 0.8, 0.75, 0.8])
        assert int(768 * combined) >= 512


class TestMidsentenceCutoffTrim:
    def test_complete_reply_untouched(self):
        text = "I checked the logs. Everything looks stable now."
        trimmed, changed = _trim_midsentence_cutoff(text)
        assert trimmed == text
        assert changed is False

    def test_dangling_fragment_trimmed_to_last_sentence(self):
        text = (
            "Intuition pumps and thought experiments are tools that nudge me "
            "out of default patterns. They push toward something more "
            "reflective or critical. Weighted against"
        )
        trimmed, changed = _trim_midsentence_cutoff(text)
        assert changed is True
        assert trimmed.endswith("reflective or critical.")
        assert "Weighted against" not in trimmed

    def test_no_boundary_keeps_partial_text(self):
        text = "a partial fragment with no sentence boundary at all"
        trimmed, changed = _trim_midsentence_cutoff(text)
        assert trimmed == text
        assert changed is False

    def test_early_boundary_salvages_complete_answer(self):
        # A complete answer is preferable to losing the entire turn because a
        # much longer unfinished tail reached the token budget.
        text = "Short opener. " + "then a very long unfinished clause " * 5
        trimmed, changed = _trim_midsentence_cutoff(text)
        assert trimmed == "Short opener."
        assert changed is True

    def test_empty_and_whitespace_safe(self):
        assert _trim_midsentence_cutoff("") == ("", False)
        assert _trim_midsentence_cutoff("   ") == ("", False)

    def test_code_fence_end_is_terminal(self):
        text = "Here is the fix:\n```python\nprint('ok')\n```"
        trimmed, changed = _trim_midsentence_cutoff(text)
        assert trimmed == text
        assert changed is False


class TestProbeHarnessMemoryHygiene:
    def test_harness_turn_pattern_detected(self):
        reason = _probe_harness_reason(
            "(turn 169) In a few sentences, how does a refrigerator move heat?",
            {},
        )
        assert reason == "harness_turn_pattern"

    def test_probe_session_ids_detected(self):
        for session in ("endurance-20260719-2140", "soak_run_4", "probe-a", "bench-x"):
            assert _probe_harness_reason("hello", {"session_id": session}), session

    def test_explicit_ephemeral_flag_detected(self):
        assert (
            _probe_harness_reason("hello", {"ephemeral_probe_session": True})
            == "ephemeral_probe_session"
        )

    def test_ordinary_conversation_passes(self):
        assert _probe_harness_reason("Bryan asked about the boot failure.", {}) == ""
        assert (
            _probe_harness_reason("hello", {"session_id": "desktop-main-session"}) == ""
        )

    def test_turn_pattern_only_matches_harness_shape(self):
        # Prose that merely mentions turns must not be quarantined.
        assert _probe_harness_reason("We took turns speaking.", {}) == ""

    @pytest.mark.asyncio
    async def test_add_memory_refuses_probe_content(self):
        from core.memory.memory_facade import MemoryFacade

        facade = MemoryFacade.__new__(MemoryFacade)
        facade._last_add_memory_status = {}
        # Only the early-refusal path runs; heavier machinery would fail loudly
        # if the refusal did not happen first.
        facade._merge_unity_metadata = lambda metadata: dict(metadata or {})
        facade._stamp_welfare_context = lambda payload: payload
        facade._welfare_should_block_write = lambda payload: ""

        ok = await MemoryFacade.add_memory(
            facade,
            "(turn 42) What was the phrase from earlier in this probe?",
            {"session_id": "endurance-test"},
        )
        assert ok is False
        assert facade._last_add_memory_status["reason"].startswith("probe_hygiene:")
