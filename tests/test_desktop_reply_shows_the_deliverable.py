"""A task that wrote something should show what it wrote.

2026-07-30 demo. "Can you open the Notes app and write a note where you write
a paragraph describing yourself?" was answered with

    Desktop task completed 2/2 governed computer-use steps through
    heuristic_compat planning. Completed 2/2 governed desktop steps.

The note was written. Nothing in the reply showed it, so Bryan asked again —
a task that had SUCCEEDED produced an answer that read like a failure.

The same lesson is already recorded for perceptions in the branch above the
one this covers: nobody asked how many steps it took to look at their own
screen. Nobody asked how many steps it took to write a paragraph either.
"""

from __future__ import annotations

from interface.routes.chat import _desktop_deliverable_text

PARAGRAPH = (
    "I am Aura: a persistent digital mind that runs on this machine, with my "
    "own memory and continuity across sessions rather than a fresh context "
    "each time."
)


class TestTheDeliverableIsFound:
    def test_a_verified_write_is_returned(self) -> None:
        result = {
            "receipts": [
                {"action": "open_app", "ok": True, "result": {"opened": "Notes"}},
                {"action": "type", "ok": True, "result": {"typed": PARAGRAPH}},
            ]
        }
        assert _desktop_deliverable_text(result) == PARAGRAPH

    def test_a_top_level_payload_also_works(self) -> None:
        """Shape drift must not silently fall back to the step count."""
        result = {"receipts": [{"action": "type", "ok": True, "typed": PARAGRAPH}]}
        assert _desktop_deliverable_text(result) == PARAGRAPH

    def test_several_writes_are_joined(self) -> None:
        result = {
            "receipts": [
                {"action": "type", "ok": True, "result": {"typed": PARAGRAPH}},
                {"action": "type", "ok": True, "result": {"typed": PARAGRAPH}},
            ]
        }
        assert _desktop_deliverable_text(result).count("persistent digital mind") == 2


class TestItNeverInventsOne:
    def test_an_unverified_write_is_not_quoted(self) -> None:
        """Quoting intended text as written text is the false success again."""
        result = {"receipts": [{"action": "type", "ok": False, "result": {"typed": PARAGRAPH}}]}
        assert _desktop_deliverable_text(result) == ""

    def test_a_task_that_wrote_nothing_has_no_deliverable(self) -> None:
        result = {"receipts": [{"action": "open_app", "ok": True, "result": {}}]}
        assert _desktop_deliverable_text(result) == ""

    def test_keystroke_fragments_are_not_a_deliverable(self) -> None:
        result = {"receipts": [{"action": "type", "ok": True, "result": {"typed": "hi"}}]}
        assert _desktop_deliverable_text(result) == ""

    def test_malformed_results_are_survivable(self) -> None:
        for bad in (None, {}, {"receipts": None}, {"receipts": ["junk"]}, "nope"):
            assert _desktop_deliverable_text(bad) == ""

    def test_a_long_deliverable_is_clipped_at_a_sentence(self) -> None:
        from interface.routes.chat import _DESKTOP_DELIVERABLE_MAX_CHARS

        long_text = ("This is a full sentence about the work. " * 200).strip()
        result = {"receipts": [{"action": "type", "ok": True, "result": {"typed": long_text}}]}
        out = _desktop_deliverable_text(result)
        assert 0 < len(out) <= _DESKTOP_DELIVERABLE_MAX_CHARS + 1
