"""A step count is not an answer, and an unread screen is not a blank one.

Both found in the live desktop on 2026-08-10.

1. "can you read text that is only pixels — words inside a video frame? answer
   yes or no first, then tell me how you know" came back as:

       Desktop task completed 1/1 governed computer-use steps through
       heuristic_compat planning. Completed 1/1 governed desktop steps.

   The step count, twice, in the branch whose own comment reads "What was
   PRODUCED, not how many steps produced it" — because the lane's `summary`
   is itself a step-count sentence, so it got pasted in front of the step
   sentence. This is the third recorded appearance of this defect shape
   (2026-07-27, 2026-07-30, 2026-08-04 all sit in comments at that call site).

2. ScreenSnapshot.screen_text is populated only from OCR of a screenshot. For
   the entire life of the governance defect that blocked take_screenshot, OCR
   never ran, screen_text was always "", and every consumer read that as "the
   screen has no text on it". Absence of a reading is not a reading of
   absence.
"""

from __future__ import annotations

import pytest


# ── 1. Bookkeeping must not be served as an answer ─────────────────────────

@pytest.mark.parametrize(
    "summary",
    [
        "Desktop task completed 1/1 governed computer-use steps through heuristic_compat planning.",
        "Desktop task completed 2/2 governed computer-use steps through heuristic_compat planning",
        "Completed 2/2 governed desktop steps.",
        "completed 10/10 governed desktop steps",
        "",
    ],
)
def test_pure_step_reports_are_recognised_as_bookkeeping(summary: str) -> None:
    from interface.routes.chat import _is_step_bookkeeping_only

    assert _is_step_bookkeeping_only(summary) is True


@pytest.mark.parametrize(
    "summary",
    [
        "Chrome is in front, showing the YouTube documentary.",
        "I wrote the note in Notes. Completed 2/2 governed desktop steps.",
        "Completed 1/1 governed desktop steps. The file is at /tmp/x.txt",
        "Opened Notes and typed the paragraph.",
        "Completed the research and found three sources.",
    ],
)
def test_summaries_about_the_world_are_never_suppressed(summary: str) -> None:
    """The dangerous direction: silencing a real answer for mentioning steps."""
    from interface.routes.chat import _is_step_bookkeeping_only

    assert _is_step_bookkeeping_only(summary) is False


def test_bookkeeping_only_result_defers_to_cognition() -> None:
    """No observation, no deliverable, only a step count ⇒ hand off the turn."""
    import inspect

    from interface.routes import chat

    source = inspect.getsource(chat)
    marker = "_is_step_bookkeeping_only(summary)"
    assert marker in source
    branch = source[source.find(marker) : source.find(marker) + 1400]
    # The branch must return no reply so cognition answers the real question,
    # exactly as the perception branch above it already does.
    assert 'response = ""' in branch


# ── 2. An unread screen must not read as a blank screen ────────────────────

def test_screen_text_status_defaults_to_not_attempted() -> None:
    from core.perception.screen_perception import ScreenSnapshot

    assert ScreenSnapshot().screen_text_status == "not_attempted"
    assert ScreenSnapshot().screen_text == ""


def test_blank_screen_and_unread_screen_are_distinguishable() -> None:
    """The two states that were identical for the life of the capture defect."""
    from core.perception.screen_perception import ScreenSnapshot

    genuinely_blank = ScreenSnapshot()
    genuinely_blank.screen_text = ""
    genuinely_blank.screen_text_status = "read_empty"

    never_read = ScreenSnapshot()
    never_read.screen_text = ""
    never_read.screen_text_status = "unreadable:capture_failed"

    assert genuinely_blank.screen_text == never_read.screen_text
    assert genuinely_blank.screen_text_status != never_read.screen_text_status


def test_capture_path_sets_a_status_in_every_branch() -> None:
    """A silent "" is the whole defect; every path must say which it is."""
    import inspect

    from core.perception.screen_perception import ScreenPerception

    source = inspect.getsource(ScreenPerception.capture)

    assert 'screen_text_status = "read"' in source
    assert "unreadable:" in source
    assert 'screen_text_status = "not_attempted"' in source
