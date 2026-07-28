"""What she saw, not how many steps it took.

Live 2026-07-27: "Read my screen and tell me what you actually see on it
right now." The task SUCCEEDED — status desktop_objective_completed, 1/1
governed steps — and what reached the person was:

    Desktop task completed 1/1 governed computer-use steps through
    heuristic_compat planning. Completed 1/1 governed desktop steps.

The observation was in the receipt the whole time. A step count is what the
machine did; the answer is what it found, and the question was the second
one. This is the session's recurring shape once more — the content exists and
the surface reports plumbing.
"""

import pytest

from interface.routes.chat import _desktop_task_observation

pytestmark = pytest.mark.unit


def test_a_step_observation_is_found():
    assert _desktop_task_observation(
        {
            "steps": [{"observation": "Google Chrome — Aura Luna | Talk to Aura..."}],
            "summary": "Desktop task completed 1/1 governed computer-use steps",
        }
    ) == "Google Chrome — Aura Luna | Talk to Aura..."


def test_a_top_level_screen_text_is_found():
    assert _desktop_task_observation({"screen_text": "Finder — Documents"}) == (
        "Finder — Documents"
    )


def test_executor_bookkeeping_is_not_an_observation():
    """The mechanism must never masquerade as the finding."""
    assert _desktop_task_observation(
        {"summary": "Desktop task completed 1/1 governed computer-use steps"}
    ) == ""
    assert _desktop_task_observation(
        {"steps": [{"output": "Completed 1/1 steps"}]}
    ) == ""


def test_nothing_observed_reports_nothing():
    assert _desktop_task_observation({}) == ""
    assert _desktop_task_observation({"steps": []}) == ""
