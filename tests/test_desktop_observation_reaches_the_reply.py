"""What Aura saw must reach the person who asked, not the step count.

Live 2026-08-03. Bryan asked "Can you see what's on the screen and tell me
what you see?" and, after the routing fix, the read completed 1/1 governed
steps and returned:

    active_app:   Google Chrome
    window_title: Grant application — Corrigibility in a persistent AI agent…

He was handed: "Desktop task completed 1/1 governed computer-use steps
through heuristic_compat planning."

The composer already meant to lead with the observation — the comment above
it says so, from a 2026-07-27 fix of the same symptom. The extractor just
could not find this receipt: desktop_task returns `receipts`, which was not
one of the containers it searched, and the reading sits one level down under
each receipt's own `result` dict, which it only ever tested for a string.

A step count is what the machine did. The answer is what it found.
"""
from __future__ import annotations

import pytest

from interface.routes.chat import _desktop_task_observation

CHROME_WINDOW = (
    "Grant application — Corrigibility in a persistent AI agent - "
    "youngbryan97@gmail.com - Gmail - Google Chrome - Bryan"
)
SCREEN_TEXT = f"Active app: Google Chrome\nWindow: {CHROME_WINDOW}"


def live_payload(**receipt_result) -> dict:
    """The exact shape the live desktop_task returned for a screen read."""

    payload = {
        "ok": True,
        "status": "success_verified",
        "objective": "Can you see what's on the screen and tell me what you see?",
        "steps_requested": 1,
        "steps_completed": 1,
        "summary": (
            "Desktop task completed 1/1 governed computer-use steps through "
            "heuristic_compat planning."
        ),
        "receipts": [
            {
                "index": 1,
                "action": "read_screen_text",
                "ok": True,
                "effect_verified": True,
                "effect_evidence": "screen_text_returned;frontmost_app=Google Chrome",
                "result": {
                    "ok": True,
                    "status": "limited",
                    "source": "screen_perception",
                    "active_app": "Google Chrome",
                    "window_title": CHROME_WINDOW,
                    "accessibility_text": "",
                    "screen_text": "",
                    "text": SCREEN_TEXT,
                    **receipt_result,
                },
            }
        ],
    }
    return payload


class TestTheLiveReceipt:
    def test_the_observation_is_found(self):
        assert _desktop_task_observation(live_payload()) == SCREEN_TEXT

    def test_it_is_not_the_step_count(self):
        observation = _desktop_task_observation(live_payload())
        assert "governed computer-use steps" not in observation
        assert not observation.casefold().startswith("desktop task completed")


class TestEveryReceiptShape:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"screen_text": "Finder — Documents"},
            {"accessibility_text": "Safari — Apple"},
            {"text": SCREEN_TEXT},
        ],
        ids=["screen_text", "accessibility_text", "text"],
    )
    def test_each_observation_field_is_read(self, overrides):
        payload = live_payload(**overrides)
        found = _desktop_task_observation(payload)
        assert found, f"nothing extracted for receipt carrying {sorted(overrides)}"

    def test_receipts_is_searched_like_steps(self):
        """`receipts` is the key desktop_task actually returns."""

        as_steps = {"ok": True, "steps": [{"text": SCREEN_TEXT}]}
        as_receipts = {"ok": True, "receipts": [{"text": SCREEN_TEXT}]}
        assert _desktop_task_observation(as_steps) == SCREEN_TEXT
        assert _desktop_task_observation(as_receipts) == SCREEN_TEXT

    def test_a_receipt_with_nothing_observed_yields_nothing(self):
        payload = {
            "ok": True,
            "receipts": [{"index": 1, "action": "open_app", "ok": True, "result": {"ok": True}}],
        }
        assert _desktop_task_observation(payload) == ""

    def test_bookkeeping_phrasing_is_never_offered_as_an_observation(self):
        payload = {
            "ok": True,
            "receipts": [
                {"result": {"text": "Desktop task completed 1/1 governed computer-use steps."}}
            ],
        }
        assert _desktop_task_observation(payload) == ""
