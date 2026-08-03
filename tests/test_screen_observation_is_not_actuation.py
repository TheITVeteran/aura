"""Reading the screen is not acting on it, and must not be routed as if it were.

Live on 2026-08-03, Bryan asked:

    "Hey, Aura. Mostly just curious. Can you see what's on the screen and tell
     me what you see?"

It routed through desktop_task into os_automation, which refused: "OS
automation refused to act because the objective has no complete observable
acceptance contract." That refusal is correct for os_automation — a
description is not an effect it can verify — so the defect was upstream.

"Is this a screen observation?" was answered in two places that disagreed.
desktop_task held a literal-substring list containing "what's on my screen"
and "look at the screen"; Bryan said "the screen", matching neither, so the
read escalated into the actuation lane. The router's regex had matched the
sentence all along. There is one definition now.
"""
from __future__ import annotations

import pytest

from core.runtime.desktop_objective_intent import (
    looks_like_desktop_objective,
    looks_like_screen_observation,
)
from core.skills.desktop_task import DesktopTaskSkill

BRYAN_LIVE_MESSAGE = (
    "Hey, Aura. Mostly just curious. Can you see what's on the screen and "
    "tell me what you see?"
)


class TestTheSentenceThatFailedLive:
    def test_the_live_message_is_recognised_as_observation(self):
        assert looks_like_screen_observation(BRYAN_LIVE_MESSAGE)

    def test_the_live_message_does_not_escalate_to_os_automation(self):
        assert DesktopTaskSkill._objective_requests_observation_only(BRYAN_LIVE_MESSAGE), (
            "this is what sent a read into the actuation lane and got it refused"
        )

    def test_it_still_needs_the_desktop_body(self):
        """Observation is not conversation — it still requires the screen."""
        assert looks_like_desktop_objective(BRYAN_LIVE_MESSAGE)


class TestOneDefinitionNotTwo:
    """The two predicates must never diverge again."""

    PHRASINGS = (
        "what is on my screen",
        "what's on my screen",
        "what's on the screen",
        "what do you see on the screen",
        "read my screen",
        "read the screen",
        "look at the screen and tell me what is there",
        "describe the screen",
        "inspect the screen",
        "take a screenshot",
        "can you see what is on the screen",
    )

    @pytest.mark.parametrize("phrasing", PHRASINGS)
    def test_every_phrasing_agrees_across_both_callers(self, phrasing):
        shared = looks_like_screen_observation(phrasing)
        assert shared is True, f"not recognised as observation: {phrasing!r}"
        assert DesktopTaskSkill._objective_requests_observation_only(phrasing) is shared


class TestObservationPlusActionIsStillAction:
    """A read that also changes something belongs in the actuation lane."""

    MIXED = (
        "look at the screen and close the Chrome window",
        "read my screen then save it to a file",
        "check the screen and click the submit button",
        "describe the screen and then open Notes",
    )

    @pytest.mark.parametrize("objective", MIXED)
    def test_mixed_intent_is_not_observation_only(self, objective):
        assert looks_like_screen_observation(objective) is False, (
            "an objective that also mutates something needs the lane that can "
            "verify the effect"
        )
        assert DesktopTaskSkill._objective_requests_observation_only(objective) is False


class TestConversationAboutTheScreenIsNotALook:
    NARRATION = (
        "earlier you described my screen and I decided you had made it up",
        "you said you could read my screen, is that true",
        "what is the weather today",
    )

    @pytest.mark.parametrize("message", NARRATION)
    def test_recounting_is_not_a_request_to_look(self, message):
        assert looks_like_screen_observation(message) is False
