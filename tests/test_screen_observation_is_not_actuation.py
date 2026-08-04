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


class TestThePhrasingsBryanActuallyUsed:
    """Regression pins, one per live failure.

    This request has now failed live twice, weeks apart, both times because the
    matcher enumerated phrasings and the next phrasing fell outside. Each entry
    below is a sentence a real person actually typed, kept verbatim.

    The failure mode is worse than a refusal: Aura answers "I can't see your
    screen — I don't have that kind of access", which is FALSE. She has the
    capability; the router never offered her the chance to use it, so she
    explained the absence as a limitation and then as a preference.
    """

    def test_the_2026_08_03_1328_phrasing(self):
        from core.runtime.desktop_objective_intent import looks_like_screen_observation

        # "what you see", not "what DO you see" — one auxiliary verb away from
        # a false denial.
        assert looks_like_screen_observation(
            "Hey, Aura can you tell me what you see on my screen currently?"
        )

    def test_the_earlier_the_screen_phrasing_still_matches(self):
        from core.runtime.desktop_objective_intent import looks_like_screen_observation

        assert looks_like_screen_observation(
            "Can you see what's on the screen and tell me what you see?"
        )

    @pytest.mark.parametrize(
        "message",
        [
            "what's on my screen",
            "what do you see on screen right now",
            "can you see my monitor?",
            "describe my display",
            "read the screen and tell me",
            "take a screenshot",
            "my screen — what do you see?",
        ],
    )
    def test_ordinary_ways_of_asking(self, message):
        from core.runtime.desktop_objective_intent import looks_like_screen_observation

        assert looks_like_screen_observation(message), message

    @pytest.mark.parametrize(
        "message",
        [
            "close the window on my screen",
            "open Chrome and look at the screen",
            "earlier my screen showed an error",
            "what should we talk about",
        ],
    )
    def test_still_not_an_observation(self, message):
        """Widening the cue must not swallow actuation or ordinary talk."""
        from core.runtime.desktop_objective_intent import looks_like_screen_observation

        assert not looks_like_screen_observation(message), message


class TestARequestedDelayIsPartOfTheRequest:
    """"Wait 5 seconds, then tell me what is on my screen" waited 0 seconds.

    Live 2026-08-03: one read_screen_text step, answered in 1s, reported as
    "Completed 1/1 governed desktop steps". The observation was of the wrong
    moment and nothing said the delay had been dropped.
    """

    def _plan(self, objective: str):
        from core.skills.desktop_task import DesktopTaskSkill

        return DesktopTaskSkill()._derive_single_objective_steps(objective, {})

    def test_a_stated_delay_becomes_a_wait_step_first(self):
        steps = self._plan("Please wait 5 seconds, then tell me what is on my screen.")
        assert steps[0].action == "wait"
        assert steps[0].target == "5"
        assert any(step.action == "read_screen_text" for step in steps)

    def test_no_delay_means_no_wait(self):
        steps = self._plan("what is on my screen")
        assert not any(step.action == "wait" for step in steps)

    def test_an_unquantified_delay_invents_no_duration(self):
        """"Wait a moment" leaves the quantity unspecified; picking one would
        be answering a request nobody made."""
        steps = self._plan("wait a moment and tell me what is on my screen")
        assert not any(step.action == "wait" for step in steps)

    @pytest.mark.parametrize(
        ("objective", "seconds"),
        [
            ("wait 10s and read the screen", 10.0),
            ("give it 3 secs then look at the screen", 3.0),
            ("in 2 minutes tell me what is on my screen", 120.0),
        ],
    )
    def test_units_are_read_from_the_request(self, objective, seconds):
        from core.skills.desktop_task import _requested_wait_seconds

        assert _requested_wait_seconds(objective) == seconds


class TestBeingShownHerOwnCodeIsNotDesktopWork:
    """"Show me a piece of your own code and which file it lives in" carries an
    action word and a surface word, so it was sent to os_automation and refused
    for having no observable effect — while the conversational floor had a real
    1999-character excerpt ready. Measured live 2026-08-03 21:42.
    """

    BRYAN_ASKED = (
        "Show me a piece of your own code that you find interesting. Tell me "
        "which file it lives in, what it does, and why it interests you."
    )

    def test_the_desktop_lane_keeps_its_hands_off(self):
        assert looks_like_desktop_objective(self.BRYAN_ASKED) is False

    def test_the_source_floor_answers_it(self):
        from core.conversation.response_reliability import own_source_excerpt_floor

        assert own_source_excerpt_floor(self.BRYAN_ASKED).strip()

    def test_somebody_elses_code_is_not_hers(self):
        from core.conversation.response_reliability import own_source_excerpt_floor
        from core.utils.own_source_intent import asks_for_own_source

        assert asks_for_own_source("show me the actual code for numpy") is False
        assert own_source_excerpt_floor("show me the actual code for numpy") == ""

    def test_real_desktop_work_still_routes(self):
        assert looks_like_desktop_objective("open Chrome and show me the file on my desktop")

    def test_the_two_layers_share_one_definition(self):
        """Two copies of this judgement drift, and then one layer answers a
        question the other was going to answer properly."""
        import inspect

        from core.conversation import response_reliability
        from core.runtime import desktop_objective_intent

        for module in (response_reliability, desktop_objective_intent):
            assert "own_source_intent" in inspect.getsource(module)
