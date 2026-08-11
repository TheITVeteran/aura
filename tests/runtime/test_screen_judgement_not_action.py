"""An opinion about the screen is not an action on it.

LIVE DEFECT, 2026-08-10. Asked:

    "look at my screen again, then give me an opinion rather than a
     description. of everything you can see open right now, which window would
     you close first if you were me, and why that one? I want your actual
     judgement, not a list."

the router matched its screen-observation branch, sent the turn to the desktop
lane, and the person was handed:

    "os_automation failed: OS automation refused to act because the objective
     has no complete observable acceptance contract … Completed 0/1 steps. I am
     not claiming the desktop action finished."

os_automation was right to refuse. Nothing was asked to happen — nobody asked
her to close a window, they asked which one she WOULD close. The request needs
screen data as evidence and a judgement as the answer, and only the
conversational lane can produce the second half.

Her agency rules already said it: "Hypotheticals, quoted requests, negated
actions, and recalled evidence are not execution requests merely because they
name a tool." That lived in the identity contract, where the router could not
act on it.
"""

from __future__ import annotations

import pytest


LIVE_MESSAGE = (
    "look at my screen again, then give me an opinion rather than a "
    "description. of everything you can see open right now, which window would "
    "you close first if you were me, and why that one? I want your actual "
    "judgement, not a list."
)


def _is_desktop(text: str) -> bool:
    from core.phases.response_contract import looks_like_desktop_objective

    return looks_like_desktop_objective(text)


@pytest.mark.parametrize(
    "message",
    [
        LIVE_MESSAGE,
        "which window would you close first?",
        "if you were me which app would you quit",
        "what do you think of the windows I have open",
        "should i close chrome or keep it",
        "give me your take on my screen layout",
        "of these windows which one is worth keeping",
    ],
)
def test_judgement_requests_do_not_go_to_the_desktop_lane(message: str) -> None:
    assert _is_desktop(message) is False


@pytest.mark.parametrize(
    "message",
    [
        "close Chrome",
        "close the front window",
        "open Notes and write a paragraph",
        "read my screen and tell me what you see",
        "what is on my screen",
        "take a screenshot",
    ],
)
def test_real_desktop_work_is_untouched(message: str) -> None:
    assert _is_desktop(message) is True


def test_an_imperative_outranks_an_opinion_in_the_same_message() -> None:
    """"close the window and give me your opinion" must still close it.

    The dangerous direction for this guard: silently dropping real work
    because the sentence also asked for a view.
    """
    assert _is_desktop("close the window and give me your opinion on the article") is True
    assert _is_desktop("minimize everything then tell me what you think") is True


def test_observation_verbs_are_not_effects() -> None:
    """"look at my screen" changes nothing, so it must not read as imperative.

    If it did, the live message above would be classified as desktop work by
    its own first two words.
    """
    from core.utils.screen_judgement_intent import EFFECT_IMPERATIVE_RE

    for observation in ("look at my screen", "read my screen", "show me the window",
                        "tell me what is open"):
        assert not EFFECT_IMPERATIVE_RE.search(observation), observation
    for effect in ("close the window", "open Notes", "minimize everything"):
        assert EFFECT_IMPERATIVE_RE.search(effect), effect


def test_a_choice_needs_no_screen_noun() -> None:
    """"should I close chrome or keep it" names no surface and is still a question."""
    from core.utils.screen_judgement_intent import asks_for_screen_judgement

    assert asks_for_screen_judgement("should i close chrome or keep it") is True


def test_plural_surfaces_reach_the_desktop_lane() -> None:
    """Pre-existing, found the same day: every surface term was singular.

    "minimize the window" routed to the desktop lane and "minimize all windows"
    did not — the same request, declined for its grammatical number.
    """
    assert _is_desktop("minimize all windows") is True
    assert _is_desktop("close all my tabs") is True
    assert _is_desktop("close all the apps") is True


def test_predicate_lives_in_core_utils() -> None:
    """core/runtime may not import cognition; a second copy is how they drift.

    Same reason own_source_intent and occluded_view_intent live there.
    """
    from core.utils import screen_judgement_intent

    assert screen_judgement_intent.__name__.startswith("core.utils.")
