"""The wallpaper leg was silently dropped from the objective.

Measured live. The request was:

  "...find an orca image online, set it as my desktop wallpaper, and tell me
   the URL you got it from."

The chain reported desktop_objective_completed, the folder and PDF were real —
and the desktop picture never changed. The planner never emitted the wallpaper
steps at all, because intent detection did not recognise this phrasing, so
there was nothing to fail: the leg simply did not exist in the plan.

Two gaps, both of them ordinary English:
  - "an orca IMAGE" (attributive) vs "an image OF an orca";
  - "set IT as my wallpaper", where the referent is earlier in the sentence.
"""

from __future__ import annotations

import pytest

from core.skills.os_affordances import detect_os_settings


@pytest.mark.parametrize(
    "objective,topic",
    [
        (
            "find an orca image online, set it as my desktop wallpaper, and tell "
            "me the URL you got it from",
            "orca",
        ),
        ("find an orca image online and set it as my wallpaper", "orca"),
        ("Change my wallpaper to an orca and show me where you found it.", "orca"),
        ("change my background to an orca", "orca"),
        ("set my desktop background to an orca", "orca"),
        ("make my wallpaper an orca", "orca"),
        (
            "find a picture of a humpback whale and make it my background",
            "humpback whale",
        ),
    ],
)
def test_wallpaper_requests_are_recognised(objective, topic):
    assert detect_os_settings(objective) == [("wallpaper", topic)], objective


@pytest.mark.parametrize(
    "objective",
    [
        "what's my wallpaper?",
        "tell me about orcas",
        "do you like my desktop background?",
        "write three sentences about orcas in a note",
    ],
)
def test_non_requests_do_not_change_anything(objective):
    """Asking ABOUT the wallpaper must never change it."""
    assert detect_os_settings(objective) == [], objective


def test_a_bare_pronoun_with_no_referent_is_not_invented():
    """Resolving a pronoun must not become guessing at one."""
    assert detect_os_settings("set it as my wallpaper") == []
