"""A click chosen again on a screen whose words the last choice of it left as they were is worth less each time.

Offline 2026-10-12 an alien builder's corner button was pressed six times running, its worth the same each time, while
I'M DONE! waited one place below it.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.skills.screen_pursuit_decision_steps import remember_the_pick, set_aside_what_changed_nothing

pytestmark = pytest.mark.unit

BUILDER = {"text": "HEIGHT TORSO TEXTURES STRENGTH SPEED AGILITY FLIGHT RANDOMIZE VIEW GALLERY I'M DONE!"}
CORNER = 'click "the one that stands out at 4% across, 94% down"'
DONE = "click \"I'M DONE!\""


def _worths(run, observation):
    ahead = {CORNER: (2.174, "to see what it does"), DONE: (1.021, "to see what it does"), "right": (1.5, "")}
    return {name: value for name, (value, _why) in set_aside_what_changed_nothing(run, ahead, observation).items()}


def test_a_click_that_left_the_screen_as_it_was_gives_way_to_another():
    run = SimpleNamespace()
    first = _worths(run, BUILDER)
    assert max(first, key=first.get) == CORNER
    for _ in range(2):
        remember_the_pick(run, CORNER, BUILDER)
        worths = _worths(run, BUILDER)
    assert max(worths, key=worths.get) == "right" or worths[DONE] > worths[CORNER]
    assert worths[DONE] == pytest.approx(1.021) and worths["right"] == pytest.approx(1.5)   # keys and others as they were


def test_on_another_screen_what_changed_nothing_is_forgotten():
    run = SimpleNamespace()
    remember_the_pick(run, CORNER, BUILDER)
    _worths(run, BUILDER)
    remember_the_pick(run, CORNER, BUILDER)
    named = {"text": "NAME YOUR ALIEN SAVE TO THE GALLERY"}
    assert _worths(run, named)[CORNER] == pytest.approx(2.174)
