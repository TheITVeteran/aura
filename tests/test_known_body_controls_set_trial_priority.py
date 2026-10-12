"""Named body controls outrank clickable interface furniture in the real control loop."""
import time
from types import SimpleNamespace

import pytest

from core.agency import playing_as_it_happens as play
from core.agency.how_a_game_works import (
    AIMS_FROM_A_BODY,
    FOLLOWS,
    STEERED,
    Answer,
    HowItWorks,
    understood,
)
from core.agency.which_one_answers_to_her import WhichIsHers
from core.cognition.a_guide_to_a_place import PROGRAM, THE_GUIDE, TOLD, Guide

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("body", [STEERED, AIMS_FROM_A_BODY])
def test_known_keyboard_body_overrides_incidental_pointer_guess(body):
    how = HowItWorks(me=Answer(body, .9, ["observed control"]))
    assert how.try_pointer_first(True) is False


@pytest.mark.parametrize("guess", [True, False])
def test_unmeasured_input_priority_remains_unmeasured(guess):
    assert HowItWorks().try_pointer_first(guess) is guess
    assert HowItWorks(me=Answer(STEERED, .4, ["program event"])).try_pointer_first(guess) is guess
    assert HowItWorks(me=Answer(FOLLOWS, .9, ["observed following"])).try_pointer_first(guess) is True


def test_the_registered_priority_invariant():
    from core.agency.how_a_game_works import _known_body_controls_invariant

    assert _known_body_controls_invariant() == ()


def guide():
    current = Guide(place="An unfamiliar interactive floor")
    current.take_in(PROGRAM, "The program listens for mouse clicks.")
    current.take_in(TOLD, "Use the up and down arrow keys to move your body. Click the menu button to change settings.")
    return current


@pytest.mark.asyncio
async def test_actual_actor_calibrates_a_named_key_before_pointer_sweeps_or_favored_taps(monkeypatch):
    current = guide()
    how = understood(current)
    assert how.me.known() and how.me.value == STEERED
    hands = []

    class Hands:
        async def down(self, key):
            hands.append(("down", key))

        async def up(self, key):
            hands.append(("up", key))

        async def tap(self, key):
            hands.append(("tap", key))

        async def point(self, *point):
            hands.append(("point", point))

    monkeypatch.setattr(play, "a_key_that_pays", lambda keys: "down")
    run = play._Run(keys=["up", "down"], began=time.monotonic(),
                    pointer_first=how.try_pointer_first(True))
    choosing = SimpleNamespace(mine=None, ways={}, lifts=[])
    moves = SimpleNamespace(seeing=True, things={})
    token = THE_GUIDE.set(current)
    try:
        await play._act(Hands(), run, moves, WhichIsHers(), None, choosing, time.monotonic())
    finally:
        THE_GUIDE.reset(token)
    assert hands == [("down", "up")]
    assert run.trying == 1 and run.held_trying
