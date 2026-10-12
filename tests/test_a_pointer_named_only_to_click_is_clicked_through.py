"""What a place's words say the pointer is for decides whether it is played in time: moved, steered, aimed, or clicked
to throw, it is; clicked on things, dragged, or clicked to start, it is clicked through. A press held on a ball in a
passage about putting is a send.

Offline 2026-10-11 (the 56-game campaign, pass 3): a hidden-object game whose rooms animated was handed to her fast play
thirty-two times, three seconds at a time; a golf game's "click on your ball and hold down the mouse button" was never
putted; a food fight was handed back three seconds into its first quiet and lost with nothing thrown.
"""
from __future__ import annotations

import pytest

from core.agency.playing_by_shots import sends_by_letting_go
from core.agency.the_controls_a_game_names import the_pointer_only_clicks
from core.agency.the_way_it_is_played import AS_IT_HAPPENS, SEND, ways_asked

pytestmark = pytest.mark.unit

CLICKED_THROUGH = ["Click on items as you find them. Then drag them here and assemble them before time runs out!",
                   "Click to Start", "Click on the doors to see who lives there."]
IN_TIME = ["TAKE AIM WITH YOUR MOUSE AND CLICK TO THROW. DODGE FOOD BY PRESSING THE SPACE BAR.",
           "Use the mouse to move your basket and catch the eggs.",
           "Move your mouse to stack crystals of the same color into a tower.",
           "The mouse moves Bloo; click to jump over the boxes."]


@pytest.mark.parametrize("said", CLICKED_THROUGH)
def test_a_pointer_named_only_to_click_or_drag_is_not_played_in_time(said):
    assert the_pointer_only_clicks(said)
    assert AS_IT_HAPPENS not in ways_asked(said)


@pytest.mark.parametrize("said", IN_TIME)
def test_a_pointer_moved_aimed_or_clicked_to_act_is_played_in_time(said):
    assert not the_pointer_only_clicks(said)
    assert AS_IT_HAPPENS in ways_asked(said)


def test_keys_named_for_moving_are_played_in_time_whatever_the_pointer_does():
    said = "Use the arrow keys to move. Click on the gems to collect them."
    assert not the_pointer_only_clicks(said)


def test_a_press_held_on_a_ball_in_a_passage_about_putting_is_a_send():
    said = "HOW TO USE THE PUTTER: 1. Click on your ball and hold down the mouse button."
    assert sends_by_letting_go(said)
    assert ways_asked(said)[0] == SEND
    # Held to use something for as long as it is held is no send.
    assert not sends_by_letting_go("While flying, click and hold your mouse button to use Glide Power.")


def _a_run(**kwargs):
    from core.agency.playing_as_it_happens import _Run

    return _Run(keys=["space"], began=0.0, last_moving=0.0, **kwargs)


class _Moves:
    """A picture in which a thing moves at the given times and nothing moves at the others."""

    def __init__(self, moving_at):
        self.moving_at = set(moving_at)
        self.at = 0.0

    def moving(self, faster_than=0.0):
        from types import SimpleNamespace

        return [SimpleNamespace(seen=self.at)] if self.at in self.moving_at else []


def _ended(run, moves, until):
    from core.agency.playing_as_it_happens import _over

    at = 0.0
    while at <= until:
        moves.at = at
        why = _over(run, moves, [], at)
        if why:
            return at, why
        at = round(at + 0.25, 2)
    return None


def test_a_place_played_in_time_is_not_handed_back_in_its_first_quiet():
    # Things come at 1 s, then not again until 7 s: a quiet of six seconds is how this place goes.
    times = [1.0, 7.0, 7.25]
    assert _ended(_a_run(), _Moves(times), 7.5) is not None                         # not said to be played in time
    assert _ended(_a_run(in_time=True), _Moves(times), 7.5) is None


def test_the_quiet_a_place_has_kept_while_played_sets_how_long_a_stillness_must_last():
    run = _a_run(in_time=True)
    times = [0.5, 0.75, 4.25, 4.5]                                                  # a quiet of three and a half
    assert _ended(run, _Moves(times), 4.5) is None
    assert run.longest_still == pytest.approx(3.5)
    later = _a_run(longest_still=run.longest_still)
    assert _ended(later, _Moves([0.5]), 5.0) is None                               # 1.5 x 3.5 = 5.25 s of patience
