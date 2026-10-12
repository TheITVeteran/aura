"""Actual actor controls are learned before a finite drop ends its observations."""
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency import playing_as_it_happens as play
from core.agency.what_meeting_things_does import AVOID, WhatMeetingDoes
from core.agency.which_one_answers_to_her import WhichIsHers
from core.perception.what_moves_in_the_picture import Kind, Thing

pytestmark = pytest.mark.unit


def _body(number, kind, position, colour):
    look = np.zeros(64)
    look[kind] = 1.
    return Thing(number, *position, 14., 14., look, colour, np.zeros((1, 1, 3), np.uint8),
                 born=.0, seen=.0, kind=kind, moved=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("axis,sign,keys", [(1, 1, ("a", "b", "c")), (1, -1, ("j", "l", "k")),
                                         (0, 1, ("z", "x", "y")), (0, -1, ("m", "o", "n"))])
async def test_cold_actor_repeats_an_observed_control_then_measures_the_other_controls(monkeypatch, axis, sign, keys):
    now = [0.0]
    monkeypatch.setattr(play.time, "monotonic", lambda: now[0])
    actions = []

    class Hands:
        async def down(self, key):
            actions.append((now[0], key))

        async def up(self, key):
            actions.append((now[0], ""))

    shape = (260, 400) if axis == 1 else (400, 260)
    position, other = [100., 100.], [260., 260.]
    position[axis], other[axis] = (50., 20.) if sign == 1 else (210., 240.)
    mine = _body(1, 0, position, (170, 220, 140))
    stone = _body(2, 1, other, (80, 80, 90))
    moves = SimpleNamespace(shape=shape, things={1: mine, 2: stone}, seeing=True,
                            kinds=[Kind(0, mine.look, 196, mine.colour), Kind(1, stone.look, 196, stone.colour)])
    hers = WhichIsHers()
    meeting = WhatMeetingDoes()
    meeting.told[1] = AVOID
    run = play._Run(keys=list(keys), began=0.)
    hands = Hands()
    response_at, identity_at = None, None
    assert hers.number is None and not hers._by_thing
    for step in range(1, 289):
        now[0] = step / 60
        for thing, push in ((mine, -110. if run.held == keys[0] else 90.), (stone, 90.)):
            velocity = [thing.vx, thing.vy]
            velocity[axis] += sign * push / 60
            thing.vx, thing.vy = velocity
            thing.x += thing.vx / 60
            thing.y += thing.vy / 60
            thing.seen = now[0]
            thing.path.append((now[0], thing.x, thing.y))
        if not 0 <= (stone.x, stone.y)[axis] <= 260:
            moves.things.pop(2, None)
        hers.saw(moves, [], now[0])
        if hers.number is not None and identity_at is None:
            assert hers.number == mine.number
            identity_at = now[0]
        choosing = play._Choosing(moves, hers, meeting, run.keys)
        await play._act(hands, run, moves, hers, meeting, choosing, now[0])
        if run.counter_trial.response is not None and response_at is None:
            response_at = now[0]
            assert hers.number is None
            assert run.counter_trial.response.number == mine.number
        location = (mine.x, mine.y)[axis]
        assert 7 < location < 253, "the track left the finite observation area before control was learned"
    assert response_at is not None and identity_at is not None and response_at < identity_at < 3.0
    assert hers.carried and hers.keys_known(list(keys))
    assert run.counter_trial.done
    assert all(any(key == other_key for _at, key in actions) for other_key in keys[1:])
    assert now[0] == 4.8
