"""The actual actor learns an invented plant and reaches targets without losing its body."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency import playing_as_it_happens as play
from core.agency.a_thing_gone_for import GoneFor
from core.agency.what_comes_for_her import WhatComesForHer
from core.agency.what_meeting_things_does import WhatMeetingDoes
from core.agency.what_the_place_does_to_her import Pressures
from core.agency.which_one_answers_to_her import WhichIsHers
from core.agency.which_way_she_faces import WhichWaySheFaces
from core.perception.what_moves_in_the_picture import Kind, Thing

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize('hz', [10., 30.])
@pytest.mark.parametrize('angle', [0., 1.2])
@pytest.mark.parametrize('rolling', [False, True])
async def test_four_contacts_keep_the_continuously_identified_body(monkeypatch, hz, angle, rolling):
    now = [0.]
    monkeypatch.setattr(play.time, 'monotonic', lambda: now[0])

    class Hands:
        async def down(self, key):
            pass

        async def up(self, key):
            pass

        async def tap(self, key):
            raise AssertionError('a separate input interrupted navigation')

    look = np.zeros(64)
    look[1] = 1.
    body = Thing(1, 1500., 1500., 14., 10., look, (140, 170, 220),
                 np.zeros((10, 14, 3), np.uint8), 0., 0., kind=1, moved=True)
    moves = SimpleNamespace(shape=(4000, 4000), things={1: body}, seeing=True,
                            kinds=[Kind(1, look, 140, body.colour)])
    hers, meeting = WhichIsHers(), WhatMeetingDoes()
    run = play._Run(keys=['drive', 'back', 'yaw-a', 'yaw-b'], began=0., faces=WhichWaySheFaces(),
                    pressures=Pressures(), comes=WhatComesForHer(), gone_for=GoneFor())
    speed, contacts, target, identified = 0., 0, None, False
    offsets = [(200., 150.), (-200., 150.), (-200., -150.), (200., -150.)]
    for step in range(1, int(90 * hz) + 1):
        now[0] = step / hz
        run.intervals.append(1 / hz)
        if run.held == 'yaw-a' and (not rolling or speed):
            angle -= 2.8 / hz
        if run.held == 'yaw-b' and (not rolling or speed):
            angle += 2.8 / hz
        speed = (120. if run.held == 'drive' else -60. if run.held == 'back'
                 else speed * math.exp(-.7 / hz) if rolling else 0.)
        if rolling and abs(speed) < 5.:
            speed = 0.
        body.vx, body.vy = speed * math.cos(angle), speed * math.sin(angle)
        body.x, body.y = body.x + body.vx / hz, body.y + body.vy / hz
        body.seen = now[0]
        body.path.append((now[0], body.x, body.y))
        if target is not None:
            target.seen = now[0]
        hers.saw(moves, [], now[0])
        if identified:
            assert hers.number == 1, 'the actor lost the body during its own trajectory'
        identified |= hers.number == 1
        choosing = play._Choosing(moves, hers, meeting, run.keys)
        play._what_the_place_does(run, choosing, moves, hers, meeting, now[0])
        await play._act(Hands(), run, moves, hers, meeting, choosing, now[0])
        if not run.faces.turns_her():
            continue
        if (target is not None and abs(body.x - target.x) <= (body.w + target.w) / 2
                and abs(body.y - target.y) <= (body.h + target.h) / 2):
            contacts += 1
            target = None
            moves.things.pop(2)
            if contacts == len(offsets):
                break
        if target is None:
            dx, dy = offsets[contacts]
            target_look = np.zeros(64)
            target_look[20] = 1.
            target = Thing(2, body.x + dx, body.y + dy, 40., 40., target_look, (80, 160, 80),
                           np.zeros((40, 40, 3), np.uint8), now[0], now[0], kind=2)
            moves.things[2] = target
            moves.kinds.append(Kind(2, target_look, 1600, target.colour))
            meeting.told[2] = 'meet'
    assert contacts == 4 and identified and hers.number == 1, str({
        'body': (body.x, body.y), 'target': (target.x, target.y) if target else None,
        'held': run.held, 'turners': run.faces.turning_keys(), 'ahead': run.faces.ahead_key(),
        'goal': choosing.target()[0], 'trials': run.facing_trials, 'turns': run.faces.turns,
        'facing': run.faces.facing_now(), 'real_heading': angle,
    })
    assert not hers.not_mine and hers.identification.epoch == 0
