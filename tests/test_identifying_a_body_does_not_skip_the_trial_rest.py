"""The actual actor closes its probe before learning and using body-relative steering."""
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
async def test_the_actual_actor_finishes_its_motion_probe_and_learns_the_turning_controls(monkeypatch, hz, angle, rolling):
    now, actions = [0.], []
    monkeypatch.setattr(play.time, 'monotonic', lambda: now[0])

    class Hands:
        async def down(self, key):
            actions.append((now[0], key))

        async def up(self, key):
            actions.append((now[0], ''))

        async def tap(self, key):
            raise AssertionError('a separate tap interrupted the measurement')

    look = np.zeros(64)
    look[1] = 1.
    body = Thing(1, 1500., 1500., 14., 10., look, (140, 170, 220),
                 np.zeros((10, 14, 3), np.uint8), 0., 0., kind=1, moved=True)
    moves = SimpleNamespace(shape=(4000, 4000), things={1: body}, seeing=True,
                            kinds=[Kind(1, look, 140, body.colour)])
    hers, meeting = WhichIsHers(), WhatMeetingDoes()
    run = play._Run(keys=['drive', 'back', 'yaw-a', 'yaw-b'], began=0., faces=WhichWaySheFaces(),
                    pressures=Pressures(), comes=WhatComesForHer(), gone_for=GoneFor())
    speed = 0.
    for step in range(1, int(20 * hz) + 1):
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
        hers.saw(moves, [], now[0])
        choosing = play._Choosing(moves, hers, meeting, run.keys)
        play._what_the_place_does(run, choosing, moves, hers, meeting, now[0])
        await play._act(Hands(), run, moves, hers, meeting, choosing, now[0])
        if run.faces.turns_her():
            break
    assert hers.number == body.number and run.faces.turns_her() and now[0] < 20.
    assert run.faces.ahead_key() == 'drive'
    assert run.faces.turning_keys()['yaw-a'] == pytest.approx(-2.8)
    assert len(actions) > 9 and any(key == '' for _at, key in actions)


def test_the_rest_obligation_has_a_registered_measured_claim():
    from core.agency.inputs_while_measuring_control import _body_trial_rest_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _body_trial_rest_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'an_observed_body_trial_finishes_with_its_rest')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
