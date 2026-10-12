"""A succession of similar objects cannot share a pointer-control experiment."""
import math
from types import SimpleNamespace

import pytest

from core.agency.which_one_answers_to_her import WhichIsHers

pytestmark = pytest.mark.unit


def _thing(number, x, at, *, born=0., measured=True):
    return SimpleNamespace(number=number, kind=3, born=born, seen=at, x=x, y=80.,
                           w=12., h=16., moved=True, motion_measured_at=at if measured else None,
                           motion_evidence='measured' if measured else 'unavailable')


def test_nearest_members_of_a_stream_cannot_supply_a_body_that_follows_the_pointer():
    hers = WhichIsHers()
    for step in range(150):
        at = step * .04
        pointer = 120. + 75. * math.sin(step / 6.)
        hers.pointed(pointer, 80., at)
        things = {n: _thing(n, 10. + n * 40. - 3. * step, at) for n in range(1, 18)
                  if 0. < 10. + n * 40. - 3. * step < 240.}
        hers._what_follows_the_pointer(SimpleNamespace(things=things, shape=(160, 240)), at)
        assert hers.number is None and not hers.follows_pointer


@pytest.mark.parametrize('breaks', ['unmeasured', 'gap', 'birth', 'number'])
def test_separate_short_sightings_cannot_establish_pointer_control(breaks):
    hers = WhichIsHers()
    for step in range(120):
        segment, index = divmod(step, 10)
        at = step * .04 + (segment if breaks == 'gap' else 0.)
        x = 120. + 70. * math.sin(step / 2.)
        hers.pointed(x, 80., at)
        body = _thing(segment + 1 if breaks == 'number' else 1, x, at,
                      born=float(segment) if breaks == 'birth' else 0.,
                      measured=breaks != 'unmeasured' or index != 9)
        hers._what_follows_the_pointer(SimpleNamespace(things={body.number: body}, shape=(160, 240)), at)
        assert hers.number is None and not hers.follows_pointer


def test_one_measured_body_can_follow_the_pointer_through_repeated_turns():
    hers = WhichIsHers()
    for step in range(120):
        at, x = step * .04, 120. + 75. * math.sin(step / 6.)
        hers.pointed(x, 80., at)
        body = _thing(1, x, at)
        hers._what_follows_the_pointer(SimpleNamespace(things={1: body}, shape=(160, 240)), at)
    assert hers.number == 1 and hers.follows_pointer


def test_registered_pointer_continuity_invariant():
    from core.agency.what_follows_the_pointer import _pointer_body_continuity_invariant

    assert _pointer_body_continuity_invariant() == ()


def test_recovering_a_pointer_body_uses_observed_reach_instead_of_the_nearest_lookalike():
    hers = WhichIsHers()
    hers.kind, hers.last_seen, hers.last_seen_at = 3, (40., 80.), 1.
    hers.last_shape, hers.last_size = (12., 16.), 192.
    hers.pointed(180., 80., 1.1)
    near, far = _thing(1, 46., 1.1), _thing(2, 180., 1.1)
    moves = SimpleNamespace(things={1: near, 2: far})
    assert hers._under_the_pointer(moves, 1.1) == 1
    del moves.things[1]
    assert hers._under_the_pointer(moves, 1.1) is None
    near.motion_measured_at, near.motion_evidence = None, 'unavailable'
    moves.things[1] = near
    assert hers._under_the_pointer(moves, 1.1) is None
