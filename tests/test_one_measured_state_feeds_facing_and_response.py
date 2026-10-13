"""Facing and control feedback read the same current measured state as identification."""
import math

import pytest

from core.agency.facing_seen_under_its_input import observe_facing
from core.agency.which_one_answers_to_her import WhichIsHers
from core.agency.which_way_she_faces import WhichWaySheFaces
from core.perception.motion_kept_between_pictures import (
    current_motion_estimate,
    observe_thing_motion,
)
from tests.test_measured_motion_proposes_visual_correspondence import _tracked_body


def _owned_body():
    eyes, body = _tracked_body()
    hers = WhichIsHers()
    hers.number, hers.kind = body.number, body.kind
    hers.holding('drive', 1., trying=True)
    return eyes, body, hers


def test_current_measured_motion_reaches_facing_and_continuity_instead_of_a_false_raw_direction():
    eyes, body, hers = _owned_body()
    state = current_motion_estimate(body, body.seen)
    assert state is not None and state.velocity_error < .5 * math.hypot(*state.velocity)
    body.vx, body.vy = -800., 800.
    faces = WhichWaySheFaces()
    observe_facing(faces, body, hers, body.seen)
    hers.saw(eyes, [], body.seen)
    assert faces.facing == pytest.approx(0., abs=.01)
    assert faces.going[-1] == pytest.approx(state.velocity)
    assert hers.last_velocity == pytest.approx(state.velocity)
    assert (body.vx, body.vy) == (-800., 800.)


def test_a_current_state_prevents_false_control_contradiction_from_the_older_velocity():
    eyes, body, hers = _owned_body()
    hers.lowest, hers.highest = [0., 0.], [2000., 2000.]
    for began in (-2., -1.):
        for _ in range(8):
            hers._hers.add('drive', 80., 0., press=began)
    for _ in range(40):
        body.x += 4.
        body.seen += .05
        body.motion_measured_at = body.seen
        body.vx, body.vy = -800., 800.
        observe_thing_motion(body, body.seen, eyes.shape)
        hers.saw(eyes, [], body.seen)
    assert hers.number == body.number
    assert hers._answered['drive'] == pytest.approx([1.] * len(hers._answered['drive']), abs=.02)
    assert not any(r['reason'] == 'observed control contradiction' for r in hers.identification.receipts)


@pytest.mark.parametrize('departure', ['birth', 'missing', 'time'])
def test_a_departed_state_cannot_supply_a_heading_measurement(departure):
    _eyes, body, hers = _owned_body()
    at = body.seen
    body.vx, body.vy = 0., 80.
    if departure == 'birth':
        body.born += .5
    elif departure == 'missing':
        body.motion_evidence, body.motion_measured_at = 'unavailable', None
    else:
        at += .1
    assert current_motion_estimate(body, at) is None
    faces = WhichWaySheFaces()
    observe_facing(faces, body, hers, at)
    if departure == 'birth':
        assert faces.facing == pytest.approx(math.pi / 2)
    else:
        assert faces.facing is None and not faces.going
