"""A heading response cannot lend another body its controls or hide a contradiction."""
import math
from types import SimpleNamespace

import pytest

from core.agency.controls_ready_for_the_motion_model import has_a_measured_response
from core.agency.facing_seen_under_its_input import observe_facing
from core.agency.response_in_the_body_frame import bound_facing_response, expected_travel
from core.agency.turning_while_travelling import TravelTurn
from core.agency.which_one_answers_to_her import WhichIsHers
from core.agency.which_way_she_faces import WhichWaySheFaces

pytestmark = pytest.mark.unit


def _bound():
    hers, faces = WhichIsHers(), WhichWaySheFaces()
    hers.number, hers.kind = 1, 1
    hers.holding('drive', 0.)
    body = SimpleNamespace(number=1, vx=80., vy=0., seen=.3, motion_evidence='direct')
    observe_facing(faces, body, hers, .3)
    faces.along['drive'] = [80., 80.]
    faces.along['back'] = [-60., -60.]
    for key, rate in (('', 0.), ('bend', 1.4)):
        faces.travel_turns.trials.extend(TravelTurn(key, began, began + .6, rate) for began in (-2., -1.))
    return hers, faces, body


@pytest.mark.parametrize('change', ['body', 'kind', 'epoch', 'lost'])
def test_changed_body_or_experiment_cannot_use_old_heading_measurements(change):
    hers, faces, body = _bound()
    assert has_a_measured_response(hers, 'bend')
    if change == 'body':
        hers.number = body.number = 2
    elif change == 'kind':
        hers.kind = 2
    elif change == 'epoch':
        hers.identification.reset('new experiment')
    else:
        hers.number = None
    assert bound_facing_response(hers) is None
    assert not has_a_measured_response(hers, 'bend')
    observe_facing(faces, body if hers.number is not None else None, hers, .4)
    assert not faces.along and not faces.turns and not faces.travel_turns.trials
    assert not faces.turning_keys()


def test_a_known_travel_turn_predicts_the_tangent_before_this_picture():
    hers, faces, body = _bound()
    expected = expected_travel(faces, 'bend', .4)
    assert expected == pytest.approx((80 * math.cos(.14), 80 * math.sin(.14)))
    assert has_a_measured_response(hers, 'bend')
    assert faces.way_of('bend') is None
    assert expected_travel(faces, 'bend', .7) is None
    body.vx = body.vy = 0.
    body.seen = .4
    observe_facing(faces, body, hers, .4)
    assert expected_travel(faces, 'bend', .5) is None


def test_turning_and_driving_supply_two_axes_only_when_both_controls_are_available():
    hers, _faces, _body = _bound()
    assert hers.axes(['drive', 'bend']) == (True, True)
    assert hers.axes(['bend']) == (False, False)


def test_repeated_straight_travel_does_not_forget_observed_heading_dependence():
    faces = WhichWaySheFaces()
    faces.turns = [({'yaw': .5}, -1.)] * 2
    angles = [0., math.pi / 2, math.pi] + [0.] * 25
    for index, angle in enumerate(angles):
        began = index * .8
        for after in (0., .25, .3, .4):
            faces.saw(80 * math.cos(angle), 80 * math.sin(angle), 'drive', began + after, began=began)
        faces.saw(0., 0., '', began + .5, began=began + .5)
    assert not faces.spread() and faces.heading_dependent and faces.turns_her()
    faces.turns.clear()
    assert not faces.turns_her()


def test_a_prediction_or_unclaimed_body_cannot_supply_the_heading_model():
    hers, faces, body = _bound()
    body.seen, body.motion_evidence, body.vx, body.vy = .4, 'unseen', 0., 80.
    body.motion_measured_at = None
    observe_facing(faces, body, hers, .4)
    assert expected_travel(faces, 'bend', .5) is None
    hers.number = None
    body.seen, body.motion_evidence = .5, 'direct'
    observe_facing(faces, body, hers, .5)
    assert faces.facing is None


def test_body_frame_checks_still_reject_two_controls_that_stop_answering():
    hers, faces, body = _bound()
    body.x = body.y = 50.
    body.vx = body.vy = 0.
    for key in ('drive', 'back'):
        hers._hers.add(key, 80. if key == 'drive' else -60., 0.)
    for step in range(1, 121):
        at = .3 + step * .03
        key = 'drive' if step < 61 else 'back'
        # Freeze the established heading while its claimed body stands still.
        # Current stationary motion is passed to the identity check, without
        # allowing that check to overwrite its own expectation first.
        faces.travel_turns.last_at = at - .03
        hers._answering(body, key, .5, at)
        if hers.number is None:
            break
    assert hers.number is None and 1 in hers.not_mine


def test_the_body_heading_boundary_has_a_registered_measured_claim():
    from core.agency.response_in_the_body_frame import _body_heading_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _body_heading_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'heading_response_belongs_to_its_body_and_experiment')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
