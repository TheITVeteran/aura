"""Filtered motion uncertainty and explicit drive roles survive their controller boundary."""
import math
from types import SimpleNamespace

import pytest

from core.agency.a_model_for_steering import ProvisionalSteeringBank
from core.agency.steering_from_a_model import _turn_keys
from core.agency.which_way_she_faces import WhichWaySheFaces
from core.cognition.a_guide_to_a_place import THE_GUIDE
from core.perception.how_a_program_plays import AKey, HowItPlays


def test_an_estimated_velocity_keeps_its_error_instead_of_being_differentiated_again():
    bank = ProvisionalSteeringBank()
    bank.arm(['bend'], body_extent=12.)
    for at in (1., 1.01, 1.02, 1.03):
        bank.saw((1, 0., 0), 'drive', .5, at, 50 + 30 * (at - 1), 50., 30., 0.,
                 position_error=2., measured_velocity_error=3.)
    # Two-pixel errors over .01 s would imply 400 px/s uncertainty if the
    # already fitted velocity were treated as a new two-picture difference.
    assert bank.driver(['drive', 'bend']) == 'drive'
    assert bank._pose.speed_error == 3.
    assert bank._pose.heading_error == pytest.approx(math.asin(.1))
    action = bank.action(['drive', 'bend'], 'drive', (90., 60.), now=1.03)
    assert action is not None
    assert all(prediction.linear_error >= 3. for prediction in action.predictions)


def test_the_first_state_and_large_uncertainty_do_not_create_a_confident_heading():
    bank = ProvisionalSteeringBank()
    bank.saw((1, 0., 0), 'drive', .5, 1., 50., 50., 30., 0., measured_velocity_error=3.)
    assert bank._pose.heading_error == pytest.approx(math.asin(.1))
    assert bank.driver(['drive']) is None  # no complete owned input interval
    bank.saw((1, 0., 0), 'drive', .5, 1.1, 51., 50., 10., 0., measured_velocity_error=30.)
    assert bank._drivers['drive'].anchored is False


@pytest.mark.parametrize('error', [math.nan, math.inf, -1., False])
def test_invalid_uncertainty_retires_the_current_pose(error):
    bank = ProvisionalSteeringBank()
    bank.saw((1, 0., 0), 'drive', .5, 1., 50., 50., 30., 0., measured_velocity_error=3.)
    bank.saw((1, 0., 0), 'drive', .5, 1.1, 53., 50., 30., 0., measured_velocity_error=error)
    assert bank._pose is None and not bank._drivers


def _source(receiver='Body', method='accelerate'):
    return HowItPlays(hers={'class': 'Body'}, keys={
        'drive': AKey('drive', 'moves', f'{receiver}.{method}()', f'{receiver}.controls'),
        'bend': AKey('bend', 'moves', 'Body.rotate()', 'Body.controls')})


def test_a_curved_drive_does_not_erase_the_explicit_non_rotation_control():
    faces = WhichWaySheFaces()
    faces.travel_turns = SimpleNamespace(curved=lambda _key: True)
    guide = SimpleNamespace(code_says=_source(), controls={})
    token = THE_GUIDE.set(guide)
    try:
        assert _turn_keys(SimpleNamespace(keys=['drive', 'bend']), faces) == {'bend'}
    finally:
        THE_GUIDE.reset(token)


def test_measured_velocity_uncertainty_has_a_registered_claim():
    from core.agency.a_model_for_steering import _velocity_uncertainty_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _velocity_uncertainty_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'steering_preserves_measured_velocity_uncertainty')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())


@pytest.mark.parametrize('receiver,method', [('Background', 'accelerate'), ('Body', 'getAcceleration'), ('Body', 'turnAndAccelerate')])
def test_a_mismatched_query_or_explicit_combined_role_cannot_exclude_a_turn_prior(receiver, method):
    faces = WhichWaySheFaces()
    faces.travel_turns = SimpleNamespace(curved=lambda _key: True)
    token = THE_GUIDE.set(SimpleNamespace(code_says=_source(receiver, method), controls={}))
    try:
        assert 'drive' in _turn_keys(SimpleNamespace(keys=['drive', 'bend']), faces)
    finally:
        THE_GUIDE.reset(token)
