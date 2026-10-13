"""Slow observed travel can support an uncertain physical hypothesis."""
import math

import pytest

from core.agency.a_model_for_steering import ProvisionalSteeringBank, _arc, _slow_motion_invariant

OWNER = (7, 1., 2)
KEYS = ('drive', 'turn')


def model():
    result = ProvisionalSteeringBank()
    result.arm(['turn'], body_extent=16.)
    return result


def slow_path(bank, *, key='drive', began=0., start=0., frames=21, dt=.05, speed=4., error=1., owner=OWNER):
    for index in range(frames):
        at = start + index * dt
        bank.saw(owner, key, began, at, speed * at, 0., 0., 0., position_error=error)
    return at


@pytest.mark.parametrize('speed', [3., 4., 7.])
@pytest.mark.parametrize('cadence', [1 / 120, 1 / 40, .1])
def test_accumulated_travel_recovers_a_driver_when_each_velocity_is_unresolved(speed, cadence):
    bank = model()
    at = slow_path(bank, dt=cadence, frames=math.ceil(1.5 / cadence) + 1, speed=speed)
    assert bank.driver(KEYS, now=at) == 'drive'
    assert not bank.estimates and not bank._samples
    action = bank.action(KEYS, 'drive', (50., 30.), now=at)
    assert action is not None and action.status == 'provisional'
    assert all(p.linear_error > 0 for p in action.predictions)


def test_unresolved_raster_jitter_never_becomes_a_driver():
    bank = model()
    for index in range(100):
        at = .025 * index
        bank.saw(OWNER, 'drive', 0., at, 10. + index % 2, 10., 0., 0., position_error=1.)
    assert bank.driver(KEYS) is None
    assert bank.action(KEYS, 'drive', (50., 30.)) is None


def test_endpoint_error_must_be_overcome_before_a_slow_heading_exists():
    bank = model()
    slow_path(bank, frames=11, speed=4.)
    assert bank.driver(KEYS) is None  # Two pixels of travel and two pixels of endpoint error.
    bank.saw(OWNER, 'drive', 0., .6, 2.4, 0., 0., 0., position_error=1.)
    assert bank.driver(KEYS) == 'drive'


def test_resolved_turn_receipts_do_not_acquire_a_driver_role():
    bank = model()
    slow_path(bank, key='turn')
    assert bank._pose.heading is not None
    assert bank.driver(KEYS) is None
    assert bank.action(KEYS, 'turn', (50., 30.)) is None


def test_arming_a_previous_drive_as_a_turn_removes_its_drive_role():
    bank = model()
    slow_path(bank)
    assert bank.driver(KEYS) == 'drive'
    bank.arm(KEYS, body_extent=16.)
    assert bank.driver(KEYS) is None and 'drive' not in bank._drivers


def test_crossing_input_receipts_cannot_accumulate_a_slow_response():
    bank = model()
    for index in range(30):
        at = .1 * index
        bank.saw(OWNER, 'drive', at, at, 4 * at, 0., 0., 0., position_error=1.)
    assert bank.driver(KEYS) is None


@pytest.mark.parametrize('new_owner', [(7, 2., 2), (7, 1., 3), (8, 1., 2)])
def test_displacement_before_an_owner_change_cannot_complete_the_new_anchor(new_owner):
    bank = model()
    slow_path(bank, frames=10)
    bank.saw(new_owner, 'drive', 0., .5, 2., 0., 0., 0., position_error=1.)
    assert bank.driver(KEYS) is None
    assert len(bank._positions) == 1


def test_a_long_gap_discards_the_anchor_and_driver():
    bank = model()
    slow_path(bank)
    bank.saw(OWNER, 'drive', 0., 1.5, 6., 0., 0., 0., position_error=1.)
    assert bank.driver(KEYS) is None and len(bank._positions) == 1


def test_missing_observation_requires_fresh_displacement_evidence():
    bank = model()
    slow_path(bank)
    bank.saw(OWNER, 'drive', 0., 1.05, 4.2, 0., 0., 0., measured=False)
    bank.saw(OWNER, 'drive', 0., 1.1, 4.4, 0., 0., 0., position_error=1.)
    assert bank.driver(KEYS) is None and len(bank._positions) == 1


def test_a_stale_picture_or_response_cannot_select_a_driver():
    bank = model()
    slow_path(bank)
    assert bank.driver(KEYS, now=1.4) is None
    assert bank.driver(KEYS, now=.9) is None
    bank.saw(OWNER, 'turn', 1., 1.3, 5.2, 0., 0., 0., position_error=1.)
    bank.saw(OWNER, 'turn', 1., 1.4, 5.6, 0., 0., 0., position_error=1.)
    assert bank.driver(KEYS, now=1.4) is None


def test_accumulation_has_bounded_storage_and_time_at_a_high_frame_rate():
    bank = model()
    at = slow_path(bank, dt=.001, frames=5000)
    assert bank.driver(KEYS, now=at) == 'drive'
    assert len(bank._positions) <= 128
    assert at - bank._positions[0][0] <= 8 * bank.longest_gap


def test_chord_and_pixel_uncertainty_reach_each_forecast_and_clearance_check():
    bank = model()
    slow_path(bank)
    seen = []

    def clearance(x, y, error):
        seen.append((x, y, error))
        return error <= 1.

    assert bank.action(KEYS, 'drive', (50., 30.), clearance=clearance) is None
    assert seen and seen[0][2] > 1.
    action = bank.action(KEYS, 'drive', (50., 30.))
    assert action is not None
    for prediction in action.predictions:
        assert prediction.path[-1][3] >= 1. + prediction.linear_error * action.duration


def test_the_chord_heading_keeps_a_curvature_allowance():
    bank = model()
    speed, rate = 4., .2
    for index in range(31):
        at = .05 * index
        x, y = _arc(0., 0., 0., speed, rate, at)
        bank.saw(OWNER, 'drive', 0., at, x, y, 0., 0., position_error=.05)
    pose = bank._pose
    assert pose is not None and pose.heading is not None
    assert abs(pose.heading - rate * pose.at) <= pose.heading_error
    assert pose.speed_error > 0
    assert not bank.estimates


def test_a_velocity_crossing_input_delivery_does_not_select_a_driver():
    bank = model()
    bank.saw(OWNER, '', 0., .1, 0., 0., 0., 0., position_error=.01)
    bank.saw(OWNER, 'drive', .15, .2, 1., 0., 10., 0., position_error=.01)
    assert bank.driver(KEYS) is None
    bank.saw(OWNER, 'drive', .15, .3, 2., 0., 10., 0., position_error=.01)
    assert bank.driver(KEYS) == 'drive'


def test_slow_motion_invariant():
    assert _slow_motion_invariant() == ()
