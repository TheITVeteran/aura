"""Arrival needs enough time for the rest of its state to become ready."""
import math

import pytest

from core.agency.controls_that_take_time_to_build import (
    ChannelRule,
    GradualControls,
    MotionState,
    _stored_response_invariant,
)
from core.agency.room_to_finish_a_required_motion import (
    AfterBraking,
    _preparation_room_invariant,
    after_braking,
    time_to_prepare,
    transfer_from_rest,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("direction", [-1., 1.])
@pytest.mark.parametrize("rate,brake", [(4., 2.), (3., .5), (0., 2.)])
def test_cancelling_movement_projects_its_coupled_state(direction, rate, brake):
    initial = 10.
    stopped = after_braking(initial, rate * direction, brake, other_values=(6., 8.), other_changes=(-3., -.5))
    assert stopped is not None
    position, speed, first, second = initial, rate * direction, 6., 8.
    dt = stopped.time / 100 if stopped.time else 0.
    for _ in range(100):
        position += speed * dt - .5 * direction * brake * dt ** 2
        speed -= direction * brake * dt
        first -= 3. * dt
        second -= .5 * dt
    assert stopped.position == pytest.approx(position)
    assert stopped.other_values == pytest.approx((first, second))
    assert speed == pytest.approx(0.)


@pytest.mark.parametrize("distance,acceleration,limit", [(8., 2., 4.), (40., 2., 4.),
                                                        (-40., 2., 4.), (2., 3., 20.), (0., 2., 4.)])
def test_a_rest_to_rest_transfer_has_the_claimed_time(distance, acceleration, limit):
    seconds = transfer_from_rest(distance, acceleration, limit)
    ramp = min(math.sqrt(abs(distance) / acceleration), limit / acceleration)
    top = acceleration * ramp
    coast = max(0., seconds - 2 * ramp)
    # Independent areas under the accelerating, coasting and braking rates.
    travelled = .5 * acceleration * ramp ** 2 + top * coast + .5 * top * ramp
    assert travelled == pytest.approx(abs(distance))
    assert top <= limit


def test_preparation_includes_stopping_before_transferring():
    stopped = after_braking(10., 4., 2.)
    assert time_to_prepare(stopped, 50., 4., 2., 4.) == pytest.approx(12.)
    assert time_to_prepare(stopped, 16., 4., 2., 4.) == 2.


@pytest.mark.parametrize("bad", [0., -1., math.nan, math.inf])
def test_unmeasured_braking_cannot_supply_a_stopping_place(bad):
    assert after_braking(10., 4., bad) is None
    assert transfer_from_rest(10., bad, 4.) is None


def test_missing_coupling_and_invalid_tolerance_remain_unavailable():
    assert after_braking(10., 4., 2., other_values=(6.,), other_changes=()) is None
    assert time_to_prepare(AfterBraking(14., (), 2.), 50., -1., 2., 4.) is None


@pytest.mark.parametrize("decay,drift", [(1., .2), (.8, 0.), (.5, .1), (0., .2), (-.5, .1), (1.0000001, .01)])
@pytest.mark.parametrize("ticks", [0, 1, 20, 200])
def test_a_fast_forecast_agrees_with_separate_physical_ticks(decay, drift, ticks):
    model = GradualControls(((ChannelRule(decay=decay, drift=drift), ChannelRule(decay=.7)),),
                           ((1., -1.), (-.5, 2.)), ((0., 1.),))
    state = MotionState((10., 20.), (1., 2.))
    predicted = model.after(state, 0, ticks)
    for _ in range(ticks):
        state = model.advance(state, 0)
    assert predicted.position == pytest.approx(state.position, abs=1e-7)
    assert predicted.channels == pytest.approx(state.channels, abs=1e-7)


def test_a_targeted_channel_uses_bounded_ticks_instead_of_a_wrong_affine_sum():
    model = GradualControls(((ChannelRule(target=4., change=.3),),), ((1.,),))
    state = MotionState((0.,), (0.,))
    predicted = model.after(state, 0, 40)
    for _ in range(40):
        state = model.advance(state, 0)
    assert predicted == state


def test_a_forecast_over_preparation_can_show_an_earlier_constraint_crossing():
    model = GradualControls(((ChannelRule(drift=.1), ChannelRule(decay=.8)),), ((1., -1.),))
    state = MotionState((0.,), (1., 0.))
    needed = time_to_prepare(AfterBraking(0., (), 0.), 40., 0., 2., 4.)
    assert needed == 12.
    future = model.after(state, 0, math.ceil(needed))
    assert future.position[0] > 15., "contact would precede the time needed to align"
    # Keeping enough stored opposing response buys the time instead.
    supported = model.after(MotionState((0.,), (1., 4.)), 0, math.ceil(needed))
    assert supported.position[0] < 15.


@pytest.mark.parametrize("ticks", [-1, 1.5, True, 10001])
def test_a_forecast_cannot_allocate_unbounded_physical_ticks(ticks):
    model = GradualControls(((ChannelRule(),),), ((1.,),))
    with pytest.raises(ValueError):
        model.after(MotionState((0.,), (1.,)), 0, ticks)


def test_even_a_zero_tick_forecast_requires_a_measured_finite_state():
    model = GradualControls(((ChannelRule(),),), ((1.,),))
    with pytest.raises(ValueError):
        model.after(MotionState((math.nan,), (1.,)), 0, 0)


def test_large_finite_parameters_cannot_publish_an_infinite_forecast():
    model = GradualControls(((ChannelRule(decay=2.),),), ((1.,),))
    with pytest.raises(ValueError, match="not finite"):
        model.after(MotionState((0.,), (1.,)), 0, 10000)
    assert transfer_from_rest(8., 2., 1e200) == 4.
    assert time_to_prepare(AfterBraking(0., (), 1e308), 1e308, 0., 1., 1.) is None


def test_the_registered_mathematics_runs_both_measurements():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _stored_response_invariant() == _preparation_room_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    for name in ("motion_forecasts_keep_released_response", "arrival_preparation_includes_the_time_to_stop"):
        check = next(t for t in suite.tests() if t.name == name)
        assert check.predict(None) is True
        assert any(c.test == name for c in suite.claims())
