"""A usable kinematic hypothesis remains separate from measured control authority."""
import math

import pytest

from core.agency.a_model_for_steering import ProvisionalSteeringBank, _arc

OWNER = (7, 1., 2)


def bank(speed=60., error=.01):
    result = ProvisionalSteeringBank()
    result.arm(['opaque-z', 'opaque-a'], body_extent=24.)
    result.saw(OWNER, 'opaque-drive', 0., .3, 0., 0., speed, 0., position_error=error)
    return result


def observe_arc(model, key, began, angle, rate, speed=60., error=.01, count=4, dt=.1, origin=(0., 0.)):
    for index in range(1, count + 1):
        after = dt * index
        x, y = _arc(*origin, angle, speed, rate, after)
        direction = angle + rate * after
        model.saw(OWNER, key, began, began + after, x, y,
                  speed * math.cos(direction), speed * math.sin(direction), position_error=error)


def test_one_current_drive_response_supports_a_bounded_provisional_action():
    model = bank()
    action = model.action(['opaque-z', 'opaque-drive', 'opaque-a'], 'opaque-drive', (0., 100.))
    assert action is not None and action.key in model.turn_keys
    assert action.owner == OWNER and action.status == 'provisional'
    assert 0 < action.duration <= .2
    assert {p.model for p in action.predictions} >= {'moving turn prior', 'no effect', 'autonomous curve'}
    assert not model.estimates


def test_unarmed_controls_do_not_acquire_turn_roles():
    model = bank()
    model.arm([])
    action = model.action(['opaque-drive', 'fire-looking-but-opaque'], 'opaque-drive', (0., 100.))
    assert action is not None and action.key == 'opaque-drive'


def test_wrong_sign_prior_is_corrected_by_actual_owned_motion():
    model = bank()
    action = model.action(['opaque-z', 'opaque-drive', 'opaque-a'], 'opaque-drive', (0., 100.))
    assert action is not None
    assert model.delivered(action, .3)
    observe_arc(model, action.key, .3, 0., -1.3, count=3)
    fit = model.estimates[action.key]
    assert fit.rate == pytest.approx(-1.3) and fit.uncertainty < .1
    observe_arc(model, '', .6, -.39, 0., count=3, origin=_arc(0., 0., 0., 60., -1.3, .3))
    next_action = model.action(['opaque-z', 'opaque-drive', 'opaque-a'], 'opaque-drive', (0., 100.))
    assert next_action is not None and next_action.key != action.key


@pytest.mark.parametrize('rate', [-4.7, -.8, .35, 2.4])
@pytest.mark.parametrize('speed', [5., 30., 80.])
def test_current_rate_fit_is_not_tied_to_a_game_or_speed(rate, speed):
    model = bank(speed)
    observe_arc(model, 'opaque-a', .3, 0., rate, speed=speed, count=5)
    assert model.estimates['opaque-a'].rate == pytest.approx(rate)
    assert model.estimates['opaque-a'].observations >= 2


@pytest.mark.parametrize('scale', [1., .375, .2])
@pytest.mark.parametrize('cadence', [1 / 30, .1, .25])
def test_quantized_picture_uncertainty_bounds_the_fitted_rate(scale, cadence):
    speed, rate, error = 80 * scale, 2.4, math.sqrt(.5)
    model = bank(speed, error)
    previous = (0., 0.)
    for index in range(1, 13):
        after = cadence * index
        x, y = _arc(0., 0., 0., speed, rate, after)
        current = (round(x), round(y))
        observed = ((current[0] - previous[0]) / cadence, (current[1] - previous[1]) / cadence)
        model.saw(OWNER, 'opaque-a', .3, .3 + after, *current, *observed, position_error=error)
        previous = current
    fit = model.estimates.get('opaque-a')
    if scale == 1. and cadence == .1:
        assert fit is not None
    if fit is not None:
        assert abs(fit.rate - rate) <= fit.uncertainty + 1e-8


def test_neutral_curve_competes_with_and_retires_controlled_turn_prior():
    model = bank()
    observe_arc(model, 'opaque-a', .3, 0., .8)
    endpoint = _arc(0., 0., 0., 60., .8, .4)
    observe_arc(model, '', .7, .32, .8, origin=endpoint)
    assert 'opaque-a' in model.retired_keys
    assert model.estimates['opaque-a'].rate == pytest.approx(model.estimates[''].rate)


def test_straight_no_effect_is_not_kept_as_a_turn_control():
    model = bank()
    observe_arc(model, 'opaque-a', .3, 0., 0.)
    observe_arc(model, '', .7, 0., 0., origin=(24., 0.))
    assert 'opaque-a' in model.retired_keys


@pytest.mark.parametrize('new_owner', [(7, 2., 2), (7, 1., 3), (8, 1., 2)])
def test_birth_epoch_and_track_changes_invalidate_frozen_predictions(new_owner):
    model = bank()
    action = model.action(['opaque-a', 'opaque-drive'], 'opaque-drive', (0., 100.))
    model.saw(new_owner, 'opaque-drive', .3, .4, 6., 0., 60., 0., position_error=.01)
    assert model.pending is None and not model.estimates
    assert not model.delivered(action, .3)


def test_missing_motion_and_a_long_gap_do_not_verify_a_prediction():
    model = bank()
    action = model.action(['opaque-a', 'opaque-drive'], 'opaque-drive', (0., 100.))
    model.saw(OWNER, action.key, .3, .4, 0., 0., 0., 0., measured=False)
    assert model.pending is None and model.last_result == 'unobserved'
    assert model.action(['opaque-a', 'opaque-drive'], 'opaque-drive', (0., 100.)) is None
    model.saw(OWNER, 'opaque-drive', 1., 1.1, 0., 0., 60., 0., position_error=.01)
    action = model.action(['opaque-a', 'opaque-drive'], 'opaque-drive', (0., 100.))
    model.saw(OWNER, action.key, 1.1, 2., 0., 0., 60., 0., position_error=.01)
    assert model.pending is None and model.last_result == 'observation gap'


def test_a_body_always_seen_stationary_has_no_invented_heading():
    model = ProvisionalSteeringBank()
    model.arm(['opaque-a'], body_extent=24.)
    for at in (.3, .4, .5):
        model.saw(OWNER, 'opaque-drive', 0., at, 10., 10., 0., 0., position_error=.01)
    assert model.action(['opaque-a', 'opaque-drive'], 'opaque-drive', (0., 100.)) is None


def test_clearance_applies_to_all_competing_paths():
    model = bank()
    seen = []
    def clearance(x, y, error):
        seen.append((x, y, error))
        return y >= -1e-8
    assert model.action(['opaque-a', 'opaque-drive'], 'opaque-drive', (0., 100.), clearance=clearance) is None
    assert seen and all(math.isfinite(v) for row in seen for v in row)


def test_dispatch_delay_and_changed_input_cannot_verify_an_old_forecast():
    model = bank()
    action = model.action(['opaque-a', 'opaque-drive'], 'opaque-drive', (0., 100.), delay_s=.05)
    assert action is not None and model.delivered(action, .35)
    model.saw(OWNER, 'opaque-drive', 0., .34, 2.4, 0., 60., 0., position_error=.01)
    assert model.pending is action
    model.saw(OWNER, 'another-input', .36, .5, 12., 0., 60., 0., position_error=.01)
    assert model.pending is None and model.last_result == 'input changed'


@pytest.mark.parametrize('horizon', [0., -1., math.inf, math.nan])
def test_invalid_or_unbounded_action_times_are_not_used(horizon):
    model = bank()
    assert model.action(['opaque-a', 'opaque-drive'], 'opaque-drive', (0., 100.), horizon=horizon) is None


@pytest.mark.parametrize('speed,turn_rate', [(5., .35), (30., .8), (80., 2.4), (30., 4.7)])
def test_one_controller_makes_progress_on_different_invented_plants(speed, turn_rate):
    model = bank(speed)
    x, y, heading, at = 0., 0., 0., .3
    target = (0., 50.)
    best = math.dist((x, y), target)
    keys = ['opaque-z', 'opaque-drive', 'opaque-a']
    for _ in range(80):
        action = model.action(keys, 'opaque-drive', target)
        assert action is not None
        began = at
        assert model.delivered(action, began)
        actual_rate = {'opaque-a': -turn_rate, 'opaque-z': turn_rate}.get(action.key, 0.)
        start = x, y, heading
        for index in (1, 2):
            after = action.duration * index / 2
            x, y = _arc(start[0], start[1], start[2], speed, actual_rate, after)
            heading = start[2] + actual_rate * after
            at = began + after
            model.saw(OWNER, action.key, began, at, x, y, speed * math.cos(heading),
                      speed * math.sin(heading), position_error=.001)
        best = min(best, math.dist((x, y), target))
        if best < 10:
            break
    assert best < 25


def test_cartesian_response_contradicts_a_body_turn_forecast():
    model = bank()
    action = model.action(['opaque-a', 'opaque-drive'], 'opaque-drive', (0., 100.))
    assert model.delivered(action, .3)
    for at, y in ((.4, 6.), (.5, 12.)):
        model.saw(OWNER, action.key, .3, at, 0., y, 0., 60., position_error=.01)
    assert model.last_result == 'contradicted'
    observe_arc(model, '', .5, math.pi / 2, 0., count=3, origin=(0., 12.))
    assert action.key in model.retired_keys


def test_stationary_turn_prior_demands_a_following_drive_observation():
    model = bank()
    model.saw(OWNER, '', .3, .4, 0., 0., 0., 0., position_error=.01)
    action = model.action(['opaque-a', 'opaque-drive'], 'opaque-drive', (0., 100.))
    assert action is not None and action.key == 'opaque-a'
    assert action.predictions[0].model == 'stationary turn prior'
    assert model.delivered(action, .4)
    for at in (.5, .6):
        model.saw(OWNER, action.key, .4, at, 0., 0., 0., 0., position_error=.01)
    assert model.last_result == 'heading unobserved'
    following = model.action(['opaque-a', 'opaque-drive'], 'opaque-drive', (0., 100.))
    assert following is not None and following.key == 'opaque-drive'


def test_an_explicitly_stale_picture_cannot_issue_an_action():
    model = bank()
    assert model.action(['opaque-a', 'opaque-drive'], 'opaque-drive', (0., 100.), now=.8) is None


def test_a_later_incompatible_picture_retires_a_prefix_rate_fit():
    model = bank()
    observe_arc(model, 'opaque-a', .3, 0., .8, count=2)
    assert 'opaque-a' in model.estimates
    model.saw(OWNER, 'opaque-a', .3, .6, 18., 0., 0., 60., position_error=.001)
    assert 'opaque-a' not in model.estimates


def test_a_stop_cannot_join_the_angular_samples_before_and_after_it():
    model = bank()
    observe_arc(model, 'opaque-a', .3, 0., .8, count=2)
    model.saw(OWNER, 'opaque-a', .3, .6, 12., 0., 0., 0., position_error=.001)
    assert not model._samples
    model.saw(OWNER, 'opaque-a', .3, .7, 12., 6., 0., 60., position_error=.001)
    assert len(model._samples) == 1
