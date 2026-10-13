"""Physical release closes an owned forecast at its measured observation span."""
import math
from dataclasses import replace

import pytest

from core.agency.a_model_for_steering import ProvisionalSteeringBank, _arc

OWNER = (13, 4., 7)


def pulse(*, turning=False, began=.3):
    model = ProvisionalSteeringBank()
    model.arm(['opaque-turn'], body_extent=24.)
    model.saw(OWNER, 'opaque-drive', 0., .3, 0., 0., 60., 0., position_error=.01)
    keys = ['opaque-drive', 'opaque-turn'] if turning else ['opaque-drive']
    target = (0., 100.) if turning else (100., 0.)
    action = model.action(keys, 'opaque-drive', target)
    assert action is not None and model.delivered(action, began)
    return model, action


def picture(model, action, at, *, began=.3, rate=0., x=None, y=None, error=.01):
    after = at - began
    observed = _arc(60 * max(0., began - .3), 0., 0., 60., rate, after)
    model.saw(OWNER, action.key, began, at,
              observed[0] if x is None else x, observed[1] if y is None else y,
              60 * math.cos(rate * after), 60 * math.sin(rate * after), position_error=error)


def test_a_pulse_ending_before_the_full_next_picture_keeps_its_partial_feedback():
    model, action = pulse()
    picture(model, action, .4)
    assert model.pending is action and model.last_result == 'partial consistent'
    assert model.ended(action, .3, .45)
    assert model.pending is None and model.covered_span == pytest.approx((.3, .4))
    model.saw(OWNER, '', .45, .6, 18., 0., 60., 0., position_error=.01)
    assert model.last_result == 'partial consistent'
    assert model.covered_span == pytest.approx((.3, .4))


def test_an_end_with_no_picture_during_the_pulse_is_unobserved():
    model, action = pulse()
    assert model.ended(action, .3, .45)
    assert model.last_result == 'unobserved' and model.covered_span is None
    assert not model.estimates
    model.saw(OWNER, '', .45, .6, 18., 0., 60., 0., position_error=.01)
    assert model.last_result == 'unobserved' and model.covered_span is None


def test_a_picture_after_release_cannot_teach_or_contradict_the_old_receipt():
    model, action = pulse(turning=True)
    picture(model, action, .35, rate=action.predictions[0].rate)
    assert model.ended(action, .3, .36)
    observed = model.covered_span
    assert model.last_result == 'partial consistent'
    # Even a delayed caller labelling a later picture with the old receipt
    # cannot reopen the acknowledged physical input interval.
    picture(model, action, .4, x=500., y=500.)
    picture(model, action, .45, x=-500., y=-500.)
    assert model.last_result == 'partial consistent' and model.covered_span == observed
    assert action.key not in model.estimates
    assert not model.delivered(action, .3)
    assert not model.ended(action, .3, .5)


def test_partial_uncertainty_scales_with_its_elapsed_time():
    model, action = pulse(turning=True)
    assert action.predictions[0].path[-1][3] > 3.
    # One pixel would fit the full-pulse uncertainty. At a quarter of the
    # duration its angular contribution is only one sixteenth as large.
    picture(model, action, .35, x=3., y=1.)
    assert model.pending is action and model.last_result == 'partial contradicted'
    assert model.ended(action, .3, .36)
    assert model.last_result == 'partial contradicted'
    assert model.covered_span == pytest.approx((.3, .35))


@pytest.mark.parametrize('changed_key', ['', 'opaque-drive', 'another-key'])
def test_input_change_keeps_only_the_previous_receipts_picture(changed_key):
    model, action = pulse(turning=True)
    picture(model, action, .4, rate=action.predictions[0].rate)
    model.saw(OWNER, changed_key, .45, .6, 500., 500., 60., 0., position_error=.01)
    assert model.pending is None and model.last_result == 'partial consistent'
    assert model.covered_span == pytest.approx((.3, .4))


def test_a_new_receipt_that_predates_saved_feedback_cannot_retain_it():
    model, action = pulse()
    picture(model, action, .4)
    model.saw(OWNER, '', .35, .45, 9., 0., 60., 0., position_error=.01)
    assert model.pending is None and model.last_result == 'unobserved'
    assert model.covered_span is None


def test_a_release_that_predates_saved_feedback_cannot_retain_it():
    model, action = pulse()
    picture(model, action, .4)
    assert model.ended(action, .3, .35)
    assert model.last_result == 'unobserved' and model.covered_span is None


@pytest.mark.parametrize('bad_end', [(.31, .45), (.3, .29), (.3, math.inf), (.3, math.nan)])
def test_an_end_must_match_a_real_active_receipt_and_finite_window(bad_end):
    model, action = pulse()
    picture(model, action, .4)
    assert not model.ended(action, *bad_end)
    assert model.pending is action and model.last_result == 'partial consistent'
    assert model.covered_span == pytest.approx((.3, .4))


def test_a_different_prediction_cannot_close_the_active_pulse():
    model, action = pulse()
    other = replace(action, prediction_id=action.prediction_id + 1)
    assert not model.ended(other, .3, .45)
    assert model.pending is action
    assert model.ended(action, .3, .45)


def test_delivery_cannot_change_the_receipt_of_an_active_prediction():
    model, action = pulse()
    assert model.delivered(action, .3)
    assert not model.delivered(action, .35)
    picture(model, action, .4)
    assert model.ended(action, .3, .45)
    assert model.last_result == 'partial consistent'


@pytest.mark.parametrize('owner', [(13, 5., 7), (13, 4., 8), (14, 4., 7), None])
def test_changed_ownership_invalidates_partial_feedback_and_the_end_acknowledgement(owner):
    model, action = pulse()
    picture(model, action, .4)
    model.saw(owner, action.key, .3, .45, 9., 0., 60., 0., position_error=.01)
    assert model.pending is None and model.last_result == 'unobserved'
    assert model.covered_span is None and not model.ended(action, .3, .46)


@pytest.mark.parametrize('at,measured,result', [(.45, False, 'unobserved'), (.8, True, 'observation gap')])
def test_missing_or_gapped_photos_invalidate_the_saved_partial_window(at, measured, result):
    model, action = pulse()
    picture(model, action, .4)
    model.saw(OWNER, action.key, .3, at, 9., 0., 60., 0., position_error=.01, measured=measured)
    assert model.pending is None and model.last_result == result
    assert model.covered_span is None and not model.ended(action, .3, at)


def test_a_full_forecast_requires_a_picture_while_the_same_receipt_is_still_active():
    model, action = pulse()
    picture(model, action, .4)
    picture(model, action, .5)
    assert model.pending is None and model.last_result == 'consistent'
    assert model.covered_span == pytest.approx((.3, .5))
    assert model.ended(action, .3, .55)
    assert model.last_result == 'consistent'
    assert model.covered_span == pytest.approx((.3, .5))
    assert not model.ended(action, .3, .6)


def test_a_completed_forecasts_release_closes_future_rate_samples_too():
    model, action = pulse(turning=True)
    rate = action.predictions[0].rate
    picture(model, action, .4, rate=rate)
    picture(model, action, .5, rate=rate)
    assert model.pending is None and model.last_result == 'consistent'
    measured = model.estimates[action.key]
    assert model.ended(action, .3, .55)
    picture(model, action, .6, rate=-7.)
    picture(model, action, .65, rate=-7.)
    assert model.last_result == 'consistent'
    assert model.estimates[action.key] == measured and not model._samples
    assert model.covered_span == pytest.approx((.3, .5))


def test_actual_dispatch_time_and_interval_boundary_bound_partial_coverage():
    model, action = pulse(began=.35)
    model.saw(OWNER, 'opaque-drive', 0., .34, 2.4, 0., 60., 0., position_error=.01)
    picture(model, action, .4, began=.35)
    assert model.covered_span == pytest.approx((.35, .4))
    assert not model.estimates
    picture(model, action, .45, began=.35)
    assert model.ended(action, .35, .46)
    assert model.last_result == 'partial consistent'
    assert model.covered_span == pytest.approx((.35, .45))


def test_first_post_delivery_position_verifies_without_fitting_crossing_velocity():
    model, action = pulse(turning=True, began=.305)
    picture(model, action, .4, began=.305, rate=action.predictions[0].rate)
    assert model.pending is action and model.last_result == 'partial consistent'
    assert model.covered_span == pytest.approx((.305, .4))
    assert not model.estimates and not model._samples
    assert model.ended(action, .305, .45)
    assert model.last_result == 'partial consistent'


def test_an_action_not_acknowledged_as_delivered_cannot_receive_forecast_feedback():
    model, delivered = pulse()
    assert model.ended(delivered, .3, .35)
    action = model.action(['opaque-drive'], 'opaque-drive', (100., 0.))
    assert action is not None
    picture(model, action, .4, began=.36)
    assert model.pending is action and model.covered_span is None
    assert model.last_result == 'unobserved' and not model.ended(action, .36, .45)
