"""A response forecast uses actual input intervals rather than a delayed velocity."""
import math

import pytest

from core.agency.what_acknowledged_inputs_predict import (
    VectorInput,
    _acknowledged_prediction_invariant,
    input_clock_uncertainty,
    position_from_inputs,
)

pytestmark = pytest.mark.unit


def test_reversal_and_release_are_integrated_in_their_actual_intervals():
    history = (VectorInput(1., (100.,)), VectorInput(1.05, (-100.,)), VectorInput(1.17, (0.,)))
    actual = position_from_inputs((40.,), 1., 1.2, history, ((0., 100.),))
    assert actual is not None and math.isclose(actual[0], 33.)
    assert _acknowledged_prediction_invariant() == ()


def test_saturation_is_applied_before_the_following_reverse_input():
    history = (VectorInput(1., (100.,)), VectorInput(1.1, (-100.,)))
    actual = position_from_inputs((95.,), 1., 1.2, history, ((0., 100.),))
    assert actual is not None and math.isclose(actual[0], 90.)


def test_vector_responses_can_use_different_state_channels():
    history = (VectorInput(1., (10., -5., 2.)), VectorInput(1.1, (0., 0., 0.)))
    actual = position_from_inputs((1., 2., 3.), 1., 1.2, history, ((0., 10.),) * 3)
    assert actual == pytest.approx((2., 1.5, 3.2))


def test_request_acknowledgement_uncertainty_bounds_an_earlier_actual_reversal():
    received = (VectorInput(1., (100.,)), VectorInput(1.17, (-100.,), 1.05))
    earliest = (VectorInput(1., (100.,)), VectorInput(1.05, (-100.,)))
    nominal = position_from_inputs((40.,), 1., 1.2, received, ((0., 100.),))
    actual = position_from_inputs((40.,), 1., 1.2, earliest, ((0., 100.),))
    uncertainty = input_clock_uncertainty(1., 1.2, received)
    assert uncertainty == pytest.approx((24.,))
    assert abs(nominal[0] - actual[0]) <= uncertainty[0]


def test_an_unknown_response_cannot_receive_zero_clock_uncertainty():
    assert input_clock_uncertainty(1., 1.2, (VectorInput(1., None),)) is None
    assert input_clock_uncertainty(1., 1.2, (VectorInput(1., (float('nan'),)),)) is None
    assert input_clock_uncertainty(1., 1.2, (VectorInput(1., (1.,)), VectorInput(1.1, (2.,), True))) is None


@pytest.mark.parametrize('history', [(), (VectorInput(1.1, (1.,)),),
    (VectorInput(1., (1.,)), VectorInput(1.1, None)),
    (VectorInput(1., (float('nan'),)),), (VectorInput(1.1, (1.,)), VectorInput(1., (1.,)))])
def test_missing_nonfinite_and_reordered_intervals_remain_unknown(history):
    assert position_from_inputs((4.,), 1., 1.2, history, ((0., 10.),)) is None


@pytest.mark.parametrize('bounds', [None, ((0.,),), ((10., 0.),), ((0., float('nan')),), 'bounds'])
def test_unmeasured_or_malformed_bounds_do_not_produce_a_forecast(bounds):
    assert position_from_inputs((4.,), 1., 1.2, (VectorInput(1., (1.,)),), bounds) is None


def test_the_input_interval_claim_is_registered():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'acknowledged_inputs_cover_the_response_interval')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
