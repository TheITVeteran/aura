"""Motion from noisy pixels, with bounded predictions and current control custody."""
import math
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency.steering_from_a_model import observed_steering_model
from core.agency.which_one_answers_to_her import WhichIsHers
from core.agency.which_way_she_faces import WhichWaySheFaces
from core.perception.motion_kept_between_pictures import (
    MotionBetweenPictures,
    _motion_belief_invariant,
    current_motion_estimate,
    observe_thing_motion,
)


@pytest.mark.parametrize('acceleration', [0., 20., -30., 80.])
@pytest.mark.parametrize('seed', [4, 17, 41])
def test_position_only_observations_reconstruct_motion_better_than_two_frame_differences(acceleration, seed):
    rng = np.random.default_rng(seed)
    belief = MotionBetweenPictures(scale=240.)
    estimated, differences, forecasts, baseline = [], [], [], []
    last = None
    at = 1.
    for index in range(80):
        dt = rng.uniform(.035, .12)
        at += dt
        t = at - 1
        truth = np.array((30. + 40 * t + .5 * acceleration * t * t, 60. + 10 * t))
        velocity = np.array((40. + acceleration * t, 10.))
        pixels = truth + rng.normal(0., .75, 2)
        state = belief.observe((1, 0.), ('pixels',), at, tuple(pixels), error=.75)
        if index >= 12:
            estimated.append(np.linalg.norm(state.velocity - velocity))
            raw = (pixels - last) / dt
            differences.append(np.linalg.norm(raw - velocity))
            future = belief.forecast(at + .2)
            actual = truth + velocity * .2 + np.array((.5 * acceleration * .2 ** 2, 0.))
            forecasts.append(np.linalg.norm(future.position - actual))
            baseline.append(np.linalg.norm(pixels + raw * .2 - actual))
            assert future.observed_at == at and not future.current
        last = pixels
    assert np.mean(estimated) < .5 * np.mean(differences)
    assert np.mean(forecasts) < .65 * np.mean(baseline)


def test_missing_pictures_grow_uncertainty_without_refreshing_the_state():
    belief = MotionBetweenPictures(scale=100.)
    for at in np.arange(1., 2., .1):
        belief.observe((7, 0.), ('pixels',), float(at), (40. * at, 10.))
    last = belief.at
    before = belief.forecast(last)
    short = belief.forecast(last + .2)
    longer = belief.forecast(last + .6)
    assert before.current and not short.current and not longer.current
    assert before.position_error < short.position_error < longer.position_error
    assert belief.at == last and belief.observations == 10
    assert belief.forecast(last + 1.1) is None and belief.forecast(last - .01) is None


@pytest.mark.parametrize('change', ['owner', 'basis', 'gap'])
def test_new_identity_reference_or_long_gap_cannot_inherit_measured_derivatives(change):
    belief = MotionBetweenPictures(scale=100.)
    for at in (1., 1.1, 1.2, 1.3):
        belief.observe((1, 0.), ('points',), at, (30. * at, 10.))
    owner, basis, at = (1, 1.4) if change == 'owner' else (1, 0.), ('box',) if change == 'basis' else ('points',), 3. if change == 'gap' else 1.4
    state = belief.observe(owner, basis, at, (50., 10.))
    assert state.observations == 1 and not state.current
    assert state.velocity == (0., 0.)


def test_a_jump_that_neither_motion_law_explains_restarts_derivatives():
    belief = MotionBetweenPictures(scale=100.)
    for at in np.arange(1., 3., .1):
        belief.observe((1, 0.), ('pixels',), float(at), (20. * at, 10.), error=.25)
    state = belief.observe((1, 0.), ('pixels',), 3., (900., -700.), error=.25)
    assert state.status == 'motion changed' and state.observations == 1 and not state.current
    assert state.velocity == (0., 0.)


def test_repeated_and_invalid_measurements_do_not_change_history():
    belief = MotionBetweenPictures()
    assert belief.observe((1,), ('pixels',), 1., (0., 0.))
    for at, position in [(1., (20., 30.)), (.9, (20., 30.)), (math.nan, (0., 0.)), (2., (math.inf, 0.))]:
        assert belief.observe((1,), ('pixels',), at, position) is None
    assert belief.at == 1. and belief.observations == 1
    assert _motion_belief_invariant() == ()


def test_resolution_scaling_keeps_the_same_motion_and_uncertainty_in_view_units():
    beliefs = [MotionBetweenPictures(scale=scale) for scale in (100., 400.)]
    for at in np.arange(1., 2., .1):
        states = [b.observe((1,), ('pixels',), float(at), (b.scale * at, b.scale * .3), error=b.scale * .005)
                  for b in beliefs]
    assert np.allclose(np.asarray(states[0].velocity) / 100., np.asarray(states[1].velocity) / 400.)
    assert states[0].position_error / 100. == pytest.approx(states[1].position_error / 400.)


def test_current_public_state_reaches_the_steering_model_and_missing_motion_teaches_nothing():
    hers, faces = WhichIsHers(), WhichWaySheFaces()
    hers.number, hers.kind = 1, 0
    hers.holding('opaque', 0., trying=True)
    body = SimpleNamespace(number=1, born=0., x=0., y=40., w=12., h=12., kind=0,
                           vx=800., vy=-800., seen=0., motion_evidence='measured', motion_measured_at=0.,
                           motion_position=None, dims=deque(), motion_belief=None, motion_estimate=None)
    for at in np.arange(1., 2., .1):
        body.seen = body.motion_measured_at = float(at)
        body.x = 60. * at
        observe_thing_motion(body, float(at), (160, 240))
    at = body.seen
    estimate = current_motion_estimate(body, at)
    assert estimate is not None and estimate.velocity_error < .5 * math.hypot(*estimate.velocity)
    observed_steering_model(faces, body, hers, SimpleNamespace(), at)
    assert faces.steering_model._pose.vx == pytest.approx(estimate.velocity[0])
    assert faces.steering_model._pose.vy == pytest.approx(estimate.velocity[1])
    assert faces.steering_model._pose.speed_error == pytest.approx(estimate.velocity_error)
    body.motion_evidence, body.motion_measured_at = 'unavailable', None
    assert current_motion_estimate(body, at) is None
    observed_steering_model(faces, body, hers, SimpleNamespace(), at + .1)
    assert faces.steering_model._pose is None


def test_rendered_rotation_and_acceleration_feed_state_through_the_ordinary_visual_observer():
    from core.perception.what_moves_in_the_picture import WhatMoves
    from tests.test_source_points_keep_their_visible_detail import render

    eyes = WhatMoves(width=160)
    for index in range(10):
        eyes.see(render(present=False), 1. + index * .1)
    identity, estimates = None, []
    for index in range(40):
        elapsed, at = index * .06, 2. + index * .06
        center = np.array((100 + 35 * elapsed, 60 + 4 * elapsed + 10 * elapsed ** 2))
        eyes.see(render(center, angle=.01 * index), at)
        visible = [t for t in eyes.things.values() if t.seen == at]
        assert visible
        # The oracle selects the rendered body for this component test;
        # neither the detector nor the estimator receives its true state.
        body = min(visible, key=lambda t: math.dist((t.x, t.y), center * .5))
        identity = identity or (body.number, body.born)
        assert (body.number, body.born) == identity
        estimate = current_motion_estimate(body, at)
        if estimate is not None:
            estimates.append(estimate)
    assert len(estimates) >= 30
    expected = np.array((17.5, 2 + 10 * elapsed))
    assert np.linalg.norm(np.asarray(estimates[-1].velocity) - expected) < 3.
    hers, faces = WhichIsHers(), WhichWaySheFaces()
    hers.number, hers.kind = body.number, body.kind
    hers.holding('opaque', 0., trying=True)
    observed_steering_model(faces, body, hers, eyes, at)
    assert faces.steering_model._pose.vx == pytest.approx(estimates[-1].velocity[0])
    assert faces.steering_model._pose.vy == pytest.approx(estimates[-1].velocity[1])
    forecast = body.where_at(.2)
    displacement = np.subtract(body.motion_belief.forecast(at + .2).position, estimates[-1].position)
    assert np.allclose(forecast, np.asarray((body.x, body.y)) + displacement)


def test_motion_belief_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'motion_forecasts_do_not_renew_observations')
    assert check.predict(None)
    assert any(c.test == check.name for c in suite.claims())


def test_preparing_the_observer_does_not_establish_a_picture_or_body():
    from core.perception.what_moves_in_the_picture import WhatMoves

    eyes = WhatMoves()
    before = eyes.pictures, eyes.has_looked, dict(eyes.things), eyes.activity.at
    from core.perception.what_moves_in_the_picture import prepare_play_vision

    prepare_play_vision()
    assert (eyes.pictures, eyes.has_looked, dict(eyes.things), eyes.activity.at) == before


def _reference_body_with_history():
    from tests.test_a_measured_patch_can_keep_a_point_reference import _blob, _established

    eyes, body = _established()
    observe_thing_motion(body, 1.2, (100, 180))
    for at, x in ((1.3, 45.), (1.4, 47.)):
        eyes._moved(body, _blob(x, 22), at)
        observe_thing_motion(body, at, (100, 180))
    assert current_motion_estimate(body, 1.4) is not None
    return eyes, body


def test_an_independently_bridged_replacement_keeps_the_motion_reference():
    from tests.test_a_measured_patch_can_keep_a_point_reference import _blob, _motion

    eyes, body = _reference_body_with_history()
    basis = body.motion_estimate.basis
    for at, total, x in ((1.5, 8., 49.), (1.6, 10., 51.)):
        eyes._point_motion[1] = _motion(body.seen, at, total, names=(5, 6, 7, 8))
        eyes._moved(body, _blob(x, 22), at)
        observe_thing_motion(body, at, (100, 180))
        assert body.point_pose.reference_at == body.visible_reference.reference_at == 1.1
        assert body.motion_estimate.basis == basis
    assert body.motion_estimate.observations == 5


def test_an_unbridged_replacement_cannot_inherit_the_old_motion():
    from tests.test_a_measured_patch_can_keep_a_point_reference import _blob, _motion

    eyes, body = _reference_body_with_history()
    old = body.motion_estimate.basis
    for at, total, x in ((1.5, 8., 49.), (1.6, 10., 51.)):
        eyes._point_motion[1] = _motion(body.seen, at, total, names=(5, 6, 7, 8))
        eyes._moved(body, _blob(x, 22, colour=70), at)
        observe_thing_motion(body, at, (100, 180))
        assert current_motion_estimate(body, at) is None
    assert body.motion_estimate.basis != old
    assert body.motion_estimate.basis[-1] == 1.5
    assert body.motion_estimate.observations == 1
    assert body.motion_estimate.velocity == (0., 0.)


@pytest.mark.parametrize('change', ['unknown', 'different'])
def test_a_current_estimate_cannot_qualify_under_a_changed_or_unknown_reference(change):
    from dataclasses import replace

    _eyes, body = _reference_body_with_history()
    body.visible_reference = replace(body.visible_reference, reference_at=None if change == 'unknown' else 1.2)
    assert current_motion_estimate(body, body.seen) is None
    observe_thing_motion(body, body.seen, (100, 180))
    assert current_motion_estimate(body, body.seen) is None


def test_a_replaced_coordinate_reference_also_retires_the_controller_history():
    from dataclasses import replace

    _eyes, body = _reference_body_with_history()
    hers, faces = WhichIsHers(), WhichWaySheFaces()
    hers.number, hers.kind = body.number, body.kind
    hers.holding('opaque', 0., trying=True)
    observed_steering_model(faces, body, hers, SimpleNamespace(), body.seen)
    old = faces.steering_model
    body.seen = body.motion_measured_at = 1.5
    body.visible_reference = replace(body.visible_reference, at=1.5, reference_at=1.5)
    observed_steering_model(faces, body, hers, SimpleNamespace(), body.seen)
    assert faces.steering_model is not old
    assert faces.steering_basis == ('points', 1.5)
    current = faces.steering_model
    body.seen = body.motion_measured_at = 1.6
    body.visible_reference = replace(body.visible_reference, at=1.6, reference_at=None)
    observed_steering_model(faces, body, hers, SimpleNamespace(), body.seen)
    assert faces.steering_model is not current and faces.steering_model._pose is None
    observe_thing_motion(body, body.seen, (100, 180))
    assert current_motion_estimate(body, body.seen) is None
