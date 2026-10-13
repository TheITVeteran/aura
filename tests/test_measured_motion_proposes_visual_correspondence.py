"""Motion proposals keep appearance and observation custody as independent checks."""
import copy
import math
from collections import deque

import numpy as np
import pytest

from core.perception.motion_kept_between_pictures import (
    _motion_proposal_invariant,
    observe_thing_motion,
    predicted_box_from_motion,
)
from core.perception.what_moves_in_the_picture import Thing, WhatMoves, _back_in_sight


def _tracked_body(number=1):
    eyes = WhatMoves(width=200)
    look = np.full(64, 1 / 64)
    body = Thing(number, 20., 40., 10., 12., look, (60, 170, 220),
                 np.full((12, 10, 3), (60, 170, 220), np.uint8), born=0., seen=1., kind=0)
    body.path = deque(maxlen=100)
    for at in np.arange(1., 2., .05):
        body.seen = body.motion_measured_at = float(at)
        body.motion_evidence = 'measured'
        body.x, body.y = 20. + 80. * (at - 1.), 40.
        body.path.append((float(at), body.x, body.y))
        observe_thing_motion(body, float(at), (120, 200))
    eyes.things[number] = body
    eyes.shape = (120, 200)
    return eyes, body


def _blob(body, x):
    return dict(x=x, y=body.y, w=body.w, h=body.h, look=body.look.copy(),
                colour=body.colour, patch=body.patch.copy())


def test_measured_motion_reaches_both_candidate_search_and_association_cost():
    eyes, body = _tracked_body()
    # A changing outline can contaminate an older two-picture derivative.
    body.vx, body.vy = -400., 0.
    at = body.seen + .15
    prediction = predicted_box_from_motion(body, at)
    assert prediction is not None
    assert prediction[0] == pytest.approx(body.x + 12., abs=1.)
    blob = _blob(body, prediction[0])
    assert math.isfinite(eyes._cost(body, blob, at))
    matched, remaining = eyes._match([blob], at)
    assert matched == {body.number: 0} and not remaining
    # The erroneous raw speed cannot expand a fallback leap to unrelated pixels.
    distant = _blob(body, 190.)
    assert not math.isfinite(eyes._cost(body, distant, at))
    assert not math.isfinite(eyes._leap_cost(body, distant, at))
    assert body.seen == body.motion_measured_at < at


def test_the_last_measured_box_survives_projection_of_a_missing_picture():
    _eyes, body = _tracked_body()
    before = predicted_box_from_motion(body, body.seen + .15)
    body.x, body.y = -900., 500.
    after = predicted_box_from_motion(body, body.seen + .15)
    assert after == before


def test_an_unmeasured_detection_can_keep_a_proposal_without_refreshing_control_state():
    from core.perception.motion_kept_between_pictures import current_motion_estimate

    _eyes, body = _tracked_body()
    observed_at = body.seen
    before = predicted_box_from_motion(body, observed_at + .2)
    body.seen += .1
    body.x, body.y = -900., 500.
    body.motion_evidence, body.motion_measured_at = 'unavailable', None
    body.motion_estimate = None
    assert predicted_box_from_motion(body, observed_at + .2) == before
    assert current_motion_estimate(body, body.seen) is None
    assert body.motion_belief.at == observed_at
    assert body.measured_motion_box.state.observed_at == observed_at
    assert predicted_box_from_motion(body, observed_at + 1.1) is None


def test_an_unmeasured_replacement_cannot_borrow_the_proposal_of_another_birth():
    _eyes, body = _tracked_body()
    body.born = body.seen
    body.motion_evidence, body.motion_measured_at = 'unavailable', None
    assert predicted_box_from_motion(body, body.seen + .2) is None


@pytest.mark.parametrize('departure', ['shape', 'appearance', 'time', 'reference'])
def test_a_trajectory_cannot_override_departed_visible_evidence(departure):
    eyes, body = _tracked_body()
    at = body.seen + .15
    blob = _blob(body, predicted_box_from_motion(body, at)[0])
    if departure == 'shape':
        blob['w'], blob['h'] = 1., 100.
    elif departure == 'appearance':
        blob['look'] = np.zeros(64)
        blob['look'][0] = 1.
    elif departure == 'time':
        assert predicted_box_from_motion(body, body.seen + 1.1) is None
        return
    else:
        body.motion_position = (body.x, body.y)
        assert predicted_box_from_motion(body, at) is None
        return
    assert not math.isfinite(eyes._cost(body, blob, at))


def test_an_observed_return_can_keep_identity_across_visual_reclassification():
    eyes, body = _tracked_body()
    eyes.things.clear()
    eyes.out_of_sight[body.number] = (body, body.seen + .26)
    at = body.seen + .35
    candidate = copy.deepcopy(body)
    candidate.number, candidate.kind = 99, 4
    candidate.x = predicted_box_from_motion(body, at)[0]
    assert _back_in_sight(eyes, candidate, at) is body
    assert body.seen < at and body.motion_measured_at < at


def test_overlapping_predictions_cannot_choose_an_identity_by_iteration_order():
    eyes, body = _tracked_body()
    _other_eyes, other = _tracked_body(2)
    eyes.things.clear()
    eyes.out_of_sight = {1: (body, body.seen + .26), 2: (other, body.seen + .26)}
    at = body.seen + .35
    candidate = copy.deepcopy(body)
    candidate.number = 99
    candidate.x = predicted_box_from_motion(body, at)[0]
    assert _back_in_sight(eyes, candidate, at) is None
    assert set(eyes.out_of_sight) == {1, 2}


def test_the_motion_proposal_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _motion_proposal_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'motion_proposals_keep_the_last_measured_box')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())


def test_pixels_and_independent_input_trials_identify_the_body_and_feed_steering():
    from core.agency.steering_from_a_model import observed_steering_model
    from core.agency.which_one_answers_to_her import WhichIsHers
    from core.agency.which_way_she_faces import WhichWaySheFaces
    from core.perception.motion_kept_between_pictures import current_motion_estimate
    from tests.test_source_points_keep_their_visible_detail import render

    eyes, hers, faces = WhatMoves(width=160), WhichIsHers(), WhichWaySheFaces()
    at, step, x = 1., .05, 80.
    for _ in range(12):
        at += step
        hers.saw(eyes, eyes.see(render(present=False), at), at)
    qualified = []
    for key, duration in [('', .6)] + [trial for _ in range(3)
            for trial in [('opaque-a', .8), ('', .6), ('opaque-b', .8), ('', .6)]]:
        hers.holding(key, at, trying=True)
        for _ in range(round(duration / step)):
            at += step
            x += (60. if key == 'opaque-a' else -60. if key == 'opaque-b' else 0.) * step
            pixels = render((x, 70.), angle=.01 * at)
            distractor = render((220 + 5 * np.sin(at), 140.), seed=41, replacement=True)
            mask = np.any(distractor != 25, axis=2)
            pixels[mask] = distractor[mask]
            hers.saw(eyes, eyes.see(pixels, at), at)
            body = hers.thing(eyes)
            observed_steering_model(faces, body, hers, eyes, at)
            estimate = current_motion_estimate(body, at) if body is not None else None
            if estimate is not None and faces.steering_model._pose is not None:
                qualified.append((body.x, body.y, x * .5))
    # No oracle set hers.number; known coordinates only adjudicate the result.
    assert len(qualified) >= 70
    assert all(abs(px - expected) < 2. and abs(py - 35.) < 2. for px, py, expected in qualified)
    decisions = [r for r in hers.identification.receipts if r.get('selected') is not None]
    assert decisions and decisions[0]['reason'] == 'independent repeated trials'
    assert sum(n >= 2 for n in decisions[0]['independent_trials'].values()) >= 2
