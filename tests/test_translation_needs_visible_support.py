"""Moving proposals cannot teach controls or physics without measured translation."""
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception.points_that_keep_their_motion import PointMotion
from core.perception.translation_from_visible_support import translation_is_observed
from core.perception.what_moves_in_the_picture import Thing, WhatMoves

pytestmark = pytest.mark.unit


def _boundary(width=12, height=12):
    from core.perception.boundary_support_for_translation import visible_boundary

    image = np.full((120, 200, 3), 30, np.uint8)
    image[40:40 + height, 30:30 + width] = 200
    return visible_boundary(image, 30, 40, 30 + width, 40 + height, np.ones((height, width), bool))


def _thing():
    patch = np.full((12, 12, 3), 200, np.uint8)
    body = Thing(1, 40., 50., 12., 12., np.array([1., 0.]), (200, 200, 200), patch, 0., 1., kind=0,
                 support=np.ones((12, 12), bool), boundary=_boundary())
    body.path.append((1., 40., 50.))
    body.motion_path.extend([(.9, 40., 50.), (1., 40., 50.)])
    body.motion_measured_at, body.motion_evidence = 1., 'measured'
    return body


def _blob(body, *, x=None, w=12.):
    return dict(x=body.x if x is None else x, y=body.y, w=w, h=12., look=body.look.copy(),
                colour=body.colour, patch=np.full((12, int(w), 3), 200, np.uint8),
                support=np.ones((12, int(w)), bool), boundary=_boundary(int(w)))


@pytest.mark.parametrize('dx,dy', [(5., 0.), (-5., 0.), (0., 5.), (0., -5.)])
def test_plain_translated_objects_have_measured_movement_without_texture_points(dx, dy):
    body = _thing()
    blob = _blob(body, x=body.x + dx)
    blob['y'] += dy
    assert translation_is_observed(body, blob, 1.1)
    WhatMoves()._moved(body, blob, 1.1)
    assert body.motion_measured_at == 1.1
    assert body.vx == pytest.approx(dx / .1) and body.vy == pytest.approx(dy / .1)


def test_a_changed_box_around_unmoved_points_does_not_measure_translation():
    body = _thing()
    moves = WhatMoves()
    blob = _blob(body, x=45., w=22.)
    moves._point_motion[1] = PointMotion(0., 0., .01, 4, 1., 1.1)
    assert moves._cost(body, blob, 1.1) < float('inf')
    moves._moved(body, blob, 1.1)
    assert body.x == 45. and body.seen == 1.1
    assert body.motion_measured_at is None and not body.motion_path
    assert body.motion_evidence == 'contradicted' and (body.vx, body.vy) == (0., 0.)


def test_animated_texture_inside_a_fixed_outline_is_not_body_translation():
    body = _thing()
    assert not translation_is_observed(body, _blob(body), 1.1, PointMotion(5., 0., .01, 4, 1., 1.1))


@pytest.mark.parametrize('step', [.5, 1., 2.])
def test_pixel_tolerance_cannot_accumulate_box_drift_over_stationary_points(step):
    body, eyes = _thing(), WhatMoves()
    body.boundary = None
    for index in range(1, 9):
        at = 1. + index * .05
        blob = _blob(body, x=body.x + step)
        blob['boundary'] = None
        eyes._point_motion[1] = PointMotion(0., 0., .01, 4, body.seen, at)
        eyes._moved(body, blob, at)
        assert body.motion_measured_at is None and not body.motion_path
        assert (body.vx, body.vy) == (0., 0.)


@pytest.mark.parametrize('axis,direction', [('x', -1), ('x', 1), ('y', -1), ('y', 1)])
@pytest.mark.parametrize('points_present', [False, True])
def test_growth_of_one_edge_does_not_certify_centroid_motion(axis, direction, points_present):
    body = _thing()
    blob = _blob(body)
    blob[axis] += direction
    blob['w' if axis == 'x' else 'h'] += 2
    width, height = int(blob['w']), int(blob['h'])
    blob.update(patch=np.full((height, width, 3), 200, np.uint8),
                support=np.ones((height, width), bool), boundary=_boundary(width, height))
    points = PointMotion(0., 0., .01, 4, 1., 1.1) if points_present else None
    assert not translation_is_observed(body, blob, 1.1, points)


@pytest.mark.parametrize('missing', ['mask', 'pixels', 'time', 'shape', 'colour'])
def test_without_agreeing_points_unobserved_or_changed_support_cannot_measure_motion(missing):
    body = _thing()
    blob = _blob(body, x=45.)
    at = 1.1
    if missing == 'mask':
        blob['support'] = None
    elif missing == 'pixels':
        blob['patch'] = None
    elif missing == 'time':
        at = 2.
    elif missing == 'shape':
        blob = _blob(body, x=45., w=24.)
    else:
        blob['patch'][:] = 20
    assert not translation_is_observed(body, blob, at)


def test_bad_motion_cannot_contaminate_the_next_measured_velocity():
    body = _thing()
    eyes = WhatMoves()
    eyes._moved(body, _blob(body, x=48., w=28.), 1.1)
    assert body.motion_measured_at is None
    # A new object extent becomes stable; this measured interval stands on its own.
    eyes._moved(body, _blob(body, x=49., w=28.), 1.2)
    assert body.motion_measured_at == 1.2
    assert body.vx == pytest.approx(10.)
    assert body.motion_path[0][0] == 1.1


def test_a_contradicted_motion_interval_retires_the_whole_press_not_just_one_sample():
    from core.agency.which_one_answers_to_her import _Speeds

    samples = _Speeds()
    for at in (1.1, 1.2):
        samples.add('a', 30., 0., press=1., when=at)
    samples.unseen('a', 1., contradicted=True)
    samples.add('a', 30., 0., press=1., when=1.4)
    assert not samples.by_press and not samples.timed
    samples.add('a', 30., 0., press=2., when=2.1)
    assert ('a', 2.) in samples.by_press


def test_a_gap_starts_a_new_observed_segment_of_the_same_held_press():
    from core.agency.which_one_answers_to_her import _Speeds

    samples = _Speeds()
    samples.add('a', 90., 0., press=1., when=1.1)
    samples.unseen('a', 1.)
    for at in (1.4, 1.5):
        samples.add('a', 30., 0., press=1., when=at)
    assert samples.by_press[('a', 1.)] == [(30., 0.), (30., 0.)]
    assert [at for at, *_ in samples.timed[('a', 1.)]] == [1.4, 1.5]


def test_a_stationary_observation_does_not_certify_an_unmeasured_prediction():
    body = _thing()
    eyes = WhatMoves()
    eyes._moved(body, _blob(body, x=48., w=28.), 1.1)
    eyes.things[1] = body
    eyes._backdrop = np.zeros((120, 200, 3), np.float32)
    image = np.zeros((120, 200, 3), np.uint8)
    left, top, right, bottom = (int(v) for v in body.box())
    image[top:bottom, left:right] = 200
    eyes._the_unmatched(image, {}, 1.2)
    assert body.motion_measured_at == 1.2 and (body.vx, body.vy) == (0., 0.)
    assert list(body.motion_path) == [(1.1, 48., 50.), (1.2, 48., 50.)]


def test_a_brief_unavailable_pose_reanchors_discovery_and_records_only_a_measured_segment():
    from core.agency.what_a_press_does import WhatAPressDoes

    body, eyes, presses = _thing(), WhatMoves(), WhatAPressDoes()
    presses.pressed('a', 1., [body], epoch=1, independent=True)
    presses.released('a', 1.2)
    for index, dy in enumerate((-5., -12., -18., -12., -5., 0., 0., 0., 0., 0., 0., 0.), 1):
        at = 1. + index * .06
        blob = _blob(body, x=40.)
        blob['y'] = 50. + dy
        if index == 3:
            blob['support'] = None
        eyes._moved(body, blob, at)
        presses.saw({1: body}, at)
    assert presses.lifts(0, 12., 1) == ['a']
    assert len(presses.entity_presses) == 1
    assert all(point[0] >= .3 - 1e-8 for point in presses.entity_presses[0].samples)


def test_unmeasured_box_history_cannot_supply_baseline_drift():
    from core.agency.what_a_press_does import _going_on

    body = _thing()
    body.path.extend([(1.1, 42., 50.), (1.2, 44., 50.), (1.3, 46., 50.)])
    body.vx, body.seen = 20., 1.3
    assert _going_on(body) == (0., 0.)


def test_an_unmeasured_zero_velocity_cannot_start_a_stationary_press_trial():
    from core.agency.what_a_press_does import WhatAPressDoes

    body, presses = _thing(), WhatAPressDoes()
    body.motion_measured_at = None
    presses.pressed('a', 1., [body], epoch=1, independent=True)
    assert not presses.watching


def test_settled_behavior_keeps_observing_the_independent_trial():
    from core.agency.what_a_press_does import WhatAPressDoes

    body, eyes, presses = _thing(), WhatMoves(), WhatAPressDoes()
    presses.pressed('a', 1., [body], epoch=1, independent=True)
    for index in range(1, 13):
        at = 1. + index * .05
        eyes._moved(body, _blob(body), at)
        presses.saw({1: body}, at)
    assert len(presses.by_number[('a', 1)]) == 1
    assert presses.by_number[('a', 1)][0][-1][0] == pytest.approx(.25)
    assert presses.entity_presses[0].ended == pytest.approx(1.6)
    assert len(presses.entity_presses[0].samples) == 12
    presses.pressed('b', 1.61, [body], epoch=1, independent=True)
    assert all(w.key == 'b' for w in presses.watching)
    assert presses.entity_presses[0].ended == pytest.approx(1.6)


def test_a_held_unsettled_input_still_keeps_its_measured_trial():
    from core.agency.what_a_press_does import WhatAPressDoes

    body, eyes, presses = _thing(), WhatMoves(), WhatAPressDoes()
    presses.pressed('a', 1., [body], epoch=1, independent=True)
    for index in range(1, 17):
        at = 1. + index * .1
        eyes._moved(body, _blob(body, x=body.x + 3.), at)
        presses.saw({1: body}, at)
    assert not presses.watching and not presses.by_number
    assert presses.entity_presses[0].ended == pytest.approx(2.5)


def test_a_measured_prefix_is_kept_when_later_pixels_are_unavailable():
    from core.agency.what_a_press_does import WhatAPressDoes

    body, eyes, presses = _thing(), WhatMoves(), WhatAPressDoes()
    presses.pressed('a', 1., [body], epoch=1, independent=True)
    for index in range(1, 5):
        at = 1. + index * .06
        eyes._moved(body, _blob(body, x=body.x + 3.), at)
        presses.saw({1: body}, at)
    body.motion_measured_at = None
    body.motion_evidence, body.seen = 'unavailable', 1.3
    presses.saw({1: body}, 1.3)
    assert len(presses.entity_presses) == 1
    receipt = presses.entity_presses[0]
    assert receipt.ended == pytest.approx(1.24) and len(receipt.samples) == 4
    presses.saw({1: body}, 1.4)
    assert len(presses.entity_presses) == 1
    body.motion_evidence, body.seen = 'contradicted', 1.5
    presses.saw({1: body}, 1.5)
    assert not presses.entity_presses


def test_identification_uses_true_translation_but_not_key_correlated_box_changes():
    from core.agency.which_one_answers_to_her import WhichIsHers

    outcomes = []
    for real_translation in (False, True):
        body, eyes, hers = _thing(), WhatMoves(), WhichIsHers()
        eyes.things[1] = body
        at = 2.
        for trial in range(8):
            key = 'a' if trial % 2 == 0 else 'b'
            hers.holding(key, at, trying=True)
            for _ in range(7):
                at += .1
                shift = 3. if key == 'a' else -3.
                blob = _blob(body, x=body.x + shift)
                # A detector drifts along stationary texture in one world;
                # in the other the same visible texture moves with its box.
                eyes._point_motion[1] = PointMotion(shift if real_translation else 0., 0., .01, 4, body.seen, at)
                eyes._moved(body, blob, at)
                hers.saw(eyes, [], at)
        outcomes.append(hers.number)
    assert outcomes == [None, 1]


def test_unmeasured_motion_cannot_become_physics_or_a_press_path():
    from core.agency.what_a_press_does import WhatAPressDoes
    from core.perception.how_things_move_here import HowThingsMoveHere

    body = _thing()
    body.moved = True
    physics, presses = HowThingsMoveHere(), WhatAPressDoes()
    moves = SimpleNamespace(things={1: body}, shape=(120, 200))
    nobody = SimpleNamespace(thing=lambda _moves: None)
    presses.pressed('a', 1., [body], epoch=1, independent=True)
    for i in range(1, 7):
        body.seen = 1. + i * .05
        body.x += 3.
        body.vx += 60.
        physics.saw(moves, nobody, [], body.seen)
        presses.saw(moves.things, body.seen)
    assert not physics.kinds and not presses.curves and not presses.entity_presses


def test_registered_translation_support_invariant():
    from core.perception.translation_from_visible_support import _translation_support_invariant

    assert _translation_support_invariant() == ()


def test_registered_motion_segments_invariant():
    from core.agency.which_one_answers_to_her import _motion_segments_invariant

    assert _motion_segments_invariant() == ()


def test_registered_settled_trial_invariant():
    from core.agency.what_a_press_does import _settled_trial_invariant

    assert _settled_trial_invariant() == ()
