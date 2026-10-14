"""Position windows pair a travel chord with the phase observed at its midpoint."""
from __future__ import annotations

import math

import pytest

from core.agency.direction_from_a_measured_pose_path import DirectionFromPosePath
from core.agency.direction_linked_to_observed_pose import DirectionFromPose

pytestmark = pytest.mark.unit


def circle(phase, *, offset=.3, speed=80., rate=.7):
    radius = speed/rate
    return (30.+radius*(math.sin(phase+offset)-math.sin(offset)),
            30.-radius*(math.cos(phase+offset)-math.cos(offset)))


def test_curved_travel_pairs_heading_with_midpoint_phase_before_predicting_current_heading():
    path, relation = DirectionFromPosePath(), DirectionFromPose()
    for i in range(20):
        at, phase = 1.+i*.1, i*.07
        pair = path.observe('owned-press', at, phase, .01, circle(phase), .25, ready_after=1.)
        if pair is not None:
            assert pair.direction-pair.phase == pytest.approx(.3)
            assert 0. < pair.uncertainty < .12
            relation.observe('owned-body', at, phase, .01, direction=pair.direction,
                             sample_phase=pair.phase, sample_uncertainty=pair.uncertainty)
    assert relation.current == pytest.approx(1.33+.3)
    relation.observe('owned-body', 3., 1.6, .01, direction=None)
    assert relation.current == pytest.approx(1.9)


def test_rotation_without_resolved_translation_supplies_no_direction_sample():
    path = DirectionFromPosePath()
    for i in range(30):
        assert path.observe('press', i*.1, i*.07, .01, (30., 30.), .25, ready_after=0.) is None
    assert len(path.seen) == 30


@pytest.mark.parametrize('departure', ['input', 'body', 'pose-basis', 'gap', 'settling'])
def test_a_direction_window_cannot_cross_an_unmeasured_or_different_input_interval(departure):
    path = DirectionFromPosePath()
    for i in range(4):
        path.observe(('body', 1., 'drive', 0.), i*.1, i*.07, .01, circle(i*.07), .25, ready_after=0.)
    context, at, ready_after = ('body', 1., 'drive', 0.), .4, 0.
    if departure == 'input':
        context = ('body', 1., 'drive', .4)
    elif departure == 'body':
        context = ('new-body', 1., 'drive', 0.)
    elif departure == 'pose-basis':
        context = ('body', 2., 'drive', 0.)
    elif departure == 'gap':
        at = .9
    else:
        ready_after = .4
    assert path.observe(context, at, .28, .01, circle(.28), .25, ready_after=ready_after) is None


def test_uncertain_positions_cannot_qualify_a_phase_relation():
    path, relation = DirectionFromPosePath(), DirectionFromPose()
    for i in range(20):
        phase, at = i*.07, i*.1
        pair = path.observe('press', at, phase, .01, circle(phase), 10., ready_after=0.)
        if pair is not None:
            assert pair.uncertainty > .12
            relation.observe('body', at, phase, .01, direction=pair.direction,
                             sample_phase=pair.phase, sample_uncertainty=pair.uncertainty)
    assert relation.current is None and not relation.samples


def test_phase_interpolation_wraps_at_the_image_angle_boundary():
    path = DirectionFromPosePath()
    pair = None
    for i in range(6):
        phase = 3.+i*.07
        pair = path.observe('press', i*.1, (phase+math.pi) % (2.*math.pi)-math.pi,
                            .01, circle(phase), .25, ready_after=0.)
    assert pair is not None
    offset = (pair.direction-pair.phase+math.pi) % (2.*math.pi)-math.pi
    assert offset == pytest.approx(.3)


def test_an_unresolved_rotation_jump_does_not_create_a_midpoint_heading():
    path = DirectionFromPosePath()
    assert path.observe('press', 0., 0., .01, (0., 0.), .25, ready_after=0.) is None
    assert path.observe('press', .2, 2., .01, (20., 0.), .25, ready_after=0.) is None


@pytest.mark.parametrize('field', ['at', 'phase', 'uncertainty', 'position_error', 'ready_after', 'position'])
@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), True, None])
def test_unmeasured_numeric_values_remain_unknown(field, invalid):
    args = dict(context='press', at=1., phase=.2, uncertainty=.01,
                position=(10., 10.), position_error=.25, ready_after=0.)
    args[field] = (invalid, 10.) if field == 'position' else invalid
    path = DirectionFromPosePath()
    assert path.observe(**args) is None and not path.seen


def test_an_ambiguous_context_retires_its_position_window():
    path = DirectionFromPosePath()
    path.observe('press', 0., 0., .01, (0., 0.), .25, ready_after=0.)
    assert path.observe(['press'], .2, 0., .01, (20., 0.), .25, ready_after=0.) is None
    assert not path.seen
