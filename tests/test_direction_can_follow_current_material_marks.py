"""Rotation has to earn its relation to travel before it can update a facing."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from core.agency.direction_linked_to_observed_pose import DirectionFromPose
from core.agency.which_way_she_faces import WhichWaySheFaces
from core.perception.orientation_from_continuing_marks import current_angular_phase
from core.perception.position_from_the_same_points import PointPose
from core.perception.rigid_motion_from_named_points import RigidMotion

pytestmark = pytest.mark.unit


def body(at=2., angle=.4, position=(30., 30.)):
    starts = {i: p for i, p in enumerate(((0., 0.), (12., 0.), (0., 12.), (12., 12.)))}
    rotated_origin = RigidMotion(angle, (0., 0.), (), 0.).apply((6., 6.))
    fit = RigidMotion(angle, tuple(position[i]-rotated_origin[i] for i in range(2)), (), 0.)
    current = {i: tuple(fit.apply(p)) for i, p in starts.items()}
    pose = PointPose(1., at, (6., 6.), starts, tuple(fit.apply((6., 6.))),
                     reference_at=1., visible_marks=current, angular_reference_at=1.)
    return SimpleNamespace(number=1, kind=2, born=1., seen=at, motion_measured_at=at,
                           motion_evidence='points', point_pose=pose)


def test_geometry_reports_rotation_without_calling_it_a_facing():
    phase = current_angular_phase(body(), 2.)
    assert phase is not None and phase.angle == pytest.approx(.4)
    assert phase.reference_at == 1. and phase.uncertainty > 0.


@pytest.mark.parametrize('departure', ['stale', 'unmeasured', 'new-birth', 'no-reference', 'prediction', 'reflection'])
def test_unqualified_geometry_cannot_supply_an_angular_phase(departure):
    thing = body()
    if departure == 'stale':
        thing.point_pose.at = 1.9
    elif departure == 'unmeasured':
        thing.motion_measured_at = None
    elif departure == 'new-birth':
        thing.born = 1.5
    elif departure == 'no-reference':
        thing.point_pose.reference_at = None
    elif departure == 'prediction':
        thing.motion_evidence = 'unavailable'
    else:
        thing.point_pose.visible_marks = {i: (-p[0], p[1]) for i, p in thing.point_pose.anchors.items()}
    assert current_angular_phase(thing, 2.) is None


def qualified():
    relation = DirectionFromPose()
    for i, phase in enumerate((0., .4, .8)):
        relation.observe(('body', 1.), 1.+i, phase, .01, direction=phase+.3)
    return relation


def test_observed_pose_can_follow_direction_when_the_body_stops_translating():
    relation = qualified()
    assert relation.current == pytest.approx(1.1)
    relation.observe(('body', 1.), 4., 1.2, .01, direction=None)
    assert relation.current == pytest.approx(1.5)
    faces = WhichWaySheFaces(facing=.8, now=4., pose_direction=relation)
    assert faces.facing_now() == pytest.approx(1.5)
    faces.now = 4.1
    assert faces.facing_now() == pytest.approx(.8)  # A later picture cannot borrow the old phase.


def test_a_rotating_animation_without_changed_travel_cannot_qualify_facing():
    relation = DirectionFromPose()
    for i, phase in enumerate((0., .4, .8)):
        relation.observe('body', 1.+i, phase, .01, direction=.2)
    assert relation.current is None


def test_repeated_nearby_samples_do_not_qualify_a_direction_relation():
    relation = DirectionFromPose()
    for i in range(20):
        relation.observe('body', 1.+i, .001*i, .01, direction=.3+.001*i)
    assert relation.current is None and len(relation.samples) == 16


@pytest.mark.parametrize('departure', ['owner', 'reference', 'uncertain', 'nonfinite'])
def test_phase_continuation_keeps_context_and_measurement_quality(departure):
    relation = qualified()
    context, at, phase, uncertainty = ('body', 1.), 4., 1.2, .01
    if departure == 'owner':
        context = ('other', 1.)
    elif departure == 'reference':
        context = ('body', 2.)
    elif departure == 'uncertain':
        uncertainty = .4
    else:
        phase = float('nan')
    relation.observe(context, at, phase, uncertainty, direction=None)
    assert relation.current is None


def test_a_repeated_or_older_capture_does_not_discard_a_current_qualified_direction():
    import copy

    relation = qualified()
    before = copy.deepcopy(relation)
    for at in (3., 2.):
        relation.observe(('body', 1.), at, 1.2, .01, direction=None)
        assert relation == before


def test_a_legacy_point_pose_without_visible_marks_stays_unknown():
    thing = body()
    del thing.point_pose.visible_marks
    assert current_angular_phase(thing, 2.) is None


def test_relation_wraps_at_the_phase_boundary():
    relation = DirectionFromPose()
    for i, phase in enumerate((2.7, 3.1, -2.8)):
        relation.observe('body', 1.+i, phase, .01, direction=phase-.5)
    relation.observe('body', 4., -2.4, .01, direction=None)
    assert relation.current == pytest.approx(-2.9)


def test_real_input_binding_calibrates_only_current_owned_motion(monkeypatch):
    from core.agency import facing_seen_under_its_input as binding

    faces = WhichWaySheFaces()
    hers = SimpleNamespace(number=1, kind=2, identification=SimpleNamespace(epoch=3),
                           _held_at=lambda _at: (0., 'drive'), _beside_until=0.,
                           _hers=SimpleNamespace(free={}, typical=lambda _key: None), _by_thing={})
    for i in range(16):
        phase = i*.07
        at = 2.+i*.1
        radius = 80./.7
        position = (30.+radius*(math.sin(phase+.3)-math.sin(.3)),
                    30.-radius*(math.cos(phase+.3)-math.cos(.3)))
        thing = body(at, phase, position)
        monkeypatch.setattr(binding, 'measured_velocity',
                            lambda _mine, _at, p=phase: (80.*math.cos(p+.3), 80.*math.sin(p+.3), 1.))
        binding.observe_facing(faces, thing, hers, at)
    assert faces.pose_direction.current == pytest.approx(1.35)
    monkeypatch.setattr(binding, 'measured_velocity', lambda *_a: (0., 0., None))
    hers._held_at = lambda _at: (3.6, 'turn')
    binding.observe_facing(faces, body(3.7, 1.4, position), hers, 3.7)
    assert faces.facing_now() == pytest.approx(1.7)
    faces.along['drive'] = [80., 80.]
    assert faces.way_of('drive') == pytest.approx((80.*math.cos(1.7), 80.*math.sin(1.7)))


def test_unknown_velocity_error_does_not_become_a_zero_error_calibration(monkeypatch):
    from core.agency import facing_seen_under_its_input as binding

    faces = WhichWaySheFaces()
    hers = SimpleNamespace(number=1, kind=2, identification=SimpleNamespace(epoch=3),
                           _held_at=lambda _at: (0., 'drive'), _beside_until=0.,
                           _hers=SimpleNamespace(free={}, typical=lambda _key: None), _by_thing={})
    monkeypatch.setattr(binding, 'measured_velocity', lambda *_a: (80., 0., None))
    for i, phase in enumerate((0., .4, .8, 1.2)):
        binding.observe_facing(faces, body(2.+i*.1, phase), hers, 2.+i*.1)
    assert faces.pose_direction.current is None and not faces.pose_direction.samples


def reseeded(at=2.1, angle=1., *, names=None):
    thing = body(at, angle)
    pose = thing.point_pose
    if names is not None:
        pose.visible_marks = dict(zip(names, pose.visible_marks.values(), strict=True))
    pose.origin = pose.position
    pose.anchors = dict(pose.visible_marks)
    pose.reference_at = pose.angular_reference_at = at
    return thing


def test_angular_reference_crosses_new_anchors_only_through_current_continuing_marks():
    from core.perception.position_from_the_same_points import _bridge_angular_basis

    previous, thing = body(2., .8), reseeded()
    _bridge_angular_basis(previous.point_pose, thing.point_pose)
    phase = current_angular_phase(thing, 2.1)
    assert phase is not None and phase.reference_at == 1. and phase.angle == pytest.approx(1.)
    assert phase.uncertainty > current_angular_phase(previous, 2.).uncertainty


def test_a_position_reference_does_not_grant_an_angular_reference():
    from core.perception.position_from_the_same_points import _bridge_angular_basis

    previous, thing = body(2., .8), reseeded(names=(10, 11, 12, 13))
    _bridge_angular_basis(previous.point_pose, thing.point_pose)
    thing.point_pose.reference_at = 1.  # An independent translation kept the positional origin.
    phase = current_angular_phase(thing, 2.1)
    assert phase is not None and phase.reference_at == 2.1 and phase.angle == pytest.approx(0.)


@pytest.mark.parametrize('departure', ['two', 'reflection', 'deformation', 'gap', 'birth'])
def test_an_unmeasured_angular_bridge_cannot_continue_the_old_basis(departure):
    from core.perception.position_from_the_same_points import _bridge_angular_basis

    previous, thing = body(2., .8), reseeded()
    if departure == 'two':
        thing.point_pose.visible_marks = dict(list(thing.point_pose.visible_marks.items())[:2])
    elif departure == 'reflection':
        thing.point_pose.visible_marks = {i: (-p[0], p[1]) for i, p in thing.point_pose.visible_marks.items()}
    elif departure == 'deformation':
        thing.point_pose.visible_marks = {i: (p[0]*1.8, p[1]*1.8) for i, p in thing.point_pose.visible_marks.items()}
    elif departure == 'gap':
        thing.point_pose.at = 3.
    else:
        thing.point_pose.born = 1.5
    _bridge_angular_basis(previous.point_pose, thing.point_pose)
    assert thing.point_pose.angular_reference_at == 2.1 and thing.point_pose.angular_offset == 0.


def test_the_observation_and_direction_canaries_have_registered_claims():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    for name in ('angular_phase_requires_current_material_marks', 'pose_direction_requires_observed_travel_agreement'):
        check = next(t for t in suite.tests() if t.name == name)
        assert check.predict(None) is True
        assert any(c.test == name for c in suite.claims())
