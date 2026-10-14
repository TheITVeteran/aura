"""Bind observed facing to the input that was delivered before its picture."""
from __future__ import annotations

import math
from typing import Any

from core.agency.an_input_answers_relative_to_motion import moving_inputs_in_trials
from core.agency.response_in_the_body_frame import body_epoch
from core.agency.which_way_she_faces import MOVING, SETTLE_S
from core.perception.motion_kept_between_pictures import measured_velocity
from core.perception.orientation_from_continuing_marks import current_angular_phase
from core.perception.translation_from_visible_support import motion_was_measured
from core.verify import invariant


def observe_facing(faces: Any, mine: Any, hers: Any, at: float) -> None:
    """A prediction or an overlapping input cannot teach a direction or a stationary turn."""
    identity = body_epoch(hers)
    birth = _current_birth(mine, hers, at)
    changed = faces.body_epoch != identity
    continuing = (changed and birth is not None and faces.body_birth == birth
                  and faces.body_epoch is not None and identity is not None
                  and faces.body_epoch[0] == identity[0] and faces.body_epoch[2] == identity[2])
    replacement = (identity is not None and mine is not None and getattr(mine, 'number', None) == hers.number
                   and getattr(mine, 'seen', None) == at
                   and faces.body_birth != birth)
    if (changed and not continuing) or replacement:
        faces.__init__()
        if identity is not None:
            # These are experiment requests, not heading measurements.
            faces.pressed.update(key for key in hers._hers.free if key)
            faces.heading_hints.update(key for key in hers._hers.free if key
                and (velocity := hers._hers.typical(key)) is not None
                and all(math.isfinite(v) for v in velocity) and math.hypot(*velocity) >= MOVING)
            own_trials = getattr(hers, '_by_thing', {}).get(hers.number)
            faces.heading_hints.update(moving_inputs_in_trials(own_trials or hers._hers, least_speed=MOVING))
    faces.body_epoch = identity
    if (identity is not None and mine is not None and getattr(mine, 'number', None) == hers.number
            and getattr(mine, 'seen', None) == at):
        faces.body_birth = birth
    hers.facing_controls = faces
    receipt = hers._held_at(at)
    key, began = (receipt[1], receipt[0]) if receipt is not None else ("", None)
    measured = (identity is not None and receipt is not None and mine is not None
                and mine.number == hers.number and motion_was_measured(mine, at)
                and at >= hers._beside_until)
    vx, vy, _uncertainty = measured_velocity(mine, at)
    faces.saw(vx, vy, key, at, measured=measured, began=began)
    phase = current_angular_phase(mine, at) if measured else None
    if phase is not None:
        direction = None
        context = identity[0], birth, identity[2], phase.reference_at
        pair = None
        if key and began is not None and faces.sign is not None and key not in faces.turning_keys():
            pair = faces.pose_path.observe((context, key, began), at, phase.angle, phase.uncertainty,
                phase.position, phase.position_error, ready_after=max(began+SETTLE_S, hers._beside_until))
        if pair is not None:
            direction = pair.direction+(math.pi if faces.sign < 0 else 0.)
        faces.pose_direction.observe(context, at, phase.angle, phase.uncertainty, direction=direction,
                                     sample_phase=pair.phase if pair is not None else None,
                                     sample_uncertainty=pair.uncertainty if pair is not None else None)


def _current_birth(mine: Any, hers: Any, at: float) -> float | None:
    """A visible track keeps its birth while its colour or pose is classified anew.

    Offline 2026-10-13 a continuing turning body changed visual kind. Its
    heading model was cleared despite the same number, birth and experiment.
    """
    born = getattr(mine, 'born', None)
    if (mine is None or hers.number is None or getattr(mine, 'number', None) != hers.number
            or getattr(mine, 'seen', None) != at
            or isinstance(born, bool) or not isinstance(born, (int, float)) or not math.isfinite(born)):
        return None
    return float(born)


@invariant('agency.a_continuing_body_keeps_heading_when_its_look_changes', scope='agency',
           owner='core/agency/facing_seen_under_its_input.py', observational=False)
def _heading_lineage_invariant() -> tuple:
    from types import SimpleNamespace

    from core.agency.which_one_answers_to_her import WhichIsHers
    from core.agency.which_way_she_faces import WhichWaySheFaces

    hers, faces = WhichIsHers(), WhichWaySheFaces()
    hers.number, hers.kind = 1, 1
    hers.holding('drive', 0.)
    body = SimpleNamespace(number=1, born=0., seen=.3, vx=80., vy=0., motion_evidence='direct')
    observe_facing(faces, body, hers, .3)
    faces.along['drive'] = [80., 80.]
    hers.kind, body.seen = 2, .4
    observe_facing(faces, body, hers, .4)
    assert faces.along['drive'] == [80., 80.]
    body.born, body.seen = .45, .5
    observe_facing(faces, body, hers, .5)
    assert not faces.along
    return ()


@invariant("agency.facing_keeps_observed_motion_and_input_custody", scope="agency",
           owner="core/agency/facing_seen_under_its_input.py", observational=False)
def _facing_input_invariant() -> tuple:
    from types import SimpleNamespace

    from core.agency.which_one_answers_to_her import WhichIsHers
    from core.agency.which_way_she_faces import WhichWaySheFaces

    hers, faces = WhichIsHers(), WhichWaySheFaces()
    hers.number, hers.kind = 1, 1
    hers.holding("drive", 0.)
    hers.holding("turn", .5)
    thing = SimpleNamespace(number=1, vx=80., vy=0., seen=.3, motion_evidence="pixels", motion_measured_at=.3)
    observe_facing(faces, thing, hers, .3)
    assert faces.held == "drive" and faces.held_since == 0. and len(faces.going) == 1
    thing.seen = .35
    observe_facing(faces, thing, hers, .35)
    assert not faces.going and faces.settled_pictures == 0
    assert not faces.turns_measured, "unmeasured motion was read as stationary turning"
    from core.perception.motion_kept_between_pictures import MotionEstimate

    thing.born, thing.seen, thing.motion_measured_at = 0., .4, .4
    thing.motion_estimate = MotionEstimate((1, 0.), ((100, 100), 'box'), .4, .4, (30., 30.),
                                         (0., 80.), (0., 0.), 1., 2., 10, 'observed', (1.,))
    observe_facing(faces, thing, hers, .4)
    assert math.isclose(faces.facing, math.pi / 2) and faces.going == [(0., 80.)]
    return ()


@invariant('agency.a_new_heading_requires_a_current_direction_experiment', scope='agency',
           owner='core/agency/facing_seen_under_its_input.py', observational=False)
def _fresh_heading_invariant() -> tuple:
    from types import SimpleNamespace

    from core.agency.which_one_answers_to_her import WhichIsHers
    from core.agency.which_way_she_faces import WhichWaySheFaces

    hers, faces = WhichIsHers(), WhichWaySheFaces()
    hers.number, hers.kind = 2, 1
    hers.holding('', 1.)
    for began in (-3., -2.):
        for after in (.25, .3, .35, .4):
            hers._hers.add('a', 80., 0., press=began, when=began + after)
    body = SimpleNamespace(number=2, vx=0., vy=0., seen=1.3, motion_evidence='direct')
    observe_facing(faces, body, hers, 1.3)
    assert faces.trial(['a']) == 'a'
    assert faces.facing is None and not faces.along and not faces.turning_keys()
    hers.number = None
    observe_facing(faces, None, hers, 1.4)
    assert not faces.heading_hints and faces.trial(['a']) is None
    hers, faces = WhichIsHers(), WhichWaySheFaces()
    hers.number, hers.kind = 3, 1
    hers.holding('', 1.)
    for began, speed in ((-3., 80.), (-2., -80.)):
        for after in (.25, .3, .35, .4):
            hers._by_thing[3].add('opaque', speed, 0., press=began, when=began + after)
    body = SimpleNamespace(number=3, vx=0., vy=0., seen=1.3, motion_evidence='direct')
    observe_facing(faces, body, hers, 1.3)
    assert faces.trial(['opaque']) == 'opaque' and faces.facing is None and not faces.along
    return ()
