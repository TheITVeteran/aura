"""Bind observed facing to the input that was delivered before its picture."""
from __future__ import annotations

import math
from typing import Any

from core.agency.response_in_the_body_frame import body_epoch
from core.agency.which_way_she_faces import MOVING
from core.perception.translation_from_visible_support import motion_was_measured
from core.verify import invariant


def observe_facing(faces: Any, mine: Any, hers: Any, at: float) -> None:
    """A prediction or an overlapping input cannot teach a direction or a stationary turn."""
    identity = body_epoch(hers)
    if faces.body_epoch != identity:
        faces.__init__()
        faces.body_epoch = identity
        if identity is not None:
            # These are experiment requests, not heading measurements.
            faces.pressed.update(key for key in hers._hers.free if key)
            faces.heading_hints.update(key for key in hers._hers.free if key
                and (velocity := hers._hers.typical(key)) is not None
                and all(math.isfinite(v) for v in velocity) and math.hypot(*velocity) >= MOVING)
    hers.facing_controls = faces
    receipt = hers._held_at(at)
    key, began = (receipt[1], receipt[0]) if receipt is not None else ("", None)
    measured = (identity is not None and receipt is not None and mine is not None
                and mine.number == hers.number and motion_was_measured(mine, at)
                and at >= hers._beside_until)
    vx, vy = (float(mine.vx), float(mine.vy)) if mine is not None else (0., 0.)
    faces.saw(vx, vy, key, at, measured=measured, began=began)


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
    return ()
