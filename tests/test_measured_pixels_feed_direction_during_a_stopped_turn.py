"""Exercise pixel observation and facing binding with an explicitly supplied body identity."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage

from core.agency.facing_seen_under_its_input import observe_facing
from core.agency.which_way_she_faces import WhichWaySheFaces
from core.perception.orientation_from_continuing_marks import current_angular_phase
from core.perception.what_moves_in_the_picture import WhatMoves

pytestmark = pytest.mark.unit


def picture(tile, x, y, angle):
    frame = np.full((240, 320, 3), 24, np.uint8)
    rotated = ndimage.rotate(tile, -angle*180/math.pi, axes=(1, 0), reshape=True,
                             order=1, mode='constant', cval=24)
    top, left = int(round(y-rotated.shape[0]/2)), int(round(x-rotated.shape[1]/2))
    frame[top:top+rotated.shape[0], left:left+rotated.shape[1]] = rotated
    return frame


@pytest.mark.parametrize('seed', [762, 831])
@pytest.mark.parametrize('coupled', [True, False])
def test_a_measured_pose_follows_direction_only_if_the_visible_travel_agrees(seed, coupled):
    tile = np.full((25, 33, 3), (220, 125, 45), np.uint8)
    tile[3:22, 4:29] = np.random.default_rng(seed).integers(90, 240, (19, 25, 3), dtype=np.uint8)
    observer, faces = WhatMoves(), WhichWaySheFaces()
    clock = [0., 'drive']
    hers = SimpleNamespace(number=1, kind=None, identification=SimpleNamespace(epoch=1),
                           _held_at=lambda _at: tuple(clock), _beside_until=0.,
                           _hers=SimpleNamespace(free={}, typical=lambda _key: None), _by_thing={})
    x, y, angle, phases = 55., 65., 0., 0
    stopped_errors, calibrated = [], 0
    for i in range(60):
        at = 1.+i*.1
        if i < 40:
            angle += .035
            x += 4.*math.cos(angle) if coupled else 4.
            y += 4.*math.sin(angle) if coupled else 0.
        else:
            if i == 40:
                clock[:] = [at, 'turn']
            angle += .055
        observer.see(picture(tile, x, y, angle), at)
        # This component owns entity 1 before observing. Position and angle
        # belong only to the renderer and the assertions, never the controller.
        thing = observer.things.get(1)
        if thing is None or thing.seen != at:
            continue
        hers.kind = thing.kind
        phases += current_angular_phase(thing, at) is not None
        observe_facing(faces, thing, hers, at)
        if faces.pose_direction.current is not None:
            calibrated += 1
            if i >= 40:
                stopped_errors.append(abs((faces.facing_now()-angle+math.pi) % (2.*math.pi)-math.pi))
    assert phases >= 40
    if coupled:
        assert len(stopped_errors) >= 18
        assert max(stopped_errors) < .12
    else:
        assert calibrated == 0, 'an animation cannot supply the travel heading'
