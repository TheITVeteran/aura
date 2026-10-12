"""Unresolved movement cannot be reported as a continuously still screen."""
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency.playing_as_it_happens import _over, _Run
from core.perception.activity_in_the_picture import PictureActivity, _picture_activity_invariant
from core.perception.what_moves_in_the_picture import WhatMoves

pytestmark = pytest.mark.unit


def _picture():
    return np.full((80, 120, 3), 30, np.uint8)


def _run():
    return _Run(keys=['a'], began=1., last_moving=1.)


def _eyes(activity):
    return SimpleNamespace(activity=activity, moving=lambda **_kwargs: [])


def test_visible_articulation_keeps_play_going_without_any_certified_body_velocity():
    activity, run = PictureActivity(), _run()
    for index in range(30):
        at = 1. + index * .2
        picture = _picture()
        picture[30:40 + index % 4, 30:40] = 220
        activity.saw(picture, at)
        assert not _over(run, _eyes(activity), [], at)
    assert run.last_moving == at


def test_a_current_covered_quiet_interval_can_still_end_play():
    activity, run = PictureActivity(), _run()
    endings = []
    for index in range(30):
        at = 1. + index * .2
        activity.saw(_picture(), at)
        ending = _over(run, _eyes(activity), [], at)
        if ending:
            endings.append((at, ending))
    assert endings[0][0] > 4.
    assert endings[0][1] == 'nothing on the screen has moved for a while'


@pytest.mark.parametrize('interruption', ['gap', 'resize', 'invalid', 'reversal'])
def test_an_interrupted_capture_cannot_supply_the_missing_quiet(interruption):
    activity = PictureActivity()
    picture = _picture()
    activity.saw(picture, 1.)
    activity.saw(picture, 1.2)
    assert activity.quiet_for(1.2) == pytest.approx(.2)
    at = 1.4
    if interruption == 'gap':
        at = 5.
    elif interruption == 'resize':
        picture = picture[:40]
    elif interruption == 'invalid':
        picture = np.zeros((0, 0, 3))
    else:
        at = 1.1
    activity.saw(picture, at)
    assert activity.quiet_for(at) is None
    assert not _over(_run(), _eyes(activity), [], 5.)


def test_missing_or_stale_activity_is_unknown_instead_of_zero_motion():
    activity = PictureActivity()
    assert activity.quiet_for(9.) is None
    for index in range(20):
        activity.saw(_picture(), 1. + index * .2)
    assert activity.quiet_for(activity.at) > 3.
    assert not _over(_run(), _eyes(activity), [], 9.)


def test_isolated_pixel_noise_and_small_channel_noise_do_not_reset_quiet():
    activity = PictureActivity()
    picture = _picture()
    activity.saw(picture, 1.)
    noise = picture.copy() + 3
    noise[20, 20] = 255
    activity.saw(noise, 1.2)
    assert activity.changed is False and activity.quiet_for(1.2) == pytest.approx(.2)


def test_real_motion_eyes_record_activity_before_a_repeated_view_returns_early(monkeypatch):
    moves = WhatMoves()
    monkeypatch.setattr(moves, '_with_the_view', lambda _picture: True)
    picture = _picture()
    moves.see(picture, 1.)
    moves.see(picture, 1.2)
    assert moves.activity.changed is False
    picture[20:30, 20:30] = 220
    moves.see(picture, 1.4)
    assert moves.activity.changed is True
    assert not moves.things and not moves._point_motion


def test_registered_activity_invariant():
    assert _picture_activity_invariant() == ()
