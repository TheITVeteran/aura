"""What a pastime is played with, and where that goes, is known from its name before its course is seen.

Offline 2026-10-11 a mini golf course was shot at from every round thing on it but the ball, and never at the hole: a
person told "mini golf" knows there is a ball, that it is theirs, and that it goes into the hole.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.cognition.what_a_pastime_is_played_with import pastime_named_in, where_it_goes


@pytest.mark.unit
def test_a_place_names_its_pastime_and_the_pastime_its_ball_and_where_the_ball_goes():
    golf = pastime_named_in("Cartoon Network: Cartoon Cove Mini Golf")
    assert golf is not None and (golf.played_with, golf.toward) == ("ball", "hole")
    assert pastime_named_in("HOW TO USE THE PUTTER: 1. Click on your ball").name == "golf"
    assert pastime_named_in("turning the chore into a game of basketball").toward == "hoop"
    assert pastime_named_in("Earn points by knocking other tops out of the stadium").played_with == "top"
    assert pastime_named_in("Use the arrow keys to move and space to jump") is None


def _course(holes: list[tuple[int, int]]) -> np.ndarray:
    picture = np.zeros((300, 400, 3), dtype=np.uint8)
    picture[:] = (190, 170, 110)                                  # the green, sandy here
    picture[:40] = (90, 170, 230)                                 # the sky, running off the top
    yy, xx = np.mgrid[0:300, 0:400]
    for cx, cy in holes:
        picture[((xx - cx) / 12.0) ** 2 + ((yy - cy) / 7.0) ** 2 <= 1.0] = (15, 15, 15)
    return picture


@pytest.mark.unit
def test_the_one_dark_mouth_on_a_course_is_where_the_ball_goes_and_two_are_no_aim():
    golf = pastime_named_in("mini golf")
    x, y = where_it_goes(golf, _course([(230, 200)]))
    assert x == pytest.approx(230 / 400, abs=0.02) and y == pytest.approx(200 / 300, abs=0.02)
    assert where_it_goes(golf, _course([(120, 150), (300, 220)])) is None
    assert where_it_goes(pastime_named_in("basketball"), _course([(230, 200)])) is None   # a hoop is not a mouth
