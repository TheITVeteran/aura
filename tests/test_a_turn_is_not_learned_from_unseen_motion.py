"""A press with no settled observations is not measured as standing still."""
import math

import pytest

from core.agency.which_way_she_faces import WhichWaySheFaces

pytestmark = pytest.mark.unit


def moving(faces, key, began, velocity):
    for after in (0, .25, .3, .35, .4):
        faces.saw(*velocity, key, began + after)


def test_a_press_too_short_to_observe_is_not_taken_to_turn_a_body():
    faces = WhichWaySheFaces()
    moving(faces, "drive", 0, (80, 0))
    faces.saw(0, 0, "yaw", .5)
    moving(faces, "drive", .55, (0, 80))
    assert not faces.turns


def test_one_settled_picture_does_not_measure_a_stationary_turn():
    faces = WhichWaySheFaces()
    moving(faces, "drive", 0, (80, 0))
    faces.saw(0, 0, "yaw", .5)
    faces.saw(0, 0, "yaw", .8)
    moving(faces, "drive", 1, (0, 80))
    assert not faces.turns


def test_an_observed_still_press_between_goings_can_measure_a_turn():
    faces = WhichWaySheFaces()
    moving(faces, "drive", 0, (80, 0))
    for at in (.5, .75, .8, .85, .9):
        faces.saw(0, 0, "yaw", at)
    moving(faces, "drive", 1, (0, 80))
    assert len(faces.turns) == 1
    assert faces.turns[0][0] == {"yaw": .5}


def test_the_registered_turn_observation_invariant():
    from core.agency.which_way_she_faces import _turn_observation_invariant

    assert _turn_observation_invariant() == ()


def test_a_short_unmeasured_press_keeps_what_a_known_rate_predicts():
    faces = WhichWaySheFaces(facing=0)
    faces.turns = [({"yaw": .5}, math.pi / 2)] * 2
    faces.saw(0, 0, "yaw", 0)
    faces.saw(0, 0, "", .1)
    assert faces.facing_now() == pytest.approx(math.pi * .1)
    assert not faces.turns_measured
    assert len(faces.turns) == 2
