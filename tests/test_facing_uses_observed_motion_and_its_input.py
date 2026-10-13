"""Facing observations retain their measured motion, time and delivered input."""
import copy
import math
from types import SimpleNamespace as Namespace

import pytest

from core.agency import playing_as_it_happens as play
from core.agency.facing_seen_under_its_input import observe_facing
from core.agency.which_one_answers_to_her import WhichIsHers
from core.agency.which_way_she_faces import WhichWaySheFaces

pytestmark = pytest.mark.unit


def _place():
    faces, hers = WhichWaySheFaces(), WhichIsHers()
    mine = Namespace(number=1, x=20., y=20., vx=0., vy=0., seen=0.,
              motion_evidence="pixels", motion_measured_at=None)
    pressures = Namespace(saw=lambda **_kw: None, urgency=lambda: 0.,
                   pull_at=lambda _x, _y: (0., 0.), harder=lambda _at: 1.)
    run = Namespace(faces=faces, pressures=pressures, held="", letting_go=False, gone_for=None,
             comes=Namespace(saw=lambda *_a: None), way_through=None)
    choosing = Namespace(mine=mine, pointing=False, caution=1.)
    moves, meeting = Namespace(things={1: mine}), Namespace(readouts=Namespace(current={}), vitals=1.)
    return run, choosing, moves, hers, meeting, mine


def _pictures(place, key, began, velocity, *, measured=True, delays=(0., .25, .3, .35, .4)):
    run, choosing, moves, hers, meeting, mine = place
    hers.holding(key, began)
    run.held = key
    for after in delays:
        at = began + after
        mine.seen, mine.vx, mine.vy = at, *velocity
        mine.motion_measured_at = at if measured else None
        play._what_the_place_does(run, choosing, moves, hers, meeting, at)


@pytest.mark.parametrize("predicted", [(0., 0.), (0., 80.), (-80., 0.)])
def test_the_actual_observer_cannot_learn_a_turn_from_predicted_motion(predicted):
    place = _place()
    _pictures(place, "drive", 0., (80., 0.))
    _pictures(place, "yaw", .5, predicted, measured=False)
    faces = place[0].faces
    assert faces.facing == 0. and not faces.going and faces.settled_pictures == 0
    _pictures(place, "drive", 1., (0., 80.))
    assert not faces.turns and "yaw" not in faces.ways
    assert faces.facing == pytest.approx(math.pi / 2)


def test_the_actual_observer_can_learn_from_current_observed_stationary_motion():
    place = _place()
    _pictures(place, "drive", 0., (80., 0.))
    _pictures(place, "yaw", .5, (0., 0.))
    _pictures(place, "drive", 1., (0., 80.))
    assert place[0].faces.turns == [({"yaw": .5}, pytest.approx(math.pi / 2))]


def test_a_missing_body_censors_the_actual_observer_and_advances_its_input_clock():
    place = _place()
    _pictures(place, "drive", 0., (80., 0.))
    run, choosing, moves, hers, meeting, _mine = place
    hers.holding("yaw", .5)
    run.held, choosing.mine = "yaw", None
    play._what_the_place_does(run, choosing, moves, hers, meeting, .8)
    assert run.faces.held == "yaw" and run.faces.held_since == .5
    assert run.faces.now == .8 and not run.faces.turns_measured
    assert run.faces.settled_pictures == 0


def test_a_captured_picture_uses_the_earlier_input_receipt():
    run, choosing, moves, hers, meeting, mine = _place()
    hers.holding("drive", 0.)
    hers.holding("yaw", .5)
    run.held = "yaw"
    mine.seen = mine.motion_measured_at = .3
    mine.vx = 80.
    play._what_the_place_does(run, choosing, moves, hers, meeting, .3)
    assert run.faces.held == "drive" and run.faces.held_since == 0.
    assert run.faces.going == [(80., 0.)] and run.faces.settled_pictures == 1


def test_delayed_first_pictures_do_not_change_the_measured_turn_duration():
    place = _place()
    _pictures(place, "drive", 0., (80., 0.), delays=(.21, .27, .35, .46))
    _pictures(place, "yaw", .5, (0., 0.), delays=(.3, .35, .4))
    _pictures(place, "drive", 1., (0., 80.), delays=(.21, .27, .35, .4))
    assert place[0].faces.turns == [({"yaw": .5}, pytest.approx(math.pi / 2))]


def test_an_overlapping_input_cannot_teach_a_stationary_turn():
    place = _place()
    _pictures(place, "drive", 0., (80., 0.))
    place[3].pressed_beside(.95)
    _pictures(place, "yaw", .5, (0., 0.))
    _pictures(place, "drive", 1., (0., 80.))
    assert not place[0].faces.turns


def test_motion_without_an_input_receipt_remains_unmeasured():
    faces, hers = WhichWaySheFaces(), WhichIsHers()
    mine = Namespace(vx=80., vy=0., seen=.3, motion_evidence="direct")
    observe_facing(faces, mine, hers, .3)
    assert faces.facing is None and not faces.going and not faces.turns_measured


def _going(faces, key, began, velocity):
    for after in (0., .25, .3, .35, .4):
        faces.saw(*velocity, key, began + after)


def test_a_gap_between_current_pictures_cannot_establish_a_continuous_turn():
    faces = WhichWaySheFaces()
    _going(faces, "drive", 0., (80., 0.))
    for at in (.5, .75, 1.15, 1.2, 1.25):
        faces.saw(0., 0., "yaw", at)
    _going(faces, "drive", 1.35, (0., 80.))
    assert not faces.turns


def test_duplicate_pictures_cannot_supply_two_stationary_observations():
    faces = WhichWaySheFaces()
    _going(faces, "drive", 0., (80., 0.))
    for at in (.5, .75, .75, .75):
        faces.saw(0., 0., "yaw", at)
    _going(faces, "drive", 1., (0., 80.))
    assert not faces.turns


@pytest.mark.parametrize("at", [.3, .2, float("nan"), float("inf"), -float("inf")])
def test_out_of_order_or_invalid_pictures_cannot_change_the_input_clock(at):
    faces = WhichWaySheFaces()
    faces.saw(80., 0., "drive", 0.)
    faces.saw(80., 0., "drive", .3)
    before = copy.deepcopy(faces)
    faces.saw(0., 80., "yaw", at)
    assert faces == before


@pytest.mark.parametrize("velocity", [(float("nan"), 0.), (0., float("inf")), (0., -float("inf"))])
def test_invalid_velocity_censors_measurement_without_stopping_the_input_clock(velocity):
    faces = WhichWaySheFaces()
    _going(faces, "drive", 0., (80., 0.))
    faces.saw(*velocity, "yaw", .75, began=.5)
    assert faces.now == .75 and faces.held_since == .5
    assert faces.settled_pictures == 0 and not faces.going and not faces.turns_measured


def test_a_movement_gap_does_not_join_two_separate_samples_into_a_press():
    faces = WhichWaySheFaces()
    faces.saw(80., 0., "drive", 0.)
    faces.saw(80., 0., "drive", .25)
    faces.saw(80., 0., "drive", .75)
    faces.saw(0., 0., "", .8)
    assert "drive" not in faces.ways and "drive" not in faces.along


def test_known_turn_rates_predict_through_missing_observations_without_new_learning():
    faces = WhichWaySheFaces(facing=0.)
    faces.turns = [({"yaw": .5}, math.pi / 2)] * 2
    before = copy.deepcopy(faces.turns)
    faces.saw(0., 0., "yaw", .05, measured=False, began=0.)
    faces.saw(0., 0., "yaw", .25, measured=False, began=0.)
    assert faces.facing_now() == pytest.approx(math.pi / 4)
    assert faces.turns == before and faces.settled_pictures == 0


def test_a_new_receipt_for_the_same_input_starts_a_distinct_press():
    faces = WhichWaySheFaces()
    faces.saw(80., 0., "drive", .25, began=0.)
    faces.saw(80., 0., "drive", .3, began=0.)
    faces.saw(80., 0., "drive", .5, began=.45)
    assert faces.held_since == .45 and not faces.going and faces.settled_pictures == 0
    assert len(faces.ways["drive"]) == 1


def test_fresh_movement_after_a_gap_can_anchor_the_next_observed_turn():
    faces = WhichWaySheFaces()
    _going(faces, "drive", 0., (80., 0.))
    faces.saw(0., 0., "drive", .5, measured=False)
    for at in (.55, .6, .65, .7):
        faces.saw(80., 0., "drive", at)
    for at in (.8, 1.05, 1.1, 1.15, 1.2):
        faces.saw(0., 0., "yaw", at)
    _going(faces, "drive", 1.3, (0., 80.))
    assert faces.turns == [({"yaw": .5}, pytest.approx(math.pi / 2))]


def test_the_input_and_motion_boundary_has_a_registered_measured_claim():
    from core.agency.facing_seen_under_its_input import _facing_input_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _facing_input_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == "facing_keeps_observed_motion_and_input_custody")
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
