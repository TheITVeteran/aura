"""A carried body's first contact must be gentle, rather than a later return."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency.arriving_by_her_pushes import (
    _arrival_braking_invariant,
    arrival_excess,
    arrival_holds,
    arrival_key,
)
from core.agency.choosing_where_to_be import _Choosing
from core.agency.imagining_ahead import _the_scene
from core.agency.what_meeting_things_does import WhatMeetingDoes
from core.agency.which_one_answers_to_her import WhichIsHers
from core.perception.what_moves_in_the_picture import Thing

pytestmark = pytest.mark.unit


def _body(number, kind, x, y, width=12.):
    return Thing(number, x, y, width, width, np.zeros(64), (180, 180, 180),
                 np.zeros((1, 1, 3), np.uint8), 0., 0., kind=kind, moved=number == 1)


def _learn(force, axis):
    body = _body(1, 0, 400., 400.)
    moves = SimpleNamespace(shape=(800, 800), things={1: body}, kinds=[], out_of_sight={})
    hers, at = WhichIsHers(), 0.
    for _ in range(4):
        for key in ("first", "", "second", ""):
            hers.holding(key, at, trying=True)
            acceleration = {"first": -force, "second": force}.get(key, 0.)
            for _ in range(28):
                step = .025
                at += step
                place, speed = ("x", "vx") if axis == 0 else ("y", "vy")
                setattr(body, place, getattr(body, place) + getattr(body, speed) * step + .5 * acceleration * step ** 2)
                setattr(body, speed, getattr(body, speed) + acceleration * step)
                body.seen = at
                hers.saw(moves, [], at)
    assert hers.number == body.number and hers.carried
    assert hers.push_of("first")[axis] == pytest.approx(-force)
    assert hers.push_of("second")[axis] == pytest.approx(force)
    return body, moves, hers


def _advance(body, acceleration, seconds, target, axis, limit):
    """Inspect within an input interval, including contact before its final sample."""
    place, speed = ("x", "vx") if axis == 0 else ("y", "vy")
    for _ in range(20):
        dt = seconds / 20
        setattr(body, place, getattr(body, place) + getattr(body, speed) * dt + .5 * acceleration * dt ** 2)
        setattr(body, speed, getattr(body, speed) + acceleration * dt)
        body.seen += dt
        if abs(getattr(body, place) - target) <= 24.:
            assert abs(getattr(body, speed)) <= limit + 1e-6, "first contact was too fast"
            return True
    return False


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("start,speed,target,force,step,delay", [
    (100., 0., 500., 100., .1, 0.),
    (700., 0., 300., 100., .1, 0.),
    (100., 80., 500., 100., .1, 0.),
    (620., -90., 300., 70., .1, 0.),
    (400., 100., 500., 100., .1, 0.),
    (500., -100., 400., 100., .1, 0.),
    (100., 0., 500., 60., .05, 0.),
    (100., 0., 500., 100., .1, .05),
])
def test_measured_pushes_bring_a_body_to_its_first_contact_gently(axis, start, speed, target, force, step, delay):
    body, moves, hers = _learn(force, axis)
    body.x = body.y = 100.
    body.vx = body.vy = 0.
    place, velocity = ("x", "vx") if axis == 0 else ("y", "vy")
    setattr(body, place, start)
    setattr(body, velocity, speed)
    destination = _body(2, 1, target if axis == 0 else 100., target if axis == 1 else 100., 36.)
    moves.things[2] = destination
    meeting = WhatMeetingDoes()
    meeting.stance = lambda kind, fixture=False: "meet"
    meeting.known = lambda kind: True
    meeting.gentle_below = lambda kind: 12.
    held = ""
    for _ in range(round(12 / step)):
        choosing = _Choosing(moves, hers, meeting, ["first", "second"], next_picture_s=step, response_s=delay)
        chosen = choosing.key(held)[0]
        if delay and _advance(body, hers.push_of(held)[axis], delay, target, axis, 12.):
            return
        if _advance(body, hers.push_of(chosen)[axis], step - delay, target, axis, 12.):
            return
        held = chosen
    pytest.fail("the body did not arrive in the bounded run")


@pytest.mark.parametrize("change", ["no_rest", "one_way", "nonfinite", "no_goal", "bad_step"])
def test_an_arrival_path_needs_a_measured_way_to_brake(change):
    pushes = {"": (0., 0.), "a": (-100., 0.), "b": (100., 0.)}
    goal, step = (200., None), .1
    if change == "no_rest":
        pushes.pop("")
    elif change == "one_way":
        pushes.pop("a")
    elif change == "nonfinite":
        pushes["a"] = (math.nan, 0.)
    elif change == "no_goal":
        goal = None, None
    elif change == "bad_step":
        step = 0.
    assert arrival_key((100., 100.), (20., 0.), goal, pushes, step) is None
    assert arrival_holds((100., 100.), (20., 0.), goal, pushes, [.1, .2, .3], step, "") is None


def test_braking_room_is_checked_beyond_the_imagined_horizon():
    pushes = {"": (0., 0.), "a": (-100., 0.), "b": (100., 0.)}
    assert arrival_excess((100., 100.), (200., 0.), (200., None), pushes, (24., 24.), 12.) > 70.
    assert arrival_excess((100., 100.), (20., 0.), (200., None), pushes, (24., 24.), 12.) == 0.


def test_the_old_push_goes_on_during_delivery():
    body, moves, hers = _learn(100., 0)
    body.x, body.y, body.vx, body.vy = 400., 100., 80., 0.
    choosing = _Choosing(moves, hers, WhatMeetingDoes(), ["first", "second"], response_s=.2)
    choosing.held_now = "first"
    scene = _the_scene(choosing, 1.2)
    assert scene.start == pytest.approx((414., 100.))
    assert scene.going == pytest.approx((60., 0.))


def test_unknown_held_push_cannot_supply_a_delivery_prediction():
    body, moves, hers = _learn(100., 0)
    choosing = _Choosing(moves, hers, WhatMeetingDoes(), ["first", "second"], response_s=.2)
    choosing.held_now = "unmeasured"
    assert _the_scene(choosing, 1.2) is None


def test_other_bodies_also_move_while_a_command_is_being_delivered():
    body, moves, hers = _learn(100., 0)
    body.x, body.y, body.vx = 400., 100., 0.
    other = _body(2, 1, 430., 100., 24.)
    other.moved, other.vx = True, -20.
    moves.things[2] = other
    meeting = WhatMeetingDoes()
    meeting.stance = lambda kind, fixture=False: "avoid"
    choosing = _Choosing(moves, hers, meeting, ["first", "second"], response_s=.2)
    scene = _the_scene(choosing, 1.2)
    assert scene.where[1, 0, 0, 0] == pytest.approx(430. - 20. * (.2 + scene.times[0]))


def test_arrival_evidence_invariant():
    assert _arrival_braking_invariant() == ()


@pytest.mark.parametrize("pull", [0., 60.])
def test_two_axes_and_a_passive_pull_share_one_arrival_plan(pull):
    body, destination = _body(1, 0, 400., 400.), _body(2, 1, 500., 500., 36.)
    moves = SimpleNamespace(shape=(800, 800), things={1: body}, kinds=[], out_of_sight={})
    pushes = {"": (0., pull), "a": (-120., pull), "b": (120., pull),
              "c": (0., pull - 160.), "d": (0., pull + 160.)}
    hers, at = WhichIsHers(), 0.
    for _ in range(4):
        for key in ("a", "", "c", "", "b", "", "d", ""):
            hers.holding(key, at, trying=True)
            for _ in range(28):
                at += .025
                ax, ay = pushes[key]
                body.x += body.vx * .025 + .5 * ax * .025 ** 2
                body.y += body.vy * .025 + .5 * ay * .025 ** 2
                body.vx, body.vy = body.vx + ax * .025, body.vy + ay * .025
                body.seen = at
                hers.saw(moves, [], at)
    assert hers.number == 1 and hers.carried
    body.x, body.y, body.vx, body.vy = 100., 200., 0., 0.
    moves.things[2] = destination
    meeting = WhatMeetingDoes()
    meeting.stance = lambda kind, fixture=False: "meet"
    meeting.known = lambda kind: True
    meeting.gentle_below = lambda kind: 40.
    held = ""
    for _ in range(160):
        choosing = _Choosing(moves, hers, meeting, ["a", "b", "c", "d"], next_picture_s=.1)
        held = choosing.key(held)[0]
        ax, ay = pushes[held]
        for _ in range(20):
            body.x += body.vx * .005 + .5 * ax * .005 ** 2
            body.y += body.vy * .005 + .5 * ay * .005 ** 2
            body.vx, body.vy = body.vx + ax * .005, body.vy + ay * .005
            body.seen += .005
            if abs(body.x - 500.) <= 24. and abs(body.y - 500.) <= 24.:
                assert math.hypot(body.vx, body.vy) <= 40. + 1e-6
                return
    pytest.fail("the two-axis body did not arrive in the bounded run")
