"""Keys that turn her and a key that takes her the way she faces are told apart from keys that move her one way, and she
is steered by them (core/agency/which_way_she_faces.py).

A body in a world of its own: up takes it the way it faces, down backs it up, left and right turn it on the spot. She
tries keys as play does at the start, then the trials the facing asks for, and is then asked to drive to place after
place. And a body whose
keys move it one way on the screen, whatever it looks like, is not taken for one that turns.
"""
from __future__ import annotations

import math
import random

import pytest

from core.agency.which_way_she_faces import WhichWaySheFaces

pytestmark = pytest.mark.unit

STEP = 1 / 30


class _Turning:
    def __init__(self, facing: float = 0.0) -> None:
        self.x, self.y, self.facing, self.vx, self.vy = 240.0, 160.0, facing, 0.0, 0.0

    def step(self, key: str) -> None:
        if key == "left":
            self.facing -= 2.8 * STEP
        if key == "right":
            self.facing += 2.8 * STEP
        speed = 120.0 if key == "up" else -60.0 if key == "down" else 0.0
        self.vx, self.vy = speed * math.cos(self.facing), speed * math.sin(self.facing)
        self.x, self.y = self.x + self.vx * STEP, self.y + self.vy * STEP


class _OneWay(_Turning):
    def step(self, key: str) -> None:
        way = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}.get(key, (0, 0))
        self.vx, self.vy = 120.0 * way[0], 120.0 * way[1]
        self.x, self.y = self.x + self.vx * STEP, self.y + self.vy * STEP


KEYS = ["up", "down", "left", "right"]


def _held(body: _Turning, faces: WhichWaySheFaces, key: str, at: float, seconds: float = 0.45) -> float:
    end = at + seconds
    while at < end:
        body.step(key)
        faces.saw(body.vx, body.vy, key, at)
        at += STEP
    return at


def _tried(body: _Turning, faces: WhichWaySheFaces, seconds: float, rand: random.Random) -> float:
    """Her keys tried as her play tries them, each in turn with a rest between; then, for what is left of ``seconds``,
    a trial where the facing asks for one, as her play asks at each choice (ten at most), and a key at random else."""
    at = 0.0
    for key in KEYS:
        at = _held(body, faces, key, at)
        at = _held(body, faces, "", at)
    trials = 0
    while at < seconds:
        key = faces.trial(KEYS) if trials < 10 else None
        trials += key is not None
        at = _held(body, faces, key if key is not None else rand.choice([*KEYS, ""]), at)
    return at


def test_a_body_that_turns_is_found_to_and_driven_to_one_place_after_another():
    reached = []
    for seed in range(6):
        rand, body, faces = random.Random(seed), _Turning(), WhichWaySheFaces()
        at = _tried(body, faces, 10.0, rand)
        assert faces.turns_her(), (faces.turns, {k: len(v) for k, v in faces.ways.items()})
        assert faces.ahead_key() == "up"
        # Each key found to turn her is one that does, and turns her the way it does: one is enough to steer by.
        turning = faces.turning_keys()
        assert turning and set(turning) <= {"left", "right"}, turning
        assert all((rate < 0) == (key == "left") for key, rate in turning.items()), turning
        got, target, began = 0, (rand.uniform(40, 440), rand.uniform(40, 280)), at
        while at - began < 20.0:
            key = faces.key_toward(body.x, body.y, *target) or ""
            body.step(key)
            faces.saw(body.vx, body.vy, key, at)
            at += STEP
            if math.hypot(body.x - target[0], body.y - target[1]) < 12:
                got += 1
                target = (rand.uniform(40, 440), rand.uniform(40, 280))
        reached.append(got)
    assert min(reached) >= 6, reached


def test_a_body_whose_keys_move_it_one_way_is_not_taken_to_turn():
    rand, body, faces = random.Random(3), _OneWay(), WhichWaySheFaces()
    _tried(body, faces, 12.0, rand)
    assert not faces.turns_her()
    assert faces.key_toward(0, 0, 100, 100) is None or not faces.turns_her()
