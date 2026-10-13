"""A thing that comes for her is imagined turning toward where she will be only as fast as it has been seen to turn.

Camp Lazlo: Paintcan Panic, game 25 of the 56, is played here from the card its code was read into by hand
(tests/fixtures/flash_games_as_their_code_says/25.json, tests/fixtures/worlds_that_move/paintcan.html): its chasers
push toward her by 0.4 pixels a frame each frame and keep 0.97 of their going, and reach twice her speed. Imagined going
straight on, they were walked into; imagined turning to her at once, a step aside never cleared one. Measured on its
seventh level (three chasers, 65% to paint, a touch costing 1000 of 5000 paint), seeds 1 to 12, three decisions a
second, each a decision late and the waits uneven by up to 30%, with what her reading of a Flash program gives her: she
won 1 with chasers imagined going straight on and 6 with each turning as fast as it had been seen to. On its
twentieth level at eight decisions a second, answering in a quarter of a second, it was 12 and 11: what a chaser does
next is seen in time, so it is imagined turning only where she answers more slowly than that.
"""
from __future__ import annotations

import math

import pytest

from core.agency.what_comes_for_her import how_it_turns

pytestmark = pytest.mark.unit


def _a_path(push: float, keep: float, frames: int, fps: float = 31.0, every: int = 4):
    """A chaser's path, seen every few frames: pushed toward (300, 0) from (0, 0) as the card's chasers are."""
    x = y = vx = vy = 0.0
    out = []
    for frame in range(frames):
        dx, dy = 300.0 - x, 100.0 - y
        d = math.hypot(dx, dy) or 1.0
        vx, vy = vx * keep + dx / d * push, vy * keep + dy / d * push
        x, y = x + vx, y + vy
        if frame % every == 0:
            out.append((frame / fps, x, y))
    return out


def test_how_fast_a_chaser_turns_and_goes_is_read_off_its_path():
    turn, top = how_it_turns(_a_path(0.4, 0.97, 60))
    # Its going changes by its push, 0.4 a frame each frame (384 pixels a second each second at 31 frames a second),
    # and by what keeping 0.97 takes off, at most 0.03 of its greatest speed of 0.4 / 0.03 a frame.
    assert 0.4 * 31 * 31 * 0.6 < turn <= (0.4 + 0.03 * 0.4 / 0.03) * 31 * 31
    assert top > 0.4 / 0.03 * 31 * 0.5                   # past half its greatest speed, in a second and a half
    slow_turn, _top = how_it_turns(_a_path(0.15, 0.97, 60))
    assert slow_turn < turn


def test_a_chaser_is_imagined_turning_only_where_she_answers_slowly():
    from types import SimpleNamespace

    from core.agency.imagining_ahead import CHASED_WHERE_ANSWERING_S, _the_scene
    from core.agency.what_meeting_things_does import AVOID

    path = [(at, 200.0 - 40.0 * at, 100.0 + 20.0 * at * at) for at in (0.0, 0.15, 0.3, 0.45, 0.6, 0.75)]
    chaser = SimpleNamespace(number=2, kind=2, x=170.0, y=111.0, w=8.0, h=8.0, vx=-40.0, vy=30.0, moved=True, path=path,
                             where_at=lambda after: (170.0 - 40.0 * after, 111.0 + 30.0 * after))
    mine = SimpleNamespace(number=1, kind=1, x=100.0, y=100.0, w=8.0, h=8.0, vx=0.0, vy=0.0)

    def chasing(answer_s):
        choosing = SimpleNamespace(
            mine=mine, ways={"left": (-80.0, 0.0), "right": (80.0, 0.0)}, moves=SimpleNamespace(shape=(160, 240)),
            others=lambda: [chaser], stance=lambda thing: AVOID, physics=None, caution=1.0, response_s=0.0,
            lifts=[], press_paths=None, rest=(0.0, 0.0), barred=None, answer_s=lambda: answer_s,
            comes=SimpleNamespace(comes_for_her=lambda thing, size: True))
        scene = _the_scene(choosing, 1.2)
        return bool(scene.chasing.any()), scene

    slow, scene = chasing(CHASED_WHERE_ANSWERING_S + 0.1)
    assert slow and math.isfinite(scene.chase_turn[0]) and scene.chase_turn[0] > 0
    assert not chasing(CHASED_WHERE_ANSWERING_S - 0.1)[0]


def test_too_little_of_a_path_says_nothing():
    assert how_it_turns([]) is None and how_it_turns([(0.0, 1.0, 1.0), (0.05, 2.0, 2.0)]) is None


def test_a_fast_thing_seen_between_two_choices_is_the_same_thing():
    from tools.oracles.a_world_run_from_its_code import node_is_here
    from tools.oracles.her_choices_on_a_world import HerChoicesOnAWorld

    if not node_is_here():
        pytest.skip("node is not installed")
    with HerChoicesOnAWorld("paintcan", query="level=20", seed=1, decide_every=1 / 8, late_by=1, jitter=0.3,
                            reading=_paintcan_as_read()) as play:
        play.run(10.0)
        numbered = play._numbered - 10                   # noqa: SLF001 - how many numbers its chasers were given
    assert numbered == 3


def _paintcan_as_read():
    """What her reading of the Flash program gives her of game 25, in the world's own pixels: her pace from its
    greatest speed a frame at its 31 frames a second, the field she is kept in, and its 30 by 25 cells; of its chasers,
    only that touching one costs her, as play comes to know."""
    from core.perception.a_world_as_its_code_makes_it import (
        AKindOfThing,
        AWorldAsItsCodeMakesIt,
        GroundToCover,
        HerBody,
    )

    scale, fps = 0.8, 31.0
    pace = 7 * fps * scale
    field = (10 * scale, 110 * scale, 590 * scale, 390 * scale)
    return AWorldAsItsCodeMakesIt(
        stage=(480.0, 320.0),
        hers=HerBody(name="me", size=(16.0, 16.0),
                     paces={"left": (-pace, 0.0), "right": (pace, 0.0), "up": (0.0, -pace), "down": (0.0, pace)},
                     kept_inside=(field[0] - 8, field[1] - 8, field[2] - 8, field[3] - 8)),
        ground=GroundToCover(cell=(30 * scale, 25 * scale), origin=(field[0], field[1]), cells=(20, 12),
                             marks_at=(8.0, 8.0), kept_in="painted", counted_by="count"),
        things=[AKindOfThing(name="chasers", touching="costs a life", moves="chases her", size=(14.0, 14.0))],
        keys={"left": "moves", "right": "moves", "up": "moves", "down": "moves", "z": "jumps"})


def _played(comes_known: bool) -> list[dict]:
    from tools.oracles.her_choices_on_a_world import HerChoicesOnAWorld

    out = []
    for seed in range(1, 13):
        with HerChoicesOnAWorld("paintcan", query="level=7", seed=seed, decide_every=1 / 3, late_by=1, jitter=0.3,
                                reading=_paintcan_as_read(), comes_known=comes_known) as play:
            out.append(play.run(150.0))
    return out


@pytest.mark.slow
def test_chasers_imagined_turning_as_fast_as_seen_are_got_away_from():
    from tools.oracles.a_world_run_from_its_code import node_is_here

    if not node_is_here():
        pytest.skip("node is not installed")

    def won(played):
        return sum(1 for came in played if came["ended"] and came["ended"][-1]["why"] == "won")

    turning, straight = _played(True), _played(False)
    assert won(turning) >= 5 and won(straight) <= 2, (won(turning), won(straight))
