"""How a program plays is read from its code: what each key does, what the keys move, and what passes a level, fails
it and wins the game.

Read against real games where their files are here (the offline museum of Flash games), each checked against what a
reading of its code found by hand (tests/fixtures/flash_games_as_their_code_says/); and against invented code for each
way a program says these things.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from core.perception.how_a_program_plays import functions_of, how_it_plays

pytestmark = pytest.mark.unit

_MUSEUM = Path(os.path.expanduser("~/.aura/flash-campaign/offline-bundle/programs"))
_AS_THEIR_CODE_SAYS = Path(__file__).parent / "fixtures" / "flash_games_as_their_code_says"


def _played(game: int):
    from core.perception.reading_a_flash_program import read_flash

    path = _MUSEUM / f"{game}.swf"
    if not path.exists():
        pytest.skip("the offline museum of Flash games is not on this machine")
    return how_it_plays(read_flash(path.read_bytes())["code"])


def test_a_coverage_game_is_read_as_its_code_says():
    """Camp Lazlo: Paintcan Panic, against the card written from its code."""
    card = json.loads((_AS_THEIR_CODE_SAYS / "25.json").read_text())
    plays = _played(25)
    for key in ("left", "up", "right", "down"):
        assert plays.keys[key].does == "moves", key
    assert plays.keys["z"].does == "jumps"
    assert plays.keys["p"].does == "pauses" and plays.keys["m"].does == "sounds"
    # Its cheat keys end the level outright: no part of play.
    assert plays.keys["w"].does == plays.keys["l"].does == "ends the level"
    assert set(plays.played_with()) >= {"left", "up", "right", "down", "z"} and "w" not in plays.played_with()
    assert plays.hers["class"] == card["hers"]["class"] and plays.hers["symbol"] == card["hers"]["symbol"]
    assert plays.hers["numbers"]["_speedMax"] == card["hers"]["speed_max"]
    assert "getCoverage() >= _quota" in plays.when("passed")
    assert "_paintBucket < 0" in plays.when("failed")
    assert "_levelIndex == 20" in plays.when("won")
    passed = next(end for end in plays.ends if end.kind == "passed" and "getCoverage" in end.when)
    assert passed.made_of["getCoverage"] == "_paintCount / 2.4"
    # The cheat keys' endings are kept apart from play's.
    assert all("Key." not in when for kind in ("passed", "failed", "won") for when in plays.when(kind))


def test_keys_named_by_constants_and_set_into_flags_are_read():
    code = [
        "// frame script in the main timeline",
        "KJUMP = 32",
        "  // anonymous function ()",
        "  key_left = Key.isDown(37)",
        "  if !Key.isDown(r1.KJUMP) goto +30",
        "  r1.player.startJump()",
        "  if !Key.isDown(80) goto +12",
        "  r1.paused = !r1.paused",
        "Game.readKeys = function () {...}",
    ]
    plays = how_it_plays(code)
    assert plays.keys["left"].does == "moves"
    assert plays.keys["space"].does == "jumps" and plays.keys["space"].by == "player.startJump()"
    assert plays.keys["p"].does == "pauses"
    assert plays.hers["class"] == "player"


def test_what_a_method_reads_is_its_class_s_and_ends_are_read_the_right_way_round():
    code = [
        "// init script for symbol 7",
        "  // anonymous function (game, mc)",
        "  r1._mc = r2",
        "_global.Rover = function (game, mc) {...}",
        "function (game, mc) {...}.prototype",
        "  // anonymous function ()",
        "  if !Key.isDown(39) goto +20",
        "  r1.moveBy(r1._speed, 0)",
        "function (game, mc) {...}.prototype.steer = function () {...}",
        "function (game, mc) {...}.prototype._speed = 5",
        "  // anonymous function ()",
        "  if !!(r1.cratesLeft() > 0) goto +16",
        "  r1.finishLevel('passed')",
        "  if !(r1._fuel <= 0) goto +12",
        "  r1.finishLevel('failed')",
        "function (game, mc) {...}.prototype.check = function () {...}",
        "  // anonymous function ()",
        "  return r1._crates.length",
        "function (game, mc) {...}.prototype.cratesLeft = function () {...}",
        "game.attachMovie('roverArt', 'rover_mc', 3)",
        "r1.rover = new Rover(r1, game.attachMovie('roverArt', 'rover_mc', 3))",
    ]
    names = {(f.owner, f.name) for f in functions_of(code)}
    assert ("Rover", "steer") in names and ("Rover", "check") in names
    plays = how_it_plays(code)
    assert plays.keys["right"].does == "moves"
    assert plays.hers == {"class": "Rover", "symbol": "roverArt", "numbers": {"_speed": 5.0}}
    assert plays.when("passed") == ["cratesLeft() <= 0"] and plays.when("failed") == ["_fuel <= 0"]
    assert plays.ends[0].made_of == {"cratesLeft": "_crates.length"}


def test_a_sound_named_for_failing_ends_nothing():
    code = ["  // anonymous function ()", "  if !(r1.tries > 3) goto +9", "  r1.sounds.PlayMusic('fail')",
            "Board.lost = function () {...}"]
    assert how_it_plays(code).ends == []


def test_what_its_code_says_is_the_surest_thing_her_guide_knows_of_how_it_is_played():
    """The keys its code moves its hero by are the keys of play, whatever else is tested; a cheat, a pause and the
    sound are no part of it; and how a level is passed and failed is said in plain words."""
    from core.cognition.a_guide_to_a_place import Guide
    from core.cognition.what_its_code_says import CODE, in_plain_words, take_in_how_it_plays

    code = [
        "  // anonymous function ()",
        "  if !Key.isDown(37) goto +20",
        "  r1.hero.moveLeft()",
        "  if !Key.isDown(88) goto +20",
        "  r1.hero.throwPie()",
        "  if !Key.isDown(80) goto +12",
        "  r1.paused = !r1.paused",
        "  if !Key.isDown(75) goto +12",
        "  r1.board.skipLevel()",
        "Board.keys = function () {...}",
        "  // anonymous function ()",
        "  if !!(r1.board.piesLeft > 0) goto +12",
        "  r1.board.endRound('lose')",
        "  if !!(r1.board.getHits() < r1.board._target) goto +12",
        "  r1.board.endRound('pass')",
        "Board.check = function () {...}",
    ]
    guide = Guide(place="Pie Toss")
    take_in_how_it_plays(guide, how_it_plays(code), 0.0)
    assert guide.controls["left"].act == "move" and guide.controls["x"].act == "act"
    assert {c.source for c in guide.controls.values()} == {CODE}
    assert guide.keys_for_play() == ["left", "x"]
    assert "pause (p)" in guide.not_for_play and "a cheat (k)" in guide.not_for_play
    assert guide.win == ["a level is passed when hits ≥ target"]
    assert guide.lose == ["a level is failed when pies left ≤ 0"]
    assert in_plain_words("_paintCount / 2.4") == "paint count / 2.4"


def test_what_a_program_says_of_how_it_plays_is_kept_with_its_read():
    from core.perception.how_a_program_plays import kept, unkept

    plays = how_it_plays(["  // anonymous function ()", "  if !Key.isDown(32) goto +9", "  r1.jump()",
                          "Hero.prototype.keys = function () {...}"])
    again = unkept(kept(plays))
    assert again.keys["space"].does == "jumps" and again.hers == plays.hers
    assert unkept({}) is None and kept(None) == {}
