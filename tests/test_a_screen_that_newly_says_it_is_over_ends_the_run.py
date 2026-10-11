"""A run ends where the screen says so in words it did not show in play, though its way to start again is not new.

LIVE 2026-10-11 a two-minute game ended on "Your final score : 6500" with the Play button its title screen had shown.
A way back is news only when it was not there before, so the end was passed over, she played on into the next game,
and each of her three runs was ended part-way through a game by her own clock: "Run 2 of 3: scored 1000" was said
with 37 seconds of that game still to run. The same day a second game of falling pieces ended on the screen the first
had ended on, and she played the end screen for a minute and a half.
"""
from __future__ import annotations

import time

import pytest

from core.language.how_a_game_ended import what_says_the_game_is_over
from core.skills.screen_pursuit_as_it_happens import PlayingAsItHappens

pytestmark = pytest.mark.unit


def _played(words_seen: list[str]) -> PlayingAsItHappens:
    reflexes = PlayingAsItHappens(page=None, band=(0, 0, 1, 1), goal="play it", ends_at=time.monotonic() + 600)
    reflexes.stretches.append({"pictures": 400, "words_seen": words_seen})
    return reflexes


def _screen(text: str, *buttons: str) -> dict:
    return {"text": text, "layout": [{"text": label} for label in buttons]}


def test_the_words_of_an_end_are_known():
    assert what_says_the_game_is_over("Your final score : 6500") == "your final score"
    assert what_says_the_game_is_over("GAME OVER - BLOCKED DROP") == "game over"
    assert what_says_the_game_is_over("SCORE 0001000 PTS TIME LEFT 01:25") == ""
    # A level finished is not the game over.
    assert what_says_the_game_is_over("LEVEL COMPLETE! Well done. NEXT") == ""


def test_a_final_score_with_a_way_back_seen_before_ends_the_run():
    reflexes = _played(["play", "score pts bombs shield speed time left"])
    assert reflexes.run_is_over(_screen("Your final score : 6500 Play", "Play"))
    assert "your final score" in reflexes.over_because and "6500" in reflexes.ending_words


def test_a_way_back_on_screen_all_game_long_with_nothing_said_does_not():
    reflexes = _played(["play again", "score 120 level 2"])
    assert not reflexes.run_is_over(_screen("SCORE 120 LEVEL 2 Play Again", "Play Again"))


def test_words_of_an_end_that_were_up_while_she_played_do_not():
    # A game that keeps "Retry" and "Game over in 3 misses" in view all game long has not ended by showing them.
    reflexes = _played(["retry", "game over in 3 misses score 4"])
    assert not reflexes.run_is_over(_screen("Game over in 3 misses SCORE 5 Retry", "Retry"))


def test_play_that_knows_its_game_stopped_ends_the_run_at_a_way_back_seen_before():
    reflexes = _played(["play again"])
    reflexes.play_stopped = True
    assert reflexes.run_is_over(_screen("SONIC LINK N SMASH LEVEL SCORE 2440 Play Again", "Play Again"))
