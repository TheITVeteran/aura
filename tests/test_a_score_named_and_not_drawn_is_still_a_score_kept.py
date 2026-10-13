"""A game that names a score on its last screen and draws no number for it is still a game that only keeps a score.

Offline 2026-10-12 a game ended "GAME OVER  YOUR SCORE:  YOUR RATING: CRACKED UP  PLAY AGAIN", the number left blank by
the player it ran in. Four finished runs were counted as four games lost, and the best of three was never told.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.language.how_a_game_ended import names_a_measure, the_rating_in, the_score_in
from core.skills.sovereign_browser_drawing import _only_a_score, _scored, _the_best_of

pytestmark = pytest.mark.unit

BLANK = "GAME OVER YOUR SCORE: YOUR RATING: CRACKED UP PLAY AGAIN"


def test_a_measure_is_named_whether_or_not_its_number_is_drawn():
    assert names_a_measure(BLANK) and the_score_in(BLANK) is None
    assert names_a_measure("Game Over Your score: Your rating: LONG SHOT Play Again")
    assert names_a_measure("FINAL SCORE: 120")
    assert not names_a_measure("GAME OVER TRY AGAIN")
    assert not names_a_measure("Score a goal to win the match")          # told to score, not shown a score


def test_the_rating_is_read_in_its_own_words():
    assert the_rating_in(BLANK) == "CRACKED UP"
    assert the_rating_in("Game Over Your score: Your rating: LONG SHOT Play Again") == "LONG SHOT"
    assert the_rating_in("GAME OVER TRY AGAIN") == ""


def test_a_run_that_ends_on_a_blank_score_is_finished_not_lost():
    playing = SimpleNamespace(words=["HOW TO PLAY Use the mouse to move. Click to throw the ball."])
    assert _only_a_score(playing, {"words": BLANK, "score": None})
    # Where the game holds out a win, a game over is a loss as before.
    against = SimpleNamespace(words=["Beat your opponent to win the match"])
    assert not _only_a_score(against, {"words": BLANK, "score": None})
    # And a last screen that names no measure says nothing of a score.
    assert not _only_a_score(playing, {"words": "GAME OVER TRY AGAIN", "score": None})


def test_the_best_of_runs_with_no_number_drawn_is_told_by_her_own_count_and_their_ratings():
    runs = [{"words": BLANK, "score": None, "gains": 4, "took_s": 40.0},
            {"words": BLANK.replace("CRACKED UP", "SUNNY SIDE UP"), "score": None, "gains": 9, "took_s": 55.0},
            {"words": BLANK, "score": None, "gains": 6, "took_s": 61.0}]
    assert _scored(runs[1]) == "9 by my own count, rated SUNNY SIDE UP"
    said = _the_best_of(runs)
    assert said.startswith("That's 3 runs. No score was drawn; by my own count my best was run 2, with 9, rated SUNNY SIDE UP")
    assert "4 by my own count, rated CRACKED UP" in said and "6 by my own count" in said


def test_a_number_drawn_is_still_what_the_best_is_told_by():
    runs = [{"words": "GAME OVER YOUR SCORE: 120", "score": 120, "gains": 3, "took_s": 30.0},
            {"words": "GAME OVER YOUR SCORE: 90", "score": 90, "gains": 8, "took_s": 30.0}]
    assert _the_best_of(runs).startswith("That's 2 runs. My best was run 1, with 120")


def test_runs_with_no_score_drawn_or_counted_are_told_by_what_the_game_rated_them():
    runs = [{"words": BLANK, "score": None, "took_s": 40.0}, {"words": "Game Over Your score: Your rating: LONG SHOT",
             "score": None, "took_s": 30.0}]
    assert _the_best_of(runs) == "That's 2 runs. No score was drawn; run 1 was rated CRACKED UP; run 2 was rated LONG SHOT."
    same = [{"words": BLANK, "score": None}, {"words": BLANK, "score": None}]
    assert _the_best_of(same) == "That's 2 runs. No score was drawn; each was rated CRACKED UP."
