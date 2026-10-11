"""A current turn and instructions cannot become a completed game through a predicted ending."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.language.how_a_game_ended import (
    declares_a_win,
    how_it_ended,
    how_it_ended_in,
    says_a_round_is_over,
    says_play_is_ongoing,
)
from core.skills.sovereign_browser_drawing import _current_play_veto_invariant, _how_the_run_went

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("words", [
    "Your turn to make a move", "Computer's turn", "Your move", "Round is in progress",
    "Waiting for the opponent to move", "Resume the game",
    "You win! YOUR TURN TO MAKE A MOVE", "You lost. Game continues",
    "Lives: 0 Your turn", "You 8 Opponent 2 Your turn",
])
def test_current_play_contradicts_a_terminal_claim(words):
    assert says_play_is_ongoing(words)
    assert how_it_ended(words) == ""
    assert not says_a_round_is_over(words)
    assert not declares_a_win(words)


@pytest.mark.parametrize("words", [
    "To win, capture every piece.", "If you win, advance to the next level.",
    "You win when you reach the last row.", "First to 5 points wins.",
    "You will win if you collect enough points.", "You can win the game.",
    "You did not win.", "You didn't lose.", "Don't crash into the wall.",
    "Avoid the guards or you lose.", "Before time runs out, try again.",
    "The previous round you won.", "Last game you lost.",
    "When you have 0 lives left, the game is over.", "Health: 0 if the enemy hits you.",
    "You 5 Computer 3", "Player 2 Computer 9", "Energy: 0",
])
def test_rules_possibilities_history_and_unsettled_scores_do_not_end_play(words):
    assert how_it_ended(words) == ""
    assert not says_a_round_is_over(words)
    assert not declares_a_win(words)


@pytest.mark.parametrize(("words", "expected"), [
    ("YOU WIN!", "won"), ("You have won!", "won"), ("The computer wins.", "lost"),
    ("You were defeated.", "lost"), ("The computer was defeated.", "won"), ("VICTORY!", "won"),
    ("You lose!", "lost"), ("You have failed.", "lost"), ("You died.", "lost"),
    ("GAME OVER. You 5 Computer 3", "won"), ("GAME OVER. You 2 Computer 5", "lost"),
    ("You 9 Computer 4 Press SPACE to play again", "won"),
    ("Congratulations! Level 4 complete.", "won"), ("Score: 120 Lives: 0", "lost"),
])
def test_current_terminal_statements_still_settle_the_run(words, expected):
    assert how_it_ended(words) == expected


def test_split_ocr_readings_keep_current_play_above_a_predicted_result():
    parts = ["BIG SHOT CHECKERS", "BIOD mAC", "YOUR TURN TO MAKE A MOVE"]
    assert how_it_ended_in(parts) == ""
    assert how_it_ended_in(["You win!", *parts]) == ""
    assert how_it_ended_in(["The computer wins.", "YOUR", "TURN TO MAKE A MOVE"]) == ""


def test_split_terminal_words_and_scores_are_read_together():
    assert how_it_ended_in(["You win!", "You 5 Computer 3", "Play again"]) == "won"
    assert how_it_ended_in(["GAME OVER", "You 2 Computer 9"]) == "lost"
    assert how_it_ended_in(["You win!", "The computer wins."]) == ""


def test_future_and_historical_turns_do_not_override_a_current_terminal_statement():
    assert not says_play_is_ongoing("If it is your turn, make a move")
    assert not says_play_is_ongoing("Your turn is over")
    assert how_it_ended("You win! In the previous game it was your turn.") == "won"


def test_the_adjacent_invariant_checks_the_actual_completion_seam():
    assert _current_play_veto_invariant() == ()


def _reflexes(parts, *, inferred="won", violations=()):
    return SimpleNamespace(
        ending_words=" ".join(parts), ending_parts=parts,
        stretches=[{"pictures": 16, "settled": inferred, "runtime_checks": {"violations": list(violations)}}],
        what_it_came_to=lambda: "Sixteen moves were made.",
    )


@pytest.mark.parametrize("inferred", ["won", "lost"])
def test_a_board_predictions_terminal_result_is_vetoed_by_the_current_turn(inferred):
    parts = ["BIG SHOT CHECKERS", "BIOD mAC", "YOUR TURN TO MAKE A MOVE"]
    run = _how_the_run_went(_reflexes(parts, inferred=inferred), {"last_seen": " ".join(parts)})
    assert run["ended"] == ""


def test_the_current_screen_vetoes_an_older_ending_reading():
    run = _how_the_run_went(_reflexes(["YOU WIN!"]), {"last_seen": "Your turn to make a move"})
    assert run["ended"] == ""


def test_runtime_violations_keep_their_interruption_even_with_an_ongoing_turn():
    run = _how_the_run_went(_reflexes(["Your turn"], violations=["unowned input"]), {})
    assert run["ended"] == "interrupted"


@pytest.mark.parametrize("words", [
    "Johnny wins!", "Mara won!", "Winner: Johnny", "Victory for Mara!", "Mara is victorious.",
    "Johnny lost!", "Mara was defeated.", "Johnny has failed.",
    "GAME OVER. Johnny wins!", "Congratulations! Mara is the winner.",
    "GAME OVER. You 5 Johnny 3",
])
def test_an_unknown_participants_result_does_not_establish_the_players_result(words):
    assert how_it_ended(words) == ""
    assert how_it_ended_in([words]) == ""
    assert not declares_a_win(words)


@pytest.mark.parametrize(("words", "expected"), [
    ("Player one wins!", "won"), ("Player 1 won.", "won"), ("P1 is victorious.", "won"),
    ("Winner: You", "won"), ("Victory for you!", "won"), ("You are the winner.", "won"),
    ("The opponent wins!", "lost"), ("The enemy won.", "lost"), ("Player 2 wins.", "lost"),
    ("Player two is the winner.", "lost"), ("P2 is victorious.", "lost"),
    ("The other player wins.", "lost"), ("Winner: Computer", "lost"),
    ("The computer loses.", "won"), ("The opponent was defeated.", "won"),
    ("You've won!", "won"), ("You've been caught.", "lost"), ("You got captured.", "lost"),
    ("The computer got defeated.", "won"),
])
def test_explicit_participant_roles_settle_the_players_result(words, expected):
    assert how_it_ended(words) == expected
    assert how_it_ended_in([words]) == expected


@pytest.mark.parametrize("parts", [
    ["You win!", "The opponent wins!"], ["Player 1 won.", "Player 2 won."],
    ["You won.", "You lost."], ["You win! The opponent wins!"],
    ["You win!", "Johnny wins!"], ["The computer wins!", "Mara wins!"],
    ["You won, Johnny wins!"], ["You won and the computer won."],
    ["You won, you lost."], ["The computer won and Mara wins."],
])
def test_conflicting_or_unbound_current_results_abstain(parts):
    assert how_it_ended_in(parts) == ""
    assert how_it_ended("\n".join(parts)) == ""


def test_multiple_consistent_asserted_results_in_one_clause_settle_the_run():
    assert how_it_ended("You won and the computer lost.") == "won"
    assert how_it_ended("You were defeated and the opponent won.") == "lost"


@pytest.mark.parametrize("words", [
    "Defeat the robots", "Defeat robots", "Please defeat every target", "Objective: defeat all enemies",
    "Crash the enemy vehicle", "Win the round", "Beat the opponent", "Lose all health to restart",
])
def test_subject_free_action_directives_are_not_terminal_assertions(words):
    assert how_it_ended(words) == ""
    assert how_it_ended_in([words]) == ""
    assert not says_a_round_is_over(words)
    assert not declares_a_win(words)


@pytest.mark.parametrize(("words", "expected"), [
    ("Defeat!", "lost"), ("You defeat the opponent.", "won"),
    ("The enemy defeats you.", "lost"), ("You crashed.", "lost"),
])
def test_terminal_labels_and_explicit_participant_assertions_are_preserved(words, expected):
    assert how_it_ended(words) == expected
    assert says_a_round_is_over(words)


def test_instruction_read_during_play_does_not_end_the_actual_stretch():
    from core.agency.playing_as_it_happens import (
        SAID_IT_IS_OVER,
        _Run,
        _said_it_is_over,
        _the_words_of_play,
    )

    run = _Run(keys=["up"], began=0.0)
    run.first_moving = 0.5
    run.words_read = [(1.0, "landings: 0"), (5.0, "defeat the robots"), (9.0, "fuel")]
    assert _said_it_is_over(run) == ""
    run.words_read += [(12.0, "ouch!"), (12.0, "try again")]
    assert _said_it_is_over(run) == SAID_IT_IS_OVER
    assert "defeat the robots" in _the_words_of_play(run, SAID_IT_IS_OVER)
    assert "try again" not in _the_words_of_play(run, SAID_IT_IS_OVER)
