"""A current turn and instructions cannot become a completed game through a predicted ending."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.language.how_a_game_ended import (
    declares_a_win,
    how_it_ended,
    how_it_ended_in,
    says_a_round_is_over,
    says_an_unresolved_result,
    says_play_is_ongoing,
    what_says_the_game_is_over,
)
from core.skills.sovereign_browser_drawing import (
    _current_play_veto_invariant,
    _how_the_run_went,
    _measure_runs_count_invariant,
)

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


@pytest.mark.parametrize("words", [
    "Your turn. Your final score: 120", "If you lose, your final score is shown.",
    "When the game is over, try again.", "Your score will be shown when you finish.",
])
def test_whole_game_end_words_use_the_same_current_assertion_scope(words):
    assert what_says_the_game_is_over(words) == ""


async def _score_play(monkeypatch, slices, *, conditions=None):
    """Exercise the production completion route with returned observations and a bounded clock."""
    from core.skills import sovereign_browser_drawing as drawing

    clock, calls, restarted, said = [0.0], [], [], []
    queue = list(slices)

    async def stock(*args):
        return None

    async def one(page, band, goal, url, deadline, keep):
        item = queue.pop(0)
        calls.append(item)
        clock[0] += item.get("elapsed", 10.0)
        words = item["words"]
        reflexes = SimpleNamespace(
            keep=keep, words=["Collect as many points as possible."], ending_words=words, ending_parts=[words],
            over_because=item.get("over", "observed terminal"), new_screens=0,
            stretches=[{"pictures": 20, "counters": {"score": item.get("score", 120)},
                        "runtime_checks": {"violations": item.get("violations", [])}}],
            what_it_came_to=lambda: "observed play",
        )
        return {"moves": [{"key": "left"}], "last_seen": words, "completed": item.get("completed", False)}, reflexes

    monkeypatch.setattr(drawing, "_a_stocktaking", stock)
    monkeypatch.setattr(drawing, "_one_run", one)
    monkeypatch.setattr(drawing, "_tell", said.append)
    monkeypatch.setattr("core.skills.screen_pursuit_as_it_happens.begin_run", lambda *args: restarted.append(args))
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    conditions = conditions or (
        "If a game can be won, keep playing it until you win; if it only keeps a score, "
        "play it three times and tell me your best."
    )
    result = await drawing._played(None, (0, 0, 1, 1), "Play this game. " + conditions, "https://a.example/game", {})
    return result, calls, restarted, said


@pytest.mark.asyncio
async def test_an_ongoing_scored_game_at_the_deadline_is_not_a_finished_attempt(monkeypatch):
    done, calls, restarted, said = await _score_play(monkeypatch, [
        {"words": "YOUR TURN TO MAKE A MOVE SCORE: 120", "over": "", "elapsed": 100000, "completed": True},
    ])
    assert len(calls) == 1 and restarted == []
    assert done["runs"] == ["unread"] and done["completed_runs"] == 0
    assert not done["completed"] and not done["ok"] and done["best_score"] is None
    assert not done["the_ask_does_not_apply"]
    assert not any("That's 1 run" in line for line in said)


@pytest.mark.asyncio
@pytest.mark.parametrize("conditions", [
    None, "If it only keeps a score, play it three times and tell me your best.",
])
async def test_one_ended_scored_run_at_the_deadline_does_not_satisfy_three(monkeypatch, conditions):
    done, calls, restarted, _said = await _score_play(monkeypatch, [
        {"words": "GAME OVER YOUR SCORE: 120", "elapsed": 100000},
    ], conditions=conditions)
    assert len(calls) == 1 and restarted == []
    assert done["completed_runs"] == 1 and done["requested_measure_runs"] == 3
    assert done["best_score"] == 120 and not done["completed"] and not done["ok"]
    assert "only 1 of 3" in done["stopped_because"]
    assert not done["the_ask_does_not_apply"]


@pytest.mark.asyncio
async def test_ongoing_score_slices_continue_the_same_run_without_becoming_attempts(monkeypatch):
    done, calls, restarted, said = await _score_play(monkeypatch, [
        {"words": "Your turn SCORE: 999", "score": 999, "over": ""},
        {"words": "TIME LEFT 01:25 SCORE: 150", "score": 150, "over": ""},
        *[{"words": f"Your final score: {score}", "score": score} for score in (120, 340, 200)],
    ])
    assert len(calls) == 5 and len(restarted) == 2
    assert done["completed_runs"] == 3 and done["completed"] and done["ok"]
    assert done["runs"][:2] == ["unread", "unread"]
    assert done["best_score"] == 340
    assert any("That's 3 runs. My best was run 2, with 340" in line for line in said)


@pytest.mark.asyncio
async def test_runtime_violations_cannot_be_skipped_by_the_score_completion_branch(monkeypatch):
    done, calls, restarted, _said = await _score_play(monkeypatch, [
        {"words": "GAME OVER YOUR SCORE: 120", "violations": ["unowned input"]},
    ])
    assert len(calls) == 1 and restarted == []
    assert not done["completed"] and not done["ok"]
    assert done["runtime_violations"] == ["unowned input"]


def test_the_score_count_canary_checks_only_completed_runs():
    assert _measure_runs_count_invariant() == ()


@pytest.mark.parametrize("parts", [
    ["Mara wins!"], ["Johnny lost!"], ["You won, the opponent won."],
    ["You win!", "Mara wins!"], ["GAME OVER. You 5 Mara 3"],
])
def test_an_unresolved_current_result_vetoes_the_real_callers_board_fallback(parts):
    assert says_an_unresolved_result("\n".join(parts))
    reflexes = _reflexes(parts)
    reflexes.over_because = "the board game ended: won"
    assert _how_the_run_went(reflexes, {"last_seen": " ".join(parts)})["ended"] == ""


@pytest.mark.parametrize("counter_values", [[90, 85], ["01:30", "01:25"]])
@pytest.mark.parametrize("predicted", ["won", "lost"])
def test_a_still_decreasing_observed_clock_vetoes_an_inferred_terminal(predicted, counter_values):
    reflexes = _reflexes(["SCORE 120 TIME LEFT 01:25"], inferred=predicted)
    reflexes.over_because = f"the board game ended: {predicted}"
    reflexes.stretches = [{"pictures": 16, "counters": {"time": value}, "settled": predicted} for value in counter_values]
    assert _how_the_run_went(reflexes, {"last_seen": "SCORE 120 TIME LEFT 01:25"})["ended"] == ""


def test_two_current_ocr_clock_readings_veto_an_inferred_terminal_without_counter_labels():
    reflexes = _reflexes(["SCORE 120 TIME LEFT 01:25"])
    reflexes.words = ["SCORE 100 TIME LEFT 01:30", "SCORE 120 TIME LEFT 01:25"]
    assert _how_the_run_went(reflexes, {"last_seen": reflexes.words[-1]})["ended"] == ""


@pytest.mark.parametrize("predicted", ["won", "lost", "drawn"])
def test_terminal_board_evidence_is_retained_without_current_contradictions(predicted):
    reflexes = _reflexes(["board"], inferred=predicted)
    assert _how_the_run_went(reflexes, {})["ended"] == predicted


def test_a_frozen_positive_clock_remainder_does_not_erase_terminal_board_evidence():
    reflexes = _reflexes(["SCORE 120 TIME LEFT 01:25"])
    reflexes.words = ["SCORE 100 TIME LEFT 01:30", "SCORE 120 TIME LEFT 01:25"]
    reflexes.stretches = [{"pictures": 16, "counters": {"time": value}, "settled": "won"} for value in (90, 85, 85)]
    assert _how_the_run_went(reflexes, {"last_seen": reflexes.words[-1]})["ended"] == "won"


def test_an_explicit_terminal_declaration_retains_authority_over_an_earlier_running_clock():
    reflexes = _reflexes(["YOU WIN! TIME LEFT 01:25"], inferred="lost")
    reflexes.stretches = [{"pictures": 16, "counters": {"time": value}, "settled": "lost"} for value in (90, 85)]
    assert _how_the_run_went(reflexes, {"last_seen": "YOU WIN! TIME LEFT 01:25"})["ended"] == "won"


@pytest.mark.parametrize("reason", ["out of time", "no move is left to her", "the picture could not be taken"])
def test_a_controller_stop_reason_is_not_a_terminal_outcome(reason):
    assert _how_the_run_went(_reflexes(["board"], inferred=reason), {})["ended"] == ""
