"""A pass of play over many games is turned into what her self-improvement weighs: a scorecard with the held-out
games kept apart, regressions against the pass before, and a reason to improve for each game short of what was asked.

Nothing here is a game: an invented pass of four made-up games.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.learning.what_a_pass_of_play_came_to import (
    AS_ASKED,
    GAINED,
    NEVER_PLAYED,
    PLAYED,
    WON,
    evaluator_for,
    levels_in,
    recorded,
    scorecard,
    signals,
)

pytestmark = pytest.mark.unit

STRETCH = "x a stretch played as it happened: {'seconds': 9.0, 'ended': 'out of time', 'pictures_a_second': 20.0, 'gains': %d}\n"


def _a_pass(where: Path, games: dict[int, tuple[dict, str]]) -> Path:
    (where / "w0").mkdir(parents=True)
    (where / "w0" / "run.log").write_text("".join(json.dumps({"game": game, **result}) + "\n" for game, (result, _) in games.items()))
    for game, (_result, log) in games.items():
        (where / "w0" / f"game{game}.log").write_text(log)
    return where


def test_how_far_she_got_is_read_off_what_she_did_and_said(tmp_path):
    played = _a_pass(tmp_path / "p1", {
        1: ({"won": True, "runs": ["won"]}, STRETCH % 3),
        2: ({"runs": ["finished"]}, "x saying out loud: 'That's 3 runs. My best was run 2, with 40.'\n"),
        3: ({"runs": ["lost"]}, STRETCH % 2),
        4: ({"runs": ["unread"]}, STRETCH % 0),
        5: ({"runs": []}, "nothing happened\n"),
    })
    levels = {game: how.level for game, how in levels_in(played).items()}
    assert levels == {1: WON, 2: AS_ASKED, 3: GAINED, 4: PLAYED, 5: NEVER_PLAYED}


def test_the_scorecard_keeps_held_out_games_apart_and_names_what_went_worse(tmp_path):
    before = _a_pass(tmp_path / "p1", {1: ({"runs": ["finished"]}, "x saying out loud: 'That's 3 runs. My best was run 1, with 9.'\n"),
                                       2: ({"runs": ["lost"]}, STRETCH % 2), 9: ({"won": True}, "")})
    after = _a_pass(tmp_path / "p2", {1: ({"runs": ["lost"]}, STRETCH % 0), 2: ({"runs": ["lost"]}, STRETCH % 4),
                                      9: ({"runs": ["lost"]}, STRETCH % 1)})
    card = scorecard(after, held_out={9}, before=before)
    assert card["metrics"]["learned_from_mean"] == pytest.approx((PLAYED + GAINED) / 2)
    assert card["metrics"]["held_out_mean"] == pytest.approx(GAINED)
    assert card["score"] == pytest.approx((PLAYED + GAINED) / 2 / WON)
    assert len(card["regressions"]) == 2 and card["regressions"][0].startswith("game 1: did what was asked before")
    said = signals(after, held_out={9})
    assert [s["evidence"]["game"] for s in said] == [1, 2]            # held-out games give no reason to improve
    assert said[0]["kind"] == "played without gaining" and said[0]["severity"] > said[1]["severity"]


def test_the_latest_pass_is_what_the_self_improvement_loop_evaluates(tmp_path):
    import os
    import time

    campaign = tmp_path / "campaign"
    first = _a_pass(campaign / "p1", {1: ({"runs": ["lost"]}, STRETCH % 2)})
    second = _a_pass(campaign / "p2", {1: ({"won": True}, "")})
    os.utime(first, (time.time() - 100, time.time() - 100))
    evaluate = evaluator_for(campaign, held_out=set())
    assert evaluate()["score"] == pytest.approx(1.0)

    class Loop:
        def __init__(self):
            self.signals = []

        def record_signal(self, source, kind, **rest):
            self.signals.append((source, kind, rest))

    loop = Loop()
    assert recorded(loop, first, held_out=set()) == 1 and loop.signals[0][1] == "gained but did not do what was asked"
    assert second.exists()


def test_how_far_she_got_told_as_her_time_ran_out_is_not_doing_what_was_asked(tmp_path):
    played = _a_pass(tmp_path / "p1", {1: ({"runs": ["lost", "lost"]}, STRETCH % 2 + "x saying out loud: \"I've given this "
                                           "one its share of the time, and I leave it here. That's 2 runs. My best was run 1, "
                                           "with 3.\"\n")})
    assert levels_in(played)[1].level == GAINED
