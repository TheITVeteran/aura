"""A bar that fills and empties is read as it goes: a fall of what she has left is a loss, and low, she is careful.

LIVE 2026-10-09 three heroes shared a health bar and she played as if nothing could hurt her; another game drained a
bucket of paint at every knock, and she stood still while it emptied. Nothing here is either game: a strip on a
panel that shortens in steps, a plain strip that never changes, a band of sky, and a thing crossing the picture.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.perception.how_full_a_bar_is import BarsOnTheScreen

pytestmark = pytest.mark.unit


def _picture(left_of_bar: int, thing_x: int) -> np.ndarray:
    picture = np.full((240, 320, 3), (250, 160, 60), np.uint8)       # sky, running off both sides
    picture[150:240] = (60, 150, 60)                                  # ground
    picture[10:24, 20:140] = (90, 30, 60)                             # the bar's track
    picture[13:21, 22:22 + left_of_bar] = (255, 200, 210)             # the bar, emptying from its right end
    picture[40:48, 200:300] = (30, 30, 200)                           # a plain strip that never changes
    picture[100:130, thing_x:thing_x + 20] = (200, 30, 30)            # something crossing
    return picture


def _played(levels):
    bars, changes = BarsOnTheScreen(), []
    at = 0.0
    for level in levels:
        for _ in range(6):
            at += 0.1
            changes += bars.read(_picture(level, int(at * 40) % 280), at)
    return bars, changes


def test_a_strip_that_shortens_in_steps_and_holds_between_is_a_bar_and_nothing_else_is():
    bars, changes = _played([116, 116, 100, 100, 84, 84, 60, 60, 40])
    found = bars.bars()
    assert len(found) == 1 and found[0].anchored == "start", [(b.row, b.start, b.end) for b in bars.strips]
    # Three changes, each after the level held, make it a bar; from then on each change is told.
    assert [round(c["to"], 2) for c in changes] == [round(60 / 116, 2), round(40 / 116, 2)]
    assert all(c["to"] < c["from"] for c in changes)


def test_a_fall_in_a_bar_named_health_is_a_loss_and_low_she_is_careful():
    from types import SimpleNamespace

    from core.agency.playing_as_it_happens import _Choosing, _read_the_bars, _Run
    from core.agency.what_meeting_things_does import WhatMeetingDoes

    run = _Run(keys=["left"], began=0.0, last_moving=0.0)
    run.regions_read = [{"text": "HEALTH", "center_x": 0.03, "center_y": 0.07}]
    meeting = WhatMeetingDoes()
    moves = SimpleNamespace(shape=(160, 240), things={}, share=lambda x, y: (x / 240, y / 160))
    hers = SimpleNamespace(thing=lambda _moves: None)
    said = []
    at = 0.0
    for level in [116, 116, 100, 100, 70, 70, 40, 40, 20]:
        for _ in range(6):
            at += 0.1
            _read_the_bars(run, meeting, hers, moves, _picture(level, 10), at, said.append)
    assert run.losses >= 2 and [v["what"] for v in meeting.verdicts] == ["loss"] * len(meeting.verdicts)
    assert run.vitals < 0.25 and meeting.vitals == run.vitals
    assert said and "HEALTH bar" in said[0]
    choosing = _Choosing(SimpleNamespace(things={}), SimpleNamespace(thing=lambda _m: None, keys_that_move_her=lambda _k: {},
                                                                     follows_pointer=False, axes=lambda _k: (False, False),
                                                                     makes={}), meeting, [])
    assert choosing.caution > 2.0


def test_a_stretch_with_nothing_of_hers_that_goes_against_her_is_handed_back_and_one_that_pays_is_not():
    """LIVE 2026-10-09 a game's bar went down twice while nothing answered to her, and she played on without anything
    of her own for three minutes."""
    from types import SimpleNamespace

    from core.agency.playing_as_it_happens import _nothing_answers, _Run
    from core.agency.which_one_answers_to_her import WhichIsHers

    run = _Run(keys=[], began=0)
    meeting = SimpleNamespace(verdicts=[{"what": "loss"}], evidence={})
    run.gains, run.losses = 1, 2
    assert not _nothing_answers(run, WhichIsHers(), meeting, 20)
    assert _nothing_answers(run, WhichIsHers(), meeting, 40) == "nothing here answers to me, and it is going against me"
    run.gains = 3
    assert not _nothing_answers(run, WhichIsHers(), meeting, 300)


def test_writing_that_says_the_round_is_over_ends_the_stretch_and_is_not_a_word_of_play():
    """LIVE 2026-10-09 an end screen's stars twinkled under "TRY AGAIN", and she played on at it for half a minute."""
    from core.agency.playing_as_it_happens import (
        SAID_IT_IS_OVER,
        _Run,
        _said_it_is_over,
        _the_words_of_play,
    )

    run = _Run(keys=["up"], began=0.0)
    run.first_moving = 0.5
    run.words_read = [(1.0, "landings: 0"), (5.0, "defeat the robots"), (9.0, "fuel")]
    assert not _said_it_is_over(run)
    run.words_read += [(12.0, "ouch!"), (12.0, "try again")]
    assert _said_it_is_over(run) == SAID_IT_IS_OVER
    assert "try again" not in _the_words_of_play(run, SAID_IT_IS_OVER) and "fuel" in _the_words_of_play(run, SAID_IT_IS_OVER)
    begun_on_one = _Run(keys=["up"], began=0.0)
    begun_on_one.words_read = [(0.5, "play again"), (6.0, "play again")]        # the last round's end, still up
    assert not _said_it_is_over(begun_on_one)


def test_the_end_screen_a_stretch_begins_on_is_not_its_end_however_long_the_first_reading_takes():
    """LIVE 2026-10-09 a stretch begun under an end screen still up read its "TRY AGAIN" first at 2.5 s, took it for
    the round's end, and ended at once, round after round."""
    from core.agency.playing_as_it_happens import _Run, _said_it_is_over

    run = _Run(keys=["up"], began=0.0)
    run.words_read = [(2.5, "landings: 1"), (2.5, "try again"), (5.0, "try again")]
    assert not _said_it_is_over(run)


def test_a_thing_her_eyes_took_for_hers_that_nothing_of_hers_moves_answers_to_nothing():
    """LIVE 2026-10-11 the boy on a game's animated title was taken for her by her eyes, no key moved him, and she
    played the title for a minute before its START: a thing taken for hers answers to her only once something of hers
    moves it."""
    from types import SimpleNamespace

    from core.agency.playing_as_it_happens import _nothing_answers, _Run
    from core.agency.which_one_answers_to_her import WhichIsHers

    run = _Run(keys=[], began=0)
    meeting = SimpleNamespace(verdicts=[], evidence={})
    hers = WhichIsHers()
    hers.seen_to_be_her(SimpleNamespace(number=4, kind=2), 1.0, "child character")
    assert hers.kind == 2
    assert _nothing_answers(run, hers, meeting, 40) == "nothing here answers to me while it moves"
    hers.follows_pointer = True
    assert not _nothing_answers(run, hers, meeting, 40)


def test_a_thing_that_answered_once_and_stands_still_under_press_after_press_answers_to_nothing_now():
    """Offline 2026-10-11 a lander's crash screen kept the lander drawn on it; the keys that had flown it were known, and
    she held them at the picture for three minutes."""
    from types import SimpleNamespace

    from core.agency.playing_as_it_happens import UNANSWERED_PRESSES, _nothing_answers, _Run

    class Hers:
        kind, follows_pointer = 3, False

        def __init__(self):
            self._hers = SimpleNamespace(moved_at=39.0)

        def keys_that_move_her(self, keys):
            return {"up": (0.0, -20.0), "left": (-20.0, 0.0)}

    run = _Run(keys=["up", "left"], began=0.0)
    meeting = SimpleNamespace(verdicts=[], evidence={})
    hers = Hers()
    run.input_key_downs.update({"up": 40, "left": 40})
    assert not _nothing_answers(run, hers, meeting, 40.0)                # it went a moment ago
    run.input_key_downs.update({"up": UNANSWERED_PRESSES})
    assert not _nothing_answers(run, hers, meeting, 45.0)                # pressed, but not long since it went
    assert _nothing_answers(run, hers, meeting, 60.0) == "nothing here answers to me while it moves"
    hers._hers.moved_at = 59.0                                           # it goes again under her keys
    assert not _nothing_answers(run, hers, meeting, 60.0)
    run.input_key_downs.update({"space": 50})                            # keys that never moved it say nothing
    assert not _nothing_answers(run, hers, meeting, 75.0)


def test_a_gauge_that_fills_from_one_end_is_not_her():
    """LIVE 2026-10-11 a shooter's super-shot bar filled while space was held and emptied as the shot went; it answered
    to space and x, and she took it for herself ("I'm the white bar at the top left, moved by space and x"). A body goes
    about whole; a gauge fills from one end that stays put."""
    from types import SimpleNamespace

    from core.agency.which_one_answers_to_her import WhichIsHers

    def thing(number, x, y, w, h, at):
        return SimpleNamespace(number=number, kind=number, x=x, y=y, w=w, h=h, vx=0.0, vy=0.0, seen=at, born=0.0, size=w * h,
                               moved=True)

    hers = WhichIsHers()
    hers.number, hers.kind = 1, 1                         # taken for her, as it was
    at = 1.0
    for step in range(8):
        at += 0.1
        width = 20.0 + 6.0 * step                          # filling rightward from x = 10
        bar = thing(1, 10.0 + width / 2, 12.0, width, 6.0, at)
        body = thing(2, 100.0 + 5.0 * step, 150.0, 24.0, 30.0, at)
        moves = SimpleNamespace(things={1: bar, 2: body}, shape=(300, 400), kinds=[])
        hers.saw(moves, [], at)
    assert hers.number != 1 and 1 in hers.not_mine
    assert not hers._outlines.a_gauge(2)


def test_what_her_presses_bring_out_is_not_taken_for_her():
    """Offline 2026-10-11 each shot of a flying shooter was taken for her as it was fired, nine in sixteen seconds: what
    her presses bring out goes the same way at every press, and she is the thing that was there before the press."""
    from types import SimpleNamespace

    from core.agency.playing_as_it_happens import _found_by_her_presses
    from core.agency.which_one_answers_to_her import Makes, WhichIsHers

    hers = WhichIsHers()
    hers.makes["x"] = Makes("x", 5, 300.0, 0.0, 6)
    run = SimpleNamespace(presses=SimpleNamespace(curves={"x": object()}, kinds_it_moves=lambda far: {5, 6}))
    shot = SimpleNamespace(number=40, kind=5, born=9.8)
    girl = SimpleNamespace(number=3, kind=6, born=1.0)
    _found_by_her_presses(run, SimpleNamespace(things={40: shot, 3: girl}), hers, 10.0)
    assert hers.number == 3


def test_a_body_hovering_where_it_is_is_no_gauge():
    """A figure flying in place moves within itself, its edges stirring on every side: it is not a gauge for doing so
    (offline 2026-10-11 each girl of a flying shooter was taken for a gauge as she hovered)."""
    import random
    from types import SimpleNamespace

    from core.agency.which_one_answers_to_her import _HowOutlinesGo

    outlines, rng = _HowOutlinesGo(), random.Random(3)
    for _ in range(30):
        left, right = 100.0 + rng.choice((0.0, 0.0, -3.0, 3.0)), 140.0 + rng.choice((0.0, 0.0, -3.0, 3.0))
        top, bottom = 50.0 + rng.choice((0.0, -3.0)), 90.0 + rng.choice((0.0, 3.0))
        outlines.saw(SimpleNamespace(number=7, x=(left + right) / 2, y=(top + bottom) / 2, w=right - left, h=bottom - top))
    assert outlines.from_an_end[7] >= 3 and not outlines.a_gauge(7)


def test_a_thing_whose_outlines_were_forgotten_is_no_gauge_and_no_fault():
    """Offline 2026-10-11 a thing's outlines were forgotten with its counts kept, and asking whether it was a gauge
    ended the game with "max() iterable argument is empty"."""
    from types import SimpleNamespace

    from core.agency.which_one_answers_to_her import _HowOutlinesGo

    outlines = _HowOutlinesGo()
    for step in range(6):
        outlines.saw(SimpleNamespace(number=4, x=10.0 + 3.0 * step, y=5.0, w=6.0 * step + 20.0, h=4.0))
    outlines.from_an_end[9] = 5                              # counts of a thing whose outlines are gone
    assert outlines.a_gauge(9) is False
    outlines.forget_all_but({})
    assert outlines.a_gauge(4) is False and not outlines.from_an_end


def test_a_bar_named_as_a_score_is_not_what_she_has_left(monkeypatch):
    """LIVE 2026-10-11 "The SCORE bar goes down: it's what I have left." """
    from types import SimpleNamespace

    from core.agency import reading_while_playing as reading
    from core.agency import what_she_has_left

    monkeypatch.setattr(what_she_has_left, "bars_read", lambda *a, **k: [{"what": "loss", "counter": "the SCORE bar",
                                                                          "at": 1.0}])
    monkeypatch.setattr(what_she_has_left, "what_is_left", lambda bars: None)
    said, verdicts = [], []
    run = SimpleNamespace(bars=object(), regions_read=[], gains=0, losses=0, presses=None, reach=None, vitals=None)
    meeting = SimpleNamespace(_verdict=verdicts.append, _met={}, vitals=None)
    hers = SimpleNamespace(kind=1, follows_pointer=False, thing=lambda moves: None)
    moves = SimpleNamespace(shape=(300, 400))
    reading._read_the_bars(run, meeting, hers, moves, None, 1.0, said.append)
    assert run.losses == 0 and not verdicts and not said


def test_a_bar_named_by_a_control_is_not_what_she_has_left(monkeypatch):
    """LIVE 2026-10-11 "The SPACE TO STOMP bar goes down: it's what I have left." """
    from types import SimpleNamespace

    from core.agency import reading_while_playing as reading
    from core.agency import what_she_has_left

    monkeypatch.setattr(what_she_has_left, "bars_read", lambda *a, **k: [{"what": "loss", "counter": "the SPACE TO STOMP bar",
                                                                          "at": 1.0}])
    monkeypatch.setattr(what_she_has_left, "what_is_left", lambda bars: None)
    said, verdicts = [], []
    run = SimpleNamespace(bars=object(), regions_read=[], gains=0, losses=0, presses=None, reach=None, vitals=None)
    meeting = SimpleNamespace(_verdict=verdicts.append, _met={}, vitals=None)
    hers = SimpleNamespace(kind=1, follows_pointer=False, thing=lambda moves: None)
    reading._read_the_bars(run, meeting, hers, SimpleNamespace(shape=(300, 400)), None, 1.0, said.append)
    assert run.losses == 0 and not verdicts and not said


def test_where_no_body_of_hers_is_in_the_picture_nothing_answering_does_not_end_play(monkeypatch):
    """Offline 2026-10-12 a food fight she was holding her own in, aiming from outside the picture, was handed back
    mid-round, every round, at the moment a title screen would have been."""
    from types import SimpleNamespace

    import core.agency.playing_as_it_happens as play
    from core.agency.which_one_answers_to_her import WhichIsHers

    run = play._Run(keys=["space"], began=0.0)
    meeting = SimpleNamespace(verdicts=[], evidence={})
    assert play._nothing_answers(run, WhichIsHers(), meeting, 40.0) == "nothing here answers to me while it moves"
    monkeypatch.setattr(play, "_how_it_stands", lambda _run, _at: (True, True))
    assert not play._nothing_answers(run, WhichIsHers(), meeting, 40.0)
