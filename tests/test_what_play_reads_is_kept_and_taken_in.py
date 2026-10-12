"""What play reads is kept however long it goes, and what it tells her to do is taken in as it is read.

Offline 2026-10-11 a stretch stopped keeping what it read at its four hundredth reading, about a minute in, and a GAME
OVER after that was never read; and a bowling game's second lesson ("NOW USE YOUR MOUSE TO AIM AND SHOOT THE EVIL
PIN!") came up mid-stretch, was never taken in, and she never aimed.
"""
from __future__ import annotations

import pytest


@pytest.mark.unit
def test_an_end_read_after_hundreds_of_readings_is_read():
    from core.agency.playing_as_it_happens import SAID_IT_IS_OVER, _Run, _said_it_is_over
    from core.agency.reading_while_playing import READINGS_KEPT, kept_reading

    run = _Run(keys=["up"], began=0.0)
    kept_reading(run, 0.5, "try again")                       # the last round's end, still up as it began
    for n in range(2 * READINGS_KEPT):
        kept_reading(run, 3.0 + n * 0.25, "score: 120")
    assert not _said_it_is_over(run)
    kept_reading(run, 400.0, "game over")
    assert _said_it_is_over(run) == SAID_IT_IS_OVER
    assert len(run.words_read) == READINGS_KEPT and run.words_read[0] == (0.5, "try again")


@pytest.mark.unit
def test_a_lesson_read_in_play_is_taken_in_as_it_comes():
    from core.agency.playing_as_it_happens import _Run
    from core.agency.reading_while_playing import kept_reading, rules_as_told_since
    from core.agency.what_the_rules_said import WhatTheRulesSaid

    run = _Run(keys=["w", "a", "s", "d"], began=0.0)
    rules = WhatTheRulesSaid.read("use wasd to move")
    kept_reading(run, 1.0, "score: 0")
    assert rules_as_told_since(run, rules) is rules           # nothing told: the rules as they were
    kept_reading(run, 9.0, "good job! now use your mouse to")
    kept_reading(run, 9.0, "and shoot the evil pin!")
    rules = rules_as_told_since(run, rules)
    assert rules.a_click_is_a_shot and run.pointer_trigger
    assert "use wasd to move" in rules.words
    again = rules_as_told_since(run, rules)
    assert again is rules                                     # read once, not again each look
