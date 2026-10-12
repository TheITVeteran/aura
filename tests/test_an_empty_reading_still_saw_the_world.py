"""Text-free play has a measured first screen; absent readings establish nothing."""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency.words_read_with_their_picture import (
    _word_picture_start_invariant,
    first_word_picture_at,
    kept_word_picture,
)

pytestmark = pytest.mark.unit


def _reading_world():
    from core.agency.playing_as_it_happens import _Run

    run = _Run(keys=[], began=0.0)
    meeting = SimpleNamespace(
        read=lambda *args: [], writing=set(),
        readouts=SimpleNamespace(current={}, where={}, out_of={}),
    )
    hers = SimpleNamespace(thing=lambda moves: None)
    moves = SimpleNamespace(shape=(24, 32), things={})
    return run, meeting, hers, moves


def test_completed_empty_reading_stops_play_when_new_terminal_words_appear():
    from core.agency.playing_as_it_happens import SAID_IT_IS_OVER, _said_it_is_over
    from core.agency.reading_while_playing import _read_the_words

    run, meeting, hers, moves = _reading_world()
    _read_the_words(run, meeting, hers, moves, [], 0.25)
    assert run.words_read == []
    assert first_word_picture_at(run) == 0.25
    terminal = [{"text": "try again", "x": .2, "y": .4, "width": .4, "height": .1}]
    _read_the_words(run, meeting, hers, moves, terminal, 4.0)
    assert _said_it_is_over(run) == SAID_IT_IS_OVER
    assert run.over_read_at == 4.0


def test_delayed_first_reading_of_an_initial_end_screen_keeps_its_initial_status():
    from core.agency.playing_as_it_happens import _said_it_is_over
    from core.agency.reading_while_playing import _read_the_words

    run, meeting, hers, moves = _reading_world()
    terminal = [{"text": "try again", "x": .2, "y": .4, "width": .4, "height": .1}]
    _read_the_words(run, meeting, hers, moves, terminal, 2.5)
    _read_the_words(run, meeting, hers, moves, terminal, 5.0)
    assert first_word_picture_at(run) == 2.5
    assert _said_it_is_over(run) == ""


@pytest.mark.asyncio
async def test_no_reader_does_not_turn_an_unread_picture_into_an_empty_reading(monkeypatch):
    from core.agency import reading_while_playing as reading

    run, meeting, hers, moves = _reading_world()
    monkeypatch.setattr(reading, "_read_the_bars", lambda *args: None)
    monkeypatch.setattr("core.perception.the_drawing_as_objects.words_in", lambda picture: None)
    await reading._keep_reading(run, meeting, hers, moves, np.zeros((24, 32, 3), dtype=np.uint8), 0.25, None)
    assert first_word_picture_at(run) is None
    assert run.words_read == []


@pytest.mark.asyncio
async def test_a_renderer_with_no_words_provides_a_completed_reading(monkeypatch):
    from core.agency import reading_while_playing as reading

    run, meeting, hers, moves = _reading_world()
    monkeypatch.setattr(reading, "_read_the_bars", lambda *args: None)
    monkeypatch.setattr("core.perception.the_drawing_as_objects.words_in", lambda picture: [])
    await reading._keep_reading(run, meeting, hers, moves, np.zeros((24, 32, 3), dtype=np.uint8), 0.25, None)
    assert first_word_picture_at(run) == 0.25


@pytest.mark.asyncio
async def test_a_pending_ocr_reading_is_unmeasured_until_its_empty_result_arrives(monkeypatch):
    from core.agency import reading_while_playing as reading

    run, meeting, hers, moves = _reading_world()
    monkeypatch.setattr(reading, "_read_the_bars", lambda *args: None)
    monkeypatch.setattr("core.perception.the_drawing_as_objects.words_in", lambda picture: None)
    picture = np.zeros((24, 32, 3), dtype=np.uint8)
    pending = asyncio.get_running_loop().create_future()
    run.reading, run.read_at = pending, 0.25
    await reading._keep_reading(run, meeting, hers, moves, picture, 0.3, lambda picture: [])
    assert first_word_picture_at(run) is None
    pending.set_result(([], 0.25))
    await reading._keep_reading(run, meeting, hers, moves, picture, 0.4, lambda picture: [])
    assert first_word_picture_at(run) == 0.25
    assert run.reading is None


def test_picture_times_survive_out_of_order_completion_and_word_history_compaction():
    from core.agency.reading_while_playing import READINGS_KEPT, kept_reading

    run, _meeting, _hers, _moves = _reading_world()
    kept_word_picture(run, 1.0)
    kept_word_picture(run, 0.25)
    for n in range(READINGS_KEPT * 2):
        kept_reading(run, 3.0 + n * .25, "score: 120")
    assert len(run.words_read) == READINGS_KEPT
    assert first_word_picture_at(run) == .25


@pytest.mark.parametrize("when", [float("nan"), float("inf"), float("-inf"), -1.0])
def test_invalid_or_prior_picture_times_do_not_establish_a_measured_start(when):
    run, _meeting, _hers, _moves = _reading_world()
    kept_word_picture(run, when)
    assert first_word_picture_at(run) is None


def test_the_registered_empty_reading_invariant():
    assert _word_picture_start_invariant() == ()
