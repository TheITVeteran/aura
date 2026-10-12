"""A picture whose words were read counts even when it has no words.

Offline 2026-10-12 play began on a screen without text. Its first words said
the round had ended, but were treated as the screen play had begun on.
Keep the first completed reading apart from the words it found.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from core.verify import invariant

if TYPE_CHECKING:
    from core.agency.playing_as_it_happens import _Run


def kept_word_picture(run: _Run, when: float) -> None:
    """Keep when a completed reading's picture was taken, including a reading that found no words."""
    if not math.isfinite(when) or when < run.began:
        return
    first = getattr(run, "word_picture_read_at", None)
    run.word_picture_read_at = when if first is None else min(first, when)


def first_word_picture_at(run: _Run) -> float | None:
    """The first measured reading, or the first retained words for older callers; None means no reading."""
    readings = [when for when, _said in run.words_read if math.isfinite(when) and when >= run.began]
    first = getattr(run, "word_picture_read_at", None)
    if first is not None and math.isfinite(first) and first >= run.began:
        readings.append(first)
    return min(readings, default=None)


@invariant("agency.an_empty_reading_still_measures_the_initial_screen", scope="agency",
           owner="core/agency/words_read_with_their_picture.py", observational=False)
def _word_picture_start_invariant() -> tuple:
    from types import SimpleNamespace

    run = SimpleNamespace(began=10.0, words_read=[], word_picture_read_at=None)
    assert first_word_picture_at(run) is None, "an unread picture became a measured start"
    kept_word_picture(run, 10.2)
    run.words_read.append((14.0, "try again"))
    assert first_word_picture_at(run) == 10.2, "the first words erased an earlier empty reading"
    run = SimpleNamespace(began=10.0, words_read=[(12.5, "try again")], word_picture_read_at=None)
    assert first_word_picture_at(run) == 12.5, "a delayed first reading was invented at the start"
    return ()
