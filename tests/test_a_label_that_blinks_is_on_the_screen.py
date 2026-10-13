"""A label that blinks to be noticed is on the screen between its flashes.

Offline 2026-10-13 a game's CONTINUE blinked: it was read in every other picture, the picture each look ended on was
mostly a dark one, and she pressed keys at the page for four minutes. Within one look, writing an earlier reading of
the same screen had is kept; writing of another screen is not, and writing read another way in its place is not added.
"""
from __future__ import annotations

from core.perception.what_the_pixels_show import _with_what_blinked


def _label(text: str, x: float, y: float) -> dict:
    return {"text": text, "center_x": x, "center_y": y, "x": x - 0.05, "y": y - 0.02, "width": 0.1, "height": 0.04}


_PAGE = [_label("Instructions", 0.5, 0.1), _label("Find the lost robot", 0.5, 0.3), _label("Use your mouse", 0.5, 0.4),
         _label("Skip instructions", 0.2, 0.9)]


def test_a_label_caught_dark_in_the_last_picture_is_still_read():
    lit = [*_PAGE, _label("CONTINUE", 0.8, 0.9)]
    last = {"text": " ".join(r["text"] for r in _PAGE), "layout": list(_PAGE)}
    _with_what_blinked(last, [lit, list(_PAGE)])
    assert [r["text"] for r in last["layout"]][-1] == "CONTINUE" and last["layout"][-1]["blinked"]
    assert last["text"].endswith("CONTINUE")


def test_the_writing_of_another_screen_is_not_carried_onto_this_one():
    title = [_label("TEAM WORK", 0.5, 0.2), _label("PLAY", 0.5, 0.6), _label("HELP", 0.5, 0.8)]
    last = {"text": "", "layout": list(_PAGE)}
    _with_what_blinked(last, [title])
    assert [r["text"] for r in last["layout"]] == [r["text"] for r in _PAGE]


def test_writing_read_another_way_in_its_place_is_not_added_twice():
    earlier = [*_PAGE[:3], _label("SKIP INSTRUCTI0NS", 0.2, 0.9)]
    last = {"text": "", "layout": list(_PAGE)}
    _with_what_blinked(last, [earlier])
    assert len(last["layout"]) == len(_PAGE)


def test_a_screen_of_few_words_that_stood_on_the_screen_before_is_another_screen():
    """Offline 2026-10-13 a game's first picture after PLAY AGAIN shared its two words with the end screen, and was given
    the end screen's PLAY AGAIN: the next end, its PLAY AGAIN no longer new, was played on for three minutes."""
    end = [_label("GAME OVER", 0.5, 0.2), _label("YOUR SCORE", 0.5, 0.4), _label("LEVEL", 0.3, 0.05),
           _label("SCORE", 0.7, 0.05), _label("Madame Foster will not be pleased", 0.5, 0.6), _label("PLAY AGAIN", 0.5, 0.8)]
    game = {"text": "LEVEL SCORE", "layout": [_label("LEVEL", 0.3, 0.05), _label("SCORE", 0.7, 0.05)]}
    _with_what_blinked(game, [end])
    assert [r["text"] for r in game["layout"]] == ["LEVEL", "SCORE"]
