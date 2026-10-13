"""A picture already looked at is not looked at again, and nothing about what is found changes but the time."""
from __future__ import annotations

import numpy as np

from core.perception.a_picture_read_once import KEPT, how_often_again, picture_key, read_once


def _picture(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(40, 60, 3), dtype=np.uint8)


def test_the_same_pixels_are_looked_at_once():
    looked = []

    def look():
        looked.append(1)
        return [{"text": "Play", "center_x": 0.5}]

    picture = _picture(1)
    first = read_once("a test of looking once", picture, look)
    again = read_once("a test of looking once", picture.copy(), look)
    assert first == again and len(looked) == 1
    assert how_often_again()["a test of looking once"] == (1, 1)


def test_one_pixel_different_is_another_picture():
    looked = []
    picture = _picture(2)
    read_once("a test of one pixel", picture, lambda: looked.append(1) or "first")
    changed = picture.copy()
    changed[10, 10, 0] ^= 1
    read_once("a test of one pixel", changed, lambda: looked.append(1) or "second")
    assert len(looked) == 2
    assert picture_key(picture) != picture_key(changed)
    assert picture_key(picture) != picture_key(picture.astype(np.int16))      # the same numbers held otherwise


def test_what_is_given_out_is_a_copy():
    picture = _picture(3)
    given = read_once("a test of copies", picture, lambda: [{"text": "Next"}])
    given[0]["answers_the_pointer"] = True
    assert read_once("a test of copies", picture, lambda: []) == [{"text": "Next"}]


def test_what_else_a_finding_depends_on_is_part_of_it():
    picture = _picture(4)
    assert read_once("a test of also", picture, lambda: "with a", "a") == "with a"
    assert read_once("a test of also", picture, lambda: "with b", "b") == "with b"


def test_only_so_many_pictures_are_kept():
    for seed in range(KEPT + 5):
        read_once("a test of keeping", _picture(100 + seed), lambda seed=seed: seed)
    looked = []
    read_once("a test of keeping", _picture(100), lambda: looked.append(1) or 0)      # the oldest was let go
    assert looked == [1]


def test_a_picture_read_twice_gives_the_same_words_and_shapes_as_reading_it_anew():
    """The ways of looking kept by picture give the same findings from memory as from the pixels."""
    from core.perception.drawn_objects import drawn_objects
    from core.perception.shapes_that_look_pressable import pressable_shapes

    picture = np.full((120, 200, 3), 30, np.uint8)
    picture[40:70, 60:140] = (220, 40, 40)                                    # a red button
    picture[90:110, 20:50] = (40, 200, 60)
    assert pressable_shapes(picture) == pressable_shapes(picture.copy())
    assert drawn_objects(picture) == drawn_objects(picture.copy())
