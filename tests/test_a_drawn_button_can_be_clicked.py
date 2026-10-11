"""A button drawn with no words on it is something she can click.

A game's rules page with an arrow in its corner offered her nothing to click,
because what she could click was only writing.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from core.perception.shapes_that_look_pressable import pressable_shapes
from core.skills.screen_pursuit_bearings import things_to_click, where_to_click


def _rules_page() -> np.ndarray:
    picture = Image.new("RGB", (640, 480), (30, 60, 120))
    draw = ImageDraw.Draw(picture)
    draw.rectangle((40, 40, 600, 52), fill=(255, 255, 255))  # a line of the rules
    draw.polygon([(560, 400), (560, 450), (605, 425)], fill=(250, 200, 0))  # the arrow on
    return np.asarray(picture)


def test_the_arrow_is_found_and_the_writing_is_not_taken_for_a_button():
    rules = {"text": "Throw the ball where the mouse points", "x": 0.06, "y": 0.08, "width": 0.88, "height": 0.03}
    shapes = pressable_shapes(_rules_page(), apart_from=[rules])
    assert [s["text"] for s in shapes] == ["the shape at 90% across, 90% down"]
    assert abs(shapes[0]["center_x"] - 0.9) < 0.03 and abs(shapes[0]["center_y"] - 0.885) < 0.03


def test_a_blank_or_busy_picture_has_no_buttons():
    assert pressable_shapes(np.zeros((480, 640, 3), dtype=np.uint8)) == []
    noise = np.random.default_rng(1).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    assert len(pressable_shapes(noise)) <= 6


def test_a_drawn_button_is_offered_inside_a_game_and_clicked_where_it_is():
    seen = {"layout": [], "shapes": pressable_shapes(_rules_page())}
    offered = things_to_click(seen, drawn_where=(0, 0, 1, 1))
    assert offered == ('click "the shape at 90% across, 90% down"',)
    at = where_to_click(seen, "the shape at 90% across, 90% down")
    assert at is not None and at[0] > 0.85 and at[1] > 0.85
    assert things_to_click(seen, drawn_where=None) == ()


def test_the_words_on_a_drawn_button_no_reading_of_the_screen_made_out_name_it_and_put_it_first():
    """LIVE-like 2026-10-11 a title's PLAY, small red letters in a red frame, was not read from the whole screen, and
    the shape it stood in was offered after five others, none of them it. Read from the shape itself it is PLAY."""
    picture = Image.new("RGB", (600, 400), (250, 240, 240))
    draw = ImageDraw.Draw(picture)
    draw.ellipse((60, 60, 260, 300), fill=(240, 120, 40))                 # a figure on the title, larger than the button
    draw.rectangle((470, 340, 580, 380), outline=(220, 30, 30), width=4)  # the button's frame
    draw.rectangle((490, 352, 560, 368), fill=(220, 30, 30))              # its lettering, as a reader would see it
    pixels = np.asarray(picture)

    def read(patch):
        patch = np.asarray(patch)
        red = ((patch[..., 0] > 180) & (patch[..., 1] < 80)).mean()
        return [{"text": "e"}, {"text": "PLAY"}] if 0.15 < red < 0.9 else []

    shapes = pressable_shapes(pixels, read=read)
    assert shapes[0]["text"] == "PLAY" and shapes[0]["written_on_it"], [s["text"] for s in shapes]
    assert abs(shapes[0]["center_x"] - 0.875) < 0.05 and abs(shapes[0]["center_y"] - 0.9) < 0.05
    assert where_to_click({"shapes": shapes}, "PLAY") is not None
