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


def test_a_button_comes_before_the_title_it_stands_in():
    """Offline 2026-10-11 a title read as "Coco's Scrambler PLAY" was offered before the PLAY button inside it."""
    from core.perception.shapes_that_look_pressable import _named_by_their_writing

    pixels = np.full((400, 600, 3), 250, np.uint8)
    pixels[230:395, 100:590] = (240, 120, 40)                              # the title
    pixels[340:380, 470:580] = (230, 40, 40)                               # its PLAY button

    def read(patch):
        patch = np.asarray(patch)
        return [{"text": "Coco's Scramble PLAY"}] if (patch[..., 1] > 100).mean() > 0.5 else [{"text": "PLAY"}]

    title = {"text": "the shape at 55% across, 80% down", "x": 0.167, "y": 0.575, "width": 0.817, "height": 0.41}
    button = {"text": "the shape at 85% across, 90% down", "x": 0.783, "y": 0.85, "width": 0.183, "height": 0.1}
    named = _named_by_their_writing(pixels, [(0.9, title), (0.6, button)], read)
    assert [region["text"] for _share, region in named] == ["PLAY", "Coco's Scramble PLAY"]


def test_a_story_pages_arrow_in_its_corner_is_found_among_a_busy_drawing_and_leads_on():
    """LIVE 2026-10-11 a comic before play had its magenta arrow in the bottom right corner, and she clicked a shape in
    the middle of the drawing: a button of one strong colour is found where neither its brightness nor its outline
    stands out, and the corner a page's way on is put in leads on."""
    from core.language.a_way_on import how_much_it_leads_on

    rng = np.random.default_rng(7)
    picture = Image.fromarray(rng.integers(90, 200, (256, 384, 3), dtype=np.uint8))   # a busy drawing
    draw = ImageDraw.Draw(picture)
    draw.rectangle((350, 222, 372, 244), fill=(230, 40, 200), outline=(255, 255, 255), width=2)   # the arrow's button
    shapes = pressable_shapes(np.asarray(picture))
    corner = [s for s in shapes if s["center_x"] > 0.85 and s["center_y"] > 0.8]
    assert corner and shapes[0] is corner[0], [s["text"] for s in shapes]
    assert how_much_it_leads_on(corner[0]["text"]) > how_much_it_leads_on("the shape at 50% across, 50% down")
