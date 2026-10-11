"""Where Apple's Vision is not installed (Linux, a container), the tesseract program reads the words on a screen.

Without it every counter and every line of a game's rules read as nothing, and her play in a test world on Linux was
play without a score: what meeting a thing does is learned from the score going up or a life going.
"""
from __future__ import annotations

import shutil

import pytest

pytestmark = pytest.mark.unit


def _a_screen(lines: list[tuple[str, tuple[int, int]]]):
    """A game-sized screen with words drawn in a proportional typeface, as a browser draws a canvas's text."""
    np = pytest.importorskip("numpy")
    image = pytest.importorskip("PIL.Image")
    draw = pytest.importorskip("PIL.ImageDraw")
    font = pytest.importorskip("PIL.ImageFont").load_default(size=16)
    picture = image.new("RGB", (480, 320), (20, 40, 60))
    pen = draw.Draw(picture)
    for said, (x, y) in lines:
        pen.text((x, y), said, fill=(255, 255, 255), font=font)
    return np.asarray(picture)[:, :, ::-1].copy()


def test_counters_and_a_sentence_are_read_as_runs_where_they_sit():
    if shutil.which("tesseract") is None:
        pytest.skip("no tesseract program here")
    from core.perception.what_the_pixels_show import _recognized_by_tesseract

    read = _recognized_by_tesseract(_a_screen([("SCORE 12", (10, 8)), ("LIVES 3", (400, 8)),
                                                ("Catch the fruit, not the bombs.", (60, 152))]))
    said = {region["text"]: region for region in read}
    assert "SCORE 12" in said and "LIVES 3" in said, read
    assert "Catch the fruit, not the bombs." in said, read
    # Where each sits, as fractions from the top left, as Vision gives them.
    assert said["SCORE 12"]["center_y"] < 0.15 and said["SCORE 12"]["center_x"] < 0.3
    assert said["LIVES 3"]["center_x"] > 0.7
    assert 0.4 < said["Catch the fruit, not the bombs."]["center_y"] < 0.6


def test_no_reader_installed_reads_nothing_rather_than_failing(monkeypatch):
    from core.perception import what_the_pixels_show

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert what_the_pixels_show._recognized_by_tesseract(_a_screen([("SCORE 1", (10, 8))])) == []
