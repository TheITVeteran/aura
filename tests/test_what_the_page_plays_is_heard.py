"""What a page plays is heard as events, the same sound known again by its signature, without changing what the page
does (core/perception/what_the_page_plays.py)."""
from __future__ import annotations

import pytest

from core.perception.what_the_page_plays import sounds_in

pytestmark = pytest.mark.unit


def test_what_is_handed_over_is_put_on_her_clock_and_anything_else_is_nothing_heard():
    taken = {"now": 5000.0, "heard": [{"at": 4500.0, "kind": "tone", "sig": "sine 440"},
                                     {"at": "x", "kind": "tone", "sig": "sine 440"},
                                     {"at": 4900.0, "kind": "drum", "sig": "?"}]}
    [sound] = sounds_in(taken, 100.0)
    assert sound.at == pytest.approx(99.5) and sound.sig == "sine 440"
    assert sounds_in(None, 1.0) == [] and sounds_in({"now": "x", "heard": []}, 1.0) == []


@pytest.mark.asyncio
async def test_a_page_heard_with_each_frame_as_its_sounds_start():
    from playwright.async_api import async_playwright

    from core.perception.frames_as_they_are_drawn import CanvasFrames

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 320, "height": 200})
        await page.set_content('<style>body{margin:0}</style><canvas width="320" height="200"></canvas>')
        await page.evaluate("() => { const c=document.querySelector('canvas').getContext('2d');"
                            "c.fillStyle='#123';c.fillRect(0,0,320,200);c.fillStyle='#fff';c.fillRect(20,20,40,40); }")
        frames = CanvasFrames(page)
        clip = {"x": 0, "y": 0, "width": 320, "height": 200}
        try:
            await frames.look(clip)
            played = await page.evaluate("""() => {
              const audio = new AudioContext();
              const tone = (pitch, wave) => { const o = audio.createOscillator(); o.type = wave;
                o.frequency.value = pitch; o.connect(audio.destination); o.start(); o.stop(audio.currentTime + 0.05);
                return o; };
              tone(440, 'square'); tone(1320, 'sine'); tone(440, 'square');
              const b = audio.createBuffer(1, 800, 8000), d = b.getChannelData(0);
              for (let i = 0; i < d.length; i++) d[i] = Math.sin(i / 3);
              const s = audio.createBufferSource(); s.buffer = b; s.connect(audio.destination); s.start();
              return typeof s.start === 'function';
            }""")
            picture, _at = await frames.look(clip)
            heard = [(sound.kind, sound.sig) for sound in picture.sounds]
            assert played
            assert heard[:3] == [("tone", "square 440"), ("tone", "sine 1320"), ("tone", "square 440")], heard
            assert heard[3][0] == "buffer" and heard[3][1].startswith("buffer 800 8000 "), heard
            # Taken once: the next frame hears nothing new, and is the canvas's picture as before.
            later, _at = await frames.look(clip)
            assert not getattr(later, "sounds", ())
            assert later.shape[:2] == (200, 320)
        finally:
            await frames.close()
            await browser.close()
