"""A tapped key is held down a moment, as a hand taps one, so that a game asking each frame which keys are down sees it.

Until 14 October 2026 her tap was a key down and up sent together. A world that reads its held keys at each frame
(tests/fixtures/worlds_that_move/foodfight.html, as Flash's Key.isDown is read) saw none of ten such taps in Chromium,
and every one held 20 ms or more.
"""
from __future__ import annotations

import asyncio

import pytest

from core.skills.keys_tapped_as_a_hand_taps import TAP_HELD_S, KeysTapped

pytestmark = pytest.mark.unit


def _hands():
    happened = []

    async def down(key):
        happened.append(("down", key, asyncio.get_running_loop().time()))

    async def up(key):
        happened.append(("up", key, asyncio.get_running_loop().time()))

    return KeysTapped(down, up), happened


@pytest.mark.asyncio
async def test_a_tap_is_let_go_of_a_moment_later_and_she_does_not_wait_for_it():
    taps, happened = _hands()
    began = asyncio.get_running_loop().time()
    await taps.tap("Space")
    assert [h[:2] for h in happened] == [("down", "Space")]
    assert asyncio.get_running_loop().time() - began < TAP_HELD_S / 2
    await asyncio.sleep(TAP_HELD_S * 2)
    assert [h[:2] for h in happened] == [("down", "Space"), ("up", "Space")]
    assert happened[1][2] - happened[0][2] >= TAP_HELD_S - 0.01
    assert 1.0 / TAP_HELD_S < 12.0                      # past one frame at Flash's own rate of 12 a second


@pytest.mark.asyncio
async def test_a_key_tapped_again_while_still_down_is_pressed_afresh():
    taps, happened = _hands()
    await taps.tap("x")
    await taps.tap("x")
    await asyncio.sleep(TAP_HELD_S * 2)
    assert [h[0] for h in happened] == ["down", "up", "down", "up"]


@pytest.mark.asyncio
async def test_a_key_held_after_a_tap_is_not_let_go_by_the_tap():
    taps, happened = _hands()
    await taps.tap("ArrowLeft")
    taps.held("ArrowLeft")                               # she holds it now, to go left
    await asyncio.sleep(TAP_HELD_S * 2)
    assert [h[0] for h in happened] == ["down"]


@pytest.mark.asyncio
async def test_every_key_tapped_is_let_go_of_when_play_ends():
    taps, happened = _hands()
    await taps.tap("a")
    await taps.tap("b")
    await taps.let_go_of_all()
    assert sorted(h[:2] for h in happened if h[0] == "up") == [("up", "a"), ("up", "b")] and not taps.letting_go


@pytest.mark.asyncio
async def test_her_hands_tap_a_key_a_game_reading_its_held_keys_each_frame_sees():
    playwright_api = pytest.importorskip("playwright.async_api")
    from core.capabilities.phantom_browser import _chromium_graphics_arguments
    from core.skills.screen_pursuit_as_it_happens import PlayingAsItHappens
    from tools.measure_playing_as_it_happens import WORLDS

    async with playwright_api.async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(headless=True, args=_chromium_graphics_arguments())
        except Exception as why:  # noqa: BLE001 - no engine installed here
            pytest.skip(f"no browser engine: {why}")
        try:
            page = await browser.new_page(viewport={"width": 800, "height": 600})
            await page.goto(f"file://{WORLDS / 'foodfight.html'}?start=1&seed=1")
            await page.wait_for_timeout(300)
            await page.evaluate("""() => { window.__ducks = 0; const g = __world.game, step = g.step.bind(g);
                let was = false; g.step = (dt, w) => { step(dt, w); const now = g.ducked > 0;
                if (now && !was) window.__ducks++; was = now; }; }""")
            hands = PlayingAsItHappens(page, (0.0, 0.0, 1.0, 1.0), "play", 0.0)
            hands._focused = True                        # noqa: SLF001 - the page has nothing to focus but itself
            for _ in range(6):
                await hands.tap("space")
                await page.wait_for_timeout(900)         # past the world's wait between ducks
            assert await page.evaluate("window.__ducks") == 6
            await hands.close()
        finally:
            await browser.close()
