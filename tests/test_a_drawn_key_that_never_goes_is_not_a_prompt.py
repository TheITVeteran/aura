"""A key drawn on the screen that is still there after being pressed again and again is asking for nothing.

LIVE 2026-10-11 something drawn like a down key showed over a lander a moment into play. It was taken for a prompt,
and she pressed down at it, fast, until the lander was on the ground, in each of six goes.
"""
from __future__ import annotations

import asyncio

import pytest

from core.agency.pressing_what_is_shown import PRESSED_TO_NO_END, KeysShown

pytestmark = pytest.mark.unit


class _Hands:
    def __init__(self) -> None:
        self.pressed: list[str] = []

    async def down(self, key: str) -> None:
        self.pressed.append(key)

    async def up(self, key: str) -> None:
        return None


def _drawn(*keys: str) -> list[dict]:
    return [{"key": key, "center_x": 0.5} for key in keys]


def test_pressed_to_no_end_it_is_let_alone_and_stays_let_alone() -> None:
    shown, hands = KeysShown(), _Hands()

    async def go() -> float:
        at = 0.0
        shown._saw(_drawn(), at)                      # play began with no key drawn
        for _ in range(PRESSED_TO_NO_END + 2):
            at += 0.15
            shown._saw(_drawn("down"), at)            # then one is, and stays, whatever she presses
            if shown.asks(at):
                await shown.press(hands)
        return at

    at = asyncio.run(go())
    assert len(hands.pressed) == PRESSED_TO_NO_END
    assert not shown.asks(at + 0.15) and "down" in shown.asks_nothing
    # Kept for the next stretch of the same game, with what was never absent.
    again = KeysShown(shown.done())
    again._saw(_drawn("down"), at + 1.0)
    again._saw(_drawn("down"), at + 1.15)
    assert not again.asks(at + 1.15)


def test_a_prompt_that_goes_when_answered_is_still_pressed() -> None:
    shown, hands = KeysShown(), _Hands()

    async def go() -> None:
        at = 0.0
        shown._saw(_drawn(), at)
        for _round in range(PRESSED_TO_NO_END):     # shown, pressed twice, gone; and shown again
            for _ in range(2):
                at += 0.15
                shown._saw(_drawn("left"), at)
                if shown.asks(at):
                    await shown.press(hands)
            at += 0.6
            shown._saw(_drawn(), at)

    asyncio.run(go())
    assert len(hands.pressed) == 2 * PRESSED_TO_NO_END and not shown.asks_nothing
