"""A key tapped is held down a moment, as a hand taps one, and let go of while she goes on.

A game that asks at each of its frames which keys are down (Flash's Key.isDown, a canvas game's set of held keys) sees
only a key that is down when its frame comes, and 12 frames a second is Flash's own default. Measured 14 October 2026
in Chromium on a world that reads its held keys each frame (tests/fixtures/worlds_that_move/foodfight.html): let go at
once, a tap reached it 0 times in 10; held 20 ms, 10 times in 10. A person's tap is held about a tenth of a second.
Held in her loop, a tap would cost her a picture or two, so the letting go comes on its own.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger("Aura.KeysTapped")

__all__ = ["TAP_HELD_S", "KeysTapped"]

#: How long a tapped key stays down, in seconds: past one frame at 12 a second.
TAP_HELD_S = 0.09


class KeysTapped:
    """The taps whose letting go is still to come, one a key."""

    def __init__(self, down: Callable[[str], Awaitable[None]], up: Callable[[str], Awaitable[None]],
                 held_s: float = TAP_HELD_S) -> None:
        self.down, self.up, self.held_s = down, up, held_s
        self.letting_go: dict[str, asyncio.Task] = {}

    async def tap(self, key: str) -> None:
        if key in self.letting_go:
            await self.let_go(key)                       # pressed afresh, not a held key's repeat
        await self.down(key)
        self.letting_go[key] = asyncio.get_running_loop().create_task(self._let_go_after(key))

    def held(self, key: str) -> None:
        """A key pressed or let go of as a hold: no tap's letting go comes after it."""
        letting = self.letting_go.pop(key, None)
        if letting is not None:
            letting.cancel()

    async def let_go(self, key: str) -> None:
        self.held(key)
        await self.up(key)

    async def let_go_of_all(self) -> None:
        for key in list(self.letting_go):
            await self.let_go(key)

    async def _let_go_after(self, key: str) -> None:
        await asyncio.sleep(self.held_s)
        self.letting_go.pop(key, None)
        try:
            await self.up(key)
        except Exception as why:  # noqa: BLE001 - a page closed under a tap's letting go
            logger.info("letting go of %s after a tap: %s", key, why)
