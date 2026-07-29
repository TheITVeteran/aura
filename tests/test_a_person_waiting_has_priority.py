"""A person at the keyboard outranks background work.

Live 2026-07-28, mid-demo rehearsal:

    [MLX] Waiting for foreground owner Cortex to release (held 58.7s)

and the follow-up message typed during that window came back as

    I still have the previous turn open. I am not going to fake a new answer
    over it; the next clean reply should land from the active turn.

Two separate faults produced that. The chat route waited 2.0 seconds for the
lane before refusing, against holds that legitimately run for a minute — so
the person did nothing wrong and lost their message. And the MLX foreground
guard compared ages only, with no notion of WHO was holding, so a typed turn
queued behind autonomous loops that had nobody waiting on them.

Background work is interruptible by design. Someone sitting at the keyboard
is not. The second half matters as much as the first: a user turn must never
steal the lane from ANOTHER user turn, or two people (or a turn and its own
follow-up) would cut each other off mid-answer.
"""

import asyncio

import pytest

from core.brain.llm import mlx_client
from interface.routes import chat as chat_routes

pytestmark = pytest.mark.unit


def test_a_user_turn_waits_long_enough_to_be_answered():
    """Two seconds against a minute-long hold is a refusal, not a wait."""
    assert chat_routes._FOREGROUND_CHAT_BUSY_WAIT_S >= 60.0


def test_the_wait_never_outlasts_the_preemption_threshold():
    """Past that point the holder is treated as stuck and cleared anyway, so
    waiting longer would only delay a recovery that is already coming."""
    assert (
        chat_routes._FOREGROUND_CHAT_BUSY_WAIT_S
        <= chat_routes._FOREGROUND_CHAT_LOCK_PREEMPT_AFTER_S
    )


def test_a_person_takes_the_lane_from_background_work():
    async def scenario() -> str:
        async with mlx_client._foreground_owner_context(
            "background_loop", foreground_request=False, stale_after=600.0
        ):
            try:
                async with asyncio.timeout(6):
                    async with mlx_client._foreground_owner_context(
                        "user_turn", foreground_request=True, stale_after=60.0
                    ):
                        return mlx_client._FOREGROUND_OWNER_NAME or ""
            except (TimeoutError, asyncio.TimeoutError):
                return "blocked"

    assert asyncio.run(scenario()) == "user_turn"


def test_a_person_does_not_cut_off_another_person():
    async def scenario() -> str:
        async with mlx_client._foreground_owner_context(
            "user_a", foreground_request=True, stale_after=600.0
        ):
            try:
                async with asyncio.timeout(4):
                    async with mlx_client._foreground_owner_context(
                        "user_b", foreground_request=True, stale_after=60.0
                    ):
                        return "stolen"
            except (TimeoutError, asyncio.TimeoutError):
                return "waited"

    assert asyncio.run(scenario()) == "waited"
