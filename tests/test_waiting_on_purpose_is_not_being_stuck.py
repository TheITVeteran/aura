"""Waiting on purpose is not being stuck, and a game she was playing is played on when she cannot think what next.

LIVE 2026-10-10 she read a game's instructions as they played, pressing nothing, and her run was ended as having no
move available three looks in; then, her model busy, the answer to "what next" was refused and she closed the game.
"""
from __future__ import annotations

import pytest

from core.skills.fluid_executor import WAITING, FluidExecutor

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_a_decision_that_waits_does_not_end_the_run_as_stuck():
    looks = {"n": 0}

    async def observe():
        looks["n"] += 1
        return looks["n"]

    async def decide(seen):
        return WAITING if seen <= 8 else None          # eight looks of a lesson playing, then nothing to do

    receipt = await FluidExecutor(stall_window=3).pursue("play", observe=observe, decide=decide,
                                                         is_satisfied=lambda seen: False, max_cycles=50, max_seconds=5)
    assert looks["n"] == 11 and receipt.outcome == "no_move_available"     # stuck only after the waiting was done


def test_a_game_she_has_been_playing_and_not_finished_is_played_on_without_asking_what_next():
    from core.skills import sovereign_browser_drawing as drawing

    url = "https://example.org/details/a-game"
    goal = "Go to it and play the game until you win."
    assert drawing.played_first({"url": url}, goal) is None             # never played here: nothing to go on with
    drawing._PLAYED_HERE.add(drawing._address(url))
    try:
        for _ in range(drawing.PLAY_ON_AT_MOST):
            on = drawing.played_first({"url": url}, goal)
            assert on["resolved_actions"][0]["selector"] == drawing.DRAWING and not on["done"]
        assert drawing.played_first({"url": url}, goal) is None         # gone back to as often as it is
        drawing._PLAYED_ON.clear()
        drawing._PLAY_OVER[drawing._address(url)] = "won"
        assert drawing.played_first({"url": url}, goal)["done"] is True  # finished: said so, not played on
    finally:
        drawing._PLAYED_HERE.discard(drawing._address(url))
        drawing._PLAYED_ON.pop(drawing._address(url), None)
        drawing._PLAY_OVER.pop(drawing._address(url), None)


def test_left_alone_on_one_moving_screen_she_is_not_left_alone_on_the_next():
    """LIVE 2026-10-10 PLAY led from an animated menu into a boat game, which played itself to GAME OVER in the quiet
    she had been left in on the menu."""
    from core.skills.screen_pursuit_as_it_happens import _a_different_screen

    menu = "BONUS ITEMS Grab the rainbow monkeys to get bonuses PLAY"
    assert not _a_different_screen("BONuS ITEMS Grab the rainbow monkeys to get bonuses PLAY", menu)   # read again
    assert _a_different_screen("SCORE 0 LIVES 3", menu)                                              # the game
    assert _a_different_screen("", menu)                                                             # wordless play


@pytest.mark.parametrize(("label", "read"), [("GAME", True), ("OVER", True), ("YOUR SCORE", True), ("You Win!", True),
                                             ("CONTINUE", False), ("NEW GAME", False), ("PLAY AGAIN", False)])
def test_what_an_end_announces_is_read_and_a_way_on_from_it_is_pressed(label, read):
    from core.language.a_way_on import how_much_it_leads_on

    assert (how_much_it_leads_on(label) < 0.7) is read


def test_a_screen_found_no_well_is_not_watched_again_while_it_stays_the_same_screen(monkeypatch):
    """LIVE-like 2026-10-11 a scrapbook's page was watched for eight seconds, for a well, before every click on it."""
    import asyncio
    from types import SimpleNamespace

    import core.agency.playing_a_well as wells
    import core.skills.screen_pursuit_as_it_happens as pursuit

    watched = []

    async def no_well(look, known=None, seen=None):
        watched.append(1)
        return None

    monkeypatch.setattr(wells, "a_well_with_a_piece_coming_down", no_well)
    reflexes = SimpleNamespace(no_well_here=False, keep={}, look=None, said_now="SCOOBY-DOO PUBLIC LIBRARY NEW PHOTO",
                               waited_at_a_well=0.0)
    assert not asyncio.run(pursuit._played_a_well(reflexes, 100.0))
    reflexes.said_now = "SCOOBY-D0O PUBLIC LIBRARY NEW PH0TO"                    # read a little differently
    assert not asyncio.run(pursuit._played_a_well(reflexes, 200.0))
    assert len(watched) == 1
    reflexes.said_now = "LEVEL 1 SCORE 0"                                         # another screen
    asyncio.run(pursuit._played_a_well(reflexes, 300.0))
    assert len(watched) == 2
