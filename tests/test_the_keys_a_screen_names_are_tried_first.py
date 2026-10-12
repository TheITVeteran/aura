"""The keys the screen in front of her names for moving are tried first, and keys named for an act last.

Offline 2026-10-11 a bowling game's "USE WASD TO MOVE" was tried after space and four arrows it did not use, and its
two-heart player was out before her keys were known.
"""
from __future__ import annotations

import pytest

from core.agency.the_controls_a_game_names import tried_in_turn


@pytest.mark.unit
def test_keys_named_for_moving_go_first_and_keys_for_an_act_last():
    keys = ["space", "w", "a", "s", "d", "up", "down", "left", "right"]
    assert tried_in_turn(keys, "PICK A PLAYER USE WASD TO MOVE SPACE TO STOMP") == [
        "w", "a", "s", "d", "up", "down", "left", "right", "space"]
    assert tried_in_turn(["x", "z", "left", "right", "up"], "Press the Z key to melee attack close enemies.") == [
        "left", "right", "up", "x", "z"]


@pytest.mark.unit
def test_with_nothing_named_the_order_is_kept_but_for_keys_that_act():
    assert tried_in_turn(["up", "space", "left"], "") == ["up", "left", "space"]
