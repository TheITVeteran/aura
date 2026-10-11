"""Sending is read in the words a player is told it in, and a send that never sent anything here is let go.

Offline 2026-10-10 what was looked up of a photo game said it was "released", and in a world that waited for her she
pulled and let go at forty places without sending anything, again each time the world was still.
"""
from __future__ import annotations

import pytest

from core.agency.playing_by_shots import sends_by_letting_go
from core.agency.the_way_it_is_played import SEND, HeldTo

pytestmark = pytest.mark.unit


def test_sending_is_read_as_told_not_as_told_of():
    assert sends_by_letting_go("Pull back and release to launch the ball.")
    assert sends_by_letting_go("Click and hold to aim, then release to shoot.")
    assert sends_by_letting_go("Pulling back further gives more power.")
    assert not sends_by_letting_go("The game was released in 2002 by Cartoon Network.")
    assert not sends_by_letting_go("The rocket launches when the timer hits zero and is pulled by gravity.")


def test_a_way_that_never_did_what_it_is_for_is_let_go_until_the_place_says_something_new():
    held = HeldTo()
    held.never_did_it(SEND, "a photo game's words")
    assert held.choose([SEND], "a photo game's words") == ""
    assert held.choose([SEND], "Now pull back the slingshot and release to fire.") == SEND


def test_a_send_named_is_not_a_send_told():
    # Offline 2026-10-10 "the new extreme sport of hamster launching" had her pull and let go twenty-nine times at a game
    # played by clicks.
    assert not sends_by_letting_go("Help the hamsters enjoy their time off with the new extreme sport of hamster launching!")
    assert not sends_by_letting_go("When the hamster lines up with the pillow, click again to launch the hamster.")
    assert sends_by_letting_go("Fill the power meter and release.")
