"""She learns what a kind of thing does just before it hurts her, gets clear when she sees it, and keeps clear of a
thing she does not yet know that is closing on her faster than she can answer.

The tells of action games (an enemy that stops and draws back before it lunges, a block that shakes before it drops)
are given so a player can read them; nobody learns them as a list.
"""
from __future__ import annotations

from types import SimpleNamespace as Thing

import pytest

from core.agency.warning_signs import STOPS, WarningSigns

pytestmark = pytest.mark.unit

ME = Thing(number=0, x=100.0, y=100.0, kind=0)


def _brute(at_rest: bool, x: float = 300.0) -> dict[int, Thing]:
    return {7: Thing(number=7, kind=3, x=x, y=100.0, w=20.0, h=20.0, vx=0.0 if at_rest else 30.0, vy=0.0, still=at_rest)}


def test_a_change_that_comes_before_losses_more_than_chance_becomes_a_tell_and_is_read_after():
    signs = WarningSigns()
    losses: list[dict] = []
    at = 0.0
    for _round in range(3):                                   # it paces, stops, and a moment later she is hit
        for _ in range(6):
            signs.saw(_brute(False), ME, at, losses)
            at += 0.1
        stopped = at
        losses.append({"what": "loss", "at": stopped + 0.3})
        for _ in range(25):                                   # it stands a while after, then paces on
            signs.saw(_brute(True), ME, at, losses)
            at += 0.1
        for _ in range(30):                                   # long enough for each change to be settled
            signs.saw(_brute(False), ME, at, losses)
            at += 0.1
    assert signs.is_a_tell(3, STOPS)
    said = signs.news(lambda kind: "the brutes")
    assert said == ["When the brutes stop, harm follows: I'll get clear when I see it."] and signs.news(lambda k: "") == []
    for _ in range(6):
        signs.saw(_brute(False), ME, at, losses)
        at += 0.1
    for _ in range(5):
        shown = signs.saw(_brute(True), ME, at, losses)
        at += 0.1
        if shown:
            break
    assert (3, STOPS) in shown and signs.showing(_brute(True)[7], at)


def test_a_change_after_which_nothing_comes_is_no_tell_and_a_thing_coming_fast_is_kept_clear_of():
    signs = WarningSigns()
    at = 0.0
    for _round in range(4):
        for at_rest in (False,) * 6 + (True,) * 3:
            signs.saw(_brute(at_rest), ME, at, [])
            at += 0.1
        for _ in range(25):
            signs.saw(_brute(False), ME, at, [])
            at += 0.1
    assert not signs.is_a_tell(3, STOPS)
    rushing = {9: Thing(number=9, kind=5, x=160.0, y=100.0, w=10.0, h=10.0, vx=-200.0, vy=0.0, still=False)}
    signs.saw(rushing, ME, at, [])
    signs.saw(rushing, ME, at + 0.05, [])
    assert signs.closing_in(rushing[9], ME, within_s=0.5)                      # 60 px at 200 px/s: 0.3 s away
    leaving = Thing(number=9, kind=5, x=160.0, y=100.0, w=10.0, h=10.0, vx=200.0, vy=0.0, still=False)
    assert not signs.closing_in(leaving, ME, within_s=0.5)


def test_what_she_keeps_clear_of_that_went_out_of_sight_is_still_danger_where_it_was_going():
    from core.agency.playing_as_it_happens import AVOID, _Choosing
    from core.perception.what_moves_in_the_picture import Thing as Seen

    def seen(number, x, y, vx=0.0):
        import numpy as np
        return Seen(number, x, y, 10.0, 10.0, np.zeros(4), (200, 30, 30), np.zeros((1, 1, 3)), born=0.0, seen=0.0, kind=4,
                    vx=vx)

    choosing = object.__new__(_Choosing)
    choosing.mine, choosing.at, choosing.reach = seen(0, 100.0, 50.0), 10.0, None
    choosing.moves = Thing(shape=(120, 200), out_of_sight={9: (seen(9, 40.0, 50.0, vx=60.0), 9.5)})
    choosing.meeting = Thing(stance=lambda kind: AVOID if kind == 4 else "meet")
    choosing.others = lambda: []
    staying = choosing.danger_along(lambda after: (0.0, 0.0))
    keeping_ahead = choosing.danger_along(lambda after: (60.0 * after, 0.0))
    assert staying > 0.0 and keeping_ahead < staying                            # it comes out where she stands


def test_many_changes_standing_out_at_once_each_need_more_before_they_are_tells_and_are_said_one_at_a_time():
    """Offline 2026-10-10 five tells were learned in one second from the same few losses, and all five were said."""
    signs = WarningSigns()
    signs._by_chance = lambda: 0.1  # what a loss follows by chance does not decide this
    for kind in range(5):
        signs.counts[(kind, STOPS)] = (2, 0)
    assert not any(signs.is_a_tell(kind, STOPS) for kind in range(5))
    signs.counts[(0, STOPS)] = (4, 0)
    assert signs.is_a_tell(0, STOPS)
    signs.counts[(1, STOPS)] = (4, 1)
    said = signs.news(lambda kind: f"the kind {kind} things")
    assert said == ["When the kind 0 things stop, harm follows: I'll get clear when I see it."]
    assert signs.news(lambda kind: f"the kind {kind} things") == ["When the kind 1 things stop, harm follows: I'll get clear when I see it."]


def test_a_change_across_the_picture_is_explained_by_the_one_beside_her_and_is_no_tell():
    """Offline 2026-10-12 coins, bars and dogs all over a busy picture each changed before the same few losses, and
    eight tells were learned and said in five seconds. The thing beside her when she is hurt is the likelier cause."""
    signs = WarningSigns()
    losses: list[dict] = []
    at = 0.0

    def picture(near_at_rest: bool, far_at_rest: bool) -> dict[int, Thing]:
        near = Thing(number=7, kind=3, x=130.0, y=100.0, w=20.0, h=20.0, vx=0.0 if near_at_rest else 30.0, vy=0.0,
                     still=near_at_rest)
        far = Thing(number=8, kind=6, x=600.0, y=400.0, w=20.0, h=20.0, vx=0.0 if far_at_rest else 30.0, vy=0.0,
                    still=far_at_rest)
        return {7: near, 8: far}

    for _round in range(4):                                   # both stop, and a moment later she is hit
        for _ in range(6):
            signs.saw(picture(False, False), ME, at, losses)
            at += 0.1
        losses.append({"what": "loss", "at": at + 0.3})
        for _ in range(25):
            signs.saw(picture(True, True), ME, at, losses)
            at += 0.1
        for _ in range(30):
            signs.saw(picture(False, False), ME, at, losses)
            at += 0.1
    assert signs.is_a_tell(3, STOPS)
    assert not signs.is_a_tell(6, STOPS)


def test_tells_found_together_are_said_some_seconds_apart():
    signs = WarningSigns()
    signs._by_chance = lambda: 0.1
    for kind in range(2):
        signs.counts[(kind, STOPS)] = (5, 0)
    assert len(signs.news(lambda kind: f"the kind {kind} things", at=10.0)) == 1
    assert signs.news(lambda kind: f"the kind {kind} things", at=11.0) == []
    assert len(signs.news(lambda kind: f"the kind {kind} things", at=30.0)) == 1


def test_with_no_body_of_hers_and_a_key_to_duck_she_learns_the_tell_and_ducks_as_it_begins():
    """Offline 2026-10-12 a food fight's opponent drew back before every throw, the throw landed a sixth of a second
    after it left his hand, and she, with a key to duck and no body in the picture, never learned to duck as he drew
    back. Nothing here is that game: someone across the way stands still a moment before each hit to her bar."""
    from types import SimpleNamespace

    from core.agency.warning_signs import read_the_signs

    def across(at_rest: bool) -> SimpleNamespace:
        return SimpleNamespace(kinds=[], things={7: Thing(number=7, kind=3, x=200.0, y=80.0, w=30.0, h=50.0,
                                                vx=0.0 if at_rest else 40.0, vy=0.0, still=at_rest)})

    said: list[str] = []
    signs, losses, at = WarningSigns(), [], 0.0
    read_the_signs(signs, across(False), None, losses, at, said.append)        # no body, no key: nothing is read
    assert signs.began is None
    for _round in range(4):
        for _ in range(8):
            read_the_signs(signs, across(False), None, losses, at, said.append, out_of_the_way=True)
            at += 0.1
        losses.append({"what": "loss", "at": at + 0.4})
        for _ in range(6):
            read_the_signs(signs, across(True), None, losses, at, said.append, out_of_the_way=True)
            at += 0.1
        for _ in range(25):
            read_the_signs(signs, across(False), None, losses, at, said.append, out_of_the_way=True)
            at += 0.1
    assert signs.is_a_tell(3, STOPS) and said
    for _ in range(8):
        read_the_signs(signs, across(False), None, losses, at, said.append, out_of_the_way=True)
        at += 0.1
    read_the_signs(signs, across(True), None, losses, at, said.append, out_of_the_way=True)
    assert signs.just_shown(at + 0.1) and not signs.just_shown(at + 0.15)       # once for each tell
