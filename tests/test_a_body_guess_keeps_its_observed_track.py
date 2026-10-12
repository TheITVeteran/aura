"""A visual box cannot keep naming whatever later occupies its old place."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency.a_body_seen_then import ABodySeenThen, _body_seen_then_invariant
from core.perception.where_i_am_on_screen import Avatar

pytestmark = pytest.mark.unit


def _body(number=1, *, x=15., at=2., look=(1., 0.)):
    return SimpleNamespace(number=number, kind=3, x=x, y=15., w=8., h=8., vx=5., vy=0.,
                           seen=at, look=np.asarray(look))


def _moves(*things, at=2., measured=True):
    return SimpleNamespace(things={thing.number: thing for thing in things},
                           share=lambda x, y: (x / 100, y / 100), seeing=measured, _last_at=at)


def test_a_delayed_box_selects_the_captured_track_at_its_new_position():
    body = _body()
    moves = _moves(body)
    guess = ABodySeenThen(moves, 2.)
    for at in (3., 4.):
        body.x, body.seen = 15. + 5. * (at - 2.), at
        moves._last_at = at
        guess.follow(moves, at, measured=True)
    replacement = _body(2, x=15., at=4., look=(0., 1.))
    moves.things[2] = replacement
    guess.bind(Avatar("body", (.1, .1, .2, .2)))
    assert guess.thing(moves, 4.) is body
    assert guess.observed[1].place == (15., 15.)
    assert body.x == 25.


def test_measured_absence_retires_the_box_before_its_answer_arrives():
    body = _body()
    guess = ABodySeenThen(_moves(body), 2.)
    guess.follow(_moves(at=3.), 3., measured=True)
    replacement = _body(2, at=4.)
    moves = _moves(replacement, at=4.)
    guess.follow(moves, 4., measured=True)
    guess.bind(Avatar("body", (.1, .1, .2, .2)))
    assert guess.thing(moves, 4.) is None
    assert guess.retired == {1}


def test_an_unread_picture_neither_proves_absence_nor_supplies_current_identity():
    body = _body()
    guess = ABodySeenThen(_moves(body), 2.)
    guess.bind(Avatar("body", (.1, .1, .2, .2)))
    guess.follow(_moves(at=2.1, measured=False), 2.1, measured=False)
    assert not guess.retired and guess.thing(_moves(body), 2.1) is None
    body.x, body.seen = 16., 2.2
    moves = _moves(body, at=2.2)
    guess.follow(moves, 2.2, measured=True)
    assert guess.thing(moves, 2.2) is body


def test_a_track_number_with_changed_pixel_colours_cannot_keep_the_guess():
    body = _body()
    guess = ABodySeenThen(_moves(body), 2.)
    guess.bind(Avatar("body", (.1, .1, .2, .2)))
    body.look[:] = (0., 1.)
    body.seen = 2.1
    moves = _moves(body, at=2.1)
    guess.follow(moves, 2.1, measured=True)
    assert guess.thing(moves, 2.1) is None
    assert guess.observed[1].look == (1., 0.)


def test_an_extrapolated_track_is_not_a_current_observation():
    body = _body()
    moves = _moves(body)
    guess = ABodySeenThen(moves, 2.)
    guess.bind(Avatar("body", (.1, .1, .2, .2)))
    guess.follow(moves, 2.1, measured=True)
    assert guess.thing(moves, 2.1) is None
    assert not guess.retired


def test_the_visual_track_invariant():
    assert _body_seen_then_invariant() == ()


@pytest.mark.asyncio
async def test_live_naming_bootstraps_once_and_never_overrides_causal_identity(monkeypatch):
    from core.agency.naming_what_she_sees import SeeingInPlay, _who_i_am_to_my_eyes
    from core.agency.which_one_answers_to_her import WhichIsHers
    from core.cognition.a_guide_to_a_place import TOLD, Guide
    from core.cognition.guidance_lifetime import guidance_work
    from core.perception import where_i_am_on_screen

    async def eyes(*_args):
        await asyncio.sleep(0)
        return Avatar("body", (.1, .1, .2, .2))

    monkeypatch.setattr(where_i_am_on_screen, "where_the_avatar_is", eyes)
    guide = Guide()
    guide.take_in(TOLD, "Use the arrow keys to move.")
    seeing = SeeingInPlay(began=0.)
    run = SimpleNamespace(keys=["left", "right"], trying=4,
                          presses=SimpleNamespace(moves_anything=lambda _px: True))
    meeting = SimpleNamespace(stance=lambda _kind: "", known=lambda _kind: False)
    body, hers = _body(), WhichIsHers()
    moves = _moves(body)
    picture = np.zeros((100, 100, 3), np.uint8)
    def say(*_args):
        return None

    async with guidance_work():
        _who_i_am_to_my_eyes(run, seeing, guide, moves, hers, meeting, picture, 2., say)
        await asyncio.wait_for(seeing.avatar_asking, 1.)
        for at in range(3, 9):
            body.x, body.seen = 15. + 5. * (at - 2), float(at)
            moves._last_at = float(at)
            _who_i_am_to_my_eyes(run, seeing, guide, moves, hers, meeting, picture, float(at), say)
        assert hers.number == body.number
        hers.number, hers.kind = 7, 4
        body.seen, moves._last_at = 8.1, 8.1
        _who_i_am_to_my_eyes(run, seeing, guide, moves, hers, meeting, picture, 8.1, say)
        assert hers.number == 7
        hers.number = None
        moves.things, moves._last_at = {}, 8.2
        _who_i_am_to_my_eyes(run, seeing, guide, moves, hers, meeting, picture, 8.2, say)
        replacement = _body(2, at=8.3)
        moves.things, moves._last_at = {2: replacement}, 8.3
        _who_i_am_to_my_eyes(run, seeing, guide, moves, hers, meeting, picture, 8.3, say)
        assert hers.number is None
