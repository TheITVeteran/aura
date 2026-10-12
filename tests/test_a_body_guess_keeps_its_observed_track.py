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


def test_a_lost_reading_waits_for_measured_absence_and_never_duplicates_a_request():
    from core.perception.reading_again_after_loss import ReadingAgain, _reading_renewal_invariant

    assert _reading_renewal_invariant() == ()
    reading = ReadingAgain()
    absent = dict(pending=False, measured=True, usable=False, needed=True)
    assert reading.due(1., **absent)
    reading.began(1.)
    assert not reading.due(20., pending=True, measured=True, usable=False, needed=True)
    assert not reading.due(21., **absent)
    assert not reading.due(35., **absent)
    assert reading.due(36., **absent)
    assert not reading.due(37., pending=False, measured=False, usable=False, needed=True)
    assert not reading.due(37.1, **absent)
    assert reading.due(37.5, **absent)
    assert not reading.due(38., pending=False, measured=True, usable=True, needed=True)
    assert not reading.due(39., pending=False, measured=True, usable=False, needed=False)
    assert not reading.due(float('nan'), **absent)
    assert not reading.due(1., **absent)


@pytest.mark.asyncio
async def test_a_retired_visual_answer_is_reobserved_on_a_fresh_frame(monkeypatch):
    from core.agency.naming_what_she_sees import SeeingInPlay, _who_i_am_to_my_eyes
    from core.agency.which_one_answers_to_her import WhichIsHers
    from core.cognition.a_guide_to_a_place import TOLD, Guide
    from core.cognition.guidance_lifetime import guidance_work
    from core.perception import where_i_am_on_screen

    unblock = asyncio.Event()
    frames = []
    async def eyes(picture, _names):
        frames.append(picture.copy())
        await unblock.wait()
        return Avatar("body", (.1, .1, .2, .2))
    monkeypatch.setattr(where_i_am_on_screen, "where_the_avatar_is", eyes)
    guide = Guide()
    guide.take_in(TOLD, "Use the arrow keys to move.")
    seeing = SeeingInPlay(began=0.)
    run = SimpleNamespace(keys=["left", "right"], trying=4,
                          presses=SimpleNamespace(moves_anything=lambda _px: True))
    meeting = SimpleNamespace(stance=lambda _kind: "", known=lambda _kind: False)
    body, hers = _body(), WhichIsHers()
    picture = np.zeros((100, 100, 3), np.uint8)
    def look(moves, at):
        _who_i_am_to_my_eyes(run, seeing, guide, moves, hers, meeting, picture, at, lambda *_args: None)

    async with guidance_work():
        look(_moves(body), 2.)
        first_request = seeing.avatar_asking
        await asyncio.sleep(0)
        look(_moves(at=2.1), 2.1)
        unblock.set()
        await asyncio.wait_for(first_request, 1.)
        replacement = _body(2, at=2.2)
        look(_moves(replacement, at=2.2), 2.2)
        assert hers.number is None and seeing.avatar_observation.thing(_moves(replacement), 2.2) is None
        replacement.seen = 4.
        look(_moves(replacement, at=4.), 4.)
        assert seeing.avatar_asking is first_request
        picture[:] = 120
        replacement.seen = 4.3
        look(_moves(replacement, at=4.3), 4.3)
        assert seeing.avatar_asking is not first_request
        await asyncio.wait_for(seeing.avatar_asking, 1.)
        assert len(frames) == 2 and frames[0].max() == 0 and frames[1].min() == 120
        for tick in range(44, 91):
            at = tick / 10
            replacement.x, replacement.seen = 15. + 5. * (at - 4.3), at
            look(_moves(replacement, at=at), at)
        assert hers.number == 2
        hers.number = 7
        replacement.seen = 12.
        look(_moves(replacement, at=12.), 12.)
        assert hers.number == 7 and len(frames) == 2


def test_refreshing_an_observation_has_a_registered_measurement():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(item for item in suite.tests() if item.name == "lost_readings_are_refreshed_without_duplicate_requests")
    assert check.predict(None) is True
