"""Optional geometric discovery respects the activity already described by a place."""
import time
from types import SimpleNamespace

import pytest

from core.agency.how_a_game_works import _picture_checks_invariant, how_it_works, understood
from core.cognition.a_guide_to_a_place import SCREEN, THE_GUIDE, Guide
from core.skills import screen_pursuit_as_it_happens as pursuit


@pytest.mark.parametrize('said', [
    'Use the arrow keys to guide a robot safely onto its charging platform.',
    'Drive your vehicle into the marked space.',
    'Use the mouse to catch the falling fruit before it hits the ground.',
])
def test_continuous_movement_gets_control_before_incompatible_geometry(said):
    how = how_it_works(said)
    assert how.picture_check_fits('board') is False
    assert how.picture_check_fits('well') is False


@pytest.mark.parametrize('said', ['Use the arrow keys. Gravity pulls things down.', 'A grid of squares.', 'SCORE 0'])
def test_shared_controls_and_incidental_geometry_leave_discovery_unresolved(said):
    how = how_it_works(said)
    assert how.picture_check_fits('board') is None and how.picture_check_fits('well') is None


def test_specific_turn_and_falling_piece_rules_keep_their_checks_available():
    assert how_it_works('YOUR TURN: move a piece.').picture_check_fits('board') is True
    assert how_it_works('Use arrow keys to move falling blocks and clear rows.').picture_check_fits('well') is True
    mixed = how_it_works('Guide your craft onto the platform. Next clear rows of falling blocks.')
    assert mixed.picture_check_fits('well') is True
    assert _picture_checks_invariant() == ()


def test_replacing_instruction_content_cannot_keep_an_old_cached_activity():
    guide = SimpleNamespace(_read={'Guide your craft onto the platform.'}, controls={}, pointer={}, how=None)
    assert understood(guide).picture_check_fits('board') is False
    guide._read = {'YOUR TURN: move a piece.'}
    assert understood(guide).picture_check_fits('board') is True


@pytest.mark.asyncio
async def test_the_actual_route_enters_control_using_instructions_kept_before_recent_screens(monkeypatch):
    guide = Guide(place='an invented moving world')
    guide.take_in(SCREEN, 'Use the arrow keys to guide a robot safely onto its charging platform.')
    token = THE_GUIDE.set(guide)
    reflexes = pursuit.PlayingAsItHappens(object(), (0., 0., 1., 1.), 'Reach the platform.', time.monotonic() + 30.)
    reflexes.words = ['SCORE 0'] * 8
    reflexes.said_now = 'SCORE 0'
    reflexes.under_her_hand = True

    async def absent(*_args, **_kwargs):
        return False

    async def incompatible(*_args, **_kwargs):
        pytest.fail('an incompatible picture check delayed control')

    class ControlEnteredError(Exception):
        pass

    async def control(*_args, **kwargs):
        assert 'up' in kwargs['keys'] and 'down' in kwargs['keys']
        raise ControlEnteredError

    from core.agency import playing_as_it_happens

    for name in ('_hers_to_send_is_in_sight', '_chose_what_it_asks', '_searched_the_places'):
        monkeypatch.setattr(pursuit, name, absent)
    for name in ('_typed_what_it_asks', '_did_again_what_it_showed'):
        monkeypatch.setattr(pursuit.PlayingAsItHappens, name, absent)
    monkeypatch.setattr(pursuit, '_played_a_board', incompatible)
    monkeypatch.setattr(pursuit, '_played_a_well', incompatible)
    monkeypatch.setattr(playing_as_it_happens, 'play_as_it_happens', control)
    try:
        with pytest.raises(ControlEnteredError):
            await reflexes.while_it_moves()
    finally:
        THE_GUIDE.reset(token)


def test_activity_routing_is_a_registered_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'picture_checks_follow_the_activity')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())
