"""Source-bound body controls reach the finite discovery pass without a body claim."""
from types import SimpleNamespace

import pytest

from core.agency.how_a_game_works import POINTS, SEEN_SURE, Answer, understood
from core.cognition.a_guide_to_a_place import Guide
from core.perception.how_a_program_plays import AKey, HowItPlays
from core.perception.reading_a_program import ProgramRead


def parsed_guide(by='Rover.moveBy(1, 0)', *, body='Rover', role='moves'):
    guide = Guide()
    guide.take_in_program(ProgramRead(keys={'j': ''}, plays=HowItPlays(
        keys={'j': AKey('j', role, by, 'Rover.update')}, hers={'class': body})), at=1.)
    return guide


def test_a_known_method_reaches_discovery_even_when_the_callers_keys_omit_it():
    guide = parsed_guide()
    guide.how_seen['me'] = Answer(POINTS, SEEN_SURE, ['earlier clicks changed a menu'])
    how = understood(guide)
    assert how.keys_for_body_trials(['return']) == ['return', 'j']
    assert not how.try_pointer_first(True)
    assert how.me.value == POINTS  # an input experiment does not replace observed identity or role


def test_an_unresolved_or_other_objects_method_cannot_supply_body_keys():
    for guide in (parsed_guide(body=''), parsed_guide(by='Decoration.moveBy(1, 0)'),
                  parsed_guide(by='Rover.moveBy(1, 0)', role='does something')):
        assert not understood(guide).body_keys_from_code


def test_a_changed_source_binding_retires_the_experiment_keys_from_the_cached_understanding():
    guide = parsed_guide()
    how = understood(guide)
    assert how.body_keys_from_code == ('j',)
    guide.code_says.hers['class'] = 'another object'
    assert understood(guide) is how and not how.body_keys_from_code


def test_a_source_binding_without_a_current_control_does_not_get_synthesized():
    guide = parsed_guide()
    guide.controls.clear()
    assert not understood(guide).body_keys_from_code


def test_only_a_qualified_lexical_body_can_resolve_this():
    assert understood(parsed_guide(by='this.moveBy(1, 0)')).body_keys_from_code == ('j',)
    guide = parsed_guide(by='this.moveBy(1, 0)')
    guide.code_says.keys['j'].where = 'another method'
    assert not understood(guide).body_keys_from_code
    guide.code_says.keys['j'].where = 'Rover.Decoration.update'
    assert not understood(guide).body_keys_from_code


def test_an_opaque_body_key_can_be_experimentally_qualified_without_default_arrows():
    guide = SimpleNamespace(controls={}, code_says=None, how=None, _read=(), pointer={}, how_seen={})
    assert not understood(guide).body_keys_from_code


def test_parsed_body_experiments_have_a_registered_measured_claim():
    from core.agency.how_a_game_works import _parsed_body_trial_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _parsed_body_trial_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'parsed_body_controls_reach_finite_discovery')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())


@pytest.mark.asyncio
async def test_the_actual_pixel_loop_tries_the_parsed_key_omitted_by_its_caller(monkeypatch):
    import asyncio
    import time

    import numpy as np

    from core.agency import playing_as_it_happens as playing
    from core.cognition.a_guide_to_a_place import SCREEN, THE_GUIDE, Control
    from core.perception.what_moves_in_the_picture import WhatMoves

    guide = parsed_guide()
    guide.take_in(SCREEN, ['Press RETURN to play this level again.'], at=1.)
    guide.controls['return'] = Control('return', 'play this level again', SCREEN, 1.)
    held, actions = set(), []
    x, last = 20., time.monotonic()

    class Hands:
        async def down(self, key):
            held.add(key)
            actions.append(('down', key))

        async def up(self, key):
            held.discard(key)
            actions.append(('up', key))

        async def click(self, *_args):
            actions.append(('click', ''))

        async def point(self, *_args):
            pass

        async def tap(self, key):
            actions.append(('tap', key))

    async def look():
        nonlocal x, last
        await asyncio.sleep(.02)
        now = time.monotonic()
        x += 20. * (now - last) if 'j' in held else 0.
        last = now
        picture = np.zeros((64, 96, 3), dtype=np.uint8)
        col = min(80, int(x))
        picture[30:38, col:col + 8] = (30, 210, 90)
        picture[32:36, col + 2:col + 6] = (220, 240, 180)
        return picture, now

    monkeypatch.setattr(playing, 'eyes_for_play', lambda **kwargs: WhatMoves(**kwargs))
    token = THE_GUIDE.set(guide)
    try:
        await asyncio.wait_for(playing.play_as_it_happens(
            look, Hands(), keys=['return'], pointer_first=True, seconds=2.), 4.)
    finally:
        THE_GUIDE.reset(token)
    assert ('down', 'j') in actions and ('up', 'j') in actions and x > 20.
    assert ('down', 'return') not in actions and not held
