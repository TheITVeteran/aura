"""Movement discovery can finish without requiring an emission to move its owner."""
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency import playing_as_it_happens as play
from core.agency.how_a_game_works import understood
from core.agency.what_meeting_things_does import SHOOT, WhatMeetingDoes
from core.agency.which_one_answers_to_her import Makes, WhichIsHers
from core.cognition.a_guide_to_a_place import COUNSEL, PROGRAM, SCREEN, THE_GUIDE, Control, Guide
from core.perception.a_world_as_its_code_makes_it import (
    AKindOfThing,
    AWorldAsItsCodeMakesIt,
    HerBody,
    kept,
    unkept,
)
from core.perception.what_moves_in_the_picture import Kind, Thing

pytestmark = pytest.mark.unit


def guide(*, combined=False):
    current = Guide()
    current.controls = {key: Control(key, '', PROGRAM, 1.) for key in ('a', 'd', 'j', 'unknown')}
    paces = {'a': (-80., 0.), 'd': (80., 0.)}
    if combined:
        paces['j'] = (0., -100.)
    current.world_its_code_makes = AWorldAsItsCodeMakesIt(
        hers=HerBody(name='body', paces=paces), sends={'j': 'markers'})
    return current


def test_only_emission_effects_leave_body_trials_and_remain_in_the_guide():
    current = guide()
    how = understood(current)
    assert how.keys_for_body_trials(['a', 'd', 'j', 'unknown']) == ['a', 'd', 'unknown']
    assert 'j' in current.controls and current.world_its_code_makes.sends == {'j': 'markers'}
    assert how.has_a_body() is None


def test_combined_movement_and_emission_still_needs_a_movement_trial():
    assert understood(guide(combined=True)).keys_for_body_trials(['j']) == ['j', 'a', 'd']


def test_changed_or_missing_source_restores_the_unresolved_control():
    current = guide()
    how = understood(current)
    current.world_its_code_makes = None
    assert understood(current) is how
    assert how.keys_for_body_trials(['j']) == ['j']
    current.world_its_code_makes = AWorldAsItsCodeMakesIt(sends={'j': 'markers'})
    assert understood(current).keys_for_body_trials(['j']) == ['j']


def test_source_findings_require_a_current_control_binding():
    current = guide()
    current.controls.clear()
    how = understood(current)
    assert not how.body_keys_from_code and not how.emission_keys_from_code
    assert how.keys_for_body_trials(['j']) == ['j']


def test_saving_and_loading_preserves_every_declared_world_field():
    import json

    world = guide().world_its_code_makes
    world.things = [AKindOfThing('markers', moves='rises', speed=100.),
                    AKindOfThing('obstacles', touching='costs a life', past_an_edge='costs a life')]
    world.read_at = {'body': 'update', 'sends': 'emit'}
    assert unkept(json.loads(json.dumps(kept(world)))) == world


def test_external_counsel_cannot_add_controls_when_this_place_names_them():
    current = Guide()
    current.take_in(SCREEN, 'Use the left and right arrow keys to move. Press SPACE to fire.')
    current.take_in(COUNSEL, 'Press J to jump and H to hide.')
    assert set(current.keys_for_play()) == {'left', 'right', 'space'}
    assert 'j' in current.controls and current.controls['j'].source == COUNSEL


def test_counsel_can_supply_unresolved_controls_when_no_local_instructions_exist():
    current = Guide()
    current.take_in(COUNSEL, 'Press J to jump.')
    assert current.keys_for_play() == ['j']


def test_the_play_handoff_retires_unqualified_remembered_names():
    from core.agency.body_trials_keep_the_world import (
        _control_source_invariant,
        keys_with_current_sources,
    )

    current = Guide()
    current.take_in(SCREEN, 'Use left and right to move. Press SPACE to fire.')
    current.take_in(COUNSEL, 'Press J to jump.')
    assert keys_with_current_sources(current, ['j', 'h'], ['i'], current.keys_for_play()) == ['left', 'right', 'space']
    assert _control_source_invariant() == ()


@pytest.mark.asyncio
async def test_actual_actor_reaches_its_trigger_after_only_movement_is_measured(monkeypatch):
    current = guide()
    # These settled measurements supply control responses; source findings never create them.
    hers = WhichIsHers()
    hers.number, hers.kind = 1, 0
    monkeypatch.setattr(hers, 'way_of', lambda key: {'a': (-80., 0.), 'd': (80., 0.)}.get(key))
    hers.makes['j'] = Makes('j', 2, 0., -100., 0)
    mine = Thing(1, 50., 100., 12., 12., np.zeros(64), (0, 240, 30),
                 np.zeros((1, 1, 3), dtype=np.uint8), 1., 10., kind=0)
    target = Thing(2, 50., 20., 12., 12., np.zeros(64), (240, 30, 0),
                   np.zeros((1, 1, 3), dtype=np.uint8), 1., 10., kind=1)
    moves = SimpleNamespace(seeing=True, shape=(150, 150), things={1: mine, 2: target},
                            kinds=[Kind(k, np.zeros(64), 144., (0, 0, 0)) for k in range(3)])
    meeting = WhatMeetingDoes()
    meeting.told[1] = SHOOT
    run = play._Run(keys=understood(current).keys_for_body_trials(['a', 'd', 'j']), began=1.)
    actions = []

    class Hands:
        async def tap(self, key):
            actions.append(('tap', key))

        async def down(self, key):
            actions.append(('down', key))

        async def up(self, key):
            actions.append(('up', key))

    monkeypatch.setattr(play.time, 'monotonic', lambda: 10.)
    token = THE_GUIDE.set(current)
    try:
        choosing = play._Choosing(moves, hers, meeting, run.keys)
        assert not play._still_trying_keys(run, hers, choosing, False, at=10.)
        await play._act(Hands(), run, moves, hers, meeting, choosing, 10.)
    finally:
        THE_GUIDE.reset(token)
    assert ('tap', 'j') in actions and run.taps == 1
    assert hers.way_of('j') is None and run.trying == 0


def test_registered_claim_measures_the_effect_partition():
    from core.agency.how_a_game_works import _motion_and_emission_trials_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite
    from core.perception.a_world_as_its_code_makes_it import _saved_world_effects_invariant

    assert _motion_and_emission_trials_invariant() == ()
    assert _saved_world_effects_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    for name in ('body_motion_trials_keep_emission_controls_separate', 'current_controls_keep_their_sources',
                 'saved_world_readings_preserve_action_effects'):
        check = next(t for t in suite.tests() if t.name == name)
        assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
