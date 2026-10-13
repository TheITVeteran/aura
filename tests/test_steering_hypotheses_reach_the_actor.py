"""A provisional physical forecast reaches real input delivery before facing calibration."""
from types import SimpleNamespace

import pytest

from core.agency import playing_as_it_happens as play
from core.agency.steering_from_a_model import (
    _turn_keys,
    observed_steering_model,
    steer_with_a_model,
)
from core.agency.which_one_answers_to_her import WhichIsHers
from core.agency.which_way_she_faces import WhichWaySheFaces
from core.cognition.a_guide_to_a_place import THE_GUIDE

pytestmark = pytest.mark.unit


def _setup(monkeypatch):
    now = [.3]
    monkeypatch.setattr(play.time, 'monotonic', lambda: now[0])
    faces = WhichWaySheFaces(facing=0.)
    faces.along['opaque-drive'] = [60.]
    run = play._Run(keys=['opaque-drive', 'opaque-a', 'opaque-b'], began=0., faces=faces,
                    trying=12, held='opaque-drive', held_since=0., held_trying=True)
    run.how_it_stands = (.3, False, False)
    hers = WhichIsHers()
    hers.number, hers.kind = 1, 0
    hers.holding('opaque-drive', 0., trying=True)
    mine = SimpleNamespace(number=1, kind=0, born=0., seen=.3, x=150., y=150.,
                           w=20., h=20., vx=60., vy=0., motion_evidence='direct')
    moves = SimpleNamespace(shape=(300, 300), things={1:mine}, seeing=True)
    choosing = SimpleNamespace(mine=mine, moves=moves, next_picture_s=.1, response_s=0.,
                               way_through=None, target=lambda: ((150., 240.), 'meet', None),
                               ways={'opaque-drive':(60., 0.)}, lifts=[])
    actions = []

    class Hands:
        async def down(self, key):
            actions.append(('down', key))
        async def up(self, key):
            actions.append(('up', key))

    guide = SimpleNamespace(controls={
        'opaque-a':SimpleNamespace(act='turn left', on=''),
        'opaque-b':SimpleNamespace(act='turn right', on='')})
    observed_steering_model(faces, mine, hers, moves, .3)
    return now, run, hers, mine, moves, choosing, actions, Hands(), guide


@pytest.mark.asyncio
async def test_the_actor_delivers_a_provisional_turn_before_complete_calibration(monkeypatch):
    now, run, hers, _mine, moves, choosing, actions, hands, guide = _setup(monkeypatch)
    token = THE_GUIDE.set(guide)
    try:
        await play._act(hands, run, moves, hers, None, choosing, now[0])
    finally:
        THE_GUIDE.reset(token)
    assert actions[0] == ('up', 'opaque-drive')
    assert actions[1][0] == 'down' and actions[1][1] in ('opaque-a', 'opaque-b')
    assert run._modelled_steering is not None and run.facing_trials == 0
    assert not run.held_trying and hers._held[-1][2] is False
    assert not run.faces.turns_her() and not run.faces.turning_keys()


@pytest.mark.asyncio
async def test_a_delivered_forecast_retains_its_finite_input_window(monkeypatch):
    now, run, hers, _mine, _moves, choosing, actions, hands, guide = _setup(monkeypatch)
    token = THE_GUIDE.set(guide)
    try:
        assert await steer_with_a_model(hands, run, hers, choosing, now[0], hold=play._hold)
        now[0] += .05
        assert await steer_with_a_model(hands, run, hers, choosing, now[0], hold=play._hold)
    finally:
        THE_GUIDE.reset(token)
    assert len(actions) == 2
    assert 0 < run._modelled_steering[1] - .3 <= .35


@pytest.mark.asyncio
async def test_the_actor_uses_accumulated_slow_motion_without_a_calibrated_driver(monkeypatch):
    now, run, hers, mine, moves, choosing, actions, hands, guide = _setup(monkeypatch)
    run.faces.along.clear()
    run.faces.steering_model = None
    choosing.ways.clear()
    token = THE_GUIDE.set(guide)
    try:
        for index in range(13):
            now[0] = mine.seen = index * .1
            mine.x, mine.vx = 150. + 4. * now[0], 4.
            observed_steering_model(run.faces, mine, hers, moves, now[0], run=run)
        assert run.faces.ahead_key(least=1) is None
        assert run.faces.steering_model.driver(run.keys, now=now[0]) == 'opaque-drive'
        assert await steer_with_a_model(hands, run, hers, choosing, now[0], hold=play._hold)
    finally:
        THE_GUIDE.reset(token)
    assert run._modelled_steering is not None and not run.faces.turns_her()
    assert actions[0] == ('up', 'opaque-drive') and actions[1][0] == 'down'


@pytest.mark.asyncio
async def test_changed_birth_ends_the_previous_forecast_and_releases_its_input(monkeypatch):
    now, run, hers, mine, _moves, choosing, actions, hands, guide = _setup(monkeypatch)
    token = THE_GUIDE.set(guide)
    try:
        assert await steer_with_a_model(hands, run, hers, choosing, now[0], hold=play._hold)
        mine.born, now[0] = .31, .35
        assert await steer_with_a_model(hands, run, hers, choosing, now[0], hold=play._hold)
    finally:
        THE_GUIDE.reset(token)
    assert actions[-1] == ('up', actions[1][1]) and not run.held
    assert run._modelled_steering is None


@pytest.mark.asyncio
async def test_an_active_calibration_interval_keeps_its_observation_time(monkeypatch):
    from core.agency.giving_a_trial_time import ControlTrial

    now, run, hers, _mine, _moves, choosing, actions, hands, guide = _setup(monkeypatch)
    run.facing_trial = ControlTrial(run.held, 0., .7, hers.identification.epoch)
    token = THE_GUIDE.set(guide)
    try:
        assert not await steer_with_a_model(hands, run, hers, choosing, now[0], hold=play._hold)
    finally:
        THE_GUIDE.reset(token)
    assert not actions


@pytest.mark.asyncio
async def test_an_unrelated_switch_cannot_arm_a_turning_hypothesis(monkeypatch):
    now, run, hers, _mine, _moves, choosing, actions, hands, guide = _setup(monkeypatch)
    for control in guide.controls.values():
        control.act = 'turn on music'
    token = THE_GUIDE.set(guide)
    try:
        assert not await steer_with_a_model(hands, run, hers, choosing, now[0], hold=play._hold)
    finally:
        THE_GUIDE.reset(token)
    assert not actions


@pytest.mark.asyncio
async def test_a_cartesian_body_without_turn_evidence_keeps_ordinary_planning(monkeypatch):
    now, run, hers, _mine, _moves, choosing, actions, hands, guide = _setup(monkeypatch)
    for control in guide.controls.values():
        control.act = 'move left'
    token = THE_GUIDE.set(guide)
    try:
        assert not await steer_with_a_model(hands, run, hers, choosing, now[0], hold=play._hold)
    finally:
        THE_GUIDE.reset(token)
    assert not actions


def test_changing_coordinate_basis_discards_the_old_physical_forecast(monkeypatch):
    _now, run, hers, mine, moves, _choosing, _actions, _hands, _guide = _setup(monkeypatch)
    previous = run.faces.steering_model
    mine.motion_position, mine.motion_measured_at, mine.seen = (151., 151.), .4, .4
    observed_steering_model(run.faces, mine, hers, moves, .4)
    assert run.faces.steering_model is not previous


def test_a_numeric_capture_receipt_can_date_the_owned_bodys_birth(monkeypatch):
    from core.perception.a_clock_for_drawn_frames import MeasuredFrameTime

    _now, run, hers, mine, moves, _choosing, _actions, _hands, _guide = _setup(monkeypatch)
    mine.born = MeasuredFrameTime(0., lower=0., upper=.001, arrived=.01,
                                 source='browser-wall', source_at=1000., epoch=1)
    observed_steering_model(run.faces, mine, hers, moves, .3)
    assert run.faces.steering_model.owner == (1, 0., hers.identification.epoch)
    mine.born = False
    observed_steering_model(run.faces, mine, hers, moves, .4)
    assert run.faces.steering_model.owner is None


@pytest.mark.asyncio
async def test_an_observed_contradiction_ends_the_current_forecast(monkeypatch):
    now, run, hers, _mine, _moves, choosing, actions, hands, guide = _setup(monkeypatch)
    token = THE_GUIDE.set(guide)
    try:
        assert await steer_with_a_model(hands, run, hers, choosing, now[0], hold=play._hold)
        run.faces.steering_model.pending = None
        run.faces.steering_model.last_result = 'contradicted'
        now[0] += .05
        assert await steer_with_a_model(hands, run, hers, choosing, now[0], hold=play._hold)
    finally:
        THE_GUIDE.reset(token)
    assert actions[-1] == ('up', actions[1][1]) and not run.held


@pytest.mark.asyncio
async def test_a_stale_picture_cannot_extend_the_physical_input_window(monkeypatch):
    now, run, hers, _mine, _moves, choosing, actions, hands, guide = _setup(monkeypatch)
    token = THE_GUIDE.set(guide)
    try:
        assert await steer_with_a_model(hands, run, hers, choosing, now[0], hold=play._hold)
        now[0] += 1.
        assert await steer_with_a_model(hands, run, hers, choosing, .3, hold=play._hold)
    finally:
        THE_GUIDE.reset(token)
    assert actions[-1] == ('up', actions[1][1]) and not run.held


def test_the_provisional_input_boundary_has_a_registered_measured_claim():
    from core.agency.steering_from_a_model import _provisional_steering_input_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _provisional_steering_input_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'provisional_steering_keeps_input_ownership')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())


@pytest.mark.asyncio
async def test_a_provisional_driver_action_does_not_extend_an_independent_input_trial(monkeypatch):
    now, run, hers, _mine, _moves, choosing, actions, hands, guide = _setup(monkeypatch)
    choosing.target = lambda: ((240., 150.), 'meet', None)
    token = THE_GUIDE.set(guide)
    try:
        assert await steer_with_a_model(hands, run, hers, choosing, now[0], hold=play._hold)
    finally:
        THE_GUIDE.reset(token)
    assert actions == [('up', 'opaque-drive'), ('down', 'opaque-drive')]
    assert not run.held_trying and hers._held[-1][2] is False


@pytest.mark.asyncio
async def test_a_model_can_request_and_deliver_its_neutral_comparison(monkeypatch):
    from core.agency.a_model_for_steering import RateEstimate

    now, run, hers, _mine, _moves, choosing, actions, hands, guide = _setup(monkeypatch)
    run.faces.steering_model.estimates['opaque-a'] = RateEstimate(1., .1, 3, 0.)
    token = THE_GUIDE.set(guide)
    try:
        assert await steer_with_a_model(hands, run, hers, choosing, now[0], hold=play._hold)
    finally:
        THE_GUIDE.reset(token)
    assert actions == [('up', 'opaque-drive')]
    assert not run.held and not run.held_trying
    assert run._modelled_steering[0].key == ''


@pytest.mark.asyncio
@pytest.mark.parametrize('neutral', [False, True])
async def test_actor_expiry_keeps_the_forecasts_pre_release_feedback(monkeypatch, neutral):
    from core.agency.a_model_for_steering import RateEstimate

    now, run, hers, _mine, _moves, choosing, actions, hands, guide = _setup(monkeypatch)
    model = run.faces.steering_model
    if neutral:
        model.estimates['opaque-a'] = RateEstimate(1., .1, 3, 0.)
    token = THE_GUIDE.set(guide)
    try:
        assert await steer_with_a_model(hands, run, hers, choosing, now[0], hold=play._hold)
        action, _until, since, _model = run._modelled_steering
        now[0] = .4
        model.saw(action.owner, action.key, since, .4, 156., 150., 60., 0., position_error=.01)
        assert model.pending is action and model.last_result == 'partial consistent'
        now[0] = .55
        assert await steer_with_a_model(hands, run, hers, choosing, now[0], hold=play._hold)
    finally:
        THE_GUIDE.reset(token)
    assert run._modelled_steering is None and not run.held and model.pending is None
    assert model.last_result == 'partial consistent' and model.covered_span == pytest.approx((.3, .4))
    assert sum(kind == 'up' for kind, _key in actions) == (1 if neutral else 2)


@pytest.mark.asyncio
async def test_an_old_forecast_cannot_release_a_new_press_of_the_same_control(monkeypatch):
    now, run, hers, mine, _moves, choosing, actions, hands, guide = _setup(monkeypatch)
    token = THE_GUIDE.set(guide)
    try:
        assert await steer_with_a_model(hands, run, hers, choosing, now[0], hold=play._hold)
        key = run.held
        now[0] = .31
        await play._hold(hands, run, hers, '', now[0])
        now[0] = .32
        await play._hold(hands, run, hers, key, now[0], trying=True)
        before = list(actions)
        # The next sighting has no current motion on which to plan another input.
        now[0] = .33
        mine.seen = .3
        assert not await steer_with_a_model(hands, run, hers, choosing, now[0], hold=play._hold)
    finally:
        THE_GUIDE.reset(token)
    assert actions == before and run.held == key and run.held_since == .32
    assert run._modelled_steering is None


def _source_guide(by, *, owner='Actuator', where='Controls.update', does='moves'):
    from core.perception.how_a_program_plays import AKey, HowItPlays

    source = HowItPlays(hers={'class':owner}, keys={'opaque-a':AKey('opaque-a', does, by, where)})
    return SimpleNamespace(controls={'opaque-a':SimpleNamespace(act='move', on='')}, code_says=source)


@pytest.mark.parametrize('method', ['mTurn', 'turn_left', 'RotateClockwise', 'setRotation', 'mTURN'])
def test_resolved_source_body_rotation_survives_a_generic_move_role(method):
    guide = _source_guide(f'Actuator.{method}(1)')
    run = SimpleNamespace(keys=['opaque-a', 'opaque-b'])
    faces = SimpleNamespace(travel_turns=SimpleNamespace(curved=lambda _key: False))
    token = THE_GUIDE.set(guide)
    try:
        assert _turn_keys(run, faces) == {'opaque-a'}
    finally:
        THE_GUIDE.reset(token)


@pytest.mark.parametrize('by', [
    'Actuator.turnOn()', 'Actuator.turnOff()', 'Actuator.turnOnMusic()', 'Actuator.rotateAudio()',
    'Actuator.turnButton()', 'Actuator.turnFurniture()', 'Actuator.getRotation()', 'Actuator.checkTurn()',
    'Actuator.returnLeft()', 'Furniture.rotateLeft()', 'Buttons.turnRight()', 'Music.rotatePitch()',
    'rotateRight()', 'r17.turnLeft()', 'Actuator.moveLeft()',
])
def test_source_switches_other_objects_and_unresolved_calls_do_not_arm_body_turns(by):
    guide = _source_guide(by)
    run = SimpleNamespace(keys=['opaque-a'])
    faces = SimpleNamespace(travel_turns=SimpleNamespace(curved=lambda _key: False))
    token = THE_GUIDE.set(guide)
    try:
        assert not _turn_keys(run, faces)
    finally:
        THE_GUIDE.reset(token)


@pytest.mark.parametrize('owner,does', [(None, 'moves'), ('Elsewhere', 'moves'), ('Actuator', 'does something')])
def test_source_rotation_requires_a_resolved_controlled_class_and_movement_role(owner, does):
    guide = _source_guide('Actuator.turnRight(1)', owner=owner, does=does)
    run = SimpleNamespace(keys=['opaque-a'])
    faces = SimpleNamespace(travel_turns=SimpleNamespace(curved=lambda _key: False))
    token = THE_GUIDE.set(guide)
    try:
        assert not _turn_keys(run, faces)
    finally:
        THE_GUIDE.reset(token)


@pytest.mark.parametrize('receiver', ['this', 'self'])
@pytest.mark.parametrize('where,expected', [('Actuator.update', {'opaque-a'}), ('Other.update', set()),
                                          ('Actuator.Decoration.update', set())])
def test_a_local_rotation_call_keeps_its_lexical_class_owner(receiver, where, expected):
    guide = _source_guide(f'{receiver}.rotateLeft(1)', where=where)
    run = SimpleNamespace(keys=['opaque-a'])
    faces = SimpleNamespace(travel_turns=SimpleNamespace(curved=lambda _key: False))
    token = THE_GUIDE.set(guide)
    try:
        assert _turn_keys(run, faces) == expected
    finally:
        THE_GUIDE.reset(token)


def test_source_roles_preserve_explicit_instructions_and_observed_curvature():
    guide = _source_guide('Furniture.rotateLeft()')
    guide.controls['opaque-a'].act = 'turn left'
    run = SimpleNamespace(keys=['opaque-a', 'opaque-b'])
    faces = SimpleNamespace(travel_turns=SimpleNamespace(curved=lambda key: key == 'opaque-b'))
    token = THE_GUIDE.set(guide)
    try:
        assert _turn_keys(run, faces) == {'opaque-a', 'opaque-b'}
    finally:
        THE_GUIDE.reset(token)


@pytest.mark.asyncio
async def test_the_actor_uses_source_rotation_as_a_provisional_hypothesis(monkeypatch):
    now, run, hers, _mine, _moves, choosing, actions, hands, _guide = _setup(monkeypatch)
    guide = _source_guide('Actuator.mTurn(1)')
    before = hers.number, hers.kind, hers.identification.epoch
    token = THE_GUIDE.set(guide)
    try:
        assert await steer_with_a_model(hands, run, hers, choosing, now[0], hold=play._hold)
    finally:
        THE_GUIDE.reset(token)
    assert actions == [('up', 'opaque-drive'), ('down', 'opaque-a')]
    assert run._modelled_steering[0].status == 'provisional'
    assert not run.faces.turns_her() and not run.faces.turning_keys()
    assert (hers.number, hers.kind, hers.identification.epoch) == before


def test_source_turn_role_invariant():
    from core.agency.steering_from_a_model import _source_turn_role_invariant

    assert _source_turn_role_invariant() == ()
