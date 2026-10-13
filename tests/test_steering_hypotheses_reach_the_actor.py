"""A provisional physical forecast reaches real input delivery before facing calibration."""
from types import SimpleNamespace

import pytest

from core.agency import playing_as_it_happens as play
from core.agency.steering_from_a_model import observed_steering_model, steer_with_a_model
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
