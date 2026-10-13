"""A visual classification changes without creating a new controlled body."""
import math
from types import SimpleNamespace

import pytest

from core.agency import playing_as_it_happens as play
from core.agency.facing_seen_under_its_input import _heading_lineage_invariant, observe_facing
from core.agency.response_in_the_body_frame import bound_facing_response
from core.agency.steering_from_a_model import observed_steering_model
from core.agency.which_one_answers_to_her import WhichIsHers
from core.agency.which_way_she_faces import WhichWaySheFaces
from core.cognition.a_guide_to_a_place import THE_GUIDE

pytestmark = pytest.mark.unit


def _body():
    hers, faces = WhichIsHers(), WhichWaySheFaces()
    hers.number, hers.kind = 1, 0
    hers.holding('drive', 0., trying=True)
    body = SimpleNamespace(number=1, kind=0, born=0., seen=.3, x=100., y=100.,
                           vx=80., vy=0., w=12., h=12., motion_evidence='direct')
    observe_facing(faces, body, hers, .3)
    faces.along['drive'] = [80., 80.]
    return hers, faces, body


def test_current_visual_reclassification_keeps_the_continuing_body_response():
    hers, faces, body = _body()
    along = faces.along['drive']
    for index in range(6):
        hers.kind = body.kind = index + 1
        body.seen = .4 + index * .1
        assert bound_facing_response(hers) is None
        observe_facing(faces, body, hers, body.seen)
        assert bound_facing_response(hers) is faces
        assert faces.along['drive'] is along and along == [80., 80.]
        assert faces.body_birth == 0. and faces.facing == pytest.approx(0.)


@pytest.mark.parametrize('change', ('birth', 'epoch', 'number'))
def test_a_replacement_or_new_experiment_cannot_inherit_the_heading(change):
    hers, faces, body = _body()
    if change == 'birth':
        body.born = .35
    elif change == 'epoch':
        hers.identification.reset('new input experiment')
    else:
        hers.number = body.number = 2
    body.seen = .4
    observe_facing(faces, body, hers, .4)
    assert not faces.along and not faces.turns and not faces.travel_turns.trials


@pytest.mark.parametrize('birth', (False, math.nan, math.inf, '0'))
def test_a_current_invalid_birth_cannot_preserve_an_earlier_owned_model(birth):
    hers, faces, body = _body()
    body.born, body.seen = birth, .4
    hers.kind += 1
    observe_facing(faces, body, hers, .4)
    assert faces.body_birth is None and not faces.along


def test_a_missing_or_stale_observation_retains_history_without_teaching_motion():
    hers, faces, body = _body()
    for mine, at in ((None, .4), (body, .5)):
        observe_facing(faces, mine, hers, at)
        assert faces.along['drive'] == [80., 80.]
        assert faces.body_birth == 0. and not faces.going
        assert not faces.turns_measured
    body.seen = .6
    hers.kind = body.kind = 2
    observe_facing(faces, body, hers, .6)
    assert faces.along['drive'] == [80., 80.] and len(faces.going) == 1


def test_unmeasured_motion_during_a_changed_look_supplies_no_new_direction():
    hers, faces, body = _body()
    hers.kind = body.kind = 2
    body.seen, body.motion_evidence = .4, 'unavailable'
    body.motion_measured_at = None
    body.vx, body.vy = 0., 80.
    observe_facing(faces, body, hers, .4)
    assert faces.along['drive'] == [80., 80.]
    assert faces.facing == pytest.approx(0.) and not faces.going


def test_without_a_continuing_birth_the_earlier_kind_boundary_remains_in_force():
    hers, faces, body = _body()
    del body.born
    body.seen = .4
    observe_facing(faces, body, hers, .4)
    faces.along['drive'] = [80., 80.]
    hers.kind += 1
    body.seen = .5
    observe_facing(faces, body, hers, .5)
    assert not faces.along


def test_an_absent_body_description_cannot_feed_or_arm_a_steering_model():
    hers, faces, body = _body()
    observed_steering_model(faces, body, hers, SimpleNamespace(), .3)
    assert faces.steering_model.owner is not None
    observed_steering_model(faces, object(), hers, SimpleNamespace(), .4, run=SimpleNamespace())
    assert faces.steering_model.owner is None and faces.steering_model._pose is None


@pytest.mark.asyncio
async def test_observed_slow_motion_survives_changing_looks_and_reaches_the_actor(monkeypatch):
    hers, _earlier, body = _body()
    faces = WhichWaySheFaces()
    now = [0.]
    monkeypatch.setattr(play.time, 'monotonic', lambda: now[0])
    run = play._Run(keys=['drive', 'bend_a', 'bend_b'], began=0., faces=faces,
                    held='drive', held_since=0., held_trying=True, trying=12)
    moves = SimpleNamespace(shape=(300, 300), things={1: body}, seeing=True)
    choosing = SimpleNamespace(mine=body, moves=moves, next_picture_s=.1, response_s=0.,
                               way_through=None, target=lambda: ((100., 200.), 'meet', None),
                               ways={}, lifts=[])
    guide = SimpleNamespace(controls={key: SimpleNamespace(act=action, on='')
                                      for key, action in (('bend_a', 'turn left'), ('bend_b', 'turn right'))})
    actions = []

    class Hands:
        async def down(self, key):
            actions.append(('down', key))

        async def up(self, key):
            actions.append(('up', key))

    token = THE_GUIDE.set(guide)
    try:
        model = None
        for index in range(14):
            now[0] = body.seen = index * .1
            hers.kind = body.kind = index % 3
            body.x, body.vx = 100. + 4. * now[0], 4.
            observe_facing(faces, body, hers, now[0])
            observed_steering_model(faces, body, hers, moves, now[0], run=run)
            if model is None:
                model = faces.steering_model
            assert faces.steering_model is model
        assert faces.ahead_key(least=1) is None
        assert model.driver(run.keys, now=now[0]) == 'drive'
        await play._act(Hands(), run, moves, hers, None, choosing, now[0])
    finally:
        THE_GUIDE.reset(token)
    assert run._modelled_steering is not None
    assert actions[0] == ('up', 'drive') and actions[1][0] == 'down'
    assert not faces.turns_her() and not run.held_trying


def test_the_heading_lineage_boundary_has_a_registered_measurement():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _heading_lineage_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'a_continuing_body_keeps_heading_when_its_look_changes')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
