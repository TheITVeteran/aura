"""Prior controls can request measurements after a track changes, without lending its heading."""
from types import SimpleNamespace

import pytest

from core.agency.facing_seen_under_its_input import observe_facing
from core.agency.which_one_answers_to_her import WhichIsHers
from core.agency.which_way_she_faces import WhichWaySheFaces

pytestmark = pytest.mark.unit


def _waiting_body():
    hers, faces = WhichIsHers(), WhichWaySheFaces()
    hers.number, hers.kind = 2, 1
    hers.holding('', 1.)
    for key, velocity in (('drive', (80., 0.)), ('turn-a', (0., 0.)), ('turn-b', (0., 0.))):
        for repeat in range(2):
            for sample in range(4):
                hers._hers.add(key, *velocity, press=-10. + repeat, when=-9.7 + repeat + .05 * sample)
    body = SimpleNamespace(number=2, vx=0., vy=0., seen=1.3, motion_evidence='direct')
    observe_facing(faces, body, hers, 1.3)
    return hers, faces, body


def test_a_stopped_reidentified_body_requests_its_first_observed_direction():
    _hers, faces, _body = _waiting_body()
    assert not faces.along and faces.facing is None and not faces.turning_keys()
    assert faces.trial(['turn-a', 'drive', 'turn-b']) == 'drive'
    assert faces.facing is None and not faces.along and faces.ahead_key() is None


def test_requested_driver_measurements_end_with_a_rest_and_need_current_motion():
    _hers, faces, _body = _waiting_body()
    assert faces.trial(['drive', 'turn-a', 'turn-b']) == 'drive'
    for at in (1.4, 1.65, 1.7, 1.8):
        faces.saw(80., 0., 'drive', at, began=1.4)
    assert faces.trial(['drive', 'turn-a', 'turn-b']) == ''
    faces.saw(0., 0., '', 2., began=2.)
    assert faces.ahead_key(least=1) == 'drive'
    assert faces.trial(['drive', 'turn-a', 'turn-b']) == 'drive'


def test_missing_motion_cannot_make_bootstrap_trials_repeat_without_a_bound():
    _hers, faces, _body = _waiting_body()
    requests = []
    for step in range(12):
        key = faces.trial(['drive', 'turn-a', 'turn-b'])
        requests.append(key)
        if key is not None:
            faces.saw(0., 0., key, 2. + step, began=2. + step, measured=False)
    assert requests.count('drive') == 2 and requests[-1] is None
    assert not faces.along and faces.facing is None


def test_hints_cannot_add_a_key_that_the_current_environment_does_not_offer():
    _hers, faces, _body = _waiting_body()
    assert faces.trial(['different']) is None


@pytest.mark.parametrize('departure', [None, 'single', 'interrupted', 'clock'])
def test_opposed_trial_directions_request_fresh_motion_without_declaring_a_heading(departure):
    hers, faces = WhichIsHers(), WhichWaySheFaces()
    hers.number, hers.kind = 2, 1
    hers.holding('', 1.)
    evidence = hers._by_thing[2]
    for index, began in enumerate((-3., -2.)):
        for elapsed in (.25, .3, .35, .4):
            evidence.add('opaque', 80. * (-1 if index else 1), 0., press=began, when=began + elapsed)
    if departure == 'single':
        evidence.timed.pop(('opaque', -2.))
    elif departure == 'interrupted':
        evidence.interrupted.add(('opaque', -2.))
    elif departure == 'clock':
        values = evidence.timed['opaque', -2.]
        values[-1] = (values[0][0], *values[-1][1:])
    body = SimpleNamespace(number=2, vx=0., vy=0., seen=1.3, motion_evidence='direct')
    observe_facing(faces, body, hers, 1.3)
    assert faces.trial(['opaque']) == ('opaque' if departure is None else None)
    assert faces.facing is None and not faces.along and not faces.turning_keys()


def test_the_fresh_heading_experiment_has_a_registered_measured_claim():
    from core.agency.facing_seen_under_its_input import _fresh_heading_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _fresh_heading_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'a_new_heading_requires_a_current_direction_experiment')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
