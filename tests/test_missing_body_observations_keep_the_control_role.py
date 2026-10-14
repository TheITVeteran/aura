"""Missing visual identity cannot substitute an unproved timing-only control role."""
from types import SimpleNamespace

import pytest

from core.agency.a_body_still_expected_when_it_is_unseen import (
    _expected_control_invariant,
    body_control_is_still_expected,
)
from core.agency.how_a_game_works import (
    AIMS_FROM_A_BODY,
    AIMS_FROM_HERE,
    CODED_SURE,
    FOLLOWS,
    MAKES,
    SAID_SURE,
    STEERED,
    TIMES,
    Answer,
    HowItWorks,
)


@pytest.mark.parametrize('role', [STEERED, AIMS_FROM_A_BODY, FOLLOWS])
def test_established_body_control_survives_unavailable_body_measurement(role):
    how = HowItWorks(me=Answer(role, SAID_SURE))
    assert body_control_is_still_expected(how, timing_pays=False)
    assert how.me.value == role
    assert not body_control_is_still_expected(how, timing_pays=True)


@pytest.mark.parametrize('role', [TIMES, AIMS_FROM_HERE, MAKES])
def test_a_stronger_other_role_outweighs_incidental_coded_body_keys(role):
    how = HowItWorks(me=Answer(role, SAID_SURE), body_keys_from_code=('opaque',))
    assert not body_control_is_still_expected(how, timing_pays=False)


def test_an_unsupported_guess_does_not_block_timing_but_bound_body_inputs_qualify_discovery():
    how = HowItWorks(me=Answer(STEERED, CODED_SURE))
    assert not body_control_is_still_expected(how, timing_pays=False)
    how.body_keys_from_code = ('opaque',)
    assert body_control_is_still_expected(how, timing_pays=False)
    assert not body_control_is_still_expected(None, timing_pays=False)
    assert _expected_control_invariant() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize('role,paid,expected', [(STEERED, False, False), (TIMES, False, True),
                                             ('', False, True), (STEERED, True, True)])
async def test_the_actual_timing_wait_cannot_postpone_body_rediscovery(role, paid, expected):
    from core.agency.playing_as_it_happens import _press_in_time, _Run

    run = _Run(keys=['opaque'], began=1., last_moving=1.)
    run.how = HowItWorks(me=Answer(role, SAID_SURE))
    run.timing = SimpleNamespace(pays=lambda: paid, press_now=lambda *_args: False,
                                 has_something_to_time=lambda *_args: True)
    assert await _press_in_time(None, run, SimpleNamespace(things={}), 2., None) is expected


def test_expected_body_control_has_a_registered_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'missing_body_observations_keep_the_control_role')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())
