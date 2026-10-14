"""Curved travel retains speed; autonomous curves do not acquire control identity."""
import math
from collections import defaultdict

import pytest

from core.agency.an_input_answers_relative_to_motion import (
    _travel_speed_invariant,
    response_by_travel_speed,
)
from core.agency.causal_identification import CausalIdentification
from core.agency.controls_measured_on_one_thing import own_control_witness
from core.agency.which_one_answers_to_her import _contingency, _Speeds


def _curves(*, autonomous=False, copies=3):
    evidence = _Speeds()
    for index in range(2 * copies):
        key = 'opaque' if index % 2 == 0 else ''
        began, heading = 1. + index, index * 1.1
        speed = 90. if autonomous or key else 0.
        for step in range(8):
            angle = heading + step * math.pi / 4
            position = (30. + 8. * math.sin(angle), 30. - 8. * math.cos(angle))
            evidence.add(key, speed * math.cos(angle), speed * math.sin(angle),
                         press=began, when=began + .25 + .05 * step, position=position)
    return evidence


def test_closed_travel_supplies_a_repeated_witness_when_screen_vectors_cancel():
    evidence = _curves()
    groups = defaultdict(list)
    for (key, _began), values in evidence.by_press.items():
        groups[key].append(tuple(sum(v[axis] for v in values) / len(values) for axis in (0, 1)))
    assert _contingency(groups)[1] < 1.e-10
    statistic, effect = response_by_travel_speed(evidence)
    assert statistic > 20. and effect == pytest.approx(90.)
    witness = own_control_witness('body', evidence, epoch=2, movement_over=4.)
    assert witness is not None
    ledger = CausalIdentification(statistic_over=20., effect_over=20.)
    ledger.epoch = 2
    assert ledger.choose([witness], visible={'body'}, established=None, excluded=set()) == 'body'


def test_an_autonomous_curve_shared_by_all_inputs_cannot_supply_a_witness():
    evidence = _curves(autonomous=True)
    statistic, effect = response_by_travel_speed(evidence)
    assert statistic < 1. and effect < 1.
    witness = own_control_witness('autonomous', evidence, epoch=0, movement_over=4.)
    ledger = CausalIdentification(statistic_over=20., effect_over=20.)
    assert ledger.choose([witness] if witness else [], visible={'autonomous'}, established=None, excluded=set()) is None


@pytest.mark.parametrize('departure', ['single', 'missing-neutral', 'interrupted', 'clock', 'nonfinite'])
def test_unknown_or_departed_trial_evidence_cannot_supply_speed_credit(departure):
    evidence = _curves(copies=2)
    if departure == 'single':
        evidence.timed.pop(('opaque', 3.))
    elif departure == 'missing-neutral':
        evidence.timed = {press: values for press, values in evidence.timed.items() if press[0]}
    else:
        for press, values in evidence.timed.items():
            if not press[0]:
                continue
            if departure == 'interrupted':
                evidence.interrupted.add(press)
            elif departure == 'clock':
                values[-1] = (values[0][0], *values[-1][1:])
            else:
                values[-1] = (values[-1][0], math.nan, 0.)
    assert response_by_travel_speed(evidence) == (0., 0.)


def test_speed_witness_still_requires_visible_displacement_and_the_current_epoch():
    evidence = _curves()
    evidence.lowest, evidence.highest = [30., 30.], [31., 31.]
    assert own_control_witness('unresolved', evidence, epoch=0, movement_over=4.) is None
    evidence.lowest, evidence.highest = [20., 20.], [40., 40.]
    witness = own_control_witness('body', evidence, epoch=0, movement_over=4.)
    ledger = CausalIdentification(statistic_over=20., effect_over=20.)
    ledger.reset('a different body experiment')
    assert ledger.choose([witness], visible={'body'}, established=None, excluded=set()) is None


def test_travel_speed_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _travel_speed_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'travel_keeps_its_speed_when_it_changes_direction')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())
