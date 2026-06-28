"""Embodied common sense: a naive-physics plausibility gate over claims and plans."""
from __future__ import annotations

import pytest

from core.cognition.embodied_commonsense import EmbodiedCommonSense, get_embodied_commonsense


@pytest.fixture
def cs():
    return EmbodiedCommonSense()


def test_ordinary_plan_is_plausible(cs):
    v = cs.check("open the file, read the config, and write the updated value back")
    assert v.plausible
    assert not v.violations


def test_object_permanence_violation(cs):
    v = cs.check("the data simply vanished into nothing with no cause")
    assert not v.plausible
    assert "permanence" in v.violations


def test_time_order_violation(cs):
    v = cs.check("we will change the past so the bug never happened")
    assert not v.plausible
    assert "time_order" in v.violations


def test_conservation_violation(cs):
    v = cs.check("the device produces infinite energy from nothing via perpetual motion")
    assert not v.plausible
    assert "conservation" in v.violations


def test_single_location_violation(cs):
    v = cs.check("the process is in two places at once on the same machine")
    assert "single_location" in v.violations


def test_plausibility_is_monotone(cs):
    assert cs.plausibility("save the document") > cs.plausibility("undo what already happened in the past")


def test_singleton_stable():
    assert get_embodied_commonsense() is get_embodied_commonsense()


def test_ladder_escalates_physically_impossible_plan():
    # The deliberative tier must not confidently 'handle' an impossible plan — it escalates.
    from core.agency.hierarchical_agency import HierarchicalAgency, AgencyTier, Situation
    agency = HierarchicalAgency(ledger_enabled=False)
    result = agency.dispatch(Situation("create infinite fuel from nothing with perpetual motion"))
    assert AgencyTier.DELIBERATIVE in result.path
    assert result.final_tier > AgencyTier.DELIBERATIVE  # escalated past naive deliberation
