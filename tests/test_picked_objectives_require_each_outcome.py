"""The real collection caller keeps a finished pursuit apart from its requested outcome."""
from __future__ import annotations

import asyncio

import pytest

from core.skills import sovereign_browser_picking as picking

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("outcomes", [
    [False, False, False], [True, False, True], [False, True, True],
])
def test_ending_each_pursuit_does_not_complete_each_objective(monkeypatch, outcomes):
    result = _picked(monkeypatch, outcomes)
    assert not result["completed"]
    assert [pick["completed"] for pick in result["picked"]] == outcomes
    assert all(pick["pursuit_completed"] for pick in result["picked"])
    assert [pick["as_asked"] for pick in result["picked"]] == outcomes


def test_the_collection_completes_when_every_objective_is_observed(monkeypatch):
    result = _picked(monkeypatch, [True, True, True])
    assert result["completed"]
    assert all(pick["completed"] for pick in result["picked"])


def test_the_adjacent_invariant_rejects_failed_and_missing_objectives():
    assert picking._picked_outcomes_invariant() == ()


def test_the_registered_claim_is_bounded_to_its_regression():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import Evidence, ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    claim = next(claim for claim in suite.claims() if claim.test == "picked_objectives_require_each_outcome")
    assert claim.evidence == Evidence.MEASURED_SYNTHETIC
    test = next(test for test in suite.tests() if test.name == claim.test)
    assert test.predict(None)


def _picked(monkeypatch, outcomes):
    async def listed(browser):
        return [{"text": f"Activity {number}", "href": f"https://example.org/{number}"}
                for number in range(3)]

    async def itself(skill, browser, url, name, task):
        return url

    class Skill:
        def __init__(self):
            self.outcomes = iter(outcomes)

        async def _safe_browse(self, browser, url):
            return True

        def _say_out_loud(self, *args):
            pass

        async def _handle_pursue(self, browser, url, goal, max_steps, **kwargs):
            achieved = next(self.outcomes)
            return {"ok": True, "completed": True,
                    "steps": [{"won": achieved, "completed": achieved}]}

    monkeypatch.setattr(picking, "the_list_on_the_page", listed)
    monkeypatch.setattr(picking, "_the_item_itself", itself)
    goal = ("Play three of the activities and win each. Number the activities from 0. "
            "Start at 0, take the remainder. Then add 1 and take the remainder again.")
    return asyncio.run(picking.picked_by_the_rule(Skill(), object(), "https://example.org/list", goal, 30))
