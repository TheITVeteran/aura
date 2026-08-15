"""Forming an autonomous goal: one object, one durable home, one review.

`add_goal` cloned the caller's dict, overwrote the clone's status to "pending",
and left the caller holding the original — goal genesis then sent that
original, still marked "incubating", to identity. One goal existed as two
objects disagreeing about its lifecycle state.

Its docstring promised "a persistent goal that survives across conversations"
and it appended to a Pydantic list on AgencyState: no write, no reload, no
crash-recovery contract.

And when the moral review raised, the code logged "continuing with caution" and
created the durable goal anyway, so a review that could not run was
indistinguishable from one that approved.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.agency.agency_core import _MAX_PENDING_GOALS, AgencyCore, AgencyState
from core.container import ServiceContainer


class IdentityProbe:
    def __init__(self, disposition: str = "saved"):
        self.goals: list[dict] = []
        self._disposition = disposition

    def add_long_term_goal(self, goal, *, source="identity", importance=0.75):
        self.goals.append(goal)
        return self._disposition

    @staticmethod
    def score_goal(_text):
        return 0.5


def _core() -> AgencyCore:
    core = AgencyCore.__new__(AgencyCore)
    core.state = AgencyState()
    core._approve_agency_state_mutation = lambda **_kw: True
    core._coerce_priority = staticmethod(lambda v, default=0.6: float(v))
    return core


@pytest.fixture(autouse=True)
def _clean_container():
    ServiceContainer.clear()
    yield
    ServiceContainer.clear()


def test_the_stored_goal_and_the_callers_goal_are_the_same_object():
    identity = IdentityProbe()
    ServiceContainer.register_instance("identity_service", identity, required=False)
    core = _core()
    goal = {"id": "g1", "text": "Mastery of: x", "status": "incubating", "priority": 0.6}

    assert core.add_goal(goal) is True

    assert core.state.pending_goals[-1] is goal
    assert identity.goals[-1] is goal
    assert goal["status"] == "pending", "the caller kept a stale lifecycle state"


def test_a_goal_reaches_the_durable_store():
    identity = IdentityProbe()
    ServiceContainer.register_instance("identity_service", identity, required=False)
    core = _core()
    goal = {"id": "g1", "text": "Mastery of: x", "priority": 0.6}

    core.add_goal(goal)

    assert identity.goals == [goal]
    assert goal["durable"] is True


def test_a_goal_that_could_not_be_made_durable_says_so():
    """A working set that cannot be persisted is a memo, and it should not
    claim otherwise."""
    core = _core()
    goal = {"id": "g1", "text": "Mastery of: x", "priority": 0.6}

    assert core.add_goal(goal) is True
    assert goal["durable"] is False


def test_a_refused_durable_write_is_not_reported_as_durable():
    identity = IdentityProbe(disposition="denied")
    ServiceContainer.register_instance("identity_service", identity, required=False)
    core = _core()
    goal = {"id": "g1", "text": "Mastery of: x", "priority": 0.6}

    core.add_goal(goal)

    assert goal["durable"] is False


def test_the_working_set_ceiling_is_named_and_enforced():
    core = _core()
    core.state.pending_goals = [{"text": f"g{i}"} for i in range(_MAX_PENDING_GOALS)]

    assert core.add_goal({"text": "one more"}) is False


def test_an_unavailable_moral_review_forms_no_goal(monkeypatch):
    """"Continuing with caution" created the goal anyway. The caution was in
    the log message."""
    import core.morality.moral_reasoning as moral_module

    def _broken():
        raise RuntimeError("moral reasoning unavailable")

    monkeypatch.setattr(moral_module, "get_moral_reasoning", _broken)

    identity = IdentityProbe()
    ServiceContainer.register_instance("identity_service", identity, required=False)

    core = _core()
    core.orch = None
    core.swarm = None
    core.state.curiosity_pressure = 0.95
    core.state.last_goal_genesis_time = 0.0
    core._approve_agency_state_mutation = lambda **_kw: True

    result = asyncio.run(
        AgencyCore._pathway_goal_genesis(core, now=10_000.0, idle_seconds=1200.0)
    )

    assert result is not None
    assert result["type"] == "deferred_goal"
    assert result["source"] == "goal_genesis_audit_unavailable"
    assert core.state.pending_goals == [], "a durable goal was formed without a review"
    assert identity.goals == []


def test_the_genesis_cooldown_does_not_advance_on_a_deferral(monkeypatch):
    """Deferring must cost one cycle, not the full 600s window — otherwise a
    broken audit silently suppresses goal formation for ten minutes at a time.
    """
    import core.morality.moral_reasoning as moral_module

    monkeypatch.setattr(
        moral_module,
        "get_moral_reasoning",
        lambda: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )

    core = _core()
    core.orch = None
    core.swarm = None
    core.state.curiosity_pressure = 0.95
    core.state.last_goal_genesis_time = 0.0

    asyncio.run(AgencyCore._pathway_goal_genesis(core, now=10_000.0, idle_seconds=1200.0))

    assert core.state.last_goal_genesis_time == 0.0
