"""Having a faculty and being able to use it are different facts.

Bryan: "A human is capable of searching too but likely won't get confused if
they try to use google with no internet. They would just know 'it won't
work.'... she should logically be able to infer 'Hey, I can't use web search
right now' and stick to that because she can logically separate tool
capability from the external factors needed to use that tool."

Aura had one axis. The registry knows which skills are registered and
enabled; nothing in it knows whether the world outside the process can
currently support them. A missing network therefore looked exactly like a
missing skill, so she would either claim a capability that could not possibly
work or deny one she has.

The second axis is preconditions, probed from the world. The verdict is
COMPOSED — capability AND preconditions — which is what makes it reasoning
rather than phrasing: unplug the network and the conclusion changes by
itself, with nothing anywhere mentioning networks.

The distinction these tests protect is the one a person makes without
effort: the faculty stays intact while the world is broken.
"""

import pytest

import core.conversation.capability_preconditions as preconditions
from core.conversation.capability_condition import (
    CapabilityStanding,
    capability_condition_evidence,
    condition_for,
)

pytestmark = pytest.mark.unit


class _Engine:
    def iter_tool_catalog(self, *, include_inactive: bool = True):
        return [{"name": "web_search", "available": True}]


@pytest.fixture
def offline(monkeypatch):
    preconditions.reset_precondition_cache()
    monkeypatch.setitem(
        preconditions._PROBES,
        "network",
        lambda: preconditions.PreconditionState(
            "network", False, "there is no network connection right now"
        ),
    )
    yield
    preconditions.reset_precondition_cache()


@pytest.fixture
def online(monkeypatch):
    preconditions.reset_precondition_cache()
    monkeypatch.setitem(
        preconditions._PROBES,
        "network",
        lambda: preconditions.PreconditionState("network", True, "there is a network"),
    )
    yield
    preconditions.reset_precondition_cache()


class TestTheTwoAxesCompose:
    def test_with_the_world_intact_it_is_simply_ready(self, online):
        assert condition_for("web_search", capability_engine=_Engine()).standing is (
            CapabilityStanding.READY
        )

    def test_without_the_network_it_is_blocked_not_absent(self, offline):
        condition = condition_for("web_search", capability_engine=_Engine())
        assert condition.standing is CapabilityStanding.BLOCKED_BY_PRECONDITION
        assert condition.standing is not CapabilityStanding.ABSENT

    def test_the_faculty_survives_the_outage(self, offline):
        """She does not forget how to search when the wifi drops."""
        condition = condition_for("web_search", capability_engine=_Engine())
        assert condition.faculty_intact
        assert condition.is_transient
        assert "network" in condition.missing_preconditions

    def test_the_conclusion_changes_with_the_world(self, monkeypatch):
        """The point of composing rather than asserting."""
        engine = _Engine()
        preconditions.reset_precondition_cache()
        monkeypatch.setitem(
            preconditions._PROBES,
            "network",
            lambda: preconditions.PreconditionState("network", True, "there is a network"),
        )
        assert condition_for("web_search", capability_engine=engine).standing is (
            CapabilityStanding.READY
        )
        preconditions.reset_precondition_cache()
        monkeypatch.setitem(
            preconditions._PROBES,
            "network",
            lambda: preconditions.PreconditionState("network", False, "no network"),
        )
        assert condition_for("web_search", capability_engine=engine).standing is (
            CapabilityStanding.BLOCKED_BY_PRECONDITION
        )


class TestWhatSheIsToldAboutIt:
    def test_the_evidence_separates_faculty_from_world(self, offline):
        block = capability_condition_evidence(
            "can you look up the weather", capability_engine=_Engine()
        )
        assert "YOU HAVE THIS, BUT IT CANNOT WORK RIGHT NOW" in block
        assert "The capability is intact" in block
        assert "no network connection" in block

    def test_it_is_still_facts_not_a_script(self, offline):
        block = capability_condition_evidence(
            "can you look up the weather", capability_engine=_Engine()
        )
        assert "your own words" in block
        for canned in ("I can't access", "I'm sorry", "Unfortunately"):
            assert canned not in block


class TestAnUnknownWorldIsNotABrokenOne:
    def test_a_probe_that_cannot_answer_is_not_a_failure(self, monkeypatch):
        """Reporting "there's no network" because a socket raised something
        unexpected is the same confident lie as reporting a missing skill
        because a registry read failed."""
        preconditions.reset_precondition_cache()
        monkeypatch.setitem(
            preconditions._PROBES,
            "network",
            lambda: preconditions.PreconditionState(
                "network", False, "unknown", unknown=True
            ),
        )
        assert preconditions.failing_preconditions("web_search") == ()
        assert condition_for("web_search", capability_engine=_Engine()).standing is (
            CapabilityStanding.READY
        )
        preconditions.reset_precondition_cache()


class TestPreconditionsAreDeclared:
    def test_a_capability_declares_what_the_world_must_provide(self):
        assert "network" in preconditions.declared_preconditions("web_search")
        assert "desktop_session" in preconditions.declared_preconditions("computer_use")
        assert preconditions.declared_preconditions("some_unknown_skill") == ()
