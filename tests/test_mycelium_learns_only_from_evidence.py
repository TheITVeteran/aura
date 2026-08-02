"""Routing may not become permanently confident on "nothing threw".

The live defect: the response lane passed ``success=True`` immediately
after a tool call that had merely not raised, so a tool returning a
FAILURE result still strengthened the pathway that chose it. Confidence
then rose, was written to the vault, and outlived the process — the route
became durably more trusted for producing a failed outcome.

The fix is scope, not silence. An unverified outcome still steers this
session; it just cannot rewrite what Aura believes tomorrow.
"""
from __future__ import annotations

import pytest

from core.mycelium import HardwiredPathway, MycelialNetwork


@pytest.fixture
def network():
    # The network is a singleton and reinforce() delegates to whichever
    # instance owns the topology. Without this reset the fixture is not the
    # owner, its session ledger is not the one being written, and the
    # assertions below read a different object than the code under test.
    MycelialNetwork._instance = None
    MycelialNetwork._initialized = False
    net = MycelialNetwork()
    # register_pathway, not a raw dict insert: matching walks _pathway_order,
    # so a pathway poked straight into .pathways is never a routing candidate
    # and every assertion about routing would pass vacuously.
    net.register_pathway("route-1", r"^do the thing$", "do_thing")
    net.pathways["route-1"].confidence = 0.5
    return net


def test_nothing_threw_does_not_move_persisted_confidence(network):
    before = network.pathways["route-1"].confidence
    for _ in range(20):
        network.reinforce("route-1", True)  # no evidence: the live bug
    assert network.pathways["route-1"].confidence == before, (
        "twenty unverified successes rewrote durable routing confidence"
    )


def test_but_it_still_steers_this_session(network):
    """Scope, not silence. Unverified evidence is useful, not believed."""
    for _ in range(5):
        network.reinforce("route-1", True)
    assert network.session_confidence_delta("route-1") > 0
    assert network.effective_confidence("route-1") > network.pathways["route-1"].confidence


def test_a_tool_returning_failure_never_strengthens_the_route_that_chose_it(network):
    """The exact live case: success=True alongside a failing result."""
    before = network.pathways["route-1"].confidence
    for _ in range(10):
        network.reinforce("route-1", True, evidence={"ok": False})
    assert network.pathways["route-1"].confidence <= before


def test_a_checked_success_does_persist(network):
    """The control: verified learning must still work, or this is a freeze."""
    before = network.pathways["route-1"].confidence
    network.reinforce(
        "route-1",
        True,
        evidence={
            "ok": True,
            "verification_grade": "postcondition_verified",
            "evidence_id": "ev-1",
        },
    )
    assert network.pathways["route-1"].confidence > before


def test_an_observed_failure_weakens_durably_and_immediately(network):
    """A broken route must not stay alive waiting for stronger proof."""
    before = network.pathways["route-1"].confidence
    network.reinforce("route-1", False, evidence={"ok": False, "evidence_id": "ev-2"})
    assert network.pathways["route-1"].confidence < before


def test_an_unverified_failure_still_makes_the_route_unattractive_now(network):
    """Session weakening preserves the safety property without persisting."""
    network.reinforce("route-1", False)
    assert network.effective_confidence("route-1") < network.pathways["route-1"].confidence


def test_session_confidence_is_never_written_into_the_vault(network):
    """The boundary is structural: session state is not a Pathway field."""
    network.reinforce("route-1", True)
    assert network.session_confidence_delta("route-1") != 0.0
    dumped = network.pathways["route-1"].to_dict()
    assert "session_confidence" not in dumped
    assert not any("session" in key for key in dumped), (
        "session state leaked into the serialized pathway; after a restart it "
        "would be indistinguishable from verified confidence"
    )


def test_the_unverified_tally_still_moves_so_the_record_stays_honest(network):
    """An operator must be able to see confidence resting on assertions."""
    before = network.pathways["route-1"].unverified_reinforcements
    network.reinforce("route-1", True)
    assert network.pathways["route-1"].unverified_reinforcements == before + 1
    assert network.pathways["route-1"].evidence_grade in {"untested", "asserted_only"}


def test_routing_skips_a_route_this_session_proved_bad(network):
    """Effective confidence, not durable, decides whether a route is offered."""
    network.pathways["route-1"].confidence = HardwiredPathway.MIN_CONFIDENCE + 0.01
    assert network.match_hardwired("do the thing") is not None
    for _ in range(3):
        network.reinforce("route-1", False)  # unverified failures, this session
    assert network.match_hardwired("do the thing") is None, (
        "a route that failed repeatedly this session was still offered because "
        "only its persisted confidence was consulted"
    )
