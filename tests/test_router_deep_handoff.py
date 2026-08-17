"""The 72B Solver must be unreachable without an explicit deep handoff.

Confirmed by adversarial probe: "The router can invoke the 72B Solver even when
secondary/deep handoff was disabled."

The mechanism is a nice illustration of how a guard becomes a fiction:

  1. DEEP_ENDPOINT ("Solver") is registered on LLMTier.SECONDARY, deliberately,
     with the comment "Moved to SECONDARY to prevent accidental promotion".
  2. Ordinary local secondary endpoints use the same tier.
  3. SECONDARY is included in PRIMARY's local failover chain.
  4. So (1) put the Solver in exactly the tier that (3) always includes. The
     chain came out [Cortex, Solver, Brainstem] and any Cortex failure invoked
     the 72B — the precise outcome (1) intended to prevent.
  5. `allow_secondary` looked like it controlled this. It did nothing: the
     `prefer_tier == PRIMARY and not allow_secondary` and `... and
     allow_secondary` branches produced identical tier lists.

A flag that is read and then ignored is worse than no flag, because callers
believe they disabled something.
"""
from __future__ import annotations

import pytest

from core.brain.llm.llm_router import IntelligentLLMRouter, LLMEndpoint, LLMTier
from core.brain.llm.model_registry import DEEP_ENDPOINT, PRIMARY_ENDPOINT


@pytest.fixture
def router():
    r = IntelligentLLMRouter()
    # Solver and ordinary local fallback lanes can share SECONDARY.
    r.endpoints = {
        PRIMARY_ENDPOINT: LLMEndpoint(name=PRIMARY_ENDPOINT, tier=LLMTier.PRIMARY),
        DEEP_ENDPOINT: LLMEndpoint(name=DEEP_ENDPOINT, tier=LLMTier.SECONDARY),
        "Local-Secondary": LLMEndpoint(name="Local-Secondary", tier=LLMTier.SECONDARY),
        "Brainstem": LLMEndpoint(name="Brainstem", tier=LLMTier.TERTIARY),
        "Static-Reflex": LLMEndpoint(name="Static-Reflex", tier=LLMTier.EMERGENCY),
    }
    return r


def test_solver_is_absent_from_the_chain_without_deep_handoff(router):
    """The headline. Not un-preferred — absent."""
    chain = router._get_ordered_endpoints(LLMTier.PRIMARY, allow_secondary=False)
    assert DEEP_ENDPOINT not in chain, (
        f"the 72B Solver is reachable with deep handoff disabled: {chain}"
    )


def test_solver_is_reachable_with_an_explicit_deep_handoff(router):
    """A disable that cannot be lifted is a removal, not a control."""
    chain = router._get_ordered_endpoints(LLMTier.PRIMARY, allow_secondary=True)
    assert DEEP_ENDPOINT in chain


def test_the_flag_actually_changes_the_chain(router):
    """It used to produce byte-identical chains either way."""
    off = router._get_ordered_endpoints(LLMTier.PRIMARY, allow_secondary=False)
    on = router._get_ordered_endpoints(LLMTier.PRIMARY, allow_secondary=True)
    assert off != on, "allow_secondary is inert — it is read and then ignored"


def test_local_secondary_fallback_survives_the_solver_gate(router):
    """Filtering Solver must preserve ordinary local secondary failover.

    Excluding the whole tier would discard a healthy local fallback; only the
    Solver endpoint is governed by explicit deep handoff.
    """
    chain = router._get_ordered_endpoints(LLMTier.PRIMARY, allow_secondary=False)
    assert "Local-Secondary" in chain, (
        "excluding Solver also removed the ordinary local secondary fallback"
    )
    assert chain.index(PRIMARY_ENDPOINT) < chain.index("Local-Secondary")
    assert chain.index("Local-Secondary") < chain.index("Brainstem")


def test_solver_cannot_be_smuggled_in_as_a_preferred_endpoint(router):
    """prefer_endpoint is prepended before ordering — it must not bypass the gate."""
    chain = router._get_ordered_endpoints(
        LLMTier.PRIMARY, prefer_endpoint=DEEP_ENDPOINT, allow_secondary=False
    )
    assert DEEP_ENDPOINT not in chain, (
        "naming the Solver as prefer_endpoint bypassed the deep-handoff gate"
    )


def test_solver_excluded_from_the_default_chain_too(router):
    """The no-prefer_tier branch also includes SECONDARY."""
    chain = router._get_ordered_endpoints(None, allow_secondary=False)
    assert DEEP_ENDPOINT not in chain


def test_background_work_never_reaches_the_solver(router):
    """Background is TERTIARY-only, and must stay that way."""
    chain = router._get_ordered_endpoints(
        LLMTier.PRIMARY, allow_secondary=False, is_background=True
    )
    assert DEEP_ENDPOINT not in chain


def test_secondary_preference_still_excludes_the_solver_without_handoff(router):
    """Asking for the SECONDARY tier is not the same as asking for the 72B."""
    chain = router._get_ordered_endpoints(LLMTier.SECONDARY, allow_secondary=False)
    assert DEEP_ENDPOINT not in chain, (
        "preferring the SECONDARY tier pulled in the Solver — the tier is mixed "
        "with ordinary local fallback, so tier preference must not imply deep handoff"
    )


def test_guard_redirects_an_explicit_solver_request_without_handoff():
    """Defence in depth: the call-site guard is independent of the chain filter."""
    from core.brain.llm.model_registry import guard_solver_request

    guarded = guard_solver_request(DEEP_ENDPOINT, deep_handoff=False)
    assert guarded["redirected"] is True
    assert guarded["endpoint"] == PRIMARY_ENDPOINT

    allowed = guard_solver_request(DEEP_ENDPOINT, deep_handoff=True)
    assert allowed["redirected"] is False
    assert allowed["endpoint"] == DEEP_ENDPOINT
