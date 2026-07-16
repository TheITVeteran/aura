"""The 72B Solver must be unreachable without an explicit deep handoff.

Confirmed by adversarial probe: "The router can invoke the 72B Solver even when
secondary/deep handoff was disabled."

The mechanism is a nice illustration of how a guard becomes a fiction:

  1. DEEP_ENDPOINT ("Solver") is registered on LLMTier.SECONDARY, deliberately,
     with the comment "Moved to SECONDARY to prevent accidental promotion".
  2. The Gemini cloud endpoints are also on SECONDARY.
  3. SECONDARY is included in PRIMARY's failover chain ON PURPOSE, so the 32B
     degrades to Gemini instead of dropping to the 7B brainstem.
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
    # The real lane layout: Solver AND the cloud endpoints share SECONDARY.
    r.endpoints = {
        PRIMARY_ENDPOINT: LLMEndpoint(name=PRIMARY_ENDPOINT, tier=LLMTier.PRIMARY, url="x"),
        DEEP_ENDPOINT: LLMEndpoint(name=DEEP_ENDPOINT, tier=LLMTier.SECONDARY, url="x"),
        "Gemini-Fast": LLMEndpoint(name="Gemini-Fast", tier=LLMTier.SECONDARY, url="x"),
        "Brainstem": LLMEndpoint(name="Brainstem", tier=LLMTier.TERTIARY, url="x"),
        "Static-Reflex": LLMEndpoint(name="Static-Reflex", tier=LLMTier.EMERGENCY, url="x"),
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


def test_the_cloud_fallback_survives_the_fix(router):
    """The fix must not re-break what the chain was designed for.

    SECONDARY is in PRIMARY's failover chain so the 32B degrades to Gemini
    instead of dropping to the 7B brainstem. Excluding the whole tier would
    have traded one defect for another; only the Solver endpoint is excluded.
    """
    chain = router._get_ordered_endpoints(LLMTier.PRIMARY, allow_secondary=False)
    assert "Gemini-Fast" in chain, (
        "excluding the Solver also removed the cloud fallback — the 32B now "
        "drops straight to the 7B brainstem"
    )
    assert chain.index(PRIMARY_ENDPOINT) < chain.index("Gemini-Fast")
    assert chain.index("Gemini-Fast") < chain.index("Brainstem")


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
        "(cloud + deep), so tier preference must not imply deep handoff"
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
