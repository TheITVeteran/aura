"""Mint the capability an actuator authority decision is supposed to carry.

Fake gateways in the actuator tests used to return ``SimpleNamespace(approved=
True, capability_token_id="cap-test")`` and nothing else. That worked only
because ``ActuatorRegistry`` treated "no signed capability was minted at all"
as a degraded-but-permitted path, so the tests were exercising the bypass —
and would have kept passing if the binding check had been deleted outright.

Once absence became a refusal (as ``AuthorityGateway._mint_signed_capability``
always said it would be: "a mint failure fails the action closed rather than
degrading it into an unauthenticated execution"), those fakes had to start
carrying a real capability. They mint one HERE, through the production issuer,
over the production digest — so the assertions now depend on the real chain
verifying, which is what they were meant to be showing.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

__all__ = ["bound_authority_decision", "signed_capability_for"]


def signed_capability_for(
    action: str,
    params: dict[str, Any],
    *,
    outcome: str = "proceed",
    domain: str = "tool_execution",
    ttl_s: float = 300.0,
) -> dict[str, Any]:
    """A real signed capability bound to exactly ``(action, params)``."""
    from core.governance.capability_chain import (
        compute_action_digest,
        get_capability_issuer,
    )

    capability = get_capability_issuer().issue_from_decision(
        SimpleNamespace(outcome=outcome, domain=domain),
        action=action,
        action_digest=compute_action_digest(action, params),
        scope=f"test:{action}",
        ttl_s=ttl_s,
    )
    return capability.to_dict()


def bound_authority_decision(
    action: str,
    params: dict[str, Any],
    **extra: Any,
) -> SimpleNamespace:
    """An approving decision whose capability actually binds these parameters.

    ``extra`` overrides or adds fields (``capability_token_id``,
    ``executive_intent_id``, ``standing_authority_token``, …) the way each test
    needs them.
    """
    fields: dict[str, Any] = {
        "approved": True,
        "reason": "approved",
        "capability_token_id": "cap-test",
        "signed_capability": signed_capability_for(action, params),
    }
    fields.update(extra)
    return SimpleNamespace(**fields)
