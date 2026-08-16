"""core/skills/action_gateway.py — the default approver for a FluidExecutor step.

`FluidExecutor` took an optional `gateway` and treated its absence as approval:

    if self._gateway is None:
        return True, ""

`DesktopPlanner` constructs its executor without one, and so do
`GoalPursuitEngine` and `ParallelExecutor`. So the default construction of the
agency lane approved every step it was asked about, and the sentence "no
consequential action occurs outside the canonical governance lane" was not true
for any of them.

Deleting the fail-open without supplying a default would block every step in
those three callers, which is why it survived. This is the default: a thin
adapter over the two admission surfaces that already exist, rather than a
fourth opinion about what is allowed.

    Standing directives   deny-only durable prohibitions the user has set.
                          A match is a refusal and nothing here can grant.
    Constitutional guard  destructive-shell and critical-path blocks.

Neither can approve anything on its own — they can only decline — so an
`allowed` result here means "no prohibition matched", not "somebody authorised
this". The authorisation is the governance receipt required at the skill
boundary in `core/skills/base_skill.py`, which is the check that actually
guards the effect. This gateway is the earlier, cheaper refusal that stops a
plan before it runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.ActionGateway")


@dataclass(frozen=True)
class ActionDecision:
    """A step-level verdict. `reason` is empty only when allowed."""

    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


class ActionGateway:
    """Approve a named step against the standing prohibitions in force."""

    def approve(self, name: str, args: dict[str, Any] | None = None) -> ActionDecision:
        payload = dict(args or {})

        directive = self._standing_directive_refusal(name, payload)
        if directive is not None:
            return directive

        constitutional = self._constitutional_refusal(name, payload)
        if constitutional is not None:
            return constitutional

        return ActionDecision(allowed=True)

    def _standing_directive_refusal(
        self, name: str, args: dict[str, Any]
    ) -> ActionDecision | None:
        try:
            from core.governance.standing_directives import get_standing_directives

            match, loaded = get_standing_directives().check(
                tool_name=name, args=args, effect_scope="write"
            )
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as exc:
            record_degradation(
                "action_gateway",
                exc,
                action="refused a step because standing directives could not be read",
            )
            return ActionDecision(
                allowed=False,
                reason=f"standing directives unreadable: {exc}",
            )
        if getattr(loaded, "unreadable", False):
            # A prohibition file that cannot be parsed is a prohibition whose
            # contents are unknown, which is not the same as no prohibitions.
            return ActionDecision(
                allowed=False, reason="standing directives file is unreadable"
            )
        if match is not None:
            directive = getattr(match, "directive", None)
            detail = getattr(directive, "value", "") or getattr(match, "matched_on", "")
            return ActionDecision(
                allowed=False, reason=f"standing directive forbids this: {detail}"
            )
        return None

    def _constitutional_refusal(
        self, name: str, args: dict[str, Any]
    ) -> ActionDecision | None:
        try:
            from core.security.constitutional_guard import ConstitutionalGuard

            if not ConstitutionalGuard().check_action(name, args):
                return ActionDecision(
                    allowed=False, reason="constitutional guard refused this action"
                )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "action_gateway",
                exc,
                action="refused a step because the constitutional guard was unavailable",
            )
            return ActionDecision(
                allowed=False, reason=f"constitutional guard unavailable: {exc}"
            )
        return None


_GATEWAY: ActionGateway | None = None


def get_action_gateway() -> ActionGateway:
    """The process-wide default approver."""
    global _GATEWAY
    if _GATEWAY is None:
        _GATEWAY = ActionGateway()
    return _GATEWAY


__all__ = ["ActionDecision", "ActionGateway", "get_action_gateway"]
