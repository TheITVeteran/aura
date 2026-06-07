"""Canonical source-promotion policy for self-modification.

Normal Aura runtime may diagnose, draft, stage, and validate repairs. Writing a
validated patch back into the live source tree is a separate promotion action
and must only happen in an operator-controlled repair-lab or supervised flow.
"""
from __future__ import annotations

from dataclasses import dataclass
import os

RUNTIME_SELF_MODIFICATION_ENV = "AURA_ALLOW_RUNTIME_SELF_MODIFICATION"
AUTONOMOUS_PATCH_PROMOTION_ENV = "AURA_ALLOW_AUTONOMOUS_PATCH_PROMOTION"
REPAIR_LAB_SOURCE_PROMOTION_ENV = "AURA_ALLOW_REPAIR_LAB_SOURCE_PROMOTION"
SUPERVISED_SELF_MODIFICATION_ENV = "AURA_ALLOW_SUPERVISED_SELF_MODIFICATION"


@dataclass(frozen=True)
class PromotionPolicyDecision:
    allowed: bool
    reason: str
    required_env: tuple[str, ...]


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def autonomous_source_promotion_decision() -> PromotionPolicyDecision:
    required = (
        RUNTIME_SELF_MODIFICATION_ENV,
        AUTONOMOUS_PATCH_PROMOTION_ENV,
        REPAIR_LAB_SOURCE_PROMOTION_ENV,
    )
    missing = tuple(name for name in required if not env_flag(name, False))
    if missing:
        return PromotionPolicyDecision(
            allowed=False,
            reason=(
                "autonomous source promotion requires an explicit repair-lab "
                f"profile with {', '.join(required)}=1"
            ),
            required_env=required,
        )
    return PromotionPolicyDecision(
        allowed=True,
        reason="autonomous repair-lab source promotion enabled",
        required_env=required,
    )


def supervised_source_promotion_decision() -> PromotionPolicyDecision:
    required = (SUPERVISED_SELF_MODIFICATION_ENV,)
    if not env_flag(SUPERVISED_SELF_MODIFICATION_ENV, False):
        return PromotionPolicyDecision(
            allowed=False,
            reason=f"{SUPERVISED_SELF_MODIFICATION_ENV}=1 is required for supervised source promotion",
            required_env=required,
        )
    return PromotionPolicyDecision(
        allowed=True,
        reason="supervised source promotion enabled",
        required_env=required,
    )


def source_promotion_decision(*, supervised: bool) -> PromotionPolicyDecision:
    if supervised:
        return supervised_source_promotion_decision()
    return autonomous_source_promotion_decision()


__all__ = [
    "AUTONOMOUS_PATCH_PROMOTION_ENV",
    "PromotionPolicyDecision",
    "REPAIR_LAB_SOURCE_PROMOTION_ENV",
    "RUNTIME_SELF_MODIFICATION_ENV",
    "SUPERVISED_SELF_MODIFICATION_ENV",
    "autonomous_source_promotion_decision",
    "env_flag",
    "source_promotion_decision",
    "supervised_source_promotion_decision",
]
