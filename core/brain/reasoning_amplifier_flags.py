"""Canonical runtime controls owned by the reasoning amplifier."""

from __future__ import annotations

from core.runtime.flags import FlagKind, declare

_REASONING_AMPLIFIER_V2 = declare(
    "AURA_REASONING_AMPLIFIER_V2",
    kind=FlagKind.STRING,
    default="1",
    description="Enable verifier-backed reasoning amplification for eligible turns.",
    owner="core.brain.reasoning_amplifier_v2",
)


def reasoning_amplifier_v2_enabled() -> bool:
    """Return the live typed switch without creating a second flag owner."""

    return str(_REASONING_AMPLIFIER_V2.value()).strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }
