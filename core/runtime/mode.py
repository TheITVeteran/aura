"""core/runtime/mode.py -- Canonical Runtime Mode Enforcement
================================================================

Aura supports exactly five runtime modes. Production mode is the default.
Every module that needs to check "am I in production?" or "is research enabled?"
must use these helpers. No ad-hoc os.environ checks.

Modes:
    production  — Default. Research features disabled. Fail-closed. Unsigned
                  skills do not load. Self-modification disabled. Cloud fallback
                  requires explicit opt-in.
    research    — Research features enabled. Self-repair allowed (sandboxed).
                  Experimental subsystems active. Must be explicitly set.
    dev         — Full access. Self-modification allowed (sandboxed). Debug
                  endpoints active. Hot-reload enabled. Must be explicitly set.
    simulation  — Sandboxed environment for testing. No real side effects.
                  All tool calls are mocked. Must be explicitly set.
    safe        — Emergency lockdown. All autonomous behavior disabled. No tools.
                  No cloud. No background tasks. Foreground-only chat.

Invariant:
    os.environ["AURA_MODE"] is the single source of truth.
    If unset, mode is "production".
"""
from __future__ import annotations

import logging
import os
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Runtime.Mode")


class AuraMode(StrEnum):
    """The five canonical runtime modes."""
    PRODUCTION = "production"
    RESEARCH = "research"
    DEV = "dev"
    SIMULATION = "simulation"
    SAFE = "safe"


_VALID_MODES = frozenset(m.value for m in AuraMode)

# Modes that allow potentially dangerous features
_RESEARCH_MODES = frozenset({AuraMode.RESEARCH, AuraMode.DEV})
_DANGEROUS_MODES = frozenset({AuraMode.DEV})
_SANDBOXED_MODES = frozenset({AuraMode.SIMULATION})
_RESTRICTED_MODES = frozenset({AuraMode.SAFE})


def get_mode() -> AuraMode:
    """Return the current runtime mode. Default is production."""
    raw = os.environ.get("AURA_MODE", "production").strip().lower()
    if raw not in _VALID_MODES:
        logger.warning(
            "Unknown AURA_MODE=%r; falling back to 'production'. Valid modes: %s",
            raw,
            ", ".join(sorted(_VALID_MODES)),
        )
        return AuraMode.PRODUCTION
    return AuraMode(raw)


def is_production() -> bool:
    """True if running in production mode (default)."""
    return get_mode() == AuraMode.PRODUCTION


def is_research() -> bool:
    """True if running in research mode."""
    return get_mode() in _RESEARCH_MODES


def is_dev() -> bool:
    """True if running in developer mode."""
    return get_mode() == AuraMode.DEV


def is_simulation() -> bool:
    """True if running in simulation mode (all side effects mocked)."""
    return get_mode() == AuraMode.SIMULATION


def is_safe() -> bool:
    """True if running in safe mode (emergency lockdown)."""
    return get_mode() == AuraMode.SAFE


def allows_self_modification() -> bool:
    """True if the current mode allows self-modification."""
    return get_mode() in _DANGEROUS_MODES


def allows_research_features() -> bool:
    """True if the current mode allows research/experimental features."""
    return get_mode() in _RESEARCH_MODES


def allows_autonomous_behavior() -> bool:
    """True if the current mode allows autonomous background behavior."""
    return get_mode() not in _RESTRICTED_MODES


def allows_tool_execution() -> bool:
    """True if the current mode allows tool/skill execution."""
    mode = get_mode()
    if mode == AuraMode.SAFE:
        return False
    return True


def allows_cloud_fallback() -> bool:
    """True if the current mode allows cloud model fallback."""
    mode = get_mode()
    if mode in (AuraMode.SAFE, AuraMode.SIMULATION):
        return False
    return True


def allows_unsigned_skills() -> bool:
    """True if the current mode allows unsigned/unmanifested skills."""
    return get_mode() in (AuraMode.DEV, AuraMode.RESEARCH)


def max_autonomy_level() -> int:
    """Return the maximum autonomy level for the current mode.

    0 = disabled, 1 = passive, 2 = maintenance, 3 = proactive,
    4 = self-repair, 5 = self-modification
    """
    mode = get_mode()
    return {
        AuraMode.SAFE: 0,
        AuraMode.PRODUCTION: 2,
        AuraMode.SIMULATION: 3,
        AuraMode.RESEARCH: 4,
        AuraMode.DEV: 5,
    }.get(mode, 2)


def enforce_production_gate(feature_name: str) -> None:
    """Raise RuntimeError if a research/dev feature is used in production mode.

    Call this at the entry point of any feature that must be gated in production.
    """
    mode = get_mode()
    if mode == AuraMode.PRODUCTION:
        raise RuntimeError(
            f"Feature '{feature_name}' is not available in production mode. "
            f"Set AURA_MODE=research or AURA_MODE=dev to enable it."
        )


def mode_context() -> dict[str, Any]:
    """Return a dict describing the current mode for logging/diagnostics."""
    mode = get_mode()
    return {
        "mode": mode.value,
        "allows_research": allows_research_features(),
        "allows_self_modification": allows_self_modification(),
        "allows_autonomous": allows_autonomous_behavior(),
        "allows_tools": allows_tool_execution(),
        "allows_cloud": allows_cloud_fallback(),
        "allows_unsigned_skills": allows_unsigned_skills(),
        "max_autonomy_level": max_autonomy_level(),
    }


def validate_mode_at_startup() -> None:
    """Log the current mode and validate configuration consistency."""
    mode = get_mode()
    ctx = mode_context()
    logger.info("🔧 Runtime mode: %s", mode.value)
    for key, value in ctx.items():
        if key != "mode":
            logger.debug("  %s: %s", key, value)

    # Validate environment variable consistency
    autonomy_env = os.environ.get("AURA_AUTONOMY_LEVEL")
    if autonomy_env is not None:
        try:
            level = int(autonomy_env)
            if level > max_autonomy_level():
                logger.warning(
                    "AURA_AUTONOMY_LEVEL=%d exceeds max for mode %s (%d); clamping.",
                    level,
                    mode.value,
                    max_autonomy_level(),
                )
                os.environ["AURA_AUTONOMY_LEVEL"] = str(max_autonomy_level())
        except ValueError:
            logger.warning("Invalid AURA_AUTONOMY_LEVEL=%r; ignoring.", autonomy_env)

    if mode == AuraMode.SAFE:
        logger.warning("🔒 SAFE MODE ACTIVE: All autonomous behavior disabled.")
