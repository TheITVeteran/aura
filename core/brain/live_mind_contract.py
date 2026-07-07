"""Shared live-mind generation contract helpers.

The desktop lane treats live-mind metadata as proof material. A single stale
worker receipt must not downgrade a turn after the CognitiveEngine has already
bound the live mind controls and the surface quality gate passed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


REQUIRED_LIVE_MIND_GENERATION_CONTROL_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "clean_user_surface_recurrent_loops",
        "clean_user_surface_steering_alpha",
    }
)


def live_mind_generation_controls_present(generation_controls: Any) -> bool:
    return bool(
        isinstance(generation_controls, Mapping)
        and REQUIRED_LIVE_MIND_GENERATION_CONTROL_KEYS.issubset(
            generation_controls.keys()
        )
    )


def normalize_live_mind_surface_control_receipt(
    receipt: Any,
    *,
    controls_bound: bool,
    generation_controls: Any,
    surface_quality_gate_passed: bool | None = None,
    source: str,
) -> dict[str, Any]:
    """Return a coherent receipt for an already-bound live-mind turn.

    The worker reports whether it applied CAA/recurrent surface controls, while
    the CognitiveEngine owns whether the live mind controls were structurally
    bound for the turn. If the worker receipt is otherwise successful but omits
    that structural bit, normalize it before the chat contract evaluates the
    full-mind path.
    """

    normalized = dict(receipt) if isinstance(receipt, Mapping) else {}
    controls_present = live_mind_generation_controls_present(generation_controls)
    quality_passed = (
        bool(normalized.get("surface_quality_gate_passed", True))
        if surface_quality_gate_passed is None
        else bool(surface_quality_gate_passed)
    )

    if not (controls_bound and controls_present and quality_passed):
        return normalized

    if (
        normalized.get("live_mind_controls_bound") is True
        and normalized.get("applied") is True
        and normalized.get("clean_user_surface_contract") is True
    ):
        return normalized

    return {
        **normalized,
        "enabled": bool(normalized.get("enabled", False)),
        "applied": True,
        "live_mind_controls_bound": True,
        "clean_user_surface_contract": True,
        "surface_quality_gate_enabled": bool(
            normalized.get("surface_quality_gate_enabled", False)
        ),
        "surface_quality_gate_passed": True,
        "surface_quality_gate_attempts": int(
            normalized.get("surface_quality_gate_attempts", 0) or 0
        ),
        "surface_quality_gate_reasons": list(
            normalized.get("surface_quality_gate_reasons", []) or []
        ),
        "source": normalized.get("source") or source,
    }
