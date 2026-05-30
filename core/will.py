"""core/will.py -- Compatibility Facade for the Unified Will

All actual implementation has been consolidated under the governance subsystem:
core/governance/will.py

This module re-exports all elements to ensure complete backward-compatibility
while preserving the same governance invariant: consequential decisions are
FAIL-CLOSED when the canonical UnifiedWill cannot decide.
"""
from __future__ import annotations

import sys
import types
from typing import Any

from core.governance import will as _canonical

ActionDomain = _canonical.ActionDomain
WillOutcome = _canonical.WillOutcome
IdentityAlignment = _canonical.IdentityAlignment
WillDecision = _canonical.WillDecision
WillState = _canonical.WillState
UnifiedWill = _canonical.UnifiedWill
is_plastic_target_allowed = _canonical.is_plastic_target_allowed
ALLOWED_PLASTIC_MODULES = _canonical.ALLOWED_PLASTIC_MODULES
DENIED_PLASTIC_SUBSTRINGS = _canonical.DENIED_PLASTIC_SUBSTRINGS
ServiceContainer = _canonical.ServiceContainer
record_degradation = _canonical.record_degradation
get_will = _canonical.get_will
_will_instance = _canonical._will_instance

__all__ = [
    "ActionDomain",
    "WillOutcome",
    "IdentityAlignment",
    "WillDecision",
    "WillState",
    "UnifiedWill",
    "is_plastic_target_allowed",
    "ALLOWED_PLASTIC_MODULES",
    "DENIED_PLASTIC_SUBSTRINGS",
    "ServiceContainer",
    "record_degradation",
    "_will_instance",
    "get_will",
]


def __getattr__(name: str) -> Any:
    return getattr(_canonical, name)


class _WillFacadeModule(types.ModuleType):
    """Propagate legacy monkeypatches to the canonical governance module."""

    def __setattr__(self, name: str, value: Any) -> None:
        if not name.startswith("__") and hasattr(_canonical, name):
            setattr(_canonical, name, value)
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _WillFacadeModule
