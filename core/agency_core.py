"""core/agency_core.py -- Compatibility Facade for AgencyCore

All actual implementation has been consolidated under the agency subsystem:
core/agency/agency_core.py

This module re-exports all elements to ensure complete backward-compatibility.
"""
from __future__ import annotations

import sys
import types
from typing import Any

from core.agency import agency_core as _canonical

AgencyBus = _canonical.AgencyBus
EngagementMode = _canonical.EngagementMode
AgencyState = _canonical.AgencyState
SovereignSwarm = _canonical.SovereignSwarm
AgencyCore = _canonical.AgencyCore
get_task_tracker = _canonical.get_task_tracker
_schedule_agency_task = _canonical._schedule_agency_task

__all__ = [
    "AgencyBus",
    "EngagementMode",
    "AgencyState",
    "SovereignSwarm",
    "AgencyCore",
    "get_task_tracker",
    "_schedule_agency_task",
]


def __getattr__(name: str) -> Any:
    return getattr(_canonical, name)


class _AgencyCoreFacadeModule(types.ModuleType):
    """Propagate legacy monkeypatches to the canonical agency module."""

    def __setattr__(self, name: str, value: Any) -> None:
        if not name.startswith("__") and hasattr(_canonical, name):
            setattr(_canonical, name, value)
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _AgencyCoreFacadeModule
