"""LAMP/AuraNow binding layer.

This package provides Aura's canonical state-grounded "now" surface: a
continuous self-field, interoception, affective control variables, workspace
ignition, ownership attribution, higher-order observation, protected continuity,
and introspection verification.
"""

from .aura_now import (
    AffectiveState,
    AttentionState,
    AuraNow,
    BodyState,
    MemoryContext,
    OwnershipState,
    PredictionState,
    ReportBoundary,
    SelfState,
    WillStateSnapshot,
    WorkspaceState,
    WorldState,
)
from .runtime import BeingRuntime, get_being_runtime, reset_being_runtime_for_test

__all__ = [
    "AffectiveState",
    "AttentionState",
    "AuraNow",
    "BeingRuntime",
    "BodyState",
    "MemoryContext",
    "OwnershipState",
    "PredictionState",
    "ReportBoundary",
    "SelfState",
    "WillStateSnapshot",
    "WorldState",
    "WorkspaceState",
    "get_being_runtime",
    "reset_being_runtime_for_test",
]
