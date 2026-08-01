# ruff: noqa: N999
"""Typed physical reachability, experimentation, and evidence contracts."""

from core.reality_reach.contracts import (
    ChannelDeclaration,
    ChannelKind,
    Constraint,
    ConstraintKind,
    CouplingClass,
    EvidenceLevel,
    FailureCode,
    NumericDomain,
    ObjectiveKind,
    ProofRequirement,
    ReachabilityCertificate,
    ReachabilityFailure,
    ReachabilityStatus,
    RealityIR,
    RealityLayer,
)
from core.reality_reach.live import (
    ChannelReading,
    HostResourceAdapter,
    ReadingStatus,
    RealityReachService,
    get_reality_reach_service,
    register_reality_reach_service,
)
from core.reality_reach.reachability import ChannelRegistry, ReachabilityEngine

__all__ = [
    "ChannelDeclaration",
    "ChannelKind",
    "ChannelReading",
    "ChannelRegistry",
    "Constraint",
    "ConstraintKind",
    "CouplingClass",
    "EvidenceLevel",
    "FailureCode",
    "HostResourceAdapter",
    "NumericDomain",
    "ObjectiveKind",
    "ProofRequirement",
    "RealityIR",
    "RealityLayer",
    "ReadingStatus",
    "ReachabilityCertificate",
    "ReachabilityEngine",
    "ReachabilityFailure",
    "ReachabilityStatus",
    "RealityReachService",
    "get_reality_reach_service",
    "register_reality_reach_service",
]
