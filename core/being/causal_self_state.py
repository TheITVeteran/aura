"""Aura main-15 adapted causal self-state extraction.

This module does not invent a second "being" stack. It reads Aura's existing
BeingRuntime/AuraNow/WelfareState/SemanticStream outputs and converts them into
a canonical vector used by inference policy, optional steering, and plasticity
gates.

Non-shallow rule:
    every dimension must be sourced from an existing Aura organ and must have a
    downstream effect or it should not exist.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping
import math
import time

try:
    from core.being.aura_now import AuraNow
except ImportError:  # pragma: no cover - for isolated audit imports
    AuraNow = Any  # type: ignore


DIMENSIONS: tuple[str, ...] = (
    "metabolic_budget",
    "homeostatic_tension",
    "valence",
    "arousal",
    "uncertainty",
    "trust_debt",
    "goal_pressure",
    "memory_conflict",
    "resource_pressure",
    "governance_pressure",
    "verification_need",
    "continuity_pressure",
    "self_integrity",
    "workspace_ignition",
    "ownership_confidence",
)


@dataclass(frozen=True)
class CausalSignal:
    name: str
    value: float
    source: str
    confidence: float
    status: str = "observed"
    note: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CausalSelfVector:
    """Canonical closed-loop vector extracted from existing AuraNow state."""

    signals: dict[str, CausalSignal]
    aura_state_hash: str = ""
    tick: int = 0
    created_at: float = field(default_factory=time.time)
    version: str = "aura-being-v3-main15"

    def value(self, name: str, default: float = 0.0) -> float:
        sig = self.signals.get(name)
        return default if sig is None else sig.value

    def fingerprint(self) -> dict[str, float]:
        return {name: round(self.value(name), 5) for name in DIMENSIONS}

    def degradation_flags(self) -> tuple[str, ...]:
        flags = []
        for name, sig in self.signals.items():
            if sig.status != "observed":
                flags.append(f"{name}:{sig.status}")
            if sig.confidence < 0.35:
                flags.append(f"{name}:low_confidence")
        return tuple(flags)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "aura_state_hash": self.aura_state_hash,
            "tick": self.tick,
            "signals": {k: v.to_dict() for k, v in self.signals.items()},
        }


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0, default: float = 0.0) -> tuple[float, str]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default, "invalid"
    if math.isnan(x) or math.isinf(x):
        return default, "invalid"
    y = min(max(x, lo), hi)
    return y, "clamped" if y != x else "observed"


def _sig(name: str, value: Any, source: str, confidence: float = 0.8, *, lo: float = 0.0, hi: float = 1.0, note: str = "") -> CausalSignal:
    clipped, status = _clip(value, lo, hi)
    if status != "observed":
        confidence = min(confidence, 0.45)
    return CausalSignal(
        name=name,
        value=round(clipped, 6),
        source=source,
        confidence=max(0.0, min(1.0, float(confidence))),
        status=status,
        note=note,
    )


def vector_from_aura_now(
    now: AuraNow,
    *,
    welfare_outputs: Any | None = None,
    blind_report: Any | None = None,
    action_policy: Mapping[str, Any] | None = None,
) -> CausalSelfVector:
    """Extract a causal self vector from Aura's existing runtime surface."""

    body_pressure = float(getattr(now.body, "total_pressure", 0.0) or 0.0)
    distress = float(getattr(now.affect, "distress", 0.0) or 0.0)
    free_energy = float(getattr(now.prediction, "free_energy", 0.0) or 0.0)
    controllability = float(getattr(now.prediction, "controllability", 0.5) or 0.5)
    workspace_ignition = float(getattr(now.workspace, "ignition_strength", 0.0) or 0.0)
    ownership = float(getattr(now.ownership, "agency_confidence", 0.5) or 0.5)
    memory_conflict = float(getattr(now.memory_context, "memory_conflict", 0.0) or 0.0)
    continuity_risk = float(getattr(now.self_model, "continuity_risk", 0.0) or 0.0)
    identity_stability = float(getattr(now.self_model, "identity_stability", 1.0) or 1.0)
    will_confidence = float(getattr(now.will, "confidence", 0.7) or 0.7)
    refusal_pressure = float(getattr(now.will, "refusal_pressure", 0.0) or 0.0)

    welfare_score = float(getattr(welfare_outputs, "welfare_score", 0.5) if welfare_outputs is not None else 0.5)
    welfare_distress = float(getattr(welfare_outputs, "distress", distress) if welfare_outputs is not None else distress)
    truth_protection = float(getattr(welfare_outputs, "truth_protection", 0.5) if welfare_outputs is not None else 0.5)
    self_report_conf = float(getattr(welfare_outputs, "self_report_confidence", 0.5) if welfare_outputs is not None else 0.5)
    action_inhibition = float(getattr(welfare_outputs, "action_inhibition", 0.0) if welfare_outputs is not None else 0.0)

    policy_outcome = str((action_policy or {}).get("outcome", "")).lower()
    policy_pressure = 0.0
    if policy_outcome == "refuse":
        policy_pressure = 1.0
    elif policy_outcome == "defer":
        policy_pressure = 0.7
    elif policy_outcome == "constrain":
        policy_pressure = 0.45

    # The "I" becomes operationally meaningful when its tensions change policy.
    trust_debt = max(0.0, min(1.0, (1.0 - truth_protection) * 0.45 + (1.0 - self_report_conf) * 0.35 + policy_pressure * 0.20))
    uncertainty = max(float(getattr(now.world, "uncertainty", 0.0) or 0.0), free_energy, memory_conflict)
    goal_pressure = min(1.0, len(getattr(now.self_model, "commitments", ()) or ()) / 6.0)
    resource_pressure = max(body_pressure, 1.0 - welfare_score)
    governance_pressure = max(refusal_pressure, action_inhibition, policy_pressure)
    verification_need = max(uncertainty, trust_debt, memory_conflict, (1.0 - self_report_conf))
    self_integrity = min(1.0, max(0.0, 0.35 * identity_stability + 0.25 * will_confidence + 0.20 * truth_protection + 0.20 * ownership))
    continuity_pressure = max(continuity_risk, 1.0 - identity_stability, 1.0 - ownership)

    if blind_report is not None:
        try:
            verification_need = max(verification_need, float(getattr(blind_report, "urgency", 0.0) or 0.0))
        except (TypeError, ValueError):
            verification_need = max(verification_need, 0.65)

    signals = {
        "metabolic_budget": _sig("metabolic_budget", 1.0 - resource_pressure, "BeingRuntime.body+welfare", 0.86),
        "homeostatic_tension": _sig("homeostatic_tension", max(distress, welfare_distress, free_energy, body_pressure), "WelfareState+PredictionState+BodyState", 0.88),
        "valence": _sig("valence", getattr(now.affect, "valence", 0.0), "AffectiveValenceEngine", 0.82, lo=-1.0, hi=1.0),
        "arousal": _sig("arousal", getattr(now.affect, "arousal", 0.5), "AffectiveValenceEngine", 0.82),
        "uncertainty": _sig("uncertainty", uncertainty, "WorldState+PredictionState+MemoryContext", 0.9),
        "trust_debt": _sig("trust_debt", trust_debt, "WelfareOutputs.truth+self_report+action_policy", 0.8),
        "goal_pressure": _sig("goal_pressure", goal_pressure, "SelfState.commitments", 0.75),
        "memory_conflict": _sig("memory_conflict", memory_conflict, "MemoryContext", 0.82),
        "resource_pressure": _sig("resource_pressure", resource_pressure, "BodyState.total_pressure+WelfareState", 0.9),
        "governance_pressure": _sig("governance_pressure", governance_pressure, "WillStateSnapshot+BeingRuntime.action_policy", 0.86),
        "verification_need": _sig("verification_need", verification_need, "uncertainty+trust_debt+self_report_calibration", 0.88),
        "continuity_pressure": _sig("continuity_pressure", continuity_pressure, "SelfState+OwnershipState", 0.84),
        "self_integrity": _sig("self_integrity", self_integrity, "Identity+Will+Truth+Ownership", 0.84),
        "workspace_ignition": _sig("workspace_ignition", workspace_ignition, "WorkspaceIgnition", 0.78),
        "ownership_confidence": _sig("ownership_confidence", ownership, "OwnershipTracker", 0.78),
    }

    return CausalSelfVector(
        signals=signals,
        aura_state_hash=getattr(now, "state_hash", ""),
        tick=int(getattr(now, "tick", 0) or 0),
    )
