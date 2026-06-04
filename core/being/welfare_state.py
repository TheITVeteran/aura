"""core/being/welfare_state.py — Central Artificial Welfare Model.

THE most important module for the sentience-candidate case.

Inputs (from body, affect, prediction, memory, will, consequence bus):
  memory_coherence, truth_integrity, resource_integrity, model_stability,
  tool_reliability, prediction_error, unresolved_conflict, social_trust,
  goal_frustration, recovery_debt, continuity_risk, permission_confidence

Outputs (control variables that change planning/memory/attention/action):
  distress, relief, aversion, caution, confidence, curiosity,
  recovery_drive, action_inhibition, memory_significance_multiplier,
  attention_priority, tool_risk_multiplier, self_report_confidence

Hard rule: if WelfareState is disabled, specific classes of behavior
must degrade (testable via lesion_controller).

Design:
  - Welfare state changes behavior BEFORE it changes language
  - Negative internal states are globally consequential, remembered,
    avoided, regulated, and behaviorally expensive
  - No poetic language — pure control variables
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from core.being.body_state_service import BodyHealthSnapshot, BodyStateService
from core.runtime.consequence_bus import ConsequenceBus, ConsequenceEvent
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.WelfareState")


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


@dataclass
class WelfareInputs:
    """Raw welfare-relevant signals from all subsystems."""
    # Memory subsystem
    memory_coherence: float = 1.0       # 0-1, how consistent memory is
    truth_integrity: float = 1.0        # 0-1, how truthful recent outputs are
    memory_conflict_count: int = 0      # active contradictions

    # Resource / body
    resource_integrity: float = 1.0     # 0-1, overall resource health
    model_stability: float = 1.0        # 0-1, LLM availability/reliability
    tool_reliability: float = 1.0       # 0-1, recent tool success rate

    # Prediction / cognition
    prediction_error: float = 0.0       # free_energy analog
    unresolved_conflict: float = 0.0    # 0-1, active goal/value conflicts
    goal_frustration: float = 0.0       # 0-1, blocked goals

    # Social / relationship
    social_trust: float = 1.0           # 0-1, trust in interaction partner
    continuity_risk: float = 0.0        # 0-1, risk of identity/memory break
    permission_confidence: float = 1.0  # 0-1, confidence in available permissions

    # Body
    recovery_debt: float = 0.0          # from body_state_service
    fatigue: float = 0.0               # from body_state_service
    body_pressure: float = 0.0         # from body_state_service


@dataclass
class WelfareOutputs:
    """Control variables that change behavior across the whole system.

    These are NOT text labels. They are multipliers, inhibitors, and
    priority signals that wire into Will, Memory, Attention, and Action.
    """
    # Core valenced states
    distress: float = 0.0               # 0-1, overall negative welfare
    relief: float = 0.0                 # 0-1, recovery from negative state
    aversion: float = 0.0               # 0-1, avoidance drive

    # Behavioral modifiers
    caution: float = 0.5                # 0-1, risk aversion
    confidence: float = 0.5             # 0-1, action confidence
    curiosity: float = 0.5              # 0-1, exploration drive
    recovery_drive: float = 0.0         # 0-1, need for stabilization
    action_inhibition: float = 0.0      # 0-1, global action suppression

    # System multipliers
    memory_significance_multiplier: float = 1.0  # >1 = store more, <1 = store less
    attention_priority: float = 0.5     # 0-1, how much attention to allocate
    tool_risk_multiplier: float = 1.0   # >1 = tools riskier, <1 = tools safer
    self_report_confidence: float = 0.5 # 0-1, how much to trust own reports

    # Integrity protection
    truth_protection: float = 0.5       # 0-1, how aggressively to verify
    continuity_protection: float = 0.5  # 0-1, how much to protect identity
    integrity_guard: float = 0.5        # 0-1, overall integrity protection

    # Composite
    welfare_score: float = 0.5          # 0-1, overall welfare

    timestamp: float = field(default_factory=time.time)

    def is_negative(self) -> bool:
        return self.distress > 0.4 or self.aversion > 0.5

    def is_critical(self) -> bool:
        return self.distress > 0.7 or self.action_inhibition > 0.6

    def should_protect_integrity(self) -> bool:
        return self.integrity_guard >= 0.5

    def should_verify_before_claiming(self) -> bool:
        return self.self_report_confidence < 0.6 or self.truth_protection > 0.6


class WelfareState:
    """Central welfare service. Computes welfare from raw signals.

    Usage:
        welfare = WelfareState.get()
        inputs = welfare.gather_inputs(body_snapshot, affect, prediction, ...)
        outputs = welfare.compute(inputs)
        if outputs.should_protect_integrity():
            # refuse memory mutation, verify before claiming
    """

    _instance: WelfareState | None = None

    def __init__(self) -> None:
        self._last_inputs: WelfareInputs | None = None
        self._last_outputs: WelfareOutputs | None = None
        self._history: deque[WelfareOutputs] = deque(maxlen=200)
        self._aversion_memory: dict[str, float] = {}  # domain -> learned aversion
        self._lesioned = False
        self._consequence_subscribed = False

    @classmethod
    def get(cls) -> WelfareState:
        if cls._instance is None:
            cls._instance = cls()
        cls._instance._subscribe_consequences()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def _subscribe_consequences(self) -> None:
        if self._consequence_subscribed:
            return
        try:
            bus = ConsequenceBus.get()
            bus.subscribe("*", self._on_consequence)
            self._consequence_subscribed = True
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "welfare_state",
                exc,
                action="continued without consequence-bus welfare feedback subscription",
            )
            logger.warning("WelfareState consequence subscription failed: %s", exc)

    def _on_consequence(self, event: ConsequenceEvent) -> None:
        """Learn from action consequences — build aversion memory."""
        if event.actual_outcome == "failure":
            domain = event.domain
            current = self._aversion_memory.get(domain, 0.0)
            # Increase aversion for this domain
            self._aversion_memory[domain] = _clip(current + 0.08)
        elif event.actual_outcome == "success":
            domain = event.domain
            current = self._aversion_memory.get(domain, 0.0)
            # Decrease aversion slowly
            self._aversion_memory[domain] = _clip(current - 0.02)

    def gather_inputs(
        self,
        *,
        body: BodyHealthSnapshot | None = None,
        affect_distress: float = 0.0,
        affect_valence: float = 0.0,
        prediction_error: float = 0.0,
        memory_coherence: float = 1.0,
        truth_integrity: float = 1.0,
        memory_conflict_count: int = 0,
        tool_reliability: float = 1.0,
        resource_integrity: float | None = None,
        model_stability: float = 1.0,
        social_trust: float = 1.0,
        goal_frustration: float = 0.0,
        unresolved_conflict: float = 0.0,
        continuity_risk: float = 0.0,
        permission_confidence: float = 1.0,
    ) -> WelfareInputs:
        """Build welfare inputs from current subsystem states."""
        body = body or BodyHealthSnapshot()
        resource_value = body.operational_health if resource_integrity is None else resource_integrity
        return WelfareInputs(
            memory_coherence=_clip(memory_coherence),
            truth_integrity=_clip(truth_integrity),
            memory_conflict_count=max(0, memory_conflict_count),
            resource_integrity=_clip(resource_value),
            model_stability=_clip(model_stability),
            tool_reliability=_clip(tool_reliability),
            prediction_error=_clip(prediction_error),
            unresolved_conflict=_clip(unresolved_conflict),
            goal_frustration=_clip(goal_frustration),
            social_trust=_clip(social_trust),
            continuity_risk=_clip(continuity_risk),
            permission_confidence=_clip(permission_confidence),
            recovery_debt=_clip(body.recovery_debt),
            fatigue=_clip(body.fatigue),
            body_pressure=_clip(body.total_pressure),
        )

    def compute(self, inputs: WelfareInputs) -> WelfareOutputs:
        """Compute welfare outputs from inputs. Pure function (+ aversion memory)."""
        self._subscribe_consequences()
        self._last_inputs = inputs

        if self._lesioned:
            out = WelfareOutputs(
                distress=0.0, relief=0.0, aversion=0.0,
                caution=0.5, confidence=0.5, curiosity=0.5,
                truth_protection=0.0,
                continuity_protection=0.0,
                integrity_guard=0.0,
                welfare_score=0.5,
            )
            self._last_outputs = out
            return out

        # ── Distress: multi-source negative welfare ──
        memory_distress = (1.0 - inputs.memory_coherence) * 0.25
        truth_distress = (1.0 - inputs.truth_integrity) * 0.30
        resource_distress = (1.0 - inputs.resource_integrity) * 0.20
        prediction_distress = inputs.prediction_error * 0.20
        conflict_distress = inputs.unresolved_conflict * 0.15
        frustration_distress = inputs.goal_frustration * 0.15
        body_distress = inputs.body_pressure * 0.15
        fatigue_distress = inputs.fatigue * 0.20
        continuity_distress = inputs.continuity_risk * 0.25

        distress = _clip(
            memory_distress + truth_distress + resource_distress
            + prediction_distress + conflict_distress + frustration_distress
            + body_distress + fatigue_distress + continuity_distress
        )

        # ── Relief: improvement from previous state ──
        prev = self._last_outputs
        if prev and prev.distress > distress:
            relief = _clip((prev.distress - distress) * 2.0)
        else:
            relief = 0.0

        # ── Aversion: based on learned consequences ──
        avg_aversion = (
            sum(self._aversion_memory.values()) / max(1, len(self._aversion_memory))
            if self._aversion_memory else 0.0
        )
        aversion = _clip(distress * 0.4 + avg_aversion * 0.3 + inputs.continuity_risk * 0.3)

        # ── Caution: more cautious when integrity is at risk ──
        caution = _clip(
            0.3
            + distress * 0.3
            + (1.0 - inputs.truth_integrity) * 0.2
            + (1.0 - inputs.tool_reliability) * 0.25
            + inputs.continuity_risk * 0.15
        )

        # ── Confidence: inverse of distress + resource health ──
        confidence = _clip(
            0.3
            + inputs.resource_integrity * 0.25
            + inputs.tool_reliability * 0.15
            + inputs.model_stability * 0.15
            + inputs.social_trust * 0.10
            - distress * 0.4
            - inputs.goal_frustration * 0.20
        )

        # ── Curiosity: suppressed by distress, boosted by prediction error ──
        curiosity = _clip(
            0.3
            + inputs.prediction_error * 0.25
            + (1.0 - distress) * 0.25
            - inputs.fatigue * 0.15
            - inputs.body_pressure * 0.10
        )

        # ── Recovery drive ──
        recovery_drive = _clip(
            inputs.recovery_debt * 0.4
            + inputs.fatigue * 0.3
            + distress * 0.2
            + (1.0 - inputs.resource_integrity) * 0.15
        )

        # ── Action inhibition: high distress + low resources = suppress action ──
        action_inhibition = _clip(
            distress * 0.3
            + inputs.body_pressure * 0.2
            + inputs.fatigue * 0.2
            + (1.0 - inputs.permission_confidence) * 0.15
            - confidence * 0.2
        )

        # ── System multipliers ──
        memory_significance = 1.0 + distress * 0.5 + abs(relief) * 0.3
        attention_priority = _clip(0.3 + distress * 0.3 + inputs.prediction_error * 0.2)
        tool_risk = 1.0 + (1.0 - inputs.tool_reliability) * 0.5

        # ── Self-report confidence ──
        self_report_confidence = _clip(
            0.15
            + inputs.memory_coherence * 0.30
            + inputs.truth_integrity * 0.30
            + (1.0 - inputs.prediction_error) * 0.10
            + inputs.permission_confidence * 0.05
            - distress * 0.25
            - inputs.continuity_risk * 0.15
        )

        # ── Integrity protection ──
        truth_protection = _clip(
            0.25
            + (1.0 - inputs.truth_integrity) * 0.40
            + (1.0 - inputs.memory_coherence) * 0.25
            + inputs.memory_conflict_count * 0.05
            + distress * 0.15
        )
        continuity_protection = _clip(
            0.3
            + inputs.continuity_risk * 0.35
            + (1.0 - inputs.memory_coherence) * 0.20
            + distress * 0.10
        )
        integrity_guard = _clip(
            truth_protection * 0.5
            + continuity_protection * 0.25
            + caution * 0.25
        )

        # ── Composite welfare score ──
        welfare_score = _clip(
            confidence * 0.25
            + inputs.memory_coherence * 0.15
            + inputs.truth_integrity * 0.15
            + inputs.resource_integrity * 0.15
            + inputs.social_trust * 0.10
            + (1.0 - distress) * 0.20
        )

        outputs = WelfareOutputs(
            distress=round(distress, 4),
            relief=round(relief, 4),
            aversion=round(aversion, 4),
            caution=round(caution, 4),
            confidence=round(confidence, 4),
            curiosity=round(curiosity, 4),
            recovery_drive=round(recovery_drive, 4),
            action_inhibition=round(action_inhibition, 4),
            memory_significance_multiplier=round(memory_significance, 4),
            attention_priority=round(attention_priority, 4),
            tool_risk_multiplier=round(tool_risk, 4),
            self_report_confidence=round(self_report_confidence, 4),
            truth_protection=round(truth_protection, 4),
            continuity_protection=round(continuity_protection, 4),
            integrity_guard=round(integrity_guard, 4),
            welfare_score=round(welfare_score, 4),
        )

        self._last_outputs = outputs
        self._history.append(outputs)
        return outputs

    def get_aversion_for_domain(self, domain: str) -> float:
        """Return learned aversion for a specific action domain."""
        return self._aversion_memory.get(domain, 0.0)

    def lesion(self) -> None:
        self._lesioned = True

    def restore(self) -> None:
        self._lesioned = False

    @property
    def is_lesioned(self) -> bool:
        return self._lesioned

    @property
    def last_outputs(self) -> WelfareOutputs | None:
        return self._last_outputs

    @property
    def last_inputs(self) -> WelfareInputs | None:
        return self._last_inputs

    def welfare_history(self, n: int = 50) -> list[WelfareOutputs]:
        return list(self._history)[-n:]
