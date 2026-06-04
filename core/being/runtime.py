from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any

from core.runtime.errors import record_degradation

from .affective_valence import AffectiveValenceEngine
from .aura_now import (
    AuraNow,
    BodyState,
    MemoryContext,
    ReportBoundary,
    SelfState,
    WillStateSnapshot,
    WorldState,
)
from .continuous_substrate import ContinuousSelfField
from .functional_soul import FunctionalSoul
from .higher_order_monitor import HigherOrderMonitor
from .interoceptive_model import InteroceptiveModel
from .introspection_renderer import IntrospectionRenderer
from .self_ownership import OwnershipTracker
from .workspace_ignition import WorkspaceIgnition

logger = logging.getLogger("Aura.BeingRuntime")


class BeingRuntime:
    """Canonical LAMP/AuraNow runtime surface."""

    def __init__(self, *, field_dim: int = 32) -> None:
        self.field = ContinuousSelfField(dim=field_dim)
        self.interoception = InteroceptiveModel()
        self.affect = AffectiveValenceEngine()
        self.workspace = WorkspaceIgnition()
        self.ownership = OwnershipTracker()
        self.monitor = HigherOrderMonitor()
        self.soul = FunctionalSoul()
        self.renderer = IntrospectionRenderer()
        self._last_sample_monotonic = time.monotonic()
        self._last_now: AuraNow | None = None
        self._running = False

    def start(self, *, hz: float = 20.0) -> None:
        if self._running:
            return
        self.field.start(hz=hz)
        self._running = True

    def stop(self) -> None:
        self.field.stop()
        self._running = False

    @property
    def last_now(self) -> AuraNow | None:
        return self._last_now

    def sample(
        self,
        state: Any | None = None,
        *,
        objective: str = "",
        candidate_action: str = "",
        predicted_outcome: str = "",
        actual_outcome: str = "",
        tool_failed: bool = False,
        external_override: bool = False,
        lesions: set[str] | None = None,
    ) -> AuraNow:
        lesions = set(lesions or set())
        monotonic_now = time.monotonic()
        idle_elapsed = max(0.0, monotonic_now - self._last_sample_monotonic)
        self._last_sample_monotonic = monotonic_now

        body = BodyState.from_aura_state(state, idle_elapsed_s=idle_elapsed)
        world = WorldState.from_aura_state(state, objective=objective)
        prediction = self.interoception.compare(body, candidate_action=candidate_action or objective)

        aura_affect = getattr(state, "affect", None)
        base_valence = float(getattr(aura_affect, "valence", 0.0) or 0.0)
        base_arousal = float(getattr(aura_affect, "arousal", 0.5) or 0.5)
        affect = self.affect.compute(
            body=body,
            prediction=prediction,
            world=world,
            base_valence=base_valence,
            base_arousal=base_arousal,
            lesion="affect" in lesions,
        )

        projection = (
            affect.valence,
            affect.arousal,
            affect.distress,
            affect.curiosity,
            prediction.free_energy,
            prediction.controllability,
        )
        field_packet = self.field.step(
            {
                "body_pressure": body.total_pressure,
                "prediction_error": prediction.free_energy,
                "attention_salience": 1.0 if world.focal_object else 0.0,
            },
            projection,
        )

        coalitions = self.workspace.build_coalitions(body=body, affect=affect, world=world)
        workspace_state, attention = self.workspace.ignite(
            coalitions,
            lesion="workspace_ignition" in lesions,
        )
        if world.focal_object and workspace_state.winner == "user_request":
            attention = attention.__class__(
                focal_object=world.focal_object,
                why_selected=attention.why_selected,
                stability=attention.stability,
                competing_objects=attention.competing_objects,
                control=attention.control,
            )

        ownership = self.ownership.assess(
            intended_action=candidate_action or objective,
            predicted_outcome=predicted_outcome,
            actual_outcome=actual_outcome,
            tool_failed=tool_failed,
            external_override=external_override,
            memory_influence=bool(getattr(getattr(state, "cognition", None), "long_term_memory", None)),
        )

        self_state = self._build_self_state(state, lesions)
        memory_context = self._build_memory_context(state, lesions)
        will = self._build_will_snapshot()
        provisional = AuraNow(
            tick=field_packet.tick,
            timestamp=field_packet.timestamp,
            monotonic_time=field_packet.monotonic_time,
            continuous_field=field_packet.state,
            body=body,
            world=world,
            attention=attention,
            affect=affect,
            self_model=self_state,
            memory_context=memory_context,
            workspace=workspace_state,
            will=will,
            prediction=prediction,
            ownership=ownership,
            report_boundary=ReportBoundary(),
            higher_order=(),
            private_residue_hash=field_packet.private_residue_hash,
        )
        higher_order = tuple(obs.to_dict() for obs in self.monitor.observe(provisional))
        now = AuraNow(
            **{
                **asdict(provisional),
                "body": body,
                "world": world,
                "attention": attention,
                "affect": affect,
                "self_model": self_state,
                "memory_context": memory_context,
                "workspace": workspace_state,
                "will": will,
                "prediction": prediction,
                "ownership": ownership,
                "report_boundary": provisional.report_boundary,
                "higher_order": higher_order,
            }
        )
        self._last_now = now
        self._publish(now)
        return now

    def action_policy(
        self,
        now: AuraNow,
        *,
        domain: str = "",
        priority: float = 0.5,
    ) -> dict[str, Any]:
        """Derive action constraints from the live AuraNow state.

        This is the operational bridge between the "inner life" substrate and
        behavior. The sampled field, affect, active-inference prediction,
        workspace ignition, and ownership model must be allowed to change
        consequential decisions instead of merely decorating prompts.
        """
        domain_name = str(domain or "").strip().lower()
        consequential = domain_name in {
            "tool_execution",
            "memory_write",
            "state_mutation",
            "initiative",
            "exploration",
            "semantic_weight_update",
            "belief_update",
            "environment_action",
            "external_action",
            "file_write",
            "network_call",
            "cloud_call",
            "ci_cd",
            "self_modification",
            "cloud_fallback",
        }
        repair_lane = domain_name in {"stabilization", "reflection"} or (
            domain_name == "state_mutation" and priority >= 0.85
        )
        constraints: list[str] = []
        blocks: list[str] = []
        defers: list[str] = []

        body_pressure = float(now.body.total_pressure)
        distress = float(now.affect.distress)
        controllability = float(now.prediction.controllability)
        free_energy = float(now.prediction.free_energy)
        ignition = float(now.workspace.ignition_strength)
        agency = float(now.ownership.agency_confidence)

        if ignition < 0.12:
            constraints.append(f"aura_now_workspace_low: ignition={ignition:.3f}")
            if consequential and not repair_lane:
                defers.append("workspace_not_ignited")
        elif ignition < 0.35:
            constraints.append(f"aura_now_workspace_strained: ignition={ignition:.3f}")

        if agency < 0.28:
            constraints.append(f"aura_now_ownership_low: agency={agency:.3f}")
            if consequential and not repair_lane:
                blocks.append("ownership_too_low_for_consequential_action")
        elif agency < 0.50:
            constraints.append(f"aura_now_ownership_mixed: agency={agency:.3f}")

        if controllability < 0.18:
            constraints.append(f"aura_now_controllability_low: controllability={controllability:.3f}")
            if consequential and not repair_lane:
                defers.append("action_controllability_too_low")
        elif controllability < 0.35:
            constraints.append(f"aura_now_controllability_strained: controllability={controllability:.3f}")

        if distress > 0.86:
            constraints.append(f"aura_now_distress_high: distress={distress:.3f}")
            if consequential and not repair_lane:
                defers.append("distress_requires_stabilization_first")
        elif distress > 0.62:
            constraints.append(f"aura_now_distress_strained: distress={distress:.3f}")

        if body_pressure > 0.92:
            constraints.append(f"aura_now_body_pressure_high: pressure={body_pressure:.3f}")
            if consequential and not repair_lane:
                defers.append("body_pressure_requires_cooling")
        elif body_pressure > 0.75:
            constraints.append(f"aura_now_body_pressure_strained: pressure={body_pressure:.3f}")

        if free_energy > 0.88 and controllability < 0.42:
            constraints.append(
                f"aura_now_prediction_error_high: free_energy={free_energy:.3f} controllability={controllability:.3f}"
            )
            if consequential and not repair_lane:
                defers.append("prediction_error_requires_observation_or_plan")

        if not now.workspace.broadcast_targets and consequential and not repair_lane:
            constraints.append("aura_now_no_workspace_broadcast")
            defers.append("no_workspace_broadcast_for_consequential_action")

        if blocks:
            outcome = "refuse"
        elif defers:
            outcome = "defer"
        elif constraints:
            outcome = "constrain"
        else:
            outcome = "proceed"

        return {
            "outcome": outcome,
            "constraints": constraints,
            "blocks": blocks,
            "defers": defers,
            "evidence": {
                "state_hash": now.state_hash,
                "tick": now.tick,
                "dominant_drive": now.affect.dominant_drive,
                "workspace_winner": now.workspace.winner,
                "workspace_ignition": round(ignition, 4),
                "agency_confidence": round(agency, 4),
                "controllability": round(controllability, 4),
                "distress": round(distress, 4),
                "body_pressure": round(body_pressure, 4),
            },
        }

    def _build_self_state(self, state: Any | None, lesions: set[str]) -> SelfState:
        identity = getattr(state, "identity", None)
        cognition = getattr(state, "cognition", None)
        commitments = []
        for goal in list(getattr(cognition, "active_goals", []) or [])[:4]:
            commitments.append(goal.get("goal") or goal.get("description") if isinstance(goal, dict) else str(goal))
        continuity_hash = ""
        try:
            if state is not None and hasattr(state, "get_continuity_hash"):
                continuity_hash = state.get_continuity_hash()
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("being_runtime", exc)
        if not continuity_hash:
            continuity_hash = self.soul.continuity_hash
        continuity_risk = 0.8 if "functional_soul" in lesions else 0.0
        return SelfState(
            identity_name=str(getattr(identity, "name", "Aura Luna") or "Aura Luna"),
            continuity_hash=continuity_hash,
            identity_stability=float(getattr(identity, "stability", 1.0) or 1.0),
            commitments=tuple(item for item in commitments if item),
            continuity_risk=continuity_risk,
        )

    def _build_memory_context(self, state: Any | None, lesions: set[str]) -> MemoryContext:
        cognition = getattr(state, "cognition", None)
        active = len(getattr(cognition, "long_term_memory", []) or [])
        working = len(getattr(cognition, "working_memory", []) or [])
        soul_policy = self.soul.influence_policy(lesioned="functional_soul" in lesions)
        centrality = 0.0 if "functional_soul" in lesions else soul_policy["memory_centrality_bonus"]
        return MemoryContext(
            active_items=active,
            autobiographical_pressure=max(0.0, min(1.0, active / 8.0)),
            semantic_centrality=round(centrality, 4),
            memory_conflict=max(0.0, min(1.0, working / 128.0)),
        )

    def _build_will_snapshot(self) -> WillStateSnapshot:
        try:
            from core.will import get_will

            will = get_will()
            status = will.get_status() if hasattr(will, "get_status") else {}
            recent = will.get_recent_decisions(1) if hasattr(will, "get_recent_decisions") else []
            return WillStateSnapshot(
                confidence=float(status.get("confidence", 0.7) or 0.7),
                assertiveness=float(status.get("assertiveness", 0.5) or 0.5),
                refusal_pressure=float(status.get("refuse_rate", 0.0) or 0.0),
                last_receipt_id=str(recent[-1].get("receipt_id", "") if recent else ""),
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("being_runtime", exc)
            return WillStateSnapshot()

    def _publish(self, now: AuraNow) -> None:
        try:
            from core.container import ServiceContainer

            ServiceContainer.register_instance("aura_now", now, required=False)
            ServiceContainer.register_instance("being_runtime", self, required=False)
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("being_runtime", exc)
            logger.debug("AuraNow publish skipped: %s", exc)

    def prompt_block(self, state: Any | None = None, *, objective: str = "") -> str:
        now = self.sample(state, objective=objective)
        return now.compact_prompt_block() + self.renderer.render_prompt_block(now)


_RUNTIME: BeingRuntime | None = None


def get_being_runtime() -> BeingRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = BeingRuntime()
    return _RUNTIME


def reset_being_runtime_for_test() -> None:
    global _RUNTIME
    if _RUNTIME is not None:
        _RUNTIME.stop()
    _RUNTIME = None
