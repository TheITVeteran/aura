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
from .blind_introspection import BlindIntrospector
from .body_state_service import BodyStateService
from .continuous_substrate import ContinuousSelfField
from .functional_soul import FunctionalSoul
from .higher_order_monitor import HigherOrderMonitor
from .interoceptive_model import InteroceptiveModel
from .introspection_renderer import IntrospectionRenderer
from .self_ownership import OwnershipTracker
from .self_report_calibrator import SelfReportCalibrator
from .semantic_stream import SemanticStream
from .welfare_state import WelfareState
from .workspace_ignition import WorkspaceIgnition

logger = logging.getLogger("Aura.BeingRuntime")


class BeingRuntime:
    """Canonical LAMP/AuraNow runtime surface.

    Welfare, body, semantic stream, blind introspection, and self-report
    calibration are wired into every sample() call. The BeingRuntime
    action-constraint path consumes these signals before consequential
    behavior.
    """

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

        # Welfare/body/introspection subsystems.
        self.body_service = BodyStateService.get()
        self.welfare = WelfareState.get()
        self.blind_introspector = BlindIntrospector()
        self.self_report_calibrator = SelfReportCalibrator()
        self.semantic_stream = SemanticStream.get()
        self._last_welfare: Any | None = None
        self._last_blind_report: Any | None = None
        self._last_body_snapshot: Any | None = None
        self._last_causal_self_vector: Any | None = None
        self._last_causal_valenced_workspace: Any | None = None
        self._lesion_controller_registered = False

    def start(self, *, hz: float = 20.0) -> None:
        if self._running:
            return
        self.field.start(hz=hz)
        self._register_lesion_targets()
        self._running = True

    def stop(self) -> None:
        self.field.stop()
        self._running = False

    def _register_lesion_targets(self) -> None:
        """Register all subsystems with the canonical LesionController."""
        if self._lesion_controller_registered:
            return
        try:
            from core.runtime.lesion_controller import LesionController
        except (ImportError, RuntimeError) as exc:
            record_degradation(
                "being_runtime",
                exc,
                severity="warning",
                action="deferred lesion-controller registration during BeingRuntime start",
            )
            logger.warning("LesionController registration deferred: %s", exc)
            return

        ctrl = LesionController.get()
        failed_targets: list[str] = []
        for name, subsystem in (
            ("welfare", self.welfare),
            ("body", self.body_service),
            ("introspection", self.blind_introspector),
            ("self_report", self.self_report_calibrator),
            ("semantic_stream", self.semantic_stream),
            ("affect", self.affect),
            ("workspace", self.workspace),
        ):
            try:
                ctrl.register(name, subsystem)
            except (AttributeError, TypeError, ValueError) as exc:
                failed_targets.append(name)
                record_degradation(
                    "being_runtime",
                    exc,
                    severity="warning",
                    action=f"left lesion target {name} unregistered until interface is fixed",
                    extra={"lesion_target": name},
                )

        if not failed_targets:
            self._lesion_controller_registered = True
        else:
            logger.warning("Lesion target registration incomplete: %s", failed_targets)

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

        # Feed raw body state into the body service before welfare is computed.
        self.body_service.update_body(body)
        body_snapshot = self.body_service.snapshot()

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

        self_state = self._build_self_state(state, lesions)
        memory_context = self._build_memory_context(state, lesions)

        # Compute welfare from body, affect, prediction, memory, and continuity.
        welfare_inputs = self.welfare.gather_inputs(
            body=body_snapshot,
            affect_distress=affect.distress,
            affect_valence=affect.valence,
            prediction_error=prediction.free_energy,
            memory_coherence=1.0 - memory_context.memory_conflict,
            tool_reliability=max(0.0, 1.0 - body.tool_failure_pressure),
            model_stability=1.0,
            continuity_risk=self_state.continuity_risk,
        )
        welfare_outputs = self.welfare.compute(welfare_inputs)

        # Update the semantic stream with current welfare/body state.
        self.semantic_stream.update_welfare(
            welfare_score=welfare_outputs.welfare_score,
            distress=welfare_outputs.distress,
            fatigue=body_snapshot.fatigue,
            recovery_drive=welfare_outputs.recovery_drive,
            body_health=body_snapshot.operational_health,
        )
        if objective:
            self.semantic_stream.update_situation("active_task")
        self.semantic_stream.evolve()

        # Classify internal state from control variables, without label prompting.
        blind_trace = self.blind_introspector.build_trace(
            distress=welfare_outputs.distress,
            body_pressure=body_snapshot.total_pressure,
            prediction_error=prediction.free_energy,
            memory_coherence=welfare_inputs.memory_coherence,
            tool_reliability=welfare_inputs.tool_reliability,
            goal_frustration=welfare_inputs.goal_frustration,
            social_trust=welfare_inputs.social_trust,
            continuity_risk=welfare_inputs.continuity_risk,
            fatigue=body_snapshot.fatigue,
            recovery_debt=body_snapshot.recovery_debt,
            curiosity=affect.curiosity,
            confidence=welfare_outputs.confidence,
        )
        blind_report = self.blind_introspector.introspect(blind_trace)

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
        self._last_welfare = welfare_outputs
        self._last_blind_report = blind_report
        self._last_body_snapshot = body_snapshot
        try:
            from core.being.causal_self_state import vector_from_aura_now

            causal_vector = vector_from_aura_now(
                now,
                welfare_outputs=welfare_outputs,
                blind_report=blind_report,
            )
            self._last_causal_self_vector = causal_vector
            self._last_causal_valenced_workspace = causal_vector.causal_valenced_workspace
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "being_runtime",
                exc,
                severity="warning",
                action="continued without causal valenced workspace vector for this sample",
            )
        self._publish(now)
        return now

    def action_policy(
        self,
        now: AuraNow,
        *,
        domain: str = "",
        priority: float = 0.5,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Derive action constraints from the live AuraNow state + welfare.

        This is the operational bridge between the "inner life" substrate and
        behavior. Welfare, body, affect, active-inference prediction,
        workspace ignition, and ownership model MUST change consequential
        decisions — not merely decorate prompts.

        MANDATORY: Body cost is paid for every consequential action.
        """
        domain_name = str(domain or "").strip().lower()
        context = dict(context or {})
        continuity_memory_write = bool(
            domain_name == "memory_write"
            and (
                context.get("conversation_continuity")
                or context.get("explicit_observational_memory_write")
            )
            and not context.get("high_risk_memory_write")
        )
        foreground_continuity_state = bool(
            domain_name == "state_mutation"
            and context.get("foreground_continuity_state")
        )
        explicit_foreground_desktop_tool = bool(
            domain_name == "tool_execution"
            and context.get("desktop_execution_contract")
            and context.get("foreground_request")
            and context.get("user_explicitly_authorized")
            and context.get("user_visible_desktop_action")
            and context.get("verification_required")
        )
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

        # Welfare-driven constraints.
        welfare = getattr(self, "_last_welfare", None)
        if welfare is not None:
            if welfare.action_inhibition > 0.5 and consequential and not repair_lane:
                defers.append(f"welfare_action_inhibition={welfare.action_inhibition:.3f}")
                constraints.append(f"welfare_inhibition_high: {welfare.action_inhibition:.3f}")

            if welfare.should_protect_integrity() and domain_name in {
                "memory_write", "self_modification", "belief_update",
            }:
                constraints.append(f"welfare_integrity_guard={welfare.integrity_guard:.3f}")
                if welfare.integrity_guard > 0.8 and not repair_lane:
                    defers.append("welfare_integrity_protection_active")

            if welfare.recovery_drive > 0.6 and consequential and not repair_lane:
                constraints.append(f"welfare_recovery_drive={welfare.recovery_drive:.3f}")
                defers.append("welfare_recovery_required_before_action")

            if welfare.should_verify_before_claiming() and domain_name == "response":
                constraints.append(f"welfare_verify_before_claim: self_report_conf={welfare.self_report_confidence:.3f}")

        # Every consequential action must pay a body cost. If accounting fails,
        # the action is constrained/deferred instead of silently proceeding.
        body_cost: dict[str, float] = {}
        if consequential:
            try:
                body_cost = self.body_service.spend(domain_name, cost_multiplier=priority)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation(
                    "being_runtime",
                    exc,
                    severity="degraded",
                    action="constrained consequential action after body-cost accounting failure",
                    extra={"domain": domain_name, "priority": priority},
                )
                constraints.append("body_cost_accounting_failed")
                if not repair_lane:
                    defers.append("body_cost_accounting_required_before_action")

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

        if explicit_foreground_desktop_tool and defers and not blocks:
            desktop_soft_defers = {
                "action_controllability_too_low",
                "workspace_not_ignited",
                "no_workspace_broadcast_for_consequential_action",
                "prediction_error_requires_observation_or_plan",
            }
            hard_defers = [item for item in defers if item not in desktop_soft_defers]
            if not hard_defers:
                constraints.append("foreground_desktop_action_constrained:not_deferred")
                constraints.extend(f"foreground_desktop_note:{item}" for item in defers[:4])
                defers = []

        if blocks:
            outcome = "refuse"
        elif defers:
            if continuity_memory_write or foreground_continuity_state:
                constraints.append(
                    "continuity_memory_write_constrained:not_deferred"
                    if continuity_memory_write
                    else "foreground_state_commit_constrained:not_deferred"
                )
                note_prefix = "continuity_memory_note" if continuity_memory_write else "foreground_state_note"
                constraints.extend(f"{note_prefix}:{item}" for item in defers[:4])
                outcome = "constrain"
            else:
                outcome = "defer"
        elif constraints:
            outcome = "constrain"
        else:
            outcome = "proceed"

        last_body_snapshot = getattr(self, "_last_body_snapshot", None)
        body_fatigue = float(getattr(last_body_snapshot, "fatigue", 0.0) or 0.0)

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
                "body_cost_applied": {key: round(float(value), 4) for key, value in body_cost.items()},
                "welfare_score": round(welfare.welfare_score, 4) if welfare else 0.5,
                "welfare_distress": round(welfare.distress, 4) if welfare else 0.0,
                "welfare_integrity_guard": round(welfare.integrity_guard, 4) if welfare else 0.5,
                "welfare_truth_protection": round(welfare.truth_protection, 4) if welfare else 0.5,
                "welfare_action_inhibition": round(welfare.action_inhibition, 4) if welfare else 0.0,
                "welfare_recovery_drive": round(welfare.recovery_drive, 4) if welfare else 0.0,
                "welfare_self_report_confidence": round(welfare.self_report_confidence, 4) if welfare else 0.5,
                "body_fatigue": round(body_fatigue, 4),
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
        """Full prompt block: AuraNow + renderer + semantic stream + blind introspection."""
        now = self.sample(state, objective=objective)
        parts = [now.compact_prompt_block()]
        organismal_block = self.organismal_workspace_prompt_block()
        if organismal_block:
            parts.append(organismal_block)

        # Semantic stream context.
        parts.append(self.semantic_stream.state.to_prompt_block())

        # Blind introspection (structured, non-persona).
        blind = getattr(self, "_last_blind_report", None)
        if blind and blind.confidence > 0:
            parts.append(
                f"## BLIND INTROSPECTION (non-persona)\n"
                f"- State: {blind.predicted_state_class} (confidence={blind.confidence:.2f})\n"
                f"- Behavior: {', '.join(blind.expected_behavior_shifts)}\n"
                f"- Welfare estimate: {blind.welfare_estimate:.2f}\n"
                f"- Urgency: {blind.urgency:.2f}\n\n"
            )

        # Welfare summary.
        welfare = getattr(self, "_last_welfare", None)
        if welfare:
            parts.append(
                f"## WELFARE STATE\n"
                f"- Score: {welfare.welfare_score:.2f}, Distress: {welfare.distress:.2f}\n"
                f"- Integrity guard: {welfare.integrity_guard:.2f}, "
                f"Truth protection: {welfare.truth_protection:.2f}\n"
                f"- Recovery drive: {welfare.recovery_drive:.2f}, "
                f"Action inhibition: {welfare.action_inhibition:.2f}\n"
                f"- Self-report confidence: {welfare.self_report_confidence:.2f}\n\n"
            )

        # State-grounded introspection.
        parts.append(self.renderer.render_prompt_block(now))

        return "".join(parts)

    def organismal_workspace_prompt_block(self, *, compact: bool = False) -> str:
        """Return the latest Causal Valenced Workspace block for prompts."""
        cvw = getattr(self, "_last_causal_valenced_workspace", None)
        if cvw is None or not hasattr(cvw, "prompt_block"):
            return ""
        return cvw.prompt_block(compact=compact)


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
