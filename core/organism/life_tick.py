"""core/organism/life_tick.py
Execution steps for a single tick of the canonical organism loop.
Maps: Perceive -> Body -> Beliefs -> Goals -> Attention -> Deliberate -> Act -> Verify -> Memory -> Welfare -> Values -> Identity -> Sleep.
"""
import logging
import time
from typing import Dict, Any, Optional

from core.organism.life_state import LifeState
from core.runtime.errors import record_degradation

logger = logging.getLogger("Organism.LifeTick")

_LIFE_TICK_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


class LifeTickProcessor:
    """Coordinates the exact chronological transitions of a single life tick."""

    def __init__(self, container: Optional[Any] = None):
        self.container = container

    async def execute_tick(self, state: LifeState) -> None:
        """Executes the full 13-stage organic pipeline."""
        state.timestamp = time.time()
        state.tick_count += 1

        # 1. Perceive world
        await self._perceive_world(state)

        # 2. Update body/interoception
        await self._update_body(state)

        # 3. Update beliefs/world model
        await self._update_beliefs(state)

        # 4. Update goals/drives/preferences
        await self._update_goals_drives_preferences(state)

        # 5. Choose attention
        await self._choose_attention(state)

        # 6. Deliberate
        await self._deliberate(state)

        # 7. Act or inhibit action
        action_intent = await self._determine_action(state)
        receipt = None
        if action_intent:
            receipt = await self._act_or_inhibit(state, action_intent)

        # 8. Verify consequence
        if receipt:
            await self._verify_consequence(state, receipt)

        # 9. Update memory
        await self._update_memory(state, receipt)

        # 10. Update welfare
        await self._update_welfare(state)

        # 11. Update values
        await self._update_values(state)

        # 12. Consolidate identity
        await self._consolidate_identity(state)

        # 13. Sleep/dream/train/repair
        await self._offline_cycle(state)

    async def _perceive_world(self, state: LifeState) -> None:
        try:
            # Poll screen/processes/accessibility sensors
            from core.body.body_runtime import get_body_runtime
            body = get_body_runtime()
            observations = await body.perceive_all()
            state.world_model["last_observations"] = observations
        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.perceive", exc)
            logger.warning("Perception tick step failed: %s", exc)

    async def _update_body(self, state: LifeState) -> None:
        try:
            from core.body.body_runtime import get_body_runtime
            body = get_body_runtime()
            status = await body.get_system_status()
            state.body.battery_level = status.get("battery", 100.0)
            state.body.cpu_usage = status.get("cpu", 10.0)
            state.body.memory_usage = status.get("memory", 50.0)
            state.body.current_focus_app = status.get("focus_app", "Terminal")
            state.body.clipboard_content = status.get("clipboard", "")
        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.body", exc)

    async def _update_beliefs(self, state: LifeState) -> None:
        try:
            # Reconstruct causal and entity graphs based on observations
            from core.world.belief_revision import BeliefRevisionEngine
            engine = BeliefRevisionEngine()
            await engine.revise_beliefs(state)
        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.beliefs", exc)

    async def _update_goals_drives_preferences(self, state: LifeState) -> None:
        try:
            from core.agency.mission_manager import get_mission_manager
            manager = get_mission_manager()
            await manager.update_goals_and_drives(state)
        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.goals", exc)

    async def _choose_attention(self, state: LifeState) -> None:
        try:
            from core.executive.attention_controller import AttentionController
            controller = AttentionController()
            state.cognition.active_attention = await controller.focus_attention(state)
        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.attention", exc)

    async def _deliberate(self, state: LifeState) -> None:
        try:
            from core.executive.executive_kernel import DeliberationEngine
            engine = DeliberationEngine()
            await engine.deliberate(state)
        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.deliberate", exc)

    async def _determine_action(self, state: LifeState) -> Optional[Dict[str, Any]]:
        if not state.cognition.pending_actions:
            return None
        return state.cognition.pending_actions.pop(0)

    async def _act_or_inhibit(self, state: LifeState, intent: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from core.executive.inhibition_system import ActionInhibitor
            inhibitor = ActionInhibitor()
            if await inhibitor.should_inhibit(state, intent):
                logger.info("Action inhibited by governance/moral safety kernel: %s", intent)
                return {"status": "inhibited", "intent": intent}

            from core.body.action_body import get_action_body
            action_body = get_action_body()
            return await action_body.execute_action(intent, state)
        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.act", exc)
            return {"status": "failed", "error": str(exc), "intent": intent}

    async def _verify_consequence(self, state: LifeState, receipt: Dict[str, Any]) -> None:
        try:
            from core.body.action_postcondition import ActionPostconditionVerifier
            verifier = ActionPostconditionVerifier()
            await verifier.verify(receipt, state)
        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.verify", exc)

    async def _update_memory(self, state: LifeState, receipt: Optional[Dict[str, Any]]) -> None:
        try:
            from core.memory.autobiography import AutobiographyEngine
            engine = AutobiographyEngine()
            await engine.record_tick_event(state, receipt)
        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.memory", exc)

    async def _update_welfare(self, state: LifeState) -> None:
        try:
            from core.welfare.welfare_bus import WelfareBus
            bus = WelfareBus()
            await bus.evaluate_welfare(state)
        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.welfare", exc)

    async def _update_values(self, state: LifeState) -> None:
        try:
            from core.values.preference_provenance import PreferenceProvenanceManager
            manager = PreferenceProvenanceManager()
            await manager.evaluate_preferences(state)
        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.values", exc)

    async def _consolidate_identity(self, state: LifeState) -> None:
        try:
            from core.identity.identity_kernel import IdentityKernel
            kernel = IdentityKernel()
            await kernel.guard_identity_continuity(state)
        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.identity", exc)

    async def _offline_cycle(self, state: LifeState) -> None:
        try:
            from core.sleep.sleep_cycle import SleepManager
            manager = SleepManager()
            if await manager.should_trigger_sleep(state):
                await manager.execute_sleep_cycle(state)
        except _LIFE_TICK_RECOVERABLE_ERRORS as exc:
            record_degradation("organism.sleep", exc)
