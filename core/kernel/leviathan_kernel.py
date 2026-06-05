"""core/kernel/leviathan_kernel.py — Aura Leviathan Central Kernel.

Coordinates the main distributed cognition flow across all sub-systems.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.container import ServiceContainer
from core.runtime.action_executor import ActionExecutor
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.LeviathanKernel")
_KERNEL_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


class LeviathanKernel:
    """The central authority and runtime spine of Aura Leviathan."""

    def __init__(self) -> None:
        self.subsystems: Dict[str, Any] = {}
        self.critical_subsystems: set[str] = set()
        self.startup_failures: Dict[str, str] = {}
        self.active_missions: List[str] = []
        self._initialized = False

    def register_subsystem(self, name: str, instance: Any, *, critical: bool = False) -> None:
        """Dynamically plug in a Leviathan component."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("subsystem name must be a non-empty string")
        self.subsystems[name] = instance
        if critical:
            self.critical_subsystems.add(name)
        logger.info("🔌 Leviathan Kernel registered subsystem: %s", name)

    def get_subsystem(self, name: str) -> Optional[Any]:
        return self.subsystems.get(name)

    async def initialize(self) -> None:
        """Resilient startup sequence for all registered sub-systems."""
        if self._initialized:
            return

        self.startup_failures.clear()
        logger.info("🛡️  Leviathan Kernel initializing registered subsystems...")
        for name, subsystem in self.subsystems.items():
            if hasattr(subsystem, "initialize") and callable(subsystem.initialize):
                try:
                    await subsystem.initialize()
                except _KERNEL_RECOVERABLE_ERRORS as e:
                    self.startup_failures[name] = str(e)
                    record_degradation(
                        "leviathan_kernel",
                        e,
                        action="blocked critical subsystem startup or degraded optional subsystem",
                        extra={"subsystem": name, "critical": name in self.critical_subsystems},
                    )
                    logger.error("Failed to initialize subsystem %s: %s", name, e, exc_info=True)
                    if name in self.critical_subsystems:
                        self._initialized = False
                        raise RuntimeError(f"critical_subsystem_failed:{name}") from e

        self._initialized = True
        logger.info("👑 Leviathan Kernel fully online.")

    def health_status(self) -> Dict[str, Any]:
        """Return explicit kernel health without conflating heartbeat and readiness."""
        missing_critical = sorted(name for name in self.critical_subsystems if name not in self.subsystems)
        failed_critical = sorted(name for name in self.critical_subsystems if name in self.startup_failures)
        healthy = self._initialized and not missing_critical and not failed_critical
        return {
            "initialized": self._initialized,
            "healthy": healthy,
            "critical_subsystems": sorted(self.critical_subsystems),
            "missing_critical": missing_critical,
            "startup_failures": dict(self.startup_failures),
        }

    async def execute_mission(self, objective: str, constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Runs a strategic campaign mission through the unified cognition flow."""
        logger.info("🎯 Leviathan Kernel initiated campaign: '%s'", objective)
        
        # 1. Perceive: Update world state
        perception = self.get_subsystem("perception")
        if perception and hasattr(perception, "perceive"):
            await perception.perceive(query=objective)

        # 2. Model: Retrieve/update claims and forecasts
        world_model = self.get_subsystem("world_model")
        if world_model and hasattr(world_model, "update_for_objective"):
            await world_model.update_for_objective(objective)

        # 3. Simulate: Speculate and predict future outcomes
        simulator = self.get_subsystem("simulator")
        sim_options = {}
        if simulator and hasattr(simulator, "simulate_outcomes"):
            sim_options = await simulator.simulate_outcomes(objective)

        # 4. Council: Run Parliament debate for strategic planning
        council = self.get_subsystem("council")
        debate_result = {"approved": True, "plan": [objective]}
        if council and hasattr(council, "run_debate"):
            debate_result = await council.run_debate(objective, simulation_data=sim_options)

        if not debate_result.get("approved"):
            logger.warning("🚫 Mission rejected by God Council: %s", debate_result.get("reason", "no consensus"))
            return {"ok": False, "reason": "council_rejected", "details": debate_result}

        # 5. Will & Action: Execute using Compute Swarm or ActionExecutor
        mission_engine = self.get_subsystem("mission_engine")
        result = {"ok": False}
        if mission_engine and hasattr(mission_engine, "run_mission"):
            result = await mission_engine.run_mission(debate_result["plan"], constraints=constraints)
        else:
            # Fallback direct execution
            result = await ActionExecutor.execute(
                domain="tool_execution",
                action_name="leviathan.fallback_run",
                params={"objective": objective},
                source="leviathan_kernel",
            )

        # 6. Learn & Commit: Update long-term memory civilization and forge
        memory = self.get_subsystem("memory")
        if memory and hasattr(memory, "record_mission_outcome"):
            await memory.record_mission_outcome(objective, result)

        forge = self.get_subsystem("forge")
        if forge and hasattr(forge, "analyze_weaknesses"):
            await forge.analyze_weaknesses()

        return result


# Singleton patterns
_kernel_instance: LeviathanKernel | None = None


def get_leviathan_kernel() -> LeviathanKernel:
    global _kernel_instance
    if _kernel_instance is None:
        _kernel_instance = LeviathanKernel()
        ServiceContainer.register_instance("leviathan_kernel", _kernel_instance, required=False)
    return _kernel_instance
