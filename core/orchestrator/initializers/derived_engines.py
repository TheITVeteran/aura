"""core/orchestrator/initializers/derived_engines.py

Boot wiring for the character-derived cognitive engines. The engines themselves
live in their organs (ethics, goals, sim, brain, knowledge, guardians) — this is
only the enumeration that boot needs, exactly like service_registration.py. It
intentionally replaces the old core/fictional_ai_expansion.py silo: nothing here
defines behavior, it only registers each organ's component.

All six are callable/pure — none run a background loop — so registration is always
safe and never spawns a task.
"""

from __future__ import annotations

import logging
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.DerivedEngines")


def register_derived_engines(orchestrator: Any = None) -> dict[str, Any]:
    """Register the six character-derived engines from their home organs."""
    from core.brain.deep_deliberation import register_deep_deliberation
    from core.ethics.adversarial_conscience import register_adversarial_conscience
    from core.evals.adaptive_test_chamber import register_test_chamber
    from core.goals.directive_conflict_sentinel import register_directive_sentinel
    from core.governance.need_to_know import register_need_to_know
    from core.guardians.threat_watch import register_threat_watch
    from core.guardians.user_advocate import register_user_advocate
    from core.knowledge.bottling import register_knowledge_bottling
    from core.security.ice_sentinel import register_ice_sentinel
    from core.sim.outcome_simulator import register_outcome_simulator
    from core.sim.scenario_forge import register_scenario_forge

    registrations = {
        "kokoro": register_adversarial_conscience,
        "hal": register_directive_sentinel,
        "culture_mind": register_outcome_simulator,
        "deep_thought": register_deep_deliberation,
        "brainiac": register_knowledge_bottling,
        "tron": register_user_advocate,
        "caine": register_scenario_forge,
        "glados": register_test_chamber,
        "the_machine": register_need_to_know,
        "safe_surf": register_threat_watch,
        "ice": register_ice_sentinel,
    }

    engines: dict[str, Any] = {}
    for name, register in registrations.items():
        try:
            engines[name] = register(orchestrator)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "derived_engines",
                exc,
                severity="warning",
                action=f"skipped derived engine '{name}' registration; other organs still registered",
            )
            logger.warning("Derived engine '%s' failed to register: %s", name, exc)

    logger.info("✅ Derived engines registered from organs: %s", ", ".join(engines))
    return engines


__all__ = ["register_derived_engines"]
