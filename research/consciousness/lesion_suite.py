"""research/consciousness/lesion_suite.py
Lesion suite measuring behavioral degradation when specific modules are disabled.
"""
from typing import Dict, Any, List
from core.organism.life_state import LifeState
from core.welfare.lesion_tests import WelfareLesionSuite


class LesionBehaviorTester:
    """Verifies that welfare/memory lesions alter execution patterns correctly."""

    def __init__(self):
        self.suite = WelfareLesionSuite()

    def run_lesion_behavior_test(self, state: LifeState) -> Dict[str, Any]:
        # Lock energy to extremely low value
        self.suite.apply_lesion("energy", 5.0)

        # Trigger welfare evaluation
        from core.welfare.welfare_bus import WelfareBus
        bus = WelfareBus()
        bus.lesions = self.suite
        
        # Run test
        import asyncio
        asyncio.run(bus.evaluate_welfare(state))

        # Check if the policy limits adapted
        policy_limits = state.world_model.get("active_policy_limits", {})
        max_allowed_risk = policy_limits.get("max_tool_risk", 5)

        # Restore baseline
        self.suite.clear_lesions()

        # Success condition: low energy must restrict max tool risk
        passed = max_allowed_risk <= 2
        return {
            "test_name": "lesion_low_energy_behavior_adaptation",
            "active_risk_limit_under_lesion": max_allowed_risk,
            "passed": passed
        }
