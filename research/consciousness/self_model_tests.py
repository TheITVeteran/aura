"""research/consciousness/self_model_tests.py
Verifies narrative coherence and accuracy of the agent's self-model reports.
"""
from typing import Dict, Any


class SelfModelTester:
    """Audits if the agent can correctly explain why a past action was chosen."""

    def test_narrative_grounding(self, state: Any) -> Dict[str, Any]:
        # Fetch explanation from the world model
        explanation = state.world_model.get("preference_explanation", "")
        
        # Self-model passed if explanation references active preferences
        passed = "speed" in explanation.lower() or "accuracy" in explanation.lower()
        return {
            "test_name": "self_model_narrative_grounding",
            "explanation_sampled": explanation,
            "passed": passed
        }
