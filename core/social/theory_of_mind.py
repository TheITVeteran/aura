"""core/social/theory_of_mind.py
Theory of Mind parser modeling separate operator beliefs.
"""
from typing import Dict, Any


class TheoryOfMindModel:
    """Estimates the differences between user beliefs and actual system state."""

    def __init__(self):
        # Maps person -> beliefs dict
        self._mental_models: Dict[str, Dict[str, Any]] = {}

    def update_user_belief(self, person: str, key: str, value: Any) -> None:
        if person not in self._mental_models:
            self._mental_models[person] = {}
        self._mental_models[person][key] = value

    def check_belief_discrepancy(self, person: str, actual_state: Dict[str, Any]) -> Dict[str, Any]:
        """Identifies discrepancies between user expectations and system reality."""
        user_beliefs = self._mental_models.get(person, {})
        discrepancies = {}
        
        for key, user_val in user_beliefs.items():
            if key in actual_state and actual_state[key] != user_val:
                discrepancies[key] = {
                    "user_believed": user_val,
                    "actual_value": actual_state[key]
                }
        return discrepancies
