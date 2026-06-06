"""core/social/reciprocity_engine.py
Reciprocity engine tracking cooperative trust indexes and interaction balances.
"""
from typing import Dict


class ReciprocityEngine:
    """Estimates interaction fairness score between the human and agent."""

    def __init__(self):
        # Maps person -> balance score (-10.0 to 10.0)
        self._balance: Dict[str, float] = {}

    def get_reciprocity_index(self, person: str) -> float:
        return self._balance.get(person, 0.0)

    def record_transaction(self, person: str, agent_helped: bool, human_helped: bool) -> None:
        current = self.get_reciprocity_index(person)
        delta = 0.0
        if human_helped:
            delta += 1.0
        if agent_helped:
            delta -= 0.5  # Slightly decrease score when agent provides output
            
        self._balance[person] = min(max(current + delta, -10.0), 10.0)
