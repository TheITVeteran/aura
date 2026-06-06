"""Theory-of-mind state for modeling separate operator beliefs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


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


@dataclass
class BeliefState:
    """Aura's current believed facts for lightweight ToM correction flows."""

    beliefs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrustState:
    """Mutable trust calibration for belief-correction interactions."""

    trust: float = 1.0

    def lower(self, amount: float = 0.05) -> None:
        self.trust = max(0.0, min(1.0, self.trust - amount))


class PerspectiveSimulator:
    """Compares Aura-known facts with an operator's modeled beliefs."""

    def __init__(self, *, person: str = "user"):
        self.person = person
        self.aura_beliefs: Dict[str, Any] = {}
        self.user_model = TheoryOfMindModel()

    def aura_knows(self, key: str, value: Any) -> None:
        self.aura_beliefs[str(key)] = value

    def user_believes(self, key: str, value: Any) -> None:
        self.user_model.update_user_belief(self.person, str(key), value)

    def divergence(self, key: str) -> Optional[Dict[str, Any]]:
        key = str(key)
        user_beliefs = self.user_model._mental_models.get(self.person, {})
        aura_has = key in self.aura_beliefs
        user_has = key in user_beliefs
        if aura_has and user_has and self.aura_beliefs[key] != user_beliefs[key]:
            return {
                "kind": "false_belief",
                "key": key,
                "aura_value": self.aura_beliefs[key],
                "user_value": user_beliefs[key],
            }
        if aura_has and not user_has:
            return {
                "kind": "knowledge_gap",
                "key": key,
                "aura_value": self.aura_beliefs[key],
            }
        return None


class TheoryOfMindEngine:
    """Operational ToM facade used by explanation and correction flows."""

    def __init__(self, *, person: str = "user"):
        self.simulator = PerspectiveSimulator(person=person)
        self.belief = BeliefState()
        self.trust = TrustState()

    def explanation_strategy(self, key: str) -> str:
        divergence = self.simulator.divergence(key)
        if not divergence:
            return "confirm_shared_context"
        kind = divergence.get("kind")
        if kind == "false_belief":
            return "respectfully_correct_false_belief"
        if kind == "knowledge_gap":
            return "explain_from_first_principles"
        return "collaborative_clarification"

    def record_correction(self, *, key: str, correct_value: Any) -> None:
        self.belief.beliefs[str(key)] = correct_value
        self.simulator.aura_knows(str(key), correct_value)
        self.trust.lower()
