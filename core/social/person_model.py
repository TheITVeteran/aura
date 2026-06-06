"""core/social/person_model.py
Person Model coordinating social mind representations.
"""
from typing import Dict, Any, Optional

from core.social.relationship_graph import RelationshipGraph
from core.social.trust_model import TrustModel
from core.social.user_preference_model import UserPreferenceModel
from core.social.theory_of_mind import TheoryOfMindModel
from core.social.social_memory import SocialMemoryStore
from core.social.reciprocity_engine import ReciprocityEngine
from core.social.boundary_respect import BoundaryRespectChecker


class PersonModel:
    """Canonical representation of the human operator (Bryan)."""

    def __init__(self, name: str = "Bryan"):
        self.name = name
        self.relationships = RelationshipGraph()
        self.trust = TrustModel()
        self.preferences = UserPreferenceModel()
        self.theory_of_mind = TheoryOfMindModel()
        self.memory = SocialMemoryStore()
        self.reciprocity = ReciprocityEngine()
        self.boundary = BoundaryRespectChecker()

        # Seed Bryan in relations
        self.relationships.add_person(name, {"role": "owner_operator"})

    def get_social_status(self) -> Dict[str, Any]:
        """Consolidates current social attributes for the operator."""
        return {
            "name": self.name,
            "trust_score": self.trust.get_trust(self.name),
            "verbosity_pref": self.preferences.get_preference(self.name, "verbosity", "concise"),
            "reciprocity_index": self.reciprocity.get_reciprocity_index(self.name)
        }

    def validate_action(self, channel: str, params: Dict[str, Any]) -> bool:
        """Helper to verify actions don't violate boundaries."""
        return not self.boundary.check_boundary_violation(channel, params)
