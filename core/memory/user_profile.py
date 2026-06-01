"""User Profile System - Persistent model of the user across conversations

Stores and updates:
- Preferences (response format, style, interests)
- Characteristics (background, expertise, learning style)
- Learnings (facts about user discovered from conversations)
- Relationship history (key moments, shared interests)
- Behavioral patterns (what works, what doesn't)

Updated continuously from SemanticFacts extracted from conversations.
Queryable for context injection into new conversations.
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.runtime.errors import record_degradation
from core.utils.exceptions import capture_and_log

logger = logging.getLogger("Memory.UserProfile")


@dataclass
class ProfileFact:
    """A fact stored in user profile with tracking."""
    category: str  # "preference", "characteristic", "learning", "relationship"
    key: str       # "response_format", "timezone", "expertise", etc
    value: str
    confidence: float = 0.8
    last_updated: float = field(default_factory=time.time)
    source_fact_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UserProfile:
    """Persistent model of a user's preferences, characteristics, and learnings."""
    
    _instance: Optional["UserProfile"] = None
    _lock = asyncio.Lock()
    
    def __init__(self, storage_path: Optional[str] = None):
        self._storage_path = Path(
            storage_path or (Path.home() / ".aura" / "data" / "user_profile.json")
        )
        self._profile_data: Dict[str, List[ProfileFact]] = {
            "preferences": [],
            "characteristics": [],
            "learnings": [],
            "relationship": [],
        }
        self._load_from_disk()
    
    @classmethod
    async def get_instance(cls, storage_path: Optional[str] = None) -> "UserProfile":
        """Get or create singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = UserProfile(storage_path)
        return cls._instance
    
    def _load_from_disk(self):
        """Load user profile from disk if it exists."""
        try:
            if self._storage_path.exists():
                with open(self._storage_path, 'r') as f:
                    data = json.load(f)
                    # Deserialize back to ProfileFact objects
                    for category, facts_list in data.items():
                        if category in self._profile_data:
                            self._profile_data[category] = [
                                ProfileFact(**fact) for fact in facts_list
                            ]
                logger.debug(f"✓ Loaded user profile from {self._storage_path}")
        except Exception as e:
            record_degradation("user_profile", e)
            logger.debug("Failed to load user profile: %s", e)
    
    def _save_to_disk(self):
        """Persist user profile to disk."""
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._storage_path, 'w') as f:
                # Serialize to JSON
                data = {
                    category: [fact.to_dict() for fact in facts]
                    for category, facts in self._profile_data.items()
                }
                json.dump(data, f, indent=2)
            logger.debug(f"✓ Saved user profile to {self._storage_path}")
        except Exception as e:
            record_degradation("user_profile", e)
            logger.warning("Failed to save user profile: %s", e)
    
    def add_or_update_fact(
        self,
        category: str,
        key: str,
        value: str,
        confidence: float = 0.8,
        source_fact_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add or update a fact in the user profile.
        
        Args:
            category: "preferences", "characteristics", "learnings", "relationship"
            key: Fact key (e.g., "response_format", "timezone")
            value: Fact value
            confidence: 0-1 confidence score
            source_fact_id: ID of source SemanticFact
            metadata: Additional metadata
            
        Returns:
            True if added/updated
        """
        if category not in self._profile_data:
            logger.warning(f"Unknown profile category: {category}")
            return False
        
        metadata = metadata or {}
        
        # Check if fact already exists
        existing = None
        for fact in self._profile_data[category]:
            if fact.key == key:
                existing = fact
                break
        
        if existing:
            # Update existing fact if new one is more confident
            if confidence >= existing.confidence:
                existing.value = value
                existing.confidence = confidence
                existing.last_updated = time.time()
                existing.source_fact_id = source_fact_id
                existing.metadata.update(metadata)
                logger.debug(f"✓ Updated {category}.{key} = '{value}' (confidence={confidence:.2f})")
            else:
                logger.debug(f"Skipped lower-confidence update for {category}.{key}")
                return False
        else:
            # Add new fact
            new_fact = ProfileFact(
                category=category,
                key=key,
                value=value,
                confidence=confidence,
                source_fact_id=source_fact_id,
                metadata=metadata,
            )
            self._profile_data[category].append(new_fact)
            logger.debug(f"✓ Added {category}.{key} = '{value}' (confidence={confidence:.2f})")
        
        # Persist changes
        self._save_to_disk()
        return True
    
    def get_fact(self, category: str, key: str) -> Optional[ProfileFact]:
        """Retrieve a specific fact."""
        for fact in self._profile_data.get(category, []):
            if fact.key == key:
                return fact
        return None
    
    def get_facts_by_category(self, category: str) -> List[ProfileFact]:
        """Get all facts in a category."""
        return self._profile_data.get(category, [])
    
    def get_high_confidence_facts(self, category: Optional[str] = None, threshold: float = 0.75) -> List[ProfileFact]:
        """Get facts above confidence threshold."""
        facts = []
        
        if category:
            categories = [category]
        else:
            categories = self._profile_data.keys()
        
        for cat in categories:
            for fact in self._profile_data.get(cat, []):
                if fact.confidence >= threshold:
                    facts.append(fact)
        
        return sorted(facts, key=lambda f: f.confidence, reverse=True)
    
    def to_context_block(self) -> str:
        """Format user profile as context block for LLM injection."""
        blocks = []
        
        # Preferences block
        prefs = self.get_facts_by_category("preferences")
        if prefs:
            pref_lines = [f"- {fact.key}: {fact.value}" for fact in prefs]
            blocks.append(f"[User Preferences]\n" + "\n".join(pref_lines))
        
        # Characteristics block
        chars = self.get_facts_by_category("characteristics")
        if chars:
            char_lines = [f"- {fact.value}" for fact in chars]
            blocks.append(f"[About the User]\n" + "\n".join(char_lines))
        
        # Learnings block
        learnings = self.get_facts_by_category("learnings")
        if learnings:
            learning_lines = [f"- {fact.value}" for fact in learnings]
            blocks.append(f"[Known About User]\n" + "\n".join(learning_lines))
        
        # Relationship block
        rel = self.get_facts_by_category("relationship")
        if rel:
            rel_lines = [f"- {fact.value}" for fact in rel]
            blocks.append(f"[Relationship Notes]\n" + "\n".join(rel_lines))
        
        return "\n\n".join(blocks) if blocks else ""
    
    def summary(self) -> str:
        """Get a human-readable summary of the profile."""
        lines = ["=== User Profile ==="]
        
        for category, facts in self._profile_data.items():
            if facts:
                lines.append(f"\n{category.upper()}:")
                for fact in sorted(facts, key=lambda f: f.confidence, reverse=True):
                    lines.append(f"  • {fact.key}: {fact.value} ({fact.confidence:.0%})")
        
        if sum(len(f) for f in self._profile_data.values()) == 0:
            lines.append("\n(No profile data yet)")
        
        return "\n".join(lines)
