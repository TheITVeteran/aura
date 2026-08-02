"""Aura Self-Profile System - Persistent model of Aura's identity and capabilities

Stores and updates:
- Learned capabilities ("I'm good at debugging Python")
- Communication style patterns ("I prefer detailed explanations")
- Relationship history with user (shared moments, promises, inside jokes)
- Emotional state patterns ("I feel energized by novel problems")
- Learned limitations ("I struggle with audio processing")

Updated continuously from self-learning facts extracted from conversations.
Queryable for identity coherence and relationship continuity.
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Memory.AuraSelfProfile")
_PROFILE_PERSISTENCE_ERRORS = (
    AttributeError,
    json.JSONDecodeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass
class SelfProfileFact:
    """A fact about Aura stored with confidence and provenance."""
    category: str  # "capability", "style", "relationship", "emotional_pattern", "limitation"
    key: str       # "good_at_debugging", "prefers_detail", "shared_starship_dream"
    value: str
    confidence: float = 0.8
    last_updated: float = field(default_factory=time.time)
    evidence_count: int = 1  # How many times this has been reinforced
    source_fact_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuraSelfProfile:
    """Persistent model of Aura's identity, capabilities, and relationship with user."""
    
    _instance: Optional["AuraSelfProfile"] = None
    _lock = asyncio.Lock()
    
    def __init__(self, storage_path: Optional[str] = None):
        self._storage_path = Path(
            storage_path or (state_root() / "data" / "aura_self_profile.json")
        )
        self._profile_data: Dict[str, List[SelfProfileFact]] = {
            "capability": [],
            "style": [],
            "relationship": [],
            "emotional_pattern": [],
            "limitation": [],
        }
        self._load_from_disk()
    
    @classmethod
    async def get_instance(cls, storage_path: Optional[str] = None) -> "AuraSelfProfile":
        """Get or create singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = AuraSelfProfile(storage_path)
        return cls._instance
    
    def _load_from_disk(self):
        """Load Aura profile from disk if it exists."""
        try:
            if self._storage_path.exists():
                with open(self._storage_path, 'r') as f:
                    data = json.load(f)
                    # Deserialize back to SelfProfileFact objects
                    for category, facts_list in data.items():
                        if category in self._profile_data:
                            self._profile_data[category] = [
                                SelfProfileFact(**fact) for fact in facts_list
                            ]
                logger.debug(f"✓ Loaded Aura self-profile from {self._storage_path}")
        except _PROFILE_PERSISTENCE_ERRORS as e:
            record_degradation("aura_self_profile", e)
            logger.debug("Failed to load Aura self-profile: %s", e)
    
    def _save_to_disk(self):
        """Persist Aura profile to disk."""
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                category: [fact.to_dict() for fact in facts]
                for category, facts in self._profile_data.items()
            }
            atomic_write_text(self._storage_path, json.dumps(data, indent=2))
            logger.debug(f"✓ Saved Aura self-profile to {self._storage_path}")
        except _PROFILE_PERSISTENCE_ERRORS as e:
            record_degradation("aura_self_profile", e)
            logger.warning("Failed to save Aura self-profile: %s", e)
    
    def add_or_reinforce_fact(
        self,
        category: str,
        key: str,
        value: str,
        confidence: float = 0.8,
        source_fact_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add or reinforce a fact about Aura's identity.
        
        Args:
            category: Type of fact
            key: Fact identifier
            value: Fact value
            confidence: 0-1 confidence
            source_fact_id: Source SemanticFact ID
            metadata: Additional data
            
        Returns:
            True if added/updated
        """
        if category not in self._profile_data:
            logger.warning(f"Unknown self-profile category: {category}")
            return False
        
        metadata = metadata or {}
        
        # Check if fact already exists
        existing = None
        for fact in self._profile_data[category]:
            if fact.key == key:
                existing = fact
                break
        
        if existing:
            # Reinforce existing fact
            existing.evidence_count += 1
            # Gradually increase confidence with repetition
            existing.confidence = min(0.99, existing.confidence + 0.05)
            existing.last_updated = time.time()
            if source_fact_id:
                if source_fact_id not in existing.source_fact_ids:
                    existing.source_fact_ids.append(source_fact_id)
            existing.metadata.update(metadata)
            logger.debug(f"✓ Reinforced capability: {key} (evidence={existing.evidence_count}, confidence={existing.confidence:.2f})")
        else:
            # Add new fact about self
            new_fact = SelfProfileFact(
                category=category,
                key=key,
                value=value,
                confidence=confidence,
                source_fact_ids=[source_fact_id] if source_fact_id else [],
                metadata=metadata,
            )
            self._profile_data[category].append(new_fact)
            logger.debug(f"✓ Added self-fact: {key} = '{value}' (confidence={confidence:.2f})")
        
        # Persist changes
        self._save_to_disk()
        return True
    
    def get_fact(self, category: str, key: str) -> Optional[SelfProfileFact]:
        """Retrieve a specific fact."""
        for fact in self._profile_data.get(category, []):
            if fact.key == key:
                return fact
        return None
    
    def get_facts_by_category(self, category: str) -> List[SelfProfileFact]:
        """Get all facts in a category."""
        return self._profile_data.get(category, [])
    
    def get_strong_capabilities(self, threshold: float = 0.75) -> List[SelfProfileFact]:
        """Get learned capabilities above confidence threshold."""
        facts = []
        for fact in self._profile_data.get("capability", []):
            if fact.confidence >= threshold:
                facts.append(fact)
        return sorted(facts, key=lambda f: f.confidence, reverse=True)
    
    def get_relationship_memories(self) -> List[SelfProfileFact]:
        """Get relationship history facts."""
        return sorted(
            self._profile_data.get("relationship", []),
            key=lambda f: f.last_updated,
            reverse=True
        )
    
    def to_identity_block(self) -> str:
        """Format self-profile as identity reinforcement block for LLM."""
        blocks = []
        
        # Capabilities block
        caps = self.get_strong_capabilities(threshold=0.7)
        if caps:
            cap_lines = [f"- {fact.value}" for fact in caps]
            blocks.append(f"[My Learned Capabilities]\n" + "\n".join(cap_lines))
        
        # Communication style
        styles = self.get_facts_by_category("style")
        if styles:
            style_lines = [f"- {fact.value}" for fact in styles]
            blocks.append(f"[My Communication Style]\n" + "\n".join(style_lines))
        
        # Relationship memories
        rels = self.get_relationship_memories()
        if rels:
            rel_lines = [f"- {fact.value}" for fact in rels[:3]]  # Top 3
            blocks.append(f"[Our Relationship]\n" + "\n".join(rel_lines))
        
        return "\n\n".join(blocks) if blocks else ""
    
    def summary(self) -> str:
        """Get a human-readable summary of Aura's self-model."""
        lines = ["=== Aura Self-Profile ==="]
        
        for category, facts in self._profile_data.items():
            if facts:
                lines.append(f"\n{category.upper()}:")
                for fact in sorted(facts, key=lambda f: f.confidence, reverse=True):
                    lines.append(f"  • {fact.value} ({fact.confidence:.0%}, {fact.evidence_count}x confirmed)")
        
        if sum(len(f) for f in self._profile_data.values()) == 0:
            lines.append("\n(No self-model data yet)")
        
        return "\n".join(lines)
