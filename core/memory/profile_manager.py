"""Profile Manager - Continuously learns and updates user and Aura profiles

Integrates:
1. SemanticFactExtractor - pulls facts from conversations
2. UserProfile - stores user learnings
3. AuraSelfProfile - stores Aura learnings

Runs as background task during chat turns to update both profiles.
"""

import asyncio
import logging
from typing import Optional

from core.memory.semantic_fact_extractor import SemanticFactExtractor, FactType
from core.memory.user_profile import UserProfile
from core.memory.aura_self_profile import AuraSelfProfile
from core.runtime.errors import record_degradation

logger = logging.getLogger("Memory.ProfileManager")


class ProfileManager:
    """Manages continuous profile learning from conversations."""
    
    _instance: Optional["ProfileManager"] = None
    _lock = asyncio.Lock()
    
    def __init__(self):
        self._fact_extractor = SemanticFactExtractor()
        self._user_profile: Optional[UserProfile] = None
        self._aura_profile: Optional[AuraSelfProfile] = None
    
    @classmethod
    async def get_instance(cls) -> "ProfileManager":
        """Get or create singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = ProfileManager()
                    await cls._instance._initialize()
        return cls._instance
    
    async def _initialize(self):
        """Initialize profile instances."""
        try:
            self._user_profile = await UserProfile.get_instance()
            self._aura_profile = await AuraSelfProfile.get_instance()
            logger.debug("✓ ProfileManager initialized")
        except Exception as e:
            record_degradation("profile_manager", e)
            logger.warning("ProfileManager initialization failed: %s", e)
    
    async def learn_from_turn(
        self,
        user_message: str,
        aura_response: str,
        session_id: str = "default",
    ) -> tuple[int, int]:
        """Learn from a conversation turn.
        
        Args:
            user_message: What the user said
            aura_response: Aura's response
            session_id: Current session ID
            
        Returns:
            Tuple of (user_facts_learned, aura_facts_learned)
        """
        if not self._user_profile or not self._aura_profile:
            logger.debug("ProfileManager not initialized, skipping learning")
            return (0, 0)
        
        user_count = 0
        aura_count = 0
        
        try:
            # Extract facts from conversation
            facts = self._fact_extractor.extract_facts(
                user_message=user_message,
                aura_response=aura_response,
                session_id=session_id,
            )
            
            logger.debug(f"Extracted {len(facts)} facts from turn, learning...")
            
            # Process each fact
            for fact in facts:
                try:
                    if fact.fact_type == FactType.USER_PREFERENCE:
                        # Store user preference
                        self._user_profile.add_or_update_fact(
                            category="preferences",
                            key=fact.predicate,
                            value=fact.object,
                            confidence=fact.confidence,
                            metadata={"source": fact.source_text[:100]},
                        )
                        user_count += 1
                    
                    elif fact.fact_type == FactType.USER_CHARACTERISTIC:
                        # Store user characteristic
                        self._user_profile.add_or_update_fact(
                            category="characteristics",
                            key=f"{fact.predicate}_in_{fact.object.replace(' ', '_')}",
                            value=f"The user {fact.predicate} {fact.object}",
                            confidence=fact.confidence,
                            metadata={"predicate": fact.predicate},
                        )
                        user_count += 1
                    
                    elif fact.fact_type == FactType.USER_LEARNING:
                        # Store what user learned
                        self._user_profile.add_or_update_fact(
                            category="learnings",
                            key=f"learned_{fact.object[:20].replace(' ', '_')}",
                            value=f"The user learned that {fact.object}",
                            confidence=fact.confidence,
                            metadata={"learned_recently": True},
                        )
                        user_count += 1
                    
                    elif fact.fact_type == FactType.AURA_LEARNING:
                        # Store what Aura learned about user
                        if "you" in fact.source_text.lower() or "user" in fact.source_text.lower():
                            self._user_profile.add_or_update_fact(
                                category="learnings",
                                key=f"aura_observed_{fact.object[:20].replace(' ', '_')}",
                                value=f"I noticed that you {fact.object}",
                                confidence=fact.confidence,
                                metadata={"aura_observed": True},
                            )
                            user_count += 1
                        else:
                            # Aura learning about herself
                            self._aura_profile.add_or_reinforce_fact(
                                category="capability",
                                key=f"learned_{fact.object[:20].replace(' ', '_')}",
                                value=f"I learned that {fact.object}",
                                confidence=fact.confidence,
                                metadata={"learned_recently": True},
                            )
                            aura_count += 1
                    
                    elif fact.fact_type == FactType.GENERAL_KNOWLEDGE:
                        # Could store in a knowledge graph, for now store in Aura's style
                        if "recommend" in fact.predicate.lower():
                            self._aura_profile.add_or_reinforce_fact(
                                category="style",
                                key=f"recommends_{fact.object[:20].replace(' ', '_')}",
                                value=f"I recommend to {fact.object}",
                                confidence=fact.confidence,
                            )
                            aura_count += 1
                    
                    elif fact.fact_type == FactType.RELATIONSHIP_FACT:
                        # Store relationship memory
                        self._aura_profile.add_or_reinforce_fact(
                            category="relationship",
                            key=f"shared_{fact.object[:20].replace(' ', '_')}",
                            value=f"We {fact.object}",
                            confidence=fact.confidence,
                            metadata={"relationship_memory": True},
                        )
                        aura_count += 1
                
                except Exception as e:
                    record_degradation("profile_manager", e)
                    logger.debug("Failed to process fact: %s", e)
                    continue
            
            if user_count > 0 or aura_count > 0:
                logger.info(f"✓ Learned {user_count} user facts and {aura_count} self facts")
            
            return (user_count, aura_count)
        
        except Exception as e:
            record_degradation("profile_manager", e)
            logger.warning("Profile learning failed: %s", e)
            return (0, 0)
    
    async def get_context_injection(self) -> str:
        """Get formatted context blocks for chat preflight injection."""
        blocks = []
        
        try:
            if self._user_profile:
                user_context = self._user_profile.to_context_block()
                if user_context:
                    blocks.append(user_context)
            
            if self._aura_profile:
                aura_context = self._aura_profile.to_identity_block()
                if aura_context:
                    blocks.append(aura_context)
        except Exception as e:
            record_degradation("profile_manager", e)
            logger.debug("Failed to generate context injection: %s", e)
        
        if blocks:
            return "\n\n".join(blocks)
        return ""
    
    def get_user_profile(self) -> Optional[UserProfile]:
        """Get user profile instance."""
        return self._user_profile
    
    def get_aura_profile(self) -> Optional[AuraSelfProfile]:
        """Get Aura profile instance."""
        return self._aura_profile


async def learn_from_turn_auto(
    user_message: str,
    aura_response: str,
    session_id: str = "default",
) -> tuple[int, int]:
    """Convenience function to trigger profile learning."""
    try:
        manager = await ProfileManager.get_instance()
        return await manager.learn_from_turn(
            user_message=user_message,
            aura_response=aura_response,
            session_id=session_id,
        )
    except Exception as e:
        record_degradation("profile_manager", e)
        logger.warning("Auto-learning failed: %s", e)
        return (0, 0)
