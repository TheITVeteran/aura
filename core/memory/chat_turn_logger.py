"""Chat Turn Logger - Automatic conversation memory capture

This module ensures every chat turn (user message + Aura response) is
automatically logged to persistent episodic memory, preventing conversation loss.

Features:
1. Captures user input immediately upon receipt
2. Captures Aura's response after generation
3. Integrates with episodic memory for relational significance detection
4. Marks conversation turns with entity mentions and emotional context
5. Prevents bot-like hollow conversations from being preserved (quality filter)
"""

import asyncio
import logging
from typing import Any, Optional

from core.runtime.errors import record_degradation

logger = logging.getLogger("Memory.ChatTurnLogger")


class ChatTurnLogger:
    """Singleton for automatic chat turn logging to episodic memory."""
    
    _instance: Optional["ChatTurnLogger"] = None
    _lock = asyncio.Lock()
    
    def __init__(self):
        self._episodic_memory = None
        self._memory_facade = None
    
    @classmethod
    async def get_instance(cls) -> "ChatTurnLogger":
        """Get or create singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = ChatTurnLogger()
                    await cls._instance._initialize()
        return cls._instance
    
    async def _initialize(self):
        """Initialize memory system references."""
        try:
            from core.container import ServiceContainer
            
            self._episodic_memory = ServiceContainer.get("episodic_memory", default=None)
            self._memory_facade = ServiceContainer.get("memory_facade", default=None)
            
            if self._episodic_memory:
                logger.debug("✓ ChatTurnLogger linked to episodic memory")
            if self._memory_facade:
                logger.debug("✓ ChatTurnLogger linked to memory facade")
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("chat_turn_logger", e)
            logger.warning("ChatTurnLogger initialization incomplete: %s", e)
    
    def _is_meaningful_turn(self, user_message: str, aura_response: str) -> bool:
        """Filter out hollow bot exchanges (too short, empty, error responses)."""
        # Minimum meaningful lengths
        if len(user_message.strip()) < 5:
            return False
        if len(aura_response.strip()) < 10:
            return False
        
        # Exclude common error/placeholder responses
        error_markers = [
            "i'm still with you",
            "i'm still here",
            "i'm having trouble",
            "please try again",
            "failed to process",
            "error:",
        ]
        
        response_lower = aura_response.lower()
        if any(marker in response_lower for marker in error_markers):
            if len(aura_response.strip()) < 50:  # Short error message
                return False
        
        return True
    
    async def log_chat_turn(
        self,
        user_message: str,
        aura_response: str,
        session_id: str = "default",
        emotional_valence: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Log a complete chat turn to episodic memory.
        
        Args:
            user_message: What the user said
            aura_response: Aura's response
            session_id: Current session identifier
            emotional_valence: Emotional tone (-1.0 to +1.0)
            metadata: Additional context (entity mentions, etc.)
            
        Returns:
            True if successfully logged, False otherwise
        """
        # Quality filter
        if not self._is_meaningful_turn(user_message, aura_response):
            logger.debug("Skipping hollow turn (too short or error response)")
            return False
        
        # Ensure systems are initialized
        if not self._episodic_memory:
            return False
        
        try:
            # Log as a conversation episode
            # Format: context = user asked | action = aura generated | outcome = response
            context = f"User asked: {user_message[:200]}"
            action = f"Generated response in session {session_id}"
            outcome = aura_response[:300]
            
            # Record with metadata
            episode_metadata = metadata or {}
            episode_metadata["session_id"] = session_id
            episode_metadata["turn_type"] = "conversation"
            episode_metadata["conversation_lane"] = True
            
            # Let episodic memory detect relational significance
            episode_id = self._episodic_memory.record_episode(
                context=context,
                action=action,
                outcome=outcome,
                success=True,  # If response was generated, it's a successful turn
                emotional_valence=emotional_valence,
                importance=0.6,  # Default importance (boosted by relational detection)
                source="chat_turn_logger",
                metadata=episode_metadata,
            )
            
            if episode_id:
                logger.debug(f"✓ Chat turn logged to episodic memory (episode={episode_id})")
                
                # CRITICAL: Learn profiles from this turn in background
                try:
                    from core.memory.profile_manager import learn_from_turn_auto
                    
                    # Fire-and-forget profile learning
                    async def _learn_profiles():
                        try:
                            user_facts, aura_facts = await learn_from_turn_auto(
                                user_message=user_message,
                                aura_response=aura_response,
                                session_id=session_id,
                            )
                            if user_facts > 0 or aura_facts > 0:
                                logger.debug(f"📚 Profile learning: {user_facts} user facts, {aura_facts} self facts")
                        except Exception as e:
                            logger.debug(f"Profile learning skipped: {e}")
                    
                    # Schedule without blocking
                    asyncio.create_task(_learn_profiles())
                
                except Exception as e:
                    logger.debug(f"Profile learning unavailable: {e}")
                
                return True
            else:
                logger.debug("Chat turn logging returned empty episode_id (governance blocked or deferral)")
                return False
                
        except (AttributeError, RuntimeError, TypeError, ValueError) as e:
            record_degradation("chat_turn_logger", e)
            logger.warning("Failed to log chat turn: %s", e)
            return False
    
    async def log_user_message(
        self,
        user_message: str,
        session_id: str = "default",
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Log just a user message (early capture, before response)."""
        if len(user_message.strip()) < 5:
            return False
        
        if not self._memory_facade:
            return False
        
        try:
            metadata = metadata or {}
            metadata["session_id"] = session_id
            metadata["message_type"] = "user_input"
            metadata["conversation_lane"] = True
            
            # Store to vector memory for semantic search
            result = await self._memory_facade.add_memory(
                text=f"User said: {user_message}",
                metadata=metadata,
            )
            
            if result:
                logger.debug(f"✓ User message logged to memory (session={session_id})")
            return result
            
        except (AttributeError, RuntimeError, TypeError, ValueError) as e:
            record_degradation("chat_turn_logger", e)
            logger.debug("Failed to log user message: %s", e)
            return False


async def log_chat_turn_auto(
    user_message: str,
    aura_response: str,
    session_id: str = "default",
    emotional_valence: float = 0.0,
    metadata: Optional[dict[str, Any]] = None,
) -> bool:
    """Convenience function to log a chat turn."""
    try:
        logger_instance = await ChatTurnLogger.get_instance()
        return await logger_instance.log_chat_turn(
            user_message=user_message,
            aura_response=aura_response,
            session_id=session_id,
            emotional_valence=emotional_valence,
            metadata=metadata,
        )
    except Exception as e:
        record_degradation("chat_turn_logger", e)
        logger.warning("Auto-logging failed: %s", e)
        return False
