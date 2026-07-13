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
from __future__ import annotations

import asyncio
import logging
from typing import Any

from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Memory.ChatTurnLogger")

_CHAT_TURN_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    asyncio.TimeoutError,
)


class ChatTurnLogger:
    """Singleton for automatic chat turn logging to episodic memory."""
    
    _instance: ChatTurnLogger | None = None
    _lock = asyncio.Lock()
    
    def __init__(self) -> None:
        self._episodic_memory = None
        self._memory_facade = None
    
    @classmethod
    async def get_instance(cls) -> ChatTurnLogger:
        """Get or create singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = ChatTurnLogger()
                    await cls._instance._initialize()
        return cls._instance
    
    async def _initialize(self) -> None:
        """Initialize memory system references."""
        self._refresh_services()

    def _refresh_services(self) -> None:
        """Resolve memory services that may have completed boot after this singleton."""
        try:
            from core.container import ServiceContainer

            if self._episodic_memory is None:
                self._episodic_memory = ServiceContainer.get("episodic_memory", default=None)
            if self._memory_facade is None:
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
        
        # Exclude common fallback responses that should not become memories.
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

    def _schedule_profile_learning(
        self,
        *,
        user_id: str,
        user_message: str,
        aura_response: str,
        session_id: str,
    ) -> bool:
        """Schedule exact-agent learning independently of episodic availability."""
        normalized_user_id = " ".join(str(user_id or "").strip().split())[:160]
        if not normalized_user_id:
            return False
        try:
            from core.memory.profile_manager import learn_from_turn_auto

            async def _learn_profiles() -> None:
                try:
                    user_facts, self_facts = await learn_from_turn_auto(
                        user_id=normalized_user_id,
                        user_message=user_message,
                        aura_response=aura_response,
                        session_id=session_id,
                    )
                    if user_facts > 0 or self_facts > 0:
                        logger.debug(
                            "Profile learning: %d user facts, %d self facts",
                            user_facts,
                            self_facts,
                        )
                except _CHAT_TURN_RECOVERABLE_ERRORS as exc:
                    record_degradation("chat_turn_logger.profile_learning", exc)
                    logger.debug("Profile learning skipped: %s", exc)

            get_task_tracker().create_task(
                _learn_profiles(),
                name=f"profile_learning_{session_id}",
            )
            return True
        except _CHAT_TURN_RECOVERABLE_ERRORS as exc:
            record_degradation("chat_turn_logger.profile_learning", exc)
            logger.debug("Profile learning unavailable: %s", exc)
            return False
    
    async def log_chat_turn(
        self,
        user_message: str,
        aura_response: str,
        session_id: str = "default",
        emotional_valence: float = 0.0,
        metadata: dict[str, Any] | None = None,
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

        episode_metadata = dict(metadata or {})
        episode_metadata["session_id"] = session_id
        episode_metadata["turn_type"] = "conversation"
        episode_metadata["conversation_lane"] = True
        profile_user_id = str(episode_metadata.get("user_id") or "").strip()[:160]
        self._schedule_profile_learning(
            user_id=profile_user_id,
            user_message=user_message,
            aura_response=aura_response,
            session_id=session_id,
        )

        self._refresh_services()
        if not self._episodic_memory:
            return False
        
        try:
            # Log as a conversation episode
            # Format: context = user asked | action = aura generated | outcome = response
            context = f"User asked: {user_message[:200]}"
            action = f"Generated response in session {session_id}"
            outcome = aura_response[:300]
            
            # Record with metadata
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
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Log just a user message (early capture, before response)."""
        if len(user_message.strip()) < 5:
            return False

        self._refresh_services()
        if not self._memory_facade:
            return False
        
        try:
            metadata = dict(metadata or {})
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
            return bool(result)
            
        except (AttributeError, RuntimeError, TypeError, ValueError) as e:
            record_degradation("chat_turn_logger", e)
            logger.debug("Failed to log user message: %s", e)
            return False


async def log_chat_turn_auto(
    user_message: str,
    aura_response: str,
    session_id: str = "default",
    emotional_valence: float = 0.0,
    metadata: dict[str, Any] | None = None,
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
    except _CHAT_TURN_RECOVERABLE_ERRORS as e:
        record_degradation("chat_turn_logger", e)
        logger.warning("Auto-logging failed: %s", e)
        return False
