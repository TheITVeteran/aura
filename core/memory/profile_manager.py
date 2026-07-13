"""Exact-agent profile learning from explicit user-origin conversation evidence."""
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from core.memory.semantic_fact_extractor import FactType, SemanticFact, SemanticFactExtractor
from core.memory.user_profile import UserProfile
from core.runtime.errors import record_degradation
from core.social.relational_memory import (
    RelationalMemoryAuthority,
    get_relational_memory_authority,
)

logger = logging.getLogger("Memory.ProfileManager")

_PROFILE_MANAGER_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
)


def _normalize_user_id(value: Any) -> str:
    return " ".join(str(value or "").strip().split())[:160]


def _fact_digest(user_id: str, session_id: str, fact: SemanticFact) -> str:
    material = "\n".join(
        (
            user_id,
            str(session_id or "")[:120],
            fact.fact_type.value,
            str(fact.subject or "")[:80],
            str(fact.predicate or "")[:120],
            str(fact.object or "")[:320],
        )
    )
    return hashlib.sha256(material.encode("utf-8", errors="replace")).hexdigest()


class ProfileManager:
    """Projects explicit user statements into the canonical relational authority."""

    _instance: ProfileManager | None = None
    _lock = asyncio.Lock()

    def __init__(
        self,
        *,
        authority: RelationalMemoryAuthority | None = None,
        user_profile: UserProfile | None = None,
    ) -> None:
        self._authority = authority or get_relational_memory_authority()
        self._fact_extractor = SemanticFactExtractor()
        self._user_profile: UserProfile | None = user_profile
        self._learning_attempts = 0
        self._learning_failures = 0
        self._fact_processing_failures = 0
        self._identity_skips = 0
        self._consent_skips = 0
        self._unsupported_fact_skips = 0
        self._last_failure_reason = ""

    def _record_learning_failure(
        self,
        exc: BaseException,
        *,
        fact_level: bool = False,
    ) -> None:
        self._learning_failures += 1
        if fact_level:
            self._fact_processing_failures += 1
        self._last_failure_reason = f"{type(exc).__name__}: {exc}"
        record_degradation("profile_manager", exc)

    @classmethod
    async def get_instance(cls) -> ProfileManager:
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    await cls._instance._initialize()
        return cls._instance

    async def _initialize(self) -> None:
        try:
            if self._user_profile is None:
                self._user_profile = await UserProfile.get_instance()
            logger.debug("ProfileManager initialized with exact-agent user authority")
        except _PROFILE_MANAGER_RECOVERABLE_ERRORS as exc:
            self._record_learning_failure(exc)
            logger.warning("ProfileManager initialization failed: %s", exc)

    @staticmethod
    def _profile_mapping(fact: SemanticFact) -> tuple[str, str, str] | None:
        source_role = str(fact.metadata.get("source_role") or "")
        if source_role != "user":
            return None
        object_text = " ".join(str(fact.object or "").strip().split())[:320]
        predicate = " ".join(str(fact.predicate or "").strip().split())[:100]
        if not object_text or not predicate:
            return None
        object_digest = hashlib.sha256(
            object_text.encode("utf-8", errors="replace")
        ).hexdigest()[:16]
        supplied_profile_key = " ".join(
            str(fact.metadata.get("profile_key") or "").strip().split()
        )[:160]
        profile_key = (
            hashlib.sha256(supplied_profile_key.encode("utf-8")).hexdigest()[:16]
            if supplied_profile_key
            else object_digest
        )
        if fact.fact_type == FactType.USER_PREFERENCE:
            return "preferences", f"{predicate}:{profile_key}", object_text
        if fact.fact_type == FactType.USER_CHARACTERISTIC:
            return "characteristics", f"{predicate}:{object_digest}", object_text
        if fact.fact_type == FactType.USER_LEARNING:
            return "learnings", f"{predicate}:{object_digest}", object_text
        if fact.fact_type == FactType.RELATIONSHIP_FACT:
            return "relationship", f"{predicate}:{object_digest}", object_text
        return None

    async def learn_from_turn(
        self,
        user_id: str,
        user_message: str,
        aura_response: str,
        session_id: str,
    ) -> tuple[int, int]:
        """Learn only explicit exact-agent facts; returns `(user_facts, self_facts)`."""
        normalized_user_id = _normalize_user_id(user_id)
        if not normalized_user_id:
            self._identity_skips += 1
            record_degradation(
                "profile_manager.identity",
                ValueError("profile learning skipped without exact user identity"),
                severity="warning",
                action="skipped unscoped profile inference",
            )
            return (0, 0)
        if not self._authority.allows(normalized_user_id, "derived_profile", "recall"):
            self._consent_skips += 1
            return (0, 0)
        if self._user_profile is None:
            self._record_learning_failure(RuntimeError("user profile adapter is unavailable"))
            return (0, 0)

        self._learning_attempts += 1
        learned = 0
        try:
            facts = self._fact_extractor.extract_user_facts(
                user_message,
                session_id=session_id,
            )
            session_digest = hashlib.sha256(
                str(session_id or "").encode("utf-8", errors="replace")
            ).hexdigest()
            for fact in facts:
                mapping = self._profile_mapping(fact)
                if mapping is None:
                    self._unsupported_fact_skips += 1
                    continue
                category, key, value = mapping
                evidence_digest = _fact_digest(normalized_user_id, session_id, fact)
                try:
                    changed = self._user_profile.add_or_update_fact(
                        normalized_user_id,
                        category=category,
                        key=key,
                        value=value,
                        confidence=fact.confidence,
                        source_fact_id=f"semantic-{evidence_digest[:24]}",
                        evidence_digest=evidence_digest,
                        metadata={
                            "source_role": "user",
                            "explicit_user_statement": True,
                            "predicate": fact.predicate,
                            "session_digest": session_digest,
                            "correction": fact.metadata.get("correction") is True,
                        },
                    )
                    learned += int(changed)
                except _PROFILE_MANAGER_RECOVERABLE_ERRORS as exc:
                    self._record_learning_failure(exc, fact_level=True)
                    logger.debug("Failed to project explicit user fact: %s", exc)
            if learned:
                logger.info(
                    "Learned %d consented explicit facts for exact agent %s",
                    learned,
                    hashlib.sha256(normalized_user_id.encode("utf-8")).hexdigest()[:12],
                )
            return (learned, 0)
        except _PROFILE_MANAGER_RECOVERABLE_ERRORS as exc:
            self._record_learning_failure(exc)
            logger.warning("Profile learning failed: %s", exc)
            return (0, 0)

    async def get_context_injection(self, user_id: str) -> str:
        normalized_user_id = _normalize_user_id(user_id)
        if not normalized_user_id or self._user_profile is None:
            return ""
        try:
            return self._user_profile.to_context_block(normalized_user_id)
        except _PROFILE_MANAGER_RECOVERABLE_ERRORS as exc:
            self._record_learning_failure(exc)
            logger.debug("Failed to generate exact-agent profile context: %s", exc)
            return ""

    def get_user_profile(self) -> UserProfile | None:
        return self._user_profile

    def get_aura_profile(self) -> None:
        """Generated conversation text is not an authoritative self-learning source."""
        return None

    def get_status(self) -> dict[str, object]:
        return {
            "initialized": self._user_profile is not None,
            "learning_attempts": self._learning_attempts,
            "learning_failures": self._learning_failures,
            "fact_processing_failures": self._fact_processing_failures,
            "identity_skips": self._identity_skips,
            "consent_skips": self._consent_skips,
            "unsupported_fact_skips": self._unsupported_fact_skips,
            "last_failure_reason": self._last_failure_reason,
            "canonical_owner": "relational_memory",
        }


async def learn_from_turn_auto(
    user_id: str,
    user_message: str,
    aura_response: str,
    session_id: str,
) -> tuple[int, int]:
    try:
        manager = await ProfileManager.get_instance()
        return await manager.learn_from_turn(
            user_id=user_id,
            user_message=user_message,
            aura_response=aura_response,
            session_id=session_id,
        )
    except _PROFILE_MANAGER_RECOVERABLE_ERRORS as exc:
        record_degradation("profile_manager", exc)
        logger.warning("Auto-learning failed: %s", exc)
        return (0, 0)
