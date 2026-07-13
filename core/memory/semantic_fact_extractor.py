"""Semantic Fact Extractor - Extract learning facts from conversations

Pulls structured knowledge from conversation text:
- User preferences ("I prefer X")
- User characteristics ("I'm a developer who specializes in Y")
- User learnings ("I just learned that X is better than Y")
- Aura learnings ("I notice that X works better for you")
- General learnings ("Best practice: do X before Y")

Facts are extracted with confidence scores and can update/contradict existing knowledge.
"""

import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Memory.SemanticFactExtractor")

_FACT_EXTRACTOR_RECOVERABLE_ERRORS = (
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    re.error,
)


class FactType(Enum):
    """Types of facts that can be extracted."""
    USER_PREFERENCE = "user_preference"
    USER_CHARACTERISTIC = "user_characteristic"
    USER_LEARNING = "user_learning"
    AURA_LEARNING = "aura_learning"
    GENERAL_KNOWLEDGE = "general_knowledge"
    RELATIONSHIP_FACT = "relationship_fact"


@dataclass
class SemanticFact:
    """A single extracted fact with confidence and provenance."""
    fact_type: FactType
    subject: str                  # "user", "aura", or concept name
    predicate: str                # "prefers", "is", "learned", etc
    object: str                   # The value/claim
    confidence: float = 0.75      # 0-1 confidence score
    source_text: str = ""         # Original text it came from
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    contradicts: str | None = None  # ID of fact this contradicts
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for storage."""
        return {
            **asdict(self),
            "fact_type": self.fact_type.value,
        }
    
    def to_natural_language(self) -> str:
        """Format as readable fact."""
        if self.subject == "user":
            return f"User {self.predicate} {self.object}"
        elif self.subject == "aura":
            return f"I {self.predicate} {self.object}"
        else:
            return f"{self.subject.title()} {self.predicate} {self.object}"


class SemanticFactExtractor:
    """Extract structured facts from conversation text."""
    
    # Pattern templates for extraction
    USER_PREFERENCE_PATTERNS = [
        (
            r"\bi\s+(?:now\s+)?prefer\s+(?:the\s+)?([^.,;!?]+?)\s+over\s+([^.,;!?]+)",
            "user",
            "prefers_over",
        ),
        (
            r"\bi\s+(?:now\s+)?prefer(?:ence)?\s+(?:to\s+)?"
            r"((?:(?!\s+over\b)[^.,;!?])+)",
            "user",
            "prefers",
        ),
        (r"i\s+like\s+([^.,;!?]+)", "user", "likes"),
        (r"i\s+(?:do\s+not|don't|dont)\s+like\s+([^.,;!?]+)", "user", "dislikes"),
        (r"my\s+(?:preferred|favorite)\s+(?:way|style|format)\s+(?:is|:)\s+([^.,;!?]+)", "user", "prefers"),
    ]
    
    USER_CHARACTERISTIC_PATTERNS = [
        (r"i\s+(?:am|'m)\s+a\s+([^.,;!?]+)", "user", "is"),
        (r"i\s+(?:work|specialize)\s+in\s+([^.,;!?]+)", "user", "specializes_in"),
        (r"i\s+(?:know|understand)\s+(?:about\s+)?([^.,;!?]+)", "user", "knows_about"),
        (r"my\s+(?:background|experience)\s+(?:is|in)\s+([^.,;!?]+)", "user", "experienced_in"),
        (r"i\s+(?:have|possess)\s+([^.,;!?]+)\s+(?:skills?|expertise)", "user", "has_expertise_in"),
    ]

    USER_LEARNING_PATTERNS = [
        (r"\bi\s+(?:just\s+)?learned\s+that\s+([^.,;!?]+)", "user", "learned_that"),
        (r"\bi\s+(?:just\s+)?discovered\s+that\s+([^.,;!?]+)", "user", "discovered_that"),
    ]
    
    AURA_LEARNING_PATTERNS = [
        (r"i\s+(?:notice|learned|found|discovered)\s+that\s+(?:you\s+)?([^.,;!?]+)", "aura", "learned_that"),
        (r"i've?\s+observed\s+that\s+(?:you\s+)?([^.,;!?]+)", "aura", "observed_that"),
        (r"(?:it\s+seems|appears)\s+(?:you|that)\s+([^.,;!?]+)", "aura", "observed_that"),
        (r"you\s+(?:seem|appear)\s+to\s+(?:prefer|like|be)\s+([^.,;!?]+)", "aura", "observed_preference"),
    ]
    
    BEST_PRACTICE_PATTERNS = [
        (r"best\s+practice\s+(?:is\s+)?(?:to\s+)?([^.,;!?]+)", "general", "best_practice"),
        (r"(?:typically|usually|generally)\s+(?:best\s+to|works\s+better\s+to|should)\s+([^.,;!?]+)", "general", "best_practice"),
        (r"i\s+(?:recommend|suggest|advise)\s+(?:to\s+)?([^.,;!?]+)", "aura", "recommends"),
    ]
    
    RELATIONSHIP_PATTERNS = [
        (r"(?:you|we)\s+(?:should\s+)?(?:build|create|make|have)\s+([^.,;!?]+)\s+together", "relationship", "shared_goal"),
        (r"i\s+(?:promise|will|want)\s+to\s+([^.,;!?]+)\s+(?:with\s+)?you", "relationship", "promise"),
        (r"(?:we're|we're?)\s+([^.,;!?]+)", "relationship", "state"),
    ]
    
    def __init__(self) -> None:
        self._all_patterns: list[tuple[FactType, str, str, str]] = []
        
        for pattern, subject, predicate in self.USER_PREFERENCE_PATTERNS:
            self._all_patterns.append((FactType.USER_PREFERENCE, pattern, subject, predicate))
        
        for pattern, subject, predicate in self.USER_CHARACTERISTIC_PATTERNS:
            self._all_patterns.append((FactType.USER_CHARACTERISTIC, pattern, subject, predicate))

        for pattern, subject, predicate in self.USER_LEARNING_PATTERNS:
            self._all_patterns.append((FactType.USER_LEARNING, pattern, subject, predicate))
        
        for pattern, subject, predicate in self.AURA_LEARNING_PATTERNS:
            self._all_patterns.append((FactType.AURA_LEARNING, pattern, subject, predicate))
        
        for pattern, subject, predicate in self.BEST_PRACTICE_PATTERNS:
            self._all_patterns.append((FactType.GENERAL_KNOWLEDGE, pattern, subject, predicate))
        
        for pattern, subject, predicate in self.RELATIONSHIP_PATTERNS:
            self._all_patterns.append((FactType.RELATIONSHIP_FACT, pattern, subject, predicate))
    
    def extract_facts(
        self,
        user_message: str,
        aura_response: str,
        session_id: str = "default",
    ) -> list[SemanticFact]:
        """Extract all facts from a conversation turn.
        
        Args:
            user_message: What the user said
            aura_response: Aura's response
            session_id: Current session identifier
            
        Returns:
            List of extracted semantic facts
        """
        facts = self.extract_user_facts(user_message, session_id=session_id)
        
        # Extract from Aura response
        try:
            aura_facts = self._extract_from_text(
                aura_response,
                source_role="aura",
                session_id=session_id
            )
            facts.extend(aura_facts)
        except _FACT_EXTRACTOR_RECOVERABLE_ERRORS as e:
            record_degradation("fact_extractor", e)
            logger.debug("Aura fact extraction failed: %s", e)
        
        # Deduplicate and score
        facts = self._deduplicate_and_score(facts)
        
        logger.debug(f"Extracted {len(facts)} semantic facts from conversation turn")
        return facts

    def extract_user_facts(
        self,
        user_message: str,
        *,
        session_id: str = "default",
    ) -> list[SemanticFact]:
        """Extract only literal user-origin facts for exact-agent projections."""
        try:
            return self._deduplicate_and_score(
                self._extract_from_text(
                    user_message,
                    source_role="user",
                    session_id=session_id,
                )
            )
        except _FACT_EXTRACTOR_RECOVERABLE_ERRORS as exc:
            record_degradation("fact_extractor", exc)
            logger.debug("User fact extraction failed: %s", exc)
            return []
    
    def _extract_from_text(
        self,
        text: str,
        source_role: str = "user",
        session_id: str = "default",
    ) -> list[SemanticFact]:
        """Extract facts from a single text block."""
        facts: list[SemanticFact] = []
        text_lower = text.lower()
        
        for fact_type, pattern, subject, predicate in self._all_patterns:
            if source_role == "user" and fact_type in {
                FactType.AURA_LEARNING,
                FactType.GENERAL_KNOWLEDGE,
            }:
                continue
            if source_role == "aura" and fact_type in {
                FactType.USER_PREFERENCE,
                FactType.USER_CHARACTERISTIC,
                FactType.USER_LEARNING,
                FactType.RELATIONSHIP_FACT,
            }:
                continue
            try:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    if predicate == "prefers" and re.match(
                        r"\s+over\b",
                        text[match.end() :],
                        re.IGNORECASE,
                    ):
                        continue
                    # Extract the captured group(s)
                    groups = match.groups()
                    if not groups:
                        continue
                    
                    obj = groups[0].strip()
                    profile_key = ""
                    if predicate == "prefers_over" and len(groups) > 1:
                        alternative = str(groups[1] or "").strip()
                        if not alternative:
                            continue
                        obj = f"{obj} over {alternative}"
                        normalized_pair = sorted(
                            {
                                " ".join(groups[0].lower().split()),
                                " ".join(alternative.lower().split()),
                            }
                        )
                        profile_key = "|".join(normalized_pair)
                    
                    # Skip very short or generic objects
                    if len(obj) < 3 or obj in ("things", "stuff", "it", "this", "that"):
                        continue
                    
                    # Determine confidence based on pattern type
                    confidence = 0.8
                    if "prefer" in predicate.lower():
                        confidence = 0.9  # Strong signal
                    elif "learned" in predicate.lower() or "discovered" in predicate.lower():
                        confidence = 0.85
                    elif "best" in text_lower[max(0, match.start()-10):match.end()]:
                        confidence = 0.75
                    
                    preceding = text[max(0, match.start() - 48) : match.start()]
                    matched_text = match.group(0)
                    correction = bool(
                        re.search(
                            r"(?:\bactually\b|\bcorrection\b|\binstead\b|"
                            r"\bfrom\s+now\s+on\b|\bi\s+now\s+prefer\b)",
                            f"{preceding} {matched_text}",
                            re.IGNORECASE,
                        )
                    )
                    metadata: dict[str, Any] = {
                        "source_role": source_role,
                        "session_id": session_id,
                    }
                    if source_role == "user" and fact_type in {
                        FactType.USER_PREFERENCE,
                        FactType.USER_CHARACTERISTIC,
                        FactType.USER_LEARNING,
                        FactType.RELATIONSHIP_FACT,
                    }:
                        metadata["explicit_user_statement"] = True
                    if correction:
                        metadata["correction"] = True
                    if profile_key:
                        metadata["profile_key"] = profile_key

                    fact = SemanticFact(
                        fact_type=fact_type,
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        confidence=confidence,
                        source_text=text[max(0, match.start()-30):min(len(text), match.end()+30)],
                        metadata=metadata,
                    )
                    facts.append(fact)
            except _FACT_EXTRACTOR_RECOVERABLE_ERRORS as e:
                record_degradation("fact_extractor.pattern", e)
                logger.debug("Pattern matching failed for %s: %s", pattern, e)
                continue
        
        return facts
    
    def _deduplicate_and_score(self, facts: list[SemanticFact]) -> list[SemanticFact]:
        """Remove duplicates and boost confidence for repeated facts."""
        # Group by (subject, predicate, object)
        grouped: dict[tuple[str, str, str], list[SemanticFact]] = {}
        
        for fact in facts:
            key = (fact.subject, fact.predicate, fact.object)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(fact)
        
        # For duplicates, boost confidence
        deduped: list[SemanticFact] = []
        for group in grouped.values():
            if len(group) > 1:
                # Boost confidence for repeated facts
                primary = group[0]
                primary.confidence = min(0.95, primary.confidence + (0.05 * len(group)))
                primary.metadata["repetition_count"] = len(group)
                deduped.append(primary)
            else:
                deduped.append(group[0])
        
        return deduped


async def extract_facts_auto(
    user_message: str,
    aura_response: str,
    session_id: str = "default",
) -> list[SemanticFact]:
    """Convenience function to extract facts."""
    try:
        extractor = SemanticFactExtractor()
        return extractor.extract_facts(
            user_message=user_message,
            aura_response=aura_response,
            session_id=session_id,
        )
    except _FACT_EXTRACTOR_RECOVERABLE_ERRORS as e:
        record_degradation("fact_extractor", e)
        logger.warning("Fact extraction failed: %s", e)
        return []
