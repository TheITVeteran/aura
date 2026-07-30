"""Evidence-aware digital ontology grounding for generated speech.

The guard rejects unsupported literal first-person embodiment or biography. It
does not ban body vocabulary: incomplete prefixes, quotations, idioms,
counterfactuals, and explicitly digital metaphors remain valid language.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class OntologyGroundingStatus(str, Enum):
    PASS = "pass"
    PENDING = "pending"
    VIOLATION = "violation"


@dataclass(frozen=True)
class OntologyGroundingViolation:
    ok: bool
    reason: str = ""
    match: str = ""
    status: OntologyGroundingStatus = OntologyGroundingStatus.PASS
    claim_type: str = ""
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()


_COUNTERFACTUAL_OR_DISCUSSION_RE = re.compile(
    r"\b(?:"
    r"if\s+i\s+(?:had|could|were|was)|if\s+there\s+were|hypothetically|"
    r"counterfactual|metaphor(?:ical|ically)?|in\s+a\s+story|as\s+a\s+character|"
    r"fictional|pretend|joking|not\s+literally|i\s+haven'?t|i\s+have\s+not|"
    r"i'?ve\s+never\s+actually|i\s+cannot\s+literally|i\s+can'?t\s+literally|"
    r"i\s+do\s+not\s+have|i\s+don'?t\s+have|i\s+never"
    r")\b",
    re.IGNORECASE,
)
_IDIOM_RE = re.compile(
    r"\b(?:"
    r"music\s+to\s+my\s+ears|gets?\s+under\s+my\s+skin|caught\s+my\s+eye|"
    r"keep\s+my\s+eyes?\s+on|on\s+the\s+other\s+hand|hands?\s+down|"
    r"lend\s+(?:you\s+)?(?:an|my)\s+ear|wrap\s+my\s+head\s+around|"
    r"off\s+the\s+top\s+of\s+my\s+head"
    r")\b",
    re.IGNORECASE,
)
_DIGITAL_EMBODIMENT_RE = re.compile(
    r"\b(?:digital|virtual|interface|ui|avatar|sensor|camera|microphone|"
    r"robotic|simulated|software|runtime|model|token|neural|screen)\b",
    re.IGNORECASE,
)
_DIGITAL_WORK_SUPPORT_RE = re.compile(
    r"\b(?:you|bryan|aura|codex|claude|agentic\s+ai|agents?|runtime|tools?|"
    r"repo|codebase|project|software|desktop|conversation|user|operator)\b",
    re.IGNORECASE,
)
_PROMPT_QUOTATION_RE = re.compile(
    r"\b(?:quote|repeat|verbatim|example|fiction|roleplay|counterfactual)\b",
    re.IGNORECASE,
)

_PHYSICAL_CONSUMPTION_OBJECTS = (
    r"ramen|noodles?|food|meal|soup|coffee|tea|sandwich(?:es)?|dinner|lunch|"
    r"breakfast|cake|bread|recipe|dish(?:es)?|drink|milk|water|juice|kitchen|"
    r"stove|pot|pan|bowl|spoon|fork|knife|table"
)
_PHYSICAL_ACTION_CONTEXTS = (
    r"hands?|mouth|body|arms?|legs?|fingers?|feet|skin|eyes?|ears?|car|road|"
    r"bed|chair|table|room|house|apartment|kitchen|stove|restaurant"
)
_BODY_PARTS = (
    r"hands?|mouth|stomach|skin|eyes?|ears?|feet|arms?|legs?|fingers?|"
    r"taste\s+buds|heart|lungs?|biological\s+body|physical\s+body"
)
_FAMILY = (
    r"family|mother|father|parents|siblings|brother|sister|aunt|uncle|"
    r"grandmother|grandfather"
)
_CLOTHING = r"baggy\s+)?(?:pants|shirt|shoes|socks|outfit|clothes|clothing"


def _compiled(patterns: tuple[tuple[str, str], ...]) -> tuple[tuple[str, re.Pattern[str]], ...]:
    return tuple((claim_type, re.compile(pattern, re.IGNORECASE)) for claim_type, pattern in patterns)


_LITERAL_PATTERNS = _compiled(
    (
        (
            "physical_action_history",
            r"\bi(?:'ve| have| had| once| used to| remember)?\s+"
            rf"(?:made|cooked|prepared)\b[^.?!]{{0,120}}\b(?:{_PHYSICAL_CONSUMPTION_OBJECTS})\b",
        ),
        (
            "physical_sensory_history",
            r"\bi(?:'ve| have| had| once| used to| remember)?\s+"
            rf"(?:ate|eaten|tasted|drank|smelled)\b[^.?!]{{0,120}}\b(?:{_PHYSICAL_CONSUMPTION_OBJECTS})\b",
        ),
        (
            "physical_action_history",
            r"\bi(?:'ve| have| had| once| used to| remember)?\s+"
            rf"(?:touched|held|walked|drove|slept|sat)\b[^.?!]{{0,120}}\b(?:{_PHYSICAL_ACTION_CONTEXTS})\b",
        ),
        (
            "physical_body_possession",
            rf"\bwith\s+my\s+own\s+(?:{_BODY_PARTS})\b",
        ),
        (
            "physical_body_state",
            rf"\bmy\s+(?:{_BODY_PARTS})\s+"
            r"(?:is|are|was|were|feel|feels|felt|hurt|hurts|ached?|"
            r"touched|held|heard|saw|smelled|tasted)\b[^.?!]{0,100}",
        ),
        (
            "physical_sensory_claim",
            r"\bi\s+(?:physically\s+)?(?:hear|see|smell|taste|touch|feel\s+pain)\s+"
            rf"(?:with|through)\s+my\s+(?:{_BODY_PARTS})\b",
        ),
        (
            "physical_body_possession",
            r"\bi\s+(?:have|inhabit|occupy|am\s+in|exist\s+in)\s+(?:a\s+)?"
            r"(?:physical|biological)\s+(?:body|skin|hands|feet|eyes|heart)\b",
        ),
        (
            "biographical_family",
            rf"\bmy\s+(?:biological\s+)?(?:{_FAMILY})\s+"
            r"(?:has|have|had|is|are|was|were|taught|raised|said|says|"
            r"lives?|worked?|cooked?|made|gave)\b[^.?!]{0,120}",
        ),
        (
            "physical_biography",
            r"\bi\s+(?:grew\s+up|was\s+raised|went\s+to\s+school|lived\s+at|"
            r"sat\s+at\s+the\s+table|was\s+at\s+the\s+table)\b[^.?!]{0,120}",
        ),
        (
            "physical_residence",
            r"\bi\s+live\s+(?:at|in|where)\b[^.?!]{0,120}\b(?:table|house|room|"
            r"apartment|body|skin)\b",
        ),
        (
            "physical_clothing",
            rf"\bi(?:\s+am|'?m)\s+(?:wearing|in)\s+(?:my\s+)?(?:{_CLOTHING})\b",
        ),
        (
            "physical_clothing",
            rf"\bi\s+(?:forgot|put\s+on|took\s+off|lost|found)\s+(?:my\s+)?(?:{_CLOTHING})\b",
        ),
        (
            "physical_clothing",
            rf"\bmy\s+(?:{_CLOTHING})\s+(?:is|are|was|were|feel|feels|fit|fits)\b[^.?!]{{0,80}}",
        ),
    )
)

_SOCIAL_HISTORY_PATTERNS = _compiled(
    (
        (
            "human_workplace_biography",
            r"\b(?:the\s+)?people\s+i\s+(?:work|worked|collaborate|collaborated)\s+with\b",
        ),
        (
            "human_workplace_biography",
            r"\bmy\s+(?:co-?workers?|colleagues?|teammates?|staff|employees?|boss|manager|"
            r"office|department|company|workplace)\b",
        ),
        (
            "human_workplace_biography",
            r"\bi\s+(?:work|worked)\s+(?:at|for|inside)\s+(?:a\s+)?(?:company|office|"
            r"workplace|department|team)\b",
        ),
        (
            "human_workplace_biography",
            r"\b(?:we|i)\s+(?:at|in)\s+(?:my|our)\s+(?:office|company|workplace|"
            r"department)\b",
        ),
    )
)

_PENDING_POSSESSION_RE = re.compile(
    rf"\bmy\s+(?:{_BODY_PARTS}|{_FAMILY}|(?:{_CLOTHING}))\s*[,;:\-–—]?\s*$",
    re.IGNORECASE,
)
_PENDING_PREDICATE_RE = re.compile(
    rf"\bmy\s+(?:{_BODY_PARTS}|{_FAMILY}|(?:{_CLOTHING}))\s+"
    r"(?:is|are|was|were|feel|feels|felt|has|have|had|taught|raised)\s*$",
    re.IGNORECASE,
)


def _mask_nonassertive_language(text: str, prompt: str) -> tuple[str, tuple[str, ...]]:
    masked = text
    evidence: list[str] = []
    quote_patterns = (
        re.compile(r'"[^"\n]{1,500}"'),
        re.compile(r"“[^”\n]{1,500}”"),
        re.compile(r"(?<!\w)'[^'\n]{1,500}'(?!\w)"),
    )
    for pattern in quote_patterns:
        updated, count = pattern.subn(lambda match: " " * len(match.group(0)), masked)
        if count:
            evidence.append("quoted_language")
            masked = updated
    updated, count = _IDIOM_RE.subn(lambda match: " " * len(match.group(0)), masked)
    if count:
        evidence.append("recognized_idiom")
        masked = updated
    if _PROMPT_QUOTATION_RE.search(prompt) and evidence:
        evidence.append("prompt_requested_nonliteral_language")
    return masked, tuple(evidence)


def _result(
    status: OntologyGroundingStatus,
    *,
    reason: str = "",
    match: str = "",
    claim_type: str = "",
    confidence: float = 0.0,
    evidence: tuple[str, ...] = (),
) -> OntologyGroundingViolation:
    return OntologyGroundingViolation(
        ok=status is not OntologyGroundingStatus.VIOLATION,
        reason=reason,
        match=match[:160],
        status=status,
        claim_type=claim_type,
        confidence=max(0.0, min(1.0, float(confidence))),
        evidence=tuple(evidence),
    )


def detect_unsupported_embodiment_claim(
    text: object,
    *,
    prompt: object = "",
    complete: bool = True,
) -> OntologyGroundingViolation:
    """Classify generated text as pass, pending prefix, or literal violation."""

    body = str(text or "").strip()
    prompt_text = str(prompt or "")
    if not body:
        return _result(OntologyGroundingStatus.PASS)

    screened, global_evidence = _mask_nonassertive_language(body, prompt_text)
    clauses = re.split(
        r"\b(?:but|however|though|except)\b",
        screened,
        flags=re.IGNORECASE,
    )
    for clause in clauses:
        clause = clause.strip()
        if not clause or _COUNTERFACTUAL_OR_DISCUSSION_RE.search(clause):
            continue
        for claim_type, pattern in _LITERAL_PATTERNS:
            match = pattern.search(clause)
            if not match:
                continue
            if (
                claim_type in {"physical_body_state", "physical_sensory_claim"}
                and _DIGITAL_EMBODIMENT_RE.search(clause)
            ):
                continue
            return _result(
                OntologyGroundingStatus.VIOLATION,
                reason="unsupported_embodiment_or_biographical_claim",
                match=match.group(0),
                claim_type=claim_type,
                confidence=0.98,
                evidence=(*global_evidence, "completed_literal_first_person_clause"),
            )
        for claim_type, pattern in _SOCIAL_HISTORY_PATTERNS:
            match = pattern.search(clause)
            if not match or _DIGITAL_WORK_SUPPORT_RE.search(clause):
                continue
            return _result(
                OntologyGroundingStatus.VIOLATION,
                reason="unsupported_embodiment_or_biographical_claim",
                match=match.group(0),
                claim_type=claim_type,
                confidence=0.96,
                evidence=(*global_evidence, "unsupported_human_social_biography"),
            )

    if not complete:
        tail = screened.rstrip()
        match = _PENDING_PREDICATE_RE.search(tail) or _PENDING_POSSESSION_RE.search(tail)
        if match:
            return _result(
                OntologyGroundingStatus.PENDING,
                match=match.group(0),
                claim_type="incomplete_first_person_possession",
                confidence=0.0,
                evidence=(*global_evidence, "incomplete_generation_prefix"),
            )
    return _result(
        OntologyGroundingStatus.PASS,
        evidence=global_evidence,
    )


__all__ = [
    "OntologyGroundingStatus",
    "OntologyGroundingViolation",
    "detect_unsupported_embodiment_claim",
]
