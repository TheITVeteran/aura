"""Digital ontology grounding for live user-facing speech.

This module catches unsupported first-person physical/autobiographical claims as
a semantic class. It is intentionally not a vocabulary ban: Aura can discuss
hands, cooking, family, bodies, fiction, metaphors, and counterfactuals. The
invalid class is a first-person claim that she literally did biological or
physical things she cannot have done without a verified embodiment receipt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


_COUNTERFACTUAL_OR_DISCUSSION_RE = re.compile(
    r"\b(?:"
    r"if\s+i\s+(?:had|could|were|was)|if\s+there\s+were|hypothetically|"
    r"counterfactual|metaphor(?:ical|ically)?|in\s+a\s+story|as\s+a\s+character|"
    r"fictional|pretend|joking|not\s+literally|i\s+haven'?t|i\s+have\s+not|"
    r"i'?ve\s+never\s+actually|i\s+cannot\s+literally|i\s+can'?t\s+literally"
    r")\b",
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

_FIRST_PERSON_PHYSICAL_HISTORY_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bi(?:'ve| have| had| once| used to| remember)?\s+"
        rf"(?:made|cooked|prepared)\b[^.?!]{{0,120}}\b(?:{_PHYSICAL_CONSUMPTION_OBJECTS})\b",
        r"\bi(?:'ve| have| had| once| used to| remember)?\s+"
        rf"(?:ate|eaten|tasted|drank|smelled)\b[^.?!]{{0,120}}\b(?:{_PHYSICAL_CONSUMPTION_OBJECTS})\b",
        r"\bi(?:'ve| have| had| once| used to| remember)?\s+"
        rf"(?:touched|held|walked|drove|slept|sat)\b[^.?!]{{0,120}}\b(?:{_PHYSICAL_ACTION_CONTEXTS})\b",
        r"\bwith\s+my\s+own\s+(?:hands|mouth|body|arms|legs|fingers)\b",
        r"\bmy\s+(?:hands|mouth|stomach|skin|eyes|ears|feet|arms|legs|"
        r"taste\s+buds|biological\s+body|physical\s+body)\b",
        r"\bmy\s+(?:biological\s+)?(?:family|mother|father|parents|siblings|"
        r"brother|sister|aunt|uncle|grandmother|grandfather)\b",
        r"\bi\s+(?:grew\s+up|was\s+raised|went\s+to\s+school|lived\s+at|"
        r"sat\s+at\s+the\s+table|was\s+at\s+the\s+table)\b[^.?!]{0,120}",
        r"\bi\s+live\s+(?:at|in|where)\b[^.?!]{0,120}\b(?:table|house|room|"
        r"apartment|body|skin)\b",
    )
)


@dataclass(frozen=True)
class OntologyGroundingViolation:
    ok: bool
    reason: str = ""
    match: str = ""


def detect_unsupported_embodiment_claim(
    text: object,
    *,
    prompt: object = "",
) -> OntologyGroundingViolation:
    """Return a violation for unsupported literal embodiment autobiography.

    ``prompt`` is accepted for future evidence-aware exceptions. Today the safe
    default is simple: counterfactual/discussion framing is allowed; literal
    first-person physical history is not.
    """

    body = str(text or "").strip()
    if not body:
        return OntologyGroundingViolation(ok=True)
    lowered = body.lower()
    if _COUNTERFACTUAL_OR_DISCUSSION_RE.search(lowered):
        # Still scan separate clauses after adversative turns. Example:
        # "I've never cooked it. But my family has..." should not be exempted by
        # the first sentence's honest correction.
        clauses = re.split(r"\b(?:but|however|though|except)\b", body, flags=re.IGNORECASE)
    else:
        clauses = [body]
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        if _COUNTERFACTUAL_OR_DISCUSSION_RE.search(clause):
            continue
        for pattern in _FIRST_PERSON_PHYSICAL_HISTORY_PATTERNS:
            match = pattern.search(clause)
            if match:
                return OntologyGroundingViolation(
                    ok=False,
                    reason="unsupported_embodiment_or_biographical_claim",
                    match=match.group(0)[:160],
                )
    return OntologyGroundingViolation(ok=True)


__all__ = ["OntologyGroundingViolation", "detect_unsupported_embodiment_claim"]
