"""Conservative semantic coverage for compound user requests.

This module is shared by both phase-level dialogue validation and the desktop
reliability gate. A prompt contract that detects multiple asks is useful only
if every production response path checks the same contract before surfacing a
reply.
"""

from __future__ import annotations

import re
from typing import Any

from core.conversation.requested_reply_shape import (
    is_reply_shape_constraint_segment,
)

_COVERAGE_STOPWORDS = frozenset(
    {
        "about",
        "actually",
        "again",
        "and",
        "answer",
        "any",
        "anything",
        "are",
        "ask",
        "asked",
        "aura",
        "because",
        "been",
        "being",
        "both",
        "but",
        "can",
        "chatgpt",
        "current",
        "could",
        "did",
        "does",
        "doing",
        "done",
        "for",
        "from",
        "give",
        "had",
        "has",
        "have",
        "her",
        "here",
        "him",
        "his",
        "how",
        "its",
        "just",
        "hey",
        "like",
        "make",
        "many",
        "may",
        "mean",
        "might",
        "more",
        "most",
        "much",
        "must",
        "not",
        "now",
        "one",
        "only",
        "other",
        "out",
        "over",
        "naturally",
        "own",
        "really",
        "right",
        "say",
        "see",
        "separately",
        "she",
        "should",
        "some",
        "something",
        "still",
        "such",
        "take",
        "tell",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "thing",
        "things",
        "think",
        "this",
        "those",
        "through",
        "too",
        "use",
        "very",
        "want",
        "was",
        "way",
        "well",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
    }
)

_MIN_COVERAGE_TOKENS = 2

_RELATION_REQUEST_RE = re.compile(
    r"\b(?:distinguish|differentiate|separate|compare|contrast)\b"
    r"(?P<left>.+?)"
    r"(?:\bfrom\b|\bwith\b|\bversus\b|\bvs\.?\b|\band\b)"
    r"(?P<right>.+)",
    re.IGNORECASE,
)

# Surface forms that prove the same side of a requested distinction.  These
# are deliberately narrow semantic families, not a general synonym table.
# The important case is epistemic provenance: saying the word ``state`` does
# not satisfy "distinguish what you know from what you can only infer".
_COVERAGE_EQUIVALENCE = {
    # Natural check-ins commonly restate one intent twice: "Are you okay?
    # Feeling fine?" is not two independent tasks.  Collapsing these surface
    # forms into one semantic side lets concise direct answers satisfy the
    # request without disabling coverage for genuinely compound turns.
    "okay": "self_condition",
    "fine": "self_condition",
    "feel": "self_condition",
    "feeling": "self_condition",
    "steady": "self_condition",
    "condition": "self_condition",
    "know": "epistemic_known",
    "known": "epistemic_known",
    "knowing": "epistemic_known",
    "observe": "epistemic_known",
    "observed": "epistemic_known",
    "observation": "epistemic_known",
    "observations": "epistemic_known",
    "measure": "epistemic_known",
    "measured": "epistemic_known",
    "measurement": "epistemic_known",
    "measurements": "epistemic_known",
    "confirmed": "epistemic_known",
    "direct": "epistemic_known",
    "directly": "epistemic_known",
    "evidence": "epistemic_known",
    "infer": "epistemic_inferred",
    "inferred": "epistemic_inferred",
    "inference": "epistemic_inferred",
    "inferences": "epistemic_inferred",
    "inferential": "epistemic_inferred",
    "inferentially": "epistemic_inferred",
    "estimate": "epistemic_inferred",
    "estimated": "epistemic_inferred",
    "apparently": "epistemic_inferred",
    "likely": "epistemic_inferred",
    "presumably": "epistemic_inferred",
    "probably": "epistemic_inferred",
    "perhaps": "epistemic_inferred",
    "maybe": "epistemic_inferred",
    "seem": "epistemic_inferred",
    "seems": "epistemic_inferred",
    "seemed": "epistemic_inferred",
    "uncertain": "epistemic_inferred",
}

_EPISTEMIC_SIDES = frozenset({"epistemic_known", "epistemic_inferred"})
_CLAUSE_BOUNDARY_RE = re.compile(r"(?:[.!?;]+|\n+)")
_DIRECT_ASSERTION_RE = re.compile(
    r"\b(?:"
    r"i(?:'m|\s+am|\s+have|\s+feel|\s+see|\s+observe|\s+remember|"
    r"\s+can|\s+cannot|\s+do|\s+don't)|"
    r"my\s+[a-z][a-z'-]*(?:\s+[a-z][a-z'-]*){0,3}\s+"
    r"(?:is|are|was|were|has|had|feels?|remains?|ended)|"
    r"(?:the|this|that|these|those)\s+[a-z][a-z'-]*"
    r"(?:\s+[a-z][a-z'-]*){0,4}\s+"
    r"(?:is|are|was|were|has|have|shows?|reads?|reports?|contains?|remains?|ended)"
    r")\b",
    re.IGNORECASE,
)


def coverage_tokens(text: Any) -> set[str]:
    """Return distinctive words that can prove an ask was engaged."""

    words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", str(text or "").lower())
    return {
        _COVERAGE_EQUIVALENCE.get(
            word.split("'", 1)[0], word.split("'", 1)[0]
        )
        for word in words
        if word not in _COVERAGE_STOPWORDS and len(word.split("'", 1)[0]) > 2
    }


def _epistemic_partition_is_covered(body: Any) -> bool:
    """Return whether prose separates asserted evidence from inference.

    A direct assertion is not automatically synonymous with knowledge. It is
    admitted here only as one side of an explicit epistemic partition and only
    when another substantive clause marks itself as inference. This recognizes
    natural discourse such as ``I am steady. Inferentially, that may persist``
    without accepting a wholly speculative answer or requiring magic words.
    """

    direct_witness = False
    inferred_witness = False
    for raw_clause in _CLAUSE_BOUNDARY_RE.split(str(body or "")):
        clause = raw_clause.strip()
        words = re.findall(r"[A-Za-z][A-Za-z'-]{1,}", clause)
        if len(words) < 4:
            continue
        tokens = coverage_tokens(clause)
        if "epistemic_inferred" in tokens:
            inferred_witness = True
            continue
        if "epistemic_known" in tokens or _DIRECT_ASSERTION_RE.search(clause):
            direct_witness = True
    return direct_witness and inferred_witness


def _relation_sides_are_covered(
    segment: Any,
    body: Any,
    answered: set[str],
) -> bool | None:
    """Return whether both sides of an explicit relation were addressed.

    ``None`` means the segment is not an explicit compare/contrast request.
    A relation is stronger than ordinary lexical engagement: each named side
    needs its own witness in the answer.  Otherwise one shared context word
    can make a reply look complete while the requested distinction is absent.
    """

    match = _RELATION_REQUEST_RE.search(str(segment or ""))
    if match is None:
        return None
    left = coverage_tokens(match.group("left"))
    right = coverage_tokens(match.group("right"))
    if not left or not right:
        return None
    if (left | right) == _EPISTEMIC_SIDES and left != right:
        return _epistemic_partition_is_covered(body)
    return bool(left & answered) and bool(right & answered)


def unanswered_question_parts(body: Any, contract: object | None) -> list[str]:
    """Return substantive asks a reply never engages with at all.

    This intentionally fails open unless the upstream prompt-shape contract
    already classified the turn as requiring single-reply coverage. A segment
    is missing only when it has at least two distinctive words and shares none
    with the reply. The check catches a wholly dropped part without grading
    answer quality or punishing concise prose.
    """

    if not getattr(contract, "requires_single_reply_coverage", False):
        return []
    segments = tuple(getattr(contract, "question_segments", ()) or ())
    if len(segments) < 2:
        return []

    answered = coverage_tokens(body)
    missed: list[str] = []
    for segment in segments:
        if is_reply_shape_constraint_segment(segment):
            continue
        relation_covered = _relation_sides_are_covered(segment, body, answered)
        if relation_covered is False:
            missed.append(str(segment))
            continue
        if relation_covered is True:
            continue
        wanted = coverage_tokens(segment)
        if len(wanted) < _MIN_COVERAGE_TOKENS:
            continue
        if not (wanted & answered):
            missed.append(str(segment))
    return missed


__all__ = ["coverage_tokens", "unanswered_question_parts"]
