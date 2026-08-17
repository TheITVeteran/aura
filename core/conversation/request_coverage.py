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
    "failed": "failure",
    "fails": "failure",
    "failing": "failure",
    "weighted": "weight",
    "weights": "weight",
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


def requested_epistemic_partition_is_covered(request: Any, body: Any) -> bool:
    """Return whether a requested known/inferred distinction has both sides."""

    match = _RELATION_REQUEST_RE.search(str(request or ""))
    if match is None:
        return True
    left = coverage_tokens(match.group("left"))
    right = coverage_tokens(match.group("right"))
    if (left | right) != _EPISTEMIC_SIDES or left == right:
        return True
    return _epistemic_partition_is_covered(body)


def complete_epistemic_partition_from_evidence(
    request: Any,
    body: Any,
    evidence_body: Any,
) -> str:
    """Append only missing epistemic witnesses from an authoritative answer.

    This is a semantic merge, not a model instruction or a regenerated reply.
    It is deliberately usable only when the evidence answer itself proves both
    sides of the requested distinction.  A model-authored direct answer is
    retained; the smallest evidence clauses needed to satisfy the omitted
    known/inferred predicate are appended in their original wording.
    """

    draft = str(body or "").strip()
    evidence = str(evidence_body or "").strip()
    if requested_epistemic_partition_is_covered(request, draft):
        return draft
    if not draft or not evidence:
        return draft
    if not requested_epistemic_partition_is_covered(request, evidence):
        return draft

    draft_tokens = coverage_tokens(draft)
    needed = set(_EPISTEMIC_SIDES - draft_tokens)
    evidence_clauses = [
        clause.strip()
        for clause in _CLAUSE_BOUNDARY_RE.split(evidence)
        if clause.strip()
    ]
    explicit: dict[str, str] = {}
    fallback_known = ""
    for clause in evidence_clauses:
        explicit_sides = coverage_tokens(clause) & _EPISTEMIC_SIDES
        for side in explicit_sides:
            explicit.setdefault(side, clause)
        if not fallback_known and _DIRECT_ASSERTION_RE.search(clause):
            fallback_known = clause

    additions: list[str] = []
    for side in ("epistemic_known", "epistemic_inferred"):
        if side not in needed:
            continue
        clause = explicit.get(side) or (
            fallback_known if side == "epistemic_known" else ""
        )
        if not clause:
            continue
        additions.append(clause.rstrip(".!?; ") + ".")
        needed.remove(side)
    if needed:
        return draft
    merged = " ".join((draft, *additions)).strip()
    return (
        merged
        if requested_epistemic_partition_is_covered(request, merged)
        else draft
    )


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
    numbered_markers = re.findall(r"(?:^|\n|\s)\d+\s*[.)]", str(body or ""))
    if len(numbered_markers) >= 2:
        answered.add("numbered")
    try:
        numbered_parts = max(0, int(getattr(contract, "numbered_parts", 0) or 0))
    except (TypeError, ValueError):
        numbered_parts = 0
    numbered_start = max(0, len(segments) - numbered_parts)
    missed: list[str] = []
    for index, segment in enumerate(segments):
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
        overlap = wanted & answered
        # Numbered multipart requests carry independent explicit obligations.
        # One shared context word cannot prove one of those obligations was
        # answered: "weights" in a graph example must not satisfy a later ask
        # about negative-weight failure. Ordinary short conversation keeps the
        # one-anchor rule so concise natural answers remain valid.
        required_anchors = (
            min(2, len(wanted))
            if numbered_parts >= 3 and index >= numbered_start
            else 1
        )
        if len(overlap) < required_anchors:
            missed.append(str(segment))
    return missed


__all__ = [
    "complete_epistemic_partition_from_evidence",
    "coverage_tokens",
    "requested_epistemic_partition_is_covered",
    "unanswered_question_parts",
]
