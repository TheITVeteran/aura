"""Conservative semantic coverage for compound user requests.

This module is shared by both phase-level dialogue validation and the desktop
reliability gate. A prompt contract that detects multiple asks is useful only
if every production response path checks the same contract before surfacing a
reply.
"""

from __future__ import annotations

import re
from typing import Any

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
        "know",
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


def coverage_tokens(text: Any) -> set[str]:
    """Return distinctive words that can prove an ask was engaged."""

    words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", str(text or "").lower())
    return {
        word.split("'", 1)[0]
        for word in words
        if word not in _COVERAGE_STOPWORDS and len(word.split("'", 1)[0]) > 2
    }


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
        wanted = coverage_tokens(segment)
        if len(wanted) < _MIN_COVERAGE_TOKENS:
            continue
        if not (wanted & answered):
            missed.append(str(segment))
    return missed


__all__ = ["coverage_tokens", "unanswered_question_parts"]
