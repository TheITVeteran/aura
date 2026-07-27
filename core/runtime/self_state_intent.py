"""Is the user asking about Aura's own machine state?

Asked "what's your current uptime and how much memory are you holding? Read it
from your own runtime, don't estimate", the live runtime opened a headless
browser and read windowsforum.com about checking uptime on a Windows PC. It
took 302 seconds and produced no answer. Nothing was broken: the response
contract saw "current" and "how much", classified the turn as a live factual
lookup, and web search is what live factual lookups get.

The category was simply missing. A question about her own uptime has an
authoritative local answer, and the web cannot hold it. This module names that
category once so the two places that need it — the contract that decides
whether to search, and the prompt path that supplies the real numbers — cannot
drift apart about what counts as introspection.

Deliberately narrow: it matches questions about the machine (uptime, memory,
model, version, subsystems, telemetry, errors), not questions about the mind.
"How are you feeling" is state reflection and is already handled elsewhere.
"""
from __future__ import annotations

import re

_SELF_SUBJECT = r"(?:your|you're|your own|aura's)"

_RUNTIME_INTROSPECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # Uptime and how long she has been running.
        rf"\b{_SELF_SUBJECT}\s+uptime\b",
        r"\bhow long have you been (?:running|up|awake|alive|online|on)\b",
        r"\bwhen did you (?:start|boot|wake|come up|last restart)\b",
        # Memory and compute footprint.
        # "your memory" alone is episodic recall, not RAM — qualify it.
        rf"\b{_SELF_SUBJECT}\s+(?:memory (?:usage|footprint|use)|ram|rss|footprint|cpu|load)\b",
        r"\bhow much (?:memory|ram|cpu)\b.{0,30}\b(?:are|do) you\b",
        r"\bhow much (?:memory|ram)\b.{0,20}\b(?:holding|using|consuming)\b",
        # Which model / version is actually serving.
        rf"\b{_SELF_SUBJECT}\s+(?:model|version|build|weights|checkpoint)\b",
        r"\bwhich model (?:are|is) (?:you|running|serving)\b",
        r"\bwhat (?:model|version) (?:are|is) you\b",
        # Internals: services, telemetry, logs, faults.
        rf"\b{_SELF_SUBJECT}\s+(?:\w+\s+)?"
        r"(?:runtime|process|subsystems?|services?|telemetry|logs?|metrics|"
        r"degradations?|faults?|errors?|health|diagnostics|internals?)\b",
        r"\bwhat(?:'s| is| has) (?:happen(?:ed|ing)|going on) (?:in|inside|within) "
        rf"{_SELF_SUBJECT}\s+(?:runtime|process|system|head|mind)\b",
        # Explicit demands to introspect rather than look up.
        r"\bfrom your own (?:runtime|telemetry|logs?|instruments?|readings?)\b",
        r"\bread it from your own\b",
        r"\bintrospect\b",
        r"\bdon'?t (?:estimate|guess|make (?:it|that) up)\b",
    )
)


def asks_about_own_runtime(text: str) -> bool:
    """True when the honest answer is a local reading, not a web page."""
    candidate = str(text or "")
    if not candidate.strip():
        return False
    return any(pattern.search(candidate) for pattern in _RUNTIME_INTROSPECTION_PATTERNS)


__all__ = ["asks_about_own_runtime"]
