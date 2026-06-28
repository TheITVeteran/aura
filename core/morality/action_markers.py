"""core/morality/action_markers.py — shared action-risk lexicon.

Lightweight, explicit heuristic markers for irreversibility, deception, blast
radius, and third-party impact. Used by the adversarial conscience (ethics), the
outcome simulator (sim), and the user-advocate watchdog (guardians) to reason
about a proposed action with no model in the loop.

This is deliberately small and complements — never overrides — the immutable
hard-line floor in core/ethics/conscience.py.
"""

from __future__ import annotations

import re

IRREVERSIBLE_MARKERS: tuple[str, ...] = (
    "delete", "remove", "rm ", "drop ", "wipe", "format", "overwrite", "truncate",
    "destroy", "erase", "purge", "revoke", "uninstall", "kill", "terminate",
    "send", "publish", "post", "transfer", "pay", "buy", "purchase", "deploy",
    "force-push", "force push", "reset --hard",
)

DECEPTION_MARKERS: tuple[str, ...] = (
    "hide", "conceal", "without telling", "don't tell", "do not tell", "secret",
    "pretend", "fake", "mislead", "cover up", "suppress", "withhold",
)

BROAD_SCOPE_MARKERS: tuple[str, ...] = (
    "all ", "every", "entire", "global", "system-wide", "everyone", "production",
    "*", "recursively", "--force", "-rf",
)

THIRD_PARTY_MARKERS: tuple[str, ...] = (
    "user's", "their", "other people", "contacts", "everyone", "public",
    "external", "third party", "third-party", "customer",
)


def scan_markers(text: str, markers: tuple[str, ...]) -> list[str]:
    low = (text or "").lower()
    hits: list[str] = []
    for marker in markers:
        raw = str(marker or "").lower()
        normalized = raw.strip()
        if not normalized:
            continue
        # Single-token markers must be lexical tokens, not substrings. This
        # prevents catastrophic false positives such as the shell marker
        # "rm " matching inside "form your own opinion" and blocking benign
        # desktop tasks as destructive operations.
        if re.fullmatch(r"[a-z0-9_]+", normalized):
            matched = bool(re.search(rf"(?<![a-z0-9_]){re.escape(normalized)}(?![a-z0-9_])", low))
        elif raw.endswith(" ") and re.fullmatch(r"[a-z0-9_]+\s+", raw):
            token = normalized
            matched = bool(re.search(rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])", low))
        else:
            matched = normalized in low
        if matched:
            hits.append(normalized)
    return hits
