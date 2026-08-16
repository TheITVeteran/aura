"""Parse explicit speaker introductions without changing the utterance itself.

Conversation ingress has two different facts to preserve: what arrived on the
wire and what the speaker asked Aura to answer. A leading introduction such as
``ChatGPT here.`` belongs to the first fact, while the sentence after it is the
semantic user turn. Keeping both in one string lets a decoder bind a later
``you`` to the introduced speaker instead of to Aura.

This parser recognizes only an explicit leading declaration with a hard
sentence boundary and a name-like identifier. It does not infer identity from
prose, authenticate the declaration, or rewrite text elsewhere in a message.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = ["InterlocutorTurn", "parse_interlocutor_introduction"]


_NAME_TOKEN = r"(?:@?[A-Z][A-Za-z0-9_+-]*|[A-Z0-9][A-Z0-9_+-]{1,})"
_NAME = rf"(?P<name>{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,3}})"
_BOUNDARY = r"(?P<boundary>\s*(?:[.!?:]|[-\u2013\u2014])\s+)"
_BODY = r"(?P<body>\S[\s\S]*)"
_INTRO_PATTERNS = (
    re.compile(rf"^\s*{_NAME}\s+(?:here|speaking){_BOUNDARY}{_BODY}$"),
    re.compile(
        rf"^\s*(?i:this\s+is|it\s+is|it['\u2019]s|i\s+am|i['\u2019]m)\s+"
        rf"{_NAME}{_BOUNDARY}{_BODY}$"
    ),
)

_NON_IDENTITIES = frozenset(
    {
        "a test",
        "an example",
        "the problem",
        "the question",
        "the answer",
        "the result",
        "what",
        "who",
        "someone",
        "anyone",
        "everyone",
    }
)


@dataclass(frozen=True)
class InterlocutorTurn:
    """Wire text, semantic utterance, and optional self-declared speaker."""

    raw_message: str
    utterance: str
    declared_name: str | None = None
    declaration: str = ""
    declaration_end: int = 0

    @property
    def has_declaration(self) -> bool:
        return bool(self.declared_name)

    def evidence(self) -> dict[str, Any]:
        """Return bounded provenance data; the declaration is never identity proof."""

        if not self.has_declaration:
            return {}
        return {
            "display_name": self.declared_name,
            "speaking_role": "user",
            "source": "message_prefix_self_declaration",
            "authenticated": False,
            "declaration": self.declaration,
            "raw_span": [0, self.declaration_end],
        }


def _valid_name(value: str) -> bool:
    normalized = " ".join(str(value or "").split())
    if not normalized or normalized.casefold() in _NON_IDENTITIES:
        return False
    if len(normalized) > 80:
        return False
    return any(character.isalpha() for character in normalized)


def parse_interlocutor_introduction(message: Any) -> InterlocutorTurn:
    """Split one explicit leading speaker declaration from its utterance.

    The returned ``raw_message`` is always equal to the caller's string input.
    On ambiguity, ``utterance`` is the raw message and no identity evidence is
    emitted.
    """

    raw = str(message or "")
    for pattern in _INTRO_PATTERNS:
        match = pattern.match(raw)
        if match is None:
            continue
        name = " ".join(match.group("name").split())
        body = str(match.group("body") or "").strip()
        if not _valid_name(name) or not body:
            continue
        declaration_end = match.start("body")
        declaration = raw[:declaration_end].strip()
        return InterlocutorTurn(
            raw_message=raw,
            utterance=body,
            declared_name=name,
            declaration=declaration,
            declaration_end=declaration_end,
        )
    return InterlocutorTurn(raw_message=raw, utterance=raw)
