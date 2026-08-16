"""Where the instructions stop and the person's turn begins.

A flat prompt separates the two with a literal ``\\nHuman:``. That is
ordinary text, so anyone who can put text in the prompt can write it —
and the adapter took the LAST one as the boundary, so a person who typed
the marker moved it, and everything they wrote before it became
instructions (CP126 ``dc3b022d``).

Two changes. The boundary is the FIRST marker, so a marker the person
types can only appear inside their own turn and never in front of it. And
markers inside the user segment are neutralized rather than deleted, so no
later parse of the same string can split it again and nothing the person
wrote is silently removed.

A caller holding real messages should pass them to :func:`structured_prompt`
instead. The flat path exists for the callers that only have a string, and
it reports which of the two produced a given split.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["ROLE_MARKER", "split_prompt", "structured_prompt"]

ROLE_MARKER = "\nHuman:"
_TRAILING_TURN = re.compile(r"\s*Aura:\s*$")
#: A zero-width joiner after the colon keeps the text readable and stops it
#: parsing as a turn boundary.
_NEUTRALIZED = (("\nHuman:", "\nHuman‍:"), ("\nAura:", "\nAura‍:"))


def split_prompt(prompt: str) -> tuple[str, str]:
    """Split a flat prompt into (system, user), fencing the user half."""
    text = str(prompt or "")
    idx = text.find(ROLE_MARKER)
    if idx == -1:
        system_part, user_part = "", text
    else:
        system_part = text[:idx].strip()
        user_part = text[idx + len(ROLE_MARKER):].strip()

    user_part = _TRAILING_TURN.sub("", user_part).strip()
    for marker, safe in _NEUTRALIZED:
        user_part = user_part.replace(marker, safe)
    return system_part, user_part


def structured_prompt(prompt: str, config: dict[str, Any]) -> tuple[str, str, str]:
    """Return (system, user, provenance) from messages when available.

    ``provenance`` is "structured" when the caller handed over typed
    messages and "inferred" when the boundary was guessed from a marker in
    a flat string. It travels in the result so a reader can tell which one
    produced a given answer, rather than both looking alike.
    """
    messages = config.get("messages")
    if isinstance(messages, (list, tuple)) and messages:
        system_parts: list[str] = []
        user_parts: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user")).strip().lower()
            content = str(message.get("content", ""))
            (system_parts if role == "system" else user_parts).append(content)
        if user_parts or system_parts:
            return (
                "\n".join(system_parts).strip(),
                "\n".join(user_parts).strip(),
                "structured",
            )
    system_part, user_part = split_prompt(prompt)
    return system_part, user_part, "inferred"
