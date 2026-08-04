"""Telling what the turn attached apart from what the person said.

A live turn assembles a prompt around the visible message: the desktop
full-mind contract directives, active grounding evidence, a screen reading,
excerpts of her own source. All of it is bracketed with a banner so it is
legible in a prompt — and all of it used to be recorded as ``role: user``,
because the augmented objective was what got appended to working memory.

Measured live 2026-08-04. Two turns about her source attached real excerpts
as evidence; the third turn asked "what's 17 times 4?" and came back with a
function from ``core/memory/associative_entity_memory.py``. The excerpts
were still in working memory, and text in working memory is material a model
continues — the same mechanism that made a screen capture come back as the
reply.

Recording the visible message stops NEW pollution. This module is the other
half: what is already in memory is scrubbed on the way past, so a
conversation that was contaminated before the fix heals instead of carrying
those blocks for the rest of its life.

Pure string handling, no imports from the runtime: this has to be safe to
call from memory paths that must not pull cognition in behind them.
"""
from __future__ import annotations

import re
from typing import Any

__all__ = [
    "INJECTED_BANNERS",
    "contains_injected_block",
    "strip_injected_blocks",
]

#: The banners a turn attaches. Each is either a fenced block with a matching
#: ``[END …]`` line, or a single labelled section that runs to the end.
INJECTED_BANNERS: tuple[str, ...] = (
    "YOUR OWN RECENT PERCEPTION",
    "YOUR OWN SOURCE",
    "LIVE DESKTOP FULL-MIND CONTRACT",
    "ACTIVE GROUNDING EVIDENCE",
    "OBSERVATION",
    "DIRECT RESULT",
    "SKILL OUTPUT",
    "SKILL RESULT",
    "INTERNAL MEMORY RECALL",
)

_BANNER_ALTERNATION = "|".join(re.escape(name) for name in INJECTED_BANNERS)

#: A fenced block: ``[NAME …]`` through ``[END NAME…]``, inclusive.
_FENCED_RE = re.compile(
    rf"\[\s*(?:{_BANNER_ALTERNATION})\b[^\]]*\]" r".*?" rf"\[\s*END\b[^\]]*\]",
    re.DOTALL | re.IGNORECASE,
)

#: An unfenced banner — ``[DIRECT RESULT]: …`` — which runs to the end of the
#: text. Applied only after the fenced form has been removed, so a properly
#: closed block is never over-consumed.
_TRAILING_RE = re.compile(
    rf"\[\s*(?:{_BANNER_ALTERNATION})\b[^\]]*\].*\Z",
    re.DOTALL | re.IGNORECASE,
)


def strip_injected_blocks(text: Any) -> str:
    """The person's words, with everything the turn attached removed.

    Returns the input unchanged when it carries no banner, and never returns
    an empty string for non-empty input that was ENTIRELY injected — the
    caller needs to be able to tell "nothing was said" from "all of it was
    machinery", and an empty user message is its own defect downstream.
    """
    body = str(text or "")
    if not body.strip():
        return body
    cleaned = _FENCED_RE.sub("", body)
    cleaned = _TRAILING_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned or body.strip()


def contains_injected_block(text: Any) -> bool:
    """Whether this text carries machinery that was never spoken."""
    body = str(text or "")
    if not body.strip():
        return False
    return bool(_FENCED_RE.search(body) or _TRAILING_RE.search(body))
