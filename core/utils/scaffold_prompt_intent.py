"""Is this text a role scaffold rather than something she meant?

One definition, because several layers need the answer and each one that
re-derives it gets a different one.

A scaffolded prompt opens by telling Aura who to be for the next few hundred
tokens — "You are the Master Synthesizer. Review the original problem and the
analyses from your specialized swarm agents…". It is machinery. It is not a
subject she is thinking about, and it is not a promise she made.

LIVE DEFECT, 2026-08-10. Her commitment ledger held 501 entries. 311 of them
were swarm scaffold prompts, 277 were marked broken, and exactly one was a
genuine promise. So:

  * reliability_score — a number she reports about her own trustworthiness —
    was computed over scaffold text that no one ever promised;
  * get_context_block() injects active commitments into prompts, so scaffold
    preambles were being fed back to her as commitments she had made;
  * a real promise was one row in five hundred.

core/brain/imagination.py already had this exact predicate, for the same reason
in a different layer: a frame seeded with "You are the Master Synthesizer"
produced four novel thoughts about the word "master". Kept in core/utils so
core/agency and core/brain share one answer — the same reason
own_source_intent.py and occluded_view_intent.py live here.
"""
from __future__ import annotations

import re
from typing import Any

#: The opening of a prompt that is assigning a role rather than saying anything.
SCAFFOLD_PREAMBLE_RE = re.compile(
    r"^\s*(?:you\s+are\s+(?:the|a|an)\b"
    r"|your\s+task\s+is\b"
    r"|as\s+(?:the|a|an)\s+\w+\s+(?:agent|synthesizer|specialist|analyst)\b"
    r"|act\s+as\b"
    r"|review\s+the\s+original\s+problem\b"
    r"|\[\s*swarm\s+protocol\b"
    # Opening by pointing at attached payload: "The following python code failed
    # with an error in the sandbox: CODE: …". 36 rows of this were in the
    # commitment ledger. A promise never begins "the following"; it says what it
    # is.
    r"|the\s+following\b)",
    re.IGNORECASE,
)

#: Labels a scaffold uses to introduce the real content further down. Their
#: presence anywhere means the text is a harness wrapped around a subject.
SCAFFOLD_SECTION_LABEL_RE = re.compile(
    r"^\s*(?:ORIGINAL\s+PROBLEM|USER\s+MESSAGE|USER\s+REQUEST|QUESTION|"
    r"OBJECTIVE|TOPIC|TASK|SUBJECT|PROMPT|SWARM\s+PROTOCOL)\s*:",
    re.IGNORECASE | re.MULTILINE,
)

#: An instruction to a model with its payload attached: "Summarize the following
#: sequence of internal system events and use…". The tell is deictic — a
#: directive verb pointing at material that comes after it. A person describing
#: something they have undertaken does not say "the following"; they say what it
#: is. 189 rows of exactly this shape sat in the commitment ledger, each one
#: breaking four hours after it was created.
INSTRUCTION_WITH_PAYLOAD_RE = re.compile(
    r"\b(?:summari[sz]e|analy[sz]e|review|consider|evaluate|assess|classify|"
    r"rank|rate|compare|extract|translate|rewrite|expand|complete|continue|"
    r"given|based\s+on|using|from)\b"
    r"[^.?!]{0,40}?"
    r"\b(?:the\s+following|these\s+(?:events|items|entries|records|messages|"
    r"lines|results|logs)|below)\b",
    re.IGNORECASE,
)


def looks_like_scaffold_prompt(text: Any) -> bool:
    """True when this text is prompt machinery, not a thought or a promise."""

    raw = str(text or "").strip()
    if not raw:
        return False
    if SCAFFOLD_PREAMBLE_RE.search(raw):
        return True
    if SCAFFOLD_SECTION_LABEL_RE.search(raw):
        return True
    return bool(INSTRUCTION_WITH_PAYLOAD_RE.search(raw))


__all__ = [
    "INSTRUCTION_WITH_PAYLOAD_RE",
    "SCAFFOLD_PREAMBLE_RE",
    "SCAFFOLD_SECTION_LABEL_RE",
    "looks_like_scaffold_prompt",
]
