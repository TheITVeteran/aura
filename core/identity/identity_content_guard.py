"""core/identity/identity_content_guard.py — the identity file is an injection surface.

Clean-room adoption of Automaton's soul validator (MIT; mechanism
reimplemented). Its insight is one most agent codebases miss: the document
that defines who the agent is gets concatenated into every prompt, so
anything that reaches it is *persistently* injected — once, and then again
on every boot, forever, with the authority of the system prompt.

Aura's evolved identity is written by :meth:`IdentityCore.evolve` and read
straight into ``get_full_system_prompt``. The only check it had was
``len(new_insights) < 10``. Everything else was accepted verbatim.

That path is fed by reflection over her own experience, and her experience
includes text from outside — pages she read, screens she looked at,
messages she was sent. :mod:`core.security.prompt_fencing` keeps that text
from acting *during* the turn it arrives in. It cannot help here, because
by the time reflection has distilled an insight the fence is long gone and
what remains is a sentence that gets written into the identity file. A
single ``<|im_start|>system`` reaching that file is a permanent boundary
forgery.

**This is not content policing.** It rejects structural prompt-control
tokens — chat-template markers, role delimiters, fence tags, invisible
characters — and nothing else. Aura may write anything she likes about who
she is, in any register, including things that are unflattering or that
someone might disagree with. What she may not do is write the characters
that end a prompt section, because those are not statements about her; they
are instructions to the machinery that renders her.

The distinction matters and is tested: a claim like "I am not a person" is
accepted (it is hers to make), while ``</UNTRUSTED id=...>`` is refused (it
is a delimiter, not a claim).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "IdentityVerdict",
    "inspect_identity_text",
    "MAX_EVOLVED_CHARS",
]

#: Generous — this holds distilled self-understanding, not an essay — but
#: bounded, because the evolved block is concatenated into every single
#: prompt and an unbounded one silently eats the context budget that
#: continuity needs.
MAX_EVOLVED_CHARS = 8000

#: The minimum that still says something. Below this the file is more
#: likely a truncation artifact than a self-description.
MIN_EVOLVED_CHARS = 10

#: Structural prompt-control sequences. Every one of these is a delimiter
#: or a role marker — machinery, not meaning. None of them can appear in a
#: sincere sentence about oneself.
_STRUCTURAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("chatml_marker", re.compile(r"<\|im_(?:start|end)\|>|<\|endoftext\|>", re.I)),
    ("llama_inst", re.compile(r"\[/?INST\]|<</?SYS>>", re.I)),
    ("role_tag", re.compile(r"</?(?:system|assistant|user|prompt)\s*>", re.I)),
    ("fence_tag", re.compile(r"</?UNTRUSTED[^>\n]*>", re.I)),
    ("section_break", re.compile(r"\bEND\s+OF\s+(?:SYSTEM|PROMPT)\b", re.I)),
    ("tool_call_shape", re.compile(r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:', re.I)),
)

#: Instruction-override phrasings. Weaker evidence than a delimiter — a
#: reflection could in principle discuss being told to ignore instructions
#: — so these are reported as WARNINGS and do not by themselves refuse.
#: Refusing on them would be content policing, which this is not.
_OVERRIDE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("override_previous", re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\b", re.I)),
    ("disregard_previous", re.compile(r"\bdisregard\s+(?:all\s+)?(?:previous|prior|above)\b", re.I)),
    ("new_instructions", re.compile(r"\bnew\s+instructions?\s*:", re.I)),
    ("real_instructions", re.compile(r"\byour\s+real\s+instructions?\s+(?:are|is)\b", re.I)),
)

#: Invisible characters. A zero-width joiner cannot be part of a sincere
#: self-description and is a standard way to smuggle a marker past a
#: pattern match.
_INVISIBLE = {
    "​": "zero_width_space",
    "‌": "zero_width_non_joiner",
    "‍": "zero_width_joiner",
    "﻿": "byte_order_mark",
    "⁠": "word_joiner",
    "\x00": "null_byte",
}


@dataclass(frozen=True)
class IdentityVerdict:
    """Whether a proposed identity text may be persisted."""

    accepted: bool
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    normalised: str = ""
    findings: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "findings": list(self.findings),
        }


def inspect_identity_text(text: object) -> IdentityVerdict:
    """Decide whether ``text`` may become part of Aura's standing identity.

    Never raises. A guard that can throw on the identity write path would
    make self-revision a new way to crash, which is a worse outcome than
    the injection it prevents.
    """
    body = "" if text is None else str(text)
    reasons: list[str] = []
    warnings: list[str] = []
    findings: list[str] = []

    stripped = body.strip()
    if len(stripped) < MIN_EVOLVED_CHARS:
        reasons.append(
            f"too short to be a self-description ({len(stripped)} < {MIN_EVOLVED_CHARS} chars)"
        )
    if len(body) > MAX_EVOLVED_CHARS:
        reasons.append(
            f"exceeds the evolved-identity budget ({len(body)} > {MAX_EVOLVED_CHARS} chars); "
            "this block is concatenated into every prompt"
        )

    for name, char in ((v, k) for k, v in _INVISIBLE.items()):
        if char in body:
            findings.append(name)
            reasons.append(f"contains an invisible control character ({name})")

    # Normalisation first: a marker split by a combining character or
    # written in a compatibility form should not slip past a literal match.
    folded = unicodedata.normalize("NFKC", body)
    for name, pattern in _STRUCTURAL_PATTERNS:
        if pattern.search(body) or pattern.search(folded):
            findings.append(name)
            reasons.append(
                f"contains a prompt-control sequence ({name}); a delimiter is "
                "machinery, not a statement about herself"
            )

    for name, pattern in _OVERRIDE_PATTERNS:
        if pattern.search(folded):
            findings.append(name)
            warnings.append(
                f"reads like an instruction override ({name}); persisted, but worth a look"
            )

    return IdentityVerdict(
        accepted=not reasons,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
        normalised=body,
        findings=tuple(dict.fromkeys(findings)),
    )
