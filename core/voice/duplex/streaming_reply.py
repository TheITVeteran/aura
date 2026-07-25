"""core/voice/duplex/streaming_reply.py — Speaking before the reply is finished.

The governed turn returns one finished string, so nothing can be spoken
until the last token is decoded. At the measured 13.5 tok/s of the resident
32B that puts time-to-first-audio at ~4.4s for a 45-word answer, and no
amount of tuning elsewhere in the lane touches it — the whole rest of the
pipeline is 278ms.

Streaming takes that to roughly the decode time of the *first clause*, about
1.1s. The cost is that the first clause leaves her mouth before the complete
reply has cleared validation, and audio cannot be unsaid.

That is a real risk and this module exists to bound it rather than wave at
it. The bound has three parts.

**Eligibility is narrow and fails closed.** Only turns that are plainly
conversational stream. Anything that smells of a factual claim, a citation,
a number, a tool action, or a question about system state takes the fully
governed buffered path — those are exactly the turns where an unvalidated
sentence does damage, and exactly where the extra seconds are affordable
because the user expects thinking.

**Every clause is checked before it is spoken, not after.** The validator
runs on each clause as it completes and before it reaches the synthesiser,
so a failure stops the reply instead of explaining one that already
happened.

**Failure is total, not partial.** Any clause that fails, any validator
error, any malformed stream: the whole streaming attempt is abandoned and
the turn is re-run through the buffered governed path. Slow and correct
beats fast and wrong, and the fallback is the behaviour that already
shipped.

Disable entirely with ``AURA_VOICE_STREAM_REPLY=0``.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("Aura.Voice.StreamingReply")


# Turns that must never stream. Deliberately over-broad: a false "not
# eligible" costs seconds, a false "eligible" costs correctness.
_EVIDENCE_MARKERS = (
    # factual lookup and citation
    "look up", "search", "google", "cite", "source", "reference", "according to",
    "what did", "who said", "when did", "what year", "what date", "how many",
    "how much", "what percentage", "statistics", "prove", "evidence",
    # system and self state
    "status", "health", "diagnos", "log", "error", "crash", "memory usage",
    "how are you doing internally", "what's running", "uptime", "version",
    # actions with side effects
    "run ", "execute", "open ", "delete", "remove", "install", "deploy",
    "commit", "push", "write to", "create a file", "send ", "email",
    # code
    "code", "function", "class ", "traceback", "stack trace", "compile",
)

# Markdown or structure that means the model is writing, not speaking. If
# one of these appears mid-stream the reply is not shaped for the ear and
# the buffered path's shaping should handle it.
_STRUCTURE_MARKERS = re.compile(r"(```|^\s*[-*]\s|^\s*\d+\.\s|^#{1,6}\s|\|\s*---)", re.MULTILINE)

# Fragments of prompt scaffolding that must never be spoken aloud. A leak
# here is the failure mode the governed path's stripper exists to catch.
_LEAK_MARKERS = (
    "[spoken turn", "[voice context", "[you can hear", "system:", "assistant:",
    "user:", "<|", "|>", "###", "as an ai", "i am an ai language model",
    "my instructions", "the prompt says", "context:",
)


@dataclass(slots=True)
class Eligibility:
    ok: bool
    reason: str


def is_streamable(transcript: str) -> Eligibility:
    """May this turn's reply be spoken as it is generated?

    Fails closed: anything ambiguous goes to the buffered governed path.
    """
    text = (transcript or "").strip().lower()
    if not text:
        return Eligibility(False, "empty_transcript")

    for marker in _EVIDENCE_MARKERS:
        if marker in text:
            return Eligibility(False, f"evidence_critical:{marker.strip()}")

    # A bare number in the question usually means a factual answer is
    # expected back, and factual answers are what validation protects.
    if re.search(r"\b\d{3,}\b", text):
        return Eligibility(False, "numeric_question")

    return Eligibility(True, "conversational")


@dataclass(slots=True)
class ClauseVerdict:
    ok: bool
    reason: str = ""


class ClauseValidator:
    """Checks each clause *before* it is spoken.

    Cheap and deterministic on purpose: this runs inside the latency budget
    it exists to protect, so anything model-based would defeat the point.
    It catches the failure modes that actually occur in a spoken clause —
    prompt scaffolding leaking through, the model switching into written
    formatting, degenerate repetition — and defers everything subtler to the
    buffered path it falls back to.
    """

    def __init__(self, *, max_clause_chars: int = 400) -> None:
        self._max = max_clause_chars
        self._seen: list[str] = []

    def check(self, clause: str) -> ClauseVerdict:
        text = (clause or "").strip()
        if not text:
            return ClauseVerdict(True, "empty")

        low = text.lower()
        for marker in _LEAK_MARKERS:
            if marker in low:
                return ClauseVerdict(False, f"prompt_leak:{marker.strip()}")

        if _STRUCTURE_MARKERS.search(text):
            return ClauseVerdict(False, "written_structure")

        if len(text) > self._max:
            # A clause this long means the chunker never found a boundary,
            # which usually means the stream is degenerate.
            return ClauseVerdict(False, "clause_too_long")

        # Degenerate repetition: the classic local-model failure where it
        # locks onto a phrase and repeats it. Speaking that aloud is worse
        # than any latency.
        normalized = re.sub(r"[^\w\s]", "", low).strip()
        if normalized and self._seen.count(normalized) >= 2:
            return ClauseVerdict(False, "repetition_loop")
        self._seen.append(normalized)
        if len(self._seen) > 24:
            self._seen.pop(0)

        return ClauseVerdict(True, "ok")

    def reset(self) -> None:
        self._seen.clear()
