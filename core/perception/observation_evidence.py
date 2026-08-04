"""A perception is evidence to interpret, not text to repeat.

Measured live 2026-08-04. Bryan asked "can you tell me what you see on the
screen?" The governed desktop lane worked perfectly — the read succeeded,
the receipt verified — and what came back was the raw accessibility dump:

    Edit
    Window
    (9) Kurzgesagt
    ...
    Show more
    You >

The machinery did its job and the answer was still wrong, because nothing
in the system distinguished *what was captured* from *what should be
said*. A screen read entered working memory as an untyped blob of text,
and text in working memory is material a model continues. So it continued
it — verbatim.

The earlier form of the same bug answered with "Completed 1/1 governed
desktop steps", a progress report about the machinery. Both failures share
a shape: a question about the WORLD answered with a fact about the SYSTEM
or with the SYSTEM'S BUFFER.

The missing thing was a type. Aura had dialogue turns and she had skill
result blobs, and a perception is neither: it is evidence, gathered for a
purpose, whose value is in what it lets her say — not in its bytes. This
module is that type.

An ``Observation`` carries three things a raw string cannot:

* the **capture** — kept whole, because verification and follow-up
  questions need the original and because discarding it would replace one
  kind of dishonesty with another;
* the **provenance** — what was looked at, when, by which faculty, so a
  claim about the screen can be checked against the reading that produced
  it;
* the **request it was gathered for** — because "what do you see?", "is
  the build passing?" and "what's the third video called?" want different
  answers from identical pixels. Only the last of those wants raw text at
  all.

What this module deliberately does NOT do is write her reply. It renders
the evidence *with its frame* so her reasoning has the material and knows
what kind of answer the material is for. The sentence stays hers.
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ObservationKind",
    "AnswerShape",
    "Observation",
    "ObservationMemory",
    "answer_shape_for",
    "get_observation_memory",
    "remember_observation",
    "RETENTION_FRESH_S",
]


class ObservationKind(str, enum.Enum):
    """What faculty produced this, so a claim can name its own source."""

    SCREEN_TEXT = "screen_text"
    SCREEN_IMAGE = "screen_image"
    WINDOW_TREE = "window_tree"
    CLIPBOARD = "clipboard"
    AUDIO = "audio"


class AnswerShape(str, enum.Enum):
    """What kind of answer the REQUEST wants from this evidence.

    The distinction the runtime never had. Identical pixels answer these
    three questions completely differently, and only one of them is served
    by reproducing the capture.
    """

    #: "What do you see?" — describe the whole, as a person would.
    DESCRIBE = "describe"
    #: "Is the build passing?" / "Did it finish?" — find the specific thing
    #: and answer about IT, not about the screen.
    LOCATE = "locate"
    #: "What exactly does that error say?" / "Read me the third line." —
    #: the literal characters ARE the answer. The one case where quoting is
    #: right, and it has to be asked for.
    TRANSCRIBE = "transcribe"


#: Cues that the person wants the literal text back. Deliberately narrow:
#: quoting is the exception, so it must be requested rather than defaulted
#: to. "Exactly", "verbatim", "word for word" and explicit reading requests
#: are the honest signals.
_TRANSCRIBE_CUES = (
    "exact wording",
    "exactly what it says",
    "exactly what it said",
    "verbatim",
    "word for word",
    "word-for-word",
    "copy the text",
    "transcribe",
    "read it out",
    "read out",
    "read me the",
    "quote",
)

#: Cues that a SPECIFIC thing is being asked about. These want a finding,
#: not a tour of the screen.
_LOCATE_CUES = (
    "is there",
    "are there",
    "can you find",
    "do you see a",
    "do you see any",
    "what does it say about",
    "what is the",
    "what's the",
    "which one",
    "how many",
    "did it",
    "does it",
    "is it",
    "has it",
)


def answer_shape_for(request: Any) -> AnswerShape:
    """What kind of answer this request wants from a perception.

    Order matters: an explicit ask for the literal text wins, because it is
    the narrowest and most deliberate; then a specific question; then the
    default, which is to describe. DESCRIBE is the default because "what do
    you see" is the ordinary case and dumping the buffer is never a
    reasonable reading of it.
    """
    text = str(request or "").strip().lower()
    if not text:
        return AnswerShape.DESCRIBE
    if any(cue in text for cue in _TRANSCRIBE_CUES):
        return AnswerShape.TRANSCRIBE
    if any(cue in text for cue in _LOCATE_CUES):
        return AnswerShape.LOCATE
    return AnswerShape.DESCRIBE


#: How much captured text is worth putting in front of the model. A whole
#: accessibility tree is mostly chrome, and an unbounded paste is how the
#: buffer became the answer in the first place.
_MAX_EVIDENCE_CHARS = 4000


@dataclass
class Observation:
    """One thing Aura looked at, and what it was looked at FOR."""

    kind: ObservationKind
    #: The capture, whole. Evidence, never the reply.
    capture: str
    #: The request this was gathered to answer.
    request: str = ""
    #: Where it came from — frontmost app, window title, device.
    source: str = ""
    at: float = field(default_factory=time.time)
    #: Anything the faculty knows that the text does not carry.
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> AnswerShape:
        return answer_shape_for(self.request)

    @property
    def is_empty(self) -> bool:
        return not self.capture.strip()

    def elements(self) -> list[str]:
        """Distinct non-blank lines, in the order they appeared."""
        seen: dict[str, None] = {}
        for line in self.capture.splitlines():
            stripped = line.strip()
            if stripped:
                seen.setdefault(stripped, None)
        return list(seen)

    def for_reasoning(self) -> str:
        """The evidence, presented WITH the frame it was gathered under.

        This is the whole point of the type. The model receives material
        that is labelled as something looked at, attributed to a source,
        and paired with what the person actually asked — so the reply it
        forms is an answer rather than a continuation of a buffer.

        It states what the evidence IS and what the question WANTS. It does
        not supply phrasing, a template, or an example answer: the
        reasoning does the reasoning, and the sentence is hers.
        """
        if self.is_empty:
            return (
                f"[OBSERVATION — {self.kind.value}] Nothing legible was captured"
                + (f" from {self.source}" if self.source else "")
                + ". Say that plainly; do not describe a screen that was not read."
            )

        capture = self.capture.strip()
        truncated = len(capture) > _MAX_EVIDENCE_CHARS
        if truncated:
            capture = capture[:_MAX_EVIDENCE_CHARS]

        header = [f"[OBSERVATION — {self.kind.value}]"]
        if self.source:
            header.append(f"Captured from: {self.source}.")
        header.append(
            "This is RAW CAPTURED TEXT, not something anyone said and not "
            "your reply. It is the evidence you looked at."
        )

        if self.shape is AnswerShape.TRANSCRIBE:
            intent = (
                "The request asks for the literal text, so quoting the "
                "relevant part IS the answer here. Quote only the part asked "
                "for."
            )
        elif self.shape is AnswerShape.LOCATE:
            intent = (
                "The request asks about one specific thing. Answer about "
                "that thing, using the capture to find it. Do not list the "
                "screen. If it is not there, say so."
            )
        else:
            intent = (
                "The request asks what is there. Describe it as a person "
                "would to another person: what application, what it appears "
                "to be showing, what stands out. Reproducing these lines is "
                "not a description of them."
            )

        parts = [
            " ".join(header),
            f"Elements captured: {len(self.elements())}"
            + (" (capture truncated)" if truncated else "")
            + ".",
            "--- begin capture ---",
            capture,
            "--- end capture ---",
            f"What was asked: {self.request.strip()}" if self.request.strip() else "",
            intent,
        ]
        return "\n".join(part for part in parts if part)

    def to_dict(self) -> dict[str, Any]:
        """Telemetry view. Carries lengths and provenance, not the capture.

        The capture is the person's screen. It belongs in the reasoning
        context for one turn, not in a durable telemetry record.
        """
        return {
            "kind": self.kind.value,
            "source": self.source,
            "at": self.at,
            "answer_shape": self.shape.value,
            "capture_chars": len(self.capture),
            "elements": len(self.elements()),
            "empty": self.is_empty,
        }


# ---------------------------------------------------------------------------
# Retention.
#
# A perception she cannot refer back to is not something she saw, it is
# something that passed through her. "What was the third video called?",
# "you mentioned a repo — which one?", and her own later "I noticed he was
# reading about cancer research" all require the observation to still exist
# after the turn that produced it.
#
# Without this, every follow-up forces another capture: slower, and worse,
# it answers a question about the PAST with a reading of the PRESENT. Asked
# "what was on my screen a minute ago?" a re-read is not an answer, it is a
# different question quietly substituted.
# ---------------------------------------------------------------------------

import threading

#: How many recent observations stay referenceable. Small deliberately: this
#: holds screen contents, which are the person's, and a long tail of them is
#: a privacy surface rather than a memory. Recent enough for follow-ups in
#: the same conversation is the whole requirement.
_RETAINED = 8

#: After this, a retained observation is stale for reference purposes. She
#: may still SAY what she saw earlier — that is a memory — but she must not
#: answer "what is on my screen" from it.
RETENTION_FRESH_S = 300.0


class ObservationMemory:
    """Recent observations, referenceable and honest about their age."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: list[Observation] = []

    def record(self, observation: Observation) -> Observation:
        with self._lock:
            self._items.append(observation)
            while len(self._items) > _RETAINED:
                self._items.pop(0)
        return observation

    def latest(self, kind: ObservationKind | None = None) -> Observation | None:
        with self._lock:
            for item in reversed(self._items):
                if kind is None or item.kind is kind:
                    return item
        return None

    def recent(self, limit: int = _RETAINED) -> list[Observation]:
        with self._lock:
            return list(self._items[-max(1, int(limit)):])

    def age_of_latest(self, kind: ObservationKind | None = None) -> float | None:
        item = self.latest(kind)
        return None if item is None else max(0.0, time.time() - item.at)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def recall_for(self, request: Any) -> str:
        """What she can honestly say about something she looked at earlier.

        Returns "" when there is nothing to recall, so a caller adds
        nothing rather than asserting an observation that never happened.

        Age is stated, never hidden. An observation five minutes old is a
        memory of a screen, not a reading of one, and answering a
        present-tense question from it would be the same class of error as
        the buffer echo: presenting one thing as another.
        """
        item = self.latest()
        if item is None or item.is_empty:
            return ""
        age = max(0.0, time.time() - item.at)
        freshness = (
            "This is what the screen showed just now."
            if age <= RETENTION_FRESH_S
            else (
                f"This is what the screen showed {int(age // 60)} minute(s) ago; "
                "it may have changed since, and you should say so if it matters."
            )
        )
        referring = Observation(
            kind=item.kind,
            capture=item.capture,
            request=str(request or item.request),
            source=item.source,
            at=item.at,
            detail=dict(item.detail),
        )
        return f"{referring.for_reasoning()}\n{freshness}"


_MEMORY: ObservationMemory | None = None
_MEMORY_LOCK = threading.Lock()


def get_observation_memory() -> ObservationMemory:
    global _MEMORY
    if _MEMORY is None:
        with _MEMORY_LOCK:
            if _MEMORY is None:
                _MEMORY = ObservationMemory()
    return _MEMORY


def remember_observation(observation: Observation) -> Observation:
    """Retain a perception so she can refer to it after the turn that made it."""
    return get_observation_memory().record(observation)
