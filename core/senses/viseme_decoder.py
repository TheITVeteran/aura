"""core/senses/viseme_decoder.py
─────────────────────────────
Bounded-vocabulary lip reading from viseme sequences.

Honest scope, stated plainly: this is NOT open-vocabulary sentence
lip-reading (that needs the trained VSR checkpoint on the governed
seam). It IS real, working lip reading for a small command vocabulary:
landmark lip geometry (aperture, width) is classified per-frame into
coarse viseme classes; a run-length-collapsed viseme string is matched
against per-word templates with edit distance; matches report
calibrated confidence and refuse when ambiguous.

Viseme classes from geometry:
  C  closed        (bilabials / rest:   aperture low)
  O  open          (vowels a/e:         aperture high, width mid)
  R  rounded       (o/u/w:              aperture mid, width LOW)
  W  wide          (i/ee/s:             aperture low-mid, width HIGH)
"""
from __future__ import annotations

from dataclasses import dataclass

# Geometry thresholds on the normalized (face-height-relative) metrics
# produced by core/senses/visual_speech.py landmarks.
_APERTURE_CLOSED = 0.015
_APERTURE_OPEN = 0.055
_WIDTH_ROUNDED = 0.34
_WIDTH_WIDE = 0.44

# Viseme templates for the default command vocabulary. Templates are
# collapsed viseme strings; several words share visemes with others in
# English generally, but within THIS vocabulary each is distinct.
DEFAULT_VOCABULARY: dict[str, str] = {
    "aura": "OCO",       # a-u-ra: open, rounding/closure, open
    "yes": "WOC",        # y(wide)-e(open)-s(closed-ish)
    "no": "CR",          # n(closed)-o(rounded)
    "stop": "WCRC",      # s-t(closed)-o(rounded)-p(bilabial)
    "hello": "COR",      # h-e(open)-llo(rounded)
    "open": "RCOC",      # o(rounded)-p(bilabial)-e(open)-n
    "close": "CWRW",     # c-l(wide-ish)-o(rounded)-se(wide)
    "wake": "ROC",       # w(rounded)-a(open)-ke(closed)
}

_MIN_FRAMES = 4
_MAX_TEMPLATE_DISTANCE = 1     # edit distance tolerance
_AMBIGUITY_MARGIN = 1          # runner-up must be worse by this much


def classify_viseme(aperture: float, width: float) -> str:
    """One frame of lip geometry → coarse viseme class."""
    if aperture < _APERTURE_CLOSED:
        return "C"
    if width < _WIDTH_ROUNDED:
        return "R"
    if aperture > _APERTURE_OPEN:
        return "O"
    if width > _WIDTH_WIDE:
        return "W"
    return "O" if aperture > (_APERTURE_CLOSED + _APERTURE_OPEN) / 2 else "W"


def collapse_runs(visemes: str) -> str:
    """Run-length collapse: articulation dwell time is speaker-dependent;
    the class SEQUENCE is what carries the word."""
    collapsed: list[str] = []
    for viseme in visemes:
        if not collapsed or collapsed[-1] != viseme:
            collapsed.append(viseme)
    return "".join(collapsed)


def _edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (char_a != char_b),
            ))
        previous = current
    return previous[-1]


@dataclass
class LipReadResult:
    word: str | None
    confidence: float
    viseme_sequence: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "confidence": round(self.confidence, 4),
            "viseme_sequence": self.viseme_sequence,
            "reason": self.reason,
            "honest_scope": "bounded_command_vocabulary_not_open_speech",
        }


class VisemeDecoder:
    """Sequence-in, word-out decoder over a fixed vocabulary."""

    def __init__(self, vocabulary: dict[str, str] | None = None):
        self.vocabulary = dict(vocabulary or DEFAULT_VOCABULARY)
        self._frames: list[str] = []

    def feed(self, aperture: float, width: float, *, speaking: bool) -> None:
        """Accumulate frames while the speech-activity gate is open."""
        if speaking:
            self._frames.append(classify_viseme(float(aperture), float(width)))

    def reset(self) -> None:
        self._frames.clear()

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def decode(self) -> LipReadResult:
        """Match the collapsed utterance against the vocabulary and
        reset for the next utterance. Refuses honestly when the evidence
        is short or ambiguous."""
        sequence = collapse_runs("".join(self._frames))
        frames = len(self._frames)
        self.reset()
        if frames < _MIN_FRAMES or len(sequence) < 2:
            return LipReadResult(None, 0.0, sequence, "insufficient_articulation")
        scored = sorted(
            (( _edit_distance(sequence, template), word)
             for word, template in self.vocabulary.items()),
        )
        best_distance, best_word = scored[0]
        runner_distance = scored[1][0] if len(scored) > 1 else best_distance + 99
        if best_distance > _MAX_TEMPLATE_DISTANCE:
            return LipReadResult(None, 0.0, sequence, "no_vocabulary_match")
        if runner_distance - best_distance < _AMBIGUITY_MARGIN:
            return LipReadResult(None, 0.0, sequence, "ambiguous_between_candidates")
        template_length = max(len(self.vocabulary[best_word]), 1)
        confidence = max(0.0, 1.0 - best_distance / template_length)
        return LipReadResult(best_word, confidence, sequence, "matched")
