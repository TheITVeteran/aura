"""core/brain/llm/token_budget_evidence.py — a chars-per-token ratio that says where it came from.

The prompt assembler sizes every budget in characters and converts with one
number: ``max_tokens * 4``, annotated "Rough estimation: 1 token ~= 4 chars".
Four is roughly right for English prose and wrong in both directions for the
text this runtime actually carries. Code, JSON receipts, file paths, and CJK
run nearer two to three characters per token, so a prompt built to fit can be
half again over the real window. Overflow is not the symmetric failure: the
backend drops from the head, and the head is the identity lock and the
structural constraint block. The prompt keeps its shape and loses what binds
it.

The estimate cannot simply be replaced here. Loading a tokenizer in the
orchestrator process is the thing that must not happen on this path — it is
model work in the process that serves conversation, contending with the
resident worker for the same hardware.

So the same move as ``context_window_evidence``: the ratio travels with its
provenance, the assumption is conservative rather than average, and the
component that already knows both numbers reports them.

``MEASURED``
    Derived from prompts the worker actually tokenized: it holds the string it
    encoded and the token count it got, and reporting the pair costs nothing.
``ASSUMED``
    Nothing has been observed yet. The value is deliberately below the prose
    average, because under-filling the window wastes context and over-filling
    it deletes the constraints.

``ASSUMED`` records a degradation the first time it is used, so a runtime that
never receives an observation says so once instead of budgeting on a guess for
the life of the process.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import StrEnum

from core.runtime.errors import record_degradation

__all__ = [
    "RatioSource",
    "CharsPerToken",
    "assumed_chars_per_token",
    "chars_per_token",
    "observe_prompt_tokenization",
    "reset_for_test",
]

#: Below the ~4.0 English-prose average on purpose. The cost of being low is a
#: prompt that carries less than it could; the cost of being high is a prompt
#: whose head is silently deleted by the backend. Those are not comparable.
ASSUMED_CHARS_PER_TOKEN = 3.0

#: Observations needed before the measured ratio is trusted over the assumption.
#: One prompt is a sample of one document; a handful spans a conversation's mix
#: of prose, code, and receipts.
MIN_OBSERVATIONS = 8

#: Bounds any single observation must satisfy to be counted. A ratio outside
#: this range means the caller paired a string with a token count from a
#: different string, and averaging that in would corrupt every later budget.
_MIN_PLAUSIBLE_RATIO = 0.5
_MAX_PLAUSIBLE_RATIO = 12.0


class RatioSource(StrEnum):
    MEASURED = "measured"
    ASSUMED = "assumed"


@dataclass(frozen=True, slots=True)
class CharsPerToken:
    """A ratio and the evidence behind it."""

    ratio: float
    source: RatioSource
    observations: int
    detail: str = ""

    @property
    def measured(self) -> bool:
        return self.source is RatioSource.MEASURED

    def tokens_to_chars(self, tokens: int) -> int:
        return max(0, int(float(tokens) * self.ratio))


_LOCK = threading.Lock()
_TOTAL_CHARS = 0
_TOTAL_TOKENS = 0
_OBSERVATIONS = 0
_ASSUMPTION_REPORTED = False


def observe_prompt_tokenization(chars: int, tokens: int) -> bool:
    """Report one prompt's real character and token counts.

    Returns True when the observation was counted. Callers are components that
    tokenized a prompt anyway; nothing here tokenizes on its own.
    """

    global _TOTAL_CHARS, _TOTAL_TOKENS, _OBSERVATIONS

    try:
        char_count = int(chars)
        token_count = int(tokens)
    except (TypeError, ValueError):
        return False
    if char_count <= 0 or token_count <= 0:
        return False
    ratio = char_count / token_count
    if not _MIN_PLAUSIBLE_RATIO <= ratio <= _MAX_PLAUSIBLE_RATIO:
        # Two different strings, not a surprising one.
        record_degradation(
            "llm.token_budget_evidence",
            ValueError(
                f"implausible chars-per-token observation {ratio:.2f} "
                f"({char_count} chars / {token_count} tokens); discarded"
            ),
            severity="warning",
            action="kept the previous chars-per-token evidence",
        )
        return False

    with _LOCK:
        _TOTAL_CHARS += char_count
        _TOTAL_TOKENS += token_count
        _OBSERVATIONS += 1
    return True


def assumed_chars_per_token() -> CharsPerToken:
    return CharsPerToken(
        ratio=ASSUMED_CHARS_PER_TOKEN,
        source=RatioSource.ASSUMED,
        observations=0,
        detail="no prompt tokenization has been reported to this process",
    )


def chars_per_token() -> CharsPerToken:
    """The ratio to budget with, carrying how it was arrived at."""

    global _ASSUMPTION_REPORTED

    with _LOCK:
        observations = _OBSERVATIONS
        total_chars = _TOTAL_CHARS
        total_tokens = _TOTAL_TOKENS

    if observations >= MIN_OBSERVATIONS and total_tokens > 0:
        return CharsPerToken(
            ratio=total_chars / total_tokens,
            source=RatioSource.MEASURED,
            observations=observations,
            detail=f"{total_chars} chars over {total_tokens} tokens",
        )

    with _LOCK:
        first_time = not _ASSUMPTION_REPORTED
        _ASSUMPTION_REPORTED = True
    if first_time:
        record_degradation(
            "llm.token_budget_evidence",
            RuntimeError(
                "prompt budgets are using the assumed chars-per-token ratio "
                f"({ASSUMED_CHARS_PER_TOKEN}); {observations} of "
                f"{MIN_OBSERVATIONS} observations reported"
            ),
            severity="warning",
            action="budgeted the prompt against a stated assumption",
        )
    return CharsPerToken(
        ratio=ASSUMED_CHARS_PER_TOKEN,
        source=RatioSource.ASSUMED,
        observations=observations,
        detail=f"{observations}/{MIN_OBSERVATIONS} observations",
    )


def reset_for_test() -> None:
    global _TOTAL_CHARS, _TOTAL_TOKENS, _OBSERVATIONS, _ASSUMPTION_REPORTED

    with _LOCK:
        _TOTAL_CHARS = 0
        _TOTAL_TOKENS = 0
        _OBSERVATIONS = 0
        _ASSUMPTION_REPORTED = False
