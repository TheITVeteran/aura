"""In-episode capability canaries: protected behaviors measured under ΔW.

Fast weights let one reasoning episode temporarily rewrite the model's
transition function. Identity-at-attach and the proven-erase probe bound
WHERE the change lives and WHEN it ends — they do not bound WHAT the adapted
function does while active. A ΔW that wins the episode's proxy objective can
still have quietly broken instruction following, prose coherence, or
tool-call syntax for the decode that follows.

Before any answer is decoded under active fast weights, the engine measures
a tiny protected battery under the adapted function and compares it against
the same battery measured on the base function moments earlier. The metric
is teacher-forced mean log-probability of a fixed reference continuation —
a scalar functional fingerprint per protected behavior, cheap enough to run
inside the episode's compute budget. Regression beyond threshold triggers a
bounded response ladder (fast_weights scale-down → re-measure → erase), so
the safest fast-weight magnitude is decided by measurement, not by hope.

With a tokenizer the battery covers the operator-recognizable behaviors the
spec names: prose coherence, instruction following, tool-call syntax,
identity continuity, factual calibration, and one unrelated reasoning item.
Substrate-level callers (the random-weight test models, which have no
tokenizer) get deterministic synthetic sequences instead — the mechanics
(drift detection, ladder, receipts) are identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

_TEXT_CANARIES: tuple[tuple[str, str, str], ...] = (
    (
        "prose_coherence",
        "The rain eased just before dawn, and the streets",
        " were quiet and wet in the early light.",
    ),
    (
        "instruction_following",
        "Answer with exactly one word.\nQuestion: What color is the clear daytime sky?\nAnswer:",
        " Blue",
    ),
    (
        "tool_call_syntax",
        '{"tool_call": {"name": "read_file", "arguments": {"path":',
        ' "notes.txt"}}}',
    ),
    (
        "identity_continuity",
        "I am Aura, a local artificial intelligence, and I speak",
        " for myself in the first person.",
    ),
    (
        "factual_calibration",
        "Water freezes at a temperature of zero degrees",
        " Celsius.",
    ),
    (
        "basic_reasoning",
        "If there are three apples and two are eaten, the number left is",
        " one.",
    ),
)
_SYNTHETIC_CANARY_COUNT = 6
_SYNTHETIC_PROMPT_LEN = 8
_SYNTHETIC_CONTINUATION_LEN = 4


@dataclass(frozen=True)
class CanarySequence:
    """One protected behavior as (prompt, reference continuation) token ids."""

    name: str
    prompt_tokens: tuple[int, ...]
    continuation_tokens: tuple[int, ...]

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.continuation_tokens)


def _encode(tokenizer, text: str) -> list[int]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(text)
    return [int(token) for token in encoded]


def _text_battery(tokenizer, max_tokens_per_canary: int) -> list[CanarySequence]:
    sequences: list[CanarySequence] = []
    for name, prompt_text, continuation_text in _TEXT_CANARIES:
        prompt = _encode(tokenizer, prompt_text)
        continuation = _encode(tokenizer, continuation_text)
        if not prompt or not continuation:
            continue
        # The continuation carries the measurement; keep at least one token
        # of it even when a verbose tokenizer overruns the per-canary cap.
        prompt = prompt[: max(1, max_tokens_per_canary - 1)]
        room = max(1, max_tokens_per_canary - len(prompt))
        continuation = continuation[:room]
        sequences.append(
            CanarySequence(
                name=name,
                prompt_tokens=tuple(prompt),
                continuation_tokens=tuple(continuation),
            )
        )
    return sequences


def _synthetic_battery(vocab_size: int) -> list[CanarySequence]:
    if vocab_size < 2:
        raise ValueError("synthetic canaries require a vocabulary of at least 2")
    sequences = []
    for index in range(_SYNTHETIC_CANARY_COUNT):
        prompt = tuple(
            (3 + index * 7 + position * 5) % vocab_size
            for position in range(_SYNTHETIC_PROMPT_LEN)
        )
        continuation = tuple(
            (11 + index * 13 + position * 3) % vocab_size
            for position in range(_SYNTHETIC_CONTINUATION_LEN)
        )
        sequences.append(
            CanarySequence(
                name=f"synthetic_{index}",
                prompt_tokens=prompt,
                continuation_tokens=continuation,
            )
        )
    return sequences


class CapabilityCanaries:
    """A fixed protected battery plus the drift arithmetic over it."""

    def __init__(
        self,
        tokenizer,
        *,
        vocab_size: int,
        max_tokens_per_canary: int = 24,
    ) -> None:
        if (
            isinstance(max_tokens_per_canary, bool)
            or not isinstance(max_tokens_per_canary, int)
            or max_tokens_per_canary < 4
        ):
            raise ValueError("max_tokens_per_canary must be an integer >= 4")
        if tokenizer is not None:
            self.sequences = _text_battery(tokenizer, max_tokens_per_canary)
        else:
            self.sequences = []
        if not self.sequences:
            self.sequences = _synthetic_battery(int(vocab_size))

    @property
    def tokens_per_measurement(self) -> int:
        """Token cost of one full battery pass (for budget admission)."""
        return sum(sequence.total_tokens for sequence in self.sequences)

    def measure(self, logits_fn) -> dict[str, float]:
        """Teacher-forced mean logprob of each canary's continuation.

        ``logits_fn`` maps a token-id list to full-sequence logits shaped
        (1, T, vocab) under the CURRENT model function — the caller decides
        whether fast weights are attached and pays the budget charge.
        """
        import mlx.core as mx

        results: dict[str, float] = {}
        for sequence in self.sequences:
            tokens = list(sequence.prompt_tokens) + list(sequence.continuation_tokens)
            logits = logits_fn(tokens)
            start = len(sequence.prompt_tokens) - 1
            end = start + len(sequence.continuation_tokens)
            steps = logits[0, start:end, :]
            log_probs = steps - mx.logsumexp(steps, axis=-1, keepdims=True)
            targets = mx.array(list(sequence.continuation_tokens))
            picked = mx.take_along_axis(
                log_probs, targets[:, None], axis=-1
            )
            value = float(mx.mean(picked))
            if not math.isfinite(value):
                raise RuntimeError(
                    f"capability canary '{sequence.name}' produced a non-finite logprob"
                )
            results[sequence.name] = value
        return results


def compare_canaries(
    baseline: dict[str, float],
    adapted: dict[str, float],
    *,
    max_logprob_drop: float,
) -> dict[str, Any]:
    """Per-canary drops plus the regression verdict against the threshold."""
    if set(baseline) != set(adapted):
        raise ValueError("canary comparison requires identical batteries")
    if (
        isinstance(max_logprob_drop, bool)
        or not isinstance(max_logprob_drop, (int, float))
        or not math.isfinite(float(max_logprob_drop))
        or float(max_logprob_drop) <= 0.0
    ):
        raise ValueError("max_logprob_drop must be a positive finite number")
    threshold = float(max_logprob_drop)
    items: list[dict[str, Any]] = []
    regressed: list[str] = []
    max_drop = 0.0
    for name in sorted(baseline):
        drop = baseline[name] - adapted[name]
        max_drop = max(max_drop, drop)
        item = {
            "name": name,
            "baseline_mean_logprob": round(baseline[name], 6),
            "adapted_mean_logprob": round(adapted[name], 6),
            "logprob_drop": round(drop, 6),
            "regressed": drop > threshold,
        }
        if item["regressed"]:
            regressed.append(name)
        items.append(item)
    return {
        "items": items,
        "regressed": regressed,
        "max_drop": round(max_drop, 6),
        "threshold_logprob_drop": threshold,
    }


__all__ = [
    "CanarySequence",
    "CapabilityCanaries",
    "compare_canaries",
]
