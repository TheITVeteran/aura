"""How likely each candidate next turn is after a bare transcript, on her fast model.

core/together/what_was_proposed.py reads what an utterance asks of her from
which answers her own model expects. That needs likelihoods, and a generation
gives one sampled string. This holds her fast model in this process and runs
one forward pass over the transcript with every candidate appended: no
sampling, no decoding, no text written, and nothing told to the model. The
transcript is the one utterance in the bare chat markup with no system
message; the fast model's own template would write a default one.

The model is small (her reflex lane, about a gigabyte) and is loaded the first
time it is needed, on a thread of its own so the event loop never waits on
it, and only when the host has room: free memory of at least twice the
weights, once for the weights and once for the pass. `AURA_EAR=off` leaves it
unloaded, and then nothing is read.

It does not load in the desktop's parent process, which keeps MLX on the CPU
and leaves model work to the worker that owns the lane. So on the live
desktop nothing is read by this yet: the same pass has to become a job of the
cortex worker. A tool or a test in a process of its own runs it as written.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.Brain.LLM.Expects")

__all__ = ["ExpectsNext", "get_expects_next", "transcript_of"]

#: The most candidates one pass scores. The readers that use this score a few
#: second parts for each of a handful of kinds.
MOST_CANDIDATES = 64


def transcript_of(said: str) -> str:
    """One utterance and her turn opened after it, with no system message."""
    return f"<|im_start|>user\n{' '.join(str(said or '').split())}<|im_end|>\n<|im_start|>assistant\n"


class ExpectsNext:
    """Her fast model, held for scoring and nothing else."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._path = Path(model_path) if model_path else None
        self._thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aura-expects-next")
        self._model: Any = None
        self._tokenizer: Any = None
        self._refused = ""

    def _where(self) -> Path:
        if self._path is None:
            from core.brain.llm.model_registry import FALLBACK_ENDPOINT, get_lane_runtime_model_path

            self._path = Path(get_lane_runtime_model_path(FALLBACK_ENDPOINT))
        return self._path

    def _has_room(self, path: Path) -> bool:
        import psutil

        weights = sum(item.stat().st_size for item in path.glob("*.safetensors"))
        return weights > 0 and psutil.virtual_memory().available >= 2 * weights

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if self._refused:
            return False
        if os.environ.get("AURA_EAR", "").strip().lower() == "off":
            self._refused = "switched off"
            return False
        try:
            path = self._where()
            if not self._has_room(path):
                self._refused = "the host has no room for it"
                logger.info("👂 Her fast model is not loaded for scoring: %s.", self._refused)
                return False
            import mlx.core as mx

            if mx.default_device() != mx.gpu:
                # The desktop's parent process keeps MLX on the CPU, and model
                # work there belongs to the worker that owns the lane
                # (core/runtime/desktop_boot_safety.py). Live, 13 October, this
                # loaded there anyway and ground one core for minutes.
                self._refused = "this process keeps MLX off the GPU"
                logger.info("👂 Her fast model is not loaded for scoring: %s.", self._refused)
                return False
            from mlx_lm import load

            self._model, self._tokenizer = load(str(path))
            logger.info("👂 Her fast model is loaded for scoring next turns (%s).", path.name)
            return True
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self._refused = f"{type(exc).__name__}: {exc}"
            record_degradation(
                "expects_next", exc, severity="warning",
                action="what an utterance asks of her is not read; her fast model did not load",
            )
            return False

    def _score(self, said: str, candidates: Sequence[str]) -> list[float]:
        if not self._load():
            return []
        import mlx.core as mx

        tokenizer = self._tokenizer
        prefix = list(tokenizer.encode(transcript_of(said), add_special_tokens=False))
        rows = [list(tokenizer.encode(str(c), add_special_tokens=False)) for c in candidates[:MOST_CANDIDATES]]
        if not prefix or any(not row for row in rows):
            return []
        longest = max(len(row) for row in rows)
        # Padded on the right: attention runs left to right, so what follows a
        # candidate cannot reach back into it.
        padded = [row + [row[-1]] * (longest - len(row)) for row in rows]
        logits = self._model(mx.array([prefix + row for row in padded]))
        # Only the positions that predict a candidate's tokens are read, and
        # the whole answer comes back as one small array: one evaluation a
        # pass, where reading it a number at a time was a hundred.
        start = len(prefix) - 1
        at = logits[:, start:start + longest, :]
        logprobs = at - mx.logsumexp(at, axis=-1, keepdims=True)
        chosen = mx.take_along_axis(logprobs, mx.array(padded)[..., None], axis=-1)[..., 0]
        real = mx.array([[1.0] * len(row) + [0.0] * (longest - len(row)) for row in rows])
        return [float(total) for total in mx.sum(chosen * real, axis=-1).tolist()]

    def score(self, said: str, candidates: Sequence[str]) -> list[float]:
        """The log-likelihood of each candidate as her next turn after `said`, in nats.

        Empty when the model is not loaded. Waits on the model's own thread,
        so call it from a worker thread, never from the event loop.
        """
        try:
            return self._thread.submit(self._score, said, list(candidates)).result()
        except (RuntimeError, TypeError, ValueError, IndexError) as exc:
            record_degradation(
                "expects_next", exc, severity="warning",
                action="one utterance was not read for what it asks of her",
            )
            return []

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self._model is not None,
            "refused": self._refused,
            "model": self._path.name if self._path else "",
        }


_EXPECTS: ExpectsNext | None = None
_EXPECTS_LOCK = checked_lock("brain.llm.expects_next.singleton")


def get_expects_next() -> ExpectsNext:
    global _EXPECTS
    with _EXPECTS_LOCK:
        if _EXPECTS is None:
            _EXPECTS = ExpectsNext()
        return _EXPECTS
