"""core/brain/llm/interoception_tap.py — per-token substrate interoception (worker side).

Every decode step, the resident model computes a full log-probability
distribution over its vocabulary — its actual, momentary belief about what
comes next. Until now that signal was discarded in the worker loop, and the
parent's "surprise" feedback ran on a unique-word-ratio heuristic over the
finished text with ``logprobs=None``. This module captures the real thing.

The tap is a **pure observer** with three hard guarantees:

1. It can never alter generation — it only reads ``response.token`` /
   ``response.logprobs`` / ``response.text`` after the sampler has already run.
2. It can never raise into the token loop — every per-token failure is
   swallowed, counted in the payload as ``dropped``, and generation continues.
3. It is bounded — at most :data:`_HARD_TOKEN_CAP` per-token records, a
   fixed-size spike list, and a payload that stays a few KB of JSON no matter
   how long the generation ran.

What it measures, per sampled token:

* **surprisal** ``-log p(token)`` in nats — how unexpected the chosen word was
  to the model that chose it;
* **entropy** of the full next-token distribution — how open the moment was;
* **top-2 probability gap** — how contested the choice was (felt ambivalence);
* whether the sampled token was the argmax (exploration vs conviction).

Interpretation (fluency, felt confidence, strain) happens parent-side in
:mod:`core.being.thought_interoception`; the worker ships measurements only.

Kill switch: ``AURA_INTEROCEPTION=0`` disables the tap entirely
(:func:`maybe_build_tap` returns ``None``).
"""
from __future__ import annotations

import logging
import math
import os
import time
from typing import Any

logger = logging.getLogger("Aura.Brain.InteroceptionTap")

# Mirrors the worker's absolute generation cap; the tap never stores more.
_HARD_TOKEN_CAP = 8192

# A token only counts as a reportable "spike" above this surprisal (nats):
# 2.0 nats ⇒ the model gave the sampled word < ~14% probability. Below that,
# naming the word as "contested" would be noise, not sensation.
_SPIKE_FLOOR_NATS = 2.0

# Per-token failures we absorb (count + continue) rather than raise into the
# decode loop. Deliberately broad short of BaseException: a dropped
# measurement is always preferable to a damaged generation.
_STEP_RECOVERABLE = (
    AttributeError,
    IndexError,
    KeyError,
    OverflowError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    ZeroDivisionError,
)

PAYLOAD_VERSION = 1


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


def _finite(x: Any) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError, OverflowError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _step_stats(logprobs: Any, token_id: int) -> tuple[float, float, float, float, bool] | None:
    """Compute (surprisal, entropy, top1_logprob, top2_logprob, argmax_hit).

    Accepts an MLX array (live worker) or anything ``numpy.asarray`` can take
    (tests, CPU fallback). Returns ``None`` when the distribution is unusable;
    the caller counts it as a dropped measurement.
    """
    stats = None
    try:
        import mlx.core as mx  # noqa: PLC0415 — worker-local heavy import

        if isinstance(logprobs, mx.array):
            lp = logprobs.reshape(-1)
            size = int(lp.shape[0])
            if not (0 <= int(token_id) < size):
                return None
            tok_lp = lp[int(token_id)]
            ent = -mx.sum(mx.exp(lp) * lp)
            pair = mx.topk(lp, k=2) if size >= 2 else mx.stack([lp[0], lp[0]])
            arg = mx.argmax(lp)
            # One materialization for the whole step: five scalars.
            packed = mx.stack(
                [tok_lp, ent, pair.reshape(-1)[0], pair.reshape(-1)[1], arg.astype(mx.float32)]
            ).tolist()
            stats = (packed[0], packed[1], packed[2], packed[3], packed[4])
    except ImportError:
        stats = None

    if stats is None:
        import numpy as np  # noqa: PLC0415

        arr = np.asarray(logprobs, dtype=np.float64).reshape(-1)
        size = int(arr.shape[0])
        if size == 0 or not (0 <= int(token_id) < size):
            return None
        tok_lp = float(arr[int(token_id)])
        with np.errstate(over="ignore", invalid="ignore"):
            probs = np.exp(arr)
            ent = float(-np.sum(probs * arr))
        if size >= 2:
            top_idx = np.argpartition(arr, -2)[-2:]
            a, b = float(arr[top_idx[0]]), float(arr[top_idx[1]])
        else:
            a = b = tok_lp
        stats = (tok_lp, ent, a, b, float(int(np.argmax(arr))))

    tok_lp = _finite(stats[0])
    ent = _finite(stats[1])
    a = _finite(stats[2])
    b = _finite(stats[3])
    arg_raw = _finite(stats[4])
    if tok_lp is None or ent is None or a is None or b is None or arg_raw is None:
        return None
    top1, top2 = (a, b) if a >= b else (b, a)  # mx.topk does not guarantee order
    surprisal = max(0.0, -tok_lp)
    entropy = max(0.0, ent)
    argmax_hit = int(arg_raw) == int(token_id)
    return surprisal, entropy, top1, top2, argmax_hit


def _downsample_mean(values: list[float], points: int) -> list[float]:
    """Bucket-mean a series down to at most ``points`` values."""
    n = len(values)
    if n == 0 or points <= 0:
        return []
    if n <= points:
        return [round(v, 4) for v in values]
    out: list[float] = []
    for b in range(points):
        lo = (b * n) // points
        hi = max(lo + 1, ((b + 1) * n) // points)
        chunk = values[lo:hi]
        out.append(round(sum(chunk) / len(chunk), 4))
    return out


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[idx]


class InteroceptionTap:
    """Accumulates per-token substrate measurements for one generation attempt."""

    def __init__(
        self,
        *,
        spike_k: int | None = None,
        curve_points: int | None = None,
        sample_points: int | None = None,
        near_tie_gap: float = 0.10,
    ) -> None:
        self._spike_k = spike_k if spike_k is not None else _env_int(
            "AURA_INTEROCEPTION_TOPK_SPIKES", 8, 0, 32
        )
        self._curve_points = curve_points if curve_points is not None else _env_int(
            "AURA_INTEROCEPTION_CURVE_POINTS", 32, 0, 128
        )
        self._sample_points = sample_points if sample_points is not None else _env_int(
            "AURA_INTEROCEPTION_SAMPLE_POINTS", 128, 8, 512
        )
        self._near_tie_gap = max(0.0, min(1.0, float(near_tie_gap)))

        self._surprisals: list[float] = []
        self._entropies: list[float] = []
        self._prob_gaps: list[float] = []  # p(top1) - p(top2) per token
        self._argmax_hits = 0
        self._token_ids: list[int] = []
        self._token_lps: list[float] = []
        self._texts: list[str] = []
        self._dropped = 0
        self._first_feed_at: float | None = None
        self._last_feed_at: float | None = None
        self._step_error_logged = False

    # ── ingestion ────────────────────────────────────────────────────────────
    def feed(self, token_id: Any, logprobs: Any, token_text: Any) -> None:
        """Record one decode step. Never raises."""
        try:
            if logprobs is None or len(self._surprisals) >= _HARD_TOKEN_CAP:
                self._dropped += 1
                return
            stats = _step_stats(logprobs, int(token_id))
            if stats is None:
                self._dropped += 1
                return
            surprisal, entropy, top1, top2, argmax_hit = stats
            now = time.monotonic()
            if self._first_feed_at is None:
                self._first_feed_at = now
            self._last_feed_at = now
            self._surprisals.append(surprisal)
            self._entropies.append(entropy)
            # exp of values ≤ 0 is safe; top1/top2 are logprobs of a normalized dist.
            self._prob_gaps.append(
                max(0.0, math.exp(min(0.0, top1)) - math.exp(min(0.0, top2)))
            )
            if argmax_hit:
                self._argmax_hits += 1
            self._token_ids.append(int(token_id))
            self._token_lps.append(-surprisal)
            self._texts.append(str(token_text or ""))
        except _STEP_RECOVERABLE as exc:
            self._dropped += 1
            if not self._step_error_logged:
                self._step_error_logged = True
                logger.debug("Interoception step measurement failed (once-logged): %s", exc)

    @property
    def measured_tokens(self) -> int:
        return len(self._surprisals)

    # ── live snapshot (rides progress IPC messages) ─────────────────────────
    def live_snapshot(self) -> dict[str, Any] | None:
        """Cheap running summary for mid-generation pulses. Never raises."""
        try:
            n = len(self._surprisals)
            if n == 0:
                return None
            return {
                "token_count": n,
                "mean_surprisal": round(sum(self._surprisals) / n, 4),
                "mean_entropy": round(sum(self._entropies) / n, 4),
            }
        except _STEP_RECOVERABLE:
            return None

    # ── finalize ─────────────────────────────────────────────────────────────
    def finalize(self, *, attempt: int = 0, generation_tps: float | None = None) -> dict[str, Any] | None:
        """Distil the attempt into a compact, JSON-safe payload. Never raises.

        Returns ``None`` when nothing was measured, so callers can simply omit
        the field and the parent treats interoception as absent.
        """
        try:
            n = len(self._surprisals)
            if n == 0:
                return None
            duration = 0.0
            if self._first_feed_at is not None and self._last_feed_at is not None:
                duration = max(0.0, self._last_feed_at - self._first_feed_at)
            tps = generation_tps
            if tps is None and duration > 0 and n > 1:
                tps = (n - 1) / duration

            sorted_surprisal = sorted(self._surprisals)
            near_ties = sum(1 for gap in self._prob_gaps if gap < self._near_tie_gap)
            tail = self._entropies[-16:]

            stride = max(1, (n + self._sample_points - 1) // self._sample_points)
            payload: dict[str, Any] = {
                "version": PAYLOAD_VERSION,
                "attempt": int(attempt),
                "token_count": n,
                "dropped": self._dropped,
                "duration_s": round(duration, 3),
                "tokens_per_s": round(tps, 2) if tps is not None else None,
                "mean_surprisal": round(sum(self._surprisals) / n, 4),
                "p90_surprisal": round(_percentile(sorted_surprisal, 0.90), 4),
                "max_surprisal": round(sorted_surprisal[-1], 4),
                "mean_entropy": round(sum(self._entropies) / n, 4),
                "peak_entropy": round(max(self._entropies), 4),
                "tail_entropy": round(sum(tail) / len(tail), 4),
                "mean_top2_gap": round(sum(self._prob_gaps) / n, 4),
                "near_tie_rate": round(near_ties / n, 4),
                "argmax_rate": round(self._argmax_hits / n, 4),
                "curve": _downsample_mean(self._surprisals, self._curve_points),
                "spikes": self._top_spikes(),
                "token_ids_sample": self._token_ids[::stride][: self._sample_points],
                "logprob_sample": [
                    round(v, 4) for v in self._token_lps[::stride][: self._sample_points]
                ],
            }
            return payload
        except _STEP_RECOVERABLE as exc:
            logger.debug("Interoception finalize failed: %s", exc)
            return None

    def _top_spikes(self) -> list[dict[str, Any]]:
        """The K most surprising moments, with the words they landed on.

        Position 0 is excluded from *selection* (the first token reflects the
        prompt boundary, not content uncertainty) but stays in the aggregates.
        """
        if self._spike_k <= 0:
            return []
        candidates = sorted(
            (
                i for i in range(1, len(self._surprisals))
                if self._surprisals[i] >= _SPIKE_FLOOR_NATS
            ),
            key=lambda i: self._surprisals[i],
            reverse=True,
        )[: self._spike_k]
        spikes = []
        for i in sorted(candidates):
            context = "".join(self._texts[max(0, i - 3): i + 3])
            context = " ".join(context.split())[:60]
            spikes.append(
                {
                    "pos": i,
                    "text": self._texts[i].strip()[:24],
                    "context": context,
                    "surprisal": round(self._surprisals[i], 4),
                }
            )
        return spikes


def interoception_enabled() -> bool:
    return _env_flag("AURA_INTEROCEPTION", True)


def maybe_build_tap() -> InteroceptionTap | None:
    """Env-gated constructor. Returns ``None`` (tap disabled) on any failure."""
    try:
        if not interoception_enabled():
            return None
        return InteroceptionTap()
    except _STEP_RECOVERABLE as exc:
        logger.debug("Interoception tap unavailable: %s", exc)
        return None
