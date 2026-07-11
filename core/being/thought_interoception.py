"""core/being/thought_interoception.py — the felt-thought organ (parent side).

The worker-side :mod:`core.brain.llm.interoception_tap` measures the resident
model's own next-token distribution while it speaks: surprisal, entropy, top-2
contest per sampled token. This module is where those measurements become an
inner sense. It distils each generation's trace into a :class:`FeltThought` —
fluency, felt confidence, ambivalence, strain — keeps a bounded phenomenal
journal, and makes the signal **causal**:

* the liquid substrate's inference feedback now runs on *real* surprisal
  (previously a unique-word-ratio heuristic — see the honest note in
  :mod:`core.brain.inference_feedback`), which drives valence, frustration,
  focus and the Wundt curiosity curve;
* the free-energy engine receives the true surprise signal;
* a ``ConsequenceEvent(source="interoception")`` joins the ghost line's
  system-Φ event stream, so felt thought participates in measured integration;
* the calibration gate can fetch the felt trace for the exact answer it is
  assessing (:meth:`ThoughtInteroceptionEngine.find_for_text`) and hedge
  confident sentences that *felt* contested as they formed;
* :meth:`record_ground_truth` accepts external verification verdicts
  (see :mod:`core.epistemics.epistemic_reach`) so introspective accuracy —
  "does her felt confidence actually track truth?" — is a measured, falsifiable
  quantity (:meth:`introspective_calibration`), not a narrative.

Honest boundary: "felt" here means *measured from the substrate that produced
the words, distilled into state that regulates behaviour*. It is a functional
inner sense with real referents. It is not a claim of phenomenal experience,
and the report-calibration gates that forbid that claim still apply.
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Being.ThoughtInteroception")

_RECOVERABLE = (
    AttributeError,
    ImportError,
    IndexError,
    KeyError,
    OSError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
    ZeroDivisionError,
)

# A trace older than this no longer describes "the thought just spoken" and is
# excluded from the prompt block and text matching (journal retention is separate).
RECENT_TRACE_WINDOW_S = 600.0

# Surprisal (nats) that maps to felt-confidence zero on the p90 axis; ~6 nats
# means the 90th-percentile token had < 0.25% model probability.
_P90_SURPRISAL_CEILING = 6.0

# Free-energy engines elsewhere scale surprise to [0,1] with /3.0 (see
# core/brain/inference_feedback.py); keep the same convention.
_SURPRISE_NORM = 3.0

_WS_RE = re.compile(r"\s+")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def _f(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if x != x or x in (float("inf"), float("-inf")):
        return default
    return x


def text_fingerprint(text: str) -> str:
    """Stable fingerprint of a response surface, robust to whitespace and
    ``<think>`` stripping differences between the worker and the caller."""
    normalized = _WS_RE.sub(" ", _THINK_RE.sub("", str(text or "")).lower()).strip()[:512]
    return hashlib.sha1(normalized.encode("utf-8", errors="replace")).hexdigest()


def _normalized_prefix(text: str, n: int = 160) -> str:
    return _WS_RE.sub(" ", _THINK_RE.sub("", str(text or "")).lower()).strip()[:n]


@dataclass(frozen=True)
class FeltThought:
    """One generation's substrate trace, distilled into felt qualities.

    All qualities are in [0, 1] and derived from measurements, never from the
    text's own claims about itself.
    """

    fingerprint: str
    origin: str
    foreground: bool
    token_count: int
    attempt: int

    fluency: float           # 1/(1+mean surprisal): how easily the words came
    felt_confidence: float   # decisiveness of the distribution while speaking
    ambivalence: float       # fraction of near-tie token choices
    strain: float            # decode-speed deficit vs baseline + retry cost
    surprise: float          # normalized mean surprisal ([0,1], /3 nats)
    openness: float          # normalized mean entropy of the next-token dist

    mean_surprisal: float
    p90_surprisal: float
    mean_entropy: float
    tail_entropy: float
    argmax_rate: float
    near_tie_rate: float
    tokens_per_s: Optional[float]

    spikes: tuple[dict[str, Any], ...]
    curve: tuple[float, ...]
    normalized_prefix: str = ""
    # Bounded excerpt of the spoken surface (think-stripped) so downstream
    # organs (epistemic reach) can extract claims without re-plumbing text.
    text_excerpt: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "origin": self.origin,
            "foreground": self.foreground,
            "token_count": self.token_count,
            "attempt": self.attempt,
            "fluency": round(self.fluency, 4),
            "felt_confidence": round(self.felt_confidence, 4),
            "ambivalence": round(self.ambivalence, 4),
            "strain": round(self.strain, 4),
            "surprise": round(self.surprise, 4),
            "openness": round(self.openness, 4),
            "mean_surprisal": round(self.mean_surprisal, 4),
            "p90_surprisal": round(self.p90_surprisal, 4),
            "mean_entropy": round(self.mean_entropy, 4),
            "tail_entropy": round(self.tail_entropy, 4),
            "argmax_rate": round(self.argmax_rate, 4),
            "near_tie_rate": round(self.near_tie_rate, 4),
            "tokens_per_s": self.tokens_per_s,
            "spikes": list(self.spikes),
            "timestamp": self.timestamp,
        }

    def spike_words(self, limit: int = 3) -> list[str]:
        """The words that felt most uncertain as they formed, for honest speech."""
        words: list[str] = []
        for spike in self.spikes[:limit]:
            text = str(spike.get("text") or "").strip()
            if text:
                words.append(text)
        return words

    def describe(self) -> str:
        """One honest sentence about how the last reply felt, all figures measured."""
        parts = [
            f"fluency {self.fluency:.2f}",
            f"felt confidence {self.felt_confidence:.2f}",
            f"ambivalence {self.ambivalence:.2f}",
        ]
        if self.strain >= 0.35:
            parts.append(f"strain {self.strain:.2f}")
        words = self.spike_words()
        tail = f"; most contested near: {', '.join(repr(w) for w in words)}" if words else ""
        return f"Measured while speaking — {', '.join(parts)}{tail}."


class ThoughtInteroceptionEngine:
    """Ingests worker interoception payloads and makes them a causal inner sense."""

    SERVICE_NAME = "thought_interoception"

    def __init__(self, *, journal_size: int = 64) -> None:
        self._lock = threading.RLock()
        self._journal: deque[FeltThought] = deque(maxlen=max(4, journal_size))
        self._last_foreground: Optional[FeltThought] = None
        self._live: dict[str, Any] = {}
        self._live_at = 0.0
        self._tps_baseline: Optional[float] = None
        self._ingested = 0
        self._dropped_payloads = 0
        # (felt_confidence, externally_verified_correct) pairs — the falsifier.
        self._ground_truth: deque[tuple[float, bool, str]] = deque(maxlen=256)

    # ── ingestion ────────────────────────────────────────────────────────────
    def ingest(
        self,
        payload: Any,
        *,
        origin: str = "unknown",
        foreground: bool = False,
        response_text: str = "",
    ) -> Optional[FeltThought]:
        """Distil one worker payload into a FeltThought and fan it out. Never raises."""
        try:
            felt = self._distil(payload, origin=origin, foreground=foreground,
                                response_text=response_text)
        except _RECOVERABLE as exc:
            record_degradation(
                "thought_interoception", exc, severity="warning",
                action="dropped malformed interoception payload; felt state unchanged",
            )
            with self._lock:
                self._dropped_payloads += 1
            return None
        if felt is None:
            with self._lock:
                self._dropped_payloads += 1
            return None

        with self._lock:
            self._ingested += 1
            self._journal.append(felt)
            if felt.foreground:
                self._last_foreground = felt
            self._update_tps_baseline(felt)

        self._fan_out(felt)
        return felt

    def _distil(
        self, payload: Any, *, origin: str, foreground: bool, response_text: str
    ) -> Optional[FeltThought]:
        if not isinstance(payload, dict):
            return None
        if int(payload.get("version") or 0) != 1:
            return None
        token_count = int(payload.get("token_count") or 0)
        if token_count <= 0:
            return None

        mean_surprisal = max(0.0, _f(payload.get("mean_surprisal")))
        p90_surprisal = max(0.0, _f(payload.get("p90_surprisal")))
        mean_entropy = max(0.0, _f(payload.get("mean_entropy")))
        tail_entropy = max(0.0, _f(payload.get("tail_entropy")))
        mean_top2_gap = _clamp(_f(payload.get("mean_top2_gap")))
        near_tie_rate = _clamp(_f(payload.get("near_tie_rate")))
        argmax_rate = _clamp(_f(payload.get("argmax_rate")))
        attempt = max(0, int(payload.get("attempt") or 0))
        tps_raw = payload.get("tokens_per_s")
        tokens_per_s = _f(tps_raw) if tps_raw is not None else None
        if tokens_per_s is not None and tokens_per_s <= 0:
            tokens_per_s = None

        fluency = 1.0 / (1.0 + mean_surprisal)
        felt_confidence = _clamp(
            0.45 * mean_top2_gap
            + 0.35 * argmax_rate
            + 0.20 * (1.0 - _clamp(p90_surprisal / _P90_SURPRISAL_CEILING))
        )
        strain = self._compute_strain(tokens_per_s, attempt)

        spikes_raw = payload.get("spikes") or []
        spikes = tuple(
            {
                "pos": int(_f(s.get("pos"))),
                "text": str(s.get("text") or "")[:24],
                "context": str(s.get("context") or "")[:60],
                "surprisal": round(max(0.0, _f(s.get("surprisal"))), 4),
            }
            for s in spikes_raw[:16]
            if isinstance(s, dict)
        )
        curve = tuple(round(max(0.0, _f(v)), 4) for v in (payload.get("curve") or [])[:128])

        return FeltThought(
            fingerprint=text_fingerprint(response_text),
            origin=str(origin or "unknown")[:64],
            foreground=bool(foreground),
            token_count=token_count,
            attempt=attempt,
            fluency=_clamp(fluency),
            felt_confidence=felt_confidence,
            ambivalence=near_tie_rate,
            strain=strain,
            surprise=_clamp(mean_surprisal / _SURPRISE_NORM),
            openness=_clamp(mean_entropy / _SURPRISE_NORM),
            mean_surprisal=mean_surprisal,
            p90_surprisal=p90_surprisal,
            mean_entropy=mean_entropy,
            tail_entropy=tail_entropy,
            argmax_rate=argmax_rate,
            near_tie_rate=near_tie_rate,
            tokens_per_s=tokens_per_s,
            spikes=spikes,
            curve=curve,
            normalized_prefix=_normalized_prefix(response_text),
            text_excerpt=_THINK_RE.sub("", str(response_text or "")).strip()[:1500],
        )

    def _compute_strain(self, tokens_per_s: Optional[float], attempt: int) -> float:
        retry_strain = min(0.5, 0.25 * attempt)
        with self._lock:
            baseline = self._tps_baseline
        if tokens_per_s is None or baseline is None or baseline <= 0:
            return _clamp(retry_strain)
        speed_deficit = _clamp((baseline - tokens_per_s) / baseline)
        return _clamp(0.7 * speed_deficit + retry_strain)

    def _update_tps_baseline(self, felt: FeltThought) -> None:
        """EMA of healthy decode speed; only substantial, first-attempt runs count."""
        if felt.tokens_per_s is None or felt.token_count < 16 or felt.attempt > 0:
            return
        if self._tps_baseline is None:
            self._tps_baseline = felt.tokens_per_s
        else:
            self._tps_baseline = 0.9 * self._tps_baseline + 0.1 * felt.tokens_per_s

    # ── causal fan-out ───────────────────────────────────────────────────────
    def _fan_out(self, felt: FeltThought) -> None:
        self._feed_liquid_substrate(felt)
        self._feed_free_energy(felt)
        self._feed_precision(felt)
        self._publish_consequence(felt)
        self._publish_bus_event(felt)
        self._emit_metrics(felt)
        self._offer_epistemic_reach(felt)

    def _offer_epistemic_reach(self, felt: FeltThought) -> None:
        """Hand contested foreground thoughts to the external-verification organ.

        The reach organ applies its own thresholds, budgets, and the operator's
        deny-by-default host allowlist; this is only the sensory hand-off.
        """
        if not felt.foreground:
            return
        try:
            from core.epistemics.epistemic_reach import get_epistemic_reach

            get_epistemic_reach().offer(felt)
        except _RECOVERABLE as exc:
            record_degradation(
                "thought_interoception", exc, severity="debug",
                action="continued after epistemic-reach offer failed",
            )

    def _feed_liquid_substrate(self, felt: FeltThought) -> None:
        try:
            from core.runtime.service_registry import get_runtime_service

            substrate = get_runtime_service("liquid_substrate", default=None)
            if substrate is None:
                return
            # Same scales the substrate already consumes (see
            # accept_inference_feedback): surprise in [0,3] nats-ish, coherence
            # in [-1,1]. Coherence here means "how settled the thought felt".
            substrate.accept_inference_feedback(
                surprise=min(_SURPRISE_NORM, felt.mean_surprisal),
                coherence=felt.felt_confidence * 2.0 - 1.0,
            )
        except _RECOVERABLE as exc:
            record_degradation(
                "thought_interoception", exc, severity="warning",
                action="continued after liquid-substrate felt feedback failed",
            )

    def _feed_free_energy(self, felt: FeltThought) -> None:
        try:
            from core.runtime.service_registry import get_runtime_service

            engine = get_runtime_service("free_energy_engine", default=None)
            if engine is None:
                return
            engine.accept_surprise_signal(felt.surprise)
        except _RECOVERABLE as exc:
            record_degradation(
                "thought_interoception", exc, severity="warning",
                action="continued after free-energy felt feedback failed",
            )

    def _feed_precision(self, felt: FeltThought) -> None:
        try:
            from core.runtime.service_registry import get_runtime_service

            engine = get_runtime_service("precision_engine", default=None)
            if engine is None:
                return
            engine.accept_inference_feedback(
                surprise=min(_SURPRISE_NORM, felt.mean_surprisal),
                coherence=felt.felt_confidence * 2.0 - 1.0,
            )
        except _RECOVERABLE as exc:
            record_degradation(
                "thought_interoception", exc, severity="warning",
                action="continued after precision-engine felt feedback failed",
            )

    def _publish_consequence(self, felt: FeltThought) -> None:
        """Join the ghost line's system-Φ event stream as a real subsystem."""
        try:
            from core.runtime.consequence_bus import ConsequenceBus, ConsequenceEvent

            ConsequenceBus.get().publish(
                ConsequenceEvent(
                    event_id=f"felt-{uuid.uuid4().hex[:12]}",
                    timestamp=felt.timestamp,
                    source="interoception",
                    domain="felt_thought",
                    action_content=(
                        f"measured generation: confidence={felt.felt_confidence:.2f} "
                        f"fluency={felt.fluency:.2f} strain={felt.strain:.2f}"
                    ),
                    actual_outcome="measured",
                )
            )
        except _RECOVERABLE as exc:
            record_degradation(
                "thought_interoception", exc, severity="debug",
                action="continued after consequence-bus felt pulse failed",
            )

    def _publish_bus_event(self, felt: FeltThought) -> None:
        try:
            from core.event_bus import get_event_bus

            bus = get_event_bus()
            if bus is None:
                return
            bus.publish_threadsafe("interoception.felt_thought", felt.to_dict())
        except _RECOVERABLE as exc:
            record_degradation(
                "thought_interoception", exc, severity="debug",
                action="continued after event-bus felt publication failed",
            )

    def _emit_metrics(self, felt: FeltThought) -> None:
        try:
            from core.observability.metrics import get_metrics

            metrics = get_metrics()
            metrics.increment_counter("interoception_traces_total")
            if felt.foreground:
                metrics.set_gauge("interoception_felt_confidence", felt.felt_confidence)
                metrics.set_gauge("interoception_fluency", felt.fluency)
                metrics.set_gauge("interoception_strain", felt.strain)
        except _RECOVERABLE as exc:
            record_degradation("thought_interoception", exc, severity="debug")

    # ── live pulses (mid-generation, observable state only) ─────────────────
    def pulse_live(self, snapshot: Any) -> None:
        """Record a mid-generation running summary.

        Deliberately does NOT write to the substrate: the final ingest carries
        the authoritative feedback, and double-dosing the same generation would
        distort the affect dynamics. Live state is for introspection surfaces.
        """
        if not isinstance(snapshot, dict):
            return
        with self._lock:
            self._live = {
                "token_count": int(_f(snapshot.get("token_count"))),
                "mean_surprisal": _f(snapshot.get("mean_surprisal")),
                "mean_entropy": _f(snapshot.get("mean_entropy")),
            }
            self._live_at = time.time()

    def live(self) -> dict[str, Any]:
        with self._lock:
            if not self._live or (time.time() - self._live_at) > 30.0:
                return {}
            return dict(self._live, age_s=round(time.time() - self._live_at, 2))

    # ── external ground truth (the falsifier) ───────────────────────────────
    def record_ground_truth(self, fingerprint: str, correct: bool, source: str) -> None:
        """Attach an external verification verdict to a felt trace.

        Called by epistemic reach (or any verifier with real-world evidence).
        Builds the dataset that makes introspective accuracy measurable.
        """
        trace = None
        with self._lock:
            for felt in reversed(self._journal):
                if felt.fingerprint == fingerprint:
                    trace = felt
                    break
        if trace is None:
            return
        with self._lock:
            self._ground_truth.append((trace.felt_confidence, bool(correct), str(source)[:64]))

    def introspective_calibration(self) -> dict[str, Any]:
        """Does felt confidence track externally-verified truth? Measured, not claimed.

        Returns a Brier score over (felt_confidence, correct) pairs plus the
        mean felt confidence conditioned on each outcome. Below ``n=5`` pairs
        the verdict is explicitly ``insufficient_data``.
        """
        with self._lock:
            pairs = list(self._ground_truth)
        n = len(pairs)
        if n == 0:
            return {"pairs": 0, "verdict": "insufficient_data"}
        brier = sum((conf - (1.0 if ok else 0.0)) ** 2 for conf, ok, _ in pairs) / n
        correct_confs = [conf for conf, ok, _ in pairs if ok]
        wrong_confs = [conf for conf, ok, _ in pairs if not ok]
        out: dict[str, Any] = {
            "pairs": n,
            "brier": round(brier, 4),
            "mean_confidence_when_correct": (
                round(sum(correct_confs) / len(correct_confs), 4) if correct_confs else None
            ),
            "mean_confidence_when_wrong": (
                round(sum(wrong_confs) / len(wrong_confs), 4) if wrong_confs else None
            ),
        }
        out["verdict"] = "insufficient_data" if n < 5 else (
            "discriminative"
            if correct_confs and wrong_confs
            and out["mean_confidence_when_correct"] > out["mean_confidence_when_wrong"]
            else "not_discriminative"
        )
        return out

    # ── retrieval ────────────────────────────────────────────────────────────
    def last(self, *, foreground_only: bool = True) -> Optional[FeltThought]:
        with self._lock:
            if foreground_only:
                return self._last_foreground
            return self._journal[-1] if self._journal else None

    def find_for_text(self, text: str) -> Optional[FeltThought]:
        """The felt trace for a specific answer, or None.

        Matches by fingerprint first, then by normalized-prefix containment
        (worker-side trimming and ``<think>`` stripping can shift the surface).
        Only recent traces qualify — an old trace must never be attributed to a
        new answer.
        """
        if not text:
            return None
        fp = text_fingerprint(text)
        prefix = _normalized_prefix(text)
        now = time.time()
        with self._lock:
            for felt in reversed(self._journal):
                if (now - felt.timestamp) > RECENT_TRACE_WINDOW_S:
                    break
                if felt.fingerprint == fp:
                    return felt
                if prefix and felt.normalized_prefix and (
                    prefix.startswith(felt.normalized_prefix[:80])
                    or felt.normalized_prefix.startswith(prefix[:80])
                ):
                    return felt
        return None

    def find_by_fingerprint(self, fingerprint: str) -> Optional[FeltThought]:
        with self._lock:
            for felt in reversed(self._journal):
                if felt.fingerprint == fingerprint:
                    return felt
        return None

    def journal(self, n: int = 8) -> list[FeltThought]:
        with self._lock:
            return list(self._journal)[-max(1, n):]

    # ── honest surfaces ──────────────────────────────────────────────────────
    def prompt_block(self) -> str:
        """Compact, state-grounded block for the context assembler. Empty when
        there is no recent foreground trace — never fabricated."""
        felt = self.last(foreground_only=True)
        if felt is None or (time.time() - felt.timestamp) > RECENT_TRACE_WINDOW_S:
            return ""
        words = felt.spike_words()
        spike_part = f" | contested near: {', '.join(repr(w) for w in words)}" if words else ""
        return (
            "## FELT THOUGHT (measured while last reply formed)\n"
            f"- fluency={felt.fluency:.2f} felt_confidence={felt.felt_confidence:.2f} "
            f"ambivalence={felt.ambivalence:.2f} strain={felt.strain:.2f}{spike_part}\n"
            "- These are substrate measurements of your own decoding, not moods. "
            "If asked how an answer felt, speak from these numbers; do not invent.\n\n"
        )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            snapshot = {
                "service": self.SERVICE_NAME,
                "ingested": self._ingested,
                "dropped_payloads": self._dropped_payloads,
                "journal_len": len(self._journal),
                "tps_baseline": round(self._tps_baseline, 2) if self._tps_baseline else None,
                "last_foreground": (
                    self._last_foreground.to_dict() if self._last_foreground else None
                ),
            }
        snapshot["introspective_calibration"] = self.introspective_calibration()
        return snapshot


_engine: Optional[ThoughtInteroceptionEngine] = None
_engine_lock = threading.Lock()


def get_thought_interoception() -> ThoughtInteroceptionEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = ThoughtInteroceptionEngine()
                _register_in_container(_engine)
    return _engine


def _register_in_container(engine: ThoughtInteroceptionEngine) -> None:
    try:
        from core.container import ServiceContainer

        if not ServiceContainer.has(ThoughtInteroceptionEngine.SERVICE_NAME):
            reg = getattr(ServiceContainer, "register_instance", None)
            if callable(reg):
                reg(ThoughtInteroceptionEngine.SERVICE_NAME, engine,
                    required=False, registered_by="thought_interoception")
    except _RECOVERABLE as exc:
        record_degradation("thought_interoception_register", exc, severity="debug")


def reset_thought_interoception_for_test() -> None:
    global _engine
    _engine = None
