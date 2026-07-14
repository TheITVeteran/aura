"""core/kernel/feedback_observer.py - The Causal Chain Observer."""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.state.aura_state import phenomenal_text

logger = logging.getLogger("Aura.FeedbackObserver")

_FEEDBACK_RECOVERABLE_ERRORS = (RuntimeError, AttributeError, TypeError, ValueError)
_CALLBACK_FAILURE_LIMIT = 3

if TYPE_CHECKING:
    from core.state.aura_state import AuraState


def _record_feedback_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "feedback_observer",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.AUDIT_GAP,
        receipt_required=severity in {"degraded", "critical"},
        extra=extra,
    )


def _callback_label(callback: Callable[[TickEntry], None]) -> str:
    module = getattr(callback, "__module__", "") or ""
    name = getattr(callback, "__qualname__", None) or getattr(callback, "__name__", None)
    if name:
        return f"{module}.{name}" if module else str(name)
    return type(callback).__qualname__


@dataclass
class ObserverCallback:
    callback: Callable[[TickEntry], None]
    label: str
    consecutive_failures: int = 0
    disabled: bool = False
    disabled_reason: str = ""


@dataclass
class TickEntry:
    """One complete tick in the causal chain."""
    tick_id:       int
    timestamp:     float
    objective:     str

    # ── Snapshot before phases run ───────────────────────────────────────────
    phi_before:          float
    mode_before:         str
    valence_before:      float
    arousal_before:      float
    curiosity_before:    float
    dominant_before:     str
    phenomenal_before:   Any | None
    top_emotions_before: dict[str, float]   # top-3 by value
    origin:              str                 = ""
    priority:            bool                = False

    # ── Set after phases complete ────────────────────────────────────────────
    phi_after:           float               = 0.0
    mode_after:          str                 = ""
    valence_after:       float               = 0.0
    arousal_after:       float               = 0.0
    curiosity_after:     float               = 0.0
    dominant_after:      str                 = ""
    phenomenal_after:    Any | None          = None
    top_emotions_after:  dict[str, float]    = field(default_factory=dict)
    response_preview:    str                 = ""   # first 120 chars
    tick_duration_ms:    float               = 0.0
    priority_tick:       bool                = False
    phase_durations_ms:  dict[str, float]    = field(default_factory=dict)

    # ── Derived deltas ───────────────────────────────────────────────────────
    @property
    def valence_delta(self) -> float:
        """Change in valence across this tick (after minus before)."""
        return round(self.valence_after - self.valence_before, 4)

    @property
    def phi_delta(self) -> float:
        """Change in phi (integrated information) across this tick."""
        return round(self.phi_after - self.phi_before, 4)

    @property
    def curiosity_delta(self) -> float:
        """Change in curiosity level across this tick."""
        return round(self.curiosity_after - self.curiosity_before, 4)

    @property
    def mode_changed(self) -> bool:
        """True if the cognitive mode changed during this tick."""
        return self.mode_before != self.mode_after

    @property
    def affect_drove_mode(self) -> bool:
        """True if phi change was large enough to have gated mode."""
        return abs(self.phi_delta) > 0.05 and self.mode_changed

    @property
    def is_user_facing(self) -> bool:
        if self.priority:
            return True
        return str(self.origin or "").strip().lower() in {
            "user", "voice", "admin", "api", "gui", "ws", "websocket", "direct", "external",
        }

    def summary(self) -> str:
        """One-line causal chain summary - log this each tick."""
        mood_change = (
            f"{self.dominant_before}→{self.dominant_after}"
            if self.dominant_before != self.dominant_after
            else self.dominant_after
        )
        mode_str = (
            f"{self.mode_before}→{self.mode_after}"
            if self.mode_changed else self.mode_after
        )
        phi_str  = f"phi={self.phi_before:.3f}→{self.phi_after:.3f}"
        val_str  = f"val={self.valence_before:+.2f}→{self.valence_after:+.2f}"
        resp_str = f'"{self.response_preview[:60]}..."' if self.response_preview else "(no response)"
        return (
            f"[tick={self.tick_id}  {phi_str}  {val_str}  "
            f"mood={mood_change}  mode={mode_str}  {int(self.tick_duration_ms)}ms]\n"
            f"  obj: {self.objective[:80]}\n"
            f"  phen: {phenomenal_text(self.phenomenal_after) or '—'}\n"
            f"  resp: {resp_str}"
        )

    def to_dict(self) -> dict:
        """Serialize the full tick entry to a JSON-safe dictionary."""
        return {
            "tick_id":          self.tick_id,
            "timestamp":        self.timestamp,
            "objective":        self.objective[:200],
            "origin":           self.origin,
            "priority":         self.priority,
            "is_user_facing":   self.is_user_facing,
            "phi":              {"before": self.phi_before,
                                 "after":  self.phi_after,
                                 "delta":  self.phi_delta},
            "valence":          {"before": self.valence_before,
                                 "after":  self.valence_after,
                                 "delta":  self.valence_delta},
            "curiosity":        {"before": self.curiosity_before,
                                 "after":  self.curiosity_after,
                                 "delta":  self.curiosity_delta},
            "mood":             {"before": self.dominant_before,
                                 "after":  self.dominant_after},
            "mode":             {"before": self.mode_before,
                                 "after":  self.mode_after,
                                 "changed": self.mode_changed},
            "phenomenal":       phenomenal_text(self.phenomenal_after),
            "affect_drove_mode": self.affect_drove_mode,
            "response_preview": self.response_preview[:120],
            "tick_duration_ms": round(self.tick_duration_ms, 1),
            "priority_tick":    self.priority_tick,
            "phase_durations_ms": dict(self.phase_durations_ms),
        }


class FeedbackObserver:
    """
    Tracks the causal chain across ticks.

    The loop is:
      affect(N) -> phi(N) -> phenomenal_state(N) -> system_prompt(N)
               -> response(N) -> affect_delta(N->N+1)

    Call begin_tick() before the phase pipeline runs.
    Call end_tick() after the pipeline completes.
    """

    MAX_HISTORY = 200   # Keep last 200 ticks in memory

    def __init__(self):
        """Initialize the observer with an empty history and no registered callbacks."""
        self._history:   deque[TickEntry] = deque(maxlen=self.MAX_HISTORY)
        self._tick_id:   int = 0
        self._callbacks: list[ObserverCallback] = []

    # ── Public API ──────────────────────────────────────────────────────────────

    def begin_tick(
        self,
        state: AuraState,
        objective: str,
        *,
        origin: str = "",
        priority: bool = False,
    ) -> TickEntry:
        """
        Snapshot state BEFORE the phase pipeline runs.
        Returns a TickEntry that must be passed to end_tick().
        """
        self._tick_id += 1
        e      = state.affect.emotions
        top3   = dict(sorted(e.items(), key=lambda x: x[1], reverse=True)[:3])

        entry = TickEntry(
            tick_id        = self._tick_id,
            timestamp      = time.time(),
            objective      = objective[:200],
            origin         = str(origin or ""),
            priority       = bool(priority),
            phi_before     = state.phi,
            mode_before    = state.cognition.current_mode.value,
            valence_before = state.affect.valence,
            arousal_before = state.affect.arousal,
            curiosity_before = state.affect.curiosity,
            dominant_before  = state.affect.dominant_emotion,
            phenomenal_before = state.cognition.phenomenal_state,
            top_emotions_before = top3,
            priority_tick   = bool(priority),
        )
        return entry

    def end_tick(
        self,
        entry: TickEntry,
        response: str | None,
        new_state: AuraState,
        start_time: float,
    ) -> TickEntry:
        """
        Snapshot state AFTER the phase pipeline completes.
        Fires all registered callbacks. Stores in history.
        """
        e    = new_state.affect.emotions
        top3 = dict(sorted(e.items(), key=lambda x: x[1], reverse=True)[:3])

        entry.phi_after         = new_state.phi
        entry.mode_after        = new_state.cognition.current_mode.value
        entry.valence_after     = new_state.affect.valence
        entry.arousal_after     = new_state.affect.arousal
        entry.curiosity_after   = new_state.affect.curiosity
        entry.dominant_after    = new_state.affect.dominant_emotion
        entry.phenomenal_after  = new_state.cognition.phenomenal_state
        entry.top_emotions_after = top3
        entry.response_preview  = (response or "")[:120]
        entry.tick_duration_ms  = (time.time() - start_time) * 1000.0

        self._history.append(entry)

        # Fire callbacks without letting optional observers destabilize the tick.
        for binding in self._callbacks:
            if binding.disabled:
                continue
            try:
                binding.callback(entry)
            except _FEEDBACK_RECOVERABLE_ERRORS as _e:
                binding.consecutive_failures += 1
                extra = {
                    "callback": binding.label,
                    "tick_id": entry.tick_id,
                    "consecutive_failures": binding.consecutive_failures,
                    "failure_limit": _CALLBACK_FAILURE_LIMIT,
                }
                if binding.consecutive_failures >= _CALLBACK_FAILURE_LIMIT:
                    binding.disabled = True
                    binding.disabled_reason = f"{type(_e).__qualname__}: {str(_e)[:160]}"
                    action = (
                        "disabled failing feedback callback after repeated failures; "
                        "tick history and other observers continued"
                    )
                else:
                    action = "isolated feedback callback failure; tick history and other observers continued"
                _record_feedback_degradation(_e, action=action, extra=extra)
                logger.debug("Feedback callback %s failed: %s", binding.label, _e)
            else:
                binding.consecutive_failures = 0

        return entry

    def on_tick(self, callback: Callable[[TickEntry], None]) -> None:
        """Register a callback that fires after every tick completes."""
        if not callable(callback):
            raise TypeError("feedback observer callback must be callable")
        self._callbacks.append(ObserverCallback(callback=callback, label=_callback_label(callback)))

    def get_callback_status(self) -> list[dict[str, Any]]:
        """Return callback health for dashboards and runtime diagnostics."""
        return [
            {
                "label": binding.label,
                "consecutive_failures": binding.consecutive_failures,
                "disabled": binding.disabled,
                "disabled_reason": binding.disabled_reason,
            }
            for binding in self._callbacks
        ]

    def get_last_trace(self, n: int = 10) -> list[TickEntry]:
        """Return the last N TickEntry records."""
        return list(self._history)[-n:]

    def get_current_loop_state(self) -> dict[str, Any]:
        """
        Snapshot of the live loop: latest tick + a short causal summary.
        Useful for dashboards and debug endpoints.
        """
        if not self._history:
            return {
                "status": "no_ticks_yet",
                "callbacks": self.get_callback_status(),
            }

        last = self._history[-1]
        recent = list(self._history)[-5:]

        # Compute running averages
        avg_phi_delta     = sum(t.phi_delta     for t in recent) / len(recent)
        avg_valence_delta = sum(t.valence_delta for t in recent) / len(recent)

        return {
            "tick_id":          last.tick_id,
            "current_phi":      last.phi_after,
            "current_valence":  last.valence_after,
            "current_mood":     last.dominant_after,
            "current_mode":     last.mode_after,
            "phenomenal_state": phenomenal_text(last.phenomenal_after),
            "last_response":    last.response_preview,
            "affect_loop_active": abs(avg_valence_delta) > 0.005,
            "phi_trending":     "rising" if avg_phi_delta > 0.01
                                else "falling" if avg_phi_delta < -0.01
                                else "stable",
            "recent_ticks":     len(recent),
            "avg_phi_delta_5":  round(avg_phi_delta, 4),
            "avg_val_delta_5":  round(avg_valence_delta, 4),
            "callbacks":        self.get_callback_status(),
        }

    def print_loop(self, n: int = 5) -> None:
        """Print the last N ticks to stdout — call this from a debug shell."""
        entries = self.get_last_trace(n)
        if not entries:
            logger.info("[FeedbackObserver] No ticks recorded yet.")
            return
        logger.info("\n%s", '─'*70)
        logger.info("  AURA FEEDBACK LOOP — last %s ticks", len(entries))
        logger.info("%s", '─'*70)
        for e in entries:
            logger.info(e.summary())
            logger.info("")
        loop = self.get_current_loop_state()
        logger.info(f"  Loop active: {loop.get('affect_loop_active')}  "
              f"Phi trend: {loop.get('phi_trending')}  "
              f"avg Δval/tick: {loop.get('avg_val_delta_5')}")
        logger.info("%s\n", '─'*70)
