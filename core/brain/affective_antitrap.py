"""core/brain/affective_antitrap.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Anti-trap governor for affect-modulated sampling: breaks the
digital-depression feedback loop.

The trap (external review, July 3): an error cascade spikes distress
telemetry (cortisol, error pressure, low coherence) → the latent bridge
clamps sampling temperature to its deterministic floor for safety → at
floor temperature the model loses the semantic variance needed to invent
the fix → errors persist → distress persists → the clamp never lifts.
The system wedges itself in a hyper-deterministic depression.

Three mechanisms, all bounded and observable:

1. TRAP DETECTION — a ring of recent (temperature, distress) observations.
   Trap = temperature pinned at/near floor for the whole window AND
   distress not improving across it. Both conditions required: pinned
   temperature while distress RECOVERS is the clamp working; distress
   without pinning is ordinary stress.
2. GOVERNED ESCAPE — when trapped, raise the temperature floor to an
   exploration band for a bounded number of computations (annealing
   escape), record an AFFECT-TRAP fault occurrence with the observed
   window, then re-measure: the escape's efficacy (distress delta) is
   written into the fault record. A cooldown prevents oscillation.
3. LANE DECOUPLING — self-repair/ideation lanes get a guaranteed
   exploration floor at ALL times: safety clamps may govern speech, but
   the lanes that must invent a way out are never starved of variance.
   (Deterministic final code EMISSION stays at temp 0 by design — the
   floor protects idea search, not token-exact codegen.)
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.AffectiveAntitrap")

# Trap detection window and thresholds.
WINDOW = 8                       # observations examined for the trap signature
FLOOR_EPSILON = 0.05             # "pinned" = within this of the hard floor
TEMP_HARD_FLOOR = 0.15           # latent bridge's deterministic floor
ESCAPE_TEMP_FLOOR = 0.45         # exploration band floor during escape
ESCAPE_SPAN = 6                  # computations the escape floor persists for
COOLDOWN_S = 600.0               # min seconds between escapes
LANE_EXPLORATION_FLOOR = 0.50    # repair/ideation lanes never sample below this

# Lanes whose purpose is inventing a way out — never variance-starved.
EXPLORATION_LANES = frozenset({"repair", "ideation", "discovery", "self_repair"})


@dataclass
class _Observation:
    at: float
    temperature: float
    distress: float


def distress_score(substrate: dict[str, float]) -> float:
    """Collapse the distress-relevant substrate axes to one 0..1 score."""
    cortisol = float(substrate.get("cortisol", 0.3))
    error_pressure = float(substrate.get("active_error_pressure", 0.0))
    frustration = float(substrate.get("frustration", 0.0))
    incoherence = 1.0 - float(substrate.get("organismal_coherence", 1.0))
    raw = 0.35 * cortisol + 0.30 * error_pressure + 0.20 * frustration + 0.15 * incoherence
    return max(0.0, min(1.0, raw))


class AffectiveTrapGuard:
    """Observes every sampling-parameter computation; breaks closed loops.

    Thread-safe. observe_and_adjust() is O(WINDOW) on a tiny deque —
    negligible on the inference-parameter path.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ring: deque[_Observation] = deque(maxlen=WINDOW)
        self._escape_remaining = 0
        self._last_escape_at = 0.0
        self._escape_entry_distress = 0.0
        self._escapes_fired = 0

    # ── the seam: called by the latent bridge per computation ────────

    def observe_and_adjust(
        self,
        temperature: float,
        substrate: dict[str, float],
        *,
        lane: str = "speech",
    ) -> tuple[float, str]:
        """Record the observation; return (adjusted_temperature, note).

        note is empty when nothing changed — the latent bridge appends it
        to its rationale so every intervention is visible in telemetry.
        """
        score = distress_score(substrate)
        now = time.monotonic()

        with self._lock:
            self._ring.append(_Observation(at=now, temperature=temperature, distress=score))

            # Lane decoupling comes first and is unconditional.
            if lane in EXPLORATION_LANES and temperature < LANE_EXPLORATION_FLOOR:
                return (
                    LANE_EXPLORATION_FLOOR,
                    f"antitrap: lane '{lane}' exploration floor "
                    f"{LANE_EXPLORATION_FLOOR:.2f} (was {temperature:.2f})",
                )

            # Active escape window.
            if self._escape_remaining > 0:
                self._escape_remaining -= 1
                if self._escape_remaining == 0:
                    self._complete_escape(score)
                if temperature < ESCAPE_TEMP_FLOOR:
                    return (
                        ESCAPE_TEMP_FLOOR,
                        f"antitrap: escape floor {ESCAPE_TEMP_FLOOR:.2f} "
                        f"({self._escape_remaining} computations remain)",
                    )
                return (temperature, "")

            if self._is_trapped() and (now - self._last_escape_at) >= COOLDOWN_S:
                self._begin_escape(score)
                return (
                    max(temperature, ESCAPE_TEMP_FLOOR),
                    "antitrap: TRAP DETECTED — temperature pinned at floor "
                    f"with non-improving distress ({score:.2f}); escape band "
                    f"opened for {ESCAPE_SPAN} computations",
                )

        return (temperature, "")

    # ── trap signature ────────────────────────────────────────────────

    def _is_trapped(self) -> bool:
        if len(self._ring) < WINDOW:
            return False
        observations = list(self._ring)
        pinned = all(
            o.temperature <= TEMP_HARD_FLOOR + FLOOR_EPSILON for o in observations
        )
        if not pinned:
            return False
        half = WINDOW // 2
        early = sum(o.distress for o in observations[:half]) / half
        late = sum(o.distress for o in observations[half:]) / (WINDOW - half)
        # Distress not improving: the clamp is no longer buying recovery.
        return late >= early - 0.02

    # ── escape lifecycle ──────────────────────────────────────────────

    def _begin_escape(self, entry_distress: float) -> None:
        self._escape_remaining = ESCAPE_SPAN
        self._last_escape_at = time.monotonic()
        self._escape_entry_distress = entry_distress
        self._escapes_fired += 1
        logger.warning(
            "AFFECTIVE TRAP: temperature pinned at floor with non-improving "
            "distress (%.2f) — opening exploration escape band for %d computations",
            entry_distress, ESCAPE_SPAN,
        )
        self._record_fault(
            details=(
                f"trap window: temp<=~{TEMP_HARD_FLOOR + FLOOR_EPSILON:.2f} x{WINDOW}, "
                f"distress {entry_distress:.2f} non-improving; escape opened"
            ),
            recovered=False,
        )

    def _complete_escape(self, exit_distress: float) -> None:
        delta = self._escape_entry_distress - exit_distress
        logger.info(
            "AFFECTIVE TRAP escape complete: distress %.2f → %.2f (delta %+.2f)",
            self._escape_entry_distress, exit_distress, delta,
        )
        self._record_fault(
            details=(
                f"escape complete: distress {self._escape_entry_distress:.2f}"
                f" → {exit_distress:.2f} (delta {delta:+.2f})"
            ),
            recovered=delta > 0.0,
        )

    def _record_fault(self, *, details: str, recovered: bool) -> None:
        try:
            from core.resilience.fault_taxonomy import get_fault_registry
            get_fault_registry().record_fault(
                "AFFECT-TRAP", subsystem="latent_bridge.antitrap",
                details=details, recovered=recovered,
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.debug("Fault registry unavailable for affect trap: %s", exc)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "window_filled": len(self._ring),
                "escape_active": self._escape_remaining > 0,
                "escape_remaining": self._escape_remaining,
                "escapes_fired": self._escapes_fired,
                "cooldown_s": COOLDOWN_S,
                "lane_floor": LANE_EXPLORATION_FLOOR,
            }


_guard: AffectiveTrapGuard | None = None
_guard_lock = threading.Lock()


def get_affective_trap_guard() -> AffectiveTrapGuard:
    global _guard
    if _guard is None:
        with _guard_lock:
            if _guard is None:
                _guard = AffectiveTrapGuard()
    return _guard
