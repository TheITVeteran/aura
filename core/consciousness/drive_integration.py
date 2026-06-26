"""Drive-integration volition — a competition of accumulating drives, not VAD thresholds.

The critique of AttractorVolitionEngine: "still uses simple VAD thresholds and a 30-second
refractory period. It's a start, not a will." Two concrete weaknesses: a drive fires the instant
an (valence, arousal, dominance) point crosses a hard line (a transient spike acts the same as a
sustained pull), and a flat 30-second clock gates everything regardless of how strong the urge is.

This replaces both with a small dynamical system:

  * Temporal integration — each drive is a leaky integrator accumulating its instantaneous
    activation over time. A sustained moderate pull builds to action; a one-off spike decays
    before it commits. Volition reflects *history*, not a single sample.
  * Competition — drives mutually inhibit (soft winner-take-all over integrated activation), so a
    decision is the resolution of competing urges, not independent thresholds each firing freely.
  * Hysteresis instead of a flat refractory — a drive commits when its integrated activation
    crosses a high threshold and is then suppressed until it falls below a lower one (a Schmitt
    trigger). The "cooldown" is a consequence of the dynamics (the winner is depleted on firing),
    not a hard-coded 30 s — so a strong drive can re-assert sooner and a weak one waits longer.

Drives are grounded in real signals — VAD from the substrate, plus nociception pressure and
value-model urges where available — not VAD alone. This is a genuine drive-arbitration mechanism;
it is deliberately *not* called free will. It decides which urge wins right now and when.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from core.runtime.errors import record_degradation

logger = logging.getLogger("Volition.DriveIntegration")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class Drive:
    """One motivational channel: a leaky integrator with Schmitt-trigger commitment.

    ``activation`` accumulates instantaneous drive (``leak`` pulls it back toward 0 each step).
    It *fires* when activation crosses ``fire_threshold`` and is then ``suppressed`` (won't fire
    again) until it falls below ``release_threshold`` — hysteresis, not a flat clock.
    """

    name: str
    action: str
    gain: float = 1.0               # how fast this drive accumulates from its signal
    leak: float = 0.15              # fraction of activation lost per second
    fire_threshold: float = 0.7
    release_threshold: float = 0.35
    activation: float = 0.0
    suppressed: bool = False
    last_fired: float = 0.0

    def integrate(self, signal: float, dt: float) -> None:
        """Accumulate ``signal`` (its instantaneous pull) with leak over ``dt`` seconds."""
        decay = math.exp(-self.leak * max(0.0, dt))
        self.activation = _clamp(self.activation * decay + self.gain * _clamp(signal) * dt)
        if self.suppressed and self.activation <= self.release_threshold:
            self.suppressed = False  # hysteresis: re-armed once it has cooled

    def ready(self) -> bool:
        return (not self.suppressed) and self.activation >= self.fire_threshold

    def fire(self, now: float) -> None:
        self.suppressed = True
        self.last_fired = now
        self.activation *= 0.4  # firing depletes the urge (the cooldown emerges from this)

    def to_dict(self) -> Dict[str, float]:
        return {"name": self.name, "action": self.action,
                "activation": round(self.activation, 4), "suppressed": self.suppressed}


@dataclass
class VolitionDecision:
    action: Optional[str]
    drive: Optional[str]
    activation: float
    competitors: Dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "action": self.action,
            "drive": self.drive,
            "activation": round(self.activation, 4),
            "competitors": {k: round(v, 4) for k, v in self.competitors.items()},
            "reason": self.reason,
        }


# Each drive maps VAD (+ extra signals) to an instantaneous pull in [0,1].
def _curiosity_signal(s: Dict[str, float]) -> float:
    return _clamp(0.5 * max(0.0, s.get("arousal", 0.0)) + 0.5 * max(0.0, s.get("valence", 0.0))
                  + 0.4 * s.get("novelty", 0.0))


def _boredom_signal(s: Dict[str, float]) -> float:
    return _clamp(0.6 * max(0.0, -s.get("arousal", 0.0)) + 0.4 * max(0.0, -s.get("valence", 0.0))
                  - 0.3 * s.get("novelty", 0.0))


def _reflection_signal(s: Dict[str, float]) -> float:
    return _clamp(0.6 * max(0.0, s.get("dominance", 0.0)) + 0.4 * max(0.0, -s.get("arousal", 0.0)))


def _relief_signal(s: Dict[str, float]) -> float:
    # Pain/nociception pressure drives a regulatory (relief/stabilize) urge.
    return _clamp(s.get("pain", 0.0))


_DRIVE_SIGNALS: Dict[str, Callable[[Dict[str, float]], float]] = {
    "curiosity": _curiosity_signal,
    "boredom": _boredom_signal,
    "reflection": _reflection_signal,
    "relief": _relief_signal,
}


class DriveIntegrationEngine:
    """Integrates competing drives over time and arbitrates which urge acts now."""

    def __init__(self, *, inhibition: float = 0.5) -> None:
        self._inhibition = inhibition
        self._drives: Dict[str, Drive] = {
            "curiosity": Drive("curiosity", "explore_knowledge", gain=1.0, leak=0.15),
            "boredom": Drive("boredom", "seek_novelty", gain=0.8, leak=0.1),
            "reflection": Drive("reflection", "deep_reflection", gain=0.7, leak=0.12),
            "relief": Drive("relief", "stabilize", gain=1.4, leak=0.2,
                            fire_threshold=0.6, release_threshold=0.3),
        }
        self._last_step = time.time()

    def add_drive(self, drive: Drive, signal_fn: Callable[[Dict[str, float]], float]) -> None:
        self._drives[drive.name] = drive
        _DRIVE_SIGNALS[drive.name] = signal_fn

    def step(self, signals: Dict[str, float], *, now: Optional[float] = None,
             dt: Optional[float] = None) -> VolitionDecision:
        """Advance the dynamics one tick and return the winning drive's action (or none).

        ``signals`` carries the grounding (valence/arousal/dominance/novelty/pain). The winner is
        the highest-integrated *ready* drive after mutual inhibition; firing depletes it.
        """
        now = time.time() if now is None else now
        if dt is None:
            dt = max(1e-3, now - self._last_step)
        self._last_step = now

        # 1) integrate each drive from its instantaneous signal
        for name, drive in self._drives.items():
            sig_fn = _DRIVE_SIGNALS.get(name)
            raw = sig_fn(signals) if sig_fn else 0.0
            drive.integrate(raw, dt)

        # 2) mutual inhibition: each drive is suppressed by the strongest *other* drive
        activations = {n: d.activation for n, d in self._drives.items()}
        strongest = max(activations.values()) if activations else 0.0
        effective: Dict[str, float] = {}
        for name, drive in self._drives.items():
            others_max = max((a for n, a in activations.items() if n != name), default=0.0)
            effective[name] = _clamp(drive.activation - self._inhibition * others_max)

        # 3) the winner must be the strongest AND past its own fire threshold (hysteresis-armed)
        ready = [(n, d) for n, d in self._drives.items() if d.ready()]
        if not ready:
            return VolitionDecision(action=None, drive=None, activation=strongest,
                                    competitors=activations, reason="no_drive_ready")
        winner_name, winner = max(ready, key=lambda nd: effective[nd[0]])
        # Another drive may dominate after inhibition even if this one is "ready".
        if effective[winner_name] <= 0.0:
            return VolitionDecision(action=None, drive=None, activation=strongest,
                                    competitors=activations, reason="inhibited")
        winner.fire(now)
        return VolitionDecision(
            action=winner.action, drive=winner_name, activation=winner.activation,
            competitors=activations, reason="drive_won_competition",
        )

    # ── grounding: gather real signals best-effort ────────────────────────

    def gather_signals(self, base: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Assemble the signal vector from the substrate + nociception (best-effort)."""
        signals = dict(base or {})
        signals.setdefault("valence", 0.0)
        signals.setdefault("arousal", 0.0)
        signals.setdefault("dominance", 0.0)
        signals.setdefault("novelty", 0.0)
        try:
            from core.affect.nociception import get_nociception_engine
            signals["pain"] = _clamp(get_nociception_engine().nociceptive_pressure())
        except Exception as exc:  # noqa: BLE001 - grounding is best-effort
            record_degradation("drive_integration", exc, severity="debug")
            signals.setdefault("pain", 0.0)
        return signals

    def state(self) -> Dict[str, object]:
        return {"drives": {n: d.to_dict() for n, d in self._drives.items()}}


_instance: Optional[DriveIntegrationEngine] = None


def get_drive_integration_engine() -> DriveIntegrationEngine:
    global _instance
    if _instance is None:
        _instance = DriveIntegrationEngine()
    return _instance
