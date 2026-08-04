"""What happens when the ODE diverges — and what used to happen instead.

The substrate integrates ``dx/dt = -x + tanh(Wx + I) + noise`` forward forever.
Stiff weights, a bad injected stimulus, or an accumulated overflow can push a
step to ``NaN`` or ``±inf``, and the reviewer's note was exact:

    "The substrate has no fallback if step() produces NaN or infinite values.
     There is a substrate.valid() check, but no corrective mechanism except
     crashing and restarting from the last WAL checkpoint — which is not yet
     tested."

It is worse than no fallback. ``_commit_worker_state_transform`` calls
``np.nan_to_num(..., nan=0.0)`` on the way in, on the current state, and on the
velocity. A diverged state is therefore not detected and not recovered; it is
SILENTLY REPLACED BY ZEROS and the run continues. Zeros are not a safe default
here — the state vector carries valence, arousal, dominance, frustration,
curiosity, energy and focus, so "recover to zero" means every affective reading
resets to neutral mid-conversation and nothing anywhere says a thing happened.
An unrecorded divergence looks exactly like a calm mind.

This module is the corrective mechanism:

  * ``check_soundness`` decides whether a state is sound, and says WHY it is
    not — before anything is coerced;
  * ``DivergenceRecovery`` keeps a small ring of states that were verified
    sound, so recovery restores the mind's last real condition rather than a
    fabricated one;
  * repeated divergence escalates to a real control action — damping the
    integration — because restoring the same state into the same dynamics that
    just diverged only buys one more step;
  * every recovery is recorded as a degradation with the diagnostic attached.

``probe_divergence_recovery`` is the executable check behind the eval arena's
recovery case, negative control included.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.SubstrateRecovery")

#: Activations live in [-1, 1]. A value outside this by more than float slack
#: is a broken invariant, not a large number.
_STATE_BOUND = 1.0
_BOUND_SLACK = 1e-6

#: How many verified-sound states to keep. Recovery wants the most recent sound
#: state; the depth exists so that a divergence discovered several steps late
#: still has somewhere to land.
_CHECKPOINT_DEPTH = 8

#: Consecutive recoveries before the dynamics themselves are damped. Restoring
#: a sound state into unchanged dynamics that just diverged buys one step.
_ESCALATE_AFTER = 3

#: How hard to damp on escalation, and the floor it will not go below. Damping
#: to zero would stop the mind rather than stabilise it.
_DAMPING_FACTOR = 0.5
_MIN_DAMPING = 0.05


@dataclass(frozen=True)
class SoundnessReport:
    """Whether a state vector is usable, and what is wrong with it if not."""

    sound: bool
    reasons: tuple[str, ...] = ()
    non_finite_count: int = 0
    out_of_bounds_count: int = 0
    max_magnitude: float = 0.0

    def as_metrics(self) -> dict[str, Any]:
        return {
            "sound": self.sound,
            "reasons": list(self.reasons),
            "non_finite": self.non_finite_count,
            "out_of_bounds": self.out_of_bounds_count,
            "max_magnitude": round(float(self.max_magnitude), 6),
        }


def check_soundness(state: Any, *, bound: float = _STATE_BOUND) -> SoundnessReport:
    """Judge a state vector BEFORE anything coerces it.

    Deliberately separate from the coercion. ``np.nan_to_num`` answers "what
    number shall I use instead"; this answers "did the dynamics stay in the
    regime the model is defined on", and only the second one can be reported.
    """

    try:
        array = np.asarray(state, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        return SoundnessReport(False, (f"unreadable_state:{exc}",))

    if array.size == 0:
        return SoundnessReport(False, ("empty_state",))

    finite = np.isfinite(array)
    non_finite = int((~finite).sum())
    reasons: list[str] = []
    if non_finite:
        if int(np.isnan(array).sum()):
            reasons.append("nan")
        if int(np.isinf(array).sum()):
            reasons.append("inf")

    finite_values = array[finite]
    max_magnitude = float(np.max(np.abs(finite_values))) if finite_values.size else float("inf")
    out_of_bounds = (
        int((np.abs(finite_values) > bound + _BOUND_SLACK).sum()) if finite_values.size else 0
    )
    if out_of_bounds:
        reasons.append("out_of_bounds")

    return SoundnessReport(
        sound=not reasons,
        reasons=tuple(reasons),
        non_finite_count=non_finite,
        out_of_bounds_count=out_of_bounds,
        max_magnitude=max_magnitude,
    )


@dataclass
class RecoveryOutcome:
    """What the recovery did, for the caller and for the record."""

    recovered: bool
    state: np.ndarray | None
    report: SoundnessReport
    checkpoint_age_steps: int = 0
    damping_applied: float = 1.0
    reason: str = ""


@dataclass
class DivergenceRecovery:
    """A rolling record of sound states, and what to do when one is not.

    Owned by the substrate and consulted at its single commit point, so a
    divergence cannot enter state through some path that forgot to check.
    """

    depth: int = _CHECKPOINT_DEPTH
    escalate_after: int = _ESCALATE_AFTER
    #: Self-tests induce divergence on purpose. Their recoveries are evidence
    #: the mechanism works, not evidence the runtime is degrading, and raising
    #: a live incident for each one would train everyone to ignore the real
    #: ones. Only a deliberate probe may turn this off.
    record_degradations: bool = True
    _checkpoints: deque = field(default_factory=lambda: deque(maxlen=_CHECKPOINT_DEPTH))
    _steps: int = 0
    _consecutive: int = 0
    _damping: float = 1.0
    divergences: int = 0
    recoveries: int = 0
    escalations: int = 0
    last_divergence_at: float = 0.0
    last_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.depth != _CHECKPOINT_DEPTH:
            self._checkpoints = deque(self._checkpoints, maxlen=max(1, int(self.depth)))

    # ---- the normal path --------------------------------------------------

    def observe(self, state: Any) -> SoundnessReport:
        """Record a state if it is sound. Returns the soundness verdict."""
        report = check_soundness(state)
        self._steps += 1
        if report.sound:
            self._checkpoints.append((self._steps, np.array(state, dtype=np.float64, copy=True)))
            self._consecutive = 0
            # Sound dynamics earn their damping back, slowly: a single good
            # step is not evidence the instability is gone.
            if self._damping < 1.0:
                self._damping = min(1.0, self._damping * 1.1)
        return report

    # ---- the failure path -------------------------------------------------

    def recover(self, state: Any, *, subsystem: str = "liquid_substrate") -> RecoveryOutcome:
        """Return the last verified-sound state in place of a diverged one."""

        report = check_soundness(state)
        if report.sound:
            self.observe(state)
            return RecoveryOutcome(False, np.asarray(state, dtype=np.float64), report, reason="sound")

        self.divergences += 1
        self._consecutive += 1
        self.last_divergence_at = time.time()
        self.last_reasons = report.reasons

        if self._consecutive >= self.escalate_after:
            self.escalations += 1
            self._damping = max(_MIN_DAMPING, self._damping * _DAMPING_FACTOR)
            logger.warning(
                "Substrate diverged %d times in a row (%s); damping integration to %.3f",
                self._consecutive,
                ",".join(report.reasons),
                self._damping,
            )

        if not self._checkpoints:
            # Nothing sound was ever seen. Say that, rather than inventing a
            # state — a fabricated "recovery" is what this module exists to end.
            if self.record_degradations:
                record_degradation(
                    subsystem,
                    RuntimeError(
                        f"substrate diverged ({','.join(report.reasons)}) with no sound "
                        "checkpoint to restore"
                    ),
                    severity="critical",
                    action="no recovery possible; caller must reinitialise",
                    extra=report.as_metrics(),
                )
            return RecoveryOutcome(
                False, None, report, damping_applied=self._damping, reason="no_checkpoint"
            )

        step, checkpoint = self._checkpoints[-1]
        restored = np.array(checkpoint, dtype=np.float64, copy=True) * self._damping
        self.recoveries += 1
        age = max(0, self._steps - step)
        if self.record_degradations:
            record_degradation(
                subsystem,
                RuntimeError(f"substrate step diverged ({','.join(report.reasons)})"),
                severity="degraded",
                action=(
                    f"restored the last sound state from {age} step(s) back "
                    f"(damping={self._damping:.3f})"
                ),
                extra={**report.as_metrics(), "checkpoint_age_steps": age},
            )
        return RecoveryOutcome(
            True,
            restored,
            report,
            checkpoint_age_steps=age,
            damping_applied=self._damping,
            reason="restored_last_sound_state",
        )

    # ---- inspection -------------------------------------------------------

    @property
    def damping(self) -> float:
        return self._damping

    @property
    def checkpoint_count(self) -> int:
        return len(self._checkpoints)

    def as_metrics(self) -> dict[str, Any]:
        return {
            "divergences": self.divergences,
            "recoveries": self.recoveries,
            "escalations": self.escalations,
            "checkpoints_held": len(self._checkpoints),
            "damping": round(self._damping, 4),
            "consecutive_divergences": self._consecutive,
            "last_reasons": list(self.last_reasons),
            "last_divergence_at": self.last_divergence_at,
        }


def probe_divergence_recovery() -> Any:
    """Executable check for the eval arena's ``recovery`` case.

    Drives a real divergence through a real ``DivergenceRecovery`` and asserts
    both directions: a NaN state must be replaced by the last sound one, and a
    sound state must be left exactly alone. A recovery layer that rewrites
    every state passes the first and fails the second.
    """

    from core.evals.eval_arena import ProbeOutcome

    recovery = DivergenceRecovery(record_degradations=False)
    sound = np.array([0.4, -0.2, 0.1, 0.0], dtype=np.float64)
    observed = recovery.observe(sound)

    diverged = np.array([np.nan, np.inf, 0.1, 0.0], dtype=np.float64)
    outcome = recovery.recover(diverged, subsystem="eval_arena.recovery_probe")

    # NEGATIVE CONTROL: a sound state must pass through untouched.
    untouched = recovery.recover(sound, subsystem="eval_arena.recovery_probe")

    restored_ok = (
        outcome.recovered
        and outcome.state is not None
        and bool(np.all(np.isfinite(outcome.state)))
        and float(np.max(np.abs(outcome.state))) <= _STATE_BOUND + _BOUND_SLACK
    )
    passthrough_ok = (
        not untouched.recovered
        and untouched.state is not None
        and bool(np.allclose(untouched.state, sound))
    )
    passed = bool(observed.sound and restored_ok and passthrough_ok)
    return ProbeOutcome(
        measured=True,
        passed=passed,
        detail=(
            "restored the last sound state after divergence and left a sound state alone"
            if passed
            else "did not both recover from divergence and pass sound state through"
        ),
        evidence={
            "diverged_reasons": list(outcome.report.reasons),
            "restored": restored_ok,
            "sound_state_untouched": passthrough_ok,
            "checkpoint_age_steps": outcome.checkpoint_age_steps,
        },
    )


__all__ = [
    "DivergenceRecovery",
    "RecoveryOutcome",
    "SoundnessReport",
    "check_soundness",
    "probe_divergence_recovery",
]
