"""Attach learned halting to the live engine (CP234).

The engine already halts, but on RESIDUAL CONVERGENCE: it watches
``branch.halting.residual_trail`` and stops when the state stops moving.
That is fixed-point detection, not thought allocation. It answers "has this
loop finished changing?" when Anima Rationis line 765 asks "how much
thought does THIS problem deserve?" -- expected difficulty, uncertainty,
stakes, expected value of more computation.

The two come apart in exactly the case that matters. CP226 measured a loop
that kept moving (relative deltas 0.55, 0.50, 0.32) while accuracy fell to
zero. Residual halting sees healthy motion and keeps going; that is the
overthinking failure the document names at line 425.

This bridge lets the engine consult a learned head WITHOUT replacing the
existing behaviour:

* Default is ``residual`` -- byte-for-byte the current engine. A live
  cortex is not the place to discover that a new halting policy is worse.
* ``learned`` consults the head, but the residual rule remains a floor: if
  the state has genuinely converged, more steps cannot help regardless of
  what the head wants.
* Every decision carries its reason, so a receipt says WHY a branch stopped
  rather than only that it did.

The head is zero-initialized (CP230), so even in ``learned`` mode an
untrained head reproduces the residual policy exactly. Allocation is a
capability the model earns; attaching the mechanism grants nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

HALTING_BRIDGE_SCHEMA = "aura.learned_halting_bridge.v1"

RESIDUAL = "residual"
LEARNED = "learned"
MODES = (RESIDUAL, LEARNED)


@dataclass(frozen=True)
class HaltingBridgeConfig:
    """How the engine should decide it has thought enough."""

    mode: str = RESIDUAL
    # Below this relative residual the state has stopped moving, so further
    # steps recompute a fixed point. Enforced in BOTH modes: it is a fact
    # about the dynamics, not a policy preference.
    convergence_residual: float = 0.01
    min_steps: int = 1
    max_steps: int = 8

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        for name in ("min_steps", "max_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_steps > self.max_steps:
            raise ValueError("min_steps cannot exceed max_steps")
        if (
            isinstance(self.convergence_residual, bool)
            or not isinstance(self.convergence_residual, (int, float))
            or not 0.0 <= float(self.convergence_residual) < 1.0
        ):
            raise ValueError("convergence_residual must be inside [0, 1)")


def should_halt(
    *,
    step: int,
    residual_trail: list[float],
    config: HaltingBridgeConfig,
    head: Any = None,
    state: Any = None,
) -> dict[str, Any]:
    """Decide whether this branch has thought enough, and say why.

    ``step`` is 1-based (steps completed). Returns the verdict plus the
    reason, because a halting decision nobody can attribute is
    indistinguishable from a constant -- and this codebase has repeatedly
    shipped mechanisms that were present without firing.
    """
    if type(step) is not int or step < 1:
        raise ValueError("step must be a positive integer (1-based)")
    if config.mode == LEARNED and head is None:
        raise ValueError(
            "learned mode requires a halting head; falling back silently "
            "would report learned allocation while running the residual rule"
        )

    if step < config.min_steps:
        return _verdict(False, "below_min_steps", step, None)
    if step >= config.max_steps:
        return _verdict(True, "max_steps_reached", step, None)

    converged = bool(
        residual_trail
        and float(residual_trail[-1]) < config.convergence_residual
    )
    if converged:
        # True in both modes: a loop at its fixed point has stopped
        # computing, whatever the compute budget or the head says.
        return _verdict(True, "converged", step, None)

    if config.mode == RESIDUAL:
        return _verdict(False, "still_moving", step, None)

    if state is None:
        raise ValueError("learned mode requires the current latent state")
    probability = float(head.halt_probability(state))
    if probability >= head.threshold:
        return _verdict(True, "head_satisfied", step, probability)
    return _verdict(False, "head_wants_more", step, probability)


def _verdict(
    halt: bool, reason: str, step: int, probability: float | None
) -> dict[str, Any]:
    return {
        "schema": HALTING_BRIDGE_SCHEMA,
        "halt": bool(halt),
        "reason": reason,
        "step": step,
        "halt_probability": (
            round(probability, 6) if probability is not None else None
        ),
    }


def bridge_receipt(
    verdicts: list[dict[str, Any]], config: HaltingBridgeConfig
) -> dict[str, Any]:
    """Summarize an episode's halting decisions for the receipt.

    Reports whether the LEARNED head actually determined the outcome. A
    run in learned mode whose every stop came from the residual floor or
    the step cap is running the old policy under a new name, and the
    receipt should say so rather than let the mode string imply otherwise.
    """
    if not verdicts:
        raise ValueError("no halting verdicts to summarize")
    stops = [v for v in verdicts if v["halt"]]
    reasons: dict[str, int] = {}
    for verdict in verdicts:
        reasons[verdict["reason"]] = reasons.get(verdict["reason"], 0) + 1
    head_decided = sum(1 for v in stops if v["reason"] == "head_satisfied")
    return {
        "schema": HALTING_BRIDGE_SCHEMA,
        "mode": config.mode,
        "decisions": len(verdicts),
        "steps_taken": max(v["step"] for v in verdicts),
        "reasons": reasons,
        "stopped_by_head": head_decided,
        # The honest question about a learned policy: did it do anything?
        "head_was_causal": bool(config.mode == LEARNED and head_decided > 0),
    }


__all__ = [
    "HALTING_BRIDGE_SCHEMA",
    "LEARNED",
    "MODES",
    "RESIDUAL",
    "HaltingBridgeConfig",
    "bridge_receipt",
    "should_halt",
]
