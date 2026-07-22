"""Controlled attractor escape: a structured ladder, never raw noise first.

The recurrence core already has divergence guards, best-state reversion, and
branch-decorrelation jitter. What they lack is a SECOND LIFE: a branch that
diverges or stalls inside an attractor simply halts. This module gives the
engine a governed escape ladder, ordered from least to most destructive:

    1. revert to the best verified state seen so far;
    2. reseed the branch around its role anchor (fresh basin, same prompt);
    3. shift the branch's cognitive role (different anchor direction);
    4. inject a very small matched-magnitude perturbation (last resort —
       raw activation noise can just as easily destroy a nearly correct
       computation, so it comes after everything principled).

"Suspend fast weights" — the spec's remaining rung — lives where fast
weights actually run: the capability-canary ladder erases a regressing ΔW
at the fast-weight phase (fast weights are not active during recurrence).

Every rung starts a PROBATION: the branch has a bounded number of further
steps to beat its pre-escape best score. Improvement retains the escape;
anything else reverts to the pre-escape best and halts honestly with an
``escape_failed_*`` reason. Attempts, rungs, triggers, and outcomes are all
receipted per branch.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.LatentCortex.Escape")

ESCAPE_RUNGS: tuple[str, ...] = (
    "revert_best",
    "reseed_anchor",
    "role_shift",
    "matched_perturbation",
)


@dataclass
class EscapeConfig:
    """Bounds for the per-branch escape ladder."""

    enabled: bool = True
    # Steps without a best-score improvement before a live branch counts as
    # stalled inside an attractor.
    stall_patience: int = 3
    # Total escape attempts per branch across all rungs.
    max_attempts: int = 3
    # Post-escape steps the branch gets to beat its pre-escape best.
    probation_steps: int = 2
    # Matched-magnitude scale for the last-resort perturbation rung.
    perturbation_scale: float = 0.02
    # Score improvement that counts as escaping the attractor.
    min_improvement: float = 1e-4


@dataclass
class EscapeAttempt:
    rung: str
    trigger: str
    at_step: int
    pre_best_score: float
    outcome: str = "probation"  # probation | retained | failed | unresolved

    def to_receipt(self) -> dict[str, Any]:
        return {
            "rung": self.rung,
            "trigger": self.trigger,
            "at_step": self.at_step,
            "pre_best_score": round(self.pre_best_score, 6),
            "outcome": self.outcome,
        }


class BranchEscapeLadder:
    """One branch's escape state machine (owned by BranchState)."""

    def __init__(self, config: EscapeConfig, branch_index: int) -> None:
        self.config = config
        self.branch_index = int(branch_index)
        self.attempts: list[EscapeAttempt] = []
        self._used_rungs: set[str] = set()
        self._probation: dict[str, Any] | None = None

    # ── Queries ─────────────────────────────────────────────────────────
    def can_attempt(self) -> bool:
        return (
            len(self.attempts) < max(0, int(self.config.max_attempts))
            and len(self._used_rungs) < len(ESCAPE_RUNGS)
        )

    def snapshot(self) -> dict[str, Any]:
        """Capture the complete mutable escape state for transactional rewind."""

        attempts = tuple(
            {
                "rung": attempt.rung,
                "trigger": attempt.trigger,
                "at_step": attempt.at_step,
                "pre_best_score": attempt.pre_best_score,
                "outcome": attempt.outcome,
            }
            for attempt in self.attempts
        )
        probation = None
        if self._probation is not None:
            attempt = self._probation["attempt"]
            try:
                attempt_index = self.attempts.index(attempt)
            except ValueError as exc:  # pragma: no cover - internal invariant
                raise RuntimeError("escape probation attempt is not registered") from exc
            probation = {
                "attempt_index": attempt_index,
                "pre_best_score": float(self._probation["pre_best_score"]),
                "pre_best_state": self._probation["pre_best_state"],
                "deadline_step": int(self._probation["deadline_step"]),
            }
        return {
            "attempts": attempts,
            "used_rungs": frozenset(self._used_rungs),
            "probation": probation,
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore a snapshot while retaining this branch's fixed configuration."""

        required = {"attempts", "used_rungs", "probation"}
        if not isinstance(snapshot, dict) or set(snapshot) != required:
            raise ValueError("invalid escape-ladder snapshot")
        attempts = [
            EscapeAttempt(
                rung=str(item["rung"]),
                trigger=str(item["trigger"]),
                at_step=int(item["at_step"]),
                pre_best_score=float(item["pre_best_score"]),
                outcome=str(item["outcome"]),
            )
            for item in snapshot["attempts"]
        ]
        used_rungs = {str(value) for value in snapshot["used_rungs"]}
        if any(rung not in ESCAPE_RUNGS for rung in used_rungs):
            raise ValueError("escape snapshot contains an unknown rung")
        probation_data = snapshot["probation"]
        probation = None
        if probation_data is not None:
            index = int(probation_data["attempt_index"])
            if index < 0 or index >= len(attempts):
                raise ValueError("escape snapshot probation index is invalid")
            probation = {
                "attempt": attempts[index],
                "pre_best_score": float(probation_data["pre_best_score"]),
                "pre_best_state": probation_data["pre_best_state"],
                "deadline_step": int(probation_data["deadline_step"]),
            }
        self.attempts = attempts
        self._used_rungs = used_rungs
        self._probation = probation

    def _stalled(self, branch) -> bool:
        patience = max(1, int(self.config.stall_patience))
        trail = branch.halting.residual_trail
        if not trail or trail[-1] < branch.halting.config.convergence_eps:
            return False
        return (branch.steps - 1) - branch.halting.best_step >= patience

    def _next_rung(self) -> str | None:
        for rung in ESCAPE_RUNGS:
            if rung not in self._used_rungs:
                return rung
        return None

    # ── Event handlers (called by BranchEnsemble.step_all) ─────────────
    def on_divergence(self, branch, reason: str) -> str:
        """A halting-guard divergence fired. Returns 'escaped' or 'halt:<reason>'."""
        if self._probation is not None:
            self._fail_probation(branch)
            return f"halt:escape_failed_{reason}"
        if not self.can_attempt():
            return f"halt:{reason}"
        return self._attempt(branch, reason)

    def on_step(self, branch) -> str:
        """A live step finished. Returns '', 'retained', 'escaped', or 'halt:...'."""
        if self._probation is not None:
            attempt: EscapeAttempt = self._probation["attempt"]
            improvement = branch.halting.best_score - self._probation["pre_best_score"]
            if improvement > float(self.config.min_improvement):
                attempt.outcome = "retained"
                self._probation = None
                return "retained"
            if branch.steps >= self._probation["deadline_step"]:
                rung = attempt.rung
                self._fail_probation(branch)
                return f"halt:escape_failed_{rung}"
            return ""
        if self._stalled(branch) and self.can_attempt():
            return self._attempt(branch, "stalled")
        return ""

    def finalize(self) -> None:
        """Episode over: an attempt still on probation is unresolved, not won."""
        if self._probation is not None:
            self._probation["attempt"].outcome = "unresolved"
            self._probation = None

    # ── Internals ───────────────────────────────────────────────────────
    def _attempt(self, branch, trigger: str) -> str:
        rung = self._next_rung()
        if rung is None:
            return f"halt:{trigger}"
        pre_best_score = float(branch.halting.best_score)
        pre_best_state = (
            branch.halting.best_state
            if branch.halting.best_state is not None
            else branch.z
        )
        self._apply_rung(branch, rung)
        attempt = EscapeAttempt(
            rung=rung,
            trigger=trigger,
            at_step=branch.steps,
            pre_best_score=(
                pre_best_score if math.isfinite(pre_best_score) else -1e9
            ),
        )
        self.attempts.append(attempt)
        self._used_rungs.add(rung)
        self._probation = {
            "attempt": attempt,
            "pre_best_score": attempt.pre_best_score,
            "pre_best_state": pre_best_state,
            "deadline_step": branch.steps + max(1, int(self.config.probation_steps)),
        }
        logger.debug(
            "Branch %d escape rung=%s trigger=%s at step %d",
            self.branch_index,
            rung,
            trigger,
            branch.steps,
        )
        return "escaped"

    def _fail_probation(self, branch) -> None:
        probation = self._probation
        self._probation = None
        if probation is None:
            return
        probation["attempt"].outcome = "failed"
        branch.z = probation["pre_best_state"]
        branch.workspace.update(branch.z)

    def _apply_rung(self, branch, rung: str) -> None:
        import mlx.core as mx

        from core.brain.llm.latent_cortex.workspace import (
            per_position_rms,
            role_anchor,
        )

        base = (
            branch.halting.best_state
            if branch.halting.best_state is not None
            else branch.z
        )
        attempt_index = len(self.attempts)
        if rung == "revert_best":
            z = base
        elif rung == "reseed_anchor":
            key = mx.random.key(7001 + 97 * self.branch_index + attempt_index)
            jitter = mx.random.normal(branch.anchor.shape, key=key)
            jitter = jitter * (
                0.05
                * per_position_rms(branch.anchor)
                / mx.maximum(per_position_rms(jitter), 1e-6)
            )
            z = 0.5 * base + 0.5 * branch.anchor + jitter
        elif rung == "role_shift":
            from core.brain.llm.latent_cortex.branches import BRANCH_ROLES
            from core.brain.llm.latent_cortex.cognitive_operators import (
                operator_for_role,
            )

            new_role = next(
                (role for role in BRANCH_ROLES if role != branch.role),
                branch.role,
            )
            dim = int(base.shape[-1])
            direction = role_anchor(
                f"escape:{new_role}", dim, base_seed=self.branch_index
            )
            scale = 0.1 * mx.mean(per_position_rms(base))
            z = base + scale * direction[None, None, :]
            branch.role = new_role
            branch.operator = operator_for_role(new_role)
        elif rung == "matched_perturbation":
            key = mx.random.key(9001 + 97 * self.branch_index + attempt_index)
            noise = mx.random.normal(base.shape, key=key)
            noise = noise * (
                float(self.config.perturbation_scale)
                * per_position_rms(base)
                / mx.maximum(per_position_rms(noise), 1e-6)
            )
            z = base + noise
        else:  # pragma: no cover - rung set is closed
            raise ValueError(f"unknown escape rung: {rung!r}")
        branch.z = z
        branch.workspace.update(z)
        mx.eval(branch.z)

    def to_receipt(self) -> dict[str, Any]:
        return {
            "attempts": [attempt.to_receipt() for attempt in self.attempts],
            "rungs_used": sorted(self._used_rungs),
            "on_probation": self._probation is not None,
        }


__all__ = ["ESCAPE_RUNGS", "BranchEscapeLadder", "EscapeAttempt", "EscapeConfig"]
