"""Virtual width: a tied-weight latent society.

K branches are K concurrent dynamical states of the SAME neural operator —
not K models. Each branch gets a distinct role (constructive solution,
counterexample search, constraint checking, …) purely through its workspace
seed basin; the weights are identical and the prompt KV is shared read-only.

Exchange: every E steps the branches communicate through a designated
communication slot — an agreement-weighted consensus of branch summaries is
blended into each branch's comm slot, so useful partial results propagate
without collapsing the ensemble.

Anti-collapse: if two branch summaries become near-parallel, deterministic
decorrelation jitter is injected into the later branch. Diversity is a
maintained invariant, not a hope.

Honest accounting: the ensemble reports total token-layer applications so
Experiment 4 can compare against equal-FLOP self-consistency sampling. If
branches don't beat sampling at equal compute, they are expensive theater —
the harness is allowed to say so.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from core.brain.llm.latent_cortex.recurrence import (
    HaltingController,
    WindowRunner,
    recurrence_step,
    relative_residual,
)
from core.brain.llm.latent_cortex.types import BranchConfig, ComputeBudget, RecurrenceConfig
from core.brain.llm.latent_cortex.workspace import LatentWorkspace, per_position_rms

logger = logging.getLogger("Aura.LatentCortex.Branches")

# Cognitive roles for branch seeding, in priority order (from the spec's
# "tied-weight latent society"). Branch k takes BRANCH_ROLES[k % len].
BRANCH_ROLES: tuple[str, ...] = (
    "constructive_solution",
    "counterexample_search",
    "constraint_checking",
    "causal_reconstruction",
    "analogy",
    "reverse_reasoning",
    "simplification",
    "adversarial_criticism",
)


@dataclass
class BranchState:
    index: int
    role: str
    workspace: LatentWorkspace
    halting: HaltingController
    z: Any = None
    anchor: Any = None
    halted: bool = False
    halt_reason: str = ""
    steps: int = 0
    score: float = 0.0

    def to_receipt(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "role": self.role,
            "steps": self.steps,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "score": round(float(self.score), 6),
            "residual_trail": [round(r, 5) for r in self.halting.residual_trail],
        }


class BranchEnsemble:
    """K latent branches stepping over shared frozen weights + shared prompt KV.

    The ensemble serializes branch window passes (memory-light: one branch's
    activations at a time) and rewinds slot KV after every pass, so branches
    never see each other's cache side effects. Only the winner's final state
    is persisted — by the engine, not here.
    """

    def __init__(
        self,
        branches: list[BranchState],
        config: BranchConfig,
        recurrence: RecurrenceConfig,
    ) -> None:
        self.branches = branches
        self.config = config
        self.recurrence = recurrence
        self.exchanges = 0

    # ── Construction ────────────────────────────────────────────────────
    @classmethod
    def seed(
        cls,
        prompt_embeddings,
        workspace_cfg,
        branch_cfg: BranchConfig,
        recurrence_cfg: RecurrenceConfig,
        runner: WindowRunner,
        cache,
        prelude_end: int,
    ) -> "BranchEnsemble":
        import mlx.core as mx

        branches: list[BranchState] = []
        for k in range(branch_cfg.n_branches):
            role = BRANCH_ROLES[k % len(BRANCH_ROLES)]
            ws = LatentWorkspace.from_prompt_embeddings(
                prompt_embeddings, workspace_cfg, branch_role=role
            )
            # Prelude pass: persist=False for every branch — the engine
            # persists the WINNER's prelude at selection time. Rationale: all
            # branches must see identical caches; only one set of slot KV may
            # survive into decode.
            z0 = runner.run(ws.z, cache, 0, prelude_end, persist=False)
            ws.update(z0)
            halting = HaltingController(
                config=recurrence_cfg,
                baseline_rms=float(mx.mean(per_position_rms(z0))),
            )
            branches.append(
                BranchState(
                    index=k, role=role, workspace=ws, halting=halting, z=z0, anchor=z0
                )
            )
        return cls(branches, branch_cfg, recurrence_cfg)

    # ── Stepping ────────────────────────────────────────────────────────
    def active(self) -> list[BranchState]:
        return [b for b in self.branches if not b.halted]

    def step_all(
        self,
        runner: WindowRunner,
        cache,
        start: int,
        end: int,
        *,
        budget: ComputeBudget,
        alpha_override: float | None = None,
        score_fn: Callable[[BranchState], float] | None = None,
    ) -> None:
        """Advance every unhalted branch one controlled recurrence step."""
        for branch in self.active():
            z_next = recurrence_step(
                branch.z,
                runner,
                cache,
                start,
                end,
                self.recurrence,
                branch.steps,
                anchor=branch.anchor,
                alpha_override=alpha_override,
            )
            residual = relative_residual(z_next, branch.z)
            score = score_fn(branch) if score_fn is not None else None
            decision = branch.halting.observe(
                branch.steps, z_next, residual, score=score, budget=budget
            )
            branch.z = z_next
            branch.workspace.update(z_next)
            branch.steps += 1
            if decision.should_halt:
                final, reverted = branch.halting.final_state(branch.z)
                branch.z = final
                branch.workspace.update(final)
                branch.halted = True
                branch.halt_reason = decision.reason + ("_reverted" if reverted else "")

        if (
            len(self.active()) > 1
            and self.exchanges * self.config.exchange_interval
            < max(b.steps for b in self.branches)
            and max(b.steps for b in self.branches) % self.config.exchange_interval == 0
        ):
            self.exchange()
            self.maintain_diversity()

    # ── Communication ───────────────────────────────────────────────────
    def exchange(self) -> None:
        """Blend the agreement-weighted consensus into each comm slot.

        Weights favor branches whose summaries agree with the ensemble mean —
        an outlier branch still RECEIVES the consensus but contributes little
        to it, which lets adversarial/counterexample roles stay adversarial
        without dragging the consensus around.
        """
        import mlx.core as mx

        live = self.active()
        if len(live) < 2:
            return
        summaries = [b.workspace.summary() for b in live]  # (1,1,D) each
        stack = mx.concatenate(summaries, axis=1)  # (1,K,D)
        mean = mx.mean(stack, axis=1, keepdims=True)  # (1,1,D)

        def _cos(a, b):
            num = mx.sum(a * b)
            den = mx.maximum(mx.linalg.norm(a) * mx.linalg.norm(b), 1e-6)
            return num / den

        agreements = mx.stack([_cos(s, mean) for s in summaries])  # (K,)
        weights = mx.softmax(agreements, axis=0)
        consensus = sum(w * s for w, s in zip(weights, summaries))

        gamma = float(self.config.exchange_gamma)
        slot = int(self.config.comm_slot)
        for branch in live:
            z = branch.z
            comm = (1.0 - gamma) * z[:, slot : slot + 1, :] + gamma * consensus
            branch.z = mx.concatenate([z[:, :slot, :], comm, z[:, slot + 1 :, :]], axis=1)
            branch.workspace.update(branch.z)
        mx.eval(*[b.z for b in live])
        self.exchanges += 1

    def maintain_diversity(self) -> None:
        """Decorrelate near-parallel branch pairs with deterministic jitter."""
        import mlx.core as mx

        live = self.active()
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                a, b = live[i], live[j]
                sa, sb = a.workspace.summary(), b.workspace.summary()
                cos = float(
                    mx.sum(sa * sb)
                    / mx.maximum(mx.linalg.norm(sa) * mx.linalg.norm(sb), 1e-6)
                )
                if cos <= self.config.collapse_cos_threshold:
                    continue
                key = mx.random.key(1000 + 31 * a.index + b.index + b.steps)
                jitter = mx.random.normal(b.z.shape, key=key)
                jitter = jitter * (
                    float(self.config.jitter_scale)
                    * per_position_rms(b.z)
                    / mx.maximum(per_position_rms(jitter), 1e-6)
                )
                b.z = b.z + jitter
                b.workspace.update(b.z)
                mx.eval(b.z)
                logger.debug(
                    "Branch diversity jitter: %s↔%s cos=%.4f", a.index, b.index, cos
                )

    # ── Selection ───────────────────────────────────────────────────────
    def all_halted(self) -> bool:
        return all(b.halted for b in self.branches)

    def select(self, score_fn: Callable[[BranchState], float] | None = None) -> BranchState:
        """Pick the winning branch: external score if given, else convergence.

        Convergence quality = negative last residual — a branch that settled
        into a fixed point beats one still wandering when no verifier exists.
        """
        for branch in self.branches:
            if score_fn is not None:
                branch.score = float(score_fn(branch))
            else:
                trail = branch.halting.residual_trail
                branch.score = -trail[-1] if trail else float("-inf")
        return max(self.branches, key=lambda b: b.score)

    def to_receipt(self) -> dict[str, Any]:
        return {
            "n_branches": len(self.branches),
            "exchanges": self.exchanges,
            "branches": [b.to_receipt() for b in self.branches],
        }


__all__ = ["BRANCH_ROLES", "BranchEnsemble", "BranchState"]
