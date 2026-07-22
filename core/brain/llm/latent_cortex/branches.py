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

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.brain.llm.latent_cortex.escape import BranchEscapeLadder, EscapeConfig
from core.brain.llm.latent_cortex.recurrence import (
    HaltingController,
    WindowRunner,
    recurrence_step,
    relative_residual,
    rms_match,
)
from core.brain.llm.latent_cortex.types import BranchConfig, ComputeBudget, RecurrenceConfig
from core.brain.llm.latent_cortex.workspace import (
    LatentWorkspace,
    _role_seed,
    per_position_rms,
)

logger = logging.getLogger("Aura.LatentCortex.Branches")

BRANCH_ISOLATION_SCHEMA = "aura.rlc.branch_isolation.v1"

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


def _tensor_sha256(array: Any) -> str:
    import numpy as np

    data = np.asarray(array)
    hasher = hashlib.sha256()
    hasher.update(str(data.dtype).encode("ascii"))
    hasher.update(str(data.shape).encode("ascii"))
    hasher.update(data.tobytes())
    return hasher.hexdigest()


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
    escape: BranchEscapeLadder | None = None
    # Neural-bytecode savepoint: one snapshot slot per branch (later
    # savepoints overwrite). verify_probe(revert_on_drop) restores it.
    savepoint: Any = None
    savepoint_steps: int = 0
    seed_sha256: str = ""
    candidate_sha256: str = ""
    candidate_step: int = 0
    rng_stream_sha256: str = ""

    def to_receipt(self) -> dict[str, Any]:
        receipt = {
            "index": self.index,
            "role": self.role,
            "steps": self.steps,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "score": round(float(self.score), 6),
            "residual_trail": [round(r, 5) for r in self.halting.residual_trail],
        }
        if self.escape is not None and self.escape.attempts:
            receipt["escape"] = self.escape.to_receipt()
        return receipt


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
        # Optional per-episode observers, attached by the engine.
        self.telemetry: Any = None
        self._isolation_sealed = False
        self._isolation_failure = ""
        self._blocked_cross_exposures = 0
        self._cross_exposure_started = False
        self._first_exchange_step: int | None = None
        self._context_sha256 = ""
        self._configured_role_lesion = len({branch.role for branch in branches}) != len(
            branches
        )
        self._seed_alias_free = (
            len({id(branch.workspace) for branch in branches}) == len(branches)
            and len({id(branch.z) for branch in branches}) == len(branches)
        )
        self._seed_states_unique = len(
            {branch.seed_sha256 for branch in branches}
        ) == len(branches)
        self._rng_streams_unique = len(
            {branch.rng_stream_sha256 for branch in branches}
        ) == len(branches)

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
        *,
        context_seeds: list[tuple[str, Any]] | None = None,
        escape_cfg: EscapeConfig | None = None,
    ) -> BranchEnsemble:
        import mlx.core as mx

        branches: list[BranchState] = []
        context_sha256 = _tensor_sha256(prompt_embeddings)
        role_override = tuple(branch_cfg.roles or ())
        if role_override and len(role_override) != branch_cfg.n_branches:
            raise ValueError(
                "BranchConfig.roles must name exactly n_branches roles, got "
                f"{len(role_override)} for {branch_cfg.n_branches} branches"
            )
        for k in range(branch_cfg.n_branches):
            role = (
                role_override[k]
                if role_override
                else BRANCH_ROLES[k % len(BRANCH_ROLES)]
            )
            ws = LatentWorkspace.from_prompt_embeddings(
                prompt_embeddings,
                workspace_cfg,
                branch_role=role,
                context_seeds=context_seeds,
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
                best_state=z0,
            )
            branches.append(
                BranchState(
                    index=k,
                    role=role,
                    workspace=ws,
                    halting=halting,
                    z=z0,
                    anchor=z0,
                    escape=(
                        BranchEscapeLadder(escape_cfg, k)
                        if escape_cfg is not None and escape_cfg.enabled
                        else None
                    ),
                    seed_sha256=_tensor_sha256(z0),
                    rng_stream_sha256=hashlib.sha256(
                        f"{role}:{_role_seed(role, workspace_cfg.seed)}".encode()
                    ).hexdigest(),
                )
            )
        ensemble = cls(branches, branch_cfg, recurrence_cfg)
        ensemble._context_sha256 = context_sha256
        return ensemble

    def _seal_isolation_if_ready(self) -> None:
        if self._isolation_sealed or self._isolation_failure:
            return
        required = int(self.config.isolation_steps)
        short_halts = [
            branch.index
            for branch in self.branches
            if branch.halted and branch.steps < required
        ]
        if short_halts:
            self._isolation_failure = "branch_halted_before_candidate"
            return
        if any(branch.steps < required for branch in self.branches):
            return
        for branch in self.branches:
            branch.candidate_sha256 = _tensor_sha256(branch.z)
            branch.candidate_step = branch.steps
        if not self._seed_alias_free:
            self._isolation_failure = "branch_state_alias_detected"
            return
        if not self._configured_role_lesion and not self._seed_states_unique:
            self._isolation_failure = "seed_state_collision"
            return
        if not self._configured_role_lesion and not self._rng_streams_unique:
            self._isolation_failure = "rng_stream_collision"
            return
        if not self._configured_role_lesion and len(
            {branch.candidate_sha256 for branch in self.branches}
        ) != len(self.branches):
            self._isolation_failure = "candidate_state_collision"
            return
        self._isolation_sealed = True

    def isolation_receipt(self, cache_discipline: dict[str, Any]) -> dict[str, Any]:
        """Return the public proof that candidates preceded peer exposure."""

        self._seal_isolation_if_ready()
        cache_proven = (
            isinstance(cache_discipline, dict)
            and cache_discipline.get("all_restored") is True
            and cache_discipline.get("restore_failures") == 0
            and cache_discipline.get("restored_calls")
            == cache_discipline.get("nonpersistent_calls")
        )
        candidates = [
            {
                "index": branch.index,
                "role": branch.role,
                "context_sha256": self._context_sha256,
                "rng_stream_sha256": branch.rng_stream_sha256,
                "seed_sha256": branch.seed_sha256,
                "candidate_sha256": branch.candidate_sha256,
                "candidate_step": branch.candidate_step,
            }
            for branch in self.branches
        ]
        certified = (
            self._isolation_sealed
            and not self._isolation_failure
            and not self._configured_role_lesion
            and self._seed_alias_free
            and self._seed_states_unique
            and self._rng_streams_unique
            and cache_proven
            and all(branch.candidate_sha256 for branch in self.branches)
            and (
                self._first_exchange_step is None
                or self._first_exchange_step >= int(self.config.isolation_steps)
            )
        )
        if certified:
            reason = "certified"
        elif self._isolation_failure:
            reason = self._isolation_failure
        elif self._configured_role_lesion:
            reason = "configured_role_lesion"
        elif not cache_proven:
            reason = "cache_restoration_unproven"
        else:
            reason = "isolation_incomplete"
        return {
            "schema": BRANCH_ISOLATION_SCHEMA,
            "n_branches": len(self.branches),
            "required_steps": int(self.config.isolation_steps),
            "sealed": self._isolation_sealed,
            "certified": certified,
            "reason": reason,
            "configured_role_lesion": self._configured_role_lesion,
            "seed_alias_free": self._seed_alias_free,
            "seed_states_unique": self._seed_states_unique,
            "rng_streams_unique": self._rng_streams_unique,
            "cross_exposure_started": self._cross_exposure_started,
            "first_exchange_step": self._first_exchange_step,
            "blocked_cross_exposures": self._blocked_cross_exposures,
            "candidates": candidates,
            "cache_discipline": dict(cache_discipline),
        }

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
        reserve_layer_apps: int = 0,
    ) -> bool:
        """Advance every live branch, or none when the whole round cannot fit."""
        active = self.active()
        round_cost = sum(int(branch.z.shape[1]) * (end - start) for branch in active)
        if round_cost + reserve_layer_apps > budget.remaining_layer_apps:
            return False
        deferred_fixed_depth_halts: list[tuple[BranchState, str]] = []
        for branch in active:
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
            score = (
                self._score_candidate(branch, z_next, score_fn)
                if score_fn is not None
                else None
            )
            decision = branch.halting.observe(
                branch.steps, z_next, residual, score=score, budget=budget
            )
            branch.z = z_next
            branch.workspace.update(z_next)
            branch.steps += 1
            if self.telemetry is not None:
                self.telemetry.record_step(
                    branch.index, branch.z, branch.anchor, residual
                )
            if decision.should_halt:
                if self.recurrence.fixed_depth and decision.reason == "max_steps":
                    deferred_fixed_depth_halts.append((branch, decision.reason))
                    continue
                # Divergence gets a second life through the escape ladder;
                # legitimate halts (converged / max_steps / budget) do not.
                if (
                    branch.escape is not None
                    and decision.reason.startswith("diverged")
                ):
                    action = branch.escape.on_divergence(branch, decision.reason)
                    if action == "escaped":
                        continue
                    self._halt(branch, action.removeprefix("halt:"))
                    continue
                self._halt(branch, decision.reason)
                continue
            if branch.escape is not None:
                action = branch.escape.on_step(branch)
                if action.startswith("halt:"):
                    self._halt(branch, action.removeprefix("halt:"))

        self._seal_isolation_if_ready()

        if (
            len(self.active()) > 1
            and self.exchanges * self.config.exchange_interval
            < max(b.steps for b in self.branches)
            and max(b.steps for b in self.branches) % self.config.exchange_interval == 0
        ):
            if self.exchange():
                self.maintain_diversity()
        for branch, reason in deferred_fixed_depth_halts:
            self._halt(branch, reason)
        return True

    @staticmethod
    def _score_candidate(
        branch: BranchState,
        z_next: Any,
        score_fn: Callable[[BranchState], float],
    ) -> float:
        """Evaluate a candidate through the existing branch-scoring contract.

        The callback historically receives ``BranchState``. Project the
        candidate into that view only for the call, then restore the committed
        state even if the verifier raises. This prevents a score for ``z_t``
        from being attached to ``z_(t+1)`` without changing public callers.
        """

        prior_z = branch.z
        prior_steps = branch.steps
        branch.z = z_next
        branch.workspace.update(z_next)
        branch.steps = prior_steps + 1
        try:
            return float(score_fn(branch))
        finally:
            branch.z = prior_z
            branch.workspace.update(prior_z)
            branch.steps = prior_steps

    # ── Neural-bytecode instructions ────────────────────────────────────
    def exchange_now(self) -> bool:
        """Bytecode-forced exchange: communicate immediately when ≥2 live."""
        if len(self.active()) < 2:
            return False
        if not self.exchange():
            return False
        self.maintain_diversity()
        return True

    def savepoint_all(self) -> int:
        """Snapshot every live branch's complete mutable execution state."""
        saved = 0
        for branch in self.active():
            branch.savepoint = {
                "z": branch.z,
                "role": branch.role,
                "halted": branch.halted,
                "halt_reason": branch.halt_reason,
                "steps": branch.steps,
                "score": branch.score,
                "halting": branch.halting.snapshot(),
                "escape": branch.escape.snapshot() if branch.escape is not None else None,
            }
            branch.savepoint_steps = branch.steps
            saved += 1
        return saved

    def revert_branch_to_savepoint(self, branch: BranchState) -> bool:
        """Transactionally restore one branch to its most recent savepoint."""

        snapshot = branch.savepoint
        if not isinstance(snapshot, dict):
            return False
        required = {
            "z",
            "role",
            "halted",
            "halt_reason",
            "steps",
            "score",
            "halting",
            "escape",
        }
        if set(snapshot) != required:
            raise ValueError("invalid branch savepoint")
        if (snapshot["escape"] is None) != (branch.escape is None):
            raise ValueError("branch escape configuration changed after savepoint")
        branch.z = snapshot["z"]
        branch.workspace.update(branch.z)
        branch.role = str(snapshot["role"])
        branch.halted = bool(snapshot["halted"])
        branch.halt_reason = str(snapshot["halt_reason"])
        branch.steps = int(snapshot["steps"])
        branch.score = float(snapshot["score"])
        branch.halting.restore(snapshot["halting"])
        if branch.escape is not None:
            branch.escape.restore(snapshot["escape"])
        return True

    def revert_all_to_savepoint(self) -> int:
        """Backtrack every branch that holds a savepoint transactionally."""
        reverted = 0
        for branch in self.branches:
            if self.revert_branch_to_savepoint(branch):
                reverted += 1
        return reverted

    def inject_control(self, control, *, strength: float = 0.12) -> int:
        """Causally write one bounded operator vector into each live workspace."""

        import mlx.core as mx

        if (
            isinstance(strength, bool)
            or not isinstance(strength, (int, float))
            or not 0.0 < float(strength) <= 0.5
        ):
            raise ValueError("control strength must be inside (0, 0.5]")
        changed = 0
        for branch in self.active():
            z = branch.z
            vector = mx.reshape(control, (1, 1, int(z.shape[-1])))
            slot = min(int(self.config.comm_slot), int(z.shape[1]) - 1)
            prior = z[:, slot : slot + 1, :]
            blended = (1.0 - float(strength)) * prior + float(strength) * vector
            blended = rms_match(blended, prior, self.recurrence.rms_clip_ratio)
            branch.z = mx.concatenate(
                [z[:, :slot, :], blended, z[:, slot + 1 :, :]],
                axis=1,
            )
            branch.workspace.update(branch.z)
            changed += 1
        if changed:
            mx.eval(*[branch.z for branch in self.active()])
        return changed

    def compress_state(self, *, strength: float = 0.25) -> int:
        """Fold global branch summaries into comm slots without erasing detail."""

        import mlx.core as mx

        if (
            isinstance(strength, bool)
            or not isinstance(strength, (int, float))
            or not 0.0 < float(strength) <= 0.5
        ):
            raise ValueError("compression strength must be inside (0, 0.5]")
        live = self.active()
        if not live:
            return 0
        if len(live) > 1 and not self._isolation_sealed:
            self._blocked_cross_exposures += 1
            return 0
        summaries = [branch.workspace.summary() for branch in live]
        global_summary = sum(summaries) / len(summaries)
        for branch in live:
            z = branch.z
            slot = min(int(self.config.comm_slot), int(z.shape[1]) - 1)
            prior = z[:, slot : slot + 1, :]
            compressed = (
                (1.0 - float(strength)) * prior
                + float(strength) * global_summary
            )
            compressed = rms_match(
                compressed,
                prior,
                self.recurrence.rms_clip_ratio,
            )
            branch.z = mx.concatenate(
                [z[:, :slot, :], compressed, z[:, slot + 1 :, :]],
                axis=1,
            )
            branch.workspace.update(branch.z)
        mx.eval(*[branch.z for branch in live])
        return len(live)

    def disagreement(self) -> float:
        """Mean pairwise cosine distance between active branch summaries."""

        import mlx.core as mx

        live = self.active()
        if len(live) < 2:
            return 0.0
        distances: list[float] = []
        for left_index, left in enumerate(live):
            left_summary = left.workspace.summary()
            for right in live[left_index + 1 :]:
                right_summary = right.workspace.summary()
                cosine = float(
                    mx.sum(left_summary * right_summary)
                    / mx.maximum(
                        mx.linalg.norm(left_summary)
                        * mx.linalg.norm(right_summary),
                        1e-6,
                    )
                )
                distances.append(max(0.0, min(1.0, 0.5 * (1.0 - cosine))))
        return sum(distances) / max(1, len(distances))

    def halt_all(self, reason: str) -> int:
        """Stop every live branch through the same best-state finalizer."""

        live = list(self.active())
        for branch in live:
            self._halt(branch, reason)
        return len(live)

    def _halt(self, branch: BranchState, reason: str) -> None:
        """Halt one branch, shipping the best state when it beats the last."""
        final, reverted = branch.halting.final_state(branch.z)
        branch.z = final
        branch.workspace.update(final)
        branch.halted = True
        branch.halt_reason = reason + ("_reverted" if reverted else "")
        if branch.escape is not None:
            branch.escape.finalize()

    # ── Communication ───────────────────────────────────────────────────
    def exchange(self) -> bool:
        """Blend the agreement-weighted consensus into each comm slot.

        Weights favor branches whose summaries agree with the ensemble mean —
        an outlier branch still RECEIVES the consensus but contributes little
        to it, which lets adversarial/counterexample roles stay adversarial
        without dragging the consensus around.
        """
        import mlx.core as mx

        live = self.active()
        if len(live) < 2:
            return False
        if not self._isolation_sealed:
            self._blocked_cross_exposures += 1
            return False
        summaries = [b.workspace.summary() for b in live]  # (1,1,D) each
        if self.telemetry is not None:
            self.telemetry.record_exchange(summaries)
        stack = mx.concatenate(summaries, axis=1)  # (1,K,D)
        mean = mx.mean(stack, axis=1, keepdims=True)  # (1,1,D)

        def _cos(a, b):
            num = mx.sum(a * b)
            den = mx.maximum(mx.linalg.norm(a) * mx.linalg.norm(b), 1e-6)
            return num / den

        agreements = mx.stack([_cos(s, mean) for s in summaries])  # (K,)
        weights = mx.softmax(agreements, axis=0)
        consensus = sum(w * s for w, s in zip(weights, summaries, strict=True))

        gamma = float(self.config.exchange_gamma)
        slot = int(self.config.comm_slot)
        for branch in live:
            z = branch.z
            comm = (1.0 - gamma) * z[:, slot : slot + 1, :] + gamma * consensus
            branch.z = mx.concatenate([z[:, :slot, :], comm, z[:, slot + 1 :, :]], axis=1)
            branch.workspace.update(branch.z)
        mx.eval(*[b.z for b in live])
        self.exchanges += 1
        self._cross_exposure_started = True
        if self._first_exchange_step is None:
            self._first_exchange_step = min(branch.steps for branch in live)
        return True

    def maintain_diversity(self) -> bool:
        """Decorrelate near-parallel branch pairs with deterministic jitter."""
        import mlx.core as mx

        live = self.active()
        if len(live) > 1 and not self._isolation_sealed:
            self._blocked_cross_exposures += 1
            return False
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
        return True

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


__all__ = [
    "BRANCH_ISOLATION_SCHEMA",
    "BRANCH_ROLES",
    "BranchEnsemble",
    "BranchState",
]
