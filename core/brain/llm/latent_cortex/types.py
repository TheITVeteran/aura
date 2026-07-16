"""Configuration, budget, and receipt types for the Recursive Latent Cortex.

Everything the engine does is parameterized here and everything it did is
reported here. Receipts are the honesty spine: an episode that diverged,
blew its budget, or fell back to the vanilla path says so in machine-readable
form — downstream consumers (health, ledgers, the experiment harness) never
have to infer what happened.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# Hard ceilings no configuration may exceed. These protect the live host:
# a runaway schedule on the resident 32B is a memory/latency incident, not
# an experiment. Operators may lower them via config, never raise them.
ABSOLUTE_MAX_RECURRENT_STEPS = 64
ABSOLUTE_MAX_SLOTS = 128
ABSOLUTE_MAX_BRANCHES = 8
ABSOLUTE_MAX_LAYER_APPS = 500_000_000  # token-layer applications per episode

# Default per-episode compute in token-layer applications. Sized so a 64-layer
# model with a 2k prompt (prefill 2048*64 ≈ 131k) plus 32 slots recurring over
# a 32-layer window for 16 steps (32*32*16 ≈ 16k) fits with wide margin.
DEFAULT_EPISODE_LAYER_APPS = 4_000_000


@dataclass
class WorkspaceConfig:
    """Writable latent workspace: M continuous thought slots."""

    n_slots: int = 16
    seed: int = 0
    # Role names seed deterministic anchor vectors; slot i takes roles[i % len].
    roles: tuple[str, ...] = (
        "objective",
        "constraints",
        "hypothesis",
        "counterexample",
        "world_state",
        "subgoal",
        "uncertainty",
        "self_monitor",
    )
    # Scale of the role-anchor perturbation applied on top of the pooled
    # prompt embedding (relative to embedding RMS).
    anchor_scale: float = 0.05


@dataclass
class RecurrenceConfig:
    """Controlled recurrence — the anti-naive-looping controls."""

    max_steps: int = 12
    min_steps: int = 2
    alpha: float = 0.5
    alpha_schedule: str = "constant"  # constant | cosine
    # RMSMatch ratio clamp: new-state per-position RMS may move at most this
    # factor from the previous state's RMS in a single step.
    rms_clip_ratio: float = 3.0
    # Fixed-point convergence: relative residual below eps ⇒ converged.
    convergence_eps: float = 0.02
    # Divergence guard: mean-RMS growth beyond this factor of the post-seed
    # state (or any non-finite value) ⇒ halt and revert to best state.
    divergence_ratio: float = 10.0


@dataclass
class BranchConfig:
    """Virtual width: K concurrent latent trajectories of the same weights."""

    n_branches: int = 1
    exchange_interval: int = 4
    # Blend factor when writing the cross-branch consensus into each branch's
    # communication slot.
    exchange_gamma: float = 0.35
    comm_slot: int = 0
    # Anti-collapse: if two branch summaries exceed this cosine similarity,
    # deterministic decorrelation jitter is applied to the later branch.
    collapse_cos_threshold: float = 0.98
    jitter_scale: float = 0.02


@dataclass
class LatentOptConfig:
    """Gradient descent over thoughts (frozen weights, Z is the variable)."""

    enabled: bool = False
    steps: int = 4
    lr: float = 0.05
    lambda_reconstruct: float = 1.0
    lambda_manifold: float = 0.5
    max_grad_norm: float = 1.0
    # When True the optimizer applies matched-magnitude RANDOM perturbations
    # instead of gradient steps — the Experiment-5 control arm.
    control_mode: bool = False


@dataclass
class FastWeightsConfig:
    """Episode-scoped low-rank ΔW = s·U Vᵀ on selected window-layer linears."""

    enabled: bool = False
    rank: int = 2
    scale: float = 1.0
    target: str = "o_proj"  # o_proj | down_proj
    opt_steps: int = 4
    lr: float = 0.01
    # Layers (within the recurrent window) that receive fast weights; None ⇒
    # every window layer. Keep small on big models.
    max_wrapped_layers: int = 8


@dataclass
class ComputeBudget:
    """Episode compute economy in token-layer applications + wall clock."""

    max_layer_apps: int = DEFAULT_EPISODE_LAYER_APPS
    wall_clock_s: float = 120.0
    started_monotonic: float = field(default_factory=time.monotonic)
    spent_layer_apps: int = 0

    def charge(self, tokens: int, layers: int) -> None:
        self.spent_layer_apps += int(tokens) * int(layers)

    @property
    def exhausted(self) -> bool:
        if self.spent_layer_apps >= min(self.max_layer_apps, ABSOLUTE_MAX_LAYER_APPS):
            return True
        return (time.monotonic() - self.started_monotonic) >= self.wall_clock_s

    @property
    def remaining_layer_apps(self) -> int:
        return max(0, min(self.max_layer_apps, ABSOLUTE_MAX_LAYER_APPS) - self.spent_layer_apps)

    def to_receipt(self) -> dict[str, Any]:
        return {
            "max_layer_apps": self.max_layer_apps,
            "spent_layer_apps": self.spent_layer_apps,
            "wall_clock_s": self.wall_clock_s,
            "elapsed_s": round(time.monotonic() - self.started_monotonic, 3),
        }


@dataclass
class CortexConfig:
    """The integrated machine's full configuration."""

    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    recurrence: RecurrenceConfig = field(default_factory=RecurrenceConfig)
    branches: BranchConfig = field(default_factory=BranchConfig)
    latent_opt: LatentOptConfig = field(default_factory=LatentOptConfig)
    fast_weights: FastWeightsConfig = field(default_factory=FastWeightsConfig)
    # Prelude/coda split as layer fractions (window = the middle region).
    prelude_frac: float = 0.25
    coda_frac: float = 0.25
    # Explicit schedule program (schedules.LayerSchedule.to_dict() form); None
    # ⇒ single-window default derived from recurrence.max_steps.
    schedule: dict[str, Any] | None = None
    # Decode settings for the answer produced after latent computation.
    decode_max_tokens: int = 512
    decode_temperature: float = 0.0

    def validate(self) -> list[str]:
        """Return a list of human-readable violations (empty ⇒ valid)."""
        problems: list[str] = []
        if not 1 <= self.workspace.n_slots <= ABSOLUTE_MAX_SLOTS:
            problems.append(f"n_slots {self.workspace.n_slots} outside [1, {ABSOLUTE_MAX_SLOTS}]")
        if not 1 <= self.recurrence.max_steps <= ABSOLUTE_MAX_RECURRENT_STEPS:
            problems.append(
                f"max_steps {self.recurrence.max_steps} outside [1, {ABSOLUTE_MAX_RECURRENT_STEPS}]"
            )
        if self.recurrence.min_steps > self.recurrence.max_steps:
            problems.append("min_steps exceeds max_steps")
        if not 0.0 < self.recurrence.alpha <= 1.0:
            problems.append(f"alpha {self.recurrence.alpha} outside (0, 1]")
        if self.recurrence.rms_clip_ratio < 1.0:
            problems.append("rms_clip_ratio must be >= 1.0")
        if not 1 <= self.branches.n_branches <= ABSOLUTE_MAX_BRANCHES:
            problems.append(
                f"n_branches {self.branches.n_branches} outside [1, {ABSOLUTE_MAX_BRANCHES}]"
            )
        if self.branches.comm_slot >= self.workspace.n_slots:
            problems.append("comm_slot index outside workspace")
        if not 0.0 < self.prelude_frac < 0.5:
            problems.append(f"prelude_frac {self.prelude_frac} outside (0, 0.5)")
        if not 0.0 < self.coda_frac < 0.5:
            problems.append(f"coda_frac {self.coda_frac} outside (0, 0.5)")
        if self.fast_weights.enabled and self.fast_weights.rank < 1:
            problems.append("fast_weights.rank must be >= 1")
        return problems


@dataclass
class EpisodeReceipt:
    """Everything one reasoning episode actually did — the honesty record."""

    episode_id: str = ""
    started_at: float = field(default_factory=time.time)
    # Invariant proofs (governance.CheckpointInvariant fills these).
    checkpoint_fingerprint: str = ""
    params_unchanged: bool | None = None
    fast_weights_erased: bool | None = None
    # Topology actually used.
    n_layers: int = 0
    prelude_end: int = 0
    coda_start: int = 0
    n_slots: int = 0
    n_branches: int = 0
    schedule_hash: str = ""
    # Trajectory evidence.
    steps_taken: int = 0
    residual_trail: list[float] = field(default_factory=list)
    halting_reason: str = ""
    best_step: int = -1
    reverted_to_best: bool = False
    branch_scores: list[float] = field(default_factory=list)
    selected_branch: int = 0
    exchanges: int = 0
    # Optimization evidence.
    latent_opt_applied: bool = False
    latent_opt_mode: str = ""  # gradient | control | off
    latent_opt_loss_trail: list[float] = field(default_factory=list)
    fast_weights_applied: bool = False
    fast_weights_layers: int = 0
    # Economy.
    budget: dict[str, Any] = field(default_factory=dict)
    # Honesty flags: anything a consumer must know before trusting the output
    # ("diverged_reverted", "budget_exhausted", "fallback_vanilla", ...).
    honest_flags: list[str] = field(default_factory=list)

    def flag(self, name: str) -> None:
        if name not in self.honest_flags:
            self.honest_flags.append(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "started_at": self.started_at,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "params_unchanged": self.params_unchanged,
            "fast_weights_erased": self.fast_weights_erased,
            "n_layers": self.n_layers,
            "prelude_end": self.prelude_end,
            "coda_start": self.coda_start,
            "n_slots": self.n_slots,
            "n_branches": self.n_branches,
            "schedule_hash": self.schedule_hash,
            "steps_taken": self.steps_taken,
            "residual_trail": [round(r, 6) for r in self.residual_trail],
            "halting_reason": self.halting_reason,
            "best_step": self.best_step,
            "reverted_to_best": self.reverted_to_best,
            "branch_scores": [round(s, 6) for s in self.branch_scores],
            "selected_branch": self.selected_branch,
            "exchanges": self.exchanges,
            "latent_opt_applied": self.latent_opt_applied,
            "latent_opt_mode": self.latent_opt_mode,
            "latent_opt_loss_trail": [round(v, 6) for v in self.latent_opt_loss_trail],
            "fast_weights_applied": self.fast_weights_applied,
            "fast_weights_layers": self.fast_weights_layers,
            "budget": dict(self.budget),
            "honest_flags": list(self.honest_flags),
        }


@dataclass
class LatentReasoningResult:
    """What the engine returns to the worker/caller."""

    ok: bool
    text: str
    receipt: EpisodeReceipt
    reason: str = ""  # populated when ok is False or a fallback occurred

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "text": self.text,
            "reason": self.reason,
            "receipt": self.receipt.to_dict(),
        }


__all__ = [
    "ABSOLUTE_MAX_BRANCHES",
    "ABSOLUTE_MAX_LAYER_APPS",
    "ABSOLUTE_MAX_RECURRENT_STEPS",
    "ABSOLUTE_MAX_SLOTS",
    "BranchConfig",
    "ComputeBudget",
    "CortexConfig",
    "DEFAULT_EPISODE_LAYER_APPS",
    "EpisodeReceipt",
    "FastWeightsConfig",
    "LatentOptConfig",
    "LatentReasoningResult",
    "RecurrenceConfig",
    "WorkspaceConfig",
]
