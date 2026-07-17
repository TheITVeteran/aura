"""Configuration, budget, and receipt types for the Recursive Latent Cortex.

Everything the engine does is parameterized here and everything it did is
reported here. Receipts are the honesty spine: an episode that diverged,
blew its budget, or fell back to the vanilla path says so in machine-readable
form — downstream consumers (health, ledgers, the experiment harness) never
have to infer what happened.
"""
from __future__ import annotations

import math
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
ABSOLUTE_MAX_WALL_CLOCK_S = 900.0

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

    def __post_init__(self) -> None:
        if isinstance(self.max_layer_apps, bool) or not isinstance(self.max_layer_apps, int):
            raise TypeError("max_layer_apps must be an integer")
        if self.max_layer_apps <= 0:
            raise ValueError("max_layer_apps must be positive")
        self.max_layer_apps = min(self.max_layer_apps, ABSOLUTE_MAX_LAYER_APPS)
        if isinstance(self.wall_clock_s, bool) or not isinstance(self.wall_clock_s, (int, float)):
            raise TypeError("wall_clock_s must be numeric")
        self.wall_clock_s = float(self.wall_clock_s)
        if not math.isfinite(self.wall_clock_s) or self.wall_clock_s <= 0.0:
            raise ValueError("wall_clock_s must be finite and positive")
        self.wall_clock_s = min(self.wall_clock_s, ABSOLUTE_MAX_WALL_CLOCK_S)
        if self.spent_layer_apps < 0:
            raise ValueError("spent_layer_apps cannot be negative")

    def charge(self, tokens: int, layers: int) -> None:
        if (
            isinstance(tokens, bool)
            or isinstance(layers, bool)
            or not isinstance(tokens, int)
            or not isinstance(layers, int)
            or tokens < 0
            or layers < 0
        ):
            raise ValueError("budget charges require non-negative integer tokens and layers")
        self.charge_layer_apps(tokens * layers)

    def charge_layer_apps(self, layer_apps: int) -> None:
        if isinstance(layer_apps, bool) or not isinstance(layer_apps, int) or layer_apps < 0:
            raise ValueError("layer-app charge must be a non-negative integer")
        if layer_apps > self.remaining_layer_apps:
            raise RuntimeError(
                f"compute budget exhausted: requested={layer_apps} "
                f"remaining={self.remaining_layer_apps}"
            )
        self.spent_layer_apps += layer_apps

    def can_afford(self, tokens: int, layers: int, *, reserve_layer_apps: int = 0) -> bool:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (tokens, layers, reserve_layer_apps)
        ):
            return False
        return (
            not self.exhausted
            and tokens * layers + reserve_layer_apps <= self.remaining_layer_apps
        )

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
            "exhausted": self.exhausted,
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
    decode_top_p: float = 1.0
    decode_bridge_policy: str = "none"
    input_context_max_chars: int = 0
    allow_vanilla_fallback: bool = True

    def validate(self) -> list[str]:
        """Return a list of human-readable violations (empty ⇒ valid)."""
        problems: list[str] = []

        def finite(value: Any) -> bool:
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            )

        def integer_in(value: Any, minimum: int, maximum: int) -> bool:
            return type(value) is int and minimum <= value <= maximum

        if not integer_in(self.workspace.n_slots, 1, ABSOLUTE_MAX_SLOTS):
            problems.append(f"n_slots {self.workspace.n_slots} outside [1, {ABSOLUTE_MAX_SLOTS}]")
        if not isinstance(self.workspace.roles, (list, tuple)) or not self.workspace.roles or any(
            not isinstance(role, str) or not role.strip() for role in self.workspace.roles
        ) or len(self.workspace.roles) > ABSOLUTE_MAX_SLOTS:
            problems.append("workspace roles must be non-empty strings")
        if not integer_in(self.workspace.seed, -(2**63), 2**63 - 1):
            problems.append("workspace seed must be a signed 64-bit integer")
        if not finite(self.workspace.anchor_scale) or not 0.0 <= self.workspace.anchor_scale <= 1.0:
            problems.append("anchor_scale must be finite and inside [0, 1]")
        if not integer_in(
            self.recurrence.max_steps, 1, ABSOLUTE_MAX_RECURRENT_STEPS
        ):
            problems.append(
                f"max_steps {self.recurrence.max_steps} outside [1, {ABSOLUTE_MAX_RECURRENT_STEPS}]"
            )
        if not (
            type(self.recurrence.min_steps) is int
            and type(self.recurrence.max_steps) is int
            and 1 <= self.recurrence.min_steps <= self.recurrence.max_steps
        ):
            problems.append("min_steps must be inside [1, max_steps]")
        if not isinstance(self.recurrence.alpha_schedule, str) or self.recurrence.alpha_schedule not in {
            "constant",
            "cosine",
        }:
            problems.append("alpha_schedule must be constant or cosine")
        if not finite(self.recurrence.alpha) or not 0.0 < self.recurrence.alpha <= 1.0:
            problems.append(f"alpha {self.recurrence.alpha} outside (0, 1]")
        if not finite(self.recurrence.rms_clip_ratio) or not 1.0 <= self.recurrence.rms_clip_ratio <= 100.0:
            problems.append("rms_clip_ratio must be finite and inside [1, 100]")
        if not finite(self.recurrence.convergence_eps) or not 0.0 < self.recurrence.convergence_eps <= 1.0:
            problems.append("convergence_eps must be finite and inside (0, 1]")
        if not finite(self.recurrence.divergence_ratio) or not 1.0 < self.recurrence.divergence_ratio <= 1000.0:
            problems.append("divergence_ratio must be finite and inside (1, 1000]")
        if not integer_in(self.branches.n_branches, 1, ABSOLUTE_MAX_BRANCHES):
            problems.append(
                f"n_branches {self.branches.n_branches} outside [1, {ABSOLUTE_MAX_BRANCHES}]"
            )
        if not integer_in(
            self.branches.exchange_interval, 1, ABSOLUTE_MAX_RECURRENT_STEPS
        ):
            problems.append("exchange_interval outside recurrent-step limits")
        if not finite(self.branches.exchange_gamma) or not 0.0 <= self.branches.exchange_gamma <= 1.0:
            problems.append("exchange_gamma must be finite and inside [0, 1]")
        if not (
            type(self.branches.comm_slot) is int
            and type(self.workspace.n_slots) is int
            and 0 <= self.branches.comm_slot < self.workspace.n_slots
        ):
            problems.append("comm_slot index outside workspace")
        if not finite(self.branches.collapse_cos_threshold) or not -1.0 <= self.branches.collapse_cos_threshold <= 1.0:
            problems.append("collapse_cos_threshold must be finite and inside [-1, 1]")
        if not finite(self.branches.jitter_scale) or not 0.0 <= self.branches.jitter_scale <= 1.0:
            problems.append("jitter_scale must be finite and inside [0, 1]")
        if type(self.latent_opt.enabled) is not bool:
            problems.append("latent_opt.enabled must be boolean")
        if type(self.latent_opt.control_mode) is not bool:
            problems.append("latent_opt.control_mode must be boolean")
        if not integer_in(
            self.latent_opt.steps, 1, ABSOLUTE_MAX_RECURRENT_STEPS
        ):
            problems.append("latent_opt.steps outside recurrent-step limits")
        if not finite(self.latent_opt.lr) or not 0.0 < self.latent_opt.lr <= 1.0:
            problems.append("latent_opt.lr must be finite and inside (0, 1]")
        if not finite(self.latent_opt.lambda_reconstruct) or self.latent_opt.lambda_reconstruct < 0.0:
            problems.append("latent_opt.lambda_reconstruct must be finite and non-negative")
        if not finite(self.latent_opt.lambda_manifold) or self.latent_opt.lambda_manifold < 0.0:
            problems.append("latent_opt.lambda_manifold must be finite and non-negative")
        if not finite(self.latent_opt.max_grad_norm) or not 0.0 < self.latent_opt.max_grad_norm <= 1000.0:
            problems.append("latent_opt.max_grad_norm must be finite and inside (0, 1000]")
        if not finite(self.prelude_frac) or not 0.0 < self.prelude_frac < 0.5:
            problems.append(f"prelude_frac {self.prelude_frac} outside (0, 0.5)")
        if not finite(self.coda_frac) or not 0.0 < self.coda_frac < 0.5:
            problems.append(f"coda_frac {self.coda_frac} outside (0, 0.5)")
        if finite(self.prelude_frac) and finite(self.coda_frac) and self.prelude_frac + self.coda_frac >= 1.0:
            problems.append("prelude_frac + coda_frac must be < 1")
        if self.schedule is not None and not isinstance(self.schedule, dict):
            problems.append("schedule must be a mapping or null")
        if not integer_in(self.decode_max_tokens, 1, 8192):
            problems.append("decode_max_tokens outside [1, 8192]")
        if not finite(self.decode_temperature) or not 0.0 <= self.decode_temperature <= 2.0:
            problems.append("decode_temperature must be finite and inside [0, 2]")
        if not finite(self.decode_top_p) or not 0.0 < self.decode_top_p <= 1.0:
            problems.append("decode_top_p must be finite and inside (0, 1]")
        if self.decode_bridge_policy not in {"none", "assistant_answer_v1"}:
            problems.append(
                "decode_bridge_policy must be none or assistant_answer_v1"
            )
        if not (
            type(self.input_context_max_chars) is int
            and (
                self.input_context_max_chars == 0
                or 2048 <= self.input_context_max_chars <= 65536
            )
        ):
            problems.append(
                "input_context_max_chars must be 0 or inside [2048, 65536]"
            )
        if type(self.allow_vanilla_fallback) is not bool:
            problems.append("allow_vanilla_fallback must be boolean")
        if type(self.fast_weights.enabled) is not bool:
            problems.append("fast_weights.enabled must be boolean")
        if not integer_in(self.fast_weights.rank, 1, 64):
            problems.append("fast_weights.rank outside [1, 64]")
        if not finite(self.fast_weights.scale) or not 0.0 < self.fast_weights.scale <= 16.0:
            problems.append("fast_weights.scale must be finite and inside (0, 16]")
        if not isinstance(self.fast_weights.target, str) or self.fast_weights.target not in {
            "o_proj",
            "down_proj",
        }:
            problems.append("fast_weights.target must be o_proj or down_proj")
        if not integer_in(
            self.fast_weights.opt_steps, 1, ABSOLUTE_MAX_RECURRENT_STEPS
        ):
            problems.append("fast_weights.opt_steps outside recurrent-step limits")
        if not finite(self.fast_weights.lr) or not 0.0 < self.fast_weights.lr <= 1.0:
            problems.append("fast_weights.lr must be finite and inside (0, 1]")
        if not integer_in(
            self.fast_weights.max_wrapped_layers,
            1,
            ABSOLUTE_MAX_RECURRENT_STEPS,
        ):
            problems.append("fast_weights.max_wrapped_layers outside [1, 64]")
        return problems


@dataclass
class EpisodeReceipt:
    """Everything one reasoning episode actually did — the honesty record."""

    episode_id: str = ""
    started_at: float = field(default_factory=time.time)
    # Invariant proofs (governance.CheckpointInvariant fills these).
    checkpoint_fingerprint: str = ""
    checkpoint_fingerprint_method: str = ""
    checkpoint_file_count: int = 0
    worker_boot_id: str = ""
    worker_pid: int = 0
    worker_model_path: str = ""
    worker_model_parameter_count: int = 0
    worker_model_stored_parameter_element_count: int = 0
    worker_model_parameter_count_basis: str = ""
    worker_source_sha256: str = ""
    worker_affective_steering_active: bool = False
    worker_affective_steering_alpha: float = 0.0
    episode_affective_steering_applied: bool = False
    episode_affective_steering_alpha: float = 0.0
    request_payload_sha256: str = ""
    input_tokens_sha256: str = ""
    input_token_count: int = 0
    input_context_compaction: dict[str, Any] = field(default_factory=dict)
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
    # Digest of the first-decode logits (next-token distribution conditioned
    # on [prompt; refined thoughts]) — the cheap causal audit surface: any
    # change to the latent computation shows up here even when greedy tokens
    # collapse into the same attractor.
    first_logits_digest: str = ""
    latent_opt_applied: bool = False
    latent_opt_mode: str = ""  # gradient | control | off
    latent_opt_loss_trail: list[float] = field(default_factory=list)
    latent_opt_attempts: int = 0
    latent_opt_steps: int = 0
    latent_opt_rejected: int = 0
    latent_opt_budget_exhausted: bool = False
    fast_weights_applied: bool = False
    fast_weights_layers: int = 0
    fast_weight_optimization_attempts: int = 0
    fast_weight_optimized_steps: int = 0
    fast_weight_rejected_steps: int = 0
    fast_weight_budget_exhausted: bool = False
    fast_weight_optimizer: str = ""
    fast_weight_loss_trail: list[float] = field(default_factory=list)
    fast_weight_gradient_norm_trail: list[float] = field(default_factory=list)
    fast_weight_accepted_step_sizes: list[float] = field(default_factory=list)
    fast_weight_line_search_backtracks: int = 0
    # Decode completeness. A token-limit or EOS stop is complete; a budget stop
    # is a truncated answer and cannot satisfy the production receipt contract.
    decode_requested_tokens: int = 0
    decode_generated_tokens: int = 0
    decode_termination: str = "not_started"
    decode_temperature: float = 0.0
    decode_top_p: float = 1.0
    decode_bridge_applied: bool = False
    decode_bridge_policy: str = "none"
    decode_bridge_token_count: int = 0
    decode_bridge_tokens_sha256: str = ""
    decode_bridge_logits_digest: str = ""
    output_quality: dict[str, Any] = field(default_factory=dict)
    # Runtime lifecycle evidence. Timings are stage-local wall-clock seconds;
    # progress messages use the same stage names so a parent can distinguish a
    # slow live episode from a wedged worker without peeking into model state.
    last_stage: str = "not_started"
    stage_timings_s: dict[str, float] = field(default_factory=dict)
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
            "checkpoint_fingerprint_method": self.checkpoint_fingerprint_method,
            "checkpoint_file_count": self.checkpoint_file_count,
            "worker_boot_id": self.worker_boot_id,
            "worker_pid": self.worker_pid,
            "worker_model_path": self.worker_model_path,
            "worker_model_parameter_count": self.worker_model_parameter_count,
            "worker_model_stored_parameter_element_count": (
                self.worker_model_stored_parameter_element_count
            ),
            "worker_model_parameter_count_basis": (
                self.worker_model_parameter_count_basis
            ),
            "worker_source_sha256": self.worker_source_sha256,
            "worker_affective_steering_active": self.worker_affective_steering_active,
            "worker_affective_steering_alpha": self.worker_affective_steering_alpha,
            "episode_affective_steering_applied": self.episode_affective_steering_applied,
            "episode_affective_steering_alpha": self.episode_affective_steering_alpha,
            "request_payload_sha256": self.request_payload_sha256,
            "input_tokens_sha256": self.input_tokens_sha256,
            "input_token_count": self.input_token_count,
            "input_context_compaction": dict(self.input_context_compaction),
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
            "first_logits_digest": self.first_logits_digest,
            "best_step": self.best_step,
            "reverted_to_best": self.reverted_to_best,
            "branch_scores": [round(s, 6) for s in self.branch_scores],
            "selected_branch": self.selected_branch,
            "exchanges": self.exchanges,
            "latent_opt_applied": self.latent_opt_applied,
            "latent_opt_mode": self.latent_opt_mode,
            "latent_opt_loss_trail": [round(v, 6) for v in self.latent_opt_loss_trail],
            "latent_opt_attempts": self.latent_opt_attempts,
            "latent_opt_steps": self.latent_opt_steps,
            "latent_opt_rejected": self.latent_opt_rejected,
            "latent_opt_budget_exhausted": self.latent_opt_budget_exhausted,
            "fast_weights_applied": self.fast_weights_applied,
            "fast_weights_layers": self.fast_weights_layers,
            "fast_weight_optimization_attempts": self.fast_weight_optimization_attempts,
            "fast_weight_optimized_steps": self.fast_weight_optimized_steps,
            "fast_weight_rejected_steps": self.fast_weight_rejected_steps,
            "fast_weight_budget_exhausted": self.fast_weight_budget_exhausted,
            "fast_weight_optimizer": self.fast_weight_optimizer,
            "fast_weight_loss_trail": [
                round(v, 6) for v in self.fast_weight_loss_trail
            ],
            "fast_weight_gradient_norm_trail": [
                round(v, 6) for v in self.fast_weight_gradient_norm_trail
            ],
            "fast_weight_accepted_step_sizes": [
                round(v, 12) for v in self.fast_weight_accepted_step_sizes
            ],
            "fast_weight_line_search_backtracks": (
                self.fast_weight_line_search_backtracks
            ),
            "decode_requested_tokens": self.decode_requested_tokens,
            "decode_generated_tokens": self.decode_generated_tokens,
            "decode_termination": self.decode_termination,
            "decode_temperature": self.decode_temperature,
            "decode_top_p": self.decode_top_p,
            "decode_bridge_applied": self.decode_bridge_applied,
            "decode_bridge_policy": self.decode_bridge_policy,
            "decode_bridge_token_count": self.decode_bridge_token_count,
            "decode_bridge_tokens_sha256": self.decode_bridge_tokens_sha256,
            "decode_bridge_logits_digest": self.decode_bridge_logits_digest,
            "output_quality": dict(self.output_quality),
            "last_stage": self.last_stage,
            "stage_timings_s": {
                str(name): round(float(seconds), 6)
                for name, seconds in self.stage_timings_s.items()
            },
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
    # Raw output token ids — substrate-level callers (the experiments harness
    # driving random-weight models with synthetic vocabularies) verify these.
    tokens: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "text": self.text,
            "reason": self.reason,
            "tokens": list(self.tokens),
            "receipt": self.receipt.to_dict(),
        }


__all__ = [
    "ABSOLUTE_MAX_BRANCHES",
    "ABSOLUTE_MAX_LAYER_APPS",
    "ABSOLUTE_MAX_RECURRENT_STEPS",
    "ABSOLUTE_MAX_SLOTS",
    "ABSOLUTE_MAX_WALL_CLOCK_S",
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
