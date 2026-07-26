"""Group-relative policy optimization through Aura's recurrent live path.

This module is the missing objective bridge between verifier rewards and the
Recursive Latent Cortex. It does not generate or grade tasks. It consumes exact
completion tokens, the branch that produced each completion, frozen old-policy
log-probabilities captured at sampling time, and programmatic rewards. Current
policy probabilities are recomputed by ``live_path_forward`` so their gradient
passes through latent slots, recurrent window layers, branch exchange, and the
persisted answer path.

The adapter-disabled reference still executes the same RLC graph. This keeps KL
anchoring from comparing a recurrent policy to an architecturally different
standard decoder.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.brain.llm.latent_cortex.recurrence_adapter import (
    recurrence_adapter_disabled,
)
from core.learning.grpo import group_advantages
from core.learning.recurrence_native_objective_v2 import (
    LivePathForward,
    live_path_forward,
)

RECURRENT_GRPO_SCHEMA = "aura.recurrent_grpo.v1"
RECURRENT_SAMPLING_SCHEMA = "aura.recurrent_sampling_behavior.v3"


@dataclass(frozen=True, slots=True)
class RecurrentSamplingConfig:
    """Proof-grade sampling controls for the fixed recurrent policy graph."""

    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    # KV-cached and full teacher-forced quantized kernels are behaviorally
    # equivalent but not numerically identical, especially at the first token.
    # PPO's exact behavior ratio and clipped-token bound are the primary safety
    # contract; these wider secondary bounds catch wholesale graph divergence.
    max_abs_logprob_drift: float = 4.0
    max_mean_abs_logprob_drift: float = 0.5
    clip_epsilon: float = 0.2
    max_clipped_token_fraction: float = 0.25
    max_old_policy_approx_kl: float = 0.1

    def __post_init__(self) -> None:
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 8192:
            raise ValueError("max_tokens must be inside [1, 8192]")
        for name, value in (
            ("temperature", self.temperature),
            ("top_p", self.top_p),
            ("max_abs_logprob_drift", self.max_abs_logprob_drift),
            ("max_mean_abs_logprob_drift", self.max_mean_abs_logprob_drift),
            ("clip_epsilon", self.clip_epsilon),
            ("max_clipped_token_fraction", self.max_clipped_token_fraction),
            ("max_old_policy_approx_kl", self.max_old_policy_approx_kl),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite")
        # Temperature scaling, nucleus truncation, repetition penalties, and
        # minimum-token masks define a different behavior policy from the raw
        # recurrent model scored by the exact adjoint. Proof campaigns use the
        # unmodified categorical distribution; product decoding stays free to
        # use its richer controls outside this training surface.
        if float(self.temperature) != 1.0:
            raise ValueError("proof-grade recurrent sampling requires temperature=1")
        if float(self.top_p) != 1.0:
            raise ValueError("proof-grade recurrent sampling requires top_p=1")
        if not 0.0 <= float(self.max_abs_logprob_drift) <= 100.0:
            raise ValueError("max_abs_logprob_drift must be inside [0, 100]")
        if not 0.0 <= float(self.max_mean_abs_logprob_drift) <= 100.0:
            raise ValueError("max_mean_abs_logprob_drift must be inside [0, 100]")
        if not 0.0 < float(self.clip_epsilon) <= 1.0:
            raise ValueError("clip_epsilon must be inside (0, 1]")
        if not 0.0 <= float(self.max_clipped_token_fraction) <= 1.0:
            raise ValueError("max_clipped_token_fraction must be inside [0, 1]")
        if not 0.0 <= float(self.max_old_policy_approx_kl) <= 100.0:
            raise ValueError("max_old_policy_approx_kl must be inside [0, 100]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "max_abs_logprob_drift": float(self.max_abs_logprob_drift),
            "max_mean_abs_logprob_drift": float(
                self.max_mean_abs_logprob_drift
            ),
            "clip_epsilon": float(self.clip_epsilon),
            "max_clipped_token_fraction": float(
                self.max_clipped_token_fraction
            ),
            "max_old_policy_approx_kl": float(self.max_old_policy_approx_kl),
        }


@dataclass(frozen=True, slots=True)
class RecurrentPolicySample:
    tokens: tuple[int, ...]
    branch_index: int
    behavior_logprobs: tuple[float, ...]
    differentiable_logprobs: tuple[float, ...]
    max_abs_logprob_drift: float
    mean_abs_logprob_drift: float
    max_abs_logprob_drift_token_index: int
    clipped_token_fraction: float
    old_policy_approx_kl: float
    behavior_admitted: bool
    execution_spec_sha256: str
    policy_sha256: str
    prompt_tokens_sha256: str
    seed: int
    sampling_config: RecurrentSamplingConfig
    episode_receipt: dict[str, Any]

    def receipt(self) -> dict[str, Any]:
        encoded = json.dumps(
            list(self.tokens), separators=(",", ":"), allow_nan=False
        ).encode("ascii")
        behavior_encoded = json.dumps(
            list(self.behavior_logprobs),
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        differentiable_encoded = json.dumps(
            list(self.differentiable_logprobs),
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        return {
            "schema": RECURRENT_SAMPLING_SCHEMA,
            "execution_spec_sha256": self.execution_spec_sha256,
            "policy_sha256": self.policy_sha256,
            "prompt_tokens_sha256": self.prompt_tokens_sha256,
            "seed": self.seed,
            "sampling_config": self.sampling_config.to_dict(),
            "token_count": len(self.tokens),
            "tokens_sha256": hashlib.sha256(encoded).hexdigest(),
            "behavior_logprobs_sha256": hashlib.sha256(
                behavior_encoded
            ).hexdigest(),
            "differentiable_logprobs_sha256": hashlib.sha256(
                differentiable_encoded
            ).hexdigest(),
            "branch_index": self.branch_index,
            "max_abs_logprob_drift": self.max_abs_logprob_drift,
            "mean_abs_logprob_drift": self.mean_abs_logprob_drift,
            "max_abs_logprob_drift_token_index": (
                self.max_abs_logprob_drift_token_index
            ),
            "clipped_token_fraction": self.clipped_token_fraction,
            "old_policy_approx_kl": self.old_policy_approx_kl,
            "behavior_admitted": self.behavior_admitted,
            "cached_decode_termination": self.episode_receipt.get(
                "decode_termination", ""
            ),
            "cached_params_unchanged": self.episode_receipt.get(
                "params_unchanged"
            ),
            "cached_runtime_integrity": dict(
                self.episode_receipt.get("runtime_integrity", {})
            ),
            "cached_nonparametric_memory_status": str(
                self.episode_receipt.get("nonparametric_memory", {}).get(
                    "status",
                    "",
                )
            ),
            "cached_recurrence_adapter": dict(
                self.episode_receipt.get("recurrence_adapter", {})
            ),
        }


class RecurrentSamplingAdmissionError(RuntimeError):
    """Raised when cached behavior is outside the bounded PPO contract."""

    def __init__(self, sample: RecurrentPolicySample) -> None:
        self.sample = sample
        super().__init__(
            "cached recurrent behavior failed PPO admission: "
            f"max={sample.max_abs_logprob_drift:.6f} "
            f"mean={sample.mean_abs_logprob_drift:.6f} "
            f"clipped={sample.clipped_token_fraction:.6f}"
        )


# Compatibility for CP256 callers. New receipts and failures use the accurate
# behavior-policy name because the cached and adjoint kernels are not identical.
RecurrentSamplingParityError = RecurrentSamplingAdmissionError


class RecurrentGroupClipAdmissionError(RuntimeError):
    """Raised before adjoint replay when the cached policy is too far away."""

    def __init__(
        self,
        *,
        clip_fraction: float,
        max_clip_fraction: float,
        old_policy_approx_kl: float,
    ) -> None:
        self.clip_fraction = float(clip_fraction)
        self.max_clip_fraction = float(max_clip_fraction)
        self.old_policy_approx_kl = float(old_policy_approx_kl)
        super().__init__(
            "cached behavior policy exceeds recurrent PPO clip admission: "
            f"observed={self.clip_fraction:.6f} "
            f"limit={self.max_clip_fraction:.6f}"
        )


def _tokens_sha256(tokens: Sequence[int]) -> str:
    encoded = json.dumps(
        list(tokens), separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def recurrent_policy_sha256(model: Any, spec: RLCExecutionSpec) -> str:
    """Hash the exact trainable tensor tree plus its recurrent graph."""

    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten

    tensors = sorted(tree_flatten(model.trainable_parameters()))
    if not tensors:
        raise ValueError("recurrent policy has no trainable parameters")
    mx.eval([value for _name, value in tensors])
    digest = hashlib.sha256()
    digest.update(b"aura.recurrent_policy.v1\0")
    digest.update(spec.sha256.encode("ascii"))
    for name, value in tensors:
        name_bytes = name.encode("utf-8")
        dtype = str(value.dtype).encode("ascii")
        shape = json.dumps(list(value.shape), separators=(",", ":")).encode(
            "ascii"
        )
        try:
            array = np.asarray(value)
        except RuntimeError:
            # NumPy lacks a portable bfloat16 buffer code. Conversion to
            # float32 is exact for every bfloat16 value; the original dtype is
            # already bound separately above.
            array = np.asarray(value.astype(mx.float32))
        payload = array.tobytes(order="C")
        for part in (name_bytes, dtype, shape, payload):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def cortex_config_from_execution_spec(
    spec: RLCExecutionSpec,
    *,
    sampling: RecurrentSamplingConfig | None = None,
) -> Any:
    """Construct the fixed live-engine graph named by an execution spec."""

    from core.brain.llm.latent_cortex.types import (
        BranchConfig,
        CortexConfig,
        RecurrenceConfig,
        WorkspaceConfig,
    )

    problems = spec.validate()
    if problems:
        raise ValueError(f"invalid execution spec: {problems}")
    if spec.decode_bridge_policy != "none":
        raise ValueError(
            "recurrent sampling v1 supports decode_bridge_policy=none only"
        )
    resolved = sampling or RecurrentSamplingConfig()
    config = CortexConfig(
        workspace=WorkspaceConfig(
            n_slots=spec.n_slots,
            seed=spec.slot_seed,
            roles=spec.slot_roles,
            anchor_scale=spec.anchor_scale,
        ),
        recurrence=RecurrenceConfig(
            max_steps=spec.recurrent_steps,
            min_steps=spec.recurrent_steps,
            alpha=spec.alpha,
            alpha_schedule=spec.alpha_schedule,
            rms_clip_ratio=spec.rms_clip_ratio,
            convergence_eps=1e-9,
            divergence_ratio=1000.0,
            fixed_depth=True,
        ),
        branches=BranchConfig(
            n_branches=len(spec.branch_roles),
            exchange_interval=spec.exchange_interval,
            exchange_gamma=spec.exchange_gamma,
            comm_slot=spec.comm_slot,
            collapse_cos_threshold=spec.collapse_cos_threshold,
            jitter_scale=spec.jitter_scale,
            roles=spec.branch_roles,
        ),
        prelude_frac=spec.prelude_frac,
        coda_frac=spec.coda_frac,
        decode_max_tokens=resolved.max_tokens,
        decode_temperature=resolved.temperature,
        decode_top_p=resolved.top_p,
        decode_bridge_policy="none",
        decode_contract="none",
        decode_min_tokens=0,
        decode_repetition_penalty=1.0,
        allow_vanilla_fallback=False,
        escape={"enabled": False},
        telemetry_enabled=False,
        probe_cache_enabled=False,
    )
    config_problems = config.validate()
    if config_problems:
        raise ValueError(f"execution spec produced invalid CortexConfig: {config_problems}")
    return config


def sample_recurrent_completion(
    model: Any,
    prompt_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    seed: int,
    sampling: RecurrentSamplingConfig | None = None,
    tokenizer: Any | None = None,
    model_path: str | None = None,
    require_admission: bool = True,
) -> RecurrentPolicySample:
    """Sample through cached RLC, then admit it as a bounded behavior policy."""

    import mlx.core as mx

    from core.brain.llm.latent_cortex.engine import LatentCortexEngine

    resolved = sampling or RecurrentSamplingConfig()
    prompt = tuple(prompt_tokens)
    if not prompt or any(type(token) is not int or token < 0 for token in prompt):
        raise ValueError("prompt_tokens must contain non-negative integers")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    policy_sha256 = recurrent_policy_sha256(model, spec)
    mx.random.seed(seed)
    engine = LatentCortexEngine(
        model,
        tokenizer=tokenizer,
        config=cortex_config_from_execution_spec(spec, sampling=resolved),
        model_path=model_path,
        schedule_library=None,
    )
    result = engine.reason(
        token_ids=list(prompt),
        decode_max_tokens=resolved.max_tokens,
        capture_decode_logprobs=True,
        decode_sentence_grace_tokens=0,
        nonparametric_memory_enabled=False,
    )
    if not result.ok:
        raise RuntimeError(f"cached recurrent sampling failed: {result.reason}")
    if not result.tokens:
        raise RuntimeError("cached recurrent sampling produced no completion tokens")
    if len(result.tokens) != len(result.decode_token_logprobs):
        raise RuntimeError("cached recurrent sampling logprobs do not align with tokens")
    branch_index = int(result.receipt.selected_branch)
    differentiable = recurrent_completion_token_logprobs(
        model,
        prompt,
        result.tokens,
        spec=spec,
        branch_index=branch_index,
    )
    mx.eval(differentiable)
    if recurrent_policy_sha256(model, spec) != policy_sha256:
        raise RuntimeError("recurrent policy changed during cached sampling")
    target = tuple(float(value) for value in differentiable)
    behavior = tuple(float(value) for value in result.decode_token_logprobs)
    if any(not math.isfinite(value) for value in (*target, *behavior)):
        raise FloatingPointError("recurrent sampling produced non-finite logprobs")
    differences = tuple(
        abs(left - right)
        for left, right in zip(behavior, target, strict=True)
    )
    maximum = max(differences)
    maximum_index = differences.index(maximum)
    mean = sum(differences) / len(differences)
    ratios = tuple(
        math.exp(target_value - behavior_value)
        for target_value, behavior_value in zip(target, behavior, strict=True)
    )
    clipped_fraction = sum(
        abs(ratio - 1.0) > float(resolved.clip_epsilon) for ratio in ratios
    ) / len(ratios)
    old_policy_approx_kl = sum(
        (ratio - 1.0) - math.log(ratio) for ratio in ratios
    ) / len(ratios)
    from core.brain.llm.latent_cortex.runtime_integrity import (
        runtime_integrity_safe,
    )

    measured_runtime_safe = runtime_integrity_safe(
        result.receipt.runtime_integrity,
        require_worker=False,
        expected_episode_id=result.receipt.episode_id,
        expected_input_tokens_sha256=result.receipt.input_tokens_sha256,
        expected_fast_weights_applied=result.receipt.fast_weights_applied,
        expected_checkpoint_fingerprint=(
            result.receipt.checkpoint_fingerprint
        ),
        expected_checkpoint_method=(
            result.receipt.checkpoint_fingerprint_method
        ),
        expected_checkpoint_file_count=(
            result.receipt.checkpoint_file_count
        ),
    )
    admitted = (
        maximum <= float(resolved.max_abs_logprob_drift)
        and mean <= float(resolved.max_mean_abs_logprob_drift)
        and clipped_fraction <= float(resolved.max_clipped_token_fraction)
        and old_policy_approx_kl <= float(resolved.max_old_policy_approx_kl)
        and measured_runtime_safe
        and not any(
            flag.startswith("fallback_")
            for flag in result.receipt.honest_flags
        )
    )
    sample = RecurrentPolicySample(
        tokens=tuple(result.tokens),
        branch_index=branch_index,
        behavior_logprobs=behavior,
        differentiable_logprobs=target,
        max_abs_logprob_drift=maximum,
        mean_abs_logprob_drift=mean,
        max_abs_logprob_drift_token_index=maximum_index,
        clipped_token_fraction=clipped_fraction,
        old_policy_approx_kl=old_policy_approx_kl,
        behavior_admitted=admitted,
        execution_spec_sha256=spec.sha256,
        policy_sha256=policy_sha256,
        prompt_tokens_sha256=_tokens_sha256(prompt),
        seed=seed,
        sampling_config=resolved,
        episode_receipt=result.receipt.to_dict(),
    )
    if require_admission and not admitted:
        raise RecurrentSamplingAdmissionError(sample)
    return sample


@dataclass(frozen=True, slots=True)
class RecurrentGRPOConfig:
    clip_epsilon: float = 0.2
    kl_coefficient: float = 0.04
    advantage_clip: float = 4.0
    max_initial_clip_fraction: float = 0.25
    max_initial_old_policy_approx_kl: float = 0.1

    def __post_init__(self) -> None:
        for name, value in (
            ("clip_epsilon", self.clip_epsilon),
            ("kl_coefficient", self.kl_coefficient),
            ("advantage_clip", self.advantage_clip),
            ("max_initial_clip_fraction", self.max_initial_clip_fraction),
            (
                "max_initial_old_policy_approx_kl",
                self.max_initial_old_policy_approx_kl,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < float(self.clip_epsilon) <= 1.0:
            raise ValueError("clip_epsilon must be inside (0, 1]")
        if not 0.0 <= float(self.kl_coefficient) <= 100.0:
            raise ValueError("kl_coefficient must be inside [0, 100]")
        if not 0.0 < float(self.advantage_clip) <= 100.0:
            raise ValueError("advantage_clip must be inside (0, 100]")
        if not 0.0 <= float(self.max_initial_clip_fraction) <= 1.0:
            raise ValueError("max_initial_clip_fraction must be inside [0, 1]")
        if not 0.0 <= float(self.max_initial_old_policy_approx_kl) <= 100.0:
            raise ValueError(
                "max_initial_old_policy_approx_kl must be inside [0, 100]"
            )


@dataclass(frozen=True, slots=True)
class RecurrentGRPOObjective:
    loss: Any
    policy_loss: Any
    reference_kl: Any
    old_policy_approx_kl: Any
    clip_fraction: Any
    completion_count: int
    token_count: int

    def receipt(self) -> dict[str, Any]:
        import mlx.core as mx

        mx.eval(
            self.loss,
            self.policy_loss,
            self.reference_kl,
            self.old_policy_approx_kl,
            self.clip_fraction,
        )
        return {
            "schema": RECURRENT_GRPO_SCHEMA,
            "loss": float(self.loss),
            "policy_loss": float(self.policy_loss),
            "reference_kl": float(self.reference_kl),
            "old_policy_approx_kl": float(self.old_policy_approx_kl),
            "clip_fraction": float(self.clip_fraction),
            "completion_count": self.completion_count,
            "token_count": self.token_count,
        }


@dataclass(frozen=True, slots=True)
class ExactAdjointRecurrentGRPOResult:
    gradients: Any | None
    advantage_report: dict[str, Any]
    reference_kl: float
    old_policy_approx_kl: float
    clip_fraction: float
    policy_loss: float
    objective_at_sampling: float
    gradient_surrogate_value: float
    completion_count: int
    token_count: int
    branch_indices: tuple[int, ...]

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": RECURRENT_GRPO_SCHEMA,
            "mode": "exact_adjoint_single_update",
            "advantage_report": self.advantage_report,
            "reference_kl": self.reference_kl,
            "old_policy_approx_kl": self.old_policy_approx_kl,
            "clip_fraction": self.clip_fraction,
            "policy_loss": self.policy_loss,
            "objective_at_sampling": self.objective_at_sampling,
            "gradient_surrogate_value": self.gradient_surrogate_value,
            "completion_count": self.completion_count,
            "token_count": self.token_count,
            "branch_indices": list(self.branch_indices),
            "has_gradient": self.gradients is not None,
        }


def branch_token_logprobs(
    forward: LivePathForward,
    answer_tokens: Sequence[int],
    *,
    branch_index: int,
) -> Any:
    """Teacher-forced per-token log-probabilities for one producing branch."""
    import mlx.core as mx
    import mlx.nn as nn

    tokens = tuple(answer_tokens)
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("answer_tokens must contain non-negative integers")
    if type(branch_index) is not int or not 0 <= branch_index < len(
        forward.branch_logits
    ):
        raise ValueError("branch_index is outside the live-path branch set")
    logits = forward.branch_logits[branch_index]
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("branch logits must have shape [1, tokens, vocabulary]")
    if int(logits.shape[1]) != len(tokens):
        raise ValueError("branch logits and answer tokens do not align")
    targets = mx.array(tokens)[None, :]
    losses = nn.losses.cross_entropy(
        logits.astype(mx.float32), targets, reduction="none"
    )
    return -mx.squeeze(losses, axis=0)


def recurrent_completion_token_logprobs(
    model: Any,
    prompt_tokens: Sequence[int],
    completion_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    branch_index: int,
    bridge_tokens: Sequence[int] = (),
    adapters_on: bool = True,
) -> Any:
    """Score exact completion tokens through the true recurrent hidden path."""
    boundary = nullcontext() if adapters_on else recurrence_adapter_disabled()
    with boundary:
        forward = live_path_forward(
            model,
            prompt_tokens,
            completion_tokens,
            spec=spec,
            bridge_tokens=bridge_tokens,
        )
    return branch_token_logprobs(
        forward, completion_tokens, branch_index=branch_index
    )


def clipped_recurrent_grpo_objective(
    policy_logprobs: Sequence[Any],
    old_policy_logprobs: Sequence[Any],
    advantages: Sequence[float],
    *,
    reference_logprobs: Sequence[Any] | None = None,
    config: RecurrentGRPOConfig | None = None,
) -> RecurrentGRPOObjective:
    """Token-normalized clipped GRPO objective with same-RLC reference KL."""
    import mlx.core as mx

    resolved = config or RecurrentGRPOConfig()
    count = len(policy_logprobs)
    if count < 2:
        raise ValueError("recurrent GRPO needs at least two completions")
    if len(old_policy_logprobs) != count or len(advantages) != count:
        raise ValueError("policy, old policy, and advantages must align")
    if reference_logprobs is not None and len(reference_logprobs) != count:
        raise ValueError("reference policy must align with completions")
    normalized_advantages = [float(value) for value in advantages]
    if any(not math.isfinite(value) for value in normalized_advantages):
        raise ValueError("advantages must be finite")

    policy_terms: list[Any] = []
    old_kl_terms: list[Any] = []
    reference_kl_terms: list[Any] = []
    clipped_tokens: list[Any] = []
    token_count = 0
    references: Sequence[Any | None] = (
        reference_logprobs
        if reference_logprobs is not None
        else (None,) * count
    )
    for policy, old_policy, advantage, reference in zip(
        policy_logprobs,
        old_policy_logprobs,
        normalized_advantages,
        references,
        strict=True,
    ):
        if policy.ndim != 1 or old_policy.ndim != 1 or policy.shape != old_policy.shape:
            raise ValueError("current and old token logprobs must be aligned vectors")
        if int(policy.shape[0]) < 1:
            raise ValueError("completion token logprobs cannot be empty")
        if reference is not None and (
            reference.ndim != 1 or reference.shape != policy.shape
        ):
            raise ValueError("reference token logprobs must align with policy")
        token_count += int(policy.shape[0])
        old = mx.stop_gradient(old_policy)
        log_ratio = policy - old
        ratio = mx.exp(log_ratio)
        clipped_ratio = mx.clip(
            ratio,
            1.0 - float(resolved.clip_epsilon),
            1.0 + float(resolved.clip_epsilon),
        )
        surrogate = mx.minimum(ratio * advantage, clipped_ratio * advantage)
        policy_terms.append(-mx.mean(surrogate))
        old_kl_terms.append(mx.mean((ratio - 1.0) - log_ratio))
        clipped_tokens.append(
            mx.mean(
                (mx.abs(ratio - 1.0) > resolved.clip_epsilon).astype(mx.float32)
            )
        )
        if reference is not None:
            delta = mx.stop_gradient(reference) - policy
            reference_kl_terms.append(mx.mean(mx.exp(delta) - delta - 1.0))

    policy_loss = mx.mean(mx.stack(policy_terms))
    old_policy_approx_kl = mx.mean(mx.stack(old_kl_terms))
    clip_fraction = mx.mean(mx.stack(clipped_tokens))
    reference_kl = (
        mx.mean(mx.stack(reference_kl_terms))
        if reference_kl_terms
        else mx.zeros(())
    )
    loss = policy_loss + float(resolved.kl_coefficient) * reference_kl
    return RecurrentGRPOObjective(
        loss=loss,
        policy_loss=policy_loss,
        reference_kl=reference_kl,
        old_policy_approx_kl=old_policy_approx_kl,
        clip_fraction=clip_fraction,
        completion_count=count,
        token_count=token_count,
    )


def verifier_group_objective(
    policy_logprobs: Sequence[Any],
    old_policy_logprobs: Sequence[Any],
    rewards: Sequence[float],
    *,
    reference_logprobs: Sequence[Any] | None = None,
    config: RecurrentGRPOConfig | None = None,
) -> tuple[RecurrentGRPOObjective, dict[str, Any]]:
    """Derive group advantages only from verifier rewards, then optimize."""
    resolved = config or RecurrentGRPOConfig()
    advantage_report = group_advantages(
        rewards, clip=resolved.advantage_clip
    )
    objective = clipped_recurrent_grpo_objective(
        policy_logprobs,
        old_policy_logprobs,
        advantage_report["advantages"],
        reference_logprobs=reference_logprobs,
        config=resolved,
    )
    return objective, advantage_report


def exact_adjoint_verifier_group_value_and_grad(
    model: Any,
    prompt_tokens: Sequence[int],
    completion_tokens: Sequence[Sequence[int]],
    branch_indices: Sequence[int],
    rewards: Sequence[float],
    *,
    spec: RLCExecutionSpec,
    behavior_logprobs: Sequence[Sequence[float]] | None = None,
    bridge_tokens: Sequence[int] = (),
    config: RecurrentGRPOConfig | None = None,
) -> ExactAdjointRecurrentGRPOResult:
    """Exact first-update GRPO gradient with one recurrent graph resident.

    Completions must come from the bounded cached behavior policy and this
    result must feed exactly one optimizer update. The
    cached behavior probabilities are supplied, so the clipped ratio is
    measured rather than assumed across numerically different MLX kernels. A
    verifier advantage ``A`` weights answer CE by the exact unclipped derivative
    ``A * p/q``; clipped tokens receive zero policy-gradient coefficient. The
    same-RLC k3 KL derivative is represented exactly by the fixed
    token coefficient ``exp(log p_ref - log p_policy) - 1``. The existing
    discrete adjoint then backpropagates each weighted completion through the
    recurrent graph before its activations are released.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map

    from core.learning.recurrence_native_objective_v2 import (
        exact_adjoint_live_path_value_and_grad,
    )

    resolved = config or RecurrentGRPOConfig()
    count = len(completion_tokens)
    if count < 2:
        raise ValueError("recurrent GRPO needs at least two completions")
    if len(branch_indices) != count or len(rewards) != count:
        raise ValueError("completions, branches, and rewards must align")
    if behavior_logprobs is not None and len(behavior_logprobs) != count:
        raise ValueError("behavior logprobs must align with completions")
    normalized_branches = tuple(branch_indices)
    if any(type(index) is not int or index < 0 for index in normalized_branches):
        raise ValueError("branch indices must be non-negative integers")
    advantage_report = group_advantages(
        rewards, clip=resolved.advantage_clip
    )
    token_count = sum(len(tokens) for tokens in completion_tokens)
    if any(not tokens for tokens in completion_tokens):
        raise ValueError("completion token sequences cannot be empty")
    if advantage_report["degenerate"]:
        return ExactAdjointRecurrentGRPOResult(
            gradients=None,
            advantage_report=advantage_report,
            reference_kl=0.0,
            old_policy_approx_kl=0.0,
            clip_fraction=0.0,
            policy_loss=0.0,
            objective_at_sampling=0.0,
            gradient_surrogate_value=0.0,
            completion_count=count,
            token_count=token_count,
            branch_indices=normalized_branches,
        )

    accumulated: Any | None = None
    reference_kl = 0.0
    old_policy_approx_kl = 0.0
    clip_fraction = 0.0
    policy_loss = 0.0
    surrogate_value = 0.0
    group_scale = 1.0 / count
    behaviors: Sequence[Sequence[float] | None] = (
        behavior_logprobs
        if behavior_logprobs is not None
        else (None,) * count
    )
    for tokens, branch_index, advantage, behavior_values in zip(
        completion_tokens,
        normalized_branches,
        advantage_report["advantages"],
        behaviors,
        strict=True,
    ):
        policy = recurrent_completion_token_logprobs(
            model,
            prompt_tokens,
            tokens,
            spec=spec,
            branch_index=branch_index,
            bridge_tokens=bridge_tokens,
            adapters_on=True,
        )
        reference = recurrent_completion_token_logprobs(
            model,
            prompt_tokens,
            tokens,
            spec=spec,
            branch_index=branch_index,
            bridge_tokens=bridge_tokens,
            adapters_on=False,
        )
        mx.eval(policy, reference)
        if behavior_values is None:
            behavior = mx.stop_gradient(policy)
        else:
            normalized_behavior = tuple(float(value) for value in behavior_values)
            if len(normalized_behavior) != len(tokens) or any(
                not math.isfinite(value) for value in normalized_behavior
            ):
                raise ValueError("behavior logprobs must be finite and token-aligned")
            behavior = mx.array(normalized_behavior, dtype=mx.float32)
        log_ratio = policy - behavior
        ratio = mx.exp(log_ratio)
        clipped_ratio = mx.clip(
            ratio,
            1.0 - float(resolved.clip_epsilon),
            1.0 + float(resolved.clip_epsilon),
        )
        surrogate = mx.minimum(
            ratio * float(advantage),
            clipped_ratio * float(advantage),
        )
        active = (
            ratio <= 1.0 + float(resolved.clip_epsilon)
            if float(advantage) >= 0.0
            else ratio >= 1.0 - float(resolved.clip_epsilon)
        )
        policy_coefficients = (
            float(advantage) * ratio * active.astype(mx.float32)
        )
        delta = reference - policy
        k3_tokens = mx.exp(delta) - delta - 1.0
        mx.eval(k3_tokens)
        reference_kl += group_scale * float(mx.mean(k3_tokens))
        old_policy_approx_kl += group_scale * float(
            mx.mean((ratio - 1.0) - log_ratio)
        )
        clip_fraction += group_scale * float(
            mx.mean(
                (mx.abs(ratio - 1.0) > resolved.clip_epsilon).astype(
                    mx.float32
                )
            )
        )
        policy_loss += group_scale * float(-mx.mean(surrogate))
        coefficients = policy_coefficients + float(resolved.kl_coefficient) * (
            mx.exp(delta) - 1.0
        )
        mx.eval(coefficients)
        weights = [group_scale * float(value) for value in coefficients]
        value, gradients, _base_value, _cosines = (
            exact_adjoint_live_path_value_and_grad(
                model,
                prompt_tokens,
                tokens,
                spec=spec,
                bridge_tokens=bridge_tokens,
                token_loss_weights=weights,
                branch_index=branch_index,
            )
        )
        surrogate_value += float(value)
        accumulated = (
            gradients
            if accumulated is None
            else tree_map(
                lambda previous, current: previous + current,
                accumulated,
                gradients,
            )
        )
        mx.eval(accumulated)
        del (
            policy,
            reference,
            behavior,
            log_ratio,
            ratio,
            clipped_ratio,
            surrogate,
            active,
            policy_coefficients,
            delta,
            k3_tokens,
            coefficients,
            gradients,
        )
        mx.clear_cache()

    if accumulated is None:
        raise RuntimeError("recurrent GRPO exact-adjoint gradient is empty")
    finite = [
        mx.all(mx.isfinite(gradient))
        for _name, gradient in tree_flatten(accumulated)
    ]
    mx.eval(finite)
    if not finite or not all(bool(flag) for flag in finite):
        raise FloatingPointError("recurrent GRPO gradient is non-finite")
    if clip_fraction > float(resolved.max_initial_clip_fraction):
        raise RuntimeError(
            "cached behavior policy exceeds recurrent PPO clip admission: "
            f"observed={clip_fraction:.6f} "
            f"limit={resolved.max_initial_clip_fraction:.6f}"
        )
    if old_policy_approx_kl > float(resolved.max_initial_old_policy_approx_kl):
        raise RuntimeError(
            "cached behavior policy exceeds recurrent PPO KL admission: "
            f"observed={old_policy_approx_kl:.6f} "
            f"limit={resolved.max_initial_old_policy_approx_kl:.6f}"
        )
    objective_at_sampling = (
        policy_loss + float(resolved.kl_coefficient) * reference_kl
    )
    return ExactAdjointRecurrentGRPOResult(
        gradients=accumulated,
        advantage_report=advantage_report,
        reference_kl=reference_kl,
        old_policy_approx_kl=old_policy_approx_kl,
        clip_fraction=clip_fraction,
        policy_loss=policy_loss,
        objective_at_sampling=objective_at_sampling,
        gradient_surrogate_value=surrogate_value,
        completion_count=count,
        token_count=token_count,
        branch_indices=normalized_branches,
    )


def exact_adjoint_sampled_group_value_and_grad(
    model: Any,
    prompt_tokens: Sequence[int],
    samples: Sequence[RecurrentPolicySample],
    rewards: Sequence[float],
    *,
    spec: RLCExecutionSpec,
    bridge_tokens: Sequence[int] = (),
    config: RecurrentGRPOConfig | None = None,
) -> ExactAdjointRecurrentGRPOResult:
    """Validate immutable sample provenance before one recurrent update."""

    resolved = config or RecurrentGRPOConfig()
    if len(samples) < 2 or len(samples) != len(rewards):
        raise ValueError("samples and rewards must align with at least two entries")
    prompt_sha256 = _tokens_sha256(prompt_tokens)
    expected_policy = recurrent_policy_sha256(model, spec)
    old_policy_approx_kl = 0.0
    clip_fraction = 0.0
    group_scale = 1.0 / len(samples)
    for index, sample in enumerate(samples):
        if not isinstance(sample, RecurrentPolicySample):
            raise TypeError(f"sample {index} is not a RecurrentPolicySample")
        if not sample.behavior_admitted:
            raise RecurrentSamplingAdmissionError(sample)
        if sample.execution_spec_sha256 != spec.sha256:
            raise ValueError(f"sample {index} execution spec differs")
        if sample.prompt_tokens_sha256 != prompt_sha256:
            raise ValueError(f"sample {index} prompt differs")
        if sample.policy_sha256 != expected_policy:
            raise ValueError(f"sample {index} recurrent policy differs")
        if not math.isclose(
            float(sample.sampling_config.clip_epsilon),
            float(resolved.clip_epsilon),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"sample {index} PPO clip epsilon differs")
        if not (
            len(sample.tokens)
            == len(sample.behavior_logprobs)
            == len(sample.differentiable_logprobs)
        ):
            raise ValueError(f"sample {index} token probabilities differ")
        ratios = []
        differences = []
        for policy, behavior in zip(
            sample.differentiable_logprobs,
            sample.behavior_logprobs,
            strict=True,
        ):
            if not math.isfinite(policy) or not math.isfinite(behavior):
                raise ValueError(f"sample {index} token probabilities are non-finite")
            differences.append(abs(policy - behavior))
            ratios.append(math.exp(policy - behavior))
        maximum = max(differences)
        mean = sum(differences) / len(differences)
        maximum_index = differences.index(maximum)
        if not math.isclose(
            maximum,
            float(sample.max_abs_logprob_drift),
            rel_tol=0.0,
            abs_tol=1e-9,
        ) or not math.isclose(
            mean,
            float(sample.mean_abs_logprob_drift),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"sample {index} drift receipt differs")
        if maximum_index != sample.max_abs_logprob_drift_token_index:
            raise ValueError(f"sample {index} maximum-drift index differs")
        old_policy_approx_kl += group_scale * sum(
            (ratio - 1.0) - math.log(ratio) for ratio in ratios
        ) / len(ratios)
        sample_clip_fraction = sum(
            abs(ratio - 1.0) > float(resolved.clip_epsilon)
            for ratio in ratios
        ) / len(ratios)
        if not math.isclose(
            sample_clip_fraction,
            float(sample.clipped_token_fraction),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"sample {index} clipped-token receipt differs")
        if (
            maximum > float(sample.sampling_config.max_abs_logprob_drift)
            or mean > float(sample.sampling_config.max_mean_abs_logprob_drift)
            or sample_clip_fraction
            > float(sample.sampling_config.max_clipped_token_fraction)
            or float(sample.old_policy_approx_kl)
            > float(sample.sampling_config.max_old_policy_approx_kl)
        ):
            raise ValueError(f"sample {index} admission receipt is inconsistent")
        clip_fraction += group_scale * sample_clip_fraction
    if clip_fraction > float(resolved.max_initial_clip_fraction):
        raise RecurrentGroupClipAdmissionError(
            clip_fraction=clip_fraction,
            max_clip_fraction=float(resolved.max_initial_clip_fraction),
            old_policy_approx_kl=old_policy_approx_kl,
        )
    if old_policy_approx_kl > float(resolved.max_initial_old_policy_approx_kl):
        raise RuntimeError(
            "cached behavior policy exceeds recurrent PPO KL admission: "
            f"observed={old_policy_approx_kl:.6f} "
            f"limit={resolved.max_initial_old_policy_approx_kl:.6f}"
        )
    return exact_adjoint_verifier_group_value_and_grad(
        model,
        prompt_tokens,
        [sample.tokens for sample in samples],
        [sample.branch_index for sample in samples],
        rewards,
        spec=spec,
        behavior_logprobs=[sample.behavior_logprobs for sample in samples],
        bridge_tokens=bridge_tokens,
        config=resolved,
    )


__all__ = [
    "RECURRENT_GRPO_SCHEMA",
    "RECURRENT_SAMPLING_SCHEMA",
    "ExactAdjointRecurrentGRPOResult",
    "RecurrentGRPOConfig",
    "RecurrentGRPOObjective",
    "RecurrentGroupClipAdmissionError",
    "RecurrentPolicySample",
    "RecurrentSamplingAdmissionError",
    "RecurrentSamplingConfig",
    "RecurrentSamplingParityError",
    "branch_token_logprobs",
    "clipped_recurrent_grpo_objective",
    "cortex_config_from_execution_spec",
    "exact_adjoint_sampled_group_value_and_grad",
    "exact_adjoint_verifier_group_value_and_grad",
    "recurrent_policy_sha256",
    "recurrent_completion_token_logprobs",
    "sample_recurrent_completion",
    "verifier_group_objective",
]
