"""Causal, episode-local virtual compute quanta for recurrent reasoning.

A quantum is a short-lived latent intervention derived from the episode's
already-admitted prompt/context state. It is not text, a caller-provided
vector, a belief, or durable parameter state. The guided intervention earns
one use only when repeated fixed-compute probes beat both no-intervention and
norm-matched orthogonal controls under an independently admitted verifier.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.brain.llm.latent_cortex.counterfactual_probe import (
    CounterfactualProbeResult,
    is_sha256,
)
from core.brain.llm.latent_cortex.loop_core import canonical_sha256
from core.brain.llm.latent_cortex.resource_accounting import RESOURCE_COUNTERS
from core.brain.llm.latent_cortex.verified_best import (
    tensor_sha256,
    validate_observation,
)

VIRTUAL_QUANTA_SCHEMA = "aura.latent_cortex.virtual_quanta.v2"
VIRTUAL_QUANTA_RECEIPT_SCHEMA = "aura.latent_cortex.virtual_quanta.receipt.v2"

DISABLED = "disabled"
COUNTERFACTUAL = "counterfactual"
ARM_NAMES = ("no_quantum", "matched_random", "guided_quantum")
MAX_REPLICATES = 4
MAX_TTL_STEPS = 4
MAX_ACTION_STEP = 1_000_000
MAX_STATE_ELEMENTS = 100_000_000


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _bounded_step(value: Any, *, name: str) -> int:
    if type(value) is not int or not 0 <= value <= MAX_ACTION_STEP:
        raise ValueError(f"{name} is invalid")
    return value


def _positions(
    value: Sequence[int],
    *,
    n_slots: int,
    name: str,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} positions are invalid")
    normalized = tuple(value)
    if any(
        type(position) is not int or not 0 <= position < n_slots for position in normalized
    ) or normalized != tuple(sorted(set(normalized))):
        raise ValueError(f"{name} positions are invalid")
    return normalized


def _receipt_positions(
    protected_value: Any,
    source_value: Any,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if (
        isinstance(protected_value, (str, bytes))
        or not isinstance(protected_value, Sequence)
        or isinstance(source_value, (str, bytes))
        or not isinstance(source_value, Sequence)
    ):
        raise ValueError("virtual quanta receipt positions are invalid")
    protected_raw = tuple(protected_value)
    source_raw = tuple(source_value)
    combined = protected_raw + source_raw
    if any(
        type(position) is not int or not 0 <= position <= MAX_ACTION_STEP for position in combined
    ):
        raise ValueError("virtual quanta receipt positions are invalid")
    n_slots = max(combined, default=0) + 1
    return (
        _positions(protected_raw, n_slots=n_slots, name="protected"),
        _positions(source_raw, n_slots=n_slots, name="source"),
    )


def _state(value: Any, *, name: str) -> np.ndarray:
    state = np.asarray(value)
    if (
        state.ndim != 3
        or state.shape[0] != 1
        or state.shape[1] < 1
        or state.shape[2] < 1
        or state.size > MAX_STATE_ELEMENTS
        or not np.issubdtype(state.dtype, np.floating)
        or not np.all(np.isfinite(state))
    ):
        raise ValueError(f"{name} latent state is invalid")
    return np.array(state, copy=True)


def _rms(value: np.ndarray) -> float:
    if value.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))


def _resource_snapshot(value: Any) -> dict[str, int]:
    ledger = getattr(value, "resource_ledger", None)
    totals = ledger.totals() if ledger is not None else None
    if not isinstance(totals, Mapping) or set(totals) != set(RESOURCE_COUNTERS):
        raise ValueError("virtual quantum resource ledger is unavailable")
    normalized: dict[str, int] = {}
    for name in RESOURCE_COUNTERS:
        amount = totals[name]
        if type(amount) is not int or amount < 0:
            raise ValueError("virtual quantum resource counter is invalid")
        normalized[name] = amount
    return normalized


def _resource_delta(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, int]:
    if set(before) != set(RESOURCE_COUNTERS) or set(after) != set(RESOURCE_COUNTERS):
        raise ValueError("virtual quantum resource windows differ")
    delta = {name: int(after[name]) - int(before[name]) for name in RESOURCE_COUNTERS}
    if any(amount < 0 for amount in delta.values()):
        raise ValueError("virtual quantum resource counters regressed")
    return delta


def _fully_metered(
    delta: Mapping[str, int],
    *,
    probe: Mapping[str, Any],
) -> bool:
    return bool(
        set(delta) == set(RESOURCE_COUNTERS)
        and delta["transformer_layer_apps"] == probe["layer_apps"]
        and delta["output_head_tokens"] == probe["probe_token_count"]
        and delta["attention_query_key_pairs"] > 0
        and delta["verifier_calls"] >= 1
        and delta["verifier_input_bytes"] > 0
        and delta["verifier_output_bytes"] > 0
    )


@dataclass(frozen=True, slots=True)
class VirtualQuantaConfig:
    """Trust boundary for one bounded, one-use latent compute quantum."""

    mode: str = COUNTERFACTUAL
    max_relative_delta_rms: float = 0.05
    min_verifier_margin: float = 0.01
    replicates: int = 2
    ttl_steps: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {DISABLED, COUNTERFACTUAL}:
            raise ValueError("virtual quanta mode is invalid")
        if (
            not _finite(self.max_relative_delta_rms)
            or not 0.0 < float(self.max_relative_delta_rms) <= 0.25
        ):
            raise ValueError("virtual quanta delta bound must be inside (0, 0.25]")
        if (
            not _finite(self.min_verifier_margin)
            or not 0.0 <= float(self.min_verifier_margin) <= 0.25
        ):
            raise ValueError("virtual quanta verifier margin must be inside [0, 0.25]")
        if type(self.replicates) is not int or not 2 <= self.replicates <= MAX_REPLICATES:
            raise ValueError(f"virtual quanta replicates must be inside [2, {MAX_REPLICATES}]")
        if type(self.ttl_steps) is not int or not 1 <= self.ttl_steps <= MAX_TTL_STEPS:
            raise ValueError(f"virtual quanta TTL must be inside [1, {MAX_TTL_STEPS}]")
        if type(self.seed) is not int or not -(2**63) <= self.seed <= 2**63 - 1:
            raise ValueError("virtual quanta seed must be signed 64-bit")

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | None) -> VirtualQuantaConfig:
        raw = dict(value or {})
        unknown = set(raw) - {
            "mode",
            "max_relative_delta_rms",
            "min_verifier_margin",
            "replicates",
            "ttl_steps",
            "seed",
        }
        if unknown:
            raise ValueError(f"virtual quanta has unknown keys: {sorted(unknown)}")
        return cls(
            mode=raw.get("mode", COUNTERFACTUAL),
            max_relative_delta_rms=raw.get("max_relative_delta_rms", 0.05),
            min_verifier_margin=raw.get("min_verifier_margin", 0.01),
            replicates=raw.get("replicates", 2),
            ttl_steps=raw.get("ttl_steps", 1),
            seed=raw.get("seed", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "max_relative_delta_rms": round(float(self.max_relative_delta_rms), 10),
            "min_verifier_margin": round(float(self.min_verifier_margin), 10),
            "replicates": self.replicates,
            "ttl_steps": self.ttl_steps,
            "seed": self.seed,
        }


@dataclass(slots=True)
class _PrivateQuantum:
    quantum_id: str
    direction: np.ndarray | None
    created_step: int
    expires_step: int
    direction_sha256: str
    direction_shape: list[int]
    direction_dtype: str
    uses: int = 0

    def consume(self, *, step: int) -> None:
        if self.direction is None or self.uses != 0:
            raise RuntimeError("virtual quantum authority is unavailable")
        if not self.created_step <= step < self.expires_step:
            raise RuntimeError("virtual quantum authority expired")
        self.uses = 1

    def erase(self, *, reason: str) -> dict[str, Any]:
        if self.direction is None:
            raise RuntimeError("virtual quantum direction was already released")
        prior = tensor_sha256(self.direction)
        self.direction.fill(0.0)
        zero_sha256 = tensor_sha256(self.direction)
        all_zero = bool(np.count_nonzero(self.direction) == 0)
        self.direction = None
        payload = {
            "quantum_id": self.quantum_id,
            "reason": reason,
            "prior_direction_sha256": prior,
            "zeroized_direction_sha256": zero_sha256,
            "direction_shape": list(self.direction_shape),
            "direction_dtype": self.direction_dtype,
            "all_zero_before_release": all_zero,
            "private_reference_released": self.direction is None,
        }
        return {**payload, "erasure_sha256": canonical_sha256(payload)}


def _candidate_states(
    baseline_state: Any,
    anchor_state: Any,
    *,
    protected_positions: Sequence[int],
    source_positions: Sequence[int],
    objective_sha256: str,
    subject_sha256: str,
    branch_index: int,
    created_step: int,
    config: VirtualQuantaConfig,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[str, Any]],
    _PrivateQuantum,
    str,
]:
    baseline = _state(baseline_state, name="virtual quantum baseline")
    anchor = _state(anchor_state, name="virtual quantum anchor")
    if baseline.shape != anchor.shape:
        raise ValueError("virtual quantum baseline/anchor shapes differ")
    protected = _positions(
        protected_positions,
        n_slots=baseline.shape[1],
        name="protected",
    )
    source = _positions(
        source_positions,
        n_slots=baseline.shape[1],
        name="source",
    )
    if any(position not in protected for position in source):
        raise ValueError("virtual quantum source is not immutable evidence")
    mutable = tuple(position for position in range(baseline.shape[1]) if position not in protected)
    if not mutable:
        raise ValueError("virtual quantum has no mutable workspace")
    if (
        not is_sha256(objective_sha256)
        or not is_sha256(subject_sha256)
        or type(branch_index) is not int
        or branch_index < 0
    ):
        raise ValueError("virtual quantum source identity is invalid")

    direction = np.zeros_like(baseline)
    if source:
        evidence_vector = np.mean(
            baseline[:, source, :],
            axis=1,
            keepdims=True,
            dtype=np.float64,
        ).astype(baseline.dtype)
        anchor_vector = np.mean(
            anchor[:, mutable, :],
            axis=1,
            keepdims=True,
            dtype=np.float64,
        ).astype(baseline.dtype)
        target = 0.75 * evidence_vector + 0.25 * anchor_vector
        direction[:, mutable, :] = target - baseline[:, mutable, :]
        source_kind = "immutable_context_projection"
    else:
        direction[:, mutable, :] = anchor[:, mutable, :] - baseline[:, mutable, :]
        source_kind = "prompt_anchor_projection"
    direction[:, protected, :] = 0.0

    mutable_state = baseline[:, mutable, :]
    mutable_direction = direction[:, mutable, :]
    if not source and _rms(mutable_direction) <= 1e-12:
        rotated = np.roll(mutable_state, shift=1, axis=-1)
        direction[:, mutable, :] = rotated - mutable_state
        mutable_direction = direction[:, mutable, :]
        source_kind = "prompt_self_projection"
    state_rms = max(_rms(mutable_state), 1e-6)
    direction_rms = _rms(mutable_direction)
    max_delta_rms = float(config.max_relative_delta_rms) * state_rms
    if direction_rms <= 1e-12 or max_delta_rms <= 1e-12:
        raise ValueError("virtual quantum source direction degenerated")
    delta_rms = min(direction_rms, max_delta_rms)
    direction *= delta_rms / direction_rms
    direction[:, protected, :] = 0.0

    seed_material = (
        f"{objective_sha256}:{subject_sha256}:{branch_index}:{config.seed}:"
        f"{tensor_sha256(baseline)}:{tensor_sha256(anchor)}"
    )
    seed = int.from_bytes(hashlib.sha256(seed_material.encode("ascii")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    random_direction = np.zeros_like(baseline)
    random_mutable = rng.standard_normal(mutable_direction.shape).astype(
        baseline.dtype,
        copy=False,
    )
    random_mutable -= np.mean(random_mutable, axis=-1, keepdims=True)
    guided_flat = direction[:, mutable, :].reshape(-1).astype(np.float64)
    random_flat = random_mutable.reshape(-1).astype(np.float64)
    guided_energy = float(np.dot(guided_flat, guided_flat))
    random_flat -= (
        float(np.dot(random_flat, guided_flat)) / max(guided_energy, 1e-18)
    ) * guided_flat
    random_mutable = random_flat.reshape(random_mutable.shape).astype(baseline.dtype)
    random_rms = _rms(random_mutable)
    if random_rms <= 1e-12:
        raise ValueError("virtual quantum matched-random control degenerated")
    random_mutable *= delta_rms / random_rms
    random_direction[:, mutable, :] = random_mutable

    states = {
        "no_quantum": np.array(baseline, copy=True),
        "matched_random": np.array(baseline + random_direction, copy=True),
        "guided_quantum": np.array(baseline + direction, copy=True),
    }
    summaries: dict[str, dict[str, Any]] = {}
    for name, candidate in states.items():
        delta = candidate - baseline
        delta_mutable = delta[:, mutable, :]
        changed_positions = [
            position
            for position in range(baseline.shape[1])
            if not np.array_equal(candidate[:, position, :], baseline[:, position, :])
        ]
        delta_flat = delta_mutable.reshape(-1).astype(np.float64)
        if not np.any(delta_flat):
            cosine = None
        else:
            cosine = round(
                float(np.dot(delta_flat, guided_flat))
                / max(
                    math.sqrt(float(np.dot(delta_flat, delta_flat)) * guided_energy),
                    1e-18,
                ),
                12,
            )
        summaries[name] = {
            "name": name,
            "state_sha256": tensor_sha256(candidate),
            "delta_rms": round(_rms(delta_mutable), 12),
            "relative_mutable_delta_rms": round(
                _rms(delta_mutable) / state_rms,
                12,
            ),
            "cosine_to_guided": cosine,
            "changed_positions": changed_positions,
            "protected_positions_unchanged": all(
                np.array_equal(candidate[:, position, :], baseline[:, position, :])
                for position in protected
            ),
        }
    if summaries["no_quantum"]["state_sha256"] != tensor_sha256(baseline):
        raise RuntimeError("virtual quantum no-op changed state")
    if (
        summaries["matched_random"]["state_sha256"]
        in {
            summaries["no_quantum"]["state_sha256"],
            summaries["guided_quantum"]["state_sha256"],
        }
        or summaries["guided_quantum"]["state_sha256"] == summaries["no_quantum"]["state_sha256"]
    ):
        raise RuntimeError("virtual quantum arms are not distinct")
    if not math.isclose(
        summaries["matched_random"]["delta_rms"],
        summaries["guided_quantum"]["delta_rms"],
        rel_tol=1e-5,
        abs_tol=1e-8,
    ):
        raise RuntimeError("virtual quantum random arm is not norm matched")
    if any(
        not summary["protected_positions_unchanged"]
        or summary["relative_mutable_delta_rms"] > float(config.max_relative_delta_rms) + 1e-6
        for summary in summaries.values()
    ):
        raise RuntimeError("virtual quantum escaped its state bound")

    direction_sha256 = tensor_sha256(direction)
    identity = {
        "schema": VIRTUAL_QUANTA_SCHEMA,
        "objective_sha256": objective_sha256,
        "subject_sha256": subject_sha256,
        "branch_index": branch_index,
        "baseline_state_sha256": tensor_sha256(baseline),
        "anchor_state_sha256": tensor_sha256(anchor),
        "direction_sha256": direction_sha256,
        "protected_positions": list(protected),
        "source_positions": list(source),
        "created_step": created_step,
        "expires_step": created_step + config.ttl_steps,
        "config": config.to_dict(),
    }
    quantum = _PrivateQuantum(
        quantum_id="vq-" + canonical_sha256(identity)[:24],
        direction=direction,
        created_step=created_step,
        expires_step=created_step + config.ttl_steps,
        direction_sha256=direction_sha256,
        direction_shape=list(direction.shape),
        direction_dtype=str(direction.dtype),
    )
    return states, summaries, quantum, source_kind


def _arm_order(
    *,
    episode_id: str,
    objective_sha256: str,
    replicate: int,
    seed: int,
) -> tuple[str, ...]:
    digest = hashlib.sha256(f"{episode_id}:{objective_sha256}:{seed}".encode()).digest()
    base_offset = int.from_bytes(digest[:2], "big") % len(ARM_NAMES)
    offset = (base_offset + replicate) % len(ARM_NAMES)
    return ARM_NAMES[offset:] + ARM_NAMES[:offset]


def _empty_receipt(
    *,
    config: VirtualQuantaConfig,
    episode_id: str,
    objective_sha256: str,
    subject_sha256: str,
    branch_index: int,
    source_kv_boundary_sha256: str,
    protected_positions: Sequence[int],
    source_positions: Sequence[int],
    created_step: int,
    status: str,
    reason: str,
) -> dict[str, Any]:
    payload = {
        "schema": VIRTUAL_QUANTA_RECEIPT_SCHEMA,
        "config": config.to_dict(),
        "episode_id": episode_id,
        "objective_sha256": objective_sha256,
        "subject_sha256": subject_sha256,
        "branch_index": branch_index,
        "source_kv_boundary_sha256": source_kv_boundary_sha256,
        "verifier_policy_sha256": "",
        "verifier_preflight_sha256": "",
        "protected_positions": list(protected_positions),
        "source_positions": list(source_positions),
        "created_step": created_step,
        "expires_step": created_step + config.ttl_steps,
        "quantum_id": "",
        "source_kind": "",
        "baseline_state_sha256": "",
        "anchor_state_sha256": "",
        "direction_sha256": "",
        "direction_shape": [],
        "direction_dtype": "",
        "arms": [],
        "execution_order": [],
        "all_arms_stable": False,
        "all_arms_equal_resources": False,
        "all_arms_fully_metered": False,
        "guided_beats_controls": False,
        "contribution": {},
        "application": {
            "attempted": False,
            "applied": False,
            "uses": 0,
            "one_use": True,
            "ttl_valid": False,
            "pre_state_sha256": "",
            "post_state_sha256": "",
            "protected_positions_unchanged": False,
        },
        "erasure": {},
        "status": status,
        "reason": reason,
        "authority_scope": "episode_objective_branch_kv_one_use",
        "critic_prose_authority": False,
        "caller_vector_authority": False,
        "durable_weight_change": False,
        "answer_text_stored": False,
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def build_empty_virtual_quanta_receipt(
    *,
    episode_id: str,
    objective_sha256: str,
    subject_sha256: str,
    branch_index: int,
    source_kv_boundary_sha256: str,
    protected_positions: Sequence[int],
    source_positions: Sequence[int],
    config: VirtualQuantaConfig | None = None,
    status: str = "unavailable",
    reason: str = "counterfactual_authority_unavailable",
) -> dict[str, Any]:
    """Build an explicit no-authority receipt for non-verifier episodes."""

    active_config = config or VirtualQuantaConfig()
    if not isinstance(active_config, VirtualQuantaConfig):
        raise TypeError("empty virtual quanta config is invalid")
    if (
        status not in {"disabled", "unavailable"}
        or (status == "disabled") is not (active_config.mode == DISABLED)
        or not isinstance(reason, str)
        or not reason
    ):
        raise ValueError("empty virtual quanta status is invalid")
    if (
        not isinstance(episode_id, str)
        or not episode_id
        or not is_sha256(objective_sha256)
        or not is_sha256(subject_sha256)
        or not is_sha256(source_kv_boundary_sha256)
        or type(branch_index) is not int
        or branch_index < 0
    ):
        raise ValueError("empty virtual quanta identity is invalid")
    protected, source = _receipt_positions(
        protected_positions,
        source_positions,
    )
    if any(position not in protected for position in source):
        raise ValueError("empty virtual quanta source positions are invalid")
    return _empty_receipt(
        config=active_config,
        episode_id=episode_id,
        objective_sha256=objective_sha256,
        subject_sha256=subject_sha256,
        branch_index=branch_index,
        source_kv_boundary_sha256=source_kv_boundary_sha256,
        protected_positions=protected,
        source_positions=source,
        created_step=0,
        status=status,
        reason=reason,
    )


def run_virtual_quanta(
    *,
    baseline_state: Any,
    anchor_state: Any,
    branch_index: int,
    protected_positions: Sequence[int],
    source_positions: Sequence[int],
    episode_id: str,
    objective_sha256: str,
    subject_sha256: str,
    source_kv_boundary_sha256: str,
    verifier_policy_sha256: str,
    verifier_preflight_sha256: str,
    created_step: int,
    config: VirtualQuantaConfig,
    evaluate: Callable[[str, Any, int], CounterfactualProbeResult] | None,
    apply_state: Callable[[Any], Any] | None,
    restore_state: Callable[[Any], Any] | None,
    budget: Any,
    unavailable_reason: str = "",
) -> dict[str, Any]:
    """Evaluate, apply, and erase one latent quantum transaction."""

    if not isinstance(config, VirtualQuantaConfig):
        raise TypeError("virtual quanta config is invalid")
    baseline = _state(baseline_state, name="virtual quantum baseline")
    protected = _positions(
        protected_positions,
        n_slots=baseline.shape[1],
        name="protected",
    )
    source = _positions(
        source_positions,
        n_slots=baseline.shape[1],
        name="source",
    )
    step = _bounded_step(created_step, name="virtual quantum created step")
    if step + config.ttl_steps > MAX_ACTION_STEP:
        raise ValueError("virtual quantum lifetime exceeds action-step bound")
    if (
        not isinstance(episode_id, str)
        or not episode_id
        or not is_sha256(objective_sha256)
        or not is_sha256(subject_sha256)
        or not is_sha256(source_kv_boundary_sha256)
        or type(branch_index) is not int
        or branch_index < 0
        or any(position not in protected for position in source)
    ):
        raise ValueError("virtual quantum episode identity is invalid")
    common = {
        "config": config,
        "episode_id": str(episode_id),
        "objective_sha256": objective_sha256,
        "subject_sha256": subject_sha256,
        "branch_index": branch_index,
        "source_kv_boundary_sha256": source_kv_boundary_sha256,
        "protected_positions": protected,
        "source_positions": source,
        "created_step": step,
    }
    if config.mode == DISABLED:
        return _empty_receipt(
            **common,
            status="disabled",
            reason="configured_disabled",
        )
    if (
        evaluate is None
        or apply_state is None
        or restore_state is None
        or not is_sha256(verifier_policy_sha256)
        or not is_sha256(verifier_preflight_sha256)
    ):
        return _empty_receipt(
            **common,
            status="unavailable",
            reason=unavailable_reason or "counterfactual_authority_unavailable",
        )

    quantum: _PrivateQuantum | None = None
    erasure: dict[str, Any] = {}
    states: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, Any]] = {}
    source_kind = ""
    arms = {name: [] for name in ARM_NAMES}
    arm_rows: list[dict[str, Any]] = []
    execution_order: list[dict[str, Any]] = []
    status = "restored"
    reason = "counterfactual_not_admitted"
    contribution: dict[str, Any] = {}
    application = {
        "attempted": False,
        "applied": False,
        "uses": 0,
        "one_use": True,
        "ttl_valid": False,
        "pre_state_sha256": tensor_sha256(baseline),
        "post_state_sha256": tensor_sha256(baseline),
        "protected_positions_unchanged": True,
    }
    all_stable = all_equal = all_metered = guided_wins = False
    try:
        states, summaries, quantum, source_kind = _candidate_states(
            baseline,
            anchor_state,
            protected_positions=protected,
            source_positions=source,
            objective_sha256=objective_sha256,
            subject_sha256=subject_sha256,
            branch_index=branch_index,
            created_step=step,
            config=config,
        )
        budget.charge_tensor_work(
            "virtual_quanta_candidate_generation",
            element_reads=int(baseline.size * 8),
            element_writes=int(baseline.size * 7),
            scalar_ops=int(baseline.size * 18),
            host_scalar_ops=128,
        )
        for replicate in range(config.replicates):
            order = _arm_order(
                episode_id=episode_id,
                objective_sha256=objective_sha256,
                replicate=replicate,
                seed=config.seed,
            )
            execution_order.append(
                {
                    "replicate": replicate,
                    "arms": list(order),
                }
            )
            for name in order:
                before = _resource_snapshot(budget)
                raw = evaluate(name, states[name], replicate)
                after = _resource_snapshot(budget)
                probe = raw.normalized()
                delta = _resource_delta(before, after)
                row = {
                    "replicate": replicate,
                    "probe_tokens_sha256": probe["probe_tokens_sha256"],
                    "probe_token_count": probe["probe_token_count"],
                    "observation": probe["observation"],
                    "layer_apps": probe["layer_apps"],
                    "resource_before": before,
                    "resource_after": after,
                    "resource_delta": delta,
                    "fully_metered": _fully_metered(delta, probe=probe),
                }
                arms[name].append(row)
        arm_rows = []
        for name in ARM_NAMES:
            observations = [validate_observation(row["observation"]) for row in arms[name]]
            common_lower = max(float(row["lower_bound"]) for row in observations)
            common_upper = min(float(row["upper_bound"]) for row in observations)
            stable = common_lower <= common_upper + 1e-9
            arm_rows.append(
                {
                    **summaries[name],
                    "replicates": arms[name],
                    "complete": len(arms[name]) == config.replicates,
                    "stable": stable,
                    "score_mean": round(
                        sum(float(row["score"]) for row in observations) / len(observations),
                        10,
                    ),
                    "lower_bound": round(
                        min(float(row["lower_bound"]) for row in observations),
                        10,
                    ),
                    "upper_bound": round(
                        max(float(row["upper_bound"]) for row in observations),
                        10,
                    ),
                }
            )
        by_name = {row["name"]: row for row in arm_rows}
        all_stable = all(
            row["stable"]
            and all(trial["observation"]["authoritative"] is True for trial in row["replicates"])
            for row in arm_rows
        )
        resource_vectors = [
            tuple(trial["resource_delta"][counter] for counter in RESOURCE_COUNTERS)
            for row in arm_rows
            for trial in row["replicates"]
        ]
        all_equal = len(set(resource_vectors)) == 1
        all_metered = all(trial["fully_metered"] for row in arm_rows for trial in row["replicates"])
        control_upper = max(
            float(by_name["no_quantum"]["upper_bound"]),
            float(by_name["matched_random"]["upper_bound"]),
        )
        guided_lower = float(by_name["guided_quantum"]["lower_bound"])
        margin = guided_lower - control_upper
        guided_wins = bool(
            all_stable
            and all_equal
            and all_metered
            and margin > float(config.min_verifier_margin) + 1e-9
        )
        contribution = {
            "score": round(
                float(by_name["guided_quantum"]["score_mean"])
                - max(
                    float(by_name["no_quantum"]["score_mean"]),
                    float(by_name["matched_random"]["score_mean"]),
                ),
                10,
            ),
            "lower_bound": round(margin, 10),
            "measured_before_credit": True,
            "basis": "matched_counterfactual_verifier_margin",
        }
        if not all_stable:
            reason = "counterfactual_observations_unstable_or_untrusted"
        elif not all_equal:
            reason = "counterfactual_resource_mismatch"
        elif not all_metered:
            reason = "counterfactual_resource_accounting_incomplete"
        elif not guided_wins:
            reason = "guided_quantum_did_not_beat_controls"
        else:
            application["attempted"] = True
            quantum.consume(step=step)
            applied = _state(
                apply_state(np.array(states["guided_quantum"], copy=True)),
                name="virtual quantum applied",
            )
            if tensor_sha256(applied) != summaries["guided_quantum"]["state_sha256"]:
                raise RuntimeError("virtual quantum applied state differs")
            application = {
                **application,
                "applied": True,
                "uses": quantum.uses,
                "ttl_valid": quantum.created_step <= step < quantum.expires_step,
                "post_state_sha256": tensor_sha256(applied),
                "protected_positions_unchanged": all(
                    np.array_equal(
                        applied[:, position, :],
                        baseline[:, position, :],
                    )
                    for position in protected
                ),
            }
            if not application["protected_positions_unchanged"]:
                raise RuntimeError("virtual quantum changed protected evidence")
            status = "applied"
            reason = "guided_quantum_verified"
        if not guided_wins:
            restored = _state(
                restore_state(np.array(baseline, copy=True)),
                name="virtual quantum restored",
            )
            if tensor_sha256(restored) != tensor_sha256(baseline):
                raise RuntimeError("virtual quantum rollback failed")
            application["post_state_sha256"] = tensor_sha256(restored)
    except Exception as exc:  # noqa: BLE001 - transactional rollback owns extension failures
        restored = _state(
            restore_state(np.array(baseline, copy=True)),
            name="virtual quantum restored",
        )
        if tensor_sha256(restored) != tensor_sha256(baseline):
            raise RuntimeError("virtual quantum rollback failed") from exc
        status = "restored"
        reason = f"counterfactual_failed:{type(exc).__name__}"
        application["applied"] = False
        if quantum is not None:
            application["uses"] = quantum.uses
            application["ttl_valid"] = bool(
                quantum.uses == 1 and quantum.created_step <= step < quantum.expires_step
            )
        application["post_state_sha256"] = tensor_sha256(restored)
        if not arm_rows and summaries:
            arm_rows = [
                {
                    **summaries[name],
                    "replicates": arms[name],
                    "complete": False,
                    "stable": False,
                    "score_mean": None,
                    "lower_bound": None,
                    "upper_bound": None,
                }
                for name in ARM_NAMES
            ]
    finally:
        if quantum is not None and quantum.direction is not None:
            erasure = quantum.erase(
                reason=(
                    "consumed_after_verified_application"
                    if status == "applied"
                    else "trial_restored_or_failed"
                )
            )

    if quantum is None:
        return _empty_receipt(
            **common,
            status="unavailable",
            reason=reason,
        )

    payload = {
        "schema": VIRTUAL_QUANTA_RECEIPT_SCHEMA,
        "config": config.to_dict(),
        "episode_id": episode_id,
        "objective_sha256": objective_sha256,
        "subject_sha256": subject_sha256,
        "branch_index": branch_index,
        "source_kv_boundary_sha256": source_kv_boundary_sha256,
        "verifier_policy_sha256": verifier_policy_sha256,
        "verifier_preflight_sha256": verifier_preflight_sha256,
        "protected_positions": list(protected),
        "source_positions": list(source),
        "created_step": step,
        "expires_step": step + config.ttl_steps,
        "quantum_id": quantum.quantum_id if quantum is not None else "",
        "source_kind": source_kind,
        "baseline_state_sha256": tensor_sha256(baseline),
        "anchor_state_sha256": tensor_sha256(_state(anchor_state, name="anchor")),
        "direction_sha256": quantum.direction_sha256 if quantum is not None else "",
        "direction_shape": list(quantum.direction_shape) if quantum is not None else [],
        "direction_dtype": quantum.direction_dtype if quantum is not None else "",
        "arms": arm_rows,
        "execution_order": execution_order,
        "all_arms_stable": all_stable,
        "all_arms_equal_resources": all_equal,
        "all_arms_fully_metered": all_metered,
        "guided_beats_controls": guided_wins,
        "contribution": contribution,
        "application": application,
        "erasure": erasure,
        "status": status,
        "reason": reason,
        "authority_scope": "episode_objective_branch_kv_one_use",
        "critic_prose_authority": False,
        "caller_vector_authority": False,
        "durable_weight_change": False,
        "answer_text_stored": False,
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def _validate_external_bindings(
    receipt: Mapping[str, Any],
    *,
    episode_id: str,
    objective_sha256: str,
    protected: tuple[int, ...],
    source: tuple[int, ...],
    cognitive_slots: Sequence[Mapping[str, Any]] | None,
    verifier_preflight: Mapping[str, Any] | None,
    information_accounting: Mapping[str, Any] | None,
    resource_accounting: Mapping[str, Any] | None,
    kv_state_tree: Mapping[str, Any] | None,
    verifier_required: bool,
) -> dict[str, int]:
    if (
        not isinstance(information_accounting, Mapping)
        or not isinstance(resource_accounting, Mapping)
        or not isinstance(kv_state_tree, Mapping)
        or not isinstance(cognitive_slots, Sequence)
        or isinstance(cognitive_slots, (str, bytes))
    ):
        raise ValueError("virtual quanta external evidence is absent")
    from core.brain.llm.latent_cortex.blind_review import (
        validate_decoy_preflight_receipt,
    )
    from core.brain.llm.latent_cortex.resource_accounting import (
        validate_information_receipt,
        validate_resource_receipt,
    )

    information = validate_information_receipt(information_accounting)
    resources = validate_resource_receipt(resource_accounting)
    preflight = None
    if isinstance(verifier_preflight, Mapping) and verifier_preflight:
        preflight = validate_decoy_preflight_receipt(
            dict(verifier_preflight),
            episode_id=episode_id,
            objective_sha256=objective_sha256,
        )
    nodes = kv_state_tree.get("nodes")
    cognitive_positions = sorted(
        row.get("slot")
        for row in cognitive_slots
        if isinstance(row, Mapping) and type(row.get("slot")) is int
    )
    information_policies = information.get("policies")
    expected_verifier_policy = (
        information_policies.get("verifier") if isinstance(information_policies, Mapping) else None
    )
    if (
        information["accounting_complete"] is not True
        or resources["accounting_complete"] is not True
        or not isinstance(nodes, list)
        or not any(
            isinstance(node, Mapping)
            and node.get("node_sha256") == receipt["source_kv_boundary_sha256"]
            and node.get("branch_index") in {None, receipt["branch_index"]}
            for node in nodes
        )
        or list(source) != cognitive_positions
        or any(position not in protected for position in cognitive_positions)
        or (
            verifier_required
            and (
                preflight is None
                or preflight["verifier_admitted"] is not True
                or receipt["verifier_preflight_sha256"] != preflight["receipt_sha256"]
                or receipt["verifier_policy_sha256"] != expected_verifier_policy
            )
        )
    ):
        raise ValueError("virtual quanta external source binding differs")
    return dict(resources["totals"])


def validate_virtual_quanta_receipt(
    value: Any,
    *,
    episode_id: str,
    objective_sha256: str,
    n_branches: int,
    expected_config: VirtualQuantaConfig | None = None,
    cognitive_slots: Sequence[Mapping[str, Any]] | None = None,
    verifier_preflight: Mapping[str, Any] | None = None,
    information_accounting: Mapping[str, Any] | None = None,
    resource_accounting: Mapping[str, Any] | None = None,
    kv_state_tree: Mapping[str, Any] | None = None,
    require_external_bindings: bool = False,
) -> dict[str, Any]:
    """Reconstruct authority, parity, contribution, use, and erasure."""

    fields = {
        "schema",
        "config",
        "episode_id",
        "objective_sha256",
        "subject_sha256",
        "branch_index",
        "source_kv_boundary_sha256",
        "verifier_policy_sha256",
        "verifier_preflight_sha256",
        "protected_positions",
        "source_positions",
        "created_step",
        "expires_step",
        "quantum_id",
        "source_kind",
        "baseline_state_sha256",
        "anchor_state_sha256",
        "direction_sha256",
        "direction_shape",
        "direction_dtype",
        "arms",
        "execution_order",
        "all_arms_stable",
        "all_arms_equal_resources",
        "all_arms_fully_metered",
        "guided_beats_controls",
        "contribution",
        "application",
        "erasure",
        "status",
        "reason",
        "authority_scope",
        "critic_prose_authority",
        "caller_vector_authority",
        "durable_weight_change",
        "answer_text_stored",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("virtual quanta receipt fields differ")
    receipt = dict(value)
    payload = {name: receipt[name] for name in fields - {"receipt_sha256"}}
    config = expected_config or VirtualQuantaConfig()
    if (
        receipt["schema"] != VIRTUAL_QUANTA_RECEIPT_SCHEMA
        or receipt["config"] != config.to_dict()
        or receipt["episode_id"] != episode_id
        or receipt["objective_sha256"] != objective_sha256
        or receipt["receipt_sha256"] != canonical_sha256(payload)
        or receipt["authority_scope"] != "episode_objective_branch_kv_one_use"
        or receipt["critic_prose_authority"] is not False
        or receipt["caller_vector_authority"] is not False
        or receipt["durable_weight_change"] is not False
        or receipt["answer_text_stored"] is not False
        or type(n_branches) is not int
        or n_branches < 1
        or type(receipt["branch_index"]) is not int
        or not 0 <= receipt["branch_index"] < n_branches
        or not is_sha256(receipt["subject_sha256"])
        or not is_sha256(receipt["source_kv_boundary_sha256"])
    ):
        raise ValueError("virtual quanta receipt identity is invalid")
    protected, source = _receipt_positions(
        receipt["protected_positions"],
        receipt["source_positions"],
    )
    if any(position not in protected for position in source):
        raise ValueError("virtual quanta source positions are invalid")
    created_step = _bounded_step(
        receipt["created_step"],
        name="virtual quantum created step",
    )
    if (
        created_step + config.ttl_steps > MAX_ACTION_STEP
        or receipt["expires_step"] != created_step + config.ttl_steps
    ):
        raise ValueError("virtual quanta expiry differs")
    if receipt["status"] not in {"disabled", "unavailable", "applied", "restored"}:
        raise ValueError("virtual quanta status is invalid")
    external_totals = (
        _validate_external_bindings(
            receipt,
            episode_id=episode_id,
            objective_sha256=objective_sha256,
            protected=protected,
            source=source,
            cognitive_slots=cognitive_slots,
            verifier_preflight=verifier_preflight,
            information_accounting=information_accounting,
            resource_accounting=resource_accounting,
            kv_state_tree=kv_state_tree,
            verifier_required=bool(receipt["arms"]),
        )
        if require_external_bindings
        else None
    )

    if receipt["status"] in {"disabled", "unavailable"}:
        inactive_application = {
            "attempted": False,
            "applied": False,
            "uses": 0,
            "one_use": True,
            "ttl_valid": False,
            "pre_state_sha256": "",
            "post_state_sha256": "",
            "protected_positions_unchanged": False,
        }
        if (
            (receipt["status"] == "disabled" and config.mode != DISABLED)
            or (receipt["status"] == "unavailable" and config.mode == DISABLED)
            or receipt["arms"]
            or receipt["execution_order"]
            or receipt["quantum_id"]
            or receipt["source_kind"]
            or receipt["baseline_state_sha256"]
            or receipt["anchor_state_sha256"]
            or receipt["direction_sha256"]
            or receipt["direction_shape"]
            or receipt["direction_dtype"]
            or receipt["verifier_policy_sha256"]
            or receipt["verifier_preflight_sha256"]
            or receipt["all_arms_stable"] is not False
            or receipt["all_arms_equal_resources"] is not False
            or receipt["all_arms_fully_metered"] is not False
            or receipt["contribution"]
            or receipt["application"] != inactive_application
            or receipt["erasure"]
            or receipt["guided_beats_controls"] is not False
            or not isinstance(receipt["reason"], str)
            or not receipt["reason"]
        ):
            raise ValueError("inactive virtual quanta receipt minted authority")
        return receipt
    if (
        not receipt["quantum_id"].startswith("vq-")
        or not is_sha256(receipt["baseline_state_sha256"])
        or not is_sha256(receipt["anchor_state_sha256"])
        or not is_sha256(receipt["direction_sha256"])
        or not is_sha256(receipt["verifier_policy_sha256"])
        or not is_sha256(receipt["verifier_preflight_sha256"])
        or config.mode != COUNTERFACTUAL
        or receipt["source_kind"]
        not in {
            "immutable_context_projection",
            "prompt_anchor_projection",
            "prompt_self_projection",
        }
        or not isinstance(receipt["direction_shape"], list)
        or len(receipt["direction_shape"]) != 3
        or any(type(size) is not int or size < 1 for size in receipt["direction_shape"])
        or any(position >= receipt["direction_shape"][1] for position in protected)
        or not isinstance(receipt["direction_dtype"], str)
        or not receipt["direction_dtype"]
    ):
        raise ValueError("virtual quanta private authority evidence is invalid")
    quantum_identity = {
        "schema": VIRTUAL_QUANTA_SCHEMA,
        "objective_sha256": receipt["objective_sha256"],
        "subject_sha256": receipt["subject_sha256"],
        "branch_index": receipt["branch_index"],
        "baseline_state_sha256": receipt["baseline_state_sha256"],
        "anchor_state_sha256": receipt["anchor_state_sha256"],
        "direction_sha256": receipt["direction_sha256"],
        "protected_positions": list(protected),
        "source_positions": list(source),
        "created_step": created_step,
        "expires_step": receipt["expires_step"],
        "config": config.to_dict(),
    }
    if receipt["quantum_id"] != "vq-" + canonical_sha256(quantum_identity)[:24]:
        raise ValueError("virtual quanta private authority identity differs")
    try:
        zero = np.zeros(
            tuple(receipt["direction_shape"]),
            dtype=np.dtype(receipt["direction_dtype"]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("virtual quanta direction dtype is invalid") from exc
    erasure = receipt["erasure"]
    erasure_fields = {
        "quantum_id",
        "reason",
        "prior_direction_sha256",
        "zeroized_direction_sha256",
        "direction_shape",
        "direction_dtype",
        "all_zero_before_release",
        "private_reference_released",
        "erasure_sha256",
    }
    if (
        not isinstance(erasure, Mapping)
        or set(erasure) != erasure_fields
        or erasure["quantum_id"] != receipt["quantum_id"]
        or erasure["prior_direction_sha256"] != receipt["direction_sha256"]
        or erasure["zeroized_direction_sha256"] != tensor_sha256(zero)
        or erasure["direction_shape"] != receipt["direction_shape"]
        or erasure["direction_dtype"] != receipt["direction_dtype"]
        or erasure["all_zero_before_release"] is not True
        or erasure["private_reference_released"] is not True
        or erasure["erasure_sha256"]
        != canonical_sha256({name: erasure[name] for name in erasure_fields - {"erasure_sha256"}})
    ):
        raise ValueError("virtual quanta erasure is invalid")

    if (
        not isinstance(receipt["arms"], list)
        or [row.get("name") for row in receipt["arms"]] != list(ARM_NAMES)
        or not isinstance(receipt["execution_order"], list)
        or len(receipt["execution_order"]) > config.replicates
    ):
        raise ValueError("virtual quanta counterfactual arms are invalid")
    for replicate, execution in enumerate(receipt["execution_order"]):
        if (
            not isinstance(execution, Mapping)
            or set(execution) != {"replicate", "arms"}
            or execution["replicate"] != replicate
            or execution["arms"]
            != list(
                _arm_order(
                    episode_id=episode_id,
                    objective_sha256=objective_sha256,
                    replicate=replicate,
                    seed=config.seed,
                )
            )
        ):
            raise ValueError("virtual quanta execution order differs")
    complete_counterfactual = bool(
        len(receipt["execution_order"]) == config.replicates
        and all(
            isinstance(arm, Mapping)
            and arm.get("complete") is True
            and isinstance(arm.get("replicates"), list)
            and len(arm["replicates"]) == config.replicates
            for arm in receipt["arms"]
        )
    )
    if not complete_counterfactual:
        application = receipt["application"]
        if (
            receipt["status"] != "restored"
            or not str(receipt["reason"]).startswith("counterfactual_failed:")
            or receipt["all_arms_stable"] is not False
            or receipt["all_arms_equal_resources"] is not False
            or receipt["all_arms_fully_metered"] is not False
            or receipt["guided_beats_controls"] is not False
            or receipt["contribution"] != {}
            or not isinstance(application, Mapping)
            or application.get("attempted") is not False
            or application.get("applied") is not False
            or application.get("uses") != 0
            or application.get("post_state_sha256") != receipt["baseline_state_sha256"]
        ):
            raise ValueError("failed virtual quanta transaction minted authority")
        for arm in receipt["arms"]:
            trials = arm.get("replicates") if isinstance(arm, Mapping) else None
            if (
                not isinstance(trials, list)
                or len(trials) > config.replicates
                or arm.get("complete") is not False
                or arm.get("stable") is not False
                or arm.get("score_mean") is not None
                or arm.get("lower_bound") is not None
                or arm.get("upper_bound") is not None
                or arm.get("protected_positions_unchanged") is not True
                or not _finite(arm.get("relative_mutable_delta_rms"))
                or arm["relative_mutable_delta_rms"] > float(config.max_relative_delta_rms) + 1e-6
            ):
                raise ValueError("failed virtual quanta arm evidence is invalid")
            seen_replicates: set[int] = set()
            for trial in trials:
                replicate = trial.get("replicate") if isinstance(trial, Mapping) else None
                if (
                    type(replicate) is not int
                    or not 0 <= replicate < config.replicates
                    or replicate in seen_replicates
                    or set(trial.get("resource_before", {})) != set(RESOURCE_COUNTERS)
                    or set(trial.get("resource_after", {})) != set(RESOURCE_COUNTERS)
                    or set(trial.get("resource_delta", {})) != set(RESOURCE_COUNTERS)
                ):
                    raise ValueError("failed virtual quanta trial evidence is invalid")
                seen_replicates.add(replicate)
                validate_observation(trial["observation"])
                expected_delta = _resource_delta(
                    trial["resource_before"],
                    trial["resource_after"],
                )
                expected_metered = _fully_metered(
                    expected_delta,
                    probe={
                        "layer_apps": trial["layer_apps"],
                        "probe_token_count": trial["probe_token_count"],
                    },
                )
                if (
                    trial["resource_delta"] != expected_delta
                    or trial["fully_metered"] is not expected_metered
                    or not is_sha256(trial["probe_tokens_sha256"])
                    or (
                        external_totals is not None
                        and any(
                            trial["resource_after"][name] > external_totals[name]
                            for name in RESOURCE_COUNTERS
                        )
                    )
                ):
                    raise ValueError("failed virtual quanta trial resources are invalid")
        flattened_order = [
            (execution["replicate"], arm)
            for execution in receipt["execution_order"]
            for arm in execution["arms"]
        ]
        observed_trials = {
            (trial["replicate"], arm["name"])
            for arm in receipt["arms"]
            for trial in arm["replicates"]
        }
        if observed_trials != set(flattened_order[: len(observed_trials)]):
            raise ValueError("failed virtual quanta trial prefix differs")
        return receipt
    resource_vectors: list[tuple[int, ...]] = []
    stable = True
    metered = True
    by_name: dict[str, Mapping[str, Any]] = {}
    for arm in receipt["arms"]:
        if (
            not isinstance(arm, Mapping)
            or not isinstance(arm.get("replicates"), list)
            or len(arm["replicates"]) != config.replicates
            or arm.get("complete") is not True
            or type(arm.get("stable")) is not bool
            or not _finite(arm.get("score_mean"))
            or not _finite(arm.get("lower_bound"))
            or not _finite(arm.get("upper_bound"))
            or arm["lower_bound"] > arm["upper_bound"]
            or type(arm.get("protected_positions_unchanged")) is not bool
            or arm["protected_positions_unchanged"] is not True
            or not _finite(arm.get("relative_mutable_delta_rms"))
            or arm["relative_mutable_delta_rms"] > float(config.max_relative_delta_rms) + 1e-6
        ):
            raise ValueError("virtual quanta arm evidence is invalid")
        observations = []
        for replicate, trial in enumerate(arm["replicates"]):
            if (
                not isinstance(trial, Mapping)
                or trial.get("replicate") != replicate
                or set(trial.get("resource_before", {})) != set(RESOURCE_COUNTERS)
                or set(trial.get("resource_after", {})) != set(RESOURCE_COUNTERS)
                or set(trial.get("resource_delta", {})) != set(RESOURCE_COUNTERS)
            ):
                raise ValueError("virtual quanta trial evidence is invalid")
            observation = validate_observation(trial["observation"])
            observations.append(observation)
            expected_delta = _resource_delta(
                trial["resource_before"],
                trial["resource_after"],
            )
            probe = {
                "layer_apps": trial["layer_apps"],
                "probe_token_count": trial["probe_token_count"],
            }
            expected_metered = _fully_metered(expected_delta, probe=probe)
            if (
                trial["resource_delta"] != expected_delta
                or trial["fully_metered"] is not expected_metered
                or not is_sha256(trial["probe_tokens_sha256"])
                or (
                    external_totals is not None
                    and any(
                        trial["resource_after"][name] > external_totals[name]
                        for name in RESOURCE_COUNTERS
                    )
                )
            ):
                raise ValueError("virtual quanta trial resources are invalid")
            resource_vectors.append(tuple(expected_delta[counter] for counter in RESOURCE_COUNTERS))
            metered = metered and expected_metered
        expected_stable = max(float(row["lower_bound"]) for row in observations) <= min(
            float(row["upper_bound"]) for row in observations
        ) + 1e-9 and all(row["authoritative"] is True for row in observations)
        expected_mean = round(
            sum(float(row["score"]) for row in observations) / len(observations),
            10,
        )
        if (
            arm["stable"] is not expected_stable
            or arm["score_mean"] != expected_mean
            or arm["lower_bound"]
            != round(min(float(row["lower_bound"]) for row in observations), 10)
            or arm["upper_bound"]
            != round(max(float(row["upper_bound"]) for row in observations), 10)
        ):
            raise ValueError("virtual quanta arm aggregation differs")
        stable = stable and expected_stable
        by_name[arm["name"]] = arm
    expected_equal = len(set(resource_vectors)) == 1
    control_upper = max(
        float(by_name["no_quantum"]["upper_bound"]),
        float(by_name["matched_random"]["upper_bound"]),
    )
    margin = float(by_name["guided_quantum"]["lower_bound"]) - control_upper
    expected_win = bool(
        stable and expected_equal and metered and margin > float(config.min_verifier_margin) + 1e-9
    )
    expected_contribution = {
        "score": round(
            float(by_name["guided_quantum"]["score_mean"])
            - max(
                float(by_name["no_quantum"]["score_mean"]),
                float(by_name["matched_random"]["score_mean"]),
            ),
            10,
        ),
        "lower_bound": round(margin, 10),
        "measured_before_credit": True,
        "basis": "matched_counterfactual_verifier_margin",
    }
    if (
        receipt["all_arms_stable"] is not stable
        or receipt["all_arms_equal_resources"] is not expected_equal
        or receipt["all_arms_fully_metered"] is not metered
        or receipt["guided_beats_controls"] is not expected_win
        or receipt["contribution"] != expected_contribution
    ):
        raise ValueError("virtual quanta decision evidence differs")

    application = receipt["application"]
    application_fields = {
        "attempted",
        "applied",
        "uses",
        "one_use",
        "ttl_valid",
        "pre_state_sha256",
        "post_state_sha256",
        "protected_positions_unchanged",
    }
    expected_applied = receipt["status"] == "applied"
    failed_after_verified_win = bool(
        expected_win
        and not expected_applied
        and receipt["status"] == "restored"
        and str(receipt["reason"]).startswith("counterfactual_failed:")
    )
    if (
        not isinstance(application, Mapping)
        or set(application) != application_fields
        or application["attempted"] is not expected_win
        or application["applied"] is not expected_applied
        or application["uses"] != (1 if expected_applied or failed_after_verified_win else 0)
        or application["one_use"] is not True
        or application["ttl_valid"] is not (expected_applied or failed_after_verified_win)
        or application["pre_state_sha256"] != receipt["baseline_state_sha256"]
        or application["post_state_sha256"]
        != (
            by_name["guided_quantum"]["state_sha256"]
            if expected_applied
            else receipt["baseline_state_sha256"]
        )
        or application["protected_positions_unchanged"] is not True
        or (expected_applied is not expected_win and not failed_after_verified_win)
    ):
        raise ValueError("virtual quanta application evidence is invalid")

    return receipt


__all__ = [
    "ARM_NAMES",
    "COUNTERFACTUAL",
    "DISABLED",
    "VIRTUAL_QUANTA_RECEIPT_SCHEMA",
    "VIRTUAL_QUANTA_SCHEMA",
    "VirtualQuantaConfig",
    "build_empty_virtual_quanta_receipt",
    "run_virtual_quanta",
    "validate_virtual_quanta_receipt",
]
