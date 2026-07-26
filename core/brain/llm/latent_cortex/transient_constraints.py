"""Verified, branch-local, one-use negative constraints for recurrent search.

The controller never turns critic prose into hidden-state authority.  A
constraint can be admitted only from an independently bounded verifier
observation and an equal-compute counterfactual trial showing that removing a
specific failed latent direction beats both the failed state and a
magnitude-matched orthogonal sham.

All tensors remain worker-private.  Public receipts contain commitments,
bounded geometry, verifier observations, scope, lifetime, and dispositions so
the parent service can independently reconstruct every authority decision.
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
    VerifierObservation,
    tensor_sha256,
    validate_observation,
)

TRANSIENT_CONSTRAINT_SCHEMA = "aura.rlc.transient_negative_constraints.v1"
TRANSIENT_CONSTRAINT_ATTEMPT_SCHEMA = "aura.rlc.transient_constraint_attempt.v1"
TRANSIENT_CONSTRAINT_APPLICATION_SCHEMA = "aura.rlc.transient_constraint_application.v1"
ARM_NAMES = ("failed_no_op", "matched_orthogonal_sham", "negative_direction")
CONSTRAINT_PARITY_COUNTERS = tuple(RESOURCE_COUNTERS)
MAX_REPLICATES = 3
MAX_CONSTRAINTS = 16
MAX_ACTION_STEP = 10_000
_EPSILON = 1e-9


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _bounded_action(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 80
        or not all(character.isalnum() or character in "._-" for character in value)
    ):
        raise ValueError("transient constraint action is invalid")
    return value


def _bounded_step(value: Any, *, name: str) -> int:
    if type(value) is not int or not 0 <= value <= MAX_ACTION_STEP:
        raise ValueError(f"{name} is invalid")
    return value


def _as_state(value: Any, *, name: str) -> np.ndarray:
    state = np.asarray(value)
    if (
        state.ndim != 3
        or state.shape[0] != 1
        or state.shape[1] < 1
        or state.shape[2] < 2
        or state.size > 100_000_000
        or not np.issubdtype(state.dtype, np.floating)
        or not np.all(np.isfinite(state))
    ):
        raise ValueError(f"{name} latent state is invalid")
    return np.array(state, copy=True)


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))


def _observation_signature(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        value["score"],
        value["lower_bound"],
        value["upper_bound"],
        value["sample_count"],
        value["basis"],
        value["independent"],
        value["authoritative"],
    )


def _normalized_observation(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping) and "observation_sha256" in value:
        return validate_observation(value)
    return VerifierObservation.from_value(value).to_dict()


def _resource_totals(budget: Any | None) -> dict[str, int] | None:
    ledger = getattr(budget, "resource_ledger", None)
    totals = getattr(ledger, "totals", None)
    if not callable(totals):
        return None
    value = totals()
    if set(value) != set(RESOURCE_COUNTERS):
        raise ValueError("constraint resource ledger counters differ")
    return {name: int(value[name]) for name in RESOURCE_COUNTERS}


def _resource_delta(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, int]:
    if set(before) != set(RESOURCE_COUNTERS) or set(after) != set(RESOURCE_COUNTERS):
        raise ValueError("constraint resource snapshots differ")
    delta = {name: int(after[name]) - int(before[name]) for name in RESOURCE_COUNTERS}
    if any(value < 0 for value in delta.values()):
        raise ValueError("constraint resource counters moved backwards")
    return delta


def _protected_map(
    value: Mapping[int, Sequence[int]] | None,
    *,
    n_branches: int,
) -> dict[int, tuple[int, ...]]:
    raw = dict(value or {})
    if set(raw) != set(range(n_branches)):
        raise ValueError("protected-position map must cover every branch")
    result: dict[int, tuple[int, ...]] = {}
    for branch_index, positions in raw.items():
        if isinstance(positions, (str, bytes)):
            raise ValueError("protected positions must be a sequence")
        normalized = tuple(sorted(set(positions)))
        if any(type(position) is not int or position < 0 for position in normalized):
            raise ValueError("protected position is invalid")
        result[branch_index] = normalized
    return result


@dataclass(frozen=True, slots=True)
class TransientConstraintConfig:
    """Bounded authority and lifetime for transient negative steering."""

    max_relative_delta_rms: float = 0.08
    min_verifier_margin: float = 0.01
    replicates: int = 2
    ttl_action_steps: int = 3
    max_constraints: int = 8

    def __post_init__(self) -> None:
        if (
            not _finite(self.max_relative_delta_rms)
            or not 0.0 < float(self.max_relative_delta_rms) <= 0.20
        ):
            raise ValueError("constraint delta bound must be inside (0, 0.20]")
        if (
            not _finite(self.min_verifier_margin)
            or not 0.0 <= float(self.min_verifier_margin) <= 0.25
        ):
            raise ValueError("constraint verifier margin must be inside [0, 0.25]")
        if type(self.replicates) is not int or not 2 <= self.replicates <= MAX_REPLICATES:
            raise ValueError(f"constraint replicates must be inside [2, {MAX_REPLICATES}]")
        if type(self.ttl_action_steps) is not int or not 1 <= self.ttl_action_steps <= 16:
            raise ValueError("constraint TTL must be inside [1, 16]")
        if (
            type(self.max_constraints) is not int
            or not 1 <= self.max_constraints <= MAX_CONSTRAINTS
        ):
            raise ValueError(f"constraint count must be inside [1, {MAX_CONSTRAINTS}]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_relative_delta_rms": round(float(self.max_relative_delta_rms), 10),
            "min_verifier_margin": round(float(self.min_verifier_margin), 10),
            "replicates": self.replicates,
            "ttl_action_steps": self.ttl_action_steps,
            "max_constraints": self.max_constraints,
        }

    @classmethod
    def from_value(
        cls,
        value: Mapping[str, Any] | None,
    ) -> TransientConstraintConfig:
        raw = dict(value or {})
        allowed = {
            "max_relative_delta_rms",
            "min_verifier_margin",
            "replicates",
            "ttl_action_steps",
            "max_constraints",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"transient constraints have unknown keys: {sorted(unknown)}")
        return cls(
            max_relative_delta_rms=raw.get("max_relative_delta_rms", 0.08),
            min_verifier_margin=raw.get("min_verifier_margin", 0.01),
            replicates=raw.get("replicates", 2),
            ttl_action_steps=raw.get("ttl_action_steps", 3),
            max_constraints=raw.get("max_constraints", 8),
        )


@dataclass(slots=True)
class _PrivateConstraint:
    constraint_id: str
    branch_index: int
    source_action: str
    created_action_step: int
    expires_after_action_step: int
    source_failure_upper_bound: float
    direction: np.ndarray | None
    direction_sha256: str
    direction_shape: tuple[int, ...]
    source_kv_boundary_sha256: str
    source_parent_state_sha256: str
    status: str = "active"
    applied_action_step: int | None = None
    reservation_id: str = ""


def _failure_authority(
    observation: Mapping[str, Any],
    incumbent: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    candidate = validate_observation(observation)
    incumbent_row = validate_observation(incumbent) if incumbent else {}
    if candidate["authoritative"] is not True:
        return "non_authoritative_failure_observation", incumbent_row
    if candidate["basis"] == "deterministic_exact" and float(candidate["upper_bound"]) <= _EPSILON:
        return "deterministic_exact_rejection", incumbent_row
    if candidate["basis"] == "calibrated_interval" and float(candidate["upper_bound"]) <= _EPSILON:
        return "calibrated_zero_rejection", incumbent_row
    if (
        incumbent_row
        and incumbent_row["authoritative"] is True
        and float(candidate["upper_bound"]) + _EPSILON < float(incumbent_row["lower_bound"])
    ):
        return "confidence_interval_regression", incumbent_row
    return "failure_not_verified", incumbent_row


def _candidate_states(
    parent: Any,
    failed: Any,
    *,
    protected_positions: Sequence[int],
    max_relative_delta_rms: float,
    seed_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], np.ndarray]:
    parent_state = _as_state(parent, name="constraint parent")
    failed_state = _as_state(failed, name="constraint failed proposal")
    if parent_state.shape != failed_state.shape:
        raise ValueError("constraint parent/proposal shapes differ")
    protected = tuple(sorted(set(protected_positions)))
    if any(
        type(position) is not int or not 0 <= position < failed_state.shape[1]
        for position in protected
    ):
        raise ValueError("constraint protected position is invalid")
    if not is_sha256(seed_sha256):
        raise ValueError("constraint seed binding is invalid")

    failure_direction = failed_state - parent_state
    if protected:
        failure_direction[:, protected, :] = 0.0
    mutable_positions = [
        position for position in range(failed_state.shape[1]) if position not in set(protected)
    ]
    direction_rms = _rms(failure_direction[:, mutable_positions, :]) if mutable_positions else 0.0
    if not mutable_positions or direction_rms <= 1e-12:
        raise ValueError("verified failure has no writable latent direction")
    mutable_state = failed_state[:, mutable_positions, :]
    state_rms = max(_rms(mutable_state), 1e-6)
    target_delta_rms = min(
        direction_rms,
        float(max_relative_delta_rms) * state_rms,
    )
    if target_delta_rms <= 1e-12:
        raise ValueError("verified failure direction is below numeric resolution")
    negative_delta = -failure_direction * (target_delta_rms / direction_rms)

    rng = np.random.default_rng(int.from_bytes(bytes.fromhex(seed_sha256)[:8], "big", signed=False))
    random_delta = rng.standard_normal(failed_state.shape).astype(failed_state.dtype)
    if protected:
        random_delta[:, protected, :] = 0.0
    random_flat = random_delta.reshape(-1).astype(np.float64)
    guided_flat = negative_delta.reshape(-1).astype(np.float64)
    guided_energy = float(np.dot(guided_flat, guided_flat))
    random_flat -= (
        float(np.dot(random_flat, guided_flat)) / max(guided_energy, 1e-18)
    ) * guided_flat
    random_delta = random_flat.reshape(random_delta.shape).astype(failed_state.dtype)
    if protected:
        random_delta[:, protected, :] = 0.0
    random_rms = _rms(random_delta[:, mutable_positions, :])
    if random_rms <= 1e-12:
        raise ValueError("constraint sham direction degenerated")
    random_delta *= target_delta_rms / random_rms

    states = {
        "failed_no_op": np.array(failed_state, copy=True),
        "matched_orthogonal_sham": failed_state + random_delta,
        "negative_direction": failed_state + negative_delta,
    }
    baseline_sha256 = tensor_sha256(failed_state)
    summaries: dict[str, dict[str, Any]] = {}
    for name, state in states.items():
        delta = state - failed_state
        changed_positions = [
            position
            for position in range(state.shape[1])
            if not np.array_equal(
                state[:, position, :],
                failed_state[:, position, :],
            )
        ]
        delta_rms = _rms(delta[:, mutable_positions, :])
        denominator = math.sqrt(float(np.dot(delta.reshape(-1), delta.reshape(-1))) * guided_energy)
        cosine = (
            None
            if delta_rms <= 1e-12
            else round(
                float(np.dot(delta.reshape(-1), guided_flat)) / max(denominator, 1e-18),
                12,
            )
        )
        summaries[name] = {
            "name": name,
            "state_sha256": tensor_sha256(state),
            "delta_rms": round(delta_rms, 12),
            "relative_mutable_delta_rms": round(delta_rms / state_rms, 12),
            "cosine_to_negative_direction": cosine,
            "changed_positions": changed_positions,
            "protected_positions_unchanged": all(
                np.array_equal(state[:, position, :], failed_state[:, position, :])
                for position in protected
            ),
        }
    if (
        summaries["failed_no_op"]["state_sha256"] != baseline_sha256
        or summaries["matched_orthogonal_sham"]["state_sha256"] == baseline_sha256
        or summaries["negative_direction"]["state_sha256"] == baseline_sha256
        or summaries["matched_orthogonal_sham"]["state_sha256"]
        == summaries["negative_direction"]["state_sha256"]
    ):
        raise RuntimeError("constraint intervention arms are not distinct")
    if not math.isclose(
        summaries["matched_orthogonal_sham"]["delta_rms"],
        summaries["negative_direction"]["delta_rms"],
        rel_tol=1e-5,
        abs_tol=1e-8,
    ):
        raise RuntimeError("constraint sham is not magnitude matched")
    if any(
        not summary["protected_positions_unchanged"]
        or summary["relative_mutable_delta_rms"] > float(max_relative_delta_rms) + 1e-6
        for summary in summaries.values()
    ):
        raise RuntimeError("constraint intervention escaped its bound")
    return states, summaries, negative_delta


class TransientConstraintLedger:
    """Episode-local authority ledger and private one-use direction store."""

    def __init__(
        self,
        *,
        episode_id: str,
        objective_sha256: str,
        n_branches: int,
        protected_positions: Mapping[int, Sequence[int]],
        config: TransientConstraintConfig | None = None,
    ) -> None:
        if not isinstance(episode_id, str) or not episode_id or len(episode_id) > 160:
            raise ValueError("constraint episode identity is invalid")
        if not is_sha256(objective_sha256):
            raise ValueError("constraint objective commitment is invalid")
        if type(n_branches) is not int or not 1 <= n_branches <= 64:
            raise ValueError("constraint branch count is invalid")
        self.episode_id = episode_id
        self.objective_sha256 = objective_sha256
        self.n_branches = n_branches
        self.protected_positions = _protected_map(
            protected_positions,
            n_branches=n_branches,
        )
        self.config = config or TransientConstraintConfig()
        self._private: dict[str, _PrivateConstraint] = {}
        self._critic_rejections: list[dict[str, Any]] = []
        self._attempts: list[dict[str, Any]] = []
        self._applications: list[dict[str, Any]] = []
        self._reservation_rollbacks: list[dict[str, Any]] = []
        self._erasures: list[dict[str, Any]] = []

    @property
    def private_direction_count(self) -> int:
        return sum(constraint.direction is not None for constraint in self._private.values())

    def reject_critic_proposal(
        self,
        prose: str,
        *,
        branch_index: int,
        action_step: int,
    ) -> None:
        """Record that unverified prose was denied control authority."""

        if (
            not isinstance(prose, str)
            or not prose
            or len(prose) > 16_384
            or type(branch_index) is not int
            or not 0 <= branch_index < self.n_branches
        ):
            raise ValueError("critic constraint proposal is invalid")
        _bounded_step(action_step, name="critic proposal action step")
        payload = {
            "branch_index": branch_index,
            "action_step": action_step,
            "prose_sha256": hashlib.sha256(prose.encode("utf-8")).hexdigest(),
            "prose_chars": len(prose),
            "decision": "rejected",
            "reason": "critic_prose_has_no_constraint_authority",
            "constraint_created": False,
            "text_stored": False,
        }
        self._critic_rejections.append({**payload, "event_sha256": canonical_sha256(payload)})

    def _record_inactive_attempt(
        self,
        *,
        source: dict[str, Any],
        status: str,
        reason: str,
    ) -> dict[str, Any]:
        payload = {
            **source,
            "status": status,
            "reason": reason,
            "evaluation_order": [],
            "arms": [],
            "all_arms_equal_compute": False,
            "all_arms_equal_tokens": False,
            "all_arms_equal_allocated_resources": False,
            "all_arms_fully_metered": False,
            "all_observations_authoritative": False,
            "repeat_stable": False,
            "controls_repeat_failure": False,
            "guided_beats_controls": False,
            "constraint_id": "",
            "expires_after_action_step": None,
        }
        attempt = {**payload, "attempt_sha256": canonical_sha256(payload)}
        self._attempts.append(attempt)
        return attempt

    def consider_verified_failure(
        self,
        *,
        parent_state: Any,
        failed_state: Any,
        branch_index: int,
        source_action: str,
        action_step: int,
        source_kv_boundary_sha256: str,
        observation: Mapping[str, Any],
        incumbent_observation: Mapping[str, Any] | None,
        verifier_policy_sha256: str,
        verifier_preflight_sha256: str,
        evaluate: Callable[[str, Any, int], CounterfactualProbeResult] | None,
        evaluation_unavailable_reason: str = "",
        budget: Any | None = None,
    ) -> dict[str, Any]:
        """Admit one constraint only after verified failure and control trial."""

        if type(branch_index) is not int or not 0 <= branch_index < self.n_branches:
            raise ValueError("constraint branch index is invalid")
        source_action = _bounded_action(source_action)
        action_step = _bounded_step(action_step, name="constraint source action step")
        if not is_sha256(verifier_policy_sha256):
            raise ValueError("constraint verifier policy commitment is invalid")
        if not is_sha256(verifier_preflight_sha256):
            raise ValueError("constraint verifier preflight commitment is invalid")
        if not is_sha256(source_kv_boundary_sha256):
            raise ValueError("constraint source KV boundary is invalid")
        candidate_observation = validate_observation(observation)
        failure_kind, incumbent = _failure_authority(
            candidate_observation,
            incumbent_observation,
        )
        parent = _as_state(parent_state, name="constraint parent")
        failed = _as_state(failed_state, name="constraint failed proposal")
        if parent.shape != failed.shape:
            raise ValueError("constraint source shapes differ")
        source = {
            "schema": TRANSIENT_CONSTRAINT_ATTEMPT_SCHEMA,
            "ordinal": len(self._attempts),
            "episode_id": self.episode_id,
            "objective_sha256": self.objective_sha256,
            "branch_index": branch_index,
            "source_action": source_action,
            "created_action_step": action_step,
            "source_kv_boundary_sha256": source_kv_boundary_sha256,
            "parent_state_sha256": tensor_sha256(parent),
            "failed_state_sha256": tensor_sha256(failed),
            "source_observation": candidate_observation,
            "incumbent_observation": incumbent,
            "failure_kind": failure_kind,
            "protected_positions": list(self.protected_positions[branch_index]),
            "verifier_policy_sha256": verifier_policy_sha256,
            "verifier_preflight_sha256": verifier_preflight_sha256,
            "probe_cache_policy": "fresh_cache_exact_token_probe",
            "resource_parity_counters": list(CONSTRAINT_PARITY_COUNTERS),
            "direction_sha256": "",
            "direction_shape": [],
            "answer_text_stored": False,
            "critic_prose_used": False,
        }
        if failure_kind not in {
            "deterministic_exact_rejection",
            "calibrated_zero_rejection",
            "confidence_interval_regression",
        }:
            return self._record_inactive_attempt(
                source=source,
                status="rejected",
                reason=failure_kind,
            )
        if len(self._private) >= self.config.max_constraints:
            return self._record_inactive_attempt(
                source=source,
                status="skipped",
                reason="constraint_capacity_exhausted",
            )
        if evaluate is None:
            return self._record_inactive_attempt(
                source=source,
                status="skipped",
                reason=(
                    evaluation_unavailable_reason or "matched_counterfactual_evaluator_unavailable"
                ),
            )

        seed_sha256 = canonical_sha256(
            {
                "episode_id": self.episode_id,
                "objective_sha256": self.objective_sha256,
                "branch_index": branch_index,
                "source_action": source_action,
                "action_step": action_step,
                "parent_state_sha256": source["parent_state_sha256"],
                "failed_state_sha256": source["failed_state_sha256"],
                "source_observation_sha256": candidate_observation["observation_sha256"],
            }
        )
        try:
            states, summaries, private_direction = _candidate_states(
                parent,
                failed,
                protected_positions=self.protected_positions[branch_index],
                max_relative_delta_rms=self.config.max_relative_delta_rms,
                seed_sha256=seed_sha256,
            )
        except ValueError as exc:
            return self._record_inactive_attempt(
                source=source,
                status="skipped",
                reason=str(exc).replace(" ", "_"),
            )
        if budget is not None:
            elements = int(failed.size)
            budget.charge_tensor_work(
                "transient_negative_constraint_candidates",
                element_reads=7 * elements,
                element_writes=4 * elements,
                scalar_ops=10 * elements,
                host_scalar_ops=256,
            )

        order_rng = np.random.default_rng(
            int.from_bytes(bytes.fromhex(seed_sha256)[8:16], "big", signed=False)
        )
        evaluation_order: list[list[str]] = []
        results: dict[str, list[dict[str, Any]]] = {name: [] for name in ARM_NAMES}
        try:
            for replicate in range(self.config.replicates):
                order = list(ARM_NAMES)
                order_rng.shuffle(order)
                evaluation_order.append(order)
                for name in order:
                    resource_before = _resource_totals(budget)
                    normalized = evaluate(
                        name,
                        states[name],
                        replicate,
                    ).normalized()
                    resource_after = _resource_totals(budget)
                    normalized["resource_before"] = dict(resource_before or {})
                    normalized["resource_after"] = dict(resource_after or {})
                    normalized["resource_delta"] = (
                        _resource_delta(resource_before, resource_after)
                        if (resource_before is not None and resource_after is not None)
                        else {}
                    )
                    results[name].append(normalized)
        except Exception as exc:
            return self._record_inactive_attempt(
                source=source,
                status="restored",
                reason=f"evaluation_failed:{type(exc).__name__}",
            )

        arms: list[dict[str, Any]] = []
        all_authoritative = True
        repeat_stable = True
        layer_apps: list[int] = []
        token_counts: list[int] = []
        resource_deltas: list[dict[str, int]] = []
        fully_metered = True
        for name in ARM_NAMES:
            observations = [row["observation"] for row in results[name]]
            all_authoritative = all_authoritative and all(
                row["authoritative"] is True for row in observations
            )
            signatures = [_observation_signature(row) for row in observations]
            repeat_stable = repeat_stable and all(
                signature == signatures[0] for signature in signatures[1:]
            )
            layer_apps.extend(row["layer_apps"] for row in results[name])
            token_counts.extend(row["probe_token_count"] for row in results[name])
            for result in results[name]:
                resource_delta = result["resource_delta"]
                resource_deltas.append(resource_delta)
                fully_metered = fully_metered and (
                    set(resource_delta) == set(RESOURCE_COUNTERS)
                    and resource_delta["transformer_layer_apps"] == result["layer_apps"]
                    and resource_delta["output_head_tokens"] == result["probe_token_count"]
                    and resource_delta["attention_query_key_pairs"] > 0
                    and resource_delta["verifier_calls"] >= 1
                    and resource_delta["verifier_input_bytes"] > 0
                    and resource_delta["verifier_output_bytes"] > 0
                )
            arms.append({**summaries[name], "replicates": results[name]})
        all_equal_compute = bool(layer_apps) and len(set(layer_apps)) == 1
        all_equal_tokens = bool(token_counts) and len(set(token_counts)) == 1
        all_equal_resources = bool(resource_deltas) and (
            all(resource_deltas)
            and all(
                all(
                    resource_delta[name] == resource_deltas[0][name]
                    for name in CONSTRAINT_PARITY_COUNTERS
                )
                for resource_delta in resource_deltas[1:]
            )
        )
        source_upper = float(candidate_observation["upper_bound"])
        control_upper = max(
            float(row["observation"]["upper_bound"])
            for name in ("failed_no_op", "matched_orthogonal_sham")
            for row in results[name]
        )
        guided_lower = min(
            float(row["observation"]["lower_bound"]) for row in results["negative_direction"]
        )
        controls_repeat_failure = control_upper <= source_upper + _EPSILON
        guided_beats_controls = (
            all_authoritative
            and repeat_stable
            and all_equal_compute
            and all_equal_tokens
            and all_equal_resources
            and fully_metered
            and controls_repeat_failure
            and guided_lower > control_upper + float(self.config.min_verifier_margin) + _EPSILON
        )
        attempt_payload = {
            **source,
            "status": "admitted" if guided_beats_controls else "restored",
            "reason": (
                "guided_lower_bound_beats_repeating_controls"
                if guided_beats_controls
                else "non_authoritative_verifier_observation"
                if not all_authoritative
                else "verifier_repeat_instability"
                if not repeat_stable
                else "control_compute_mismatch"
                if not all_equal_compute
                else "control_token_count_mismatch"
                if not all_equal_tokens
                else "control_resource_mismatch"
                if not all_equal_resources
                else "control_resource_accounting_incomplete"
                if not fully_metered
                else "controls_did_not_repeat_verified_failure"
                if not controls_repeat_failure
                else "guided_candidate_did_not_beat_controls"
            ),
            "evaluation_order": evaluation_order,
            "arms": arms,
            "direction_sha256": tensor_sha256(private_direction),
            "direction_shape": list(private_direction.shape),
            "all_arms_equal_compute": all_equal_compute,
            "all_arms_equal_tokens": all_equal_tokens,
            "all_arms_equal_allocated_resources": all_equal_resources,
            "all_arms_fully_metered": fully_metered,
            "all_observations_authoritative": all_authoritative,
            "repeat_stable": repeat_stable,
            "controls_repeat_failure": controls_repeat_failure,
            "guided_beats_controls": guided_beats_controls,
            "constraint_id": "",
            "expires_after_action_step": (
                min(MAX_ACTION_STEP, action_step + self.config.ttl_action_steps)
                if guided_beats_controls
                else None
            ),
        }
        pre_id_payload = dict(attempt_payload)
        constraint_id = canonical_sha256(pre_id_payload) if guided_beats_controls else ""
        attempt_payload["constraint_id"] = constraint_id
        attempt = {
            **attempt_payload,
            "attempt_sha256": canonical_sha256(attempt_payload),
        }
        self._attempts.append(attempt)
        if guided_beats_controls:
            self._private[constraint_id] = _PrivateConstraint(
                constraint_id=constraint_id,
                branch_index=branch_index,
                source_action=source_action,
                created_action_step=action_step,
                expires_after_action_step=int(attempt["expires_after_action_step"]),
                source_failure_upper_bound=source_upper,
                direction=private_direction,
                direction_sha256=tensor_sha256(private_direction),
                direction_shape=tuple(private_direction.shape),
                source_kv_boundary_sha256=source_kv_boundary_sha256,
                source_parent_state_sha256=source["parent_state_sha256"],
            )
        return attempt

    def _erase_private(
        self,
        constraint: _PrivateConstraint,
        *,
        reason: str,
    ) -> None:
        direction = constraint.direction
        if direction is None:
            return
        prior_sha256 = tensor_sha256(direction)
        if prior_sha256 != constraint.direction_sha256:
            raise RuntimeError("constraint private direction changed before erasure")
        direction.fill(0.0)
        zeroized_sha256 = tensor_sha256(direction)
        all_zero = bool(np.count_nonzero(direction) == 0)
        payload = {
            "constraint_id": constraint.constraint_id,
            "reason": reason,
            "prior_direction_sha256": prior_sha256,
            "zeroized_direction_sha256": zeroized_sha256,
            "direction_shape": list(constraint.direction_shape),
            "all_zero_before_release": all_zero,
            "private_reference_released": True,
        }
        if not all_zero:
            raise RuntimeError("constraint private direction zeroization failed")
        constraint.direction = None
        self._erasures.append({**payload, "erasure_sha256": canonical_sha256(payload)})

    def _expire_before(self, *, branch_index: int, action_step: int) -> None:
        for constraint in self._private.values():
            if (
                constraint.status == "active"
                and constraint.branch_index == branch_index
                and action_step > constraint.expires_after_action_step
            ):
                constraint.status = "expired_ttl"
                self._erase_private(constraint, reason="ttl_expired")

    def abort_all(self) -> None:
        """Zeroize every private direction when an episode cannot finalize."""

        for constraint in self._private.values():
            if constraint.direction is not None:
                constraint.status = "aborted_episode_failure"
                self._erase_private(
                    constraint,
                    reason="episode_aborted",
                )

    def pending_action(
        self,
        *,
        branch_index: int,
        action_step: int,
        kv_boundary_sha256: str,
        state_sha256: str,
    ) -> str | None:
        """Return the next constraint-bound action, expiring stale authority."""

        if type(branch_index) is not int or not 0 <= branch_index < self.n_branches:
            raise ValueError("constraint pending branch index is invalid")
        action_step = _bounded_step(action_step, name="constraint pending action step")
        if not is_sha256(kv_boundary_sha256):
            raise ValueError("constraint pending KV boundary is invalid")
        if not is_sha256(state_sha256):
            raise ValueError("constraint pending state is invalid")
        self._expire_before(branch_index=branch_index, action_step=action_step)
        candidates = sorted(
            (
                constraint
                for constraint in self._private.values()
                if constraint.status == "active"
                and constraint.branch_index == branch_index
                and constraint.created_action_step < action_step
                and action_step <= constraint.expires_after_action_step
            ),
            key=lambda item: (item.created_action_step, item.constraint_id),
        )
        for stale in candidates:
            if stale.source_kv_boundary_sha256 != kv_boundary_sha256:
                stale.status = "expired_stale_kv"
                self._erase_private(stale, reason="stale_kv_boundary")
            elif stale.source_parent_state_sha256 != state_sha256:
                stale.status = "expired_stale_state"
                self._erase_private(stale, reason="stale_parent_state")
        candidate = next(
            (
                item
                for item in candidates
                if item.status == "active"
                and item.source_kv_boundary_sha256 == kv_boundary_sha256
                and item.source_parent_state_sha256 == state_sha256
            ),
            None,
        )
        return candidate.source_action if candidate is not None else None

    def apply_next(
        self,
        state: Any,
        *,
        branch_index: int,
        action: str,
        action_step: int,
        branch_step: int,
        kv_boundary_sha256: str,
        budget: Any | None = None,
    ) -> tuple[Any, dict[str, Any] | None]:
        """Reserve and apply at most one constraint; caller commits recurrence."""

        if type(branch_index) is not int or not 0 <= branch_index < self.n_branches:
            raise ValueError("constraint application branch is invalid")
        action = _bounded_action(action)
        action_step = _bounded_step(action_step, name="constraint application step")
        branch_step = _bounded_step(branch_step, name="constraint branch step")
        if not is_sha256(kv_boundary_sha256):
            raise ValueError("constraint application KV boundary is invalid")
        current = _as_state(state, name="constraint application")
        pre_sha256 = tensor_sha256(current)
        protected = self.protected_positions[branch_index]
        if any(position >= current.shape[1] for position in protected):
            raise ValueError("constraint protected position exceeds state")
        self._expire_before(branch_index=branch_index, action_step=action_step)
        candidates = sorted(
            [
                constraint
                for constraint in self._private.values()
                if (
                    constraint.status == "active"
                    and constraint.branch_index == branch_index
                    and constraint.source_action == action
                    and constraint.created_action_step < action_step
                    and action_step <= constraint.expires_after_action_step
                )
            ],
            key=lambda item: item.created_action_step,
        )
        for stale in candidates:
            if stale.source_kv_boundary_sha256 != kv_boundary_sha256:
                stale.status = "expired_stale_kv"
                self._erase_private(stale, reason="stale_kv_boundary")
            elif stale.source_parent_state_sha256 != pre_sha256:
                stale.status = "expired_stale_state"
                self._erase_private(stale, reason="stale_parent_state")
        constraint = next(
            (
                candidate
                for candidate in candidates
                if candidate.source_kv_boundary_sha256 == kv_boundary_sha256
                and candidate.source_parent_state_sha256 == pre_sha256
            ),
            None,
        )
        if constraint is None:
            return state, None
        if constraint.direction is None:
            raise RuntimeError("active constraint has no private direction")
        direction = np.array(constraint.direction, copy=True)
        if direction.shape != current.shape:
            raise ValueError("constraint direction shape changed")
        if protected:
            direction[:, protected, :] = 0.0
        mutable_positions = [
            position for position in range(current.shape[1]) if position not in set(protected)
        ]
        mutable_rms = max(_rms(current[:, mutable_positions, :]), 1e-6)
        direction_rms = _rms(direction[:, mutable_positions, :])
        max_delta_rms = float(self.config.max_relative_delta_rms) * mutable_rms
        if direction_rms > max_delta_rms:
            direction *= max_delta_rms / max(direction_rms, 1e-18)
        output = current + direction
        if protected:
            output[:, protected, :] = current[:, protected, :]
        post_sha256 = tensor_sha256(output)
        if pre_sha256 == post_sha256 or not np.all(np.isfinite(output)):
            raise RuntimeError("transient constraint did not produce a finite causal change")
        if budget is not None:
            elements = int(current.size)
            budget.charge_tensor_work(
                "transient_negative_constraint_apply",
                element_reads=3 * elements,
                element_writes=2 * elements,
                scalar_ops=3 * elements,
                host_scalar_ops=64,
            )
        reservation_id = canonical_sha256(
            {
                "constraint_id": constraint.constraint_id,
                "branch_index": branch_index,
                "action": action,
                "action_step": action_step,
                "branch_step": branch_step,
                "kv_boundary_sha256": kv_boundary_sha256,
                "pre_state_sha256": pre_sha256,
                "post_state_sha256": post_sha256,
            }
        )
        constraint.status = "reserved"
        constraint.reservation_id = reservation_id
        payload = {
            "schema": TRANSIENT_CONSTRAINT_APPLICATION_SCHEMA,
            "ordinal": len(self._applications),
            "reservation_id": reservation_id,
            "constraint_id": constraint.constraint_id,
            "branch_index": branch_index,
            "source_action": constraint.source_action,
            "applied_action": action,
            "created_action_step": constraint.created_action_step,
            "applied_action_step": action_step,
            "expires_after_action_step": constraint.expires_after_action_step,
            "branch_step_before": branch_step,
            "branch_step_after": None,
            "kv_boundary_before_sha256": kv_boundary_sha256,
            "kv_boundary_after_sha256": "",
            "pre_state_sha256": pre_sha256,
            "post_state_sha256": post_sha256,
            "post_recurrence_state_sha256": "",
            "delta_rms": round(
                _rms((output - current)[:, mutable_positions, :]),
                12,
            ),
            "relative_mutable_delta_rms": round(
                _rms((output - current)[:, mutable_positions, :]) / mutable_rms,
                12,
            ),
            "protected_positions": list(protected),
            "protected_positions_unchanged": all(
                np.array_equal(output[:, position, :], current[:, position, :])
                for position in protected
            ),
            "recurrence_committed": False,
            "one_use_consumed": False,
            "followup_observation": {},
            "outcome": "awaiting_verification",
            "failure_reduced": False,
            "failure_repeated": False,
            "answer_text_stored": False,
        }
        application = {
            **payload,
            "application_sha256": canonical_sha256(payload),
        }
        self._applications.append(application)
        return output, application

    def commit_application(
        self,
        *,
        reservation_id: str,
        branch_step_after: int,
        kv_boundary_after_sha256: str,
        recurrence_state: Any,
    ) -> dict[str, Any]:
        """Consume a reserved constraint only after recurrence committed."""

        if not is_sha256(reservation_id):
            raise ValueError("constraint reservation identity is invalid")
        branch_step_after = _bounded_step(
            branch_step_after,
            name="constraint committed branch step",
        )
        if not is_sha256(kv_boundary_after_sha256):
            raise ValueError("constraint committed KV boundary is invalid")
        application = next(
            (row for row in self._applications if row["reservation_id"] == reservation_id),
            None,
        )
        if application is None or application["recurrence_committed"] is not False:
            raise ValueError("constraint reservation is absent or closed")
        constraint = self._private[application["constraint_id"]]
        if (
            constraint.status != "reserved"
            or constraint.reservation_id != reservation_id
            or branch_step_after != application["branch_step_before"] + 1
            or kv_boundary_after_sha256 != application["kv_boundary_before_sha256"]
        ):
            raise ValueError("constraint recurrence commitment differs")
        recurrence_sha256 = tensor_sha256(
            _as_state(recurrence_state, name="constraint recurrence result")
        )
        payload = dict(application)
        payload.pop("application_sha256")
        payload.update(
            {
                "branch_step_after": branch_step_after,
                "kv_boundary_after_sha256": kv_boundary_after_sha256,
                "post_recurrence_state_sha256": recurrence_sha256,
                "recurrence_committed": True,
                "one_use_consumed": True,
            }
        )
        committed = {
            **payload,
            "application_sha256": canonical_sha256(payload),
        }
        self._applications[self._applications.index(application)] = committed
        constraint.status = "consumed"
        constraint.applied_action_step = application["applied_action_step"]
        constraint.reservation_id = ""
        self._erase_private(constraint, reason="one_use_consumed")
        return committed

    def rollback_application(
        self,
        *,
        reservation_id: str,
        restored_state: Any,
        branch_step_after: int,
        kv_boundary_after_sha256: str,
        reason: str,
    ) -> dict[str, Any]:
        """Restore a refused/failed recurrence without consuming authority."""

        if not is_sha256(reservation_id):
            raise ValueError("constraint rollback reservation is invalid")
        if reason not in {
            "budget_refused",
            "recurrence_failed",
            "cancelled",
        }:
            raise ValueError("constraint rollback reason is invalid")
        branch_step_after = _bounded_step(
            branch_step_after,
            name="constraint rollback branch step",
        )
        if not is_sha256(kv_boundary_after_sha256):
            raise ValueError("constraint rollback KV boundary is invalid")
        application = next(
            (row for row in self._applications if row["reservation_id"] == reservation_id),
            None,
        )
        if application is None or application["recurrence_committed"] is not False:
            raise ValueError("constraint rollback reservation is absent or closed")
        restored_sha256 = tensor_sha256(_as_state(restored_state, name="constraint rollback state"))
        constraint = self._private[application["constraint_id"]]
        if (
            constraint.status != "reserved"
            or constraint.reservation_id != reservation_id
            or restored_sha256 != application["pre_state_sha256"]
            or branch_step_after != application["branch_step_before"]
            or kv_boundary_after_sha256 != application["kv_boundary_before_sha256"]
        ):
            raise RuntimeError("constraint rollback did not restore exact parent")
        self._applications.remove(application)
        constraint.status = "active"
        constraint.reservation_id = ""
        payload = {
            "ordinal": len(self._reservation_rollbacks),
            "reservation_id": reservation_id,
            "constraint_id": constraint.constraint_id,
            "branch_index": constraint.branch_index,
            "action_step": application["applied_action_step"],
            "branch_step": branch_step_after,
            "kv_boundary_sha256": kv_boundary_after_sha256,
            "pre_state_sha256": application["pre_state_sha256"],
            "reserved_state_sha256": application["post_state_sha256"],
            "restored_state_sha256": restored_sha256,
            "reason": reason,
            "authority_consumed": False,
        }
        rollback = {**payload, "rollback_sha256": canonical_sha256(payload)}
        self._reservation_rollbacks.append(rollback)
        return rollback

    def observe_followup(
        self,
        *,
        branch_index: int,
        action_step: int,
        observation: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Bind the next matching verified result to its consumed constraint."""

        normalized = validate_observation(observation)
        application = next(
            (
                row
                for row in reversed(self._applications)
                if (
                    row["branch_index"] == branch_index
                    and row["applied_action_step"] == action_step
                    and row["recurrence_committed"] is True
                    and row["outcome"] == "awaiting_verification"
                )
            ),
            None,
        )
        if application is None:
            return None
        constraint = self._private[application["constraint_id"]]
        reduced = (
            normalized["authoritative"] is True
            and float(normalized["lower_bound"])
            > constraint.source_failure_upper_bound
            + float(self.config.min_verifier_margin)
            + _EPSILON
        )
        repeated = (
            normalized["authoritative"] is True
            and float(normalized["upper_bound"]) <= constraint.source_failure_upper_bound + _EPSILON
        )
        payload = dict(application)
        payload.pop("application_sha256")
        payload.update(
            {
                "followup_observation": normalized,
                "outcome": (
                    "verified_failure_reduced"
                    if reduced
                    else "verified_failure_repeated"
                    if repeated
                    else "followup_inconclusive"
                ),
                "failure_reduced": reduced,
                "failure_repeated": repeated,
            }
        )
        updated = {**payload, "application_sha256": canonical_sha256(payload)}
        self._applications[self._applications.index(application)] = updated
        return updated

    def finalize(self, *, final_action_step: int) -> dict[str, Any]:
        final_action_step = _bounded_step(
            final_action_step,
            name="constraint final action step",
        )
        for constraint in self._private.values():
            if constraint.status == "reserved":
                raise RuntimeError("cannot finalize transient constraints with an open reservation")
            if constraint.status == "active":
                constraint.status = "expired_episode_end"
                self._erase_private(
                    constraint,
                    reason="episode_ended",
                )
        constraints = [
            {
                "constraint_id": constraint.constraint_id,
                "branch_index": constraint.branch_index,
                "source_action": constraint.source_action,
                "created_action_step": constraint.created_action_step,
                "expires_after_action_step": constraint.expires_after_action_step,
                "source_failure_upper_bound": round(
                    constraint.source_failure_upper_bound,
                    10,
                ),
                "source_kv_boundary_sha256": (constraint.source_kv_boundary_sha256),
                "direction_sha256": constraint.direction_sha256,
                "direction_shape": list(constraint.direction_shape),
                "status": constraint.status,
                "applied_action_step": constraint.applied_action_step,
                "max_uses": 1,
                "private_direction_erased": constraint.direction is None,
            }
            for constraint in sorted(
                self._private.values(),
                key=lambda item: item.created_action_step,
            )
        ]
        payload = {
            "schema": TRANSIENT_CONSTRAINT_SCHEMA,
            "config": self.config.to_dict(),
            "episode_id": self.episode_id,
            "objective_sha256": self.objective_sha256,
            "n_branches": self.n_branches,
            "protected_positions": {
                str(index): list(positions) for index, positions in self.protected_positions.items()
            },
            "final_action_step": final_action_step,
            "critic_rejections": list(self._critic_rejections),
            "attempts": list(self._attempts),
            "constraints": constraints,
            "applications": list(self._applications),
            "reservation_rollbacks": list(self._reservation_rollbacks),
            "erasures": list(self._erasures),
            "aggregates": {
                "critic_rejection_count": len(self._critic_rejections),
                "attempt_count": len(self._attempts),
                "admitted_count": len(constraints),
                "application_count": len(self._applications),
                "reservation_rollback_count": len(self._reservation_rollbacks),
                "erasure_count": len(self._erasures),
                "verified_reduction_count": sum(
                    row["failure_reduced"] for row in self._applications
                ),
                "verified_repeat_count": sum(row["failure_repeated"] for row in self._applications),
                "active_after_episode": 0,
                "private_directions_after_episode": (self.private_direction_count),
            },
            "authority_scope": "episode_objective_branch_action_one_use",
            "critic_prose_authority": False,
            "answer_text_stored": False,
        }
        receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
        return validate_transient_constraint_receipt(
            receipt,
            episode_id=self.episode_id,
            objective_sha256=self.objective_sha256,
            n_branches=self.n_branches,
            protected_positions=self.protected_positions,
            expected_config=self.config,
        )


def build_empty_transient_constraint_receipt(
    *,
    episode_id: str,
    objective_sha256: str,
    n_branches: int,
    protected_positions: Mapping[int, Sequence[int]],
    config: TransientConstraintConfig | None = None,
) -> dict[str, Any]:
    return TransientConstraintLedger(
        episode_id=episode_id,
        objective_sha256=objective_sha256,
        n_branches=n_branches,
        protected_positions=protected_positions,
        config=config,
    ).finalize(final_action_step=0)


def _validate_attempt(
    value: Any,
    *,
    ordinal: int,
    episode_id: str,
    objective_sha256: str,
    n_branches: int,
    protected_positions: Mapping[int, tuple[int, ...]],
    config: TransientConstraintConfig,
) -> dict[str, Any]:
    fields = {
        "schema",
        "ordinal",
        "episode_id",
        "objective_sha256",
        "branch_index",
        "source_action",
        "created_action_step",
        "source_kv_boundary_sha256",
        "parent_state_sha256",
        "failed_state_sha256",
        "source_observation",
        "incumbent_observation",
        "failure_kind",
        "protected_positions",
        "verifier_policy_sha256",
        "verifier_preflight_sha256",
        "probe_cache_policy",
        "resource_parity_counters",
        "direction_sha256",
        "direction_shape",
        "answer_text_stored",
        "critic_prose_used",
        "status",
        "reason",
        "evaluation_order",
        "arms",
        "all_arms_equal_compute",
        "all_arms_equal_tokens",
        "all_arms_equal_allocated_resources",
        "all_arms_fully_metered",
        "all_observations_authoritative",
        "repeat_stable",
        "controls_repeat_failure",
        "guided_beats_controls",
        "constraint_id",
        "expires_after_action_step",
        "attempt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("transient constraint attempt fields differ")
    row = dict(value)
    payload = {key: row[key] for key in fields - {"attempt_sha256"}}
    branch_index = row["branch_index"]
    if (
        row["schema"] != TRANSIENT_CONSTRAINT_ATTEMPT_SCHEMA
        or row["ordinal"] != ordinal
        or row["episode_id"] != episode_id
        or row["objective_sha256"] != objective_sha256
        or type(branch_index) is not int
        or not 0 <= branch_index < n_branches
        or row["protected_positions"] != list(protected_positions[branch_index])
        or not is_sha256(row["parent_state_sha256"])
        or not is_sha256(row["failed_state_sha256"])
        or not is_sha256(row["source_kv_boundary_sha256"])
        or not is_sha256(row["verifier_policy_sha256"])
        or not is_sha256(row["verifier_preflight_sha256"])
        or row["probe_cache_policy"] != "fresh_cache_exact_token_probe"
        or row["resource_parity_counters"] != list(CONSTRAINT_PARITY_COUNTERS)
        or row["answer_text_stored"] is not False
        or row["critic_prose_used"] is not False
        or row["attempt_sha256"] != canonical_sha256(payload)
    ):
        raise ValueError("transient constraint attempt identity is invalid")
    _bounded_action(row["source_action"])
    created_step = _bounded_step(
        row["created_action_step"],
        name="constraint attempt action step",
    )
    source_observation = validate_observation(row["source_observation"])
    incumbent = (
        validate_observation(row["incumbent_observation"]) if row["incumbent_observation"] else {}
    )
    failure_kind, expected_incumbent = _failure_authority(
        source_observation,
        incumbent or None,
    )
    if row["failure_kind"] != failure_kind or incumbent != expected_incumbent:
        raise ValueError("transient constraint failure authority differs")

    if not row["arms"]:
        if (
            row["status"] not in {"rejected", "skipped", "restored"}
            or row["evaluation_order"]
            or row["all_arms_equal_compute"] is not False
            or row["all_arms_equal_tokens"] is not False
            or row["all_arms_equal_allocated_resources"] is not False
            or row["all_arms_fully_metered"] is not False
            or row["all_observations_authoritative"] is not False
            or row["repeat_stable"] is not False
            or row["controls_repeat_failure"] is not False
            or row["guided_beats_controls"] is not False
            or row["direction_sha256"]
            or row["direction_shape"]
            or row["constraint_id"]
            or row["expires_after_action_step"] is not None
        ):
            raise ValueError("inactive transient constraint claims authority")
        return row

    if (
        row["status"] not in {"admitted", "restored"}
        or len(row["arms"]) != len(ARM_NAMES)
        or len(row["evaluation_order"]) != config.replicates
        or any(sorted(order) != sorted(ARM_NAMES) for order in row["evaluation_order"])
    ):
        raise ValueError("transient constraint trial structure is invalid")
    arms = {arm.get("name"): arm for arm in row["arms"] if isinstance(arm, Mapping)}
    if set(arms) != set(ARM_NAMES):
        raise ValueError("transient constraint control arms are incomplete")
    layer_apps: list[int] = []
    token_counts: list[int] = []
    resource_deltas: list[dict[str, int]] = []
    fully_metered = True
    all_authoritative = True
    repeat_stable = True
    observations: dict[str, list[dict[str, Any]]] = {}
    for name, arm in arms.items():
        required = {
            "name",
            "state_sha256",
            "delta_rms",
            "relative_mutable_delta_rms",
            "cosine_to_negative_direction",
            "changed_positions",
            "protected_positions_unchanged",
            "replicates",
        }
        if (
            set(arm) != required
            or not is_sha256(arm["state_sha256"])
            or not _finite(arm["delta_rms"])
            or float(arm["delta_rms"]) < 0.0
            or not _finite(arm["relative_mutable_delta_rms"])
            or float(arm["relative_mutable_delta_rms"]) < 0.0
            or float(arm["relative_mutable_delta_rms"])
            > float(config.max_relative_delta_rms) + 1e-6
            or arm["protected_positions_unchanged"] is not True
            or len(arm["replicates"]) != config.replicates
            or not isinstance(arm["changed_positions"], list)
            or any(
                position in set(protected_positions[branch_index])
                for position in arm["changed_positions"]
            )
        ):
            raise ValueError("transient constraint arm geometry is invalid")
        normalized: list[dict[str, Any]] = []
        for replicate in arm["replicates"]:
            if not isinstance(replicate, Mapping) or set(replicate) != {
                "probe_tokens_sha256",
                "probe_token_count",
                "observation",
                "layer_apps",
                "resource_before",
                "resource_after",
                "resource_delta",
            }:
                raise ValueError("transient constraint replicate fields differ")
            observation = validate_observation(replicate["observation"])
            if (
                not is_sha256(replicate["probe_tokens_sha256"])
                or type(replicate["probe_token_count"]) is not int
                or replicate["probe_token_count"] <= 0
                or type(replicate["layer_apps"]) is not int
                or replicate["layer_apps"] <= 0
                or not isinstance(replicate["resource_before"], Mapping)
                or set(replicate["resource_before"]) != set(RESOURCE_COUNTERS)
                or not isinstance(replicate["resource_after"], Mapping)
                or set(replicate["resource_after"]) != set(RESOURCE_COUNTERS)
                or not isinstance(replicate["resource_delta"], Mapping)
                or set(replicate["resource_delta"]) != set(RESOURCE_COUNTERS)
                or any(
                    type(counter) is not int or counter < 0
                    for counters in (
                        replicate["resource_before"],
                        replicate["resource_after"],
                        replicate["resource_delta"],
                    )
                    for counter in counters.values()
                )
                or any(
                    replicate["resource_after"][counter] - replicate["resource_before"][counter]
                    != replicate["resource_delta"][counter]
                    for counter in RESOURCE_COUNTERS
                )
            ):
                raise ValueError("transient constraint replicate is invalid")
            normalized.append(observation)
            layer_apps.append(replicate["layer_apps"])
            token_counts.append(replicate["probe_token_count"])
            resource_deltas.append(dict(replicate["resource_delta"]))
            fully_metered = fully_metered and (
                replicate["resource_delta"]["transformer_layer_apps"] == replicate["layer_apps"]
                and replicate["resource_delta"]["output_head_tokens"]
                == replicate["probe_token_count"]
                and replicate["resource_delta"]["attention_query_key_pairs"] > 0
                and replicate["resource_delta"]["verifier_calls"] >= 1
                and replicate["resource_delta"]["verifier_input_bytes"] > 0
                and replicate["resource_delta"]["verifier_output_bytes"] > 0
            )
        observations[name] = normalized
        all_authoritative = all_authoritative and all(
            item["authoritative"] is True for item in normalized
        )
        signatures = [_observation_signature(item) for item in normalized]
        repeat_stable = repeat_stable and all(
            signature == signatures[0] for signature in signatures[1:]
        )
    if (
        arms["failed_no_op"]["state_sha256"] != row["failed_state_sha256"]
        or float(arms["failed_no_op"]["delta_rms"]) != 0.0
        or arms["failed_no_op"]["changed_positions"]
        or arms["failed_no_op"]["cosine_to_negative_direction"] is not None
        or arms["matched_orthogonal_sham"]["state_sha256"]
        in {
            row["failed_state_sha256"],
            arms["negative_direction"]["state_sha256"],
        }
        or arms["negative_direction"]["state_sha256"] == row["failed_state_sha256"]
        or not math.isclose(
            float(arms["matched_orthogonal_sham"]["delta_rms"]),
            float(arms["negative_direction"]["delta_rms"]),
            rel_tol=1e-5,
            abs_tol=1e-8,
        )
        or not _finite(arms["matched_orthogonal_sham"]["cosine_to_negative_direction"])
        or abs(float(arms["matched_orthogonal_sham"]["cosine_to_negative_direction"])) > 1e-5
        or not _finite(arms["negative_direction"]["cosine_to_negative_direction"])
        or float(arms["negative_direction"]["cosine_to_negative_direction"]) < 0.999
    ):
        raise ValueError("transient constraint arm controls are invalid")
    all_equal_compute = bool(layer_apps) and len(set(layer_apps)) == 1
    all_equal_tokens = bool(token_counts) and len(set(token_counts)) == 1
    all_equal_resources = bool(resource_deltas) and all(
        all(resource_delta[name] == resource_deltas[0][name] for name in CONSTRAINT_PARITY_COUNTERS)
        for resource_delta in resource_deltas[1:]
    )
    ordered_resource_rows = [
        arms[name]["replicates"][replicate]
        for replicate, order in enumerate(row["evaluation_order"])
        for name in order
    ]
    if any(
        current["resource_before"] != previous["resource_after"]
        for previous, current in zip(
            ordered_resource_rows,
            ordered_resource_rows[1:],
            strict=False,
        )
    ):
        raise ValueError("transient constraint resource window is discontinuous")
    source_upper = float(source_observation["upper_bound"])
    control_upper = max(
        float(item["upper_bound"])
        for name in ("failed_no_op", "matched_orthogonal_sham")
        for item in observations[name]
    )
    guided_lower = min(float(item["lower_bound"]) for item in observations["negative_direction"])
    controls_repeat = control_upper <= source_upper + _EPSILON
    guided_beats = (
        all_authoritative
        and repeat_stable
        and all_equal_compute
        and all_equal_tokens
        and all_equal_resources
        and fully_metered
        and controls_repeat
        and guided_lower > control_upper + float(config.min_verifier_margin) + _EPSILON
    )
    if (
        row["all_arms_equal_compute"] is not all_equal_compute
        or row["all_arms_equal_tokens"] is not all_equal_tokens
        or row["all_arms_equal_allocated_resources"] is not all_equal_resources
        or row["all_arms_fully_metered"] is not fully_metered
        or row["all_observations_authoritative"] is not all_authoritative
        or row["repeat_stable"] is not repeat_stable
        or row["controls_repeat_failure"] is not controls_repeat
        or row["guided_beats_controls"] is not guided_beats
        or (row["status"] == "admitted") is not guided_beats
        or (
            guided_beats
            and (
                not is_sha256(row["direction_sha256"])
                or not isinstance(row["direction_shape"], list)
                or len(row["direction_shape"]) != 3
                or row["direction_shape"][0] != 1
                or any(type(size) is not int or size <= 0 for size in row["direction_shape"])
                or any(
                    position >= row["direction_shape"][1]
                    for position in protected_positions[branch_index]
                )
            )
        )
    ):
        raise ValueError("transient constraint trial decision differs")
    expected_expiry = (
        min(MAX_ACTION_STEP, created_step + config.ttl_action_steps) if guided_beats else None
    )
    pre_id_payload = dict(payload)
    pre_id_payload["constraint_id"] = ""
    expected_id = canonical_sha256(pre_id_payload) if guided_beats else ""
    if row["constraint_id"] != expected_id or row["expires_after_action_step"] != expected_expiry:
        raise ValueError("transient constraint authority identity differs")
    return row


def validate_transient_constraint_receipt(
    value: Any,
    *,
    episode_id: str,
    objective_sha256: str,
    n_branches: int,
    protected_positions: Mapping[int, Sequence[int]],
    expected_config: TransientConstraintConfig | None = None,
    cognitive_action_trace: Sequence[Mapping[str, Any]] | None = None,
    verifier_preflight: Mapping[str, Any] | None = None,
    information_accounting: Mapping[str, Any] | None = None,
    resource_accounting: Mapping[str, Any] | None = None,
    kv_state_tree: Mapping[str, Any] | None = None,
    verified_best_state: Mapping[str, Any] | None = None,
    loop_stability: Mapping[str, Any] | None = None,
    require_verified_best_binding: bool = False,
    require_external_bindings: bool = False,
) -> dict[str, Any]:
    """Independently reconstruct transient authority, lifetime, and outcomes."""

    fields = {
        "schema",
        "config",
        "episode_id",
        "objective_sha256",
        "n_branches",
        "protected_positions",
        "final_action_step",
        "critic_rejections",
        "attempts",
        "constraints",
        "applications",
        "reservation_rollbacks",
        "erasures",
        "aggregates",
        "authority_scope",
        "critic_prose_authority",
        "answer_text_stored",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("transient constraint receipt fields differ")
    receipt = dict(value)
    payload = {key: receipt[key] for key in fields - {"receipt_sha256"}}
    config = expected_config or TransientConstraintConfig()
    protected = _protected_map(protected_positions, n_branches=n_branches)
    expected_protected = {str(index): list(positions) for index, positions in protected.items()}
    if (
        receipt["schema"] != TRANSIENT_CONSTRAINT_SCHEMA
        or receipt["config"] != config.to_dict()
        or receipt["episode_id"] != episode_id
        or receipt["objective_sha256"] != objective_sha256
        or receipt["n_branches"] != n_branches
        or receipt["protected_positions"] != expected_protected
        or receipt["authority_scope"] != "episode_objective_branch_action_one_use"
        or receipt["critic_prose_authority"] is not False
        or receipt["answer_text_stored"] is not False
        or receipt["receipt_sha256"] != canonical_sha256(payload)
    ):
        raise ValueError("transient constraint receipt identity is invalid")
    final_step = _bounded_step(
        receipt["final_action_step"],
        name="constraint receipt final action step",
    )

    for rejection in receipt["critic_rejections"]:
        if not isinstance(rejection, Mapping):
            raise ValueError("critic constraint rejection is invalid")
        fields = {
            "branch_index",
            "action_step",
            "prose_sha256",
            "prose_chars",
            "decision",
            "reason",
            "constraint_created",
            "text_stored",
            "event_sha256",
        }
        if set(rejection) != fields:
            raise ValueError("critic constraint rejection fields differ")
        rejection_payload = {key: rejection[key] for key in fields - {"event_sha256"}}
        if (
            type(rejection["branch_index"]) is not int
            or not 0 <= rejection["branch_index"] < n_branches
            or not is_sha256(rejection["prose_sha256"])
            or type(rejection["prose_chars"]) is not int
            or not 1 <= rejection["prose_chars"] <= 16_384
            or rejection["decision"] != "rejected"
            or rejection["reason"] != "critic_prose_has_no_constraint_authority"
            or rejection["constraint_created"] is not False
            or rejection["text_stored"] is not False
            or rejection["event_sha256"] != canonical_sha256(rejection_payload)
        ):
            raise ValueError("critic constraint rejection is invalid")
        _bounded_step(rejection["action_step"], name="critic rejection step")

    attempts = [
        _validate_attempt(
            row,
            ordinal=ordinal,
            episode_id=episode_id,
            objective_sha256=objective_sha256,
            n_branches=n_branches,
            protected_positions=protected,
            config=config,
        )
        for ordinal, row in enumerate(receipt["attempts"])
    ]
    admitted = {row["constraint_id"]: row for row in attempts if row["status"] == "admitted"}
    if len(admitted) > config.max_constraints:
        raise ValueError("transient constraint receipt exceeds capacity")
    constraints = receipt["constraints"]
    if not isinstance(constraints, list) or len(constraints) != len(admitted):
        raise ValueError("transient constraint inventory differs")
    constraint_rows: dict[str, dict[str, Any]] = {}
    for row in constraints:
        fields = {
            "constraint_id",
            "branch_index",
            "source_action",
            "created_action_step",
            "expires_after_action_step",
            "source_failure_upper_bound",
            "source_kv_boundary_sha256",
            "direction_sha256",
            "direction_shape",
            "status",
            "applied_action_step",
            "max_uses",
            "private_direction_erased",
        }
        if not isinstance(row, Mapping) or set(row) != fields:
            raise ValueError("transient constraint inventory fields differ")
        source = admitted.get(row["constraint_id"])
        if (
            source is None
            or row["constraint_id"] in constraint_rows
            or row["branch_index"] != source["branch_index"]
            or row["source_action"] != source["source_action"]
            or row["created_action_step"] != source["created_action_step"]
            or row["expires_after_action_step"] != source["expires_after_action_step"]
            or row["source_failure_upper_bound"] != source["source_observation"]["upper_bound"]
            or row["source_kv_boundary_sha256"] != source["source_kv_boundary_sha256"]
            or row["direction_sha256"] != source["direction_sha256"]
            or row["direction_shape"] != source["direction_shape"]
            or not isinstance(row["direction_shape"], list)
            or len(row["direction_shape"]) != 3
            or row["direction_shape"][0] != 1
            or any(type(size) is not int or size <= 0 for size in row["direction_shape"])
            or row["status"]
            not in {
                "consumed",
                "expired_ttl",
                "expired_episode_end",
                "expired_stale_kv",
                "expired_stale_state",
                "aborted_episode_failure",
            }
            or row["max_uses"] != 1
            or row["private_direction_erased"] is not True
        ):
            raise ValueError("transient constraint inventory is invalid")
        constraint_rows[row["constraint_id"]] = dict(row)

    applications = receipt["applications"]
    if not isinstance(applications, list):
        raise ValueError("transient constraint applications must be a list")
    used_ids: set[str] = set()
    for ordinal, row in enumerate(applications):
        fields = {
            "schema",
            "ordinal",
            "reservation_id",
            "constraint_id",
            "branch_index",
            "source_action",
            "applied_action",
            "created_action_step",
            "applied_action_step",
            "expires_after_action_step",
            "branch_step_before",
            "branch_step_after",
            "kv_boundary_before_sha256",
            "kv_boundary_after_sha256",
            "pre_state_sha256",
            "post_state_sha256",
            "post_recurrence_state_sha256",
            "delta_rms",
            "relative_mutable_delta_rms",
            "protected_positions",
            "protected_positions_unchanged",
            "recurrence_committed",
            "one_use_consumed",
            "followup_observation",
            "outcome",
            "failure_reduced",
            "failure_repeated",
            "answer_text_stored",
            "application_sha256",
        }
        if not isinstance(row, Mapping) or set(row) != fields:
            raise ValueError("transient constraint application fields differ")
        source = admitted.get(row["constraint_id"])
        inventory = constraint_rows.get(row["constraint_id"])
        application_payload = {key: row[key] for key in fields - {"application_sha256"}}
        if (
            source is None
            or inventory is None
            or row["constraint_id"] in used_ids
            or row["schema"] != TRANSIENT_CONSTRAINT_APPLICATION_SCHEMA
            or row["ordinal"] != ordinal
            or not is_sha256(row["reservation_id"])
            or row["branch_index"] != source["branch_index"]
            or row["source_action"] != source["source_action"]
            or row["applied_action"] != source["source_action"]
            or row["created_action_step"] != source["created_action_step"]
            or row["expires_after_action_step"] != source["expires_after_action_step"]
            or not source["created_action_step"]
            < row["applied_action_step"]
            <= source["expires_after_action_step"]
            or row["protected_positions"] != list(protected[source["branch_index"]])
            or row["protected_positions_unchanged"] is not True
            or row["recurrence_committed"] is not True
            or row["one_use_consumed"] is not True
            or type(row["branch_step_before"]) is not int
            or row["branch_step_after"] != row["branch_step_before"] + 1
            or row["kv_boundary_before_sha256"] != source["source_kv_boundary_sha256"]
            or row["kv_boundary_after_sha256"] != row["kv_boundary_before_sha256"]
            or row["pre_state_sha256"] != source["parent_state_sha256"]
            or not is_sha256(row["post_recurrence_state_sha256"])
            or not is_sha256(row["pre_state_sha256"])
            or not is_sha256(row["post_state_sha256"])
            or row["pre_state_sha256"] == row["post_state_sha256"]
            or not _finite(row["delta_rms"])
            or float(row["delta_rms"]) <= 0.0
            or not _finite(row["relative_mutable_delta_rms"])
            or not 0.0
            < float(row["relative_mutable_delta_rms"])
            <= float(config.max_relative_delta_rms) + 1e-6
            or row["answer_text_stored"] is not False
            or row["application_sha256"] != canonical_sha256(application_payload)
            or inventory["status"] != "consumed"
            or inventory["applied_action_step"] != row["applied_action_step"]
        ):
            raise ValueError("transient constraint application is invalid")
        used_ids.add(row["constraint_id"])
        if row["followup_observation"]:
            observation = validate_observation(row["followup_observation"])
            source_upper = float(source["source_observation"]["upper_bound"])
            reduced = (
                observation["authoritative"] is True
                and float(observation["lower_bound"])
                > source_upper + float(config.min_verifier_margin) + _EPSILON
            )
            repeated = (
                observation["authoritative"] is True
                and float(observation["upper_bound"]) <= source_upper + _EPSILON
            )
            expected_outcome = (
                "verified_failure_reduced"
                if reduced
                else "verified_failure_repeated"
                if repeated
                else "followup_inconclusive"
            )
        else:
            reduced = False
            repeated = False
            expected_outcome = "awaiting_verification"
        if (
            row["failure_reduced"] is not reduced
            or row["failure_repeated"] is not repeated
            or row["outcome"] != expected_outcome
        ):
            raise ValueError("transient constraint followup decision differs")

    rollbacks = receipt["reservation_rollbacks"]
    if not isinstance(rollbacks, list):
        raise ValueError("constraint reservation rollbacks must be a list")
    for ordinal, row in enumerate(rollbacks):
        fields = {
            "ordinal",
            "reservation_id",
            "constraint_id",
            "branch_index",
            "action_step",
            "branch_step",
            "kv_boundary_sha256",
            "pre_state_sha256",
            "reserved_state_sha256",
            "restored_state_sha256",
            "reason",
            "authority_consumed",
            "rollback_sha256",
        }
        if not isinstance(row, Mapping) or set(row) != fields:
            raise ValueError("constraint reservation rollback fields differ")
        source = admitted.get(row["constraint_id"])
        rollback_payload = {key: row[key] for key in fields - {"rollback_sha256"}}
        if (
            source is None
            or row["ordinal"] != ordinal
            or not is_sha256(row["reservation_id"])
            or row["branch_index"] != source["branch_index"]
            or row["action_step"] <= source["created_action_step"]
            or row["action_step"] > source["expires_after_action_step"]
            or type(row["branch_step"]) is not int
            or row["branch_step"] < 0
            or row["kv_boundary_sha256"] != source["source_kv_boundary_sha256"]
            or not is_sha256(row["pre_state_sha256"])
            or not is_sha256(row["reserved_state_sha256"])
            or row["restored_state_sha256"] != row["pre_state_sha256"]
            or row["reserved_state_sha256"] == row["pre_state_sha256"]
            or row["reason"] not in {"budget_refused", "recurrence_failed", "cancelled"}
            or row["authority_consumed"] is not False
            or row["rollback_sha256"] != canonical_sha256(rollback_payload)
        ):
            raise ValueError("constraint reservation rollback is invalid")

    erasures = receipt["erasures"]
    if not isinstance(erasures, list) or len(erasures) != len(admitted):
        raise ValueError("constraint private erasure inventory differs")
    erased_ids: set[str] = set()
    for row in erasures:
        fields = {
            "constraint_id",
            "reason",
            "prior_direction_sha256",
            "zeroized_direction_sha256",
            "direction_shape",
            "all_zero_before_release",
            "private_reference_released",
            "erasure_sha256",
        }
        if not isinstance(row, Mapping) or set(row) != fields:
            raise ValueError("constraint erasure fields differ")
        inventory = constraint_rows.get(row["constraint_id"])
        expected_reason = (
            {
                "consumed": "one_use_consumed",
                "expired_ttl": "ttl_expired",
                "expired_episode_end": "episode_ended",
                "expired_stale_kv": "stale_kv_boundary",
                "expired_stale_state": "stale_parent_state",
                "aborted_episode_failure": "episode_aborted",
            }.get(inventory["status"])
            if inventory is not None
            else None
        )
        erasure_payload = {key: row[key] for key in fields - {"erasure_sha256"}}
        if (
            inventory is None
            or row["constraint_id"] in erased_ids
            or row["prior_direction_sha256"] != inventory["direction_sha256"]
            or row["direction_shape"] != inventory["direction_shape"]
            or not is_sha256(row["zeroized_direction_sha256"])
            or row["reason"] != expected_reason
            or row["all_zero_before_release"] is not True
            or row["private_reference_released"] is not True
            or row["erasure_sha256"] != canonical_sha256(erasure_payload)
        ):
            raise ValueError("constraint erasure is invalid")
        erased_ids.add(row["constraint_id"])

    aggregates = receipt["aggregates"]
    expected_aggregates = {
        "critic_rejection_count": len(receipt["critic_rejections"]),
        "attempt_count": len(attempts),
        "admitted_count": len(admitted),
        "application_count": len(applications),
        "reservation_rollback_count": len(rollbacks),
        "erasure_count": len(erasures),
        "verified_reduction_count": sum(row["failure_reduced"] for row in applications),
        "verified_repeat_count": sum(row["failure_repeated"] for row in applications),
        "active_after_episode": 0,
        "private_directions_after_episode": 0,
    }
    if aggregates != expected_aggregates:
        raise ValueError("transient constraint aggregates differ")
    if any(
        row["status"] == "expired_episode_end" and row["expires_after_action_step"] < final_step
        for row in constraints
    ):
        raise ValueError("expired transient constraint missed its TTL")
    if require_external_bindings and attempts:
        if (
            not isinstance(cognitive_action_trace, Sequence)
            or isinstance(cognitive_action_trace, (str, bytes))
            or not isinstance(verifier_preflight, Mapping)
            or not isinstance(information_accounting, Mapping)
            or not isinstance(resource_accounting, Mapping)
            or not isinstance(kv_state_tree, Mapping)
            or (
                require_verified_best_binding
                and (
                    not isinstance(verified_best_state, Mapping)
                    or not isinstance(loop_stability, Mapping)
                )
            )
        ):
            raise ValueError("transient constraint external evidence is absent")
        from core.brain.llm.latent_cortex.blind_review import (
            validate_decoy_preflight_receipt,
        )
        from core.brain.llm.latent_cortex.resource_accounting import (
            validate_information_receipt,
            validate_resource_receipt,
        )

        validated_preflight = validate_decoy_preflight_receipt(
            dict(verifier_preflight),
            episode_id=episode_id,
            objective_sha256=objective_sha256,
        )
        validated_information = validate_information_receipt(information_accounting)
        validated_resources = validate_resource_receipt(resource_accounting)
        if (
            validated_preflight["verifier_admitted"] is not True
            or validated_information["accounting_complete"] is not True
            or validated_resources["accounting_complete"] is not True
        ):
            raise ValueError("transient constraint verifier was not admitted")
        verifier_policy_sha256 = validated_information["policies"].get("verifier")
        action_rows: dict[int, Mapping[str, Any]] = {}
        for action_row in cognitive_action_trace:
            transition = action_row.get("transition") if isinstance(action_row, Mapping) else None
            step = transition.get("step_index") if isinstance(transition, Mapping) else None
            if type(step) is not int or step in action_rows:
                raise ValueError("transient constraint action source is invalid")
            action_rows[step] = action_row
        verified_decisions: dict[tuple[int, int], Mapping[str, Any]] = {}
        if require_verified_best_binding:
            from core.brain.llm.latent_cortex.verified_best import (
                validate_verified_best_receipt,
            )

            validated_verified_best = validate_verified_best_receipt(
                verified_best_state,
                cognitive_action_trace=list(cognitive_action_trace),
                loop_stability=dict(loop_stability),
                expected_n_branches=n_branches,
            )
            for branch in validated_verified_best["branches"]:
                branch_index = branch["branch_index"]
                for decision in branch["decisions"]:
                    key = (branch_index, decision["action_step"])
                    if key in verified_decisions:
                        raise ValueError("transient verified-best source overlaps")
                    verified_decisions[key] = decision
        node_rows = kv_state_tree.get("nodes")
        if not isinstance(node_rows, list):
            raise ValueError("transient constraint KV source is invalid")
        kv_nodes = {
            row.get("node_sha256"): row
            for row in node_rows
            if isinstance(row, Mapping) and is_sha256(row.get("node_sha256"))
        }
        if len(kv_nodes) != len(node_rows):
            raise ValueError("transient constraint KV source is invalid")
        for attempt in attempts:
            action_row = action_rows.get(attempt["created_action_step"])
            transition = action_row.get("transition") if isinstance(action_row, Mapping) else None
            verification = (
                action_row.get("verification") if isinstance(action_row, Mapping) else None
            )
            source_node = kv_nodes.get(attempt["source_kv_boundary_sha256"])
            verified_decision = verified_decisions.get(
                (attempt["branch_index"], attempt["created_action_step"])
            )
            if (
                not isinstance(transition, Mapping)
                or not isinstance(verification, Mapping)
                or action_row.get("transient_constraint_attempt") != attempt
                or transition.get("action") != attempt["source_action"]
                or transition.get("step_index") != attempt["created_action_step"]
                or verification.get("target_branch") != attempt["branch_index"]
                or verification.get("observation") != attempt["source_observation"]
                or verification.get("candidate_state_sha256") != attempt["failed_state_sha256"]
                or verification.get("kv_boundary_before_sha256")
                != attempt["source_kv_boundary_sha256"]
                or verification.get("decision")
                not in {"preserve_verified", "reject_verified_failure"}
                or attempt["verifier_policy_sha256"] != verifier_policy_sha256
                or attempt["verifier_preflight_sha256"] != validated_preflight["receipt_sha256"]
                or source_node is None
                or source_node.get("branch_index") not in {None, attempt["branch_index"]}
                or (
                    require_verified_best_binding
                    and (
                        not isinstance(verified_decision, Mapping)
                        or verified_decision.get("candidate_state_sha256")
                        != attempt["failed_state_sha256"]
                        or verified_decision.get("resulting_state_sha256")
                        != attempt["parent_state_sha256"]
                        or verified_decision.get("observation") != attempt["source_observation"]
                        or verified_decision.get("decision") != verification.get("decision")
                        or verified_decision.get("branch_step")
                        != verification.get("branch_step_after")
                    )
                )
            ):
                raise ValueError("transient constraint source binding differs")
        for application in applications:
            action_row = action_rows.get(application["applied_action_step"])
            transition = action_row.get("transition") if isinstance(action_row, Mapping) else None
            if (
                not isinstance(transition, Mapping)
                or action_row.get("transient_constraint") != application
                or transition.get("action") != application["applied_action"]
                or transition.get("step_index") != application["applied_action_step"]
                or not isinstance(action_row.get("verification"), Mapping)
                or action_row["verification"].get("constraint_input_state_sha256")
                != application["pre_state_sha256"]
                or action_row["verification"].get("candidate_state_sha256")
                != application["post_recurrence_state_sha256"]
                or action_row["verification"].get("kv_boundary_before_sha256")
                != application["kv_boundary_before_sha256"]
                or action_row["verification"].get("kv_boundary_after_sha256")
                != application["kv_boundary_after_sha256"]
                or action_row["verification"].get("branch_step_before")
                != application["branch_step_before"]
                or action_row["verification"].get("branch_step_after")
                != application["branch_step_after"]
                or application["kv_boundary_before_sha256"] not in kv_nodes
                or application["kv_boundary_after_sha256"] not in kv_nodes
            ):
                raise ValueError("transient constraint application binding differs")
        observed_resource_totals = {name: 0 for name in RESOURCE_COUNTERS}
        for attempt in attempts:
            for arm in attempt["arms"]:
                for replicate in arm["replicates"]:
                    if any(
                        replicate["resource_after"][name] > validated_resources["totals"][name]
                        for name in RESOURCE_COUNTERS
                    ):
                        raise ValueError(
                            "transient constraint resource window exceeds episode totals"
                        )
                    for name, amount in replicate["resource_delta"].items():
                        observed_resource_totals[name] += amount
        if any(
            amount > validated_resources["totals"][name]
            for name, amount in observed_resource_totals.items()
        ):
            raise ValueError("transient constraint resources exceed episode totals")
    return receipt


__all__ = [
    "ARM_NAMES",
    "TRANSIENT_CONSTRAINT_SCHEMA",
    "TransientConstraintConfig",
    "TransientConstraintLedger",
    "build_empty_transient_constraint_receipt",
    "validate_transient_constraint_receipt",
]
