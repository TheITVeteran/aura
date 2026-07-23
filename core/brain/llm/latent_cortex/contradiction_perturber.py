"""Bounded, counterfactually admitted latent repair from contradiction evidence."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.brain.llm.latent_cortex.counterfactual_probe import (
    CounterfactualProbeResult,
)
from core.brain.llm.latent_cortex.loop_core import canonical_sha256
from core.brain.llm.latent_cortex.verified_best import (
    tensor_sha256,
    validate_observation,
)

CONTRADICTION_PERTURBATION_SCHEMA = "aura.rlc.contradiction_perturbation_receipt.v1"
DISABLED = "disabled"
COUNTERFACTUAL = "counterfactual"
ARM_NAMES = ("no_op", "matched_random", "contradiction_guided")
MAX_REPLICATES = 4


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


@dataclass(frozen=True, slots=True)
class ContradictionPerturberConfig:
    """Trust boundary for one localized, reversible latent intervention."""

    mode: str = COUNTERFACTUAL
    max_relative_delta_rms: float = 0.08
    min_verifier_margin: float = 0.01
    replicates: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {DISABLED, COUNTERFACTUAL}:
            raise ValueError("contradiction perturber mode is invalid")
        if (
            not _finite(self.max_relative_delta_rms)
            or not 0.0 < float(self.max_relative_delta_rms) <= 0.25
        ):
            raise ValueError("contradiction perturber delta bound must be inside (0, 0.25]")
        if (
            not _finite(self.min_verifier_margin)
            or not 0.0 <= float(self.min_verifier_margin) <= 0.25
        ):
            raise ValueError("contradiction perturber verifier margin must be inside [0, 0.25]")
        if type(self.replicates) is not int or not 2 <= self.replicates <= MAX_REPLICATES:
            raise ValueError(
                f"contradiction perturber replicates must be inside [2, {MAX_REPLICATES}]"
            )
        if type(self.seed) is not int or not -(2**63) <= self.seed <= 2**63 - 1:
            raise ValueError("contradiction perturber seed must be signed 64-bit")

    @classmethod
    def from_value(
        cls,
        value: Mapping[str, Any] | None,
    ) -> ContradictionPerturberConfig:
        raw = dict(value or {})
        unknown = set(raw) - {
            "mode",
            "max_relative_delta_rms",
            "min_verifier_margin",
            "replicates",
            "seed",
        }
        if unknown:
            raise ValueError(f"contradiction perturber has unknown keys: {sorted(unknown)}")
        return cls(
            mode=raw.get("mode", COUNTERFACTUAL),
            max_relative_delta_rms=raw.get("max_relative_delta_rms", 0.08),
            min_verifier_margin=raw.get("min_verifier_margin", 0.01),
            replicates=raw.get("replicates", 2),
            seed=raw.get("seed", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "max_relative_delta_rms": round(float(self.max_relative_delta_rms), 10),
            "min_verifier_margin": round(float(self.min_verifier_margin), 10),
            "replicates": self.replicates,
            "seed": self.seed,
        }


PerturbationArmResult = CounterfactualProbeResult


def _as_state(value: Any, *, name: str) -> np.ndarray:
    state = np.asarray(value)
    if (
        state.ndim != 3
        or state.shape[0] != 1
        or state.shape[1] < 1
        or state.shape[2] < 1
        or state.size > 100_000_000
        or not np.issubdtype(state.dtype, np.floating)
        or not np.all(np.isfinite(state))
    ):
        raise ValueError(f"{name} latent state is invalid")
    return np.array(state, copy=True)


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))


def _candidate_states(
    baseline: Any,
    anchor: Any,
    *,
    position_index: int,
    protected_positions: Sequence[int],
    config: ContradictionPerturberConfig,
    seed_material: str,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    base = _as_state(baseline, name="baseline")
    anchor_state = _as_state(anchor, name="anchor")
    if base.shape != anchor_state.shape:
        raise ValueError("contradiction perturbation baseline/anchor shapes differ")
    if type(position_index) is not int or not 0 <= position_index < base.shape[1]:
        raise ValueError("contradiction perturbation position is invalid")
    protected = tuple(sorted(set(protected_positions)))
    if any(type(index) is not int or not 0 <= index < base.shape[1] for index in protected):
        raise ValueError("contradiction perturbation protected positions are invalid")
    if position_index in protected:
        raise ValueError("contradiction perturbation targeted immutable evidence")
    if not _is_sha256(seed_material):
        raise ValueError("contradiction perturbation seed binding is invalid")

    slot = base[:, position_index : position_index + 1, :]
    anchor_slot = anchor_state[:, position_index : position_index + 1, :]
    direction = anchor_slot - slot
    direction_rms = _rms(direction)
    slot_rms = max(_rms(slot), 1e-6)
    max_delta_rms = float(config.max_relative_delta_rms) * slot_rms
    if direction_rms <= 1e-12 or max_delta_rms <= 1e-12:
        raise ValueError("contradiction perturbation has no guided direction")
    delta_rms = min(direction_rms, max_delta_rms)
    guided_delta = direction * (delta_rms / direction_rms)

    seed_payload = hashlib.sha256(
        f"{seed_material}:{config.seed}:{position_index}".encode("ascii")
    ).digest()
    seed = int.from_bytes(seed_payload[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    random_delta = rng.standard_normal(slot.shape).astype(base.dtype, copy=False)
    random_delta -= np.mean(random_delta, axis=-1, keepdims=True)
    guided_flat = guided_delta.reshape(-1).astype(np.float64)
    random_flat = random_delta.reshape(-1).astype(np.float64)
    guided_energy = float(np.dot(guided_flat, guided_flat))
    if guided_energy > 1e-18:
        random_flat -= (float(np.dot(random_flat, guided_flat)) / guided_energy) * guided_flat
        random_delta = random_flat.reshape(random_delta.shape).astype(base.dtype)
    random_rms = _rms(random_delta)
    if random_rms <= 1e-12:
        raise ValueError("contradiction perturbation random control degenerated")
    random_delta *= delta_rms / random_rms

    states = {
        "no_op": np.array(base, copy=True),
        "matched_random": np.array(base, copy=True),
        "contradiction_guided": np.array(base, copy=True),
    }
    states["matched_random"][:, position_index : position_index + 1, :] += random_delta
    states["contradiction_guided"][:, position_index : position_index + 1, :] += guided_delta

    summaries: dict[str, dict[str, Any]] = {}
    baseline_sha256 = tensor_sha256(base)
    for name in ARM_NAMES:
        state = states[name]
        delta = state - base
        target_delta = delta[:, position_index : position_index + 1, :]
        target_delta_flat = target_delta.reshape(-1).astype(np.float64)
        target_delta_rms = _rms(target_delta)
        if target_delta_rms <= 1e-12:
            cosine_to_guided = None
        else:
            denominator = math.sqrt(
                float(np.dot(target_delta_flat, target_delta_flat)) * guided_energy
            )
            cosine_to_guided = round(
                float(np.dot(target_delta_flat, guided_flat)) / max(denominator, 1e-18),
                12,
            )
        changed_positions = [
            index
            for index in range(base.shape[1])
            if not np.array_equal(state[:, index, :], base[:, index, :])
        ]
        summaries[name] = {
            "name": name,
            "state_sha256": tensor_sha256(state),
            "delta_rms": round(_rms(delta), 12),
            "target_delta_rms": round(target_delta_rms, 12),
            "relative_target_delta_rms": round(
                target_delta_rms / slot_rms,
                12,
            ),
            "cosine_to_guided_delta": cosine_to_guided,
            "changed_positions": changed_positions,
            "protected_positions_unchanged": all(
                np.array_equal(state[:, index, :], base[:, index, :]) for index in protected
            ),
        }
    if summaries["no_op"]["state_sha256"] != baseline_sha256:
        raise RuntimeError("contradiction no-op control changed state")
    if (
        summaries["matched_random"]["state_sha256"] == baseline_sha256
        or summaries["contradiction_guided"]["state_sha256"] == baseline_sha256
        or summaries["matched_random"]["state_sha256"]
        == summaries["contradiction_guided"]["state_sha256"]
    ):
        raise RuntimeError("contradiction perturbation controls are not distinct")
    guided_rms = summaries["contradiction_guided"]["target_delta_rms"]
    random_rms = summaries["matched_random"]["target_delta_rms"]
    if not math.isclose(guided_rms, random_rms, rel_tol=1e-5, abs_tol=1e-8):
        raise RuntimeError("contradiction random control is not magnitude matched")
    if any(
        summary["changed_positions"] not in ([], [position_index])
        or not summary["protected_positions_unchanged"]
        or summary["relative_target_delta_rms"] > float(config.max_relative_delta_rms) + 1e-6
        for summary in summaries.values()
    ):
        raise RuntimeError("contradiction perturbation escaped its state bound")
    return states, summaries


def _empty_receipt(
    *,
    config: ContradictionPerturberConfig,
    contradiction_tensor: Mapping[str, Any],
    selected_branch: int,
    verifier_policy_sha256: str,
    decoy_review_sha256: str,
    protected_positions: Sequence[int],
    status: str,
    reason: str,
) -> dict[str, Any]:
    branches = contradiction_tensor.get("branches")
    candidate_probability = (
        branches[selected_branch].get("candidate_probability")
        if (
            isinstance(branches, list)
            and 0 <= selected_branch < len(branches)
            and isinstance(branches[selected_branch], Mapping)
            and contradiction_tensor.get("selected_branch_candidate") is not None
        )
        else None
    )
    payload = {
        "schema": CONTRADICTION_PERTURBATION_SCHEMA,
        "config": config.to_dict(),
        "contradiction_tensor_sha256": contradiction_tensor.get("receipt_sha256", ""),
        "selected_branch": selected_branch,
        "candidate": contradiction_tensor.get("selected_branch_candidate"),
        "candidate_probability": candidate_probability,
        "protected_positions": sorted(set(protected_positions)),
        "verifier_policy_sha256": verifier_policy_sha256,
        "decoy_review_sha256": decoy_review_sha256,
        "status": status,
        "reason": reason,
        "target_kind": "latent_workspace_sequence_position",
        "baseline_state_sha256": "",
        "resulting_state_sha256": "",
        "evaluation_order": [],
        "arms": [],
        "all_arms_equal_compute": False,
        "all_observations_authoritative": False,
        "repeat_stable": False,
        "guided_beats_controls": False,
        "state_mutation_applied": False,
        "rollback_proven": False,
        "answer_text_stored": False,
        "authority_scope": "none",
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def run_contradiction_perturbation(
    *,
    baseline: Any,
    anchor: Any,
    protected_positions: Sequence[int],
    contradiction_tensor: Mapping[str, Any],
    selected_branch: int,
    config: ContradictionPerturberConfig,
    verifier_policy_sha256: str,
    decoy_review_sha256: str,
    evaluate: Callable[[str, Any, int], PerturbationArmResult] | None,
    evaluation_unavailable_reason: str = "",
    budget: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Evaluate bounded arms and return either the admitted state or baseline."""

    if (
        not isinstance(contradiction_tensor, Mapping)
        or not _is_sha256(contradiction_tensor.get("receipt_sha256"))
        or type(selected_branch) is not int
        or selected_branch != contradiction_tensor.get("selected_branch")
    ):
        raise ValueError("contradiction perturbation source is invalid")
    if config.mode == DISABLED:
        return baseline, _empty_receipt(
            config=config,
            contradiction_tensor=contradiction_tensor,
            selected_branch=selected_branch,
            verifier_policy_sha256=verifier_policy_sha256,
            decoy_review_sha256=decoy_review_sha256,
            protected_positions=protected_positions,
            status="disabled",
            reason="configured_disabled",
        )
    candidate = contradiction_tensor.get("selected_branch_candidate")
    branches = contradiction_tensor.get("branches")
    if (
        contradiction_tensor.get("mode") != "learned"
        or not isinstance(candidate, Mapping)
        or not isinstance(branches, list)
        or not 0 <= selected_branch < len(branches)
    ):
        return baseline, _empty_receipt(
            config=config,
            contradiction_tensor=contradiction_tensor,
            selected_branch=selected_branch,
            verifier_policy_sha256=verifier_policy_sha256,
            decoy_review_sha256=decoy_review_sha256,
            protected_positions=protected_positions,
            status="skipped",
            reason="learned_localized_candidate_unavailable",
        )
    probability = branches[selected_branch].get("candidate_probability")
    if not _finite(probability) or not 0.0 <= float(probability) <= 1.0:
        raise ValueError("contradiction perturbation probability is invalid")
    position_index = candidate.get("position_index")
    transition_index = candidate.get("transition_index")
    if type(position_index) is not int or type(transition_index) is not int:
        raise ValueError("contradiction perturbation coordinate is invalid")
    if position_index in set(protected_positions):
        return baseline, _empty_receipt(
            config=config,
            contradiction_tensor=contradiction_tensor,
            selected_branch=selected_branch,
            verifier_policy_sha256=verifier_policy_sha256,
            decoy_review_sha256=decoy_review_sha256,
            protected_positions=protected_positions,
            status="skipped",
            reason="candidate_targets_immutable_evidence",
        )
    if (
        evaluate is None
        or not _is_sha256(verifier_policy_sha256)
        or not _is_sha256(decoy_review_sha256)
    ):
        return baseline, _empty_receipt(
            config=config,
            contradiction_tensor=contradiction_tensor,
            selected_branch=selected_branch,
            verifier_policy_sha256=verifier_policy_sha256,
            decoy_review_sha256=decoy_review_sha256,
            protected_positions=protected_positions,
            status="skipped",
            reason=(evaluation_unavailable_reason or "independent_admitted_verifier_unavailable"),
        )

    try:
        states, summaries = _candidate_states(
            baseline,
            anchor,
            position_index=position_index,
            protected_positions=protected_positions,
            config=config,
            seed_material=contradiction_tensor["receipt_sha256"],
        )
    except ValueError as exc:
        if str(exc) != "contradiction perturbation has no guided direction":
            raise
        return baseline, _empty_receipt(
            config=config,
            contradiction_tensor=contradiction_tensor,
            selected_branch=selected_branch,
            verifier_policy_sha256=verifier_policy_sha256,
            decoy_review_sha256=decoy_review_sha256,
            protected_positions=protected_positions,
            status="skipped",
            reason="guided_direction_unavailable",
        )
    if budget is not None:
        elements = int(np.asarray(baseline).size)
        budget.charge_tensor_work(
            "contradiction_perturbation_candidates",
            element_reads=6 * elements,
            element_writes=3 * elements,
            scalar_ops=8 * elements,
            host_scalar_ops=256,
        )

    seed = hashlib.sha256(
        (
            f"{contradiction_tensor['receipt_sha256']}:{config.seed}:"
            f"{selected_branch}:{position_index}"
        ).encode("ascii")
    ).digest()
    orders: list[list[str]] = []
    rng = np.random.default_rng(int.from_bytes(seed[:8], "big"))
    for _ in range(config.replicates):
        order = list(ARM_NAMES)
        rng.shuffle(order)
        orders.append(order)

    results: dict[str, list[dict[str, Any]]] = {name: [] for name in ARM_NAMES}
    try:
        for replicate, order in enumerate(orders):
            for name in order:
                results[name].append(evaluate(name, states[name], replicate).normalized())
    except Exception as exc:
        payload = _empty_receipt(
            config=config,
            contradiction_tensor=contradiction_tensor,
            selected_branch=selected_branch,
            verifier_policy_sha256=verifier_policy_sha256,
            decoy_review_sha256=decoy_review_sha256,
            protected_positions=protected_positions,
            status="restored",
            reason=f"evaluation_failed:{type(exc).__name__}",
        )
        payload = dict(payload)
        payload.pop("receipt_sha256")
        payload["baseline_state_sha256"] = tensor_sha256(baseline)
        payload["resulting_state_sha256"] = tensor_sha256(baseline)
        payload["rollback_proven"] = True
        return baseline, {**payload, "receipt_sha256": canonical_sha256(payload)}

    arm_rows: list[dict[str, Any]] = []
    all_authoritative = True
    repeat_stable = True
    layer_apps: list[int] = []
    for name in ARM_NAMES:
        observations = [row["observation"] for row in results[name]]
        all_authoritative = all_authoritative and all(
            row["authoritative"] is True for row in observations
        )
        repeat_stable = repeat_stable and all(row == observations[0] for row in observations[1:])
        layer_apps.extend(row["layer_apps"] for row in results[name])
        arm_rows.append(
            {
                **summaries[name],
                "replicates": results[name],
            }
        )
    all_equal_compute = bool(layer_apps) and len(set(layer_apps)) == 1
    observations_by_name = {
        name: [row["observation"] for row in results[name]] for name in ARM_NAMES
    }
    guided_lower = min(
        float(row["lower_bound"]) for row in observations_by_name["contradiction_guided"]
    )
    control_upper = max(
        float(row["upper_bound"])
        for name in ("no_op", "matched_random")
        for row in observations_by_name[name]
    )
    guided_beats_controls = (
        all_authoritative
        and repeat_stable
        and all_equal_compute
        and guided_lower > control_upper + float(config.min_verifier_margin) + 1e-12
    )
    baseline_sha256 = tensor_sha256(baseline)
    if guided_beats_controls:
        resulting = states["contradiction_guided"]
        status = "retained"
        reason = "guided_lower_bound_beats_both_controls"
        authority_scope = "selected_branch_target_position_only"
    else:
        resulting = baseline
        status = "restored"
        if not all_authoritative:
            reason = "non_authoritative_verifier_observation"
        elif not repeat_stable:
            reason = "verifier_repeat_instability"
        elif not all_equal_compute:
            reason = "control_compute_mismatch"
        else:
            reason = "guided_candidate_did_not_beat_controls"
        authority_scope = "none"
    resulting_sha256 = tensor_sha256(resulting)
    payload = {
        "schema": CONTRADICTION_PERTURBATION_SCHEMA,
        "config": config.to_dict(),
        "contradiction_tensor_sha256": contradiction_tensor["receipt_sha256"],
        "selected_branch": selected_branch,
        "candidate": {
            "transition_index": transition_index,
            "position_index": position_index,
        },
        "candidate_probability": round(float(probability), 10),
        "protected_positions": sorted(set(protected_positions)),
        "verifier_policy_sha256": verifier_policy_sha256,
        "decoy_review_sha256": decoy_review_sha256,
        "status": status,
        "reason": reason,
        "target_kind": "latent_workspace_sequence_position",
        "baseline_state_sha256": baseline_sha256,
        "resulting_state_sha256": resulting_sha256,
        "evaluation_order": orders,
        "arms": arm_rows,
        "all_arms_equal_compute": all_equal_compute,
        "all_observations_authoritative": all_authoritative,
        "repeat_stable": repeat_stable,
        "guided_beats_controls": guided_beats_controls,
        "state_mutation_applied": guided_beats_controls,
        "rollback_proven": (
            resulting_sha256 == baseline_sha256 if not guided_beats_controls else False
        ),
        "answer_text_stored": False,
        "authority_scope": authority_scope,
    }
    receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
    return resulting, validate_contradiction_perturbation_receipt(
        receipt,
        expected_config=config,
        contradiction_tensor=contradiction_tensor,
        expected_selected_branch=selected_branch,
        expected_protected_positions=protected_positions,
        verifier_policy_sha256=verifier_policy_sha256,
        decoy_review_sha256=decoy_review_sha256,
    )


def validate_contradiction_perturbation_receipt(
    value: Any,
    *,
    expected_config: ContradictionPerturberConfig,
    contradiction_tensor: Mapping[str, Any],
    expected_selected_branch: int,
    expected_protected_positions: Sequence[int],
    verifier_policy_sha256: str,
    decoy_review_sha256: str,
) -> dict[str, Any]:
    fields = {
        "schema",
        "config",
        "contradiction_tensor_sha256",
        "selected_branch",
        "candidate",
        "candidate_probability",
        "protected_positions",
        "verifier_policy_sha256",
        "decoy_review_sha256",
        "status",
        "reason",
        "target_kind",
        "baseline_state_sha256",
        "resulting_state_sha256",
        "evaluation_order",
        "arms",
        "all_arms_equal_compute",
        "all_observations_authoritative",
        "repeat_stable",
        "guided_beats_controls",
        "state_mutation_applied",
        "rollback_proven",
        "answer_text_stored",
        "authority_scope",
        "receipt_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or not isinstance(contradiction_tensor, Mapping)
        or not _is_sha256(contradiction_tensor.get("receipt_sha256"))
    ):
        raise ValueError("contradiction perturbation receipt fields/source differ")
    receipt = dict(value)
    payload = {key: receipt[key] for key in fields - {"receipt_sha256"}}
    protected = sorted(set(expected_protected_positions))
    source_branches = contradiction_tensor.get("branches")
    source_probability = (
        source_branches[expected_selected_branch].get("candidate_probability")
        if (
            isinstance(source_branches, list)
            and 0 <= expected_selected_branch < len(source_branches)
            and isinstance(source_branches[expected_selected_branch], Mapping)
            and contradiction_tensor.get("selected_branch_candidate") is not None
        )
        else None
    )
    if (
        receipt["schema"] != CONTRADICTION_PERTURBATION_SCHEMA
        or receipt["config"] != expected_config.to_dict()
        or receipt["contradiction_tensor_sha256"] != contradiction_tensor["receipt_sha256"]
        or receipt["selected_branch"] != expected_selected_branch
        or receipt["candidate"] != contradiction_tensor.get("selected_branch_candidate")
        or receipt["candidate_probability"] != source_probability
        or receipt["protected_positions"] != protected
        or receipt["verifier_policy_sha256"] != verifier_policy_sha256
        or receipt["decoy_review_sha256"] != decoy_review_sha256
        or receipt["target_kind"] != "latent_workspace_sequence_position"
        or receipt["answer_text_stored"] is not False
        or receipt["receipt_sha256"] != canonical_sha256(payload)
    ):
        raise ValueError("contradiction perturbation identity is invalid")
    if receipt["status"] in {"disabled", "skipped"}:
        if (
            receipt["arms"]
            or receipt["evaluation_order"]
            or receipt["baseline_state_sha256"] != ""
            or receipt["resulting_state_sha256"] != ""
            or receipt["all_arms_equal_compute"] is not False
            or receipt["all_observations_authoritative"] is not False
            or receipt["repeat_stable"] is not False
            or receipt["guided_beats_controls"] is not False
            or receipt["state_mutation_applied"] is not False
            or receipt["rollback_proven"] is not False
            or receipt["authority_scope"] != "none"
        ):
            raise ValueError("inactive contradiction perturbation claims authority")
        if receipt["status"] == "disabled" and (
            expected_config.mode != DISABLED or receipt["reason"] != "configured_disabled"
        ):
            raise ValueError("contradiction perturbation disabled state differs")
        if receipt["status"] == "skipped" and receipt["reason"] not in {
            "learned_localized_candidate_unavailable",
            "candidate_targets_immutable_evidence",
            "independent_admitted_verifier_unavailable",
            "counterfactual_probe_budget_unavailable",
            "guided_direction_unavailable",
        }:
            raise ValueError("contradiction perturbation skip reason is invalid")
        if receipt["reason"] == "candidate_targets_immutable_evidence" and (
            not isinstance(receipt["candidate"], Mapping)
            or receipt["candidate"].get("position_index") not in protected
        ):
            raise ValueError("contradiction perturbation evidence protection differs")
        return receipt
    if receipt["status"] == "restored" and not receipt["arms"]:
        failure_kind = str(receipt["reason"]).removeprefix("evaluation_failed:")
        if (
            not str(receipt["reason"]).startswith("evaluation_failed:")
            or not failure_kind.isidentifier()
            or len(failure_kind) > 128
            or not _is_sha256(receipt["baseline_state_sha256"])
            or receipt["resulting_state_sha256"] != receipt["baseline_state_sha256"]
            or receipt["evaluation_order"]
            or receipt["all_arms_equal_compute"] is not False
            or receipt["all_observations_authoritative"] is not False
            or receipt["repeat_stable"] is not False
            or receipt["guided_beats_controls"] is not False
            or receipt["rollback_proven"] is not True
            or receipt["state_mutation_applied"] is not False
            or receipt["authority_scope"] != "none"
        ):
            raise ValueError("failed contradiction perturbation did not roll back")
        return receipt
    if receipt["status"] not in {"retained", "restored"}:
        raise ValueError("contradiction perturbation status is invalid")
    if (
        not _is_sha256(receipt["baseline_state_sha256"])
        or not _is_sha256(receipt["resulting_state_sha256"])
        or len(receipt["arms"]) != len(ARM_NAMES)
        or len(receipt["evaluation_order"]) != expected_config.replicates
        or any(sorted(order) != sorted(ARM_NAMES) for order in receipt["evaluation_order"])
    ):
        raise ValueError("contradiction perturbation evaluated evidence is invalid")
    rows = {row.get("name"): row for row in receipt["arms"] if isinstance(row, Mapping)}
    if set(rows) != set(ARM_NAMES):
        raise ValueError("contradiction perturbation controls are incomplete")
    observations: dict[str, list[dict[str, Any]]] = {}
    layer_apps: list[int] = []
    for name, row in rows.items():
        required = {
            "name",
            "state_sha256",
            "delta_rms",
            "target_delta_rms",
            "relative_target_delta_rms",
            "cosine_to_guided_delta",
            "changed_positions",
            "protected_positions_unchanged",
            "replicates",
        }
        if (
            set(row) != required
            or not _is_sha256(row["state_sha256"])
            or not _finite(row["delta_rms"])
            or float(row["delta_rms"]) < 0.0
            or not _finite(row["target_delta_rms"])
            or float(row["target_delta_rms"]) < 0.0
            or not _finite(row["relative_target_delta_rms"])
            or float(row["relative_target_delta_rms"]) < 0.0
            or row["relative_target_delta_rms"]
            > float(expected_config.max_relative_delta_rms) + 1e-6
            or row["protected_positions_unchanged"] is not True
            or len(row["replicates"]) != expected_config.replicates
        ):
            raise ValueError("contradiction perturbation arm geometry is invalid")
        normalized = []
        for result in row["replicates"]:
            if not isinstance(result, Mapping):
                raise ValueError("contradiction perturbation replicate is invalid")
            result_fields = {
                "probe_tokens_sha256",
                "probe_token_count",
                "observation",
                "layer_apps",
            }
            if set(result) != result_fields:
                raise ValueError("contradiction perturbation replicate fields differ")
            normalized.append(validate_observation(result["observation"]))
            if (
                not _is_sha256(result["probe_tokens_sha256"])
                or type(result["probe_token_count"]) is not int
                or result["probe_token_count"] <= 0
                or type(result["layer_apps"]) is not int
                or result["layer_apps"] <= 0
            ):
                raise ValueError("contradiction perturbation replicate is invalid")
            layer_apps.append(result["layer_apps"])
        observations[name] = normalized
    if (
        rows["no_op"]["state_sha256"] != receipt["baseline_state_sha256"]
        or float(rows["no_op"]["delta_rms"]) != 0.0
        or float(rows["no_op"]["target_delta_rms"]) != 0.0
        or float(rows["no_op"]["relative_target_delta_rms"]) != 0.0
        or rows["no_op"]["changed_positions"] != []
        or rows["no_op"]["cosine_to_guided_delta"] is not None
        or rows["matched_random"]["state_sha256"]
        in {
            receipt["baseline_state_sha256"],
            rows["contradiction_guided"]["state_sha256"],
        }
        or rows["contradiction_guided"]["state_sha256"] == receipt["baseline_state_sha256"]
        or float(rows["matched_random"]["target_delta_rms"]) <= 0.0
        or float(rows["contradiction_guided"]["target_delta_rms"]) <= 0.0
        or rows["matched_random"]["changed_positions"] != [receipt["candidate"]["position_index"]]
        or not _finite(rows["matched_random"]["cosine_to_guided_delta"])
        or abs(float(rows["matched_random"]["cosine_to_guided_delta"])) > 1e-5
        or rows["contradiction_guided"]["changed_positions"]
        != [receipt["candidate"]["position_index"]]
        or not math.isclose(
            float(rows["contradiction_guided"]["cosine_to_guided_delta"]),
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        or not math.isclose(
            float(rows["matched_random"]["target_delta_rms"]),
            float(rows["contradiction_guided"]["target_delta_rms"]),
            rel_tol=1e-5,
            abs_tol=1e-8,
        )
    ):
        raise ValueError("contradiction perturbation geometry reconstruction failed")
    authoritative = all(
        observation["authoritative"] is True
        for values in observations.values()
        for observation in values
    )
    stable = all(
        all(value == values[0] for value in values[1:]) for values in observations.values()
    )
    equal_compute = bool(layer_apps) and len(set(layer_apps)) == 1
    guided_lower = min(float(row["lower_bound"]) for row in observations["contradiction_guided"])
    control_upper = max(
        float(row["upper_bound"])
        for name in ("no_op", "matched_random")
        for row in observations[name]
    )
    beats = (
        authoritative
        and stable
        and equal_compute
        and guided_lower > control_upper + float(expected_config.min_verifier_margin) + 1e-12
    )
    expected_reason = (
        "guided_lower_bound_beats_both_controls"
        if beats
        else (
            "non_authoritative_verifier_observation"
            if not authoritative
            else (
                "verifier_repeat_instability"
                if not stable
                else (
                    "control_compute_mismatch"
                    if not equal_compute
                    else "guided_candidate_did_not_beat_controls"
                )
            )
        )
    )
    if (
        receipt["all_arms_equal_compute"] is not equal_compute
        or receipt["all_observations_authoritative"] is not authoritative
        or receipt["repeat_stable"] is not stable
        or receipt["guided_beats_controls"] is not beats
        or receipt["state_mutation_applied"] is not beats
        or receipt["reason"] != expected_reason
    ):
        raise ValueError("contradiction perturbation decision reconstruction failed")
    if beats:
        if (
            receipt["status"] != "retained"
            or receipt["resulting_state_sha256"] != rows["contradiction_guided"]["state_sha256"]
            or receipt["rollback_proven"] is not False
            or receipt["authority_scope"] != "selected_branch_target_position_only"
        ):
            raise ValueError("contradiction perturbation retained wrong state")
    elif (
        receipt["status"] != "restored"
        or receipt["resulting_state_sha256"] != receipt["baseline_state_sha256"]
        or receipt["rollback_proven"] is not True
        or receipt["authority_scope"] != "none"
    ):
        raise ValueError("contradiction perturbation failed to restore baseline")
    return receipt


__all__ = [
    "ARM_NAMES",
    "CONTRADICTION_PERTURBATION_SCHEMA",
    "COUNTERFACTUAL",
    "DISABLED",
    "ContradictionPerturberConfig",
    "PerturbationArmResult",
    "run_contradiction_perturbation",
    "validate_contradiction_perturbation_receipt",
]
