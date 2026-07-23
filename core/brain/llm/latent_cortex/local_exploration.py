"""Locally conditioned, counterfactually admitted latent exploration."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.brain.llm.latent_cortex.counterfactual_probe import (
    CounterfactualProbeResult,
    is_sha256,
)
from core.brain.llm.latent_cortex.loop_core import canonical_sha256
from core.brain.llm.latent_cortex.verified_best import (
    tensor_sha256,
    validate_observation,
)

LOCAL_EXPLORATION_SCHEMA = "aura.rlc.local_exploration_receipt.v1"
DISABLED = "disabled"
COUNTERFACTUAL = "counterfactual"
FAMILIES = ("baseline", "stable_sham", "conditioned_target")
MAX_CANDIDATES = 4
MAX_REPLICATES = 3
SKIP_REASONS = {
    "learned_sources_are_unavailable",
    "contradiction_candidate_is_unavailable",
    "contradiction_probability_must_be_a_finite_probability",
    "uncertainty_branch_is_malformed",
    "uncertainty_observation_is_unavailable",
    "uncertainty_observation_is_malformed",
    "uncertainty_is_unsupported",
    "predictive_entropy_must_be_a_finite_probability",
    "entropy_is_below_admission_floor",
    "contradiction_tensor_is_empty",
    "contradiction_tensor_is_malformed",
    "contradiction_cell_is_malformed",
    "cell_contradiction_probability_must_be_a_finite_probability",
    "position_identity_is_invalid",
    "stable_sham_position_is_unavailable",
    "target_is_not_writable",
    "uncertainty_source_precedes_retained_perturbation",
    "independent_admitted_verifier_unavailable",
    "counterfactual_probe_budget_unavailable",
    "conditioned_radius_is_zero",
    "candidate_diversity_below_dtype_resolution",
}


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _probability(value: Any, *, name: str) -> float:
    if not _finite(value) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be a finite probability")
    return float(value)


@dataclass(frozen=True, slots=True)
class LocalExplorationConfig:
    """Hard bounds for one uncertainty-conditioned local search."""

    mode: str = COUNTERFACTUAL
    candidates: int = 3
    replicates: int = 2
    max_relative_delta_rms: float = 0.06
    min_predictive_entropy: float = 0.25
    max_stable_contradiction_probability: float = 0.25
    min_verifier_margin: float = 0.01
    min_unique_conditioned_probes: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {DISABLED, COUNTERFACTUAL}:
            raise ValueError("local exploration mode is invalid")
        if type(self.candidates) is not int or not 2 <= self.candidates <= MAX_CANDIDATES:
            raise ValueError(f"local exploration candidates must be inside [2, {MAX_CANDIDATES}]")
        if type(self.replicates) is not int or not 2 <= self.replicates <= MAX_REPLICATES:
            raise ValueError(f"local exploration replicates must be inside [2, {MAX_REPLICATES}]")
        if (
            not _finite(self.max_relative_delta_rms)
            or not 0.0 < float(self.max_relative_delta_rms) <= 0.20
        ):
            raise ValueError("local exploration delta bound must be inside (0, 0.20]")
        if (
            not _finite(self.min_predictive_entropy)
            or not 0.0 <= float(self.min_predictive_entropy) <= 1.0
        ):
            raise ValueError("local exploration entropy floor must be inside [0, 1]")
        if (
            not _finite(self.max_stable_contradiction_probability)
            or not 0.0 <= float(self.max_stable_contradiction_probability) <= 1.0
        ):
            raise ValueError("local exploration stable-region ceiling must be inside [0, 1]")
        if (
            not _finite(self.min_verifier_margin)
            or not 0.0 <= float(self.min_verifier_margin) <= 0.25
        ):
            raise ValueError("local exploration verifier margin must be inside [0, 0.25]")
        if (
            type(self.min_unique_conditioned_probes) is not int
            or not 2 <= self.min_unique_conditioned_probes <= self.candidates
        ):
            raise ValueError("local exploration unique-probe floor must be inside [2, candidates]")
        if type(self.seed) is not int or not -(2**63) <= self.seed <= 2**63 - 1:
            raise ValueError("local exploration seed must be signed 64-bit")

    @classmethod
    def from_value(
        cls,
        value: Mapping[str, Any] | None,
    ) -> LocalExplorationConfig:
        raw = dict(value or {})
        allowed = {
            "mode",
            "candidates",
            "replicates",
            "max_relative_delta_rms",
            "min_predictive_entropy",
            "max_stable_contradiction_probability",
            "min_verifier_margin",
            "min_unique_conditioned_probes",
            "seed",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"local exploration has unknown keys: {sorted(unknown)}")
        return cls(
            mode=raw.get("mode", COUNTERFACTUAL),
            candidates=raw.get("candidates", 3),
            replicates=raw.get("replicates", 2),
            max_relative_delta_rms=raw.get("max_relative_delta_rms", 0.06),
            min_predictive_entropy=raw.get("min_predictive_entropy", 0.25),
            max_stable_contradiction_probability=raw.get(
                "max_stable_contradiction_probability",
                0.25,
            ),
            min_verifier_margin=raw.get("min_verifier_margin", 0.01),
            min_unique_conditioned_probes=raw.get(
                "min_unique_conditioned_probes",
                2,
            ),
            seed=raw.get("seed", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "candidates": self.candidates,
            "replicates": self.replicates,
            "max_relative_delta_rms": round(
                float(self.max_relative_delta_rms),
                10,
            ),
            "min_predictive_entropy": round(
                float(self.min_predictive_entropy),
                10,
            ),
            "max_stable_contradiction_probability": round(
                float(self.max_stable_contradiction_probability),
                10,
            ),
            "min_verifier_margin": round(
                float(self.min_verifier_margin),
                10,
            ),
            "min_unique_conditioned_probes": (self.min_unique_conditioned_probes),
            "seed": self.seed,
        }


def _as_state(value: Any, *, name: str) -> np.ndarray:
    state = np.asarray(value)
    if (
        state.ndim != 3
        or state.shape[0] != 1
        or state.shape[1] < 2
        or state.shape[2] < 3
        or state.size > 100_000_000
        or not np.issubdtype(state.dtype, np.floating)
        or not np.all(np.isfinite(state))
    ):
        raise ValueError(f"{name} latent state is invalid")
    return np.array(state, copy=True)


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))


def _entropy_bits(values: Sequence[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return round(
        -sum((count / total) * math.log2(count / total) for count in counts.values()),
        10,
    )


def _source_signals(
    *,
    contradiction_tensor: Mapping[str, Any],
    neural_uncertainty: Mapping[str, Any],
    selected_branch: int,
    protected_positions: Sequence[int],
    config: LocalExplorationConfig,
) -> tuple[int, int, float, float, float]:
    branches = contradiction_tensor.get("branches")
    uncertainty_branches = neural_uncertainty.get("branches")
    if (
        contradiction_tensor.get("mode") != "learned"
        or neural_uncertainty.get("mode") != "learned"
        or type(selected_branch) is not int
        or selected_branch != contradiction_tensor.get("selected_branch")
        or selected_branch != neural_uncertainty.get("selected_branch")
        or not isinstance(branches, list)
        or not 0 <= selected_branch < len(branches)
        or not isinstance(uncertainty_branches, list)
        or not 0 <= selected_branch < len(uncertainty_branches)
    ):
        raise ValueError("local exploration learned sources are unavailable")
    branch = branches[selected_branch]
    candidate = contradiction_tensor.get("selected_branch_candidate")
    if (
        not isinstance(branch, Mapping)
        or not isinstance(candidate, Mapping)
        or branch.get("candidate") != candidate
    ):
        raise ValueError("local exploration contradiction candidate is unavailable")
    target = candidate.get("position_index")
    probability = _probability(
        branch.get("candidate_probability"),
        name="contradiction probability",
    )
    uncertainty_branch = uncertainty_branches[selected_branch]
    if not isinstance(uncertainty_branch, Mapping):
        raise ValueError("local exploration uncertainty branch is malformed")
    observations = uncertainty_branch.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("local exploration uncertainty observation is unavailable")
    latest_observation = observations[-1]
    if not isinstance(latest_observation, Mapping):
        raise ValueError("local exploration uncertainty observation is malformed")
    estimate = latest_observation.get("estimate")
    if not isinstance(estimate, Mapping) or estimate.get("supported") is not True:
        raise ValueError("local exploration uncertainty is unsupported")
    predictive_entropy = _probability(
        estimate.get("predictive_entropy"),
        name="predictive entropy",
    )
    if predictive_entropy < float(config.min_predictive_entropy):
        raise ValueError("local exploration entropy is below admission floor")
    tensor = branch.get("tensor")
    if not isinstance(tensor, list) or not tensor:
        raise ValueError("local exploration contradiction tensor is empty")
    protected = set(protected_positions)
    if type(target) is not int or target in protected:
        raise ValueError("local exploration target is not writable")
    position_scores: dict[int, float] = {}
    for transition in tensor:
        if not isinstance(transition, list):
            raise ValueError("local exploration contradiction tensor is malformed")
        for cell in transition:
            if not isinstance(cell, Mapping):
                raise ValueError("local exploration contradiction cell is malformed")
            position = cell.get("position_index")
            score = _probability(
                cell.get("contradiction_probability"),
                name="cell contradiction probability",
            )
            if type(position) is not int:
                raise ValueError("local exploration position identity is invalid")
            position_scores[position] = max(
                score,
                position_scores.get(position, 0.0),
            )
    eligible = [
        (score, position)
        for position, score in position_scores.items()
        if (
            position != target
            and position not in protected
            and score <= float(config.max_stable_contradiction_probability)
        )
    ]
    if not eligible:
        raise ValueError("local exploration stable sham position is unavailable")
    stable_score, sham = min(eligible, key=lambda row: (row[0], row[1]))
    return (
        target,
        sham,
        probability,
        predictive_entropy,
        stable_score,
    )


def _directions(
    *,
    width: int,
    count: int,
    seed_sha256: str,
) -> list[np.ndarray]:
    if (
        type(width) is not int
        or width < count + 1
        or width > 16_384
        or type(count) is not int
        or not 2 <= count <= MAX_CANDIDATES
        or not is_sha256(seed_sha256)
    ):
        raise ValueError("local exploration direction shape is invalid")
    rng = np.random.default_rng(int.from_bytes(bytes.fromhex(seed_sha256)[:8], "big"))
    directions: list[np.ndarray] = []
    for _ in range(count):
        vector = rng.standard_normal(width).astype(np.float64)
        vector -= float(np.mean(vector))
        for prior in directions:
            vector -= float(np.dot(vector, prior)) * prior
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("local exploration direction basis degenerated")
        directions.append(vector / norm)
    return directions


def _seed_sha256(
    *,
    contradiction_sha256: str,
    perturbation_sha256: str,
    uncertainty_sha256: str,
    selected_branch: int,
    target_position: int,
    sham_position: int,
    seed: int,
) -> str:
    values = (
        contradiction_sha256,
        perturbation_sha256,
        uncertainty_sha256,
    )
    if any(not is_sha256(value) for value in values):
        raise ValueError("local exploration source commitment is invalid")
    return hashlib.sha256(
        (":".join(values) + f":{selected_branch}:{target_position}:{sham_position}:{seed}").encode(
            "ascii"
        )
    ).hexdigest()


def _candidate_states(
    baseline: Any,
    *,
    target_position: int,
    sham_position: int,
    protected_positions: Sequence[int],
    conditioned_signal: float,
    config: LocalExplorationConfig,
    seed_sha256: str,
) -> tuple[
    dict[str, np.ndarray],
    list[dict[str, Any]],
    dict[str, Any],
]:
    base = _as_state(baseline, name="baseline")
    positions = int(base.shape[1])
    protected = tuple(sorted(set(protected_positions)))
    if (
        type(target_position) is not int
        or type(sham_position) is not int
        or target_position == sham_position
        or target_position in protected
        or sham_position in protected
        or not 0 <= target_position < positions
        or not 0 <= sham_position < positions
        or any(type(position) is not int or not 0 <= position < positions for position in protected)
    ):
        raise ValueError("local exploration positions are invalid")
    signal = _probability(conditioned_signal, name="conditioned signal")
    target_slot = base[:, target_position : target_position + 1, :]
    sham_slot = base[:, sham_position : sham_position + 1, :]
    target_rms = max(_rms(target_slot), 1e-6)
    sham_rms = max(_rms(sham_slot), 1e-6)
    effective_delta_rms = min(
        float(config.max_relative_delta_rms) * signal * target_rms,
        float(config.max_relative_delta_rms) * sham_rms,
    )
    if effective_delta_rms <= 1e-12:
        raise ValueError("local exploration radius is zero")
    raw_directions = _directions(
        width=int(base.shape[-1]),
        count=config.candidates,
        seed_sha256=seed_sha256,
    )
    directions = [direction.astype(base.dtype, copy=False) for direction in raw_directions]
    states: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    baseline_sha256 = tensor_sha256(base)
    for family in FAMILIES:
        for index, direction in enumerate(directions):
            label = f"{family}:{index}"
            state = np.array(base, copy=True)
            position = None
            if family == "conditioned_target":
                position = target_position
            elif family == "stable_sham":
                position = sham_position
            if position is not None:
                # The same immutable direction is reused by target and sham
                # families. Scaling a reshape in place would mutate the
                # source vector and silently give later arms a different
                # radius, defeating the equal-entropy control.
                delta = np.array(
                    direction.reshape(1, 1, -1),
                    copy=True,
                )
                delta *= effective_delta_rms / max(_rms(delta), 1e-12)
                state[:, position : position + 1, :] += delta
            changed = [
                position_index
                for position_index in range(positions)
                if not np.array_equal(
                    state[:, position_index, :],
                    base[:, position_index, :],
                )
            ]
            slot_delta_rms = (
                0.0
                if position is None
                else _rms(
                    state[:, position : position + 1, :] - base[:, position : position + 1, :]
                )
            )
            slot_rms = (
                1.0
                if position is None
                else (target_rms if position == target_position else sham_rms)
            )
            row = {
                "label": label,
                "family": family,
                "candidate_index": index,
                "position_index": position,
                "state_sha256": tensor_sha256(state),
                "direction_sha256": tensor_sha256(direction.reshape(1, 1, -1)),
                "delta_rms": round(slot_delta_rms, 12),
                "relative_delta_rms": round(
                    slot_delta_rms / slot_rms,
                    12,
                ),
                "changed_positions": changed,
                "protected_positions_unchanged": all(
                    np.array_equal(
                        state[:, protected_position, :],
                        base[:, protected_position, :],
                    )
                    for protected_position in protected
                ),
                "replicates": [],
            }
            if family != "baseline" and row["state_sha256"] == baseline_sha256:
                raise ValueError("local exploration candidate diversity degenerated")
            if (family == "baseline" and row["state_sha256"] != baseline_sha256) or (
                family != "baseline"
                and (
                    changed != [position]
                    or not row["protected_positions_unchanged"]
                    or row["relative_delta_rms"] > float(config.max_relative_delta_rms) + 1e-6
                )
            ):
                raise RuntimeError("local exploration escaped its state boundary")
            states[label] = state
            rows.append(row)
    direction_hashes = [tensor_sha256(direction.reshape(1, 1, -1)) for direction in directions]
    cosine_matrix = [
        [round(float(np.dot(left, right)), 12) for right in raw_directions]
        for left in raw_directions
    ]
    metadata = {
        "position_count": positions,
        "hidden_width": int(base.shape[-1]),
        "state_dtype": str(base.dtype),
        "effective_delta_rms": round(effective_delta_rms, 12),
        "target_slot_rms": round(target_rms, 12),
        "sham_slot_rms": round(sham_rms, 12),
        "direction_sha256s": direction_hashes,
        "direction_cosine_matrix": cosine_matrix,
        "latent_direction_entropy_bits": round(
            math.log2(config.candidates),
            10,
        ),
    }
    for family in ("stable_sham", "conditioned_target"):
        hashes = {row["state_sha256"] for row in rows if row["family"] == family}
        if len(hashes) != config.candidates:
            raise ValueError("local exploration candidate diversity degenerated")
    return states, rows, metadata


def _empty_receipt(
    *,
    config: LocalExplorationConfig,
    contradiction_tensor: Mapping[str, Any],
    contradiction_perturbation: Mapping[str, Any],
    neural_uncertainty: Mapping[str, Any],
    selected_branch: int,
    protected_positions: Sequence[int],
    verifier_policy_sha256: str,
    decoy_review_sha256: str,
    status: str,
    reason: str,
) -> dict[str, Any]:
    payload = {
        "schema": LOCAL_EXPLORATION_SCHEMA,
        "config": config.to_dict(),
        "contradiction_tensor_sha256": contradiction_tensor.get(
            "receipt_sha256",
            "",
        ),
        "contradiction_perturbation_sha256": contradiction_perturbation.get(
            "receipt_sha256",
            "",
        ),
        "neural_uncertainty_sha256": neural_uncertainty.get(
            "receipt_sha256",
            "",
        ),
        "selected_branch": selected_branch,
        "target_position": None,
        "sham_position": None,
        "contradiction_probability": None,
        "predictive_entropy": None,
        "stable_contradiction_probability": None,
        "conditioned_signal": None,
        "protected_positions": sorted(set(protected_positions)),
        "verifier_policy_sha256": verifier_policy_sha256,
        "decoy_review_sha256": decoy_review_sha256,
        "seed_sha256": "",
        "status": status,
        "reason": reason,
        "baseline_state_sha256": "",
        "resulting_state_sha256": "",
        "generator": {},
        "evaluation_order": [],
        "candidates": [],
        "all_candidates_equal_compute": False,
        "all_observations_authoritative": False,
        "repeat_deterministic": False,
        "baseline_control_deterministic": False,
        "generator_replay_proven": False,
        "conditioned_probe_entropy_bits": 0.0,
        "stable_probe_entropy_bits": 0.0,
        "baseline_probe_entropy_bits": 0.0,
        "conditioned_unique_probe_count": 0,
        "regressing_conditioned_candidates": 0,
        "selected_candidate": None,
        "conditioned_beats_controls": False,
        "state_mutation_applied": False,
        "rollback_proven": False,
        "answer_text_stored": False,
        "authority_scope": "none",
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def run_local_exploration(
    *,
    baseline: Any,
    protected_positions: Sequence[int],
    contradiction_tensor: Mapping[str, Any],
    contradiction_perturbation: Mapping[str, Any],
    neural_uncertainty: Mapping[str, Any],
    selected_branch: int,
    config: LocalExplorationConfig,
    verifier_policy_sha256: str,
    decoy_review_sha256: str,
    evaluate: Callable[
        [str, Any, int],
        CounterfactualProbeResult,
    ]
    | None,
    evaluation_unavailable_reason: str = "",
    budget: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Search only an admitted local region; retain nothing without controls."""

    sources = (
        contradiction_tensor,
        contradiction_perturbation,
        neural_uncertainty,
    )
    if (
        any(
            not isinstance(source, Mapping) or not is_sha256(source.get("receipt_sha256"))
            for source in sources
        )
        or type(selected_branch) is not int
    ):
        raise ValueError("local exploration source is invalid")
    if config.mode == DISABLED:
        return baseline, _empty_receipt(
            config=config,
            contradiction_tensor=contradiction_tensor,
            contradiction_perturbation=contradiction_perturbation,
            neural_uncertainty=neural_uncertainty,
            selected_branch=selected_branch,
            protected_positions=protected_positions,
            verifier_policy_sha256=verifier_policy_sha256,
            decoy_review_sha256=decoy_review_sha256,
            status="disabled",
            reason="configured_disabled",
        )
    if contradiction_perturbation.get("state_mutation_applied") is True:
        return baseline, _empty_receipt(
            config=config,
            contradiction_tensor=contradiction_tensor,
            contradiction_perturbation=contradiction_perturbation,
            neural_uncertainty=neural_uncertainty,
            selected_branch=selected_branch,
            protected_positions=protected_positions,
            verifier_policy_sha256=verifier_policy_sha256,
            decoy_review_sha256=decoy_review_sha256,
            status="skipped",
            reason="uncertainty_source_precedes_retained_perturbation",
        )
    try:
        (
            target_position,
            sham_position,
            contradiction_probability,
            predictive_entropy,
            stable_probability,
        ) = _source_signals(
            contradiction_tensor=contradiction_tensor,
            neural_uncertainty=neural_uncertainty,
            selected_branch=selected_branch,
            protected_positions=protected_positions,
            config=config,
        )
    except ValueError as exc:
        return baseline, _empty_receipt(
            config=config,
            contradiction_tensor=contradiction_tensor,
            contradiction_perturbation=contradiction_perturbation,
            neural_uncertainty=neural_uncertainty,
            selected_branch=selected_branch,
            protected_positions=protected_positions,
            verifier_policy_sha256=verifier_policy_sha256,
            decoy_review_sha256=decoy_review_sha256,
            status="skipped",
            reason=str(exc).replace("local exploration ", "").replace(" ", "_"),
        )
    if (
        evaluate is None
        or not is_sha256(verifier_policy_sha256)
        or not is_sha256(decoy_review_sha256)
    ):
        return baseline, _empty_receipt(
            config=config,
            contradiction_tensor=contradiction_tensor,
            contradiction_perturbation=contradiction_perturbation,
            neural_uncertainty=neural_uncertainty,
            selected_branch=selected_branch,
            protected_positions=protected_positions,
            verifier_policy_sha256=verifier_policy_sha256,
            decoy_review_sha256=decoy_review_sha256,
            status="skipped",
            reason=(evaluation_unavailable_reason or "independent_admitted_verifier_unavailable"),
        )
    conditioned_signal = contradiction_probability * predictive_entropy
    seed_sha256 = _seed_sha256(
        contradiction_sha256=contradiction_tensor["receipt_sha256"],
        perturbation_sha256=contradiction_perturbation["receipt_sha256"],
        uncertainty_sha256=neural_uncertainty["receipt_sha256"],
        selected_branch=selected_branch,
        target_position=target_position,
        sham_position=sham_position,
        seed=config.seed,
    )
    try:
        states, candidate_rows, generator = _candidate_states(
            baseline,
            target_position=target_position,
            sham_position=sham_position,
            protected_positions=protected_positions,
            conditioned_signal=conditioned_signal,
            config=config,
            seed_sha256=seed_sha256,
        )
        replay_states, replay_rows, replay_generator = _candidate_states(
            baseline,
            target_position=target_position,
            sham_position=sham_position,
            protected_positions=protected_positions,
            conditioned_signal=conditioned_signal,
            config=config,
            seed_sha256=seed_sha256,
        )
    except ValueError as exc:
        if str(exc) not in {
            "local exploration radius is zero",
            "local exploration candidate diversity degenerated",
        }:
            raise
        return baseline, _empty_receipt(
            config=config,
            contradiction_tensor=contradiction_tensor,
            contradiction_perturbation=contradiction_perturbation,
            neural_uncertainty=neural_uncertainty,
            selected_branch=selected_branch,
            protected_positions=protected_positions,
            verifier_policy_sha256=verifier_policy_sha256,
            decoy_review_sha256=decoy_review_sha256,
            status="skipped",
            reason=(
                "conditioned_radius_is_zero"
                if str(exc) == "local exploration radius is zero"
                else "candidate_diversity_below_dtype_resolution"
            ),
        )
    replay_proven = (
        generator == replay_generator
        and [(row["label"], row["state_sha256"], row["direction_sha256"]) for row in candidate_rows]
        == [(row["label"], row["state_sha256"], row["direction_sha256"]) for row in replay_rows]
        and all(np.array_equal(states[label], replay_states[label]) for label in states)
    )
    if not replay_proven:
        raise RuntimeError("local exploration generator replay failed")
    if budget is not None:
        elements = int(np.asarray(baseline).size)
        candidate_total = len(states)
        budget.charge_tensor_work(
            "local_exploration_candidates",
            # Candidate generation is replayed once for determinism proof;
            # copies, comparisons, hashes, and both target/sham families are
            # conservatively charged rather than treating replay as free.
            element_reads=8 * candidate_total * elements,
            element_writes=2 * candidate_total * elements,
            scalar_ops=8 * candidate_total * elements,
            host_scalar_ops=512 + config.candidates**2,
        )
    labels = sorted(states)
    rng = np.random.default_rng(int.from_bytes(bytes.fromhex(seed_sha256)[8:16], "big"))
    orders: list[list[str]] = []
    for _ in range(config.replicates):
        order = list(labels)
        rng.shuffle(order)
        orders.append(order)
    results: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
    try:
        for replicate, order in enumerate(orders):
            for label in order:
                results[label].append(evaluate(label, states[label], replicate).normalized())
    except Exception as exc:
        payload = _empty_receipt(
            config=config,
            contradiction_tensor=contradiction_tensor,
            contradiction_perturbation=contradiction_perturbation,
            neural_uncertainty=neural_uncertainty,
            selected_branch=selected_branch,
            protected_positions=protected_positions,
            verifier_policy_sha256=verifier_policy_sha256,
            decoy_review_sha256=decoy_review_sha256,
            status="restored",
            reason=f"evaluation_failed:{type(exc).__name__}",
        )
        payload = dict(payload)
        payload.pop("receipt_sha256")
        payload["baseline_state_sha256"] = tensor_sha256(baseline)
        payload["resulting_state_sha256"] = tensor_sha256(baseline)
        payload["rollback_proven"] = True
        return baseline, {**payload, "receipt_sha256": canonical_sha256(payload)}

    observations: dict[str, list[dict[str, Any]]] = {}
    layer_apps: list[int] = []
    repeat_deterministic = True
    authoritative = True
    row_by_label = {row["label"]: row for row in candidate_rows}
    for label in labels:
        normalized = results[label]
        row_by_label[label]["replicates"] = normalized
        observations[label] = [row["observation"] for row in normalized]
        layer_apps.extend(row["layer_apps"] for row in normalized)
        authoritative = authoritative and all(
            row["observation"]["authoritative"] is True for row in normalized
        )
        repeat_deterministic = repeat_deterministic and all(
            (
                row["observation"],
                row["probe_tokens_sha256"],
                row["probe_token_count"],
            )
            == (
                normalized[0]["observation"],
                normalized[0]["probe_tokens_sha256"],
                normalized[0]["probe_token_count"],
            )
            for row in normalized[1:]
        )
    equal_compute = bool(layer_apps) and len(set(layer_apps)) == 1
    baseline_identities = {
        canonical_sha256(
            {
                "observation": row["observation"],
                "probe_tokens_sha256": row["probe_tokens_sha256"],
                "probe_token_count": row["probe_token_count"],
            }
        )
        for index in range(config.candidates)
        for row in results[f"baseline:{index}"]
    }
    baseline_control_deterministic = len(baseline_identities) == 1
    first_probe_hashes = {label: results[label][0]["probe_tokens_sha256"] for label in labels}
    entropy_by_family = {
        family: _entropy_bits(
            [first_probe_hashes[f"{family}:{index}"] for index in range(config.candidates)]
        )
        for family in FAMILIES
    }
    conditioned_unique = len(
        {first_probe_hashes[f"conditioned_target:{index}"] for index in range(config.candidates)}
    )
    baseline_lower = min(
        float(observation["lower_bound"])
        for index in range(config.candidates)
        for observation in observations[f"baseline:{index}"]
    )
    control_upper = max(
        float(observation["upper_bound"])
        for family in ("baseline", "stable_sham")
        for index in range(config.candidates)
        for observation in observations[f"{family}:{index}"]
    )
    conditioned_bounds = []
    for index in range(config.candidates):
        values = observations[f"conditioned_target:{index}"]
        conditioned_bounds.append(
            (
                min(float(value["lower_bound"]) for value in values),
                max(float(value["upper_bound"]) for value in values),
                index,
            )
        )
    selected_lower, _selected_upper, selected_index = max(
        conditioned_bounds,
        key=lambda row: (row[0], -row[2]),
    )
    regressions = sum(
        upper < baseline_lower - float(config.min_verifier_margin) - 1e-12
        for _lower, upper, _index in conditioned_bounds
    )
    beats = (
        authoritative
        and repeat_deterministic
        and baseline_control_deterministic
        and equal_compute
        and replay_proven
        and conditioned_unique >= config.min_unique_conditioned_probes
        and selected_lower > control_upper + float(config.min_verifier_margin) + 1e-12
    )
    baseline_sha256 = tensor_sha256(baseline)
    selected_label = f"conditioned_target:{selected_index}"
    if beats:
        resulting = states[selected_label]
        status = "retained"
        reason = "conditioned_candidate_beats_no_op_and_stable_sham"
        authority_scope = "selected_branch_target_position_only"
    else:
        resulting = baseline
        status = "restored"
        if not authoritative:
            reason = "non_authoritative_verifier_observation"
        elif not repeat_deterministic:
            reason = "counterfactual_repeat_nondeterminism"
        elif not baseline_control_deterministic:
            reason = "baseline_control_order_leakage"
        elif not equal_compute:
            reason = "control_compute_mismatch"
        elif conditioned_unique < config.min_unique_conditioned_probes:
            reason = "conditioned_exploration_did_not_increase_output_diversity"
        else:
            reason = "conditioned_candidate_did_not_beat_controls"
        authority_scope = "none"
    resulting_sha256 = tensor_sha256(resulting)
    payload = {
        "schema": LOCAL_EXPLORATION_SCHEMA,
        "config": config.to_dict(),
        "contradiction_tensor_sha256": contradiction_tensor["receipt_sha256"],
        "contradiction_perturbation_sha256": contradiction_perturbation["receipt_sha256"],
        "neural_uncertainty_sha256": neural_uncertainty["receipt_sha256"],
        "selected_branch": selected_branch,
        "target_position": target_position,
        "sham_position": sham_position,
        "contradiction_probability": round(
            contradiction_probability,
            10,
        ),
        "predictive_entropy": round(predictive_entropy, 10),
        "stable_contradiction_probability": round(stable_probability, 10),
        "conditioned_signal": round(conditioned_signal, 10),
        "protected_positions": sorted(set(protected_positions)),
        "verifier_policy_sha256": verifier_policy_sha256,
        "decoy_review_sha256": decoy_review_sha256,
        "seed_sha256": seed_sha256,
        "status": status,
        "reason": reason,
        "baseline_state_sha256": baseline_sha256,
        "resulting_state_sha256": resulting_sha256,
        "generator": generator,
        "evaluation_order": orders,
        "candidates": candidate_rows,
        "all_candidates_equal_compute": equal_compute,
        "all_observations_authoritative": authoritative,
        "repeat_deterministic": repeat_deterministic,
        "baseline_control_deterministic": (baseline_control_deterministic),
        "generator_replay_proven": replay_proven,
        "conditioned_probe_entropy_bits": entropy_by_family["conditioned_target"],
        "stable_probe_entropy_bits": entropy_by_family["stable_sham"],
        "baseline_probe_entropy_bits": entropy_by_family["baseline"],
        "conditioned_unique_probe_count": conditioned_unique,
        "regressing_conditioned_candidates": regressions,
        "selected_candidate": selected_index,
        "conditioned_beats_controls": beats,
        "state_mutation_applied": beats,
        "rollback_proven": (resulting_sha256 == baseline_sha256 if not beats else False),
        "answer_text_stored": False,
        "authority_scope": authority_scope,
    }
    receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
    return resulting, validate_local_exploration_receipt(
        receipt,
        expected_config=config,
        contradiction_tensor=contradiction_tensor,
        contradiction_perturbation=contradiction_perturbation,
        neural_uncertainty=neural_uncertainty,
        expected_selected_branch=selected_branch,
        expected_protected_positions=protected_positions,
        verifier_policy_sha256=verifier_policy_sha256,
        decoy_review_sha256=decoy_review_sha256,
    )


def validate_local_exploration_receipt(
    value: Any,
    *,
    expected_config: LocalExplorationConfig,
    contradiction_tensor: Mapping[str, Any],
    contradiction_perturbation: Mapping[str, Any],
    neural_uncertainty: Mapping[str, Any],
    expected_selected_branch: int,
    expected_protected_positions: Sequence[int],
    verifier_policy_sha256: str,
    decoy_review_sha256: str,
) -> dict[str, Any]:
    """Reconstruct source conditioning, controls, metrics, and authority."""

    fields = {
        "schema",
        "config",
        "contradiction_tensor_sha256",
        "contradiction_perturbation_sha256",
        "neural_uncertainty_sha256",
        "selected_branch",
        "target_position",
        "sham_position",
        "contradiction_probability",
        "predictive_entropy",
        "stable_contradiction_probability",
        "conditioned_signal",
        "protected_positions",
        "verifier_policy_sha256",
        "decoy_review_sha256",
        "seed_sha256",
        "status",
        "reason",
        "baseline_state_sha256",
        "resulting_state_sha256",
        "generator",
        "evaluation_order",
        "candidates",
        "all_candidates_equal_compute",
        "all_observations_authoritative",
        "repeat_deterministic",
        "baseline_control_deterministic",
        "generator_replay_proven",
        "conditioned_probe_entropy_bits",
        "stable_probe_entropy_bits",
        "baseline_probe_entropy_bits",
        "conditioned_unique_probe_count",
        "regressing_conditioned_candidates",
        "selected_candidate",
        "conditioned_beats_controls",
        "state_mutation_applied",
        "rollback_proven",
        "answer_text_stored",
        "authority_scope",
        "receipt_sha256",
    }
    sources = (
        contradiction_tensor,
        contradiction_perturbation,
        neural_uncertainty,
    )
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or any(
            not isinstance(source, Mapping) or not is_sha256(source.get("receipt_sha256"))
            for source in sources
        )
    ):
        raise ValueError("local exploration receipt fields/source differ")
    receipt = dict(value)
    payload = {key: receipt[key] for key in fields - {"receipt_sha256"}}
    protected = sorted(set(expected_protected_positions))
    if (
        receipt["schema"] != LOCAL_EXPLORATION_SCHEMA
        or receipt["config"] != expected_config.to_dict()
        or receipt["contradiction_tensor_sha256"] != contradiction_tensor["receipt_sha256"]
        or receipt["contradiction_perturbation_sha256"]
        != contradiction_perturbation["receipt_sha256"]
        or receipt["neural_uncertainty_sha256"] != neural_uncertainty["receipt_sha256"]
        or receipt["selected_branch"] != expected_selected_branch
        or receipt["protected_positions"] != protected
        or receipt["verifier_policy_sha256"] != verifier_policy_sha256
        or receipt["decoy_review_sha256"] != decoy_review_sha256
        or receipt["answer_text_stored"] is not False
        or receipt["receipt_sha256"] != canonical_sha256(payload)
    ):
        raise ValueError("local exploration receipt identity is invalid")
    if receipt["status"] in {"disabled", "skipped"}:
        inactive_defaults = {
            "target_position": None,
            "sham_position": None,
            "contradiction_probability": None,
            "predictive_entropy": None,
            "stable_contradiction_probability": None,
            "conditioned_signal": None,
            "seed_sha256": "",
            "baseline_state_sha256": "",
            "resulting_state_sha256": "",
            "generator": {},
            "evaluation_order": [],
            "candidates": [],
            "all_candidates_equal_compute": False,
            "all_observations_authoritative": False,
            "repeat_deterministic": False,
            "baseline_control_deterministic": False,
            "generator_replay_proven": False,
            "conditioned_probe_entropy_bits": 0.0,
            "stable_probe_entropy_bits": 0.0,
            "baseline_probe_entropy_bits": 0.0,
            "conditioned_unique_probe_count": 0,
            "regressing_conditioned_candidates": 0,
            "selected_candidate": None,
            "conditioned_beats_controls": False,
            "state_mutation_applied": False,
            "rollback_proven": False,
            "authority_scope": "none",
        }
        if any(receipt[key] != expected for key, expected in inactive_defaults.items()):
            raise ValueError("inactive local exploration claims evidence or authority")
        if receipt["status"] == "disabled" and (
            expected_config.mode != DISABLED or receipt["reason"] != "configured_disabled"
        ):
            raise ValueError("local exploration disabled state differs")
        if receipt["status"] == "skipped" and (
            expected_config.mode == DISABLED or receipt["reason"] not in SKIP_REASONS
        ):
            raise ValueError("local exploration skip reason is invalid")
        return receipt
    if receipt["status"] == "restored" and not receipt["candidates"]:
        failure_kind = str(receipt["reason"]).removeprefix("evaluation_failed:")
        if (
            not str(receipt["reason"]).startswith("evaluation_failed:")
            or not failure_kind.isidentifier()
            or len(failure_kind) > 128
            or not is_sha256(receipt["baseline_state_sha256"])
            or receipt["resulting_state_sha256"] != receipt["baseline_state_sha256"]
            or receipt["evaluation_order"]
            or receipt["generator"]
            or receipt["target_position"] is not None
            or receipt["sham_position"] is not None
            or receipt["contradiction_probability"] is not None
            or receipt["predictive_entropy"] is not None
            or receipt["stable_contradiction_probability"] is not None
            or receipt["conditioned_signal"] is not None
            or receipt["seed_sha256"] != ""
            or receipt["all_candidates_equal_compute"] is not False
            or receipt["all_observations_authoritative"] is not False
            or receipt["repeat_deterministic"] is not False
            or receipt["baseline_control_deterministic"] is not False
            or receipt["generator_replay_proven"] is not False
            or receipt["conditioned_probe_entropy_bits"] != 0.0
            or receipt["stable_probe_entropy_bits"] != 0.0
            or receipt["baseline_probe_entropy_bits"] != 0.0
            or receipt["conditioned_unique_probe_count"] != 0
            or receipt["regressing_conditioned_candidates"] != 0
            or receipt["selected_candidate"] is not None
            or receipt["state_mutation_applied"] is not False
            or receipt["conditioned_beats_controls"] is not False
            or receipt["rollback_proven"] is not True
            or receipt["authority_scope"] != "none"
        ):
            raise ValueError("failed local exploration did not roll back")
        return receipt
    if receipt["status"] not in {"retained", "restored"}:
        raise ValueError("local exploration status is invalid")
    if expected_config.mode == DISABLED:
        raise ValueError("disabled local exploration was evaluated")
    try:
        target, sham, probability, entropy, stable_probability = _source_signals(
            contradiction_tensor=contradiction_tensor,
            neural_uncertainty=neural_uncertainty,
            selected_branch=expected_selected_branch,
            protected_positions=protected,
            config=expected_config,
        )
    except ValueError as exc:
        raise ValueError("evaluated local exploration source is unavailable") from exc
    expected_seed = _seed_sha256(
        contradiction_sha256=contradiction_tensor["receipt_sha256"],
        perturbation_sha256=contradiction_perturbation["receipt_sha256"],
        uncertainty_sha256=neural_uncertainty["receipt_sha256"],
        selected_branch=expected_selected_branch,
        target_position=target,
        sham_position=sham,
        seed=expected_config.seed,
    )
    if (
        receipt["target_position"] != target
        or receipt["sham_position"] != sham
        or receipt["contradiction_probability"] != round(probability, 10)
        or receipt["predictive_entropy"] != round(entropy, 10)
        or receipt["stable_contradiction_probability"] != round(stable_probability, 10)
        or receipt["conditioned_signal"] != round(probability * entropy, 10)
        or receipt["seed_sha256"] != expected_seed
        or not is_sha256(receipt["baseline_state_sha256"])
        or not is_sha256(receipt["resulting_state_sha256"])
    ):
        raise ValueError("local exploration conditioning reconstruction failed")
    generator = receipt["generator"]
    generator_fields = {
        "position_count",
        "hidden_width",
        "state_dtype",
        "effective_delta_rms",
        "target_slot_rms",
        "sham_slot_rms",
        "direction_sha256s",
        "direction_cosine_matrix",
        "latent_direction_entropy_bits",
    }
    if (
        not isinstance(generator, Mapping)
        or set(generator) != generator_fields
        or type(generator["position_count"]) is not int
        or generator["position_count"] < 2
        or not 0 <= target < generator["position_count"]
        or not 0 <= sham < generator["position_count"]
        or type(generator["hidden_width"]) is not int
        or not expected_config.candidates + 1 <= generator["hidden_width"] <= 16_384
        or not isinstance(generator["state_dtype"], str)
        or not _finite(generator["effective_delta_rms"])
        or float(generator["effective_delta_rms"]) <= 0.0
        or not _finite(generator["target_slot_rms"])
        or float(generator["target_slot_rms"]) <= 0.0
        or not _finite(generator["sham_slot_rms"])
        or float(generator["sham_slot_rms"]) <= 0.0
        or generator["latent_direction_entropy_bits"]
        != round(math.log2(expected_config.candidates), 10)
        or receipt["generator_replay_proven"] is not True
    ):
        raise ValueError("local exploration generator metadata is invalid")
    try:
        state_dtype = np.dtype(generator["state_dtype"])
    except TypeError as exc:
        raise ValueError("local exploration state dtype is invalid") from exc
    if not np.issubdtype(state_dtype, np.floating) or state_dtype.itemsize not in {2, 4, 8}:
        raise ValueError("local exploration state dtype is invalid")
    expected_effective_delta_rms = min(
        float(expected_config.max_relative_delta_rms)
        * probability
        * entropy
        * float(generator["target_slot_rms"]),
        float(expected_config.max_relative_delta_rms) * float(generator["sham_slot_rms"]),
    )
    if not math.isclose(
        float(generator["effective_delta_rms"]),
        expected_effective_delta_rms,
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        raise ValueError("local exploration radius reconstruction failed")
    directions = _directions(
        width=generator["hidden_width"],
        count=expected_config.candidates,
        seed_sha256=expected_seed,
    )
    expected_direction_hashes = [
        tensor_sha256(direction.astype(state_dtype).reshape(1, 1, -1)) for direction in directions
    ]
    if generator["direction_sha256s"] != expected_direction_hashes:
        raise ValueError("local exploration direction identity differs")
    expected_cosines = [
        [round(float(np.dot(left, right)), 12) for right in directions] for left in directions
    ]
    if generator["direction_cosine_matrix"] != expected_cosines:
        raise ValueError("local exploration direction geometry differs")
    labels = [
        f"{family}:{index}" for family in FAMILIES for index in range(expected_config.candidates)
    ]
    if (
        not isinstance(receipt["evaluation_order"], list)
        or len(receipt["evaluation_order"]) != expected_config.replicates
        or any(sorted(order) != sorted(labels) for order in receipt["evaluation_order"])
        or not isinstance(receipt["candidates"], list)
        or len(receipt["candidates"]) != len(labels)
    ):
        raise ValueError("local exploration evaluation coverage differs")
    rows = {row.get("label"): row for row in receipt["candidates"] if isinstance(row, Mapping)}
    if set(rows) != set(labels):
        raise ValueError("local exploration candidate set differs")
    observations: dict[str, list[dict[str, Any]]] = {}
    layer_apps: list[int] = []
    repeat_deterministic = True
    first_hashes: dict[str, str] = {}
    for label in labels:
        row = rows[label]
        family, ordinal_text = label.split(":", 1)
        ordinal = int(ordinal_text)
        required = {
            "label",
            "family",
            "candidate_index",
            "position_index",
            "state_sha256",
            "direction_sha256",
            "delta_rms",
            "relative_delta_rms",
            "changed_positions",
            "protected_positions_unchanged",
            "replicates",
        }
        expected_position = (
            None if family == "baseline" else target if family == "conditioned_target" else sham
        )
        expected_delta_rms = (
            0.0 if expected_position is None else float(generator["effective_delta_rms"])
        )
        expected_slot_rms = (
            1.0
            if expected_position is None
            else float(generator["target_slot_rms"])
            if family == "conditioned_target"
            else float(generator["sham_slot_rms"])
        )
        if (
            set(row) != required
            or row["family"] != family
            or type(row["candidate_index"]) is not int
            or row["candidate_index"] != ordinal
            or row["position_index"] != expected_position
            or not is_sha256(row["state_sha256"])
            or row["direction_sha256"] != expected_direction_hashes[ordinal]
            or not _finite(row["delta_rms"])
            or float(row["delta_rms"]) < 0.0
            or not math.isclose(
                float(row["delta_rms"]),
                expected_delta_rms,
                rel_tol=1e-5,
                abs_tol=1e-8,
            )
            or not _finite(row["relative_delta_rms"])
            or not 0.0
            <= float(row["relative_delta_rms"])
            <= float(expected_config.max_relative_delta_rms) + 1e-6
            or not math.isclose(
                float(row["relative_delta_rms"]),
                expected_delta_rms / expected_slot_rms,
                rel_tol=1e-5,
                abs_tol=1e-8,
            )
            or row["changed_positions"]
            != ([] if expected_position is None else [expected_position])
            or row["protected_positions_unchanged"] is not True
            or not isinstance(row["replicates"], list)
            or len(row["replicates"]) != expected_config.replicates
        ):
            raise ValueError("local exploration candidate geometry differs")
        if family == "baseline":
            if (
                row["state_sha256"] != receipt["baseline_state_sha256"]
                or float(row["delta_rms"]) != 0.0
                or float(row["relative_delta_rms"]) != 0.0
            ):
                raise ValueError("local exploration baseline control changed")
        elif (
            row["state_sha256"] == receipt["baseline_state_sha256"]
            or float(row["delta_rms"]) <= 0.0
        ):
            raise ValueError("local exploration candidate did not change")
        normalized: list[dict[str, Any]] = []
        replicate_identity: list[tuple[Any, str, int]] = []
        for result in row["replicates"]:
            if (
                not isinstance(result, Mapping)
                or set(result)
                != {
                    "probe_tokens_sha256",
                    "probe_token_count",
                    "observation",
                    "layer_apps",
                }
                or not is_sha256(result["probe_tokens_sha256"])
                or type(result["probe_token_count"]) is not int
                or result["probe_token_count"] <= 0
                or type(result["layer_apps"]) is not int
                or result["layer_apps"] <= 0
            ):
                raise ValueError("local exploration probe result is invalid")
            observation = validate_observation(result["observation"])
            normalized.append(observation)
            layer_apps.append(result["layer_apps"])
            replicate_identity.append(
                (
                    observation,
                    result["probe_tokens_sha256"],
                    result["probe_token_count"],
                )
            )
        observations[label] = normalized
        first_hashes[label] = row["replicates"][0]["probe_tokens_sha256"]
        repeat_deterministic = repeat_deterministic and all(
            identity == replicate_identity[0] for identity in replicate_identity[1:]
        )
    authoritative = all(
        observation["authoritative"] is True
        for values in observations.values()
        for observation in values
    )
    equal_compute = bool(layer_apps) and len(set(layer_apps)) == 1
    baseline_identities = {
        canonical_sha256(
            {
                "observation": rows[f"baseline:{index}"]["replicates"][replicate]["observation"],
                "probe_tokens_sha256": rows[f"baseline:{index}"]["replicates"][replicate][
                    "probe_tokens_sha256"
                ],
                "probe_token_count": rows[f"baseline:{index}"]["replicates"][replicate][
                    "probe_token_count"
                ],
            }
        )
        for index in range(expected_config.candidates)
        for replicate in range(expected_config.replicates)
    }
    baseline_control_deterministic = len(baseline_identities) == 1
    for family in ("stable_sham", "conditioned_target"):
        if (
            len(
                {
                    rows[f"{family}:{index}"]["state_sha256"]
                    for index in range(expected_config.candidates)
                }
            )
            != expected_config.candidates
        ):
            raise ValueError("local exploration candidate states collapsed")
    entropy_by_family = {
        family: _entropy_bits(
            [first_hashes[f"{family}:{index}"] for index in range(expected_config.candidates)]
        )
        for family in FAMILIES
    }
    conditioned_unique = len(
        {first_hashes[f"conditioned_target:{index}"] for index in range(expected_config.candidates)}
    )
    baseline_lower = min(
        float(observation["lower_bound"])
        for index in range(expected_config.candidates)
        for observation in observations[f"baseline:{index}"]
    )
    control_upper = max(
        float(observation["upper_bound"])
        for family in ("baseline", "stable_sham")
        for index in range(expected_config.candidates)
        for observation in observations[f"{family}:{index}"]
    )
    bounds = []
    for index in range(expected_config.candidates):
        values = observations[f"conditioned_target:{index}"]
        bounds.append(
            (
                min(float(value["lower_bound"]) for value in values),
                max(float(value["upper_bound"]) for value in values),
                index,
            )
        )
    selected_lower, _selected_upper, selected = max(
        bounds,
        key=lambda row: (row[0], -row[2]),
    )
    regressions = sum(
        upper < baseline_lower - float(expected_config.min_verifier_margin) - 1e-12
        for _lower, upper, _index in bounds
    )
    beats = (
        authoritative
        and repeat_deterministic
        and baseline_control_deterministic
        and equal_compute
        and receipt["generator_replay_proven"] is True
        and conditioned_unique >= expected_config.min_unique_conditioned_probes
        and selected_lower > control_upper + float(expected_config.min_verifier_margin) + 1e-12
    )
    expected_reason = (
        "conditioned_candidate_beats_no_op_and_stable_sham"
        if beats
        else (
            "non_authoritative_verifier_observation"
            if not authoritative
            else (
                "counterfactual_repeat_nondeterminism"
                if not repeat_deterministic
                else (
                    "baseline_control_order_leakage"
                    if not baseline_control_deterministic
                    else (
                        "control_compute_mismatch"
                        if not equal_compute
                        else (
                            "conditioned_exploration_did_not_increase_output_diversity"
                            if conditioned_unique < expected_config.min_unique_conditioned_probes
                            else "conditioned_candidate_did_not_beat_controls"
                        )
                    )
                )
            )
        )
    )
    if (
        receipt["all_candidates_equal_compute"] is not equal_compute
        or receipt["all_observations_authoritative"] is not authoritative
        or receipt["repeat_deterministic"] is not repeat_deterministic
        or receipt["baseline_control_deterministic"] is not baseline_control_deterministic
        or receipt["conditioned_probe_entropy_bits"] != entropy_by_family["conditioned_target"]
        or receipt["stable_probe_entropy_bits"] != entropy_by_family["stable_sham"]
        or receipt["baseline_probe_entropy_bits"] != entropy_by_family["baseline"]
        or receipt["conditioned_unique_probe_count"] != conditioned_unique
        or receipt["regressing_conditioned_candidates"] != regressions
        or receipt["selected_candidate"] != selected
        or receipt["conditioned_beats_controls"] is not beats
        or receipt["state_mutation_applied"] is not beats
        or receipt["reason"] != expected_reason
    ):
        raise ValueError("local exploration decision reconstruction failed")
    selected_state = rows[f"conditioned_target:{selected}"]["state_sha256"]
    if beats:
        if (
            receipt["status"] != "retained"
            or receipt["resulting_state_sha256"] != selected_state
            or receipt["rollback_proven"] is not False
            or receipt["authority_scope"] != "selected_branch_target_position_only"
        ):
            raise ValueError("local exploration retained wrong state")
    elif (
        receipt["status"] != "restored"
        or receipt["resulting_state_sha256"] != receipt["baseline_state_sha256"]
        or receipt["rollback_proven"] is not True
        or receipt["authority_scope"] != "none"
    ):
        raise ValueError("local exploration failed to restore baseline")
    return receipt


__all__ = [
    "COUNTERFACTUAL",
    "DISABLED",
    "FAMILIES",
    "LOCAL_EXPLORATION_SCHEMA",
    "LocalExplorationConfig",
    "run_local_exploration",
    "validate_local_exploration_receipt",
]
