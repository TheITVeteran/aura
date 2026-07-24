"""Calibrated integration of incumbent, corrected, and fused distributions."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
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

HETEROGENEOUS_INTEGRATION_SCHEMA = "aura.rlc.heterogeneous_integration_receipt.v1"
HETEROGENEOUS_DECODE_SCHEMA = "aura.rlc.heterogeneous_decode_receipt.v1"
DISABLED = "disabled"
COUNTERFACTUAL = "counterfactual"
POLICIES = ("select_old", "select_new", "probability_fusion")
MAX_REPLICATES = 3
SKIP_REASONS = {
    "requires_exactly_one_retained_source",
    "retained_source_is_invalid",
    "perturbation_arms_are_invalid",
    "exploration_candidates_are_invalid",
    "source_observations_are_invalid",
    "source_observations_are_incomplete",
    "source_observations_are_not_authoritative",
    "source_bounds_do_not_prefer_correction",
    "independent_admitted_verifier_unavailable",
    "counterfactual_probe_budget_unavailable",
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


def _canonical_receipt(value: Mapping[str, Any]) -> bool:
    digest = value.get("receipt_sha256")
    return is_sha256(digest) and digest == canonical_sha256(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    )


def _output_tokens_sha256(tokens: list[int]) -> str:
    encoded = ("[" + ",".join(str(token) for token in tokens) + "]").encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class HeterogeneousIntegrationConfig:
    """Hard bounds for final incumbent/corrected distribution arbitration."""

    mode: str = COUNTERFACTUAL
    replicates: int = 2
    min_verifier_margin: float = 0.01
    min_js_divergence_bits: float = 0.0001

    def __post_init__(self) -> None:
        if self.mode not in {DISABLED, COUNTERFACTUAL}:
            raise ValueError("heterogeneous integration mode is invalid")
        if type(self.replicates) is not int or not 2 <= self.replicates <= MAX_REPLICATES:
            raise ValueError(
                f"heterogeneous integration replicates must be inside [2, {MAX_REPLICATES}]"
            )
        if (
            not _finite(self.min_verifier_margin)
            or not 0.0 <= float(self.min_verifier_margin) <= 0.25
        ):
            raise ValueError("heterogeneous integration verifier margin must be inside [0, 0.25]")
        if (
            not _finite(self.min_js_divergence_bits)
            or not 0.0 <= float(self.min_js_divergence_bits) <= 1.0
        ):
            raise ValueError("heterogeneous integration JS floor must be inside [0, 1]")

    @classmethod
    def from_value(
        cls,
        value: Mapping[str, Any] | None,
    ) -> HeterogeneousIntegrationConfig:
        raw = dict(value or {})
        allowed = {
            "mode",
            "replicates",
            "min_verifier_margin",
            "min_js_divergence_bits",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"heterogeneous integration has unknown keys: {sorted(unknown)}")
        return cls(
            mode=raw.get("mode", COUNTERFACTUAL),
            replicates=raw.get("replicates", 2),
            min_verifier_margin=raw.get("min_verifier_margin", 0.01),
            min_js_divergence_bits=raw.get(
                "min_js_divergence_bits",
                0.0001,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "replicates": self.replicates,
            "min_verifier_margin": round(
                float(self.min_verifier_margin),
                10,
            ),
            "min_js_divergence_bits": round(
                float(self.min_js_divergence_bits),
                10,
            ),
        }


@dataclass(frozen=True, slots=True)
class IntegrationPolicyResult:
    """One dual-lane policy probe; generated answer text is discarded."""

    probe: CounterfactualProbeResult
    incumbent_state_sha256: str
    corrected_state_sha256: str
    fusion_weight: float
    old_lane_layer_apps: int
    new_lane_layer_apps: int
    old_initial_logits_sha256: str
    new_initial_logits_sha256: str
    old_logits_trace_sha256: str
    new_logits_trace_sha256: str
    policy_logits_trace_sha256: str
    mean_js_divergence_bits: float
    max_js_divergence_bits: float
    divergence_samples: int

    def normalized(self) -> dict[str, Any]:
        probe = self.probe.normalized()
        if (
            not is_sha256(self.incumbent_state_sha256)
            or not is_sha256(self.corrected_state_sha256)
            or self.incumbent_state_sha256 == self.corrected_state_sha256
            or not _finite(self.fusion_weight)
            or not 0.0 <= float(self.fusion_weight) <= 1.0
            or type(self.old_lane_layer_apps) is not int
            or self.old_lane_layer_apps <= 0
            or type(self.new_lane_layer_apps) is not int
            or self.new_lane_layer_apps <= 0
            or self.old_lane_layer_apps != self.new_lane_layer_apps
            or probe["layer_apps"] != self.old_lane_layer_apps + self.new_lane_layer_apps
            or not is_sha256(self.old_initial_logits_sha256)
            or not is_sha256(self.new_initial_logits_sha256)
            or not is_sha256(self.old_logits_trace_sha256)
            or not is_sha256(self.new_logits_trace_sha256)
            or not is_sha256(self.policy_logits_trace_sha256)
            or not _finite(self.mean_js_divergence_bits)
            or not 0.0 <= float(self.mean_js_divergence_bits) <= 1.0
            or not _finite(self.max_js_divergence_bits)
            or not float(self.mean_js_divergence_bits) <= float(self.max_js_divergence_bits) <= 1.0
            or type(self.divergence_samples) is not int
            or self.divergence_samples <= 0
        ):
            raise ValueError("heterogeneous integration policy result is invalid")
        return {
            **probe,
            "incumbent_state_sha256": self.incumbent_state_sha256,
            "corrected_state_sha256": self.corrected_state_sha256,
            "fusion_weight": round(float(self.fusion_weight), 10),
            "old_lane_layer_apps": self.old_lane_layer_apps,
            "new_lane_layer_apps": self.new_lane_layer_apps,
            "old_initial_logits_sha256": self.old_initial_logits_sha256,
            "new_initial_logits_sha256": self.new_initial_logits_sha256,
            "old_logits_trace_sha256": self.old_logits_trace_sha256,
            "new_logits_trace_sha256": self.new_logits_trace_sha256,
            "policy_logits_trace_sha256": self.policy_logits_trace_sha256,
            "mean_js_divergence_bits": round(
                float(self.mean_js_divergence_bits),
                10,
            ),
            "max_js_divergence_bits": round(
                float(self.max_js_divergence_bits),
                10,
            ),
            "divergence_samples": self.divergence_samples,
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
        raise ValueError(f"{name} integration state is invalid")
    return np.array(state, copy=True)


def _source_observations(
    *,
    contradiction_perturbation: Mapping[str, Any],
    local_exploration: Mapping[str, Any],
) -> tuple[
    str,
    str,
    str,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    sources = []
    if contradiction_perturbation.get("state_mutation_applied") is True:
        sources.append("contradiction_perturbation")
    if local_exploration.get("state_mutation_applied") is True:
        sources.append("local_exploration")
    if len(sources) != 1:
        raise ValueError("heterogeneous integration requires exactly one retained source")
    source_kind = sources[0]
    source = (
        contradiction_perturbation
        if source_kind == "contradiction_perturbation"
        else local_exploration
    )
    if (
        not isinstance(source, Mapping)
        or not _canonical_receipt(source)
        or source.get("status") != "retained"
        or not is_sha256(source.get("baseline_state_sha256"))
        or not is_sha256(source.get("resulting_state_sha256"))
        or source["baseline_state_sha256"] == source["resulting_state_sha256"]
    ):
        raise ValueError("heterogeneous integration retained source is invalid")
    old_observations: list[dict[str, Any]] = []
    new_observations: list[dict[str, Any]] = []
    if source_kind == "contradiction_perturbation":
        rows = source.get("arms")
        if (
            not isinstance(rows, list)
            or len(rows) != 3
            or any(not isinstance(row, Mapping) for row in rows)
        ):
            raise ValueError("heterogeneous integration perturbation arms are invalid")
        names = [row.get("name") for row in rows]
        if len(set(names)) != len(names) or set(names) != {
            "no_op",
            "matched_random",
            "contradiction_guided",
        }:
            raise ValueError("heterogeneous integration perturbation arms are invalid")
        by_name = {row["name"]: row for row in rows}
        old = by_name.get("no_op")
        new = by_name.get("contradiction_guided")
    else:
        rows = source.get("candidates")
        selected = source.get("selected_candidate")
        if (
            not isinstance(rows, list)
            or type(selected) is not int
            or selected < 0
            or any(not isinstance(row, Mapping) for row in rows)
        ):
            raise ValueError("heterogeneous integration exploration candidates are invalid")
        labels = [row.get("label") for row in rows]
        if any(not isinstance(label, str) or not label for label in labels) or len(
            set(labels)
        ) != len(labels):
            raise ValueError("heterogeneous integration exploration candidates are invalid")
        by_label = {row["label"]: row for row in rows}
        old = by_label.get("baseline:0")
        new = by_label.get(f"conditioned_target:{selected}")
    for row, destination in (
        (old, old_observations),
        (new, new_observations),
    ):
        replicates = row.get("replicates") if isinstance(row, Mapping) else None
        if not isinstance(replicates, list) or any(
            not isinstance(result, Mapping) for result in replicates
        ):
            raise ValueError("heterogeneous integration source observations are invalid")
        destination.extend(validate_observation(result.get("observation")) for result in replicates)
        if len(destination) < 2:
            raise ValueError("heterogeneous integration source observations are incomplete")
    if any(
        observation["authoritative"] is not True
        for observation in old_observations + new_observations
    ):
        raise ValueError("heterogeneous integration source observations are not authoritative")
    return (
        source_kind,
        source["receipt_sha256"],
        source["baseline_state_sha256"],
        old_observations,
        new_observations,
    )


def _fusion_weight(
    old_observations: list[dict[str, Any]],
    new_observations: list[dict[str, Any]],
) -> tuple[float, float, float]:
    old_upper = max(float(observation["upper_bound"]) for observation in old_observations)
    new_lower = min(float(observation["lower_bound"]) for observation in new_observations)
    if new_lower <= old_upper:
        raise ValueError("heterogeneous integration source bounds do not prefer correction")
    denominator = new_lower + old_upper
    gamma = 1.0 if denominator <= 1e-12 else new_lower / denominator
    return round(gamma, 10), round(old_upper, 10), round(new_lower, 10)


def _empty_receipt(
    *,
    config: HeterogeneousIntegrationConfig,
    contradiction_perturbation: Mapping[str, Any],
    local_exploration: Mapping[str, Any],
    verifier_policy_sha256: str,
    decoy_review_sha256: str,
    status: str,
    reason: str,
) -> dict[str, Any]:
    payload = {
        "schema": HETEROGENEOUS_INTEGRATION_SCHEMA,
        "config": config.to_dict(),
        "contradiction_perturbation_sha256": contradiction_perturbation.get(
            "receipt_sha256",
            "",
        ),
        "local_exploration_sha256": local_exploration.get(
            "receipt_sha256",
            "",
        ),
        "source_kind": "none",
        "source_receipt_sha256": "",
        "source_baseline_state_sha256": "",
        "incumbent_state_sha256": "",
        "corrected_state_sha256": "",
        "source_old_upper_bound": None,
        "source_new_lower_bound": None,
        "fusion_weight": None,
        "verifier_policy_sha256": verifier_policy_sha256,
        "decoy_review_sha256": decoy_review_sha256,
        "status": status,
        "reason": reason,
        "evaluation_order": [],
        "policies": [],
        "all_policies_equal_compute": False,
        "all_lanes_equal_compute": False,
        "all_observations_authoritative": False,
        "repeat_deterministic": False,
        "shared_lane_evidence": False,
        "mean_js_divergence_bits": None,
        "max_js_divergence_bits": None,
        "selected_policy": "select_old",
        "fusion_beats_selection": False,
        "new_beats_old": False,
        "decode_policy_applied": False,
        "rollback_proven": False,
        "answer_text_stored": False,
        "authority_scope": "none",
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def run_heterogeneous_integration(
    *,
    incumbent_state: Any,
    corrected_state: Any,
    contradiction_perturbation: Mapping[str, Any],
    local_exploration: Mapping[str, Any],
    config: HeterogeneousIntegrationConfig,
    verifier_policy_sha256: str,
    decoy_review_sha256: str,
    evaluate: Callable[
        [str, Any, Any, float, int],
        IntegrationPolicyResult,
    ]
    | None,
    evaluation_unavailable_reason: str = "",
    budget: Any | None = None,
) -> tuple[Any, str, dict[str, Any]]:
    """Admit selection or fusion only after equal-compute policy comparison."""

    sources = (contradiction_perturbation, local_exploration)
    if any(
        not isinstance(source, Mapping) or not is_sha256(source.get("receipt_sha256"))
        for source in sources
    ):
        raise ValueError("heterogeneous integration source is invalid")
    if config.mode == DISABLED:
        return (
            incumbent_state,
            "select_old",
            _empty_receipt(
                config=config,
                contradiction_perturbation=contradiction_perturbation,
                local_exploration=local_exploration,
                verifier_policy_sha256=verifier_policy_sha256,
                decoy_review_sha256=decoy_review_sha256,
                status="disabled",
                reason="configured_disabled",
            ),
        )
    try:
        (
            source_kind,
            source_receipt_sha256,
            source_baseline_sha256,
            old_observations,
            new_observations,
        ) = _source_observations(
            contradiction_perturbation=contradiction_perturbation,
            local_exploration=local_exploration,
        )
        gamma, old_upper, new_lower = _fusion_weight(
            old_observations,
            new_observations,
        )
    except ValueError as exc:
        return (
            incumbent_state,
            "select_old",
            _empty_receipt(
                config=config,
                contradiction_perturbation=contradiction_perturbation,
                local_exploration=local_exploration,
                verifier_policy_sha256=verifier_policy_sha256,
                decoy_review_sha256=decoy_review_sha256,
                status="skipped",
                reason=str(exc)
                .replace("heterogeneous integration ", "")
                .replace(
                    " ",
                    "_",
                ),
            ),
        )
    incumbent = _as_state(incumbent_state, name="incumbent")
    corrected = _as_state(corrected_state, name="corrected")
    incumbent_sha256 = tensor_sha256(incumbent)
    corrected_sha256 = tensor_sha256(corrected)
    if (
        incumbent.shape != corrected.shape
        or incumbent.dtype != corrected.dtype
        or incumbent_sha256 != source_baseline_sha256
        or corrected_sha256 == incumbent_sha256
    ):
        raise ValueError("heterogeneous integration state lineage is invalid")
    source = (
        contradiction_perturbation
        if source_kind == "contradiction_perturbation"
        else local_exploration
    )
    if corrected_sha256 != source["resulting_state_sha256"]:
        raise ValueError("heterogeneous integration corrected state differs")
    if (
        evaluate is None
        or not is_sha256(verifier_policy_sha256)
        or not is_sha256(decoy_review_sha256)
    ):
        return (
            incumbent,
            "select_old",
            _empty_receipt(
                config=config,
                contradiction_perturbation=contradiction_perturbation,
                local_exploration=local_exploration,
                verifier_policy_sha256=verifier_policy_sha256,
                decoy_review_sha256=decoy_review_sha256,
                status="skipped",
                reason=(
                    evaluation_unavailable_reason or "independent_admitted_verifier_unavailable"
                ),
            ),
        )
    seed_sha256 = hashlib.sha256(
        (f"{source_receipt_sha256}:{incumbent_sha256}:{corrected_sha256}:{gamma:.10f}").encode(
            "ascii"
        )
    ).hexdigest()
    rng = np.random.default_rng(int.from_bytes(bytes.fromhex(seed_sha256)[:8], "big"))
    orders: list[list[str]] = []
    for _ in range(config.replicates):
        order = list(POLICIES)
        rng.shuffle(order)
        orders.append(order)
    results: dict[str, list[dict[str, Any]]] = {policy: [] for policy in POLICIES}
    try:
        for replicate, order in enumerate(orders):
            for policy in order:
                results[policy].append(
                    evaluate(
                        policy,
                        incumbent,
                        corrected,
                        gamma,
                        replicate,
                    ).normalized()
                )
                evaluated = results[policy][-1]
                if (
                    evaluated["incumbent_state_sha256"] != incumbent_sha256
                    or evaluated["corrected_state_sha256"] != corrected_sha256
                    or evaluated["fusion_weight"] != gamma
                ):
                    raise ValueError("heterogeneous policy evidence state lineage differs")
    except Exception as exc:
        receipt = _empty_receipt(
            config=config,
            contradiction_perturbation=contradiction_perturbation,
            local_exploration=local_exploration,
            verifier_policy_sha256=verifier_policy_sha256,
            decoy_review_sha256=decoy_review_sha256,
            status="abstained",
            reason=f"evaluation_failed:{type(exc).__name__}",
        )
        payload = dict(receipt)
        payload.pop("receipt_sha256")
        payload.update(
            {
                "source_kind": source_kind,
                "source_receipt_sha256": source_receipt_sha256,
                "source_baseline_state_sha256": source_baseline_sha256,
                "incumbent_state_sha256": incumbent_sha256,
                "corrected_state_sha256": corrected_sha256,
                "source_old_upper_bound": old_upper,
                "source_new_lower_bound": new_lower,
                "fusion_weight": gamma,
                "rollback_proven": True,
            }
        )
        return (
            incumbent,
            "select_old",
            {
                **payload,
                "receipt_sha256": canonical_sha256(payload),
            },
        )
    policy_rows = []
    layer_apps: list[int] = []
    lane_apps: list[int] = []
    authoritative = True
    repeat_deterministic = True
    lane_evidence: set[tuple[Any, ...]] = set()
    for policy in POLICIES:
        normalized = results[policy]
        layer_apps.extend(row["layer_apps"] for row in normalized)
        lane_apps.extend(
            lane
            for row in normalized
            for lane in (
                row["old_lane_layer_apps"],
                row["new_lane_layer_apps"],
            )
        )
        authoritative = authoritative and all(
            row["observation"]["authoritative"] is True for row in normalized
        )
        identity_fields = (
            "probe_tokens_sha256",
            "probe_token_count",
            "observation",
            "incumbent_state_sha256",
            "corrected_state_sha256",
            "fusion_weight",
            "old_initial_logits_sha256",
            "new_initial_logits_sha256",
            "old_logits_trace_sha256",
            "new_logits_trace_sha256",
            "policy_logits_trace_sha256",
            "mean_js_divergence_bits",
            "max_js_divergence_bits",
            "divergence_samples",
            "layer_apps",
            "old_lane_layer_apps",
            "new_lane_layer_apps",
        )
        repeat_deterministic = repeat_deterministic and all(
            tuple(row[field] for field in identity_fields)
            == tuple(normalized[0][field] for field in identity_fields)
            for row in normalized[1:]
        )
        lane_evidence.update(
            (
                replicate,
                row["old_initial_logits_sha256"],
                row["new_initial_logits_sha256"],
            )
            for replicate, row in enumerate(normalized)
        )
        policy_rows.append(
            {
                "policy": policy,
                "replicates": normalized,
            }
        )
    equal_compute = bool(layer_apps) and len(set(layer_apps)) == 1
    equal_lanes = bool(lane_apps) and len(set(lane_apps)) == 1
    shared_lane_evidence = len(lane_evidence) == config.replicates
    divergence_rows = results["probability_fusion"]
    mean_js = round(
        sum(float(row["mean_js_divergence_bits"]) for row in divergence_rows)
        / len(divergence_rows),
        10,
    )
    max_js = round(
        max(float(row["max_js_divergence_bits"]) for row in divergence_rows),
        10,
    )
    bounds = {
        policy: (
            min(float(row["observation"]["lower_bound"]) for row in results[policy]),
            max(float(row["observation"]["upper_bound"]) for row in results[policy]),
        )
        for policy in POLICIES
    }
    common = (
        authoritative
        and repeat_deterministic
        and equal_compute
        and equal_lanes
        and shared_lane_evidence
        and mean_js + 1e-12 >= float(config.min_js_divergence_bits)
    )
    margin = float(config.min_verifier_margin)
    fusion_beats = (
        common
        and bounds["probability_fusion"][0]
        > max(
            bounds["select_old"][1],
            bounds["select_new"][1],
        )
        + margin
        + 1e-12
    )
    new_beats_old = common and bounds["select_new"][0] > bounds["select_old"][1] + margin + 1e-12
    if fusion_beats:
        selected_policy = "probability_fusion"
        selected_state = corrected
        status = "selected"
        reason = "fusion_lower_bound_beats_both_selection_policies"
        authority_scope = "final_decode_distribution_only"
    elif new_beats_old:
        selected_policy = "select_new"
        selected_state = corrected
        status = "selected"
        reason = "corrected_selection_lower_bound_beats_incumbent"
        authority_scope = "corrected_state_and_final_decode"
    else:
        selected_policy = "select_old"
        selected_state = incumbent
        status = "abstained"
        if not authoritative:
            reason = "non_authoritative_verifier_observation"
        elif not repeat_deterministic:
            reason = "policy_repeat_nondeterminism"
        elif not equal_compute or not equal_lanes:
            reason = "policy_compute_mismatch"
        elif not shared_lane_evidence:
            reason = "policy_lane_evidence_mismatch"
        elif mean_js + 1e-12 < float(config.min_js_divergence_bits):
            reason = "candidate_distributions_not_distinct"
        else:
            reason = "no_policy_earned_separated_bounds"
        authority_scope = "none"
    payload = {
        "schema": HETEROGENEOUS_INTEGRATION_SCHEMA,
        "config": config.to_dict(),
        "contradiction_perturbation_sha256": contradiction_perturbation["receipt_sha256"],
        "local_exploration_sha256": local_exploration["receipt_sha256"],
        "source_kind": source_kind,
        "source_receipt_sha256": source_receipt_sha256,
        "source_baseline_state_sha256": source_baseline_sha256,
        "incumbent_state_sha256": incumbent_sha256,
        "corrected_state_sha256": corrected_sha256,
        "source_old_upper_bound": old_upper,
        "source_new_lower_bound": new_lower,
        "fusion_weight": gamma,
        "verifier_policy_sha256": verifier_policy_sha256,
        "decoy_review_sha256": decoy_review_sha256,
        "status": status,
        "reason": reason,
        "evaluation_order": orders,
        "policies": policy_rows,
        "all_policies_equal_compute": equal_compute,
        "all_lanes_equal_compute": equal_lanes,
        "all_observations_authoritative": authoritative,
        "repeat_deterministic": repeat_deterministic,
        "shared_lane_evidence": shared_lane_evidence,
        "mean_js_divergence_bits": mean_js,
        "max_js_divergence_bits": max_js,
        "selected_policy": selected_policy,
        "fusion_beats_selection": fusion_beats,
        "new_beats_old": new_beats_old,
        "decode_policy_applied": selected_policy != "select_old",
        "rollback_proven": selected_policy == "select_old",
        "answer_text_stored": False,
        "authority_scope": authority_scope,
    }
    receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
    validate_heterogeneous_integration_receipt(
        receipt,
        expected_config=config,
        contradiction_perturbation=contradiction_perturbation,
        local_exploration=local_exploration,
        verifier_policy_sha256=verifier_policy_sha256,
        decoy_review_sha256=decoy_review_sha256,
    )
    return selected_state, selected_policy, receipt


def validate_heterogeneous_integration_receipt(
    value: Any,
    *,
    expected_config: HeterogeneousIntegrationConfig,
    contradiction_perturbation: Mapping[str, Any],
    local_exploration: Mapping[str, Any],
    verifier_policy_sha256: str,
    decoy_review_sha256: str,
) -> dict[str, Any]:
    """Reconstruct source weighting, dual-lane evidence, and policy authority."""

    fields = {
        "schema",
        "config",
        "contradiction_perturbation_sha256",
        "local_exploration_sha256",
        "source_kind",
        "source_receipt_sha256",
        "source_baseline_state_sha256",
        "incumbent_state_sha256",
        "corrected_state_sha256",
        "source_old_upper_bound",
        "source_new_lower_bound",
        "fusion_weight",
        "verifier_policy_sha256",
        "decoy_review_sha256",
        "status",
        "reason",
        "evaluation_order",
        "policies",
        "all_policies_equal_compute",
        "all_lanes_equal_compute",
        "all_observations_authoritative",
        "repeat_deterministic",
        "shared_lane_evidence",
        "mean_js_divergence_bits",
        "max_js_divergence_bits",
        "selected_policy",
        "fusion_beats_selection",
        "new_beats_old",
        "decode_policy_applied",
        "rollback_proven",
        "answer_text_stored",
        "authority_scope",
        "receipt_sha256",
    }
    sources = (contradiction_perturbation, local_exploration)
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or any(
            not isinstance(source, Mapping) or not is_sha256(source.get("receipt_sha256"))
            for source in sources
        )
    ):
        raise ValueError("heterogeneous integration receipt fields/source differ")
    receipt = dict(value)
    payload = {key: receipt[key] for key in fields - {"receipt_sha256"}}
    if (
        receipt["schema"] != HETEROGENEOUS_INTEGRATION_SCHEMA
        or receipt["config"] != expected_config.to_dict()
        or receipt["contradiction_perturbation_sha256"]
        != contradiction_perturbation["receipt_sha256"]
        or receipt["local_exploration_sha256"] != local_exploration["receipt_sha256"]
        or receipt["verifier_policy_sha256"] != verifier_policy_sha256
        or receipt["decoy_review_sha256"] != decoy_review_sha256
        or receipt["answer_text_stored"] is not False
        or receipt["receipt_sha256"] != canonical_sha256(payload)
    ):
        raise ValueError("heterogeneous integration receipt identity is invalid")
    if receipt["status"] in {"disabled", "skipped"}:
        empty = _empty_receipt(
            config=expected_config,
            contradiction_perturbation=contradiction_perturbation,
            local_exploration=local_exploration,
            verifier_policy_sha256=verifier_policy_sha256,
            decoy_review_sha256=decoy_review_sha256,
            status=receipt["status"],
            reason=receipt["reason"],
        )
        if receipt != empty:
            raise ValueError("inactive heterogeneous integration claims evidence")
        if receipt["status"] == "disabled" and (
            expected_config.mode != DISABLED or receipt["reason"] != "configured_disabled"
        ):
            raise ValueError("heterogeneous integration disabled state differs")
        if receipt["status"] == "skipped" and expected_config.mode == DISABLED:
            raise ValueError("heterogeneous integration skip state differs")
        if receipt["status"] == "skipped" and receipt["reason"] not in SKIP_REASONS:
            raise ValueError("heterogeneous integration skip reason is invalid")
        return receipt
    try:
        (
            source_kind,
            source_receipt_sha256,
            source_baseline_sha256,
            old_observations,
            new_observations,
        ) = _source_observations(
            contradiction_perturbation=contradiction_perturbation,
            local_exploration=local_exploration,
        )
        gamma, old_upper, new_lower = _fusion_weight(
            old_observations,
            new_observations,
        )
    except ValueError as exc:
        raise ValueError("evaluated heterogeneous integration source is unavailable") from exc
    if (
        receipt["source_kind"] != source_kind
        or receipt["source_receipt_sha256"] != source_receipt_sha256
        or receipt["source_baseline_state_sha256"] != source_baseline_sha256
        or receipt["source_old_upper_bound"] != old_upper
        or receipt["source_new_lower_bound"] != new_lower
        or receipt["fusion_weight"] != gamma
        or not is_sha256(receipt["incumbent_state_sha256"])
        or receipt["incumbent_state_sha256"] != source_baseline_sha256
        or not is_sha256(receipt["corrected_state_sha256"])
        or receipt["corrected_state_sha256"] == receipt["incumbent_state_sha256"]
    ):
        raise ValueError("heterogeneous integration source reconstruction failed")
    if receipt["status"] == "abstained" and not receipt["policies"]:
        failure = str(receipt["reason"]).removeprefix("evaluation_failed:")
        if (
            not str(receipt["reason"]).startswith("evaluation_failed:")
            or not failure.isidentifier()
            or len(failure) > 128
            or receipt["evaluation_order"]
            or receipt["all_policies_equal_compute"] is not False
            or receipt["all_lanes_equal_compute"] is not False
            or receipt["all_observations_authoritative"] is not False
            or receipt["repeat_deterministic"] is not False
            or receipt["shared_lane_evidence"] is not False
            or receipt["mean_js_divergence_bits"] is not None
            or receipt["max_js_divergence_bits"] is not None
            or receipt["selected_policy"] != "select_old"
            or receipt["fusion_beats_selection"] is not False
            or receipt["new_beats_old"] is not False
            or receipt["decode_policy_applied"] is not False
            or receipt["rollback_proven"] is not True
            or receipt["authority_scope"] != "none"
        ):
            raise ValueError("failed heterogeneous integration did not abstain")
        return receipt
    if receipt["status"] not in {"selected", "abstained"}:
        raise ValueError("heterogeneous integration status is invalid")
    labels = list(POLICIES)
    if (
        not isinstance(receipt["evaluation_order"], list)
        or len(receipt["evaluation_order"]) != expected_config.replicates
        or any(sorted(order) != sorted(labels) for order in receipt["evaluation_order"])
        or not isinstance(receipt["policies"], list)
        or len(receipt["policies"]) != len(labels)
    ):
        raise ValueError("heterogeneous integration evaluation coverage differs")
    rows = {row.get("policy"): row for row in receipt["policies"] if isinstance(row, Mapping)}
    if set(rows) != set(labels):
        raise ValueError("heterogeneous integration policy set differs")
    results: dict[str, list[dict[str, Any]]] = {}
    layer_apps: list[int] = []
    lane_apps: list[int] = []
    authoritative = True
    repeat_deterministic = True
    lane_evidence: set[tuple[Any, ...]] = set()
    result_fields = {
        "probe_tokens_sha256",
        "probe_token_count",
        "observation",
        "layer_apps",
        "incumbent_state_sha256",
        "corrected_state_sha256",
        "fusion_weight",
        "old_lane_layer_apps",
        "new_lane_layer_apps",
        "old_initial_logits_sha256",
        "new_initial_logits_sha256",
        "old_logits_trace_sha256",
        "new_logits_trace_sha256",
        "policy_logits_trace_sha256",
        "mean_js_divergence_bits",
        "max_js_divergence_bits",
        "divergence_samples",
    }
    identity_fields = tuple(sorted(result_fields))
    for policy in labels:
        row = rows[policy]
        if (
            set(row) != {"policy", "replicates"}
            or row["policy"] != policy
            or not isinstance(row["replicates"], list)
            or len(row["replicates"]) != expected_config.replicates
        ):
            raise ValueError("heterogeneous integration policy row is invalid")
        normalized = []
        for result in row["replicates"]:
            if not isinstance(result, Mapping) or set(result) != result_fields:
                raise ValueError("heterogeneous integration policy result fields differ")
            probe = CounterfactualProbeResult(
                probe_tokens_sha256=result["probe_tokens_sha256"],
                probe_token_count=result["probe_token_count"],
                observation=result["observation"],
                layer_apps=result["layer_apps"],
            )
            normalized.append(
                IntegrationPolicyResult(
                    probe=probe,
                    incumbent_state_sha256=result["incumbent_state_sha256"],
                    corrected_state_sha256=result["corrected_state_sha256"],
                    fusion_weight=result["fusion_weight"],
                    old_lane_layer_apps=result["old_lane_layer_apps"],
                    new_lane_layer_apps=result["new_lane_layer_apps"],
                    old_initial_logits_sha256=result["old_initial_logits_sha256"],
                    new_initial_logits_sha256=result["new_initial_logits_sha256"],
                    old_logits_trace_sha256=result["old_logits_trace_sha256"],
                    new_logits_trace_sha256=result["new_logits_trace_sha256"],
                    policy_logits_trace_sha256=result["policy_logits_trace_sha256"],
                    mean_js_divergence_bits=result["mean_js_divergence_bits"],
                    max_js_divergence_bits=result["max_js_divergence_bits"],
                    divergence_samples=result["divergence_samples"],
                ).normalized()
            )
            if (
                normalized[-1]["incumbent_state_sha256"] != receipt["incumbent_state_sha256"]
                or normalized[-1]["corrected_state_sha256"] != receipt["corrected_state_sha256"]
                or normalized[-1]["fusion_weight"] != receipt["fusion_weight"]
            ):
                raise ValueError("heterogeneous integration policy state lineage differs")
        results[policy] = normalized
        layer_apps.extend(result["layer_apps"] for result in normalized)
        lane_apps.extend(
            lane
            for result in normalized
            for lane in (
                result["old_lane_layer_apps"],
                result["new_lane_layer_apps"],
            )
        )
        authoritative = authoritative and all(
            result["observation"]["authoritative"] is True for result in normalized
        )
        repeat_deterministic = repeat_deterministic and all(
            tuple(result[field] for field in identity_fields)
            == tuple(normalized[0][field] for field in identity_fields)
            for result in normalized[1:]
        )
        lane_evidence.update(
            (
                replicate,
                result["old_initial_logits_sha256"],
                result["new_initial_logits_sha256"],
            )
            for replicate, result in enumerate(normalized)
        )
    equal_compute = len(set(layer_apps)) == 1
    equal_lanes = len(set(lane_apps)) == 1
    shared_lane_evidence = len(lane_evidence) == expected_config.replicates
    divergence = results["probability_fusion"]
    mean_js = round(
        sum(float(row["mean_js_divergence_bits"]) for row in divergence) / len(divergence),
        10,
    )
    max_js = round(
        max(float(row["max_js_divergence_bits"]) for row in divergence),
        10,
    )
    bounds = {
        policy: (
            min(float(row["observation"]["lower_bound"]) for row in results[policy]),
            max(float(row["observation"]["upper_bound"]) for row in results[policy]),
        )
        for policy in labels
    }
    common = (
        authoritative
        and repeat_deterministic
        and equal_compute
        and equal_lanes
        and shared_lane_evidence
        and mean_js + 1e-12 >= float(expected_config.min_js_divergence_bits)
    )
    margin = float(expected_config.min_verifier_margin)
    fusion_beats = (
        common
        and bounds["probability_fusion"][0]
        > max(
            bounds["select_old"][1],
            bounds["select_new"][1],
        )
        + margin
        + 1e-12
    )
    new_beats_old = common and bounds["select_new"][0] > bounds["select_old"][1] + margin + 1e-12
    if fusion_beats:
        selected = "probability_fusion"
        status = "selected"
        reason = "fusion_lower_bound_beats_both_selection_policies"
        authority = "final_decode_distribution_only"
    elif new_beats_old:
        selected = "select_new"
        status = "selected"
        reason = "corrected_selection_lower_bound_beats_incumbent"
        authority = "corrected_state_and_final_decode"
    else:
        selected = "select_old"
        status = "abstained"
        if not authoritative:
            reason = "non_authoritative_verifier_observation"
        elif not repeat_deterministic:
            reason = "policy_repeat_nondeterminism"
        elif not equal_compute or not equal_lanes:
            reason = "policy_compute_mismatch"
        elif not shared_lane_evidence:
            reason = "policy_lane_evidence_mismatch"
        elif mean_js + 1e-12 < float(expected_config.min_js_divergence_bits):
            reason = "candidate_distributions_not_distinct"
        else:
            reason = "no_policy_earned_separated_bounds"
        authority = "none"
    if (
        receipt["all_policies_equal_compute"] is not equal_compute
        or receipt["all_lanes_equal_compute"] is not equal_lanes
        or receipt["all_observations_authoritative"] is not authoritative
        or receipt["repeat_deterministic"] is not repeat_deterministic
        or receipt["shared_lane_evidence"] is not shared_lane_evidence
        or receipt["mean_js_divergence_bits"] != mean_js
        or receipt["max_js_divergence_bits"] != max_js
        or receipt["selected_policy"] != selected
        or receipt["fusion_beats_selection"] is not fusion_beats
        or receipt["new_beats_old"] is not new_beats_old
        or receipt["status"] != status
        or receipt["reason"] != reason
        or receipt["decode_policy_applied"] is not (selected != "select_old")
        or receipt["rollback_proven"] is not (selected == "select_old")
        or receipt["authority_scope"] != authority
    ):
        raise ValueError("heterogeneous integration decision reconstruction failed")
    return receipt


def build_heterogeneous_decode_receipt(
    *,
    integration: Mapping[str, Any],
    output_tokens: list[int],
    termination: str,
    first_logits_sha256: str,
    fusion_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the policy comparison to the user-visible final decode."""

    if (
        not isinstance(integration, Mapping)
        or not _canonical_receipt(integration)
        or integration.get("status") not in {"selected", "abstained"}
        or not integration.get("policies")
        or integration.get("selected_policy") not in POLICIES
        or not isinstance(output_tokens, list)
        or any(type(token) is not int or token < 0 for token in output_tokens)
        or not isinstance(termination, str)
        or not termination
        or len(termination) > 128
        or not is_sha256(first_logits_sha256)
    ):
        raise ValueError("heterogeneous final decode identity is invalid")
    selected_policy = integration["selected_policy"]
    audit = dict(fusion_audit or {})
    if selected_policy == "probability_fusion":
        required = {
            "incumbent_state_sha256",
            "corrected_state_sha256",
            "fusion_weight",
            "old_lane_layer_apps",
            "new_lane_layer_apps",
            "old_initial_logits_sha256",
            "new_initial_logits_sha256",
            "policy_initial_logits_sha256",
            "old_logits_trace_sha256",
            "new_logits_trace_sha256",
            "policy_logits_trace_sha256",
            "mean_js_divergence_bits",
            "max_js_divergence_bits",
            "divergence_samples",
        }
        if (
            set(audit) != required
            or audit["incumbent_state_sha256"] != integration["incumbent_state_sha256"]
            or audit["corrected_state_sha256"] != integration["corrected_state_sha256"]
            or audit["fusion_weight"] != integration["fusion_weight"]
            or audit["old_lane_layer_apps"] != audit["new_lane_layer_apps"]
            or type(audit["old_lane_layer_apps"]) is not int
            or audit["old_lane_layer_apps"] <= 0
            or any(
                not is_sha256(audit[field])
                for field in (
                    "old_initial_logits_sha256",
                    "new_initial_logits_sha256",
                    "policy_initial_logits_sha256",
                    "old_logits_trace_sha256",
                    "new_logits_trace_sha256",
                    "policy_logits_trace_sha256",
                )
            )
            or audit["policy_initial_logits_sha256"] != first_logits_sha256
            or not _finite(audit["mean_js_divergence_bits"])
            or not 0.0 <= float(audit["mean_js_divergence_bits"]) <= 1.0
            or not _finite(audit["max_js_divergence_bits"])
            or not float(audit["mean_js_divergence_bits"])
            <= float(audit["max_js_divergence_bits"])
            <= 1.0
            or type(audit["divergence_samples"]) is not int
            or audit["divergence_samples"] <= 0
        ):
            raise ValueError("heterogeneous fusion execution audit is invalid")
        execution_kind = "dual_lane_probability_fusion"
    else:
        if audit:
            raise ValueError("single-lane heterogeneous decode carries fusion evidence")
        execution_kind = (
            "corrected_single_lane" if selected_policy == "select_new" else "incumbent_single_lane"
        )
    payload = {
        "schema": HETEROGENEOUS_DECODE_SCHEMA,
        "integration_receipt_sha256": integration["receipt_sha256"],
        "selected_policy": selected_policy,
        "execution_kind": execution_kind,
        "first_logits_sha256": first_logits_sha256,
        "output_tokens_sha256": _output_tokens_sha256(output_tokens),
        "output_token_count": len(output_tokens),
        "termination": termination,
        "fusion_audit": audit,
        "answer_text_stored": False,
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def validate_heterogeneous_decode_receipt(
    value: Any,
    *,
    integration: Mapping[str, Any],
    expected_output_tokens: Any = ...,
) -> dict[str, Any]:
    """Validate final policy execution without retaining decoded text."""

    active = (
        isinstance(integration, Mapping)
        and integration.get("status") in {"selected", "abstained"}
        and bool(integration.get("policies"))
    )
    if not active:
        if value not in ({}, None):
            raise ValueError("inactive heterogeneous integration claims final decode")
        return {}
    fields = {
        "schema",
        "integration_receipt_sha256",
        "selected_policy",
        "execution_kind",
        "first_logits_sha256",
        "output_tokens_sha256",
        "output_token_count",
        "termination",
        "fusion_audit",
        "answer_text_stored",
        "receipt_sha256",
    }
    if (
        not _canonical_receipt(integration)
        or not isinstance(value, Mapping)
        or set(value) != fields
    ):
        raise ValueError("heterogeneous final decode receipt fields differ")
    receipt = dict(value)
    payload = {key: receipt[key] for key in fields - {"receipt_sha256"}}
    selected = integration.get("selected_policy")
    expected_kind = (
        "dual_lane_probability_fusion"
        if selected == "probability_fusion"
        else "corrected_single_lane"
        if selected == "select_new"
        else "incumbent_single_lane"
    )
    if (
        selected not in POLICIES
        or receipt["schema"] != HETEROGENEOUS_DECODE_SCHEMA
        or receipt["integration_receipt_sha256"] != integration["receipt_sha256"]
        or receipt["selected_policy"] != selected
        or receipt["execution_kind"] != expected_kind
        or not is_sha256(receipt["first_logits_sha256"])
        or not is_sha256(receipt["output_tokens_sha256"])
        or type(receipt["output_token_count"]) is not int
        or receipt["output_token_count"] < 0
        or not isinstance(receipt["termination"], str)
        or not receipt["termination"]
        or len(receipt["termination"]) > 128
        or receipt["answer_text_stored"] is not False
        or receipt["receipt_sha256"] != canonical_sha256(payload)
    ):
        raise ValueError("heterogeneous final decode identity differs")
    audit = receipt["fusion_audit"]
    if selected == "probability_fusion":
        # Reuse the builder's strict audit validation with a synthetic token
        # vector, then compare only the canonical public fields.
        rebuilt = build_heterogeneous_decode_receipt(
            integration=integration,
            output_tokens=[0] * receipt["output_token_count"],
            termination=receipt["termination"],
            first_logits_sha256=receipt["first_logits_sha256"],
            fusion_audit=audit,
        )
        if (
            rebuilt["execution_kind"] != receipt["execution_kind"]
            or rebuilt["fusion_audit"] != audit
        ):
            raise ValueError("heterogeneous final fusion audit differs")
    elif audit != {}:
        raise ValueError("single-lane heterogeneous decode carries fusion evidence")
    if expected_output_tokens is not ...:
        if (
            not isinstance(expected_output_tokens, list)
            or any(type(token) is not int or token < 0 for token in expected_output_tokens)
            or receipt["output_token_count"] != len(expected_output_tokens)
            or receipt["output_tokens_sha256"] != _output_tokens_sha256(expected_output_tokens)
        ):
            raise ValueError("heterogeneous final decode output tokens differ")
    return receipt


__all__ = [
    "COUNTERFACTUAL",
    "DISABLED",
    "HETEROGENEOUS_DECODE_SCHEMA",
    "HETEROGENEOUS_INTEGRATION_SCHEMA",
    "POLICIES",
    "HeterogeneousIntegrationConfig",
    "IntegrationPolicyResult",
    "build_heterogeneous_decode_receipt",
    "run_heterogeneous_integration",
    "validate_heterogeneous_decode_receipt",
    "validate_heterogeneous_integration_receipt",
]
