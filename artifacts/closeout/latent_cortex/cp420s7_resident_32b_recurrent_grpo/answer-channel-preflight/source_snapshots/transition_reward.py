"""Independent reward reconstruction for verified recurrent transitions.

This module is deliberately downstream of ``verified_transition_episode``.
The candidate and trainer never provide labels or scalar rewards.  A reward
batch is rebuilt from sealed verifier authorities and precommitted telemetry,
then admitted lexicographically before any gradient may be constructed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Never, cast

from core.brain.llm.latent_cortex.frontier_tasks import FrontierTask
from core.learning.verified_recurrent_transition_evidence import (
    VerifiedRecurrentTransitionEvidence,
    validate_verified_recurrent_transition_evidence,
)
from core.learning.verified_transition_episode import (
    TransitionArtifactStore,
    TransitionTrustContext,
    canonical_json_bytes,
    validate_verified_transition_episode,
)

VERIFIED_TRANSITION_REWARD_SCHEMA = "aura.verified_transition.reward_batch.v1"
VERIFIED_TRANSITION_REWARD_CONFIG_SCHEMA = (
    "aura.verified_transition.reward_config.v1"
)
MICROS = 1_000_000
_SHA256_LENGTH = 64


class VerifiedTransitionRewardError(ValueError):
    """Raised when reward evidence cannot be reconstructed exactly."""


class VerifiedTransitionRewardAdmissionError(RuntimeError):
    """Raised before gradient construction for a rejected transition group."""

    def __init__(self, receipt: Mapping[str, Any]) -> None:
        self.receipt = dict(receipt)
        reason = str(receipt.get("optimizer_admission_reason", "invalid_receipt"))
        super().__init__(f"verified transition reward group rejected: {reason}")


def _fail(code: str) -> Never:
    raise VerifiedTransitionRewardError(code)


def _require_int(
    value: Any,
    *,
    role: str,
    minimum: int = 0,
    maximum: int = MICROS,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{role}_invalid")
    return value


def _require_sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{role}_invalid")
    return value


def _seal(document: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(document)
    sealed["receipt_sha256"] = hashlib.sha256(
        canonical_json_bytes(sealed)
    ).hexdigest()
    return sealed


def _validate_seal(document: Mapping[str, Any]) -> None:
    observed = _require_sha256(
        document.get("receipt_sha256"), role="reward_batch_receipt_sha256"
    )
    unsigned = dict(document)
    unsigned.pop("receipt_sha256", None)
    expected = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if observed != expected:
        _fail("reward_batch_digest_mismatch")


def _scale_micros(value: int, weight_micros: int) -> int:
    product = value * weight_micros
    if product >= 0:
        return product // MICROS
    return -((-product) // MICROS)


@dataclass(frozen=True, slots=True)
class TransitionRewardConfig:
    """Fixed-point reward weights; all values are integer micro-units."""

    correctness_delta_weight_micros: int = MICROS
    information_gain_weight_micros: int = 200_000
    diversity_gain_weight_micros: int = 100_000
    compute_cost_weight_micros: int = 100_000
    unsupported_confidence_weight_micros: int = 250_000

    def __post_init__(self) -> None:
        for field in (
            "correctness_delta_weight_micros",
            "information_gain_weight_micros",
            "diversity_gain_weight_micros",
            "compute_cost_weight_micros",
            "unsupported_confidence_weight_micros",
        ):
            _require_int(getattr(self, field), role=field)
        if self.correctness_delta_weight_micros <= sum(
            (
                self.information_gain_weight_micros,
                self.diversity_gain_weight_micros,
                self.compute_cost_weight_micros,
                self.unsupported_confidence_weight_micros,
            )
        ):
            raise ValueError(
                "correctness delta must dominate every combined shaping term"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": VERIFIED_TRANSITION_REWARD_CONFIG_SCHEMA,
            "correctness_delta_weight_micros": (
                self.correctness_delta_weight_micros
            ),
            "information_gain_weight_micros": self.information_gain_weight_micros,
            "diversity_gain_weight_micros": self.diversity_gain_weight_micros,
            "compute_cost_weight_micros": self.compute_cost_weight_micros,
            "unsupported_confidence_weight_micros": (
                self.unsupported_confidence_weight_micros
            ),
        }


@dataclass(frozen=True, slots=True)
class VerifiedTransitionEvidence:
    """Inputs required to independently replay one CP419 episode."""

    store: TransitionArtifactStore
    episode: Mapping[str, Any]
    task: FrontierTask
    expected_authority: Mapping[str, Any]
    trust_context: TransitionTrustContext


def _authority_output(
    store: TransitionArtifactStore,
    pass_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    authority = store.read_json(
        cast(Mapping[str, Any], pass_receipt["verifier_authority_artifact"]),
        role="verifier_authority",
    )
    output = authority.get("verifier_output")
    if not isinstance(output, Mapping):
        _fail("verifier_output_missing")
    return dict(output)


def _transition_kind(initial_correct: bool, final_correct: bool) -> str:
    if initial_correct and final_correct:
        return "right_to_right"
    if initial_correct and not final_correct:
        return "right_to_wrong"
    if final_correct:
        return "wrong_to_right"
    return "wrong_to_wrong"


def _verified_information_gain_micros(
    *,
    initial_parsed: bool,
    initial_correct: bool,
    final_parsed: bool,
    final_correct: bool,
) -> int:
    def state(parsed: bool, correct: bool) -> int:
        if correct:
            return 2
        if parsed:
            return 1
        return 0

    return (state(final_parsed, final_correct) - state(initial_parsed, initial_correct)) * (
        MICROS // 2
    )


def _actual_resource_cost_micros(
    store: TransitionArtifactStore,
    pass_receipt: Mapping[str, Any],
) -> int:
    process = store.read_json(
        cast(Mapping[str, Any], pass_receipt["process_receipt_artifact"]),
        role="process_receipt",
    )
    started = process.get("started_at_unix_ns")
    finished = process.get("finished_at_unix_ns")
    if type(started) is not int or type(finished) is not int or finished < started:
        _fail("transition_reward_process_timing_invalid")
    budget = pass_receipt.get("generation_budget")
    if not isinstance(budget, Mapping):
        _fail("transition_reward_generation_budget_invalid")
    maximum_tokens = _require_int(
        budget.get("max_output_tokens"),
        role="transition_reward_max_output_tokens",
        minimum=1,
        maximum=65_536,
    )
    maximum_wall_ms = _require_int(
        budget.get("max_wall_time_ms"),
        role="transition_reward_max_wall_time_ms",
        minimum=1,
        maximum=(1 << 63) - 1,
    )
    output_tokens = pass_receipt.get("output_token_ids")
    if not isinstance(output_tokens, list) or not output_tokens:
        _fail("transition_reward_output_tokens_invalid")
    token_fraction = min(MICROS, len(output_tokens) * MICROS // maximum_tokens)
    wall_budget_ns = maximum_wall_ms * 1_000_000
    wall_fraction = min(MICROS, (finished - started) * MICROS // wall_budget_ns)
    return max(token_fraction, wall_fraction)


def _unsupported_confidence_micros(
    pass_receipt: Mapping[str, Any],
    *,
    final_correct: bool,
) -> int:
    if final_correct:
        return 0
    values = pass_receipt.get("behavior_policy_logprobs")
    if not isinstance(values, list) or not values:
        _fail("transition_reward_behavior_logprobs_invalid")
    threshold = Decimal("-0.6931471805599453")
    confident = 0
    for value in values:
        try:
            parsed = Decimal(str(value))
        except InvalidOperation as exc:
            raise VerifiedTransitionRewardError(
                "transition_reward_behavior_logprob_invalid"
            ) from exc
        if not parsed.is_finite() or parsed > 0:
            _fail("transition_reward_behavior_logprob_invalid")
        confident += parsed >= threshold
    return confident * MICROS // len(values)


def _reconstruct_transition(
    evidence: VerifiedTransitionEvidence | VerifiedRecurrentTransitionEvidence,
    *,
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
    token_encoder: Callable[[bytes], Sequence[int]],
    token_decoder: Callable[[Sequence[int]], bytes],
    config: TransitionRewardConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(evidence, VerifiedRecurrentTransitionEvidence):
        return _reconstruct_recurrent_transition(
            evidence,
            independent_scorer=independent_scorer,
            config=config,
        )
    store = evidence.store
    episode = validate_verified_transition_episode(
        store,
        evidence.episode,
        task=evidence.task,
        expected_authority=evidence.expected_authority,
        independent_scorer=independent_scorer,
        token_encoder=token_encoder,
        token_decoder=token_decoder,
        trust_context=evidence.trust_context,
    )
    first = store.read_json(
        cast(Mapping[str, Any], episode["pass_0_artifact"]), role="reasoning_pass"
    )
    second = store.read_json(
        cast(Mapping[str, Any], episode["pass_1_artifact"]), role="reasoning_pass"
    )
    initial = _authority_output(store, first)
    final = _authority_output(store, second)
    initial_correct = initial.get("correct") is True
    final_correct = final.get("correct") is True
    initial_parsed = initial.get("parsed") is True
    final_parsed = final.get("parsed") is True
    kind = _transition_kind(initial_correct, final_correct)

    correctness_delta = int(final_correct) - int(initial_correct)
    information_gain = _verified_information_gain_micros(
        initial_parsed=initial_parsed,
        initial_correct=initial_correct,
        final_parsed=final_parsed,
        final_correct=final_correct,
    )
    normalized_0 = initial.get("normalized_answer_sha256")
    normalized_1 = final.get("normalized_answer_sha256")
    diversity_gain = int(
        initial_parsed
        and final_parsed
        and normalized_0 is not None
        and normalized_1 is not None
        and normalized_0 != normalized_1
    ) * MICROS
    resource_1 = _actual_resource_cost_micros(store, second)
    unsupported_confidence = _unsupported_confidence_micros(
        second,
        final_correct=final_correct,
    )
    components = {
        "correctness_delta_micros": (
            correctness_delta * config.correctness_delta_weight_micros
        ),
        "information_gain_micros": _scale_micros(
            information_gain, config.information_gain_weight_micros
        ),
        "diversity_gain_micros": _scale_micros(
            diversity_gain, config.diversity_gain_weight_micros
        ),
        "compute_cost_micros": -_scale_micros(
            resource_1, config.compute_cost_weight_micros
        ),
        "unsupported_confidence_micros": -_scale_micros(
            unsupported_confidence,
            config.unsupported_confidence_weight_micros,
        ),
    }
    reward_micros = sum(components.values())
    record = {
        "episode_id": episode["episode_id"],
        "task_id": episode["task_id"],
        "episode_receipt_sha256": episode["receipt_sha256"],
        "pass_0_receipt_sha256": first["receipt_sha256"],
        "pass_1_receipt_sha256": second["receipt_sha256"],
        "initial_parsed": initial_parsed,
        "final_parsed": final_parsed,
        "initial_correct": initial_correct,
        "final_correct": final_correct,
        "transition_kind": kind,
        "answer_changed": normalized_0 != normalized_1,
        "initial_answer_sha256": normalized_0,
        "final_answer_sha256": normalized_1,
        "information_gain_micros": information_gain,
        "diversity_gain_micros": diversity_gain,
        "resource_1_micros": resource_1,
        "unsupported_confidence_micros": unsupported_confidence,
        "reward_components_micros": components,
        "reward_micros": reward_micros,
        "pass_1_policy_sha256": second["policy_sha256"],
        "pass_1_input_token_ids": list(second["input_token_ids"]),
        "pass_1_output_token_ids": list(second["output_token_ids"]),
        "pass_1_behavior_policy_logprobs": list(
            second["behavior_policy_logprobs"]
        ),
    }
    return record, episode


def _reconstruct_recurrent_transition(
    evidence: VerifiedRecurrentTransitionEvidence,
    *,
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
    config: TransitionRewardConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validated = validate_verified_recurrent_transition_evidence(
        evidence.store,
        evidence.document,
        task=evidence.task,
        independent_scorer=independent_scorer,
        tokenizer_trace_adapter=evidence.tokenizer_trace_adapter,
        expected_tokenizer_bundle_sha256=evidence.document[
            "tokenizer_bundle_sha256"
        ],
        campaign_trust_policy=evidence.campaign_trust_policy,
    )
    episode = dict(validated.document)
    sample = json.loads(cast(str, episode["sample_receipt_json"]))
    pair = cast(dict[str, Any], sample["causal_transition_pair"])
    parent = cast(dict[str, Any], pair["parent"])
    child = cast(dict[str, Any], pair["child"])
    initial = cast(dict[str, Any], episode["parent_score"])
    final = cast(dict[str, Any], episode["child_score"])
    initial_correct = initial["correct"] is True
    final_correct = final["correct"] is True
    initial_parsed = initial["parsed"] is True
    final_parsed = final["parsed"] is True
    information_gain = _verified_information_gain_micros(
        initial_parsed=initial_parsed,
        initial_correct=initial_correct,
        final_parsed=final_parsed,
        final_correct=final_correct,
    )
    normalized_0 = initial["normalized_answer_sha256"]
    normalized_1 = final["normalized_answer_sha256"]
    diversity_gain = int(
        initial_parsed and final_parsed and normalized_0 != normalized_1
    ) * MICROS
    token_budget = child["token_budget"]
    resource_1 = min(MICROS, child["token_count"] * MICROS // token_budget)
    unsupported_confidence = _unsupported_confidence_micros(
        {"behavior_policy_logprobs": child["behavior_logprobs"]},
        final_correct=final_correct,
    )
    components = {
        "correctness_delta_micros": (
            (int(final_correct) - int(initial_correct))
            * config.correctness_delta_weight_micros
        ),
        "information_gain_micros": _scale_micros(
            information_gain, config.information_gain_weight_micros
        ),
        "diversity_gain_micros": _scale_micros(
            diversity_gain, config.diversity_gain_weight_micros
        ),
        "compute_cost_micros": -_scale_micros(
            resource_1, config.compute_cost_weight_micros
        ),
        "unsupported_confidence_micros": -_scale_micros(
            unsupported_confidence,
            config.unsupported_confidence_weight_micros,
        ),
    }
    return (
        {
            "episode_id": episode["episode_id"],
            "task_id": episode["task_id"],
            "episode_receipt_sha256": episode["receipt_sha256"],
            "pass_0_receipt_sha256": parent["receipt_sha256"],
            "pass_1_receipt_sha256": child["receipt_sha256"],
            "initial_parsed": initial_parsed,
            "final_parsed": final_parsed,
            "initial_correct": initial_correct,
            "final_correct": final_correct,
            "transition_kind": _transition_kind(initial_correct, final_correct),
            "answer_changed": normalized_0 != normalized_1,
            "initial_answer_sha256": normalized_0,
            "final_answer_sha256": normalized_1,
            "information_gain_micros": information_gain,
            "diversity_gain_micros": diversity_gain,
            "resource_1_micros": resource_1,
            "unsupported_confidence_micros": unsupported_confidence,
            "reward_components_micros": components,
            "reward_micros": sum(components.values()),
            "pass_1_policy_sha256": pair["policy_sha256"],
            "pass_1_input_token_ids": list(episode["prompt_tokens"]),
            "pass_1_output_token_ids": list(child["tokens"]),
            "pass_1_behavior_policy_logprobs": list(
                str(value) for value in child["behavior_logprobs"]
            ),
        },
        episode,
    )


def _admission_reason(
    *,
    right_to_wrong: int,
    initially_correct: int,
    wrong_to_right: int,
    rewards: Sequence[int],
) -> str:
    if right_to_wrong:
        return "right_to_wrong_regression"
    if initially_correct == 0:
        return "eir_undefined_no_initially_correct_control"
    if wrong_to_right == 0:
        return "no_verified_wrong_to_right_improvement"
    if len(set(rewards)) < 2:
        return "degenerate_structured_reward_group"
    return "admitted"


def _assemble_reward_batch(
    *,
    records: Sequence[Mapping[str, Any]],
    episode_artifacts: Sequence[Mapping[str, Any]],
    task_id: str,
    config: TransitionRewardConfig,
    created_at_unix_ns: int,
) -> dict[str, Any]:
    if len(records) != len(episode_artifacts) or not records:
        _fail("transition_reward_aggregate_inputs_invalid")
    counts = {
        kind: sum(record["transition_kind"] == kind for record in records)
        for kind in (
            "wrong_to_right",
            "right_to_wrong",
            "right_to_right",
            "wrong_to_wrong",
        )
    }
    if sum(counts.values()) != len(records):
        _fail("transition_reward_transition_kind_invalid")
    initially_correct = counts["right_to_wrong"] + counts["right_to_right"]
    eir_defined = initially_correct > 0
    eir_micros: int | None = (
        counts["right_to_wrong"] * MICROS // initially_correct
        if eir_defined
        else None
    )
    rewards = [cast(int, record["reward_micros"]) for record in records]
    if any(type(reward) is not int for reward in rewards):
        _fail("transition_reward_scalar_invalid")
    reason = _admission_reason(
        right_to_wrong=counts["right_to_wrong"],
        initially_correct=initially_correct,
        wrong_to_right=counts["wrong_to_right"],
        rewards=rewards,
    )
    return _seal(
        {
            "schema": VERIFIED_TRANSITION_REWARD_SCHEMA,
            "task_id": task_id,
            "group_size": len(records),
            "reward_config": config.to_dict(),
            "episode_artifacts": [dict(binding) for binding in episode_artifacts],
            "transitions": [dict(record) for record in records],
            "wrong_to_right": counts["wrong_to_right"],
            "right_to_wrong": counts["right_to_wrong"],
            "right_to_right": counts["right_to_right"],
            "wrong_to_wrong": counts["wrong_to_wrong"],
            "initially_correct": initially_correct,
            "eir_defined": eir_defined,
            "eir_numerator": counts["right_to_wrong"],
            "eir_denominator": initially_correct,
            "eir_micros": eir_micros,
            "optimizer_admitted": reason == "admitted",
            "optimizer_admission_reason": reason,
            "created_at_unix_ns": created_at_unix_ns,
        }
    )


def build_verified_transition_reward_batch(
    store: TransitionArtifactStore,
    evidence: Sequence[
        VerifiedTransitionEvidence | VerifiedRecurrentTransitionEvidence
    ],
    *,
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
    token_encoder: Callable[[bytes], Sequence[int]],
    token_decoder: Callable[[Sequence[int]], bytes],
    config: TransitionRewardConfig | None = None,
    created_at_unix_ns: int,
) -> dict[str, Any]:
    """Reconstruct and seal one optimizer-admission decision."""

    if not evidence:
        _fail("transition_reward_evidence_empty")
    if type(created_at_unix_ns) is not int or created_at_unix_ns <= 0:
        _fail("transition_reward_created_at_invalid")
    resolved = config or TransitionRewardConfig()
    records: list[dict[str, Any]] = []
    episode_artifacts: list[dict[str, Any]] = []
    seen_episode_ids: set[str] = set()
    task_id: str | None = None
    for item in evidence:
        if not isinstance(
            item, (VerifiedTransitionEvidence, VerifiedRecurrentTransitionEvidence)
        ):
            _fail("transition_reward_evidence_type_invalid")
        record, episode = _reconstruct_transition(
            item,
            independent_scorer=independent_scorer,
            token_encoder=token_encoder,
            token_decoder=token_decoder,
            config=resolved,
        )
        episode_id = str(record["episode_id"])
        if episode_id in seen_episode_ids:
            _fail("transition_reward_duplicate_episode")
        seen_episode_ids.add(episode_id)
        if task_id is None:
            task_id = str(record["task_id"])
        elif record["task_id"] != task_id:
            _fail("transition_reward_mixed_tasks")
        records.append(record)
        episode_artifacts.append(store.put_json(episode))

    return _assemble_reward_batch(
        records=records,
        episode_artifacts=episode_artifacts,
        task_id=cast(str, task_id),
        config=resolved,
        created_at_unix_ns=created_at_unix_ns,
    )


def _config_from_document(value: Any) -> TransitionRewardConfig:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "correctness_delta_weight_micros",
        "information_gain_weight_micros",
        "diversity_gain_weight_micros",
        "compute_cost_weight_micros",
        "unsupported_confidence_weight_micros",
    }:
        _fail("transition_reward_config_schema_invalid")
    if value.get("schema") != VERIFIED_TRANSITION_REWARD_CONFIG_SCHEMA:
        _fail("transition_reward_config_version_invalid")
    return TransitionRewardConfig(
        correctness_delta_weight_micros=_require_int(
            value.get("correctness_delta_weight_micros"),
            role="correctness_delta_weight_micros",
        ),
        information_gain_weight_micros=_require_int(
            value.get("information_gain_weight_micros"),
            role="information_gain_weight_micros",
        ),
        diversity_gain_weight_micros=_require_int(
            value.get("diversity_gain_weight_micros"),
            role="diversity_gain_weight_micros",
        ),
        compute_cost_weight_micros=_require_int(
            value.get("compute_cost_weight_micros"),
            role="compute_cost_weight_micros",
        ),
        unsupported_confidence_weight_micros=_require_int(
            value.get("unsupported_confidence_weight_micros"),
            role="unsupported_confidence_weight_micros",
        ),
    )


def validate_verified_transition_reward_batch(
    store: TransitionArtifactStore,
    receipt: Mapping[str, Any],
    evidence: Sequence[
        VerifiedTransitionEvidence | VerifiedRecurrentTransitionEvidence
    ],
    *,
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
    token_encoder: Callable[[bytes], Sequence[int]],
    token_decoder: Callable[[Sequence[int]], bytes],
) -> dict[str, Any]:
    """Rebuild a reward batch from source evidence and compare every byte."""

    if not isinstance(receipt, Mapping):
        _fail("transition_reward_receipt_invalid")
    _validate_seal(receipt)
    if receipt.get("schema") != VERIFIED_TRANSITION_REWARD_SCHEMA:
        _fail("transition_reward_schema_invalid")
    artifacts = receipt.get("episode_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(evidence):
        _fail("transition_reward_episode_artifacts_invalid")
    for binding, item in zip(artifacts, evidence, strict=True):
        stored = store.read_json(cast(Mapping[str, Any], binding), role="paired_episode")
        if stored != dict(item.episode):
            _fail("transition_reward_episode_binding_mismatch")
    expected = build_verified_transition_reward_batch(
        store,
        evidence,
        independent_scorer=independent_scorer,
        token_encoder=token_encoder,
        token_decoder=token_decoder,
        config=_config_from_document(receipt.get("reward_config")),
        created_at_unix_ns=_require_int(
            receipt.get("created_at_unix_ns"),
            role="transition_reward_created_at",
            maximum=(1 << 63) - 1,
        ),
    )
    if expected != dict(receipt):
        _fail("transition_reward_reconstruction_mismatch")
    return dict(receipt)


def require_optimizer_admission(receipt: Mapping[str, Any]) -> None:
    """Fail closed before gradient construction for a rejected batch."""

    _validate_seal(receipt)
    if receipt.get("optimizer_admitted") is not True:
        raise VerifiedTransitionRewardAdmissionError(receipt)
    if (
        receipt.get("right_to_wrong") != 0
        or receipt.get("eir_defined") is not True
        or receipt.get("eir_micros") != 0
        or not isinstance(receipt.get("transitions"), list)
    ):
        _fail("transition_reward_admission_invariant_invalid")


def bind_rewards_to_recurrent_samples(
    receipt: Mapping[str, Any],
    samples: Sequence[Any],
    prompt_tokens: Sequence[int],
) -> tuple[float, ...]:
    """Bind final-pass reward rows to exact recurrent policy samples."""

    _validate_seal(receipt)
    transitions_value = receipt.get("transitions")
    if not isinstance(transitions_value, list):
        _fail("transition_reward_rows_invalid")
    transitions = cast(list[Mapping[str, Any]], transitions_value)
    if len(transitions) != len(samples):
        _fail("transition_reward_sample_count_mismatch")
    prompt_payload = json.dumps(
        list(prompt_tokens), separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    prompt_sha256 = hashlib.sha256(prompt_payload).hexdigest()
    rewards: list[float] = []
    for index, (transition, sample) in enumerate(
        zip(transitions, samples, strict=True)
    ):
        if tuple(transition["pass_1_input_token_ids"]) != tuple(prompt_tokens):
            _fail(f"transition_reward_prompt_tokens_mismatch_{index}")
        if getattr(sample, "prompt_tokens_sha256", None) != prompt_sha256:
            _fail(f"transition_reward_sample_prompt_mismatch_{index}")
        if tuple(transition["pass_1_output_token_ids"]) != tuple(
            getattr(sample, "tokens", ())
        ):
            _fail(f"transition_reward_sample_tokens_mismatch_{index}")
        if transition["pass_1_policy_sha256"] != getattr(
            sample, "policy_sha256", None
        ):
            _fail(f"transition_reward_sample_policy_mismatch_{index}")
        observed_logprobs = getattr(sample, "behavior_logprobs", ())
        expected_logprobs = transition["pass_1_behavior_policy_logprobs"]
        if len(observed_logprobs) != len(expected_logprobs):
            _fail(f"transition_reward_sample_logprob_count_mismatch_{index}")
        try:
            exact = all(
                Decimal(str(observed)) == Decimal(str(expected))
                for observed, expected in zip(
                    observed_logprobs, expected_logprobs, strict=True
                )
            )
        except (InvalidOperation, ValueError) as exc:
            raise VerifiedTransitionRewardError(
                f"transition_reward_sample_logprob_invalid_{index}"
            ) from exc
        if not exact or any(not math.isfinite(float(value)) for value in observed_logprobs):
            _fail(f"transition_reward_sample_logprob_mismatch_{index}")
        rewards.append(int(transition["reward_micros"]) / MICROS)
    return tuple(rewards)


def rewards_for_recurrent_samples(
    receipt: Mapping[str, Any],
    samples: Sequence[Any],
    prompt_tokens: Sequence[int],
) -> tuple[float, ...]:
    """Bind an optimizer-admitted reward batch to its policy samples."""

    require_optimizer_admission(receipt)
    return bind_rewards_to_recurrent_samples(receipt, samples, prompt_tokens)


__all__ = [
    "MICROS",
    "VERIFIED_TRANSITION_REWARD_CONFIG_SCHEMA",
    "VERIFIED_TRANSITION_REWARD_SCHEMA",
    "TransitionRewardConfig",
    "VerifiedTransitionEvidence",
    "VerifiedTransitionRewardAdmissionError",
    "VerifiedTransitionRewardError",
    "bind_rewards_to_recurrent_samples",
    "build_verified_transition_reward_batch",
    "require_optimizer_admission",
    "rewards_for_recurrent_samples",
    "validate_verified_transition_reward_batch",
]
