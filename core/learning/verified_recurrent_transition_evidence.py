"""Replay-complete evidence for one causal recurrent parent/child transition."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Never, cast

from core.brain.llm.latent_cortex.campaign_trust import (
    VerifiedCampaignTrustPolicy,
)
from core.learning.recurrent_grpo import (
    validate_causal_recurrent_transition_pair_receipt,
    validate_recurrent_policy_sample_receipt,
)
from core.learning.verified_token_trace import (
    TokenizerTraceAdapter,
    build_verified_token_trace,
    canonical_behavior_logprob,
    observable_completion_from_trace,
    validate_observable_completion,
    validate_verified_token_trace,
)
from core.learning.verified_transition_episode import (
    TransitionArtifactStore,
    canonical_json_bytes,
)

RECURRENT_TRANSITION_EVIDENCE_SCHEMA = "aura.verified_transition.recurrent_evidence.v2"
_DOCUMENT_KEYS = frozenset(
    {
        "schema",
        "episode_id",
        "task_id",
        "task_commitment",
        "task_commitment_sha256",
        "campaign_trust_policy_sha256",
        "campaign_trust_root_key_id",
        "tokenizer_bundle_sha256",
        "prompt_tokens",
        "prompt_tokens_sha256",
        "sample_receipt_json",
        "sample_receipt_sha256",
        "parent_token_trace",
        "child_token_trace",
        "parent_response_sha256",
        "child_response_sha256",
        "parent_observable_completion",
        "child_observable_completion",
        "parent_score",
        "child_score",
        "created_at_unix_ns",
        "receipt_sha256",
    }
)
_SCORE_KEYS = frozenset(
    {"parsed", "correct", "reason", "normalized_answer_sha256"}
)


class VerifiedRecurrentTransitionEvidenceError(ValueError):
    """Stable evidence construction or replay failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise VerifiedRecurrentTransitionEvidenceError(code)


def _digest(value: Any) -> str:
    try:
        return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    except (TypeError, ValueError, OverflowError, RecursionError):
        _fail("recurrent_evidence_noncanonical_value")


def _float_json(value: Any, *, role: str) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise VerifiedRecurrentTransitionEvidenceError(
            f"{role}_invalid"
        ) from exc
    return encoded


def _parse_float_json(value: Any, *, role: str) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        _fail(f"{role}_invalid")
    try:
        parsed = json.loads(
            value,
            parse_constant=lambda _value: _fail(f"{role}_nonfinite"),
        )
    except VerifiedRecurrentTransitionEvidenceError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise VerifiedRecurrentTransitionEvidenceError(f"{role}_invalid") from exc
    if not isinstance(parsed, dict) or _float_json(parsed, role=role) != value:
        _fail(f"{role}_noncanonical")
    return parsed


def _sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{role}_invalid")
    return value


def _tokens_sha256(tokens: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps(list(tokens), separators=(",", ":"), allow_nan=False).encode(
            "ascii"
        )
    ).hexdigest()


def _task_commitment(task: Any) -> dict[str, Any]:
    resolver = getattr(task, "verified_transition_task_commitment", None)
    if not callable(resolver):
        _fail("recurrent_evidence_task_commitment_missing")
    value = resolver()
    if not isinstance(value, Mapping) or not value:
        _fail("recurrent_evidence_task_commitment_invalid")
    try:
        return cast(dict[str, Any], json.loads(canonical_json_bytes(value)))
    except (TypeError, ValueError, UnicodeError, RecursionError, OverflowError):
        _fail("recurrent_evidence_task_commitment_invalid")


def _trace_responses(
    parent_trace: Any,
    child_trace: Any,
    *,
    adapter: TokenizerTraceAdapter,
    expected_tokenizer_bundle_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    try:
        parent = validate_verified_token_trace(
            parent_trace,
            adapter=adapter,
            expected_tokenizer_bundle_sha256=expected_tokenizer_bundle_sha256,
        )
        child = validate_verified_token_trace(
            child_trace,
            adapter=adapter,
            expected_tokenizer_bundle_sha256=expected_tokenizer_bundle_sha256,
        )
    except ValueError as exc:
        raise VerifiedRecurrentTransitionEvidenceError(
            "recurrent_evidence_token_trace_invalid"
        ) from exc
    parent_response = parent["generation"]["response_text"]
    child_response = child["generation"]["response_text"]
    if not isinstance(parent_response, str) or not isinstance(child_response, str):
        _fail("recurrent_evidence_response_invalid")
    return parent, child, parent_response, child_response


def _observable_completion(trace: Mapping[str, Any]) -> dict[str, Any]:
    generation = trace["generation"]
    return observable_completion_from_trace(
        token_ids=generation["token_ids"],
        streaming_deltas=generation["streaming_deltas"],
        tokenizer_bundle=trace["tokenizer_bundle"],
    )


def optimization_token_counts_from_evidence(
    transition_evidence: Sequence[Any],
) -> tuple[int, ...]:
    """Return only externally replayed child-prefix lengths for an objective."""

    counts: list[int] = []
    for evidence in transition_evidence:
        document = getattr(evidence, "document", None)
        if not isinstance(document, Mapping):
            _fail("recurrent_evidence_observable_completion_missing")
        observable = document.get("child_observable_completion")
        if not isinstance(observable, Mapping):
            _fail("recurrent_evidence_observable_completion_missing")
        count = observable.get("optimization_token_count")
        full_count = observable.get("full_token_count")
        if (
            type(count) is not int
            or type(full_count) is not int
            or not 1 <= count <= full_count
        ):
            _fail("recurrent_evidence_observable_completion_invalid")
        counts.append(count)
    if not counts:
        _fail("recurrent_evidence_observable_completion_missing")
    return tuple(counts)


def _score(
    task: Any,
    response: str,
    scorer: Callable[[Any, Any], Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        value = scorer(task, response)
    except Exception as exc:
        raise VerifiedRecurrentTransitionEvidenceError(
            "recurrent_evidence_scorer_failed"
        ) from exc
    if not isinstance(value, Mapping):
        _fail("recurrent_evidence_score_invalid")
    parsed = value.get("parsed")
    correct = value.get("correct")
    reason = value.get("reason")
    normalized = value.get("normalized_answer_sha256")
    if (
        type(parsed) is not bool
        or type(correct) is not bool
        or not isinstance(reason, str)
        or not reason
        or normalized is None
    ):
        _fail("recurrent_evidence_score_invalid")
    return {
        "parsed": parsed,
        "correct": correct,
        "reason": reason,
        "normalized_answer_sha256": _sha256(
            normalized, role="recurrent_evidence_normalized_answer_sha256"
        ),
    }


@dataclass(frozen=True, slots=True)
class VerifiedRecurrentTransitionEvidence:
    store: TransitionArtifactStore
    document: Mapping[str, Any]
    task: Any
    campaign_trust_policy: VerifiedCampaignTrustPolicy
    tokenizer_trace_adapter: TokenizerTraceAdapter

    @property
    def episode(self) -> Mapping[str, Any]:
        """Compatibility identity consumed by provider membership checks."""

        return self.document


def build_verified_recurrent_transition_evidence(
    store: TransitionArtifactStore,
    *,
    task: Any,
    prompt_text: str,
    prompt_tokens: Sequence[int],
    sample: Any,
    supplied_completion: str,
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
    tokenizer_trace_adapter: TokenizerTraceAdapter,
    expected_tokenizer_bundle_sha256: str,
    campaign_trust_policy: VerifiedCampaignTrustPolicy,
    created_at_unix_ns: int,
) -> VerifiedRecurrentTransitionEvidence:
    """Seal and immediately replay one recurrent causal transition."""

    if type(created_at_unix_ns) is not int or not 0 < created_at_unix_ns < (1 << 63):
        _fail("recurrent_evidence_created_at_invalid")
    try:
        sample_receipt = validate_recurrent_policy_sample_receipt(sample.receipt())
        pair = validate_causal_recurrent_transition_pair_receipt(
            sample_receipt["causal_transition_pair"]
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise VerifiedRecurrentTransitionEvidenceError(
            "recurrent_evidence_sample_invalid"
        ) from exc
    prompt = tuple(prompt_tokens)
    commitment = _task_commitment(task)
    task_id = getattr(task, "task_id", None)
    tokenizer_bundle_sha256 = _sha256(
        expected_tokenizer_bundle_sha256,
        role="recurrent_evidence_tokenizer_bundle_sha256",
    )
    trust_policy_sha256 = _sha256(
        getattr(campaign_trust_policy, "policy_sha256", None),
        role="recurrent_evidence_campaign_trust_policy_sha256",
    )
    trust_root_key_id = _sha256(
        getattr(campaign_trust_policy, "root_key_id", None),
        role="recurrent_evidence_campaign_trust_root_key_id",
    )
    if (
        not isinstance(task_id, str)
        or not task_id
        or commitment.get("task_id") != task_id
        or sample_receipt["prompt_tokens_sha256"] != _tokens_sha256(prompt)
        or pair["episode_id"] != sample_receipt["episode_id"]
        or pair["policy_sha256"] != sample_receipt["policy_sha256"]
        or pair["child"]["tokens"] != sample_receipt["tokens"]
        or pair["child"]["behavior_logprobs"]
        != sample_receipt["behavior_logprobs"]
        or pair["params_unchanged"] is not True
        or pair["child_behavior_admitted"] is not True
    ):
        _fail("recurrent_evidence_sample_binding_mismatch")
    if not isinstance(prompt_text, str) or not isinstance(supplied_completion, str):
        _fail("recurrent_evidence_text_input_invalid")
    try:
        parent_response = tokenizer_trace_adapter.decode_output(
            pair["parent"]["tokens"]
        )
        child_response = tokenizer_trace_adapter.decode_output(
            pair["child"]["tokens"]
        )
        parent_trace = build_verified_token_trace(
            adapter=tokenizer_trace_adapter,
            prompt_text=prompt_text,
            prompt_token_ids=prompt,
            output_token_ids=pair["parent"]["tokens"],
            behavior_logprobs=pair["parent"]["behavior_logprobs"],
            response_text=parent_response,
        )
        child_trace = build_verified_token_trace(
            adapter=tokenizer_trace_adapter,
            prompt_text=prompt_text,
            prompt_token_ids=prompt,
            output_token_ids=pair["child"]["tokens"],
            behavior_logprobs=pair["child"]["behavior_logprobs"],
            response_text=child_response,
        )
    except ValueError as exc:
        raise VerifiedRecurrentTransitionEvidenceError(
            "recurrent_evidence_token_trace_invalid"
        ) from exc
    parent_trace, child_trace, parent_response, child_response = _trace_responses(
        parent_trace,
        child_trace,
        adapter=tokenizer_trace_adapter,
        expected_tokenizer_bundle_sha256=tokenizer_bundle_sha256,
    )
    parent_observable = _observable_completion(parent_trace)
    child_observable = _observable_completion(child_trace)
    if child_observable["response_text"] != supplied_completion:
        _fail("recurrent_evidence_completion_mismatch")
    body = {
        "schema": RECURRENT_TRANSITION_EVIDENCE_SCHEMA,
        "episode_id": pair["episode_id"],
        "task_id": task_id,
        "task_commitment": commitment,
        "task_commitment_sha256": _digest(commitment),
        "campaign_trust_policy_sha256": trust_policy_sha256,
        "campaign_trust_root_key_id": trust_root_key_id,
        "tokenizer_bundle_sha256": tokenizer_bundle_sha256,
        "prompt_tokens": list(prompt),
        "prompt_tokens_sha256": _tokens_sha256(prompt),
        "sample_receipt_json": _float_json(
            sample_receipt, role="recurrent_evidence_sample_receipt"
        ),
        "sample_receipt_sha256": hashlib.sha256(
            _float_json(
                sample_receipt, role="recurrent_evidence_sample_receipt"
            ).encode("ascii")
        ).hexdigest(),
        "parent_token_trace": parent_trace,
        "child_token_trace": child_trace,
        "parent_response_sha256": hashlib.sha256(
            parent_response.encode("utf-8")
        ).hexdigest(),
        "child_response_sha256": hashlib.sha256(
            child_response.encode("utf-8")
        ).hexdigest(),
        "parent_observable_completion": parent_observable,
        "child_observable_completion": child_observable,
        "parent_score": _score(
            task,
            parent_observable["response_text"],
            independent_scorer,
        ),
        "child_score": _score(
            task,
            child_observable["response_text"],
            independent_scorer,
        ),
        "created_at_unix_ns": created_at_unix_ns,
    }
    document = {**body, "receipt_sha256": _digest(body)}
    store.put_json(document)
    return validate_verified_recurrent_transition_evidence(
        store,
        document,
        task=task,
        independent_scorer=independent_scorer,
        tokenizer_trace_adapter=tokenizer_trace_adapter,
        expected_tokenizer_bundle_sha256=tokenizer_bundle_sha256,
        campaign_trust_policy=campaign_trust_policy,
    )


def validate_verified_recurrent_transition_evidence(
    store: TransitionArtifactStore,
    document: Mapping[str, Any],
    *,
    task: Any,
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
    tokenizer_trace_adapter: TokenizerTraceAdapter,
    expected_tokenizer_bundle_sha256: str,
    campaign_trust_policy: VerifiedCampaignTrustPolicy,
) -> VerifiedRecurrentTransitionEvidence:
    if not isinstance(document, Mapping) or set(document) != _DOCUMENT_KEYS:
        _fail("recurrent_evidence_schema_invalid")
    normalized = cast(dict[str, Any], json.loads(canonical_json_bytes(document)))
    tokenizer_bundle_sha256 = _sha256(
        expected_tokenizer_bundle_sha256,
        role="recurrent_evidence_tokenizer_bundle_sha256",
    )
    unsigned = dict(normalized)
    observed = unsigned.pop("receipt_sha256")
    if (
        normalized.get("schema") != RECURRENT_TRANSITION_EVIDENCE_SCHEMA
        or observed != _digest(unsigned)
        or normalized.get("task_commitment") != _task_commitment(task)
        or normalized.get("task_commitment_sha256")
        != _digest(normalized.get("task_commitment"))
        or normalized.get("campaign_trust_policy_sha256")
        != getattr(campaign_trust_policy, "policy_sha256", None)
        or normalized.get("campaign_trust_root_key_id")
        != getattr(campaign_trust_policy, "root_key_id", None)
        or normalized.get("tokenizer_bundle_sha256")
        != tokenizer_bundle_sha256
        or normalized.get("prompt_tokens_sha256")
        != _tokens_sha256(normalized.get("prompt_tokens", []))
    ):
        _fail("recurrent_evidence_reconstruction_mismatch")
    try:
        parsed_sample = _parse_float_json(
            normalized.get("sample_receipt_json"),
            role="recurrent_evidence_sample_receipt",
        )
        if normalized.get("sample_receipt_sha256") != hashlib.sha256(
            normalized["sample_receipt_json"].encode("ascii")
        ).hexdigest():
            _fail("recurrent_evidence_reconstruction_mismatch")
        sample = validate_recurrent_policy_sample_receipt(
            parsed_sample
        )
        pair = validate_causal_recurrent_transition_pair_receipt(
            sample["causal_transition_pair"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise VerifiedRecurrentTransitionEvidenceError(
            "recurrent_evidence_reconstruction_mismatch"
        ) from exc
    parent_trace, child_trace, parent_response, child_response = _trace_responses(
        normalized.get("parent_token_trace"),
        normalized.get("child_token_trace"),
        adapter=tokenizer_trace_adapter,
        expected_tokenizer_bundle_sha256=tokenizer_bundle_sha256,
    )
    try:
        parent_observable = validate_observable_completion(
            normalized.get("parent_observable_completion"),
            token_ids=parent_trace["generation"]["token_ids"],
            streaming_deltas=parent_trace["generation"]["streaming_deltas"],
            tokenizer_bundle=parent_trace["tokenizer_bundle"],
        )
        child_observable = validate_observable_completion(
            normalized.get("child_observable_completion"),
            token_ids=child_trace["generation"]["token_ids"],
            streaming_deltas=child_trace["generation"]["streaming_deltas"],
            tokenizer_bundle=child_trace["tokenizer_bundle"],
        )
    except ValueError as exc:
        raise VerifiedRecurrentTransitionEvidenceError(
            "recurrent_evidence_observable_completion_invalid"
        ) from exc
    if (
        pair != sample.get("causal_transition_pair")
        or pair["episode_id"] != normalized.get("episode_id")
        or getattr(task, "task_id", None) != normalized.get("task_id")
        or parent_trace["prompt"]["token_ids"] != normalized.get("prompt_tokens")
        or child_trace["prompt"] != parent_trace["prompt"]
        or parent_trace["generation"]["token_ids"]
        != pair["parent"]["tokens"]
        or child_trace["generation"]["token_ids"] != pair["child"]["tokens"]
        or parent_trace["generation"]["behavior_logprobs"]
        != [
            canonical_behavior_logprob(value)
            for value in pair["parent"]["behavior_logprobs"]
        ]
        or child_trace["generation"]["behavior_logprobs"]
        != [
            canonical_behavior_logprob(value)
            for value in pair["child"]["behavior_logprobs"]
        ]
        or normalized.get("parent_response_sha256")
        != hashlib.sha256(parent_response.encode("utf-8")).hexdigest()
        or normalized.get("child_response_sha256")
        != hashlib.sha256(child_response.encode("utf-8")).hexdigest()
        or normalized.get("parent_score")
        != _score(task, parent_observable["response_text"], independent_scorer)
        or normalized.get("child_score")
        != _score(task, child_observable["response_text"], independent_scorer)
        or not isinstance(normalized.get("parent_score"), Mapping)
        or set(normalized["parent_score"]) != _SCORE_KEYS
        or not isinstance(normalized.get("child_score"), Mapping)
        or set(normalized["child_score"]) != _SCORE_KEYS
    ):
        _fail("recurrent_evidence_reconstruction_mismatch")
    return VerifiedRecurrentTransitionEvidence(
        store=store,
        document=normalized,
        task=task,
        campaign_trust_policy=campaign_trust_policy,
        tokenizer_trace_adapter=tokenizer_trace_adapter,
    )


__all__ = [
    "RECURRENT_TRANSITION_EVIDENCE_SCHEMA",
    "VerifiedRecurrentTransitionEvidence",
    "VerifiedRecurrentTransitionEvidenceError",
    "build_verified_recurrent_transition_evidence",
    "optimization_token_counts_from_evidence",
    "validate_verified_recurrent_transition_evidence",
]
