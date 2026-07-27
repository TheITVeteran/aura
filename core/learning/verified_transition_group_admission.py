"""Signed group-level admission for verified recurrent policy updates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Never, cast

from core.brain.llm.latent_cortex.campaign_trust import (
    TASK_ISSUER,
    verify_role_attestation,
)
from core.learning.verified_transition_episode import (
    TransitionArtifactStore,
    canonical_json_bytes,
)
from core.learning.verified_transition_reward import (
    VerifiedTransitionEvidence,
    require_optimizer_admission,
    rewards_for_recurrent_samples,
    validate_verified_transition_reward_batch,
)

TRANSITION_GROUP_MANIFEST_SCHEMA = "aura.verified_transition.group_manifest.v1"
TRANSITION_GROUP_ADMISSION_SCHEMA = "aura.verified_transition.group_admission.v1"
_SHA256_LENGTH = 64
_MAX_GROUP_SIZE = 1_024


class VerifiedTransitionGroupError(ValueError):
    """Raised when a planned optimizer group cannot be reconstructed."""


def _fail(code: str) -> Never:
    raise VerifiedTransitionGroupError(code)


def _require_identifier(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 192
        or not value[0].isalnum()
        or any(
            not (character.isalnum() or character in "._:/;=+-")
            for character in value
        )
    ):
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


def _require_int(
    value: Any,
    *,
    role: str,
    minimum: int = 0,
    maximum: int = (1 << 63) - 1,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{role}_invalid")
    return value


def _seal(document: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    sealed = dict(document)
    sealed[field] = hashlib.sha256(canonical_json_bytes(sealed)).hexdigest()
    return sealed


def _validate_seal(document: Mapping[str, Any], *, field: str, role: str) -> None:
    observed = _require_sha256(document.get(field), role=f"{role}_{field}")
    unsigned = dict(document)
    unsigned.pop(field, None)
    if observed != hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest():
        _fail(f"{role}_digest_mismatch")


def _sampling_config_document(sample: Any) -> dict[str, Any]:
    config = getattr(sample, "sampling_config", None)
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    if not isinstance(config, Mapping) or not config:
        _fail("group_sample_sampling_config_invalid")
    document = dict(config)
    try:
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise VerifiedTransitionGroupError(
            "group_sample_sampling_config_invalid"
        ) from exc
    return document


def sampling_config_sha256(sample: Any) -> str:
    payload = json.dumps(
        _sampling_config_document(sample),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class TransitionGroupPlanEntry:
    episode_id: str
    task_id: str
    rng_root_sha256: str
    policy_sha256: str
    recurrent_execution_spec_sha256: str
    producing_branch_index: int
    sample_seed: int
    sampling_config_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": _require_identifier(self.episode_id, role="group_episode_id"),
            "task_id": _require_identifier(self.task_id, role="group_task_id"),
            "rng_root_sha256": _require_sha256(
                self.rng_root_sha256, role="group_rng_root"
            ),
            "policy_sha256": _require_sha256(
                self.policy_sha256, role="group_policy"
            ),
            "recurrent_execution_spec_sha256": _require_sha256(
                self.recurrent_execution_spec_sha256,
                role="group_execution_spec",
            ),
            "producing_branch_index": _require_int(
                self.producing_branch_index,
                role="group_branch_index",
                maximum=255,
            ),
            "sample_seed": _require_int(
                self.sample_seed,
                role="group_sample_seed",
                maximum=(1 << 32) - 1,
            ),
            "sampling_config_sha256": _require_sha256(
                self.sampling_config_sha256,
                role="group_sampling_config",
            ),
        }


def build_transition_group_manifest(
    *,
    group_id: str,
    task_id: str,
    entries: Sequence[TransitionGroupPlanEntry],
    reward_config_sha256: str,
    planned_at_unix_ns: int,
) -> dict[str, Any]:
    """Build the payload that the external task issuer signs pre-generation."""

    if not entries or len(entries) > _MAX_GROUP_SIZE:
        _fail("group_manifest_size_invalid")
    normalized = [entry.to_dict() for entry in entries]
    episode_ids = [entry["episode_id"] for entry in normalized]
    if len(set(episode_ids)) != len(episode_ids):
        _fail("group_manifest_duplicate_episode")
    normalized_task = _require_identifier(task_id, role="group_manifest_task")
    if any(entry["task_id"] != normalized_task for entry in normalized):
        _fail("group_manifest_mixed_tasks")
    return _seal(
        {
            "schema": TRANSITION_GROUP_MANIFEST_SCHEMA,
            "group_id": _require_identifier(group_id, role="group_id"),
            "task_id": normalized_task,
            "group_size": len(normalized),
            "reward_config_sha256": _require_sha256(
                reward_config_sha256,
                role="group_reward_config",
            ),
            "entries": normalized,
            "planned_at_unix_ns": _require_int(
                planned_at_unix_ns,
                role="group_planned_at",
                minimum=1,
            ),
        },
        field="manifest_sha256",
    )


def validate_transition_group_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "group_id",
        "task_id",
        "group_size",
        "reward_config_sha256",
        "entries",
        "planned_at_unix_ns",
        "manifest_sha256",
    }:
        _fail("group_manifest_schema_invalid")
    if value.get("schema") != TRANSITION_GROUP_MANIFEST_SCHEMA:
        _fail("group_manifest_version_invalid")
    _validate_seal(value, field="manifest_sha256", role="group_manifest")
    entries = value.get("entries")
    if not isinstance(entries, list):
        _fail("group_manifest_entries_invalid")
    normalized_entries: list[TransitionGroupPlanEntry] = []
    for raw in entries:
        if not isinstance(raw, Mapping) or set(raw) != {
            "episode_id",
            "task_id",
            "rng_root_sha256",
            "policy_sha256",
            "recurrent_execution_spec_sha256",
            "producing_branch_index",
            "sample_seed",
            "sampling_config_sha256",
        }:
            _fail("group_manifest_entry_schema_invalid")
        normalized_entries.append(
            TransitionGroupPlanEntry(
                episode_id=cast(str, raw.get("episode_id")),
                task_id=cast(str, raw.get("task_id")),
                rng_root_sha256=cast(str, raw.get("rng_root_sha256")),
                policy_sha256=cast(str, raw.get("policy_sha256")),
                recurrent_execution_spec_sha256=cast(
                    str, raw.get("recurrent_execution_spec_sha256")
                ),
                producing_branch_index=cast(int, raw.get("producing_branch_index")),
                sample_seed=cast(int, raw.get("sample_seed")),
                sampling_config_sha256=cast(str, raw.get("sampling_config_sha256")),
            )
        )
    expected = build_transition_group_manifest(
        group_id=cast(str, value.get("group_id")),
        task_id=cast(str, value.get("task_id")),
        entries=normalized_entries,
        reward_config_sha256=cast(str, value.get("reward_config_sha256")),
        planned_at_unix_ns=cast(int, value.get("planned_at_unix_ns")),
    )
    if expected != dict(value):
        _fail("group_manifest_reconstruction_mismatch")
    return dict(value)


def _task_disclosure_signed_at(evidence: VerifiedTransitionEvidence) -> int:
    attestation = evidence.trust_context.task_issuer_attestation
    if not isinstance(attestation, Mapping):
        _fail("group_task_issuer_attestation_missing")
    signed_payload = attestation.get("signed_payload")
    if not isinstance(signed_payload, Mapping):
        _fail("group_task_issuer_attestation_invalid")
    return _require_int(
        signed_payload.get("signed_at_unix"),
        role="group_task_disclosure_time",
        minimum=1,
    )


def _actual_plan_entry(
    evidence: VerifiedTransitionEvidence,
    sample: Any,
) -> TransitionGroupPlanEntry:
    episode = evidence.episode
    store = evidence.store
    second = store.read_json(
        cast(Mapping[str, Any], episode["pass_1_artifact"]),
        role="reasoning_pass",
    )
    execution_spec = store.read_json(
        cast(Mapping[str, Any], second["execution_spec_artifact"]),
        role="execution_spec",
    )
    recurrent_spec = execution_spec.get("recurrent_execution_spec_sha256")
    if recurrent_spec is None:
        _fail("group_episode_recurrent_execution_spec_missing")
    latent_path = store.read_json(
        cast(Mapping[str, Any], second["latent_path_artifact"]),
        role="latent_path",
    )
    branch = _require_int(
        getattr(sample, "branch_index", None),
        role="group_sample_branch",
        maximum=255,
    )
    branch_count = _require_int(
        latent_path.get("branch_count"),
        role="group_episode_branch_count",
        minimum=1,
        maximum=256,
    )
    if branch >= branch_count:
        _fail("group_sample_branch_outside_episode")
    sample_spec = _require_sha256(
        getattr(sample, "execution_spec_sha256", None),
        role="group_sample_execution_spec",
    )
    if recurrent_spec != sample_spec:
        _fail("group_sample_execution_spec_mismatch")
    policy = _require_sha256(
        getattr(sample, "policy_sha256", None),
        role="group_sample_policy",
    )
    if policy != second.get("policy_sha256"):
        _fail("group_sample_policy_mismatch")
    return TransitionGroupPlanEntry(
        episode_id=cast(str, episode["episode_id"]),
        task_id=cast(str, episode["task_id"]),
        rng_root_sha256=cast(str, second["rng_root_sha256"]),
        policy_sha256=policy,
        recurrent_execution_spec_sha256=sample_spec,
        producing_branch_index=branch,
        sample_seed=_require_int(
            getattr(sample, "seed", None),
            role="group_sample_seed",
            maximum=(1 << 32) - 1,
        ),
        sampling_config_sha256=sampling_config_sha256(sample),
    )


def _sample_receipt(sample: Any) -> dict[str, Any]:
    receipt = sample.receipt() if hasattr(sample, "receipt") else None
    if not isinstance(receipt, Mapping):
        payload = {
            "prompt_tokens_sha256": getattr(sample, "prompt_tokens_sha256", None),
            "policy_sha256": getattr(sample, "policy_sha256", None),
            "execution_spec_sha256": getattr(
                sample, "execution_spec_sha256", None
            ),
            "branch_index": getattr(sample, "branch_index", None),
            "seed": getattr(sample, "seed", None),
            "tokens": list(getattr(sample, "tokens", ())),
            "behavior_logprobs": list(getattr(sample, "behavior_logprobs", ())),
            "sampling_config": _sampling_config_document(sample),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        return {
            "schema": "aura.verified_transition.sample_binding.v1",
            "sample_sha256": hashlib.sha256(encoded).hexdigest(),
        }
    encoded = json.dumps(
        dict(receipt),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return {
        "schema": "aura.verified_transition.sample_binding.v1",
        "sample_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def build_verified_transition_group_admission(
    store: TransitionArtifactStore,
    reward_receipt: Mapping[str, Any],
    evidence: Sequence[VerifiedTransitionEvidence],
    samples: Sequence[Any],
    prompt_tokens: Sequence[int],
    *,
    group_manifest: Mapping[str, Any],
    group_manifest_attestation: Mapping[str, Any],
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
    token_encoder: Callable[[bytes], Sequence[int]],
    token_decoder: Callable[[Sequence[int]], bytes],
    created_at_unix_ns: int,
) -> dict[str, Any]:
    """Prove exact planned membership before exposing an optimizer admission."""

    if len(evidence) != len(samples) or not evidence:
        _fail("group_admission_input_count_invalid")
    reward = validate_verified_transition_reward_batch(
        store,
        reward_receipt,
        evidence,
        independent_scorer=independent_scorer,
        token_encoder=token_encoder,
        token_decoder=token_decoder,
    )
    require_optimizer_admission(reward)
    manifest = validate_transition_group_manifest(group_manifest)
    reward_config_sha256 = hashlib.sha256(
        canonical_json_bytes(cast(Mapping[str, Any], reward["reward_config"]))
    ).hexdigest()
    if manifest["reward_config_sha256"] != reward_config_sha256:
        _fail("group_manifest_reward_config_mismatch")
    if manifest["task_id"] != reward["task_id"]:
        _fail("group_manifest_task_mismatch")
    actual_entries = [
        _actual_plan_entry(item, sample).to_dict()
        for item, sample in zip(evidence, samples, strict=True)
    ]
    if manifest["entries"] != actual_entries:
        _fail("group_manifest_membership_mismatch")
    if manifest["group_size"] != len(actual_entries):
        _fail("group_manifest_size_mismatch")
    if len({entry["policy_sha256"] for entry in actual_entries}) != 1:
        _fail("group_manifest_mixed_policies")
    if (
        len(
            {
                entry["recurrent_execution_spec_sha256"]
                for entry in actual_entries
            }
        )
        != 1
    ):
        _fail("group_manifest_mixed_execution_specs")

    policies = [item.trust_context.verified_policy() for item in evidence]
    policy_sha256 = policies[0].policy_sha256
    if any(policy.policy_sha256 != policy_sha256 for policy in policies):
        _fail("group_manifest_mixed_trust_policies")
    latest_allowed = min(_task_disclosure_signed_at(item) for item in evidence)
    signed = verify_role_attestation(
        policies[0],
        group_manifest_attestation,
        role=TASK_ISSUER,
        expected_payload=manifest,
        not_after_unix=latest_allowed,
    )
    signed_at_unix = _require_int(
        signed.get("signed_at_unix"),
        role="group_manifest_signed_at",
        minimum=1,
    )
    planned_at = _require_int(
        manifest["planned_at_unix_ns"],
        role="group_manifest_planned_at",
        minimum=1,
    )
    if planned_at // 1_000_000_000 != signed_at_unix:
        _fail("group_manifest_signature_time_mismatch")
    earliest_generation = min(
        _require_int(
            item.store.read_json(
                cast(Mapping[str, Any], item.episode["pass_0_artifact"]),
                role="reasoning_pass",
            )["generated_at_unix_ns"],
            role="group_episode_generated_at",
            minimum=1,
        )
        for item in evidence
    )
    if planned_at >= earliest_generation:
        _fail("group_manifest_not_pre_generation")

    rewards_for_recurrent_samples(reward, samples, prompt_tokens)
    prompt_payload = canonical_json_bytes(list(prompt_tokens))
    sample_bindings = [_sample_receipt(sample) for sample in samples]
    return _seal(
        {
            "schema": TRANSITION_GROUP_ADMISSION_SCHEMA,
            "group_id": manifest["group_id"],
            "task_id": manifest["task_id"],
            "group_size": len(actual_entries),
            "policy_sha256": actual_entries[0]["policy_sha256"],
            "recurrent_execution_spec_sha256": actual_entries[0][
                "recurrent_execution_spec_sha256"
            ],
            "prompt_tokens_sha256": hashlib.sha256(prompt_payload).hexdigest(),
            "reward_receipt_artifact": store.put_json(reward),
            "reward_receipt_sha256": reward["receipt_sha256"],
            "group_manifest_artifact": store.put_json(manifest),
            "group_manifest_sha256": manifest["manifest_sha256"],
            "group_manifest_attestation_artifact": store.put_json(
                cast(Mapping[str, Any], group_manifest_attestation)
            ),
            "trust_policy_sha256": policy_sha256,
            "entries": actual_entries,
            "sample_bindings": sample_bindings,
            "wrong_to_right": reward["wrong_to_right"],
            "right_to_wrong": reward["right_to_wrong"],
            "eir_defined": reward["eir_defined"],
            "eir_numerator": reward["eir_numerator"],
            "eir_denominator": reward["eir_denominator"],
            "eir_micros": reward["eir_micros"],
            "optimizer_admitted": True,
            "created_at_unix_ns": _require_int(
                created_at_unix_ns,
                role="group_admission_created_at",
                minimum=1,
            ),
        },
        field="receipt_sha256",
    )


def validate_verified_transition_group_admission(
    store: TransitionArtifactStore,
    receipt: Mapping[str, Any],
    reward_receipt: Mapping[str, Any],
    evidence: Sequence[VerifiedTransitionEvidence],
    samples: Sequence[Any],
    prompt_tokens: Sequence[int],
    *,
    group_manifest: Mapping[str, Any],
    group_manifest_attestation: Mapping[str, Any],
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]],
    token_encoder: Callable[[bytes], Sequence[int]],
    token_decoder: Callable[[Sequence[int]], bytes],
) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        _fail("group_admission_receipt_invalid")
    _validate_seal(receipt, field="receipt_sha256", role="group_admission")
    if receipt.get("schema") != TRANSITION_GROUP_ADMISSION_SCHEMA:
        _fail("group_admission_schema_invalid")
    expected = build_verified_transition_group_admission(
        store,
        reward_receipt,
        evidence,
        samples,
        prompt_tokens,
        group_manifest=group_manifest,
        group_manifest_attestation=group_manifest_attestation,
        independent_scorer=independent_scorer,
        token_encoder=token_encoder,
        token_decoder=token_decoder,
        created_at_unix_ns=_require_int(
            receipt.get("created_at_unix_ns"),
            role="group_admission_created_at",
            minimum=1,
        ),
    )
    if expected != dict(receipt):
        _fail("group_admission_reconstruction_mismatch")
    return dict(receipt)


__all__ = [
    "TRANSITION_GROUP_ADMISSION_SCHEMA",
    "TRANSITION_GROUP_MANIFEST_SCHEMA",
    "TransitionGroupPlanEntry",
    "VerifiedTransitionGroupError",
    "build_transition_group_manifest",
    "build_verified_transition_group_admission",
    "sampling_config_sha256",
    "validate_transition_group_manifest",
    "validate_verified_transition_group_admission",
]
