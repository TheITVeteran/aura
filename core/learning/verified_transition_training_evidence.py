"""Independent source-artifact replay for verified recurrent training."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Never, cast

from core.brain.llm.latent_cortex.campaign_trust import VerifiedCampaignTrustPolicy
from core.learning.verified_transition_campaign import (
    VerifiedTransitionCampaignLedger,
)
from core.learning.verified_transition_episode import (
    TransitionArtifactStore,
    canonical_json_bytes,
)
from core.learning.verified_transition_group_admission import (
    validate_verified_transition_group_admission,
)
from core.learning.verified_transition_reward import VerifiedTransitionEvidence
from core.learning.verified_transition_update import (
    VerifiedTransitionUpdateJournal,
    validate_verified_transition_update_receipt,
)

VERIFIED_TRAINING_EVIDENCE_SCHEMA = "aura.verified_transition.training_evidence.v1"


class VerifiedTransitionTrainingEvidenceError(ValueError):
    """Stable independent training-evidence failure."""


def _fail(code: str) -> Never:
    raise VerifiedTransitionTrainingEvidenceError(code)


def _seal(document: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(document)
    sealed["receipt_sha256"] = hashlib.sha256(canonical_json_bytes(sealed)).hexdigest()
    return sealed


def _sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{role}_invalid")
    return value


def validate_verified_transition_training_evidence_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the sealed replay summary embedded in an adapter identity."""

    required = {
        "schema",
        "campaign_receipt_sha256",
        "campaign_manifest_sha256",
        "updated_sequences",
        "group_admission_sha256s",
        "update_receipt_sha256s",
        "objective_receipt_sha256s",
        "optimizer_update_count",
        "initial_policy_sha256",
        "final_policy_sha256",
        "source_artifacts_replayed",
        "legacy_scalar_reward_path_used",
        "receipt_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required:
        _fail("verified_training_evidence_receipt_schema_invalid")
    normalized = dict(receipt)
    if normalized.get("schema") != VERIFIED_TRAINING_EVIDENCE_SCHEMA:
        _fail("verified_training_evidence_receipt_version_invalid")
    observed = _sha256(
        normalized.get("receipt_sha256"), role="verified_training_evidence_receipt"
    )
    unsigned = dict(normalized)
    unsigned.pop("receipt_sha256")
    if observed != hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest():
        _fail("verified_training_evidence_receipt_digest_mismatch")
    for field in ("campaign_receipt_sha256", "campaign_manifest_sha256"):
        _sha256(normalized.get(field), role=f"verified_training_{field}")
    sequences = normalized.get("updated_sequences")
    admissions = normalized.get("group_admission_sha256s")
    updates = normalized.get("update_receipt_sha256s")
    objectives = normalized.get("objective_receipt_sha256s")
    count = normalized.get("optimizer_update_count")
    if (
        not isinstance(sequences, list)
        or any(type(sequence) is not int or sequence < 0 for sequence in sequences)
        or sequences != sorted(set(sequences))
        or not isinstance(admissions, list)
        or not isinstance(updates, list)
        or not isinstance(objectives, list)
        or type(count) is not int
        or count < 1
        or not len(sequences) == len(admissions) == len(updates) == len(objectives) == count
        or normalized.get("source_artifacts_replayed") is not True
        or normalized.get("legacy_scalar_reward_path_used") is not False
    ):
        _fail("verified_training_evidence_receipt_invalid")
    for role, values in (
        ("admission", admissions),
        ("update", updates),
        ("objective", objectives),
    ):
        for value in values:
            _sha256(value, role=f"verified_training_{role}")
    _sha256(normalized.get("initial_policy_sha256"), role="verified_training_initial_policy")
    _sha256(normalized.get("final_policy_sha256"), role="verified_training_final_policy")
    return normalized


@dataclass(frozen=True, slots=True)
class VerifiedTransitionReplayGroup:
    sequence: int
    transition_store: TransitionArtifactStore
    group_admission_receipt: Mapping[str, Any]
    reward_receipt: Mapping[str, Any]
    transition_evidence: Sequence[VerifiedTransitionEvidence]
    samples: Sequence[Any]
    prompt_tokens: Sequence[int]
    group_manifest: Mapping[str, Any]
    group_manifest_attestation: Mapping[str, Any]
    independent_scorer: Callable[[Any, Any], Mapping[str, Any]]
    token_encoder: Callable[[bytes], Sequence[int]]
    token_decoder: Callable[[Sequence[int]], bytes]
    update_journal: VerifiedTransitionUpdateJournal
    update_receipt: Mapping[str, Any]


def validate_verified_transition_training_evidence(
    campaign_ledger: VerifiedTransitionCampaignLedger,
    *,
    policy: VerifiedCampaignTrustPolicy,
    groups: Sequence[VerifiedTransitionReplayGroup],
) -> dict[str, Any]:
    """Replay every source artifact behind every committed optimizer update."""

    campaign = campaign_ledger.validate_closed(policy=policy)
    close = cast(dict[str, Any], campaign["close_payload"])
    updated_sequences = [
        index
        for index, status in enumerate(cast(list[str], close["group_statuses"]))
        if status == "updated"
    ]
    if [group.sequence for group in groups] != updated_sequences:
        _fail("verified_training_updated_group_set_mismatch")
    update_receipts: list[str] = []
    admission_receipts: list[str] = []
    objective_receipts: list[str] = []
    initial_policy_before: str | None = None
    previous_policy_after: str | None = None
    for group in groups:
        start, terminal = campaign_ledger.group_records(
            sequence=group.sequence,
            policy=policy,
        )
        if terminal.get("status") != "updated":
            _fail("verified_training_group_not_updated")
        admission = validate_verified_transition_group_admission(
            group.transition_store,
            group.group_admission_receipt,
            group.reward_receipt,
            group.transition_evidence,
            group.samples,
            group.prompt_tokens,
            group_manifest=group.group_manifest,
            group_manifest_attestation=group.group_manifest_attestation,
            independent_scorer=group.independent_scorer,
            token_encoder=group.token_encoder,
            token_decoder=group.token_decoder,
        )
        update = validate_verified_transition_update_receipt(
            group.update_journal,
            group.update_receipt,
        )
        if (
            start.get("group_manifest") != dict(group.group_manifest)
            or terminal.get("group_admission_sha256")
            != admission.get("receipt_sha256")
            or terminal.get("update_receipt_sha256")
            != update.get("receipt_sha256")
            or update.get("group_admission_sha256")
            != admission.get("receipt_sha256")
            or admission.get("policy_sha256")
            != update.get("policy_before_sha256")
        ):
            _fail("verified_training_group_source_binding_mismatch")
        policy_before = cast(str, update["policy_before_sha256"])
        policy_after = cast(str, update["policy_after_sha256"])
        if initial_policy_before is None:
            initial_policy_before = policy_before
        if previous_policy_after is not None and policy_before != previous_policy_after:
            _fail("verified_training_policy_chain_mismatch")
        previous_policy_after = policy_after
        admission_receipts.append(cast(str, admission["receipt_sha256"]))
        update_receipts.append(cast(str, update["receipt_sha256"]))
        objective_receipts.append(cast(str, update["objective_receipt_sha256"]))
    if len(update_receipts) != close["updated_count"]:
        _fail("verified_training_update_count_mismatch")
    return validate_verified_transition_training_evidence_receipt(_seal(
        {
            "schema": VERIFIED_TRAINING_EVIDENCE_SCHEMA,
            "campaign_receipt_sha256": campaign["receipt_sha256"],
            "campaign_manifest_sha256": close["campaign_manifest_sha256"],
            "updated_sequences": updated_sequences,
            "group_admission_sha256s": admission_receipts,
            "update_receipt_sha256s": update_receipts,
            "objective_receipt_sha256s": objective_receipts,
            "optimizer_update_count": len(update_receipts),
            "initial_policy_sha256": initial_policy_before,
            "final_policy_sha256": previous_policy_after,
            "source_artifacts_replayed": True,
            "legacy_scalar_reward_path_used": False,
        }
    ))


__all__ = [
    "VERIFIED_TRAINING_EVIDENCE_SCHEMA",
    "VerifiedTransitionReplayGroup",
    "VerifiedTransitionTrainingEvidenceError",
    "validate_verified_transition_training_evidence_receipt",
    "validate_verified_transition_training_evidence",
]
