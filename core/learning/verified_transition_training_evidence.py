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
        ):
            _fail("verified_training_group_source_binding_mismatch")
        policy_before = cast(str, update["policy_before_sha256"])
        policy_after = cast(str, update["policy_after_sha256"])
        if previous_policy_after is not None and policy_before != previous_policy_after:
            _fail("verified_training_policy_chain_mismatch")
        previous_policy_after = policy_after
        admission_receipts.append(cast(str, admission["receipt_sha256"]))
        update_receipts.append(cast(str, update["receipt_sha256"]))
    if len(update_receipts) != close["updated_count"]:
        _fail("verified_training_update_count_mismatch")
    return _seal(
        {
            "schema": VERIFIED_TRAINING_EVIDENCE_SCHEMA,
            "campaign_receipt_sha256": campaign["receipt_sha256"],
            "campaign_manifest_sha256": close["campaign_manifest_sha256"],
            "updated_sequences": updated_sequences,
            "group_admission_sha256s": admission_receipts,
            "update_receipt_sha256s": update_receipts,
            "optimizer_update_count": len(update_receipts),
            "final_policy_sha256": previous_policy_after,
            "source_artifacts_replayed": True,
            "legacy_scalar_reward_path_used": False,
        }
    )


__all__ = [
    "VERIFIED_TRAINING_EVIDENCE_SCHEMA",
    "VerifiedTransitionReplayGroup",
    "VerifiedTransitionTrainingEvidenceError",
    "validate_verified_transition_training_evidence",
]
