"""Adversarial custody tests for the causal verified-transition ledger."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_TRUST_POLICY_SCHEMA,
    CAMPAIGN_TRUST_ROLES,
    EVIDENCE_VERIFIER,
    TASK_ISSUER,
    build_role_attestation,
    policy_signed_payload,
    validate_campaign_trust_policy,
)
from core.learning.verified_transition_causal_campaign import (
    CAUSAL_CAMPAIGN_MANIFEST_SCHEMA,
    CausalCampaignScheduleEntry,
    VerifiedTransitionCausalCampaignError,
    VerifiedTransitionCausalCampaignLedger,
    build_causal_campaign_manifest,
    validate_causal_campaign_manifest,
)
from core.learning.verified_transition_episode import canonical_json_bytes
from core.learning.verified_transition_group_admission import (
    TransitionGroupPlanEntry,
    build_transition_group_manifest,
)

BASE_SECOND = 1_800_000_000


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _public_raw(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _public_pem(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _pin(role: str, key: Ed25519PrivateKey) -> dict[str, str]:
    public = _public_raw(key)
    return {
        "signer_id": f"{role}-signer",
        "organization_id": f"{role}-external-organization",
        "public_key_b64": base64.b64encode(public).decode("ascii"),
        "key_id": hashlib.sha256(public).hexdigest(),
        "implementation_sha256": _sha(f"{role}-implementation"),
        "release_sha256": _sha(f"{role}-release"),
        "custody_class": "external_service",
        "custody_evidence_sha256": _sha(f"{role}-custody"),
    }


def _trust_material() -> tuple[Any, dict[str, Ed25519PrivateKey]]:
    root = Ed25519PrivateKey.generate()
    role_keys = {
        role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES
    }
    body = {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": "causal-transition-campaign-2026-07",
        "policy_revision": 1,
        "campaign_name": "causal-transition-campaign",
        "protocol_sha256": _sha("causal-protocol"),
        "previous_policy_sha256": None,
        "revoked_key_ids": [],
        "issued_at_unix": BASE_SECOND,
        "not_before_unix": BASE_SECOND + 100,
        "expires_at_unix": BASE_SECOND + 10_000,
        "roles": {role: _pin(role, role_keys[role]) for role in CAMPAIGN_TRUST_ROLES},
    }
    signed = canonical_json_bytes(body)
    root_raw = _public_raw(root)
    document = {
        **body,
        "root_signature": {
            "algorithm": "Ed25519",
            "key_id": hashlib.sha256(root_raw).hexdigest(),
            "signature_b64": base64.b64encode(root.sign(signed)).decode("ascii"),
            "signed_payload_sha256": hashlib.sha256(signed).hexdigest(),
        },
    }
    assert policy_signed_payload(document) == body
    policy = validate_campaign_trust_policy(
        document,
        trusted_root_public_key_pem=_public_pem(root),
        expected_campaign_name="causal-transition-campaign",
        expected_protocol_sha256=_sha("causal-protocol"),
        now_unix=BASE_SECOND + 120,
    )
    return policy, role_keys


def _make_material(tmp_path: Path, *, group_count: int = 2) -> dict[str, Any]:
    policy, role_keys = _trust_material()
    initial_policy = _sha("initial-policy")
    schedule_root = _sha("causal-schedule-root")
    schedule = tuple(
        CausalCampaignScheduleEntry(
            sequence=sequence,
            task_id=f"task-{sequence}",
            task_commitment_sha256=_sha(f"task-commitment-{sequence}"),
        )
        for sequence in range(group_count)
    )
    planned_second = BASE_SECOND + 150
    manifest = build_causal_campaign_manifest(
        campaign_id="causal-campaign-001",
        campaign_schedule_root_sha256=schedule_root,
        trust_policy_sha256=policy.policy_sha256,
        initial_policy_sha256=initial_policy,
        schedule=schedule,
        planned_at_unix_ns=planned_second * 1_000_000_000,
    )
    manifest_attestation = build_role_attestation(
        policy,
        role=TASK_ISSUER,
        payload=manifest,
        signed_at_unix=planned_second,
        private_key=role_keys[TASK_ISSUER],
    )
    root = tmp_path / "campaign"
    ledger = VerifiedTransitionCausalCampaignLedger.create(
        root,
        campaign_manifest=manifest,
        campaign_manifest_attestation=manifest_attestation,
        policy=policy,
    )
    return {
        "policy": policy,
        "role_keys": role_keys,
        "manifest": manifest,
        "initial_policy": initial_policy,
        "schedule_root": schedule_root,
        "schedule": schedule,
        "ledger": ledger,
        "root": root,
    }


@pytest.fixture
def material(tmp_path: Path) -> dict[str, Any]:
    return _make_material(tmp_path)


def _group(
    material: dict[str, Any],
    *,
    sequence: int,
    policy_before: str,
    planned_second: int | None = None,
    manifest_signed_second: int | None = None,
    lineage_signed_second: int | None = None,
) -> dict[str, Any]:
    second = BASE_SECOND + 160 + sequence * 10 if planned_second is None else planned_second
    entries = tuple(
        TransitionGroupPlanEntry(
            episode_id=f"episode-{sequence}-{index}",
            task_id=f"task-{sequence}",
            rng_root_sha256=_sha(f"rng-{sequence}-{index}"),
            policy_sha256=policy_before,
            recurrent_execution_spec_sha256=_sha("execution-spec"),
            producing_branch_index=index,
            sample_seed=1000 + sequence * 10 + index,
            sampling_config_sha256=_sha(f"sampling-{sequence}-{index}"),
        )
        for index in range(2)
    )
    manifest = build_transition_group_manifest(
        group_id=f"group-{sequence}",
        task_id=f"task-{sequence}",
        entries=entries,
        reward_config_sha256=_sha("reward-config"),
        planned_at_unix_ns=second * 1_000_000_000,
    )
    lineage = {
        "schema": "aura.verified_transition.lineage_plan.v1",
        "contract_sha256": _sha("provider-contract"),
        "campaign_id": material["manifest"]["campaign_id"],
        "campaign_schedule_root_sha256": material["schedule_root"],
        "sequence": sequence,
        "task_commitment_sha256": material["schedule"][
            sequence
        ].task_commitment_sha256,
        "policy_before_sha256": policy_before,
        "group_manifest_sha256": manifest["manifest_sha256"],
    }
    manifest_signed = second if manifest_signed_second is None else manifest_signed_second
    lineage_signed = second if lineage_signed_second is None else lineage_signed_second
    return {
        "manifest": manifest,
        "lineage": lineage,
        "manifest_attestation": build_role_attestation(
            material["policy"],
            role=TASK_ISSUER,
            payload=manifest,
            signed_at_unix=manifest_signed,
            private_key=material["role_keys"][TASK_ISSUER],
        ),
        "lineage_attestation": build_role_attestation(
            material["policy"],
            role=TASK_ISSUER,
            payload=lineage,
            signed_at_unix=lineage_signed,
            private_key=material["role_keys"][TASK_ISSUER],
        ),
        "planned_second": second,
    }


def _admit(
    material: dict[str, Any],
    *,
    sequence: int,
    policy_before: str,
    planned_second: int | None = None,
    admitted_second: int | None = None,
    manifest_signed_second: int | None = None,
    lineage_signed_second: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    group = _group(
        material,
        sequence=sequence,
        policy_before=policy_before,
        planned_second=planned_second,
        manifest_signed_second=manifest_signed_second,
        lineage_signed_second=lineage_signed_second,
    )
    admitted = (
        group["planned_second"] + 1
        if admitted_second is None
        else admitted_second
    )
    start = material["ledger"].admit_group_plan(
        sequence=sequence,
        campaign_id=material["manifest"]["campaign_id"],
        campaign_schedule_root_sha256=material["schedule_root"],
        policy_before_sha256=policy_before,
        group_manifest=group["manifest"],
        group_manifest_attestation=group["manifest_attestation"],
        lineage_plan=group["lineage"],
        lineage_attestation=group["lineage_attestation"],
        policy=material["policy"],
        admitted_at_unix_ns=admitted * 1_000_000_000,
    )
    return dict(start), group


def _finish_updated(
    material: dict[str, Any],
    *,
    sequence: int,
    policy_after: str,
) -> dict[str, Any]:
    terminal = material["ledger"].finish_group(
        sequence=sequence,
        status="updated",
        reward_receipt_sha256=_sha(f"reward-{sequence}"),
        group_admission_sha256=_sha(f"admission-{sequence}"),
        update_receipt_sha256=_sha(f"update-{sequence}"),
        policy_after_sha256=policy_after,
        terminal_reason="optimizer_update_committed",
        finished_at_unix_ns=(BASE_SECOND + 162 + sequence * 10) * 1_000_000_000,
    )
    return dict(terminal)


def _complete(material: dict[str, Any]) -> tuple[str, str]:
    policy_1 = _sha("actual-policy-after-0")
    policy_2 = _sha("actual-policy-after-1")
    _admit(material, sequence=0, policy_before=material["initial_policy"])
    _finish_updated(material, sequence=0, policy_after=policy_1)
    _admit(material, sequence=1, policy_before=policy_1)
    _finish_updated(material, sequence=1, policy_after=policy_2)
    return policy_1, policy_2


def _close(material: dict[str, Any]) -> dict[str, Any]:
    completed_second = BASE_SECOND + 181
    payload = material["ledger"].close_payload(
        completed_at_unix_ns=completed_second * 1_000_000_000,
        policy=material["policy"],
    )
    attestation = build_role_attestation(
        material["policy"],
        role=EVIDENCE_VERIFIER,
        payload=payload,
        signed_at_unix=completed_second,
        private_key=material["role_keys"][EVIDENCE_VERIFIER],
    )
    return dict(
        material["ledger"].close(
            close_payload=payload,
            evidence_verifier_attestation=attestation,
            policy=material["policy"],
        )
    )


def _rewrite_canonical(path: Path, document: dict[str, Any]) -> None:
    unsigned = dict(document)
    unsigned.pop("receipt_sha256", None)
    document["receipt_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    path.write_bytes(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )


def test_manifest_precommits_only_knowable_schedule_facts(
    material: dict[str, Any],
) -> None:
    manifest = validate_causal_campaign_manifest(material["manifest"])
    encoded = canonical_json_bytes(manifest).decode("ascii")

    assert manifest["schema"] == CAUSAL_CAMPAIGN_MANIFEST_SCHEMA
    assert manifest["initial_policy_sha256"] == material["initial_policy"]
    assert "policy_before_sha256" not in encoded
    assert "policy_after_sha256" not in encoded
    assert "group_manifest_sha256" not in encoded
    assert "group_manifest" not in encoded

    attacked = copy.deepcopy(manifest)
    attacked["future_policy_sha256"] = _sha("unknowable-future-policy")
    with pytest.raises(
        VerifiedTransitionCausalCampaignError,
        match="manifest_schema_invalid",
    ):
        validate_causal_campaign_manifest(attacked)


def test_jit_groups_follow_actual_policy_lineage_and_reopen(
    material: dict[str, Any],
) -> None:
    policy_1, policy_2 = _complete(material)
    start_0, terminal_0 = material["ledger"].group_records_unclosed(sequence=0)
    assert start_0["policy_before_sha256"] == material["initial_policy"]
    assert terminal_0["policy_after_sha256"] == policy_1

    reopened = VerifiedTransitionCausalCampaignLedger.open(
        material["root"], policy=material["policy"]
    )
    start_1, terminal_1 = reopened.group_records_unclosed(sequence=1)
    assert start_1["policy_before_sha256"] == policy_1
    assert terminal_1["policy_after_sha256"] == policy_2


def test_open_group_exposes_validated_start_without_inventing_terminal(
    material: dict[str, Any],
) -> None:
    assert material["ledger"].group_start_if_exists(sequence=0) is None
    start, _group_material = _admit(
        material, sequence=0, policy_before=material["initial_policy"]
    )

    assert material["ledger"].group_start(sequence=0) == start
    assert material["ledger"].group_start_if_exists(sequence=0) == start
    assert material["ledger"].group_terminal_if_exists(sequence=0) is None

    terminal = _finish_updated(
        material, sequence=0, policy_after=_sha("recovered-policy-after")
    )
    assert material["ledger"].group_terminal_if_exists(sequence=0) == terminal


def test_post_disclosure_group_plan_is_rejected(material: dict[str, Any]) -> None:
    boundary = BASE_SECOND + 160
    with pytest.raises(
        VerifiedTransitionCausalCampaignError,
        match="causal_group_plan_post_disclosure",
    ):
        _admit(
            material,
            sequence=0,
            policy_before=material["initial_policy"],
            planned_second=boundary,
            admitted_second=boundary,
        )
    assert not (material["root"] / "group-00000000.started.json").exists()


def test_provider_accepted_early_signatures_are_ledger_compatible(
    material: dict[str, Any],
) -> None:
    start, _group_material = _admit(
        material,
        sequence=0,
        policy_before=material["initial_policy"],
        manifest_signed_second=BASE_SECOND + 130,
        lineage_signed_second=BASE_SECOND + 131,
    )
    assert start["sequence"] == 0
    assert start["policy_before_sha256"] == material["initial_policy"]


def test_policy_lineage_substitution_is_rejected(material: dict[str, Any]) -> None:
    actual_policy = _sha("actual-policy-after-0")
    _admit(material, sequence=0, policy_before=material["initial_policy"])
    _finish_updated(material, sequence=0, policy_after=actual_policy)

    substituted = _sha("substituted-policy-after-0")
    with pytest.raises(
        VerifiedTransitionCausalCampaignError,
        match="sequence_or_lineage_invalid",
    ):
        _admit(material, sequence=1, policy_before=substituted)
    assert not (material["root"] / "group-00000001.started.json").exists()


def test_updated_terminal_requires_actual_policy_after(
    material: dict[str, Any],
) -> None:
    _admit(material, sequence=0, policy_before=material["initial_policy"])
    with pytest.raises(
        VerifiedTransitionCausalCampaignError,
        match="terminal_policy_after_invalid",
    ):
        material["ledger"].finish_group(
            sequence=0,
            status="updated",
            reward_receipt_sha256=_sha("reward-0"),
            group_admission_sha256=_sha("admission-0"),
            update_receipt_sha256=_sha("update-0"),
            terminal_reason="optimizer_update_committed",
            finished_at_unix_ns=(BASE_SECOND + 162) * 1_000_000_000,
        )


def test_indeterminate_terminal_blocks_further_policy_lineage(
    material: dict[str, Any],
) -> None:
    _admit(material, sequence=0, policy_before=material["initial_policy"])
    terminal = material["ledger"].finish_group(
        sequence=0,
        status="indeterminate",
        reward_receipt_sha256=_sha("reward-observed-before-crash"),
        group_admission_sha256=_sha("admission-observed-before-crash"),
        update_receipt_sha256=None,
        terminal_reason="optimizer_state_requires_recovery",
        finished_at_unix_ns=(BASE_SECOND + 162) * 1_000_000_000,
    )
    assert terminal["policy_after_sha256"] is None
    with pytest.raises(
        VerifiedTransitionCausalCampaignError,
        match="policy_lineage_indeterminate",
    ):
        _admit(
            material,
            sequence=1,
            policy_before=material["initial_policy"],
        )


@pytest.mark.parametrize(
    "field",
    ["reward_receipt_sha256", "update_receipt_sha256"],
)
def test_closed_campaign_rejects_terminal_evidence_substitution(
    material: dict[str, Any], field: str
) -> None:
    _complete(material)
    _close(material)
    path = material["root"] / "group-00000000.terminal.json"
    terminal = json.loads(path.read_bytes())
    terminal[field] = _sha(f"substituted-{field}")
    _rewrite_canonical(path, terminal)

    with pytest.raises(
        VerifiedTransitionCausalCampaignError,
        match="previous_terminal_mismatch|close_reconstruction_mismatch",
    ):
        material["ledger"].validate_closed(policy=material["policy"])


def test_incomplete_campaign_cannot_produce_close_payload(
    material: dict[str, Any],
) -> None:
    _admit(material, sequence=0, policy_before=material["initial_policy"])
    _finish_updated(
        material, sequence=0, policy_after=_sha("actual-policy-after-0")
    )
    with pytest.raises(
        VerifiedTransitionCausalCampaignError,
        match="causal_campaign_incomplete:sequence=1",
    ):
        material["ledger"].close_payload(
            completed_at_unix_ns=(BASE_SECOND + 181) * 1_000_000_000,
            policy=material["policy"],
        )


def test_close_requires_external_evidence_verifier_signature(
    material: dict[str, Any],
) -> None:
    _policy_1, final_policy = _complete(material)
    completed_second = BASE_SECOND + 181
    payload = material["ledger"].close_payload(
        completed_at_unix_ns=completed_second * 1_000_000_000,
        policy=material["policy"],
    )
    wrong_role = build_role_attestation(
        material["policy"],
        role=TASK_ISSUER,
        payload=payload,
        signed_at_unix=completed_second,
        private_key=material["role_keys"][TASK_ISSUER],
    )
    with pytest.raises(ValueError, match="campaign_attestation_identity_mismatch"):
        material["ledger"].close(
            close_payload=payload,
            evidence_verifier_attestation=wrong_role,
            policy=material["policy"],
        )

    verifier = build_role_attestation(
        material["policy"],
        role=EVIDENCE_VERIFIER,
        payload=payload,
        signed_at_unix=completed_second,
        private_key=material["role_keys"][EVIDENCE_VERIFIER],
    )
    receipt = material["ledger"].close(
        close_payload=payload,
        evidence_verifier_attestation=verifier,
        policy=material["policy"],
    )
    assert receipt["close_payload"]["final_policy_sha256"] == final_policy
    assert material["ledger"].validate_closed(policy=material["policy"]) == receipt
    start, terminal = material["ledger"].group_records(
        sequence=1, policy=material["policy"]
    )
    assert start["receipt_sha256"] == payload["group_start_sha256s"][1]
    assert terminal["receipt_sha256"] == payload["group_terminal_sha256s"][1]


def test_append_validation_work_is_constant_in_campaign_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    material = _make_material(tmp_path, group_count=32)
    policy_before = material["initial_policy"]
    for sequence in range(31):
        _admit(material, sequence=sequence, policy_before=policy_before)
        policy_after = _sha(f"scale-policy-{sequence}")
        _finish_updated(material, sequence=sequence, policy_after=policy_after)
        policy_before = policy_after

    observed_reads = 0
    original_read = VerifiedTransitionCausalCampaignLedger._read

    def counted_read(
        self: VerifiedTransitionCausalCampaignLedger, name: str
    ) -> dict[str, Any]:
        nonlocal observed_reads
        if self is material["ledger"]:
            observed_reads += 1
        return original_read(self, name)

    monkeypatch.setattr(VerifiedTransitionCausalCampaignLedger, "_read", counted_read)
    _admit(material, sequence=31, policy_before=policy_before)
    assert observed_reads <= 8


def test_create_once_start_survives_two_ledger_instances_racing(
    material: dict[str, Any],
) -> None:
    group = _group(
        material,
        sequence=0,
        policy_before=material["initial_policy"],
    )
    ledgers = (
        material["ledger"],
        VerifiedTransitionCausalCampaignLedger.open(
            material["root"], policy=material["policy"]
        ),
    )

    def admit(ledger: VerifiedTransitionCausalCampaignLedger) -> str:
        try:
            ledger.admit_group_plan(
                sequence=0,
                campaign_id=material["manifest"]["campaign_id"],
                campaign_schedule_root_sha256=material["schedule_root"],
                policy_before_sha256=material["initial_policy"],
                group_manifest=group["manifest"],
                group_manifest_attestation=group["manifest_attestation"],
                lineage_plan=group["lineage"],
                lineage_attestation=group["lineage_attestation"],
                policy=material["policy"],
                admitted_at_unix_ns=(group["planned_second"] + 1)
                * 1_000_000_000,
            )
        except VerifiedTransitionCausalCampaignError:
            return "rejected"
        return "created"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(admit, ledgers))
    assert sorted(outcomes) == ["created", "rejected"]
    assert (material["root"] / "group-00000000.started.json").is_file()
