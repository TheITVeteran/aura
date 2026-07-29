"""End-to-end contracts for same-host isolated campaign signers."""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

import pytest

from core.brain.llm.latent_cortex.campaign_trust import (
    EVIDENCE_VERIFIER,
    TASK_ISSUER,
    externally_custodied_roles,
    operationally_isolated_roles,
    validate_campaign_trust_policy,
)
from core.learning.verified_transition_episode import canonical_json_bytes
from core.learning.verified_transition_production_factory import (
    CommandRoleSignerBroker,
    VerifiedTransitionProductionFactoryError,
)
from tools.provision_resident_campaign_trust import provision


def _read(path: str | Path) -> dict:
    return json.loads(Path(path).read_bytes())


def test_provisioned_role_keys_never_enter_disk_and_sign_only_sealed_policy(
    tmp_path: Path,
) -> None:
    campaign_id = "resident-test-campaign"
    protocol_sha256 = "a" * 64
    preregistration = tmp_path / "preregistration.json"
    preregistration.write_bytes(
        canonical_json_bytes(
            {
                "campaign_id": campaign_id,
                "contract_sha256": protocol_sha256,
            }
        )
    )
    trust_root = tmp_path / "trust"
    result = provision(
        preregistration_path=preregistration,
        output_root=trust_root,
        ttl_seconds=3600,
        key_custody="ephemeral",
    )
    try:
        policy = validate_campaign_trust_policy(
            _read(result["trust_policy_path"]),
            trusted_root_public_key_pem=Path(result["trust_root_path"]).read_bytes(),
            expected_campaign_name=campaign_id,
            expected_protocol_sha256=protocol_sha256,
            now_unix=result["issued_at_unix"],
        )
        config = _read(result["task_issuer_signer_config_path"])
        broker = CommandRoleSignerBroker(
            identity=config["identity"],
            executable=config["executable"],
            executable_sha256=config["executable_sha256"],
            release_manifest=config["release_manifest"],
            custody_evidence=config["custody_evidence"],
            arguments=config["arguments"],
            timeout_seconds=config["timeout_millis"] / 1000,
            inherited_environment_names=config["inherited_environment_names"],
        )

        attestation = broker.attest(
            policy,
            role=TASK_ISSUER,
            payload={"schema": "fixture.group", "value": 7},
            signed_at_unix=result["issued_at_unix"],
            purpose=f"{campaign_id}:group:0:manifest",
        )

        assert attestation["signed_payload"]["role"] == TASK_ISSUER
        assert attestation["signed_payload"]["policy_sha256"] == policy.policy_sha256
        assert attestation["signed_payload"]["operation"] == "group_manifest"
        assert attestation["signed_payload"]["purpose"] == (
            f"{campaign_id}:group:0:manifest"
        )
        campaign_attestation = broker.attest(
            policy,
            role=TASK_ISSUER,
            payload={"schema": "fixture.campaign", "value": 9},
            signed_at_unix=result["issued_at_unix"],
            purpose=f"{campaign_id}:campaign-manifest",
        )
        replayed_campaign_attestation = broker.attest(
            policy,
            role=TASK_ISSUER,
            payload={"schema": "fixture.campaign", "value": 9},
            signed_at_unix=result["issued_at_unix"],
            purpose=f"{campaign_id}:campaign-manifest",
        )
        assert replayed_campaign_attestation == campaign_attestation
        assert campaign_attestation["signed_payload"]["operation"] == (
            "campaign_manifest"
        )
        assert operationally_isolated_roles(policy) is True
        assert externally_custodied_roles(policy) is False

        verifier_config = _read(result["evidence_verifier_signer_config_path"])
        verifier = CommandRoleSignerBroker(
            identity=verifier_config["identity"],
            executable=verifier_config["executable"],
            executable_sha256=verifier_config["executable_sha256"],
            release_manifest=verifier_config["release_manifest"],
            custody_evidence=verifier_config["custody_evidence"],
            arguments=verifier_config["arguments"],
            timeout_seconds=verifier_config["timeout_millis"] / 1000,
            inherited_environment_names=verifier_config[
                "inherited_environment_names"
            ],
        )
        with pytest.raises(
            VerifiedTransitionProductionFactoryError,
            match="signer_broker_rejected_request",
        ):
            verifier.attest(
                policy,
                role=EVIDENCE_VERIFIER,
                payload={
                    "schema": "fixture.close",
                    "evidence_manifest": {"schema": "fixture.evidence"},
                    "external_evidence_verification_receipt": {
                        "schema": "fixture.receipt"
                    },
                },
                signed_at_unix=result["issued_at_unix"],
                purpose="verified-recurrent-campaign-close",
            )
        with pytest.raises(
            VerifiedTransitionProductionFactoryError,
            match="signer_broker_rejected_request",
        ):
            broker.attest(
                policy,
                role=TASK_ISSUER,
                payload={"schema": "fixture.group", "value": 8},
                signed_at_unix=result["issued_at_unix"],
                purpose="unauthorized-purpose",
            )
        assert result["private_role_keys_exported"] is False
        assert result["private_role_keys_persisted_in_keychain"] is False
        assert result["root_private_key_persisted"] is False
        assert not [
            path
            for path in trust_root.rglob("*")
            if path.is_file() and ("private" in path.name or path.suffix == ".key")
        ]
        reopened = provision(
            preregistration_path=preregistration,
            output_root=trust_root,
            ttl_seconds=3600,
        )
        assert reopened["reopened"] is True
        assert reopened["policy_sha256"] == result["policy_sha256"]
    finally:
        for service in result["services"].values():
            try:
                os.kill(service["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if all(
                not Path(service["socket_path"]).exists() for service in result["services"].values()
            ):
                break
            time.sleep(0.05)
