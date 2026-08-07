from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex import live_adapter_activation as activation
from core.brain.llm.latent_cortex.campaign_journal import (
    CampaignPlan,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_TRUST_POLICY_SCHEMA,
    CAMPAIGN_TRUST_ROLES,
    EVIDENCE_VERIFIER,
    build_role_attestation,
    validate_campaign_trust_policy,
)
from tools import materialize_live_recurrent_adapter_activation as materializer


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


def _policy(
    campaign_name: str,
) -> tuple[
    dict[str, Any],
    Ed25519PrivateKey,
    dict[str, Ed25519PrivateKey],
]:
    root = Ed25519PrivateKey.generate()
    role_keys = {
        role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES
    }
    roles = {}
    for role, key in role_keys.items():
        public = _public_raw(key)
        roles[role] = {
            "signer_id": f"{role}-signer",
            "organization_id": f"{role}-organization",
            "public_key_b64": base64.b64encode(public).decode("ascii"),
            "key_id": hashlib.sha256(public).hexdigest(),
            "implementation_sha256": hashlib.sha256(
                f"{role}:implementation".encode()
            ).hexdigest(),
            "release_sha256": hashlib.sha256(
                f"{role}:release".encode()
            ).hexdigest(),
            "custody_class": "external_service",
            "custody_evidence_sha256": hashlib.sha256(
                f"{role}:custody".encode()
            ).hexdigest(),
        }
    body = {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": "live-adapter-test-policy",
        "policy_revision": 1,
        "campaign_name": campaign_name,
        "protocol_sha256": "9" * 64,
        "previous_policy_sha256": None,
        "revoked_key_ids": [],
        "issued_at_unix": 1_800_000_000,
        "not_before_unix": 1_800_000_100,
        "expires_at_unix": 1_800_086_400,
        "roles": roles,
    }
    signed = canonical_json_bytes(body)
    root_raw = _public_raw(root)
    return (
        {
            **body,
            "root_signature": {
                "algorithm": "Ed25519",
                "key_id": hashlib.sha256(root_raw).hexdigest(),
                "signature_b64": base64.b64encode(
                    root.sign(signed)
                ).decode("ascii"),
                "signed_payload_sha256": hashlib.sha256(signed).hexdigest(),
            },
        },
        root,
        role_keys,
    )


def _write(path: Path, value: Any) -> dict[str, Any]:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o600)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _fixture(
    tmp_path: Path,
    *,
    verdict_tier: str = "PROVEN",
) -> tuple[Path, bytes, dict[str, Any]]:
    campaign_name = "resident-32b-live-activation-test"
    package_manifest = {
        "schema": activation.ROLE_CONDITIONED_MANIFEST_SCHEMA,
        "adapter_id": "role-v6-test",
    }
    adapter_identity = {
        "schema": "aura.resident_recurrent_sft_adapter_identity_receipt.v1",
        "adapter_id": "role-v6-test",
        "manifest_sha256": hashlib.sha256(
            canonical_json_bytes(package_manifest)
        ).hexdigest(),
        "composite_identity_sha256": "b" * 64,
        "base_checkpoint_fingerprint": "c" * 64,
        "model_behavior_bundle_sha256": "d" * 64,
    }
    plan = CampaignPlan.build(
        campaign_name,
        [{"arm": "adapter_rlc", "task_id": "test-task"}],
        metadata={
            "claim_eligible": True,
            "adapter_identity": {
                "format": activation.ROLE_CONDITIONED_MANIFEST_SCHEMA,
                "identity_receipt": adapter_identity,
            },
            "model_identity": {
                "fingerprint": adapter_identity["base_checkpoint_fingerprint"],
            },
        },
    )
    campaign_dir = tmp_path / "campaign"
    plan_binding = _write(campaign_dir / "plan.json", plan.to_dict())
    verdict = {
        "schema": activation.INDEPENDENT_VERDICT_SCHEMA,
        "campaign_dir": str(campaign_dir.resolve()),
        "passed": True,
        "claim_tier": verdict_tier,
        "verified_verdict": "gain_proven",
        "failures": [],
        "recomputed_verdict": "gain_preverified",
        "independent_verdict": "gain_preverified",
        "published_verdict": "gain_preverified",
        "production_semantic_grade_sha256": "1" * 64,
        "independent_semantic_grade_sha256": "1" * 64,
        "production_grade_implementation_sha256": "2" * 64,
        "independent_scoring_implementation_sha256": "3" * 64,
        "verifier_implementation_sha256": "4" * 64,
        "verifier_attestation_sha256": "5" * 64,
        "plan_sha256": plan.plan_sha256,
        "answer_reveal": {"required": True, "verified": True},
        "worker_origins": {"required": True, "verified": True},
        "final_run": {"required": True, "verified": True},
    }
    verdict_binding = _write(campaign_dir / "verdict.json", verdict)

    package = tmp_path / "releases" / "role-v6-test"
    package.mkdir(parents=True)
    _write(
        package / "recurrence_adapter_manifest.json",
        package_manifest,
    )
    _write(package / "training_completion.json", {"complete": True})
    (package / "adapter.safetensors").write_bytes(b"fixture")
    (package / "adapter.safetensors").chmod(0o600)

    policy_document, root, role_keys = _policy(campaign_name)
    policy_binding = _write(tmp_path / "campaign-policy.json", policy_document)
    verified_policy = validate_campaign_trust_policy(
        policy_document,
        trusted_root_public_key_pem=_public_pem(root),
        expected_campaign_name=campaign_name,
        now_unix=1_800_000_200,
    )
    activation_document = activation.build_live_adapter_activation(
        campaign_name=campaign_name,
        policy_sha256=verified_policy.policy_sha256,
        adapter_id="role-v6-test",
        adapter_package_path=package,
        adapter_manifest_sha256=adapter_identity["manifest_sha256"],
        adapter_composite_identity_sha256=adapter_identity[
            "composite_identity_sha256"
        ],
        base_checkpoint_fingerprint=adapter_identity[
            "base_checkpoint_fingerprint"
        ],
        model_behavior_bundle_sha256=adapter_identity[
            "model_behavior_bundle_sha256"
        ],
        campaign_plan=plan_binding,
        independent_verdict=verdict_binding,
        not_before_unix=1_800_000_200,
        expires_at_unix=1_800_010_000,
    )
    attestation = build_role_attestation(
        verified_policy,
        role=EVIDENCE_VERIFIER,
        payload=activation_document,
        signed_at_unix=1_800_000_300,
        private_key=role_keys[EVIDENCE_VERIFIER],
    )
    attestation_binding = _write(
        tmp_path / "activation-attestation.json",
        attestation,
    )
    pointer = activation.build_live_adapter_pointer(
        activation=activation_document,
        campaign_policy=policy_binding,
        activation_attestation=attestation_binding,
    )
    pointer_path = tmp_path / "active.json"
    _write(pointer_path, pointer)
    return pointer_path, _public_pem(root), adapter_identity


def _admit(
    monkeypatch: pytest.MonkeyPatch,
    pointer_path: Path,
    root_pem: bytes,
    adapter_identity: dict[str, Any],
    tmp_path: Path,
) -> dict[str, Any]:
    monkeypatch.setattr(activation, "declared_bindings", lambda _manifest: ())
    monkeypatch.setattr(activation, "inspect_mlx_tensor_metadata", lambda _path: ())
    monkeypatch.setattr(
        activation,
        "validate_resident_recurrent_sft_adapter_identity",
        lambda *_args, **_kwargs: dict(adapter_identity),
    )
    return activation.admit_live_adapter_activation(
        pointer_path,
        trusted_root_public_key_pem=root_pem,
        approved_adapter_roots=[tmp_path / "releases"],
        actual_base_checkpoint={
            "fingerprint": adapter_identity["base_checkpoint_fingerprint"],
            "method": "sha256",
            "files": 1,
        },
        actual_model_behavior_bundle={
            "bundle_sha256": adapter_identity["model_behavior_bundle_sha256"],
        },
        actual_personality_adapter={
            "present": False,
            "identity_sha256": "e" * 64,
        },
        actual_runtime_environment={"identity_sha256": "f" * 64},
        now_unix=1_800_000_400,
    )


def test_live_adapter_activation_requires_signed_positive_independent_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointer, root_pem, identity = _fixture(tmp_path)

    receipt = _admit(monkeypatch, pointer, root_pem, identity, tmp_path)

    assert receipt["claim_tier"] == "PROVEN"
    assert receipt["verified_verdict"] == "gain_proven"
    assert receipt["adapter_identity"] == identity
    assert receipt["runtime_contract"] == activation._RUNTIME_CONTRACT
    assert len(receipt["receipt_sha256"]) == 64


def test_live_adapter_builders_reject_invalid_activation_window(tmp_path: Path) -> None:
    binding = {"path": str((tmp_path / "artifact").resolve()), "sha256": "a" * 64, "size_bytes": 1}

    with pytest.raises(
        activation.LiveAdapterActivationError,
        match="live_adapter_activation_window_invalid",
    ):
        activation.build_live_adapter_activation(
            campaign_name="campaign",
            policy_sha256="b" * 64,
            adapter_id="adapter",
            adapter_package_path=tmp_path,
            adapter_manifest_sha256="c" * 64,
            adapter_composite_identity_sha256="d" * 64,
            base_checkpoint_fingerprint="e" * 64,
            model_behavior_bundle_sha256="f" * 64,
            campaign_plan=binding,
            independent_verdict=binding,
            not_before_unix=10,
            expires_at_unix=10,
        )


def test_relocated_verdict_rejects_relative_declared_campaign_provenance(
    tmp_path: Path,
) -> None:
    pointer_path, _root_pem, _identity = _fixture(tmp_path)
    pointer = json.loads(pointer_path.read_text())
    activation_document = pointer["activation"]
    plan_document = json.loads(
        Path(activation_document["campaign_plan"]["path"]).read_text()
    )
    verdict_path = Path(activation_document["independent_verdict"]["path"])
    verdict = json.loads(verdict_path.read_text())
    verdict["campaign_dir"] = "relative/campaign"

    with pytest.raises(
        activation.LiveAdapterActivationError,
        match="live_adapter_positive_certificate_invalid",
    ):
        activation.validate_positive_live_adapter_evidence(
            activation_document,
            campaign_plan=plan_document,
            independent_verdict=verdict,
            independent_verdict_path=verdict_path,
        )


def test_activation_materializer_publishes_only_after_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing_pointer_path, root_pem, identity = _fixture(tmp_path)
    existing_pointer = json.loads(existing_pointer_path.read_text())
    root_path = tmp_path / "root.pem"
    root_path.write_bytes(root_pem)
    root_path.chmod(0o600)
    output = tmp_path / "prepared"
    activation_document = existing_pointer["activation"]
    adapter_freeze = {
        "schema": "aura.latent_cortex.adapter_freeze.v1",
        "adapter_id": identity["adapter_id"],
        "content_root_sha256": "6" * 64,
        "artifacts": [
            {
                "path": "adapter.safetensors",
                "sha256": "7" * 64,
                "size_bytes": 7,
            }
        ],
        "identity_receipt": identity,
        "model_identity": {},
        "validator_identity": {},
        "certificate_sha256": "8" * 64,
    }
    monkeypatch.setattr(
        materializer,
        "verify_adapter_freeze",
        lambda _path: adapter_freeze,
    )

    preparation = materializer.prepare_activation(
        campaign_plan_path=Path(activation_document["campaign_plan"]["path"]),
        independent_verdict_path=Path(
            activation_document["independent_verdict"]["path"]
        ),
        campaign_policy_path=Path(existing_pointer["campaign_policy"]["path"]),
        trusted_root_path=root_path,
        adapter_package_path=Path(activation_document["adapter_package_path"]),
        approved_adapter_roots=[tmp_path / "releases"],
        output_dir=output,
        signed_at_unix=1_800_000_300,
        not_before_unix=1_800_000_200,
        expires_at_unix=1_800_010_000,
    )

    assert preparation["publication_allowed"] is False
    assert preparation["required_next_step"].startswith("detached_evidence")
    assert not (output / "active.json").exists()
    bundle = json.loads((output / "evidence" / "evidence-bundle.json").read_text())
    prepared_activation = json.loads((output / "activation.json").read_text())
    assert (
        bundle["review_contract"]["all_documentary_signing_inputs_embedded"] is True
    )
    assert bundle["review_contract"]["adapter_payload_bytes_embedded"] is False
    assert bundle["adapter_weights_embedded"] is False
    assert preparation["portable_evidence_bundle"]["path"] == str(
        (output / "evidence" / "evidence-bundle.json").resolve()
    )
    assert prepared_activation["campaign_plan"]["path"].startswith(
        str((output / "evidence").resolve())
    )
    assert prepared_activation["campaign_plan"]["path"] != activation_document[
        "campaign_plan"
    ]["path"]
    copied_plan = Path(bundle["documents"]["campaign_plan"]["path"])
    copied_plan_raw = copied_plan.read_bytes()
    copied_plan.write_bytes(b"{}")
    with pytest.raises(
        materializer.ActivationMaterializationError,
        match="activation_materialization_evidence_bundle_binding_mismatch",
    ):
        materializer._validate_portable_evidence_bundle(
            preparation,
            activation=prepared_activation,
        )
    copied_plan.write_bytes(copied_plan_raw)
    copied_plan.chmod(0o600)
    request = json.loads((output / "activation-signature-request.json").read_text())
    assert request["signed_payload"]["operation"] == "activate_recurrent_adapter"
    assert request["signed_payload"]["purpose"] == (
        "verified-recurrent-adapter-activation"
    )
    existing_attestation = json.loads(
        Path(existing_pointer["activation_attestation"]["path"]).read_text()
    )
    monkeypatch.setattr(
        materializer,
        "assemble_role_attestation",
        lambda *_args, **_kwargs: existing_attestation,
    )
    monkeypatch.setattr(
        activation,
        "verify_role_attestation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        materializer,
        "full_weight_checkpoint_identity",
        lambda _path: {
            "fingerprint": identity["base_checkpoint_fingerprint"],
            "method": "sha256",
            "files": 1,
        },
    )
    monkeypatch.setattr(
        materializer,
        "model_behavior_bundle_identity",
        lambda _path: {"bundle_sha256": identity["model_behavior_bundle_sha256"]},
    )
    monkeypatch.setattr(
        materializer,
        "personality_bundle_identity",
        lambda _path: {"present": False, "identity_sha256": "e" * 64},
    )
    monkeypatch.setattr(
        materializer,
        "runtime_environment_identity",
        lambda: {"identity_sha256": "f" * 64},
    )
    monkeypatch.setattr(activation, "declared_bindings", lambda _manifest: ())
    monkeypatch.setattr(activation, "inspect_mlx_tensor_metadata", lambda _path: ())
    monkeypatch.setattr(
        activation,
        "validate_resident_recurrent_sft_adapter_identity",
        lambda *_args, **_kwargs: dict(identity),
    )
    signature = tmp_path / "signature.bin"
    signature.write_bytes(b"s" * 64)
    pointer_out = output / "active.json"
    receipt_out = output / "publication.json"

    publication = materializer.finalize_activation(
        preparation_path=output / "preparation.json",
        signature_path=signature,
        campaign_policy_path=Path(existing_pointer["campaign_policy"]["path"]),
        trusted_root_path=root_path,
        model_path=tmp_path / "model",
        personality_adapter_path=None,
        approved_adapter_roots=[tmp_path / "releases"],
        pointer_path=pointer_out,
        publication_receipt_path=receipt_out,
        now_unix=1_800_000_400,
    )

    assert publication["published"] is True
    assert publication["static_weight_fusion_performed"] is False
    assert publication["admission_receipt"]["verified_verdict"] == "gain_proven"
    assert pointer_out.exists()
    assert receipt_out.exists()


def test_live_adapter_activation_rejects_signed_conjecture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointer, root_pem, identity = _fixture(
        tmp_path,
        verdict_tier="CONJECTURE",
    )

    with pytest.raises(
        activation.LiveAdapterActivationError,
        match="live_adapter_positive_certificate_invalid",
    ):
        _admit(monkeypatch, pointer, root_pem, identity, tmp_path)


def test_live_adapter_activation_rejects_untrusted_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointer, _root_pem, identity = _fixture(tmp_path)
    attacker = Ed25519PrivateKey.generate()

    with pytest.raises(ValueError, match="campaign_trust_root_key_mismatch"):
        _admit(
            monkeypatch,
            pointer,
            _public_pem(attacker),
            identity,
            tmp_path,
        )


def test_attach_certified_adapter_measures_runtime_and_uses_shared_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "release"
    package.mkdir()
    model = object()
    measured = {
        "base": {"fingerprint": "a" * 64},
        "behavior": {"bundle_sha256": "b" * 64},
        "personality": {"present": False},
        "runtime": {"identity_sha256": "c" * 64},
    }
    monkeypatch.setattr(
        activation,
        "full_weight_checkpoint_identity",
        lambda _path: measured["base"],
    )
    monkeypatch.setattr(
        activation,
        "model_behavior_bundle_identity",
        lambda _path: measured["behavior"],
    )
    monkeypatch.setattr(
        activation,
        "personality_bundle_identity",
        lambda _path: measured["personality"],
    )
    monkeypatch.setattr(
        activation,
        "runtime_environment_identity",
        lambda: measured["runtime"],
    )
    observed: dict[str, Any] = {}

    def _admit_stub(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        return {
            "adapter_package_path": str(package),
            "manifest": {"lora": {"wrapped_projections": 6}},
            "adapter_identity": {
                "lora": {"wrapped_projections": 6},
            },
            "receipt_sha256": "d" * 64,
        }

    monkeypatch.setattr(
        activation,
        "admit_live_adapter_activation",
        _admit_stub,
    )
    monkeypatch.setattr(
        activation,
        "load_resident_adapter",
        lambda loaded_model, loaded_path, _manifest: (
            6
            if loaded_model is model and loaded_path == str(package)
            else 0
        ),
    )

    receipt = activation.attach_certified_live_adapter(
        model,
        model_path=tmp_path / "model",
        personality_adapter_path=None,
        pointer_path=tmp_path / "active.json",
        trusted_root_public_key_pem=b"root",
        approved_adapter_roots=[tmp_path],
        now_unix=123,
    )

    assert observed["actual_base_checkpoint"] == measured["base"]
    assert observed["actual_model_behavior_bundle"] == measured["behavior"]
    assert observed["actual_personality_adapter"] == measured["personality"]
    assert observed["actual_runtime_environment"] == measured["runtime"]
    assert receipt["loaded_projection_count"] == 6
    assert "manifest" not in receipt


def test_attach_certified_adapter_rejects_projection_contract_before_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        activation,
        "full_weight_checkpoint_identity",
        lambda _path: {},
    )
    monkeypatch.setattr(
        activation,
        "model_behavior_bundle_identity",
        lambda _path: {},
    )
    monkeypatch.setattr(
        activation,
        "personality_bundle_identity",
        lambda _path: {},
    )
    monkeypatch.setattr(
        activation,
        "runtime_environment_identity",
        lambda: {},
    )
    monkeypatch.setattr(
        activation,
        "admit_live_adapter_activation",
        lambda *_args, **_kwargs: {
            "adapter_package_path": str(tmp_path),
            "manifest": {"lora": {"wrapped_projections": 5}},
            "adapter_identity": {"lora": {"wrapped_projections": 6}},
        },
    )
    loaded = False

    def _load(*_args: Any, **_kwargs: Any) -> int:
        nonlocal loaded
        loaded = True
        return 5

    monkeypatch.setattr(activation, "load_resident_adapter", _load)

    with pytest.raises(
        activation.LiveAdapterActivationError,
        match="live_adapter_projection_contract_mismatch",
    ):
        activation.attach_certified_live_adapter(
            object(),
            model_path=tmp_path,
            personality_adapter_path=None,
            pointer_path=tmp_path / "active.json",
            trusted_root_public_key_pem=b"root",
            approved_adapter_roots=[tmp_path],
            now_unix=123,
        )
    assert loaded is False
