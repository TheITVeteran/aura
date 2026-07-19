from __future__ import annotations

import base64
import hashlib
import stat
from argparse import Namespace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.campaign_launch_bundle import (
    ADAPTER_FREEZE_FILE,
    LAUNCH_PACKET_FILE,
    CampaignLaunchBundleError,
    build_adapter_freeze_certificate,
    read_canonical_json,
    verify_adapter_freeze,
)
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_RUNNER,
    CAMPAIGN_TRUST_POLICY_SCHEMA,
    CAMPAIGN_TRUST_ROLES,
    TASK_ISSUER,
    CampaignTrustError,
    build_role_attestation,
    validate_campaign_trust_policy,
)
from tools import prepare_latent_cortex_campaign as preparation


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _binding(path: str, payload: bytes) -> dict[str, object]:
    return {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _adapter_source(root: Path, *, adapter_id: str = "resident-test") -> Path:
    artifacts = {
        "adapters.safetensors": b"adapter-tensors",
        "adapter_final.safetensors": b"adapter-tensors",
        "adapter_config.json": b'{"loader":"test"}\n',
        "receipt.json": b'{"complete":true}\n',
        "training_config.json": b'{"max_steps":8}\n',
        "dataset_manifest.json": b'{"examples":[1]}\n',
        "execution_spec.json": b'{"recurrent_steps":4}\n',
        "source_snapshots/trainer.py": b"# frozen trainer\n",
    }
    for relative, payload in artifacts.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    manifest = {
        "schema": "aura.recurrence_adapter_manifest.v2",
        "adapter_id": adapter_id,
        "adapter": _binding("adapters.safetensors", artifacts["adapters.safetensors"]),
        "adapter_alias": _binding(
            "adapter_final.safetensors", artifacts["adapter_final.safetensors"]
        ),
        "loader_config": _binding(
            "adapter_config.json", artifacts["adapter_config.json"]
        ),
        "training_receipt": _binding("receipt.json", artifacts["receipt.json"]),
        "training_config": _binding(
            "training_config.json", artifacts["training_config.json"]
        ),
        "dataset_manifest": _binding(
            "dataset_manifest.json", artifacts["dataset_manifest.json"]
        ),
        "execution_spec": _binding(
            "execution_spec.json", artifacts["execution_spec.json"]
        ),
        "sources": {
            "trainer": {
                "origin_path": "tools/trainer.py",
                "snapshot_path": "source_snapshots/trainer.py",
                "sha256": hashlib.sha256(
                    artifacts["source_snapshots/trainer.py"]
                ).hexdigest(),
                "size_bytes": len(artifacts["source_snapshots/trainer.py"]),
            }
        },
    }
    manifest_bytes = canonical_json_bytes(manifest) + b"\n"
    (root / "recurrence_adapter_manifest.json").write_bytes(manifest_bytes)
    completion = {
        "schema": "aura.recurrence_native_training_completion.v1",
        "complete": True,
        "halt_reason": "max_steps",
        "step": 8,
        "adapter_sha256": manifest["adapter"]["sha256"],
        "receipt_sha256": manifest["training_receipt"]["sha256"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    _write_json(root / "training_completion.json", completion)
    return root


def _freeze(root: Path, *, adapter_id: str = "resident-test") -> Path:
    source = _adapter_source(root / "training", adapter_id=adapter_id)
    staging = root / ".frozen.staging"
    destination = root / "frozen"
    inventory = preparation.copy_adapter_snapshot(source, staging)
    identity = {
        "schema": "aura.recurrence_adapter_identity_receipt.v2",
        "adapter_id": adapter_id,
        "complete": True,
    }
    model = {
        "fingerprint": "1" * 64,
        "files": 2,
        "model_behavior_bundle_sha256": "2" * 64,
        "runtime_bundle_sha256": "3" * 64,
        "runtime_environment_identity_sha256": "4" * 64,
        "personality_adapter_bundle_sha256": "5" * 64,
        "effective_stack_sha256": "6" * 64,
    }
    certificate = build_adapter_freeze_certificate(
        adapter_id=adapter_id,
        inventory=inventory,
        identity_receipt=identity,
        model_identity=model,
        validator_identity={"validator_sha256": "7" * 64},
    )
    preparation.seal_adapter_snapshot(staging, destination, certificate)
    return destination


def _make_writable(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda value: len(value.parts)):
        if path.is_dir():
            path.chmod(0o700)
        elif not path.is_symlink():
            path.chmod(0o600)
    if root.exists():
        root.chmod(0o700)


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
    root: Ed25519PrivateKey,
    role_keys: dict[str, Ed25519PrivateKey],
    *,
    campaign_name: str,
    protocol_sha256: str,
    implementation_sha256: dict[str, str],
) -> dict:
    roles = {}
    for role, key in role_keys.items():
        raw = _public_raw(key)
        roles[role] = {
            "signer_id": f"{role}-signer",
            "organization_id": f"{role}-organization",
            "public_key_b64": base64.b64encode(raw).decode("ascii"),
            "key_id": hashlib.sha256(raw).hexdigest(),
            "implementation_sha256": implementation_sha256.get(
                role, hashlib.sha256(f"{role}:impl".encode()).hexdigest()
            ),
            "release_sha256": hashlib.sha256(f"{role}:release".encode()).hexdigest(),
            "custody_class": "external_service",
            "custody_evidence_sha256": hashlib.sha256(
                f"{role}:custody".encode()
            ).hexdigest(),
        }
    body = {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": "launch-bundle-test",
        "policy_revision": 1,
        "campaign_name": campaign_name,
        "protocol_sha256": protocol_sha256,
        "previous_policy_sha256": None,
        "revoked_key_ids": [],
        "issued_at_unix": 1_900_000_000,
        "not_before_unix": 1_900_000_100,
        "expires_at_unix": 1_900_086_400,
        "roles": roles,
    }
    signed = canonical_json_bytes(body)
    root_raw = _public_raw(root)
    return {
        **body,
        "root_signature": {
            "algorithm": "Ed25519",
            "key_id": hashlib.sha256(root_raw).hexdigest(),
            "signature_b64": base64.b64encode(root.sign(signed)).decode("ascii"),
            "signed_payload_sha256": hashlib.sha256(signed).hexdigest(),
        },
    }


def _launch_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    freeze = _freeze(tmp_path)
    campaign_name = "resident-32b-launch-test"
    protocol = "9" * 64
    fake_runner = tmp_path / "fake_campaign_runner.py"
    fake_runner.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "print(Path(os.environ['AURA_TEST_TRUST_REQUESTS']).read_text())\n",
        encoding="ascii",
    )
    monkeypatch.setattr(preparation, "RUNNER_PATH", fake_runner)
    root_key = Ed25519PrivateKey.generate()
    role_keys = {role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES}
    policy = _policy(
        root_key,
        role_keys,
        campaign_name=campaign_name,
        protocol_sha256=protocol,
        implementation_sha256={
            TASK_ISSUER: preparation._source_sha256(preparation.TASK_ISSUER_PATH),
            CAMPAIGN_RUNNER: preparation._source_sha256(fake_runner),
        },
    )
    root_path = tmp_path / "campaign-root.pem"
    policy_path = tmp_path / "campaign-policy.json"
    root_path.write_bytes(_public_pem(root_key))
    _write_json(policy_path, policy)
    verified = validate_campaign_trust_policy(
        policy,
        trusted_root_public_key_pem=root_path.read_bytes(),
        expected_campaign_name=campaign_name,
        expected_protocol_sha256=protocol,
        now_unix=1_900_000_200,
    )
    payloads = {
        TASK_ISSUER: {
            "schema": "aura.latent_cortex.task_issuer_prelaunch.v1",
            "campaign_name": campaign_name,
            "task_commitment_sha256": "a" * 64,
        },
        CAMPAIGN_RUNNER: {
            "schema": "aura.latent_cortex.runner_prelaunch.v1",
            "campaign_name": campaign_name,
            "adapter_identity_sha256": "b" * 64,
        },
    }
    trust = {
        "schema": "aura.latent_cortex.campaign_trust_requests.v1",
        "campaign_name": campaign_name,
        "policy_sha256": verified.policy_sha256,
        "protocol_sha256": protocol,
        "unsigned_plan_sha256": "c" * 64,
        "externally_custodied": True,
        "requests": payloads,
    }
    trust_path = tmp_path / "fake-trust-requests.json"
    _write_json(trust_path, trust)
    monkeypatch.setenv("AURA_TEST_TRUST_REQUESTS", str(trust_path))
    contamination_audit = tmp_path / "contamination.json"
    contamination_root = tmp_path / "contamination-root.pem"
    _write_json(contamination_audit, {"schema": "test.contamination.v1"})
    contamination_root.write_bytes(b"independent contamination root\n")
    model = tmp_path / "model"
    model.mkdir()
    campaign_dir = tmp_path / "campaign"
    runner_argv = [
        "--campaign-dir",
        str(campaign_dir),
        "--campaign-name",
        campaign_name,
        "--model",
        str(model),
        "--adapter",
        str(freeze),
        "--adapter-id",
        "resident-test",
        "--personality-adapter",
        "trained",
        "--seeds",
        "1000003,1000033",
        "--profile",
        "full",
        "--confirmatory",
        "--contamination-audit",
        str(contamination_audit),
        "--contamination-trust-root",
        str(contamination_root),
        "--campaign-trust-policy",
        str(policy_path),
        "--campaign-trust-root",
        str(root_path),
    ]
    bundle = tmp_path / "prelaunch"
    result = preparation.prepare_bundle(
        Namespace(
            adapter_freeze=freeze,
            runner_args=runner_argv,
            prepare_timeout=20.0,
            observed_at=1_900_000_200,
            signed_at=1_900_000_150,
            bundle_dir=bundle,
        )
    )
    return {
        "freeze": freeze,
        "bundle": bundle,
        "result": result,
        "verified_policy": verified,
        "role_keys": role_keys,
        "payloads": payloads,
        "dependencies": {
            "contamination_audit": contamination_audit,
            "fake_runner": fake_runner,
        },
    }


def test_adapter_snapshot_is_exact_read_only_and_tamper_evident(tmp_path: Path):
    frozen = _freeze(tmp_path)
    try:
        certificate = verify_adapter_freeze(frozen)
        assert certificate["content_root_sha256"]
        assert stat.S_IMODE((frozen / ADAPTER_FREEZE_FILE).stat().st_mode) == 0o400

        adapter = frozen / "adapters.safetensors"
        adapter.chmod(0o600)
        adapter.write_bytes(b"changed")
        adapter.chmod(0o400)
        with pytest.raises(
            CampaignLaunchBundleError, match="adapter_adapter_binding_mismatch"
        ):
            verify_adapter_freeze(frozen)
    finally:
        _make_writable(frozen)


def test_adapter_snapshot_rejects_manifest_symlink(tmp_path: Path):
    source = _adapter_source(tmp_path / "training")
    manifest = source / "recurrence_adapter_manifest.json"
    payload = manifest.read_bytes()
    manifest.unlink()
    target = tmp_path / "outside-manifest.json"
    target.write_bytes(payload)
    manifest.symlink_to(target)

    with pytest.raises(CampaignLaunchBundleError, match="symlink"):
        preparation.copy_adapter_snapshot(source, tmp_path / "staging")


def test_real_detached_prelaunch_signatures_seal_exact_launch_packet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _launch_fixture(tmp_path, monkeypatch)
    bundle = fixture["bundle"]
    try:
        issuer = build_role_attestation(
            fixture["verified_policy"],
            role=TASK_ISSUER,
            payload=fixture["payloads"][TASK_ISSUER],
            signed_at_unix=1_900_000_150,
            private_key=fixture["role_keys"][TASK_ISSUER],
        )
        runner = build_role_attestation(
            fixture["verified_policy"],
            role=CAMPAIGN_RUNNER,
            payload=fixture["payloads"][CAMPAIGN_RUNNER],
            signed_at_unix=1_900_000_150,
            private_key=fixture["role_keys"][CAMPAIGN_RUNNER],
        )
        issuer_path = tmp_path / "external-issuer-attestation.json"
        runner_path = tmp_path / "external-runner-attestation.json"
        _write_json(issuer_path, issuer)
        _write_json(runner_path, runner)
        interrupted_issuer = bundle / preparation._ISSUER_ATTESTATION_FILE
        preparation.write_canonical_exclusive(interrupted_issuer, issuer)
        interrupted_issuer.chmod(0o400)

        admitted = preparation.admit_bundle(
            Namespace(
                bundle_dir=bundle,
                task_issuer_attestation=issuer_path,
                runner_attestation=runner_path,
                prepare_timeout=20.0,
                observed_at=1_900_000_200,
            )
        )

        assert admitted["phase"] == "ready_for_inference"
        packet = read_canonical_json(bundle / LAUNCH_PACKET_FILE, role="test_packet")
        assert packet["argv"][-4:] == [
            "--task-issuer-attestation",
            str(bundle / "task_issuer_attestation.json"),
            "--runner-attestation",
            str(bundle / "campaign_runner_attestation.json"),
        ]
        assert preparation.inspect_bundle(Namespace(bundle_dir=bundle))["phase"] == (
            "ready_for_inference"
        )
        repeated = preparation.admit_bundle(
            Namespace(
                bundle_dir=bundle,
                task_issuer_attestation=issuer_path,
                runner_attestation=runner_path,
                prepare_timeout=20.0,
                observed_at=1_900_000_200,
            )
        )
        assert repeated["packet_sha256"] == admitted["packet_sha256"]
    finally:
        _make_writable(fixture["freeze"])


@pytest.mark.parametrize(
    "attack", ["dependency", "wrong_signature", "producer", "unplanned_artifact"]
)
def test_launch_admission_rejects_changed_dependencies_keys_or_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
):
    fixture = _launch_fixture(tmp_path, monkeypatch)
    try:
        issuer = build_role_attestation(
            fixture["verified_policy"],
            role=TASK_ISSUER,
            payload=fixture["payloads"][TASK_ISSUER],
            signed_at_unix=1_900_000_150,
            private_key=fixture["role_keys"][TASK_ISSUER],
        )
        runner = build_role_attestation(
            fixture["verified_policy"],
            role=CAMPAIGN_RUNNER,
            payload=fixture["payloads"][CAMPAIGN_RUNNER],
            signed_at_unix=1_900_000_150,
            private_key=fixture["role_keys"][CAMPAIGN_RUNNER],
        )
        if attack == "wrong_signature":
            runner["signature_b64"] = base64.b64encode(b"x" * 64).decode("ascii")
        issuer_path = tmp_path / "issuer.json"
        runner_path = tmp_path / "runner.json"
        _write_json(issuer_path, issuer)
        _write_json(runner_path, runner)
        if attack == "dependency":
            fixture["dependencies"]["contamination_audit"].write_text(
                "changed\n", encoding="ascii"
            )
        elif attack == "producer":
            fixture["dependencies"]["fake_runner"].write_text(
                "raise SystemExit(9)\n", encoding="ascii"
            )
        elif attack == "unplanned_artifact":
            (fixture["bundle"] / "not-in-the-frozen-plan.json").write_text(
                "{}\n", encoding="ascii"
            )

        with pytest.raises(
            (preparation.CampaignPreparationError, CampaignTrustError)
        ):
            preparation.admit_bundle(
                Namespace(
                    bundle_dir=fixture["bundle"],
                    task_issuer_attestation=issuer_path,
                    runner_attestation=runner_path,
                    prepare_timeout=20.0,
                    observed_at=1_900_000_200,
                )
            )
    finally:
        _make_writable(fixture["freeze"])
