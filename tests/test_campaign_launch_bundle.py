from __future__ import annotations

import base64
import hashlib
import stat
from argparse import Namespace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex.campaign_journal import (
    CampaignPlan,
    canonical_json_bytes,
)
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
    prepare_role_signature_request,
    validate_campaign_trust_policy,
)
from tools import advance_latent_cortex_campaign as advancement
from tools import prepare_latent_cortex_campaign as preparation


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def test_read_canonical_json_honors_schema_specific_newline_policy(
    tmp_path: Path,
) -> None:
    sft = tmp_path / "sft.json"
    sft.write_bytes(
        canonical_json_bytes({"schema": "aura.resident_recurrent_sft_adapter_manifest.v1"})
    )
    assert read_canonical_json(sft, role="sft", trailing_newline=None)["schema"].endswith(".v1")

    legacy = tmp_path / "legacy.json"
    legacy.write_bytes(canonical_json_bytes({"schema": "legacy"}))
    with pytest.raises(CampaignLaunchBundleError, match="legacy_noncanonical"):
        read_canonical_json(legacy, role="legacy", trailing_newline=None)


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
        "loader_config": _binding("adapter_config.json", artifacts["adapter_config.json"]),
        "training_receipt": _binding("receipt.json", artifacts["receipt.json"]),
        "training_config": _binding("training_config.json", artifacts["training_config.json"]),
        "dataset_manifest": _binding("dataset_manifest.json", artifacts["dataset_manifest.json"]),
        "execution_spec": _binding("execution_spec.json", artifacts["execution_spec.json"]),
        "sources": {
            "trainer": {
                "origin_path": "tools/trainer.py",
                "snapshot_path": "source_snapshots/trainer.py",
                "sha256": hashlib.sha256(artifacts["source_snapshots/trainer.py"]).hexdigest(),
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
        "personality_adapter_bundle_sha256": "",
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
            "custody_evidence_sha256": hashlib.sha256(f"{role}:custody".encode()).hexdigest(),
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
        "campaign_dir": campaign_dir,
        "campaign_name": campaign_name,
        "dependencies": {
            "contamination_audit": contamination_audit,
            "fake_runner": fake_runner,
        },
    }


def _admit_prelaunch(fixture: dict, tmp_path: Path) -> None:
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
    issuer_path = tmp_path / "phase-prelaunch-issuer.json"
    runner_path = tmp_path / "phase-prelaunch-runner.json"
    _write_json(issuer_path, issuer)
    _write_json(runner_path, runner)
    preparation.admit_bundle(
        Namespace(
            bundle_dir=fixture["bundle"],
            task_issuer_attestation=issuer_path,
            runner_attestation=runner_path,
            prepare_timeout=20.0,
            observed_at=1_900_000_200,
        )
    )


def _hashed(document: dict, key: str) -> dict:
    material = {name: value for name, value in document.items() if name != key}
    return {**material, key: hashlib.sha256(canonical_json_bytes(material)).hexdigest()}


def _post_inference_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    fixture = _launch_fixture(tmp_path, monkeypatch)
    _admit_prelaunch(fixture, tmp_path)
    campaign_dir = fixture["campaign_dir"]
    campaign_dir.mkdir()
    answer_payload = {"expected": {"value": 42}}
    answer_commitment = hashlib.sha256(canonical_json_bytes(answer_payload)).hexdigest()
    task_commitment = "d" * 64
    plan = CampaignPlan.build(
        fixture["campaign_name"],
        [{"arm": "adapter_rlc", "task_id": "task-1"}],
        metadata={
            "task_manifest": {
                "tasks": [
                    {
                        "task_id": "task-1",
                        "answer_commitment_sha256": answer_commitment,
                    }
                ]
            },
            "task_commitment": {"commitment_sha256": task_commitment},
        },
    )
    _write_json(campaign_dir / "plan.json", plan.to_dict())
    sealed = _hashed(
        {
            "schema": "aura.latent_cortex.sealed_output_manifest.v4",
            "cell_count": 1,
        },
        "manifest_sha256",
    )
    _write_json(campaign_dir / "sealed_output_manifest.json", sealed)
    reveal_payload = {
        "schema": "aura.latent_cortex.answer_reveal_payload.v1",
        "campaign_name": fixture["campaign_name"],
        "plan_sha256": plan.plan_sha256,
        "sealed_output_manifest_sha256": sealed["manifest_sha256"],
        "task_commitment_sha256": task_commitment,
        "answers": [
            {
                "task_id": "task-1",
                "answer_commitment_sha256": answer_commitment,
                "answer_payload": answer_payload,
            }
        ],
    }
    answer_request = prepare_role_signature_request(
        fixture["verified_policy"],
        role=TASK_ISSUER,
        payload=reveal_payload,
        signed_at_unix=1_900_000_250,
    )
    _write_json(campaign_dir / "answer_reveal_request.json", answer_request)
    fixture.update(
        plan=plan,
        sealed=sealed,
        reveal_payload=reveal_payload,
        answer_request=answer_request,
    )
    return fixture


def test_adapter_snapshot_is_exact_read_only_and_tamper_evident(tmp_path: Path):
    frozen = _freeze(tmp_path)
    try:
        certificate = verify_adapter_freeze(frozen)
        assert certificate["content_root_sha256"]
        assert certificate["model_identity"]["personality_adapter_bundle_sha256"] == ""
        assert stat.S_IMODE((frozen / ADAPTER_FREEZE_FILE).stat().st_mode) == 0o400

        adapter = frozen / "adapters.safetensors"
        adapter.chmod(0o600)
        adapter.write_bytes(b"changed")
        adapter.chmod(0o400)
        with pytest.raises(CampaignLaunchBundleError, match="adapter_adapter_binding_mismatch"):
            verify_adapter_freeze(frozen)
    finally:
        _make_writable(frozen)


def test_adapter_snapshot_rejects_rehashed_malformed_model_identity(tmp_path: Path):
    frozen = _freeze(tmp_path)
    certificate_path = frozen / ADAPTER_FREEZE_FILE
    try:
        certificate = verify_adapter_freeze(frozen)
        certificate["model_identity"]["runtime_bundle_sha256"] = "not-a-sha"
        material = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
        certificate["certificate_sha256"] = hashlib.sha256(
            canonical_json_bytes(material)
        ).hexdigest()
        certificate_path.chmod(0o600)
        _write_json(certificate_path, certificate)
        certificate_path.chmod(0o400)

        with pytest.raises(
            CampaignLaunchBundleError,
            match="adapter_freeze_model_identity_invalid",
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
            fixture["dependencies"]["contamination_audit"].write_text("changed\n", encoding="ascii")
        elif attack == "producer":
            fixture["dependencies"]["fake_runner"].write_text(
                "raise SystemExit(9)\n", encoding="ascii"
            )
        elif attack == "unplanned_artifact":
            (fixture["bundle"] / "not-in-the-frozen-plan.json").write_text("{}\n", encoding="ascii")

        with pytest.raises((preparation.CampaignPreparationError, CampaignTrustError)):
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


def _admit_answer_phase(fixture: dict, tmp_path: Path) -> tuple[dict, dict]:
    attestation = build_role_attestation(
        fixture["verified_policy"],
        role=TASK_ISSUER,
        payload=fixture["reveal_payload"],
        signed_at_unix=1_900_000_250,
        private_key=fixture["role_keys"][TASK_ISSUER],
    )
    path = tmp_path / "external-answer-reveal-attestation.json"
    _write_json(path, attestation)
    result = advancement.admit(
        Namespace(
            bundle_dir=fixture["bundle"],
            attestation=path,
            observed_at=1_900_000_300,
        )
    )
    return attestation, result


def _publish_final_request(fixture: dict, issuer_attestation: dict) -> dict:
    campaign_dir = fixture["campaign_dir"]
    reveal_material = {
        "payload": fixture["reveal_payload"],
        "request_sha256": fixture["answer_request"]["request_sha256"],
        "task_issuer_attestation": issuer_attestation,
    }
    reveal = {
        "schema": "aura.latent_cortex.answer_reveal.v1",
        **reveal_material,
        "reveal_sha256": hashlib.sha256(canonical_json_bytes(reveal_material)).hexdigest(),
    }
    _write_json(campaign_dir / "answer_reveal.json", reveal)
    campaign_manifest = _hashed(
        {
            "schema": "aura.latent_cortex.campaign_manifest.v1",
            "journal_head_sha256": "1" * 64,
        },
        "manifest_sha256",
    )
    grade = _hashed(
        {"schema": "aura.latent_cortex.paired_campaign_grade.v2", "verdict": "gain_proven"},
        "grade_sha256",
    )
    worker = _hashed(
        {
            "schema": "aura.latent_cortex.worker_execution_manifest.v1",
            "detached_plan_sha256": "2" * 64,
            "detached_classification_head_sha256": "3" * 64,
            "detached_classifications_sha256": "4" * 64,
            "imports_sha256": "5" * 64,
            "excluded_attempts_sha256": "6" * 64,
        },
        "manifest_sha256",
    )
    _write_json(campaign_dir / "campaign_manifest.json", campaign_manifest)
    _write_json(campaign_dir / "grade.json", grade)
    _write_json(campaign_dir / "worker_execution_manifest.json", worker)
    payload = {
        "schema": "aura.latent_cortex.final_run_payload.v4",
        "campaign_name": fixture["campaign_name"],
        "policy_sha256": fixture["verified_policy"].policy_sha256,
        "protocol_sha256": fixture["verified_policy"].document["protocol_sha256"],
        "plan_sha256": fixture["plan"].plan_sha256,
        "sealed_output_manifest_sha256": fixture["sealed"]["manifest_sha256"],
        "answer_reveal_sha256": reveal["reveal_sha256"],
        "campaign_manifest_sha256": campaign_manifest["manifest_sha256"],
        "journal_head_sha256": campaign_manifest["journal_head_sha256"],
        "published_grade_sha256": grade["grade_sha256"],
        "worker_execution_manifest_sha256": worker["manifest_sha256"],
        "detached_plan_sha256": worker["detached_plan_sha256"],
        "detached_classification_head_sha256": worker["detached_classification_head_sha256"],
        "detached_classifications_sha256": worker["detached_classifications_sha256"],
        "worker_imports_sha256": worker["imports_sha256"],
        "worker_excluded_attempts_sha256": worker["excluded_attempts_sha256"],
    }
    request = prepare_role_signature_request(
        fixture["verified_policy"],
        role=CAMPAIGN_RUNNER,
        payload=payload,
        signed_at_unix=1_900_000_350,
    )
    _write_json(campaign_dir / "final_run_request.json", request)
    fixture.update(
        reveal=reveal,
        campaign_manifest=campaign_manifest,
        grade=grade,
        worker=worker,
        final_payload=payload,
        final_request=request,
    )
    return request


def test_post_inference_phase_packets_bind_real_reveal_and_final_signatures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _post_inference_fixture(tmp_path, monkeypatch)
    try:
        status = advancement.status(Namespace(bundle_dir=fixture["bundle"]))
        assert status["phase"] == "awaiting_answer_reveal_signature"
        issuer_attestation, answer_result = _admit_answer_phase(fixture, tmp_path)
        assert answer_result["phase"] == "ready_for_answer_reveal"
        assert answer_result["argv"][-2:] == [
            "--answer-reveal-attestation",
            str(fixture["bundle"] / advancement.ANSWER_ATTESTATION_FILE),
        ]
        repeated = _admit_answer_phase(fixture, tmp_path)[1]
        assert repeated["packet_sha256"] == answer_result["packet_sha256"]

        _publish_final_request(fixture, issuer_attestation)
        status = advancement.status(Namespace(bundle_dir=fixture["bundle"]))
        assert status["phase"] == "awaiting_final_run_signature"
        final_attestation = build_role_attestation(
            fixture["verified_policy"],
            role=CAMPAIGN_RUNNER,
            payload=fixture["final_payload"],
            signed_at_unix=1_900_000_350,
            private_key=fixture["role_keys"][CAMPAIGN_RUNNER],
        )
        final_path = tmp_path / "external-final-run-attestation.json"
        _write_json(final_path, final_attestation)
        final_result = advancement.admit(
            Namespace(
                bundle_dir=fixture["bundle"],
                attestation=final_path,
                observed_at=1_900_000_400,
            )
        )
        assert final_result["phase"] == "ready_for_final_envelope"
        assert final_result["argv"][-4:] == [
            "--answer-reveal-attestation",
            str(fixture["bundle"] / advancement.ANSWER_ATTESTATION_FILE),
            "--final-run-attestation",
            str(fixture["bundle"] / advancement.FINAL_ATTESTATION_FILE),
        ]

        envelope_material = {
            "payload": fixture["final_payload"],
            "request_sha256": fixture["final_request"]["request_sha256"],
            "campaign_runner_attestation": final_attestation,
        }
        envelope = {
            "schema": "aura.latent_cortex.final_run_envelope.v4",
            **envelope_material,
            "envelope_sha256": hashlib.sha256(canonical_json_bytes(envelope_material)).hexdigest(),
        }
        _write_json(fixture["campaign_dir"] / "final_run_envelope.json", envelope)
        assert (
            advancement.status(Namespace(bundle_dir=fixture["bundle"]))["phase"]
            == "campaign_evidence_sealed"
        )
    finally:
        _make_writable(fixture["freeze"])


def test_answer_phase_rejects_recommitted_wrong_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _post_inference_fixture(tmp_path, monkeypatch)
    try:
        attacked = dict(fixture["reveal_payload"])
        attacked["answers"] = [dict(fixture["reveal_payload"]["answers"][0])]
        attacked["answers"][0]["answer_payload"] = {"expected": {"value": 99}}
        request = prepare_role_signature_request(
            fixture["verified_policy"],
            role=TASK_ISSUER,
            payload=attacked,
            signed_at_unix=1_900_000_250,
        )
        _write_json(fixture["campaign_dir"] / "answer_reveal_request.json", request)

        with pytest.raises(
            advancement.CampaignAdvanceError,
            match="answer_reveal_commitment_invalid",
        ):
            advancement.status(Namespace(bundle_dir=fixture["bundle"]))
    finally:
        _make_writable(fixture["freeze"])


def test_final_phase_rejects_grade_changed_after_signature_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _post_inference_fixture(tmp_path, monkeypatch)
    try:
        issuer_attestation, _result = _admit_answer_phase(fixture, tmp_path)
        _publish_final_request(fixture, issuer_attestation)
        changed_grade = _hashed(
            {
                "schema": "aura.latent_cortex.paired_campaign_grade.v2",
                "verdict": "no_gain",
            },
            "grade_sha256",
        )
        _write_json(fixture["campaign_dir"] / "grade.json", changed_grade)

        with pytest.raises(
            advancement.CampaignAdvanceError,
            match="final_run_payload_binding_invalid",
        ):
            advancement.status(Namespace(bundle_dir=fixture["bundle"]))
    finally:
        _make_writable(fixture["freeze"])


def test_phase_status_rejects_attestation_changed_after_resume_packet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _post_inference_fixture(tmp_path, monkeypatch)
    try:
        _issuer, _result = _admit_answer_phase(fixture, tmp_path)
        persisted = fixture["bundle"] / advancement.ANSWER_ATTESTATION_FILE
        attacked = read_canonical_json(persisted, role="test_answer_attestation")
        attacked["signature_b64"] = base64.b64encode(b"z" * 64).decode("ascii")
        persisted.chmod(0o600)
        persisted.write_bytes(canonical_json_bytes(attacked) + b"\n")
        persisted.chmod(0o400)

        with pytest.raises(
            advancement.CampaignAdvanceError,
            match="ready_for_answer_reveal_attestation_0_changed",
        ):
            advancement.status(Namespace(bundle_dir=fixture["bundle"]))
    finally:
        _make_writable(fixture["freeze"])
