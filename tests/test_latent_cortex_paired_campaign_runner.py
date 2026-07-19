from __future__ import annotations

import base64
import hashlib
import json
import os
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_RUNNER,
    CAMPAIGN_TRUST_POLICY_SCHEMA,
    CAMPAIGN_TRUST_ROLES,
    TASK_ISSUER,
    build_role_attestation,
    validate_campaign_trust_policy,
)
from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.brain.llm.latent_cortex.frontier_tasks import generate_task_battery
from core.brain.llm.latent_cortex.paired_campaign import build_campaign_plan
from tools import run_latent_cortex_paired_campaign as runner


def _external_policy_fixture(campaign_name: str, now: int):
    root = Ed25519PrivateKey.generate()
    role_keys = {
        role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES
    }
    roles = {}
    for role, key in role_keys.items():
        raw = key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        roles[role] = {
            "signer_id": f"{role}-signer",
            "organization_id": f"{role}-organization",
            "public_key_b64": base64.b64encode(raw).decode("ascii"),
            "key_id": hashlib.sha256(raw).hexdigest(),
            "implementation_sha256": hashlib.sha256(f"{role}:impl".encode()).hexdigest(),
            "release_sha256": hashlib.sha256(f"{role}:release".encode()).hexdigest(),
            "custody_class": "remote_hsm",
            "custody_evidence_sha256": hashlib.sha256(
                f"{role}:custody".encode()
            ).hexdigest(),
        }
    body = {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": f"{campaign_name}-policy",
        "policy_revision": 1,
        "campaign_name": campaign_name,
        "protocol_sha256": runner._campaign_protocol_sha256(),
        "previous_policy_sha256": None,
        "revoked_key_ids": [],
        "issued_at_unix": now - 10,
        "not_before_unix": now - 5,
        "expires_at_unix": now + 3600,
        "roles": roles,
    }
    root_raw = root.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    signed = canonical_json_bytes(body)
    document = {
        **body,
        "root_signature": {
            "algorithm": "Ed25519",
            "key_id": hashlib.sha256(root_raw).hexdigest(),
            "signature_b64": base64.b64encode(root.sign(signed)).decode("ascii"),
            "signed_payload_sha256": hashlib.sha256(signed).hexdigest(),
        },
    }
    root_pem = root.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return (
        validate_campaign_trust_policy(
            document,
            trusted_root_public_key_pem=root_pem,
            now_unix=now,
        ),
        role_keys,
    )


def test_majority_output_uses_parsed_answer_without_gold_access():
    first = 'reasoning\nFINAL_ANSWER: {"count":2,"witness":[1,5]}'
    second = 'different\nFINAL_ANSWER: {"witness":[1,5],"count":2}'
    wrong = 'FINAL_ANSWER: {"count":9,"witness":[]}'

    assert runner._majority_output([first, wrong, second]) == first


def test_worker_reconstructs_public_tasks_without_answer_payloads():
    tasks = generate_task_battery([7], domains=("mathematics",), difficulty=1)
    plan = build_campaign_plan(
        "public-worker-test",
        tasks,
        model_identity={"model": "sealed"},
        adapter_identity={"adapter": "sealed"},
        execution_config={"difficulty": 1},
    )

    public_tasks = runner._public_tasks_from_plan(plan)
    rebuilt = build_campaign_plan(
        "public-worker-test",
        public_tasks,
        model_identity={"model": "sealed"},
        adapter_identity={"adapter": "sealed"},
        execution_config={"difficulty": 1},
    )

    assert rebuilt.to_dict() == plan.to_dict()
    assert all(not hasattr(task, "blinded_answer") for task in public_tasks)
    assert all(not hasattr(task, "score") for task in public_tasks)


def test_expected_worker_plan_never_invokes_answer_bearing_generator(
    monkeypatch: pytest.MonkeyPatch,
):
    tasks = generate_task_battery([7], domains=("mathematics",), difficulty=1)
    model_identity = {"model": "sealed"}
    adapter_identity = {"adapter": "sealed"}
    execution_config = {"difficulty": 1}
    plan = build_campaign_plan(
        "generator-isolation-test",
        tasks,
        model_identity=model_identity,
        adapter_identity=adapter_identity,
        execution_config=execution_config,
    )
    args = SimpleNamespace(campaign_name="generator-isolation-test")
    monkeypatch.setattr(
        runner,
        "generate_task_battery",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("answer-bearing generator entered worker path")
        ),
    )
    monkeypatch.setattr(
        runner, "_identity_material", lambda _args: (model_identity, adapter_identity)
    )
    monkeypatch.setattr(runner, "_contamination_audit", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "_execution_config", lambda *_args: execution_config)
    monkeypatch.setattr(runner, "_verified_campaign_trust", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_claim_eligible", lambda *_args: False)
    monkeypatch.setattr(runner, "_arms", lambda _args: runner.FULL_ARMS)

    rebuilt, public_tasks = runner._expected_worker_plan(args, plan)

    assert rebuilt.to_dict() == plan.to_dict()
    assert all(not hasattr(task, "blinded_answer") for task in public_tasks)


def test_outputs_are_sealed_before_answer_reveal_and_scoring(tmp_path: Path):
    tasks = generate_task_battery([7], domains=("mathematics",), difficulty=1)
    plan = build_campaign_plan(
        "two-phase-preflight",
        tasks,
        model_identity={"model": "sealed"},
        adapter_identity={"adapter": "sealed"},
        execution_config={
            "difficulty": 1,
            "answer_reveal_protocol": "sealed_outputs_then_issuer_reveal_v1",
        },
    )
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    runner._persist_plan(campaign_dir, plan)
    task_by_id = {task.task_id: task for task in tasks}
    with runner.CampaignJournal(campaign_dir / runner.JOURNAL_FILE, plan) as journal:
        for cell_id in plan.cell_ids:
            definition = plan.cell_definition(cell_id)
            task = task_by_id[definition["task_id"]]
            text = "FINAL_ANSWER: " + json.dumps(
                task.reveal_for_verifier()["expected"],
                sort_keys=True,
                separators=(",", ":"),
            )
            attempt_id = journal.start_cell(cell_id)
            journal.record_arm_result(
                cell_id,
                attempt_id,
                {
                    "arm": definition["arm"],
                    "text": text,
                    "layer_apps": 1,
                },
            )
        assert journal.resume().committed_cell_ids == ()
        assert journal.resume().sealed_cell_ids == plan.cell_ids

    assert all(
        runner._arm_outputs_sealed(campaign_dir, plan, arm)
        for arm in runner.FULL_ARMS
    )
    sealed = runner._seal_output_manifest(campaign_dir, plan)
    reveal = runner._admit_answer_reveal(
        SimpleNamespace(
            campaign_dir=str(campaign_dir),
            answer_reveal_attestation="",
        ),
        plan,
        tasks,
        sealed,
    )
    assert reveal is not None
    assert (campaign_dir / runner.SEALED_OUTPUT_MANIFEST_FILE).exists()
    assert (campaign_dir / runner.ANSWER_REVEAL_FILE).exists()
    assert not (campaign_dir / runner.ANSWER_REVEAL_REQUEST_FILE).exists()
    with runner.CampaignJournal(campaign_dir / runner.JOURNAL_FILE, plan) as journal:
        assert journal.resume().committed_cell_ids == ()

    runner._score_sealed_outputs(campaign_dir, plan, tasks)
    with runner.CampaignJournal(campaign_dir / runner.JOURNAL_FILE, plan) as journal:
        assert journal.resume().committed_cell_ids == plan.cell_ids
        assert all(
            record["verification"]["correct"] is True
            for record in journal.committed_records()
        )


def test_claim_reveal_pauses_for_exact_external_issuer_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    now = int(time.time())
    campaign_name = "signed-reveal-test"
    policy, role_keys = _external_policy_fixture(campaign_name, now)
    tasks = generate_task_battery([9], domains=("mathematics",), difficulty=1)
    manifest = runner.build_task_manifest(tasks)
    auditor = Ed25519PrivateKey.generate()
    auditor_der = auditor.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    audit_body = {
        "schema": runner.CONTAMINATION_AUDIT_SCHEMA,
        "task_manifest_sha256": manifest.manifest_sha256,
        "status": "passed_zero_overlap",
        "overlap_count": 0,
        "auditor_independence": "external",
        "corpora": [{"name": "training", "snapshot_sha256": "d" * 64}],
        "methods": ["exact_prompt", "normalized_prompt", "token_fivegram"],
    }
    audit_bytes = canonical_json_bytes(audit_body)
    audit_sha = hashlib.sha256(auditor_der).hexdigest()
    audit = {
        **audit_body,
        "signature": {
            "algorithm": "ed25519",
            "key_id": audit_sha,
            "signature_b64": base64.b64encode(auditor.sign(audit_bytes)).decode(
                "ascii"
            ),
            "signed_payload_sha256": hashlib.sha256(audit_bytes).hexdigest(),
            "public_key_der_b64": base64.b64encode(auditor_der).decode("ascii"),
            "trust_root_sha256": audit_sha,
            "verified": True,
        },
    }
    model_identity = {"model": "sealed"}
    adapter_identity = {"adapter": "sealed"}
    execution_config = {
        "difficulty": 1,
        "worker_task_material": "public_manifest_only",
        "answer_reveal_protocol": "sealed_outputs_then_issuer_reveal_v1",
        "generation_seed_count": 1,
        "generation_seed_min_entropy_bits": 60,
        "generation_seed_policy": "external_issuer_uniform_63bit",
        "generation_seed_disclosure": "post_seal_answer_reveal",
    }
    unsigned = build_campaign_plan(
        campaign_name,
        tasks,
        model_identity=model_identity,
        adapter_identity=adapter_identity,
        execution_config=execution_config,
        contamination_audit=audit,
        claim_eligible=False,
    )
    trust = {
        "prelaunch_verified": True,
        "externally_custodied": True,
        "policy_sha256": policy.policy_sha256,
        "unsigned_plan_sha256": unsigned.plan_sha256,
    }
    plan = build_campaign_plan(
        campaign_name,
        tasks,
        model_identity=model_identity,
        adapter_identity=adapter_identity,
        execution_config=execution_config,
        contamination_audit=audit,
        campaign_trust=trust,
        claim_eligible=True,
    )
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    runner._persist_plan(campaign_dir, plan)
    with runner.CampaignJournal(campaign_dir / runner.JOURNAL_FILE, plan) as journal:
        for cell_id in plan.cell_ids:
            definition = plan.cell_definition(cell_id)
            attempt = journal.start_cell(cell_id)
            journal.record_arm_result(
                cell_id,
                attempt,
                {"arm": definition["arm"], "text": "candidate", "layer_apps": 1},
            )
    sealed = runner._seal_output_manifest(campaign_dir, plan)
    monkeypatch.setattr(
        runner, "_load_campaign_trust_policy", lambda *_args, **_kwargs: policy
    )
    args = SimpleNamespace(
        campaign_dir=str(campaign_dir),
        answer_reveal_attestation="",
    )

    assert runner._admit_answer_reveal(args, plan, tasks, sealed) is None
    request = json.loads(
        (campaign_dir / runner.ANSWER_REVEAL_REQUEST_FILE).read_bytes()
    )
    attestation = build_role_attestation(
        policy,
        role=TASK_ISSUER,
        payload=request["signed_payload"]["payload"],
        signed_at_unix=request["signed_payload"]["signed_at_unix"],
        private_key=role_keys[TASK_ISSUER],
    )
    attestation_path = tmp_path / "answer-reveal-attestation.json"
    attestation_path.write_bytes(canonical_json_bytes(attestation) + b"\n")
    args.answer_reveal_attestation = str(attestation_path)

    reveal = runner._admit_answer_reveal(args, plan, tasks, sealed)
    assert reveal is not None
    assert reveal["request_sha256"] == request["request_sha256"]
    assert reveal["task_issuer_attestation"] == attestation


def test_atomic_plan_artifact_is_create_or_exact_verify(tmp_path: Path):
    path = tmp_path / "plan.json"
    payload = b'{"plan":1}\n'

    runner._atomic_create_or_verify(path, payload)
    runner._atomic_create_or_verify(path, payload)

    assert path.read_bytes() == payload
    with pytest.raises(runner.CampaignProducerError, match="existing artifact differs"):
        runner._atomic_create_or_verify(path, b'{"plan":2}\n')


def test_atomic_artifact_rejects_concurrent_symlink_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "reveal.json"
    victim = tmp_path / "victim.json"
    victim.write_bytes(b'{"trusted":true}\n')

    def racing_link(
        _source: os.PathLike[str] | str,
        target: os.PathLike[str] | str,
        *,
        follow_symlinks: bool,
    ) -> None:
        assert follow_symlinks is False
        Path(target).symlink_to(victim)
        raise FileExistsError

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(runner.CampaignProducerError, match="symlink artifact rejected"):
        runner._atomic_create_or_verify(destination, victim.read_bytes())

    assert victim.read_bytes() == b'{"trusted":true}\n'
    assert destination.is_symlink()
    assert not list(tmp_path.glob(".*.tmp"))


def test_implementation_identity_covers_complete_latent_cortex_source():
    observed = runner._implementation_sha256()
    latent_root = runner.REPO_ROOT / "core/brain/llm/latent_cortex"
    expected = {
        str(path.relative_to(runner.REPO_ROOT)) for path in latent_root.glob("*.py")
    }

    assert expected.issubset(observed)
    assert "core/brain/llm/latent_cortex/fast_weights.py" in observed
    assert "tools/run_latent_cortex_paired_campaign.py" in observed
    assert all(len(digest) == 64 for digest in observed.values())


def test_fresh_checkpoint_identity_detects_same_size_weight_replacement(
    tmp_path: Path,
):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"original-weight-bytes")
    first = runner._fresh_checkpoint_file_fingerprint(tmp_path)

    shard.write_bytes(b"replaced-weight-bytes")
    second = runner._fresh_checkpoint_file_fingerprint(tmp_path)

    assert first["method"] == second["method"] == "sha256"
    assert first["files"] == second["files"] == 1
    assert first["fingerprint"] != second["fingerprint"]


def test_worker_command_resolves_relative_campaign_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(
        campaign_dir="relative-campaign",
        campaign_name="test",
        model="model",
        adapter="adapter",
        adapter_id="adapter-id",
        seeds="1",
        domains="mathematics",
        difficulty=2,
        profile="full",
        n_slots=4,
        branches=2,
        rlc_steps=2,
        rlc_profile="resident_full_stack",
        decode_max_tokens=64,
        episode_timeout=10.0,
        load_timeout=10.0,
        warmup_timeout=10.0,
        arm_timeout=20.0,
        campaign_timeout=30.0,
        equal_compute_max_samples=2,
        max_infra_attempts=1,
        confirmatory=False,
        contamination_audit="",
        contamination_trust_root="",
    )

    command = runner._worker_args(args, runner.BASE_RLC)

    campaign_index = command.index("--campaign-dir") + 1
    assert command[campaign_index] == str((tmp_path / "relative-campaign").resolve())
    assert "--seeds" not in command
    assert command[command.index("--seed-count") + 1] == "1"

    policy = runner._detached_broker_policy(args)
    assert len(policy) == len(runner.FULL_ARMS)
    assert {entry["command"][-1] for entry in policy} == set(runner.FULL_ARMS)
    assert all(entry["cwd"] == str(runner.REPO_ROOT) for entry in policy)
    assert all(entry["max_invocations"] == 1 for entry in policy)


def test_run_child_uses_detached_broker_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    args = SimpleNamespace(
        campaign_dir=str(tmp_path),
        campaign_name="test",
        model="model",
        adapter="adapter",
        adapter_id="adapter-id",
        seeds="1",
        domains="mathematics",
        difficulty=2,
        profile="primary",
        n_slots=4,
        branches=2,
        rlc_steps=2,
        rlc_profile="resident_full_stack",
        decode_max_tokens=64,
        episode_timeout=10.0,
        load_timeout=10.0,
        warmup_timeout=10.0,
        arm_timeout=20.0,
        campaign_timeout=30.0,
        equal_compute_max_samples=2,
        max_infra_attempts=1,
        confirmatory=False,
        contamination_audit="",
        contamination_trust_root="",
    )
    observed: dict[str, object] = {}

    def fake_broker(command, *, cwd, stdout_path, timeout_s):
        observed.update(
            command=command,
            cwd=cwd,
            stdout_path=stdout_path,
            timeout_s=timeout_s,
        )
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr(runner, "broker_available", lambda: True)
    monkeypatch.setattr(runner, "run_brokered_process", fake_broker)

    assert runner._run_child(args, runner.BASE_RLC, 12.5) == 17
    assert observed["command"] == runner._worker_args(args, runner.BASE_RLC)
    assert observed["cwd"] == runner.REPO_ROOT
    assert observed["stdout_path"] == tmp_path / runner.LOG_FILE
    assert observed["timeout_s"] == 12.5


def test_claim_eligibility_requires_full_powered_seven_domain_protocol():
    args = SimpleNamespace(
        confirmatory=True,
        seed_values=tuple((1 << 60) + value for value in range(20)),
        profile="full",
        domain_values=runner.FRONTIER_DOMAINS,
    )
    model_identity = {
        "runtime_bundle": {
            "model_type": "qwen2",
            "logical_parameter_count": 32_000_000_000,
            "logical_parameter_count_basis": "architecture_config_logical",
        }
    }
    adapter_identity = {
        "manifest": {"dataset_manifest": {"sha256": "d" * 64}}
    }
    audit = {"status": "passed_zero_overlap", "signature": {"verified": True}}
    trust = {"prelaunch_verified": True, "externally_custodied": True}
    assert (
        runner._claim_eligible(
            args, model_identity, adapter_identity, audit, trust
        )
        is True
    )

    args.domain_values = ("mathematics", "coding")
    assert (
        runner._claim_eligible(
            args, model_identity, adapter_identity, audit, trust
        )
        is False
    )
    args.domain_values = runner.FRONTIER_DOMAINS
    args.seed_values = tuple((1 << 60) + value for value in range(19))
    assert (
        runner._claim_eligible(
            args, model_identity, adapter_identity, audit, trust
        )
        is False
    )
    args.seed_values = tuple((1 << 60) + value for value in range(20))
    assert (
        runner._claim_eligible(
            args, model_identity, adapter_identity, {}, trust
        )
        is False
    )
    assert (
        runner._claim_eligible(
            args, model_identity, adapter_identity, audit, None
        )
        is False
    )
    args.seed_values = tuple(range(20))
    assert (
        runner._claim_eligible(
            args, model_identity, adapter_identity, audit, trust
        )
        is False
    )


def test_contamination_audit_verifies_ed25519_external_trust_root(tmp_path: Path):
    tasks = generate_task_battery([7], domains=("mathematics",), difficulty=2)
    manifest = runner.build_task_manifest(tasks)
    private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    public_key = private_key.public_key()
    public_der = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    key_id = hashlib.sha256(public_der).hexdigest()
    body = {
        "schema": runner.CONTAMINATION_AUDIT_SCHEMA,
        "task_manifest_sha256": manifest.manifest_sha256,
        "status": "passed_zero_overlap",
        "overlap_count": 0,
        "auditor_independence": "external",
        "corpora": [{"name": "external-corpus", "snapshot_sha256": "f" * 64}],
        "methods": ["exact_prompt", "normalized_prompt", "token_fivegram"],
    }
    audit = {
        **body,
        "signature": {
            "algorithm": "ed25519",
            "key_id": key_id,
            "signature_b64": base64.b64encode(
                private_key.sign(canonical_json_bytes(body))
            ).decode("ascii"),
        },
    }
    audit_path = tmp_path / "audit.json"
    trust_path = tmp_path / "auditor.pem"
    audit_path.write_text(json.dumps(audit))
    trust_path.write_bytes(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    args = SimpleNamespace(
        contamination_audit=str(audit_path),
        contamination_trust_root=str(trust_path),
    )

    verified = runner._contamination_audit(args, tasks)

    assert verified["signature"]["verified"] is True
    assert verified["signature"]["trust_root_sha256"] == key_id
    assert verified["signature"]["signed_payload_sha256"] == hashlib.sha256(
        canonical_json_bytes(body)
    ).hexdigest()

    audit["overlap_count"] = 1
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(runner.CampaignProducerError):
        runner._contamination_audit(args, tasks)


def test_contamination_audit_must_cover_bound_adapter_training_corpus(
    tmp_path: Path,
):
    tasks = generate_task_battery([7], domains=("mathematics",), difficulty=2)
    manifest = runner.build_task_manifest(tasks)
    private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    public_key = private_key.public_key()
    public_der = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    body = {
        "schema": runner.CONTAMINATION_AUDIT_SCHEMA,
        "task_manifest_sha256": manifest.manifest_sha256,
        "status": "passed_zero_overlap",
        "overlap_count": 0,
        "auditor_independence": "external",
        "corpora": [{"name": "wrong-corpus", "snapshot_sha256": "f" * 64}],
        "methods": ["exact_prompt", "normalized_prompt", "token_fivegram"],
    }
    audit = {
        **body,
        "signature": {
            "algorithm": "ed25519",
            "key_id": hashlib.sha256(public_der).hexdigest(),
            "signature_b64": base64.b64encode(
                private_key.sign(canonical_json_bytes(body))
            ).decode("ascii"),
        },
    }
    audit_path = tmp_path / "audit.json"
    trust_path = tmp_path / "auditor.pem"
    audit_path.write_text(json.dumps(audit))
    trust_path.write_bytes(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    args = SimpleNamespace(
        contamination_audit=str(audit_path),
        contamination_trust_root=str(trust_path),
    )

    with pytest.raises(
        runner.CampaignProducerError,
        match="does not cover the adapter training corpus",
    ):
        runner._contamination_audit(
            args,
            tasks,
            expected_training_corpus_sha256="e" * 64,
        )


def test_prelaunch_trust_verifies_all_pinned_roles_before_inference(
    tmp_path: Path,
):
    now = int(time.time())
    root_key = Ed25519PrivateKey.generate()
    role_keys = {
        role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES
    }

    def role_pin(role: str) -> dict[str, str]:
        raw = role_keys[role].public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        implementation_sha256 = (
            runner._prelaunch_role_implementation_sha256(role)
            if role in {TASK_ISSUER, CAMPAIGN_RUNNER, "contamination_auditor"}
            else hashlib.sha256(f"{role}:implementation".encode()).hexdigest()
        )
        return {
            "signer_id": f"{role}-signer",
            "organization_id": f"{role}-organization",
            "public_key_b64": base64.b64encode(raw).decode("ascii"),
            "key_id": hashlib.sha256(raw).hexdigest(),
            "implementation_sha256": implementation_sha256,
            "release_sha256": hashlib.sha256(f"{role}:release".encode()).hexdigest(),
            "custody_class": "remote_hsm",
            "custody_evidence_sha256": hashlib.sha256(
                f"{role}:custody".encode()
            ).hexdigest(),
        }

    policy_body = {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": "resident-prelaunch-test",
        "policy_revision": 1,
        "campaign_name": "resident-prelaunch-test",
        "protocol_sha256": runner._campaign_protocol_sha256(),
        "previous_policy_sha256": None,
        "revoked_key_ids": [],
        "issued_at_unix": now - 10,
        "not_before_unix": now - 5,
        "expires_at_unix": now + 3600,
        "roles": {role: role_pin(role) for role in CAMPAIGN_TRUST_ROLES},
    }
    root_payload = canonical_json_bytes(policy_body)
    root_raw = root_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    policy_document = {
        **policy_body,
        "root_signature": {
            "algorithm": "Ed25519",
            "key_id": hashlib.sha256(root_raw).hexdigest(),
            "signature_b64": base64.b64encode(
                root_key.sign(root_payload)
            ).decode("ascii"),
            "signed_payload_sha256": hashlib.sha256(root_payload).hexdigest(),
        },
    }
    root_pem = root_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    verified_policy = validate_campaign_trust_policy(
        policy_document,
        trusted_root_public_key_pem=root_pem,
        expected_campaign_name="resident-prelaunch-test",
        expected_protocol_sha256=runner._campaign_protocol_sha256(),
        now_unix=now,
    )

    tasks = generate_task_battery([7], domains=("mathematics",), difficulty=1)
    task_manifest = runner.build_task_manifest(tasks)
    auditor_key = role_keys["contamination_auditor"]
    auditor_der = auditor_key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    audit_body = {
        "schema": runner.CONTAMINATION_AUDIT_SCHEMA,
        "task_manifest_sha256": task_manifest.manifest_sha256,
        "status": "passed_zero_overlap",
        "overlap_count": 0,
        "auditor_independence": "external",
        "corpora": [{"name": "training-corpus", "snapshot_sha256": "d" * 64}],
        "methods": ["exact_prompt", "normalized_prompt", "token_fivegram"],
    }
    audit_payload = canonical_json_bytes(audit_body)
    audit = {
        **audit_body,
        "signature": {
            "algorithm": "ed25519",
            "key_id": hashlib.sha256(auditor_der).hexdigest(),
            "signature_b64": base64.b64encode(
                auditor_key.sign(audit_payload)
            ).decode("ascii"),
            "signed_payload_sha256": hashlib.sha256(audit_payload).hexdigest(),
            "public_key_der_b64": base64.b64encode(auditor_der).decode("ascii"),
            "trust_root_sha256": hashlib.sha256(auditor_der).hexdigest(),
            "verified": True,
        },
    }
    execution_config = {
        "difficulty": 1,
        "task_registry_version": runner.REGISTRY_VERSION,
        "generation_seeds": [7],
        "domains": ["mathematics"],
    }
    unsigned_plan = build_campaign_plan(
        "resident-prelaunch-test",
        tasks,
        model_identity={"model": "sealed"},
        adapter_identity={"adapter": "sealed"},
        execution_config=execution_config,
        contamination_audit=audit,
        claim_eligible=False,
    )
    args = SimpleNamespace(campaign_name="resident-prelaunch-test")
    payloads = runner._prelaunch_payloads(
        args,
        unsigned_plan=unsigned_plan,
        policy=verified_policy,
    )
    issuer_attestation = build_role_attestation(
        verified_policy,
        role=TASK_ISSUER,
        payload=payloads[TASK_ISSUER],
        signed_at_unix=now,
        private_key=role_keys[TASK_ISSUER],
    )
    runner_attestation = build_role_attestation(
        verified_policy,
        role=CAMPAIGN_RUNNER,
        payload=payloads[CAMPAIGN_RUNNER],
        signed_at_unix=now,
        private_key=role_keys[CAMPAIGN_RUNNER],
    )
    policy_path = tmp_path / "policy.json"
    root_path = tmp_path / "root.pem"
    issuer_path = tmp_path / "issuer.json"
    runner_path = tmp_path / "runner.json"
    policy_path.write_text(json.dumps(policy_document))
    root_path.write_bytes(root_pem)
    issuer_path.write_text(json.dumps(issuer_attestation))
    runner_path.write_text(json.dumps(runner_attestation))
    args.campaign_trust_policy = str(policy_path)
    args.campaign_trust_root = str(root_path)
    args.task_issuer_attestation = str(issuer_path)
    args.runner_attestation = str(runner_path)

    trust = runner._verified_campaign_trust(
        args,
        unsigned_plan=unsigned_plan,
        contamination_audit=audit,
    )

    assert trust is not None
    assert trust["prelaunch_verified"] is True
    assert trust["externally_custodied"] is True
    assert trust["policy_sha256"] == verified_policy.policy_sha256
    assert trust["unsigned_plan_sha256"] == unsigned_plan.plan_sha256


def test_effective_full_stack_shape_is_frozen_not_cli_placeholder():
    args = SimpleNamespace(
        rlc_profile="resident_full_stack",
        n_slots=16,
        branches=7,
        rlc_steps=8,
        decode_max_tokens=512,
    )

    effective = runner._build_rlc_config(args)

    assert effective.workspace.n_slots == 4
    assert effective.branches.n_branches == 2
    assert effective.recurrence.max_steps == 2
    assert effective.latent_opt.enabled is True
    assert effective.fast_weights.enabled is True


def test_v2_execution_spec_overrides_cli_and_preserves_training_graph():
    args = SimpleNamespace(
        rlc_profile="resident_full_stack",
        n_slots=99,
        branches=7,
        rlc_steps=19,
        decode_max_tokens=128,
    )
    spec = RLCExecutionSpec(
        n_slots=6,
        branch_roles=("constructive_solution", "counterexample_search"),
        recurrent_steps=4,
        exchange_interval=1,
        alpha=0.4,
    )

    effective = runner._build_rlc_config(args, spec.to_dict())

    assert effective.workspace.n_slots == 6
    assert effective.branches.roles == spec.branch_roles
    assert effective.recurrence.max_steps == 4
    assert effective.recurrence.min_steps == 4
    assert effective.recurrence.fixed_depth is True
    assert effective.latent_opt.enabled is False
    assert effective.fast_weights.enabled is False
    assert effective.escape == {"enabled": False}


def test_projection_resolution_is_exact_and_rejects_missing_owner():
    projection = object()
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(o_proj=projection),
                )
            ]
        )
    )

    parent, leaf, observed = runner._resolve_projection(
        model, "model.layers.0.self_attn.o_proj"
    )
    assert parent is model.model.layers[0].self_attn
    assert leaf == "o_proj"
    assert observed is projection
    with pytest.raises(runner.CampaignProducerError, match="owner is missing"):
        runner._resolve_projection(model, "model.layers.0.mlp.o_proj")


def test_deadline_alarm_interrupts_and_restores_process_timer():
    before = signal.getitimer(signal.ITIMER_REAL)
    with pytest.raises(TimeoutError, match="test_stage exceeded"):
        with runner._deadline_alarm(0.02, "test_stage"):
            time.sleep(0.2)

    after = signal.getitimer(signal.ITIMER_REAL)
    assert before == (0.0, 0.0)
    assert after == (0.0, 0.0)
