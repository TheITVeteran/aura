from __future__ import annotations

import base64
import copy
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
from tools import verify_paired_campaign_evidence as evidence_verifier


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
        "worker_origin_protocol": "preauthorized_ephemeral_chain_v2",
        "worker_origin_attempt_slots": 3,
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

    runner._score_sealed_outputs(campaign_dir, plan, tasks)
    with runner.CampaignJournal(campaign_dir / runner.JOURNAL_FILE, plan) as journal:
        campaign_manifest = journal.finalize(campaign_dir / runner.MANIFEST_FILE)
    args.final_run_attestation = ""
    grade = {"grade_sha256": "e" * 64}
    worker_authorizations = {"manifest_sha256": "a" * 64}
    worker_lifecycle = {"manifest_sha256": "c" * 64}
    worker_key_erasure = {"manifest_sha256": "b" * 64}
    (campaign_dir / runner.WORKER_AUTHORIZATION_MANIFEST_FILE).write_bytes(
        canonical_json_bytes(worker_authorizations) + b"\n"
    )
    (campaign_dir / runner.WORKER_KEY_ERASURE_MANIFEST_FILE).write_bytes(
        canonical_json_bytes(worker_key_erasure) + b"\n"
    )
    (campaign_dir / runner.WORKER_LIFECYCLE_MANIFEST_FILE).write_bytes(
        canonical_json_bytes(worker_lifecycle) + b"\n"
    )
    assert (
        runner._admit_final_run_envelope(
            args,
            plan,
            sealed_outputs=sealed,
            answer_reveal=reveal,
            campaign_manifest=campaign_manifest,
            grade=grade,
            worker_authorizations=worker_authorizations,
            worker_lifecycle=worker_lifecycle,
            worker_key_erasure=worker_key_erasure,
        )
        is None
    )
    final_request = json.loads(
        (campaign_dir / runner.FINAL_RUN_REQUEST_FILE).read_bytes()
    )
    assert (
        final_request["signed_payload"]["payload"][
            "worker_lifecycle_manifest_sha256"
        ]
        == worker_lifecycle["manifest_sha256"]
    )
    final_attestation = build_role_attestation(
        policy,
        role=CAMPAIGN_RUNNER,
        payload=final_request["signed_payload"]["payload"],
        signed_at_unix=final_request["signed_payload"]["signed_at_unix"],
        private_key=role_keys[CAMPAIGN_RUNNER],
    )
    final_attestation_path = tmp_path / "final-run-attestation.json"
    final_attestation_path.write_bytes(
        canonical_json_bytes(final_attestation) + b"\n"
    )
    args.final_run_attestation = str(final_attestation_path)

    final_envelope = runner._admit_final_run_envelope(
        args,
        plan,
        sealed_outputs=sealed,
        answer_reveal=reveal,
        campaign_manifest=campaign_manifest,
        grade=grade,
        worker_authorizations=worker_authorizations,
        worker_lifecycle=worker_lifecycle,
        worker_key_erasure=worker_key_erasure,
    )
    assert final_envelope is not None
    assert final_envelope["request_sha256"] == final_request["request_sha256"]
    assert final_envelope["campaign_runner_attestation"] == final_attestation
    (campaign_dir / runner.GRADE_FILE).write_bytes(
        canonical_json_bytes(grade) + b"\n"
    )
    failures, detail = evidence_verifier._verify_final_run_envelope(
        campaign_dir,
        plan=plan,
        trusted_policy=policy,
    )
    assert failures == []
    assert detail["verified"] is True
    (campaign_dir / runner.GRADE_FILE).write_bytes(
        canonical_json_bytes({"grade_sha256": "f" * 64}) + b"\n"
    )
    failures, detail = evidence_verifier._verify_final_run_envelope(
        campaign_dir,
        plan=plan,
        trusted_policy=policy,
    )
    assert "final run payload differs from independent reconstruction" in failures
    assert detail["verified"] is False


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


def test_claim_broker_policy_covers_every_exact_worker_attempt_command(
    tmp_path: Path,
):
    args, _plan, _policy, _role_keys = _worker_origin_claim_fixture(tmp_path)
    policy = runner._detached_broker_policy(args)

    expected = {
        tuple(runner._worker_args(args, arm, worker_attempt_slot=attempt_slot))
        for arm in runner.PRIMARY_ARMS
        for attempt_slot in range(1, args.max_infra_attempts + 1)
    }
    observed = {tuple(entry["command"]) for entry in policy}
    assert observed == expected
    assert len(policy) == len(expected)
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


def _worker_origin_claim_fixture(tmp_path: Path):
    now = int(time.time())
    campaign_name = "worker-origin-claim-test"
    policy, role_keys = _external_policy_fixture(campaign_name, now)
    tasks = generate_task_battery([7], domains=("mathematics",), difficulty=1)
    task_manifest = runner.build_task_manifest(tasks)
    auditor = Ed25519PrivateKey.generate()
    auditor_der = auditor.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    audit_body = {
        "schema": runner.CONTAMINATION_AUDIT_SCHEMA,
        "task_manifest_sha256": task_manifest.manifest_sha256,
        "status": "passed_zero_overlap",
        "overlap_count": 0,
        "auditor_independence": "external",
        "corpora": [{"name": "training", "snapshot_sha256": "d" * 64}],
        "methods": ["exact_prompt", "normalized_prompt", "token_fivegram"],
    }
    audit_bytes = canonical_json_bytes(audit_body)
    audit = {
        **audit_body,
        "signature": {
            "algorithm": "ed25519",
            "key_id": hashlib.sha256(auditor_der).hexdigest(),
            "signature_b64": base64.b64encode(auditor.sign(audit_bytes)).decode(
                "ascii"
            ),
            "signed_payload_sha256": hashlib.sha256(audit_bytes).hexdigest(),
            "public_key_der_b64": base64.b64encode(auditor_der).decode("ascii"),
            "trust_root_sha256": hashlib.sha256(auditor_der).hexdigest(),
            "verified": True,
        },
    }
    execution_config = {
        "difficulty": 1,
        "task_registry_version": runner.REGISTRY_VERSION,
        "generation_seed_count": 1,
        "generation_seed_min_entropy_bits": 61,
        "generation_seed_policy": "external_issuer_uniform_63bit",
        "generation_seed_disclosure": "post_seal_answer_reveal",
        "domains": ["mathematics"],
        "worker_task_material": "public_manifest_only",
        "answer_reveal_protocol": "sealed_outputs_then_issuer_reveal_v1",
        "worker_origin_protocol": "preauthorized_ephemeral_chain_v2",
        "worker_origin_attempt_slots": 2,
        "implementation_sha256": runner._implementation_sha256(),
    }
    unsigned = build_campaign_plan(
        campaign_name,
        tasks,
        model_identity={"model": "sealed"},
        adapter_identity={"adapter": "sealed"},
        execution_config=execution_config,
        contamination_audit=audit,
        claim_eligible=False,
        arms=runner.PRIMARY_ARMS,
    )
    plan = build_campaign_plan(
        campaign_name,
        tasks,
        model_identity={"model": "sealed"},
        adapter_identity={"adapter": "sealed"},
        execution_config=execution_config,
        contamination_audit=audit,
        campaign_trust={
            "prelaunch_verified": True,
            "externally_custodied": True,
            "policy_sha256": policy.policy_sha256,
            "unsigned_plan_sha256": unsigned.plan_sha256,
        },
        claim_eligible=True,
        arms=runner.PRIMARY_ARMS,
    )
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    runner._persist_plan(campaign_dir, plan)
    args = SimpleNamespace(
        campaign_dir=str(campaign_dir),
        campaign_name=campaign_name,
        model="/sealed/model",
        adapter="/sealed/adapter",
        adapter_id="adapter-id",
        personality_adapter="none",
        seeds=str(1 << 60),
        seed_values=(1 << 60,),
        seed_count=0,
        seed_entropy_bits=0,
        domains="mathematics",
        domain_values=("mathematics",),
        difficulty=1,
        task_registry_version=runner.REGISTRY_VERSION,
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
        max_infra_attempts=2,
        confirmatory=True,
        contamination_audit="",
        contamination_trust_root="",
        campaign_trust_policy="/external/policy.json",
        campaign_trust_root="/external/root.pem",
        task_issuer_attestation="/external/issuer.json",
        runner_attestation="/external/runner.json",
        worker_arm="",
    )
    return args, plan, policy, role_keys


def _sign_worker_authorization_requests(
    args: SimpleNamespace,
    policy,
    runner_key: Ed25519PrivateKey,
) -> None:
    campaign_dir = Path(args.campaign_dir)
    for arm in runner.PRIMARY_ARMS:
        for attempt_slot in range(1, args.max_infra_attempts + 1):
            paths = runner._worker_origin_paths(campaign_dir, arm, attempt_slot)
            request = json.loads(paths["request"].read_bytes())
            attestation = build_role_attestation(
                policy,
                role=CAMPAIGN_RUNNER,
                payload=request["signed_payload"]["payload"],
                signed_at_unix=request["signed_payload"]["signed_at_unix"],
                private_key=runner_key,
            )
            paths["attestation"].write_bytes(
                canonical_json_bytes(attestation) + b"\n"
            )


def test_claim_worker_origin_lifecycle_is_signed_chained_and_erased(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    args, plan, policy, role_keys = _worker_origin_claim_fixture(tmp_path)
    monkeypatch.setattr(
        runner, "_load_campaign_trust_policy", lambda *_args, **_kwargs: policy
    )

    assert runner._admit_worker_authorizations(args, plan) is None
    _sign_worker_authorization_requests(
        args,
        policy,
        role_keys[CAMPAIGN_RUNNER],
    )
    authorization_manifest = runner._admit_worker_authorizations(args, plan)
    assert authorization_manifest is not None
    assert authorization_manifest["claim_required"] is True
    assert len(authorization_manifest["entries"]) == (
        len(runner.PRIMARY_ARMS) * args.max_infra_attempts
    )

    entries = {
        (entry["arm"], entry["attempt_slot"]): entry
        for entry in authorization_manifest["entries"]
    }
    for arm in runner.PRIMARY_ARMS:
        command = runner._worker_args(args, arm, worker_attempt_slot=1)
        launch, launch_paths = runner._record_worker_launch(
            args, arm, 1, command
        )
        runner._record_worker_exit(launch_paths, launch, returncode=0)
    with runner.CampaignJournal(
        Path(args.campaign_dir) / runner.JOURNAL_FILE, plan
    ) as journal:
        for cell_id in plan.cell_ids:
            definition = plan.cell_definition(cell_id)
            arm = definition["arm"]
            entry = entries[(arm, 1)]
            paths = runner._worker_origin_paths(Path(args.campaign_dir), arm, 1)
            private_key = runner._load_worker_private_key(paths["private_key"])
            attestation = json.loads(paths["attestation"].read_bytes())
            attempt_id = journal.start_cell(cell_id)
            result = {
                "arm": arm,
                "text": "candidate",
                "layer_apps": 1,
                "runtime_model_identity": {
                    "worker_boot_id": entry["worker_boot_id"]
                },
            }
            result["worker_origin"] = (
                runner.build_legacy_worker_result_origin(
                    authorization_attestation=attestation,
                    authorization_payload=entry["authorization_payload"],
                    private_key=private_key,
                    result_body=result,
                    cell_id=cell_id,
                    attempt_id=attempt_id,
                    worker_boot_id=entry["worker_boot_id"],
                    sequence=1,
                )
            )
            journal.record_arm_result(cell_id, attempt_id, result)

    chains = runner._verify_worker_origin_chains(args, plan)
    assert chains is not None
    assert len(chains["chains"]) == len(runner.PRIMARY_ARMS)
    assert all(chain["result_count"] == 1 for chain in chains["chains"])
    lifecycle = runner._build_worker_lifecycle_manifest(
        args,
        plan,
        authorization_manifest=authorization_manifest,
    )
    assert lifecycle is not None
    sealed = runner._seal_output_manifest(
        Path(args.campaign_dir),
        plan,
        worker_origin_chains=chains,
        worker_lifecycle=lifecycle,
    )
    assert (
        sealed["worker_lifecycle_manifest_sha256"]
        == lifecycle["manifest_sha256"]
    )
    erasure = runner._erase_worker_private_keys(
        args,
        plan,
        authorization_manifest=authorization_manifest,
        sealed_outputs=sealed,
    )
    assert erasure is not None
    assert erasure["all_private_paths_absent"] is True
    assert erasure["copy_exclusion_claimed"] is False
    for entry in authorization_manifest["entries"]:
        paths = runner._worker_origin_paths(
            Path(args.campaign_dir), entry["arm"], entry["attempt_slot"]
        )
        assert not paths["private_key"].exists()
        assert paths["erasure_intent"].exists()
        assert paths["erasure"].exists()

    first = authorization_manifest["entries"][0]
    first_paths = runner._worker_origin_paths(
        Path(args.campaign_dir), first["arm"], first["attempt_slot"]
    )
    erasure_manifest_path = (
        Path(args.campaign_dir) / runner.WORKER_KEY_ERASURE_MANIFEST_FILE
    )
    erasure_manifest_path.unlink()
    first_paths["erasure"].unlink()
    erasure = runner._erase_worker_private_keys(
        args,
        plan,
        authorization_manifest=authorization_manifest,
        sealed_outputs=sealed,
    )
    assert erasure is not None
    assert first_paths["erasure"].exists()
    assert not first_paths["private_key"].exists()

    assert runner._admit_worker_authorizations(args, plan) == authorization_manifest
    with runner.CampaignJournal(
        Path(args.campaign_dir) / runner.JOURNAL_FILE, plan
    ) as journal:
        result_records = journal.result_records()
    failures, detail = evidence_verifier._verify_worker_origin_evidence(
        Path(args.campaign_dir),
        plan=plan,
        result_records=result_records,
        trusted_policy=policy,
    )
    assert failures == [
        "worker execution origin is unproven: producer process held exportable "
        "worker signing keys"
    ]
    assert detail["verified"] is False
    assert detail["cryptographic_chain_verified"] is True
    assert detail["worker_execution_origin_proven"] is False
    assert detail["consumed_worker_attempts"] == len(runner.PRIMARY_ARMS)
    assert detail["private_key_copy_exclusion_proven"] is False

    original_launch = first_paths["launch"].read_bytes()
    original_exit = first_paths["exit"].read_bytes()
    attacked_launch = json.loads(original_launch)
    attacked_launch["launched_at_unix_ns"] += 1
    attacked_exit = json.loads(original_exit)
    attacked_exit["launch_sha256"] = hashlib.sha256(
        canonical_json_bytes(attacked_launch)
    ).hexdigest()
    attacked_exit_material = dict(attacked_exit)
    attacked_exit_material.pop("receipt_sha256")
    attacked_exit["receipt_sha256"] = hashlib.sha256(
        canonical_json_bytes(attacked_exit_material)
    ).hexdigest()
    first_paths["launch"].write_bytes(
        canonical_json_bytes(attacked_launch) + b"\n"
    )
    first_paths["exit"].write_bytes(canonical_json_bytes(attacked_exit) + b"\n")
    failures, detail = evidence_verifier._verify_worker_origin_evidence(
        Path(args.campaign_dir),
        plan=plan,
        result_records=result_records,
        trusted_policy=policy,
    )
    assert any("worker lifecycle manifest differs" in failure for failure in failures)
    assert detail["verified"] is False
    first_paths["launch"].write_bytes(original_launch)
    first_paths["exit"].write_bytes(original_exit)

    original_erasure_manifest = erasure_manifest_path.read_bytes()
    original_erasure_intent = first_paths["erasure_intent"].read_bytes()
    attacked_erasure_intent = json.loads(original_erasure_intent)
    attacked_erasure_intent["method"] = "unlink_without_write_ahead_intent"
    attacked_intent_material = dict(attacked_erasure_intent)
    attacked_intent_material.pop("intent_sha256")
    attacked_erasure_intent["intent_sha256"] = hashlib.sha256(
        canonical_json_bytes(attacked_intent_material)
    ).hexdigest()
    first_paths["erasure_intent"].write_bytes(
        canonical_json_bytes(attacked_erasure_intent) + b"\n"
    )
    with pytest.raises(
        runner.CampaignProducerError,
        match="worker key erasure intent differs",
    ):
        runner._erase_worker_private_keys(
            args,
            plan,
            authorization_manifest=authorization_manifest,
            sealed_outputs=sealed,
        )
    first_paths["erasure_intent"].write_bytes(original_erasure_intent)

    attacked_erasure_manifest = copy.deepcopy(erasure)
    attacked_erasure_manifest["copy_exclusion_claimed"] = True
    attacked_erasure_material = dict(attacked_erasure_manifest)
    attacked_erasure_material.pop("manifest_sha256")
    attacked_erasure_manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(attacked_erasure_material)
    ).hexdigest()
    erasure_manifest_path.write_bytes(
        canonical_json_bytes(attacked_erasure_manifest) + b"\n"
    )
    with pytest.raises(
        runner.CampaignProducerError,
        match="worker key erasure manifest is invalid",
    ):
        runner._erase_worker_private_keys(
            args,
            plan,
            authorization_manifest=authorization_manifest,
            sealed_outputs=sealed,
        )
    erasure_manifest_path.write_bytes(original_erasure_manifest)

    original_erasure_receipt = first_paths["erasure"].read_bytes()
    attacked_erasure_receipt = copy.deepcopy(erasure["receipts"][0])
    attacked_erasure_receipt["absence_verified"] = False
    attacked_receipt_material = dict(attacked_erasure_receipt)
    attacked_receipt_material.pop("receipt_sha256")
    attacked_erasure_receipt["receipt_sha256"] = hashlib.sha256(
        canonical_json_bytes(attacked_receipt_material)
    ).hexdigest()
    attacked_erasure_manifest = copy.deepcopy(erasure)
    attacked_erasure_manifest["receipts"][0] = attacked_erasure_receipt
    attacked_erasure_material = dict(attacked_erasure_manifest)
    attacked_erasure_material.pop("manifest_sha256")
    attacked_erasure_manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(attacked_erasure_material)
    ).hexdigest()
    first_paths["erasure"].write_bytes(
        canonical_json_bytes(attacked_erasure_receipt) + b"\n"
    )
    erasure_manifest_path.write_bytes(
        canonical_json_bytes(attacked_erasure_manifest) + b"\n"
    )
    with pytest.raises(
        runner.CampaignProducerError,
        match="worker key erasure receipt differs",
    ):
        runner._erase_worker_private_keys(
            args,
            plan,
            authorization_manifest=authorization_manifest,
            sealed_outputs=sealed,
        )
    first_paths["erasure"].write_bytes(original_erasure_receipt)
    erasure_manifest_path.write_bytes(original_erasure_manifest)

    attacked_records = copy.deepcopy(result_records)
    attacked_records[0]["result"]["text"] = "tampered after signing"
    failures, detail = evidence_verifier._verify_worker_origin_evidence(
        Path(args.campaign_dir),
        plan=plan,
        result_records=tuple(attacked_records),
        trusted_policy=policy,
    )
    assert failures
    assert detail["verified"] is False

    first_paths["private_key"].write_bytes(b"x" * 32)
    failures, detail = evidence_verifier._verify_worker_origin_evidence(
        Path(args.campaign_dir),
        plan=plan,
        result_records=result_records,
        trusted_policy=policy,
    )
    assert any("private key remains" in failure for failure in failures)
    assert detail["verified"] is False


def test_worker_attempt_slot_is_single_use_across_parent_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    args, plan, policy, role_keys = _worker_origin_claim_fixture(tmp_path)
    monkeypatch.setattr(
        runner, "_load_campaign_trust_policy", lambda *_args, **_kwargs: policy
    )
    assert runner._admit_worker_authorizations(args, plan) is None
    _sign_worker_authorization_requests(
        args,
        policy,
        role_keys[CAMPAIGN_RUNNER],
    )
    assert runner._admit_worker_authorizations(args, plan) is not None
    arm = runner.PRIMARY_ARMS[0]
    command = runner._worker_args(args, arm, worker_attempt_slot=1)
    runner._record_worker_launch(args, arm, 1, command)

    assert (
        runner._next_worker_attempt_slot(
            Path(args.campaign_dir), arm, maximum=args.max_infra_attempts
        )
        == 2
    )
    with pytest.raises(
        runner.CampaignProducerError, match="already consumed"
    ):
        runner._record_worker_launch(args, arm, 1, command)


def test_worker_launcher_failure_consumes_slot_with_explicit_exit_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    args, plan, policy, role_keys = _worker_origin_claim_fixture(tmp_path)
    monkeypatch.setattr(
        runner, "_load_campaign_trust_policy", lambda *_args, **_kwargs: policy
    )
    assert runner._admit_worker_authorizations(args, plan) is None
    _sign_worker_authorization_requests(
        args,
        policy,
        role_keys[CAMPAIGN_RUNNER],
    )
    assert runner._admit_worker_authorizations(args, plan) is not None
    monkeypatch.setattr(runner, "broker_available", lambda: True)

    def fail_launcher(*_args, **_kwargs):
        raise RuntimeError("broker unavailable")

    monkeypatch.setattr(runner, "run_brokered_process", fail_launcher)
    arm = runner.PRIMARY_ARMS[0]
    with pytest.raises(RuntimeError, match="broker unavailable"):
        runner._run_child(
            args,
            arm,
            timeout_s=1.0,
            worker_attempt_slot=1,
        )

    paths = runner._worker_origin_paths(Path(args.campaign_dir), arm, 1)
    exit_receipt = json.loads(paths["exit"].read_bytes())
    assert exit_receipt["schema"] == "aura.latent_cortex.worker_exit.v2"
    assert exit_receipt["outcome"] == "launcher_failure"
    assert exit_receipt["returncode"] is None
    assert exit_receipt["error_type"] == "RuntimeError"
    assert (
        runner._next_worker_attempt_slot(
            Path(args.campaign_dir), arm, maximum=args.max_infra_attempts
        )
        == 2
    )


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
