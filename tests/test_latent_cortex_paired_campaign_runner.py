from __future__ import annotations

import base64
import hashlib
import json
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.frontier_tasks import generate_task_battery
from tools import run_latent_cortex_paired_campaign as runner


def test_majority_output_uses_parsed_answer_without_gold_access():
    first = 'reasoning\nFINAL_ANSWER: {"count":2,"witness":[1,5]}'
    second = 'different\nFINAL_ANSWER: {"witness":[1,5],"count":2}'
    wrong = 'FINAL_ANSWER: {"count":9,"witness":[]}'

    assert runner._majority_output([first, wrong, second]) == first


def test_atomic_plan_artifact_is_create_or_exact_verify(tmp_path: Path):
    path = tmp_path / "plan.json"
    payload = b'{"plan":1}\n'

    runner._atomic_create_or_verify(path, payload)
    runner._atomic_create_or_verify(path, payload)

    assert path.read_bytes() == payload
    with pytest.raises(runner.CampaignProducerError, match="existing artifact differs"):
        runner._atomic_create_or_verify(path, b'{"plan":2}\n')


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


def test_claim_eligibility_requires_full_powered_seven_domain_protocol():
    args = SimpleNamespace(
        confirmatory=True,
        seed_values=tuple(range(20)),
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
    audit = {"status": "passed_zero_overlap", "signature": {"verified": True}}
    assert runner._claim_eligible(args, model_identity, audit) is True

    args.domain_values = ("mathematics", "coding")
    assert runner._claim_eligible(args, model_identity, audit) is False
    args.domain_values = runner.FRONTIER_DOMAINS
    args.seed_values = tuple(range(19))
    assert runner._claim_eligible(args, model_identity, audit) is False
    args.seed_values = tuple(range(20))
    assert runner._claim_eligible(args, model_identity, {}) is False


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
