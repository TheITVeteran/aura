"""Contract tests: signed zero-overlap contamination audit producer/verifier.

The confirmatory paired campaign refuses to run without a signed audit
proving the frozen evaluation battery shares nothing with the exact
training corpus. These tests prove the producer computes overlap honestly
(exact, normalized, five-gram), signs exactly the bytes the campaign
verifies, refuses to claim external independence for in-repo keys, and
that the REAL campaign consumer accepts a clean produced audit and rejects
tampered ones.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.brain.llm.latent_cortex.frontier_tasks import (
    CURRENT_REGISTRY_VERSION,
    generate_task_battery,
)
from core.learning.resident_recurrent_sft_bootstrap_authority import (
    build_dataset_commitment,
)
from tools import produce_contamination_audit as audit_tool
from tools import run_latent_cortex_paired_campaign as campaign_runner

SEEDS = "424242"
DOMAINS = "mathematics,coding"
DIFFICULTY = 1


def _battery():
    return generate_task_battery(
        (424242,),
        domains=("mathematics", "coding"),
        difficulty=DIFFICULTY,
        registry_version=CURRENT_REGISTRY_VERSION,
    )


def _write_training_manifest(path: Path, prompts: list[str]) -> Path:
    payload = {"examples": [{"prompt": prompt, "answer": "FINAL_ANSWER: {}"} for prompt in prompts]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


CLEAN_PROMPTS = [
    "zylophant quibbernack drossle fenwick umbrage tallowdeep marrow",
    "crenellated obsidian marmalade if vexing quills unspool sideways",
    "porcelain thunder hums beneath the lacquered antipode of never",
]


def _keypair(tmp_path: Path) -> tuple[Path, Path]:
    key = tmp_path / "trust" / "private.pem"
    trust = tmp_path / "trust" / "public.pem"
    result = audit_tool.cmd_keygen(SimpleNamespace(key=str(key), trust_root=str(trust)))
    assert result == 0
    return key, trust


def _produce_args(tmp_path: Path, manifest: Path, key: Path, trust: Path):
    return SimpleNamespace(
        training_manifest=str(manifest),
        seeds=SEEDS,
        domains=DOMAINS,
        difficulty=DIFFICULTY,
        task_registry_version=CURRENT_REGISTRY_VERSION,
        corpus_name="test_corpus",
        key=str(key),
        trust_root=str(trust),
        out=str(tmp_path / "audit.json"),
        report_out=str(tmp_path / "report.json"),
    )


def test_keygen_refuses_overwrite(tmp_path: Path):
    key, trust = _keypair(tmp_path)
    assert key.exists() and trust.exists()
    assert (key.stat().st_mode & 0o777) == 0o600
    again = audit_tool.cmd_keygen(SimpleNamespace(key=str(key), trust_root=str(trust)))
    assert again == 1


def test_clean_corpus_produces_passing_signed_audit(tmp_path: Path):
    manifest = _write_training_manifest(tmp_path / "dataset.json", CLEAN_PROMPTS)
    key, trust = _keypair(tmp_path)
    args = _produce_args(tmp_path, manifest, key, trust)
    assert audit_tool.cmd_produce(args) == 0
    audit = json.loads((tmp_path / "audit.json").read_text())
    assert audit["status"] == "passed_zero_overlap"
    assert audit["overlap_count"] == 0
    assert audit["auditor_independence"] == "external"
    assert audit["signature"]["algorithm"] == "ed25519"
    # Standalone verification recomputes everything from raw inputs.
    verify_args = SimpleNamespace(
        audit=str(tmp_path / "audit.json"),
        training_manifest=str(manifest),
        seeds=SEEDS,
        domains=DOMAINS,
        difficulty=DIFFICULTY,
        task_registry_version=CURRENT_REGISTRY_VERSION,
        trust_root=str(trust),
    )
    assert audit_tool.cmd_verify(verify_args) == 0


def test_campaign_consumer_accepts_produced_audit(tmp_path: Path):
    """The REAL gate: run_latent_cortex_paired_campaign._contamination_audit
    verifies the produced artifact end to end (schema, manifest binding,
    signature) and marks it verified."""
    manifest = _write_training_manifest(tmp_path / "dataset.json", CLEAN_PROMPTS)
    key, trust = _keypair(tmp_path)
    args = _produce_args(tmp_path, manifest, key, trust)
    assert audit_tool.cmd_produce(args) == 0
    consumed = campaign_runner._contamination_audit(
        SimpleNamespace(
            contamination_audit=str(tmp_path / "audit.json"),
            contamination_trust_root=str(trust),
        ),
        _battery(),
    )
    assert consumed["status"] == "passed_zero_overlap"
    assert consumed["signature"]["verified"] is True


def test_role_v6_canonical_splits_bind_raw_snapshots_and_dataset_identity(
    tmp_path: Path,
):
    train_source = [
        {
            "task_id": "train-0",
            "family": "boolean",
            "depth": 2,
            "prompt": CLEAN_PROMPTS[0],
            "answer": "FINAL_ANSWER: true",
        }
    ]
    validation_source = [
        {
            "task_id": "validation-0",
            "family": "boolean",
            "depth": 2,
            "prompt": CLEAN_PROMPTS[1],
            "answer": "FINAL_ANSWER: false",
        }
    ]
    train = [{**train_source[0], "ordinal": 0}]
    validation = [{**validation_source[0], "ordinal": 0}]
    train_path = tmp_path / "train.json"
    validation_path = tmp_path / "validation.json"
    train_path.write_text(json.dumps(train, sort_keys=True), encoding="utf-8")
    validation_path.write_text(json.dumps(validation, sort_keys=True), encoding="utf-8")
    identity = build_dataset_commitment(train_source, validation_source)["dataset_sha256"]
    key, trust = _keypair(tmp_path)
    args = _produce_args(tmp_path, train_path, key, trust)
    args.validation_manifest = str(validation_path)
    args.training_dataset_identity_sha256 = identity
    assert audit_tool.cmd_produce(args) == 0
    audit = json.loads((tmp_path / "audit.json").read_text())
    assert audit["training_dataset_identity_sha256"] == identity
    assert len(audit["corpora"]) == 2
    assert {record["snapshot_sha256"] for record in audit["corpora"]} == {
        audit_tool._sha256_bytes(train_path.read_bytes()),
        audit_tool._sha256_bytes(validation_path.read_bytes()),
    }
    assert (
        campaign_runner._contamination_audit(
            SimpleNamespace(
                contamination_audit=str(tmp_path / "audit.json"),
                contamination_trust_root=str(trust),
            ),
            _battery(),
            expected_training_dataset_identity_sha256=identity,
        )["training_dataset_identity_sha256"]
        == identity
    )


def test_role_v6_dataset_identity_mismatch_is_rejected(tmp_path: Path):
    rows = [
        {
            "task_id": "train-0",
            "family": "boolean",
            "depth": 2,
            "prompt": CLEAN_PROMPTS[0],
            "answer": "FINAL_ANSWER: true",
            "ordinal": 0,
        }
    ]
    validation = [
        {
            "task_id": "validation-0",
            "family": "boolean",
            "depth": 2,
            "prompt": CLEAN_PROMPTS[1],
            "answer": "FINAL_ANSWER: false",
            "ordinal": 0,
        }
    ]
    train_path = tmp_path / "train.json"
    validation_path = tmp_path / "validation.json"
    train_path.write_text(json.dumps(rows), encoding="utf-8")
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    key, trust = _keypair(tmp_path)
    args = _produce_args(tmp_path, train_path, key, trust)
    args.validation_manifest = str(validation_path)
    args.training_dataset_identity_sha256 = "0" * 64
    with pytest.raises(audit_tool.AuditError, match="recomputed training dataset"):
        audit_tool.cmd_produce(args)


def test_exact_prompt_contamination_is_caught(tmp_path: Path):
    leaked = _battery()[0].public.prompt
    manifest = _write_training_manifest(tmp_path / "dataset.json", CLEAN_PROMPTS + [leaked])
    key, trust = _keypair(tmp_path)
    args = _produce_args(tmp_path, manifest, key, trust)
    assert audit_tool.cmd_produce(args) == 2  # overlap exit code
    audit = json.loads((tmp_path / "audit.json").read_text())
    assert audit["status"] == "failed_overlap"
    assert audit["overlap_count"] >= 1
    report = json.loads((tmp_path / "report.json").read_text())
    hit_methods = {method for hit in report[0]["hits"] for method in hit["methods"]}
    assert "exact_prompt" in hit_methods
    # The campaign consumer must refuse a failed audit.
    with pytest.raises(campaign_runner.CampaignProducerError):
        campaign_runner._contamination_audit(
            SimpleNamespace(
                contamination_audit=str(tmp_path / "audit.json"),
                contamination_trust_root=str(trust),
            ),
            _battery(),
        )


def test_paraphrase_contamination_caught_by_normalization_and_fivegrams(
    tmp_path: Path,
):
    leaked = _battery()[0].public.prompt
    # Case/punctuation mangling defeats exact matching but not the
    # normalized or five-gram sweeps.
    mangled = leaked.upper().replace(".", " .").replace(",", " ,")
    manifest = _write_training_manifest(tmp_path / "dataset.json", CLEAN_PROMPTS + [mangled])
    key, trust = _keypair(tmp_path)
    args = _produce_args(tmp_path, manifest, key, trust)
    assert audit_tool.cmd_produce(args) == 2
    report = json.loads((tmp_path / "report.json").read_text())
    hit_methods = {method for hit in report[0]["hits"] for method in hit["methods"]}
    assert "token_fivegram" in hit_methods


def test_tampered_audit_fails_signature_verification(tmp_path: Path):
    leaked = _battery()[0].public.prompt
    manifest = _write_training_manifest(tmp_path / "dataset.json", CLEAN_PROMPTS + [leaked])
    key, trust = _keypair(tmp_path)
    args = _produce_args(tmp_path, manifest, key, trust)
    assert audit_tool.cmd_produce(args) == 2
    audit_path = tmp_path / "audit.json"
    audit = json.loads(audit_path.read_text())
    audit["status"] = "passed_zero_overlap"  # forge the verdict
    audit["overlap_count"] = 0
    audit_path.write_text(json.dumps(audit))
    verify_args = SimpleNamespace(
        audit=str(audit_path),
        training_manifest=str(manifest),
        seeds=SEEDS,
        domains=DOMAINS,
        difficulty=DIFFICULTY,
        task_registry_version=CURRENT_REGISTRY_VERSION,
        trust_root=str(trust),
    )
    assert audit_tool.cmd_verify(verify_args) == 1
    with pytest.raises(campaign_runner.CampaignProducerError):
        campaign_runner._contamination_audit(
            SimpleNamespace(
                contamination_audit=str(audit_path),
                contamination_trust_root=str(trust),
            ),
            _battery(),
        )


def test_in_repo_key_cannot_claim_external_independence(tmp_path: Path):
    manifest = _write_training_manifest(tmp_path / "dataset.json", CLEAN_PROMPTS)
    repo_key_dir = audit_tool.REPO_ROOT / ".test_tmp_contamination_keys"
    key = repo_key_dir / "private.pem"
    trust = repo_key_dir / "public.pem"
    try:
        assert audit_tool.cmd_keygen(SimpleNamespace(key=str(key), trust_root=str(trust))) == 0
        args = _produce_args(tmp_path, manifest, key, trust)
        assert audit_tool.cmd_produce(args) == 3  # internal-independence exit
        audit = json.loads((tmp_path / "audit.json").read_text())
        assert audit["auditor_independence"] == "internal"
        with pytest.raises(campaign_runner.CampaignProducerError):
            campaign_runner._contamination_audit(
                SimpleNamespace(
                    contamination_audit=str(tmp_path / "audit.json"),
                    contamination_trust_root=str(trust),
                ),
                _battery(),
            )
    finally:
        for path in (key, trust):
            path.unlink(missing_ok=True)
        if repo_key_dir.exists():
            repo_key_dir.rmdir()


def test_wrong_trust_root_is_rejected_at_produce_time(tmp_path: Path):
    manifest = _write_training_manifest(tmp_path / "dataset.json", CLEAN_PROMPTS)
    key, _trust = _keypair(tmp_path)
    other_dir = tmp_path / "other"
    other_key = other_dir / "private.pem"
    other_trust = other_dir / "public.pem"
    assert (
        audit_tool.cmd_keygen(SimpleNamespace(key=str(other_key), trust_root=str(other_trust))) == 0
    )
    args = _produce_args(tmp_path, manifest, key, other_trust)
    with pytest.raises(audit_tool.AuditError, match="trust root"):
        audit_tool.cmd_produce(args)
