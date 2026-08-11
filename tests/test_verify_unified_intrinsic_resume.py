from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import tools.verify_unified_intrinsic_resume as resume_verifier
from tools.run_detached_step import validate_resume_verdict
from tools.unified_intrinsic_resident_identity import (
    campaign_checkpoint_binding,
    canonical_bytes,
    canonical_sha256,
    trainer_model_identity_from_manifest,
)
from tools.verify_unified_intrinsic_resume import verify_resume

PLAN_SHA = "1" * 64
COMMAND_SHA = "2" * 64
JOURNAL_SHA = "3" * 64


def _model_manifest() -> dict:
    files = [
        {"path": "config.json", "size_bytes": 10, "sha256": "a" * 64},
        {
            "path": "model-00001-of-00001.safetensors",
            "size_bytes": 20,
            "sha256": "b" * 64,
        },
        {"path": "tokenizer.json", "size_bytes": 30, "sha256": "c" * 64},
        {
            "path": "tokenizer_config.json",
            "size_bytes": 40,
            "sha256": "d" * 64,
        },
    ]
    body = {
        "schema": "aura.unified_intrinsic.model_manifest.v1",
        "root": "/model",
        "file_count": len(files),
        "files": files,
        "weights": ["model-00001-of-00001.safetensors"],
        "shard_index": None,
        "dimensions": {
            "model_type": "qwen2",
            "num_hidden_layers": 64,
            "hidden_size": 5120,
            "vocab_size": 152064,
            "quantization": {"bits": 4},
        },
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _campaign_config(tmp_path: Path) -> Path:
    output = tmp_path / "training-output"
    output.mkdir()
    output.chmod(0o700)
    body = {
        "schema": "aura.unified_intrinsic.resident_campaign.v1",
        "campaign_id": "unit-resume",
        "profile": "canary",
        "source": {
            "git": {"commit": "7" * 40, "tree": "8" * 40},
            "manifest": {"manifest_sha256": "9" * 64},
        },
        "runtime": {"identity_sha256": "e" * 64},
        "paths": {"training_output": str(output)},
        "dataset": {"identity_sha256": "4" * 64},
        "tokenizer": {"identity_sha256": "5" * 64},
        "tokenized_dataset": {"identity_sha256": "6" * 64},
        "model": _model_manifest(),
        "training": {"max_steps": 2},
        "training_args": ["--max-steps", "2"],
    }
    config = {**body, "config_sha256": canonical_sha256(body)}
    path = tmp_path / "campaign.json"
    path.write_bytes(canonical_bytes(config) + b"\n")
    path.chmod(0o400)
    return path


def _bind_resume_environment(monkeypatch) -> None:
    monkeypatch.setenv("AURA_DETACHED_RESUME_EVIDENCE_TRANSPORT", "stdout-v3")
    monkeypatch.setenv("AURA_DETACHED_PLAN_SHA256", PLAN_SHA)
    monkeypatch.setenv("AURA_DETACHED_COMMAND_SHA256", COMMAND_SHA)
    monkeypatch.setenv("AURA_DETACHED_PRIOR_ATTEMPT", "1")
    monkeypatch.setenv("AURA_DETACHED_PRIOR_JOURNAL_HEAD_SHA256", JOURNAL_SHA)


def _assert_supervisor_accepts(verdict: dict[str, object]) -> None:
    accepted = validate_resume_verdict(
        verdict,
        plan_sha256=PLAN_SHA,
        command_sha256=COMMAND_SHA,
        prior_attempt=1,
        prior_journal_head_sha256=JOURNAL_SHA,
    )
    assert accepted == verdict


def test_empty_campaign_is_safe_to_replay_from_zero(tmp_path, monkeypatch) -> None:
    config = _campaign_config(tmp_path)
    _bind_resume_environment(monkeypatch)

    verdict = verify_resume(config)

    assert verdict["verdict"] == "safe_to_resume"
    assert verdict["checkpoint_sequence"] == 0
    assert verdict["evidence"]["reason"] == "no_checkpoint_deterministic_replay"
    _assert_supervisor_accepts(verdict)


def test_torn_checkpoint_state_is_indeterminate(tmp_path, monkeypatch) -> None:
    config = _campaign_config(tmp_path)
    decoded = json.loads(config.read_text(encoding="ascii"))
    output = Path(decoded["paths"]["training_output"])
    (output / "checkpoint_latest.safetensors").write_bytes(b"not-authoritative")
    _bind_resume_environment(monkeypatch)

    verdict = verify_resume(config)

    assert verdict["verdict"] == "indeterminate"
    assert verdict["checkpoint_sequence"] == 0
    assert (
        verdict["evidence"]["reason"]
        == "checkpoint_artifacts_exist_without_authoritative_pointer"
    )
    _assert_supervisor_accepts(verdict)


def test_checkpoint_pointer_without_generation_is_indeterminate(
    tmp_path,
    monkeypatch,
) -> None:
    config = _campaign_config(tmp_path)
    decoded = json.loads(config.read_text(encoding="ascii"))
    output = Path(decoded["paths"]["training_output"])
    (output / "checkpoint_latest_pointer.json").write_bytes(b"{}\n")
    _bind_resume_environment(monkeypatch)

    verdict = verify_resume(config)

    assert verdict["verdict"] == "indeterminate"
    assert verdict["evidence"]["reason"] == "checkpoint_resolution_failed:UnifiedCheckpointError"
    _assert_supervisor_accepts(verdict)


def test_unpointed_first_generation_is_ignored_for_exact_step_zero_replay(
    tmp_path,
    monkeypatch,
) -> None:
    config = _campaign_config(tmp_path)
    decoded = json.loads(config.read_text(encoding="ascii"))
    output = Path(decoded["paths"]["training_output"])
    generations = output / "checkpoint_generations"
    generations.mkdir(mode=0o700)
    orphan = generations / f"checkpoint_latest-step-00000001-{'a' * 32}"
    orphan.mkdir(mode=0o500)
    _bind_resume_environment(monkeypatch)

    verdict = verify_resume(config)

    assert verdict["verdict"] == "safe_to_resume"
    assert verdict["checkpoint_sequence"] == 0
    assert verdict["evidence"]["reason"] == (
        "unpointed_first_generation_ignored_for_deterministic_replay"
    )
    assert verdict["evidence"]["unpointed_checkpoint_inventory"] == {
        "orphan_generations": 1,
        "staged_generations": 0,
    }
    _assert_supervisor_accepts(verdict)


def test_authoritative_checkpoint_survives_torn_summary_receipt(
    tmp_path,
    monkeypatch,
) -> None:
    config = _campaign_config(tmp_path)
    decoded = json.loads(config.read_text(encoding="ascii"))
    output = Path(decoded["paths"]["training_output"])
    (output / "training_receipt.json").write_bytes(b"{torn")
    checkpoint_receipt = {
        "identity": {
            "dataset": decoded["dataset"],
            "tokenizer": decoded["tokenizer"],
            "tokenized_dataset": decoded["tokenized_dataset"],
            "model": trainer_model_identity_from_manifest(decoded["model"]),
            "campaign_binding": campaign_checkpoint_binding(decoded),
        },
        "step": 4,
        "checkpoint_sha256": "8" * 64,
        "receipt_sha256": "9" * 64,
    }
    monkeypatch.setattr(
        resume_verifier,
        "resolve_checkpoint_generation",
        lambda *_args, **_kwargs: SimpleNamespace(receipt=checkpoint_receipt),
    )
    _bind_resume_environment(monkeypatch)

    verdict = verify_resume(config)

    assert verdict["verdict"] == "safe_to_resume"
    assert verdict["checkpoint_sequence"] == 4
    assert verdict["evidence"]["training_receipt"] == {
        "binding": "ignored_non_authoritative",
        "reason": "UnifiedResumeVerificationError",
    }
    _assert_supervisor_accepts(verdict)
