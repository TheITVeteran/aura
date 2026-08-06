"""Immutable configuration and source-custody contracts for resident-v3 proof."""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Never

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.runtime.atomic_writer import atomic_write_bytes
from core.runtime.file_read_gateway import read_stable_bytes
from tools import verify_resident_pilot_preflight as pilot_preflight
from tools import verify_resident_v3_recovery_training_admission as recovery_admission

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_SCHEMA = "aura.resident_v3_post_training_pipeline_config.v2"
_MAX_JSON_BYTES = 256 * 1024 * 1024
_PIPELINE_SOURCES = {
    "pipeline": "tools/run_resident_v3_post_training_pipeline.py",
    "pipeline_contracts": "tools/resident_v3_post_training_contracts.py",
    "training_admission": "tools/verify_resident_v3_training_admission.py",
    "recovery_training_admission": "tools/verify_resident_v3_recovery_training_admission.py",
    "campaign_preparation": "tools/prepare_latent_cortex_campaign.py",
    "mechanics_verifier": "tools/verify_resident_recurrence_mechanics.py",
    "pilot_contract_builder": "tools/build_resident_v3_pilot_contract.py",
    "pilot_preflight": "tools/verify_resident_pilot_preflight.py",
    "pilot_result": "tools/verify_resident_pilot_result.py",
    "campaign_runner": "tools/run_latent_cortex_paired_campaign.py",
    "detached_supervisor": "tools/run_detached_step.py",
    "campaign_replay": "tools/verify_recurrence_v2_smoke.py",
    "campaign_evidence": "tools/verify_paired_campaign_evidence.py",
    "independent_scoring": "tools/independent_paired_campaign_scoring.py",
    "frontier_tasks": "core/brain/llm/latent_cortex/frontier_tasks.py",
    "freeze_contract": "core/brain/llm/latent_cortex/campaign_launch_bundle.py",
    "adapter_identity": "core/brain/llm/latent_cortex/recurrence_adapter_identity_v2.py",
}


class ResidentV3PostTrainingPipelineError(RuntimeError):
    """Stable fail-closed continuation error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise ResidentV3PostTrainingPipelineError(code)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _document_sha(value: Mapping[str, Any]) -> str:
    return _sha(canonical_json_bytes(dict(value)))


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    raw = read_stable_bytes(resolved, max_bytes=_MAX_JSON_BYTES)
    return {"path": str(resolved), "sha256": _sha(raw), "size_bytes": len(raw)}


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(dict(value)) + b"\n"
    if path.exists():
        if path.is_symlink() or read_stable_bytes(path, max_bytes=_MAX_JSON_BYTES) != raw:
            _fail(f"output_exists_different:{path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_write_bytes(path, raw)


def _read_config(path: Path) -> dict[str, Any]:
    raw = read_stable_bytes(path.expanduser().resolve(strict=True), max_bytes=_MAX_JSON_BYTES)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResidentV3PostTrainingPipelineError("config_invalid") from exc
    if not isinstance(value, dict):
        _fail("config_invalid")
    claimed = value.get("config_sha256")
    material = dict(value)
    material.pop("config_sha256", None)
    if value.get("schema") != CONFIG_SCHEMA or claimed != _document_sha(material):
        _fail("config_invalid")
    return value


def _source_bindings(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for role, relative in _PIPELINE_SOURCES.items():
        raw = read_stable_bytes(root / relative, max_bytes=_MAX_JSON_BYTES)
        result[role] = {"path": relative, "sha256": _sha(raw), "size_bytes": len(raw)}
    return result


def _verify_source_bindings(config: Mapping[str, Any]) -> None:
    expected = config.get("source_bindings")
    if not isinstance(expected, Mapping) or dict(expected) != _source_bindings(REPO_ROOT):
        _fail("pipeline_source_identity_changed")


def _random_seeds() -> list[int]:
    first = (1 << 62) | secrets.randbits(62)
    second = first
    while second == first:
        second = (1 << 62) | secrets.randbits(62)
    return [first, second]


def build_config(
    *,
    protocol_path: Path,
    amendment_path: Path,
    output_root: Path,
    training_source_root: Path,
    source_commit: str,
    seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    protocol = _binding(protocol_path)
    amendment = _binding(amendment_path)
    protocol_document = json.loads(
        read_stable_bytes(Path(protocol["path"]), max_bytes=_MAX_JSON_BYTES)
    )
    amendment_document = json.loads(
        read_stable_bytes(Path(amendment["path"]), max_bytes=_MAX_JSON_BYTES)
    )
    root = output_root.expanduser().resolve(strict=False)
    training_root = training_source_root.expanduser().resolve(strict=True)
    if (
        len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
        or not training_root.is_dir()
    ):
        _fail("config_arguments_invalid")
    pilot_seeds = list(seeds or _random_seeds())
    if (
        len(pilot_seeds) != 2
        or len(set(pilot_seeds)) != 2
        or any(type(seed) is not int or seed.bit_length() != 63 for seed in pilot_seeds)
    ):
        _fail("config_seed_contract_invalid")
    material = {
        "schema": CONFIG_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": source_commit,
        "source_bindings": _source_bindings(REPO_ROOT),
        "training_source_root": str(training_root),
        "protocol": protocol,
        "amendment": amendment,
        "training_runs": {
            "partial": protocol_document["detached_execution"]["partial_run_dir"],
            "partial_sentinel": str(
                Path(protocol_document["detached_execution"]["partial_run_dir"])
                .parent
                / "sentinel-partial"
            ),
            "resume": amendment_document["resume"]["run_dir"],
            "resume_sentinel": str(
                Path(amendment_document["resume"]["run_dir"]).parent
                / "sentinel-resume"
            ),
        },
        "output_root": str(root),
        "wait": {"poll_seconds": 15.0, "terminal_timeout_seconds": 172800.0},
        "mechanics": {
            "campaign_name": "cp190-resident-32b-v3-mechanics",
            "seed": 2026072001,
            "domains": ["mathematics"],
            "difficulty": 1,
            "n_slots": 4,
            "branches": 2,
            "rlc_steps": 4,
            "decode_max_tokens": 768,
            "episode_timeout_s": 900.0,
            "load_timeout_s": 900.0,
            "warmup_timeout_s": 600.0,
            "arm_timeout_s": 1800.0,
            "campaign_timeout_s": 9000.0,
            "equal_compute_max_samples": 4,
            "max_infra_attempts": 1,
            "detached_timeout_s": 10800.0,
        },
        "pilot": {
            "campaign_name": "cp190-resident-32b-v3-directional-pilot",
            "seeds": pilot_seeds,
            "domains": list(pilot_preflight.DOMAINS),
            "difficulty": 2,
            "n_slots": 4,
            "branches": 2,
            "rlc_steps": 4,
            "decode_max_tokens": 768,
            "episode_timeout_s": 1200.0,
            "load_timeout_s": 1200.0,
            "warmup_timeout_s": 600.0,
            "arm_timeout_s": 10800.0,
            "campaign_timeout_s": 43200.0,
            "equal_compute_max_samples": 8,
            "max_infra_attempts": 3,
            "detached_timeout_s": 46800.0,
        },
        "claim_policy": {
            "physical_weight_merge_allowed": False,
            "runtime_activation_before_mechanics_allowed": False,
            "runtime_activation_before_directional_gain_allowed": False,
            "frontier_claim_without_external_custody_allowed": False,
        },
    }
    return {**material, "config_sha256": _document_sha(material)}


def build_recovery_config(
    *,
    migration_path: Path,
    output_root: Path,
    training_source_root: Path,
    source_commit: str,
    seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    migration_binding = _binding(migration_path)
    migration = json.loads(
        read_stable_bytes(Path(migration_binding["path"]), max_bytes=_MAX_JSON_BYTES)
    )
    if not isinstance(migration, Mapping):
        _fail("recovery_migration_invalid")
    try:
        _verified, migration_summary = recovery_admission.recovery._migration(  # noqa: SLF001
            Path(migration_binding["path"]),
            allow_destination_pointer_advance=True,
        )
    except Exception as exc:
        raise ResidentV3PostTrainingPipelineError("recovery_migration_invalid") from exc
    protocol_binding = migration.get("protocol")
    amendment_binding = migration.get("amendment")
    destination = migration.get("destination")
    if not all(
        isinstance(value, Mapping)
        for value in (protocol_binding, amendment_binding, destination)
    ):
        _fail("recovery_migration_invalid")
    protocol_path = Path(str(protocol_binding.get("path", ""))).resolve(strict=True)
    amendment_path = Path(str(amendment_binding.get("path", ""))).resolve(strict=True)
    if (
        _binding(protocol_path) != dict(protocol_binding)
        or _binding(amendment_path) != dict(amendment_binding)
    ):
        _fail("recovery_scientific_contract_changed")
    adapter_root = Path(str(destination.get("root", ""))).resolve(strict=True)
    root = adapter_root.parent
    if not root.name.startswith("resident_32b_v3_cp"):
        _fail("recovery_destination_invalid")
    calibration_path = root / "calibration_verdict.json"
    base = build_config(
        protocol_path=protocol_path,
        amendment_path=amendment_path,
        output_root=output_root,
        training_source_root=training_source_root,
        source_commit=source_commit,
        seeds=seeds,
    )
    material = dict(base)
    material.pop("config_sha256", None)
    campaign_suffix = root.name.replace("resident_32b_v3_", "")
    material["training_runs"] = {
        "resume": str(root / "detached-resume"),
        "resume_sentinel": str(root / "sentinel-resume"),
        "recovery_controller": str(root / "detached-recovery-controller"),
        "sentinel_archive": str(root / "detached-sentinel-proof-archive"),
    }
    material["wait"] = {
        "poll_seconds": 15.0,
        "terminal_timeout_seconds": 216000.0,
    }
    material["mechanics"] = {
        **dict(material["mechanics"]),
        "campaign_name": f"{campaign_suffix}-resident-32b-v3-mechanics",
    }
    material["pilot"] = {
        **dict(material["pilot"]),
        "campaign_name": f"{campaign_suffix}-resident-32b-v3-directional-pilot",
    }
    material["recovery"] = {
        "mode": "migration_recovery",
        "migration": migration_binding,
        "migration_sha256": migration_summary["migration_sha256"],
        "calibration_verdict": _binding(calibration_path),
        "controller_verdict_path": str(root / "recovery_controller_verdict.json"),
        "operational_ring_path": str(
            adapter_root / f"physical_footprint_resume_{root.name}.jsonl"
        ),
        "archive_run_dir": str(root / "detached-sentinel-proof-archive"),
        "archive_ring_path": str(
            root / "sentinel-proof-archive/physical_footprint_resume_complete.jsonl"
        ),
        "archive_receipt_path": str(
            root / "sentinel-proof-archive/archive_receipt.json"
        ),
        "adapter_root": str(adapter_root),
    }
    return {**material, "config_sha256": _document_sha(material)}
