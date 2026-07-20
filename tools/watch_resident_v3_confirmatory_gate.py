#!/usr/bin/env python3
"""Wait for the resident-v3 pilot and seal its confirmatory disposition."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Never

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes  # noqa: E402
from core.brain.llm.latent_cortex.campaign_launch_bundle import (  # noqa: E402
    verify_adapter_freeze,
)
from core.brain.llm.latent_cortex.exact_paired_grade import (  # noqa: E402
    exact_campaign_power_plan,
)
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (  # noqa: E402
    strict_json_loads,
)
from core.runtime.atomic_writer import atomic_write_bytes  # noqa: E402
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402

CONFIG_SCHEMA = "aura.resident_v3_confirmatory_watcher_config.v1"
STATE_SCHEMA = "aura.resident_v3_confirmatory_watcher_state.v1"
HANDOFF_SCHEMA = "aura.resident_v3_external_confirmatory_handoff.v1"
VERDICT_SCHEMA = "aura.resident_v3_confirmatory_watcher_verdict.v1"
PIPELINE_CONFIG_SCHEMA = "aura.resident_v3_post_training_pipeline_config.v2"
PIPELINE_VERDICT_SCHEMA = "aura.resident_v3_post_training_pipeline_verdict.v1"
_MAX_BYTES = 512 * 1024 * 1024


class ConfirmatoryWatcherError(RuntimeError):
    """Stable fail-closed confirmatory-watcher error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise ConfirmatoryWatcherError(code)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _document_sha(value: Mapping[str, Any]) -> str:
    return _sha(canonical_json_bytes(dict(value)))


def _binding(path: Path) -> dict[str, Any]:
    supplied = path.expanduser()
    if not supplied.is_absolute() or supplied.is_symlink():
        _fail("binding_path_invalid")
    resolved = supplied.resolve(strict=True)
    if resolved != supplied or not resolved.is_file():
        _fail("binding_path_invalid")
    raw = read_stable_bytes(resolved, max_bytes=_MAX_BYTES)
    return {"path": str(resolved), "sha256": _sha(raw), "size_bytes": len(raw)}


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    binding = _binding(path)
    try:
        value = strict_json_loads(
            read_stable_bytes(Path(binding["path"]), max_bytes=_MAX_BYTES),
            role=role,
        )
    except ValueError as exc:
        raise ConfirmatoryWatcherError(f"{role}_invalid") from exc
    if not isinstance(value, dict):
        _fail(f"{role}_invalid")
    return value


def _verified_document(
    path: Path,
    *,
    role: str,
    schema: str,
    hash_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = _binding(path)
    value = _read_json(path, role=role)
    material = dict(value)
    claimed = material.pop(hash_key, None)
    if value.get("schema") != schema or claimed != _document_sha(material):
        _fail(f"{role}_invalid")
    return value, binding


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(dict(value)) + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_symlink() or read_stable_bytes(path, max_bytes=_MAX_BYTES) != raw:
            _fail(f"output_exists_different:{path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_write_bytes(path, raw, mode=0o600)


def _power_plan() -> dict[str, Any]:
    result = exact_campaign_power_plan(
        domain_count=7,
        comparison_count=6,
        arm_count=6,
        planned_observations_per_domain=411,
    )
    if (
        result.get("certified") is not True
        or result.get("powered_for_zero_loss_noninferiority") is not True
        or result.get("minimum_observations") != 411
        or result.get("planned_total_tasks") != 2_877
        or result.get("planned_total_cells") != 17_262
    ):
        _fail("exact_power_contract_invalid")
    return result


def build_config(
    *,
    pipeline_config_path: Path,
    output_root: Path,
    source_commit: str,
) -> dict[str, Any]:
    pipeline_binding = _binding(pipeline_config_path)
    pipeline = _read_json(pipeline_config_path, role="pipeline_config")
    if (
        pipeline.get("schema") != PIPELINE_CONFIG_SCHEMA
        or pipeline.get("config_sha256")
        != _document_sha({k: v for k, v in pipeline.items() if k != "config_sha256"})
        or not isinstance(pipeline.get("recovery"), Mapping)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        _fail("pipeline_config_invalid")
    sources = {
        role: _binding(ROOT / relative)
        for role, relative in {
            "watcher": "tools/watch_resident_v3_confirmatory_gate.py",
            "exact_statistics": "core/brain/llm/latent_cortex/exact_paired_grade.py",
            "campaign_preparation": "tools/prepare_latent_cortex_campaign.py",
            "campaign_trust": "core/brain/llm/latent_cortex/campaign_trust.py",
            "campaign_runner": "tools/run_latent_cortex_paired_campaign.py",
            "frontier_verifier": "tools/verify_latent_cortex_frontier.py",
        }.items()
    }
    material = {
        "schema": CONFIG_SCHEMA,
        "created_at": time.time(),
        "source_commit": source_commit,
        "source_bindings": sources,
        "pipeline_config": pipeline_binding,
        "pipeline_root": str(Path(str(pipeline["output_root"])).resolve(strict=True)),
        "output_root": str(output_root.expanduser().resolve(strict=False)),
        "wait": {"poll_seconds": 30.0, "timeout_seconds": 604800.0},
        "confirmatory_design": {
            "domains": [
                "novel_algorithms",
                "mathematics",
                "coding",
                "scientific_inference",
                "long_horizon_planning",
                "calibration",
                "misleading_premise",
            ],
            "arms": [
                "base_vanilla",
                "base_rlc",
                "adapter_vanilla",
                "adapter_rlc",
                "base_equal_compute",
                "adapter_equal_compute",
            ],
            "exact_power": _power_plan(),
            "exact_power_scope": "zero_loss_noninferiority_floor_only",
            "positive_interaction_power_simulation_required": True,
            "custodied_roles": [
                "policy_root",
                "task_issuer",
                "campaign_runner",
                "independent_verifier",
            ],
        },
    }
    return {**material, "config_sha256": _document_sha(material)}


def _read_config(path: Path) -> dict[str, Any]:
    config = _read_json(path, role="watcher_config")
    material = dict(config)
    claimed = material.pop("config_sha256", None)
    if config.get("schema") != CONFIG_SCHEMA or claimed != _document_sha(material):
        _fail("watcher_config_invalid")
    pipeline_binding = config.get("pipeline_config")
    sources = config.get("source_bindings")
    if not isinstance(pipeline_binding, Mapping) or _binding(
        Path(str(pipeline_binding.get("path", "")))
    ) != dict(pipeline_binding):
        _fail("pipeline_config_binding_changed")
    if not isinstance(sources, Mapping) or any(
        not isinstance(binding, Mapping)
        or _binding(Path(str(binding.get("path", "")))) != dict(binding)
        for binding in sources.values()
    ):
        _fail("watcher_source_binding_changed")
    return config


def _artifact_set(pipeline_root: Path) -> dict[str, dict[str, Any]]:
    documents = {
        "activation_candidate": ("activation_candidate.json", "aura.resident_v3_activation_candidate.v1", "candidate_sha256"),
        "mechanics_verdict": ("mechanics_verdict.json", "aura.latent_cortex.resident_recurrence_mechanics.v1", "verdict_sha256"),
        "pilot_contract": ("pilot_contract.json", "aura.latent_cortex.resident_pilot_contract.v1", "contract_sha256"),
        "pilot_preflight": ("pilot_preflight.json", "aura.latent_cortex.resident_pilot_preflight.v1", "preflight_sha256"),
        "pilot_result": ("pilot_result.json", "aura.latent_cortex.resident_pilot_result.v1", "verdict_sha256"),
    }
    result: dict[str, dict[str, Any]] = {}
    for role, (name, schema, hash_key) in documents.items():
        document, binding = _verified_document(
            pipeline_root / name,
            role=role,
            schema=schema,
            hash_key=hash_key,
        )
        result[role] = {"binding": binding, "document": document}
    frozen = pipeline_root / "frozen-adapter"
    freeze = verify_adapter_freeze(frozen)
    result["frozen_adapter"] = {
        "path": str(frozen.resolve(strict=True)),
        "certificate_sha256": freeze["certificate_sha256"],
        "content_root_sha256": freeze["content_root_sha256"],
        "identity_receipt": freeze["identity_receipt"],
    }
    return result


def evaluate(
    config: Mapping[str, Any],
    pipeline_config: Mapping[str, Any],
    pipeline_verdict: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if (
        pipeline_verdict.get("schema") != PIPELINE_VERDICT_SCHEMA
        or pipeline_verdict.get("config_sha256") != pipeline_config.get("config_sha256")
    ):
        _fail("pipeline_verdict_binding_invalid")
    material = dict(pipeline_verdict)
    claimed = material.pop("verdict_sha256", None)
    if claimed != _document_sha(material):
        _fail("pipeline_verdict_invalid")
    decision = pipeline_verdict.get("decision")
    common = {
        "schema": VERDICT_SCHEMA,
        "pipeline_verdict_sha256": claimed,
        "watcher_config_sha256": config["config_sha256"],
        "reasoning_gain_proven": False,
        "same_checkpoint_interaction_proven": False,
        "frontier_level_proven": False,
        "frontier_plus_proven": False,
        "external_attestation_present": False,
    }
    if decision != "directional_gate_passed_external_frontier_proof_pending":
        result_material = {
            **common,
            "decision": "directional_gate_not_admitted_to_confirmatory_campaign",
            "failure_points": list(pipeline_verdict.get("failure_points") or [str(decision)]),
            "required_next_gate": pipeline_verdict.get("required_next_gate"),
            "finished_at": time.time(),
        }
        return {
            **result_material,
            "verdict_sha256": _document_sha(result_material),
        }, None
    if (
        pipeline_verdict.get("pipeline_completed") is not True
        or pipeline_verdict.get("directional_gain_gate_passed") is not True
        or pipeline_verdict.get("reasoning_gain_proven") is not False
        or pipeline_verdict.get("frontier_level_proven") is not False
        or pipeline_verdict.get("external_attestation_present") is not False
    ):
        _fail("positive_pipeline_verdict_invalid")
    pipeline_root = Path(str(config["pipeline_root"])).resolve(strict=True)
    artifacts = _artifact_set(pipeline_root)
    pilot = artifacts["pilot_result"]["document"]
    if (
        pilot.get("pilot_advance_gate_passed") is not True
        or pilot.get("decision") != "advance_to_powered_external_frontier_campaign"
        or pilot.get("reasoning_gain_proven") is not False
        or pilot.get("frontier_gain_proven") is not False
    ):
        _fail("pilot_result_not_confirmatory_eligible")
    handoff_material = {
        "schema": HANDOFF_SCHEMA,
        "status": "external_custody_required",
        "pipeline_config": dict(config["pipeline_config"]),
        "pipeline_verdict_sha256": claimed,
        "source_commit": config["source_commit"],
        "confirmatory_design": config["confirmatory_design"],
        "evidence": {
            role: value["binding"] if "binding" in value else value
            for role, value in artifacts.items()
        },
        "required_inputs": [
            "externally_signed_revisioned_campaign_policy",
            "distinct_task_issuer_and_campaign_runner_attestations",
            "preregistered_positive_interaction_power_simulation",
            "post_freeze_hidden_task_commitment_and_zero_overlap_audit",
            "post_evidence_independent_verifier_attestation",
        ],
        "claim_authorized": False,
        "created_at": time.time(),
    }
    handoff = {
        **handoff_material,
        "handoff_sha256": _document_sha(handoff_material),
    }
    verdict_material = {
        **common,
        "decision": "external_custody_required",
        "failure_points": ["separately_administered_trust_inputs_not_present"],
        "required_next_gate": "powered_external_frontier_campaign",
        "handoff_sha256": handoff["handoff_sha256"],
        "finished_at": time.time(),
    }
    verdict = {**verdict_material, "verdict_sha256": _document_sha(verdict_material)}
    return verdict, handoff


def _state(path: Path, config: Mapping[str, Any], *, status: str, details: Mapping[str, Any]) -> None:
    material = {
        "schema": STATE_SCHEMA,
        "status": status,
        "watcher_pid": os.getpid(),
        "config_sha256": config["config_sha256"],
        "details": dict(details),
        "updated_at": time.time(),
    }
    atomic_write_bytes(
        path,
        canonical_json_bytes({**material, "state_sha256": _document_sha(material)}) + b"\n",
        mode=0o600,
    )


def run(config_path: Path) -> int:
    config = _read_config(config_path)
    root = Path(str(config["output_root"])).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / "watcher.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        verdict_path = root / "confirmatory_watcher_verdict.json"
        if verdict_path.exists():
            return 0
        pipeline_root = Path(str(config["pipeline_root"])).resolve(strict=True)
        pipeline_verdict_path = pipeline_root / "pipeline_verdict.json"
        deadline = time.monotonic() + float(config["wait"]["timeout_seconds"])
        while not pipeline_verdict_path.exists():
            if time.monotonic() >= deadline:
                _fail("pipeline_verdict_wait_timeout")
            _state(
                root / "watcher_state.json",
                config,
                status="waiting_for_directional_verdict",
                details={"pipeline_verdict_path": str(pipeline_verdict_path)},
            )
            time.sleep(float(config["wait"]["poll_seconds"]))
        pipeline_config = _read_json(
            Path(str(config["pipeline_config"]["path"])),
            role="pipeline_config",
        )
        pipeline_verdict = _read_json(pipeline_verdict_path, role="pipeline_verdict")
        verdict, handoff = evaluate(config, pipeline_config, pipeline_verdict)
        if handoff is not None:
            _write_once(root / "external_confirmatory_handoff.json", handoff)
        _write_once(verdict_path, verdict)
        _state(
            root / "watcher_state.json",
            config,
            status="complete",
            details={"decision": verdict["decision"], "verdict_sha256": verdict["verdict_sha256"]},
        )
        return 0


def launch(config_path: Path) -> dict[str, Any]:
    config = _read_config(config_path)
    root = Path(str(config["output_root"])).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    launch_path = root / "watcher_launch.json"
    if launch_path.exists():
        return _read_json(launch_path, role="watcher_launch")
    command = [
        "/usr/bin/caffeinate",
        "-i",
        str(ROOT / ".venv/bin/python"),
        str(Path(__file__).resolve()),
        "run",
        "--config",
        str(config_path.expanduser().resolve(strict=True)),
    ]
    log_path = root / "watcher.log"
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    time.sleep(1.0)
    if process.poll() is not None:
        _fail("watcher_failed_during_launch")
    material = {
        "schema": "aura.resident_v3_confirmatory_watcher_launch.v1",
        "command": command,
        "watcher_pid": process.pid,
        "config_sha256": config["config_sha256"],
        "launched_at": time.time(),
        "log_path": str(log_path),
    }
    receipt = {**material, "launch_sha256": _document_sha(material)}
    _write_once(launch_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create-config")
    create.add_argument("--pipeline-config", type=Path, required=True)
    create.add_argument("--output-root", type=Path, required=True)
    create.add_argument("--source-commit", required=True)
    create.add_argument("--output", type=Path, required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    launch_parser = commands.add_parser("launch")
    launch_parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "create-config":
            config = build_config(
                pipeline_config_path=args.pipeline_config,
                output_root=args.output_root,
                source_commit=args.source_commit,
            )
            _write_once(args.output.expanduser().resolve(strict=False), config)
            print(json.dumps(config, indent=2, sort_keys=True))
            return 0
        if args.command == "launch":
            print(json.dumps(launch(args.config), indent=2, sort_keys=True))
            return 0
        return run(args.config)
    except Exception as exc:  # noqa: BLE001 - durable fail-closed CLI boundary
        print(
            f"watch_resident_v3_confirmatory_gate: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
