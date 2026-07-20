#!/usr/bin/env python3
"""Run the detached resident-v3 admission-to-proof continuation."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    canonical_json_bytes,
)
from core.runtime.atomic_writer import atomic_write_bytes  # noqa: E402
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402
from tools import build_resident_v3_pilot_contract as pilot_contract  # noqa: E402
from tools import prepare_latent_cortex_campaign as campaign_prepare  # noqa: E402
from tools import run_detached_step as detached  # noqa: E402
from tools import verify_resident_pilot_preflight as pilot_preflight  # noqa: E402
from tools import verify_resident_pilot_result as pilot_result  # noqa: E402
from tools import verify_resident_recurrence_mechanics as mechanics  # noqa: E402
from tools import verify_resident_v3_training_admission as admission  # noqa: E402

CONFIG_SCHEMA = "aura.resident_v3_post_training_pipeline_config.v1"
STATE_SCHEMA = "aura.resident_v3_post_training_pipeline_state.v1"
EVENT_SCHEMA = "aura.resident_v3_post_training_pipeline_event.v1"
VERDICT_SCHEMA = "aura.resident_v3_post_training_pipeline_verdict.v1"
ACTIVATION_SCHEMA = "aura.resident_v3_activation_candidate.v1"
_MAX_JSON_BYTES = 256 * 1024 * 1024
_PIPELINE_SOURCES = {
    "pipeline": "tools/run_resident_v3_post_training_pipeline.py",
    "training_admission": "tools/verify_resident_v3_training_admission.py",
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


class PipelineRun:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.root = Path(str(config["output_root"])).resolve(strict=False)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.journal_path = self.root / "pipeline_events.jsonl"
        self.state_path = self.root / "pipeline_state.json"
        self.verdict_path = self.root / "pipeline_verdict.json"
        self.stage = "initializing"
        self.event_sequence = 0
        self.event_head = ""
        self.started_at = time.time()
        self._restore_journal()

    def _restore_journal(self) -> None:
        if not self.journal_path.exists():
            return
        previous = ""
        sequence = 0
        started_at: float | None = None
        raw = read_stable_bytes(self.journal_path, max_bytes=_MAX_JSON_BYTES)
        for line in raw.splitlines():
            try:
                event = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ResidentV3PostTrainingPipelineError(
                    "pipeline_journal_invalid"
                ) from exc
            if not isinstance(event, dict):
                _fail("pipeline_journal_invalid")
            body = dict(event)
            claimed = body.pop("event_sha256", None)
            if (
                event.get("schema") != EVENT_SCHEMA
                or event.get("sequence") != sequence + 1
                or event.get("previous_event_sha256") != previous
                or claimed != _document_sha(body)
            ):
                _fail("pipeline_journal_invalid")
            sequence += 1
            previous = str(claimed)
            if started_at is None:
                started_at = float(event["recorded_at"])
        self.event_sequence = sequence
        self.event_head = previous
        if started_at is not None:
            self.started_at = started_at

    def event(self, status: str, details: Mapping[str, Any] | None = None) -> None:
        self.event_sequence += 1
        body = {
            "schema": EVENT_SCHEMA,
            "sequence": self.event_sequence,
            "stage": self.stage,
            "status": status,
            "recorded_at": time.time(),
            "previous_event_sha256": self.event_head,
            "details": dict(details or {}),
        }
        event = {**body, "event_sha256": _document_sha(body)}
        with self.journal_path.open("ab") as handle:
            handle.write(canonical_json_bytes(event) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.event_head = event["event_sha256"]
        self.state(status=status, details=details)

    def state(self, *, status: str, details: Mapping[str, Any] | None = None) -> None:
        material = {
            "schema": STATE_SCHEMA,
            "stage": self.stage,
            "status": status,
            "controller_pid": os.getpid(),
            "started_at": self.started_at,
            "updated_at": time.time(),
            "event_sequence": self.event_sequence,
            "event_head_sha256": self.event_head,
            "config_sha256": self.config["config_sha256"],
            "details": dict(details or {}),
        }
        atomic_write_bytes(
            self.state_path,
            canonical_json_bytes({**material, "state_sha256": _document_sha(material)}) + b"\n",
        )

    def set_stage(self, stage: str) -> None:
        self.stage = stage
        _verify_source_bindings(self.config)
        self.event("started")

    def wait_detached(
        self,
        run_dirs: Mapping[str, Path],
        *,
        timeout_s: float,
    ) -> dict[str, dict[str, Any]]:
        deadline = time.monotonic() + timeout_s
        while True:
            statuses = {role: detached._status(path) for role, path in run_dirs.items()}
            if all(status.get("terminal") is True for status in statuses.values()):
                return statuses
            if time.monotonic() >= deadline:
                _fail(f"detached_wait_timeout:{self.stage}")
            self.state(
                status="waiting",
                details={
                    role: {
                        "state": status.get("state"),
                        "heartbeat_sequence": status.get("heartbeat_sequence"),
                    }
                    for role, status in statuses.items()
                },
            )
            time.sleep(float(self.config["wait"]["poll_seconds"]))

    def run_json_command(
        self,
        command: list[str],
        *,
        log_name: str,
        allowed_returncodes: frozenset[int] = frozenset({0}),
    ) -> dict[str, Any]:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=3600.0,
            check=False,
        )
        log = {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        _write_once(self.root / log_name, log)
        if completed.returncode not in allowed_returncodes:
            _fail(f"command_failed:{log_name}:{completed.returncode}")
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ResidentV3PostTrainingPipelineError(f"command_output_invalid:{log_name}") from exc
        if not isinstance(value, dict):
            _fail(f"command_output_invalid:{log_name}")
        return value

    def campaign_command(
        self,
        spec: Mapping[str, Any],
        *,
        campaign_dir: Path,
        model: str,
        adapter_dir: Path,
        adapter_id: str,
        pilot: bool,
    ) -> list[str]:
        seeds = spec["seeds"] if pilot else [spec["seed"]]
        return [
            sys.executable,
            str(REPO_ROOT / "tools/run_latent_cortex_paired_campaign.py"),
            "--campaign-dir",
            str(campaign_dir),
            "--campaign-name",
            str(spec["campaign_name"]),
            "--model",
            model,
            "--adapter",
            str(adapter_dir),
            "--adapter-id",
            adapter_id,
            "--personality-adapter",
            "none",
            "--seeds",
            ",".join(str(seed) for seed in seeds),
            "--domains",
            ",".join(str(domain) for domain in spec["domains"]),
            "--difficulty",
            str(spec["difficulty"]),
            "--task-registry-version",
            pilot_preflight.CURRENT_REGISTRY_VERSION,
            "--profile",
            "primary",
            "--n-slots",
            str(spec["n_slots"]),
            "--branches",
            str(spec["branches"]),
            "--rlc-steps",
            str(spec["rlc_steps"]),
            "--rlc-profile",
            "recurrence_attribution",
            "--decode-max-tokens",
            str(spec["decode_max_tokens"]),
            "--episode-timeout",
            str(spec["episode_timeout_s"]),
            "--load-timeout",
            str(spec["load_timeout_s"]),
            "--warmup-timeout",
            str(spec["warmup_timeout_s"]),
            "--arm-timeout",
            str(spec["arm_timeout_s"]),
            "--campaign-timeout",
            str(spec["campaign_timeout_s"]),
            "--equal-compute-max-samples",
            str(spec["equal_compute_max_samples"]),
            "--max-infra-attempts",
            str(spec["max_infra_attempts"]),
        ]

    def execute_campaign(self, command: list[str], campaign_dir: Path, *, role: str) -> None:
        campaign_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        plan_result = self.run_json_command(
            [*command, "--plan-only"],
            log_name=f"{role}_plan_command.json",
        )
        policy = plan_result.get("detached_broker_policy")
        if not isinstance(policy, list) or len(policy) != 4:
            _fail(f"{role}_broker_policy_invalid")
        launch = [
            sys.executable,
            str(REPO_ROOT / "tools/run_detached_step.py"),
            "launch",
            "--run-dir",
            str(campaign_dir),
            "--name",
            str(command[command.index("--campaign-name") + 1]),
            "--cwd",
            str(REPO_ROOT),
            "--timeout",
            str(self.config[role]["detached_timeout_s"]),
            "--broker-policy-json",
            json.dumps(policy, sort_keys=True, separators=(",", ":")),
            "--",
            *command,
        ]
        self.run_json_command(launch, log_name=f"{role}_launch_command.json")
        status = self.wait_detached(
            {role: campaign_dir},
            timeout_s=float(self.config[role]["detached_timeout_s"]) + 300.0,
        )[role]
        receipt = status.get("receipt")
        if (
            status.get("completion_indeterminate") is not False
            or status.get("supervisor_alive") is not False
            or status.get("child_state") != "dead"
            or not isinstance(receipt, Mapping)
            or receipt.get("returncode") not in {0, 2}
            or receipt.get("containment_verified") is not True
            or receipt.get("process_group_empty") is not True
            or receipt.get("lineage_empty") is not True
            or receipt.get("restart_count") != 0
            or receipt.get("timed_out") is not False
        ):
            _fail(f"{role}_detached_evidence_invalid")
        self.event("completed", {"receipt_sha256": receipt.get("receipt_sha256")})

    def final_verdict(
        self,
        *,
        decision: str,
        failure_points: Sequence[str],
        pilot_verdict: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        material = {
            "schema": VERDICT_SCHEMA,
            "decision": decision,
            "pipeline_completed": decision
            in {
                "directional_gate_passed_external_frontier_proof_pending",
                "directional_gate_failed_frontier_gain_not_proven",
            },
            "failure_points": list(failure_points),
            "training_admitted": (self.root / "training_admission.json").exists(),
            "immutable_freeze_present": (self.root / "frozen-adapter" / "adapter_freeze.json").exists(),
            "mechanics_proven": (self.root / "mechanics_verdict.json").exists(),
            "directional_pilot_valid": pilot_verdict is not None,
            "directional_gain_gate_passed": bool(
                pilot_verdict and pilot_verdict.get("pilot_advance_gate_passed") is True
            ),
            "reasoning_gain_proven": False,
            "same_checkpoint_interaction_proven": False,
            "frontier_level_proven": False,
            "frontier_plus_proven": False,
            "external_attestation_present": False,
            "physical_weight_merge_performed": False,
            "runtime_activation_performed": False,
            "activation_candidate_path": str(self.root / "activation_candidate.json"),
            "required_next_gate": (
                "powered_external_frontier_campaign"
                if pilot_verdict and pilot_verdict.get("pilot_advance_gate_passed") is True
                else "repair_and_preregister_recurrence_v3_directional_pilot"
            ),
            "event_head_sha256": self.event_head,
            "event_count": self.event_sequence,
            "config_sha256": self.config["config_sha256"],
            "finished_at": time.time(),
        }
        verdict = {**material, "verdict_sha256": _document_sha(material)}
        _write_once(self.verdict_path, verdict)
        return verdict

    def run(self) -> dict[str, Any]:
        protocol_path = Path(str(self.config["protocol"]["path"]))
        amendment_path = Path(str(self.config["amendment"]["path"]))
        if _binding(protocol_path) != self.config["protocol"] or _binding(
            amendment_path
        ) != self.config["amendment"]:
            _fail("training_protocol_binding_changed")
        protocol = json.loads(read_stable_bytes(protocol_path, max_bytes=_MAX_JSON_BYTES))
        amendment = json.loads(read_stable_bytes(amendment_path, max_bytes=_MAX_JSON_BYTES))
        self.set_stage("wait_for_training_terminal")
        terminal = self.wait_detached(
            {
                "resume": Path(self.config["training_runs"]["resume"]),
                "resume_sentinel": Path(self.config["training_runs"]["resume_sentinel"]),
            },
            timeout_s=float(self.config["wait"]["terminal_timeout_seconds"]),
        )
        self.event(
            "completed",
            {
                role: status.get("receipt", {}).get("receipt_sha256")
                for role, status in terminal.items()
            },
        )

        self.set_stage("training_admission")
        admission_path = self.root / "training_admission.json"
        admitted = admission.verify(
            SimpleNamespace(
                protocol=protocol_path,
                amendment=amendment_path,
                partial_sentinel_run_dir=Path(
                    self.config["training_runs"]["partial_sentinel"]
                ),
                partial_footprint_ring=Path(amendment["sentinel"]["partial_ring"]),
                resume_sentinel_run_dir=Path(
                    self.config["training_runs"]["resume_sentinel"]
                ),
                resume_footprint_ring=Path(amendment["sentinel"]["resume_ring"]),
                output=admission_path,
                training_source_root=Path(str(self.config["training_source_root"])),
            )
        )
        if admitted.get("claim_flags", {}).get("adapter_freeze_eligible") is not True:
            _fail("training_incomplete_not_freeze_eligible")
        self.event("completed", {"admission_sha256": admitted["admission_sha256"]})

        self.set_stage("immutable_adapter_freeze")
        frozen = self.root / "frozen-adapter"
        freeze_result = campaign_prepare.freeze_adapter(
            SimpleNamespace(
                source_adapter=Path(protocol["training"]["output_dir"]),
                destination=frozen,
                model=Path(protocol["model"]["path"]),
                adapter_id=protocol["training"]["adapter_id"],
                personality_adapter="none",
            )
        )
        freeze = campaign_prepare.verify_adapter_freeze(frozen)
        activation_material = {
            "schema": ACTIVATION_SCHEMA,
            "status": "sealed_pending_proof_gates",
            "adapter_path": str(frozen),
            "adapter_id": freeze["adapter_id"],
            "adapter_sha256": freeze["identity_receipt"]["adapter_sha256"],
            "adapter_identity_sha256": freeze["identity_receipt"][
                "composite_identity_sha256"
            ],
            "freeze_certificate_sha256": freeze["certificate_sha256"],
            "physical_weight_merge_allowed": False,
            "runtime_activation_allowed": False,
            "remaining_gates": [
                "resident_32b_mechanics",
                "directional_nonregression_pilot",
                "powered_external_frontier_campaign",
            ],
        }
        _write_once(
            self.root / "activation_candidate.json",
            {**activation_material, "candidate_sha256": _document_sha(activation_material)},
        )
        self.event("completed", freeze_result)

        model = str(protocol["model"]["path"])
        adapter_id = str(protocol["training"]["adapter_id"])
        self.set_stage("mechanics")
        mechanics_dir = self.root / "mechanics-campaign"
        mechanics_command = self.campaign_command(
            self.config["mechanics"],
            campaign_dir=mechanics_dir,
            model=model,
            adapter_dir=frozen,
            adapter_id=adapter_id,
            pilot=False,
        )
        self.execute_campaign(mechanics_command, mechanics_dir, role="mechanics")
        mechanics_verdict = mechanics.verify(
            promotion_path=admission_path,
            frozen_adapter=frozen,
            campaign_dir=mechanics_dir,
        )
        _write_once(self.root / "mechanics_verdict.json", mechanics_verdict)
        self.event("verified", {"verdict_sha256": mechanics_verdict["verdict_sha256"]})

        self.set_stage("pilot_preregistration")
        pilot_dir = self.root / "directional-pilot"
        pilot_command = self.campaign_command(
            self.config["pilot"],
            campaign_dir=pilot_dir,
            model=model,
            adapter_dir=frozen,
            adapter_id=adapter_id,
            pilot=True,
        )
        pilot_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.run_json_command(
            [*pilot_command, "--plan-only"],
            log_name="pilot_preregistration_plan_command.json",
        )
        contract_path = self.root / "pilot_contract.json"
        contract = pilot_contract.build_contract(
            mechanics_path=self.root / "mechanics_verdict.json",
            campaign_dir=pilot_dir,
            seeds=self.config["pilot"]["seeds"],
            contract_id="cp190-resident-32b-v3-directional-pilot",
            created_at=str(self.config["created_at"]),
            source_commit=str(self.config["source_commit"]),
            personality_adapter="none",
        )
        _write_once(contract_path, contract)
        preflight = pilot_preflight.verify_preflight(
            contract_path=contract_path,
            mechanics_path=self.root / "mechanics_verdict.json",
            campaign_dir=pilot_dir,
        )
        _write_once(self.root / "pilot_preflight.json", preflight)
        self.event("completed", {"contract_sha256": contract["contract_sha256"]})

        self.set_stage("pilot_execution")
        self.execute_campaign(pilot_command, pilot_dir, role="pilot")
        result = pilot_result.verify(
            contract_path=contract_path,
            mechanics_path=self.root / "mechanics_verdict.json",
            campaign_dir=pilot_dir,
        )
        _write_once(self.root / "pilot_result.json", result)
        self.event("completed", {"verdict_sha256": result["verdict_sha256"]})
        if result["pilot_advance_gate_passed"] is True:
            return self.final_verdict(
                decision="directional_gate_passed_external_frontier_proof_pending",
                failure_points=["external_trust_roots_and_custodied_roles_not_present"],
                pilot_verdict=result,
            )
        failed_rules = [
            name for name, passed in result["advance_rules"].items() if passed is not True
        ]
        return self.final_verdict(
            decision="directional_gate_failed_frontier_gain_not_proven",
            failure_points=[*failed_rules, *result.get("diagnoses", [])],
            pilot_verdict=result,
        )


def run_pipeline(config_path: Path) -> int:
    config = _read_config(config_path)
    root = Path(str(config["output_root"])).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = root / "controller.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        run = PipelineRun(config)
        if run.verdict_path.exists():
            return 0
        try:
            verdict = run.run()
        except Exception as exc:  # noqa: BLE001 - durable fail-closed boundary
            trace = traceback.format_exc()
            run.event(
                "failed",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_sha256": _sha(trace.encode("utf-8")),
                },
            )
            _write_once(
                root / "failure_report.json",
                {
                    "schema": "aura.resident_v3_post_training_pipeline_failure.v1",
                    "stage": run.stage,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": trace,
                },
            )
            run.final_verdict(
                decision="proof_pipeline_failed",
                failure_points=[f"{run.stage}:{type(exc).__name__}:{exc}"],
            )
            return 1
        print(json.dumps(verdict, sort_keys=True))
        return 0


def launch_pipeline(config_path: Path, source_root: Path) -> dict[str, Any]:
    config = _read_config(config_path)
    root = Path(str(config["output_root"])).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    launch_path = root / "controller_launch.json"
    if launch_path.exists():
        return json.loads(read_stable_bytes(launch_path, max_bytes=_MAX_JSON_BYTES))
    source = source_root.expanduser().resolve(strict=True)
    command = [
        str(source / ".venv/bin/python"),
        str(source / "tools/run_resident_v3_post_training_pipeline.py"),
        "run",
        "--config",
        str(config_path.expanduser().resolve(strict=True)),
    ]
    log_path = root / "controller.log"
    log = log_path.open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            command,
            cwd=source,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log.close()
    time.sleep(1.0)
    if process.poll() is not None:
        _fail("controller_failed_during_launch")
    material = {
        "schema": "aura.resident_v3_post_training_pipeline_launch.v1",
        "command": command,
        "controller_pid": process.pid,
        "launched_at": time.time(),
        "config_sha256": config["config_sha256"],
        "source_root": str(source),
        "source_commit": config["source_commit"],
        "log_path": str(log_path),
    }
    receipt = {**material, "launch_sha256": _document_sha(material)}
    _write_once(launch_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create-config")
    create.add_argument("--protocol", type=Path, required=True)
    create.add_argument("--amendment", type=Path, required=True)
    create.add_argument("--output-root", type=Path, required=True)
    create.add_argument("--training-source-root", type=Path, required=True)
    create.add_argument("--source-commit", required=True)
    create.add_argument("--seeds", default="")
    create.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    launch = commands.add_parser("launch")
    launch.add_argument("--config", type=Path, required=True)
    launch.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "create-config":
            seeds = [int(value) for value in args.seeds.split(",") if value] or None
            config = build_config(
                protocol_path=args.protocol,
                amendment_path=args.amendment,
                output_root=args.output_root,
                training_source_root=args.training_source_root,
                source_commit=args.source_commit,
                seeds=seeds,
            )
            _write_once(args.output.expanduser().resolve(strict=False), config)
            print(json.dumps(config, indent=2, sort_keys=True))
            return 0
        if args.command == "launch":
            print(
                json.dumps(
                    launch_pipeline(args.config, args.source_root),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        return run_pipeline(args.config)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(
            f"run_resident_v3_post_training_pipeline: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
