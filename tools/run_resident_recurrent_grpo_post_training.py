#!/usr/bin/env python3
"""Crash-consistent resident recurrent-GRPO train-to-proof coordinator.

The long trainer and the directional campaign are each owned by
``run_detached_step``.  This small launchd-managed controller provides the
durable stage machine between them: wait, admit, freeze, run the nonclaiming
six-arm factorial, independently replay it, and stop at the external-custody
boundary.  A directional result is evidence for deciding what to do next; it
is never promoted into a frontier or release claim by this controller.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import plistlib
import re
import secrets
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex import (  # noqa: E402
    recurrent_grpo_adapter_identity,
)
from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.frontier_tasks import (  # noqa: E402
    CURRENT_REGISTRY_VERSION,
    FRONTIER_DOMAINS,
)
from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_bytes_if_absent,
    ensure_private_directory,
)
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402
from tools import prepare_latent_cortex_campaign as campaign_prepare  # noqa: E402
from tools import prepare_resident_recurrent_grpo_campaign as prereg  # noqa: E402
from tools import run_detached_step as detached  # noqa: E402
from tools import run_latent_cortex_paired_campaign as campaign_runner  # noqa: E402
from tools import verify_paired_campaign_evidence as campaign_verifier  # noqa: E402

CONFIG_SCHEMA = "aura.resident_recurrent_grpo_post_training_config.v1"
STATE_SCHEMA = "aura.resident_recurrent_grpo_post_training_state.v1"
EVENT_SCHEMA = "aura.resident_recurrent_grpo_post_training_event.v1"
ADMISSION_SCHEMA = "aura.resident_recurrent_grpo_training_admission.v1"
VERDICT_SCHEMA = "aura.resident_recurrent_grpo_post_training_verdict.v1"
CUSTODY_REQUEST_SCHEMA = "aura.resident_recurrent_grpo_external_custody_request.v1"
LAUNCH_SCHEMA = "aura.resident_recurrent_grpo_launchd_receipt.v1"
SOURCE_RELATIVE = "tools/run_resident_recurrent_grpo_post_training.py"
DEFAULT_CONTRACT = Path(prereg.DEFAULT_CONTRACT)
DEFAULT_ROOT = Path(prereg.DEFAULT_ROOT)
DEFAULT_CONFIG = DEFAULT_ROOT / "post_training_config.json"
DEFAULT_CONTROLLER_ROOT = DEFAULT_ROOT / "post-training"
_MAX_JSON_BYTES = 256 * 1024 * 1024
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_LABEL = re.compile(r"^[a-z0-9][a-z0-9.-]{2,119}$")
_MODEL_OWNER_SCRIPTS = frozenset(
    {
        "aura_main.py",
        "train_grpo.py",
        "run_latent_cortex_paired_campaign.py",
        "recurrence_native_train.py",
        "recurrence_native_train_v2.py",
        "latent_cortex_lab.py",
    }
)
_MIN_TRAINING_AVAILABLE_BYTES = 40 * 1024**3
_MECHANISM_PROFILES = (
    "recurrence_attribution",
    "resident_full_stack_no_latent_opt",
    "resident_full_stack_no_fast_weights",
    "resident_full_stack_no_branch_exchange",
)


class PostTrainingError(RuntimeError):
    """Stable fail-closed controller error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise PostTrainingError(code)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _document_sha(value: Mapping[str, Any]) -> str:
    return _sha(canonical_json_bytes(dict(value)))


def _resolved(path: Path, *, must_exist: bool = True) -> Path:
    supplied = path.expanduser()
    if not supplied.is_absolute():
        supplied = REPO_ROOT / supplied
    if supplied.is_symlink():
        _fail("path_symlink_rejected")
    try:
        resolved = supplied.resolve(strict=must_exist)
        resolved.relative_to(REPO_ROOT)
    except (OSError, ValueError) as exc:
        raise PostTrainingError("path_outside_repository") from exc
    return resolved


def _python_launcher_path() -> Path:
    """Preserve virtualenv launchers instead of resolving to the base Python."""

    candidate = Path(sys.executable)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).absolute()
    if not candidate.exists() or not os.access(candidate, os.X_OK):
        _fail("python_launcher_missing")
    return candidate


def _binding(path: Path) -> dict[str, Any]:
    resolved = _resolved(path)
    if not resolved.is_file():
        _fail("binding_path_invalid")
    payload = read_stable_bytes(resolved, max_bytes=_MAX_JSON_BYTES)
    return {
        "path": str(resolved.relative_to(REPO_ROOT)),
        "sha256": _sha(payload),
        "size_bytes": len(payload),
    }


def _strict_json(path: Path) -> dict[str, Any]:
    resolved = _resolved(path)
    if not resolved.is_file():
        _fail("document_path_invalid")
    raw = read_stable_bytes(resolved, max_bytes=_MAX_JSON_BYTES)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                _fail("document_duplicate_key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: _fail("document_nonfinite"),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PostTrainingError("document_invalid") from exc
    if not isinstance(value, dict):
        _fail("document_invalid")
    return value


def _write_once(path: Path, document: Mapping[str, Any]) -> None:
    resolved = _resolved(path, must_exist=False)
    payload = canonical_json_bytes(dict(document)) + b"\n"
    ensure_private_directory(resolved.parent)
    if resolved.exists():
        if resolved.is_symlink() or resolved.read_bytes() != payload:
            _fail(f"immutable_output_differs:{resolved.name}")
        return
    if not atomic_write_bytes_if_absent(resolved, payload, mode=0o600):
        if resolved.is_symlink() or resolved.read_bytes() != payload:
            _fail(f"immutable_output_raced:{resolved.name}")


def _random_seeds(count: int) -> list[int]:
    values: set[int] = set()
    while len(values) < count:
        values.add((1 << 62) | secrets.randbits(62))
    return sorted(values)


def build_config(
    *,
    contract_path: Path,
    output_root: Path,
    source_commit: str,
    seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    contract = _strict_json(contract_path)
    prereg.validate_contract(contract, verify_model=True)
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        _fail("source_commit_invalid")
    directional_seeds = list(seeds or _random_seeds(8))
    if (
        len(directional_seeds) != 8
        or len(set(directional_seeds)) != 8
        or any(type(seed) is not int or seed.bit_length() != 63 for seed in directional_seeds)
    ):
        _fail("directional_seed_contract_invalid")
    root = _resolved(output_root, must_exist=False)
    artifact_root = _resolved(Path(contract["paths"]["artifact_root"]), must_exist=False)
    if root != artifact_root / "post-training":
        _fail("output_root_contract_mismatch")
    label = f"com.aura.{contract['campaign_id']}.post-training"
    material = {
        "schema": CONFIG_SCHEMA,
        "created_at_unix": time.time(),
        "source_commit": source_commit,
        "controller_source": _binding(Path(SOURCE_RELATIVE)),
        "contract": _binding(contract_path),
        "contract_sha256": contract["contract_sha256"],
        "verified_launch_bundle": _binding(
            Path(str(contract["paths"]["verified_launch_bundle"]))
        ),
        "campaign_id": contract["campaign_id"],
        "launch_label": label,
        "output_root": str(root.relative_to(REPO_ROOT)),
        "training_run_dir": contract["paths"]["detached_training"],
        "training_output": contract["paths"]["training_output"],
        "frozen_adapter": contract["paths"]["frozen_adapter"],
        "directional_campaign": contract["paths"]["directional_campaign"],
        "wait": {
            "poll_seconds": 15.0,
            "exclusive_owner_timeout_s": 86400.0,
            "training_terminal_timeout_s": 100800.0,
            "directional_terminal_timeout_s": 93600.0,
        },
        "directional": {
            "campaign_name": f"{contract['campaign_id']}-directional-factorial",
            "seeds": directional_seeds,
            "domains": list(FRONTIER_DOMAINS),
            "difficulty": 3,
            "task_registry_version": CURRENT_REGISTRY_VERSION,
            "profile": "full",
            "rlc_profile": "resident_full_stack",
            "n_slots": 16,
            "branches": 2,
            "rlc_steps": 4,
            "decode_max_tokens": 768,
            "episode_timeout_s": 1200.0,
            "load_timeout_s": 1200.0,
            "warmup_timeout_s": 600.0,
            "arm_timeout_s": 21600.0,
            "campaign_timeout_s": 86400.0,
            "equal_compute_max_samples": 8,
            "max_infra_attempts": 3,
            "detached_timeout_s": 90000.0,
            "claim_eligible": False,
        },
        "mechanism_attribution": {
            "required": True,
            "claim_eligible": False,
            "campaign_root": str((artifact_root / "mechanism-attribution").relative_to(REPO_ROOT)),
            "baseline_profile": "resident_full_stack",
            "profiles": list(_MECHANISM_PROFILES),
            "seeds": directional_seeds,
            "domains": list(FRONTIER_DOMAINS),
            "difficulty": 3,
            "task_registry_version": CURRENT_REGISTRY_VERSION,
            "profile": "full",
            "n_slots": 16,
            "branches": 2,
            "rlc_steps": 4,
            "decode_max_tokens": 768,
            "episode_timeout_s": 1200.0,
            "load_timeout_s": 1200.0,
            "warmup_timeout_s": 600.0,
            "arm_timeout_s": 21600.0,
            "campaign_timeout_s": 86400.0,
            "equal_compute_max_samples": 8,
            "max_infra_attempts": 3,
            "detached_timeout_s": 90000.0,
        },
        "claim_policy": {
            "directional_result_is_nonclaiming": True,
            "reasoning_gain_proven": False,
            "positive_interaction_proven": False,
            "frontier_level_proven": False,
            "release_eligible": False,
            "external_custody_required": True,
        },
    }
    return {**material, "config_sha256": _document_sha(material)}


def validate_config(
    config: Mapping[str, Any], *, require_live_preregistration: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = {
        "schema",
        "created_at_unix",
        "source_commit",
        "controller_source",
        "contract",
        "contract_sha256",
        "verified_launch_bundle",
        "campaign_id",
        "launch_label",
        "output_root",
        "training_run_dir",
        "training_output",
        "frozen_adapter",
        "directional_campaign",
        "wait",
        "directional",
        "mechanism_attribution",
        "claim_policy",
        "config_sha256",
    }
    material = dict(config)
    claimed = material.pop("config_sha256", None)
    if (
        set(config) != expected
        or config.get("schema") != CONFIG_SCHEMA
        or not isinstance(claimed, str)
        or claimed != _document_sha(material)
        or config.get("controller_source") != _binding(Path(SOURCE_RELATIVE))
        or not _LABEL.fullmatch(str(config.get("launch_label", "")))
    ):
        _fail("config_identity_invalid")
    contract_binding = config.get("contract")
    if not isinstance(contract_binding, Mapping):
        _fail("contract_binding_invalid")
    contract_path = Path(str(contract_binding.get("path", "")))
    if dict(contract_binding) != _binding(contract_path):
        _fail("contract_binding_changed")
    contract = _strict_json(contract_path)
    contract_material = dict(contract)
    contract_sha = contract_material.pop("contract_sha256", None)
    if (
        contract.get("schema") != prereg.CONTRACT_SCHEMA
        or contract_sha != prereg._document_sha(contract_material)
        or contract_sha != config.get("contract_sha256")
        or contract.get("campaign_id") != config.get("campaign_id")
    ):
        _fail("contract_identity_invalid")
    launch_bundle = config.get("verified_launch_bundle")
    if (
        not isinstance(launch_bundle, Mapping)
        or dict(launch_bundle)
        != _binding(Path(str(contract["paths"]["verified_launch_bundle"])))
    ):
        _fail("verified_launch_bundle_binding_changed")
    if require_live_preregistration:
        prereg.validate_contract(contract, verify_model=True)
    claim_policy = config.get("claim_policy")
    directional = config.get("directional")
    mechanism = config.get("mechanism_attribution")
    if (
        not isinstance(claim_policy, Mapping)
        or any(claim_policy.get(key) is not False for key in (
            "reasoning_gain_proven",
            "positive_interaction_proven",
            "frontier_level_proven",
            "release_eligible",
        ))
        or claim_policy.get("directional_result_is_nonclaiming") is not True
        or claim_policy.get("external_custody_required") is not True
        or not isinstance(directional, Mapping)
        or directional.get("claim_eligible") is not False
        or directional.get("profile") != "full"
        or directional.get("rlc_profile") != "resident_full_stack"
        or len(directional.get("seeds", [])) != 8
        or not isinstance(mechanism, Mapping)
        or mechanism.get("required") is not True
        or mechanism.get("claim_eligible") is not False
        or mechanism.get("baseline_profile") != "resident_full_stack"
        or tuple(mechanism.get("profiles", ())) != _MECHANISM_PROFILES
        or mechanism.get("profile") != "full"
        or mechanism.get("seeds") != directional.get("seeds")
        or mechanism.get("domains") != directional.get("domains")
        or mechanism.get("task_registry_version") != directional.get("task_registry_version")
    ):
        _fail("claim_policy_invalid")
    return dict(config), contract


def _read_config(path: Path, *, require_live_preregistration: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _strict_json(path)
    return validate_config(value, require_live_preregistration=require_live_preregistration)


def _validate_detached_terminal(
    status: Mapping[str, Any], *, role: str, allowed_returncodes: frozenset[int]
) -> dict[str, Any]:
    receipt = status.get("receipt")
    if (
        status.get("terminal") is not True
        or status.get("completion_indeterminate") is not False
        or status.get("supervisor_alive") is not False
        or status.get("child_state") != "dead"
        or not isinstance(receipt, Mapping)
        or receipt.get("returncode") not in allowed_returncodes
        or receipt.get("containment_verified") is not True
        or receipt.get("process_group_empty") is not True
        or receipt.get("lineage_empty") is not True
        or receipt.get("timed_out") is not False
        or not _HEX_64.fullmatch(str(receipt.get("receipt_sha256", "")))
    ):
        _fail(f"{role}_detached_evidence_invalid")
    return dict(receipt)


def _validate_training_completion(
    completion: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    required = contract["training"]["completion_required"]
    max_steps = contract["training"]["parameters"]["max_steps"]
    if (
        completion.get("schema") != required["schema"]
        or completion.get("complete") is not True
        or completion.get("halt_reason") != "max_steps"
        or completion.get("step") != max_steps
        or not _HEX_64.fullmatch(str(completion.get("manifest_sha256", "")))
    ):
        _fail("training_completion_not_admissible")


def _training_diagnostic_failure(
    grpo_receipt: Mapping[str, Any],
) -> list[str] | None:
    """Classify an intentional nonclaiming training stop, if present."""

    termination = grpo_receipt.get("termination")
    verdict = grpo_receipt.get("verdict")
    learning_signal = grpo_receipt.get("learning_signal")
    if (
        not isinstance(termination, Mapping)
        or not isinstance(verdict, Mapping)
        or not isinstance(learning_signal, Mapping)
    ):
        return None
    reason = str(termination.get("reason") or "")
    if reason == "no_learning_signal":
        if termination.get("completed_budget") is not False:
            _fail("training_diagnostic_claims_invalid")
    elif reason == "training_adequacy_failed":
        adequacy = grpo_receipt.get("training_adequacy")
        failed_checks = adequacy.get("failed_checks") if isinstance(adequacy, Mapping) else None
        if (
            termination.get("completed_budget") is not True
            or not isinstance(failed_checks, list)
            or not failed_checks
            or adequacy.get("admitted") is not False
        ):
            _fail("training_diagnostic_claims_invalid")
    else:
        return None
    if (
        verdict.get("had_signal") is not False
        or verdict.get("causal_gain_proven") is not False
        or learning_signal.get("learning_signal") is not False
    ):
        _fail("training_diagnostic_claims_invalid")
    diagnosis = str(learning_signal.get("diagnosis") or "unknown")
    failures = [f"training:{reason}", f"diagnosis:{diagnosis}"]
    if reason == "training_adequacy_failed":
        failures.extend(f"training_adequacy:{check}" for check in failed_checks)
    return failures


def _cmdline_script(cmdline: Sequence[str]) -> str:
    parts = [str(part) for part in cmdline if str(part)]
    if not parts:
        return ""
    executable = Path(parts[0]).name.lower()
    for index, part in enumerate(parts):
        name = Path(part).name
        if name not in _MODEL_OWNER_SCRIPTS:
            continue
        if index == 0 or executable.startswith("python") or executable in {"uv", "uvx"}:
            return name
    return ""


def _resource_blockers() -> dict[str, Any]:
    from core.runtime.resource_observation import get_resource_observer

    current_user = os.environ.get("USER") or ""
    excluded = {os.getpid(), os.getppid()}
    contenders: list[dict[str, Any]] = []
    observer = get_resource_observer()
    for process in observer.processes():
        pid = int(process.pid)
        username = str(process.username or "")
        cmdline = [str(value) for value in (process.cmdline or [])]
        if pid in excluded or (current_user and username != current_user):
            continue
        script = _cmdline_script(cmdline)
        if not script:
            continue
        contenders.append({"pid": pid, "script": script, "argv": cmdline})
    available = int(observer.memory(include_process_tree=False).available_bytes)
    return {
        "exclusive": not contenders and available >= _MIN_TRAINING_AVAILABLE_BYTES,
        "contenders": sorted(contenders, key=lambda value: value["pid"]),
        "available_memory_bytes": available,
        "minimum_available_memory_bytes": _MIN_TRAINING_AVAILABLE_BYTES,
        "memory_ready": available >= _MIN_TRAINING_AVAILABLE_BYTES,
    }


class ControllerRun:
    def __init__(self, config: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.contract = dict(contract)
        self.root = _resolved(Path(str(config["output_root"])), must_exist=False)
        ensure_private_directory(self.root)
        self.journal_path = self.root / "controller_events.jsonl"
        self.state_path = self.root / "controller_state.json"
        self.verdict_path = self.root / "controller_verdict.json"
        self.sequence = 0
        self.head = ""
        self.started_at = time.time()
        self.stage = "initializing"
        self._restore()

    def _restore(self) -> None:
        if not self.journal_path.exists():
            return
        previous = ""
        for raw_line in read_stable_bytes(
            self.journal_path, max_bytes=_MAX_JSON_BYTES
        ).splitlines():
            try:
                event = json.loads(raw_line)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise PostTrainingError("controller_journal_invalid") from exc
            body = dict(event)
            claimed = body.pop("event_sha256", None)
            if (
                event.get("schema") != EVENT_SCHEMA
                or event.get("sequence") != self.sequence + 1
                or event.get("previous_event_sha256") != previous
                or claimed != _document_sha(body)
            ):
                _fail("controller_journal_invalid")
            self.sequence += 1
            previous = str(claimed)
            self.stage = str(event.get("stage"))
            if self.sequence == 1:
                self.started_at = float(event["recorded_at"])
        self.head = previous

    def _state(self, status: str, details: Mapping[str, Any] | None = None) -> None:
        material = {
            "schema": STATE_SCHEMA,
            "stage": self.stage,
            "status": status,
            "controller_pid": os.getpid(),
            "started_at": self.started_at,
            "updated_at": time.time(),
            "event_sequence": self.sequence,
            "event_head_sha256": self.head,
            "config_sha256": self.config["config_sha256"],
            "details": dict(details or {}),
        }
        atomic_write_bytes(
            self.state_path,
            canonical_json_bytes(
                {**material, "state_sha256": _document_sha(material)}
            )
            + b"\n",
            mode=0o600,
        )

    def event(self, status: str, details: Mapping[str, Any] | None = None) -> None:
        body = {
            "schema": EVENT_SCHEMA,
            "sequence": self.sequence + 1,
            "stage": self.stage,
            "status": status,
            "recorded_at": time.time(),
            "previous_event_sha256": self.head,
            "details": dict(details or {}),
        }
        event = {**body, "event_sha256": _document_sha(body)}
        with self.journal_path.open("ab") as handle:
            handle.write(canonical_json_bytes(event) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.sequence += 1
        self.head = event["event_sha256"]
        self._state(status, details)

    def set_stage(self, stage: str) -> None:
        self.stage = stage
        self.event("started")

    def wait_for_not_before(self) -> None:
        not_before = float(self.contract["launch_not_before_unix"])
        while time.time() < not_before:
            remaining = max(0.0, not_before - time.time())
            self._state("waiting", {"remaining_seconds": remaining})
            time.sleep(min(float(self.config["wait"]["poll_seconds"]), remaining))

    def _training_attempt_root(self) -> Path:
        return _resolved(
            Path(str(self.config["training_run_dir"])),
            must_exist=False,
        )

    def _training_attempt_result(self, attempt: int) -> dict[str, Any] | None:
        path = self.root / f"training_attempt_{attempt:04d}.json"
        return _strict_json(path) if path.is_file() else None

    def _launch_or_attach_training_attempt(
        self,
        attempt: int,
    ) -> tuple[Path, dict[str, Any]]:
        attempt_root = self._training_attempt_root()
        ensure_private_directory(attempt_root)
        run_dir = attempt_root / f"attempt-{attempt:04d}"
        start_path = self.root / f"training_attempt_{attempt:04d}_start.json"
        training_output = _resolved(
            Path(str(self.config["training_output"])),
            must_exist=False,
        )
        if not start_path.exists():
            _write_once(
                start_path,
                {
                    "schema": "aura.resident_recurrent_grpo.training_attempt_start.v1",
                    "attempt": attempt,
                    "campaign_id": self.contract["campaign_id"],
                    "contract_sha256": self.contract["contract_sha256"],
                    "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                    "progress_before": prereg._training_progress_snapshot(  # noqa: SLF001
                        training_output
                    ),
                },
            )
        start = _strict_json(start_path)
        if not (run_dir / detached.PLAN_FILE).exists():
            result = prereg._launch_training(  # noqa: SLF001
                _resolved(Path(str(self.config["contract"]["path"]))),
                resume=False,
                expected_launch_bundle_sha256=str(
                    self.config["verified_launch_bundle"]["sha256"]
                ),
                run_dir_override=run_dir,
            )
            if result != 0:
                _fail(f"training_launch_failed:{attempt}:{result}")
            self.event("launched", {"attempt": attempt, "run_dir": str(run_dir)})
        return run_dir, start

    def run_training_with_recovery(self) -> dict[str, Any]:
        """Run isolated attempts under launchd and resume only sealed progress."""

        policy = self.contract["training"]["watchdog_policy"]
        max_attempts = int(policy["max_attempts"])
        no_progress_limit = int(policy["max_consecutive_no_progress_failures"])
        no_progress_failures = 0
        for attempt in range(1, max_attempts + 1):
            existing = self._training_attempt_result(attempt)
            if existing is not None:
                if existing.get("terminal_success") is True:
                    return dict(existing["detached_status"])
                no_progress_failures = int(
                    existing.get("consecutive_no_progress_failures") or 0
                )
                continue

            run_dir, start = self._launch_or_attach_training_attempt(attempt)
            status = self.wait_detached(
                run_dir,
                timeout_s=float(self.config["wait"]["training_terminal_timeout_s"]),
                role=f"training_attempt_{attempt}",
            )
            receipt = status.get("receipt")
            if not isinstance(receipt, Mapping):
                _fail(f"training_attempt_{attempt}_receipt_missing")
            returncode = receipt.get("returncode")
            if not isinstance(returncode, int) or isinstance(returncode, bool):
                _fail(f"training_attempt_{attempt}_returncode_invalid")
            _validate_detached_terminal(
                status,
                role=f"training_attempt_{attempt}",
                allowed_returncodes=frozenset({returncode}),
            )
            training_output = _resolved(
                Path(str(self.config["training_output"])),
                must_exist=False,
            )
            after = prereg._training_progress_snapshot(training_output)  # noqa: SLF001
            before = start.get("progress_before")
            if not isinstance(before, Mapping):
                _fail(f"training_attempt_{attempt}_start_invalid")
            progressed = prereg.training_progress_advanced(before, after)
            terminal_success = returncode in {0, 3}
            checkpoint_evidence: dict[str, Any] | None = None
            if not terminal_success:
                no_progress_failures = 0 if progressed else no_progress_failures + 1
                if no_progress_failures < no_progress_limit and attempt < max_attempts:
                    checkpoint_evidence = prereg.validated_training_resume_checkpoint(
                        self.contract,
                        verify_model=True,
                    )
            result = {
                "schema": "aura.resident_recurrent_grpo.training_attempt.v1",
                "attempt": attempt,
                "campaign_id": self.contract["campaign_id"],
                "contract_sha256": self.contract["contract_sha256"],
                "detached_status": status,
                "progress_before": dict(before),
                "progress_after": after,
                "durable_progress": progressed,
                "terminal_success": terminal_success,
                "consecutive_no_progress_failures": no_progress_failures,
                "resume_checkpoint": checkpoint_evidence,
            }
            _write_once(self.root / f"training_attempt_{attempt:04d}.json", result)
            self.event(
                "completed" if terminal_success else "failed",
                {
                    "attempt": attempt,
                    "returncode": returncode,
                    "receipt_sha256": receipt["receipt_sha256"],
                    "durable_progress": progressed,
                    "checkpoint_sequence": (
                        checkpoint_evidence.get("checkpoint_sequence")
                        if checkpoint_evidence is not None
                        else None
                    ),
                },
            )
            if terminal_success:
                return status
            if no_progress_failures >= no_progress_limit:
                _fail("training_no_progress_failure_limit_exhausted")
            if attempt >= max_attempts:
                _fail("training_attempt_budget_exhausted")
            time.sleep(float(policy["retry_backoff_s"]))
        _fail("training_attempt_budget_exhausted")

    def ensure_causal_learnability_gate(self) -> dict[str, Any]:
        """Run and verify the source-bound read-only gate before training."""

        artifact_root = _resolved(
            Path(str(self.contract["paths"]["artifact_root"])), must_exist=False
        )
        receipt_path = (
            artifact_root
            / "causal-learnability-preflight"
            / "causal_learnability_preflight.json"
        )
        if receipt_path.is_file() and not receipt_path.is_symlink():
            return prereg.require_causal_learnability_training_gate(self.contract)

        contract_path = _resolved(Path(str(self.config["contract"]["path"])))
        run_dir = artifact_root / "detached-causal-learnability-preflight"
        if not (run_dir / detached.PLAN_FILE).exists():
            result = prereg._launch_causal_learnability_preflight(  # noqa: SLF001
                contract_path
            )
            if result != 0:
                _fail(f"causal_learnability_preflight_launch_failed:{result}")
            self.event("launched", {"run_dir": str(run_dir)})
        status = self.wait_detached(
            run_dir,
            timeout_s=min(
                14_400.0,
                float(self.config["wait"]["training_terminal_timeout_s"]),
            ),
            role="causal_learnability_preflight",
        )
        receipt = _validate_detached_terminal(
            status,
            role="causal_learnability_preflight",
            allowed_returncodes=frozenset({0}),
        )
        gate = prereg.require_causal_learnability_training_gate(self.contract)
        self.event(
            "admitted",
            {
                "receipt_sha256": receipt["receipt_sha256"],
                "preflight_receipt_sha256": gate["receipt_sha256"],
                "optimizer_training_reachable_cells": gate[
                    "optimizer_training_reachable_cells"
                ],
            },
        )
        return gate

    def wait_for_exclusive_model_owner(self) -> dict[str, Any]:
        deadline = time.monotonic() + float(
            self.config["wait"]["exclusive_owner_timeout_s"]
        )
        while True:
            evidence = _resource_blockers()
            if evidence["exclusive"] is True:
                return evidence
            if time.monotonic() >= deadline:
                _fail("exclusive_model_owner_wait_timeout")
            self._state(
                "waiting",
                {
                    "contenders": evidence["contenders"],
                    "available_memory_bytes": evidence["available_memory_bytes"],
                    "minimum_available_memory_bytes": evidence[
                        "minimum_available_memory_bytes"
                    ],
                },
            )
            time.sleep(float(self.config["wait"]["poll_seconds"]))

    def wait_detached(self, run_dir: Path, *, timeout_s: float, role: str) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            status = detached._status(run_dir)  # noqa: SLF001
            if status.get("terminal") is True:
                return status
            if time.monotonic() >= deadline:
                _fail(f"{role}_wait_timeout")
            self._state(
                "waiting",
                {
                    "role": role,
                    "detached_state": status.get("state"),
                    "heartbeat_sequence": status.get("heartbeat_sequence"),
                },
            )
            time.sleep(float(self.config["wait"]["poll_seconds"]))

    def admit_training(self, training_receipt: Mapping[str, Any]) -> dict[str, Any]:
        output = _resolved(Path(str(self.config["training_output"])))
        completion = _strict_json(output / "training_completion.json")
        _validate_training_completion(completion, self.contract)
        model_path = _resolved(Path(str(self.contract["model"]["path"])))
        model_identity, adapter_identity = campaign_runner._identity_material(  # noqa: SLF001
            SimpleNamespace(
                model=str(model_path),
                adapter=str(output),
                adapter_id=self.contract["campaign_id"],
                personality_adapter="none",
            )
        )
        identity_receipt = adapter_identity.get("identity_receipt")
        manifest = adapter_identity.get("manifest")
        if (
            adapter_identity.get("format")
            != recurrent_grpo_adapter_identity.MANIFEST_SCHEMA
            or not isinstance(identity_receipt, Mapping)
            or identity_receipt.get("training_method") != "recurrent_grpo"
            or identity_receipt.get("causal_gain_proven") is not False
            or not isinstance(manifest, Mapping)
            or manifest.get("adapter_id") != self.contract["campaign_id"]
            or manifest.get("dataset_sha256")
            != self.contract["training"]["dataset"]["sha256"]
            or manifest.get("execution_spec_sha256")
            != self.contract["execution_spec"]["semantic_sha256"]
            or model_identity.get("fingerprint")
            != self.contract["model"]["base_checkpoint"]["fingerprint"]
            or model_identity.get("model_behavior_bundle", {}).get("bundle_sha256")
            != self.contract["model"]["behavior_bundle"]["bundle_sha256"]
        ):
            _fail("recurrent_grpo_identity_not_admissible")
        material = {
            "schema": ADMISSION_SCHEMA,
            "campaign_id": self.contract["campaign_id"],
            "contract_sha256": self.contract["contract_sha256"],
            "detached_receipt_sha256": training_receipt["receipt_sha256"],
            "completion": _binding(output / "training_completion.json"),
            "manifest": _binding(output / recurrent_grpo_adapter_identity.MANIFEST_FILE),
            "training_receipt": _binding(output / "campaign_adapter/grpo_receipt.json"),
            "identity_receipt_sha256": _document_sha(identity_receipt),
            "adapter_identity_sha256": _document_sha(adapter_identity),
            "causal_gain_proven": False,
            "reasoning_gain_proven": False,
            "frontier_level_proven": False,
        }
        admission = {**material, "admission_sha256": _document_sha(material)}
        _write_once(self.root / "training_admission.json", admission)
        return admission

    def freeze(self) -> dict[str, Any]:
        result = campaign_prepare.freeze_adapter(
            SimpleNamespace(
                source_adapter=_resolved(Path(str(self.config["training_output"]))),
                destination=_resolved(Path(str(self.config["frozen_adapter"])), must_exist=False),
                model=_resolved(Path(str(self.contract["model"]["path"]))),
                adapter_id=self.contract["campaign_id"],
                personality_adapter="none",
            )
        )
        certificate = campaign_prepare.verify_adapter_freeze(
            _resolved(Path(str(self.config["frozen_adapter"])))
        )
        if certificate.get("adapter_id") != self.contract["campaign_id"]:
            _fail("frozen_adapter_identity_mismatch")
        return {**result, "verified_certificate_sha256": certificate["certificate_sha256"]}

    def custody_request(self) -> dict[str, Any]:
        required = list(self.contract["independent_custody"]["required_roles"])
        material = {
            "schema": CUSTODY_REQUEST_SCHEMA,
            "campaign_id": self.contract["campaign_id"],
            "contract_sha256": self.contract["contract_sha256"],
            "required_roles": required,
            "distinct_keys_and_organizations_required": True,
            "producer_private_key_access_disqualifies_claim": True,
            "required_inputs": [
                "fresh externally issued task manifest",
                "contamination audit signed by an external auditor",
                "campaign trust policy with distinct role key pins",
                "prelaunch issuer and runner attestations",
                "post-seal answer reveal and final-run attestations",
                "independent verifier attestation and replay receipt",
                "named contemporaneous frontier-provider outputs",
            ],
            "claim_state": {
                "external_trust_present": False,
                "positive_interaction_proven": False,
                "frontier_level_proven": False,
                "release_eligible": False,
            },
        }
        request = {**material, "request_sha256": _document_sha(material)}
        _write_once(self.root / "external_custody_request.json", request)
        return request

    def campaign_command(
        self,
        *,
        spec: Mapping[str, Any],
        campaign_dir: Path,
        campaign_name: str,
        rlc_profile: str,
    ) -> list[str]:
        return [
            str(_python_launcher_path()),
            str(REPO_ROOT / "tools/run_latent_cortex_paired_campaign.py"),
            "--campaign-dir",
            str(_resolved(campaign_dir, must_exist=False)),
            "--campaign-name",
            campaign_name,
            "--model",
            str(_resolved(Path(str(self.contract["model"]["path"])))),
            "--adapter",
            str(_resolved(Path(str(self.config["frozen_adapter"])))),
            "--adapter-id",
            self.contract["campaign_id"],
            "--personality-adapter",
            "none",
            "--seeds",
            ",".join(str(seed) for seed in spec["seeds"]),
            "--domains",
            ",".join(spec["domains"]),
            "--difficulty",
            str(spec["difficulty"]),
            "--task-registry-version",
            str(spec["task_registry_version"]),
            "--profile",
            "full",
            "--n-slots",
            str(spec["n_slots"]),
            "--branches",
            str(spec["branches"]),
            "--rlc-steps",
            str(spec["rlc_steps"]),
            "--rlc-profile",
            rlc_profile,
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

    def directional_command(self) -> list[str]:
        spec = self.config["directional"]
        return self.campaign_command(
            spec=spec,
            campaign_dir=Path(str(self.config["directional_campaign"])),
            campaign_name=str(spec["campaign_name"]),
            rlc_profile=str(spec["rlc_profile"]),
        )

    def mechanism_campaign_dir(self, profile: str) -> Path:
        if profile not in _MECHANISM_PROFILES:
            _fail("mechanism_profile_invalid")
        root = _resolved(
            Path(str(self.config["mechanism_attribution"]["campaign_root"])),
            must_exist=False,
        )
        return root / profile

    def mechanism_command(self, profile: str) -> list[str]:
        if profile not in _MECHANISM_PROFILES:
            _fail("mechanism_profile_invalid")
        spec = self.config["mechanism_attribution"]
        return self.campaign_command(
            spec=spec,
            campaign_dir=self.mechanism_campaign_dir(profile),
            campaign_name=f"{self.contract['campaign_id']}-mechanism-{profile}",
            rlc_profile=profile,
        )

    def ensure_campaign_launched(
        self,
        *,
        role: str,
        campaign_dir: Path,
        campaign_name: str,
        command: Sequence[str],
        plan_log_name: str,
        detached_timeout_s: float,
    ) -> None:
        campaign_dir = _resolved(campaign_dir, must_exist=False)
        run_dir = campaign_dir / "detached-supervisor"
        if (run_dir / detached.PLAN_FILE).exists():
            return
        ensure_private_directory(campaign_dir)
        completed = subprocess.run(
            [*command, "--plan-only"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=3600.0,
            check=False,
        )
        log = {
            "schema": "aura.resident_recurrent_grpo_campaign_plan_command.v1",
            "role": role,
            "command": [*command, "--plan-only"],
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        _write_once(self.root / plan_log_name, log)
        if completed.returncode != 0:
            _fail(f"{role}_plan_failed:{completed.returncode}")
        try:
            plan = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise PostTrainingError(f"{role}_plan_output_invalid") from exc
        policy = plan.get("detached_broker_policy") if isinstance(plan, Mapping) else None
        if (
            not isinstance(policy, list)
            or len(policy) != 6
            or plan.get("claim_eligible") is not False
            or plan.get("cell_count") != len(FRONTIER_DOMAINS) * 8 * 6
        ):
            _fail(f"{role}_plan_contract_invalid")
        argv = [
            "launch",
            "--run-dir",
            str(run_dir),
            "--name",
            campaign_name,
            "--cwd",
            str(REPO_ROOT),
            "--timeout",
            str(detached_timeout_s),
            "--broker-policy-json",
            json.dumps(policy, sort_keys=True, separators=(",", ":")),
            "--",
            *command,
        ]
        if detached.main(argv) != 0:
            _fail(f"{role}_detached_launch_failed")
        self.event("launched", {"plan_sha256": plan["plan_sha256"]})

    def ensure_directional_launched(self) -> None:
        spec = self.config["directional"]
        self.ensure_campaign_launched(
            role="directional",
            campaign_dir=Path(str(self.config["directional_campaign"])),
            campaign_name=str(spec["campaign_name"]),
            command=self.directional_command(),
            plan_log_name="directional_plan_command.json",
            detached_timeout_s=float(spec["detached_timeout_s"]),
        )

    def ensure_mechanism_launched(self, profile: str) -> None:
        spec = self.config["mechanism_attribution"]
        self.ensure_campaign_launched(
            role=f"mechanism_{profile}",
            campaign_dir=self.mechanism_campaign_dir(profile),
            campaign_name=f"{self.contract['campaign_id']}-mechanism-{profile}",
            command=self.mechanism_command(profile),
            plan_log_name=f"mechanism_plan_command_{profile}.json",
            detached_timeout_s=float(spec["detached_timeout_s"]),
        )

    def final_verdict(
        self,
        *,
        directional_evidence: Mapping[str, Any] | None,
        mechanism_evidence: Mapping[str, Mapping[str, Any]] | None = None,
        failure_points: Sequence[str],
    ) -> dict[str, Any]:
        grade_path = _resolved(
            Path(str(self.config["directional_campaign"])), must_exist=False
        ) / "grade.json"
        grade = _strict_json(grade_path) if grade_path.exists() else None
        material = {
            "schema": VERDICT_SCHEMA,
            "campaign_id": self.contract["campaign_id"],
            "contract_sha256": self.contract["contract_sha256"],
            "stage": self.stage,
            "pipeline_complete_through_directional": directional_evidence is not None,
            "directional_evidence_passed": bool(
                directional_evidence and directional_evidence.get("passed") is True
            ),
            "directional_observed_verdict": (
                grade.get("verdict") if isinstance(grade, Mapping) else None
            ),
            "directional_claim_eligible": False,
            "mechanism_attribution_required": True,
            "mechanism_profiles": list(_MECHANISM_PROFILES),
            "mechanism_evidence_passed": {
                profile: bool(evidence.get("passed") is True)
                for profile, evidence in (mechanism_evidence or {}).items()
            },
            "mechanism_attribution_complete": (
                mechanism_evidence is not None
                and set(mechanism_evidence) == set(_MECHANISM_PROFILES)
                and all(
                    evidence.get("passed") is True
                    for evidence in mechanism_evidence.values()
                )
            ),
            "mechanism_claim_eligible": False,
            "training_mechanics_admitted": (
                self.root / "training_admission.json"
            ).exists(),
            "immutable_freeze_present": _resolved(
                Path(str(self.config["frozen_adapter"])), must_exist=False
            ).exists(),
            "external_trust_present": False,
            "reasoning_gain_proven": False,
            "positive_interaction_proven": False,
            "frontier_level_proven": False,
            "frontier_plus_proven": False,
            "release_eligible": False,
            "failure_points": list(failure_points),
            "required_next_gate": "external_custody_and_powered_confirmatory_factorial",
            "event_count": self.sequence,
            "event_head_sha256": self.head,
            "config_sha256": self.config["config_sha256"],
            "finished_at": time.time(),
        }
        verdict = {**material, "verdict_sha256": _document_sha(material)}
        _write_once(self.verdict_path, verdict)
        return verdict

    def run(self) -> dict[str, Any]:
        if self.verdict_path.exists():
            return _strict_json(self.verdict_path)
        self.set_stage("wait_for_launch_window")
        self.wait_for_not_before()
        self.set_stage("exclusive_model_owner_admission")
        resource_evidence = self.wait_for_exclusive_model_owner()
        self.event(
            "admitted",
            {
                "available_memory_bytes": resource_evidence[
                    "available_memory_bytes"
                ],
                "minimum_available_memory_bytes": resource_evidence[
                    "minimum_available_memory_bytes"
                ],
            },
        )
        self.set_stage("causal_learnability_preflight")
        self.ensure_causal_learnability_gate()
        self.set_stage("detached_training")
        training_status = self.run_training_with_recovery()
        training_receipt = _validate_detached_terminal(
            training_status, role="training", allowed_returncodes=frozenset({0, 3})
        )
        self.event("completed", {"receipt_sha256": training_receipt["receipt_sha256"]})
        if training_receipt.get("returncode") == 3:
            self.set_stage("training_diagnostic_terminal")
            output = _resolved(Path(str(self.config["training_output"])))
            grpo_receipt = _strict_json(output / "grpo_receipt.json")
            failure_points = _training_diagnostic_failure(grpo_receipt)
            if failure_points is None:
                _fail("training_diagnostic_not_admissible")
            self.event(
                "blocked",
                {
                    "receipt_sha256": training_receipt["receipt_sha256"],
                    "failure_points": failure_points,
                },
            )
            return self.final_verdict(
                directional_evidence=None,
                failure_points=failure_points,
            )

        if self.contract.get("campaign_profile") == prereg.UPDATE_CANARY_PROFILE:
            self.set_stage("update_canary_verification")
            canary = prereg.build_update_canary_verdict(
                self.contract,
                verify_model=True,
            )
            _write_once(self.root / "update_canary_verdict.json", canary)
            self.event("completed", {"verdict_sha256": canary["verdict_sha256"]})
            material = {
                "schema": "aura.resident_recurrent_grpo.canary_controller_verdict.v1",
                "campaign_id": self.contract["campaign_id"],
                "contract_sha256": self.contract["contract_sha256"],
                "training_receipt_sha256": training_receipt["receipt_sha256"],
                "canary_verdict_sha256": canary["verdict_sha256"],
                "canary_passed": True,
                "reasoning_gain_proven": False,
                "frontier_level_proven": False,
                "config_sha256": self.config["config_sha256"],
                "finished_at": time.time(),
            }
            verdict = {**material, "verdict_sha256": _document_sha(material)}
            _write_once(self.verdict_path, verdict)
            return verdict

        self.set_stage("training_admission")
        admission = self.admit_training(training_receipt)
        self.event("completed", {"admission_sha256": admission["admission_sha256"]})

        self.set_stage("immutable_adapter_freeze")
        freeze = self.freeze()
        self.event("completed", freeze)

        self.set_stage("external_custody_request")
        request = self.custody_request()
        self.event("completed", {"request_sha256": request["request_sha256"]})

        self.set_stage("directional_six_arm_factorial")
        self.ensure_directional_launched()
        campaign_dir = _resolved(Path(str(self.config["directional_campaign"])))
        directional_status = self.wait_detached(
            campaign_dir / "detached-supervisor",
            timeout_s=float(self.config["wait"]["directional_terminal_timeout_s"]),
            role="directional",
        )
        directional_receipt = _validate_detached_terminal(
            directional_status,
            role="directional",
            allowed_returncodes=frozenset({0}),
        )
        self.event("completed", {"receipt_sha256": directional_receipt["receipt_sha256"]})

        self.set_stage("independent_directional_replay")
        evidence = campaign_verifier.verify_campaign_evidence(campaign_dir)
        _write_once(self.root / "directional_evidence_verdict.json", evidence)
        if evidence.get("passed") is not True:
            _fail("directional_independent_replay_failed")
        self.event(
            "completed",
            {
                "verified_verdict": evidence.get("verified_verdict"),
                "claim_tier": evidence.get("claim_tier"),
            },
        )
        mechanism_evidence: dict[str, Mapping[str, Any]] = {}
        for profile in _MECHANISM_PROFILES:
            self.set_stage(f"mechanism_attribution_{profile}")
            self.ensure_mechanism_launched(profile)
            mechanism_dir = self.mechanism_campaign_dir(profile)
            mechanism_status = self.wait_detached(
                mechanism_dir / "detached-supervisor",
                timeout_s=float(
                    self.config["mechanism_attribution"]["detached_timeout_s"]
                ),
                role=f"mechanism_{profile}",
            )
            mechanism_receipt = _validate_detached_terminal(
                mechanism_status,
                role=f"mechanism_{profile}",
                allowed_returncodes=frozenset({0}),
            )
            mechanism_result = campaign_verifier.verify_campaign_evidence(mechanism_dir)
            _write_once(
                self.root / f"mechanism_evidence_verdict_{profile}.json",
                mechanism_result,
            )
            if mechanism_result.get("passed") is not True:
                _fail(f"mechanism_independent_replay_failed:{profile}")
            mechanism_evidence[profile] = mechanism_result
            self.event(
                "completed",
                {
                    "profile": profile,
                    "receipt_sha256": mechanism_receipt["receipt_sha256"],
                    "verified_verdict": mechanism_result.get("verified_verdict"),
                    "claim_tier": mechanism_result.get("claim_tier"),
                },
            )
        self.set_stage("awaiting_external_custody")
        verdict = self.final_verdict(
            directional_evidence=evidence,
            mechanism_evidence=mechanism_evidence,
            failure_points=["external_trust_roots_and_distinct_custodied_roles_not_present"],
        )
        self.event("terminal", {"verdict_sha256": verdict["verdict_sha256"]})
        return verdict


def _launchd_payload(config_path: Path, config: Mapping[str, Any]) -> bytes:
    label = str(config["launch_label"])
    if not _LABEL.fullmatch(label):
        _fail("launch_label_invalid")
    root = _resolved(Path(str(config["output_root"])), must_exist=False)
    python = _python_launcher_path()
    tool = Path(__file__).resolve(strict=True)
    payload = {
        "Label": label,
        "ProgramArguments": [
            "/usr/bin/caffeinate",
            "-i",
            str(python),
            str(tool),
            "run",
            "--config",
            str(_resolved(config_path)),
        ],
        "WorkingDirectory": str(REPO_ROOT),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 30,
        "ProcessType": "Background",
        "StandardOutPath": str(root / "controller.log"),
        "StandardErrorPath": str(root / "controller.log"),
    }
    return plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)


def install_launchd(config_path: Path) -> dict[str, Any]:
    config, _contract = _read_config(
        config_path, require_live_preregistration=True
    )
    ensure_private_directory(_resolved(Path(str(config["output_root"])), must_exist=False))
    uid = os.getuid()
    domain = f"gui/{uid}"
    launch_agents = Path.home() / "Library/LaunchAgents"
    ensure_private_directory(launch_agents)
    plist_path = launch_agents / f"{config['launch_label']}.plist"
    payload = _launchd_payload(config_path, config)
    atomic_write_bytes(plist_path, payload, mode=0o600)
    subprocess.run(
        ["/bin/launchctl", "bootout", domain, str(plist_path)],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    bootstrap = subprocess.run(
        ["/bin/launchctl", "bootstrap", domain, str(plist_path)],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    if bootstrap.returncode != 0:
        _fail(f"launchd_bootstrap_failed:{bootstrap.returncode}:{bootstrap.stderr.strip()}")
    material = {
        "schema": LAUNCH_SCHEMA,
        "label": config["launch_label"],
        "plist_path": str(plist_path),
        "plist_sha256": _sha(payload),
        "config_sha256": config["config_sha256"],
        "installed_at": time.time(),
        "launch_domain": domain,
    }
    receipt = {**material, "launch_sha256": _document_sha(material)}
    _write_once(
        _resolved(Path(str(config["output_root"])), must_exist=False)
        / "launchd_receipt.json",
        receipt,
    )
    return receipt


def run_controller(config_path: Path) -> int:
    config_probe = _strict_json(config_path)
    training_run = _resolved(
        Path(str(config_probe.get("training_run_dir", ""))), must_exist=False
    )
    started = (training_run / detached.PLAN_FILE).exists()
    config, contract = validate_config(
        config_probe, require_live_preregistration=not started
    )
    root = _resolved(Path(str(config["output_root"])), must_exist=False)
    ensure_private_directory(root)
    with (root / "controller.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        run = ControllerRun(config, contract)
        try:
            verdict = run.run()
        except PostTrainingError as exc:
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
                    "schema": "aura.resident_recurrent_grpo_post_training_failure.v1",
                    "stage": run.stage,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": trace,
                },
            )
            run.final_verdict(
                directional_evidence=None,
                failure_points=[f"{run.stage}:{exc.code}"],
            )
            return 0
        except BaseException:  # launchd must restart unexpected crashes
            run._state("crashed", {"traceback_sha256": _sha(traceback.format_exc().encode())})
            raise
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    prepare.add_argument("--output-root", type=Path, default=DEFAULT_CONTROLLER_ROOT)
    prepare.add_argument("--source-commit", required=True)
    prepare.add_argument("--seeds", default="")
    prepare.add_argument("--output", type=Path, default=DEFAULT_CONFIG)
    verify = commands.add_parser("verify")
    verify.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    launch = commands.add_parser("install-launchd")
    launch.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "prepare":
            seeds = [int(value) for value in args.seeds.split(",") if value] or None
            config = build_config(
                contract_path=args.contract,
                output_root=args.output_root,
                source_commit=args.source_commit,
                seeds=seeds,
            )
            _write_once(args.output, config)
            print(json.dumps(config, indent=2, sort_keys=True))
            return 0
        if args.action == "verify":
            config, contract = _read_config(
                args.config, require_live_preregistration=True
            )
            print(
                json.dumps(
                    {
                        "schema": "aura.resident_recurrent_grpo_post_training_config_receipt.v1",
                        "config_sha256": config["config_sha256"],
                        "contract_sha256": contract["contract_sha256"],
                        "claim_eligible": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.action == "install-launchd":
            print(json.dumps(install_launchd(args.config), indent=2, sort_keys=True))
            return 0
        return run_controller(args.config)
    except (OSError, PostTrainingError, TypeError, ValueError) as exc:
        print(f"resident recurrent GRPO post-training: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
