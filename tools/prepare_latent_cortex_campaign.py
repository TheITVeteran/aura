#!/usr/bin/env python
"""Freeze a scoped recurrence adapter and prepare an externally signed launch.

The tool never generates or reads a private signing key.  It snapshots a
completed supervised-v2 or recurrent-GRPO adapter, persists exact prelaunch
signature requests, and emits a launch packet only after the separately
supplied role signatures verify against the frozen campaign bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes  # noqa: E402
from core.brain.llm.latent_cortex.campaign_launch_bundle import (  # noqa: E402
    ADAPTER_FREEZE_FILE,
    LAUNCH_PACKET_FILE,
    LAUNCH_PACKET_SCHEMA,
    PRELAUNCH_BUNDLE_SCHEMA,
    PRELAUNCH_MANIFEST_FILE,
    CampaignLaunchBundleError,
    adapter_artifact_inventory,
    build_adapter_freeze_certificate,
    read_canonical_json,
    sha256_bytes,
    verify_adapter_freeze,
)
from core.brain.llm.latent_cortex.campaign_trust import (  # noqa: E402
    CAMPAIGN_RUNNER,
    TASK_ISSUER,
    CampaignTrustError,
    prepare_role_signature_request,
    validate_campaign_trust_policy,
    verify_role_attestation,
)
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (  # noqa: E402
    MANIFEST_SCHEMA_V2,
)
from core.brain.llm.latent_cortex.recurrent_grpo_adapter_identity import (  # noqa: E402
    MANIFEST_SCHEMA as RECURRENT_GRPO_MANIFEST_SCHEMA,
)
from core.brain.llm.latent_cortex.resident_recurrent_sft_adapter_identity import (  # noqa: E402
    MANIFEST_SCHEMA as RESIDENT_RECURRENT_SFT_MANIFEST_SCHEMA,
)
from core.runtime.file_read_gateway import (  # noqa: E402
    open_stable_readonly_binary,
    read_stable_bytes,
)
from tools import run_latent_cortex_paired_campaign as campaign_runner  # noqa: E402

RUNNER_PATH = REPO_ROOT / "tools/run_latent_cortex_paired_campaign.py"
IDENTITY_PATHS = {
    "recurrence_v2_identity_validator_sha256": (
        REPO_ROOT / "core/brain/llm/latent_cortex/recurrence_adapter_identity_v2.py"
    ),
    "recurrent_grpo_identity_validator_sha256": (
        REPO_ROOT / "core/brain/llm/latent_cortex/recurrent_grpo_adapter_identity.py"
    ),
    "resident_recurrent_sft_identity_validator_sha256": (
        REPO_ROOT / "core/brain/llm/latent_cortex/resident_recurrent_sft_adapter_identity.py"
    ),
}
# Backward-compatible name consumed by the supervised-v2 promotion verifier.
IDENTITY_PATH = IDENTITY_PATHS["recurrence_v2_identity_validator_sha256"]
TASK_ISSUER_PATH = REPO_ROOT / "core/brain/llm/latent_cortex/frontier_tasks.py"
FREEZE_PATH = REPO_ROOT / "core/brain/llm/latent_cortex/campaign_launch_bundle.py"

_LAUNCH_SPEC_FILE = "launch_spec.json"
_TRUST_REQUESTS_FILE = "trust_requests.json"
_ISSUER_PAYLOAD_FILE = "task_issuer_payload.json"
_RUNNER_PAYLOAD_FILE = "campaign_runner_payload.json"
_ISSUER_REQUEST_FILE = "task_issuer_signature_request.json"
_RUNNER_REQUEST_FILE = "campaign_runner_signature_request.json"
_ISSUER_ATTESTATION_FILE = "task_issuer_attestation.json"
_RUNNER_ATTESTATION_FILE = "campaign_runner_attestation.json"
_MAX_JSON_BYTES = 256 * 1024 * 1024
_MAX_DEPENDENCY_BYTES = 1024 * 1024 * 1024
_MAX_ADAPTER_ARTIFACT_BYTES = 1 << 40
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_PREPARE_SCHEMA = "aura.latent_cortex.campaign_trust_requests.v1"
_LAUNCH_SPEC_SCHEMA = "aura.latent_cortex.launch_spec.v1"


class CampaignPreparationError(RuntimeError):
    """Stable operator-facing preparation error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise CampaignPreparationError(code)


def _sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _emit(document: dict[str, Any]) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(document) + b"\n")


def write_canonical_exclusive(path: Path, document: dict[str, Any]) -> None:
    payload = canonical_json_bytes(document) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | _NOFOLLOW,
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail("launch_artifact_short_write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def copy_adapter_snapshot(source: Path, staging: Path) -> list[dict[str, Any]]:
    """Copy exactly one stable manifest generation into private staging."""

    supplied_source = source.expanduser()
    if supplied_source.is_symlink():
        _fail("adapter_source_symlink_rejected")
    source_root = supplied_source.resolve(strict=True)
    if staging.exists() or staging.is_symlink():
        _fail("adapter_staging_already_exists")
    source_inventory = adapter_artifact_inventory(source_root, reject_unplanned=False)
    staging.mkdir(parents=False, mode=0o700)
    try:
        for binding in source_inventory:
            relative = Path(binding["path"])
            source_path = (source_root / relative).resolve(strict=True)
            if source_path.parent != source_root and source_root not in source_path.parents:
                _fail("adapter_snapshot_path_escape")
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            digest = hashlib.sha256()
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | _NOFOLLOW,
                0o600,
            )
            try:
                with open_stable_readonly_binary(
                    source_path, max_bytes=_MAX_ADAPTER_ARTIFACT_BYTES
                ) as (handle, identity):
                    copied = 0
                    for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                        copied += len(chunk)
                        digest.update(chunk)
                        view = memoryview(chunk)
                        while view:
                            written = os.write(descriptor, view)
                            if written <= 0:
                                _fail("adapter_snapshot_short_write")
                            view = view[written:]
                    if copied != identity.size:
                        _fail("adapter_snapshot_size_changed")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if copied != binding["size_bytes"] or digest.hexdigest() != binding["sha256"]:
                _fail("adapter_snapshot_source_changed")
        copied_inventory = adapter_artifact_inventory(staging, reject_unplanned=False)
        if copied_inventory != source_inventory:
            _fail("adapter_snapshot_copy_mismatch")
        return copied_inventory
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def seal_adapter_snapshot(
    staging: Path,
    destination: Path,
    certificate: dict[str, Any],
) -> None:
    """Seal and publish a staged adapter generation."""

    write_canonical_exclusive(staging / ADAPTER_FREEZE_FILE, certificate)
    try:
        for path in sorted(staging.rglob("*"), key=lambda value: len(value.parts), reverse=True):
            if path.is_file():
                path.chmod(0o400)
            elif path.is_dir():
                path.chmod(0o500)
        staging.chmod(0o500)
        verify_adapter_freeze(staging)
        os.rename(staging, destination)
    except BaseException as exc:
        try:
            staging.chmod(0o700)
            for path in staging.rglob("*"):
                if path.is_dir():
                    path.chmod(0o700)
                elif not path.is_symlink():
                    path.chmod(0o600)
        except OSError:
            pass
        if isinstance(exc, FileExistsError):
            _fail("adapter_freeze_destination_exists")
        raise
    directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _source_sha256(path: Path) -> str:
    return sha256_bytes(read_stable_bytes(path.resolve(strict=True), max_bytes=_MAX_JSON_BYTES))


def _artifact_binding(path: Path, *, role: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    payload = read_stable_bytes(resolved, max_bytes=_MAX_DEPENDENCY_BYTES)
    return {
        "role": role,
        "path": str(resolved),
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _option_values(argv: list[str], option: str) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == option:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                _fail(f"runner_option_value_missing:{option}")
            values.append(argv[index + 1])
            index += 2
            continue
        prefix = f"{option}="
        if token.startswith(prefix):
            values.append(token[len(prefix) :])
        index += 1
    return values


def _one_option(argv: list[str], option: str) -> str:
    values = _option_values(argv, option)
    if len(values) != 1 or not values[0]:
        _fail(f"runner_option_cardinality_invalid:{option}")
    return values[0]


def _has_flag(argv: list[str], option: str) -> bool:
    return sum(token == option for token in argv) == 1


def _validate_runner_argv(
    argv: list[str],
    *,
    freeze_dir: Path,
    certificate: dict[str, Any],
) -> dict[str, str]:
    if not argv or any(not isinstance(value, str) or "\x00" in value for value in argv):
        _fail("runner_argv_invalid")
    forbidden = {
        "--prepare-trust",
        "--task-issuer-attestation",
        "--runner-attestation",
        "--answer-reveal-attestation",
        "--final-run-attestation",
        "--worker-arm",
        "--worker-attempt-slot",
        "--worker-stage-journal",
        "--plan-only",
    }
    for option in forbidden:
        if _option_values(argv, option) or option in argv:
            _fail(f"runner_option_forbidden:{option}")
    if not _has_flag(argv, "--confirmatory"):
        _fail("confirmatory_campaign_required")
    required = (
        "--campaign-dir",
        "--campaign-name",
        "--model",
        "--adapter",
        "--adapter-id",
        "--seeds",
        "--contamination-audit",
        "--contamination-trust-root",
        "--campaign-trust-policy",
        "--campaign-trust-root",
    )
    values = {option: _one_option(argv, option) for option in required}
    for option in (
        "--model",
        "--adapter",
        "--contamination-audit",
        "--contamination-trust-root",
        "--campaign-trust-policy",
        "--campaign-trust-root",
    ):
        supplied = Path(values[option]).expanduser()
        if supplied.is_symlink() or not supplied.is_absolute():
            _fail(f"runner_path_not_canonical:{option}")
        resolved = supplied.resolve(strict=True)
        if str(supplied) != str(resolved):
            _fail(f"runner_path_not_canonical:{option}")
    actual_freeze = Path(values["--adapter"])
    if actual_freeze != freeze_dir.resolve(strict=True):
        _fail("runner_adapter_not_frozen_snapshot")
    if values["--adapter-id"] != certificate["adapter_id"]:
        _fail("runner_adapter_id_mismatch")
    supplied_campaign_dir = Path(values["--campaign-dir"]).expanduser()
    if not supplied_campaign_dir.is_absolute() or supplied_campaign_dir.is_symlink():
        _fail("campaign_directory_not_canonical")
    campaign_dir = supplied_campaign_dir.resolve(strict=False)
    if str(supplied_campaign_dir) != str(campaign_dir):
        _fail("campaign_directory_not_canonical")
    if campaign_dir.exists() or campaign_dir.is_symlink():
        _fail("campaign_directory_must_not_exist")
    if campaign_dir == freeze_dir or freeze_dir in campaign_dir.parents:
        _fail("campaign_directory_overlaps_adapter_freeze")
    return values


def _strict_process_json(raw: bytes, *, role: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                _fail(f"{role}_duplicate_key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_float=lambda value: (
                float(value) if math.isfinite(float(value)) else _fail(f"{role}_number_invalid")
            ),
            parse_constant=lambda _value: _fail(f"{role}_number_invalid"),
        )
    except CampaignPreparationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        _fail(f"{role}_json_invalid")
    if not isinstance(value, dict):
        _fail(f"{role}_not_object")
    return value


def _run_prepare_trust(argv: list[str], *, timeout: float) -> dict[str, Any]:
    command = [sys.executable, str(RUNNER_PATH), *argv, "--prepare-trust"]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        _fail("campaign_prepare_timeout")
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
        raise CampaignPreparationError(f"campaign_prepare_failed:{completed.returncode}:{detail}")
    trust = _strict_process_json(completed.stdout, role="campaign_trust_requests")
    requests = trust.get("requests")
    if (
        trust.get("schema") != _PREPARE_SCHEMA
        or not isinstance(requests, dict)
        or set(requests) != {TASK_ISSUER, CAMPAIGN_RUNNER}
        or not all(isinstance(payload, dict) for payload in requests.values())
    ):
        _fail("campaign_trust_requests_invalid")
    return trust


def _selected_model_identity(model: dict[str, Any]) -> dict[str, Any]:
    try:
        selected = {
            "fingerprint": model["fingerprint"],
            "files": model["files"],
            "model_behavior_bundle_sha256": model["model_behavior_bundle"]["bundle_sha256"],
            "runtime_bundle_sha256": model["runtime_bundle"]["bundle_sha256"],
            "runtime_environment_identity_sha256": model["runtime_environment"]["identity_sha256"],
            "personality_adapter_bundle_sha256": model["personality_adapter"]["bundle_sha256"],
            "effective_stack_sha256": model["effective_stack_sha256"],
        }
    except (KeyError, TypeError):
        _fail("validated_model_identity_incomplete")
    return selected


def freeze_adapter(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_adapter.expanduser()
    if source.is_symlink():
        _fail("adapter_source_symlink_rejected")
    source = source.resolve(strict=True)
    destination = args.destination.expanduser().resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        certificate = verify_adapter_freeze(destination)
        if certificate["adapter_id"] != args.adapter_id:
            _fail("existing_freeze_adapter_id_mismatch")
        return {
            "schema": "aura.latent_cortex.adapter_freeze_result.v1",
            "status": "already_frozen",
            "destination": str(destination),
            "certificate_sha256": certificate["certificate_sha256"],
        }
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.parent.is_symlink():
        _fail("adapter_freeze_parent_symlink_rejected")
    staging = destination.with_name(f".{destination.name}.{os.getpid()}.staging")
    inventory = copy_adapter_snapshot(source, staging)
    try:
        model_identity, adapter_identity = campaign_runner._identity_material(
            SimpleNamespace(
                model=str(args.model.expanduser().resolve(strict=True)),
                adapter=str(staging),
                adapter_id=args.adapter_id,
                personality_adapter=args.personality_adapter,
            )
        )
        if adapter_identity.get("format") not in {
            MANIFEST_SCHEMA_V2,
            RECURRENT_GRPO_MANIFEST_SCHEMA,
            RESIDENT_RECURRENT_SFT_MANIFEST_SCHEMA,
        }:
            _fail("supported_scoped_adapter_required")
        certificate = build_adapter_freeze_certificate(
            adapter_id=args.adapter_id,
            inventory=inventory,
            identity_receipt=adapter_identity["identity_receipt"],
            model_identity=_selected_model_identity(model_identity),
            validator_identity={
                "campaign_runner_sha256": _source_sha256(RUNNER_PATH),
                "freeze_contract_sha256": _source_sha256(FREEZE_PATH),
                **{role: _source_sha256(path) for role, path in IDENTITY_PATHS.items()},
            },
        )
        seal_adapter_snapshot(staging, destination, certificate)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    verified = verify_adapter_freeze(destination)
    return {
        "schema": "aura.latent_cortex.adapter_freeze_result.v1",
        "status": "frozen",
        "destination": str(destination),
        "certificate_sha256": verified["certificate_sha256"],
        "content_root_sha256": verified["content_root_sha256"],
        "artifact_count": len(verified["artifacts"]),
    }


def _bundle_binding(path: Path) -> dict[str, Any]:
    payload = read_stable_bytes(path, max_bytes=_MAX_JSON_BYTES)
    return {
        "path": path.name,
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _write_bundle_generation(
    bundle_dir: Path,
    documents: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if bundle_dir.exists() or bundle_dir.is_symlink():
        _fail("prelaunch_bundle_destination_exists")
    bundle_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = bundle_dir.with_name(f".{bundle_dir.name}.{os.getpid()}.staging")
    if staging.exists():
        _fail("prelaunch_bundle_staging_exists")
    staging.mkdir(mode=0o700)
    try:
        for name, document in documents.items():
            write_canonical_exclusive(staging / name, document)
        artifacts = [_bundle_binding(staging / name) for name in sorted(documents)]
        material = {
            "schema": PRELAUNCH_BUNDLE_SCHEMA,
            "phase": "awaiting_prelaunch_signatures",
            "artifacts": artifacts,
            "artifact_root_sha256": _sha256(
                {
                    "schema": "aura.latent_cortex.prelaunch_artifacts.v1",
                    "artifacts": artifacts,
                }
            ),
        }
        manifest = {**material, "manifest_sha256": _sha256(material)}
        write_canonical_exclusive(staging / PRELAUNCH_MANIFEST_FILE, manifest)
        for path in staging.iterdir():
            path.chmod(0o400)
        os.rename(staging, bundle_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _dependency_bindings(values: dict[str, str]) -> list[dict[str, Any]]:
    return [
        _artifact_binding(
            Path(values[option]),
            role=role,
        )
        for option, role in (
            ("--contamination-audit", "contamination_audit"),
            ("--contamination-trust-root", "contamination_trust_root"),
            ("--campaign-trust-policy", "campaign_trust_policy"),
            ("--campaign-trust-root", "campaign_trust_root"),
        )
    ]


def prepare_bundle(args: argparse.Namespace) -> dict[str, Any]:
    supplied_freeze = args.adapter_freeze.expanduser()
    if supplied_freeze.is_symlink():
        _fail("adapter_freeze_symlink_rejected")
    freeze_dir = supplied_freeze.resolve(strict=True)
    certificate = verify_adapter_freeze(freeze_dir)
    runner_argv = list(args.runner_args)
    if runner_argv and runner_argv[0] == "--":
        runner_argv = runner_argv[1:]
    values = _validate_runner_argv(
        runner_argv,
        freeze_dir=freeze_dir,
        certificate=certificate,
    )
    trust = _run_prepare_trust(runner_argv, timeout=args.prepare_timeout)
    if trust.get("campaign_name") != values["--campaign-name"]:
        _fail("prepared_campaign_name_mismatch")
    if trust.get("externally_custodied") is not True:
        _fail("external_campaign_custody_required")
    policy_path = Path(values["--campaign-trust-policy"]).expanduser().resolve(strict=True)
    root_path = Path(values["--campaign-trust-root"]).expanduser().resolve(strict=True)
    policy = validate_campaign_trust_policy(
        read_canonical_json(policy_path, role="campaign_trust_policy"),
        trusted_root_public_key_pem=read_stable_bytes(root_path, max_bytes=64 * 1024),
        expected_campaign_name=values["--campaign-name"],
        expected_protocol_sha256=trust["protocol_sha256"],
        now_unix=args.observed_at,
    )
    expected_implementations = {
        TASK_ISSUER: _source_sha256(TASK_ISSUER_PATH),
        CAMPAIGN_RUNNER: _source_sha256(RUNNER_PATH),
    }
    if any(
        policy.role_pin(role)["implementation_sha256"] != expected
        for role, expected in expected_implementations.items()
    ):
        _fail("prelaunch_role_implementation_mismatch")
    payloads = trust["requests"]
    issuer_request = prepare_role_signature_request(
        policy,
        role=TASK_ISSUER,
        payload=payloads[TASK_ISSUER],
        signed_at_unix=args.signed_at,
    )
    runner_request = prepare_role_signature_request(
        policy,
        role=CAMPAIGN_RUNNER,
        payload=payloads[CAMPAIGN_RUNNER],
        signed_at_unix=args.signed_at,
    )
    dependencies = _dependency_bindings(values)
    launch_material = {
        "schema": _LAUNCH_SPEC_SCHEMA,
        "working_directory": str(REPO_ROOT.resolve(strict=True)),
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "python_executable_sha256": _source_sha256(Path(sys.executable)),
        "campaign_runner": str(RUNNER_PATH.resolve(strict=True)),
        "campaign_runner_sha256": _source_sha256(RUNNER_PATH),
        "runner_argv": runner_argv,
        "runner_argv_sha256": _sha256(runner_argv),
        "campaign_dir": str(Path(values["--campaign-dir"]).expanduser().resolve(strict=False)),
        "adapter_freeze_dir": str(freeze_dir),
        "adapter_freeze_certificate_sha256": certificate["certificate_sha256"],
        "protocol_sha256": trust["protocol_sha256"],
        "unsigned_plan_sha256": trust["unsigned_plan_sha256"],
        "policy_sha256": trust["policy_sha256"],
        "dependency_artifacts": dependencies,
    }
    launch_spec = {**launch_material, "launch_spec_sha256": _sha256(launch_material)}
    documents = {
        ADAPTER_FREEZE_FILE: certificate,
        _LAUNCH_SPEC_FILE: launch_spec,
        _TRUST_REQUESTS_FILE: trust,
        _ISSUER_PAYLOAD_FILE: payloads[TASK_ISSUER],
        _RUNNER_PAYLOAD_FILE: payloads[CAMPAIGN_RUNNER],
        _ISSUER_REQUEST_FILE: issuer_request,
        _RUNNER_REQUEST_FILE: runner_request,
    }
    bundle_dir = args.bundle_dir.expanduser().resolve(strict=False)
    manifest = _write_bundle_generation(bundle_dir, documents)
    return {
        "schema": "aura.latent_cortex.prelaunch_bundle_result.v1",
        "phase": manifest["phase"],
        "bundle_dir": str(bundle_dir),
        "manifest_sha256": manifest["manifest_sha256"],
        "issuer_request": str(bundle_dir / _ISSUER_REQUEST_FILE),
        "runner_request": str(bundle_dir / _RUNNER_REQUEST_FILE),
    }


def _verify_bundle_artifacts(bundle_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_canonical_json(bundle_dir / PRELAUNCH_MANIFEST_FILE, role="prelaunch_manifest")
    material = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        set(manifest) != {"schema", "phase", "artifacts", "artifact_root_sha256", "manifest_sha256"}
        or manifest.get("schema") != PRELAUNCH_BUNDLE_SCHEMA
        or manifest.get("phase") != "awaiting_prelaunch_signatures"
        or manifest.get("manifest_sha256") != _sha256(material)
        or not isinstance(manifest.get("artifacts"), list)
    ):
        _fail("prelaunch_manifest_invalid")
    observed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in manifest["artifacts"]:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "sha256", "size_bytes"}
            or not isinstance(record.get("path"), str)
            or Path(record["path"]).name != record["path"]
            or record["path"] in seen
        ):
            _fail("prelaunch_artifact_binding_invalid")
        seen.add(record["path"])
        actual = _bundle_binding(bundle_dir / record["path"])
        if actual != record:
            _fail("prelaunch_artifact_changed")
        observed.append(actual)
    required_artifacts = {
        ADAPTER_FREEZE_FILE,
        _LAUNCH_SPEC_FILE,
        _TRUST_REQUESTS_FILE,
        _ISSUER_PAYLOAD_FILE,
        _RUNNER_PAYLOAD_FILE,
        _ISSUER_REQUEST_FILE,
        _RUNNER_REQUEST_FILE,
    }
    if seen != required_artifacts:
        _fail("prelaunch_artifact_set_invalid")
    observed_files: set[str] = set()
    for path in bundle_dir.iterdir():
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            _fail("prelaunch_storage_invalid")
        observed_files.add(path.name)
    allowed_runtime = {
        _ISSUER_ATTESTATION_FILE,
        _RUNNER_ATTESTATION_FILE,
        LAUNCH_PACKET_FILE,
        "answer_reveal_attestation.json",
        "answer_reveal_resume_packet.json",
        "final_run_attestation.json",
        "final_run_resume_packet.json",
    }
    if not observed_files.issubset(seen | {PRELAUNCH_MANIFEST_FILE} | allowed_runtime):
        _fail("prelaunch_unplanned_artifact")
    if LAUNCH_PACKET_FILE in observed_files and not {
        _ISSUER_ATTESTATION_FILE,
        _RUNNER_ATTESTATION_FILE,
    }.issubset(observed_files):
        _fail("prelaunch_admission_incomplete")
    if manifest.get("artifact_root_sha256") != _sha256(
        {"schema": "aura.latent_cortex.prelaunch_artifacts.v1", "artifacts": observed}
    ):
        _fail("prelaunch_artifact_root_invalid")
    launch_spec = read_canonical_json(bundle_dir / _LAUNCH_SPEC_FILE, role="launch_spec")
    return manifest, launch_spec


def _create_or_verify(path: Path, document: dict[str, Any], *, role: str) -> None:
    if path.exists() or path.is_symlink():
        if read_canonical_json(path, role=role) != document:
            _fail(f"{role}_conflict")
        return
    write_canonical_exclusive(path, document)
    path.chmod(0o400)


def _verify_dependency_bindings(bindings: Any) -> None:
    if not isinstance(bindings, list) or len(bindings) != 4:
        _fail("launch_dependency_set_invalid")
    observed = [_artifact_binding(Path(record["path"]), role=record["role"]) for record in bindings]
    if observed != bindings:
        _fail("launch_dependency_changed")


def admit_bundle(args: argparse.Namespace) -> dict[str, Any]:
    bundle_dir = args.bundle_dir.expanduser().resolve(strict=True)
    manifest, launch_spec = _verify_bundle_artifacts(bundle_dir)
    freeze_dir = Path(launch_spec["adapter_freeze_dir"])
    certificate = verify_adapter_freeze(freeze_dir)
    if certificate["certificate_sha256"] != launch_spec.get("adapter_freeze_certificate_sha256"):
        _fail("launch_adapter_freeze_changed")
    material = {key: value for key, value in launch_spec.items() if key != "launch_spec_sha256"}
    if (
        launch_spec.get("schema") != _LAUNCH_SPEC_SCHEMA
        or launch_spec.get("launch_spec_sha256") != _sha256(material)
        or launch_spec.get("campaign_runner_sha256") != _source_sha256(RUNNER_PATH)
        or launch_spec.get("python_executable_sha256") != _source_sha256(Path(sys.executable))
    ):
        _fail("launch_spec_invalid")
    _verify_dependency_bindings(launch_spec.get("dependency_artifacts"))
    runner_argv = launch_spec.get("runner_argv")
    if not isinstance(runner_argv, list) or _sha256(runner_argv) != launch_spec.get(
        "runner_argv_sha256"
    ):
        _fail("launch_argv_invalid")
    values = _validate_runner_argv(
        runner_argv,
        freeze_dir=freeze_dir,
        certificate=certificate,
    )
    prepared = _run_prepare_trust(runner_argv, timeout=args.prepare_timeout)
    expected_prepared = read_canonical_json(
        bundle_dir / _TRUST_REQUESTS_FILE, role="trust_requests"
    )
    if prepared != expected_prepared:
        _fail("campaign_preparation_replay_mismatch")
    policy_path = Path(values["--campaign-trust-policy"]).expanduser().resolve(strict=True)
    root_path = Path(values["--campaign-trust-root"]).expanduser().resolve(strict=True)
    policy = validate_campaign_trust_policy(
        read_canonical_json(policy_path, role="campaign_trust_policy"),
        trusted_root_public_key_pem=read_stable_bytes(root_path, max_bytes=64 * 1024),
        expected_campaign_name=values["--campaign-name"],
        expected_protocol_sha256=prepared["protocol_sha256"],
        now_unix=args.observed_at,
    )
    if prepared.get("externally_custodied") is not True or any(
        policy.role_pin(role)["implementation_sha256"] != expected
        for role, expected in (
            (TASK_ISSUER, _source_sha256(TASK_ISSUER_PATH)),
            (CAMPAIGN_RUNNER, _source_sha256(RUNNER_PATH)),
        )
    ):
        _fail("prelaunch_role_implementation_mismatch")
    issuer = read_canonical_json(args.task_issuer_attestation, role="task_issuer_attestation")
    runner = read_canonical_json(args.runner_attestation, role="campaign_runner_attestation")
    requests = {
        TASK_ISSUER: read_canonical_json(
            bundle_dir / _ISSUER_REQUEST_FILE, role="task_issuer_request"
        ),
        CAMPAIGN_RUNNER: read_canonical_json(
            bundle_dir / _RUNNER_REQUEST_FILE, role="campaign_runner_request"
        ),
    }
    attestations = {TASK_ISSUER: issuer, CAMPAIGN_RUNNER: runner}
    for role in (TASK_ISSUER, CAMPAIGN_RUNNER):
        expected_payload = prepared["requests"][role]
        signed = verify_role_attestation(
            policy,
            attestations[role],
            role=role,
            expected_payload=expected_payload,
        )
        if signed != requests[role].get("signed_payload"):
            _fail(f"{role}_signature_request_mismatch")
    issuer_path = bundle_dir / _ISSUER_ATTESTATION_FILE
    runner_path = bundle_dir / _RUNNER_ATTESTATION_FILE
    _create_or_verify(issuer_path, issuer, role="persisted_task_issuer_attestation")
    _create_or_verify(runner_path, runner, role="persisted_campaign_runner_attestation")
    argv = [
        str(Path(sys.executable).resolve(strict=True)),
        str(RUNNER_PATH.resolve(strict=True)),
        *runner_argv,
        "--task-issuer-attestation",
        str(issuer_path),
        "--runner-attestation",
        str(runner_path),
    ]
    packet_material = {
        "schema": LAUNCH_PACKET_SCHEMA,
        "phase": "ready_for_inference",
        "prelaunch_manifest_sha256": manifest["manifest_sha256"],
        "adapter_freeze_certificate_sha256": certificate["certificate_sha256"],
        "task_issuer_attestation_sha256": _bundle_binding(issuer_path)["sha256"],
        "campaign_runner_attestation_sha256": _bundle_binding(runner_path)["sha256"],
        "working_directory": str(REPO_ROOT.resolve(strict=True)),
        "campaign_dir": launch_spec["campaign_dir"],
        "argv": argv,
        "argv_sha256": _sha256(argv),
    }
    packet = {**packet_material, "packet_sha256": _sha256(packet_material)}
    _create_or_verify(
        bundle_dir / LAUNCH_PACKET_FILE,
        packet,
        role="persisted_launch_packet",
    )
    return {
        "schema": "aura.latent_cortex.launch_admission_result.v1",
        "phase": packet["phase"],
        "bundle_dir": str(bundle_dir),
        "packet_sha256": packet["packet_sha256"],
        "launch_packet": str(bundle_dir / LAUNCH_PACKET_FILE),
        "argv": argv,
    }


def inspect_bundle(args: argparse.Namespace) -> dict[str, Any]:
    bundle_dir = args.bundle_dir.expanduser().resolve(strict=True)
    manifest, launch_spec = _verify_bundle_artifacts(bundle_dir)
    material = {key: value for key, value in launch_spec.items() if key != "launch_spec_sha256"}
    if (
        launch_spec.get("schema") != _LAUNCH_SPEC_SCHEMA
        or launch_spec.get("launch_spec_sha256") != _sha256(material)
        or launch_spec.get("campaign_runner_sha256") != _source_sha256(RUNNER_PATH)
        or launch_spec.get("python_executable_sha256") != _source_sha256(Path(sys.executable))
    ):
        _fail("launch_spec_invalid")
    _verify_dependency_bindings(launch_spec.get("dependency_artifacts"))
    freeze = verify_adapter_freeze(Path(launch_spec["adapter_freeze_dir"]))
    if freeze["certificate_sha256"] != launch_spec.get("adapter_freeze_certificate_sha256"):
        _fail("launch_adapter_freeze_changed")
    result: dict[str, Any] = {
        "schema": "aura.latent_cortex.prelaunch_bundle_inspection.v1",
        "ok": True,
        "phase": manifest["phase"],
        "manifest_sha256": manifest["manifest_sha256"],
        "launch_spec_sha256": launch_spec["launch_spec_sha256"],
    }
    packet_path = bundle_dir / LAUNCH_PACKET_FILE
    if packet_path.exists():
        packet = read_canonical_json(packet_path, role="launch_packet")
        packet_material = {key: value for key, value in packet.items() if key != "packet_sha256"}
        issuer_path = bundle_dir / _ISSUER_ATTESTATION_FILE
        runner_path = bundle_dir / _RUNNER_ATTESTATION_FILE
        issuer = read_canonical_json(issuer_path, role="persisted_task_issuer_attestation")
        runner = read_canonical_json(runner_path, role="persisted_campaign_runner_attestation")
        runner_argv = launch_spec.get("runner_argv")
        expected_argv = [
            str(Path(sys.executable).resolve(strict=True)),
            str(RUNNER_PATH.resolve(strict=True)),
            *runner_argv,
            "--task-issuer-attestation",
            str(issuer_path),
            "--runner-attestation",
            str(runner_path),
        ]
        values = {
            option: _one_option(runner_argv, option)
            for option in (
                "--campaign-name",
                "--campaign-trust-policy",
                "--campaign-trust-root",
            )
        }
        policy = validate_campaign_trust_policy(
            read_canonical_json(
                Path(values["--campaign-trust-policy"]).expanduser().resolve(strict=True),
                role="campaign_trust_policy",
            ),
            trusted_root_public_key_pem=read_stable_bytes(
                Path(values["--campaign-trust-root"]).expanduser().resolve(strict=True),
                max_bytes=64 * 1024,
            ),
            expected_campaign_name=values["--campaign-name"],
            expected_protocol_sha256=launch_spec["protocol_sha256"],
        )
        expected_payloads = read_canonical_json(
            bundle_dir / _TRUST_REQUESTS_FILE, role="trust_requests"
        )["requests"]
        verify_role_attestation(
            policy,
            issuer,
            role=TASK_ISSUER,
            expected_payload=expected_payloads[TASK_ISSUER],
        )
        verify_role_attestation(
            policy,
            runner,
            role=CAMPAIGN_RUNNER,
            expected_payload=expected_payloads[CAMPAIGN_RUNNER],
        )
        if (
            packet.get("schema") != LAUNCH_PACKET_SCHEMA
            or packet.get("packet_sha256") != _sha256(packet_material)
            or packet.get("prelaunch_manifest_sha256") != manifest["manifest_sha256"]
            or packet.get("adapter_freeze_certificate_sha256") != freeze["certificate_sha256"]
            or packet.get("task_issuer_attestation_sha256")
            != _bundle_binding(issuer_path)["sha256"]
            or packet.get("campaign_runner_attestation_sha256")
            != _bundle_binding(runner_path)["sha256"]
            or packet.get("argv") != expected_argv
            or packet.get("argv_sha256") != _sha256(expected_argv)
        ):
            _fail("launch_packet_invalid")
        result.update(phase=packet["phase"], packet_sha256=packet["packet_sha256"])
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze", help="snapshot one completed scoped recurrence adapter")
    freeze.add_argument("--source-adapter", type=Path, required=True)
    freeze.add_argument("--destination", type=Path, required=True)
    freeze.add_argument("--model", type=Path, required=True)
    freeze.add_argument("--adapter-id", required=True)
    freeze.add_argument("--personality-adapter", default="trained")

    prepare = commands.add_parser(
        "prepare", help="persist exact prelaunch payloads and signature requests"
    )
    prepare.add_argument("--bundle-dir", type=Path, required=True)
    prepare.add_argument("--adapter-freeze", type=Path, required=True)
    prepare.add_argument("--signed-at", type=int, required=True)
    prepare.add_argument("--observed-at", type=int, default=None)
    prepare.add_argument("--prepare-timeout", type=float, default=3600.0)
    prepare.add_argument("runner_args", nargs=argparse.REMAINDER)

    admit = commands.add_parser(
        "admit", help="verify detached prelaunch signatures and seal launch argv"
    )
    admit.add_argument("--bundle-dir", type=Path, required=True)
    admit.add_argument("--task-issuer-attestation", type=Path, required=True)
    admit.add_argument("--runner-attestation", type=Path, required=True)
    admit.add_argument("--observed-at", type=int, default=None)
    admit.add_argument("--prepare-timeout", type=float, default=3600.0)

    inspect = commands.add_parser("inspect", help="verify a prepared bundle")
    inspect.add_argument("--bundle-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command in {"prepare", "admit"} and args.observed_at is None:
        args.observed_at = int(time.time())
    try:
        if args.command == "freeze":
            document = freeze_adapter(args)
        elif args.command == "prepare":
            if args.signed_at <= 0 or args.prepare_timeout <= 0:
                _fail("campaign_prepare_arguments_invalid")
            document = prepare_bundle(args)
        elif args.command == "admit":
            if args.prepare_timeout <= 0:
                _fail("campaign_prepare_arguments_invalid")
            document = admit_bundle(args)
        else:
            document = inspect_bundle(args)
        _emit(document)
        return 0
    except (
        CampaignLaunchBundleError,
        CampaignPreparationError,
        CampaignTrustError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        _emit(
            {
                "schema": "aura.latent_cortex.campaign_preparation_error.v1",
                "ok": False,
                "reason": getattr(exc, "code", str(exc)) or type(exc).__name__,
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
