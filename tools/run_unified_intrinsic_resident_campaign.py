#!/usr/bin/env python3
"""Run one source-bound resident unified-recurrence campaign under launchd."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import json
import math
import os
import plistlib
import stat
import subprocess
import sys
import time
import traceback
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_bytes_if_absent,
    ensure_private_directory,
)
from core.runtime.mlx_memory_guard import host_pressure  # noqa: E402
from core.runtime.model_lane_control import get_model_lane_controller  # noqa: E402
from core.runtime.resource_observation import get_resource_observer  # noqa: E402
from tools import run_detached_step as detached  # noqa: E402
from tools.prepare_unified_intrinsic_resident_campaign import (  # noqa: E402
    CONFIG_SCHEMA,
    _profile_training,
    _training_cli,
)
from tools.resident_recurrent_sft_bootstrap_identity import (  # noqa: E402
    load_resident_bootstrap_tokenizer,
    resident_bootstrap_tokenizer_identity,
)
from tools.unified_intrinsic_checkpoint import (  # noqa: E402
    UnifiedCheckpointError,
    resolve_checkpoint_generation,
    unpointed_checkpoint_inventory,
)
from tools.unified_intrinsic_preload_barrier import (  # noqa: E402
    command_sha256,
    publish_release,
    verify_release,
)
from tools.unified_intrinsic_resident_identity import (  # noqa: E402
    campaign_checkpoint_binding,
    canonical_bytes,
    canonical_sha256,
    runtime_identity,
    trainer_model_identity_from_manifest,
    verify_model_manifest,
    verify_source_git_identity,
    verify_source_manifest,
)
from tools.unified_intrinsic_tokenization_contract import (  # noqa: E402
    freeze_source_dataset,
    load_source_dataset,
    verify_tokenized_dataset,
)

STATUS_SCHEMA: Final = "aura.unified_intrinsic.controller_status.v1"
ATTEMPT_RESERVATION_SCHEMA: Final = "aura.unified_intrinsic.attempt_reservation.v1"
ATTEMPT_RESULT_SCHEMA: Final = "aura.unified_intrinsic.attempt_result.v1"
COMPLETION_SCHEMA: Final = "aura.unified_intrinsic.controller_completion.v1"
LAUNCH_SCHEMA: Final = "aura.unified_intrinsic.launchd.v1"
LAUNCH_INTENT_SCHEMA: Final = "aura.unified_intrinsic.launch_intent.v1"
LAUNCH_POINTER_SCHEMA: Final = "aura.unified_intrinsic.launch_pointer.v1"
ERROR_SCHEMA: Final = "aura.unified_intrinsic.controller_error.v1"
MAX_DOCUMENT_BYTES: Final = 512 * 1024 * 1024
STARTUP_LETHAL_MB: Final = 54.0 * 1024.0
STEADY_LETHAL_MB: Final = 48.0 * 1024.0
PRELOAD_TIMEOUT_S: Final = 300.0
HOST_LEASE_HANDOFF_TIMEOUT_S: Final = 60.0
DETACHED_TERMINAL_HANDOFF_TIMEOUT_S: Final = 5.0
TRAINING_LABEL_PREFIXES: Final = (
    "com.aura.unified-intrinsic.",
    "com.aura.resident-sft.",
    "com.aura.resident-32b-recurrent-grpo",
)
TRAINING_STATE_ROOT: Final = Path.home() / ".aura/state/resident-training"
LAUNCH_AGENTS_ROOT: Final = Path.home() / "Library/LaunchAgents"
RESTARTABLE_CODES: Final = frozenset(
    {
        "detached_launch_failed",
        "detached_resume_failed",
        "attempt_timeout",
        "controller_status_temporarily_unavailable",
        "resident_training_host_busy",
    }
)


class UnifiedResidentControllerError(RuntimeError):
    """Stable fail-closed resident campaign error."""

    def __init__(self, code: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.details = dict(details or {})


def _fail(code: str, **details: Any) -> Never:
    raise UnifiedResidentControllerError(code, details=details)


def _document(value: Any) -> bytes:
    return canonical_bytes(value) + b"\n"


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_canonical(
    path: Path,
    *,
    max_bytes: int = MAX_DOCUMENT_BYTES,
    expected_mode: int | None = None,
) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink():
        _fail("artifact_path_invalid", path=str(path))
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or not 0 < before.st_size <= max_bytes
                or (
                    expected_mode is not None
                    and stat.S_IMODE(before.st_mode) != stat.S_IMODE(expected_mode)
                )
            ):
                _fail("artifact_custody_invalid", path=str(path))
            chunks: list[bytes] = []
            remaining = max_bytes + 1
            while remaining > 0:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise UnifiedResidentControllerError(
            "artifact_unreadable",
            details={"path": str(path)},
        ) from exc
    raw = b"".join(chunks)
    if len(raw) != before.st_size or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        _fail("artifact_changed_while_read", path=str(path))
    try:
        decoded = json.loads(raw.decode("ascii"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise UnifiedResidentControllerError(
            "artifact_json_invalid",
            details={"path": str(path)},
        ) from exc
    if not isinstance(decoded, dict) or raw != _document(decoded):
        _fail("artifact_not_canonical", path=str(path))
    return decoded


def _write_once(path: Path, value: Mapping[str, Any], *, mode: int = 0o400) -> None:
    payload = _document(dict(value))
    ensure_private_directory(path.parent)
    atomic_write_bytes_if_absent(path, payload, mode=mode)
    if _read_canonical(path, expected_mode=mode) != dict(value):
        _fail("immutable_artifact_drift", path=str(path))


def _private_directory(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        _fail("campaign_directory_custody_invalid", path=str(expanded))
    path = expanded.resolve(strict=True)
    observed = path.stat()
    if (
        path.is_symlink()
        or not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or stat.S_IMODE(observed.st_mode) != 0o700
    ):
        _fail("campaign_directory_custody_invalid", path=str(path))
    return path


def _key(path: Path, *, expected_sha256: str) -> bytes:
    if not path.is_absolute() or path.is_symlink():
        _fail("campaign_key_path_invalid")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            before = os.fstat(descriptor)
            chunks: list[bytes] = []
            remaining = 129
            while remaining > 0:
                chunk = os.read(descriptor, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise UnifiedResidentControllerError("campaign_key_unreadable") from exc
    raw = b"".join(chunks)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.geteuid()
        or stat.S_IMODE(before.st_mode) != 0o400
        or len(raw) != 32
        or hashlib.sha256(raw).hexdigest() != expected_sha256
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        _fail("campaign_key_identity_invalid")
    return raw


def _load_config(path: Path) -> dict[str, Any]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        _fail("campaign_config_path_is_symlink")
    path = expanded.resolve(strict=True)
    config = _read_canonical(path, expected_mode=0o400)
    required = {
        "schema",
        "campaign_id",
        "profile",
        "prepared_at",
        "source",
        "model",
        "runtime",
        "dataset",
        "tokenizer",
        "tokenized_dataset",
        "paths",
        "heartbeat_key_sha256",
        "training",
        "training_args",
        "watchdog",
        "launch",
        "claims",
        "config_sha256",
    }
    allowed_keys = (required, required | {"bootstrap"})
    body = {key: value for key, value in config.items() if key != "config_sha256"}
    if (
        set(config) not in allowed_keys
        or config.get("schema") != CONFIG_SCHEMA
        or config.get("config_sha256") != canonical_sha256(body)
        or config.get("profile")
        not in {
            "canary",
            "full",
            "process_action_canary",
            "process_answer_bridge_canary",
            "process_analytic_acquisition",
            "process_canary",
            "process_completion_acquisition",
            "process_family_acquisition",
            "process_neural_acquisition",
            "process_public_transition_acquisition",
            "process_public_transition_direct_acquisition",
            "process_public_transition_extended_acquisition",
            "recovery",
        }
        or not isinstance(config.get("campaign_id"), str)
    ):
        _fail("campaign_config_invalid")
    expected_training = _profile_training(str(config["profile"]))
    if config.get("training") != expected_training or config.get("training_args") != _training_cli(
        expected_training
    ):
        _fail("campaign_training_profile_drift")
    paths = config.get("paths")
    base_path_keys = {
        "workspace_root",
        "campaign_root",
        "inputs",
        "training_output",
        "dataset",
        "tokenized_dataset",
        "detached_attempts",
        "heartbeat_key",
    }
    bootstrap_profiles = {
        "process_action_canary",
        "process_answer_bridge_canary",
        "process_analytic_acquisition",
        "process_completion_acquisition",
        "process_family_acquisition",
        "process_neural_acquisition",
        "process_public_transition_acquisition",
        "process_public_transition_direct_acquisition",
        "process_public_transition_extended_acquisition",
        "recovery",
    }
    expected_path_keys = (
        base_path_keys | {"bootstrap_output"}
        if config["profile"] in bootstrap_profiles
        else base_path_keys
    )
    if not isinstance(paths, dict) or set(paths) != expected_path_keys:
        _fail("campaign_paths_invalid")
    root = _private_directory(Path(paths["campaign_root"]))
    inputs = _private_directory(Path(paths["inputs"]))
    output = _private_directory(Path(paths["training_output"]))
    attempts = _private_directory(Path(paths["detached_attempts"]))
    expected_paths = {
        "campaign_root": root,
        "inputs": root / "inputs",
        "training_output": output,
        "dataset": inputs / "dataset.json",
        "tokenized_dataset": inputs / "tokenized_dataset.json",
        "detached_attempts": attempts,
        "heartbeat_key": root / "heartbeat.key",
        **(
            {"bootstrap_output": inputs / "bootstrap-output"}
            if config["profile"] in bootstrap_profiles
            else {}
        ),
    }
    for name, expected in expected_paths.items():
        observed = Path(paths[name]).expanduser().resolve(strict=True)
        if observed != expected:
            _fail("campaign_path_binding_drift", role=name)
    if path != root / "campaign.json":
        _fail("campaign_config_path_drift")
    bootstrap = config.get("bootstrap")
    if config["profile"] in bootstrap_profiles:
        if not isinstance(bootstrap, dict):
            _fail("campaign_bootstrap_invalid")
        bootstrap_body = {
            key: value for key, value in bootstrap.items() if key != "bootstrap_sha256"
        }
        if (
            set(bootstrap)
            != {
                "schema",
                "stem",
                "output",
                "parent_step",
                "parent_checkpoint_sha256",
                "parent_receipt_sha256",
                "parent_identity_sha256",
                "bootstrap_sha256",
            }
            or bootstrap.get("schema") != "aura.unified_intrinsic.bootstrap_input.v1"
            or bootstrap.get("bootstrap_sha256") != canonical_sha256(bootstrap_body)
            or bootstrap.get("output") != paths["bootstrap_output"]
            or type(bootstrap.get("parent_step")) is not int
            or int(bootstrap["parent_step"]) < 0
            or any(
                not _is_sha(bootstrap.get(name))
                for name in (
                    "parent_checkpoint_sha256",
                    "parent_receipt_sha256",
                    "parent_identity_sha256",
                )
            )
        ):
            _fail("campaign_bootstrap_invalid")
        try:
            selected = resolve_checkpoint_generation(
                Path(paths["bootstrap_output"]),
                stem=str(bootstrap.get("stem") or ""),
                required=True,
            )
        except (OSError, UnifiedCheckpointError, ValueError) as exc:
            raise UnifiedResidentControllerError("campaign_bootstrap_checkpoint_invalid") from exc
        if selected is None:  # pragma: no cover - required=True is authoritative
            _fail("campaign_bootstrap_checkpoint_unavailable")
        parent_identity = selected.receipt.get("identity")
        if (
            selected.receipt.get("step") != bootstrap["parent_step"]
            or selected.receipt.get("checkpoint_sha256") != bootstrap["parent_checkpoint_sha256"]
            or selected.receipt.get("receipt_sha256") != bootstrap["parent_receipt_sha256"]
            or not isinstance(parent_identity, dict)
            or parent_identity.get("identity_sha256") != bootstrap["parent_identity_sha256"]
        ):
            _fail("campaign_bootstrap_checkpoint_drift")
    elif bootstrap is not None:
        _fail("campaign_bootstrap_not_permitted")
    label = f"com.aura.unified-intrinsic.{config['campaign_id']}"
    if config.get("launch") != {
        "label": label,
        "launchd_required": True,
        "trainer_caffeinate_required": True,
        "immutable_target_command": True,
    }:
        _fail("campaign_launch_policy_invalid")
    watchdog = config.get("watchdog")
    if (
        not isinstance(watchdog, dict)
        or set(watchdog)
        != {
            "poll_interval_s",
            "heartbeat_stale_s",
            "attempt_timeout_s",
            "max_attempts",
            "max_consecutive_no_progress",
            "retry_backoff_s",
        }
        or any(
            isinstance(watchdog[name], bool)
            or not isinstance(watchdog[name], (int, float))
            or not math.isfinite(float(watchdog[name]))
            or float(watchdog[name]) <= 0.0
            for name in (
                "poll_interval_s",
                "heartbeat_stale_s",
                "attempt_timeout_s",
                "retry_backoff_s",
            )
        )
        or type(watchdog["max_attempts"]) is not int
        or type(watchdog["max_consecutive_no_progress"]) is not int
        or not 1 <= watchdog["max_attempts"] <= 32
        or not 1 <= watchdog["max_consecutive_no_progress"] <= 4
    ):
        _fail("campaign_watchdog_invalid")
    _key(
        Path(paths["heartbeat_key"]),
        expected_sha256=str(config.get("heartbeat_key_sha256") or ""),
    )
    return config


def _bridge(config: Mapping[str, Any]) -> str:
    value = str(config["training"]["bridge"])
    return {"assistant_answer": "\n\nFINAL_ANSWER: "}.get(value, value)


def verify_package(config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source"]
    source_root = Path(source["git"]["root"]).expanduser().resolve(strict=True)
    verify_source_git_identity(source_root, source["git"])
    verify_source_manifest(source_root, source["manifest"])
    verify_model_manifest(config["model"])
    observed_runtime = runtime_identity()
    if observed_runtime != config["runtime"]:
        _fail("campaign_runtime_identity_drift")
    dataset_path = Path(config["paths"]["dataset"])
    train, holdout = load_source_dataset(dataset_path)
    dataset_identity = freeze_source_dataset(dataset_path, train, holdout)
    if dataset_identity != config["dataset"]:
        _fail("campaign_dataset_identity_drift")
    model_root = Path(config["model"]["root"])
    tokenizer = load_resident_bootstrap_tokenizer(model_root)
    tokenizer_identity = resident_bootstrap_tokenizer_identity(model_root, tokenizer)
    if tokenizer_identity != config["tokenizer"]:
        _fail("campaign_tokenizer_identity_drift")
    tokenized_identity = verify_tokenized_dataset(
        Path(config["paths"]["tokenized_dataset"]),
        tokenizer,
        train,
        holdout,
        bridge=_bridge(config),
        dataset_identity=dataset_identity,
        tokenizer_identity_sha256=tokenizer_identity["identity_sha256"],
    )
    if tokenized_identity != config["tokenized_dataset"]:
        _fail("campaign_tokenized_dataset_identity_drift")
    return {
        "source_commit": source["git"]["commit"],
        "source_manifest_sha256": source["manifest"]["manifest_sha256"],
        "model_manifest_sha256": config["model"]["manifest_sha256"],
        "runtime_identity_sha256": observed_runtime["identity_sha256"],
        "dataset_identity_sha256": dataset_identity["identity_sha256"],
        "tokenizer_identity_sha256": tokenizer_identity["identity_sha256"],
        "tokenized_dataset_identity_sha256": tokenized_identity["identity_sha256"],
        "bootstrap_sha256": (
            config["bootstrap"]["bootstrap_sha256"]
            if isinstance(config.get("bootstrap"), dict)
            else None
        ),
    }


def _owner_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for owner in get_model_lane_controller().owner_observations():
        rows.append(
            {
                "owner_id": owner.owner_id,
                "model_path": owner.model_path,
                "purpose": owner.purpose,
                "declared_gb": owner.declared_gb,
                "observed_gb": owner.observed_gb,
                "pid": owner.process.pid,
                "started_at": owner.process.started_at,
                "preemptible": owner.preemptible,
                "metadata": dict(owner.metadata),
            }
        )
    return rows


def _require_empty_model_lane() -> list[dict[str, Any]]:
    owners = _owner_rows()
    if owners:
        _fail("resident_model_lane_occupied", owners=owners)
    return owners


@contextmanager
def _file_lock(
    path: Path,
    *,
    busy_code: str,
    wait_s: float = 0.0,
) -> Iterator[int]:
    ensure_private_directory(path.parent)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    acquired = False
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode) or observed.st_uid != os.geteuid():
            _fail("controller_lock_identity_invalid")
        os.fchmod(descriptor, 0o600)
        deadline = time.monotonic() + max(0.0, wait_s)
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise UnifiedResidentControllerError(busy_code) from exc
                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        yield descriptor
    finally:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def _host_lease(
    config: Mapping[str, Any],
    *,
    wait_s: float = 0.0,
) -> Iterator[dict[str, Any]]:
    with _file_lock(
        TRAINING_STATE_ROOT / "host.lock",
        busy_code="resident_training_host_busy",
        wait_s=wait_s,
    ) as descriptor:
        body = {
            "schema": "aura.resident_training_host_lease.v1",
            "active": True,
            "label": config["launch"]["label"],
            "config_sha256": config["config_sha256"],
            "pid": os.getpid(),
            "acquired_at_unix_ns": time.time_ns(),
        }
        lease = {**body, "lease_sha256": canonical_sha256(body)}
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, _document(lease))
        os.fsync(descriptor)
        try:
            yield lease
        finally:
            released_body = {
                **body,
                "active": False,
                "released_at_unix_ns": time.time_ns(),
            }
            released = {
                **released_body,
                "lease_sha256": canonical_sha256(released_body),
            }
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, _document(released))
            os.fsync(descriptor)


def _status_signature(body: Mapping[str, Any], key: bytes) -> str:
    return hmac.new(key, canonical_bytes(dict(body)), hashlib.sha256).hexdigest()


def _read_status(config: Mapping[str, Any], *, required: bool) -> dict[str, Any] | None:
    path = Path(config["paths"]["campaign_root"]) / "controller-status.json"
    if not path.exists():
        if required:
            _fail("controller_status_unavailable")
        return None
    status = _read_canonical(path)
    body = {name: value for name, value in status.items() if name != "hmac_sha256"}
    key = _key(
        Path(config["paths"]["heartbeat_key"]),
        expected_sha256=config["heartbeat_key_sha256"],
    )
    signature = status.get("hmac_sha256")
    if (
        status.get("schema") != STATUS_SCHEMA
        or status.get("config_sha256") != config["config_sha256"]
        or type(status.get("sequence")) is not int
        or status["sequence"] < 1
        or not isinstance(signature, str)
        or not hmac.compare_digest(signature, _status_signature(body, key))
    ):
        _fail("controller_status_authentication_failed")
    return status


def _publish_status(
    config: Mapping[str, Any],
    state: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    previous = _read_status(config, required=False)
    body = {
        "schema": STATUS_SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_sha256": config["config_sha256"],
        "sequence": 1 if previous is None else int(previous["sequence"]) + 1,
        "state": state,
        "controller_pid": os.getpid(),
        "controller_start_token": detached._process_start_token(os.getpid()),  # noqa: SLF001
        "heartbeat_at": time.time(),
        "details": dict(details),
    }
    key = _key(
        Path(config["paths"]["heartbeat_key"]),
        expected_sha256=config["heartbeat_key_sha256"],
    )
    status = {**body, "hmac_sha256": _status_signature(body, key)}
    atomic_write_bytes(
        Path(config["paths"]["campaign_root"]) / "controller-status.json",
        _document(status),
        mode=0o600,
    )
    return status


def _inspect_status(config: Mapping[str, Any]) -> dict[str, Any]:
    status = _read_status(config, required=True)
    assert status is not None
    pid = int(status.get("controller_pid") or 0)
    token = str(status.get("controller_start_token") or "")
    liveness = detached._identity_state(pid, token)  # noqa: SLF001
    terminal = status.get("state") in {"completed", "failed"}
    return {
        "schema": "aura.unified_intrinsic.controller_inspection.v1",
        "authenticated_status": status,
        "controller_liveness": liveness,
        "effective_state": (status["state"] if terminal or liveness == "alive" else "stale"),
        "claims_supported": (
            ["authenticated_terminal_controller_status"]
            if terminal
            else ["authenticated_live_controller_status"]
            if liveness == "alive"
            else []
        ),
    }


def _publish_failure_status(
    config_path: Path,
    *,
    error: str,
    details: Mapping[str, Any],
) -> None:
    try:
        config = _load_config(config_path)
        _publish_status(
            config,
            "failed",
            {
                "error": error,
                "details": dict(details),
                "restartable": error in RESTARTABLE_CODES,
            },
        )
    except Exception as exc:  # noqa: BLE001 - original controller failure remains primary
        print(
            "unified resident controller could not publish secondary failure status: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return


def _launch_contract(
    config: Mapping[str, Any],
    config_path: Path,
) -> tuple[dict[str, Any], Path, bytes, dict[str, Any]]:
    label = str(config["launch"]["label"])
    python = str(config["runtime"]["interpreter"]["executable"])
    source_root = Path(config["source"]["git"]["root"])
    root = Path(config["paths"]["campaign_root"])
    payload: dict[str, Any] = {
        "Label": label,
        "ProgramArguments": [
            python,
            str(source_root / "tools/run_unified_intrinsic_resident_campaign.py"),
            "run",
            "--config",
            str(config_path),
            "--launchd-supervised",
        ],
        "WorkingDirectory": str(root),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 30,
        "ProcessType": "Background",
        "StandardOutPath": str(root / "controller.log"),
        "StandardErrorPath": str(root / "controller.log"),
    }
    behavior = config["runtime"].get("behavior_environment")
    if isinstance(behavior, dict):
        variables = {name: value for name, value in behavior.items() if isinstance(value, str)}
        if variables:
            payload["EnvironmentVariables"] = variables
    plist = plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)
    plist_path = LAUNCH_AGENTS_ROOT / f"{label}.plist"
    body = {
        "schema": LAUNCH_INTENT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_sha256": config["config_sha256"],
        "label": label,
        "plist_path": str(plist_path),
        "plist_sha256": hashlib.sha256(plist).hexdigest(),
        "program_arguments": payload["ProgramArguments"],
        "working_directory": payload["WorkingDirectory"],
        "environment_variables": payload.get("EnvironmentVariables", {}),
    }
    intent = {**body, "intent_sha256": canonical_sha256(body)}
    return payload, plist_path, plist, intent


def _verified_launch_intent(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(config["paths"]["campaign_root"])
    config_path = root / "campaign.json"
    _payload, plist_path, expected_plist, expected = _launch_contract(
        config,
        config_path,
    )
    intent = _read_canonical(root / "launch-intent.json", expected_mode=0o400)
    if intent != expected:
        _fail("launch_intent_drift")
    if plist_path.is_symlink():
        _fail("launchd_plist_is_symlink")
    try:
        descriptor = os.open(
            plist_path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            before = os.fstat(descriptor)
            observed_plist = os.read(descriptor, len(expected_plist) + 1)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise UnifiedResidentControllerError("launchd_plist_unavailable") from exc
    if (
        observed_plist != expected_plist
        or not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.geteuid()
        or stat.S_IMODE(before.st_mode) != 0o600
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        _fail("launchd_plist_identity_drift")
    return intent


def _launchd_job(label: str) -> dict[str, Any]:
    target = f"gui/{os.getuid()}/{label}"
    result = subprocess.run(
        ["/bin/launchctl", "print", target],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    if result.returncode != 0:
        _fail("launchd_job_unavailable")
    pid: int | None = None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("pid = "):
            try:
                pid = int(stripped.removeprefix("pid = "))
            except ValueError:
                _fail("launchd_job_pid_invalid")
            break
    if pid is None or pid <= 1:
        _fail("launchd_job_pid_missing")
    return {"target": target, "pid": pid}


def _verify_launchd(config: Mapping[str, Any], *, supervised: bool) -> dict[str, Any]:
    if not supervised:
        _fail("launchd_supervision_required")
    intent = _verified_launch_intent(config)
    job = _launchd_job(str(config["launch"]["label"]))
    if job["pid"] != os.getpid():
        _fail("launchd_controller_pid_mismatch", observed=job["pid"])
    start_token = detached._process_start_token(job["pid"])  # noqa: SLF001
    if not start_token:
        _fail("launchd_controller_identity_unavailable")
    return {
        "launchd_target": job["target"],
        "controller_pid": job["pid"],
        "controller_start_token": start_token,
        "launch_intent_sha256": intent["intent_sha256"],
        "plist_sha256": intent["plist_sha256"],
    }


def _checkpoint_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(config["paths"]["training_output"])
    try:
        resolved = resolve_checkpoint_generation(
            output,
            stem="checkpoint_latest",
            required=False,
        )
    except (OSError, UnifiedCheckpointError, ValueError) as exc:
        raise UnifiedResidentControllerError("checkpoint_state_invalid") from exc
    if resolved is None:
        try:
            unpointed = unpointed_checkpoint_inventory(output)
        except UnifiedCheckpointError as exc:
            raise UnifiedResidentControllerError("checkpoint_state_invalid") from exc
        snapshot = {
            "present": False,
            "step": 0,
            "checkpoint_sha256": None,
            "receipt_sha256": None,
            "complete": False,
            "training_receipt": None,
        }
        if any(unpointed.values()):
            snapshot["ignored_unpointed"] = unpointed
        return snapshot
    receipt = resolved.receipt
    identity = receipt.get("identity")
    model_identity = identity.get("model") if isinstance(identity, dict) else None
    campaign_binding = identity.get("campaign_binding") if isinstance(identity, dict) else None
    if (
        not isinstance(identity, dict)
        or identity.get("dataset") != config["dataset"]
        or identity.get("tokenizer") != config["tokenizer"]
        or identity.get("tokenized_dataset") != config["tokenized_dataset"]
        or model_identity != trainer_model_identity_from_manifest(config["model"])
        or campaign_binding != campaign_checkpoint_binding(config)
    ):
        _fail("checkpoint_campaign_binding_drift")
    summary: dict[str, Any] | None = None
    complete = False
    training_receipt_path = output / "training_receipt.json"
    if training_receipt_path.exists():
        try:
            training_receipt = _read_canonical(training_receipt_path)
        except UnifiedResidentControllerError as exc:
            summary = {"binding": "ignored_non_authoritative", "reason": exc.code}
        else:
            body = {
                key: value for key, value in training_receipt.items() if key != "receipt_sha256"
            }
            bound = (
                training_receipt.get("receipt_sha256") == canonical_sha256(body)
                and training_receipt.get("steps") == receipt.get("step")
                and training_receipt.get("latest_checkpoint", {}).get("checkpoint_sha256")
                == receipt.get("checkpoint_sha256")
            )
            summary = {
                "binding": "authoritative_checkpoint" if bound else "ignored_stale",
                "receipt_sha256": training_receipt.get("receipt_sha256"),
                "steps": training_receipt.get("steps"),
                "complete": training_receipt.get("complete"),
                "halt_reason": training_receipt.get("halt_reason"),
            }
            complete = bool(bound and training_receipt.get("complete") is True)
    return {
        "present": True,
        "step": int(receipt["step"]),
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "generation": resolved.generation_dir.name,
        "complete": complete,
        "training_receipt": summary,
    }


def _checkpoint_hint(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read the tiny pointer for progress telemetry; final claims rehash weights."""

    path = Path(config["paths"]["training_output"]) / "checkpoint_latest_pointer.json"
    if not path.exists():
        return {"present": False, "step": 0, "authoritative": False}
    try:
        pointer = _read_canonical(path, max_bytes=64 * 1024)
    except UnifiedResidentControllerError as exc:
        return {
            "present": None,
            "step": None,
            "authoritative": False,
            "error": exc.code,
        }
    return {
        "present": True,
        "step": pointer.get("step"),
        "generation": pointer.get("checkpoint"),
        "authoritative": False,
    }


def _trainer_command(
    config_path: Path,
    config: Mapping[str, Any],
    *,
    invocation_steps: int | None,
) -> list[str]:
    source_root = Path(config["source"]["git"]["root"])
    python = str(config["runtime"]["interpreter"]["executable"])
    run_dir = Path(config["paths"]["training_output"])
    pid = "{pid}"
    campaign_binding = json.dumps(
        campaign_checkpoint_binding(config),
        sort_keys=True,
        separators=(",", ":"),
    )
    command = [
        python,
        str(source_root / "tools/train_unified_intrinsic_recurrence.py"),
        "--model",
        str(config["model"]["root"]),
        "--expected-model-identity-sha256",
        trainer_model_identity_from_manifest(config["model"])["identity_sha256"],
        "--exclusive-model-lane",
        "--campaign-binding-json",
        campaign_binding,
        "--out-dir",
        str(run_dir),
        "--dataset",
        str(config["paths"]["dataset"]),
        "--tokenized-dataset",
        str(config["paths"]["tokenized_dataset"]),
        "--resume-if-available",
        "--resource-stage-path",
        str(run_dir / f"resource-stage-{pid}.json"),
        "--resource-startup-lethal-mb",
        str(STARTUP_LETHAL_MB),
        "--resource-steady-lethal-mb",
        str(STEADY_LETHAL_MB),
        "--resource-guard-timeout-s",
        str(PRELOAD_TIMEOUT_S),
        "--preload-ready-path",
        str(run_dir / f"preload-ready-{pid}.json"),
        "--preload-release-path",
        str(run_dir / f"preload-release-{pid}.json"),
        "--preload-key-path",
        str(config["paths"]["heartbeat_key"]),
        "--preload-config-sha256",
        str(config["config_sha256"]),
        *[str(value) for value in config["training_args"]],
    ]
    if config["profile"] in {
        "process_action_canary",
        "process_answer_bridge_canary",
        "process_analytic_acquisition",
        "process_completion_acquisition",
        "process_family_acquisition",
        "process_neural_acquisition",
        "process_public_transition_acquisition",
        "process_public_transition_direct_acquisition",
        "process_public_transition_extended_acquisition",
        "recovery",
    }:
        bootstrap = config["bootstrap"]
        command.extend(
            (
                "--bootstrap-output-dir",
                str(config["paths"]["bootstrap_output"]),
                "--bootstrap-stem",
                str(bootstrap["stem"]),
            )
        )
    if invocation_steps is not None:
        command.extend(("--max-invocation-steps", str(invocation_steps)))
    return command


def _main_launch_args(
    config_path: Path,
    config: Mapping[str, Any],
    run_dir: Path,
    *,
    invocation_steps: int | None,
    resume: bool,
) -> list[str]:
    source_root = Path(config["source"]["git"]["root"])
    python = str(config["runtime"]["interpreter"]["executable"])
    trainer = _trainer_command(
        config_path,
        config,
        invocation_steps=invocation_steps,
    )
    barrier = [
        python,
        str(source_root / "tools/unified_intrinsic_preload_barrier.py"),
        "--ready",
        str(Path(config["paths"]["training_output"]) / "preload-ready-{pid}.json"),
        "--release",
        str(Path(config["paths"]["training_output"]) / "preload-release-{pid}.json"),
        "--key",
        str(config["paths"]["heartbeat_key"]),
        "--config-sha256",
        str(config["config_sha256"]),
        "--timeout",
        str(PRELOAD_TIMEOUT_S),
        "--",
        *trainer,
    ]
    verifier = json.dumps(
        [
            python,
            str(source_root / "tools/verify_unified_intrinsic_resume.py"),
            "--config",
            str(config_path),
        ],
        separators=(",", ":"),
    )
    args = [
        "launch",
        "--run-dir",
        str(run_dir),
        "--name",
        f"{config['campaign_id']}-main-{run_dir.name}",
        "--cwd",
        str(config["paths"]["campaign_root"]),
        "--timeout",
        str(config["watchdog"]["attempt_timeout_s"]),
        "--resume-contract",
        "target_checkpoint",
        "--resume-verifier-json",
        verifier,
        "--execution-output-root",
        str(config["paths"]["training_output"]),
    ]
    if resume:
        args.append("--resume")
    return [*args, *barrier]


def _invoke_detached(arguments: list[str], *, failure: str) -> None:
    try:
        result = int(detached.main(arguments))
    except SystemExit as exc:
        raise UnifiedResidentControllerError(failure) from exc
    if result != 0:
        _fail(failure)


def _target_identity(status: Mapping[str, Any]) -> tuple[int, str, int]:
    pid = int(status.get("child_pid") or 0)
    token = str(status.get("child_start_token") or "")
    supervisor_pid = int(status.get("supervisor_pid") or 0)
    if pid <= 1 or supervisor_pid <= 1 or not token:
        _fail("detached_target_identity_unavailable")
    if detached._identity_state(pid, token) != "alive":  # noqa: SLF001
        _fail("detached_target_identity_not_live")
    return pid, token, supervisor_pid


def _verify_caffeinate(target_pid: int, supervisor_pid: int) -> dict[str, Any]:
    expected = ("/usr/bin/caffeinate", "-i", "-w", str(target_pid))
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        matches = [
            process
            for process in get_resource_observer().processes()
            if process.ppid == supervisor_pid and tuple(process.cmdline) == expected
        ]
        if len(matches) == 1:
            process = matches[0]
            token = detached._process_start_token(process.pid)  # noqa: SLF001
            if token:
                return {
                    "pid": process.pid,
                    "start_token": token,
                    "parent_pid": process.ppid,
                    "command": list(expected),
                }
        time.sleep(0.1)
    _fail("trainer_bound_caffeinate_missing")


def _stable_last_ring_entry(
    path: Path,
    *,
    required_stage: str | None = "startup",
) -> tuple[dict[str, Any], str]:
    if not path.is_absolute() or path.is_symlink():
        _fail("sentinel_ring_path_invalid")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            before = os.fstat(descriptor)
            if not 0 < before.st_size <= 16 * 1024 * 1024:
                _fail("sentinel_ring_size_invalid")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining > 0:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise UnifiedResidentControllerError("sentinel_ring_unreadable") from exc
    raw = b"".join(chunks)
    if (
        remaining
        or not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.geteuid()
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        _fail("sentinel_ring_custody_invalid")
    lines = [line for line in raw.splitlines() if line]
    if not lines:
        _fail("sentinel_ring_empty")
    try:
        entry = json.loads(lines[-1].decode("ascii"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise UnifiedResidentControllerError("sentinel_ring_entry_invalid") from exc
    stage = entry.get("guard_stage") if isinstance(entry, dict) else None
    lethal_mb = float(entry.get("active_lethal_mb") or 0.0) if isinstance(entry, dict) else 0.0
    if (
        not isinstance(entry, dict)
        or stage not in {"startup", "steady", "compute", "draining"}
        or (required_stage is not None and stage != required_stage)
        or (required_stage == "startup" and lethal_mb != STARTUP_LETHAL_MB)
        or (required_stage is None and lethal_mb <= 0.0)
        or int(entry.get("procs") or 0) < 1
    ):
        _fail("sentinel_startup_evidence_invalid")
    return entry, hashlib.sha256(lines[-1] + b"\n").hexdigest()


def _sentinel_launch_args(
    config: Mapping[str, Any],
    run_dir: Path,
    *,
    target_pid: int,
) -> list[str]:
    source_root = Path(config["source"]["git"]["root"])
    python = str(config["runtime"]["interpreter"]["executable"])
    sentinel_dir = run_dir / f"sentinel-{target_pid}"
    return [
        "launch",
        "--run-dir",
        str(sentinel_dir),
        "--name",
        f"{config['campaign_id']}-sentinel-{target_pid}",
        "--cwd",
        str(config["paths"]["campaign_root"]),
        "--timeout",
        str(float(config["watchdog"]["attempt_timeout_s"]) + 300.0),
        python,
        str(source_root / "tools/memory_sentinel.py"),
        "--pid",
        str(target_pid),
        "--lethal-mb",
        str(STEADY_LETHAL_MB),
        "--startup-lethal-mb",
        str(STARTUP_LETHAL_MB),
        "--steady-marker",
        str(Path(config["paths"]["training_output"]) / f"resource-stage-{target_pid}.json"),
        "--interval",
        "1.0",
        "--ring",
        str(sentinel_dir / "ring.jsonl"),
        "--ring-window-seconds",
        "7200",
        "--tombstone-dir",
        str(sentinel_dir / "tombstones"),
    ]


def _ensure_preload_release(
    config: Mapping[str, Any],
    run_dir: Path,
    status: Mapping[str, Any],
    *,
    invocation_steps: int | None,
) -> dict[str, Any]:
    target_pid, target_token, supervisor_pid = _target_identity(status)
    output = Path(config["paths"]["training_output"])
    ready = output / f"preload-ready-{target_pid}.json"
    release = output / f"preload-release-{target_pid}.json"
    expected_trainer = [
        value.replace("{pid}", str(target_pid))
        for value in _trainer_command(
            Path(config["paths"]["campaign_root"]) / "campaign.json",
            config,
            invocation_steps=invocation_steps,
        )
    ]
    expected_command_sha = command_sha256(expected_trainer)
    if release.exists():
        verified = verify_release(
            release,
            ready_path=ready,
            key_path=Path(config["paths"]["heartbeat_key"]),
            config_sha256=config["config_sha256"],
            expected_target_pid=target_pid,
            expected_target_start_token=target_token,
            expected_command_sha256=expected_command_sha,
            require_fresh=False,
        )
        caffeinate = _verify_caffeinate(target_pid, supervisor_pid)
        return {
            "target_pid": target_pid,
            "target_start_token": target_token,
            "sentinel_run_dir": str(run_dir / f"sentinel-{target_pid}"),
            "sentinel_pid": verified["sentinel_pid"],
            "sentinel_start_token": verified["sentinel_start_token"],
            "caffeinate": caffeinate,
            "release": verified,
        }
    deadline = time.monotonic() + PRELOAD_TIMEOUT_S
    while time.monotonic() < deadline and not ready.exists():
        if detached._identity_state(target_pid, target_token) != "alive":  # noqa: SLF001
            _fail("preload_target_exited_before_ready")
        time.sleep(0.1)
    if not ready.exists():
        _fail("preload_ready_timeout")
    sentinel_dir = run_dir / f"sentinel-{target_pid}"
    if not (sentinel_dir / detached.PLAN_FILE).exists():
        _invoke_detached(
            _sentinel_launch_args(config, run_dir, target_pid=target_pid),
            failure="sentinel_detached_launch_failed",
        )
    ring = sentinel_dir / "ring.jsonl"
    sentinel_status: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        try:
            sentinel_status = detached._status(sentinel_dir)  # noqa: SLF001
        except (detached.DetachedStepError, OSError, ValueError):
            time.sleep(0.1)
            continue
        if sentinel_status.get("terminal") is True:
            _fail("sentinel_exited_before_preload_release")
        if ring.exists() and int(sentinel_status.get("child_pid") or 0) > 1:
            break
        time.sleep(0.1)
    if sentinel_status is None:
        _fail("sentinel_startup_timeout")
    sentinel_pid = int(sentinel_status.get("child_pid") or 0)
    sentinel_token = str(sentinel_status.get("child_start_token") or "")
    if (
        sentinel_pid <= 1
        or not sentinel_token
        or detached._identity_state(sentinel_pid, sentinel_token) != "alive"  # noqa: SLF001
    ):
        _fail("sentinel_identity_invalid")
    ring_entry, ring_sha256 = _stable_last_ring_entry(ring)
    caffeinate = _verify_caffeinate(target_pid, supervisor_pid)
    pressure = host_pressure()
    if pressure.get("available") is not True or pressure.get("under_pressure") is not False:
        _fail("host_pressure_denied_preload", host_pressure=pressure)
    released = publish_release(
        release,
        ready_path=ready,
        key_path=Path(config["paths"]["heartbeat_key"]),
        sentinel_pid=sentinel_pid,
        sentinel_start_token=sentinel_token,
        sentinel_ring_entry_sha256=ring_sha256,
        host_pressure=pressure,
        expected_target_pid=target_pid,
        expected_target_start_token=target_token,
        expected_command_sha256=expected_command_sha,
    )
    return {
        "target_pid": target_pid,
        "target_start_token": target_token,
        "sentinel_run_dir": str(sentinel_dir),
        "sentinel_pid": sentinel_pid,
        "sentinel_start_token": sentinel_token,
        "sentinel_ring_entry": ring_entry,
        "sentinel_ring_entry_sha256": ring_sha256,
        "caffeinate": caffeinate,
        "release": released,
    }


def _stop_detached(run_dir: Path, *, code: str) -> None:
    try:
        request = detached._stop(run_dir)  # noqa: SLF001
    except (detached.DetachedStepError, OSError, ValueError) as exc:
        raise UnifiedResidentControllerError(code) from exc
    if request.get("stopped") is not True and request.get("reason") not in {
        "already_terminal",
        "supervisor_not_alive",
        "supervisor_not_reserved",
    }:
        _fail(code)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        try:
            status = detached._status(run_dir)  # noqa: SLF001
        except (detached.DetachedStepError, OSError, ValueError) as exc:
            raise UnifiedResidentControllerError(code) from exc
        child_pid = int(status.get("child_pid") or 0)
        child_group = int(status.get("child_process_group_id") or 0)
        child_start = str(status.get("child_start_token") or "")
        containment_token = str(status.get("containment_token") or "")
        receipt = status.get("receipt")
        if (
            status.get("terminal") is True
            and isinstance(receipt, dict)
            and receipt.get("containment_verified") is True
            and status.get("child_state") == "dead"
        ):
            if child_group > 1 and detached._process_group_exists(child_group):  # noqa: SLF001
                _fail(code)
            if containment_token and detached._tagged_processes(containment_token):  # noqa: SLF001
                _fail(code)
            return
        if status.get("supervisor_state") == "dead":
            if child_pid <= 1:
                return
            target = {
                "child_pid": child_pid,
                "child_process_group_id": child_group,
                "child_start_token": child_start,
                "containment_token": containment_token,
            }
            try:
                detached._terminate_stale_target(target)  # noqa: SLF001
                child_state = detached._identity_state(child_pid, child_start)  # noqa: SLF001
                group_live = detached._process_group_exists(child_group)  # noqa: SLF001
                tagged = detached._tagged_processes(containment_token)  # noqa: SLF001
            except (detached.DetachedStepError, OSError, ValueError) as exc:
                raise UnifiedResidentControllerError(code) from exc
            if child_state != "dead" or group_live or tagged:
                _fail(code)
            return
        time.sleep(0.1)
    _fail(code)


def _sentinel_is_live(evidence: Mapping[str, Any]) -> bool:
    pid = int(evidence.get("sentinel_pid") or 0)
    token = str(evidence.get("sentinel_start_token") or "")
    run_dir = Path(str(evidence.get("sentinel_run_dir") or ""))
    if pid <= 1 or not token or not run_dir.is_absolute():
        return False
    if detached._identity_state(pid, token) != "alive":  # noqa: SLF001
        return False
    try:
        status = detached._status(run_dir)  # noqa: SLF001
    except (detached.DetachedStepError, OSError, ValueError):
        return False
    if not (
        status.get("terminal") is not True
        and int(status.get("child_pid") or 0) == pid
        and status.get("child_start_token") == token
    ):
        return False
    try:
        ring_entry, _ring_sha256 = _stable_last_ring_entry(
            run_dir / "ring.jsonl",
            required_stage=None,
        )
        sampled_at = float(ring_entry.get("at") or 0.0)
    except (OSError, TypeError, ValueError, UnifiedResidentControllerError):
        return False
    return 0.0 <= time.time() - sampled_at <= 30.0


def _caffeinate_is_live(evidence: Mapping[str, Any]) -> bool:
    caffeinate = evidence.get("caffeinate")
    if not isinstance(caffeinate, Mapping):
        return False
    pid = int(caffeinate.get("pid") or 0)
    start_token = str(caffeinate.get("start_token") or "")
    parent_pid = int(caffeinate.get("parent_pid") or 0)
    command = caffeinate.get("command")
    if (
        pid <= 1
        or parent_pid <= 1
        or not start_token
        or not isinstance(command, list)
        or any(not isinstance(value, str) for value in command)
        or detached._identity_state(pid, start_token) != "alive"  # noqa: SLF001
    ):
        return False
    return any(
        process.pid == pid
        and process.ppid == parent_pid
        and tuple(process.cmdline) == tuple(command)
        for process in get_resource_observer().processes()
    )


def _target_is_live(evidence: Mapping[str, Any]) -> bool:
    pid = int(evidence.get("target_pid") or 0)
    token = str(evidence.get("target_start_token") or "")
    return (
        pid > 1 and bool(token) and detached._identity_state(pid, token) == "alive"  # noqa: SLF001
    )


def _await_detached_terminal_handoff(
    run_dir: Path,
    *,
    timeout_s: float = DETACHED_TERMINAL_HANDOFF_TIMEOUT_S,
) -> dict[str, Any] | None:
    """Allow a clean child exit to become a signed detached terminal receipt."""

    if timeout_s < 0.0:
        raise ValueError("detached terminal handoff timeout must not be negative")
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            status = detached._status(run_dir)  # noqa: SLF001
        except (detached.DetachedStepError, OSError, ValueError):
            status = None
        if isinstance(status, dict) and status.get("terminal") is True:
            return dict(status)
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.1)


def _monitor_attempt(
    config_path: Path,
    config: Mapping[str, Any],
    run_dir: Path,
    *,
    invocation_steps: int | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not (run_dir / detached.PLAN_FILE).exists():
        _require_empty_model_lane()
        _invoke_detached(
            _main_launch_args(
                config_path,
                config,
                run_dir,
                invocation_steps=invocation_steps,
                resume=False,
            ),
            failure="detached_launch_failed",
        )
    deadline = time.monotonic() + float(config["watchdog"]["attempt_timeout_s"]) + 120.0
    last_sequence: Any = None
    heartbeat_progress_at = time.monotonic()
    resume_used = False
    release_evidence: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        try:
            status = detached._status(run_dir)  # noqa: SLF001
        except (detached.DetachedStepError, OSError, ValueError):
            time.sleep(float(config["watchdog"]["poll_interval_s"]))
            continue
        if status.get("terminal") is True:
            return dict(status), release_evidence
        if status.get("completion_indeterminate") is True:
            if resume_used:
                _fail("detached_resume_failed")
            _invoke_detached(
                _main_launch_args(
                    config_path,
                    config,
                    run_dir,
                    invocation_steps=invocation_steps,
                    resume=True,
                ),
                failure="detached_resume_failed",
            )
            resume_used = True
            release_evidence = None
            continue
        if int(status.get("child_pid") or 0) > 1:
            current_pid = int(status["child_pid"])
            if release_evidence is None or release_evidence.get("target_pid") != current_pid:
                try:
                    release_evidence = _ensure_preload_release(
                        config,
                        run_dir,
                        status,
                        invocation_steps=invocation_steps,
                    )
                except BaseException:  # noqa: BLE001 - contain target before re-raising
                    _stop_detached(run_dir, code="preload_failure_containment_failed")
                    raise
            if release_evidence is not None:
                if not _target_is_live(release_evidence):
                    terminal = _await_detached_terminal_handoff(run_dir)
                    if terminal is not None:
                        return terminal, release_evidence
                    _stop_detached(run_dir, code="target_loss_containment_failed")
                    _fail("trainer_identity_lost")
                if not _sentinel_is_live(release_evidence):
                    _stop_detached(run_dir, code="sentinel_loss_containment_failed")
                    _fail("sentinel_liveness_lost")
                if not _caffeinate_is_live(release_evidence):
                    _stop_detached(run_dir, code="caffeinate_loss_containment_failed")
                    _fail("trainer_caffeinate_lost")
        sequence = status.get("heartbeat_sequence")
        if sequence != last_sequence:
            last_sequence = sequence
            heartbeat_progress_at = time.monotonic()
        heartbeat_at = status.get("heartbeat_at")
        stale = (
            isinstance(heartbeat_at, (int, float))
            and not isinstance(heartbeat_at, bool)
            and time.time() - float(heartbeat_at) > float(config["watchdog"]["heartbeat_stale_s"])
        ) or (
            time.monotonic() - heartbeat_progress_at
            > float(config["watchdog"]["heartbeat_stale_s"])
        )
        if stale:
            _stop_detached(run_dir, code="stale_supervisor_stop_failed")
        _publish_status(
            config,
            "training",
            {
                "run_dir": str(run_dir),
                "detached": status,
                "checkpoint": _checkpoint_hint(config),
                "preload_released": release_evidence is not None,
            },
        )
        time.sleep(float(config["watchdog"]["poll_interval_s"]))
    _stop_detached(run_dir, code="attempt_timeout_containment_failed")
    _fail("attempt_timeout")


def _reservation_path(run_dir: Path) -> Path:
    return run_dir / "attempt-reservation.json"


def _result_path(config: Mapping[str, Any], attempt: int) -> Path:
    return (
        Path(config["paths"]["campaign_root"]) / "attempt-results" / f"attempt-{attempt:04d}.json"
    )


def _resource_guard_intervention(
    release_evidence: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Reopen a sentinel kill as typed evidence, never as a retryable crash."""

    if not isinstance(release_evidence, Mapping):
        return None
    raw_run_dir = release_evidence.get("sentinel_run_dir")
    if not isinstance(raw_run_dir, str) or not raw_run_dir:
        return None
    run_dir = _private_directory(Path(raw_run_dir))
    tombstone_dir = run_dir / "tombstones"
    if not tombstone_dir.exists():
        return None
    tombstone_dir = _private_directory(tombstone_dir)
    tombstones = sorted(tombstone_dir.glob("sentinel_tombstone_*.json"))
    if len(tombstones) != 1:
        _fail(
            "resource_guard_evidence_ambiguous",
            tombstone_count=len(tombstones),
            sentinel_run_dir=str(run_dir),
        )
    tombstone = _read_canonical(tombstones[0], expected_mode=0o400)
    killed = tombstone.get("killed_pids")
    reason = tombstone.get("reason")
    if (
        tombstone.get("schema") != "aura.memory_sentinel.tombstone.v1"
        or not isinstance(reason, str)
        or not reason
        or not isinstance(killed, list)
        or not killed
        or any(type(pid) is not int or pid <= 1 for pid in killed)
    ):
        _fail("resource_guard_evidence_invalid", path=str(tombstones[0]))
    body = {
        "schema": "aura.unified_intrinsic.resource_guard_intervention.v1",
        "reason": reason,
        "guard_stage": tombstone.get("guard_stage"),
        "killed_pids": killed,
        "final_sample": tombstone.get("final_sample"),
        "tombstone": str(tombstones[0]),
        "tombstone_sha256": canonical_sha256(tombstone),
    }
    return {**body, "intervention_sha256": canonical_sha256(body)}


def _reserve_attempt(
    config: Mapping[str, Any],
    run_dir: Path,
    *,
    attempt: int,
    before: Mapping[str, Any],
    invocation_steps: int | None,
) -> dict[str, Any]:
    ensure_private_directory(run_dir)
    body = {
        "schema": ATTEMPT_RESERVATION_SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_sha256": config["config_sha256"],
        "attempt": attempt,
        "progress_before": dict(before),
        "invocation_steps": invocation_steps,
    }
    reservation = {**body, "reservation_sha256": canonical_sha256(body)}
    _write_once(_reservation_path(run_dir), reservation)
    return reservation


def _load_reservation(
    config: Mapping[str, Any],
    run_dir: Path,
    *,
    attempt: int,
) -> dict[str, Any]:
    reservation = _read_canonical(_reservation_path(run_dir))
    body = {key: value for key, value in reservation.items() if key != "reservation_sha256"}
    if (
        reservation.get("schema") != ATTEMPT_RESERVATION_SCHEMA
        or reservation.get("campaign_id") != config["campaign_id"]
        or reservation.get("config_sha256") != config["config_sha256"]
        or reservation.get("attempt") != attempt
        or reservation.get("reservation_sha256") != canonical_sha256(body)
        or not isinstance(reservation.get("progress_before"), dict)
    ):
        _fail("attempt_reservation_invalid")
    return reservation


def _record_attempt(
    config: Mapping[str, Any],
    *,
    attempt: int,
    reservation: Mapping[str, Any],
    status: Mapping[str, Any],
    release_evidence: Mapping[str, Any] | None,
    after: Mapping[str, Any],
) -> dict[str, Any]:
    before = reservation["progress_before"]
    before_step = int(before.get("step") or 0)
    after_step = int(after.get("step") or 0)
    receipt = status.get("receipt")
    returncode = receipt.get("returncode") if isinstance(receipt, dict) else None
    durable_progress = after_step > before_step
    resource_guard_intervention = _resource_guard_intervention(release_evidence)
    body = {
        "schema": ATTEMPT_RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_sha256": config["config_sha256"],
        "attempt": attempt,
        "reservation_sha256": reservation["reservation_sha256"],
        "progress_before": dict(before),
        "progress_after": dict(after),
        "durable_progress": durable_progress,
        "returncode": returncode,
        "detached_plan_sha256": status.get("plan_sha256"),
        "detached_receipt": receipt,
        "preload_release": dict(release_evidence or {}),
        "resource_guard_intervention": resource_guard_intervention,
        "terminal_success": bool(returncode == 0 and (durable_progress or after["complete"])),
    }
    result = {**body, "attempt_sha256": canonical_sha256(body)}
    _write_once(_result_path(config, attempt), result)
    return result


def _load_result(config: Mapping[str, Any], attempt: int) -> dict[str, Any] | None:
    path = _result_path(config, attempt)
    if not path.exists():
        return None
    result = _read_canonical(path)
    body = {key: value for key, value in result.items() if key != "attempt_sha256"}
    if (
        result.get("schema") != ATTEMPT_RESULT_SCHEMA
        or result.get("campaign_id") != config["campaign_id"]
        or result.get("config_sha256") != config["config_sha256"]
        or result.get("attempt") != attempt
        or result.get("attempt_sha256") != canonical_sha256(body)
    ):
        _fail("attempt_result_invalid")
    return result


def _trailing_no_progress(config: Mapping[str, Any], completed_attempts: int) -> int:
    count = 0
    for attempt in range(completed_attempts, 0, -1):
        result = _load_result(config, attempt)
        if result is None or result.get("durable_progress") is True:
            break
        count += 1
    return count


def _planned_invocation_steps(
    config: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> int | None:
    if config["profile"] not in {
        "canary",
        "process_action_canary",
        "process_canary",
        "recovery",
    }:
        return None
    step = int(checkpoint["step"])
    if step == 0:
        return 1
    remaining = int(config["training"]["max_steps"]) - step
    if remaining < 1:
        _fail(
            "terminal_receipt_unavailable_at_max_step",
            checkpoint_step=step,
            checkpoint_complete=checkpoint.get("complete"),
            training_receipt=checkpoint.get("training_receipt"),
        )
    return remaining


def _completion(
    config: Mapping[str, Any],
    package: Mapping[str, Any],
    launchd: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    attempts: int,
) -> dict[str, Any]:
    if (
        checkpoint.get("complete") is not True
        or checkpoint.get("step") != config["training"]["max_steps"]
    ):
        _fail("terminal_checkpoint_invalid")
    body = {
        "schema": COMPLETION_SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_sha256": config["config_sha256"],
        "profile": config["profile"],
        "package": dict(package),
        "launchd": dict(launchd),
        "checkpoint": dict(checkpoint),
        "attempt_count": attempts,
        "resident_training_complete": True,
        "reasoning_gain_proven": False,
        "frontier_level_proven": False,
        "fusion_allowed": False,
    }
    receipt = {**body, "completion_sha256": canonical_sha256(body)}
    _write_once(
        Path(config["paths"]["campaign_root"]) / "completion-receipt.json",
        receipt,
    )
    _publish_status(config, "completed", receipt)
    return receipt


def run_controller(config_path: Path, *, launchd_supervised: bool) -> dict[str, Any]:
    config_path = config_path.expanduser()
    config = _load_config(config_path)
    config_path = Path(config["paths"]["campaign_root"]) / "campaign.json"
    # install_launchd keeps the host lease until its immutable launch receipt is
    # durable. The launched controller therefore waits for that exact bounded
    # handoff instead of racing the installer and exiting before custody passes.
    with _host_lease(config, wait_s=HOST_LEASE_HANDOFF_TIMEOUT_S):
        with _file_lock(
            Path(config["paths"]["campaign_root"]) / "controller.lock",
            busy_code="controller_already_running",
        ):
            launchd = _verify_launchd(config, supervised=launchd_supervised)
            _publish_status(config, "validating", {"launchd": launchd})
            package = verify_package(config)
            max_attempts = int(config["watchdog"]["max_attempts"])
            max_no_progress = int(config["watchdog"]["max_consecutive_no_progress"])
            for attempt in range(1, max_attempts + 1):
                checkpoint = _checkpoint_snapshot(config)
                if checkpoint["complete"]:
                    return _completion(config, package, launchd, checkpoint, attempt - 1)
                result = _load_result(config, attempt)
                if result is not None:
                    continue
                if _trailing_no_progress(config, attempt - 1) >= max_no_progress:
                    _fail("consecutive_no_progress_limit_exhausted")
                run_dir = Path(config["paths"]["detached_attempts"]) / f"attempt-{attempt:04d}"
                invocation_steps = _planned_invocation_steps(config, checkpoint)
                if _reservation_path(run_dir).exists():
                    reservation = _load_reservation(
                        config,
                        run_dir,
                        attempt=attempt,
                    )
                    invocation_steps = reservation["invocation_steps"]
                else:
                    _require_empty_model_lane()
                    reservation = _reserve_attempt(
                        config,
                        run_dir,
                        attempt=attempt,
                        before=checkpoint,
                        invocation_steps=invocation_steps,
                    )
                _publish_status(
                    config,
                    "launching",
                    {
                        "attempt": attempt,
                        "progress_before": reservation["progress_before"],
                        "invocation_steps": invocation_steps,
                    },
                )
                status, release_evidence = _monitor_attempt(
                    config_path,
                    config,
                    run_dir,
                    invocation_steps=invocation_steps,
                )
                after = _checkpoint_snapshot(config)
                result = _record_attempt(
                    config,
                    attempt=attempt,
                    reservation=reservation,
                    status=status,
                    release_evidence=release_evidence,
                    after=after,
                )
                if after["complete"]:
                    return _completion(config, package, launchd, after, attempt)
                if result["resource_guard_intervention"] is not None:
                    _fail(
                        "resource_guard_intervention_requires_repair",
                        attempt=attempt,
                        intervention=result["resource_guard_intervention"],
                    )
                if result["terminal_success"] is not True:
                    if _trailing_no_progress(config, attempt) >= max_no_progress:
                        _fail("consecutive_no_progress_limit_exhausted")
                    time.sleep(float(config["watchdog"]["retry_backoff_s"]))
            _fail("attempt_budget_exhausted")


def _loaded_training_labels() -> set[str]:
    result = subprocess.run(
        ["/bin/launchctl", "list"],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    if result.returncode != 0:
        _fail("launchd_inventory_failed")
    return {
        fields[-1]
        for line in result.stdout.splitlines()
        if (fields := line.split())
        and any(fields[-1].startswith(prefix) for prefix in TRAINING_LABEL_PREFIXES)
    }


def _retire_stale_launchd_jobs(active_label: str) -> list[str]:
    domain = f"gui/{os.getuid()}"
    retired: list[str] = []
    for label in sorted(_loaded_training_labels() - {active_label}):
        result = subprocess.run(
            ["/bin/launchctl", "bootout", f"{domain}/{label}"],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        if result.returncode != 0:
            _fail("competing_launchd_retirement_failed", label=label)
        retired.append(label)
    return retired


def _contain_campaign_attempts(config: Mapping[str, Any]) -> None:
    attempts_root = Path(config["paths"]["detached_attempts"])
    failures: list[str] = []
    for attempt in sorted(attempts_root.glob("attempt-*")):
        if attempt.is_symlink() or not attempt.is_dir():
            failures.append(str(attempt))
            continue
        sentinel_dirs = sorted(attempt.glob("sentinel-*"))
        for run_dir in [*sentinel_dirs, attempt]:
            if not (run_dir / detached.PLAN_FILE).exists():
                continue
            try:
                _stop_detached(run_dir, code="launch_rollback_containment_failed")
            except UnifiedResidentControllerError:
                failures.append(str(run_dir))
    if failures:
        _fail("launch_rollback_containment_failed", run_dirs=failures)


def _rollback_launch(config: Mapping[str, Any]) -> None:
    label = str(config["launch"]["label"])
    target = f"gui/{os.getuid()}/{label}"
    subprocess.run(
        ["/bin/launchctl", "bootout", target],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    _contain_campaign_attempts(config)
    probe = subprocess.run(
        ["/bin/launchctl", "print", target],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    if probe.returncode == 0:
        _fail("launch_rollback_job_survived")


def install_launchd(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser()
    config = _load_config(config_path)
    config_path = Path(config["paths"]["campaign_root"]) / "campaign.json"
    _require_empty_model_lane()
    package = verify_package(config)
    with _file_lock(
        TRAINING_STATE_ROOT / "install.lock",
        busy_code="resident_training_install_busy",
    ):
        label = str(config["launch"]["label"])
        root = Path(config["paths"]["campaign_root"])
        _payload, plist_path, plist, intent = _launch_contract(config, config_path)
        _write_once(root / "launch-intent.json", intent)
        agents = ensure_private_directory(LAUNCH_AGENTS_ROOT)
        if plist_path.parent != agents:
            _fail("launchd_plist_path_drift")
        atomic_write_bytes(plist_path, plist, mode=0o600)
        domain = f"gui/{os.getuid()}"
        bootstrapped = False
        try:
            with _host_lease(config):
                _require_empty_model_lane()
                retired = _retire_stale_launchd_jobs(label)
                subprocess.run(
                    ["/bin/launchctl", "bootout", domain, str(plist_path)],
                    capture_output=True,
                    text=True,
                    timeout=30.0,
                    check=False,
                )
                started = subprocess.run(
                    ["/bin/launchctl", "bootstrap", domain, str(plist_path)],
                    capture_output=True,
                    text=True,
                    timeout=30.0,
                    check=False,
                )
                bootstrapped = True
                if started.returncode != 0:
                    _fail(
                        "launchd_bootstrap_failed",
                        stderr=started.stderr.strip()[:500],
                    )
                deadline = time.monotonic() + 15.0
                job: dict[str, Any] | None = None
                while time.monotonic() < deadline:
                    try:
                        job = _launchd_job(label)
                        break
                    except UnifiedResidentControllerError as exc:
                        if exc.code not in {
                            "launchd_job_unavailable",
                            "launchd_job_pid_missing",
                        }:
                            raise
                        time.sleep(0.25)
                if job is None:
                    _fail("launchd_start_timeout")
                start_token = detached._process_start_token(job["pid"])  # noqa: SLF001
                if not start_token:
                    _fail("launchd_controller_identity_unavailable")
                body = {
                    "schema": LAUNCH_SCHEMA,
                    "campaign_id": config["campaign_id"],
                    "config_sha256": config["config_sha256"],
                    "launch_intent_sha256": intent["intent_sha256"],
                    "label": label,
                    "target": job["target"],
                    "pid": job["pid"],
                    "start_token": start_token,
                    "plist_path": str(plist_path),
                    "plist_sha256": hashlib.sha256(plist).hexdigest(),
                    "retired_labels": retired,
                    "package": package,
                    "installed_at_unix_ns": time.time_ns(),
                }
                receipt = {**body, "launch_sha256": canonical_sha256(body)}
                receipt_id = f"launch-{job['pid']}-{start_token[:16]}"
                receipts = ensure_private_directory(root / "launchd-receipts")
                receipt_path = receipts / f"{receipt_id}.json"
                _write_once(receipt_path, receipt)
                pointer_body = {
                    "schema": LAUNCH_POINTER_SCHEMA,
                    "receipt": f"launchd-receipts/{receipt_path.name}",
                    "launch_sha256": receipt["launch_sha256"],
                }
                pointer = {
                    **pointer_body,
                    "pointer_sha256": canonical_sha256(pointer_body),
                }
                atomic_write_bytes(
                    root / "launchd-receipt-pointer.json",
                    _document(pointer),
                    mode=0o600,
                )
                return receipt
        except BaseException:  # noqa: BLE001 - launch must roll back as one transaction
            if bootstrapped:
                _rollback_launch(config)
            raise


def validate_campaign(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    package = verify_package(config)
    owners = _owner_rows()
    return {
        "schema": "aura.unified_intrinsic.validation.v1",
        "campaign_id": config["campaign_id"],
        "config_sha256": config["config_sha256"],
        "package": package,
        "model_lane_clear": not owners,
        "model_lane_owners": owners,
        "launch_allowed": not owners,
        "claims_supported": ["resident_campaign_package_identity_valid"],
        "claims_not_supported": [
            "resident_training_complete",
            "reasoning_gain_proven",
            "frontier_level_proven",
            "fusion_allowed",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("validate", "install", "status"):
        command = commands.add_parser(action)
        command.add_argument("--config", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--launchd-supervised", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "validate":
            payload = validate_campaign(args.config)
        elif args.action == "install":
            payload = install_launchd(args.config)
        elif args.action == "status":
            config = _load_config(args.config)
            payload = _inspect_status(config)
        else:
            payload = run_controller(
                args.config,
                launchd_supervised=args.launchd_supervised,
            )
    except UnifiedResidentControllerError as exc:
        if args.action == "run":
            _publish_failure_status(
                args.config,
                error=exc.code,
                details=exc.details,
            )
        error = {
            "schema": ERROR_SCHEMA,
            "error": exc.code,
            "details": exc.details,
            "claims_supported": [],
        }
        print(json.dumps(error, sort_keys=True), file=sys.stderr, flush=True)
        if args.action == "run" and args.launchd_supervised:
            return 1 if exc.code in RESTARTABLE_CODES else 0
        return 2
    except Exception as exc:  # noqa: BLE001 - stable launchd crash boundary
        if args.action == "run":
            _publish_failure_status(
                args.config,
                error=type(exc).__name__,
                details={"message": str(exc) or "no_message"},
            )
        error = {
            "schema": ERROR_SCHEMA,
            "error": str(exc) or "no_message",
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "claims_supported": [],
        }
        print(json.dumps(error, sort_keys=True), file=sys.stderr, flush=True)
        return 1 if args.action == "run" and args.launchd_supervised else 2
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
