#!/usr/bin/env python3
"""Durable OS-supervised controller for the resident RLC reconciliation sweep.

The sweep owns scientific cell resumption. This controller owns the process
lifecycle around it: immutable source verification, one exclusive model owner,
bounded exact-run retries, process-group wedge recovery, authenticated liveness,
and launchd/caffeinate custody. It never changes tasks, arms, grading, or gates.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import json
import os
import plistlib
import secrets
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

SCHEMA: Final = "aura.rlc_reconciliation_controller.v1"
SOURCE_SCHEMA: Final = "aura.rlc_reconciliation_source_manifest.v1"
HEARTBEAT_SCHEMA: Final = "aura.rlc_reconciliation_controller_heartbeat.v1"
STATUS_SCHEMA: Final = "aura.rlc_reconciliation_controller_status.v1"
LAUNCH_SCHEMA: Final = "aura.rlc_reconciliation_controller_launchd.v1"
TERMINAL_PHASES: Final = frozenset({"complete", "yielded", "blocked"})
CONFIG_SUFFIXES: Final = frozenset({".json", ".toml", ".yaml", ".yml", ".jinja"})
GLOBAL_MODEL_LOCK: Final = Path.home() / ".aura/state/rlc-reconciliation-model.lock"


class ControllerError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, document: Mapping[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        payload = json.dumps(document, indent=1, sort_keys=True, allow_nan=False) + "\n"
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            sink.write(payload)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ControllerError(f"{role}_unreadable:{path}") from exc
    if not isinstance(value, dict):
        raise ControllerError(f"{role}_not_object:{path}")
    return value


def _append_event(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(document, sort_keys=True, allow_nan=False) + "\n")
        sink.flush()
        os.fsync(sink.fileno())


def _source_paths(root: Path) -> list[Path]:
    paths: set[Path] = set()
    for base in (root / "core", root / "config"):
        if not base.is_dir():
            raise ControllerError(f"source_root_incomplete:{base}")
    # Every Python file is executable input because Python can import a module
    # from any package on the source root (including sitecustomize at startup).
    for candidate in root.rglob("*.py"):
        relative = candidate.relative_to(root)
        if candidate.is_file() and not any(
            part in {".git", "__pycache__", ".pytest_cache"}
            for part in relative.parts
        ):
            paths.add(relative)
    for candidate in (root / "config").rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in CONFIG_SUFFIXES:
            paths.add(candidate.relative_to(root))
    for relative in (
        Path("tools/run_rlc_reconciliation_sweep.py"),
        Path("tools/run_rlc_reconciliation_controller.py"),
        Path("pyproject.toml"),
        Path("requirements_lock.txt"),
    ):
        path = root / relative
        if not path.is_file():
            raise ControllerError(f"required_source_missing:{relative}")
        paths.add(relative)
    return sorted(paths, key=lambda item: os.fsencode(item.as_posix()))


def build_source_manifest(root: Path, *, source_commit: str) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    files = [
        {
            "path": path.as_posix(),
            "size": (root / path).stat().st_size,
            "sha256": _sha_file(root / path),
        }
        for path in _source_paths(root)
    ]
    body = {
        "schema": SOURCE_SCHEMA,
        "source_commit": source_commit,
        "files": files,
    }
    return {**body, "manifest_sha256": _sha(body)}


def verify_source_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        manifest.get("schema") != SOURCE_SCHEMA
        or manifest.get("manifest_sha256") != _sha(body)
        or not isinstance(manifest.get("files"), list)
        or not manifest["files"]
    ):
        raise ControllerError("source_manifest_invalid")
    root = root.expanduser().resolve(strict=True)
    recorded_paths = [Path(str(item.get("path", ""))) for item in manifest["files"]]
    if recorded_paths != _source_paths(root):
        raise ControllerError("source_file_set_drift")
    for item in manifest["files"]:
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise ControllerError("source_manifest_entry_invalid")
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ControllerError("source_manifest_path_invalid")
        path = root / relative
        try:
            metadata = path.stat()
        except OSError as exc:
            raise ControllerError(f"source_file_missing:{relative}") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size != item["size"]
            or _sha_file(path) != item["sha256"]
        ):
            raise ControllerError(f"source_file_drift:{relative}")


def build_model_manifest(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: os.fsencode(path.relative_to(root).as_posix()),
    )
    if not files or not (root / "config.json").is_file():
        raise ControllerError("model_directory_incomplete")
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha_file(path),
        }
        for path in files
    ]
    body = {"root": str(root), "files": entries}
    return {**body, "manifest_sha256": _sha(body)}


def verify_model_manifest(manifest: Mapping[str, Any]) -> None:
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        manifest.get("manifest_sha256") != _sha(body)
        or not isinstance(manifest.get("files"), list)
        or not manifest["files"]
    ):
        raise ControllerError("model_manifest_invalid")
    root = Path(str(manifest.get("root"))).expanduser().resolve(strict=True)
    observed = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )
    recorded = [str(item.get("path")) for item in manifest["files"]]
    if observed != recorded:
        raise ControllerError("model_file_set_drift")
    for item in manifest["files"]:
        path = root / str(item["path"])
        metadata = path.stat()
        if metadata.st_size != item["size"] or _sha_file(path) != item["sha256"]:
            raise ControllerError(f"model_file_drift:{item['path']}")


def _config_body(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "config_sha256"}


def load_config(path: Path) -> dict[str, Any]:
    config = _read_json(path.expanduser().resolve(strict=True), role="controller_config")
    if config.get("schema") != SCHEMA or config.get("config_sha256") != _sha(_config_body(config)):
        raise ControllerError("controller_config_invalid")
    required = {
        "campaign_id",
        "source_root",
        "source_commit",
        "source_manifest_path",
        "python",
        "python_sha256",
        "model",
        "model_manifest",
        "out_dir",
        "arms",
        "seed",
        "per_domain",
        "n_slots",
        "max_tokens",
        "memory_fraction",
        "episode_wall_s",
        "attempt_wall_s",
        "max_attempts",
        "poll_s",
        "stale_after_s",
        "retry_backoff_s",
        "heartbeat_key_path",
        "launch_label",
    }
    if not required.issubset(config):
        raise ControllerError("controller_config_incomplete")
    if not 1 <= int(config["max_attempts"]) <= 32:
        raise ControllerError("controller_attempt_budget_invalid")
    if float(config["stale_after_s"]) <= float(config["episode_wall_s"]):
        raise ControllerError("controller_stale_budget_too_short")
    return config


def build_config(
    *,
    source_root: Path,
    source_commit: str,
    model: Path,
    out_dir: Path,
    python: Path,
    arms: str,
    seed: int,
    per_domain: int,
    n_slots: int,
    max_tokens: int,
    memory_fraction: float,
    episode_wall_s: float,
    attempt_wall_s: float,
    max_attempts: int,
    poll_s: float,
    stale_after_s: float,
    retry_backoff_s: float,
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    source_root = source_root.expanduser().resolve(strict=True)
    model = model.expanduser().resolve(strict=True)
    # Preserve the venv entrypoint for execution. Resolving its symlink to the
    # Homebrew base binary drops pyvenv.cfg discovery and therefore the exact
    # dependency environment, even though both paths hash the same executable.
    python = python.expanduser().absolute()
    resolved_python = python.resolve(strict=True)
    out_dir = out_dir.expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(out_dir, 0o700)
    campaign_id = out_dir.name
    manifest = build_source_manifest(source_root, source_commit=source_commit)
    manifest_path = out_dir / "source_manifest.json"
    key_path = out_dir / ".heartbeat.key"
    key = secrets.token_bytes(32)
    body = {
        "schema": SCHEMA,
        "campaign_id": campaign_id,
        "source_root": str(source_root),
        "source_commit": source_commit,
        "source_manifest_path": str(manifest_path),
        "python": str(python),
        "python_sha256": _sha_file(resolved_python),
        "model": str(model),
        "model_manifest": build_model_manifest(model),
        "out_dir": str(out_dir / "sweep"),
        "arms": arms,
        "seed": int(seed),
        "per_domain": int(per_domain),
        "n_slots": int(n_slots),
        "max_tokens": int(max_tokens),
        "memory_fraction": float(memory_fraction),
        "episode_wall_s": float(episode_wall_s),
        "attempt_wall_s": float(attempt_wall_s),
        "max_attempts": int(max_attempts),
        "poll_s": float(poll_s),
        "stale_after_s": float(stale_after_s),
        "retry_backoff_s": float(retry_backoff_s),
        "heartbeat_key_path": str(key_path),
        "launch_label": f"com.aura.rlc-reconciliation.{campaign_id}",
    }
    config = {**body, "config_sha256": _sha(body)}
    return config, manifest, key


def write_prepared_campaign(
    config_path: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    key: bytes,
) -> None:
    config_path = config_path.expanduser().absolute()
    manifest_path = Path(str(config["source_manifest_path"]))
    key_path = Path(str(config["heartbeat_key_path"]))
    if config_path.exists() or manifest_path.exists() or key_path.exists():
        raise ControllerError("prepared_campaign_artifact_exists")
    _atomic_json(manifest_path, manifest)
    descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as sink:
        sink.write(key)
        sink.flush()
        os.fsync(sink.fileno())
    _atomic_json(config_path, config)


def _heartbeat_key(config: Mapping[str, Any]) -> bytes:
    path = Path(str(config["heartbeat_key_path"]))
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ControllerError("heartbeat_key_custody_invalid")
    key = path.read_bytes()
    if len(key) != 32:
        raise ControllerError("heartbeat_key_invalid")
    return key


def _signed_heartbeat(config: Mapping[str, Any], body: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": HEARTBEAT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_sha256": config["config_sha256"],
        **body,
    }
    payload["hmac_sha256"] = hmac.new(
        _heartbeat_key(config), _canonical(payload), hashlib.sha256
    ).hexdigest()
    return payload


def verify_heartbeat(config: Mapping[str, Any], heartbeat: Mapping[str, Any]) -> None:
    provided = heartbeat.get("hmac_sha256")
    body = {key: value for key, value in heartbeat.items() if key != "hmac_sha256"}
    expected = hmac.new(_heartbeat_key(config), _canonical(body), hashlib.sha256).hexdigest()
    if (
        heartbeat.get("schema") != HEARTBEAT_SCHEMA
        or heartbeat.get("campaign_id") != config["campaign_id"]
        or heartbeat.get("config_sha256") != config["config_sha256"]
        or not isinstance(provided, str)
        or not hmac.compare_digest(provided, expected)
    ):
        raise ControllerError("controller_heartbeat_invalid")


def _journal_cells(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            try:
                if json.loads(line).get("event") == "CELL":
                    count += 1
            except (AttributeError, json.JSONDecodeError):
                continue
    return count


def _progress_snapshot(config: Mapping[str, Any], log_path: Path) -> dict[str, Any]:
    out_dir = Path(str(config["out_dir"]))
    status_path = out_dir / "status.json"
    journal_path = out_dir / "journal.jsonl"
    mtimes = [
        path.stat().st_mtime
        for path in (status_path, journal_path, log_path)
        if path.is_file()
    ]
    sweep_status: dict[str, Any] = {}
    if status_path.is_file():
        try:
            sweep_status = _read_json(status_path, role="sweep_status")
        except ControllerError:
            sweep_status = {"phase": "unreadable"}
    return {
        "cells": _journal_cells(journal_path),
        "last_progress_unix": max(mtimes, default=0.0),
        "sweep_phase": sweep_status.get("phase"),
        "sweep_arm": sweep_status.get("arm"),
        "sweep_arm_progress": sweep_status.get("arm_progress"),
    }


def _sweep_command(config: Mapping[str, Any]) -> list[str]:
    return [
        str(config["python"]),
        str(Path(str(config["source_root"])) / "tools/run_rlc_reconciliation_sweep.py"),
        "--model",
        str(config["model"]),
        "--out-dir",
        str(config["out_dir"]),
        "--seed",
        str(config["seed"]),
        "--per-domain",
        str(config["per_domain"]),
        "--n-slots",
        str(config["n_slots"]),
        "--max-tokens",
        str(config["max_tokens"]),
        "--memory-fraction",
        str(config["memory_fraction"]),
        "--episode-wall-s",
        str(config["episode_wall_s"]),
        "--max-wall-s",
        str(config["attempt_wall_s"]),
        "--arms",
        str(config["arms"]),
    ]


def _terminate_exact_group(process: subprocess.Popen[Any]) -> None:
    try:
        group = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    if group != process.pid:
        raise ControllerError("sweep_process_group_identity_invalid")
    os.killpg(group, signal.SIGTERM)
    try:
        process.wait(timeout=15.0)
        return
    except subprocess.TimeoutExpired:
        pass
    os.killpg(group, signal.SIGKILL)
    process.wait(timeout=10.0)


def _write_status(config: Mapping[str, Any], *, phase: str, **fields: Any) -> None:
    out_root = Path(str(config["out_dir"])).parent
    body = {
        "schema": STATUS_SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_sha256": config["config_sha256"],
        "phase": phase,
        "updated_unix": time.time(),
        **fields,
    }
    _atomic_json(out_root / "controller_status.json", {**body, "status_sha256": _sha(body)})


def _source_is_current(config: Mapping[str, Any]) -> None:
    manifest = _read_json(Path(str(config["source_manifest_path"])), role="source_manifest")
    if manifest.get("source_commit") != config["source_commit"]:
        raise ControllerError("source_commit_binding_invalid")
    verify_source_manifest(Path(str(config["source_root"])), manifest)
    python = Path(str(config["python"])).expanduser().absolute()
    if _sha_file(python.resolve(strict=True)) != config["python_sha256"]:
        raise ControllerError("interpreter_identity_drift")
    verify_model_manifest(config["model_manifest"])


def _process_record(pid: int) -> tuple[int, str]:
    observed = subprocess.run(
        ["/bin/ps", "-ww", "-o", "ppid=", "-o", "command=", "-p", str(pid)],
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    if observed.returncode != 0 or not observed.stdout.strip():
        raise ControllerError(f"process_lineage_unavailable:{pid}")
    fields = observed.stdout.strip().split(None, 1)
    if len(fields) != 2 or not fields[0].isdigit():
        raise ControllerError(f"process_lineage_invalid:{pid}")
    return int(fields[0]), fields[1]


def _process_table() -> list[tuple[int, int, str]]:
    observed = subprocess.run(
        ["/bin/ps", "-ww", "-axo", "pid=,ppid=,command="],
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    if observed.returncode != 0:
        raise ControllerError("process_table_unavailable")
    rows: list[tuple[int, int, str]] = []
    for line in observed.stdout.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) == 3 and fields[0].isdigit() and fields[1].isdigit():
            rows.append((int(fields[0]), int(fields[1]), fields[2]))
    return rows


def _verify_launchd_lineage(config: Mapping[str, Any]) -> dict[str, Any]:
    controller_pid = os.getpid()
    controller_parent, controller_command = _process_record(controller_pid)
    controller_required = (
        str(Path(str(config["source_root"])) / "tools/run_rlc_reconciliation_controller.py"),
        str(config["campaign_id"]),
        "--launchd-supervised",
    )
    if controller_parent != 1 or any(
        value not in controller_command for value in controller_required
    ):
        raise ControllerError("controller_launchd_caffeinate_lineage_invalid")
    caffeinate_required = (
        "/usr/bin/caffeinate",
        "-dims",
        str(config["python"]),
        str(Path(str(config["source_root"])) / "tools/run_rlc_reconciliation_controller.py"),
        str(config["campaign_id"]),
    )
    caffeinate_children = [
        pid
        for pid, parent, command in _process_table()
        if parent == controller_pid
        and all(value in command for value in caffeinate_required)
    ]
    if len(caffeinate_children) != 1:
        raise ControllerError("controller_launchd_caffeinate_lineage_invalid")
    return {
        "launchd_pid": 1,
        "caffeinate_pid": caffeinate_children[0],
        "controller_pid": controller_pid,
    }


def run(config_path: Path, *, launchd_supervised: bool = False) -> int:
    config = load_config(config_path)
    if not launchd_supervised:
        raise ControllerError("controller_requires_launchd_supervision")
    lineage = _verify_launchd_lineage(config)
    root = Path(str(config["out_dir"])).parent
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_descriptor = os.open(root / ".controller.lock", os.O_RDWR | os.O_CREAT, 0o600)
    GLOBAL_MODEL_LOCK.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    global_lock_descriptor = os.open(
        GLOBAL_MODEL_LOCK, os.O_RDWR | os.O_CREAT, 0o600
    )
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ControllerError("controller_already_running") from exc
        try:
            fcntl.flock(global_lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ControllerError("another_reconciliation_model_owner_is_active") from exc
        _source_is_current(config)
        log_path = root / "sweep.log"
        events_path = root / "controller_attempts.jsonl"
        heartbeat_path = root / "controller_heartbeat.json"
        for attempt in range(1, int(config["max_attempts"]) + 1):
            if (Path(str(config["out_dir"])) / "verdict.json").is_file():
                _write_status(config, phase="complete", attempt=attempt - 1)
                return 0
            if (Path(str(config["out_dir"])) / "YIELD").exists():
                _write_status(config, phase="yielded", attempt=attempt - 1)
                return 0
            _source_is_current(config)
            command = _sweep_command(config)
            started = time.time()
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=str(config["source_root"]),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env={**os.environ, "AURA_LOG_DIR": str(root / "logs")},
                )
                _append_event(
                    events_path,
                    {
                        "event": "ATTEMPT_STARTED",
                        "attempt": attempt,
                        "pid": process.pid,
                        "process_group": process.pid,
                        "command_sha256": _sha(command),
                        "source_commit": config["source_commit"],
                        "started_unix": started,
                    },
                )
                reason = ""
                while process.poll() is None:
                    snapshot = _progress_snapshot(config, log_path)
                    heartbeat = _signed_heartbeat(
                        config,
                        {
                            "controller_pid": os.getpid(),
                            "sweep_pid": process.pid,
                            "sweep_process_group": process.pid,
                            "attempt": attempt,
                            "observed_unix": time.time(),
                            "lineage": lineage,
                            **snapshot,
                        },
                    )
                    _atomic_json(heartbeat_path, heartbeat)
                    if (
                        snapshot["last_progress_unix"] > 0
                        and time.time() - float(snapshot["last_progress_unix"])
                        > float(config["stale_after_s"])
                    ):
                        reason = "progress_stalled"
                        _terminate_exact_group(process)
                        break
                    time.sleep(float(config["poll_s"]))
                returncode = process.poll()
            snapshot = _progress_snapshot(config, log_path)
            _append_event(
                events_path,
                {
                    "event": "ATTEMPT_FINISHED",
                    "attempt": attempt,
                    "pid": process.pid,
                    "returncode": returncode,
                    "reason": reason or "process_exited",
                    "finished_unix": time.time(),
                    **snapshot,
                },
            )
            if (Path(str(config["out_dir"])) / "verdict.json").is_file():
                _write_status(config, phase="complete", attempt=attempt, **snapshot)
                return 0
            if returncode == 4 or (Path(str(config["out_dir"])) / "YIELD").exists():
                _write_status(config, phase="yielded", attempt=attempt, **snapshot)
                return 0
            if attempt < int(config["max_attempts"]):
                _write_status(
                    config,
                    phase="retrying",
                    attempt=attempt,
                    returncode=returncode,
                    reason=reason or "incomplete_exit",
                    **snapshot,
                )
                time.sleep(float(config["retry_backoff_s"]))
        _write_status(config, phase="blocked", reason="attempt_budget_exhausted")
        return 2
    finally:
        os.close(global_lock_descriptor)
        os.close(lock_descriptor)


def _launch_payload(config_path: Path, config: Mapping[str, Any]) -> bytes:
    root = Path(str(config["out_dir"])).parent
    payload = {
        "Label": config["launch_label"],
        "ProgramArguments": [
            "/usr/bin/caffeinate",
            "-dims",
            str(config["python"]),
            str(Path(str(config["source_root"])) / "tools/run_rlc_reconciliation_controller.py"),
            "run",
            "--config",
            str(config_path.expanduser().resolve(strict=True)),
            "--launchd-supervised",
        ],
        "WorkingDirectory": str(config["source_root"]),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 30,
        "ProcessType": "Background",
        "StandardOutPath": str(root / "controller.log"),
        "StandardErrorPath": str(root / "controller.log"),
    }
    return plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)


def install_launchd(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve(strict=True)
    config = load_config(config_path)
    _source_is_current(config)
    root = Path(str(config["out_dir"])).parent
    launch_agents = Path.home() / "Library/LaunchAgents"
    launch_agents.mkdir(parents=True, exist_ok=True, mode=0o700)
    plist_path = launch_agents / f"{config['launch_label']}.plist"
    payload = _launch_payload(config_path, config)
    temporary = plist_path.with_suffix(".tmp")
    temporary.write_bytes(payload)
    os.chmod(temporary, 0o600)
    os.replace(temporary, plist_path)
    domain = f"gui/{os.getuid()}"
    subprocess.run(
        ["/bin/launchctl", "bootout", domain, str(plist_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    started = subprocess.run(
        ["/bin/launchctl", "bootstrap", domain, str(plist_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if started.returncode != 0:
        raise ControllerError(f"launchd_bootstrap_failed:{started.returncode}:{started.stderr.strip()}")
    body = {
        "schema": LAUNCH_SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_sha256": config["config_sha256"],
        "label": config["launch_label"],
        "domain": domain,
        "plist_path": str(plist_path),
        "plist_sha256": hashlib.sha256(payload).hexdigest(),
        "launchd_keepalive": True,
        "caffeinate": True,
        "installed_unix": time.time(),
    }
    receipt = {**body, "launch_sha256": _sha(body)}
    _atomic_json(root / "launchd_receipt.json", receipt)
    return receipt


def status(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = Path(str(config["out_dir"])).parent
    result: dict[str, Any] = {
        "campaign_id": config["campaign_id"],
        "source_commit": config["source_commit"],
        "controller_status": None,
        "heartbeat": None,
    }
    status_path = root / "controller_status.json"
    if status_path.is_file():
        controller_status = _read_json(status_path, role="controller_status")
        body = {key: value for key, value in controller_status.items() if key != "status_sha256"}
        if controller_status.get("status_sha256") != _sha(body):
            raise ControllerError("controller_status_invalid")
        result["controller_status"] = controller_status
    heartbeat_path = root / "controller_heartbeat.json"
    if heartbeat_path.is_file():
        heartbeat = _read_json(heartbeat_path, role="controller_heartbeat")
        verify_heartbeat(config, heartbeat)
        result["heartbeat"] = heartbeat
    result["progress"] = _progress_snapshot(config, root / "sweep.log")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source-root", type=Path, required=True)
    prepare.add_argument("--source-commit", required=True)
    prepare.add_argument("--model", type=Path, required=True)
    prepare.add_argument("--out-dir", type=Path, required=True)
    prepare.add_argument("--python", type=Path, required=True)
    prepare.add_argument("--arms", default="full_stack,full_stack_oracle")
    prepare.add_argument("--seed", type=int, default=20260808)
    prepare.add_argument("--per-domain", type=int, default=4)
    prepare.add_argument("--n-slots", type=int, default=16)
    prepare.add_argument("--max-tokens", type=int, default=512)
    prepare.add_argument("--memory-fraction", type=float, default=0.40)
    prepare.add_argument("--episode-wall-s", type=float, default=900.0)
    prepare.add_argument("--attempt-wall-s", type=float, default=32_400.0)
    prepare.add_argument("--max-attempts", type=int, default=8)
    prepare.add_argument("--poll-s", type=float, default=15.0)
    prepare.add_argument("--stale-after-s", type=float, default=1_800.0)
    prepare.add_argument("--retry-backoff-s", type=float, default=30.0)
    prepare.add_argument("--output", type=Path, required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--launchd-supervised", action="store_true")
    install = commands.add_parser("install-launchd")
    install.add_argument("--config", type=Path, required=True)
    inspect = commands.add_parser("status")
    inspect.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "prepare":
            config, manifest, key = build_config(
                source_root=args.source_root,
                source_commit=args.source_commit,
                model=args.model,
                out_dir=args.out_dir,
                python=args.python,
                arms=args.arms,
                seed=args.seed,
                per_domain=args.per_domain,
                n_slots=args.n_slots,
                max_tokens=args.max_tokens,
                memory_fraction=args.memory_fraction,
                episode_wall_s=args.episode_wall_s,
                attempt_wall_s=args.attempt_wall_s,
                max_attempts=args.max_attempts,
                poll_s=args.poll_s,
                stale_after_s=args.stale_after_s,
                retry_backoff_s=args.retry_backoff_s,
            )
            write_prepared_campaign(args.output, config, manifest, key)
            print(json.dumps(config, indent=2, sort_keys=True))
            return 0
        if args.action == "run":
            return run(args.config, launchd_supervised=args.launchd_supervised)
        if args.action == "install-launchd":
            print(json.dumps(install_launchd(args.config), indent=2, sort_keys=True))
            return 0
        print(json.dumps(status(args.config), indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
        print(f"run_rlc_reconciliation_controller: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
