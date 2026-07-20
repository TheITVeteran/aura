#!/usr/bin/env python3
"""Durably verify calibration and launch exact resident-v3 recovery training."""

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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.runtime.atomic_writer import atomic_write_bytes  # noqa: E402
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402
from tools import launch_resident_v3_recovery as recovery  # noqa: E402
from tools import run_detached_step as detached  # noqa: E402

STATE_SCHEMA = "aura.resident_v3_recovery_controller_state.v1"
VERDICT_SCHEMA = "aura.resident_v3_recovery_controller_verdict.v1"
_MAX_JSON_BYTES = 256 * 1024 * 1024


class ResidentV3RecoveryControllerError(RuntimeError):
    """Stable fail-closed recovery-controller error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise ResidentV3RecoveryControllerError(code)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    raw = read_stable_bytes(path.expanduser().resolve(strict=True), max_bytes=_MAX_JSON_BYTES)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResidentV3RecoveryControllerError(f"{role}_invalid") from exc
    if not isinstance(value, dict):
        _fail(f"{role}_invalid")
    return value


def _controller_root(migration_path: Path) -> tuple[Path, Path]:
    migration = _read_json(migration_path, role="migration")
    destination = migration.get("destination")
    if not isinstance(destination, Mapping):
        _fail("migration_destination_invalid")
    adapter_root = Path(str(destination.get("root"))).resolve(strict=True)
    root = adapter_root.parent
    if not root.name.startswith("resident_32b_v3_cp"):
        _fail("migration_destination_invalid")
    return root, adapter_root


def _write_state(root: Path, *, stage: str, status: str, details: Mapping[str, Any]) -> None:
    material = {
        "schema": STATE_SCHEMA,
        "stage": stage,
        "status": status,
        "controller_pid": os.getpid(),
        "updated_at": time.time(),
        "details": dict(details),
    }
    atomic_write_bytes(
        root / "recovery_controller_state.json",
        _canonical({**material, "state_sha256": _sha(_canonical(material))}) + b"\n",
        mode=0o600,
    )


def _terminal_statuses(
    root: Path,
    *,
    phase: str,
    timeout_s: float,
    poll_s: float,
) -> dict[str, dict[str, Any]]:
    run_dirs = {
        "trainer": root / f"detached-{phase}",
        "sentinel": root / f"sentinel-{phase}",
    }
    deadline = time.monotonic() + timeout_s
    while True:
        statuses = {role: detached._status(path) for role, path in run_dirs.items()}
        if all(status.get("terminal") is True for status in statuses.values()):
            return statuses
        if time.monotonic() >= deadline:
            _fail(f"{phase}_wait_timeout")
        _write_state(
            root,
            stage=f"wait_{phase}",
            status="waiting",
            details={
                role: {
                    "state": status.get("state"),
                    "heartbeat_sequence": status.get("heartbeat_sequence"),
                }
                for role, status in statuses.items()
            },
        )
        time.sleep(poll_s)


def _receipt_summary(statuses: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role, status in statuses.items():
        receipt = status.get("receipt")
        if not isinstance(receipt, Mapping):
            _fail(f"{role}_receipt_missing")
        result[role] = {
            "returncode": receipt.get("returncode"),
            "receipt_sha256": receipt.get("receipt_sha256"),
            "containment_verified": receipt.get("containment_verified"),
            "process_group_empty": receipt.get("process_group_empty"),
            "lineage_empty": receipt.get("lineage_empty"),
            "restart_count": receipt.get("restart_count"),
        }
    return result


def _write_verdict(
    root: Path,
    *,
    decision: str,
    migration_sha256: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    material = {
        "schema": VERDICT_SCHEMA,
        "decision": decision,
        "migration_sha256": migration_sha256,
        "details": dict(details),
        "finished_at": time.time(),
    }
    verdict = {**material, "verdict_sha256": _sha(_canonical(material))}
    atomic_write_bytes(
        root / "recovery_controller_verdict.json",
        _canonical(verdict) + b"\n",
        mode=0o600,
    )
    return verdict


def _read_verdict(path: Path, *, migration_sha256: str) -> dict[str, Any]:
    verdict = _read_json(path, role="controller_verdict")
    claimed = verdict.get("verdict_sha256")
    material = dict(verdict)
    material.pop("verdict_sha256", None)
    if (
        verdict.get("schema") != VERDICT_SCHEMA
        or verdict.get("migration_sha256") != migration_sha256
        or claimed != _sha(_canonical(material))
    ):
        _fail("controller_verdict_invalid")
    return verdict


def run_controller(
    migration_path: Path,
    *,
    calibration_timeout_s: float = 7200.0,
    training_timeout_s: float = 216000.0,
    poll_s: float = 15.0,
) -> dict[str, Any]:
    root, _adapter_root = _controller_root(migration_path)
    migration = _read_json(migration_path, role="migration")
    migration_sha256 = str(migration.get("migration_sha256"))
    lock_path = root / "recovery_controller.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _fail("controller_already_running")
        verdict_path = root / "recovery_controller_verdict.json"
        if verdict_path.exists():
            return _read_verdict(verdict_path, migration_sha256=migration_sha256)
        try:
            calibration = _terminal_statuses(
                root,
                phase="calibration",
                timeout_s=calibration_timeout_s,
                poll_s=poll_s,
            )
            calibration_receipts = _receipt_summary(calibration)
            _write_state(
                root,
                stage="verify_calibration",
                status="running",
                details=calibration_receipts,
            )
            calibration_verdict = recovery.verify_calibration(migration_path)
            _write_state(
                root,
                stage="launch_resume",
                status="running",
                details={"calibration_verdict_sha256": calibration_verdict["verdict_sha256"]},
            )
            launch = recovery.launch_phase(migration_path, phase="resume")
            _write_state(
                root,
                stage="wait_resume",
                status="launched",
                details=launch,
            )
            training = _terminal_statuses(
                root,
                phase="resume",
                timeout_s=training_timeout_s,
                poll_s=poll_s,
            )
            training_receipts = _receipt_summary(training)
            trainer = training_receipts["trainer"]
            sentinel = training_receipts["sentinel"]
            if (
                trainer["returncode"] != 0
                or sentinel["returncode"] != 0
                or any(
                    record[key] is not expected
                    for record in (trainer, sentinel)
                    for key, expected in (
                        ("containment_verified", True),
                        ("process_group_empty", True),
                        ("lineage_empty", True),
                    )
                )
                or trainer["restart_count"] != 0
                or sentinel["restart_count"] != 0
            ):
                _fail("resume_terminal_evidence_invalid")
            return _write_verdict(
                root,
                decision="training_terminal_pending_strict_admission",
                migration_sha256=migration_sha256,
                details={
                    "calibration_verdict_sha256": calibration_verdict["verdict_sha256"],
                    "training_receipts": training_receipts,
                },
            )
        except Exception as exc:
            failure = {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            _write_state(root, stage="failed", status="failed", details=failure)
            return _write_verdict(
                root,
                decision="recovery_failed_closed",
                migration_sha256=migration_sha256,
                details=failure,
            )


def launch_controller(migration_path: Path) -> dict[str, Any]:
    root, _adapter_root = _controller_root(migration_path)
    run_dir = root / "detached-recovery-controller"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--migration",
        str(migration_path.expanduser().resolve(strict=True)),
    ]
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/run_detached_step.py"),
            "launch",
            "--run-dir",
            str(run_dir),
            "--name",
            f"{root.name}-recovery-controller",
            "--cwd",
            str(REPO_ROOT),
            "--timeout",
            "216000",
            "--",
            *command,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    if completed.returncode != 0:
        _fail(f"controller_launch_failed:{completed.returncode}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ResidentV3RecoveryControllerError("controller_launch_output_invalid") from exc
    if not isinstance(value, dict):
        _fail("controller_launch_output_invalid")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "launch"):
        child = subparsers.add_parser(command)
        child.add_argument("--migration", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = (
            launch_controller(args.migration)
            if args.command == "launch"
            else run_controller(args.migration)
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("decision") != "recovery_failed_closed" else 1
    except Exception as exc:
        print(
            f"run_resident_v3_recovery_controller: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
