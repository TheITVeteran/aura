#!/usr/bin/env python3
"""Drive one signed latent-cortex campaign through every durable phase.

The controller never invents campaign material. It executes only packets that
the preparation/advancement verifiers accept and signs only exact requests
already persisted by the frozen runner. Its state is durable so an OS-level
supervisor can restart it without repeating accepted campaign work.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.campaign_launch_bundle import (  # noqa: E402
    LAUNCH_PACKET_FILE,
    read_canonical_json,
)
from core.brain.llm.latent_cortex.campaign_trust import (  # noqa: E402
    CAMPAIGN_RUNNER,
    TASK_ISSUER,
)
from core.learning.verified_transition_production_factory import (  # noqa: E402
    CommandRoleSignerBroker,
)
from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_bytes,
    ensure_private_directory,
)
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402
from tools import advance_latent_cortex_campaign as advancement  # noqa: E402

CONTROLLER_STATUS_SCHEMA = "aura.latent_cortex.campaign_controller_status.v1"
SIGNER_CONFIG_SCHEMA = "aura.verified_transition.external_signer_config.v1"
_SIGNER_CONFIG_KEYS = frozenset(
    {
        "schema",
        "identity",
        "executable",
        "executable_sha256",
        "release_manifest",
        "custody_evidence",
        "arguments",
        "timeout_millis",
        "inherited_environment_names",
    }
)
_PACKET_PHASES = {
    "ready_for_inference": LAUNCH_PACKET_FILE,
    "inference_in_progress_or_resumable": LAUNCH_PACKET_FILE,
    "post_reveal_scoring_or_resume": advancement.ANSWER_RESUME_PACKET_FILE,
}
_SIGNABLE_PHASES = {
    "awaiting_answer_reveal_signature": (TASK_ISSUER, "task_issuer"),
    "awaiting_final_run_signature": (CAMPAIGN_RUNNER, "campaign_runner"),
}
_active_child: subprocess.Popen[bytes] | None = None
_stop_signal = 0


class CampaignControllerError(RuntimeError):
    """One campaign could not be advanced without weakening its contract."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise CampaignControllerError(code)


def _strict_json(path: Path, *, role: str) -> dict[str, Any]:
    resolved = path.expanduser()
    if not resolved.is_absolute() or resolved.is_symlink():
        _fail(f"{role}_path_invalid")
    raw = read_stable_bytes(resolved.resolve(strict=True), max_bytes=64 * 1024 * 1024)
    try:
        document = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise CampaignControllerError(f"{role}_json_invalid") from exc
    if not isinstance(document, dict) or raw not in {
        canonical_json_bytes(document),
        canonical_json_bytes(document) + b"\n",
    }:
        _fail(f"{role}_noncanonical")
    return document


def _load_broker(path: Path, *, policy: Any, role: str) -> CommandRoleSignerBroker:
    document = _strict_json(path, role=f"{role}_signer_config")
    if set(document) != _SIGNER_CONFIG_KEYS or document.get("schema") != SIGNER_CONFIG_SCHEMA:
        _fail(f"{role}_signer_config_schema_invalid")
    timeout = document.get("timeout_millis")
    if (
        type(timeout) is not int
        or not 100 <= timeout <= 300_000
        or not isinstance(document.get("arguments"), list)
        or not isinstance(document.get("inherited_environment_names"), list)
    ):
        _fail(f"{role}_signer_config_invalid")
    broker = CommandRoleSignerBroker(
        identity=document["identity"],
        executable=document["executable"],
        executable_sha256=document["executable_sha256"],
        release_manifest=document["release_manifest"],
        custody_evidence=document["custody_evidence"],
        arguments=document["arguments"],
        timeout_seconds=timeout / 1000,
        inherited_environment_names=document["inherited_environment_names"],
    )
    pin = policy.role_pin(role)
    if (
        broker.identity != pin["signer_id"]
        or broker.implementation_sha256 != pin["implementation_sha256"]
        or broker.release_sha256 != pin["release_sha256"]
        or broker.custody_evidence_sha256 != pin["custody_evidence_sha256"]
    ):
        _fail(f"{role}_signer_config_policy_mismatch")
    return broker


def _write_status(
    path: Path,
    *,
    phase: str,
    sequence: int,
    state: str,
    child_pid: int = 0,
    child_pgid: int = 0,
    packet_path: str = "",
    returncode: int | None = None,
    reason: str = "",
) -> None:
    document = {
        "schema": CONTROLLER_STATUS_SCHEMA,
        "controller_pid": os.getpid(),
        "phase": phase,
        "state": state,
        "sequence": sequence,
        "heartbeat_at_unix": time.time(),
        "child_pid": child_pid,
        "child_process_group_id": child_pgid,
        "packet_path": packet_path,
        "returncode": returncode,
        "reason": reason,
    }
    atomic_write_bytes(path, canonical_json_bytes(document) + b"\n", mode=0o600)


def _signal_handler(signum: int, _frame: Any) -> None:
    global _stop_signal
    _stop_signal = signum
    child = _active_child
    if child is not None and child.poll() is None:
        try:
            os.killpg(os.getpgid(child.pid), signum)
        except ProcessLookupError:
            pass


def _packet(bundle_dir: Path, packet_path: Path) -> dict[str, Any]:
    context = advancement._context(bundle_dir)
    resolved = packet_path.resolve(strict=True)
    allowed = {
        (bundle_dir / LAUNCH_PACKET_FILE).resolve(strict=True),
        *(
            path.resolve(strict=True)
            for path in (
                bundle_dir / advancement.ANSWER_RESUME_PACKET_FILE,
                bundle_dir / advancement.FINAL_RESUME_PACKET_FILE,
            )
            if path.exists()
        ),
    }
    if resolved not in allowed:
        _fail("controller_packet_outside_verified_bundle")
    packet = read_canonical_json(resolved, role="campaign_controller_packet")
    if packet_path.name == LAUNCH_PACKET_FILE:
        if packet != context["launch_packet"]:
            _fail("controller_prelaunch_packet_mismatch")
    else:
        advancement._verify_persisted_phase_packets(context)
    argv = packet.get("argv")
    working_directory = packet.get("working_directory")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(value, str) or not value for value in argv)
        or not isinstance(working_directory, str)
        or Path(working_directory).resolve(strict=True) != REPO_ROOT.resolve(strict=True)
    ):
        _fail("controller_packet_execution_invalid")
    return packet


def _execute_packet(
    *,
    bundle_dir: Path,
    packet_path: Path,
    state_dir: Path,
    status_path: Path,
    sequence: int,
    heartbeat_seconds: float,
) -> int:
    global _active_child
    packet = _packet(bundle_dir, packet_path)
    caffeinate = shutil.which("caffeinate")
    if not caffeinate:
        _fail("caffeinate_unavailable")
    log_path = state_dir / "campaign-controller.log"
    environment = dict(os.environ)
    environment["AURA_LOG_DIR"] = str(ensure_private_directory(state_dir / "aura-logs"))
    with log_path.open("ab", buffering=0) as log:
        child = subprocess.Popen(
            [caffeinate, "-dims", *packet["argv"]],
            cwd=packet["working_directory"],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _active_child = child
        pgid = os.getpgid(child.pid)
        try:
            while True:
                returncode = child.poll()
                _write_status(
                    status_path,
                    phase=str(packet.get("phase") or "inference"),
                    sequence=sequence,
                    state="running",
                    child_pid=child.pid,
                    child_pgid=pgid,
                    packet_path=str(packet_path),
                    returncode=returncode,
                )
                if returncode is not None:
                    return returncode
                if _stop_signal:
                    return child.wait(timeout=30)
                time.sleep(heartbeat_seconds)
        finally:
            _active_child = None


def _persist_attestation(
    state_dir: Path,
    *,
    request_sha256: str,
    attestation: Mapping[str, Any],
) -> Path:
    path = state_dir / "signatures" / f"{request_sha256}.json"
    ensure_private_directory(path.parent)
    payload = canonical_json_bytes(attestation) + b"\n"
    if path.exists():
        if read_stable_bytes(path, max_bytes=4 * 1024 * 1024) != payload:
            _fail("controller_attestation_replay_conflict")
    else:
        atomic_write_bytes(path, payload, mode=0o600)
    return path


def run_controller(args: argparse.Namespace) -> dict[str, Any]:
    bundle_dir = args.bundle_dir.expanduser().resolve(strict=True)
    requested_state_dir = args.state_dir.expanduser()
    if requested_state_dir.is_symlink():
        _fail("controller_state_directory_symlink_rejected")
    state_dir = ensure_private_directory(
        Path(os.path.abspath(os.fspath(requested_state_dir)))
    ).resolve(strict=True)
    status_path = state_dir / "status.json"
    lock_path = state_dir / "controller.lock"
    lock_fd = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignControllerError("controller_already_running") from exc
        os.fchmod(lock_fd, 0o600)
        context = advancement._context(bundle_dir)
        policy = advancement._policy(context["launch_spec"], observed_at=None)
        brokers = {
            TASK_ISSUER: _load_broker(
                args.task_issuer_signer_config,
                policy=policy,
                role=TASK_ISSUER,
            ),
            CAMPAIGN_RUNNER: _load_broker(
                args.campaign_runner_signer_config,
                policy=policy,
                role=CAMPAIGN_RUNNER,
            ),
        }
        sequence = 0
        phase = "initializing"
        try:
            while sequence < args.max_phase_executions:
                if _stop_signal:
                    _fail(f"controller_interrupted:{_stop_signal}")
                phase_status = advancement.status(argparse.Namespace(bundle_dir=bundle_dir))
                phase = phase_status["phase"]
                _write_status(
                    status_path,
                    phase=phase,
                    sequence=sequence,
                    state="advancing",
                )
                if phase == "campaign_evidence_sealed":
                    _write_status(
                        status_path,
                        phase=phase,
                        sequence=sequence,
                        state="complete",
                    )
                    return phase_status
                if phase == "awaiting_prelaunch_signatures":
                    _fail("prelaunch_launch_packet_missing")
                packet_path: Path
                signable = _SIGNABLE_PHASES.get(phase)
                if signable is not None:
                    role, _label = signable
                    request_path = Path(phase_status["request_path"])
                    request = read_canonical_json(
                        request_path,
                        role=f"{role}_controller_request",
                    )
                    attestation = brokers[role].attest_prepared_request(
                        policy,
                        role=role,
                        request=request,
                    )
                    attestation_path = _persist_attestation(
                        state_dir,
                        request_sha256=request["request_sha256"],
                        attestation=attestation,
                    )
                    admission = advancement.admit(
                        argparse.Namespace(
                            bundle_dir=bundle_dir,
                            attestation=attestation_path,
                            observed_at=int(time.time()),
                        )
                    )
                    packet_path = Path(admission["packet_path"])
                else:
                    packet_name = _PACKET_PHASES.get(phase)
                    if packet_name is None:
                        _fail(f"campaign_phase_unsupported:{phase}")
                    packet_path = bundle_dir / packet_name
                sequence += 1
                returncode = _execute_packet(
                    bundle_dir=bundle_dir,
                    packet_path=packet_path,
                    state_dir=state_dir,
                    status_path=status_path,
                    sequence=sequence,
                    heartbeat_seconds=args.heartbeat_seconds,
                )
                next_status = advancement.status(
                    argparse.Namespace(bundle_dir=bundle_dir)
                )
                next_phase = next_status["phase"]
                if returncode not in {0, 2, 5, 6} and next_phase == phase:
                    _fail(f"campaign_child_failed:{returncode}:{phase}")
            _fail("campaign_phase_execution_limit_exceeded")
        except BaseException as exc:
            _write_status(
                status_path,
                phase=phase,
                sequence=sequence,
                state="failed",
                reason=getattr(exc, "code", str(exc)) or type(exc).__name__,
            )
            raise
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--task-issuer-signer-config", type=Path, required=True)
    parser.add_argument("--campaign-runner-signer-config", type=Path, required=True)
    parser.add_argument("--heartbeat-seconds", type=float, default=10.0)
    parser.add_argument("--max-phase-executions", type=int, default=8)
    return parser


def main() -> int:
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _signal_handler)
    args = build_parser().parse_args()
    try:
        if not 0.25 <= args.heartbeat_seconds <= 300.0:
            _fail("heartbeat_interval_invalid")
        if not 1 <= args.max_phase_executions <= 32:
            _fail("phase_execution_limit_invalid")
        result = run_controller(args)
        sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
        return 0
    except (CampaignControllerError, OSError, ValueError, KeyError) as exc:
        reason = getattr(exc, "code", str(exc)) or type(exc).__name__
        print(f"latent_cortex_campaign_controller: {reason}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
