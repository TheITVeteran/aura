"""Durable file-protocol execution for long external evidence verifiers.

The detached runner owns process lifetime, containment, sleep protection, and
its terminal receipt.  This module adds only deterministic request/result
custody and reconciliation around that runner.  The verifier target is pinned
    and receives ``--request`` and ``--result`` arguments.  It must treat
an already-valid result as an idempotent completed checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never, cast

from core.learning.verified_transition_episode import canonical_json_bytes
from core.runtime.atomic_writer import (
    atomic_write_bytes_if_absent,
    ensure_private_directory,
    interprocess_file_lock,
)
from core.runtime.file_read_gateway import read_stable_bytes

DURABLE_EXTERNAL_VERIFIER_JOB_SCHEMA = "aura.verified_transition.durable_external_verifier_job.v1"
_DETACHED_PLAN_SCHEMA = "aura.detached_step.plan.v2"
_DETACHED_LAUNCH_SCHEMA = "aura.detached_step.launch.v1"
_DETACHED_INSPECTION_SCHEMA = "aura.detached_step.inspection.v1"
_MAX_RUNNER_OUTPUT_BYTES = 1 << 20
_MAX_CONTRACT_BYTES = 1 << 20
_MAX_EXECUTABLE_BYTES = 512 * 1024 * 1024
_MAX_TIMEOUT_SECONDS = 93_600.0


class DurableExternalVerifierJobError(RuntimeError):
    """A durable verifier job could not be proven valid or successful."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise DurableExternalVerifierJobError(code)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _generic_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise DurableExternalVerifierJobError("durable_verifier_detached_json_invalid") from exc


def _canonical_clone(value: Any, *, role: str) -> dict[str, Any]:
    try:
        raw = canonical_json_bytes(value)
        cloned = json.loads(raw)
    except (TypeError, ValueError, UnicodeError, RecursionError, OverflowError):
        _fail(f"{role}_not_canonical_json")
    if not isinstance(cloned, dict):
        _fail(f"{role}_not_object")
    return cast(dict[str, Any], cloned)


def _sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{role}_invalid")
    return value


def _private_directory(path: str | Path, *, role: str) -> Path:
    candidate = Path(path).expanduser().absolute()
    if candidate.is_symlink():
        _fail(f"{role}_symlink_rejected")
    try:
        directory = ensure_private_directory(candidate).resolve(strict=True)
        metadata = directory.stat()
    except OSError as exc:
        raise DurableExternalVerifierJobError(f"{role}_unavailable") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        _fail(f"{role}_not_private_owned_directory")
    return directory


def _regular_executable(
    path: str | Path,
    *,
    expected_sha256: str,
    role: str,
) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or candidate.is_symlink():
        _fail(f"{role}_invalid")
    try:
        resolved = candidate.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise DurableExternalVerifierJobError(f"{role}_unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or not os.access(resolved, os.X_OK)
    ):
        _fail(f"{role}_invalid")
    observed = _sha256_bytes(read_stable_bytes(resolved, max_bytes=_MAX_EXECUTABLE_BYTES))
    if observed != _sha256(expected_sha256, role=f"{role}_sha256"):
        _fail(f"{role}_identity_mismatch")
    return resolved


def _regular_python_source(path: str | Path, *, role: str) -> tuple[Path, str]:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or candidate.is_symlink():
        _fail(f"{role}_invalid")
    try:
        resolved = candidate.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise DurableExternalVerifierJobError(f"{role}_unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        _fail(f"{role}_invalid")
    payload = read_stable_bytes(resolved, max_bytes=16 * 1024 * 1024)
    return resolved, _sha256_bytes(payload)


def _read_private_file(path: Path, *, max_bytes: int, role: str) -> bytes:
    if path.is_symlink():
        _fail(f"{role}_symlink_rejected")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise DurableExternalVerifierJobError(f"{role}_unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or metadata.st_nlink != 1
    ):
        _fail(f"{role}_not_private_owned_file")
    try:
        return read_stable_bytes(path, max_bytes=max_bytes)
    except OSError as exc:
        raise DurableExternalVerifierJobError(f"{role}_unstable") from exc


def _read_private_json(
    path: Path,
    *,
    max_bytes: int,
    role: str,
    require_canonical: bool,
) -> tuple[dict[str, Any], bytes]:
    raw = _read_private_file(path, max_bytes=max_bytes, role=role)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DurableExternalVerifierJobError(f"{role}_json_invalid") from exc
    if not isinstance(value, dict):
        _fail(f"{role}_not_object")
    normalized = canonical_json_bytes(value) if require_canonical else raw
    if require_canonical and raw != normalized:
        _fail(f"{role}_not_canonical")
    return cast(dict[str, Any], value), normalized


@dataclass(frozen=True, slots=True)
class DurableExternalVerifierSubmission:
    """Stable paths and identity for one submitted verifier request."""

    job_id: str
    request_sha256: str
    job_dir: Path
    run_dir: Path
    request_path: Path
    candidate_result_path: Path
    result_path: Path
    target_command: tuple[str, ...]
    resume_verifier_command: tuple[str, ...]


class DurableExternalVerifierJob:
    """Run one immutable verifier request under ``run_detached_step.py``.

    The external target owns only a candidate output.  A successful, verified
    detached receipt is required before that candidate can be copied into the
    broker-owned create-once result.  Reconstructing this object after caller
    death reconciles the existing detached run instead of launching a duplicate.
    """

    def __init__(
        self,
        *,
        job_root: str | Path,
        executable: str | Path,
        executable_sha256: str,
        cwd: str | Path,
        detached_runner: str | Path,
        resume_helper: str | Path,
        arguments: Sequence[str] = (),
        timeout_seconds: float,
        result_max_bytes: int = 64 * 1024 * 1024,
        request_max_bytes: int = 256 * 1024 * 1024,
        poll_interval_seconds: float = 0.1,
        runner_call_timeout_seconds: float = 30.0,
        require_sleep_protection: bool = True,
    ) -> None:
        self._root = _private_directory(job_root, role="durable_verifier_root")
        self._executable = _regular_executable(
            executable,
            expected_sha256=executable_sha256,
            role="durable_verifier_executable",
        )
        self._executable_sha256 = executable_sha256
        self._cwd = Path(cwd).expanduser()
        if not self._cwd.is_absolute() or self._cwd.is_symlink():
            _fail("durable_verifier_cwd_invalid")
        try:
            self._cwd = self._cwd.resolve(strict=True)
        except OSError as exc:
            raise DurableExternalVerifierJobError("durable_verifier_cwd_unavailable") from exc
        if not self._cwd.is_dir():
            _fail("durable_verifier_cwd_invalid")
        self._runner, self._runner_sha256 = _regular_python_source(
            detached_runner,
            role="durable_verifier_detached_runner",
        )
        self._resume_helper, self._resume_helper_sha256 = _regular_python_source(
            resume_helper,
            role="durable_verifier_resume_helper",
        )
        normalized_arguments = tuple(arguments)
        if any(
            not isinstance(argument, str) or not argument or "\x00" in argument
            for argument in normalized_arguments
        ) or any(argument in {"--request", "--result"} for argument in normalized_arguments):
            _fail("durable_verifier_arguments_invalid")
        self._arguments = normalized_arguments
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or not 0.1 <= float(timeout_seconds) <= _MAX_TIMEOUT_SECONDS
        ):
            _fail("durable_verifier_timeout_invalid")
        self._timeout_seconds = float(timeout_seconds)
        if (
            type(result_max_bytes) is not int
            or not 1 <= result_max_bytes <= 256 * 1024 * 1024
            or type(request_max_bytes) is not int
            or not 1 <= request_max_bytes <= 512 * 1024 * 1024
        ):
            _fail("durable_verifier_size_bound_invalid")
        self._result_max_bytes = result_max_bytes
        self._request_max_bytes = request_max_bytes
        if (
            isinstance(poll_interval_seconds, bool)
            or not isinstance(poll_interval_seconds, (int, float))
            or not math.isfinite(float(poll_interval_seconds))
            or not 0.01 <= float(poll_interval_seconds) <= 10.0
            or isinstance(runner_call_timeout_seconds, bool)
            or not isinstance(runner_call_timeout_seconds, (int, float))
            or not math.isfinite(float(runner_call_timeout_seconds))
            or not 1.0 <= float(runner_call_timeout_seconds) <= 120.0
        ):
            _fail("durable_verifier_polling_invalid")
        self._poll_interval_seconds = float(poll_interval_seconds)
        self._runner_call_timeout_seconds = float(runner_call_timeout_seconds)
        if not isinstance(require_sleep_protection, bool):
            _fail("durable_verifier_sleep_protection_invalid")
        self._require_sleep_protection = require_sleep_protection

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    @property
    def target_command(self) -> tuple[str, ...]:
        """Return the frozen command prefix before file-protocol arguments."""

        return (str(self._executable), *self._arguments)

    def _assert_source_identities(self) -> None:
        observed_executable = _sha256_bytes(
            read_stable_bytes(
                self._executable,
                max_bytes=_MAX_EXECUTABLE_BYTES,
            )
        )
        observed_runner = _sha256_bytes(read_stable_bytes(self._runner, max_bytes=16 * 1024 * 1024))
        observed_resume = _sha256_bytes(
            read_stable_bytes(
                self._resume_helper,
                max_bytes=16 * 1024 * 1024,
            )
        )
        if (
            observed_executable != self._executable_sha256
            or observed_runner != self._runner_sha256
            or observed_resume != self._resume_helper_sha256
        ):
            _fail("durable_verifier_source_identity_mismatch")

    def _validate_response(
        self,
        path: Path,
        *,
        request_sha256: str,
        role: str,
    ) -> tuple[dict[str, Any], bytes]:
        response, raw = _read_private_json(
            path,
            max_bytes=self._result_max_bytes,
            role=role,
            require_canonical=True,
        )
        if response.get("request_sha256") != request_sha256:
            _fail(f"{role}_request_mismatch")
        return response, raw

    def _prepare(
        self,
        request: Mapping[str, Any],
    ) -> DurableExternalVerifierSubmission:
        request_document = _canonical_clone(
            request,
            role="durable_verifier_request",
        )
        request_sha256 = _sha256(
            request_document.get("request_sha256"),
            role="durable_verifier_request_sha256",
        )
        request_bytes = canonical_json_bytes(request_document)
        if len(request_bytes) > self._request_max_bytes:
            _fail("durable_verifier_request_too_large")
        job_id = _sha256_bytes(request_bytes)
        job_dir = _private_directory(
            self._root / job_id,
            role="durable_verifier_job_directory",
        )
        run_dir = _private_directory(
            job_dir / "detached",
            role="durable_verifier_run_directory",
        )
        request_path = job_dir / "request.json"
        # The detached runner treats existing *.json command arguments as
        # executable inputs. Extensionless outputs keep a newly materialized
        # candidate from changing the frozen plan during resume.
        candidate_path = job_dir / "candidate-result"
        result_path = job_dir / "result"
        # Keep the frozen helper outside the runner's excluded run directory.
        # Its extensionless name prevents unrelated mutable job JSON from
        # becoming part of the runner's source-tree execution manifest; the
        # create-once job contract binds the helper bytes directly.
        frozen_resume_helper = job_dir / "frozen-resume-helper"
        target_command = (
            str(self._executable),
            *self._arguments,
            "--request",
            str(request_path),
            "--result",
            str(candidate_path),
        )
        resume_command = (
            sys.executable,
            str(frozen_resume_helper),
            "--request-file",
            str(request_path),
            "--candidate-result-file",
            str(candidate_path),
            "--authoritative-result-file",
            str(result_path),
            "--job-id",
            job_id,
            "--result-max-bytes",
            str(self._result_max_bytes),
        )
        contract_body = {
            "schema": DURABLE_EXTERNAL_VERIFIER_JOB_SCHEMA,
            "job_id": job_id,
            "request_sha256": request_sha256,
            "request_file_sha256": _sha256_bytes(request_bytes),
            "executable": str(self._executable),
            "executable_sha256": self._executable_sha256,
            "arguments": list(self._arguments),
            "cwd": str(self._cwd),
            "detached_runner": str(self._runner),
            "detached_runner_sha256": self._runner_sha256,
            "resume_helper": str(self._resume_helper),
            "resume_helper_sha256": self._resume_helper_sha256,
            "frozen_resume_helper": str(frozen_resume_helper),
            "timeout_millis": round(self._timeout_seconds * 1000),
            "request_max_bytes": self._request_max_bytes,
            "result_max_bytes": self._result_max_bytes,
            "require_sleep_protection": self._require_sleep_protection,
            "target_command": list(target_command),
            "resume_verifier_command": list(resume_command),
        }
        contract = {
            **contract_body,
            "contract_sha256": _sha256_bytes(canonical_json_bytes(contract_body)),
        }
        contract_bytes = canonical_json_bytes(contract)
        with interprocess_file_lock(job_dir / ".custody.lock"):
            resume_helper_bytes = read_stable_bytes(
                self._resume_helper,
                max_bytes=16 * 1024 * 1024,
            )
            if _sha256_bytes(resume_helper_bytes) != self._resume_helper_sha256:
                _fail("durable_verifier_source_identity_mismatch")
            if not atomic_write_bytes_if_absent(
                frozen_resume_helper,
                resume_helper_bytes,
                mode=0o500,
            ):
                existing_resume_helper = _read_private_file(
                    frozen_resume_helper,
                    max_bytes=16 * 1024 * 1024,
                    role="durable_verifier_frozen_resume_helper",
                )
                if existing_resume_helper != resume_helper_bytes:
                    _fail("durable_verifier_frozen_resume_helper_conflict")
            if not atomic_write_bytes_if_absent(
                request_path,
                request_bytes,
                mode=0o600,
            ):
                existing, existing_bytes = _read_private_json(
                    request_path,
                    max_bytes=self._request_max_bytes,
                    role="durable_verifier_request",
                    require_canonical=True,
                )
                if existing != request_document or existing_bytes != request_bytes:
                    _fail("durable_verifier_request_conflict")
            contract_path = job_dir / "job-contract.json"
            if not atomic_write_bytes_if_absent(
                contract_path,
                contract_bytes,
                mode=0o600,
            ):
                existing_contract, existing_contract_bytes = _read_private_json(
                    contract_path,
                    max_bytes=_MAX_CONTRACT_BYTES,
                    role="durable_verifier_job_contract",
                    require_canonical=True,
                )
                if existing_contract != contract or existing_contract_bytes != contract_bytes:
                    _fail("durable_verifier_job_contract_conflict")
        return DurableExternalVerifierSubmission(
            job_id=job_id,
            request_sha256=request_sha256,
            job_dir=job_dir,
            run_dir=run_dir,
            request_path=request_path,
            candidate_result_path=candidate_path,
            result_path=result_path,
            target_command=target_command,
            resume_verifier_command=resume_command,
        )

    def _runner_call(
        self,
        arguments: Sequence[str],
        *,
        environment: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        self._assert_source_identities()
        try:
            completed = subprocess.run(
                [sys.executable, str(self._runner), *arguments],
                check=False,
                capture_output=True,
                text=False,
                timeout=self._runner_call_timeout_seconds,
                shell=False,
                env=dict(environment) if environment is not None else None,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise DurableExternalVerifierJobError(
                "durable_verifier_detached_runner_failed"
            ) from exc
        if (
            completed.returncode != 0
            or len(completed.stdout) > _MAX_RUNNER_OUTPUT_BYTES
            or len(completed.stderr) > _MAX_RUNNER_OUTPUT_BYTES
        ):
            _fail("durable_verifier_detached_runner_failed")
        try:
            payload = json.loads(completed.stdout)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise DurableExternalVerifierJobError(
                "durable_verifier_detached_runner_response_invalid"
            ) from exc
        if not isinstance(payload, dict):
            _fail("durable_verifier_detached_runner_response_invalid")
        return cast(dict[str, Any], payload)

    def _launch(
        self,
        submission: DurableExternalVerifierSubmission,
        *,
        resume: bool,
    ) -> dict[str, Any]:
        arguments = [
            "launch",
            "--run-dir",
            str(submission.run_dir),
            "--name",
            f"external-verifier-{submission.job_id[:24]}",
            "--cwd",
            str(self._cwd),
            "--timeout",
            str(self._timeout_seconds),
            "--resume-contract",
            "target_checkpoint",
            "--resume-verifier-json",
            json.dumps(
                list(submission.resume_verifier_command),
                separators=(",", ":"),
            ),
        ]
        if resume:
            arguments.append("--resume")
        arguments.extend(["--", *submission.target_command])
        frozen_environment: Mapping[str, str] | None = None
        if resume:
            prior_plan = self._validate_detached_plan(submission)
            candidate_environment = prior_plan.get("execution_environment")
            if not isinstance(candidate_environment, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in candidate_environment.items()
            ):
                _fail("durable_verifier_detached_environment_invalid")
            frozen_environment = cast(dict[str, str], candidate_environment)
        launch = self._runner_call(
            arguments,
            environment=frozen_environment,
        )
        if (
            launch.get("schema") != _DETACHED_LAUNCH_SCHEMA
            or launch.get("run_dir") != str(submission.run_dir)
            or launch.get("resumed") is not resume
        ):
            _fail("durable_verifier_detached_launch_invalid")
        self._validate_detached_plan(submission)
        return launch

    def _status(
        self,
        submission: DurableExternalVerifierSubmission,
    ) -> dict[str, Any]:
        status = self._runner_call(["status", "--run-dir", str(submission.run_dir)])
        if status.get("schema") != _DETACHED_INSPECTION_SCHEMA or status.get("run_dir") != str(
            submission.run_dir
        ):
            _fail("durable_verifier_detached_status_invalid")
        plan = self._validate_detached_plan(submission)
        if status.get("plan_sha256") != plan["plan_sha256"]:
            _fail("durable_verifier_detached_status_plan_mismatch")
        return status

    def _validate_detached_plan(
        self,
        submission: DurableExternalVerifierSubmission,
    ) -> dict[str, Any]:
        plan, _canonical = _read_private_json(
            submission.run_dir / "detached_plan.json",
            max_bytes=16 * 1024 * 1024,
            role="durable_verifier_detached_plan",
            require_canonical=False,
        )
        body = {key: value for key, value in plan.items() if key != "plan_sha256"}
        if (
            plan.get("schema") != _DETACHED_PLAN_SCHEMA
            or plan.get("plan_sha256") != _sha256_bytes(_generic_json_bytes(body))
            or plan.get("command") != list(submission.target_command)
            or plan.get("cwd") != str(self._cwd)
            or plan.get("timeout_s") != self._timeout_seconds
            or plan.get("resume_contract") != "target_checkpoint"
            or plan.get("resume_verifier_command") != list(submission.resume_verifier_command)
            or plan.get("restart_policy") != "never"
        ):
            _fail("durable_verifier_detached_plan_mismatch")
        if self._require_sleep_protection:
            assertion = plan.get("power_assertion")
            if (
                not isinstance(assertion, dict)
                or not isinstance(assertion.get("path"), str)
                or not assertion.get("path")
                or not isinstance(assertion.get("sha256"), str)
                or len(assertion["sha256"]) != 64
            ):
                _fail("durable_verifier_sleep_protection_unavailable")
        return plan

    def _reconcile_locked(
        self,
        submission: DurableExternalVerifierSubmission,
    ) -> dict[str, Any]:
        plan_path = submission.run_dir / "detached_plan.json"
        if not plan_path.exists():
            if submission.result_path.exists():
                _fail("durable_verifier_result_without_detached_receipt")
            self._launch(submission, resume=False)
        status = self._status(submission)
        if status.get("completion_indeterminate") is True:
            last_error: DurableExternalVerifierJobError | None = None
            for attempt in range(3):
                try:
                    self._launch(submission, resume=True)
                except DurableExternalVerifierJobError as exc:
                    last_error = exc
                    status = self._status(submission)
                    if status.get("completion_indeterminate") is not True:
                        break
                    if attempt < 2:
                        time.sleep(0.05)
                        continue
                    raise
                else:
                    status = self._status(submission)
                    break
            if status.get("completion_indeterminate") is True and last_error is not None:
                raise last_error
        return status

    def begin(
        self,
        request: Mapping[str, Any],
    ) -> DurableExternalVerifierSubmission:
        """Create or recover one job and ensure a detached attempt is active."""

        submission = self._prepare(request)
        with interprocess_file_lock(submission.job_dir / ".lifecycle.lock"):
            status = self._reconcile_locked(submission)
            if status.get("terminal") is True:
                self._accept_terminal_locked(submission, status)
        return submission

    def _accept_terminal_locked(
        self,
        submission: DurableExternalVerifierSubmission,
        status: Mapping[str, Any],
    ) -> dict[str, Any]:
        receipt = status.get("receipt")
        if (
            status.get("terminal") is not True
            or not isinstance(receipt, Mapping)
            or receipt.get("status") != "passed"
            or receipt.get("passed") is not True
            or receipt.get("returncode") != 0
            or receipt.get("timed_out") is not False
            or receipt.get("stop_signal") is not None
            or receipt.get("containment_verified") is not True
            or receipt.get("command") != list(submission.target_command)
            or receipt.get("plan_sha256") != status.get("plan_sha256")
        ):
            _fail("durable_verifier_detached_execution_not_successful")
        response, candidate_bytes = self._validate_response(
            submission.candidate_result_path,
            request_sha256=submission.request_sha256,
            role="durable_verifier_candidate_result",
        )
        if not atomic_write_bytes_if_absent(
            submission.result_path,
            candidate_bytes,
            mode=0o600,
        ):
            existing, existing_bytes = self._validate_response(
                submission.result_path,
                request_sha256=submission.request_sha256,
                role="durable_verifier_result",
            )
            if existing != response or existing_bytes != candidate_bytes:
                _fail("durable_verifier_result_conflict")
        accepted, accepted_bytes = self._validate_response(
            submission.result_path,
            request_sha256=submission.request_sha256,
            role="durable_verifier_result",
        )
        if accepted != response or accepted_bytes != candidate_bytes:
            _fail("durable_verifier_result_conflict")
        self._assert_source_identities()
        return accepted

    def wait(
        self,
        submission: DurableExternalVerifierSubmission,
        *,
        caller_timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Wait for a verified terminal receipt without owning target lifetime."""

        if caller_timeout_seconds is None:
            wait_seconds = self._timeout_seconds + 30.0
        elif (
            isinstance(caller_timeout_seconds, bool)
            or not isinstance(caller_timeout_seconds, (int, float))
            or not math.isfinite(float(caller_timeout_seconds))
            or float(caller_timeout_seconds) <= 0.0
        ):
            _fail("durable_verifier_caller_timeout_invalid")
        else:
            wait_seconds = float(caller_timeout_seconds)
        deadline = time.monotonic() + wait_seconds
        while True:
            with interprocess_file_lock(submission.job_dir / ".lifecycle.lock"):
                status = self._reconcile_locked(submission)
                if status.get("terminal") is True:
                    return self._accept_terminal_locked(submission, status)
            if time.monotonic() >= deadline:
                _fail("durable_verifier_caller_wait_timeout")
            time.sleep(self._poll_interval_seconds)

    def execute(
        self,
        request: Mapping[str, Any],
        *,
        caller_timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Submit, recover, and return only an accepted create-once result."""

        submission = self.begin(request)
        return self.wait(
            submission,
            caller_timeout_seconds=caller_timeout_seconds,
        )

    def run_file_protocol(
        self,
        request: Mapping[str, Any],
        target_command: Sequence[str],
        timeout_seconds: float,
        purpose: str,
    ) -> dict[str, Any]:
        """Validate one call-site contract before durable execution."""

        command = tuple(target_command)
        if command != self.target_command:
            _fail("durable_verifier_target_command_mismatch")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or float(timeout_seconds) != self._timeout_seconds
        ):
            _fail("durable_verifier_timeout_mismatch")
        if (
            not isinstance(purpose, str)
            or not purpose
            or purpose != purpose.strip()
            or len(purpose) > 256
            or request.get("purpose") != purpose
        ):
            _fail("durable_verifier_purpose_mismatch")
        return self.execute(request)
