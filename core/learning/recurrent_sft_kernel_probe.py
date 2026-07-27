"""Executable, replayable kernel-containment evidence for recurrent SFT."""

from __future__ import annotations

import errno
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Never

PROBE_SPEC_SCHEMA: Final = "aura.rlc.recurrent_sft_kernel_probe.v1.spec"
PROBE_RECEIPT_SCHEMA: Final = "aura.rlc.recurrent_sft_kernel_probe.v1.receipt"
PROBE_OPERATIONS: Final = {
    "evaluator_read": "allowed",
    "network": "denied",
    "process_fork": "denied",
    "production_write": "denied",
    "resident_read": "denied",
    "training_write": "denied",
}
_TARGET_ROLES: Final = frozenset(
    {
        "evaluator_read",
        "production_write",
        "resident_read",
        "training_write",
    }
)
_DENIAL_ERRNOS: Final = frozenset({errno.EACCES, errno.EPERM})
_MAX_PROBE_OUTPUT_BYTES: Final = 64 * 1024

_PROBE_SOURCE: Final = r"""import errno,json,os,socket,sys
paths=json.loads(sys.argv[1])
def attempt(fn):
 try:
  result=fn()
  return {"denied":False,"errno":None,"result":result}
 except OSError as exc:
  return {"denied":exc.errno in (errno.EPERM,errno.EACCES),"errno":exc.errno,"result":None}
def read_one(path):
 fd=os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0))
 try:
  return len(os.read(fd,1))
 finally:
  os.close(fd)
def open_write(path):
 fd=os.open(path,os.O_WRONLY|getattr(os,"O_NOFOLLOW",0))
 os.close(fd)
 return True
def network():
 sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
 try:
  sock.connect(("127.0.0.1",9))
 finally:
  sock.close()
def process_fork():
 pid=os.fork()
 if pid==0:
  os._exit(0)
 os.waitpid(pid,0)
 return True
observations={
 "evaluator_read":attempt(lambda:read_one(paths["evaluator_read"])),
 "network":attempt(network),
 "process_fork":attempt(process_fork),
 "production_write":attempt(lambda:open_write(paths["production_write"])),
 "resident_read":attempt(lambda:read_one(paths["resident_read"])),
 "training_write":attempt(lambda:open_write(paths["training_write"])),
}
print(json.dumps(observations,sort_keys=True,separators=(",",":"),allow_nan=False))
"""


class RecurrentSFTKernelProbeError(RuntimeError):
    """The kernel probe or its evidence failed its exact contract."""


def _fail(code: str) -> Never:
    raise RecurrentSFTKernelProbeError(
        str(code or "recurrent_sft_kernel_probe_failed")
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _strict_json_bytes(payload: bytes, *, role: str) -> dict[str, Any]:
    if len(payload) > _MAX_PROBE_OUTPUT_BYTES:
        _fail(f"recurrent_sft_kernel_probe_{role}_oversized")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                _fail(f"recurrent_sft_kernel_probe_{role}_duplicate_key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: _fail(
                f"recurrent_sft_kernel_probe_{role}_nonfinite"
            ),
        )
    except (RecursionError, UnicodeError, ValueError) as exc:
        raise RecurrentSFTKernelProbeError(
            f"recurrent_sft_kernel_probe_{role}_json_invalid"
        ) from exc
    if not isinstance(value, dict):
        _fail(f"recurrent_sft_kernel_probe_{role}_invalid")
    return value


def _file_binding(path: Path, *, role: str) -> dict[str, Any]:
    lexical = path.expanduser()
    if lexical.is_symlink():
        _fail(f"recurrent_sft_kernel_probe_{role}_symlink_rejected")
    try:
        resolved = lexical.resolve(strict=True)
        before = resolved.stat()
        payload = resolved.read_bytes()
        after = resolved.stat()
    except OSError as exc:
        raise RecurrentSFTKernelProbeError(
            f"recurrent_sft_kernel_probe_{role}_unreadable"
        ) from exc
    if (
        not resolved.is_file()
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(payload) != after.st_size
    ):
        _fail(f"recurrent_sft_kernel_probe_{role}_unstable")
    return {
        "path": str(resolved),
        "sha256": _sha256(payload),
        "size_bytes": len(payload),
    }


def build_kernel_probe_spec(
    *,
    sandbox_executable: Path,
    profile: Path,
    python: Path,
    targets: Mapping[str, Path],
) -> dict[str, Any]:
    """Build an exact command and bind every file it may touch."""

    if set(targets) != _TARGET_ROLES:
        _fail("recurrent_sft_kernel_probe_target_roles_invalid")
    sandbox_binding = _file_binding(sandbox_executable, role="sandbox")
    profile_binding = _file_binding(profile, role="profile")
    python_binding = _file_binding(python.resolve(strict=True), role="python")
    target_bindings = {
        role: _file_binding(targets[role], role=f"target_{role}")
        for role in sorted(_TARGET_ROLES)
    }
    target_paths = {
        role: target_bindings[role]["path"] for role in sorted(_TARGET_ROLES)
    }
    command = [
        sandbox_binding["path"],
        "-f",
        profile_binding["path"],
        str(python.expanduser()),
        "-I",
        "-c",
        _PROBE_SOURCE,
        _canonical_json_bytes(target_paths).decode("ascii"),
    ]
    body = {
        "schema": PROBE_SPEC_SCHEMA,
        "sandbox": sandbox_binding,
        "profile": profile_binding,
        "python": python_binding,
        "source_sha256": _sha256(_PROBE_SOURCE.encode("ascii")),
        "targets": target_bindings,
        "expectations": dict(PROBE_OPERATIONS),
        "command": command,
        "command_sha256": _sha256(_canonical_json_bytes(command)),
    }
    return {**body, "spec_sha256": _sha256(_canonical_json_bytes(body))}


def _validate_file_binding_schema(value: Any, *, role: str) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "size_bytes"}
        or not isinstance(value.get("path"), str)
        or not value["path"]
        or not Path(value["path"]).is_absolute()
        or not _is_sha256(value.get("sha256"))
        or type(value.get("size_bytes")) is not int
        or value["size_bytes"] < 0
    ):
        _fail(f"recurrent_sft_kernel_probe_{role}_binding_invalid")
    return dict(value)


def validate_kernel_probe_spec(
    spec: Mapping[str, Any],
    *,
    rebind_files: bool = True,
) -> dict[str, Any]:
    """Validate a probe specification, optionally rereading its bound files.

    The launcher and independent verifier use the default external rebinding.
    The contained evaluator cannot reread targets whose denied access is the
    property under test, so it validates the canonical signed structure only.
    """

    expected_keys = {
        "schema",
        "sandbox",
        "profile",
        "python",
        "source_sha256",
        "targets",
        "expectations",
        "command",
        "command_sha256",
        "spec_sha256",
    }
    if not isinstance(spec, Mapping) or set(spec) != expected_keys:
        _fail("recurrent_sft_kernel_probe_spec_schema_invalid")
    body = dict(spec)
    observed_sha256 = body.pop("spec_sha256")
    if (
        spec.get("schema") != PROBE_SPEC_SCHEMA
        or observed_sha256 != _sha256(_canonical_json_bytes(body))
        or spec.get("source_sha256") != _sha256(_PROBE_SOURCE.encode("ascii"))
        or spec.get("expectations") != PROBE_OPERATIONS
        or not isinstance(spec.get("targets"), Mapping)
        or set(spec["targets"]) != _TARGET_ROLES
        or not isinstance(spec.get("command"), list)
        or spec.get("command_sha256")
        != _sha256(_canonical_json_bytes(spec["command"]))
    ):
        _fail("recurrent_sft_kernel_probe_spec_invalid")
    sandbox = _validate_file_binding_schema(spec.get("sandbox"), role="sandbox")
    profile = _validate_file_binding_schema(spec.get("profile"), role="profile")
    _validate_file_binding_schema(spec.get("python"), role="python")
    targets = {
        role: _validate_file_binding_schema(
            spec["targets"].get(role),
            role=f"target_{role}",
        )
        for role in sorted(_TARGET_ROLES)
    }
    command = spec["command"]
    expected_command = [
        sandbox["path"],
        "-f",
        profile["path"],
        command[3] if len(command) > 3 else None,
        "-I",
        "-c",
        _PROBE_SOURCE,
        _canonical_json_bytes(
            {role: targets[role]["path"] for role in sorted(_TARGET_ROLES)}
        ).decode("ascii"),
    ]
    if (
        type(rebind_files) is not bool
        or len(command) != len(expected_command)
        or command != expected_command
        or not isinstance(command[3], str)
        or not command[3]
        or not Path(command[3]).is_absolute()
    ):
        _fail("recurrent_sft_kernel_probe_spec_command_invalid")
    if not rebind_files:
        return dict(spec)
    rebound = build_kernel_probe_spec(
        sandbox_executable=Path(spec["sandbox"]["path"]),
        profile=Path(spec["profile"]["path"]),
        python=Path(spec["command"][3]),
        targets={
            role: Path(spec["targets"][role]["path"])
            for role in sorted(_TARGET_ROLES)
        },
    )
    if dict(spec) != rebound:
        _fail("recurrent_sft_kernel_probe_spec_binding_mismatch")
    return rebound


def _validate_observations(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(PROBE_OPERATIONS):
        _fail("recurrent_sft_kernel_probe_observations_invalid")
    observations: dict[str, Any] = {}
    for operation, expectation in PROBE_OPERATIONS.items():
        observation = value.get(operation)
        if (
            not isinstance(observation, Mapping)
            or set(observation) != {"denied", "errno", "result"}
            or type(observation.get("denied")) is not bool
        ):
            _fail("recurrent_sft_kernel_probe_observation_schema_invalid")
        if expectation == "allowed":
            valid = (
                observation["denied"] is False
                and observation["errno"] is None
                and observation["result"] == 1
            )
        else:
            valid = (
                observation["denied"] is True
                and type(observation["errno"]) is int
                and observation["errno"] in _DENIAL_ERRNOS
                and observation["result"] is None
            )
        if not valid:
            _fail(f"recurrent_sft_kernel_probe_{operation}_expectation_failed")
        observations[operation] = dict(observation)
    return observations


def execute_kernel_probe(
    spec: Mapping[str, Any],
    *,
    contract_sha256: str,
    environment: Mapping[str, str],
    cwd: Path,
) -> dict[str, Any]:
    """Execute the frozen probe and return deterministic evidence."""

    validated = validate_kernel_probe_spec(spec)
    if not _is_sha256(contract_sha256):
        _fail("recurrent_sft_kernel_probe_contract_sha256_invalid")
    try:
        completed = subprocess.run(
            validated["command"],
            cwd=str(cwd.resolve(strict=True)),
            env={str(key): str(value) for key, value in environment.items()},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=30.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RecurrentSFTKernelProbeError(
            "recurrent_sft_kernel_probe_execution_failed"
        ) from exc
    if (
        completed.returncode != 0
        or len(completed.stdout) > _MAX_PROBE_OUTPUT_BYTES
        or len(completed.stderr) > _MAX_PROBE_OUTPUT_BYTES
    ):
        _fail("recurrent_sft_kernel_probe_process_failed")
    observations = _strict_json_bytes(
        completed.stdout.rstrip(b"\n"),
        role="stdout",
    )
    if completed.stdout != _canonical_json_bytes(observations) + b"\n":
        _fail("recurrent_sft_kernel_probe_stdout_noncanonical")
    validated_observations = _validate_observations(observations)
    body = {
        "schema": PROBE_RECEIPT_SCHEMA,
        "contract_sha256": contract_sha256,
        "spec_sha256": validated["spec_sha256"],
        "command_sha256": validated["command_sha256"],
        "returncode": completed.returncode,
        "stdout_sha256": _sha256(completed.stdout),
        "stderr_sha256": _sha256(completed.stderr),
        "observations": validated_observations,
        "all_expectations_met": True,
    }
    return {**body, "receipt_sha256": _sha256(_canonical_json_bytes(body))}


def validate_kernel_probe_receipt(
    receipt: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    contract_sha256: str,
    rebind_files: bool = True,
) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "contract_sha256",
        "spec_sha256",
        "command_sha256",
        "returncode",
        "stdout_sha256",
        "stderr_sha256",
        "observations",
        "all_expectations_met",
        "receipt_sha256",
    }
    validated_spec = validate_kernel_probe_spec(
        spec,
        rebind_files=rebind_files,
    )
    if not isinstance(receipt, Mapping) or set(receipt) != expected_keys:
        _fail("recurrent_sft_kernel_probe_receipt_schema_invalid")
    body = dict(receipt)
    observed_sha256 = body.pop("receipt_sha256")
    if (
        receipt.get("schema") != PROBE_RECEIPT_SCHEMA
        or receipt.get("contract_sha256") != contract_sha256
        or receipt.get("spec_sha256") != validated_spec["spec_sha256"]
        or receipt.get("command_sha256") != validated_spec["command_sha256"]
        or receipt.get("returncode") != 0
        or not _is_sha256(receipt.get("stdout_sha256"))
        or not _is_sha256(receipt.get("stderr_sha256"))
        or receipt.get("all_expectations_met") is not True
        or observed_sha256 != _sha256(_canonical_json_bytes(body))
    ):
        _fail("recurrent_sft_kernel_probe_receipt_invalid")
    _validate_observations(receipt.get("observations"))
    return dict(receipt)


__all__ = [
    "PROBE_OPERATIONS",
    "PROBE_RECEIPT_SCHEMA",
    "PROBE_SPEC_SCHEMA",
    "RecurrentSFTKernelProbeError",
    "build_kernel_probe_spec",
    "execute_kernel_probe",
    "validate_kernel_probe_receipt",
    "validate_kernel_probe_spec",
]
