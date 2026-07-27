#!/usr/bin/env python3
"""Prepare or launch a kernel-contained synthetic recurrent-SFT experiment.

The trainer remains governed by its exact research authority. This operator
adds an independently frozen, deny-default macOS sandbox and a sanitized
environment before handing the target to Aura's crash-observable detached
supervisor. Evaluator, replay, resident-model, adapter-registry, fusion,
promotion, network, and process-fork access remain outside the target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.structured_sft_research_authority import (  # noqa: E402
    StructuredSFTResearchAuthorityError,
    canonical_json_bytes,
    strict_json_bytes,
    validate_authority,
)
from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_bytes,
    ensure_private_directory,
)
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402
from tools import run_detached_step  # noqa: E402

CONTRACT_SCHEMA = "aura.rlc.synthetic_recurrent_sft_containment_contract.v1"
RESULT_SCHEMA = "aura.rlc.synthetic_recurrent_sft_containment_operator.v1"
PROFILE_SCHEMA = "aura.rlc.synthetic_recurrent_sft_sandbox_profile.v1"
SANDBOX_PATH = Path("/usr/bin/sandbox-exec")
_MAX_JSON_BYTES = 256 * 1024 * 1024
_MAX_TIMEOUT_S = 4 * 60 * 60
_SAFE_ACTIONS = frozenset({"prepare", "launch"})


class StructuredSFTContainmentError(RuntimeError):
    """The containment contract could not be constructed exactly."""


def _fail(code: str) -> Never:
    raise StructuredSFTContainmentError(
        str(code or "structured_sft_containment_failed")
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        return strict_json_bytes(
            read_stable_bytes(
                path.expanduser().resolve(strict=True),
                max_bytes=_MAX_JSON_BYTES,
            ),
            role=f"containment_{role}",
        )
    except (OSError, StructuredSFTResearchAuthorityError) as exc:
        raise StructuredSFTContainmentError(
            f"structured_sft_containment_{role}_unreadable"
        ) from exc


def _existing_file(path: Path, *, role: str) -> Path:
    expanded = Path(os.path.abspath(os.fspath(path.expanduser())))
    if expanded.is_symlink():
        _fail(f"structured_sft_containment_{role}_symlink_rejected")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file():
        _fail(f"structured_sft_containment_{role}_file_required")
    return resolved


def _python_launcher(path: Path) -> Path:
    launcher = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        metadata = launcher.lstat()
        resolved = launcher.resolve(strict=True)
    except OSError as exc:
        raise StructuredSFTContainmentError(
            "structured_sft_containment_python_unreadable"
        ) from exc
    if (
        not (stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode))
        or not resolved.is_file()
        or not os.access(launcher, os.X_OK)
    ):
        _fail("structured_sft_containment_python_launcher_invalid")
    return launcher


def _existing_directory(path: Path, *, role: str) -> Path:
    expanded = Path(os.path.abspath(os.fspath(path.expanduser())))
    if expanded.is_symlink():
        _fail(f"structured_sft_containment_{role}_symlink_rejected")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_dir():
        _fail(f"structured_sft_containment_{role}_directory_required")
    return resolved


def _private_directory(path: Path, *, role: str) -> Path:
    expanded = Path(os.path.abspath(os.fspath(path.expanduser())))
    if expanded.is_symlink():
        _fail(f"structured_sft_containment_{role}_symlink_rejected")
    directory = ensure_private_directory(expanded)
    resolved = directory.resolve(strict=True)
    metadata = resolved.stat()
    if (
        metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        _fail(f"structured_sft_containment_{role}_not_private")
    return resolved


def _sb_quote(value: str) -> str:
    if not value or "\x00" in value or "\n" in value or "\r" in value:
        _fail("structured_sft_containment_profile_value_invalid")
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _literal(path: Path) -> str:
    return f'(literal "{_sb_quote(str(path))}")'


def _path_rules(
    paths: Sequence[Path],
    *,
    include_ancestors: bool = False,
) -> list[str]:
    rules: list[str] = []
    unique = set(paths)
    if include_ancestors:
        for path in tuple(unique):
            unique.update(path.parents)
    for path in sorted(unique, key=lambda item: os.fsencode(item)):
        rules.append(f"    {_literal(path)}")
        if path in paths and path.is_dir():
            rules.append(f'    (subpath "{_sb_quote(str(path))}")')
    return rules


def _normalized_forbidden(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _forbidden_roots(
    *,
    candidate_dir: Path,
    model_dir: Path,
) -> tuple[Path, ...]:
    home = Path.home().resolve(strict=True)
    likely: set[Path] = {
        _normalized_forbidden(candidate_dir.parent / "evaluator"),
        _normalized_forbidden(home / ".aura/private/rlc/verified_replay_sft"),
        _normalized_forbidden(home / ".aura/private/rlc/promotion"),
        _normalized_forbidden(home / ".aura/adapters"),
        _normalized_forbidden(home / ".aura/adapter_registry"),
        _normalized_forbidden(home / ".aura/model_registry"),
        _normalized_forbidden(home / ".aura/fusion"),
        _normalized_forbidden(REPO_ROOT / "models/adapters"),
        _normalized_forbidden(REPO_ROOT / "artifacts/closeout/verified_replay"),
    }
    try:
        model_siblings = tuple(model_dir.parent.iterdir())
    except OSError as exc:
        raise StructuredSFTContainmentError(
            "structured_sft_containment_model_parent_unreadable"
        ) from exc
    likely.update(
        sibling.resolve(strict=True)
        for sibling in model_siblings
        if sibling != model_dir and sibling.is_dir()
    )
    inherited_model = str(os.environ.get("AURA_MODEL_PATH", "") or "").strip()
    if inherited_model:
        likely.add(_normalized_forbidden(Path(inherited_model)))
    return tuple(sorted(likely, key=lambda item: os.fsencode(item)))


def _python_execution_paths(python: Path) -> tuple[Path, ...]:
    paths = {python, python.resolve(strict=True)}
    base_executable = Path(
        str(getattr(sys, "_base_executable", sys.executable))
    ).resolve(strict=True)
    paths.add(base_executable)
    app_python = (
        Path(sys.base_prefix)
        / "Resources/Python.app/Contents/MacOS/Python"
    )
    if app_python.is_file():
        paths.add(app_python.resolve(strict=True))
    return tuple(sorted(paths, key=lambda item: os.fsencode(item)))


def build_sandbox_profile(
    *,
    python: Path,
    read_paths: Sequence[Path],
    write_paths: Sequence[Path],
    forbidden_roots: Sequence[Path],
    model_lane_state: Path,
) -> str:
    """Return the exact deny-default SBPL policy used by target and verifier."""

    execution_paths = _python_execution_paths(python)
    model_lane_parent = model_lane_state.parent
    model_lane_lock = model_lane_state.with_suffix(
        model_lane_state.suffix + ".lock"
    )
    escaped_lane_parent = _sb_quote(str(model_lane_parent)).replace(".", r"\.")
    lines = [
        "(version 1)",
        "(debug deny)",
        "(deny default)",
        '(import "system.sb")',
        "(deny network*)",
        "(deny process-fork)",
        "(allow process-info*)",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        "(allow iokit-open)",
        "(allow ipc-posix-shm*)",
        "(allow signal (target self))",
        "(allow process-exec",
        *_path_rules(execution_paths),
        ")",
        "(allow file-read*",
        *_path_rules(read_paths, include_ancestors=True),
        f"    {_literal(model_lane_parent)}",
        f"    {_literal(model_lane_state)}",
        f"    {_literal(model_lane_lock)}",
        ")",
        "(allow file-write*",
        *_path_rules(write_paths),
        f"    {_literal(model_lane_parent)}",
        f"    {_literal(model_lane_state)}",
        f"    {_literal(model_lane_lock)}",
        (
            '    (regex #"^'
            + escaped_lane_parent
            + r'/\.aura_atomic_[^/]+$")'
        ),
        '    (literal "/dev/null")',
        ")",
    ]
    for root in forbidden_roots:
        quoted = _sb_quote(str(root))
        lines.extend(
            (
                "(deny file-read*",
                f'    (literal "{quoted}")',
                f'    (subpath "{quoted}")',
                ")",
                "(deny file-write*",
                f'    (literal "{quoted}")',
                f'    (subpath "{quoted}")',
                ")",
            )
        )
    return "\n".join(lines) + "\n"


def _write_create_or_verify(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink():
            _fail("structured_sft_containment_artifact_symlink_rejected")
        try:
            observed = path.read_bytes()
        except OSError as exc:
            raise StructuredSFTContainmentError(
                "structured_sft_containment_artifact_unreadable"
            ) from exc
        if observed != payload:
            _fail("structured_sft_containment_artifact_commitment_mismatch")
        return
    atomic_write_bytes(path, payload, mode=0o600)


def _assert_disjoint(
    *,
    allowed_reads: Sequence[Path],
    allowed_writes: Sequence[Path],
    forbidden_roots: Sequence[Path],
) -> None:
    for forbidden in forbidden_roots:
        for allowed in (*allowed_reads, *allowed_writes):
            if allowed == forbidden or allowed.is_relative_to(forbidden):
                _fail("structured_sft_containment_forbidden_overlap")


def _sanitized_environment(
    *,
    python: Path,
    runtime_root: Path,
    model_lane_state: Path,
) -> dict[str, str]:
    environment = {
        "AURA_HOME": str(runtime_root / "home"),
        "AURA_MODEL_LANE_STATE_PATH": str(model_lane_state),
        "HF_HOME": str(runtime_root / "cache/huggingface"),
        "HOME": str(runtime_root / "home"),
        "HUGGINGFACE_HUB_CACHE": str(
            runtime_root / "cache/huggingface/hub"
        ),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MLX_METAL_CACHE_DIR": str(runtime_root / "cache/mlx"),
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(REPO_ROOT),
        "TMPDIR": str(runtime_root / "tmp"),
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_CACHE": str(runtime_root / "cache/transformers"),
        "USER": str(os.environ.get("USER", "")),
        "LOGNAME": str(os.environ.get("LOGNAME", "")),
        "VIRTUAL_ENV": str(python.parent.parent),
    }
    if any(
        key == "AURA_MODEL_PATH" or "EVALUATOR" in key or "REPLAY" in key
        for key in environment
    ):
        _fail("structured_sft_containment_environment_forbidden")
    return dict(sorted(environment.items()))


@contextmanager
def _exact_environment(environment: Mapping[str, str]) -> Iterator[None]:
    prior = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(environment)
        yield
    finally:
        os.environ.clear()
        os.environ.update(prior)


def _command_contains_forbidden(
    command: Sequence[str],
    forbidden_roots: Sequence[Path],
) -> bool:
    material = "\0".join(command)
    return any(str(path) in material for path in forbidden_roots)


def build_contract(arguments: argparse.Namespace) -> dict[str, Any]:
    if sys.platform != "darwin" or not SANDBOX_PATH.is_file():
        _fail("structured_sft_containment_macos_sandbox_required")
    python = _python_launcher(arguments.python)
    authority_path = _existing_file(arguments.authority, role="authority")
    authority = validate_authority(
        _read_json(authority_path, role="authority"),
        expected_authority_sha256=arguments.expected_authority_sha256,
        allow_expired_resume=bool(arguments.resume),
    )
    audit_packet = _existing_file(arguments.audit_packet, role="audit_packet")
    witness_bundle = _existing_file(
        arguments.witness_bundle,
        role="witness_bundle",
    )
    trusted_log_key = _existing_file(
        arguments.trusted_log_key,
        role="trusted_log_key",
    )
    execution_spec = _existing_file(
        arguments.execution_spec,
        role="execution_spec",
    )
    candidate_dir = _existing_directory(
        arguments.candidate_dir,
        role="candidate",
    )
    candidate_custody_commit = _existing_file(
        candidate_dir.parent / ".aura_structured_sft_custody.commit.json",
        role="candidate_custody_commit",
    )
    tokenizer_dir = _existing_directory(
        arguments.tokenizer_dir,
        role="tokenizer",
    )
    snapshot_root = _existing_directory(
        arguments.snapshot_root,
        role="snapshot_root",
    )
    model_dir = _existing_directory(arguments.model_dir, role="model")
    authority_model = Path(authority["model"]["directory"]).resolve(strict=True)
    authority_snapshot = Path(
        authority["tokenization"]["snapshot_path"]
    ).resolve(strict=True)
    if (
        model_dir != authority_model
        or tokenizer_dir != authority_model
        or authority_snapshot.parent != snapshot_root
        or authority["execution_spec"]["semantic_sha256"]
        != arguments.expected_execution_spec_sha256
    ):
        _fail("structured_sft_containment_authority_path_drift")

    contract_dir = _private_directory(arguments.contract_dir, role="contract")
    training_run_dir = _private_directory(
        arguments.training_run_dir,
        role="training_run",
    )
    detached_run_dir = _private_directory(
        arguments.detached_run_dir,
        role="detached_run",
    )
    if len({contract_dir, training_run_dir, detached_run_dir}) != 3:
        _fail("structured_sft_containment_run_roots_not_disjoint")
    runtime_root = _private_directory(
        training_run_dir / "runtime",
        role="runtime",
    )
    for relative in (
        "home",
        "tmp",
        "cache",
        "cache/huggingface",
        "cache/huggingface/hub",
        "cache/mlx",
        "cache/transformers",
    ):
        _private_directory(runtime_root / relative, role="runtime_child")
    model_lane_state = Path(
        os.path.abspath(os.fspath(arguments.model_lane_state.expanduser()))
    )
    if model_lane_state.is_symlink():
        _fail("structured_sft_containment_model_lane_symlink_rejected")
    model_lane_parent = _private_directory(
        model_lane_state.parent,
        role="model_lane_parent",
    )
    model_lane_state = model_lane_parent / model_lane_state.name

    forbidden_roots = _forbidden_roots(
        candidate_dir=candidate_dir,
        model_dir=model_dir,
    )
    read_paths = (
        Path("/System"),
        Path("/Library"),
        Path("/usr"),
        Path("/bin"),
        Path("/sbin"),
        Path("/opt"),
        Path("/private/etc"),
        Path("/private/var/db/timezone"),
        Path("/dev/null"),
        REPO_ROOT.resolve(strict=True),
        python.parent.parent.resolve(strict=True),
        candidate_dir,
        candidate_custody_commit,
        tokenizer_dir,
        authority_snapshot,
        model_dir,
        contract_dir,
        training_run_dir,
        detached_run_dir,
        authority_path,
        audit_packet,
        witness_bundle,
        trusted_log_key,
        execution_spec,
    )
    write_paths = (training_run_dir, detached_run_dir, runtime_root)
    _assert_disjoint(
        allowed_reads=read_paths,
        allowed_writes=write_paths,
        forbidden_roots=forbidden_roots,
    )
    profile = build_sandbox_profile(
        python=python,
        read_paths=read_paths,
        write_paths=write_paths,
        forbidden_roots=forbidden_roots,
        model_lane_state=model_lane_state,
    )
    profile_path = contract_dir / "synthetic-recurrent-sft.sb"
    _write_create_or_verify(profile_path, profile.encode("utf-8"))

    trainer_command = [
        str(SANDBOX_PATH),
        "-f",
        str(profile_path),
        str(python),
        str(REPO_ROOT / "tools/train_structured_sft_research.py"),
        "--authority",
        str(authority_path),
        "--expected-authority-sha256",
        authority["authority_sha256"],
        "--audit-packet",
        str(audit_packet),
        "--witness-bundle",
        str(witness_bundle),
        "--trusted-log-key",
        str(trusted_log_key),
        "--witness-sequence",
        str(arguments.witness_sequence),
        "--candidate-dir",
        str(candidate_dir),
        "--tokenizer-dir",
        str(tokenizer_dir),
        "--snapshot-root",
        str(snapshot_root),
        "--model-dir",
        str(model_dir),
        "--execution-spec",
        str(execution_spec),
        "--out-dir",
        str(training_run_dir),
        "--resume-policy",
        "auto",
    ]
    verifier_command = [
        str(SANDBOX_PATH),
        "-f",
        str(profile_path),
        str(python),
        str(REPO_ROOT / "tools/verify_structured_sft_research_resume.py"),
        "--authority",
        str(authority_path),
        "--expected-authority-sha256",
        authority["authority_sha256"],
        "--run-dir",
        str(training_run_dir),
    ]
    if _command_contains_forbidden(trainer_command, forbidden_roots) or (
        _command_contains_forbidden(verifier_command, forbidden_roots)
    ):
        _fail("structured_sft_containment_forbidden_path_in_command")
    environment = _sanitized_environment(
        python=python,
        runtime_root=runtime_root,
        model_lane_state=model_lane_state,
    )
    body = {
        "schema": CONTRACT_SCHEMA,
        "profile_schema": PROFILE_SCHEMA,
        "authority_sha256": authority["authority_sha256"],
        "profile_path": str(profile_path),
        "profile_sha256": _sha256_bytes(profile.encode("utf-8")),
        "sandbox_executable_sha256": _sha256_bytes(
            read_stable_bytes(SANDBOX_PATH, max_bytes=8 * 1024 * 1024)
        ),
        "trainer_command": trainer_command,
        "trainer_command_sha256": _sha256_json(trainer_command),
        "resume_verifier_command": verifier_command,
        "resume_verifier_command_sha256": _sha256_json(verifier_command),
        "environment": environment,
        "environment_sha256": _sha256_json(environment),
        "read_paths": [str(path) for path in read_paths],
        "write_paths": [str(path) for path in write_paths],
        "forbidden_roots": [str(path) for path in forbidden_roots],
        "network": "kernel_denied",
        "process_fork": "kernel_denied",
        "evaluator_access": False,
        "verified_replay_access": False,
        "resident_checkpoint_access": False,
        "production_write_access": False,
        "training_run_dir": str(training_run_dir),
        "detached_run_dir": str(detached_run_dir),
        "model_lane_state": str(model_lane_state),
        "timeout_s": float(arguments.timeout),
        "resume": bool(arguments.resume),
        "claims_not_supported": authority["claims_not_supported"],
    }
    contract = {**body, "contract_sha256": _sha256_json(body)}
    _write_create_or_verify(
        contract_dir / "containment_contract.json",
        canonical_json_bytes(contract),
    )
    return contract


def launch_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    environment = contract.get("environment")
    trainer_command = contract.get("trainer_command")
    verifier_command = contract.get("resume_verifier_command")
    if (
        not isinstance(environment, Mapping)
        or not isinstance(trainer_command, list)
        or not isinstance(verifier_command, list)
    ):
        _fail("structured_sft_containment_contract_invalid")
    argv = [
        "launch",
        "--run-dir",
        str(contract["detached_run_dir"]),
        "--name",
        "spark-059-synthetic-recurrent-sft",
        "--cwd",
        str(REPO_ROOT),
        "--timeout",
        str(contract["timeout_s"]),
        "--resume-contract",
        "target_checkpoint",
        "--containment-mode",
        "precontained-sandbox",
        "--resume-verifier-json",
        json.dumps(verifier_command, separators=(",", ":")),
    ]
    if contract.get("resume") is True:
        argv.append("--resume")
    argv.extend(["--", *trainer_command])
    parser = run_detached_step.build_parser()
    arguments = parser.parse_args(argv)
    with _exact_environment(
        {str(key): str(value) for key, value in environment.items()}
    ):
        detached = run_detached_step._launch(arguments, parser)
    return {
        "schema": RESULT_SCHEMA,
        "status": "detached_training_launched",
        "contract_sha256": contract["contract_sha256"],
        "authority_sha256": contract["authority_sha256"],
        "training_run_dir": contract["training_run_dir"],
        "detached_run_dir": contract["detached_run_dir"],
        "detached": detached,
        "claims_not_supported": contract["claims_not_supported"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=sorted(_SAFE_ACTIONS))
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--audit-packet", type=Path, required=True)
    parser.add_argument("--witness-bundle", type=Path, required=True)
    parser.add_argument("--trusted-log-key", type=Path, required=True)
    parser.add_argument("--witness-sequence", type=int, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--execution-spec", type=Path, required=True)
    parser.add_argument("--expected-execution-spec-sha256", required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--training-run-dir", type=Path, required=True)
    parser.add_argument("--detached-run-dir", type=Path, required=True)
    parser.add_argument(
        "--model-lane-state",
        type=Path,
        default=Path.home() / ".aura/run/model_lane_control.json",
    )
    parser.add_argument("--timeout", type=float, default=2 * 60 * 60)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if (
        arguments.witness_sequence < 1
        or not 1.0 <= arguments.timeout <= _MAX_TIMEOUT_S
        or len(arguments.expected_authority_sha256) != 64
        or len(arguments.expected_execution_spec_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in (
                arguments.expected_authority_sha256
                + arguments.expected_execution_spec_sha256
            )
        )
    ):
        parser.error("authority, execution program, sequence, or timeout invalid")
    try:
        contract = build_contract(arguments)
        result = (
            {
                "schema": RESULT_SCHEMA,
                "status": "containment_contract_prepared",
                "contract_sha256": contract["contract_sha256"],
                "authority_sha256": contract["authority_sha256"],
                "profile_sha256": contract["profile_sha256"],
                "training_run_dir": contract["training_run_dir"],
                "detached_run_dir": contract["detached_run_dir"],
                "claims_not_supported": contract["claims_not_supported"],
            }
            if arguments.action == "prepare"
            else launch_contract(contract)
        )
    except (
        OSError,
        StructuredSFTContainmentError,
        StructuredSFTResearchAuthorityError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {
                    "schema": "aura.rlc.synthetic_recurrent_sft_containment_error.v1",
                    "ok": False,
                    "reason": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
