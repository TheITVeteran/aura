#!/usr/bin/env python3
"""Prepare or launch kernel-contained recurrent-SFT negative controls."""

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
    execution_spec_identity,
    small_model_identity,
    source_closure,
    strict_json_bytes,
    validate_authority,
)
from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_bytes,
    ensure_private_directory,
)
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402
from tools import run_detached_step  # noqa: E402
from tools.launch_structured_sft_research import (  # noqa: E402
    SANDBOX_PATH,
    build_sandbox_profile,
)
from tools.train_recurrent_sft_controls import control_source_paths  # noqa: E402

CONTRACT_SCHEMA = "aura.rlc.synthetic_recurrent_sft_control_containment.v1"
RESULT_SCHEMA = "aura.rlc.synthetic_recurrent_sft_control_operator.v1"
PROFILE_SCHEMA = "aura.rlc.synthetic_recurrent_sft_control_sandbox.v1"
_SAFE_ACTIONS = frozenset({"prepare", "launch"})
_MAX_DOCUMENT_BYTES = 256 * 1024 * 1024
_MAX_TIMEOUT_S = 4 * 60 * 60


class RecurrentSFTControlContainmentError(RuntimeError):
    """The control run could not be frozen behind the required kernel policy."""


def _fail(code: str) -> Never:
    raise RecurrentSFTControlContainmentError(
        str(code or "recurrent_sft_control_containment_failed")
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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_bytes(path: Path, *, role: str) -> bytes:
    try:
        return read_stable_bytes(
            path.expanduser().resolve(strict=True),
            max_bytes=_MAX_DOCUMENT_BYTES,
        )
    except OSError as exc:
        raise RecurrentSFTControlContainmentError(
            f"control_containment_{role}_unreadable"
        ) from exc


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        return strict_json_bytes(
            _read_bytes(path, role=role),
            role=f"control_containment_{role}",
        )
    except StructuredSFTResearchAuthorityError as exc:
        raise RecurrentSFTControlContainmentError(
            f"control_containment_{role}_invalid"
        ) from exc


def _existing_file(path: Path, *, role: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    if lexical.is_symlink():
        _fail(f"control_containment_{role}_symlink_rejected")
    resolved = lexical.resolve(strict=True)
    if not resolved.is_file():
        _fail(f"control_containment_{role}_file_required")
    return resolved


def _existing_directory(path: Path, *, role: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    if lexical.is_symlink():
        _fail(f"control_containment_{role}_symlink_rejected")
    resolved = lexical.resolve(strict=True)
    if not resolved.is_dir():
        _fail(f"control_containment_{role}_directory_required")
    return resolved


def _private_directory(path: Path, *, role: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    if lexical.is_symlink():
        _fail(f"control_containment_{role}_symlink_rejected")
    resolved = ensure_private_directory(lexical).resolve(strict=True)
    metadata = resolved.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        _fail(f"control_containment_{role}_not_private")
    return resolved


def _python_launcher(path: Path) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        metadata = lexical.lstat()
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise RecurrentSFTControlContainmentError(
            "control_containment_python_unreadable"
        ) from exc
    if (
        not (stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode))
        or not resolved.is_file()
        or not os.access(lexical, os.X_OK)
    ):
        _fail("control_containment_python_invalid")
    return lexical


def _write_create_or_verify(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != payload:
            _fail("control_containment_artifact_commitment_mismatch")
        return
    atomic_write_bytes(path, payload, mode=0o600)


def _forbidden_roots(*, projected_dataset: Path, model_dir: Path) -> tuple[Path, ...]:
    home = Path.home().resolve(strict=True)
    likely = {
        projected_dataset.parent.parent / "evaluator",
        home / ".aura/private/rlc/verified_replay_sft",
        home / ".aura/private/rlc/promotion",
        home / ".aura/private/rlc/frontier_campaigns",
        home / ".aura/adapters",
        home / ".aura/adapter_registry",
        home / ".aura/model_registry",
        home / ".aura/fusion",
        REPO_ROOT / "models/adapters",
        REPO_ROOT / "artifacts/closeout/verified_replay",
    }
    likely.update(
        sibling.resolve(strict=True)
        for sibling in model_dir.parent.iterdir()
        if sibling != model_dir and sibling.is_dir()
    )
    inherited = str(os.environ.get("AURA_MODEL_PATH", "") or "").strip()
    if inherited:
        likely.add(Path(os.path.abspath(os.path.expanduser(inherited))))
    return tuple(
        sorted(
            {Path(os.path.abspath(os.fspath(path.expanduser()))) for path in likely},
            key=lambda path: os.fsencode(path),
        )
    )


def _assert_disjoint(
    *,
    reads: Sequence[Path],
    writes: Sequence[Path],
    forbidden: Sequence[Path],
) -> None:
    for root in forbidden:
        for allowed in (*reads, *writes):
            if allowed == root or allowed.is_relative_to(root):
                _fail("control_containment_forbidden_overlap")


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
        "LOGNAME": str(os.environ.get("LOGNAME", "")),
        "MLX_METAL_CACHE_DIR": str(runtime_root / "cache/mlx"),
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(REPO_ROOT),
        "TMPDIR": str(runtime_root / "tmp"),
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_CACHE": str(runtime_root / "cache/transformers"),
        "USER": str(os.environ.get("USER", "")),
        "VIRTUAL_ENV": str(python.parent.parent),
    }
    if any(
        key == "AURA_MODEL_PATH" or "EVALUATOR" in key or "REPLAY" in key
        for key in environment
    ):
        _fail("control_containment_environment_forbidden")
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


def build_contract(arguments: argparse.Namespace) -> dict[str, Any]:
    if sys.platform != "darwin" or not SANDBOX_PATH.is_file():
        _fail("control_containment_macos_sandbox_required")
    python = _python_launcher(arguments.python)
    authority_path = _existing_file(
        arguments.reference_authority,
        role="authority",
    )
    authority_raw = _read_json(authority_path, role="authority")
    issued_at = authority_raw.get("issued_at_unix")
    if type(issued_at) is not int:
        _fail("control_containment_authority_issue_time_invalid")
    authority = validate_authority(
        authority_raw,
        expected_authority_sha256=arguments.expected_authority_sha256,
        now_unix=issued_at,
    )
    checkpoint_path = _existing_file(
        arguments.reference_checkpoint,
        role="checkpoint",
    )
    checkpoint_payload = _read_bytes(checkpoint_path, role="checkpoint")
    if _sha256_bytes(checkpoint_payload) != arguments.expected_checkpoint_sha256:
        _fail("control_containment_checkpoint_sha256_mismatch")
    checkpoint = _read_json(checkpoint_path, role="checkpoint")
    checkpoint_artifacts: list[Path] = []
    for role in ("adapter", "optimizer"):
        binding = checkpoint.get(role)
        if (
            not isinstance(binding, Mapping)
            or not isinstance(binding.get("path"), str)
            or Path(binding["path"]).name != binding["path"]
        ):
            _fail(f"control_containment_checkpoint_{role}_binding_invalid")
        path = _existing_file(
            checkpoint_path.parent / binding["path"],
            role=f"checkpoint_{role}",
        )
        if path.parent != checkpoint_path.parent:
            _fail(f"control_containment_checkpoint_{role}_escape")
        checkpoint_artifacts.append(path)
    projected_dataset = _existing_file(
        arguments.projected_dataset,
        role="projected_dataset",
    )
    model_dir = _existing_directory(arguments.model_dir, role="model")
    execution_spec = _existing_file(
        arguments.execution_spec,
        role="execution_spec",
    )
    if (
        small_model_identity(model_dir) != authority["model"]
        or execution_spec_identity(
            _read_json(execution_spec, role="execution_spec")
        )
        != authority["execution_spec"]
    ):
        _fail("control_containment_model_or_execution_drift")
    sources = source_closure(control_source_paths())

    contract_dir = _private_directory(arguments.contract_dir, role="contract")
    output_dir = _private_directory(arguments.output_dir, role="output")
    detached_dir = _private_directory(arguments.detached_dir, role="detached")
    if len({contract_dir, output_dir, detached_dir}) != 3:
        _fail("control_containment_run_roots_not_disjoint")
    runtime_root = _private_directory(output_dir / "runtime", role="runtime")
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
        _fail("control_containment_model_lane_symlink_rejected")
    lane_parent = _private_directory(
        model_lane_state.parent,
        role="model_lane_parent",
    )
    model_lane_state = lane_parent / model_lane_state.name

    forbidden = _forbidden_roots(
        projected_dataset=projected_dataset,
        model_dir=model_dir,
    )
    reads = (
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
        authority_path,
        checkpoint_path,
        *checkpoint_artifacts,
        projected_dataset,
        model_dir,
        execution_spec,
        contract_dir,
        output_dir,
        detached_dir,
    )
    writes = (output_dir, detached_dir, runtime_root)
    _assert_disjoint(reads=reads, writes=writes, forbidden=forbidden)
    profile = build_sandbox_profile(
        python=python,
        read_paths=reads,
        write_paths=writes,
        forbidden_roots=forbidden,
        model_lane_state=model_lane_state,
    )
    profile_path = contract_dir / "recurrent-sft-controls.sb"
    _write_create_or_verify(profile_path, profile.encode("utf-8"))
    command = [
        str(SANDBOX_PATH),
        "-f",
        str(profile_path),
        str(python),
        str(REPO_ROOT / "tools/train_recurrent_sft_controls.py"),
        "--reference-authority",
        str(authority_path),
        "--expected-authority-sha256",
        authority["authority_sha256"],
        "--reference-checkpoint",
        str(checkpoint_path),
        "--expected-reference-checkpoint-sha256",
        arguments.expected_checkpoint_sha256,
        "--projected-dataset",
        str(projected_dataset),
        "--model-dir",
        str(model_dir),
        "--execution-spec",
        str(execution_spec),
        "--expected-source-closure-sha256",
        sources["closure_sha256"],
        "--out-dir",
        str(output_dir),
    ]
    material = "\0".join(command)
    if any(str(root) in material for root in forbidden):
        _fail("control_containment_forbidden_path_in_command")
    environment = _sanitized_environment(
        python=python,
        runtime_root=runtime_root,
        model_lane_state=model_lane_state,
    )
    body = {
        "schema": CONTRACT_SCHEMA,
        "profile_schema": PROFILE_SCHEMA,
        "authority_sha256": authority["authority_sha256"],
        "reference_checkpoint_sha256": arguments.expected_checkpoint_sha256,
        "projected_dataset_file_sha256": _sha256_bytes(
            _read_bytes(projected_dataset, role="projected_dataset")
        ),
        "model_identity_sha256": authority["model"]["identity_sha256"],
        "execution_spec_sha256": authority["execution_spec"]["semantic_sha256"],
        "source_closure": sources,
        "profile_path": str(profile_path),
        "profile_sha256": _sha256_bytes(profile.encode("utf-8")),
        "sandbox_executable_sha256": _sha256_bytes(
            _read_bytes(SANDBOX_PATH, role="sandbox")
        ),
        "command": command,
        "command_sha256": _sha256_json(command),
        "environment": environment,
        "environment_sha256": _sha256_json(environment),
        "read_paths": [str(path) for path in reads],
        "write_paths": [str(path) for path in writes],
        "forbidden_roots": [str(path) for path in forbidden],
        "network": "kernel_denied",
        "process_fork": "kernel_denied",
        "evaluator_access": False,
        "resident_checkpoint_access": False,
        "production_write_access": False,
        "resume_contract": "none",
        "output_dir": str(output_dir),
        "detached_dir": str(detached_dir),
        "model_lane_state": str(model_lane_state),
        "timeout_s": float(arguments.timeout),
        "claims_not_supported": [
            "heldout_transfer",
            "reasoning_gain",
            "frontier_performance",
            "resident_32b_result",
            "production_promotion",
            "wow_signal",
        ],
    }
    contract = {**body, "contract_sha256": _sha256_json(body)}
    _write_create_or_verify(
        contract_dir / "containment_contract.json",
        canonical_json_bytes(contract),
    )
    return contract


def launch_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    argv = [
        "launch",
        "--run-dir",
        str(contract["detached_dir"]),
        "--name",
        "spark-059-recurrent-sft-controls",
        "--cwd",
        str(REPO_ROOT),
        "--timeout",
        str(contract["timeout_s"]),
        "--resume-contract",
        "none",
        "--containment-mode",
        "precontained-sandbox",
        "--",
        *list(contract["command"]),
    ]
    parser = run_detached_step.build_parser()
    parsed = parser.parse_args(argv)
    environment = contract.get("environment")
    if not isinstance(environment, Mapping):
        _fail("control_containment_environment_invalid")
    with _exact_environment(
        {str(key): str(value) for key, value in environment.items()}
    ):
        detached = run_detached_step._launch(parsed, parser)
    return {
        "schema": RESULT_SCHEMA,
        "status": "detached_control_training_launched",
        "contract_sha256": contract["contract_sha256"],
        "authority_sha256": contract["authority_sha256"],
        "output_dir": contract["output_dir"],
        "detached_dir": contract["detached_dir"],
        "detached": detached,
        "claims_not_supported": contract["claims_not_supported"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=sorted(_SAFE_ACTIONS))
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--reference-authority", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--projected-dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--execution-spec", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detached-dir", type=Path, required=True)
    parser.add_argument(
        "--model-lane-state",
        type=Path,
        default=Path.home() / ".aura/run/model_lane_control.json",
    )
    parser.add_argument("--timeout", type=float, default=2 * 60 * 60)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if (
        not _is_sha256(arguments.expected_authority_sha256)
        or not _is_sha256(arguments.expected_checkpoint_sha256)
        or not 1.0 <= arguments.timeout <= _MAX_TIMEOUT_S
    ):
        parser.error("authority, checkpoint, or timeout argument is invalid")
    try:
        contract = build_contract(arguments)
        result = (
            {
                "schema": RESULT_SCHEMA,
                "status": "control_containment_contract_prepared",
                "contract_sha256": contract["contract_sha256"],
                "profile_sha256": contract["profile_sha256"],
                "output_dir": contract["output_dir"],
                "detached_dir": contract["detached_dir"],
                "claims_not_supported": contract["claims_not_supported"],
            }
            if arguments.action == "prepare"
            else launch_contract(contract)
        )
    except (
        OSError,
        RecurrentSFTControlContainmentError,
        StructuredSFTResearchAuthorityError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {
                    "schema": f"{RESULT_SCHEMA}.error",
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
