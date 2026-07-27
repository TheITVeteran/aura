#!/usr/bin/env python3
"""Prepare or launch the contained recurrent-SFT holdout evaluator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.recurrent_sft_evaluation import (  # noqa: E402
    RecurrentSFTEvaluationError,
)
from core.learning.structured_sft import (  # noqa: E402
    STRUCTURED_SFT_CANDIDATE_FILES,
    STRUCTURED_SFT_EVALUATOR_FILES,
)
from core.learning.structured_sft_research_authority import (  # noqa: E402
    StructuredSFTResearchAuthorityError,
    canonical_json_bytes,
    execution_spec_identity,
    small_model_identity,
    validate_authority,
)
from tools import run_detached_step  # noqa: E402
from tools.evaluate_recurrent_sft_falsification import (  # noqa: E402
    RecurrentSFTFalsificationEvaluationError,
    _control_adapters,
    _reference_adapter,
    evaluation_source_closure,
)
from tools.launch_recurrent_sft_controls import (  # noqa: E402
    SANDBOX_PATH,
    RecurrentSFTControlContainmentError,
    _assert_disjoint,
    _exact_environment,
    _existing_directory,
    _existing_file,
    _is_sha256,
    _private_directory,
    _python_launcher,
    _read_bytes,
    _read_json,
    _sanitized_environment,
    _sha256_bytes,
    _sha256_json,
    _write_create_or_verify,
    build_sandbox_profile,
)

CONTRACT_SCHEMA = "aura.rlc.synthetic_recurrent_sft_evaluator_containment.v1"
RESULT_SCHEMA = "aura.rlc.synthetic_recurrent_sft_evaluator_operator.v1"
PROFILE_SCHEMA = "aura.rlc.synthetic_recurrent_sft_evaluator_sandbox.v1"
_SAFE_ACTIONS = frozenset({"prepare", "launch"})
_MAX_TIMEOUT_S = 4 * 60 * 60


class RecurrentSFTEvaluatorContainmentError(RuntimeError):
    """The evaluator could not be frozen behind the required kernel policy."""


def _fail(code: str) -> Never:
    raise RecurrentSFTEvaluatorContainmentError(
        str(code or "recurrent_sft_evaluator_containment_failed")
    )


def _forbidden_roots(*, model_dir: Path) -> tuple[Path, ...]:
    home = Path.home().resolve(strict=True)
    likely = {
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


def _artifact_files(root: Path, names: Sequence[str], *, role: str) -> tuple[Path, ...]:
    directory = _existing_directory(root, role=role)
    return tuple(
        _existing_file(directory / name, role=f"{role}_{name}")
        for name in names
    )


def build_contract(arguments: argparse.Namespace) -> dict[str, Any]:
    if sys.platform != "darwin" or not SANDBOX_PATH.is_file():
        _fail("evaluator_containment_macos_sandbox_required")
    python = _python_launcher(arguments.python)
    authority_path = _existing_file(
        arguments.reference_authority,
        role="evaluator_authority",
    )
    authority_raw = _read_json(authority_path, role="evaluator_authority")
    issued_at = authority_raw.get("issued_at_unix")
    if type(issued_at) is not int:
        _fail("evaluator_containment_authority_issue_time_invalid")
    authority = validate_authority(
        authority_raw,
        expected_authority_sha256=arguments.expected_authority_sha256,
        now_unix=issued_at,
    )
    checkpoint_path = _existing_file(
        arguments.reference_checkpoint,
        role="evaluator_checkpoint",
    )
    trained_adapter, _trained_binding = _reference_adapter(
        checkpoint_path,
        expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
        authority=authority,
    )
    control_report = _existing_file(
        arguments.control_report,
        role="evaluator_control_report",
    )
    control_paths, _control_bindings = _control_adapters(
        control_report,
        expected_report_sha256=arguments.expected_control_report_sha256,
        authority=authority,
        expected_reference_checkpoint_sha256=arguments.expected_checkpoint_sha256,
    )
    candidate_dir = _existing_directory(
        arguments.candidate_dir,
        role="evaluator_candidate",
    )
    evaluator_dir = _existing_directory(
        arguments.evaluator_dir,
        role="evaluator_private",
    )
    candidate_files = _artifact_files(
        candidate_dir,
        STRUCTURED_SFT_CANDIDATE_FILES,
        role="evaluator_candidate",
    )
    evaluator_files = _artifact_files(
        evaluator_dir,
        STRUCTURED_SFT_EVALUATOR_FILES,
        role="evaluator_private",
    )
    model_dir = _existing_directory(arguments.model_dir, role="evaluator_model")
    execution_spec = _existing_file(
        arguments.execution_spec,
        role="evaluator_execution_spec",
    )
    if (
        small_model_identity(model_dir) != authority["model"]
        or execution_spec_identity(
            _read_json(execution_spec, role="evaluator_execution_spec")
        )
        != authority["execution_spec"]
    ):
        _fail("evaluator_containment_model_or_execution_drift")
    sources = evaluation_source_closure()

    contract_dir = _private_directory(
        arguments.contract_dir,
        role="evaluator_contract",
    )
    output_dir = _private_directory(
        arguments.output_dir,
        role="evaluator_output",
    )
    detached_dir = _private_directory(
        arguments.detached_dir,
        role="evaluator_detached",
    )
    if len({contract_dir, output_dir, detached_dir}) != 3:
        _fail("evaluator_containment_run_roots_not_disjoint")
    runtime_root = _private_directory(
        output_dir / "runtime",
        role="evaluator_runtime",
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
        _private_directory(
            runtime_root / relative,
            role="evaluator_runtime_child",
        )
    model_lane_state = Path(
        os.path.abspath(os.fspath(arguments.model_lane_state.expanduser()))
    )
    if model_lane_state.is_symlink():
        _fail("evaluator_containment_model_lane_symlink_rejected")
    lane_parent = _private_directory(
        model_lane_state.parent,
        role="evaluator_model_lane_parent",
    )
    model_lane_state = lane_parent / model_lane_state.name

    forbidden = _forbidden_roots(model_dir=model_dir)
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
        trained_adapter,
        control_report,
        *tuple(control_paths[arm] for arm in sorted(control_paths)),
        *candidate_files,
        *evaluator_files,
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
    profile_path = contract_dir / "recurrent-sft-evaluator.sb"
    _write_create_or_verify(profile_path, profile.encode("utf-8"))
    command = [
        str(SANDBOX_PATH),
        "-f",
        str(profile_path),
        str(python),
        str(REPO_ROOT / "tools/evaluate_recurrent_sft_falsification.py"),
        "--reference-authority",
        str(authority_path),
        "--expected-authority-sha256",
        authority["authority_sha256"],
        "--reference-checkpoint",
        str(checkpoint_path),
        "--expected-reference-checkpoint-sha256",
        arguments.expected_checkpoint_sha256,
        "--control-report",
        str(control_report),
        "--expected-control-report-sha256",
        arguments.expected_control_report_sha256,
        "--candidate-dir",
        str(candidate_dir),
        "--evaluator-dir",
        str(evaluator_dir),
        "--model-dir",
        str(model_dir),
        "--execution-spec",
        str(execution_spec),
        "--expected-source-closure-sha256",
        sources["closure_sha256"],
        "--out-dir",
        str(output_dir),
    ]
    if any(str(root) in "\0".join(command) for root in forbidden):
        _fail("evaluator_containment_forbidden_path_in_command")
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
        "control_report_file_sha256": arguments.expected_control_report_sha256,
        "candidate_files": {
            path.name: _sha256_bytes(_read_bytes(path, role="candidate_binding"))
            for path in candidate_files
        },
        "evaluator_files": {
            path.name: _sha256_bytes(_read_bytes(path, role="evaluator_binding"))
            for path in evaluator_files
        },
        "model_identity_sha256": authority["model"]["identity_sha256"],
        "execution_spec_sha256": authority["execution_spec"]["semantic_sha256"],
        "source_closure": sources,
        "profile_path": str(profile_path),
        "profile_sha256": _sha256_bytes(profile.encode("utf-8")),
        "sandbox_executable_sha256": _sha256_bytes(
            _read_bytes(SANDBOX_PATH, role="evaluator_sandbox")
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
        "evaluator_access": True,
        "training_write_access": False,
        "resident_checkpoint_access": False,
        "production_write_access": False,
        "resume_contract": "none",
        "output_dir": str(output_dir),
        "detached_dir": str(detached_dir),
        "model_lane_state": str(model_lane_state),
        "timeout_s": float(arguments.timeout),
        "claims_not_supported": [
            "broad_reasoning_gain",
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
        "spark-059-recurrent-sft-evaluator",
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
        _fail("evaluator_containment_environment_invalid")
    with _exact_environment(
        {str(key): str(value) for key, value in environment.items()}
    ):
        detached = run_detached_step._launch(parsed, parser)
    return {
        "schema": RESULT_SCHEMA,
        "status": "detached_falsification_evaluator_launched",
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
    parser.add_argument("--control-report", type=Path, required=True)
    parser.add_argument("--expected-control-report-sha256", required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--evaluator-dir", type=Path, required=True)
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
        or not _is_sha256(arguments.expected_control_report_sha256)
        or not 1.0 <= arguments.timeout <= _MAX_TIMEOUT_S
    ):
        parser.error("authority, checkpoint, report, or timeout is invalid")
    try:
        contract = build_contract(arguments)
        result = (
            {
                "schema": RESULT_SCHEMA,
                "status": "falsification_evaluator_contract_prepared",
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
        RecurrentSFTEvaluationError,
        RecurrentSFTFalsificationEvaluationError,
        RecurrentSFTEvaluatorContainmentError,
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
