#!/usr/bin/env python3
"""Independently verify a completed recurrent-SFT falsification report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.recurrent_sft_behavior_canaries import (  # noqa: E402
    RecurrentSFTBehaviorCanaryError,
    build_generated_behavior_canaries,
    build_generated_behavior_generation_contract,
    generated_behavior_verdict,
)
from core.learning.recurrent_sft_evaluation import (  # noqa: E402
    EVALUATION_SCHEMA,
    EVALUATION_SOURCE_ROLES,
    RecurrentSFTEvaluationError,
    regression_canary_verdict,
    sha256_bytes,
    strict_json_bytes,
    validate_control_report,
)
from core.learning.recurrent_sft_falsification import (  # noqa: E402
    ALL_ARMS,
    BASE_ARM,
    CONTROL_ARMS,
    TRAINED_ARM,
    build_falsification_verdict,
    sha256_json,
)
from core.learning.recurrent_sft_sampling import (  # noqa: E402
    FAMILY_BALANCED_SAMPLER,
)
from core.learning.structured_sft import (  # noqa: E402
    STRUCTURED_SFT_CANDIDATE_FILES,
    STRUCTURED_SFT_EVALUATOR_FILES,
    validate_structured_sft_custody_pair,
)
from core.learning.structured_sft_research_authority import (  # noqa: E402
    StructuredSFTResearchAuthorityError,
    canonical_json_bytes,
    small_model_identity,
    validate_authority,
)
from core.learning.structured_sft_research_state import (  # noqa: E402
    StructuredSFTResearchStateError,
    validate_checkpoint_state,
)
from core.runtime.atomic_writer import atomic_write_bytes  # noqa: E402
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402

REPORT_SCHEMA = f"{EVALUATION_SCHEMA}.report"
VERIFICATION_SCHEMA = f"{EVALUATION_SCHEMA}.independent_verification"
_MAX_FILE_BYTES = 512 * 1024 * 1024


class RecurrentSFTFalsificationVerificationError(RuntimeError):
    """The reported falsification evidence did not independently verify."""


def _fail(code: str) -> Never:
    raise RecurrentSFTFalsificationVerificationError(
        str(code or "recurrent_sft_falsification_verification_failed")
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
            max_bytes=_MAX_FILE_BYTES,
        )
    except OSError as exc:
        raise RecurrentSFTFalsificationVerificationError(
            f"recurrent_sft_verification_{role}_unreadable"
        ) from exc


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    return strict_json_bytes(_read_bytes(path, role=role), role=role)


def _verify_binding(binding: Any, *, role: str) -> dict[str, Any]:
    if (
        not isinstance(binding, Mapping)
        or set(binding) != {"path", "sha256", "size_bytes"}
        or not isinstance(binding.get("path"), str)
        or not _is_sha256(binding.get("sha256"))
        or type(binding.get("size_bytes")) is not int
        or binding["size_bytes"] < 1
    ):
        _fail(f"recurrent_sft_verification_{role}_binding_invalid")
    lexical = Path(binding["path"]).expanduser()
    if lexical.is_symlink():
        _fail(f"recurrent_sft_verification_{role}_symlink_rejected")
    path = lexical.resolve(strict=True)
    payload = _read_bytes(path, role=role)
    if len(payload) != binding["size_bytes"] or sha256_bytes(payload) != binding["sha256"]:
        _fail(f"recurrent_sft_verification_{role}_binding_mismatch")
    return {"path": path, "payload": payload}


def _verify_report_hash(report: Mapping[str, Any]) -> None:
    body = dict(report)
    observed = body.pop("report_sha256", None)
    if report.get("schema") != REPORT_SCHEMA or observed != sha256_json(body):
        _fail("recurrent_sft_verification_report_commitment_invalid")


def _verify_contract(contract: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    body = dict(contract)
    observed = body.pop("contract_sha256", None)
    if (
        observed != sha256_json(body)
        or contract.get("contract_sha256") != report.get("containment_contract_sha256")
        or contract.get("source_closure") != report.get("source_closure")
        or contract.get("authority_sha256") != report.get("authority_sha256")
        or contract.get("model_identity_sha256") != report.get("model_identity_sha256")
        or contract.get("execution_spec_sha256") != report.get("execution_spec_sha256")
        or contract.get("custody_binding_sha256") != report.get("custody_binding_sha256")
        or contract.get("custody_bindings") != report.get("custody")
        or contract.get("network") != "kernel_denied"
        or contract.get("process_fork") != "kernel_denied"
        or contract.get("evaluator_access") is not True
        or contract.get("training_write_access") is not False
        or contract.get("resident_checkpoint_access") is not False
        or contract.get("production_write_access") is not False
        or contract.get("resume_contract") != "none"
        or sha256_json(contract.get("command")) != contract.get("command_sha256")
        or sha256_json(contract.get("environment")) != contract.get("environment_sha256")
    ):
        _fail("recurrent_sft_verification_contract_invalid")
    profile_path = contract.get("profile_path")
    sandbox_sha256 = contract.get("sandbox_executable_sha256")
    command = contract.get("command")
    if (
        not isinstance(profile_path, str)
        or not _is_sha256(contract.get("profile_sha256"))
        or not _is_sha256(sandbox_sha256)
        or not isinstance(command, list)
        or len(command) < 5
        or command[0] != "/usr/bin/sandbox-exec"
        or command[1:3] != ["-f", profile_path]
        or sha256_bytes(_read_bytes(Path(profile_path), role="sandbox_profile"))
        != contract["profile_sha256"]
        or sha256_bytes(_read_bytes(Path(command[0]), role="sandbox_executable")) != sandbox_sha256
    ):
        _fail("recurrent_sft_verification_contract_execution_invalid")


def _verify_source_closure(source_closure: Any) -> None:
    if (
        not isinstance(source_closure, Mapping)
        or not isinstance(source_closure.get("files"), list)
        or not source_closure["files"]
    ):
        _fail("recurrent_sft_verification_source_closure_invalid")
    body = dict(source_closure)
    observed = body.pop("closure_sha256", None)
    if observed != sha256_json(body):
        _fail("recurrent_sft_verification_source_closure_invalid")
    roles: set[str] = set()
    ordered_roles: list[str] = []
    for record in source_closure["files"]:
        if (
            not isinstance(record, Mapping)
            or set(record) != {"role", "path", "sha256", "size_bytes"}
            or not isinstance(record.get("role"), str)
            or record["role"] in roles
        ):
            _fail("recurrent_sft_verification_source_record_invalid")
        roles.add(record["role"])
        ordered_roles.append(record["role"])
        _verify_binding(
            {
                "path": record["path"],
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
            },
            role=f"source_{record['role']}",
        )
    if roles != set(EVALUATION_SOURCE_ROLES) or ordered_roles != sorted(EVALUATION_SOURCE_ROLES):
        _fail("recurrent_sft_verification_source_roles_invalid")


def _verify_custody(report: Mapping[str, Any]) -> dict[str, Any]:
    custody = report.get("custody")
    custody_execution = report.get("custody_execution")
    if not isinstance(custody, Mapping):
        _fail("recurrent_sft_verification_custody_invalid")
    candidate_bindings = custody.get("candidate")
    evaluator_bindings = custody.get("evaluator")
    reported_custody = custody.get("custody")
    if (
        not isinstance(candidate_bindings, Mapping)
        or set(candidate_bindings) != set(STRUCTURED_SFT_CANDIDATE_FILES)
        or not isinstance(evaluator_bindings, Mapping)
        or set(evaluator_bindings) != set(STRUCTURED_SFT_EVALUATOR_FILES)
        or not isinstance(reported_custody, Mapping)
        or custody_execution
        != {
            "launcher_semantic_replay_bound": True,
            "evaluator_exact_byte_rehash": True,
            "evaluator_projection_validation": True,
            "evaluator_semantic_replay": False,
            "independent_verifier_semantic_replay_required": True,
            "reason": "kernel_process_fork_denied",
        }
    ):
        _fail("recurrent_sft_verification_custody_invalid")
    candidate = {
        name: _verify_binding(
            candidate_bindings[name],
            role=f"candidate_{name}",
        )["payload"]
        for name in STRUCTURED_SFT_CANDIDATE_FILES
    }
    evaluator = {
        name: _verify_binding(
            evaluator_bindings[name],
            role=f"evaluator_{name}",
        )["payload"]
        for name in STRUCTURED_SFT_EVALUATOR_FILES
    }
    replayed = validate_structured_sft_custody_pair(candidate, evaluator)
    if replayed != reported_custody or sha256_json(custody) != report.get("custody_binding_sha256"):
        _fail("recurrent_sft_verification_custody_replay_mismatch")
    return {"candidate": candidate, "evaluator": evaluator, "report": replayed}


def _verify_no_holdout_content_leak(
    report_payload: bytes,
    evaluator_artifacts: Mapping[str, bytes],
) -> None:
    holdout = strict_json_bytes(
        evaluator_artifacts["holdout.private.json"],
        role="verification_holdout",
    )
    seed = holdout.get("holdout_seed_hex")
    if isinstance(seed, str) and seed.encode("ascii") in report_payload:
        _fail("recurrent_sft_verification_holdout_seed_leaked")
    examples = holdout.get("examples")
    if not isinstance(examples, list):
        _fail("recurrent_sft_verification_holdout_invalid")
    for example in examples:
        if not isinstance(example, Mapping):
            _fail("recurrent_sft_verification_holdout_invalid")
        messages = example.get("messages")
        if not isinstance(messages, list):
            _fail("recurrent_sft_verification_holdout_invalid")
        for message in messages:
            content = message.get("content") if isinstance(message, Mapping) else None
            if (
                isinstance(content, str)
                and len(content) >= 32
                and content.encode("utf-8") in report_payload
            ):
                _fail("recurrent_sft_verification_holdout_content_leaked")
    forbidden_keys = {b'"messages":', b'"prompt_tokens":', b'"answer_tokens":'}
    if any(key in report_payload for key in forbidden_keys):
        _fail("recurrent_sft_verification_holdout_projection_leaked")


def _tensor_fingerprint(path: Path) -> str:
    import mlx.core as mx
    import numpy as np

    tensors = mx.load(str(path))
    if not isinstance(tensors, Mapping) or not tensors:
        _fail("recurrent_sft_verification_adapter_tensors_invalid")
    digest = hashlib.sha256()
    for name in sorted(tensors):
        array = np.asarray(tensors[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json_bytes(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _verify_adapters(report: Mapping[str, Any]) -> dict[str, str]:
    trained = report.get("trained_candidate")
    controls = report.get("controls")
    fingerprints = report.get("adapter_fingerprints")
    if (
        not isinstance(trained, Mapping)
        or set(trained)
        != {
            "checkpoint",
            "adapter",
            "optimizer_updates",
            "step",
            "trainer_config_sha256",
        }
        or not isinstance(controls, Mapping)
        or not isinstance(controls.get("report"), Mapping)
        or not isinstance(controls.get("adapters"), Mapping)
        or set(controls["adapters"]) != set(CONTROL_ARMS)
        or not isinstance(fingerprints, Mapping)
        or set(fingerprints) != set(ALL_ARMS)
        or fingerprints.get(BASE_ARM) is not None
    ):
        _fail("recurrent_sft_verification_adapter_evidence_invalid")
    _verify_binding(trained["checkpoint"], role="trained_checkpoint")
    trained_adapter = _verify_binding(
        trained["adapter"],
        role="trained_adapter",
    )["path"]
    _verify_binding(controls["report"], role="control_report")
    paths = {
        TRAINED_ARM: trained_adapter,
        **{
            arm: _verify_binding(
                controls["adapters"][arm],
                role=f"control_adapter_{arm}",
            )["path"]
            for arm in CONTROL_ARMS
        },
    }
    observed = {arm: _tensor_fingerprint(path) for arm, path in paths.items()}
    if any(observed[arm] != fingerprints.get(arm) for arm in observed):
        _fail("recurrent_sft_verification_adapter_fingerprint_mismatch")
    return observed


def _verify_equal_work(
    report: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
) -> None:
    trained = report["trained_candidate"]
    controls = report["controls"]
    checkpoint_payload = _verify_binding(
        trained["checkpoint"],
        role="equal_work_checkpoint",
    )["payload"]
    checkpoint = strict_json_bytes(
        checkpoint_payload,
        role="equal_work_checkpoint",
    )
    if authority["trainer"].get("sampler") == FAMILY_BALANCED_SAMPLER:
        validate_checkpoint_state(
            {
                key: value
                for key, value in checkpoint.items()
                if key
                not in {
                    "adapter",
                    "optimizer",
                    "checkpoint_id",
                    "created_unix",
                }
            }
        )
    if (
        checkpoint.get("optimizer_updates") != trained["optimizer_updates"]
        or checkpoint.get("step") != trained["step"]
        or checkpoint.get("trainer_config_sha256") != trained["trainer_config_sha256"]
        or trained["step"] != trained["optimizer_updates"]
    ):
        _fail("recurrent_sft_verification_reference_workload_mismatch")
    control_payload = _verify_binding(
        controls["report"],
        role="equal_work_control_report",
    )["payload"]
    control_report = strict_json_bytes(
        control_payload,
        role="equal_work_control_report",
    )
    trainer_config_sha256 = sha256_json(authority["trainer"])
    bindings = validate_control_report(
        control_report,
        report_file_sha256=sha256_bytes(control_payload),
        expected_report_file_sha256=controls["report"]["sha256"],
        expected_authority_sha256=authority["authority_sha256"],
        expected_reference_checkpoint_sha256=trained["checkpoint"]["sha256"],
        expected_model_identity_sha256=authority["model"]["identity_sha256"],
        expected_execution_spec_sha256=authority["execution_spec"]["semantic_sha256"],
        expected_reference_optimizer_updates=trained["optimizer_updates"],
        expected_trainer_config_sha256=trainer_config_sha256,
        expected_reference_initial_adapter_sha256=(
            checkpoint.get("initial_adapter_sha256")
            if authority["trainer"].get("sampler")
            == FAMILY_BALANCED_SAMPLER
            else None
        ),
    )
    for arm in CONTROL_ARMS:
        binding = bindings[arm]
        reported = controls["adapters"][arm]
        if (
            Path(reported["path"]).name != binding["filename"]
            or reported["sha256"] != binding["sha256"]
            or reported["size_bytes"] != binding["size_bytes"]
        ):
            _fail("recurrent_sft_verification_control_binding_mismatch")


def _verify_decisions(report: Mapping[str, Any]) -> dict[str, Any]:
    observations = report.get("observations")
    canary_observations = report.get("regression_likelihood_canary_observations")
    behavior_observations = report.get("generated_behavior_canary_observations")
    behavior_contract = report.get("generated_behavior_generation_contract")
    adapter_fingerprints = report.get("adapter_fingerprints")
    if (
        not isinstance(observations, Mapping)
        or set(observations) != set(ALL_ARMS)
        or not isinstance(canary_observations, Mapping)
        or set(canary_observations) != {BASE_ARM, TRAINED_ARM}
        or not isinstance(behavior_observations, Mapping)
        or set(behavior_observations) != {BASE_ARM, TRAINED_ARM}
        or not isinstance(behavior_contract, Mapping)
        or not isinstance(adapter_fingerprints, Mapping)
    ):
        _fail("recurrent_sft_verification_observations_invalid")
    expected_behavior_contract = build_generated_behavior_generation_contract(
        execution_spec_sha256=str(report.get("execution_spec_sha256") or ""),
    )
    if (
        dict(behavior_contract) != expected_behavior_contract
        or report.get("generated_behavior_generation_contract_sha256")
        != expected_behavior_contract["contract_sha256"]
        or report.get("generated_behavior_canary_count") != len(build_generated_behavior_canaries())
    ):
        _fail("recurrent_sft_verification_behavior_contract_invalid")
    falsification = build_falsification_verdict(observations)
    canaries = regression_canary_verdict(
        canary_observations[BASE_ARM],
        canary_observations[TRAINED_ARM],
    )
    behavior_canaries = generated_behavior_verdict(
        behavior_observations[BASE_ARM],
        behavior_observations[TRAINED_ARM],
        expected_generation_contract_sha256=(expected_behavior_contract["contract_sha256"]),
        expected_trained_adapter_fingerprint=adapter_fingerprints.get(TRAINED_ARM),
    )
    lexical = report.get("ordinary_lexical_hashes")
    if (
        falsification != report.get("falsification")
        or canaries != report.get("regression_likelihood_canary_verdict")
        or behavior_canaries != report.get("generated_behavior_canary_verdict")
        or not isinstance(lexical, Mapping)
        or set(lexical) != set(ALL_ARMS)
        or any(not _is_sha256(value) for value in lexical.values())
    ):
        _fail("recurrent_sft_verification_decision_replay_mismatch")
    lexical_invariance = len(set(lexical.values())) == 1
    all_passed = (
        falsification["heldout_transfer_proven"]
        and canaries["passed"]
        and behavior_canaries["passed"]
        and lexical_invariance
    )
    expected_status = (
        "small_checkpoint_transfer_with_all_regression_gates_passed"
        if all_passed
        else "small_checkpoint_transfer_not_proven"
    )
    if (
        report.get("ordinary_lexical_invariance_proven") is not lexical_invariance
        or report.get("all_small_checkpoint_gates_passed") is not all_passed
        or report.get("status") != expected_status
        or report.get("production_effect") is not False
        or report.get("promotion_allowed") is not False
        or report.get("base_weights_unchanged") is not True
        or report.get("generated_behavior_regression_tested") is not True
        or report.get("claims_not_supported")
        != [
            "broad_reasoning_gain",
            "frontier_performance",
            "resident_32b_result",
            "production_promotion",
            "wow_signal",
        ]
    ):
        _fail("recurrent_sft_verification_final_decision_invalid")
    return {
        "falsification": falsification,
        "canaries": canaries,
        "generated_behavior_canaries": behavior_canaries,
        "generated_behavior_generation_contract_sha256": (
            expected_behavior_contract["contract_sha256"]
        ),
        "lexical_invariance": lexical_invariance,
        "all_small_checkpoint_gates_passed": all_passed,
    }


def _verify_receipt(
    receipt: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> None:
    duration = receipt.get("duration_s")
    if (
        receipt.get("returncode") != 0
        or receipt.get("timed_out") is not False
        or receipt.get("process_group_empty") is not True
        or isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or duration <= 0.0
        or receipt.get("status") != "passed"
        or receipt.get("restart_count") != 0
        or receipt.get("containment_verified") is not True
        or receipt.get("command") != contract.get("command")
        or receipt.get("executed_command") != contract.get("command")
        or receipt.get("command_sha256") != contract.get("command_sha256")
        or receipt.get("lineage_empty") is not True
    ):
        _fail("recurrent_sft_verification_detached_receipt_invalid")


def _run(arguments: argparse.Namespace) -> int:
    if arguments.report.expanduser().is_symlink():
        _fail("recurrent_sft_verification_report_symlink_rejected")
    report_path = arguments.report.expanduser().resolve(strict=True)
    report_payload = _read_bytes(report_path, role="report")
    if (
        not _is_sha256(arguments.expected_report_sha256)
        or sha256_bytes(report_payload) != arguments.expected_report_sha256
    ):
        _fail("recurrent_sft_verification_report_file_sha256_mismatch")
    report = strict_json_bytes(report_payload, role="report")
    _verify_report_hash(report)
    if arguments.contract.expanduser().is_symlink():
        _fail("recurrent_sft_verification_contract_symlink_rejected")
    contract = _read_json(arguments.contract, role="contract")
    _verify_contract(contract, report)
    _verify_source_closure(report.get("source_closure"))
    custody = _verify_custody(report)
    _verify_no_holdout_content_leak(report_payload, custody["evaluator"])
    adapter_fingerprints = _verify_adapters(report)
    decisions = _verify_decisions(report)
    receipt = _read_json(arguments.detached_receipt, role="detached_receipt")
    _verify_receipt(receipt, contract=contract)
    kernel_probe = _read_json(arguments.kernel_probe, role="kernel_probe")
    if not kernel_probe or any(value is not True for value in kernel_probe.values()):
        _fail("recurrent_sft_verification_kernel_probe_invalid")

    authority_path = Path(
        contract["command"][contract["command"].index("--reference-authority") + 1]
    )
    authority_raw = _read_json(authority_path, role="authority")
    issued_at = authority_raw.get("issued_at_unix")
    if type(issued_at) is not int:
        _fail("recurrent_sft_verification_authority_invalid")
    authority = validate_authority(
        authority_raw,
        expected_authority_sha256=report["authority_sha256"],
        now_unix=issued_at,
    )
    authority_candidate = authority.get("candidate")
    custody_report = custody["report"]
    reported_candidate_bindings = report["custody"]["candidate"]
    observed_candidate_files = [
        {
            "name": name,
            "sha256": reported_candidate_bindings[name]["sha256"],
            "size_bytes": reported_candidate_bindings[name]["size_bytes"],
        }
        for name in STRUCTURED_SFT_CANDIDATE_FILES
    ]
    if (
        not isinstance(authority_candidate, Mapping)
        or observed_candidate_files != authority_candidate.get("files")
        or custody_report.get("candidate_package_sha256")
        != authority_candidate.get("candidate_package_sha256")
        or custody_report.get("evaluator_package_sha256")
        != authority_candidate.get("evaluator_package_sha256")
        or custody_report.get("custody_root_sha256")
        != authority_candidate.get("custody_root_sha256")
    ):
        _fail("recurrent_sft_verification_authority_custody_mismatch")
    model_path = Path(contract["command"][contract["command"].index("--model-dir") + 1])
    if (
        small_model_identity(model_path) != authority["model"]
        or authority["model"]["identity_sha256"] != report["model_identity_sha256"]
    ):
        _fail("recurrent_sft_verification_model_identity_mismatch")
    _verify_equal_work(report, authority=authority)

    body = {
        "schema": VERIFICATION_SCHEMA,
        "status": "independently_verified",
        "report": {
            "path": str(report_path),
            "file_sha256": arguments.expected_report_sha256,
            "report_sha256": report["report_sha256"],
        },
        "contract_sha256": contract["contract_sha256"],
        "custody_report_sha256": custody["report"]["custody_report_sha256"],
        "adapter_fingerprints": adapter_fingerprints,
        "decision_replay": decisions,
        "kernel_probe": kernel_probe,
        "detached_receipt": {
            "returncode": receipt["returncode"],
            "timed_out": receipt["timed_out"],
            "duration_s": receipt["duration_s"],
            "process_group_empty": receipt["process_group_empty"],
        },
        "holdout_content_absent_from_report": True,
        "source_closure_rehashed": True,
        "base_model_identity_rehashed": True,
        "production_effect": False,
        "promotion_allowed": False,
        "claims_not_supported": [
            "broad_reasoning_gain",
            "frontier_performance",
            "resident_32b_result",
            "production_promotion",
            "wow_signal",
        ],
    }
    verified = {**body, "verification_sha256": sha256_json(body)}
    output = arguments.output.expanduser()
    if output.exists() or output.is_symlink():
        _fail("recurrent_sft_verification_output_exists")
    atomic_write_bytes(output, canonical_json_bytes(verified), mode=0o600)
    print(json.dumps(verified, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--detached-receipt", type=Path, required=True)
    parser.add_argument("--kernel-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _run(_parser().parse_args(argv))
    except (
        ImportError,
        OSError,
        RecurrentSFTBehaviorCanaryError,
        RecurrentSFTEvaluationError,
        RecurrentSFTFalsificationVerificationError,
        StructuredSFTResearchAuthorityError,
        StructuredSFTResearchStateError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {
                    "schema": f"{VERIFICATION_SCHEMA}.error",
                    "ok": False,
                    "reason": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
