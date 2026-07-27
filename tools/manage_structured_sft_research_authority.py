#!/usr/bin/env python3
"""Create or independently reverify restricted SPARK recurrent-SFT authority."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.learning.structured_sft_research_authority import (  # noqa: E402
    MAX_AUTHORITY_TTL_S,
    SAMPLER,
    SUPPORTED_SAMPLERS,
    RecurrentSFTTrainerConfig,
    StructuredSFTResearchAuthorityError,
    build_authority,
    candidate_identity,
    canonical_json_bytes,
    execution_spec_identity,
    small_model_identity,
    source_closure,
    strict_json_bytes,
    tokenization_identity,
    upstream_witness_identity,
    validate_authority,
)
from core.runtime.atomic_writer import atomic_write_bytes  # noqa: E402
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402
from tools.build_structured_sft_dataset import (  # noqa: E402
    CandidateDatasetBuildError,
    read_candidate_dataset_directory_with_attestation,
)
from tools.validate_structured_sft_tokenization import (  # noqa: E402
    TokenizerValidationError,
    validate,
)

RESULT_SCHEMA = "aura.rlc.synthetic_recurrent_sft_authority_operator.v1"
_MAX_JSON_BYTES = 256 * 1024 * 1024


class ResearchAuthorityOperatorError(RuntimeError):
    """The research authority operator could not finish safely."""


def _fail(code: str) -> Never:
    raise ResearchAuthorityOperatorError(
        str(code or "research_authority_operator_failed")
    )


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        raw = read_stable_bytes(
            path.expanduser().resolve(strict=True),
            max_bytes=_MAX_JSON_BYTES,
        )
        return strict_json_bytes(raw, role=role)
    except StructuredSFTResearchAuthorityError:
        raise
    except OSError as exc:
        raise ResearchAuthorityOperatorError(f"{role}_unreadable") from exc


def _read_pem(path: Path) -> bytes:
    try:
        return read_stable_bytes(
            path.expanduser().resolve(strict=True),
            max_bytes=64 * 1024,
        )
    except OSError as exc:
        raise ResearchAuthorityOperatorError(
            "trusted_log_key_unreadable"
        ) from exc


def _source_paths() -> dict[str, Path]:
    return {
        "authority": (
            REPO_ROOT
            / "core/learning/structured_sft_research_authority.py"
        ),
        "trainer": REPO_ROOT / "tools/train_structured_sft_research.py",
        "containment_launcher": (
            REPO_ROOT / "tools/launch_structured_sft_research.py"
        ),
        "detached_supervisor": REPO_ROOT / "tools/run_detached_step.py",
        "checkpoint_state": (
            REPO_ROOT / "core/learning/structured_sft_research_state.py"
        ),
        "structured_sft": REPO_ROOT / "core/learning/structured_sft.py",
        "retention_curriculum": (
            REPO_ROOT / "core/learning/recurrent_sft_retention.py"
        ),
        "behavior_canaries": (
            REPO_ROOT / "core/learning/recurrent_sft_behavior_canaries.py"
        ),
        "sampling": REPO_ROOT / "core/learning/recurrent_sft_sampling.py",
        "tokenization": REPO_ROOT / "tools/validate_structured_sft_tokenization.py",
        "recurrence_objective": (
            REPO_ROOT / "core/learning/recurrence_native_objective_v2.py"
        ),
        "execution_spec": (
            REPO_ROOT / "core/brain/llm/latent_cortex/execution_spec.py"
        ),
        "recurrence_adapter": (
            REPO_ROOT / "core/brain/llm/latent_cortex/recurrence_adapter.py"
        ),
        "recurrent_sft_execution": (
            REPO_ROOT / "core/learning/recurrent_sft_execution.py"
        ),
        "resume_verifier": (
            REPO_ROOT / "tools/verify_structured_sft_research_resume.py"
        ),
    }


def _trainer_config(arguments: argparse.Namespace) -> RecurrentSFTTrainerConfig:
    targets = tuple(
        target.strip()
        for target in arguments.lora_targets.split(",")
        if target.strip()
    )
    return RecurrentSFTTrainerConfig(
        max_steps=arguments.max_steps,
        sampler=arguments.sampler,
        batch_size=1,
        learning_rate=arguments.learning_rate,
        optimizer="AdamW",
        weight_decay=arguments.weight_decay,
        lora_rank=arguments.lora_rank,
        lora_scale=arguments.lora_scale,
        lora_dropout=0.0,
        lora_targets=targets,
        checkpoint_every=arguments.checkpoint_every,
        evaluate_every=arguments.evaluate_every,
        validation_examples=arguments.validation_examples,
        max_seq_length=arguments.max_seq_length,
        max_minutes=arguments.max_minutes,
        memory_fraction=arguments.memory_fraction,
        seed=arguments.seed,
    )


def _components(arguments: argparse.Namespace) -> dict[str, Any]:
    packet = _read_json(arguments.audit_packet, role="audit_packet")
    bundle = _read_json(arguments.witness_bundle, role="witness_bundle")
    trusted_log_key = _read_pem(arguments.trusted_log_key)
    candidate_artifacts, custody = (
        read_candidate_dataset_directory_with_attestation(
            arguments.candidate_dir
        )
    )
    tokenization_report = validate(
        candidate_directory=arguments.candidate_dir,
        tokenizer_directory=arguments.tokenizer_dir,
        snapshot_root=arguments.snapshot_root,
    )
    execution_raw = _read_json(
        arguments.execution_spec,
        role="execution_spec",
    )
    return {
        "upstream_witness": upstream_witness_identity(
            audit_packet=packet,
            witness_bundle=bundle,
            trusted_log_public_key_pem=trusted_log_key,
            expected_sequence=arguments.witness_sequence,
            expected_previous_statement_sha256=(
                arguments.previous_statement_sha256
            ),
            expected_previous_rekor_uuid=arguments.previous_rekor_uuid,
            minimum_active_shard_log_index=(
                arguments.minimum_active_shard_log_index
            ),
            minimum_integrated_time=arguments.minimum_integrated_time,
        ),
        "candidate": candidate_identity(candidate_artifacts, custody),
        "tokenization": tokenization_identity(tokenization_report),
        "model": small_model_identity(arguments.model_dir),
        "execution_spec": execution_spec_identity(execution_raw),
        "sources": source_closure(_source_paths()),
        "trainer_config": _trainer_config(arguments),
    }


def _write_create_or_verify(path: Path, document: dict[str, Any]) -> None:
    target = Path(os.path.abspath(os.fspath(path.expanduser())))
    if target.is_symlink():
        _fail("authority_output_symlink_rejected")
    payload = canonical_json_bytes(document)
    if target.exists():
        observed = read_stable_bytes(target, max_bytes=_MAX_JSON_BYTES)
        if observed != payload:
            _fail("authority_output_exists_with_different_bytes")
        return
    if not target.parent.is_dir() or target.parent.is_symlink():
        _fail("authority_output_parent_invalid")
    metadata = target.parent.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o022:
        _fail("authority_output_parent_not_owner_controlled")
    atomic_write_bytes(target, payload, mode=0o600)


def _authorize(arguments: argparse.Namespace) -> dict[str, Any]:
    material = _components(arguments)
    issued = (
        arguments.issued_at_unix
        if arguments.issued_at_unix is not None
        else int(time.time())
    )
    authority = build_authority(
        issued_at_unix=issued,
        expires_at_unix=issued + arguments.ttl_seconds,
        upstream_witness=material["upstream_witness"],
        candidate=material["candidate"],
        tokenization=material["tokenization"],
        model=material["model"],
        execution_spec=material["execution_spec"],
        trainer_config=material["trainer_config"],
        sources=material["sources"],
    )
    _write_create_or_verify(arguments.out, authority)
    return {
        "schema": RESULT_SCHEMA,
        "status": "authority_created_or_identical",
        "path": str(arguments.out.expanduser().resolve(strict=True)),
        "authority_sha256": authority["authority_sha256"],
        "expires_at_unix": authority["expires_at_unix"],
        "training_authority": authority["training_authority"],
        "trainer_ready": authority["trainer_ready"],
        "production_promotion_allowed": False,
    }


def _verify(arguments: argparse.Namespace) -> dict[str, Any]:
    authority = _read_json(arguments.authority, role="authority")
    validated = validate_authority(
        authority,
        expected_authority_sha256=arguments.expected_authority_sha256,
        now_unix=(
            arguments.now_unix
            if arguments.now_unix is not None
            else int(time.time())
        ),
    )
    material = _components(arguments)
    comparisons = {
        "upstream_witness": material["upstream_witness"],
        "candidate": material["candidate"],
        "tokenization": material["tokenization"],
        "model": material["model"],
        "execution_spec": material["execution_spec"],
        "sources": material["sources"],
        "trainer": material["trainer_config"].to_dict(),
    }
    if any(validated.get(role) != value for role, value in comparisons.items()):
        _fail("authority_live_component_drift")
    return {
        "schema": RESULT_SCHEMA,
        "status": "authority_and_live_components_verified",
        "authority_sha256": validated["authority_sha256"],
        "expires_at_unix": validated["expires_at_unix"],
        "training_authority": validated["training_authority"],
        "trainer_ready": validated["trainer_ready"],
        "production_promotion_allowed": False,
    }


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--audit-packet", type=Path, required=True)
    parser.add_argument("--witness-bundle", type=Path, required=True)
    parser.add_argument("--trusted-log-key", type=Path, required=True)
    parser.add_argument("--witness-sequence", type=int, required=True)
    parser.add_argument(
        "--previous-statement-sha256",
        default="0" * 64,
    )
    parser.add_argument("--previous-rekor-uuid")
    parser.add_argument("--minimum-active-shard-log-index", type=int)
    parser.add_argument("--minimum-integrated-time", type=int)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--execution-spec", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument(
        "--sampler",
        choices=SUPPORTED_SAMPLERS,
        default=SAMPLER,
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-scale", type=float, default=20.0)
    parser.add_argument(
        "--lora-targets",
        default="q_proj,v_proj,o_proj",
    )
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--evaluate-every", type=int, default=5)
    parser.add_argument("--validation-examples", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-minutes", type=float, default=180.0)
    parser.add_argument("--memory-fraction", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=2026072701)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    authorize = subparsers.add_parser("authorize")
    _add_shared(authorize)
    authorize.add_argument("--issued-at-unix", type=int)
    authorize.add_argument(
        "--ttl-seconds",
        type=int,
        default=MAX_AUTHORITY_TTL_S,
    )
    authorize.add_argument("--out", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    _add_shared(verify)
    verify.add_argument("--authority", type=Path, required=True)
    verify.add_argument("--expected-authority-sha256", required=True)
    verify.add_argument("--now-unix", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if (
        arguments.witness_sequence < 1
        or len(arguments.previous_statement_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in arguments.previous_statement_sha256
        )
        or (
            arguments.command == "authorize"
            and not 1 <= arguments.ttl_seconds <= MAX_AUTHORITY_TTL_S
        )
        or (
            arguments.command == "verify"
            and (
                len(arguments.expected_authority_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in arguments.expected_authority_sha256
                )
            )
        )
    ):
        parser.error("authority pin or time bound is invalid")
    try:
        result = (
            _authorize(arguments)
            if arguments.command == "authorize"
            else _verify(arguments)
        )
    except (
        CandidateDatasetBuildError,
        OSError,
        ResearchAuthorityOperatorError,
        StructuredSFTResearchAuthorityError,
        TokenizerValidationError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {
                    "schema": "aura.rlc.synthetic_recurrent_sft_authority_error.v1",
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
