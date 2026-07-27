#!/usr/bin/env python3
"""Prepare, assemble, and verify externally signed structured-SFT admission.

This tool accepts public policy material and detached attestations only. It has
no private-key, key-generation, training, or candidate-renaming operation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.campaign_trust import (  # noqa: E402
    CampaignTrustError,
    validate_campaign_trust_policy,
)
from core.learning.structured_sft import StructuredSFTError  # noqa: E402
from core.learning.structured_sft_admission import (  # noqa: E402
    StructuredSFTAdmissionError,
    build_structured_sft_admission_bundle,
    structured_sft_admission_payloads,
    structured_sft_admission_protocol,
    validate_structured_sft_admission_bundle,
)
from core.runtime.file_read_gateway import (  # noqa: E402
    StableFileReadError,
    read_stable_bytes,
)
from tools.build_structured_sft_dataset import (  # noqa: E402
    CandidateDatasetBuildError,
    read_candidate_dataset_directory,
    read_evaluator_dataset_directory,
)
from tools.manage_campaign_trust import (  # noqa: E402
    CampaignTrustToolError,
    _atomic_create_or_verify,
)

_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_KEY_BYTES = 64 * 1024
_PAYLOAD_NAMES = (
    "package_declaration",
    "contamination_audit",
    "evidence_audit",
    "trainer_binding",
)


class StructuredSFTAdmissionToolError(RuntimeError):
    """Stable operator-facing admission workflow error."""


def _lexical_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    payload = read_stable_bytes(
        _lexical_path(path),
        max_bytes=_MAX_JSON_BYTES,
    )
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise StructuredSFTAdmissionToolError(f"{role}_json_invalid") from exc
    if not isinstance(value, dict):
        raise StructuredSFTAdmissionToolError(f"{role}_object_required")
    return value


def _read_root(path: Path) -> bytes:
    return read_stable_bytes(
        _lexical_path(path),
        max_bytes=_MAX_KEY_BYTES,
    )


def _emit(document: dict[str, Any], output: Path | None) -> None:
    if output is not None:
        _atomic_create_or_verify(output, document)
    sys.stdout.buffer.write(canonical_json_bytes(document) + b"\n")


def _observed_at(value: int | None) -> int:
    observed = int(time.time()) if value is None else value
    if isinstance(observed, bool) or not isinstance(observed, int) or observed <= 0:
        raise StructuredSFTAdmissionToolError("observed_at_invalid")
    return observed


def _common_documents(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "candidate_artifacts": read_candidate_dataset_directory(args.candidate_dir),
        "evaluator_artifacts": read_evaluator_dataset_directory(args.evaluator_dir),
        "tokenizer_validation": _read_json(
            args.tokenizer_validation,
            role="tokenizer_validation",
        ),
        "privacy_report": _read_json(
            args.privacy_report,
            role="privacy_report",
        ),
        "contamination_report": _read_json(
            args.contamination_report,
            role="contamination_report",
        ),
        "evidence_report": _read_json(
            args.evidence_report,
            role="evidence_report",
        ),
        "trainer_binding": _read_json(
            args.trainer_binding,
            role="trainer_binding",
        ),
    }


def _add_evidence_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--evaluator-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-validation", type=Path, required=True)
    parser.add_argument("--privacy-report", type=Path, required=True)
    parser.add_argument("--contamination-report", type=Path, required=True)
    parser.add_argument("--evidence-report", type=Path, required=True)
    parser.add_argument("--trainer-binding", type=Path, required=True)


def _add_chain_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--previous-admission-sha256", required=True)


def _add_policy_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--minimum-policy-revision", type=int, required=True)
    parser.add_argument("--observed-at", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    protocol = commands.add_parser(
        "protocol",
        help="emit the source-bound admission protocol commitment",
    )
    protocol.add_argument("--out", type=Path)

    payload = commands.add_parser(
        "payload",
        help="emit one exact payload for a detached role signature",
    )
    _add_evidence_inputs(payload)
    _add_chain_inputs(payload)
    payload.add_argument("--name", choices=_PAYLOAD_NAMES, required=True)
    payload.add_argument("--out", type=Path)

    assemble = commands.add_parser(
        "assemble",
        help="verify detached role attestations and assemble admission",
    )
    _add_evidence_inputs(assemble)
    _add_chain_inputs(assemble)
    _add_policy_inputs(assemble)
    for name in _PAYLOAD_NAMES:
        assemble.add_argument(
            f"--{name.replace('_', '-')}-attestation",
            type=Path,
            required=True,
        )
    assemble.add_argument("--out", type=Path)

    verify = commands.add_parser(
        "verify",
        help="reconstruct admission against caller-pinned monotonic state",
    )
    _add_evidence_inputs(verify)
    _add_policy_inputs(verify)
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--expected-sequence", type=int, required=True)
    verify.add_argument("--expected-previous-admission-sha256", required=True)
    verify.add_argument("--expected-policy-sha256", required=True)
    verify.add_argument("--out", type=Path)
    return parser


def _verified_policy(
    args: argparse.Namespace,
    *,
    documents: dict[str, Any],
    observed_at: int,
):
    from core.learning.structured_sft import validate_candidate_dataset_artifacts

    candidate = validate_candidate_dataset_artifacts(documents["candidate_artifacts"])
    protocol = structured_sft_admission_protocol()
    return validate_campaign_trust_policy(
        _read_json(args.policy, role="policy"),
        trusted_root_public_key_pem=_read_root(args.root),
        expected_campaign_name=f"structured-sft:{candidate['package_sha256']}",
        expected_protocol_sha256=protocol["protocol_sha256"],
        minimum_policy_revision=args.minimum_policy_revision,
        now_unix=observed_at,
    )


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "protocol":
        return structured_sft_admission_protocol()
    documents = _common_documents(args)
    if args.command == "payload":
        payloads = structured_sft_admission_payloads(
            **documents,
            sequence=args.sequence,
            previous_admission_sha256=args.previous_admission_sha256,
        )
        return payloads[args.name]
    observed_at = _observed_at(args.observed_at)
    if args.command == "assemble":
        policy = _verified_policy(
            args,
            documents=documents,
            observed_at=observed_at,
        )
        attestations = {
            name: _read_json(
                getattr(args, f"{name}_attestation"),
                role=f"{name}_attestation",
            )
            for name in _PAYLOAD_NAMES
        }
        return build_structured_sft_admission_bundle(
            **documents,
            policy=policy,
            attestations=attestations,
            sequence=args.sequence,
            previous_admission_sha256=args.previous_admission_sha256,
            observed_at_unix=observed_at,
        )
    bundle = _read_json(args.bundle, role="admission_bundle")
    if bundle.get("policy") != _read_json(args.policy, role="policy"):
        raise StructuredSFTAdmissionToolError("external_policy_bundle_mismatch")
    return validate_structured_sft_admission_bundle(
        bundle,
        **documents,
        trusted_root_public_key_pem=_read_root(args.root),
        expected_sequence=args.expected_sequence,
        expected_previous_admission_sha256=(args.expected_previous_admission_sha256),
        expected_policy_sha256=args.expected_policy_sha256,
        minimum_policy_revision=args.minimum_policy_revision,
        now_unix=observed_at,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        document = _execute(args)
        _emit(document, args.out)
        return 0
    except (
        CampaignTrustError,
        CampaignTrustToolError,
        CandidateDatasetBuildError,
        OSError,
        StableFileReadError,
        StructuredSFTAdmissionError,
        StructuredSFTAdmissionToolError,
        StructuredSFTError,
        ValueError,
    ) as exc:
        error = {
            "schema": "aura.rlc.structured_sft_admission_tool_error.v1",
            "ok": False,
            "reason": getattr(exc, "code", str(exc)) or type(exc).__name__,
        }
        sys.stdout.buffer.write(canonical_json_bytes(error) + b"\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
