#!/usr/bin/env python3
"""Prepare, assemble, and verify a combined-SFT external audit.

The tool reads a committed combined-lineage publication and accepts public
policy material plus detached attestations.  It has no key generation,
private-key loading, candidate renaming, trainer launch, or promotion command.
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
from core.learning.combined_sft_external_audit import (  # noqa: E402
    CombinedSFTExternalAuditError,
    build_combined_sft_external_audit_bundle,
    combined_sft_external_audit_payloads,
    combined_sft_external_audit_protocol,
    validate_combined_sft_external_audit_bundle,
)
from core.learning.combined_sft_lineage_publication import (  # noqa: E402
    CombinedSFTLineagePublicationError,
    read_combined_sft_lineage_publication,
)
from core.runtime.file_read_gateway import (  # noqa: E402
    StableFileReadError,
    read_stable_bytes,
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
    "runner_binding",
)


class CombinedSFTExternalAuditToolError(RuntimeError):
    """Stable operator-facing audit workflow error."""


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
    payload = read_stable_bytes(_lexical_path(path), max_bytes=_MAX_JSON_BYTES)
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {constant}")
            ),
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise CombinedSFTExternalAuditToolError(f"{role}_json_invalid") from exc
    if not isinstance(value, dict):
        raise CombinedSFTExternalAuditToolError(f"{role}_object_required")
    return value


def _read_root(path: Path) -> bytes:
    return read_stable_bytes(_lexical_path(path), max_bytes=_MAX_KEY_BYTES)


def _emit(document: dict[str, Any], output: Path | None) -> None:
    if output is not None:
        _atomic_create_or_verify(output, document)
    sys.stdout.buffer.write(canonical_json_bytes(document) + b"\n")


def _observed_at(value: int | None) -> int:
    observed = int(time.time()) if value is None else value
    if type(observed) is not int or observed <= 0:
        raise CombinedSFTExternalAuditToolError("observed_at_invalid")
    return observed


def _documents(args: argparse.Namespace) -> dict[str, Any]:
    publication = read_combined_sft_lineage_publication(
        _lexical_path(args.candidate_dir),
        evaluator_directory=_lexical_path(args.evaluator_dir),
    )
    return {
        "candidate_artifacts": publication["candidate_artifacts"],
        "evaluator_artifacts": publication["evaluator_artifacts"],
        "privacy_report": _read_json(args.privacy_report, role="privacy_report"),
        "contamination_report": _read_json(
            args.contamination_report,
            role="contamination_report",
        ),
        "execution_report": _read_json(args.execution_report, role="execution_report"),
        "runner_binding": _read_json(args.runner_binding, role="runner_binding"),
    }


def _add_documents(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--evaluator-dir", type=Path, required=True)
    parser.add_argument("--privacy-report", type=Path, required=True)
    parser.add_argument("--contamination-report", type=Path, required=True)
    parser.add_argument("--execution-report", type=Path, required=True)
    parser.add_argument("--runner-binding", type=Path, required=True)


def _add_chain(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--previous-audit-sha256", required=True)


def _add_policy(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--minimum-policy-revision", type=int, required=True)
    parser.add_argument("--observed-at", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    protocol = commands.add_parser("protocol", help="emit the source-bound protocol")
    protocol.add_argument("--out", type=Path)

    payload = commands.add_parser("payload", help="emit one detached-signature payload")
    _add_documents(payload)
    _add_chain(payload)
    payload.add_argument("--name", choices=_PAYLOAD_NAMES, required=True)
    payload.add_argument("--out", type=Path)

    assemble = commands.add_parser("assemble", help="verify signatures and assemble audit")
    _add_documents(assemble)
    _add_chain(assemble)
    _add_policy(assemble)
    for name in _PAYLOAD_NAMES:
        assemble.add_argument(
            f"--{name.replace('_', '-')}-attestation",
            type=Path,
            required=True,
        )
    assemble.add_argument("--out", type=Path)

    verify = commands.add_parser("verify", help="reconstruct against pinned trust and chain")
    _add_documents(verify)
    _add_policy(verify)
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--expected-sequence", type=int, required=True)
    verify.add_argument("--expected-previous-audit-sha256", required=True)
    verify.add_argument("--expected-policy-sha256", required=True)
    verify.add_argument("--out", type=Path)
    return parser


def _verified_policy(
    args: argparse.Namespace,
    *,
    documents: dict[str, Any],
    observed_at: int,
):
    from core.learning.combined_sft_lineage import validate_combined_sft_lineage_custody

    custody = validate_combined_sft_lineage_custody(
        documents["candidate_artifacts"],
        documents["evaluator_artifacts"],
    )
    protocol = combined_sft_external_audit_protocol()
    return validate_campaign_trust_policy(
        _read_json(args.policy, role="policy"),
        trusted_root_public_key_pem=_read_root(args.root),
        expected_campaign_name=f"combined-sft:{custody['commitment_sha256']}",
        expected_protocol_sha256=protocol["protocol_sha256"],
        minimum_policy_revision=args.minimum_policy_revision,
        now_unix=observed_at,
    )


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "protocol":
        return combined_sft_external_audit_protocol()
    documents = _documents(args)
    if args.command == "payload":
        return combined_sft_external_audit_payloads(
            **documents,
            sequence=args.sequence,
            previous_audit_sha256=args.previous_audit_sha256,
        )[args.name]
    observed_at = _observed_at(args.observed_at)
    if args.command == "assemble":
        policy = _verified_policy(args, documents=documents, observed_at=observed_at)
        attestations = {
            name: _read_json(
                getattr(args, f"{name}_attestation"),
                role=f"{name}_attestation",
            )
            for name in _PAYLOAD_NAMES
        }
        return build_combined_sft_external_audit_bundle(
            **documents,
            policy=policy,
            attestations=attestations,
            sequence=args.sequence,
            previous_audit_sha256=args.previous_audit_sha256,
            observed_at_unix=observed_at,
        )
    bundle = _read_json(args.bundle, role="audit_bundle")
    if bundle.get("policy") != _read_json(args.policy, role="policy"):
        raise CombinedSFTExternalAuditToolError("external_policy_bundle_mismatch")
    return validate_combined_sft_external_audit_bundle(
        bundle,
        **documents,
        trusted_root_public_key_pem=_read_root(args.root),
        expected_sequence=args.expected_sequence,
        expected_previous_audit_sha256=args.expected_previous_audit_sha256,
        expected_policy_sha256=args.expected_policy_sha256,
        minimum_policy_revision=args.minimum_policy_revision,
        now_unix=observed_at,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _emit(_execute(args), args.out)
        return 0
    except (
        CampaignTrustError,
        CampaignTrustToolError,
        CombinedSFTExternalAuditError,
        CombinedSFTExternalAuditToolError,
        CombinedSFTLineagePublicationError,
        OSError,
        StableFileReadError,
        ValueError,
    ) as exc:
        error = {
            "schema": "aura.rlc.combined_sft_external_audit_tool_error.v1",
            "ok": False,
            "reason": getattr(exc, "code", str(exc)) or type(exc).__name__,
        }
        sys.stdout.buffer.write(canonical_json_bytes(error) + b"\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
