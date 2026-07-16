#!/usr/bin/env python
"""Independently verify or pre-attest a raw Latent Cortex evidence package.

The tool loads no model and imports no Aura runtime service. It is intended to
run in a separate process, container, or review checkout with public trust pins
supplied outside the evidence bundle.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.frontier_verifier import (  # noqa: E402
    FrontierVerificationError,
    VERIFICATION_KERNEL_IDENTITY_SCHEMA,
    prepare_independent_attestation_request,
    verifier_implementation_sha256,
    verify_frontier_evidence_package,
)


def _positive_finite(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _add_package_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--trust", type=Path, required=True)
    parser.add_argument("--out", type=Path)


def _emit(payload: dict, out: Path | None) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if out is not None:
        from core.brain.llm.latent_cortex.persistence import (
            get_latent_cortex_persistence,
        )

        get_latent_cortex_persistence().save_frontier_verification(out, encoded)
    sys.stdout.buffer.write(encoded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser(
        "verify",
        help="recompute raw bindings and emit a final deterministic certificate",
    )
    _add_package_args(verify)

    prepare = subparsers.add_parser(
        "prepare-attestation",
        help="emit exact payload bytes for an externally held Ed25519 verifier key",
    )
    _add_package_args(prepare)
    prepare.add_argument("--verifier-id", required=True)
    prepare.add_argument("--verified-at", type=_positive_finite, required=True)

    subparsers.add_parser(
        "fingerprint",
        help="emit the local verification-kernel digest for an external trust file",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "fingerprint":
        _emit(
            {
                "schema": VERIFICATION_KERNEL_IDENTITY_SCHEMA,
                "verification_kernel_sha256": verifier_implementation_sha256(),
            },
            None,
        )
        return 0
    if args.command == "verify":
        certificate = verify_frontier_evidence_package(
            bundle_path=args.bundle,
            manifest_path=args.manifest,
            artifact_root=args.artifact_root,
            trust_path=args.trust,
        )
        _emit(certificate, args.out)
        return 0 if certificate.get("accepted") is True else 1

    try:
        request = prepare_independent_attestation_request(
            bundle_path=args.bundle,
            manifest_path=args.manifest,
            artifact_root=args.artifact_root,
            trust_path=args.trust,
            verifier_id=args.verifier_id,
            verified_at=args.verified_at,
        )
    except FrontierVerificationError as exc:
        _emit(
            {
                "schema": "aura.latent_cortex.frontier_verifier_error.v1",
                "accepted": False,
                "reason": exc.code,
            },
            args.out,
        )
        return 1
    _emit(request, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
