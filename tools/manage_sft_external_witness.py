#!/usr/bin/env python3
"""Prepare, acquire, assemble, and offline-verify SPARK-059 witnesses.

The core verifier has no network or private-key operations.  This operator
tool accepts an already detached signature, invokes Rekor without a shell,
fetches the resulting public entry, and immediately subjects it to the same
offline verification used by later consumers.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    canonical_json_bytes,
)
from core.governance_context import local_internal_governed_scope  # noqa: E402
from core.learning.external_monotonic_witness import (  # noqa: E402
    REKOR_PUBLIC_GOOD_SERVER,
    ExternalMonotonicWitnessError,
    build_external_witness_statement,
    build_rekor_witness_bundle,
    build_spark_059_production_audit_packet,
    validate_rekor_witness_bundle,
)
from core.runtime.file_read_gateway import (  # noqa: E402
    StableFileReadError,
    read_stable_bytes,
)
from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402
from tools.manage_campaign_trust import (  # noqa: E402
    CampaignTrustToolError,
    _atomic_create_or_verify,
)

_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_BINARY_BYTES = 64 * 1024
_MAX_NETWORK_BYTES = 16 * 1024 * 1024
_ARTIFACT_PATHS = {
    "combined_lineage_publication": Path(
        "artifacts/current/cp400_combined_sft_lineage_publication_evidence.json"
    ),
    "external_audit_contract": Path(
        "artifacts/current/cp401_combined_sft_external_audit_evidence.json"
    ),
    "resident_tokenizer_admission": Path(
        "artifacts/current/cp402_verified_replay_sft_tokenizer_evidence.json"
    ),
}


class ExternalWitnessToolError(RuntimeError):
    """Stable operator-facing witness workflow error."""


def _lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _json_from_bytes(payload: bytes, *, role: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {constant}")
            ),
        )
    except (UnicodeDecodeError, ValueError, RecursionError, OverflowError) as exc:
        raise ExternalWitnessToolError(f"{role}_json_invalid") from exc
    if not isinstance(value, dict):
        raise ExternalWitnessToolError(f"{role}_object_required")
    return value


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    return _json_from_bytes(
        read_stable_bytes(_lexical(path), max_bytes=_MAX_JSON_BYTES), role=role
    )


def _read_binary(path: Path, *, role: str) -> bytes:
    payload = read_stable_bytes(_lexical(path), max_bytes=_MAX_BINARY_BYTES)
    if not payload:
        raise ExternalWitnessToolError(f"{role}_empty")
    return payload


def _emit(document: dict[str, Any], output: Path | None) -> None:
    if output is not None:
        _atomic_create_or_verify(_lexical(output), document)
    sys.stdout.buffer.write(canonical_json_bytes(document) + b"\n")


def _atomic_bytes_create_or_verify(path: Path, payload: bytes) -> None:
    destination = _lexical(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ExternalWitnessToolError("symlink output rejected")
    if destination.exists():
        metadata = destination.stat(follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise ExternalWitnessToolError("signing payload is not a regular file")
        if read_stable_bytes(destination, max_bytes=_MAX_JSON_BYTES) == payload:
            return
        raise ExternalWitnessToolError("refusing to overwrite a different signing payload")
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    completed = False
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        completed = True
    finally:
        if not completed:
            destination.unlink(missing_ok=True)


def _artifact_payloads() -> dict[str, bytes]:
    return {
        role: read_stable_bytes(REPO_ROOT / path, max_bytes=_MAX_JSON_BYTES)
        for role, path in _ARTIFACT_PATHS.items()
    }


def _optional_sha(value: str | None) -> str | None:
    if value in {None, "none"}:
        return None
    return value


def _add_packet(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--packet", type=Path, required=True)


def _add_chain(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--previous-statement-sha256", required=True)
    parser.add_argument("--previous-rekor-uuid", default="none")


def _add_witness_material(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--statement", type=Path, required=True)
    parser.add_argument("--signature", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--trusted-log-key", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    packet = commands.add_parser("packet", help="commit the current production audit state")
    packet.add_argument("--source-git-commit", required=True)
    packet.add_argument("--production-replay-candidate-sha256", default="none")
    packet.add_argument("--external-audit-bundle-sha256", default="none")
    packet.add_argument("--resident-tokenizer-bundle-sha256", default="none")
    packet.add_argument("--out", type=Path)

    statement = commands.add_parser("statement", help="prepare a canonical witness statement")
    _add_packet(statement)
    _add_chain(statement)
    statement.add_argument("--issued-at", type=int)
    statement.add_argument("--signing-payload-out", type=Path)
    statement.add_argument("--out", type=Path)

    assemble = commands.add_parser("assemble", help="assemble one fetched Rekor entry")
    _add_witness_material(assemble)
    assemble.add_argument("--rekor-entry", type=Path, required=True)
    assemble.add_argument("--rekor-uuid", required=True)
    assemble.add_argument("--out", type=Path)

    submit = commands.add_parser("submit", help="submit and immediately verify a witness")
    _add_packet(submit)
    _add_chain(submit)
    _add_witness_material(submit)
    submit.add_argument("--rekor-cli", type=Path, required=True)
    submit.add_argument("--out", type=Path)

    verify = commands.add_parser("verify", help="offline-verify a committed witness")
    _add_packet(verify)
    _add_chain(verify)
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--trusted-log-key", type=Path, required=True)
    verify.add_argument("--minimum-log-index", type=int)
    verify.add_argument("--minimum-integrated-time", type=int)
    verify.add_argument("--out", type=Path)
    return parser


def _previous_uuid(value: str) -> str | None:
    return None if value == "none" else value


def _extract_uuid(value: Any) -> str:
    candidates: set[str] = set()

    def visit(current: Any, key: str | None = None) -> None:
        if isinstance(current, dict):
            for child_key, child in current.items():
                visit(child, child_key)
        elif isinstance(current, list):
            for child in current:
                visit(child)
        elif isinstance(current, str) and key is not None:
            candidate = current.rsplit("/", 1)[-1] if key.lower() == "location" else current
            if (
                key.lower() in {"uuid", "location"}
                and len(candidate) == 80
                and all(character in "0123456789abcdef" for character in candidate)
            ):
                candidates.add(candidate)

    visit(value)
    if len(candidates) != 1:
        raise ExternalWitnessToolError("rekor_upload_uuid_ambiguous")
    return next(iter(candidates))


def _fetch_rekor_entry(uuid: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{REKOR_PUBLIC_GOOD_SERVER}/api/v1/log/entries/{uuid}",
        method="GET",
        headers={"Accept": "application/json", "User-Agent": "Aura-SPARK-witness/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read(_MAX_NETWORK_BYTES + 1)
    except (OSError, urllib.error.URLError) as exc:
        raise ExternalWitnessToolError("rekor_entry_fetch_failed") from exc
    if len(payload) > _MAX_NETWORK_BYTES:
        raise ExternalWitnessToolError("rekor_entry_response_too_large")
    envelope = _json_from_bytes(payload, role="rekor_entry_response")
    if set(envelope) != {uuid} or not isinstance(envelope[uuid], dict):
        raise ExternalWitnessToolError("rekor_entry_response_binding_invalid")
    return envelope[uuid]


def _assemble(
    *,
    statement_path: Path,
    signature_path: Path,
    certificate_path: Path,
    trusted_log_key_path: Path,
    rekor_uuid: str,
    rekor_entry: dict[str, Any],
) -> dict[str, Any]:
    return build_rekor_witness_bundle(
        statement=_read_json(statement_path, role="statement"),
        producer_signature=_read_binary(signature_path, role="signature"),
        producer_certificate_pem=_read_binary(certificate_path, role="certificate"),
        rekor_uuid=rekor_uuid,
        rekor_entry=rekor_entry,
        trusted_log_public_key_pem=_read_binary(
            trusted_log_key_path, role="trusted_log_key"
        ),
    )


def _submit(args: argparse.Namespace) -> dict[str, Any]:
    if args.out is not None and _lexical(args.out).exists():
        existing = _read_json(args.out, role="existing_witness_bundle")
        packet = _read_json(args.packet, role="packet")
        validate_rekor_witness_bundle(
            existing,
            audit_packet=packet,
            trusted_log_public_key_pem=_read_binary(
                args.trusted_log_key, role="trusted_log_key"
            ),
            expected_sequence=args.sequence,
            expected_previous_statement_sha256=args.previous_statement_sha256,
            expected_previous_rekor_uuid=_previous_uuid(args.previous_rekor_uuid),
        )
        statement = _read_json(args.statement, role="statement")
        signature = _read_binary(args.signature, role="signature")
        certificate = _read_binary(args.certificate, role="certificate")
        if (
            existing.get("statement") != statement
            or existing.get("producer_signature_b64")
            != base64.b64encode(signature).decode("ascii")
            or existing.get("producer_certificate_pem_b64")
            != base64.b64encode(certificate).decode("ascii")
        ):
            raise ExternalWitnessToolError("existing_witness_material_mismatch")
        return existing
    executable = _lexical(args.rekor_cli)
    if not executable.is_file() or executable.is_symlink() or not os.access(executable, os.X_OK):
        raise ExternalWitnessToolError("rekor_cli_not_regular_executable")
    command = [
        os.fspath(executable),
        "upload",
        "--artifact",
        os.fspath(_lexical(args.statement)),
        "--signature",
        os.fspath(_lexical(args.signature)),
        "--public-key",
        os.fspath(_lexical(args.certificate)),
        "--pki-format",
        "x509",
        "--type",
        "rekord:0.0.1",
        "--rekor_server",
        REKOR_PUBLIC_GOOD_SERVER,
        "--format",
        "json",
        "--timeout",
        "30s",
    ]
    with local_internal_governed_scope(
        "spark.external_witness.rekor_upload",
        domain="tool_execution",
        constraints={
            "operator_cli": True,
            "rekor_server": REKOR_PUBLIC_GOOD_SERVER,
        },
    ):
        completed_process = get_subprocess_gateway().run(
            command,
            capture_output=True,
            check=False,
            timeout=60,
            input="",
            env={"PATH": "/usr/bin:/bin", "HOME": os.fspath(Path.home())},
            source="spark.external_witness.rekor_upload",
        )
    stdout = completed_process.stdout.encode("utf-8")
    if completed_process.returncode != 0 or len(stdout) > _MAX_NETWORK_BYTES:
        raise ExternalWitnessToolError("rekor_upload_failed")
    upload = _json_from_bytes(stdout, role="rekor_upload")
    uuid = _extract_uuid(upload)
    bundle = _assemble(
        statement_path=args.statement,
        signature_path=args.signature,
        certificate_path=args.certificate,
        trusted_log_key_path=args.trusted_log_key,
        rekor_uuid=uuid,
        rekor_entry=_fetch_rekor_entry(uuid),
    )
    packet = _read_json(args.packet, role="packet")
    validate_rekor_witness_bundle(
        bundle,
        audit_packet=packet,
        trusted_log_public_key_pem=_read_binary(
            args.trusted_log_key, role="trusted_log_key"
        ),
        expected_sequence=args.sequence,
        expected_previous_statement_sha256=args.previous_statement_sha256,
        expected_previous_rekor_uuid=_previous_uuid(args.previous_rekor_uuid),
    )
    return bundle


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "packet":
        return build_spark_059_production_audit_packet(
            source_git_commit=args.source_git_commit,
            artifact_payloads=_artifact_payloads(),
            production_replay_candidate_sha256=_optional_sha(
                args.production_replay_candidate_sha256
            ),
            external_audit_bundle_sha256=_optional_sha(
                args.external_audit_bundle_sha256
            ),
            resident_tokenizer_bundle_sha256=_optional_sha(
                args.resident_tokenizer_bundle_sha256
            ),
        )
    if args.command == "statement":
        return build_external_witness_statement(
            audit_packet=_read_json(args.packet, role="packet"),
            sequence=args.sequence,
            previous_statement_sha256=args.previous_statement_sha256,
            previous_rekor_uuid=_previous_uuid(args.previous_rekor_uuid),
            issued_at_unix=int(time.time()) if args.issued_at is None else args.issued_at,
        )
    if args.command == "assemble":
        envelope = _read_json(args.rekor_entry, role="rekor_entry")
        entry = envelope.get(args.rekor_uuid, envelope)
        if not isinstance(entry, dict):
            raise ExternalWitnessToolError("rekor_entry_object_required")
        return _assemble(
            statement_path=args.statement,
            signature_path=args.signature,
            certificate_path=args.certificate,
            trusted_log_key_path=args.trusted_log_key,
            rekor_uuid=args.rekor_uuid,
            rekor_entry=entry,
        )
    if args.command == "submit":
        return _submit(args)
    return validate_rekor_witness_bundle(
        _read_json(args.bundle, role="bundle"),
        audit_packet=_read_json(args.packet, role="packet"),
        trusted_log_public_key_pem=_read_binary(
            args.trusted_log_key, role="trusted_log_key"
        ),
        expected_sequence=args.sequence,
        expected_previous_statement_sha256=args.previous_statement_sha256,
        expected_previous_rekor_uuid=_previous_uuid(args.previous_rekor_uuid),
        minimum_log_index=args.minimum_log_index,
        minimum_integrated_time=args.minimum_integrated_time,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        document = _execute(args)
        if args.command == "statement" and args.signing_payload_out is not None:
            _atomic_bytes_create_or_verify(
                args.signing_payload_out,
                canonical_json_bytes(document),
            )
        _emit(document, args.out)
        return 0
    except (
        ExternalMonotonicWitnessError,
        ExternalWitnessToolError,
        CampaignTrustToolError,
        OSError,
        StableFileReadError,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        error = {
            "schema": "aura.rlc.external_witness_tool_error.v1",
            "ok": False,
            "reason": getattr(exc, "code", str(exc)) or type(exc).__name__,
        }
        sys.stdout.buffer.write(canonical_json_bytes(error) + b"\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
