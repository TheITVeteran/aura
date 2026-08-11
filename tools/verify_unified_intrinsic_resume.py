#!/usr/bin/env python3
"""Issue an attempt-bound resume verdict for unified recurrence training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.unified_intrinsic_checkpoint import (  # noqa: E402
    UnifiedCheckpointError,
    resolve_checkpoint_generation,
    unpointed_checkpoint_inventory,
)
from tools.unified_intrinsic_resident_identity import (  # noqa: E402
    campaign_checkpoint_binding,
    canonical_bytes,
    canonical_sha256,
    trainer_model_identity_from_manifest,
)

VERDICT_SCHEMA: Final = "aura.detached_step.resume_verdict.v3"
EVIDENCE_SCHEMA: Final = "aura.detached_step.resume_evidence.v2"


class UnifiedResumeVerificationError(RuntimeError):
    """The detached attempt cannot be bound to safe durable state."""


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _read_canonical(path: Path, *, max_bytes: int) -> tuple[dict[str, Any], bytes]:
    if not path.is_absolute() or path.is_symlink():
        raise UnifiedResumeVerificationError("resume artifact is a symlink")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) & 0o077
                or not 0 < before.st_size <= max_bytes
            ):
                raise UnifiedResumeVerificationError("resume artifact custody differs")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining > 0:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise UnifiedResumeVerificationError("resume artifact is unreadable") from exc
    raw = b"".join(chunks)
    if (
        remaining
        or len(raw) != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise UnifiedResumeVerificationError("resume artifact changed while read")
    try:
        decoded = json.loads(raw.decode("ascii"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise UnifiedResumeVerificationError("resume artifact JSON is invalid") from exc
    if not isinstance(decoded, dict) or raw != canonical_bytes(decoded) + b"\n":
        raise UnifiedResumeVerificationError("resume artifact is not canonical")
    return decoded, raw


def _environment() -> tuple[str, str, int, str]:
    if os.environ.get("AURA_DETACHED_RESUME_EVIDENCE_TRANSPORT") != "stdout-v3":
        raise UnifiedResumeVerificationError("resume transport differs")
    plan = os.environ.get("AURA_DETACHED_PLAN_SHA256", "")
    command = os.environ.get("AURA_DETACHED_COMMAND_SHA256", "")
    journal = os.environ.get("AURA_DETACHED_PRIOR_JOURNAL_HEAD_SHA256", "")
    try:
        attempt = int(os.environ.get("AURA_DETACHED_PRIOR_ATTEMPT", ""))
    except ValueError as exc:
        raise UnifiedResumeVerificationError("resume attempt differs") from exc
    if not _is_sha(plan) or not _is_sha(command) or not _is_sha(journal) or attempt < 1:
        raise UnifiedResumeVerificationError("resume environment differs")
    return plan, command, attempt, journal


def verify_resume(config_path: Path) -> dict[str, Any]:
    plan_sha, command_sha, prior_attempt, prior_journal = _environment()
    config, config_raw = _read_canonical(
        config_path.expanduser().resolve(strict=True),
        max_bytes=256 * 1024 * 1024,
    )
    config_body = {key: value for key, value in config.items() if key != "config_sha256"}
    if config.get("config_sha256") != canonical_sha256(config_body):
        raise UnifiedResumeVerificationError("campaign config identity differs")
    output = Path(config["paths"]["training_output"]).expanduser().resolve(strict=True)
    verdict_name = "safe_to_resume"
    reason = "no_checkpoint_deterministic_replay"
    sequence = 0
    checkpoint_sha256: str | None = None
    checkpoint_receipt_sha256: str | None = None
    unpointed_inventory: dict[str, int] | None = None
    try:
        resolved = resolve_checkpoint_generation(
            output,
            stem="checkpoint_latest",
            required=False,
        )
    except (OSError, UnifiedCheckpointError, ValueError) as exc:
        resolved = None
        verdict_name = "indeterminate"
        reason = f"checkpoint_resolution_failed:{type(exc).__name__}"
    if resolved is not None:
        receipt = resolved.receipt
        identity = receipt.get("identity")
        model_identity = identity.get("model") if isinstance(identity, dict) else None
        config_model = config.get("model")
        if (
            not isinstance(identity, dict)
            or identity.get("dataset") != config.get("dataset")
            or identity.get("tokenizer") != config.get("tokenizer")
            or identity.get("tokenized_dataset") != config.get("tokenized_dataset")
            or not isinstance(config_model, dict)
            or model_identity != trainer_model_identity_from_manifest(config_model)
            or identity.get("campaign_binding")
            != campaign_checkpoint_binding(config)
        ):
            verdict_name = "indeterminate"
            reason = "checkpoint_campaign_binding_differs"
        else:
            sequence = int(receipt["step"])
            checkpoint_sha256 = str(receipt["checkpoint_sha256"])
            checkpoint_receipt_sha256 = str(receipt["receipt_sha256"])
            reason = "authoritative_generation_allows_exact_resume"
    elif verdict_name == "safe_to_resume":
        suspicious = [
            output / "checkpoint_latest.json",
            output / "checkpoint_latest.safetensors",
            output / "training_receipt.json",
        ]
        try:
            unpointed_inventory = unpointed_checkpoint_inventory(output)
        except UnifiedCheckpointError:
            verdict_name = "indeterminate"
            reason = "unpointed_checkpoint_inventory_invalid"
        if any(path.exists() or path.is_symlink() for path in suspicious):
            verdict_name = "indeterminate"
            reason = "checkpoint_artifacts_exist_without_authoritative_pointer"
        elif verdict_name == "safe_to_resume" and any(unpointed_inventory.values()):
            reason = "unpointed_first_generation_ignored_for_deterministic_replay"

    training_receipt_identity: dict[str, Any] | None = None
    receipt_path = output / "training_receipt.json"
    if receipt_path.exists() and verdict_name != "indeterminate":
        try:
            training_receipt, training_raw = _read_canonical(
                receipt_path,
                max_bytes=256 * 1024 * 1024,
            )
        except UnifiedResumeVerificationError as exc:
            training_receipt_identity = {
                "binding": "ignored_non_authoritative",
                "reason": type(exc).__name__,
            }
        else:
            receipt_body = {
                key: value
                for key, value in training_receipt.items()
                if key != "receipt_sha256"
            }
            bound = (
                training_receipt.get("receipt_sha256")
                == canonical_sha256(receipt_body)
                and training_receipt.get("steps") == sequence
                and training_receipt.get("latest_checkpoint", {}).get(
                    "checkpoint_sha256"
                )
                == checkpoint_sha256
            )
            training_receipt_identity = {
                "sha256": hashlib.sha256(training_raw).hexdigest(),
                "size_bytes": len(training_raw),
                "complete": training_receipt.get("complete"),
                "steps": training_receipt.get("steps"),
                "binding": "authoritative_checkpoint" if bound else "ignored_stale",
            }
            if bound and training_receipt.get("complete") is True:
                verdict_name = "already_completed"
                reason = "training_receipt_proves_campaign_complete"

    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "evidence_kind": "aura.unified_intrinsic.authoritative_generation.v1",
        "plan_sha256": plan_sha,
        "command_sha256": command_sha,
        "prior_attempt": prior_attempt,
        "prior_journal_head_sha256": prior_journal,
        "checkpoint_sequence": sequence,
        "campaign_config_sha256": config["config_sha256"],
        "campaign_config_file_sha256": hashlib.sha256(config_raw).hexdigest(),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_receipt_sha256": checkpoint_receipt_sha256,
        "training_receipt": training_receipt_identity,
        "unpointed_checkpoint_inventory": unpointed_inventory,
        "reason": reason,
        "verdict": verdict_name,
    }
    evidence_sha256 = hashlib.sha256(canonical_bytes(evidence)).hexdigest()
    checkpoint_identity = hashlib.sha256(
        canonical_bytes(
            {
                "prior_attempt": prior_attempt,
                "prior_journal_head_sha256": prior_journal,
                "checkpoint_sequence": sequence,
                "evidence_sha256": evidence_sha256,
            }
        )
    ).hexdigest()
    return {
        "schema": VERDICT_SCHEMA,
        "plan_sha256": plan_sha,
        "command_sha256": command_sha,
        "prior_attempt": prior_attempt,
        "prior_journal_head_sha256": prior_journal,
        "checkpoint_sequence": sequence,
        "checkpoint_identity": checkpoint_identity,
        "verdict": verdict_name,
        "evidence_sha256": evidence_sha256,
        "evidence": evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        verdict = verify_resume(args.config)
    except Exception as exc:  # noqa: BLE001 - stable verifier boundary
        print(
            f"verify_unified_intrinsic_resume: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    sys.stdout.buffer.write(canonical_bytes(verdict) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
