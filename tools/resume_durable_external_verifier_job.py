#!/usr/bin/env python3
"""Validate one immutable file-protocol verifier checkpoint for safe replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _read_private_canonical(
    path: Path,
    *,
    max_bytes: int,
) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink():
        raise ValueError("symlink rejected")
    metadata = path.stat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or metadata.st_nlink != 1
        or metadata.st_size > max_bytes
    ):
        raise ValueError("file custody invalid")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        raw = os.read(descriptor, max_bytes + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if before != after or len(raw) > max_bytes or len(raw) != metadata.st_size:
        raise ValueError("file changed during read")
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != _canonical(value):
        raise ValueError("file is not canonical JSON")
    return value, raw


def _result_state(
    path: Path,
    *,
    max_bytes: int,
    request_sha256: str,
) -> tuple[str, bytes | None]:
    if not path.exists():
        if path.is_symlink():
            raise ValueError("result symlink rejected")
        return "absent", None
    value, raw = _read_private_canonical(path, max_bytes=max_bytes)
    if value.get("request_sha256") != request_sha256:
        raise ValueError("result request binding mismatch")
    return "valid", raw


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-file", required=True)
    parser.add_argument("--candidate-result-file", required=True)
    parser.add_argument("--authoritative-result-file", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--result-max-bytes", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan_sha256 = os.environ["AURA_DETACHED_PLAN_SHA256"]
        command_sha256 = os.environ["AURA_DETACHED_COMMAND_SHA256"]
        prior_attempt = int(os.environ["AURA_DETACHED_PRIOR_ATTEMPT"])
        prior_head = os.environ["AURA_DETACHED_PRIOR_JOURNAL_HEAD_SHA256"]
        if os.environ["AURA_DETACHED_RESUME_EVIDENCE_TRANSPORT"] != "stdout-v3":
            raise ValueError("resume evidence transport is not stdout-v3")
        if (
            not _is_sha(plan_sha256)
            or not _is_sha(command_sha256)
            or not _is_sha(prior_head)
            or not _is_sha(args.job_id)
            or prior_attempt < 1
            or not 1 <= args.result_max_bytes <= 256 * 1024 * 1024
        ):
            raise ValueError("resume environment invalid")
        request_path = Path(args.request_file)
        candidate_path = Path(args.candidate_result_file)
        result_path = Path(args.authoritative_result_file)
        request, request_bytes = _read_private_canonical(
            request_path,
            max_bytes=512 * 1024 * 1024,
        )
        request_sha256 = request.get("request_sha256")
        if (
            not isinstance(request_sha256, str)
            or not _is_sha(request_sha256)
            or _sha(request_bytes) != args.job_id
        ):
            raise ValueError("request binding invalid")
        candidate_state, candidate_bytes = _result_state(
            candidate_path,
            max_bytes=args.result_max_bytes,
            request_sha256=request_sha256,
        )
        result_state, result_bytes = _result_state(
            result_path,
            max_bytes=args.result_max_bytes,
            request_sha256=request_sha256,
        )
        if result_state == "valid" and (
            candidate_state != "valid" or result_bytes != candidate_bytes
        ):
            raise ValueError("authoritative result conflicts with candidate")
        checkpoint_sequence = 1 if candidate_state == "valid" else 0
        evidence = {
            "schema": "aura.detached_step.resume_evidence.v2",
            "plan_sha256": plan_sha256,
            "command_sha256": command_sha256,
            "prior_attempt": prior_attempt,
            "prior_journal_head_sha256": prior_head,
            "checkpoint_sequence": checkpoint_sequence,
            "checkpoint_state": (
                "candidate_complete_idempotent_replay"
                if candidate_state == "valid"
                else "immutable_request_no_candidate"
            ),
            "job_id": args.job_id,
            "request_sha256": request_sha256,
            "candidate_result_state": candidate_state,
            "authoritative_result_state": result_state,
        }
        evidence_sha256 = _sha(_canonical(evidence))
        checkpoint_identity = _sha(
            _canonical(
                {
                    "prior_attempt": prior_attempt,
                    "prior_journal_head_sha256": prior_head,
                    "checkpoint_sequence": checkpoint_sequence,
                    "evidence_sha256": evidence_sha256,
                }
            )
        )
        verdict = {
            "schema": "aura.detached_step.resume_verdict.v3",
            "plan_sha256": plan_sha256,
            "command_sha256": command_sha256,
            "prior_attempt": prior_attempt,
            "prior_journal_head_sha256": prior_head,
            "checkpoint_sequence": checkpoint_sequence,
            "checkpoint_identity": checkpoint_identity,
            "verdict": "safe_to_resume",
            "evidence_sha256": evidence_sha256,
            "evidence": evidence,
        }
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            f"resume_durable_external_verifier_job: {exc}",
            file=sys.stderr,
        )
        return 2
    sys.stdout.buffer.write(_canonical(verdict) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
