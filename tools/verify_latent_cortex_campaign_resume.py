#!/usr/bin/env python3
"""Issue a fail-closed detached-resume verdict for one RLC campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    CampaignJournal,
    CampaignPlan,
    canonical_json_bytes,
)

PLAN_FILE = "plan.json"
JOURNAL_FILE = "campaign.jsonl"
MANIFEST_FILE = "campaign_manifest.json"
GRADE_FILE = "grade.json"
VERDICT_SCHEMA = "aura.detached_step.resume_verdict.v3"


class ResumeVerificationError(RuntimeError):
    pass


def _stable_file_identity(path: Path, *, max_bytes: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ResumeVerificationError(f"unsafe campaign artifact: {path}")
    before = path.stat()
    if before.st_size <= 0 or before.st_size > max_bytes:
        raise ResumeVerificationError(f"campaign artifact size is invalid: {path}")
    payload = path.read_bytes()
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ResumeVerificationError(f"campaign artifact changed while reading: {path}")
    return {
        "path": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _load_plan(path: Path) -> CampaignPlan:
    identity = _stable_file_identity(path, max_bytes=64 * 1024 * 1024)
    if identity is None:
        raise ResumeVerificationError("campaign plan is missing")
    del identity
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResumeVerificationError("campaign plan is invalid") from exc
    return CampaignPlan.from_dict(payload)


def _verify_grade(path: Path, manifest_sha256: str) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResumeVerificationError("campaign grade is invalid") from exc
    if not isinstance(payload, dict):
        raise ResumeVerificationError("campaign grade must be an object")
    claimed = payload.get("grade_sha256")
    material = dict(payload)
    material.pop("grade_sha256", None)
    if (
        payload.get("campaign_manifest_sha256") != manifest_sha256
        or claimed != hashlib.sha256(canonical_json_bytes(material)).hexdigest()
    ):
        raise ResumeVerificationError("campaign grade binding is invalid")


def _bound_environment() -> tuple[str, str, int, str]:
    plan_sha = os.environ.get("AURA_DETACHED_PLAN_SHA256", "")
    command_sha = os.environ.get("AURA_DETACHED_COMMAND_SHA256", "")
    raw_attempt = os.environ.get("AURA_DETACHED_PRIOR_ATTEMPT", "")
    prior_journal_head = os.environ.get("AURA_DETACHED_PRIOR_JOURNAL_HEAD_SHA256", "")
    if os.environ.get("AURA_DETACHED_RESUME_EVIDENCE_TRANSPORT") != "stdout-v3":
        raise ResumeVerificationError("detached resume evidence transport is not stdout-v3")
    if (
        len(plan_sha) != 64
        or len(command_sha) != 64
        or len(prior_journal_head) != 64
        or any(
            character not in "0123456789abcdef"
            for character in plan_sha + command_sha + prior_journal_head
        )
    ):
        raise ResumeVerificationError("detached plan binding is unavailable")
    try:
        prior_attempt = int(raw_attempt)
    except ValueError as exc:
        raise ResumeVerificationError("detached attempt binding is invalid") from exc
    if prior_attempt <= 0:
        raise ResumeVerificationError("detached attempt binding is invalid")
    return plan_sha, command_sha, prior_attempt, prior_journal_head


@contextmanager
def _open_journal_readonly(path: Path, plan: CampaignPlan) -> Iterator[CampaignJournal]:
    lock_path = path.with_name(f"{path.name}.lock")
    lock_existed = lock_path.exists()
    try:
        with CampaignJournal(path, plan) as journal:
            yield journal
    finally:
        if not lock_existed and lock_path.exists():
            lock_stat = lock_path.lstat()
            if (
                stat.S_ISREG(lock_stat.st_mode)
                and lock_stat.st_uid == os.geteuid()
                and lock_stat.st_size == 0
            ):
                lock_path.unlink()


def verify_campaign(campaign_dir: Path) -> dict[str, Any]:
    (
        detached_plan_sha,
        command_sha,
        prior_attempt,
        prior_journal_head,
    ) = _bound_environment()
    campaign_dir = campaign_dir.expanduser().resolve(strict=False)
    plan_path = campaign_dir / PLAN_FILE
    journal_path = campaign_dir / JOURNAL_FILE
    manifest_path = campaign_dir / MANIFEST_FILE
    grade_path = campaign_dir / GRADE_FILE
    identities = {
        "plan": _stable_file_identity(plan_path, max_bytes=64 * 1024 * 1024),
        "journal": _stable_file_identity(journal_path, max_bytes=1024 * 1024 * 1024),
        "manifest": _stable_file_identity(manifest_path, max_bytes=256 * 1024 * 1024),
        "grade": _stable_file_identity(grade_path, max_bytes=256 * 1024 * 1024),
    }
    verdict = "safe_to_resume"
    reason = "campaign_not_started"
    campaign_plan_sha: str | None = None
    committed_count = 0
    runnable_count = 0
    incomplete_count = 0

    if identities["plan"] is None:
        if any(value is not None for key, value in identities.items() if key != "plan"):
            verdict = "indeterminate"
            reason = "campaign_artifacts_exist_without_plan"
    else:
        plan = _load_plan(plan_path)
        campaign_plan_sha = plan.plan_sha256
        if identities["journal"] is None:
            if identities["manifest"] is not None or identities["grade"] is not None:
                verdict = "indeterminate"
                reason = "terminal_artifact_exists_without_journal"
            else:
                reason = "frozen_plan_has_no_journal"
        else:
            with _open_journal_readonly(journal_path, plan) as journal:
                snapshot = journal.resume()
            committed_count = len(snapshot.committed_cell_ids)
            runnable_count = len(snapshot.runnable_cell_ids)
            incomplete_count = len(snapshot.incomplete_cell_ids)
            complete = committed_count == len(plan.cell_ids)
            if complete:
                if identities["manifest"] is not None and identities["grade"] is not None:
                    with _open_journal_readonly(journal_path, plan) as journal:
                        manifest = journal.finalize(manifest_path)
                    _verify_grade(grade_path, manifest["manifest_sha256"])
                    verdict = "already_completed"
                    reason = "complete_campaign_has_terminal_artifacts"
                elif identities["manifest"] is None and identities["grade"] is None:
                    reason = "cells_complete_terminal_publication_pending"
                else:
                    verdict = "indeterminate"
                    reason = "partial_terminal_publication"
            elif identities["manifest"] is not None or identities["grade"] is not None:
                verdict = "indeterminate"
                reason = "incomplete_campaign_has_terminal_artifact"
            else:
                reason = "journal_replay_allows_infrastructure_resume"

    evidence = {
        "schema": "aura.detached_step.resume_evidence.v2",
        "evidence_kind": "aura.latent_cortex.campaign_checkpoint.v1",
        "campaign_dir": str(campaign_dir),
        "plan_sha256": detached_plan_sha,
        "command_sha256": command_sha,
        "prior_attempt": prior_attempt,
        "prior_journal_head_sha256": prior_journal_head,
        "checkpoint_sequence": committed_count,
        "campaign_plan_sha256": campaign_plan_sha,
        "artifact_identities": identities,
        "committed_count": committed_count,
        "runnable_count": runnable_count,
        "incomplete_count": incomplete_count,
        "reason": reason,
        "verdict": verdict,
    }
    # The runner hashes the evidence object it receives over stdout-v3, so this
    # digest must not include the trailing newline the old file transport wrote.
    evidence_sha = hashlib.sha256(canonical_json_bytes(evidence)).hexdigest()
    checkpoint_identity = hashlib.sha256(
        canonical_json_bytes(
            {
                "prior_attempt": prior_attempt,
                "prior_journal_head_sha256": prior_journal_head,
                "checkpoint_sequence": committed_count,
                "evidence_sha256": evidence_sha,
            }
        )
    ).hexdigest()
    return {
        "schema": VERDICT_SCHEMA,
        "plan_sha256": detached_plan_sha,
        "command_sha256": command_sha,
        "prior_attempt": prior_attempt,
        "prior_journal_head_sha256": prior_journal_head,
        "checkpoint_sequence": committed_count,
        "checkpoint_identity": checkpoint_identity,
        "verdict": verdict,
        "evidence_sha256": evidence_sha,
        "evidence": evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True)
    args = parser.parse_args()
    try:
        verdict = verify_campaign(Path(args.campaign_dir))
    except Exception as exc:  # noqa: BLE001 - CLI boundary: every failure becomes exit code + stderr
        print(f"verify_latent_cortex_campaign_resume: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(verdict, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
