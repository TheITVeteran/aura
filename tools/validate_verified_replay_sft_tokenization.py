#!/usr/bin/env python3
"""Validate committed verified-replay rows against the resident tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    canonical_json_bytes,
)
from core.learning.verified_replay_sft import (  # noqa: E402
    VerifiedReplaySFTError,
    validate_verified_replay_sft_candidate_artifacts,
    validate_verified_replay_sft_tokenization,
)
from core.learning.verified_replay_sft_publication import (  # noqa: E402
    VerifiedReplaySFTPublicationError,
    read_candidate_publication_with_attestation,
)
from tools.validate_structured_sft_tokenization import (  # noqa: E402
    TokenizerValidationError,
    load_resident_tokenizer,
    resident_tokenizer_artifact_identity,
    resident_tokenizer_runtime_identity,
    resident_tokenizer_snapshot,
)

VERIFIED_REPLAY_TOKENIZER_VALIDATION_BUNDLE_SCHEMA = (
    "aura.rlc.verified_replay_sft_tokenizer_validation_bundle.v1"
)


def validate(
    *,
    candidate_directory: Path,
    tokenizer_directory: Path,
    snapshot_root: Path,
) -> dict[str, Any]:
    """Validate candidate-only replay projection under one stable snapshot."""

    artifacts, custody_commit = read_candidate_publication_with_attestation(
        candidate_directory
    )
    candidate = validate_verified_replay_sft_candidate_artifacts(artifacts)
    manifest = candidate["manifest"]
    with resident_tokenizer_snapshot(tokenizer_directory, snapshot_root) as (
        snapshot,
        tokenizer_identity,
        snapshot_manifest,
    ):
        tokenizer = load_resident_tokenizer(snapshot)
        try:
            from mlx_lm.tuner.datasets import ChatDataset
        except ImportError as exc:
            raise TokenizerValidationError(
                "tokenizer_dependency_unavailable"
            ) from exc
        chat_dataset = ChatDataset([], tokenizer, mask_prompt=True)
        runtime_before = resident_tokenizer_runtime_identity(tokenizer)
        projection = validate_verified_replay_sft_tokenization(
            artifacts,
            tokenizer=tokenizer,
            chat_dataset_process=chat_dataset.process,
        )
        runtime_after = resident_tokenizer_runtime_identity(tokenizer)
        if runtime_after != runtime_before:
            raise TokenizerValidationError(
                "tokenizer_runtime_identity_changed_during_validation"
            )
        snapshot_identity = resident_tokenizer_artifact_identity(snapshot)
        if (
            snapshot_identity["files"] != tokenizer_identity["files"]
            or snapshot_identity["sha256"] != tokenizer_identity["sha256"]
        ):
            raise TokenizerValidationError(
                "tokenizer_snapshot_changed_during_validation"
            )
    if (
        custody_commit.get("candidate_package_sha256")
        != manifest["candidate_package_sha256"]
        or custody_commit.get("custody_root_sha256") != manifest["custody_root_sha256"]
        or custody_commit.get("source_store_sha256") != manifest["source_store_sha256"]
    ):
        raise TokenizerValidationError("replay_candidate_custody_binding_invalid")
    body = {
        "schema": VERIFIED_REPLAY_TOKENIZER_VALIDATION_BUNDLE_SCHEMA,
        "projection": projection,
        "candidate_package_sha256": manifest["candidate_package_sha256"],
        "custody_root_sha256": manifest["custody_root_sha256"],
        "source_store_sha256": manifest["source_store_sha256"],
        "candidate_custody_attestation": {
            "schema": "aura.rlc.verified_replay_sft_candidate_custody.v1",
            "generation_id": custody_commit["generation_id"],
            "candidate_package_sha256": custody_commit["candidate_package_sha256"],
            "evaluator_package_sha256": custody_commit["evaluator_package_sha256"],
            "custody_root_sha256": custody_commit["custody_root_sha256"],
            "source_store_sha256": custody_commit["source_store_sha256"],
            "source_store_revision": custody_commit["source_store_revision"],
            "commit_sha256": custody_commit["commit_sha256"],
            "evaluator_filesystem_accessed": False,
        },
        "tokenizer": {
            **tokenizer_identity,
            "loaded_from_persistent_content_addressed_snapshot": True,
            "snapshot_path": str(snapshot),
            "snapshot_manifest": snapshot_manifest,
            "runtime": runtime_before,
        },
        "trainer_binding_contract": {
            **manifest["trainer_contract"],
            "tokenizer_path": str(snapshot),
            "tokenizer_identity_sha256": tokenizer_identity["sha256"],
            "runtime_identity_sha256": runtime_before["sha256"],
            "snapshot_manifest_sha256": snapshot_manifest[
                "snapshot_manifest_sha256"
            ],
            "revalidate_in_trainer_process": True,
            "candidate_only_revalidation": True,
            "evaluator_filesystem_access_required": False,
            "path_substitution_allowed": False,
        },
        "tokenization_scope": "candidate_train_validation_only",
        "holdout_tokenized": False,
        "evaluator_filesystem_accessed": False,
        "status": "passed_exact_resident_tokenizer_masked_prefix",
        "trainer_ready": False,
        "training_authority": "none_pending_external_audit_and_trainer_admission",
    }
    return {
        **body,
        "validation_bundle_sha256": hashlib.sha256(
            canonical_json_bytes(body)
        ).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        report = validate(
            candidate_directory=arguments.candidate_dir,
            tokenizer_directory=arguments.tokenizer_dir,
            snapshot_root=arguments.snapshot_root,
        )
    except (
        OSError,
        TokenizerValidationError,
        VerifiedReplaySFTError,
        VerifiedReplaySFTPublicationError,
        ValueError,
    ) as exc:
        print(
            "validate_verified_replay_sft_tokenization: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
