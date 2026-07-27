#!/usr/bin/env python3
"""Independently verify a SPARK recurrent-SFT checkpoint for detached retry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.recurrent_sft_retention import retention_manifest  # noqa: E402
from core.learning.recurrent_sft_sampling import (  # noqa: E402
    FAMILY_BALANCED_SAMPLER,
    family_balance_receipt,
    family_balanced_epoch_order,
)
from core.learning.structured_sft_research_authority import (  # noqa: E402
    StructuredSFTResearchAuthorityError,
    sha256_json,
    strict_json_bytes,
    validate_authority,
)
from core.learning.structured_sft_research_state import (  # noqa: E402
    StructuredSFTResearchStateError,
    canonical_json_bytes,
    inspect_checkpoint,
    validate_journal,
)
from core.runtime.atomic_writer import atomic_write_bytes  # noqa: E402
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402

VERDICT_SCHEMA = "aura.detached_step.resume_verdict.v2"
EVIDENCE_SCHEMA = "aura.detached_step.resume_evidence.v1"
COMPLETION_SCHEMA = "aura.rlc.synthetic_recurrent_sft_completion.v1"


class StructuredSFTResumeVerifierError(RuntimeError):
    """The detached invocation or durable checkpoint is not retry-safe."""


def _fail(code: str) -> Never:
    raise StructuredSFTResumeVerifierError(str(code or "resume_verifier_failed"))


def _sha(value: str, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"resume_verifier_{role}_invalid")
    return value


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        return strict_json_bytes(
            read_stable_bytes(
                path.expanduser().resolve(strict=True),
                max_bytes=256 * 1024 * 1024,
            ),
            role=f"resume_verifier_{role}",
        )
    except (
        FileNotFoundError,
        OSError,
        StructuredSFTResearchAuthorityError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise StructuredSFTResumeVerifierError(
            f"resume_verifier_{role}_unreadable"
        ) from exc


def _validate_dataset(raw: dict[str, Any]) -> dict[str, Any]:
    legacy_required = {
        "schema",
        "candidate_identity_sha256",
        "train",
        "validation",
        "holdout",
        "verified_replay",
        "dataset_sha256",
    }
    balanced_required = legacy_required | {"retention", "sampler"}
    schema = raw.get("schema")
    required = (
        balanced_required
        if schema == "aura.rlc.synthetic_recurrent_sft_projected_dataset.v2"
        else legacy_required
    )
    body = dict(raw)
    observed = body.pop("dataset_sha256", None)
    if (
        set(raw) != required
        or schema
        not in {
            "aura.rlc.synthetic_recurrent_sft_projected_dataset.v1",
            "aura.rlc.synthetic_recurrent_sft_projected_dataset.v2",
        }
        or not isinstance(raw.get("train"), list)
        or not raw["train"]
        or not isinstance(raw.get("validation"), list)
        or not raw["validation"]
        or raw.get("holdout") is not None
        or raw.get("verified_replay") is not None
        or observed != sha256_json(body)
    ):
        _fail("resume_verifier_dataset_manifest_invalid")
    if schema.endswith(".v2") and (
        raw.get("retention") != retention_manifest()
        or not isinstance(raw.get("sampler"), dict)
        or raw["sampler"].get("name") != FAMILY_BALANCED_SAMPLER
    ):
        _fail("resume_verifier_dataset_sampling_invalid")
    return raw


def _detached_context() -> dict[str, Any]:
    try:
        prior_attempt = int(os.environ["AURA_DETACHED_PRIOR_ATTEMPT"])
    except (KeyError, TypeError, ValueError) as exc:
        raise StructuredSFTResumeVerifierError(
            "resume_verifier_detached_context_invalid"
        ) from exc
    context = {
        "plan_sha256": _sha(
            os.environ.get("AURA_DETACHED_PLAN_SHA256", ""),
            role="plan",
        ),
        "command_sha256": _sha(
            os.environ.get("AURA_DETACHED_COMMAND_SHA256", ""),
            role="command",
        ),
        "prior_attempt": prior_attempt,
        "prior_journal_head_sha256": _sha(
            os.environ.get("AURA_DETACHED_PRIOR_JOURNAL_HEAD_SHA256", ""),
            role="journal_head",
        ),
        "evidence_path": Path(
            os.environ.get("AURA_DETACHED_RESUME_EVIDENCE_PATH", "")
        ).expanduser(),
    }
    if prior_attempt < 1 or not context["evidence_path"].is_absolute():
        _fail("resume_verifier_detached_context_invalid")
    return context


def _bindings(authority: dict[str, Any], dataset: dict[str, Any]) -> dict[str, str]:
    return {
        "authority_sha256": authority["authority_sha256"],
        "dataset_sha256": dataset["dataset_sha256"],
        "tokenization_identity_sha256": authority["tokenization"][
            "identity_sha256"
        ],
        "model_identity_sha256": authority["model"]["identity_sha256"],
        "source_closure_sha256": authority["sources"]["closure_sha256"],
        "execution_spec_sha256": authority["execution_spec"]["semantic_sha256"],
        "trainer_config_sha256": sha256_json(authority["trainer"]),
    }


def _validate_completion(
    raw: dict[str, Any],
    *,
    authority: dict[str, Any],
    dataset: dict[str, Any],
    checkpoint_id: str,
    checkpoint_step: int,
) -> dict[str, Any]:
    required = {
        "schema",
        "authority_sha256",
        "dataset_sha256",
        "model_identity_sha256",
        "execution_spec_sha256",
        "step",
        "halt_reason",
        "terminal",
        "baseline_validation",
        "final_validation",
        "checkpoint",
        "base_weights_unchanged",
        "output_disposition",
        "ordinary_lexical_adapter_activation",
        "production_effect",
        "promotion_allowed",
        "claims_not_supported",
        "completion_sha256",
    }
    body = dict(raw)
    observed = body.pop("completion_sha256", None)
    if (
        set(raw) != required
        or raw.get("schema") != COMPLETION_SCHEMA
        or observed != sha256_json(body)
        or raw.get("authority_sha256") != authority["authority_sha256"]
        or raw.get("dataset_sha256") != dataset["dataset_sha256"]
        or raw.get("model_identity_sha256")
        != authority["model"]["identity_sha256"]
        or raw.get("execution_spec_sha256")
        != authority["execution_spec"]["semantic_sha256"]
        or raw.get("step") != checkpoint_step
        or raw.get("halt_reason") != "max_steps"
        or raw.get("terminal") is not True
        or raw.get("checkpoint") != checkpoint_id
        or raw.get("base_weights_unchanged") is not True
        or raw.get("output_disposition") != "quarantined_research_only"
        or raw.get("ordinary_lexical_adapter_activation") is not False
        or raw.get("production_effect") is not False
        or raw.get("promotion_allowed") is not False
        or raw.get("claims_not_supported") != authority["claims_not_supported"]
        or not isinstance(raw.get("baseline_validation"), dict)
        or not isinstance(raw.get("final_validation"), dict)
    ):
        _fail("resume_verifier_completion_invalid")
    return raw


def build_verdict(
    *,
    authority_path: Path,
    expected_authority_sha256: str,
    run_dir: Path,
    detached_context: dict[str, Any],
) -> dict[str, Any]:
    authority = validate_authority(
        _read_json(authority_path, role="authority"),
        expected_authority_sha256=expected_authority_sha256,
        allow_expired_resume=True,
    )
    dataset = _validate_dataset(
        _read_json(
            run_dir / "projected_dataset_manifest.json",
            role="dataset_manifest",
        )
    )
    inspected = inspect_checkpoint(
        run_dir,
        expected_bindings=_bindings(authority, dataset),
    )
    events = validate_journal(run_dir)
    state = inspected.state
    if authority["trainer"]["sampler"] == FAMILY_BALANCED_SAMPLER:
        epoch_zero_order = family_balanced_epoch_order(
            dataset["train"],
            seed=authority["trainer"]["seed"],
            epoch=0,
        )
        current_order = family_balanced_epoch_order(
            dataset["train"],
            seed=authority["trainer"]["seed"],
            epoch=state["epoch"],
        )
        if (
            dataset.get("schema")
            != "aura.rlc.synthetic_recurrent_sft_projected_dataset.v2"
            or dataset.get("sampler")
            != {
                "name": FAMILY_BALANCED_SAMPLER,
                "epoch_zero_order": epoch_zero_order,
                "epoch_zero_balance": family_balance_receipt(
                    dataset["train"],
                    epoch_zero_order,
                ),
            }
            or state["sampler"] != FAMILY_BALANCED_SAMPLER
            or state["order"] != current_order
        ):
            _fail("resume_verifier_balanced_sampler_drift")
    elif dataset.get("schema") != (
        "aura.rlc.synthetic_recurrent_sft_projected_dataset.v1"
    ):
        _fail("resume_verifier_legacy_sampler_drift")
    if state["terminal"]:
        completion_path = run_dir / "research_completion.json"
        if completion_path.exists() or completion_path.is_symlink():
            _validate_completion(
                _read_json(completion_path, role="completion"),
                authority=authority,
                dataset=dataset,
                checkpoint_id=inspected.checkpoint_dir.name,
                checkpoint_step=state["step"],
            )
            verdict_name = "already_completed"
        elif events and events[-1]["event_type"] != "TERMINAL":
            verdict_name = "safe_to_resume"
        else:
            verdict_name = "indeterminate"
    elif (
        not state["last_step_committed"]
        or state["elapsed_training_s"] >= authority["trainer"]["max_minutes"] * 60.0
        or not events
        or events[-1]["event_type"] == "TERMINAL"
    ):
        verdict_name = "indeterminate"
    else:
        verdict_name = "safe_to_resume"
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "plan_sha256": detached_context["plan_sha256"],
        "command_sha256": detached_context["command_sha256"],
        "prior_attempt": detached_context["prior_attempt"],
        "prior_journal_head_sha256": detached_context[
            "prior_journal_head_sha256"
        ],
        "checkpoint_sequence": state["step"],
        "authority_sha256": authority["authority_sha256"],
        "checkpoint_id": inspected.checkpoint_dir.name,
        "checkpoint_complete_sha256": inspected.complete_sha256,
        "checkpoint_terminal": state["terminal"],
        "elapsed_training_s": state["elapsed_training_s"],
        "trainer_max_seconds": authority["trainer"]["max_minutes"] * 60.0,
        "research_journal_sequence": events[-1]["sequence"],
        "research_journal_head_sha256": events[-1]["event_sha256"],
        "checkpoint_state": verdict_name,
    }
    evidence_payload = canonical_json_bytes(evidence)
    evidence_path = detached_context["evidence_path"]
    if evidence_path.exists() or evidence_path.is_symlink():
        _fail("resume_verifier_evidence_path_preexists")
    atomic_write_bytes(evidence_path, evidence_payload, mode=0o600)
    evidence_sha256 = hashlib.sha256(evidence_payload).hexdigest()
    checkpoint_identity = hashlib.sha256(
        json.dumps(
            {
                "prior_attempt": detached_context["prior_attempt"],
                "prior_journal_head_sha256": detached_context[
                    "prior_journal_head_sha256"
                ],
                "checkpoint_sequence": state["step"],
                "evidence_sha256": evidence_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    return {
        "schema": VERDICT_SCHEMA,
        "plan_sha256": detached_context["plan_sha256"],
        "command_sha256": detached_context["command_sha256"],
        "prior_attempt": detached_context["prior_attempt"],
        "prior_journal_head_sha256": detached_context[
            "prior_journal_head_sha256"
        ],
        "checkpoint_sequence": state["step"],
        "checkpoint_identity": checkpoint_identity,
        "verdict": verdict_name,
        "evidence_path": str(evidence_path),
        "evidence_sha256": evidence_sha256,
        "evidence": evidence,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        verdict = build_verdict(
            authority_path=arguments.authority,
            expected_authority_sha256=_sha(
                arguments.expected_authority_sha256,
                role="authority",
            ),
            run_dir=arguments.run_dir.expanduser().resolve(strict=True),
            detached_context=_detached_context(),
        )
    except (
        OSError,
        StructuredSFTResearchAuthorityError,
        StructuredSFTResearchStateError,
        StructuredSFTResumeVerifierError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {
                    "schema": "aura.rlc.synthetic_recurrent_sft_resume_error.v1",
                    "ok": False,
                    "reason": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(verdict, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
