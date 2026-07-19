#!/usr/bin/env python
"""Independent raw-output / compute verification for one paired campaign.

Trusts nothing but raw disk artifacts and deterministic regeneration:

1. plan.json is parsed through the strict CampaignPlan validator; the task
   battery is REGENERATED from the plan's declared generation parameters
   (seeds, domains, difficulty, registry version) and its manifest hash
   must equal the plan's — a doctored plan cannot smuggle different tasks.
2. campaign.jsonl is replayed read-only through the hash-chained journal;
   every committed record is chain-verified on read.
3. The production grade is recomputed, then a separate kernel that imports
   none of the production parser, scorer, or statistics code independently
   reconstructs answers, effects, compute controls, and the 2x2 interaction.
4. Both implementations must agree, and the complete recomputed production
   grade must byte-agree with grade.json, including all comparison evidence,
   statistics, eligibility fields, hashes, and campaign-manifest binding.

Exit 0: every check agrees. Exit 1: any disagreement, with reasons.

Usage:
  .venv/bin/python tools/verify_paired_campaign_evidence.py \
      --campaign-dir <dir> [--contamination-trust-root <public.pem>]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import stat
import sys
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
from core.brain.llm.latent_cortex.campaign_trust import (  # noqa: E402
    CAMPAIGN_RUNNER,
    EVIDENCE_VERIFIER,
    TASK_ISSUER,
    prepare_role_signature_request,
    validate_campaign_trust_policy,
    verify_role_attestation,
)
from core.brain.llm.latent_cortex.frontier_tasks import (  # noqa: E402
    build_task_manifest,
    generate_task_battery,
)
from core.brain.llm.latent_cortex.paired_campaign import (  # noqa: E402
    grade_campaign,
)
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402
from tools.independent_paired_campaign_scoring import (  # noqa: E402
    independent_grade_campaign,
)

PLAN_FILE = "plan.json"
JOURNAL_FILE = "campaign.jsonl"
MANIFEST_FILE = "campaign_manifest.json"
GRADE_FILE = "grade.json"
SEALED_OUTPUT_MANIFEST_FILE = "sealed_output_manifest.json"
ANSWER_REVEAL_REQUEST_FILE = "answer_reveal_request.json"
ANSWER_REVEAL_FILE = "answer_reveal.json"
VERDICT_SCHEMA = "aura.latent_cortex.independent_evidence_verdict.v1"
TASK_ISSUER_PAYLOAD_SCHEMA = "aura.latent_cortex.task_issuer_prelaunch.v1"
CAMPAIGN_RUNNER_PAYLOAD_SCHEMA = "aura.latent_cortex.runner_prelaunch.v1"
FINAL_VERIFIER_PAYLOAD_SCHEMA = "aura.latent_cortex.final_verifier_payload.v1"
SEALED_OUTPUT_MANIFEST_SCHEMA = "aura.latent_cortex.sealed_output_manifest.v1"
ANSWER_REVEAL_PAYLOAD_SCHEMA = "aura.latent_cortex.answer_reveal_payload.v1"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@contextmanager
def _open_journal_readonly(path: Path, plan: CampaignPlan):
    """Replay the journal without leaving a lock behind (verifier is a reader)."""
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


def _regenerate_tasks(plan: CampaignPlan, campaign_dir: Path):
    metadata = plan.to_dict()["metadata"]
    execution_config = metadata.get("execution_config")
    if not isinstance(execution_config, dict):
        raise SystemExit("plan has no execution_config")
    if "generation_seeds" in execution_config:
        if metadata.get("claim_eligible") is True:
            raise ValueError("claim-eligible plan disclosed generation seeds prelaunch")
        seeds = tuple(int(v) for v in execution_config["generation_seeds"])
        seed_source = "legacy_prelaunch_plan"
    else:
        reveal = _canonical_artifact(
            campaign_dir / ANSWER_REVEAL_FILE,
            role="answer reveal",
        )
        payload = reveal.get("payload")
        answers = payload.get("answers") if isinstance(payload, dict) else None
        public_tasks = metadata.get("task_manifest", {}).get("tasks")
        if not isinstance(answers, list) or not isinstance(public_tasks, list):
            raise ValueError("answer reveal cannot supply regeneration seeds")
        commitments = {
            record.get("task_id"): record.get("answer_commitment_sha256")
            for record in public_tasks
            if isinstance(record, dict)
        }
        revealed_seeds: set[int] = set()
        seen_tasks: set[str] = set()
        for answer in answers:
            if not isinstance(answer, dict) or set(answer) != {
                "task_id",
                "answer_commitment_sha256",
                "answer_payload",
            }:
                raise ValueError("answer reveal entry is invalid")
            task_id = answer["task_id"]
            answer_payload = answer["answer_payload"]
            commitment = commitments.get(task_id)
            if (
                not isinstance(task_id, str)
                or task_id in seen_tasks
                or not isinstance(answer_payload, dict)
                or answer.get("answer_commitment_sha256") != commitment
                or _sha256_bytes(canonical_json_bytes(answer_payload)) != commitment
                or type(answer_payload.get("generation_seed")) is not int
            ):
                raise ValueError("answer reveal commitment is invalid")
            seen_tasks.add(task_id)
            revealed_seeds.add(answer_payload["generation_seed"])
        if seen_tasks != set(commitments) or len(revealed_seeds) != execution_config.get(
            "generation_seed_count"
        ):
            raise ValueError("answer reveal seed set is incomplete")
        seeds = tuple(sorted(revealed_seeds))
        if (
            execution_config.get("generation_seed_policy")
            != "external_issuer_uniform_63bit"
            or min(seed.bit_length() for seed in seeds)
            != execution_config.get("generation_seed_min_entropy_bits")
            or (
                metadata.get("claim_eligible") is True
                and min(seed.bit_length() for seed in seeds) < 60
            )
        ):
            raise ValueError("answer reveal seed entropy contract is invalid")
        seed_source = "post_seal_answer_reveal"
    domains = tuple(str(v) for v in execution_config["domains"])
    difficulty = int(execution_config["difficulty"])
    registry_version = str(execution_config["task_registry_version"])
    tasks = generate_task_battery(
        seeds,
        domains=domains,
        difficulty=difficulty,
        registry_version=registry_version,
    )
    return tasks, {
        "seeds": list(seeds),
        "domains": list(domains),
        "difficulty": difficulty,
        "registry_version": registry_version,
        "seed_source": seed_source,
    }


def _trust_root_sha256(path_value: str) -> str:
    from cryptography.hazmat.primitives import serialization

    trust_bytes = Path(path_value).expanduser().resolve(strict=True).read_bytes()
    public_key = serialization.load_pem_public_key(trust_bytes)
    der = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return _sha256_bytes(der)


def _campaign_protocol_sha256() -> str:
    latent_root = REPO_ROOT / "core/brain/llm/latent_cortex"
    paths = (
        REPO_ROOT / "tools/run_latent_cortex_paired_campaign.py",
        *sorted(latent_root.glob("*.py")),
    )
    identity = {
        str(path.relative_to(REPO_ROOT)): _sha256_bytes(
            read_stable_bytes(path, max_bytes=16 * 1024 * 1024)
        )
        for path in paths
    }
    return _sha256_bytes(canonical_json_bytes(identity))


def _verifier_implementation_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "tools/independent_paired_campaign_scoring.py",
        REPO_ROOT / "core/brain/llm/latent_cortex/campaign_journal.py",
        REPO_ROOT / "core/brain/llm/latent_cortex/campaign_trust.py",
        REPO_ROOT / "core/brain/llm/latent_cortex/experiments.py",
        REPO_ROOT / "core/brain/llm/latent_cortex/frontier_tasks.py",
        REPO_ROOT / "core/brain/llm/latent_cortex/paired_campaign.py",
        REPO_ROOT / "core/runtime/file_read_gateway.py",
    )
    identity = [
        {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": _sha256_bytes(
                read_stable_bytes(path, max_bytes=16 * 1024 * 1024)
            ),
        }
        for path in paths
    ]
    return _sha256_bytes(canonical_json_bytes(identity))


def _policy_auditor_matches(policy: Any, audit: Any) -> bool:
    if not isinstance(audit, dict) or not isinstance(audit.get("signature"), dict):
        return False
    encoded = audit["signature"].get("public_key_der_b64")
    if not isinstance(encoded, str):
        return False
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        public_key = serialization.load_der_public_key(
            base64.b64decode(encoded, validate=True)
        )
        if not isinstance(public_key, Ed25519PublicKey):
            return False
        raw = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    except (TypeError, ValueError):
        return False
    return (
        base64.b64encode(raw).decode("ascii")
        == policy.role_pin("contamination_auditor")["public_key_b64"]
    )


def _verify_prelaunch_trust(
    plan: CampaignPlan,
    *,
    campaign_trust_policy: str,
    campaign_trust_root: str,
) -> tuple[Any | None, list[str], dict[str, Any]]:
    metadata = plan.to_dict()["metadata"]
    claim_eligible = metadata.get("claim_eligible") is True
    trust = metadata.get("campaign_trust")
    if not claim_eligible and not isinstance(trust, dict):
        return None, [], {"required": False, "verified": False}
    if not isinstance(trust, dict):
        return None, ["claim-eligible plan has no campaign trust context"], {}
    if not campaign_trust_policy or not campaign_trust_root:
        return None, ["independent campaign trust policy/root not supplied"], {}
    policy_document = json.loads(
        read_stable_bytes(campaign_trust_policy, max_bytes=16 * 1024 * 1024)
    )
    policy = validate_campaign_trust_policy(
        policy_document,
        trusted_root_public_key_pem=read_stable_bytes(
            campaign_trust_root, max_bytes=64 * 1024
        ),
        expected_campaign_name=plan.campaign_name,
        expected_policy_sha256=trust.get("policy_sha256"),
        expected_protocol_sha256=_campaign_protocol_sha256(),
    )
    failures: list[str] = []
    if trust.get("policy") != policy.document:
        failures.append("plan-embedded campaign policy differs from external policy")
    if trust.get("root_key_id") != policy.root_key_id:
        failures.append("plan campaign root identity differs from external root")
    if trust.get("protocol_sha256") != _campaign_protocol_sha256():
        failures.append("plan campaign protocol identity differs from verifier source")
    if not _policy_auditor_matches(policy, metadata.get("contamination_audit")):
        failures.append("contamination auditor is not the pre-pinned policy role")

    unsigned_metadata = dict(metadata)
    unsigned_metadata.pop("campaign_trust", None)
    unsigned_metadata["claim_eligible"] = False
    unsigned_metadata["claim_scope"] = (
        "resident same-checkpoint causal attribution preflight"
    )
    unsigned_plan = CampaignPlan.build(
        plan.campaign_name,
        [plan.cell_definition(cell_id) for cell_id in plan.cell_ids],
        metadata=unsigned_metadata,
    )
    if trust.get("unsigned_plan_sha256") != unsigned_plan.plan_sha256:
        failures.append("runner attestation is not bound to the reconstructed plan")
    task_manifest = unsigned_metadata["task_manifest"]
    task_commitment = unsigned_metadata["task_commitment"]
    execution_config = unsigned_metadata["execution_config"]
    generation_config = {
        "difficulty": execution_config["difficulty"],
        "domains": execution_config["domains"],
        "task_registry_version": execution_config["task_registry_version"],
    }
    if "generation_seeds" in execution_config:
        generation_config["generation_seeds"] = execution_config[
            "generation_seeds"
        ]
    else:
        generation_config["generation_seed_count"] = execution_config[
            "generation_seed_count"
        ]
        generation_config["generation_seed_min_entropy_bits"] = execution_config[
            "generation_seed_min_entropy_bits"
        ]
        generation_config["generation_seed_policy"] = execution_config[
            "generation_seed_policy"
        ]
        generation_config["generation_seed_disclosure"] = execution_config[
            "generation_seed_disclosure"
        ]
    issuer_payload = {
        "schema": TASK_ISSUER_PAYLOAD_SCHEMA,
        "campaign_name": plan.campaign_name,
        "policy_sha256": policy.policy_sha256,
        "unsigned_plan_sha256": unsigned_plan.plan_sha256,
        "task_manifest_sha256": task_manifest["manifest_sha256"],
        "task_commitment_sha256": task_commitment["commitment_sha256"],
        "generation_config_sha256": _sha256_bytes(
            canonical_json_bytes(generation_config)
        ),
    }
    runner_payload = {
        "schema": CAMPAIGN_RUNNER_PAYLOAD_SCHEMA,
        "campaign_name": plan.campaign_name,
        "policy_sha256": policy.policy_sha256,
        "protocol_sha256": _campaign_protocol_sha256(),
        "unsigned_plan_sha256": unsigned_plan.plan_sha256,
        "model_identity_sha256": _sha256_bytes(
            canonical_json_bytes(unsigned_metadata["model_identity"])
        ),
        "adapter_identity_sha256": _sha256_bytes(
            canonical_json_bytes(unsigned_metadata["adapter_identity"])
        ),
        "execution_config_sha256": _sha256_bytes(
            canonical_json_bytes(execution_config)
        ),
        "contamination_audit_sha256": _sha256_bytes(
            canonical_json_bytes(unsigned_metadata["contamination_audit"])
        ),
        "arms": unsigned_metadata["arms"],
        "cell_count": len(unsigned_plan.cell_ids),
    }
    verify_role_attestation(
        policy,
        trust.get("task_issuer_attestation"),
        role=TASK_ISSUER,
        expected_payload=issuer_payload,
    )
    verify_role_attestation(
        policy,
        trust.get("runner_attestation"),
        role=CAMPAIGN_RUNNER,
        expected_payload=runner_payload,
    )
    if trust.get("task_issuer_payload_sha256") != _sha256_bytes(
        canonical_json_bytes(issuer_payload)
    ):
        failures.append("task issuer payload digest differs from reconstruction")
    if trust.get("runner_payload_sha256") != _sha256_bytes(
        canonical_json_bytes(runner_payload)
    ):
        failures.append("runner payload digest differs from reconstruction")
    return policy, failures, {
        "required": claim_eligible,
        "verified": not failures,
        "policy_sha256": policy.policy_sha256,
        "root_key_id": policy.root_key_id,
        "protocol_sha256": _campaign_protocol_sha256(),
        "unsigned_plan_sha256": unsigned_plan.plan_sha256,
    }


def _canonical_artifact(path: Path, *, role: str) -> dict[str, Any]:
    raw = read_stable_bytes(path, max_bytes=64 * 1024 * 1024)
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is not valid JSON") from exc
    if not isinstance(document, dict) or raw != canonical_json_bytes(document) + b"\n":
        raise ValueError(f"{role} is not canonical JSON")
    return document


def _verify_sealed_output_reveal(
    campaign_dir: Path,
    *,
    plan: CampaignPlan,
    tasks: tuple[Any, ...],
    result_records: tuple[dict[str, Any], ...],
    trusted_policy: Any | None,
) -> tuple[list[str], dict[str, Any]]:
    execution = plan.to_dict()["metadata"].get("execution_config")
    if not isinstance(execution, dict) or execution.get(
        "answer_reveal_protocol"
    ) != "sealed_outputs_then_issuer_reveal_v1":
        return [], {"required": False, "verified": False}
    failures: list[str] = []
    sealed = _canonical_artifact(
        campaign_dir / SEALED_OUTPUT_MANIFEST_FILE,
        role="sealed output manifest",
    )
    sealed_material = dict(sealed)
    sealed_sha = sealed_material.pop("manifest_sha256", None)
    if (
        sealed.get("schema") != SEALED_OUTPUT_MANIFEST_SCHEMA
        or sealed.get("plan_sha256") != plan.plan_sha256
        or sealed_sha != _sha256_bytes(canonical_json_bytes(sealed_material))
        or sealed.get("cell_count") != len(plan.cell_ids)
        or not isinstance(sealed.get("cells"), list)
    ):
        failures.append("sealed output manifest is invalid")
    result_by_cell = {record["cell_id"]: record for record in result_records}
    sealed_cells = sealed.get("cells") if isinstance(sealed.get("cells"), list) else []
    if len(sealed_cells) != len(plan.cell_ids):
        failures.append("sealed output manifest cell set is incomplete")
    else:
        for ordinal, cell_id in enumerate(plan.cell_ids):
            cell = sealed_cells[ordinal]
            record = result_by_cell.get(cell_id)
            if (
                not isinstance(cell, dict)
                or set(cell)
                != {
                    "cell_id",
                    "attempt_id",
                    "arm_result_event_sha256",
                    "result_sha256",
                }
                or record is None
                or cell.get("cell_id") != cell_id
                or cell.get("attempt_id") != record.get("attempt_id")
                or cell.get("arm_result_event_sha256")
                != record.get("arm_result_event_sha256")
                or cell.get("result_sha256")
                != _sha256_bytes(canonical_json_bytes(record.get("result")))
            ):
                failures.append(f"sealed output binding differs for {cell_id}")

    reveal = _canonical_artifact(campaign_dir / ANSWER_REVEAL_FILE, role="answer reveal")
    if set(reveal) != {
        "schema",
        "payload",
        "request_sha256",
        "task_issuer_attestation",
        "reveal_sha256",
    } or reveal.get("schema") != "aura.latent_cortex.answer_reveal.v1":
        failures.append("answer reveal envelope is invalid")
        return failures, {
            "required": True,
            "verified": False,
            "sealed_output_manifest_sha256": sealed_sha,
        }
    reveal_material = {
        key: reveal[key]
        for key in ("payload", "request_sha256", "task_issuer_attestation")
    }
    if reveal.get("reveal_sha256") != _sha256_bytes(
        canonical_json_bytes(reveal_material)
    ):
        failures.append("answer reveal digest is invalid")
    payload = reveal.get("payload")
    task_commitment = plan.to_dict()["metadata"]["task_commitment"]
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {
            "schema",
            "campaign_name",
            "plan_sha256",
            "sealed_output_manifest_sha256",
            "task_commitment_sha256",
            "answers",
        }
        or payload.get("schema") != ANSWER_REVEAL_PAYLOAD_SCHEMA
        or payload.get("campaign_name") != plan.campaign_name
        or payload.get("plan_sha256") != plan.plan_sha256
        or payload.get("sealed_output_manifest_sha256") != sealed_sha
        or payload.get("task_commitment_sha256")
        != task_commitment["commitment_sha256"]
        or not isinstance(payload.get("answers"), list)
    ):
        failures.append("answer reveal payload is invalid")
    else:
        expected_answers = [
            {
                "task_id": task.task_id,
                "answer_commitment_sha256": task.public.answer_commitment_sha256,
                "answer_payload": task.reveal_for_verifier(),
            }
            for task in sorted(tasks, key=lambda item: item.task_id)
        ]
        if payload["answers"] != expected_answers:
            failures.append("answer reveal differs from independent task regeneration")

    claim_eligible = plan.to_dict()["metadata"].get("claim_eligible") is True
    if claim_eligible:
        if trusted_policy is None:
            failures.append("answer reveal has no independently trusted issuer")
        else:
            request = _canonical_artifact(
                campaign_dir / ANSWER_REVEAL_REQUEST_FILE,
                role="answer reveal request",
            )
            signed_payload = request.get("signed_payload")
            signed_at = (
                signed_payload.get("signed_at_unix")
                if isinstance(signed_payload, dict)
                else None
            )
            if type(signed_at) is not int:
                failures.append("answer reveal request timestamp is invalid")
            else:
                expected_request = prepare_role_signature_request(
                    trusted_policy,
                    role=TASK_ISSUER,
                    payload=payload,
                    signed_at_unix=signed_at,
                )
                if request != expected_request:
                    failures.append("answer reveal request is not canonical")
                if reveal.get("request_sha256") != request.get("request_sha256"):
                    failures.append("answer reveal request digest is not bound")
                attestation = reveal.get("task_issuer_attestation")
                verified = verify_role_attestation(
                    trusted_policy,
                    attestation,
                    role=TASK_ISSUER,
                    expected_payload=payload,
                    not_before_unix=signed_at,
                )
                if verified != request.get("signed_payload"):
                    failures.append("answer reveal attestation differs from request")
    elif (
        reveal.get("request_sha256") is not None
        or reveal.get("task_issuer_attestation") is not None
    ):
        failures.append("preflight answer reveal contains untrusted claim material")
    return failures, {
        "required": True,
        "verified": not failures,
        "sealed_output_manifest_sha256": sealed_sha,
        "answer_reveal_sha256": reveal.get("reveal_sha256"),
        "issuer_attested": reveal.get("task_issuer_attestation") is not None,
    }


def verify_campaign_evidence(
    campaign_dir: Path,
    *,
    contamination_trust_root: str = "",
    campaign_trust_policy: str = "",
    campaign_trust_root: str = "",
    verifier_attestation: str = "",
) -> dict[str, Any]:
    failures: list[str] = []
    detail: dict[str, Any] = {}

    plan_payload = (campaign_dir / PLAN_FILE).read_bytes()
    plan = CampaignPlan.from_dict(json.loads(plan_payload))
    detail["plan_sha256"] = _sha256_bytes(plan_payload)
    trusted_policy, trust_failures, trust_detail = _verify_prelaunch_trust(
        plan,
        campaign_trust_policy=campaign_trust_policy,
        campaign_trust_root=campaign_trust_root,
    )
    failures.extend(trust_failures)
    detail["campaign_trust"] = trust_detail
    if trust_failures:
        return {
            "schema": VERDICT_SCHEMA,
            "campaign_dir": str(campaign_dir),
            "passed": False,
            "failures": failures,
            **detail,
        }

    # 1. Independent task regeneration binds the plan to real tasks.
    try:
        tasks, generation = _regenerate_tasks(plan, campaign_dir)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        failures.append(f"answer reveal task regeneration failed: {exc}")
        return {
            "schema": VERDICT_SCHEMA,
            "campaign_dir": str(campaign_dir),
            "passed": False,
            "failures": failures,
            **detail,
        }
    detail["generation"] = generation
    regenerated_manifest = build_task_manifest(tasks)
    plan_manifest = plan.to_dict()["metadata"].get("task_manifest") or {}
    declared_sha = None
    if isinstance(plan_manifest, dict):
        declared_sha = plan_manifest.get("manifest_sha256")
    if declared_sha != regenerated_manifest.manifest_sha256:
        failures.append(
            "task manifest mismatch: plan declares "
            f"{declared_sha}, independent regeneration produced "
            f"{regenerated_manifest.manifest_sha256}"
        )
        # Grading against tasks the plan cannot reproduce would be
        # meaningless; the campaign is already unverifiable.
        return {
            "schema": VERDICT_SCHEMA,
            "campaign_dir": str(campaign_dir),
            "passed": False,
            "failures": failures,
            **detail,
        }
    detail["task_count"] = len(regenerated_manifest.tasks)

    # 2. Chain-verified replay of every committed outcome.
    with _open_journal_readonly(campaign_dir / JOURNAL_FILE, plan) as journal:
        records = journal.committed_records()
        result_records = journal.result_records()
    detail["committed_records"] = len(records)
    try:
        reveal_failures, reveal_detail = _verify_sealed_output_reveal(
            campaign_dir,
            plan=plan,
            tasks=tasks,
            result_records=result_records,
            trusted_policy=trusted_policy,
        )
    except (OSError, TypeError, ValueError, KeyError) as exc:
        reveal_failures = [f"sealed output or answer reveal validation failed: {exc}"]
        reveal_detail = {"required": True, "verified": False}
    failures.extend(reveal_failures)
    detail["answer_reveal"] = reveal_detail

    # 3. Regrade through both the production implementation and a separately
    # implemented parser/scorer/statistics kernel.  Neither result can certify
    # the campaign alone; disagreement is a proof failure.
    trusted_root = (
        _trust_root_sha256(contamination_trust_root)
        if contamination_trust_root
        else None
    )
    grade = grade_campaign(
        records,
        plan=plan,
        issuer_tasks=tasks,
        trusted_contamination_root_sha256=trusted_root,
        trusted_campaign_policy_sha256=(
            trusted_policy.policy_sha256 if trusted_policy is not None else None
        ),
    )
    independent_grade = independent_grade_campaign(
        records,
        plan=plan,
        issuer_tasks=tasks,
    )
    detail["recomputed_verdict"] = grade.get("verdict")
    detail["recomputed_claim_tier"] = grade.get("claim_tier")
    detail["independent_verdict"] = independent_grade.get("verdict")
    detail["independent_claim_tier"] = independent_grade.get("claim_tier")
    detail["independent_implementation_sha256"] = independent_grade.get(
        "implementation_sha256"
    )
    for key in (
        "verdict",
        "claim_tier",
        "observed_task_count",
        "observed_cell_count",
        "domain_counts",
    ):
        if grade.get(key) != independent_grade.get(key):
            failures.append(
                f"production grade {key}={grade.get(key)!r} disagrees with "
                f"independent kernel {independent_grade.get(key)!r}"
            )
    production_comparisons = grade.get("comparisons")
    independent_comparisons = independent_grade.get("comparisons")
    if isinstance(production_comparisons, dict) and isinstance(
        independent_comparisons, dict
    ):
        if set(production_comparisons) != set(independent_comparisons):
            failures.append("production and independent comparison sets disagree")
        else:
            for name in sorted(production_comparisons):
                production_claim = production_comparisons[name]
                independent_claim = independent_comparisons[name]
                production_tier = (
                    production_claim.get("tier")
                    if isinstance(production_claim, dict)
                    else None
                )
                independent_tier = (
                    independent_claim.get("tier")
                    if isinstance(independent_claim, dict)
                    else None
                )
                if production_tier != independent_tier:
                    failures.append(
                        f"comparison {name} tier disagrees: production "
                        f"{production_tier!r}, independent {independent_tier!r}"
                    )
    else:
        failures.append("production or independent comparisons are invalid")

    # 4. Agreement with the published grade, if one exists.
    grade_path = campaign_dir / GRADE_FILE
    published_grade_sha256: str | None = None
    campaign_manifest_sha256: str | None = None
    if grade_path.exists():
        published = json.loads(grade_path.read_bytes())
        published_grade_sha256 = published.get("grade_sha256")
        detail["published_verdict"] = published.get("verdict")
        manifest_path = campaign_dir / MANIFEST_FILE
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_bytes())
            campaign_manifest_sha256 = manifest.get("manifest_sha256")
            if published.get("campaign_manifest_sha256") != manifest.get(
                "manifest_sha256"
            ):
                failures.append("published grade is not bound to the manifest")
            expected_material = dict(grade)
            expected_material.pop("grade_sha256", None)
            expected_material["campaign_manifest_sha256"] = manifest.get(
                "manifest_sha256"
            )
            if reveal_detail.get("required") is True:
                expected_material["sealed_output_manifest_sha256"] = (
                    reveal_detail.get("sealed_output_manifest_sha256")
                )
                expected_material["answer_reveal_sha256"] = reveal_detail.get(
                    "answer_reveal_sha256"
                )
            expected_grade = {
                **expected_material,
                "grade_sha256": _sha256_bytes(
                    canonical_json_bytes(expected_material)
                ),
            }
            if published != expected_grade:
                failures.append(
                    "published grade does not fully agree with raw-evidence "
                    "recomputation"
                )
        else:
            failures.append("campaign manifest missing beside published grade")
        recomputed_material = dict(published)
        published_grade_sha = recomputed_material.pop("grade_sha256", None)
        if (
            _sha256_bytes(canonical_json_bytes(recomputed_material))
            != published_grade_sha
        ):
            failures.append("published grade_sha256 does not match its content")
    else:
        detail["published_verdict"] = None

    metadata = plan.to_dict()["metadata"]
    final_attestation_verified = False
    if metadata.get("claim_eligible") is True:
        if trusted_policy is None:
            failures.append("claim-eligible verification has no trusted campaign policy")
        elif (
            trusted_policy.role_pin(EVIDENCE_VERIFIER)["implementation_sha256"]
            != _verifier_implementation_sha256()
        ):
            failures.append(
                "evidence verifier implementation differs from the pre-pinned source"
            )
        elif not isinstance(published_grade_sha256, str) or not isinstance(
            campaign_manifest_sha256, str
        ):
            failures.append(
                "final verifier attestation requires sealed manifest and published grade"
            )
        else:
            implementation_map = metadata["execution_config"].get(
                "implementation_sha256"
            )
            production_grade_sha256 = (
                implementation_map.get(
                    "core/brain/llm/latent_cortex/paired_campaign.py"
                )
                if isinstance(implementation_map, dict)
                else None
            )
            independent_scoring_sha256 = detail.get(
                "independent_implementation_sha256"
            )
            if not all(
                isinstance(value, str) and len(value) == 64
                for value in (
                    production_grade_sha256,
                    independent_scoring_sha256,
                )
            ):
                failures.append(
                    "final verifier payload lacks pinned production or independent kernel identity"
                )
                return {
                    "schema": VERDICT_SCHEMA,
                    "campaign_dir": str(campaign_dir),
                    "passed": False,
                    "failures": failures,
                    **detail,
                }
            final_payload = {
                "schema": FINAL_VERIFIER_PAYLOAD_SCHEMA,
                "campaign_name": plan.campaign_name,
                "policy_sha256": trusted_policy.policy_sha256,
                "plan_sha256": plan.plan_sha256,
                "campaign_manifest_sha256": campaign_manifest_sha256,
                "published_grade_sha256": published_grade_sha256,
                "production_protocol_sha256": _campaign_protocol_sha256(),
                "production_grade_implementation_sha256": production_grade_sha256,
                "independent_scoring_implementation_sha256": (
                    independent_scoring_sha256
                ),
                "verifier_implementation_sha256": _verifier_implementation_sha256(),
            }
            detail["verifier_attestation_request"] = {
                "role": EVIDENCE_VERIFIER,
                "signer_id": trusted_policy.role_pin(EVIDENCE_VERIFIER)[
                    "signer_id"
                ],
                "policy_sha256": trusted_policy.policy_sha256,
                "payload": final_payload,
                "payload_sha256": _sha256_bytes(
                    canonical_json_bytes(final_payload)
                ),
            }
            if not verifier_attestation:
                failures.append("final verifier attestation not supplied")
            else:
                attestation = json.loads(
                    read_stable_bytes(
                        verifier_attestation,
                        max_bytes=16 * 1024 * 1024,
                    )
                )
                prelaunch = metadata["campaign_trust"]
                earliest_final_time = max(
                    int(
                        prelaunch["task_issuer_attestation"]["signed_payload"][
                            "signed_at_unix"
                        ]
                    ),
                    int(
                        prelaunch["runner_attestation"]["signed_payload"][
                            "signed_at_unix"
                        ]
                    ),
                )
                verify_role_attestation(
                    trusted_policy,
                    attestation,
                    role=EVIDENCE_VERIFIER,
                    expected_payload=final_payload,
                    not_before_unix=earliest_final_time,
                )
                detail["verifier_attestation_sha256"] = _sha256_bytes(
                    canonical_json_bytes(attestation)
                )
                final_attestation_verified = True
    detail["verifier_implementation_sha256"] = _verifier_implementation_sha256()
    final_claim_proven = (
        final_attestation_verified
        and not failures
        and grade.get("verdict") == "gain_preverified"
    )

    return {
        "schema": VERDICT_SCHEMA,
        "campaign_dir": str(campaign_dir),
        "passed": not failures,
        "claim_tier": "PROVEN" if final_claim_proven else grade.get("claim_tier"),
        "verified_verdict": (
            "gain_proven" if final_claim_proven else grade.get("verdict")
        ),
        "failures": failures,
        **detail,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--contamination-trust-root", default="")
    parser.add_argument("--campaign-trust-policy", default="")
    parser.add_argument("--campaign-trust-root", default="")
    parser.add_argument("--verifier-attestation", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    try:
        verdict = verify_campaign_evidence(
            Path(args.campaign_dir).expanduser().resolve(strict=True),
            contamination_trust_root=args.contamination_trust_root,
            campaign_trust_policy=args.campaign_trust_policy,
            campaign_trust_root=args.campaign_trust_root,
            verifier_attestation=args.verifier_attestation,
        )
    except Exception as exc:  # noqa: BLE001 - corrupt evidence must fail closed
        verdict = {
            "schema": VERDICT_SCHEMA,
            "campaign_dir": str(args.campaign_dir),
            "passed": False,
            "failures": [
                f"evidence unreadable or invalid: {type(exc).__name__}: {exc}"
            ],
        }
    rendered = json.dumps(verdict, indent=2, sort_keys=True)
    print(rendered)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
