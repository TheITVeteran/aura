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
from collections.abc import Mapping
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
from core.brain.llm.latent_cortex.independent_worker_campaign_evidence import (  # noqa: E402
    IndependentWorkerCampaignEvidenceError,
    verify_worker_campaign_evidence,
)
from core.brain.llm.latent_cortex.paired_campaign import (  # noqa: E402
    grade_campaign,
)
from core.brain.llm.latent_cortex.worker_origin import (  # noqa: E402
    WORKER_KEY_CUSTODY_DETACHED_SUPERVISOR,
    WORKER_KEY_CUSTODY_PRODUCER_SOFTWARE,
    ZERO_SHA256,
)
from core.brain.llm.latent_cortex.worker_origin_legacy import (  # noqa: E402
    build_legacy_worker_authorization_payload,
    verify_legacy_worker_authorization,
    verify_legacy_worker_result_origin,
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
FINAL_RUN_REQUEST_FILE = "final_run_request.json"
FINAL_RUN_ENVELOPE_FILE = "final_run_envelope.json"
WORKER_AUTHORIZATION_MANIFEST_FILE = "worker_authorization_manifest.json"
WORKER_LIFECYCLE_MANIFEST_FILE = "worker_lifecycle_manifest.json"
WORKER_KEY_ERASURE_MANIFEST_FILE = "worker_key_erasure_manifest.json"
WORKER_ORIGIN_DIR = "worker_origins"
VERDICT_SCHEMA = "aura.latent_cortex.independent_evidence_verdict.v2"
TASK_ISSUER_PAYLOAD_SCHEMA = "aura.latent_cortex.task_issuer_prelaunch.v1"
CAMPAIGN_RUNNER_PAYLOAD_SCHEMA = "aura.latent_cortex.runner_prelaunch.v1"
FINAL_VERIFIER_PAYLOAD_SCHEMA = "aura.latent_cortex.final_verifier_payload.v4"
SEALED_OUTPUT_MANIFEST_SCHEMA = "aura.latent_cortex.sealed_output_manifest.v4"
ANSWER_REVEAL_PAYLOAD_SCHEMA = "aura.latent_cortex.answer_reveal_payload.v1"
FINAL_RUN_PAYLOAD_SCHEMA = "aura.latent_cortex.final_run_payload.v4"
WORKER_AUTHORIZATION_MANIFEST_SCHEMA = (
    "aura.latent_cortex.worker_authorization_manifest.v1"
)
WORKER_LIFECYCLE_MANIFEST_SCHEMA = (
    "aura.latent_cortex.worker_lifecycle_manifest.v1"
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _first_semantic_difference(
    production: Any,
    independent: Any,
    *,
    path: str = "$",
) -> str | None:
    if type(production) is not type(independent):
        return (
            f"{path}: type {type(production).__name__} != "
            f"{type(independent).__name__}"
        )
    if isinstance(production, dict):
        production_keys = set(production)
        independent_keys = set(independent)
        if production_keys != independent_keys:
            missing = sorted(production_keys - independent_keys)
            extra = sorted(independent_keys - production_keys)
            return f"{path}: missing={missing!r} extra={extra!r}"
        for key in sorted(production):
            difference = _first_semantic_difference(
                production[key],
                independent[key],
                path=f"{path}.{key}",
            )
            if difference is not None:
                return difference
        return None
    if isinstance(production, list):
        if len(production) != len(independent):
            return f"{path}: length {len(production)} != {len(independent)}"
        for index, (left, right) in enumerate(
            zip(production, independent, strict=True)
        ):
            difference = _first_semantic_difference(
                left,
                right,
                path=f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        return None
    if production != independent:
        return f"{path}: {production!r} != {independent!r}"
    return None


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
        REPO_ROOT
        / "core/brain/llm/latent_cortex/independent_worker_campaign_evidence.py",
        REPO_ROOT / "core/brain/llm/latent_cortex/paired_campaign.py",
        REPO_ROOT / "core/brain/llm/latent_cortex/detached_campaign_evidence.py",
        REPO_ROOT / "core/brain/llm/latent_cortex/worker_attempt_import.py",
        REPO_ROOT / "core/brain/llm/latent_cortex/worker_origin.py",
        REPO_ROOT / "core/runtime/detached_subprocess_broker.py",
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

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{role} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{role} contains non-finite number {value}")

    try:
        document = json.loads(
            raw,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is not valid JSON") from exc
    if not isinstance(document, dict) or raw != canonical_json_bytes(document) + b"\n":
        raise ValueError(f"{role} is not canonical JSON")
    return document


def _worker_origin_paths(
    campaign_dir: Path,
    arm: str,
    attempt_slot: int,
) -> dict[str, Path]:
    stem = f"{arm}.attempt-{attempt_slot:02d}"
    root = campaign_dir / WORKER_ORIGIN_DIR
    return {
        "private_key": root / f".{stem}.private-key.raw",
        "request": root / f"{stem}.request.json",
        "attestation": root / f"{stem}.attestation.json",
        "launch": root / f"{stem}.launch.json",
        "exit": root / f"{stem}.exit.json",
        "erasure_intent": root / f"{stem}.erasure-intent.json",
        "erasure": root / f"{stem}.erasure.json",
    }


def _command_option(command: list[str], option: str) -> str:
    positions = [index for index, value in enumerate(command) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"worker command option is invalid: {option}")
    value = command[positions[0] + 1]
    if not value or value.startswith("--"):
        raise ValueError(f"worker command value is invalid: {option}")
    return value


def _verify_worker_command(
    command: Any,
    *,
    campaign_dir: Path,
    plan: CampaignPlan,
    arm: str,
    attempt_slot: int,
    worker_boot_id: str,
) -> None:
    if (
        not isinstance(command, list)
        or len(command) < 10
        or any(not isinstance(argument, str) or not argument for argument in command)
    ):
        raise ValueError("worker command is invalid")
    runner_path = REPO_ROOT / "tools/run_latent_cortex_paired_campaign.py"
    if Path(command[1]).resolve() != runner_path.resolve():
        raise ValueError("worker command runner path differs")
    if any(
        forbidden in command
        for forbidden in (
            "--seeds",
            "--answer-reveal-attestation",
            "--final-run-attestation",
        )
    ):
        raise ValueError("worker command contains answer-bearing material")
    paths = _worker_origin_paths(campaign_dir, arm, attempt_slot)
    expected = {
        "--campaign-dir": str(campaign_dir.resolve()),
        "--campaign-name": plan.campaign_name,
        "--worker-arm": arm,
        "--worker-attempt-slot": str(attempt_slot),
        "--worker-boot-id": worker_boot_id,
        "--worker-private-key": str(paths["private_key"]),
        "--worker-authorization": str(paths["attestation"]),
    }
    metadata = plan.to_dict()["metadata"]
    model_path = metadata.get("model_identity", {}).get("model_path")
    adapter_dir = metadata.get("adapter_identity", {}).get("adapter_dir")
    if isinstance(model_path, str):
        expected["--model"] = model_path
    if isinstance(adapter_dir, str):
        expected["--adapter"] = adapter_dir
    for option, value in expected.items():
        if _command_option(command, option) != value:
            raise ValueError(f"worker command binding differs: {option}")
    execution = metadata.get("execution_config")
    if not isinstance(execution, dict):
        raise ValueError("worker command has no execution contract")
    numeric_options = {
        "--seed-count": execution.get("generation_seed_count"),
        "--seed-entropy-bits": execution.get("generation_seed_min_entropy_bits"),
        "--difficulty": execution.get("difficulty"),
        "--n-slots": execution.get("requested_rlc_shape", {}).get("n_slots"),
        "--branches": execution.get("requested_rlc_shape", {}).get("branches"),
        "--rlc-steps": execution.get("requested_rlc_shape", {}).get("rlc_steps"),
        "--decode-max-tokens": execution.get("decode_max_tokens"),
        "--max-infra-attempts": execution.get("worker_origin_attempt_slots"),
    }
    for option, value in numeric_options.items():
        if value is not None and _command_option(command, option) != str(value):
            raise ValueError(f"worker command execution value differs: {option}")
    if execution.get("profile") is not None and _command_option(
        command, "--profile"
    ) != str(execution["profile"]):
        raise ValueError("worker command profile differs")
    if "--confirmatory" not in command:
        raise ValueError("claim worker command is not confirmatory")


def _verify_worker_authorization_manifest(
    campaign_dir: Path,
    *,
    plan: CampaignPlan,
    trusted_policy: Any,
) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    manifest = _canonical_artifact(
        campaign_dir / WORKER_AUTHORIZATION_MANIFEST_FILE,
        role="worker authorization manifest",
    )
    material = dict(manifest)
    manifest_sha256 = material.pop("manifest_sha256", None)
    metadata = plan.to_dict()["metadata"]
    execution = metadata.get("execution_config")
    arms = metadata.get("arms")
    attempt_slots = (
        execution.get("worker_origin_attempt_slots")
        if isinstance(execution, dict)
        else None
    )
    if (
        set(manifest)
        != {
            "schema",
            "claim_required",
            "campaign_name",
            "policy_sha256",
            "protocol_sha256",
            "plan_sha256",
            "attempt_slots_per_arm",
            "entries",
            "manifest_sha256",
        }
        or manifest.get("schema") != WORKER_AUTHORIZATION_MANIFEST_SCHEMA
        or manifest.get("claim_required") is not True
        or manifest.get("campaign_name") != plan.campaign_name
        or manifest.get("policy_sha256") != trusted_policy.policy_sha256
        or manifest.get("protocol_sha256") != _campaign_protocol_sha256()
        or manifest.get("plan_sha256") != plan.plan_sha256
        or manifest.get("attempt_slots_per_arm") != attempt_slots
        or manifest_sha256 != _sha256_bytes(canonical_json_bytes(material))
        or not isinstance(arms, list)
        or not isinstance(attempt_slots, int)
        or attempt_slots <= 0
        or not isinstance(manifest.get("entries"), list)
    ):
        raise ValueError("worker authorization manifest is invalid")
    expected_positions = [
        (arm, attempt_slot)
        for arm in arms
        for attempt_slot in range(1, attempt_slots + 1)
    ]
    entries = manifest["entries"]
    if len(entries) != len(expected_positions):
        raise ValueError("worker authorization manifest is incomplete")
    by_position: dict[tuple[str, int], dict[str, Any]] = {}
    for entry, (arm, attempt_slot) in zip(entries, expected_positions, strict=True):
        if (
            not isinstance(entry, dict)
            or set(entry)
            != {
                "arm",
                "attempt_slot",
                "worker_boot_id",
                "worker_key_id",
                "authorization_payload",
                "request_sha256",
                "attestation_sha256",
            }
            or entry.get("arm") != arm
            or entry.get("attempt_slot") != attempt_slot
            or not isinstance(entry.get("authorization_payload"), dict)
        ):
            raise ValueError("worker authorization entry order differs")
        payload = entry["authorization_payload"]
        try:
            public_raw = base64.b64decode(
                payload.get("worker_public_key_b64"), validate=True
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("worker authorization key is invalid") from exc
        expected_payload = build_legacy_worker_authorization_payload(
            campaign_name=plan.campaign_name,
            policy_sha256=trusted_policy.policy_sha256,
            protocol_sha256=_campaign_protocol_sha256(),
            plan_sha256=plan.plan_sha256,
            arm=arm,
            worker_attempt_slot=attempt_slot,
            worker_boot_id=str(entry.get("worker_boot_id") or ""),
            worker_key_custody=WORKER_KEY_CUSTODY_PRODUCER_SOFTWARE,
            worker_source_sha256=metadata["execution_config"][
                "implementation_sha256"
            ]["tools/run_latent_cortex_paired_campaign.py"],
            worker_command=payload.get("worker_command"),
            model_identity_sha256=_sha256_bytes(
                canonical_json_bytes(metadata["model_identity"])
            ),
            adapter_identity_sha256=_sha256_bytes(
                canonical_json_bytes(metadata["adapter_identity"])
            ),
            worker_public_key_raw=public_raw,
        )
        if payload != expected_payload or entry.get("worker_key_id") != payload.get(
            "worker_key_id"
        ):
            raise ValueError("worker authorization payload differs")
        _verify_worker_command(
            payload["worker_command"],
            campaign_dir=campaign_dir,
            plan=plan,
            arm=arm,
            attempt_slot=attempt_slot,
            worker_boot_id=payload["worker_boot_id"],
        )
        paths = _worker_origin_paths(campaign_dir, arm, attempt_slot)
        request = _canonical_artifact(
            paths["request"], role="worker authorization request"
        )
        signed_payload = request.get("signed_payload")
        signed_at = (
            signed_payload.get("signed_at_unix")
            if isinstance(signed_payload, dict)
            else None
        )
        if type(signed_at) is not int:
            raise ValueError("worker authorization request timestamp is invalid")
        expected_request = prepare_role_signature_request(
            trusted_policy,
            role=CAMPAIGN_RUNNER,
            payload=expected_payload,
            signed_at_unix=signed_at,
        )
        attestation = _canonical_artifact(
            paths["attestation"], role="worker authorization attestation"
        )
        signed = verify_legacy_worker_authorization(
            trusted_policy,
            attestation,
            expected_payload=expected_payload,
        )
        if (
            request != expected_request
            or signed != request["signed_payload"]
            or entry.get("request_sha256") != request.get("request_sha256")
            or entry.get("attestation_sha256")
            != _sha256_bytes(canonical_json_bytes(attestation))
        ):
            raise ValueError("worker authorization evidence differs")
        by_position[(arm, attempt_slot)] = {
            "entry": entry,
            "payload": expected_payload,
            "attestation": attestation,
            "request": request,
        }
    return manifest, by_position


def _verify_worker_launch_receipts(
    campaign_dir: Path,
    *,
    plan: CampaignPlan,
    authorization_manifest: Mapping[str, Any],
    used_positions: set[tuple[str, int]],
    authorization_by_position: dict[tuple[str, int], dict[str, Any]],
) -> tuple[set[tuple[str, int]], dict[str, Any]]:
    consumed_positions: set[tuple[str, int]] = set()
    lifecycle_entries: list[dict[str, Any]] = []
    for position, authorization in authorization_by_position.items():
        arm, attempt_slot = position
        payload = authorization["payload"]
        paths = _worker_origin_paths(campaign_dir, arm, attempt_slot)
        if not paths["launch"].exists():
            if paths["launch"].is_symlink() or paths["exit"].exists():
                raise ValueError("worker slot has an exit without a launch")
            continue
        consumed_positions.add(position)
        launch = _canonical_artifact(paths["launch"], role="worker launch receipt")
        if set(launch) != {
            "schema",
            "arm",
            "attempt_slot",
            "worker_boot_id",
            "worker_key_id",
            "worker_command_sha256",
            "authorization_request_sha256",
            "authorization_attestation_sha256",
            "launched_at_unix_ns",
        } or (
            launch.get("schema") != "aura.latent_cortex.worker_launch.v1"
            or launch.get("arm") != arm
            or launch.get("attempt_slot") != attempt_slot
            or launch.get("worker_boot_id") != payload["worker_boot_id"]
            or launch.get("worker_key_id") != payload["worker_key_id"]
            or launch.get("worker_command_sha256")
            != payload["worker_command_sha256"]
            or launch.get("authorization_request_sha256")
            != authorization["request"]["request_sha256"]
            or launch.get("authorization_attestation_sha256")
            != _sha256_bytes(canonical_json_bytes(authorization["attestation"]))
            or type(launch.get("launched_at_unix_ns")) is not int
            or launch["launched_at_unix_ns"] <= 0
        ):
            raise ValueError("worker launch receipt differs")
        exit_receipt = _canonical_artifact(paths["exit"], role="worker exit receipt")
        exit_material = dict(exit_receipt)
        receipt_sha256 = exit_material.pop("receipt_sha256", None)
        if (
            set(exit_receipt)
            != {
                "schema",
                "launch_sha256",
                "outcome",
                "returncode",
                "error_type",
                "exited_at_unix_ns",
                "receipt_sha256",
            }
            or exit_receipt.get("schema") != "aura.latent_cortex.worker_exit.v2"
            or exit_receipt.get("launch_sha256")
            != _sha256_bytes(canonical_json_bytes(launch))
            or type(exit_receipt.get("exited_at_unix_ns")) is not int
            or exit_receipt["exited_at_unix_ns"]
            < launch["launched_at_unix_ns"]
            or receipt_sha256 != _sha256_bytes(canonical_json_bytes(exit_material))
        ):
            raise ValueError("worker exit receipt differs")
        outcome = exit_receipt.get("outcome")
        returncode = exit_receipt.get("returncode")
        error_type = exit_receipt.get("error_type")
        if outcome == "process_exit":
            if type(returncode) is not int or error_type is not None:
                raise ValueError("worker process exit receipt differs")
        elif outcome == "launcher_failure":
            if (
                returncode is not None
                or not isinstance(error_type, str)
                or not error_type
                or position in used_positions
            ):
                raise ValueError("worker launcher failure receipt differs")
        else:
            raise ValueError("worker exit outcome differs")
        lifecycle_material = {
            "arm": arm,
            "attempt_slot": attempt_slot,
            "launch": launch,
            "exit": exit_receipt,
        }
        lifecycle_entries.append(
            {
                **lifecycle_material,
                "entry_sha256": _sha256_bytes(
                    canonical_json_bytes(lifecycle_material)
                ),
            }
        )
    if not used_positions.issubset(consumed_positions):
        raise ValueError("worker result has no consumed launch slot")
    material = {
        "schema": WORKER_LIFECYCLE_MANIFEST_SCHEMA,
        "policy_sha256": authorization_manifest["policy_sha256"],
        "plan_sha256": plan.plan_sha256,
        "worker_authorization_manifest_sha256": authorization_manifest[
            "manifest_sha256"
        ],
        "entry_count": len(lifecycle_entries),
        "entries": lifecycle_entries,
    }
    expected_manifest = {
        **material,
        "manifest_sha256": _sha256_bytes(canonical_json_bytes(material)),
    }
    observed_manifest = _canonical_artifact(
        campaign_dir / WORKER_LIFECYCLE_MANIFEST_FILE,
        role="worker lifecycle manifest",
    )
    if observed_manifest != expected_manifest:
        raise ValueError("worker lifecycle manifest differs")
    return consumed_positions, expected_manifest


def _verify_worker_key_erasure(
    campaign_dir: Path,
    *,
    plan: CampaignPlan,
    authorization_manifest: Mapping[str, Any],
    sealed_outputs: Mapping[str, Any],
) -> dict[str, Any]:
    aggregate = _canonical_artifact(
        campaign_dir / WORKER_KEY_ERASURE_MANIFEST_FILE,
        role="worker key erasure manifest",
    )
    material = dict(aggregate)
    aggregate_sha256 = material.pop("manifest_sha256", None)
    entries = authorization_manifest["entries"]
    if (
        set(aggregate)
        != {
            "schema",
            "policy_sha256",
            "plan_sha256",
            "worker_authorization_manifest_sha256",
            "sealed_output_manifest_sha256",
            "receipt_count",
            "receipts",
            "all_private_paths_absent",
            "copy_exclusion_claimed",
            "manifest_sha256",
        }
        or aggregate.get("schema")
        != "aura.latent_cortex.worker_key_erasure_manifest.v2"
        or aggregate.get("policy_sha256")
        != authorization_manifest["policy_sha256"]
        or aggregate.get("plan_sha256") != plan.plan_sha256
        or aggregate.get("worker_authorization_manifest_sha256")
        != authorization_manifest["manifest_sha256"]
        or aggregate.get("sealed_output_manifest_sha256")
        != sealed_outputs["manifest_sha256"]
        or aggregate.get("receipt_count") != len(entries)
        or not isinstance(aggregate.get("receipts"), list)
        or len(aggregate["receipts"]) != len(entries)
        or aggregate.get("all_private_paths_absent") is not True
        or aggregate.get("copy_exclusion_claimed") is not False
        or aggregate_sha256 != _sha256_bytes(canonical_json_bytes(material))
    ):
        raise ValueError("worker key erasure manifest is invalid")
    for receipt, entry in zip(aggregate["receipts"], entries, strict=True):
        arm = entry["arm"]
        attempt_slot = entry["attempt_slot"]
        paths = _worker_origin_paths(campaign_dir, arm, attempt_slot)
        if paths["private_key"].exists() or paths["private_key"].is_symlink():
            raise ValueError("worker private key remains after erasure")
        intent = _canonical_artifact(
            paths["erasure_intent"], role="worker key erasure intent"
        )
        intent_material = dict(intent)
        intent_sha256 = intent_material.pop("intent_sha256", None)
        if (
            set(intent)
            != {
                "schema",
                "policy_sha256",
                "plan_sha256",
                "worker_authorization_manifest_sha256",
                "sealed_output_manifest_sha256",
                "arm",
                "attempt_slot",
                "worker_boot_id",
                "worker_key_id",
                "method",
                "intent_at_unix_ns",
                "intent_sha256",
            }
            or intent.get("schema")
            != "aura.latent_cortex.worker_key_erasure_intent.v1"
            or intent.get("policy_sha256")
            != authorization_manifest["policy_sha256"]
            or intent.get("plan_sha256") != plan.plan_sha256
            or intent.get("worker_authorization_manifest_sha256")
            != authorization_manifest["manifest_sha256"]
            or intent.get("sealed_output_manifest_sha256")
            != sealed_outputs["manifest_sha256"]
            or intent.get("arm") != arm
            or intent.get("attempt_slot") != attempt_slot
            or intent.get("worker_boot_id") != entry["worker_boot_id"]
            or intent.get("worker_key_id") != entry["worker_key_id"]
            or intent.get("method")
            != "write_ahead_intent_then_unlink_and_parent_directory_fsync"
            or type(intent.get("intent_at_unix_ns")) is not int
            or intent["intent_at_unix_ns"] <= 0
            or intent_sha256
            != _sha256_bytes(canonical_json_bytes(intent_material))
        ):
            raise ValueError("worker key erasure intent differs")
        disk_receipt = _canonical_artifact(
            paths["erasure"], role="worker key erasure receipt"
        )
        receipt_material = dict(disk_receipt)
        receipt_sha256 = receipt_material.pop("receipt_sha256", None)
        if (
            receipt != disk_receipt
            or set(disk_receipt)
            != {
                "schema",
                "intent_sha256",
                "policy_sha256",
                "plan_sha256",
                "sealed_output_manifest_sha256",
                "arm",
                "attempt_slot",
                "worker_boot_id",
                "worker_key_id",
                "absence_observed_at_unix_ns",
                "method",
                "absence_verified",
                "copy_exclusion_claimed",
                "receipt_sha256",
            }
            or disk_receipt.get("schema")
            != "aura.latent_cortex.worker_key_erasure.v2"
            or disk_receipt.get("intent_sha256") != intent["intent_sha256"]
            or disk_receipt.get("policy_sha256")
            != authorization_manifest["policy_sha256"]
            or disk_receipt.get("plan_sha256") != plan.plan_sha256
            or disk_receipt.get("sealed_output_manifest_sha256")
            != sealed_outputs["manifest_sha256"]
            or disk_receipt.get("arm") != arm
            or disk_receipt.get("attempt_slot") != attempt_slot
            or disk_receipt.get("worker_boot_id") != entry["worker_boot_id"]
            or disk_receipt.get("worker_key_id") != entry["worker_key_id"]
            or disk_receipt.get("method")
            != "write_ahead_intent_then_unlink_and_parent_directory_fsync"
            or disk_receipt.get("absence_verified") is not True
            or disk_receipt.get("copy_exclusion_claimed") is not False
            or type(disk_receipt.get("absence_observed_at_unix_ns"))
            is not int
            or disk_receipt["absence_observed_at_unix_ns"]
            < intent["intent_at_unix_ns"]
            or receipt_sha256
            != _sha256_bytes(canonical_json_bytes(receipt_material))
        ):
            raise ValueError("worker key erasure receipt differs")
    return aggregate


def _verify_legacy_worker_origin_evidence(
    campaign_dir: Path,
    *,
    plan: CampaignPlan,
    result_records: tuple[dict[str, Any], ...],
    trusted_policy: Any | None,
) -> tuple[list[str], dict[str, Any]]:
    if plan.to_dict()["metadata"].get("claim_eligible") is not True:
        return [], {"required": False, "verified": False}
    if trusted_policy is None:
        return ["worker origins have no independently trusted policy"], {
            "required": True,
            "verified": False,
        }
    failures: list[str] = []
    try:
        authorization_manifest, authorization_by_position = (
            _verify_worker_authorization_manifest(
                campaign_dir,
                plan=plan,
                trusted_policy=trusted_policy,
            )
        )
        metadata = plan.to_dict()["metadata"]
        arms = metadata["arms"]
        by_arm = {
            arm: sorted(
                (
                    record
                    for record in result_records
                    if record["definition"].get("arm") == arm
                ),
                key=lambda record: int(
                    record["definition"]["execution_ordinal_within_arm"]
                ),
            )
            for arm in arms
        }
        chains: list[dict[str, Any]] = []
        used_positions: set[tuple[str, int]] = set()
        for arm in arms:
            sequence = 0
            previous = ZERO_SHA256
            latest_slot = 0
            for record in by_arm[arm]:
                result = record["result"]
                origin = result.get("worker_origin")
                signed_payload = (
                    origin.get("signed_payload")
                    if isinstance(origin, dict)
                    else None
                )
                attempt_slot = (
                    signed_payload.get("worker_attempt_slot")
                    if isinstance(signed_payload, dict)
                    else None
                )
                if (
                    type(attempt_slot) is not int
                    or attempt_slot < latest_slot
                    or (arm, attempt_slot) not in authorization_by_position
                ):
                    raise ValueError("worker result attempt-slot order differs")
                authorization = authorization_by_position[(arm, attempt_slot)]
                sequence += 1
                verify_legacy_worker_result_origin(
                    trusted_policy,
                    authorization_attestation=authorization["attestation"],
                    expected_authorization_payload=authorization["payload"],
                    result=result,
                    expected_cell_id=record["cell_id"],
                    expected_attempt_id=record["attempt_id"],
                    expected_sequence=sequence,
                    expected_previous_origin_sha256=previous,
                )
                previous = origin["origin_sha256"]
                latest_slot = attempt_slot
                used_positions.add((arm, attempt_slot))
            expected_count = sum(
                plan.cell_definition(cell_id)["arm"] == arm
                for cell_id in plan.cell_ids
            )
            if sequence != expected_count or latest_slot <= 0:
                raise ValueError("worker result chain is incomplete")
            chains.append(
                {
                    "arm": arm,
                    "result_count": sequence,
                    "latest_attempt_slot": latest_slot,
                    "chain_head_sha256": previous,
                }
            )
        consumed_positions, lifecycle_manifest = _verify_worker_launch_receipts(
            campaign_dir,
            plan=plan,
            authorization_manifest=authorization_manifest,
            used_positions=used_positions,
            authorization_by_position=authorization_by_position,
        )
        chain_material = {
            "schema": "aura.latent_cortex.worker_origin_chains.v1",
            "policy_sha256": trusted_policy.policy_sha256,
            "plan_sha256": plan.plan_sha256,
            "chains": chains,
        }
        chain_evidence = {
            **chain_material,
            "chains_sha256": _sha256_bytes(canonical_json_bytes(chain_material)),
        }
        sealed_outputs = _canonical_artifact(
            campaign_dir / SEALED_OUTPUT_MANIFEST_FILE,
            role="sealed output manifest",
        )
        if (
            sealed_outputs.get("worker_origin_chains") != chain_evidence
            or sealed_outputs.get("worker_lifecycle_manifest_sha256")
            != lifecycle_manifest["manifest_sha256"]
        ):
            raise ValueError("sealed worker evidence differs")
        erasure = _verify_worker_key_erasure(
            campaign_dir,
            plan=plan,
            authorization_manifest=authorization_manifest,
            sealed_outputs=sealed_outputs,
        )
    except (OSError, TypeError, ValueError, KeyError) as exc:
        failures.append(f"worker origin validation failed: {exc}")
        return failures, {"required": True, "verified": False}
    custody_failure = (
        "worker execution origin is unproven: producer process held exportable "
        "worker signing keys"
    )
    return [custody_failure], {
        "required": True,
        "verified": False,
        "cryptographic_chain_verified": True,
        "worker_execution_origin_proven": False,
        "worker_key_custody": WORKER_KEY_CUSTODY_PRODUCER_SOFTWARE,
        "authorization_manifest_sha256": authorization_manifest[
            "manifest_sha256"
        ],
        "origin_chains_sha256": chain_evidence["chains_sha256"],
        "worker_lifecycle_manifest_sha256": lifecycle_manifest[
            "manifest_sha256"
        ],
        "key_erasure_manifest_sha256": erasure["manifest_sha256"],
        "used_worker_attempts": len(used_positions),
        "consumed_worker_attempts": len(consumed_positions),
        "private_key_copy_exclusion_proven": False,
    }


def _verify_worker_origin_evidence(
    campaign_dir: Path,
    *,
    plan: CampaignPlan,
    result_records: tuple[dict[str, Any], ...],
    trusted_policy: Any | None,
) -> tuple[list[str], dict[str, Any]]:
    del result_records
    if plan.to_dict()["metadata"].get("claim_eligible") is not True:
        return [], {"required": False, "verified": False}
    if trusted_policy is None:
        return ["worker origins have no independently trusted policy"], {
            "required": True,
            "verified": False,
        }
    try:
        verified = verify_worker_campaign_evidence(
            campaign_dir=campaign_dir,
            plan=plan,
            policy=trusted_policy,
            expected_protocol_sha256=_campaign_protocol_sha256(),
        )
    except (OSError, TypeError, ValueError, KeyError) as exc:
        code = (
            exc.code
            if isinstance(exc, IndependentWorkerCampaignEvidenceError)
            else f"{type(exc).__name__}:{exc}"
        )
        return [f"detached worker evidence validation failed: {code}"], {
            "required": True,
            "verified": False,
            "failure_code": code,
        }
    return [], {
        "required": True,
        "verified": True,
        "cryptographic_chain_verified": True,
        "worker_execution_origin_proven": True,
        "worker_key_custody": WORKER_KEY_CUSTODY_DETACHED_SUPERVISOR,
        "worker_execution_manifest_sha256": verified.manifest_sha256,
        "detached_plan_sha256": verified.detached_plan_sha256,
        "detached_plan_artifact_sha256": verified.detached_plan_artifact_sha256,
        "detached_journal_head_sha256": verified.detached_journal_head_sha256,
        "detached_attempts_artifact_sha256": (
            verified.detached_attempts_artifact_sha256
        ),
        "detached_classification_head_sha256": (
            verified.detached_classification_head_sha256
        ),
        "detached_classifications_sha256": (
            verified.detached_classifications_sha256
        ),
        "imports_sha256": verified.imports_sha256,
        "excluded_attempts_sha256": verified.excluded_attempts_sha256,
        "imported_attempt_count": verified.imported_attempt_count,
        "excluded_attempt_count": verified.excluded_attempt_count,
        "private_key_copy_exclusion_proven": True,
    }


def _verify_sealed_output_reveal(
    campaign_dir: Path,
    *,
    plan: CampaignPlan,
    tasks: tuple[Any, ...],
    result_records: tuple[dict[str, Any], ...],
    trusted_policy: Any | None,
    worker_evidence: Mapping[str, Any],
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
    claim_eligible = plan.to_dict()["metadata"].get("claim_eligible") is True
    if claim_eligible:
        expected_worker_binding = {
            "worker_execution_manifest_sha256": worker_evidence.get(
                "worker_execution_manifest_sha256"
            ),
            "detached_plan_sha256": worker_evidence.get("detached_plan_sha256"),
            "detached_classification_head_sha256": worker_evidence.get(
                "detached_classification_head_sha256"
            ),
            "detached_classifications_sha256": worker_evidence.get(
                "detached_classifications_sha256"
            ),
            "worker_imports_sha256": worker_evidence.get("imports_sha256"),
            "worker_excluded_attempts_sha256": worker_evidence.get(
                "excluded_attempts_sha256"
            ),
        }
        if worker_evidence.get("verified") is not True or any(
            sealed.get(key) != value
            for key, value in expected_worker_binding.items()
        ):
            failures.append("sealed output detached-worker binding differs")
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


def _verify_legacy_final_run_envelope(
    campaign_dir: Path,
    *,
    plan: CampaignPlan,
    trusted_policy: Any | None,
) -> tuple[list[str], dict[str, Any]]:
    if plan.to_dict()["metadata"].get("claim_eligible") is not True:
        return [], {"required": False, "verified": False}
    if trusted_policy is None:
        return ["final run has no independently trusted runner"], {
            "required": True,
            "verified": False,
        }
    envelope = _canonical_artifact(
        campaign_dir / FINAL_RUN_ENVELOPE_FILE,
        role="final run envelope",
    )
    if set(envelope) != {
        "schema",
        "payload",
        "request_sha256",
        "campaign_runner_attestation",
        "envelope_sha256",
    } or envelope.get("schema") != "aura.latent_cortex.final_run_envelope.v3":
        return ["final run envelope schema is invalid"], {
            "required": True,
            "verified": False,
        }
    material = {
        key: envelope[key]
        for key in ("payload", "request_sha256", "campaign_runner_attestation")
    }
    failures: list[str] = []
    if envelope.get("envelope_sha256") != _sha256_bytes(
        canonical_json_bytes(material)
    ):
        failures.append("final run envelope digest is invalid")
    sealed = _canonical_artifact(
        campaign_dir / SEALED_OUTPUT_MANIFEST_FILE,
        role="sealed output manifest",
    )
    reveal = _canonical_artifact(campaign_dir / ANSWER_REVEAL_FILE, role="answer reveal")
    manifest = _canonical_artifact(
        campaign_dir / MANIFEST_FILE,
        role="campaign manifest",
    )
    grade = _canonical_artifact(campaign_dir / GRADE_FILE, role="published grade")
    worker_authorizations = _canonical_artifact(
        campaign_dir / WORKER_AUTHORIZATION_MANIFEST_FILE,
        role="worker authorization manifest",
    )
    worker_lifecycle = _canonical_artifact(
        campaign_dir / WORKER_LIFECYCLE_MANIFEST_FILE,
        role="worker lifecycle manifest",
    )
    worker_key_erasure = _canonical_artifact(
        campaign_dir / WORKER_KEY_ERASURE_MANIFEST_FILE,
        role="worker key erasure manifest",
    )
    expected_payload = {
        "schema": FINAL_RUN_PAYLOAD_SCHEMA,
        "campaign_name": plan.campaign_name,
        "policy_sha256": trusted_policy.policy_sha256,
        "protocol_sha256": _campaign_protocol_sha256(),
        "plan_sha256": plan.plan_sha256,
        "sealed_output_manifest_sha256": sealed["manifest_sha256"],
        "answer_reveal_sha256": reveal["reveal_sha256"],
        "campaign_manifest_sha256": manifest["manifest_sha256"],
        "journal_head_sha256": manifest["journal_head_sha256"],
        "published_grade_sha256": grade["grade_sha256"],
        "worker_authorization_manifest_sha256": worker_authorizations[
            "manifest_sha256"
        ],
        "worker_lifecycle_manifest_sha256": worker_lifecycle[
            "manifest_sha256"
        ],
        "worker_key_erasure_manifest_sha256": worker_key_erasure[
            "manifest_sha256"
        ],
    }
    if envelope.get("payload") != expected_payload:
        failures.append("final run payload differs from independent reconstruction")
    request = _canonical_artifact(
        campaign_dir / FINAL_RUN_REQUEST_FILE,
        role="final run request",
    )
    signed_payload = request.get("signed_payload")
    signed_at = (
        signed_payload.get("signed_at_unix")
        if isinstance(signed_payload, dict)
        else None
    )
    if type(signed_at) is not int:
        failures.append("final run request timestamp is invalid")
    else:
        expected_request = prepare_role_signature_request(
            trusted_policy,
            role=CAMPAIGN_RUNNER,
            payload=expected_payload,
            signed_at_unix=signed_at,
        )
        if request != expected_request:
            failures.append("final run request is not canonical")
        if envelope.get("request_sha256") != request.get("request_sha256"):
            failures.append("final run request digest is not bound")
        try:
            verified = verify_role_attestation(
                trusted_policy,
                envelope.get("campaign_runner_attestation"),
                role=CAMPAIGN_RUNNER,
                expected_payload=expected_payload,
                not_before_unix=signed_at,
            )
        except ValueError:
            failures.append("final run attestation is invalid")
        else:
            if verified != request.get("signed_payload"):
                failures.append("final run attestation differs from request")
    return failures, {
        "required": True,
        "verified": not failures,
        "envelope_sha256": envelope.get("envelope_sha256"),
        "request_sha256": envelope.get("request_sha256"),
    }


def _verify_final_run_envelope(
    campaign_dir: Path,
    *,
    plan: CampaignPlan,
    trusted_policy: Any | None,
    worker_evidence: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    if plan.to_dict()["metadata"].get("claim_eligible") is not True:
        return [], {"required": False, "verified": False}
    if trusted_policy is None:
        return ["final run has no independently trusted runner"], {
            "required": True,
            "verified": False,
        }
    if worker_evidence.get("verified") is not True:
        return ["final run has no verified detached-worker evidence"], {
            "required": True,
            "verified": False,
        }
    envelope = _canonical_artifact(
        campaign_dir / FINAL_RUN_ENVELOPE_FILE,
        role="final run envelope",
    )
    if set(envelope) != {
        "schema",
        "payload",
        "request_sha256",
        "campaign_runner_attestation",
        "envelope_sha256",
    } or envelope.get("schema") != "aura.latent_cortex.final_run_envelope.v4":
        return ["final run envelope schema is invalid"], {
            "required": True,
            "verified": False,
        }
    material = {
        key: envelope[key]
        for key in ("payload", "request_sha256", "campaign_runner_attestation")
    }
    failures: list[str] = []
    if envelope.get("envelope_sha256") != _sha256_bytes(
        canonical_json_bytes(material)
    ):
        failures.append("final run envelope digest is invalid")
    sealed = _canonical_artifact(
        campaign_dir / SEALED_OUTPUT_MANIFEST_FILE,
        role="sealed output manifest",
    )
    reveal = _canonical_artifact(campaign_dir / ANSWER_REVEAL_FILE, role="answer reveal")
    manifest = _canonical_artifact(
        campaign_dir / MANIFEST_FILE,
        role="campaign manifest",
    )
    grade = _canonical_artifact(campaign_dir / GRADE_FILE, role="published grade")
    worker_manifest = _canonical_artifact(
        campaign_dir / "worker_execution_manifest.json",
        role="worker execution manifest",
    )
    expected_payload = {
        "schema": FINAL_RUN_PAYLOAD_SCHEMA,
        "campaign_name": plan.campaign_name,
        "policy_sha256": trusted_policy.policy_sha256,
        "protocol_sha256": _campaign_protocol_sha256(),
        "plan_sha256": plan.plan_sha256,
        "sealed_output_manifest_sha256": sealed["manifest_sha256"],
        "answer_reveal_sha256": reveal["reveal_sha256"],
        "campaign_manifest_sha256": manifest["manifest_sha256"],
        "journal_head_sha256": manifest["journal_head_sha256"],
        "published_grade_sha256": grade["grade_sha256"],
        "worker_execution_manifest_sha256": worker_evidence[
            "worker_execution_manifest_sha256"
        ],
        "detached_plan_sha256": worker_evidence["detached_plan_sha256"],
        "detached_classification_head_sha256": worker_evidence[
            "detached_classification_head_sha256"
        ],
        "detached_classifications_sha256": worker_evidence[
            "detached_classifications_sha256"
        ],
        "worker_imports_sha256": worker_evidence["imports_sha256"],
        "worker_excluded_attempts_sha256": worker_evidence[
            "excluded_attempts_sha256"
        ],
    }
    if worker_manifest.get("manifest_sha256") != worker_evidence.get(
        "worker_execution_manifest_sha256"
    ):
        failures.append("final run worker manifest differs from independent replay")
    if envelope.get("payload") != expected_payload:
        failures.append("final run payload differs from independent reconstruction")
    request = _canonical_artifact(
        campaign_dir / FINAL_RUN_REQUEST_FILE,
        role="final run request",
    )
    signed_payload = request.get("signed_payload")
    signed_at = (
        signed_payload.get("signed_at_unix")
        if isinstance(signed_payload, dict)
        else None
    )
    if type(signed_at) is not int:
        failures.append("final run request timestamp is invalid")
    else:
        expected_request = prepare_role_signature_request(
            trusted_policy,
            role=CAMPAIGN_RUNNER,
            payload=expected_payload,
            signed_at_unix=signed_at,
        )
        if request != expected_request:
            failures.append("final run request is not canonical")
        if envelope.get("request_sha256") != request.get("request_sha256"):
            failures.append("final run request digest is not bound")
        try:
            verified = verify_role_attestation(
                trusted_policy,
                envelope.get("campaign_runner_attestation"),
                role=CAMPAIGN_RUNNER,
                expected_payload=expected_payload,
                not_before_unix=signed_at,
            )
        except ValueError:
            failures.append("final run attestation is invalid")
        else:
            if verified != request.get("signed_payload"):
                failures.append("final run attestation differs from request")
    return failures, {
        "required": True,
        "verified": not failures,
        "envelope_sha256": envelope.get("envelope_sha256"),
        "request_sha256": envelope.get("request_sha256"),
        "worker_execution_manifest_sha256": worker_evidence.get(
            "worker_execution_manifest_sha256"
        ),
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

    plan_path = campaign_dir / PLAN_FILE
    plan_payload = read_stable_bytes(
        plan_path,
        max_bytes=64 * 1024 * 1024,
    )
    plan = CampaignPlan.from_dict(
        _canonical_artifact(plan_path, role="campaign plan")
    )
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
    worker_origin_failures, worker_origin_detail = _verify_worker_origin_evidence(
        campaign_dir,
        plan=plan,
        result_records=result_records,
        trusted_policy=trusted_policy,
    )
    failures.extend(worker_origin_failures)
    detail["worker_origins"] = worker_origin_detail
    try:
        reveal_failures, reveal_detail = _verify_sealed_output_reveal(
            campaign_dir,
            plan=plan,
            tasks=tasks,
            result_records=result_records,
            trusted_policy=trusted_policy,
            worker_evidence=worker_origin_detail,
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
    independent_result = independent_grade_campaign(
        records,
        plan=plan,
        issuer_tasks=tasks,
        trusted_contamination_root_sha256=trusted_root,
        trusted_campaign_policy_sha256=(
            trusted_policy.policy_sha256 if trusted_policy is not None else None
        ),
    )
    if (
        not isinstance(independent_result, dict)
        or set(independent_result)
        != {
            "semantic_grade",
            "semantic_grade_canonical_sha256",
            "implementation_sha256",
        }
    ):
        failures.append("independent kernel returned an invalid result envelope")
        independent_result = {}
    independent_grade = independent_result.get("semantic_grade")
    independent_grade_sha256 = independent_result.get(
        "semantic_grade_canonical_sha256"
    )
    independent_implementation_sha256 = independent_result.get(
        "implementation_sha256"
    )
    if (
        not isinstance(independent_implementation_sha256, str)
        or len(independent_implementation_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in independent_implementation_sha256
        )
    ):
        failures.append("independent kernel implementation identity is invalid")
    production_grade_bytes = canonical_json_bytes(grade)
    production_semantic_grade_sha256 = _sha256_bytes(production_grade_bytes)
    detail["recomputed_verdict"] = grade.get("verdict")
    detail["recomputed_claim_tier"] = grade.get("claim_tier")
    detail["production_semantic_grade_sha256"] = (
        production_semantic_grade_sha256
    )
    detail["independent_semantic_grade_sha256"] = independent_grade_sha256
    detail["independent_implementation_sha256"] = (
        independent_implementation_sha256
    )
    detail["independent_verdict"] = (
        independent_grade.get("verdict")
        if isinstance(independent_grade, dict)
        else None
    )
    detail["independent_claim_tier"] = (
        independent_grade.get("claim_tier")
        if isinstance(independent_grade, dict)
        else None
    )
    if not isinstance(independent_grade, dict):
        failures.append("independent kernel returned no semantic grade tree")
    else:
        independent_grade_bytes = canonical_json_bytes(independent_grade)
        actual_independent_sha256 = _sha256_bytes(independent_grade_bytes)
        if independent_grade_sha256 != actual_independent_sha256:
            failures.append(
                "independent semantic grade hash does not match its complete tree"
            )
        if (
            production_grade_bytes != independent_grade_bytes
            or production_semantic_grade_sha256 != actual_independent_sha256
        ):
            difference = _first_semantic_difference(grade, independent_grade)
            failures.append(
                "production and independent semantic grade trees differ"
                + (f": {difference}" if difference is not None else "")
            )

    # 4. Agreement with the published grade, if one exists.
    grade_path = campaign_dir / GRADE_FILE
    published_grade_sha256: str | None = None
    campaign_manifest_sha256: str | None = None
    if grade_path.exists():
        published = _canonical_artifact(
            grade_path,
            role="published grade",
        )
        published_grade_sha256 = published.get("grade_sha256")
        detail["published_verdict"] = published.get("verdict")
        manifest_path = campaign_dir / MANIFEST_FILE
        if manifest_path.exists():
            manifest = _canonical_artifact(
                manifest_path,
                role="campaign manifest",
            )
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
            if worker_origin_detail.get("required") is True:
                expected_material["worker_execution_manifest_sha256"] = (
                    worker_origin_detail.get("worker_execution_manifest_sha256")
                )
                expected_material["detached_plan_sha256"] = (
                    worker_origin_detail.get("detached_plan_sha256")
                )
                expected_material["detached_classification_head_sha256"] = (
                    worker_origin_detail.get(
                        "detached_classification_head_sha256"
                    )
                )
                expected_material["detached_classifications_sha256"] = (
                    worker_origin_detail.get("detached_classifications_sha256")
                )
                expected_material["worker_imports_sha256"] = (
                    worker_origin_detail.get("imports_sha256")
                )
                expected_material["worker_excluded_attempts_sha256"] = (
                    worker_origin_detail.get("excluded_attempts_sha256")
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

    try:
        final_run_failures, final_run_detail = _verify_final_run_envelope(
            campaign_dir,
            plan=plan,
            trusted_policy=trusted_policy,
            worker_evidence=worker_origin_detail,
        )
    except (OSError, TypeError, ValueError, KeyError) as exc:
        final_run_failures = [f"final run envelope validation failed: {exc}"]
        final_run_detail = {"required": True, "verified": False}
    failures.extend(final_run_failures)
    detail["final_run"] = final_run_detail

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
        elif failures:
            detail["verifier_attestation_request_blocked_by_failures"] = list(
                failures
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
            exact_grade_sha256 = (
                implementation_map.get(
                    "core/brain/llm/latent_cortex/exact_paired_grade.py"
                )
                if isinstance(implementation_map, dict)
                else None
            )
            exact_statistics_sha256 = (
                implementation_map.get(
                    "core/brain/llm/latent_cortex/exact_paired_statistics.py"
                )
                if isinstance(implementation_map, dict)
                else None
            )
            production_semantic_grade_sha256 = detail.get(
                "production_semantic_grade_sha256"
            )
            independent_semantic_grade_sha256 = detail.get(
                "independent_semantic_grade_sha256"
            )
            if not all(
                isinstance(value, str) and len(value) == 64
                for value in (
                    production_grade_sha256,
                    exact_grade_sha256,
                    exact_statistics_sha256,
                    independent_scoring_sha256,
                    production_semantic_grade_sha256,
                    independent_semantic_grade_sha256,
                )
            ):
                failures.append(
                    "final verifier payload lacks pinned semantic or "
                    "implementation identity"
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
                "final_run_envelope_sha256": final_run_detail.get(
                    "envelope_sha256"
                ),
                "worker_execution_manifest_sha256": worker_origin_detail.get(
                    "worker_execution_manifest_sha256"
                ),
                "detached_plan_sha256": worker_origin_detail.get(
                    "detached_plan_sha256"
                ),
                "detached_plan_artifact_sha256": worker_origin_detail.get(
                    "detached_plan_artifact_sha256"
                ),
                "detached_journal_head_sha256": worker_origin_detail.get(
                    "detached_journal_head_sha256"
                ),
                "detached_attempts_artifact_sha256": worker_origin_detail.get(
                    "detached_attempts_artifact_sha256"
                ),
                "detached_classification_head_sha256": worker_origin_detail.get(
                    "detached_classification_head_sha256"
                ),
                "detached_classifications_sha256": worker_origin_detail.get(
                    "detached_classifications_sha256"
                ),
                "worker_imports_sha256": worker_origin_detail.get(
                    "imports_sha256"
                ),
                "worker_excluded_attempts_sha256": worker_origin_detail.get(
                    "excluded_attempts_sha256"
                ),
                "production_protocol_sha256": _campaign_protocol_sha256(),
                "production_semantic_grade_sha256": (
                    production_semantic_grade_sha256
                ),
                "independent_semantic_grade_sha256": (
                    independent_semantic_grade_sha256
                ),
                "production_grade_implementation_sha256": production_grade_sha256,
                "exact_grade_implementation_sha256": exact_grade_sha256,
                "exact_statistics_implementation_sha256": (
                    exact_statistics_sha256
                ),
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
