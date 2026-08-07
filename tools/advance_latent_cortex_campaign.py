#!/usr/bin/env python
"""Advance externally signed RLC campaigns across post-inference trust phases.

This coordinator is intentionally non-executing.  It validates the campaign
runner's post-seal or post-grade signature request, admits one detached role
signature, and emits exact resume argv.  Operators can launch that packet
through the detached supervisor after deciding that no campaign process is
already active.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    CampaignJournalError,
    CampaignPlan,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.campaign_launch_bundle import (  # noqa: E402
    LAUNCH_PACKET_FILE,
    read_canonical_json,
    sha256_bytes,
)
from core.brain.llm.latent_cortex.campaign_trust import (  # noqa: E402
    CAMPAIGN_RUNNER,
    TASK_ISSUER,
    CampaignTrustError,
    prepare_role_signature_request,
    validate_campaign_trust_policy,
    verify_role_attestation,
)
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402
from tools import prepare_latent_cortex_campaign as preparation  # noqa: E402
from tools import run_latent_cortex_paired_campaign as runner  # noqa: E402

ANSWER_ATTESTATION_FILE = "answer_reveal_attestation.json"
ANSWER_RESUME_PACKET_FILE = "answer_reveal_resume_packet.json"
FINAL_ATTESTATION_FILE = "final_run_attestation.json"
FINAL_RESUME_PACKET_FILE = "final_run_resume_packet.json"
ANSWER_RESUME_SCHEMA = "aura.latent_cortex.answer_reveal_resume_packet.v1"
FINAL_RESUME_SCHEMA = "aura.latent_cortex.final_run_resume_packet.v1"


class CampaignAdvanceError(RuntimeError):
    """Stable fail-closed campaign phase error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise CampaignAdvanceError(code)


def _sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _emit(document: dict[str, Any]) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(document) + b"\n")


def _verified_hash(document: dict[str, Any], *, key: str, role: str) -> dict[str, Any]:
    material = {name: value for name, value in document.items() if name != key}
    if document.get(key) != _sha256(material):
        _fail(f"{role}_hash_invalid")
    return document


def _context(bundle_dir: Path) -> dict[str, Any]:
    bundle = bundle_dir.expanduser().resolve(strict=True)
    manifest, launch_spec = preparation._verify_bundle_artifacts(bundle)
    packet_path = bundle / LAUNCH_PACKET_FILE
    if not packet_path.exists():
        return {
            "bundle_dir": bundle,
            "manifest": manifest,
            "launch_spec": launch_spec,
            "launch_packet": None,
        }
    preparation.inspect_bundle(argparse.Namespace(bundle_dir=bundle))
    packet = read_canonical_json(packet_path, role="prelaunch_launch_packet")
    if packet.get("campaign_dir") != launch_spec.get("campaign_dir"):
        _fail("prelaunch_campaign_directory_mismatch")
    context = {
        "bundle_dir": bundle,
        "manifest": manifest,
        "launch_spec": launch_spec,
        "launch_packet": packet,
    }
    _verify_persisted_phase_packets(context)
    return context


def _runner_values(launch_spec: dict[str, Any]) -> dict[str, str]:
    argv = launch_spec.get("runner_argv")
    if not isinstance(argv, list):
        _fail("launch_runner_argv_invalid")
    return {
        option: preparation._one_option(argv, option)
        for option in (
            "--campaign-name",
            "--campaign-trust-policy",
            "--campaign-trust-root",
        )
    }


def _policy(launch_spec: dict[str, Any], *, observed_at: int | None) -> Any:
    values = _runner_values(launch_spec)
    policy_path = Path(values["--campaign-trust-policy"]).resolve(strict=True)
    root_path = Path(values["--campaign-trust-root"]).resolve(strict=True)
    return validate_campaign_trust_policy(
        read_canonical_json(policy_path, role="campaign_trust_policy"),
        trusted_root_public_key_pem=read_stable_bytes(
            root_path, max_bytes=64 * 1024
        ),
        expected_campaign_name=values["--campaign-name"],
        expected_protocol_sha256=launch_spec["protocol_sha256"],
        now_unix=observed_at,
    )


def _campaign_plan(campaign_dir: Path, launch_spec: dict[str, Any]) -> CampaignPlan:
    document = read_canonical_json(campaign_dir / runner.PLAN_FILE, role="campaign_plan")
    try:
        plan = CampaignPlan.from_dict(document)
    except (CampaignJournalError, TypeError, ValueError) as exc:
        raise CampaignAdvanceError("campaign_plan_invalid") from exc
    values = _runner_values(launch_spec)
    if plan.campaign_name != values["--campaign-name"]:
        _fail("campaign_plan_name_mismatch")
    return plan


def _role_request(
    path: Path,
    *,
    policy: Any,
    role: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = read_canonical_json(path, role=f"{role}_phase_request")
    signed = request.get("signed_payload")
    if not isinstance(signed, dict) or type(signed.get("signed_at_unix")) is not int:
        _fail(f"{role}_phase_request_invalid")
    payload = signed.get("payload")
    if not isinstance(payload, dict):
        _fail(f"{role}_phase_payload_invalid")
    expected = prepare_role_signature_request(
        policy,
        role=role,
        payload=payload,
        signed_at_unix=signed["signed_at_unix"],
    )
    if request != expected:
        _fail(f"{role}_phase_request_mismatch")
    return request, payload


def _artifact(path: Path, *, role: str) -> dict[str, Any]:
    return preparation._artifact_binding(path, role=role)


def _answer_evidence(
    campaign_dir: Path,
    launch_spec: dict[str, Any],
    *,
    policy: Any,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    plan = _campaign_plan(campaign_dir, launch_spec)
    plan_document = plan.to_dict()
    sealed_path = campaign_dir / runner.SEALED_OUTPUT_MANIFEST_FILE
    sealed = _verified_hash(
        read_canonical_json(sealed_path, role="sealed_output_manifest"),
        key="manifest_sha256",
        role="sealed_output_manifest",
    )
    request_path = campaign_dir / runner.ANSWER_REVEAL_REQUEST_FILE
    request, payload = _role_request(
        request_path, policy=policy, role=TASK_ISSUER
    )
    if (
        set(payload)
        != {
            "schema",
            "campaign_name",
            "plan_sha256",
            "sealed_output_manifest_sha256",
            "task_commitment_sha256",
            "answers",
        }
        or payload.get("schema") != runner.ANSWER_REVEAL_PAYLOAD_SCHEMA
        or payload.get("campaign_name") != plan.campaign_name
        or payload.get("plan_sha256") != plan.plan_sha256
        or payload.get("sealed_output_manifest_sha256")
        != sealed["manifest_sha256"]
        or payload.get("task_commitment_sha256")
        != plan_document["metadata"]["task_commitment"]["commitment_sha256"]
        or not isinstance(payload.get("answers"), list)
    ):
        _fail("answer_reveal_payload_binding_invalid")
    public_tasks = plan_document["metadata"]["task_manifest"]["tasks"]
    commitments = {
        task["task_id"]: task["answer_commitment_sha256"] for task in public_tasks
    }
    answers: dict[str, dict[str, Any]] = {}
    for answer in payload["answers"]:
        if (
            not isinstance(answer, dict)
            or set(answer)
            != {"task_id", "answer_commitment_sha256", "answer_payload"}
            or answer.get("task_id") in answers
        ):
            _fail("answer_reveal_set_invalid")
        task_id = answer["task_id"]
        commitment = commitments.get(task_id)
        if (
            commitment is None
            or answer.get("answer_commitment_sha256") != commitment
            or _sha256(answer.get("answer_payload")) != commitment
        ):
            _fail("answer_reveal_commitment_invalid")
        answers[task_id] = answer
    if set(answers) != set(commitments):
        _fail("answer_reveal_set_incomplete")
    evidence = [
        _artifact(campaign_dir / runner.PLAN_FILE, role="campaign_plan"),
        _artifact(sealed_path, role="sealed_output_manifest"),
        _artifact(request_path, role="answer_reveal_request"),
    ]
    return request, payload, evidence


def _validate_attestation(
    path: Path,
    *,
    policy: Any,
    role: str,
    request: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    attestation = read_canonical_json(path, role=f"{role}_phase_attestation")
    signed = verify_role_attestation(
        policy,
        attestation,
        role=role,
        expected_payload=payload,
        not_before_unix=request["signed_payload"]["signed_at_unix"],
    )
    if signed != request["signed_payload"]:
        _fail(f"{role}_phase_attestation_request_mismatch")
    return attestation


def _base_resume_argv(context: dict[str, Any]) -> list[str]:
    packet = context["launch_packet"]
    if not isinstance(packet, dict) or not isinstance(packet.get("argv"), list):
        _fail("prelaunch_packet_required")
    argv = list(packet["argv"])
    forbidden = {
        "--answer-reveal-attestation",
        "--final-run-attestation",
    }
    if any(
        token in forbidden
        or any(token.startswith(f"{option}=") for option in forbidden)
        for token in argv
    ):
        _fail("prelaunch_packet_contains_late_attestation")
    return argv


def _verify_bound_artifact(binding: Any, *, role: str) -> None:
    if (
        not isinstance(binding, dict)
        or set(binding) != {"role", "path", "sha256", "size_bytes"}
        or not isinstance(binding.get("path"), str)
        or not isinstance(binding.get("role"), str)
    ):
        _fail(f"{role}_binding_invalid")
    if _artifact(Path(binding["path"]), role=binding["role"]) != binding:
        _fail(f"{role}_changed")


def _verify_resume_packet(
    context: dict[str, Any],
    *,
    path: Path,
    schema: str,
    phase: str,
    expected_attestations: list[Path],
    expected_argv: list[str],
) -> None:
    packet = read_canonical_json(path, role=phase)
    material = {key: value for key, value in packet.items() if key != "packet_sha256"}
    if (
        packet.get("schema") != schema
        or packet.get("phase") != phase
        or packet.get("packet_sha256") != _sha256(material)
        or packet.get("prelaunch_packet_sha256")
        != context["launch_packet"]["packet_sha256"]
        or packet.get("working_directory") != str(REPO_ROOT.resolve(strict=True))
        or packet.get("campaign_dir") != context["launch_spec"]["campaign_dir"]
        or packet.get("argv") != expected_argv
        or packet.get("argv_sha256") != _sha256(expected_argv)
        or not isinstance(packet.get("attestations"), list)
        or not isinstance(packet.get("campaign_evidence"), list)
    ):
        _fail(f"{phase}_packet_invalid")
    _verify_bound_artifact(packet.get("request"), role=f"{phase}_request")
    for index, binding in enumerate(packet["attestations"]):
        _verify_bound_artifact(binding, role=f"{phase}_attestation_{index}")
    if [binding["path"] for binding in packet["attestations"]] != [
        str(value) for value in expected_attestations
    ]:
        _fail(f"{phase}_attestation_set_invalid")
    for index, binding in enumerate(packet["campaign_evidence"]):
        _verify_bound_artifact(binding, role=f"{phase}_evidence_{index}")


def _verify_persisted_phase_packets(context: dict[str, Any]) -> None:
    bundle = context["bundle_dir"]
    answer_attestation = bundle / ANSWER_ATTESTATION_FILE
    answer_packet = bundle / ANSWER_RESUME_PACKET_FILE
    if answer_packet.exists():
        _verify_resume_packet(
            context,
            path=answer_packet,
            schema=ANSWER_RESUME_SCHEMA,
            phase="ready_for_answer_reveal",
            expected_attestations=[answer_attestation],
            expected_argv=[
                *_base_resume_argv(context),
                "--answer-reveal-attestation",
                str(answer_attestation),
            ],
        )
    final_attestation = bundle / FINAL_ATTESTATION_FILE
    final_packet = bundle / FINAL_RESUME_PACKET_FILE
    if final_packet.exists():
        _verify_resume_packet(
            context,
            path=final_packet,
            schema=FINAL_RESUME_SCHEMA,
            phase="ready_for_final_envelope",
            expected_attestations=[answer_attestation, final_attestation],
            expected_argv=[
                *_base_resume_argv(context),
                "--answer-reveal-attestation",
                str(answer_attestation),
                "--final-run-attestation",
                str(final_attestation),
            ],
        )


def _resume_packet(
    *,
    schema: str,
    phase: str,
    context: dict[str, Any],
    request: Path,
    attestations: list[Path],
    evidence: list[dict[str, Any]],
    argv: list[str],
) -> dict[str, Any]:
    material = {
        "schema": schema,
        "phase": phase,
        "prelaunch_packet_sha256": context["launch_packet"]["packet_sha256"],
        "request": _artifact(request, role=f"{phase}_request"),
        "attestations": [
            _artifact(path, role=f"{phase}_attestation_{index}")
            for index, path in enumerate(attestations)
        ],
        "campaign_evidence": evidence,
        "working_directory": str(REPO_ROOT.resolve(strict=True)),
        "campaign_dir": context["launch_spec"]["campaign_dir"],
        "argv": argv,
        "argv_sha256": _sha256(argv),
    }
    return {**material, "packet_sha256": _sha256(material)}


def _persist_attestation(
    bundle_dir: Path,
    name: str,
    attestation: dict[str, Any],
) -> Path:
    path = bundle_dir / name
    preparation._create_or_verify(path, attestation, role=name.removesuffix(".json"))
    return path


def _persist_packet(bundle_dir: Path, name: str, packet: dict[str, Any]) -> Path:
    path = bundle_dir / name
    preparation._create_or_verify(path, packet, role=name.removesuffix(".json"))
    return path


def _answer_admission(
    context: dict[str, Any],
    attestation_path: Path,
    *,
    observed_at: int,
) -> dict[str, Any]:
    campaign_dir = Path(context["launch_spec"]["campaign_dir"])
    policy = _policy(context["launch_spec"], observed_at=observed_at)
    request, payload, evidence = _answer_evidence(
        campaign_dir, context["launch_spec"], policy=policy
    )
    attestation = _validate_attestation(
        attestation_path,
        policy=policy,
        role=TASK_ISSUER,
        request=request,
        payload=payload,
    )
    persisted = _persist_attestation(
        context["bundle_dir"], ANSWER_ATTESTATION_FILE, attestation
    )
    argv = [
        *_base_resume_argv(context),
        "--answer-reveal-attestation",
        str(persisted),
    ]
    packet = _resume_packet(
        schema=ANSWER_RESUME_SCHEMA,
        phase="ready_for_answer_reveal",
        context=context,
        request=campaign_dir / runner.ANSWER_REVEAL_REQUEST_FILE,
        attestations=[persisted],
        evidence=evidence,
        argv=argv,
    )
    packet_path = _persist_packet(
        context["bundle_dir"], ANSWER_RESUME_PACKET_FILE, packet
    )
    return {
        "schema": "aura.latent_cortex.campaign_phase_result.v1",
        "phase": packet["phase"],
        "request_sha256": request["request_sha256"],
        "packet_sha256": packet["packet_sha256"],
        "packet_path": str(packet_path),
        "argv": argv,
    }


def _verified_answer_reveal(
    campaign_dir: Path,
    *,
    policy: Any,
    launch_spec: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    request, payload, _evidence = _answer_evidence(
        campaign_dir, launch_spec, policy=policy
    )
    reveal = read_canonical_json(
        campaign_dir / runner.ANSWER_REVEAL_FILE, role="answer_reveal"
    )
    reveal_material = {
        "payload": reveal.get("payload"),
        "request_sha256": reveal.get("request_sha256"),
        "task_issuer_attestation": reveal.get("task_issuer_attestation"),
    }
    if (
        set(reveal)
        != {
            "schema",
            "payload",
            "request_sha256",
            "task_issuer_attestation",
            "reveal_sha256",
        }
        or reveal.get("schema") != "aura.latent_cortex.answer_reveal.v1"
        or reveal.get("reveal_sha256") != _sha256(reveal_material)
        or reveal.get("payload") != payload
        or reveal.get("request_sha256") != request["request_sha256"]
        or not isinstance(reveal.get("task_issuer_attestation"), dict)
    ):
        _fail("answer_reveal_envelope_invalid")
    _validate_attestation_document(
        reveal["task_issuer_attestation"],
        policy=policy,
        role=TASK_ISSUER,
        request=request,
        payload=payload,
    )
    return reveal, reveal["task_issuer_attestation"]


def _validate_attestation_document(
    attestation: dict[str, Any],
    *,
    policy: Any,
    role: str,
    request: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    signed = verify_role_attestation(
        policy,
        attestation,
        role=role,
        expected_payload=payload,
        not_before_unix=request["signed_payload"]["signed_at_unix"],
    )
    if signed != request["signed_payload"]:
        _fail(f"{role}_embedded_attestation_request_mismatch")


def _final_evidence(
    campaign_dir: Path,
    launch_spec: dict[str, Any],
    *,
    policy: Any,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    plan = _campaign_plan(campaign_dir, launch_spec)
    reveal, issuer_attestation = _verified_answer_reveal(
        campaign_dir, policy=policy, launch_spec=launch_spec
    )
    sealed = _verified_hash(
        read_canonical_json(
            campaign_dir / runner.SEALED_OUTPUT_MANIFEST_FILE,
            role="sealed_output_manifest",
        ),
        key="manifest_sha256",
        role="sealed_output_manifest",
    )
    campaign_manifest = _verified_hash(
        read_canonical_json(
            campaign_dir / runner.MANIFEST_FILE, role="campaign_manifest"
        ),
        key="manifest_sha256",
        role="campaign_manifest",
    )
    grade = _verified_hash(
        read_canonical_json(campaign_dir / runner.GRADE_FILE, role="published_grade"),
        key="grade_sha256",
        role="published_grade",
    )
    worker = _verified_hash(
        read_canonical_json(
            campaign_dir / runner.WORKER_EXECUTION_MANIFEST_FILE,
            role="worker_execution_manifest",
        ),
        key="manifest_sha256",
        role="worker_execution_manifest",
    )
    request_path = campaign_dir / runner.FINAL_RUN_REQUEST_FILE
    request, payload = _role_request(
        request_path, policy=policy, role=CAMPAIGN_RUNNER
    )
    sequential_binding = runner._sequential_final_evidence_binding(
        campaign_dir,
        plan,
    )
    expected = {
        "schema": runner.FINAL_RUN_PAYLOAD_SCHEMA,
        "campaign_name": plan.campaign_name,
        "policy_sha256": policy.policy_sha256,
        "protocol_sha256": launch_spec["protocol_sha256"],
        "plan_sha256": plan.plan_sha256,
        "sealed_output_manifest_sha256": sealed["manifest_sha256"],
        "answer_reveal_sha256": reveal["reveal_sha256"],
        "campaign_manifest_sha256": campaign_manifest["manifest_sha256"],
        "journal_head_sha256": campaign_manifest["journal_head_sha256"],
        "published_grade_sha256": grade["grade_sha256"],
        "worker_execution_manifest_sha256": worker["manifest_sha256"],
        "detached_plan_sha256": worker["detached_plan_sha256"],
        "detached_classification_head_sha256": worker[
            "detached_classification_head_sha256"
        ],
        "detached_classifications_sha256": worker[
            "detached_classifications_sha256"
        ],
        "worker_imports_sha256": worker["imports_sha256"],
        "worker_excluded_attempts_sha256": worker["excluded_attempts_sha256"],
        **sequential_binding,
    }
    if payload != expected:
        _fail("final_run_payload_binding_invalid")
    evidence_paths = (
        (runner.PLAN_FILE, "campaign_plan"),
        (runner.SEALED_OUTPUT_MANIFEST_FILE, "sealed_output_manifest"),
        (runner.ANSWER_REVEAL_FILE, "answer_reveal"),
        (runner.MANIFEST_FILE, "campaign_manifest"),
        (runner.GRADE_FILE, "published_grade"),
        (runner.WORKER_EXECUTION_MANIFEST_FILE, "worker_execution_manifest"),
        (runner.FINAL_RUN_REQUEST_FILE, "final_run_request"),
    )
    evidence = [
        _artifact(campaign_dir / name, role=role) for name, role in evidence_paths
    ]
    if sequential_binding:
        evidence.extend(
            _artifact(
                campaign_dir
                / runner.SEQUENTIAL_LOOK_DIR
                / f"look-{look:03d}.json",
                role=f"sequential_look_{look}",
            )
            for look in range(1, sequential_binding["sequential_look_count"] + 1)
        )
    return request, payload, issuer_attestation, evidence


def _final_admission(
    context: dict[str, Any],
    attestation_path: Path,
    *,
    observed_at: int,
) -> dict[str, Any]:
    campaign_dir = Path(context["launch_spec"]["campaign_dir"])
    policy = _policy(context["launch_spec"], observed_at=observed_at)
    request, payload, issuer_attestation, evidence = _final_evidence(
        campaign_dir, context["launch_spec"], policy=policy
    )
    issuer_path = _persist_attestation(
        context["bundle_dir"], ANSWER_ATTESTATION_FILE, issuer_attestation
    )
    runner_attestation = _validate_attestation(
        attestation_path,
        policy=policy,
        role=CAMPAIGN_RUNNER,
        request=request,
        payload=payload,
    )
    final_path = _persist_attestation(
        context["bundle_dir"], FINAL_ATTESTATION_FILE, runner_attestation
    )
    argv = [
        *_base_resume_argv(context),
        "--answer-reveal-attestation",
        str(issuer_path),
        "--final-run-attestation",
        str(final_path),
    ]
    packet = _resume_packet(
        schema=FINAL_RESUME_SCHEMA,
        phase="ready_for_final_envelope",
        context=context,
        request=campaign_dir / runner.FINAL_RUN_REQUEST_FILE,
        attestations=[issuer_path, final_path],
        evidence=evidence,
        argv=argv,
    )
    packet_path = _persist_packet(
        context["bundle_dir"], FINAL_RESUME_PACKET_FILE, packet
    )
    return {
        "schema": "aura.latent_cortex.campaign_phase_result.v1",
        "phase": packet["phase"],
        "request_sha256": request["request_sha256"],
        "packet_sha256": packet["packet_sha256"],
        "packet_path": str(packet_path),
        "argv": argv,
    }


def _verify_final_envelope(
    context: dict[str, Any],
    *,
    policy: Any,
) -> dict[str, Any]:
    campaign_dir = Path(context["launch_spec"]["campaign_dir"])
    request, payload, _issuer, _evidence = _final_evidence(
        campaign_dir, context["launch_spec"], policy=policy
    )
    envelope = read_canonical_json(
        campaign_dir / runner.FINAL_RUN_ENVELOPE_FILE,
        role="final_run_envelope",
    )
    attestation = envelope.get("campaign_runner_attestation")
    envelope_material = {
        "payload": envelope.get("payload"),
        "request_sha256": envelope.get("request_sha256"),
        "campaign_runner_attestation": attestation,
    }
    if (
        set(envelope)
        != {
            "schema",
            "payload",
            "request_sha256",
            "campaign_runner_attestation",
            "envelope_sha256",
        }
        or envelope.get("schema") != "aura.latent_cortex.final_run_envelope.v4"
        or envelope.get("envelope_sha256") != _sha256(envelope_material)
        or envelope.get("payload") != payload
        or envelope.get("request_sha256") != request["request_sha256"]
        or not isinstance(attestation, dict)
    ):
        _fail("final_run_envelope_binding_invalid")
    _validate_attestation_document(
        attestation,
        policy=policy,
        role=CAMPAIGN_RUNNER,
        request=request,
        payload=payload,
    )
    return envelope


def _phase(context: dict[str, Any]) -> dict[str, Any]:
    if context["launch_packet"] is None:
        return {"phase": "awaiting_prelaunch_signatures"}
    campaign_dir = Path(context["launch_spec"]["campaign_dir"])
    if not campaign_dir.exists():
        return {"phase": "ready_for_inference"}
    if not campaign_dir.is_dir() or campaign_dir.is_symlink():
        _fail("campaign_directory_storage_invalid")
    policy = _policy(context["launch_spec"], observed_at=None)
    final_envelope = campaign_dir / runner.FINAL_RUN_ENVELOPE_FILE
    final_request = campaign_dir / runner.FINAL_RUN_REQUEST_FILE
    answer_reveal = campaign_dir / runner.ANSWER_REVEAL_FILE
    answer_request = campaign_dir / runner.ANSWER_REVEAL_REQUEST_FILE
    if final_envelope.exists():
        envelope = _verify_final_envelope(context, policy=policy)
        return {
            "phase": "campaign_evidence_sealed",
            "envelope_sha256": envelope["envelope_sha256"],
        }
    if final_request.exists():
        request, _payload, _issuer, _evidence = _final_evidence(
            campaign_dir, context["launch_spec"], policy=policy
        )
        return {
            "phase": "awaiting_final_run_signature",
            "request_path": str(final_request),
            "request_sha256": request.get("request_sha256"),
        }
    if answer_reveal.exists():
        _verified_answer_reveal(
            campaign_dir,
            policy=policy,
            launch_spec=context["launch_spec"],
        )
        return {"phase": "post_reveal_scoring_or_resume"}
    if answer_request.exists():
        request, _payload, _evidence = _answer_evidence(
            campaign_dir, context["launch_spec"], policy=policy
        )
        return {
            "phase": "awaiting_answer_reveal_signature",
            "request_path": str(answer_request),
            "request_sha256": request.get("request_sha256"),
        }
    return {"phase": "inference_in_progress_or_resumable"}


def status(args: argparse.Namespace) -> dict[str, Any]:
    context = _context(args.bundle_dir)
    phase = _phase(context)
    return {
        "schema": "aura.latent_cortex.campaign_phase_status.v1",
        "bundle_dir": str(context["bundle_dir"]),
        "campaign_dir": context["launch_spec"]["campaign_dir"],
        **phase,
    }


def admit(args: argparse.Namespace) -> dict[str, Any]:
    context = _context(args.bundle_dir)
    phase = _phase(context)["phase"]
    if phase == "awaiting_answer_reveal_signature":
        return _answer_admission(
            context, args.attestation, observed_at=args.observed_at
        )
    if phase == "awaiting_final_run_signature":
        return _final_admission(
            context, args.attestation, observed_at=args.observed_at
        )
    _fail(f"campaign_phase_not_signable:{phase}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    status_parser = commands.add_parser("status", help="verify and report the next phase")
    status_parser.add_argument("--bundle-dir", type=Path, required=True)
    admit_parser = commands.add_parser(
        "admit", help="admit the detached signature required by the current phase"
    )
    admit_parser.add_argument("--bundle-dir", type=Path, required=True)
    admit_parser.add_argument("--attestation", type=Path, required=True)
    admit_parser.add_argument("--observed-at", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "admit" and args.observed_at is None:
        args.observed_at = int(time.time())
    try:
        document = status(args) if args.command == "status" else admit(args)
        _emit(document)
        return 0
    except (
        CampaignAdvanceError,
        CampaignJournalError,
        CampaignTrustError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        _emit(
            {
                "schema": "aura.latent_cortex.campaign_advance_error.v1",
                "ok": False,
                "reason": getattr(exc, "code", str(exc)) or type(exc).__name__,
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
