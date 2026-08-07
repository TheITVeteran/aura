from __future__ import annotations

import base64
import hashlib
import hmac
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.brain.llm.latent_cortex.campaign_journal import (
    CampaignJournal,
    CampaignPlan,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_RUNNER,
    assemble_role_attestation,
    prepare_role_signature_request,
)
from core.brain.llm.latent_cortex.independent_worker_campaign_evidence import (
    IndependentWorkerCampaignEvidenceError,
    verify_worker_campaign_evidence,
)
from core.brain.llm.latent_cortex.worker_attempt_import import (
    LIFECYCLE_ARTIFACT_SCHEMA,
    PAIRED_CAMPAIGN_CELL_TYPE,
)
from core.brain.llm.latent_cortex.worker_origin import compute_allowed_cell_digest
from core.runtime.detached_subprocess_broker import BrokeredProcessResult
from core.runtime.detached_worker_origin import DetachedWorkerOriginAuthority
from tests import test_detached_campaign_evidence as detached_fixture
from tests.test_latent_cortex_paired_campaign_runner import (
    _external_policy_fixture,
)
from tools import run_latent_cortex_paired_campaign as runner
from tools import verify_paired_campaign_evidence as campaign_verifier

ARM = runner.ADAPTER_RLC


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _hashed(body: dict, key: str) -> dict:
    return {**body, key: _sha256(body)}


def _private_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    path.chmod(0o600)


def _campaign_plan(campaign_name: str) -> CampaignPlan:
    return CampaignPlan.build(
        campaign_name,
        [
            {
                "arm": ARM,
                "domain": "mathematics",
                "seed": 101,
                "task_sha256": "d" * 64,
                "execution_ordinal_within_arm": 0,
            },
            {
                "arm": ARM,
                "domain": "coding",
                "seed": 102,
                "task_sha256": "e" * 64,
                "execution_ordinal_within_arm": 1,
            },
        ],
        metadata={
            "claim_eligible": True,
            "arms": [ARM],
            "execution_config": {
                "worker_origin_protocol": (
                    "detached_supervisor_staged_arm_import_v3"
                ),
                "worker_origin_attempt_slots": 1,
            },
            "model_identity": {"model": "sealed"},
            "adapter_identity": {"adapter": "sealed"},
        },
    )


def _detached_plan(
    root: Path,
    *,
    campaign_name: str,
    protocol_sha256: str,
    policy,
    plan: CampaignPlan,
    origin_dir: Path,
) -> dict:
    base = detached_fixture._plan(root)
    base_policy = base["broker_policy"][0]
    allowed_cells = [
        {"cell_id": cell_id, "cell_type": PAIRED_CAMPAIGN_CELL_TYPE}
        for cell_id in plan.cell_ids
    ]
    contract_body = {
        **{
            key: value
            for key, value in base_policy["worker_origin"].items()
            if key != "contract_sha256"
        },
        "campaign_name": campaign_name,
        "protocol_sha256": protocol_sha256,
        "trust_policy_sha256": policy.policy_sha256,
        "artifact_dir": str(origin_dir),
        "arm": ARM,
        "worker_attempt_slot": 1,
        "allowed_cells": allowed_cells,
        "allowed_cell_digest": compute_allowed_cell_digest(allowed_cells),
        "model_identity_sha256": _sha256(
            plan.to_dict()["metadata"]["model_identity"]
        ),
        "adapter_identity_sha256": _sha256(
            plan.to_dict()["metadata"]["adapter_identity"]
        ),
    }
    contract = _hashed(contract_body, "contract_sha256")
    policy_body = {
        key: value for key, value in base_policy.items() if key != "policy_sha256"
    }
    policy_body["worker_origin"] = contract
    broker_policy = _hashed(policy_body, "policy_sha256")
    plan_body = {key: value for key, value in base.items() if key != "plan_sha256"}
    plan_body["broker_policy"] = [broker_policy]
    plan_body["broker_policy_sha256"] = _sha256([broker_policy])
    return _hashed(plan_body, "plan_sha256")


def _signed_broker_result(
    *,
    start: dict,
    broker_policy: dict,
    broker_token: str,
    lifecycle_summary: dict,
) -> tuple[BrokeredProcessResult, dict]:
    body = {
        "schema": detached_fixture.BROKER_RESPONSE_SCHEMA,
        "request_id": start["request_id"],
        "policy_sha256": broker_policy["policy_sha256"],
        "command_sha256": broker_policy["command_sha256"],
        "worker_pid": start["worker_pid"],
        "worker_process_group_id": start["worker_process_group_id"],
        "worker_start_token": start["worker_start_token"],
        "started_at": 104.0,
        "finished_at": 105.0,
        "duration_s": 1.0,
        "returncode": 0,
        "timed_out": False,
        "cleanup_performed": True,
        "lineage_cleanup_count": 0,
        "containment_verified": True,
        "status": "passed",
        "error": None,
        "worker_origin_lifecycle": lifecycle_summary,
    }
    signed = _hashed(body, "receipt_sha256")
    response = {
        **signed,
        "response_hmac_sha256": hmac.new(
            bytes.fromhex(broker_token),
            canonical_json_bytes(signed),
            hashlib.sha256,
        ).hexdigest(),
    }
    result = BrokeredProcessResult(
        returncode=0,
        request_id=response["request_id"],
        policy_sha256=response["policy_sha256"],
        worker_pid=response["worker_pid"],
        worker_process_group_id=response["worker_process_group_id"],
        worker_start_token=response["worker_start_token"],
        started_at=response["started_at"],
        finished_at=response["finished_at"],
        duration_s=response["duration_s"],
        timed_out=False,
        containment_verified=True,
        status="passed",
        error=None,
        worker_origin_lifecycle=lifecycle_summary,
        receipt_sha256=response["receipt_sha256"],
        response_hmac_sha256=response["response_hmac_sha256"],
    )
    return result, response


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    now = int(time.time())
    campaign_name = "independent-worker-campaign"
    protocol_sha256 = runner._campaign_protocol_sha256()
    policy, role_keys, root_pem = _external_policy_fixture(campaign_name, now)
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir(mode=0o700)
    plan = _campaign_plan(campaign_name)
    runner._persist_plan(campaign_dir, plan)
    paths = runner._worker_attempt_paths(campaign_dir, ARM, 1)
    paths["origin_dir"].mkdir(mode=0o700)

    detached_run_dir = tmp_path / "detached"
    detached_run_dir.mkdir(mode=0o700)
    detached_plan = _detached_plan(
        detached_run_dir,
        campaign_name=campaign_name,
        protocol_sha256=protocol_sha256,
        policy=policy,
        plan=plan,
        origin_dir=paths["origin_dir"],
    )
    detached_plan_path = detached_run_dir / "detached_plan.json"
    detached_attempts_path = detached_run_dir / "detached_attempts.jsonl"
    _private_json(detached_plan_path, detached_plan)

    broker_policy = detached_plan["broker_policy"][0]
    allowed_cells = broker_policy["worker_origin"]["allowed_cells"]
    session_id = hashlib.sha256(str(tmp_path).encode()).hexdigest()[:32]
    authority = DetachedWorkerOriginAuthority(
        policy=policy,
        campaign_name=campaign_name,
        protocol_sha256=protocol_sha256,
        detached_plan_sha256=detached_plan["plan_sha256"],
        broker_policy_sha256=broker_policy["policy_sha256"],
        executable_binding_sha256=broker_policy["executable_binding"][
            "binding_sha256"
        ],
        environment_sha256=detached_plan["execution_environment_sha256"],
        sandbox_sha256=_sha256(detached_plan["execution_sandbox"]),
        source_manifest_sha256=broker_policy["execution_manifest"][
            "manifest_sha256"
        ],
        session_id=session_id,
        supervisor_attempt=1,
        arm=ARM,
        worker_attempt_slot=1,
        allowed_cells=allowed_cells,
        model_identity_sha256=broker_policy["worker_origin"][
            "model_identity_sha256"
        ],
        adapter_identity_sha256=broker_policy["worker_origin"][
            "adapter_identity_sha256"
        ],
    )
    signed_at = max(now, policy.document["not_before_unix"])
    authorization_request = authority.request_authorization(
        signed_at_unix=signed_at
    )
    signed_bytes = base64.b64decode(
        authorization_request["signed_payload_b64"], validate=True
    )
    authorization_attestation = assemble_role_attestation(
        policy,
        authorization_request,
        signature_b64=base64.b64encode(
            role_keys[CAMPAIGN_RUNNER].sign(signed_bytes)
        ).decode("ascii"),
        role=CAMPAIGN_RUNNER,
    )
    authority.accept_authorization(
        authorization_attestation,
        now_unix=signed_at,
    )
    authority.start()

    with CampaignJournal(paths["stage"], plan) as stage:
        for ordinal, cell_id in enumerate(plan.cell_ids):
            attempt_id = stage.start_cell(cell_id)
            result = authority.record_result(
                {
                    "cell_id": cell_id,
                    "cell_type": PAIRED_CAMPAIGN_CELL_TYPE,
                    "attempt_id": attempt_id,
                    "origin_session_id": session_id,
                    "arm": ARM,
                    "text": f"answer-{ordinal}",
                }
            )
            stage.record_arm_result(cell_id, attempt_id, result)
            stage.record_verified(cell_id, attempt_id, {"accepted": True})
            stage.commit_cell(cell_id, attempt_id, {"stage": True})
    paths["stage"].chmod(0o600)
    terminal_origin = authority.complete(occurred_at_unix=signed_at + 1)
    lifecycle_body = {
        "schema": LIFECYCLE_ARTIFACT_SCHEMA,
        "broker_policy_sha256": broker_policy["policy_sha256"],
        "authorization_payload": authority.authorization_payload,
        "authorization_request": authorization_request,
        "authorization_attestation": authorization_attestation,
        "event_origin": terminal_origin,
        "completion_error": None,
    }
    lifecycle = {
        **lifecycle_body,
        "artifact_sha256": _sha256(lifecycle_body),
    }
    lifecycle_path = Path(
        detached_fixture._origin_paths(detached_plan, 1)[1]["lifecycle_path"]
    )
    _private_json(lifecycle_path, lifecycle)
    origin_paths = detached_fixture._origin_paths(detached_plan, 1)[1]
    for key, value in (
        ("payload_path", authority.authorization_payload),
        ("request_path", authorization_request),
        ("attestation_path", authorization_attestation),
    ):
        _private_json(Path(origin_paths[key]), value)

    lifecycle_summary = {
        "artifact_path": str(lifecycle_path),
        "artifact_sha256": lifecycle["artifact_sha256"],
        "event_type": "terminal",
        "event_sha256": terminal_origin["event_sha256"],
        "result_count": len(plan.cell_ids),
        "session_id": session_id,
    }
    request_id = "3" * 32
    broker_token = "5" * 64
    start = detached_fixture._start_body(
        detached_plan,
        attempt=1,
        request_id=request_id,
        session_id=session_id,
    )
    start["worker_origin"]["authorization_payload"] = authority.authorization_payload
    start["worker_origin"]["authorization_request_sha256"] = (
        authorization_request["request_sha256"]
    )
    start["worker_origin"]["authorization_attestation_sha256"] = _sha256(
        authorization_attestation
    )
    result, response = _signed_broker_result(
        start=start,
        broker_policy=broker_policy,
        broker_token=broker_token,
        lifecycle_summary=lifecycle_summary,
    )
    bodies = [
        detached_fixture._launch_body(detached_plan, 1),
        detached_fixture._control_body(detached_plan, 1, broker_token),
        detached_fixture._target_body(detached_plan, 1),
        start,
        detached_fixture._terminal_body(detached_plan, start, response),
    ]
    detached_fixture._write_events(detached_run_dir, bodies)
    runner._persist_brokered_worker_result(paths, result)

    policy_path = tmp_path / "policy.json"
    root_path = tmp_path / "root.pem"
    _private_json(policy_path, policy.document)
    root_path.write_bytes(root_pem)
    root_path.chmod(0o600)
    args = SimpleNamespace(
        campaign_dir=str(campaign_dir),
        campaign_name=campaign_name,
        campaign_trust_policy=str(policy_path),
        campaign_trust_root=str(root_path),
        max_infra_attempts=1,
        profile="primary",
    )
    monkeypatch.setattr(runner, "_arms", lambda _args: (ARM,))
    monkeypatch.setenv(runner.DETACHED_RUN_DIR_ENV, str(detached_run_dir))
    monkeypatch.setenv(runner.DETACHED_PLAN_PATH_ENV, str(detached_plan_path))
    monkeypatch.setenv(
        runner.DETACHED_ATTEMPTS_PATH_ENV, str(detached_attempts_path)
    )
    monkeypatch.setenv(
        runner.DETACHED_PLAN_SHA256_ENV, detached_plan["plan_sha256"]
    )
    monkeypatch.setenv(runner.DETACHED_SUPERVISOR_ATTEMPT_ENV, "1")
    manifest = runner._build_worker_execution_manifest(args, plan)
    assert manifest is not None

    bodies.append(detached_fixture._outer_terminal_body(detached_plan, 1))
    detached_fixture._write_events(detached_run_dir, bodies)
    return campaign_dir, plan, policy, protocol_sha256, role_keys


def test_raw_detached_worker_campaign_replays_independently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_dir, plan, policy, protocol_sha256, _role_keys = _fixture(
        tmp_path, monkeypatch
    )

    verified = verify_worker_campaign_evidence(
        campaign_dir=campaign_dir,
        plan=plan,
        policy=policy,
        expected_protocol_sha256=protocol_sha256,
    )

    assert verified.imported_attempt_count == 1
    assert verified.excluded_attempt_count == 0
    assert verified.manifest_sha256 == runner._read_canonical_json_artifact(
        campaign_dir / runner.WORKER_EXECUTION_MANIFEST_FILE,
        role="worker manifest",
    )["manifest_sha256"]

    failures, detail = campaign_verifier._verify_worker_origin_evidence(
        campaign_dir,
        plan=plan,
        result_records=(),
        trusted_policy=policy,
    )
    assert failures == []
    assert detail["verified"] is True
    assert detail["worker_execution_manifest_sha256"] == verified.manifest_sha256


def test_rehashed_worker_manifest_cannot_hide_changed_import_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_dir, plan, policy, protocol_sha256, _role_keys = _fixture(
        tmp_path, monkeypatch
    )
    receipt_path = runner._worker_attempt_paths(campaign_dir, ARM, 1)[
        "import_receipt"
    ]
    receipt = runner._read_canonical_json_artifact(
        receipt_path,
        role="import receipt",
    )
    receipt["final_canonical_head_sha256"] = "f" * 64
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = _sha256(body)
    receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")

    with pytest.raises(IndependentWorkerCampaignEvidenceError) as exc_info:
        verify_worker_campaign_evidence(
            campaign_dir=campaign_dir,
            plan=plan,
            policy=policy,
            expected_protocol_sha256=protocol_sha256,
        )

    assert exc_info.value.code == "worker_import_receipt_binding_invalid"


def test_worker_manifest_symlink_is_rejected_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_dir, plan, policy, protocol_sha256, _role_keys = _fixture(
        tmp_path, monkeypatch
    )
    manifest_path = campaign_dir / runner.WORKER_EXECUTION_MANIFEST_FILE
    original = tmp_path / "moved-worker-manifest.json"
    manifest_path.rename(original)
    manifest_path.symlink_to(original)

    with pytest.raises(IndependentWorkerCampaignEvidenceError) as exc_info:
        verify_worker_campaign_evidence(
            campaign_dir=campaign_dir,
            plan=plan,
            policy=policy,
            expected_protocol_sha256=protocol_sha256,
        )

    assert exc_info.value.code == "worker_execution_manifest_storage_invalid"


def test_unplanned_attempt_artifact_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_dir, plan, policy, protocol_sha256, _role_keys = _fixture(
        tmp_path, monkeypatch
    )
    injected = runner._worker_attempt_paths(campaign_dir, ARM, 1)["root"] / "note.txt"
    injected.write_text("unbound\n", encoding="ascii")

    with pytest.raises(IndependentWorkerCampaignEvidenceError) as exc_info:
        verify_worker_campaign_evidence(
            campaign_dir=campaign_dir,
            plan=plan,
            policy=policy,
            expected_protocol_sha256=protocol_sha256,
        )

    assert exc_info.value.code == "worker_attempt_contains_unplanned_entry"


def test_untrusted_arm_name_is_rejected_before_path_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_dir, plan, policy, protocol_sha256, _role_keys = _fixture(
        tmp_path, monkeypatch
    )
    document = plan.to_dict()
    metadata = document["metadata"]
    metadata["arms"] = ["../../outside"]
    hostile = CampaignPlan.build(
        plan.campaign_name,
        [plan.cell_definition(cell_id) for cell_id in plan.cell_ids],
        metadata=metadata,
    )

    with pytest.raises(IndependentWorkerCampaignEvidenceError) as exc_info:
        verify_worker_campaign_evidence(
            campaign_dir=campaign_dir,
            plan=hostile,
            policy=policy,
            expected_protocol_sha256=protocol_sha256,
        )

    assert exc_info.value.code == "worker_evidence_plan_contract_invalid"


def test_v4_final_run_envelope_binds_independent_worker_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_dir, plan, policy, protocol_sha256, role_keys = _fixture(
        tmp_path, monkeypatch
    )
    worker_failures, worker_detail = (
        campaign_verifier._verify_worker_origin_evidence(
            campaign_dir,
            plan=plan,
            result_records=(),
            trusted_policy=policy,
        )
    )
    assert worker_failures == []
    artifacts = {
        runner.SEALED_OUTPUT_MANIFEST_FILE: {"manifest_sha256": "1" * 64},
        runner.ANSWER_REVEAL_FILE: {"reveal_sha256": "2" * 64},
        runner.MANIFEST_FILE: {
            "manifest_sha256": "3" * 64,
            "journal_head_sha256": "4" * 64,
        },
        runner.GRADE_FILE: {"grade_sha256": "5" * 64},
    }
    for name, document in artifacts.items():
        _private_json(campaign_dir / name, document)
    sequential_evidence = {
        "required": True,
        "verified": True,
        "look_count": 2,
        "certificate_head_sha256": "6" * 64,
        "certificate_chain_sha256": "7" * 64,
        "first_boundary_look": 1,
        "first_boundary_decision": "positive_boundary_crossed",
        "terminal_decision": "terminal_inconclusive",
    }
    expected_payload = {
        "schema": campaign_verifier.FINAL_RUN_PAYLOAD_SCHEMA,
        "campaign_name": plan.campaign_name,
        "policy_sha256": policy.policy_sha256,
        "protocol_sha256": protocol_sha256,
        "plan_sha256": plan.plan_sha256,
        "sealed_output_manifest_sha256": "1" * 64,
        "answer_reveal_sha256": "2" * 64,
        "campaign_manifest_sha256": "3" * 64,
        "journal_head_sha256": "4" * 64,
        "published_grade_sha256": "5" * 64,
        "worker_execution_manifest_sha256": worker_detail[
            "worker_execution_manifest_sha256"
        ],
        "detached_plan_sha256": worker_detail["detached_plan_sha256"],
        "detached_classification_head_sha256": worker_detail[
            "detached_classification_head_sha256"
        ],
        "detached_classifications_sha256": worker_detail[
            "detached_classifications_sha256"
        ],
        "worker_imports_sha256": worker_detail["imports_sha256"],
        "worker_excluded_attempts_sha256": worker_detail[
            "excluded_attempts_sha256"
        ],
        "sequential_look_count": sequential_evidence["look_count"],
        "sequential_certificate_head_sha256": sequential_evidence[
            "certificate_head_sha256"
        ],
        "sequential_certificate_chain_sha256": sequential_evidence[
            "certificate_chain_sha256"
        ],
        "sequential_first_boundary_look": sequential_evidence[
            "first_boundary_look"
        ],
        "sequential_first_boundary_decision": sequential_evidence[
            "first_boundary_decision"
        ],
        "sequential_terminal_decision": sequential_evidence["terminal_decision"],
    }
    signed_at = int(time.time())
    request = prepare_role_signature_request(
        policy,
        role=CAMPAIGN_RUNNER,
        payload=expected_payload,
        signed_at_unix=signed_at,
    )
    signed_bytes = base64.b64decode(request["signed_payload_b64"], validate=True)
    attestation = assemble_role_attestation(
        policy,
        request,
        signature_b64=base64.b64encode(
            role_keys[CAMPAIGN_RUNNER].sign(signed_bytes)
        ).decode("ascii"),
        role=CAMPAIGN_RUNNER,
    )
    _private_json(campaign_dir / runner.FINAL_RUN_REQUEST_FILE, request)
    envelope_material = {
        "payload": expected_payload,
        "request_sha256": request["request_sha256"],
        "campaign_runner_attestation": attestation,
    }
    envelope = {
        "schema": "aura.latent_cortex.final_run_envelope.v4",
        **envelope_material,
        "envelope_sha256": _sha256(envelope_material),
    }
    _private_json(campaign_dir / runner.FINAL_RUN_ENVELOPE_FILE, envelope)

    failures, detail = campaign_verifier._verify_final_run_envelope(
        campaign_dir,
        plan=plan,
        trusted_policy=policy,
        worker_evidence=worker_detail,
        sequential_evidence=sequential_evidence,
    )

    assert failures == []
    assert detail["verified"] is True
    assert detail["worker_execution_manifest_sha256"] == worker_detail[
        "worker_execution_manifest_sha256"
    ]
