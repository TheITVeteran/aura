"""Contracts for externally authorized RLC cognitive-action interventions."""

from __future__ import annotations

import base64
import hashlib
import json
import time
from copy import deepcopy

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex.action_calibration import (
    ACTION_RESOURCE_DIMENSIONS,
    MIN_PAIR_COUNT,
    action_calibration_issuer_payload,
    action_calibration_starting_state_payload,
    build_action_calibration_design,
    build_action_calibration_plan,
)
from core.brain.llm.latent_cortex.action_intervention import (
    CONTROL_ARM,
    TREATMENT_ARM,
    action_intervention_authority_payload,
    action_intervention_campaign_journal_sha256,
    action_intervention_engine_request_sha256,
    build_action_intervention,
    build_action_intervention_receipt,
    claim_action_intervention_execution,
    consume_action_intervention_once,
    validate_action_intervention,
    validate_action_intervention_objective,
    validate_action_intervention_receipt,
)
from core.brain.llm.latent_cortex.campaign_journal import (
    CampaignJournal,
    CampaignPlan,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_RUNNER,
    CAMPAIGN_TRUST_POLICY_SCHEMA,
    CAMPAIGN_TRUST_ROLES,
    TASK_ISSUER,
    build_role_attestation,
    validate_campaign_trust_policy,
)
from core.brain.llm.latent_cortex.engine import LatentCortexEngine
from core.brain.llm.latent_cortex.epistemic_state import OperationKind
from core.brain.llm.latent_cortex.frontier_tasks import FRONTIER_DOMAINS, generate_task
from core.brain.llm.latent_cortex.value_of_computation import (
    ACTION_TRANSITION_SCHEMA,
    CognitiveStateSignal,
    ValueOfComputationPolicy,
    build_evidence_snapshot,
    transition_reward,
    validate_action_trace,
    validate_action_transition,
)
from core.brain.llm.latent_cortex.worker_handler import budget_from_job, config_from_job

_STATE_COMPONENT_NAMES = (
    "latent_slots_sha256",
    "branch_state_sha256",
    "kv_cache_sha256",
    "evidence_state_sha256",
    "memory_state_sha256",
    "public_action_state_sha256",
    "durable_state_sha256",
    "rng_state_sha256",
)


def _sha(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _private(label: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(
        hashlib.sha256(f"action-intervention:{label}".encode()).digest()
    )


def _public_raw(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _policy_fixture(tmp_path, monkeypatch):
    root = _private("root")
    keys = {role: _private(role) for role in CAMPAIGN_TRUST_ROLES}
    roles = {}
    for role, key in keys.items():
        public = _public_raw(key)
        roles[role] = {
            "signer_id": f"{role}-signer",
            "organization_id": f"{role}-organization",
            "public_key_b64": base64.b64encode(public).decode(),
            "key_id": hashlib.sha256(public).hexdigest(),
            "implementation_sha256": hashlib.sha256(f"{role}:implementation".encode()).hexdigest(),
            "release_sha256": hashlib.sha256(f"{role}:release".encode()).hexdigest(),
            "custody_class": "external_service",
            "custody_evidence_sha256": hashlib.sha256(f"{role}:custody".encode()).hexdigest(),
        }
    now = int(time.time())
    protocol_sha256 = hashlib.sha256(b"action-intervention-protocol").hexdigest()
    body = {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": "action-intervention-test",
        "policy_revision": 1,
        "campaign_name": "action-intervention-test",
        "protocol_sha256": protocol_sha256,
        "previous_policy_sha256": None,
        "revoked_key_ids": [],
        "issued_at_unix": now - 20,
        "not_before_unix": now - 10,
        "expires_at_unix": now + 3600,
        "roles": roles,
    }
    root_raw = _public_raw(root)
    signed = canonical_json_bytes(body)
    document = {
        **body,
        "root_signature": {
            "algorithm": "Ed25519",
            "key_id": hashlib.sha256(root_raw).hexdigest(),
            "signature_b64": base64.b64encode(root.sign(signed)).decode(),
            "signed_payload_sha256": hashlib.sha256(signed).hexdigest(),
        },
    }
    root_pem = root.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    root_path = tmp_path / "action-intervention-root.pem"
    root_path.write_bytes(root_pem)
    policy_path = tmp_path / "action-intervention-policy.json"
    policy_path.write_bytes(canonical_json_bytes(document))
    monkeypatch.setenv("AURA_RLC_ACTION_CALIBRATION_TRUST_ROOT", str(root_path))
    monkeypatch.setenv("AURA_RLC_ACTION_CALIBRATION_POLICY", str(policy_path))
    monkeypatch.setenv(
        "AURA_RLC_ACTION_CALIBRATION_REPLAY_LEDGER",
        str(tmp_path / "action-intervention-replay.jsonl"),
    )
    policy = validate_campaign_trust_policy(
        document,
        trusted_root_public_key_pem=root_pem,
        expected_campaign_name=body["campaign_name"],
        now_unix=now,
    )
    return policy, keys, now


def _intervention(
    tmp_path,
    monkeypatch,
    *,
    arm=TREATMENT_ARM,
    action="search_memory",
    request_payload_sha256=None,
    attempt_number=1,
    engine_config=None,
    engine_budget=None,
    action_policy_evidence=None,
):
    policy, keys, now = _policy_fixture(tmp_path, monkeypatch)
    tasks_by_action = {
        operation: tuple(
            generate_task(
                FRONTIER_DOMAINS[(operation_ordinal + task_ordinal) % 2],
                seed=90_000 + operation_ordinal * 100 + task_ordinal,
                difficulty=2,
            )
            for task_ordinal in range(8)
        )
        for operation_ordinal, operation in enumerate(OperationKind)
    }
    execution_config = {
        "worker_task_material": "public_manifest_only",
        "answer_reveal_protocol": "sealed_outputs_then_issuer_reveal_v1",
        "answer_blind_nonce_policy": "external_issuer_csprng_256",
        "answer_blind_nonce_disclosure": "post_seal_answer_reveal",
        "answer_blind_nonce_count": MIN_PAIR_COUNT,
        "answer_blind_nonce_min_entropy_bits": 256,
        "generation_seed_policy": "external_issuer_uniform_63bit",
        "generation_seed_count": MIN_PAIR_COUNT,
        "generation_seed_min_entropy_bits": 60,
        "task_assignment_policy": ("external_issuer_stratified_random_without_replacement_v1"),
        "task_assignment_seed_sha256": "2" * 64,
        "action_cost_budget_estimated_flops": 10**12,
        "action_resource_caps": {
            name: 100 if name == "host_scalar_ops" else 10**12
            for name in ACTION_RESOURCE_DIMENSIONS
        },
        "continuation_policy_sha256": "a" * 64,
        "budget_policy_sha256": "b" * 64,
        "rng_root_sha256": "c" * 64,
        "instrumentation_sha256": "d" * 64,
        "execute_fixture_policy_sha256": "e" * 64,
        "execute_calibration_effect_class": "deterministic_sandbox",
    }
    model_identity = {
        "model_path": "/sealed/resident-32b",
        "checkpoint_sha256": "1" * 64,
        "runtime_bundle_sha256": "3" * 64,
        "logical_parameter_count": 32_763_876_352,
    }
    campaign_trust = {
        "prelaunch_verified": True,
        "externally_custodied": True,
        "policy_sha256": policy.policy_sha256,
    }
    campaign_design = build_action_calibration_design(
        policy.document["campaign_name"],
        tasks_by_action,
        model_identity=model_identity,
        execution_config=execution_config,
        calibration_bucket="b",
        campaign_trust=campaign_trust,
        claim_eligible=False,
    )
    starting_states = {}
    for operation, tasks in tasks_by_action.items():
        for task in tasks:
            components = {
                name: hashlib.sha256(
                    f"{task.task_id}:{operation.value}:{name}".encode()
                ).hexdigest()
                for name in _STATE_COMPONENT_NAMES
            }
            payload = action_calibration_starting_state_payload(
                campaign_name=policy.document["campaign_name"],
                action=operation,
                task=task.public,
                model_identity=model_identity,
                execution_config=execution_config,
                calibration_bucket="b",
                capture_id=f"capture:{task.task_id}",
                captured_at_unix=now,
                bucket_classifier_sha256="7" * 64,
                bucket_evidence_sha256=hashlib.sha256(f"{task.task_id}:b".encode()).hexdigest(),
                state_component_sha256=components,
                campaign_design_sha256=campaign_design[
                    "campaign_design_sha256"
                ],
            )
            starting_states[task.task_id] = {
                **payload,
                "capture_attestation": build_role_attestation(
                    policy,
                    role=CAMPAIGN_RUNNER,
                    payload=payload,
                    signed_at_unix=now,
                    private_key=keys[CAMPAIGN_RUNNER],
                ),
            }
    plan = build_action_calibration_plan(
        policy.document["campaign_name"],
        tasks_by_action,
        model_identity=model_identity,
        execution_config=execution_config,
        calibration_bucket="b",
        campaign_design=campaign_design,
        starting_state_receipts=starting_states,
        campaign_trust=campaign_trust,
        claim_eligible=False,
    )
    target_action = OperationKind(action).value
    cell_id = next(
        candidate
        for candidate in plan.cell_ids
        if plan.cell_definition(candidate)["action"] == target_action
        and plan.cell_definition(candidate)["arm"] == arm
    )
    definition = plan.cell_definition(cell_id)
    components = {name: definition["starting_state"][name] for name in _STATE_COMPONENT_NAMES}
    journal_path = tmp_path / f"campaign-{arm}-{target_action}-{attempt_number}.jsonl"
    with CampaignJournal(journal_path, plan) as journal:
        attempt_id = journal.start_cell(cell_id)
        if attempt_number == 2:
            journal.fail_cell(
                cell_id,
                attempt_id,
                reason="injected_pre_execution_failure",
                details={"execution_claimed": False},
            )
            attempt_id = journal.start_cell(cell_id)
        elif attempt_number != 1:
            raise ValueError("test fixture supports attempts one and two")
        snapshot = journal.resume()
    journal_prefix = [
        json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()
    ]
    monkeypatch.setenv("AURA_RLC_ACTION_CALIBRATION_JOURNAL", str(journal_path))
    task_manifest = plan.to_dict()["metadata"]["task_manifest"]["tasks"]
    task = next(row for row in task_manifest if row["task_id"] == definition["task_id"])
    prompt = task["prompt"]
    evidence = action_policy_evidence or build_evidence_snapshot(bucket="b", cells={})
    authority = action_intervention_authority_payload(
        campaign_name=policy.document["campaign_name"],
        campaign_plan_sha256=plan.plan_sha256,
        campaign_protocol_sha256=policy.document["protocol_sha256"],
        policy_sha256=policy.policy_sha256,
        policy_revision=policy.document["policy_revision"],
        cell_id=cell_id,
        definition_sha256=_sha(definition),
        pair_id=definition["pair_id"],
        task_id=definition["task_id"],
        task_payload_sha256=definition["task_payload_sha256"],
        starting_state_sha256=definition["starting_state_sha256"],
        starting_state_components=components,
        expected_pre_state_sha256=_sha({name: components[name] for name in sorted(components)}),
        expected_pre_kv_sha256=components["kv_cache_sha256"],
        action=target_action,
        arm=arm,
        execution_ordinal=definition["execution_ordinal"],
        attempt_number=attempt_number,
        attempt_id=attempt_id,
        campaign_journal_path_sha256=action_intervention_campaign_journal_sha256(journal_path),
        journal_head_sha256=snapshot.journal_head_sha256,
        journal_event_count=len(journal_prefix),
        request_payload_sha256=request_payload_sha256 or "e" * 64,
        engine_request_sha256=action_intervention_engine_request_sha256(
            prompt=prompt,
            domain="general",
            config=config_from_job(engine_config),
            budget=budget_from_job(engine_budget),
            cognitive_context=[],
            action_policy_evidence=evidence,
            external_execution_offer=None,
            verifier_present=False,
        ),
        task_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
    )
    issuer_attestation = build_role_attestation(
        policy,
        role=TASK_ISSUER,
        payload=action_calibration_issuer_payload(plan),
        signed_at_unix=now,
        private_key=keys[TASK_ISSUER],
    )
    return build_action_intervention(
        policy=policy,
        runner_private_key=keys[CAMPAIGN_RUNNER],
        signed_at_unix=now,
        authority_payload=authority,
        campaign_plan=plan,
        campaign_journal_prefix=journal_prefix,
        task_issuer_attestation=issuer_attestation,
    )


def _state() -> CognitiveStateSignal:
    return CognitiveStateSignal(
        step_index=0,
        max_steps=4,
        neural_steps=1,
        min_neural_steps=1,
        active_branches=1,
        total_branches=1,
        residual=0.4,
        residual_delta=0.0,
        verifier_score=None,
        verifier_delta=None,
        disagreement=0.0,
        uncertainty=0.5,
        budget_remaining_fraction=0.8,
        has_memory=False,
        has_evidence=False,
        has_verifier=False,
        has_savepoint=True,
        can_execute=False,
        answer_verified=False,
        irreducible_uncertainty=False,
    )


def _task_prompt(intervention) -> str:
    authority = intervention["authority_payload"]
    tasks = intervention["campaign_plan"]["metadata"]["task_manifest"]["tasks"]
    return next(row["prompt"] for row in tasks if row["task_id"] == authority["task_id"])


def _execution_claim(intervention):
    consumption = consume_action_intervention_once(intervention)
    return consumption, claim_action_intervention_execution(intervention, consumption)


def _forced_row(intervention):
    snapshot = build_evidence_snapshot(bucket="b", cells={})
    state = _state()
    decision = ValueOfComputationPolicy(snapshot).choose_forced(
        state,
        executors=(OperationKind.SEARCH_MEMORY, OperationKind.DECOMPOSE),
        action=OperationKind.SEARCH_MEMORY,
    )
    transition = {
        "schema": ACTION_TRANSITION_SCHEMA,
        "bucket": "b",
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "decision_sha256": decision["decision_sha256"],
        "step_index": 0,
        "action": OperationKind.SEARCH_MEMORY.value,
        "mode": "campaign_forced",
        "outcome": "memory_unavailable",
        "checked": False,
        "metrics": transition_reward(
            verified_delta=0.0,
            information_gain=0.0,
            diversity_gain=0.0,
            unsupported_confidence=0.0,
            cost=0.0,
        ),
    }
    row = {
        "decision": decision,
        "transition": transition,
        "state_signal": state.to_dict(),
        "state_before": {
            "residual": 0.4,
            "disagreement": 0.0,
            "verifier_score": None,
            "budget_remaining_fraction": 0.8,
        },
        "state_after": {
            "residual": 0.4,
            "disagreement": 0.0,
            "verifier_score": None,
            "observed_verifier_score": None,
        },
        "affected_branches": 0,
        "verification": {
            "target_branch": None,
            "observation": {},
            "decision": "not_run",
            "restored": False,
        },
    }
    return snapshot, row


def test_signed_intervention_admits_and_tamper_fails_after_rehash(tmp_path, monkeypatch):
    intervention = _intervention(tmp_path, monkeypatch)
    assert (
        validate_action_intervention(
            intervention,
            require_current_policy=True,
        )
        == intervention
    )
    evidence = build_evidence_snapshot(bucket="b", cells={})
    engine_request_sha256 = action_intervention_engine_request_sha256(
        prompt=_task_prompt(intervention),
        domain="general",
        config=config_from_job(None),
        budget=budget_from_job(None),
        cognitive_context=[],
        action_policy_evidence=evidence,
        external_execution_offer=None,
        verifier_present=False,
    )
    assert engine_request_sha256 == intervention["authority_payload"]["engine_request_sha256"]
    assert engine_request_sha256 != action_intervention_engine_request_sha256(
        prompt=_task_prompt(intervention),
        domain="general",
        config=config_from_job({"max_steps": 9}),
        budget=budget_from_job(None),
        cognitive_context=[],
        action_policy_evidence=evidence,
        external_execution_offer=None,
        verifier_present=False,
    )
    assert validate_action_intervention_objective(
        intervention,
        prompt=_task_prompt(intervention),
    ) == _task_prompt(intervention)
    with pytest.raises(ValueError, match="differs from preregistration"):
        validate_action_intervention_objective(
            intervention,
            prompt="a different task",
        )
    with pytest.raises(ValueError, match="prompt-only"):
        validate_action_intervention_objective(
            intervention,
            prompt=_task_prompt(intervention),
            messages=[{"role": "user", "content": _task_prompt(intervention)}],
        )

    tampered = deepcopy(intervention)
    tampered["authority_payload"]["task_id"] = "task-relabelled"
    body = {name: value for name, value in tampered.items() if name != "intervention_sha256"}
    tampered["intervention_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="trust admission failed"):
        validate_action_intervention(tampered, require_current_policy=True)


def test_current_policy_pin_and_durable_attempt_replay_are_fail_closed(
    tmp_path,
    monkeypatch,
):
    intervention = _intervention(tmp_path, monkeypatch)
    first = consume_action_intervention_once(intervention)
    assert first["attempt_id"] == intervention["authority_payload"]["attempt_id"]
    claim = claim_action_intervention_execution(intervention, first)
    assert claim["consumption_event_sha256"] == first["event_sha256"]
    with pytest.raises(ValueError, match="already claimed"):
        claim_action_intervention_execution(intervention, first)
    with pytest.raises(ValueError, match="already consumed"):
        consume_action_intervention_once(intervention)

    policy_path = tmp_path / "action-intervention-policy.json"
    policy_path.write_text('{"superseded":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="trust admission failed"):
        validate_action_intervention(
            intervention,
            require_current_policy=True,
        )


def test_journal_bound_retry_requires_latest_started_attempt(tmp_path, monkeypatch):
    intervention = _intervention(
        tmp_path,
        monkeypatch,
        attempt_number=2,
    )
    authority = intervention["authority_payload"]
    assert authority["attempt_number"] == 2
    assert intervention["campaign_journal_prefix"][-1]["event"] == "STARTED"
    assert intervention["campaign_journal_prefix"][-1]["attempt_id"] == authority["attempt_id"]
    assert (
        validate_action_intervention(
            intervention,
            require_current_policy=True,
        )
        == intervention
    )

    tampered = deepcopy(intervention)
    tampered["campaign_journal_prefix"] = tampered["campaign_journal_prefix"][:-1]
    body = {name: value for name, value in tampered.items() if name != "intervention_sha256"}
    tampered["intervention_sha256"] = _sha(body)
    with pytest.raises(ValueError, match="trust admission failed"):
        validate_action_intervention(tampered, require_current_policy=True)


def test_canonical_journal_advancement_supersedes_delayed_signed_attempt(
    tmp_path,
    monkeypatch,
):
    intervention = _intervention(tmp_path, monkeypatch)
    authority = intervention["authority_payload"]
    journal_path = tmp_path / "campaign-forced_action-search_memory-1.jsonl"
    plan = CampaignPlan.from_dict(intervention["campaign_plan"])

    with CampaignJournal(journal_path, plan) as journal:
        retry_attempt = journal.start_cell(authority["cell_id"])

    assert retry_attempt != authority["attempt_id"]
    with pytest.raises(ValueError, match="canonical journal claim failed"):
        consume_action_intervention_once(intervention)


def test_forced_action_bypasses_state_feasibility_but_not_executor_inventory(
    tmp_path,
    monkeypatch,
):
    intervention = _intervention(tmp_path, monkeypatch)
    snapshot, row = _forced_row(intervention)
    validated = validate_action_trace(
        [row],
        evidence_snapshot=snapshot,
        executors=(OperationKind.SEARCH_MEMORY, OperationKind.DECOMPOSE),
        action_intervention=intervention,
    )
    assert validated["selected_actions"] == [OperationKind.SEARCH_MEMORY.value]
    with pytest.raises(ValueError, match="mode is unsupported"):
        validate_action_transition(
            row["transition"],
            require_checked=False,
        )

    with pytest.raises(ValueError, match="no resident executor"):
        ValueOfComputationPolicy(snapshot).choose_forced(
            _state(),
            executors=(OperationKind.DECOMPOSE,),
            action=OperationKind.SEARCH_MEMORY,
        )
    with pytest.raises(ValueError, match="no resident executor"):
        validate_action_trace(
            [row],
            evidence_snapshot=snapshot,
            executors=(OperationKind.DECOMPOSE,),
            action_intervention=intervention,
        )
    bucket_tamper = deepcopy(row)
    bucket_tamper["transition"]["bucket"] = "other"
    with pytest.raises(ValueError, match="differs from decision"):
        validate_action_trace(
            [bucket_tamper],
            evidence_snapshot=snapshot,
            executors=(OperationKind.SEARCH_MEMORY, OperationKind.DECOMPOSE),
            action_intervention=intervention,
        )


def test_treatment_and_control_receipts_enforce_exact_occurrence_and_state(
    tmp_path,
    monkeypatch,
):
    treatment = _intervention(tmp_path, monkeypatch)
    _treatment_consumption, treatment_claim = _execution_claim(treatment)
    _snapshot, row = _forced_row(treatment)
    treatment_pre = treatment["authority_payload"]["starting_state_components"]
    treatment_post = {
        **treatment_pre,
        "branch_state_sha256": "f" * 64,
        "kv_cache_sha256": "0" * 64,
    }
    treatment_receipt = build_action_intervention_receipt(
        intervention=treatment,
        execution_claim=treatment_claim,
        pre_state_components=treatment_pre,
        post_state_components=treatment_post,
        pre_state_sha256=_sha(treatment_pre),
        pre_kv_sha256=treatment_pre["kv_cache_sha256"],
        post_state_sha256=_sha(treatment_post),
        post_kv_sha256=treatment_post["kv_cache_sha256"],
        decision_sha256=row["decision"]["decision_sha256"],
        cognitive_action_trace=[row],
    )
    assert treatment_receipt["selected_action_occurrences"] == 1
    assert (
        validate_action_intervention_receipt(
            treatment_receipt,
            intervention=treatment,
            cognitive_action_trace=[row],
        )
        == treatment_receipt
    )
    with pytest.raises(ValueError, match="trace differs"):
        build_action_intervention_receipt(
            intervention=treatment,
            execution_claim=treatment_claim,
            pre_state_components=treatment_pre,
            post_state_components=treatment_post,
            pre_state_sha256=_sha(treatment_pre),
            pre_kv_sha256=treatment_pre["kv_cache_sha256"],
            post_state_sha256=_sha(treatment_post),
            post_kv_sha256=treatment_post["kv_cache_sha256"],
            decision_sha256=row["decision"]["decision_sha256"],
            cognitive_action_trace=[row, row],
        )

    control = _intervention(tmp_path, monkeypatch, arm=CONTROL_ARM)
    _control_consumption, control_claim = _execution_claim(control)
    control_state = control["authority_payload"]["starting_state_components"]
    control_post = {
        **control_state,
        "public_action_state_sha256": "f" * 64,
    }
    control_receipt = build_action_intervention_receipt(
        intervention=control,
        execution_claim=control_claim,
        pre_state_components=control_state,
        post_state_components=control_post,
        pre_state_sha256=_sha(control_state),
        pre_kv_sha256=control_state["kv_cache_sha256"],
        post_state_sha256=_sha(control_post),
        post_kv_sha256=control_post["kv_cache_sha256"],
        decision_sha256="",
        cognitive_action_trace=[],
    )
    assert control_receipt["selected_action"] is None
    assert (
        validate_action_trace(
            [],
            evidence_snapshot=build_evidence_snapshot(bucket="b", cells={}),
            executors=(OperationKind.SEARCH_MEMORY, OperationKind.DECOMPOSE),
            action_intervention=control,
        )["selected_actions"]
        == []
    )
    with pytest.raises(ValueError, match="state transition differs"):
        build_action_intervention_receipt(
            intervention=control,
            execution_claim=control_claim,
            pre_state_components=control_state,
            post_state_components={
                **control_post,
                "branch_state_sha256": "f" * 64,
            },
            pre_state_sha256=_sha(control_state),
            pre_kv_sha256=control_state["kv_cache_sha256"],
            post_state_sha256="6" * 64,
            post_kv_sha256=control_post["kv_cache_sha256"],
            decision_sha256="",
            cognitive_action_trace=[],
        )


@pytest.mark.parametrize("arm", [TREATMENT_ARM, CONTROL_ARM])
def test_tiny_resident_episode_executes_signed_intervention(
    tmp_path,
    monkeypatch,
    arm,
):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.qwen2 import Model, ModelArgs

    raw_config = {
        "n_slots": 2,
        "n_branches": 1,
        "max_steps": 2,
        "decode_max_tokens": 2,
        "allow_vanilla_fallback": False,
    }
    evidence = build_evidence_snapshot(bucket="b", cells={})
    intervention = _intervention(
        tmp_path,
        monkeypatch,
        arm=arm,
        action=OperationKind.BLIND_RESOLVE.value,
        engine_config=raw_config,
        action_policy_evidence=evidence,
    )
    identity_calls = 0
    pre_components = intervention["authority_payload"]["starting_state_components"]
    post_components = {**pre_components, "public_action_state_sha256": "f" * 64}
    if arm == TREATMENT_ARM:
        post_components["branch_state_sha256"] = "f" * 64
    measured_components = []
    original_identity = LatentCortexEngine._action_intervention_state_components.__func__

    def intervention_identity(cls, **kwargs):
        nonlocal identity_calls
        measured_components.append(original_identity(cls, **kwargs))
        identity_calls += 1
        return pre_components if identity_calls == 1 else post_components

    monkeypatch.setattr(
        LatentCortexEngine,
        "_action_intervention_state_components",
        classmethod(intervention_identity),
    )
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=32,
        num_hidden_layers=8,
        intermediate_size=64,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())

    class Tokenizer:
        eos_token_id = 0

        @staticmethod
        def encode(text):
            return [ord(character) % 64 for character in text][:8]

        @staticmethod
        def decode(token_ids):
            return " ".join(str(token) for token in token_ids)

    config = config_from_job(raw_config)
    engine = LatentCortexEngine(model, Tokenizer(), config, model_path="")
    consumption = consume_action_intervention_once(intervention)
    result = engine.reason(
        prompt=_task_prompt(intervention),
        action_policy_evidence=evidence,
        action_intervention=intervention,
        action_intervention_consumption=consumption,
    )

    assert result.ok is True, result.reason
    calibration = result.receipt.value_of_computation["calibration_intervention"]
    if arm == TREATMENT_ARM:
        assert calibration["selection_mode"] == "campaign_forced"
        assert calibration["selected_action"] == OperationKind.BLIND_RESOLVE.value
        assert calibration["selected_action_occurrences"] == 1
        assert result.receipt.cognitive_action_trace[0]["decision"]["mode"] == ("campaign_forced")
    else:
        assert calibration["selection_mode"] == "matched_no_action_control"
        assert calibration["selected_action"] is None
        assert calibration["selected_action_occurrences"] == 0
        assert len(result.receipt.cognitive_action_trace) == 1
        control_followup = result.receipt.cognitive_action_trace[0]
        assert control_followup["transition"]["step_index"] == 1
        assert control_followup["state_signal"]["omitted_action_count"] == 1
        assert control_followup["decision"]["mode"] != "campaign_forced"
        assert control_followup["decision"]["action"] != OperationKind.BLIND_RESOLVE.value
    assert len(measured_components) == 2
    assert all(set(components) == set(_STATE_COMPONENT_NAMES) for components in measured_components)
    if arm == TREATMENT_ARM:
        assert (
            measured_components[0]["branch_state_sha256"]
            != measured_components[1]["branch_state_sha256"]
        )
    else:
        assert (
            measured_components[0]["branch_state_sha256"]
            == measured_components[1]["branch_state_sha256"]
        )
    assert (
        measured_components[0]["public_action_state_sha256"]
        != measured_components[1]["public_action_state_sha256"]
    )
    executors = tuple(
        OperationKind(item) for item in result.receipt.value_of_computation["executors"]
    )
    validate_action_trace(
        result.receipt.cognitive_action_trace,
        evidence_snapshot=evidence,
        executors=executors,
        action_intervention=intervention,
    )
    validate_action_intervention_receipt(
        calibration,
        intervention=intervention,
        cognitive_action_trace=result.receipt.cognitive_action_trace,
    )

    with pytest.raises(ValueError, match="already claimed"):
        engine.reason(
            prompt=_task_prompt(intervention),
            action_policy_evidence=evidence,
            action_intervention=intervention,
            action_intervention_consumption=consumption,
        )
