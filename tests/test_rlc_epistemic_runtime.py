"""Live runtime-operation admission, durability, and wire-binding contracts."""

from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.epistemic_journal import EpistemicStateJournal
from core.brain.llm.latent_cortex.epistemic_runtime import (
    RUNTIME_OPERATION_SCHEMA,
    RuntimeOperationLease,
    measured_operation_cost,
    operation_kind_for_decision,
    validate_runtime_operation_authority,
)
from core.brain.llm.latent_cortex.epistemic_state import (
    ComputeBudgetState,
    EpistemicState,
    EpistemicStateError,
    EpistemicTransaction,
    OperationKind,
    OperationOutcome,
    OperationRecord,
    ProblemFrame,
    text_sha256,
)
from core.brain.llm.latent_cortex.value_of_computation import (
    ACTION_TRANSITION_SCHEMA,
    CognitiveStateSignal,
    ValueOfComputationPolicy,
    build_evidence_snapshot,
    transition_reward,
)

OBJECTIVE = "Compare two recovery designs and select the safer one."


def _state_pair() -> tuple[EpistemicState, EpistemicState]:
    genesis = EpistemicState.genesis(
        episode_id="rlc-runtime-test",
        problem=ProblemFrame.create(OBJECTIVE),
        budget=ComputeBudgetState(total=1.0),
    )
    memory = OperationRecord.create(
        operation_id="memory-search-runtime-test",
        kind=OperationKind.SEARCH_MEMORY,
        outcome=OperationOutcome.SUCCEEDED,
        input_state_sha256=genesis.state_sha256,
        cost=0.01,
        operator_id="selective_memory_bridge",
        operator_version="v1",
        input_payload_sha256=text_sha256("runtime memory query"),
        started_at=1.0,
        completed_at=2.0,
        detail="bounded memory query completed",
    )
    state = EpistemicTransaction(genesis).add_operation(memory).commit()
    return genesis, state


def _decision(**updates):
    decision = {
        "schema": "aura.latent_execution_controller.v1",
        "bucket": "reasoning|compare,select|short|s:high|u:high",
        "arm": "base",
        "mode": "observe",
        "evidence": {},
    }
    decision.update(updates)
    return decision


def _config(**updates):
    config = {
        "n_branches": 2,
        "max_steps": 4,
        "decode_max_tokens": 256,
    }
    config.update(updates)
    return config


def _budget():
    return {"max_layer_apps": 1000, "wall_clock_s": 30.0}


def _action_policy():
    return build_evidence_snapshot(bucket=_decision()["bucket"], cells={})


def _action_transition(
    *,
    step_index: int,
    action: OperationKind,
    cost: float,
    outcome: str = "completed",
    checked: bool = True,
) -> dict:
    return {
        "schema": ACTION_TRANSITION_SCHEMA,
        "bucket": _decision()["bucket"],
        "snapshot_sha256": _action_policy()["snapshot_sha256"],
        "decision_sha256": text_sha256(f"decision:{step_index}:{action.value}"),
        "step_index": step_index,
        "action": action.value,
        "mode": "measured" if checked else "bootstrap",
        "outcome": outcome,
        "checked": checked,
        "metrics": transition_reward(
            verified_delta=0.1 if checked else 0.0,
            information_gain=0.05,
            diversity_gain=0.01,
            unsupported_confidence=0.0,
            cost=cost,
        ),
    }


def _begin(tmp_path, **updates) -> RuntimeOperationLease:
    genesis, state = _state_pair()
    values = {
        "genesis": genesis,
        "state": state,
        "decision": _decision(),
        "config": _config(),
        "budget": _budget(),
        "action_policy_evidence": _action_policy(),
        "root": tmp_path / "runtime",
        "started_at": 10.0,
    }
    values.update(updates)
    return RuntimeOperationLease.begin(**values)


def test_operation_kind_names_the_executed_policy():
    assert operation_kind_for_decision(_decision(), _config()) is OperationKind.BRANCH
    assert operation_kind_for_decision(
        _decision(), _config(n_branches=1)
    ) is OperationKind.BLIND_RESOLVE
    assert operation_kind_for_decision(
        _decision(arm="probe_guided_bytecode"), _config()
    ) is OperationKind.CHECK_ASSUMPTION


def test_begin_persists_unknown_intent_before_compute(tmp_path):
    lease = _begin(tmp_path)

    assert lease.state.version == 2
    assert lease.intent.outcome is OperationOutcome.UNKNOWN
    assert lease.intent.failure_code == "execution_pending"
    assert lease.intent.cost == 0.0
    assert lease.authority["schema"] == RUNTIME_OPERATION_SCHEMA
    assert lease.authority["operation_kind"] == "branch"
    assert lease.authority["admitted_state_sha256"] == lease.state.state_sha256

    genesis, _ = _state_pair()
    recovered = EpistemicStateJournal(
        tmp_path / "runtime" / "rlc-runtime-test.jsonl"
    ).bootstrap(genesis)
    assert recovered == lease.state
    assert recovered.operations[-1] == lease.intent


def test_state_less_episode_can_begin_from_objective_genesis(tmp_path):
    genesis, _ = _state_pair()
    lease = _begin(tmp_path, state=genesis)

    assert lease.state.version == 1
    assert len(lease.state.operations) == 1
    assert lease.state.operations[0].kind is OperationKind.BRANCH
    assert lease.state.operations[0].outcome is OperationOutcome.UNKNOWN


def test_pending_intent_recovers_without_duplicate_execution_record(tmp_path):
    first = _begin(tmp_path)
    second = _begin(tmp_path)

    assert second.intent == first.intent
    assert second.state == first.state
    assert second.authority["admission_reason"] == "recovered_pending_operation"
    assert len(second.state.operations) == 2


def test_completion_is_explicit_retry_with_measured_cost(tmp_path):
    lease = _begin(tmp_path)
    cost, cost_receipt = measured_operation_cost(
        {"budget": {"spent_layer_apps": 250, "max_layer_apps": 1000}},
        requested_budget=_budget(),
        state=lease.state,
    )
    completed = lease.complete(
        outcome=OperationOutcome.SUCCEEDED,
        cost=cost,
        completed_at=20.0,
        detail=f"cost={cost_receipt['basis']}",
    )

    assert cost == pytest.approx(0.2475)
    assert completed.version == 3
    assert completed.budget.used == pytest.approx(0.2575)
    terminal = completed.operations[-1]
    assert terminal.outcome is OperationOutcome.SUCCEEDED
    assert terminal.retry_of_operation_id == lease.intent.operation_id
    assert terminal.attempt_sha256 == lease.intent.attempt_sha256
    assert lease.to_receipt()["completed"] is True

    genesis, _ = _state_pair()
    recovered = EpistemicStateJournal(
        tmp_path / "runtime" / "rlc-runtime-test.jsonl"
    ).bootstrap(genesis)
    assert recovered == completed


def test_completion_atomically_records_cognitive_actions_without_double_charge(
    tmp_path,
):
    lease = _begin(tmp_path)
    transitions = (
        _action_transition(
            step_index=0,
            action=OperationKind.DECOMPOSE,
            cost=0.05,
        ),
        _action_transition(
            step_index=1,
            action=OperationKind.FALSIFY,
            cost=0.10,
            outcome="verified_progress_saved",
        ),
    )
    completed = lease.complete(
        outcome=OperationOutcome.SUCCEEDED,
        cost=0.30,
        action_transitions=transitions,
        action_costs=(0.05, 0.10),
        completed_at=20.0,
    )

    receipt = lease.to_receipt()
    terminal = OperationRecord.from_dict(receipt["terminal"])
    action_operations = tuple(
        OperationRecord.from_dict(row) for row in receipt["action_operations"]
    )
    assert terminal.cost == pytest.approx(0.15)
    assert tuple(item.kind for item in action_operations) == (
        OperationKind.DECOMPOSE,
        OperationKind.FALSIFY,
    )
    assert tuple(item.cost for item in action_operations) == pytest.approx((0.05, 0.10))
    assert completed.budget.used == pytest.approx(0.31)
    assert sum(item.cost for item in completed.operations) == pytest.approx(0.31)

    genesis, _ = _state_pair()
    recovered = EpistemicStateJournal(
        tmp_path / "runtime" / "rlc-runtime-test.jsonl"
    ).bootstrap(genesis)
    assert recovered == completed


def test_invalid_action_batch_fails_atomically_without_completing_lease(tmp_path):
    lease = _begin(tmp_path)
    admitted = lease.state
    transition = _action_transition(
        step_index=0,
        action=OperationKind.DECOMPOSE,
        cost=0.5,
    )

    with pytest.raises(EpistemicStateError, match="exceed measured"):
        lease.complete(
            outcome=OperationOutcome.SUCCEEDED,
            cost=0.1,
            action_transitions=(transition,),
            action_costs=(0.2,),
            completed_at=20.0,
        )
    assert lease.state == admitted
    assert lease.to_receipt()["completed"] is False
    assert lease.to_receipt()["action_operations"] == []

    invalid = {**transition, "checked": "yes"}
    with pytest.raises(EpistemicStateError, match="transition is invalid"):
        lease.complete(
            outcome=OperationOutcome.SUCCEEDED,
            cost=0.1,
            action_transitions=(invalid,),
            action_costs=(0.05,),
            completed_at=20.0,
        )
    genesis, _ = _state_pair()
    recovered = EpistemicStateJournal(
        tmp_path / "runtime" / "rlc-runtime-test.jsonl"
    ).bootstrap(genesis)
    assert recovered == admitted


def test_failure_is_durable_and_cannot_be_completed_twice(tmp_path):
    lease = _begin(tmp_path)
    state = lease.complete(
        outcome=OperationOutcome.FAILED,
        cost=0.0,
        failure_code="worker_operation_failed",
        completed_at=12.0,
    )
    assert state.operations[-1].failure_code == "worker_operation_failed"
    with pytest.raises(EpistemicStateError, match="already complete"):
        lease.complete(
            outcome=OperationOutcome.SUCCEEDED,
            cost=0.0,
            completed_at=13.0,
        )


def test_completed_attempt_cannot_be_silently_reexecuted(tmp_path):
    lease = _begin(tmp_path)
    lease.complete(
        outcome=OperationOutcome.SUCCEEDED,
        cost=0.1,
        completed_at=12.0,
    )
    with pytest.raises(EpistemicStateError, match="cannot resume"):
        _begin(tmp_path)


def test_authority_binds_objective_config_budget_and_memory_state(tmp_path):
    lease = _begin(tmp_path)
    memory_context = [
        {
            "source": "memory",
            "text": "historical observation",
            "context_role": "memory_observation",
            "epistemic_state_sha256": lease.state.state_sha256,
        }
    ]
    validated = validate_runtime_operation_authority(
        lease.authority,
        prompt=OBJECTIVE,
        messages=None,
        config=_config(),
        budget=_budget(),
        cognitive_context=memory_context,
        action_policy_evidence=_action_policy(),
    )
    assert validated == lease.authority

    cases = (
        ("objective", {"prompt": "another objective"}),
        ("config", {"config": _config(max_steps=9)}),
        ("budget", {"budget": {"max_layer_apps": 999, "wall_clock_s": 30.0}}),
        (
            "memory state",
            {
                "cognitive_context": [
                    {**memory_context[0], "epistemic_state_sha256": "f" * 64}
                ]
            },
        ),
        (
            "action policy",
            {
                "action_policy_evidence": build_evidence_snapshot(
                    bucket=_decision()["bucket"],
                    cells={
                        OperationKind.COMPARE: {
                            "n": 1,
                            "gain_sum": 0.1,
                            "gain_sq_sum": 0.01,
                            "cost_sum": 0.1,
                            "cost_sq_sum": 0.01,
                        }
                    },
                )
            },
        ),
    )
    base = {
        "prompt": OBJECTIVE,
        "messages": None,
        "config": _config(),
        "budget": _budget(),
        "cognitive_context": memory_context,
        "action_policy_evidence": _action_policy(),
    }
    for label, update in cases:
        with pytest.raises(EpistemicStateError, match=label):
            validate_runtime_operation_authority(
                lease.authority,
                **{**base, **update},
            )


def test_authority_rejects_field_and_kind_tampering(tmp_path):
    lease = _begin(tmp_path)
    base = {
        "prompt": OBJECTIVE,
        "messages": None,
        "config": _config(),
        "budget": _budget(),
        "cognitive_context": None,
        "action_policy_evidence": _action_policy(),
    }
    for tampered in (
        {**lease.authority, "extra": True},
        {**lease.authority, "operation_kind": "invent"},
        {**lease.authority, "attempt_sha256": "x" * 64},
        {**lease.authority, "admitted_state_version": True},
        {**lease.authority, "controller_evidence": {"unbound": 1}},
        {**lease.authority, "input_evidence_ids": ["evidence-unbound"]},
        {**lease.authority, "operation_id": "rlc-op-wrong-a1"},
        {**lease.authority, "action_policy_sha256": "f" * 64},
    ):
        with pytest.raises(EpistemicStateError):
            validate_runtime_operation_authority(tampered, **base)


def test_measurement_refuses_mismatched_or_impossible_worker_budget():
    _, state = _state_pair()
    for receipt in (
        {"budget": {"spent_layer_apps": 1001, "max_layer_apps": 1000}},
        {"budget": {"spent_layer_apps": 100, "max_layer_apps": 999}},
        {"budget": {"spent_layer_apps": True, "max_layer_apps": 1000}},
    ):
        with pytest.raises(EpistemicStateError, match="compute receipt"):
            measured_operation_cost(
                receipt,
                requested_budget=_budget(),
                state=state,
            )


def test_begin_rejects_state_not_directly_derived_from_genesis(tmp_path):
    _, state = _state_pair()
    alien_genesis = EpistemicState.genesis(
        episode_id="alien-runtime-test",
        problem=ProblemFrame.create(OBJECTIVE),
        budget=ComputeBudgetState(total=1.0),
    )
    with pytest.raises(EpistemicStateError, match="does not extend genesis"):
        _begin(tmp_path, genesis=alien_genesis, state=state)


@pytest.mark.asyncio
async def test_live_decode_overrides_do_not_bypass_operation_controller(
    tmp_path, monkeypatch
):
    import core.config as config_module
    from core.brain import llm_health_router
    from core.brain.latent_cortex_service import LatentCortexService
    from core.brain.llm import mlx_client
    from core.brain.llm.latent_cortex import execution_controller
    from core.brain.llm.latent_cortex.epistemic_memory import (
        MemoryQuery,
        SelectiveMemoryBridge,
        attach_memory_result,
    )

    query = MemoryQuery.create(
        OBJECTIVE,
        episode_id="rlc-live-controller-test",
        tenant_id="local",
        user_id="owner",
        session_id="test-session",
    )
    memory_result = SelectiveMemoryBridge({}).retrieve(query)
    genesis = EpistemicState.genesis(
        episode_id=query.scope.episode_id,
        problem=ProblemFrame.create(OBJECTIVE),
        budget=ComputeBudgetState(total=1.0),
    )
    state = attach_memory_result(genesis, memory_result)
    captured = {}

    class Controller:
        def choose(self, **kwargs):
            captured["controller_choose"] = kwargs
            return _decision()

        def apply_arm(self, arm, config, **kwargs):
            captured["controller_arm"] = arm
            return dict(config)

    class Client:
        def get_worker_identity_snapshot(self):
            return {"worker_model_parameter_count": 1_500_000_000}

        async def latent_reason_async(self, **kwargs):
            captured["worker_request"] = kwargs
            return {
                "ok": False,
                "reason": "bounded_test_stop",
                "receipt": {
                    "runtime_operation_authority": kwargs["operation_authority"],
                    "budget": {
                        "spent_layer_apps": 10,
                        "max_layer_apps": kwargs["budget"]["max_layer_apps"],
                    },
                },
            }

    async def acquire(**kwargs):
        return "lease-runtime-operation-test"

    monkeypatch.setattr(config_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(execution_controller, "controller_enabled", lambda: True)
    monkeypatch.setattr(
        execution_controller,
        "get_execution_controller",
        lambda: Controller(),
    )
    monkeypatch.setattr(mlx_client, "get_mlx_client", lambda: Client())
    monkeypatch.setattr(
        llm_health_router,
        "acquire_external_generation_gate_lease",
        acquire,
    )
    monkeypatch.setattr(
        llm_health_router,
        "release_external_generation_gate_lease",
        lambda _lease_id: None,
    )

    response = await LatentCortexService().deep_reason(
        OBJECTIVE,
        stakes=0.8,
        uncertainty=0.8,
        config_overrides={"decode_max_tokens": 128},
        foreground_request=True,
        epistemic_genesis=genesis,
        epistemic_state=state,
        selective_memory_result=memory_result,
    )

    assert captured["controller_choose"]["objective"] == OBJECTIVE
    worker_request = captured["worker_request"]
    assert (
        worker_request["operation_authority"]["action_policy_sha256"]
        == worker_request["action_policy_evidence"]["snapshot_sha256"]
    )
    assert worker_request["operation_authority"]["operation_kind"] in {
        "blind_resolve",
        "branch",
    }
    operation = response["receipt"]["epistemic_operation"]
    assert operation["completed"] is True
    assert operation["terminal"]["outcome"] == "failed"
    assert operation["terminal"]["failure_code"] == "worker_operation_failed"
    assert operation["current_state_version"] == 3


@pytest.mark.asyncio
async def test_service_never_marks_contract_failed_worker_as_successful_operation(
    tmp_path, monkeypatch
):
    import core.config as config_module
    from core.brain import llm_health_router
    from core.brain.latent_cortex_service import LatentCortexService
    from core.brain.llm import mlx_client
    from core.brain.llm.latent_cortex import execution_controller

    captured = {}

    class Controller:
        def choose(self, **kwargs):
            return _decision()

        def apply_arm(self, arm, config, **kwargs):
            return dict(config)

    class Client:
        def get_worker_identity_snapshot(self):
            return {"worker_model_parameter_count": 1_500_000_000}

        async def latent_reason_async(self, **kwargs):
            snapshot = kwargs["action_policy_evidence"]
            worker_max_layer_apps = kwargs["budget"]["max_layer_apps"]
            decision = ValueOfComputationPolicy(snapshot).choose(
                CognitiveStateSignal(
                    step_index=0,
                    max_steps=4,
                    neural_steps=0,
                    min_neural_steps=1,
                    active_branches=2,
                    total_branches=2,
                    residual=0.8,
                    residual_delta=0.0,
                    verifier_score=None,
                    verifier_delta=None,
                    disagreement=0.4,
                    uncertainty=0.8,
                    budget_remaining_fraction=0.9,
                    has_memory=False,
                    has_evidence=False,
                    has_verifier=False,
                    has_savepoint=False,
                    can_execute=False,
                    answer_verified=False,
                    irreducible_uncertainty=False,
                ),
                executors=(OperationKind.DECOMPOSE,),
            )
            transition = {
                "schema": ACTION_TRANSITION_SCHEMA,
                "bucket": snapshot["bucket"],
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "decision_sha256": decision["decision_sha256"],
                "step_index": decision["step_index"],
                "action": decision["action"],
                "mode": decision["mode"],
                "outcome": "completed",
                "checked": False,
                "metrics": transition_reward(
                    verified_delta=0.0,
                    information_gain=0.0,
                    diversity_gain=0.0,
                    unsupported_confidence=0.0,
                    cost=10 / worker_max_layer_apps,
                ),
            }
            captured["transition"] = transition
            return {
                "ok": True,
                "text": "This worker response intentionally lacks its full receipt.",
                "receipt": {
                    "runtime_operation_authority": kwargs["operation_authority"],
                    "budget": {
                        "spent_layer_apps": 20,
                        "max_layer_apps": worker_max_layer_apps,
                    },
                    "value_of_computation": {
                        "active": True,
                        "bucket": snapshot["bucket"],
                        "snapshot_sha256": snapshot["snapshot_sha256"],
                        "actions_selected": 1,
                        "schema": snapshot["schema"],
                        "executors": [OperationKind.DECOMPOSE.value],
                        "checked_transitions": 0,
                        "selected_actions": [decision["action"]],
                    },
                    "cognitive_action_trace": [
                        {
                            "decision": decision,
                            "transition": transition,
                            "state_signal": CognitiveStateSignal(
                                step_index=0,
                                max_steps=4,
                                neural_steps=0,
                                min_neural_steps=1,
                                active_branches=2,
                                total_branches=2,
                                residual=0.8,
                                residual_delta=0.0,
                                verifier_score=None,
                                verifier_delta=None,
                                disagreement=0.4,
                                uncertainty=0.8,
                                budget_remaining_fraction=0.9,
                                has_memory=False,
                                has_evidence=False,
                                has_verifier=False,
                                has_savepoint=False,
                                can_execute=False,
                                answer_verified=False,
                                irreducible_uncertainty=False,
                            ).to_dict(),
                            "state_before": {
                                "residual": 0.8,
                                "disagreement": 0.4,
                                "verifier_score": None,
                                "budget_remaining_fraction": 0.9,
                            },
                            "state_after": {
                                "residual": 0.8,
                                "disagreement": 0.4,
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
                    ],
                },
            }

    async def acquire(**kwargs):
        return "lease-runtime-operation-contract-test"

    monkeypatch.setattr(config_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(execution_controller, "controller_enabled", lambda: True)
    monkeypatch.setattr(
        execution_controller,
        "get_execution_controller",
        lambda: Controller(),
    )
    monkeypatch.setattr(mlx_client, "get_mlx_client", lambda: Client())
    monkeypatch.setattr(
        llm_health_router,
        "acquire_external_generation_gate_lease",
        acquire,
    )
    monkeypatch.setattr(
        llm_health_router,
        "release_external_generation_gate_lease",
        lambda _lease_id: None,
    )

    response = await LatentCortexService().deep_reason(
        OBJECTIVE,
        stakes=0.8,
        uncertainty=0.8,
        config_overrides={"decode_max_tokens": 128},
        foreground_request=True,
    )

    assert response["ok"] is False
    assert response["reason"].startswith("receipt_contract_failed:")
    operation = response["receipt"]["epistemic_operation"]
    assert operation["terminal"]["outcome"] == "failed"
    assert operation["terminal"]["failure_code"] == "worker_receipt_contract_failed"
    assert len(operation["action_operations"]) == 1
    assert operation["action_operations"][0]["kind"] == captured["transition"]["action"]
    assert operation["action_operations"][0]["outcome"] == "succeeded"
    assert operation["compute"]["action_operation_count"] == 1
