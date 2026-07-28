"""Adversarial tests for pre-objective recurrent state custody."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

mx = pytest.importorskip("mlx.core")

from core.learning.grpo import group_advantages  # noqa: E402
from core.learning.recurrence_native_objective_v2 import (  # noqa: E402
    ExactAdjointInterventionConfig,
)
from core.learning.recurrent_grpo import (  # noqa: E402
    VERIFIED_TRAJECTORY_SOURCE_SCHEMA_V2,
    RecurrentGRPOConfig,
    VerifiedTrajectoryGroupConfig,
    recurrent_policy_optimizer_config,
    recurrent_policy_tensor_map_sha256,
)
from core.learning.verified_transition_episode import (  # noqa: E402
    canonical_json_bytes,
)
from core.learning.verified_transition_measurement_chain import (  # noqa: E402
    VerifiedTransitionMeasurementChainError,
    VerifiedTransitionMeasurementChainStore,
    load_pre_measurement_state_tensors,
    recurrent_grpo_config_contract,
    recurrent_grpo_config_from_contract,
    validate_pre_measurement_intent,
)
from core.learning.verified_transition_policy_probe import (  # noqa: E402
    build_initial_policy_state_custody,
    inspect_initial_adapter_snapshot,
    inspect_initial_optimizer_snapshot,
)
from core.learning.verified_transition_transaction import (  # noqa: E402
    VerifiedTransitionTransactionError,
    VerifiedTransitionTransactionStore,
    build_pending_trainer_step,
    build_trainer_step_static,
)
from core.learning.verified_transition_update import (  # noqa: E402
    VERIFIED_TRANSITION_RESERVATION_SCHEMA_V2,
    VerifiedTransitionUpdateJournal,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical_with_floats(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _adapter(value: float = 1.0) -> dict[str, Any]:
    return {
        "model.layers.0.self_attn.q_proj.lora_a": mx.array([[value, value + 1.0]]),
        "model.layers.0.self_attn.q_proj.lora_b": mx.array([[value + 2.0], [value + 3.0]]),
    }


def _optimizer(
    adapter: dict[str, Any],
    *,
    step: int = 0,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "step": mx.array(step),
        "learning_rate": mx.array(1e-5),
    }
    for key, tensor in adapter.items():
        state[f"{key}.m"] = mx.zeros_like(tensor)
        state[f"{key}.v"] = mx.zeros_like(tensor)
    return state


def _save(path: Path, tensors: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    mx.eval(*tensors.values())
    mx.save_safetensors(str(path), tensors)
    os.chmod(path, 0o600)


def _binding(path: Path, tensors: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(tensors)
    payload = path.read_bytes()
    return {
        "path": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "tensor_count": len(keys),
        "tensor_keys_sha256": hashlib.sha256(canonical_json_bytes(keys)).hexdigest(),
    }


def _custody(
    root: Path,
    *,
    execution_spec_sha256: str,
    adapter: dict[str, Any],
    optimizer: dict[str, Any],
) -> dict[str, Any]:
    adapter_path = root / "initial_adapter.safetensors"
    optimizer_path = root / "initial_optimizer.safetensors"
    _save(adapter_path, adapter)
    _save(optimizer_path, optimizer)
    adapter_artifact = inspect_initial_adapter_snapshot(
        adapter_path,
        execution_spec_sha256=execution_spec_sha256,
    )
    optimizer_artifact = inspect_initial_optimizer_snapshot(optimizer_path)
    return build_initial_policy_state_custody(
        initial_policy_probe_sha256=_sha("initial-probe"),
        initial_policy_sha256=adapter_artifact["policy_sha256"],
        execution_spec_sha256=execution_spec_sha256,
        adapter_initialization={
            "seed": 7,
            "rank": 2,
            "layers": 1,
            "targets": ["q_proj"],
        },
        optimizer_initialization=recurrent_policy_optimizer_config(1e-5),
        initial_adapter_artifact=adapter_artifact,
        initial_optimizer_artifact=optimizer_artifact,
        initial_adapter_path=adapter_path.resolve(strict=True),
        initial_optimizer_path=optimizer_path.resolve(strict=True),
    )


def _trajectory_source(
    *,
    admission_sha256: str,
    policy_sha256: str,
    execution_spec_sha256: str,
) -> dict[str, Any]:
    config = VerifiedTrajectoryGroupConfig(
        intervention_config=ExactAdjointInterventionConfig(
            lesion_steps=(1,),
            causality_weight=0.4,
            causality_margin=0.1,
            stopping_steps=(1, 2),
            stopping_weight=0.3,
            stopping_ponder_cost=0.01,
            stopping_temperature=0.2,
        )
    )
    body = {
        "schema": VERIFIED_TRAJECTORY_SOURCE_SCHEMA_V2,
        "group_admission_sha256": admission_sha256,
        "reward_receipt_sha256": _sha(f"reward-{admission_sha256}"),
        "policy_sha256": policy_sha256,
        "execution_spec_sha256": execution_spec_sha256,
        "prompt_tokens_sha256": _sha(f"prompt-{admission_sha256}"),
        "sample_receipt_sha256s": [
            _sha(f"sample-0-{admission_sha256}"),
            _sha(f"sample-1-{admission_sha256}"),
        ],
        "completion_tokens_sha256s": [
            _sha(f"completion-0-{admission_sha256}"),
            _sha(f"completion-1-{admission_sha256}"),
        ],
        "sample_branch_indices": [0, 1],
        "execution_branch_count": 2,
        "verified_rewards": [1.0, 0.0],
        "advantage_clip": 4.0,
        "config": config.to_dict(),
    }
    return {
        **body,
        "source_sha256": hashlib.sha256(_canonical_with_floats(body)).hexdigest(),
    }


def _static() -> dict[str, Any]:
    return build_trainer_step_static(
        samples=[{"sample": 0}, {"sample": 1}],
        structured_rewards=[1.0, 0.0],
        optimizer_admission_reason="verified_improvement",
        answer_channel={"valid": True},
        advantage_report=group_advantages([1.0, 0.0]),
    )


class _FakeTransactionStore:
    def __init__(self) -> None:
        self.transactions: list[Any] = []

    def inventory(self, *, load_tensors: bool = False) -> tuple[Any, ...]:
        del load_tensors
        return tuple(self.transactions)

    def load(
        self,
        *,
        sequence: int,
        admission_sha256: str,
        load_tensors: bool = True,
    ) -> Any | None:
        del load_tensors
        for transaction in self.transactions:
            if (
                transaction.pending_step["sequence"] == sequence
                and transaction.pending_step["group_admission_sha256"] == admission_sha256
            ):
                return transaction
        return None


def _store(
    tmp_path: Path,
    *,
    transaction_store: Any,
) -> tuple[
    VerifiedTransitionMeasurementChainStore,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
]:
    execution_spec_sha256 = _sha("execution-spec")
    adapter = _adapter()
    optimizer = _optimizer(adapter)
    custody = _custody(
        tmp_path / "launch",
        execution_spec_sha256=execution_spec_sha256,
        adapter=adapter,
        optimizer=optimizer,
    )
    store = VerifiedTransitionMeasurementChainStore.open(
        tmp_path / "transactions",
        transaction_store=transaction_store,
        initial_policy_state_custody=custody,
        provider_contract_sha256=_sha("provider-contract"),
        training_protocol_sha256=_sha("training-protocol"),
    )
    return store, adapter, optimizer, custody, execution_spec_sha256


def _begin(
    store: VerifiedTransitionMeasurementChainStore,
    *,
    sequence: int,
    admission_sha256: str,
    policy_sha256: str,
    execution_spec_sha256: str,
    adapter: dict[str, Any],
    optimizer: dict[str, Any],
    recorded_at_unix_ns: int,
    reservation_sha256: str | None = None,
) -> dict[str, Any]:
    return store.begin(
        sequence=sequence,
        trainer_step=sequence + 1,
        group_admission_sha256=admission_sha256,
        reservation_sha256=(reservation_sha256 or _sha(f"reservation-{admission_sha256}")),
        policy_before_sha256=policy_sha256,
        campaign_manifest_sha256=_sha("campaign"),
        campaign_schedule_root_sha256=_sha("schedule"),
        group_manifest_sha256=_sha(f"group-{admission_sha256}"),
        execution_spec_sha256=execution_spec_sha256,
        trainer_step_static=_static(),
        trajectory_source_binding=_trajectory_source(
            admission_sha256=admission_sha256,
            policy_sha256=policy_sha256,
            execution_spec_sha256=execution_spec_sha256,
        ),
        recurrent_grpo_config=RecurrentGRPOConfig(),
        bridge_tokens=(),
        live_adapter_tensors=adapter,
        live_optimizer_tensors=optimizer,
        recorded_at_unix_ns=recorded_at_unix_ns,
    )


def test_initial_intent_precedes_objective_and_optimizer_drift_fails_closed(
    tmp_path: Path,
) -> None:
    transactions = _FakeTransactionStore()
    store, adapter, optimizer, custody, execution_spec = _store(
        tmp_path,
        transaction_store=transactions,
    )
    admission = _sha("admission-0")
    intent = _begin(
        store,
        sequence=0,
        admission_sha256=admission,
        policy_sha256=custody["initial_policy_sha256"],
        execution_spec_sha256=execution_spec,
        adapter=adapter,
        optimizer=optimizer,
        recorded_at_unix_ns=100,
    )

    assert validate_pre_measurement_intent(intent) == intent
    assert intent["state_source"]["kind"] == "initial_policy_state"
    assert intent["state_source"]["successful_update_ordinal"] == 1
    loaded_adapter, loaded_optimizer = load_pre_measurement_state_tensors(intent)
    assert set(loaded_adapter) == set(adapter)
    assert set(loaded_optimizer) == set(optimizer)
    assert all(bool(mx.array_equal(loaded_adapter[key], adapter[key])) for key in adapter)
    assert all(bool(mx.array_equal(loaded_optimizer[key], optimizer[key])) for key in optimizer)
    config = RecurrentGRPOConfig(
        clip_epsilon=0.15,
        kl_coefficient=0.03,
        advantage_clip=3.5,
        max_initial_clip_fraction=0.2,
        max_initial_old_policy_approx_kl=0.07,
    )
    assert recurrent_grpo_config_from_contract(recurrent_grpo_config_contract(config)) == config
    assert not list((tmp_path / "transactions" / "pre-measurements").rglob("*.safetensors"))
    with pytest.raises(
        VerifiedTransitionMeasurementChainError,
        match="pre_measurement_orphan_requires_reconciliation",
    ):
        store.assert_no_orphans()

    store.reconcile_orphan(
        sequence=0,
        admission_sha256=admission,
        live_adapter_tensors=adapter,
        live_optimizer_tensors=optimizer,
        reconciled_at_unix_ns=101,
    )
    store.assert_no_orphans()
    with pytest.raises(
        VerifiedTransitionMeasurementChainError,
        match="pre_measurement_admission_permanently_burned",
    ):
        _begin(
            store,
            sequence=0,
            admission_sha256=admission,
            policy_sha256=custody["initial_policy_sha256"],
            execution_spec_sha256=execution_spec,
            adapter=adapter,
            optimizer=optimizer,
            recorded_at_unix_ns=102,
        )

    fresh_store, fresh_adapter, fresh_optimizer, fresh_custody, fresh_spec = _store(
        tmp_path / "drift",
        transaction_store=_FakeTransactionStore(),
    )
    drifted_optimizer = dict(fresh_optimizer)
    drifted_optimizer["step"] = mx.array(1)
    with pytest.raises(
        VerifiedTransitionMeasurementChainError,
        match="pre_measurement_live_optimizer_state_mismatch",
    ):
        _begin(
            fresh_store,
            sequence=0,
            admission_sha256=_sha("drift-admission"),
            policy_sha256=fresh_custody["initial_policy_sha256"],
            execution_spec_sha256=fresh_spec,
            adapter=fresh_adapter,
            optimizer=drifted_optimizer,
            recorded_at_unix_ns=200,
        )
    assert fresh_store.inventory() == ()

    dtype_optimizer = dict(fresh_optimizer)
    dtype_optimizer["step"] = mx.array(0, dtype=mx.int64)
    with pytest.raises(
        VerifiedTransitionMeasurementChainError,
        match="pre_measurement_live_optimizer_state_mismatch",
    ):
        _begin(
            fresh_store,
            sequence=0,
            admission_sha256=_sha("dtype-drift-admission"),
            policy_sha256=fresh_custody["initial_policy_sha256"],
            execution_spec_sha256=fresh_spec,
            adapter=fresh_adapter,
            optimizer=dtype_optimizer,
            recorded_at_unix_ns=201,
        )
    assert fresh_store.inventory() == ()


def test_initial_state_load_rejects_post_open_artifact_substitution(
    tmp_path: Path,
) -> None:
    store, adapter, optimizer, custody, execution_spec = _store(
        tmp_path,
        transaction_store=_FakeTransactionStore(),
    )
    substituted = dict(optimizer)
    substituted["step"] = mx.array(7)
    _save(Path(custody["initial_optimizer_path"]), substituted)

    with pytest.raises(
        VerifiedTransitionMeasurementChainError,
        match="pre_measurement_initial_optimizer_digest_mismatch",
    ):
        _begin(
            store,
            sequence=0,
            admission_sha256=_sha("substituted-initial-state"),
            policy_sha256=custody["initial_policy_sha256"],
            execution_spec_sha256=execution_spec,
            adapter=adapter,
            optimizer=optimizer,
            recorded_at_unix_ns=300,
        )
    assert store.inventory() == ()


@pytest.mark.parametrize("publish_intent", [False, True])
def test_pre_stage_recovery_burns_intervention_admission_after_restore(
    tmp_path: Path,
    publish_intent: bool,
) -> None:
    transactions = _FakeTransactionStore()
    store, adapter, optimizer, custody, execution_spec = _store(
        tmp_path,
        transaction_store=transactions,
    )
    admission = _sha(f"pre-stage-{publish_intent}")
    journal = VerifiedTransitionUpdateJournal.open(tmp_path / "update-journal")
    reservation = journal.reserve(
        admission_sha256=admission,
        policy_before_sha256=custody["initial_policy_sha256"],
        reserved_at_unix_ns=100,
        campaign_sequence=0,
        execution_spec_sha256=execution_spec,
        group_manifest_sha256=_sha(f"group-{admission}"),
        pre_measurement_required=True,
    )
    assert reservation["schema"] == VERIFIED_TRANSITION_RESERVATION_SCHEMA_V2
    intent = None
    if publish_intent:
        intent = _begin(
            store,
            sequence=0,
            admission_sha256=admission,
            policy_sha256=custody["initial_policy_sha256"],
            execution_spec_sha256=execution_spec,
            adapter=adapter,
            optimizer=optimizer,
            recorded_at_unix_ns=100,
            reservation_sha256=reservation["receipt_sha256"],
        )

    recovered = store.reconcile_interrupted_admissions(
        update_journal=journal,
        live_adapter_tensors=adapter,
        live_optimizer_tensors=optimizer,
        observed_policy_sha256=custody["initial_policy_sha256"],
        reconciled_at_unix_ns=101,
    )

    assert len(recovered) == 1
    assert recovered[0]["sequence"] == 0
    assert recovered[0]["requires_fresh_campaign"] is True
    assert recovered[0]["update_reconciliation"]["classification"] == "reserved_no_policy_change"
    assert (recovered[0]["measurement_reconciliation"] is not None) is publish_intent
    inventory = journal.inventory()
    assert len(inventory) == 1
    assert inventory[0]["commit"] is None
    assert inventory[0]["reconciliation"] is not None
    if intent is not None:
        assert (
            recovered[0]["measurement_reconciliation"]["pre_measurement_sha256"]
            == intent["receipt_sha256"]
        )
    store.assert_no_orphans()
    assert (
        store.reconcile_interrupted_admissions(
            update_journal=journal,
            live_adapter_tensors=adapter,
            live_optimizer_tensors=optimizer,
            observed_policy_sha256=custody["initial_policy_sha256"],
            reconciled_at_unix_ns=102,
        )
        == recovered
    )


def test_pre_stage_recovery_records_drift_and_refuses_abandonment(
    tmp_path: Path,
) -> None:
    store, adapter, optimizer, custody, execution_spec = _store(
        tmp_path,
        transaction_store=_FakeTransactionStore(),
    )
    admission = _sha("unrestored-admission")
    journal = VerifiedTransitionUpdateJournal.open(tmp_path / "update-journal")
    reservation = journal.reserve(
        admission_sha256=admission,
        policy_before_sha256=custody["initial_policy_sha256"],
        reserved_at_unix_ns=100,
        campaign_sequence=0,
        execution_spec_sha256=execution_spec,
        group_manifest_sha256=_sha(f"group-{admission}"),
        pre_measurement_required=True,
    )
    _begin(
        store,
        sequence=0,
        admission_sha256=admission,
        policy_sha256=custody["initial_policy_sha256"],
        execution_spec_sha256=execution_spec,
        adapter=adapter,
        optimizer=optimizer,
        recorded_at_unix_ns=100,
        reservation_sha256=reservation["receipt_sha256"],
    )

    with pytest.raises(
        VerifiedTransitionMeasurementChainError,
        match="pre_measurement_checkpoint_state_not_restored",
    ):
        store.reconcile_interrupted_admissions(
            update_journal=journal,
            live_adapter_tensors=adapter,
            live_optimizer_tensors=optimizer,
            observed_policy_sha256=_sha("drifted-live-policy"),
            reconciled_at_unix_ns=101,
        )

    inventory = journal.inventory()
    assert inventory[0]["reconciliation"]["classification"] == "policy_changed_without_commit"
    with pytest.raises(
        VerifiedTransitionMeasurementChainError,
        match="pre_measurement_orphan_requires_reconciliation",
    ):
        store.assert_no_orphans()


def test_reservation_only_recovery_requires_exact_optimizer_state(
    tmp_path: Path,
) -> None:
    store, adapter, optimizer, custody, execution_spec = _store(
        tmp_path,
        transaction_store=_FakeTransactionStore(),
    )
    admission = _sha("reservation-only-optimizer-drift")
    journal = VerifiedTransitionUpdateJournal.open(tmp_path / "update-journal")
    journal.reserve(
        admission_sha256=admission,
        policy_before_sha256=custody["initial_policy_sha256"],
        reserved_at_unix_ns=100,
        campaign_sequence=0,
        execution_spec_sha256=execution_spec,
        group_manifest_sha256=_sha(f"group-{admission}"),
        pre_measurement_required=True,
    )
    drifted_optimizer = dict(optimizer)
    drifted_optimizer["step"] = mx.array(1)

    with pytest.raises(
        VerifiedTransitionMeasurementChainError,
        match="pre_measurement_checkpoint_state_not_restored",
    ):
        store.reconcile_interrupted_admissions(
            update_journal=journal,
            live_adapter_tensors=adapter,
            live_optimizer_tensors=drifted_optimizer,
            observed_policy_sha256=custody["initial_policy_sha256"],
            reconciled_at_unix_ns=101,
        )
    assert journal.inventory()[0]["reconciliation"] is None


def test_rejection_gap_references_latest_successful_post_state(
    tmp_path: Path,
) -> None:
    transactions = _FakeTransactionStore()
    store, adapter, optimizer, custody, execution_spec = _store(
        tmp_path,
        transaction_store=transactions,
    )
    first_admission = _sha("admission-0")
    first = _begin(
        store,
        sequence=0,
        admission_sha256=first_admission,
        policy_sha256=custody["initial_policy_sha256"],
        execution_spec_sha256=execution_spec,
        adapter=adapter,
        optimizer=optimizer,
        recorded_at_unix_ns=100,
    )

    post_adapter = _adapter(10.0)
    post_optimizer = _optimizer(post_adapter, step=1)
    post_policy = recurrent_policy_tensor_map_sha256(
        post_adapter,
        execution_spec,
    )
    transaction_dir = tmp_path / "transactions" / "transactions" / f"seq-{0:08d}-{first_admission}"
    stage_dir = transaction_dir / "generations" / "00000000-staged"
    adapter_path = stage_dir / "adapter.safetensors"
    optimizer_path = stage_dir / "optimizer.safetensors"
    _save(adapter_path, post_adapter)
    _save(optimizer_path, post_optimizer)
    transaction = SimpleNamespace(
        transaction_dir=transaction_dir,
        stage={
            "receipt_sha256": _sha("stage-0"),
            "adapter": _binding(adapter_path, post_adapter),
            "optimizer": _binding(optimizer_path, post_optimizer),
        },
        pending_step={
            "sequence": 0,
            "group_admission_sha256": first_admission,
            "policy_after_sha256": post_policy,
            "pre_measurement_sha256": first["receipt_sha256"],
        },
        adapter_tensors=post_adapter,
        optimizer_tensors=post_optimizer,
        events=(
            {"kind": "update_commit"},
            {"kind": "campaign_terminal"},
            {"kind": "trainer_checkpoint"},
        ),
    )
    transactions.transactions.append(transaction)

    second_admission = _sha("admission-after-two-rejections")
    second = _begin(
        store,
        sequence=3,
        admission_sha256=second_admission,
        policy_sha256=post_policy,
        execution_spec_sha256=execution_spec,
        adapter=post_adapter,
        optimizer=post_optimizer,
        recorded_at_unix_ns=300,
    )

    source = second["state_source"]
    assert source["kind"] == "prior_transaction_post_state"
    assert source["source_sequence"] == 0
    assert source["source_admission_sha256"] == first_admission
    assert source["successful_update_ordinal"] == 2
    assert source["source_receipt_sha256"] == _sha("stage-0")
    assert len(list(tmp_path.rglob("*.safetensors"))) == 4


def test_transaction_v4_requires_and_reopens_matching_intent(
    tmp_path: Path,
) -> None:
    transaction_root = tmp_path / "transactions"
    transaction_store = VerifiedTransitionTransactionStore.open(transaction_root)
    store, adapter, optimizer, custody, execution_spec = _store(
        tmp_path,
        transaction_store=transaction_store,
    )
    admission = _sha("transaction-admission")
    journal = VerifiedTransitionUpdateJournal.open(tmp_path / "transaction-journal")
    reservation = journal.reserve(
        admission_sha256=admission,
        policy_before_sha256=custody["initial_policy_sha256"],
        reserved_at_unix_ns=100,
        campaign_sequence=0,
        execution_spec_sha256=execution_spec,
        group_manifest_sha256=_sha(f"group-{admission}"),
        pre_measurement_required=True,
    )
    intent = _begin(
        store,
        sequence=0,
        admission_sha256=admission,
        policy_sha256=custody["initial_policy_sha256"],
        execution_spec_sha256=execution_spec,
        adapter=adapter,
        optimizer=optimizer,
        recorded_at_unix_ns=100,
        reservation_sha256=reservation["receipt_sha256"],
    )
    post_adapter = _adapter(20.0)
    post_optimizer = _optimizer(post_adapter, step=1)
    pending = build_pending_trainer_step(
        sequence=0,
        trainer_step=1,
        task_id="task-0",
        trainer_sample_seed=7,
        execution_spec_sha256=execution_spec,
        campaign_manifest_sha256=_sha("campaign"),
        campaign_schedule_root_sha256=_sha("schedule"),
        group_manifest_sha256=_sha(f"group-{admission}"),
        group_admission_sha256=admission,
        reward_receipt_sha256=_sha("reward"),
        policy_before_sha256=custody["initial_policy_sha256"],
        policy_after_sha256=recurrent_policy_tensor_map_sha256(
            post_adapter,
            execution_spec,
        ),
        trainer_step_static=_static(),
        pre_measurement_sha256=intent["receipt_sha256"],
        reservation_sha256=reservation["receipt_sha256"],
        created_at_unix_ns=101,
    )
    staged = transaction_store.stage(
        adapter_tensors=post_adapter,
        optimizer_tensors=post_optimizer,
        pending_trainer_step=pending,
    )
    assert staged.pending_step["pre_measurement_sha256"] == intent["receipt_sha256"]
    assert staged.pending_step["reservation_sha256"] == reservation["receipt_sha256"]
    assert (
        store.reconcile_interrupted_admissions(
            update_journal=journal,
            live_adapter_tensors=post_adapter,
            live_optimizer_tensors=post_optimizer,
            observed_policy_sha256=pending["policy_after_sha256"],
            reconciled_at_unix_ns=102,
        )
        == ()
    )
    store.assert_no_orphans()

    intent_path = (
        transaction_root
        / "pre-measurements"
        / "entries"
        / f"seq-{0:08d}-{admission}"
        / "00000000-intent"
        / "pre-measurement.json"
    )
    original = json.loads(intent_path.read_bytes())
    original["recorded_at_unix_ns"] = 999
    os.chmod(intent_path, 0o600)
    intent_path.write_bytes(_canonical_with_floats(original))
    os.chmod(intent_path, 0o400)
    with pytest.raises(
        VerifiedTransitionTransactionError,
        match="stage_pre_measurement_unavailable",
    ):
        transaction_store.load(
            sequence=0,
            admission_sha256=admission,
            load_tensors=False,
        )
