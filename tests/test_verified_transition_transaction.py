"""Crash-point and tamper tests for verified-transition trainer custody."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from core.learning import verified_transition_transaction as transaction_module  # noqa: E402
from core.learning.grpo import group_advantages  # noqa: E402
from core.learning.verified_transition_transaction import (  # noqa: E402
    TrainerCheckpointEvidence,
    VerifiedTransitionTransactionError,
    VerifiedTransitionTransactionStore,
    build_pending_trainer_step,
    build_trainer_step_static,
    build_transaction_trainer_step,
    validate_pending_trainer_step,
)


def _sha(character: str) -> str:
    return character * 64


def _canonical(value, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return payload + (b"\n" if newline else b"")


def _seal(value, *, newline: bool = False):
    result = dict(value)
    result["receipt_sha256"] = hashlib.sha256(
        _canonical(result, newline=newline)
    ).hexdigest()
    return result


def _update_receipt():
    return _seal(
        {
            "schema": "aura.verified_transition.update_receipt.v1",
            "group_admission_sha256": _sha("a"),
            "reservation_sha256": _sha("1"),
            "commit_sha256": _sha("2"),
            "objective_record_sha256": _sha("3"),
            "objective_receipt_sha256": _sha("4"),
            "policy_before_sha256": _sha("b"),
            "policy_after_sha256": _sha("c"),
            "optimizer_update_count": 1,
            "reserved_at_unix_ns": 10,
            "committed_at_unix_ns": 20,
        }
    )


def _terminal_receipt(update):
    return _seal(
        {
            "schema": "aura.verified_transition.campaign_group_terminal.v2",
            "campaign_manifest_sha256": _sha("d"),
            "sequence": 0,
            "group_id": "group-0",
            "group_manifest_sha256": _sha("e"),
            "group_start_sha256": _sha("5"),
            "status": "updated",
            "reward_receipt_sha256": _sha("f"),
            "group_admission_sha256": _sha("a"),
            "update_receipt_sha256": update["receipt_sha256"],
            "terminal_reason": "optimizer_update_committed",
            "finished_at_unix_ns": 20,
        }
    )


def _causal_terminal_receipt(update):
    return _seal(
        {
            "schema": "aura.verified_transition.causal_group_terminal.v1",
            "campaign_manifest_sha256": _sha("d"),
            "campaign_schedule_root_sha256": _sha("7"),
            "sequence": 0,
            "group_id": "group-0",
            "group_manifest_sha256": _sha("e"),
            "group_start_sha256": _sha("5"),
            "status": "updated",
            "reward_receipt_sha256": _sha("f"),
            "group_admission_sha256": _sha("a"),
            "update_receipt_sha256": update["receipt_sha256"],
            "policy_before_sha256": _sha("b"),
            "policy_after_sha256": _sha("c"),
            "terminal_reason": "optimizer_update_committed",
            "finished_at_unix_ns": 20,
        }
    )


def _evidence():
    update = _update_receipt()
    terminal = _terminal_receipt(update)
    static = build_trainer_step_static(
        samples=[
            {"schema": "sample.v1", "sample": 0},
            {"schema": "sample.v1", "sample": 1},
        ],
        structured_rewards=[1.0, 0.0],
        optimizer_admission_reason="admitted",
        answer_channel={"correct": 1, "completions": 2},
        advantage_report=group_advantages([1.0, 0.0]),
    )
    trainer_step = _seal(
        {
            "schema": "aura.verified_transition.trainer_step.v1",
            "step": 1,
            "campaign_sequence": 0,
            "task_id": "task-0",
            "sample_seed": 7,
            "execution_spec_sha256": _sha("6"),
            "samples": static["samples"],
            "structured_rewards": static["structured_rewards"],
            "group_manifest_sha256": _sha("e"),
            "reward_receipt_sha256": _sha("f"),
            "group_admission_sha256": _sha("a"),
            "update_receipt_sha256": update["receipt_sha256"],
            "optimizer_admission_reason": static[
                "optimizer_admission_reason"
            ],
            "answer_channel": static["answer_channel"],
            "advantage_report": static["advantage_report"],
            "step_kind": "verified_optimizer_update",
            "policy_before_sha256": _sha("b"),
            "policy_after_sha256": _sha("c"),
            "update": update,
            "terminal": terminal,
        },
        newline=True,
    )
    pending = build_pending_trainer_step(
        sequence=0,
        trainer_step=1,
        task_id="task-0",
        trainer_sample_seed=7,
        execution_spec_sha256=_sha("6"),
        campaign_manifest_sha256=_sha("d"),
        campaign_schedule_root_sha256=_sha("7"),
        group_manifest_sha256=_sha("e"),
        group_admission_sha256=_sha("a"),
        reward_receipt_sha256=_sha("f"),
        policy_before_sha256=_sha("b"),
        policy_after_sha256=_sha("c"),
        trainer_step_static=static,
        created_at_unix_ns=1,
    )
    return update, terminal, trainer_step, pending


def _tensor_maps():
    return (
        {
            "model.layers.1.lora_a": mx.array([[1.0, 2.0], [3.0, 4.0]]),
            "model.layers.1.lora_b": mx.array([[0.5, -0.5]]),
        },
        {
            "state.step": mx.array(1),
            "state.model.layers.1.lora_a.m": mx.array([[0.1, 0.2], [0.3, 0.4]]),
        },
    )


def _stage(tmp_path):
    store = VerifiedTransitionTransactionStore.open(tmp_path / "transactions")
    update, terminal, trainer_step, pending = _evidence()
    adapters, optimizer = _tensor_maps()
    loaded = store.stage(
        adapter_tensors=adapters,
        optimizer_tensors=optimizer,
        pending_trainer_step=pending,
    )
    return store, loaded, update, terminal, trainer_step, pending


def _checkpoint(loaded, trainer_step):
    document = {
        "schema": "aura.grpo_checkpoint.v2",
        "checkpoint_id": "step-00000001-" + "1" * 32,
        "created_unix": 1.0,
        "protocol_sha256": _sha("2"),
        "dataset_sha256": _sha("3"),
        "step": 1,
        "curriculum": {},
        "telemetry": {},
        "last_step_committed": True,
        "history": [{"step": 0, "overall": 0.25}],
        "baseline_eval": None,
        "calibration": None,
        "elapsed_training_s": 1.0,
        "invocation_count": 1,
        "rng_strategy": "stateless_sha256_step_seeded_v1",
        "optimizer_updates": 1,
        "last_step_kind": "verified_optimizer_update",
        "execution_mode": "recurrent",
        "execution_spec_sha256": _sha("6"),
        "step_receipts": [trainer_step],
        "adapter": {
            "path": "adapter.safetensors",
            "sha256": loaded.stage["adapter"]["sha256"],
            "size_bytes": loaded.stage["adapter"]["size_bytes"],
        },
        "optimizer": {
            "path": "optimizer.safetensors",
            "sha256": loaded.stage["optimizer"]["sha256"],
            "size_bytes": loaded.stage["optimizer"]["size_bytes"],
        },
    }
    return TrainerCheckpointEvidence(
        document=document,
        artifact_sha256=hashlib.sha256(_canonical(document, newline=True)).hexdigest(),
    )


def _rewrite_sealed(path: Path, mutate):
    os.chmod(path, 0o600)
    document = json.loads(path.read_text(encoding="ascii"))
    mutate(document)
    document.pop("receipt_sha256", None)
    document["receipt_sha256"] = hashlib.sha256(_canonical(document)).hexdigest()
    path.write_bytes(_canonical(document))
    os.chmod(path, 0o400)


def test_pending_step_is_strictly_sealed_and_sequence_bound():
    *_, pending = _evidence()
    assert validate_pending_trainer_step(pending) == pending
    forged = dict(pending)
    forged["trainer_step"] = 2
    with pytest.raises(VerifiedTransitionTransactionError, match="digest_mismatch"):
        validate_pending_trainer_step(forged)


def test_stage_round_trip_preserves_exact_flat_mlx_tensor_maps(tmp_path):
    store, loaded, *_ = _stage(tmp_path)
    replayed = store.load(sequence=0, admission_sha256=_sha("a"))
    assert replayed is not None
    assert replayed.transaction_dir == loaded.transaction_dir
    assert set(replayed.adapter_tensors or {}) == {
        "model.layers.1.lora_a",
        "model.layers.1.lora_b",
    }
    assert set(replayed.optimizer_tensors or {}) == {
        "state.step",
        "state.model.layers.1.lora_a.m",
    }
    assert bool(
        mx.array_equal(
            replayed.adapter_tensors["model.layers.1.lora_a"],
            mx.array([[1.0, 2.0], [3.0, 4.0]]),
        )
    )
    assert int(replayed.optimizer_tensors["state.step"]) == 1


def test_inventory_is_ordered_and_rejects_unknown_transaction_entries(tmp_path):
    store, loaded, *_ = _stage(tmp_path)
    assert store.inventory(load_tensors=False)[0].transaction_dir == (
        loaded.transaction_dir
    )
    unexpected = store.transactions / "operator-notes"
    unexpected.mkdir(mode=0o700)
    with pytest.raises(
        VerifiedTransitionTransactionError, match="inventory_name_invalid"
    ):
        store.inventory(load_tensors=False)


def test_staged_generation_survives_abrupt_process_exit(tmp_path):
    root = tmp_path / "subprocess-transactions"
    script = textwrap.dedent(
        """
        import os
        import sys

        import mlx.core as mx

        from core.learning.grpo import group_advantages
        from core.learning.verified_transition_transaction import (
            VerifiedTransitionTransactionStore,
            build_pending_trainer_step,
            build_trainer_step_static,
        )

        sha = lambda character: character * 64
        static = build_trainer_step_static(
            samples=[{"sample": 0}, {"sample": 1}],
            structured_rewards=[1.0, 0.0],
            optimizer_admission_reason="admitted",
            answer_channel={"correct_fraction": 0.5},
            advantage_report=group_advantages([1.0, 0.0]),
        )
        pending = build_pending_trainer_step(
            sequence=0,
            trainer_step=1,
            task_id="task-0",
            trainer_sample_seed=7,
            execution_spec_sha256=sha("6"),
            campaign_manifest_sha256=sha("d"),
            campaign_schedule_root_sha256=sha("7"),
            group_manifest_sha256=sha("e"),
            group_admission_sha256=sha("a"),
            reward_receipt_sha256=sha("f"),
            policy_before_sha256=sha("b"),
            policy_after_sha256=sha("c"),
            trainer_step_static=static,
            created_at_unix_ns=1,
        )
        store = VerifiedTransitionTransactionStore.open(sys.argv[1])
        store.stage(
            adapter_tensors={"model.lora_a": mx.array([[1.0]])},
            optimizer_tensors={"state.step": mx.array(1)},
            pending_trainer_step=pending,
        )
        os._exit(73)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 73, completed.stderr

    inventory = VerifiedTransitionTransactionStore.open(root).inventory(
        load_tensors=True
    )
    assert len(inventory) == 1
    assert inventory[0].pending_step["policy_after_sha256"] == _sha("c")
    assert inventory[0].adapter_tensors is not None
    assert inventory[0].optimizer_tensors is not None


def test_transaction_reconstructs_exact_trainer_step_after_terminal(tmp_path):
    store, loaded, update, terminal, trainer_step, _pending = _stage(tmp_path)
    store.record_update_commit(
        sequence=0, admission_sha256=_sha("a"), update_receipt=update
    )
    store.record_campaign_terminal(
        sequence=0, admission_sha256=_sha("a"), terminal_receipt=terminal
    )
    complete = store.load(sequence=0, admission_sha256=_sha("a"))
    assert complete is not None
    assert build_transaction_trainer_step(complete) == trainer_step


def test_stage_is_idempotent_only_for_the_exact_same_tensors(tmp_path):
    store, first, *_rest, pending = _stage(tmp_path)
    adapters, optimizer = _tensor_maps()
    second = store.stage(
        adapter_tensors=adapters,
        optimizer_tensors=optimizer,
        pending_trainer_step=pending,
    )
    assert second.stage["receipt_sha256"] == first.stage["receipt_sha256"]
    adapters["model.layers.1.lora_a"] = mx.zeros((2, 2))
    with pytest.raises(
        VerifiedTransitionTransactionError, match="adapter_identity_conflict"
    ):
        store.stage(
            adapter_tensors=adapters,
            optimizer_tensors=optimizer,
            pending_trainer_step=pending,
        )


def test_nested_optimizer_state_must_be_flattened_before_stage(tmp_path):
    store = VerifiedTransitionTransactionStore.open(tmp_path / "transactions")
    *_, pending = _evidence()
    adapters, _optimizer = _tensor_maps()
    with pytest.raises(VerifiedTransitionTransactionError, match="not_flat"):
        store.stage(
            adapter_tensors=adapters,
            optimizer_tensors={"state": {"step": mx.array(1)}},
            pending_trainer_step=pending,
        )


def test_reconcile_classifies_every_required_kill_point(tmp_path):
    store = VerifiedTransitionTransactionStore.open(tmp_path / "transactions")
    before = store.reconcile(sequence=0, admission_sha256=_sha("a"))
    assert before.classification == "before_stage"
    assert before.restore_staged_tensors is False

    store, loaded, update, terminal, trainer_step, _pending = _stage(tmp_path)
    after_stage = store.reconcile(sequence=0, admission_sha256=_sha("a"))
    assert after_stage.classification == "after_stage"
    assert after_stage.restore_staged_tensors is True

    store.record_update_commit(
        sequence=0, admission_sha256=_sha("a"), update_receipt=update
    )
    assert (
        store.reconcile(sequence=0, admission_sha256=_sha("a")).classification
        == "after_update_commit"
    )

    store.record_campaign_terminal(
        sequence=0, admission_sha256=_sha("a"), terminal_receipt=terminal
    )
    assert (
        store.reconcile(sequence=0, admission_sha256=_sha("a")).classification
        == "after_campaign_terminal"
    )

    store.record_trainer_checkpoint(
        sequence=0,
        admission_sha256=_sha("a"),
        checkpoint=_checkpoint(loaded, trainer_step),
    )
    complete = store.reconcile(sequence=0, admission_sha256=_sha("a"))
    assert complete.classification == "after_trainer_checkpoint"
    assert complete.restore_staged_tensors is False


def test_causal_campaign_terminal_is_bound_to_staged_policy_lineage(tmp_path):
    store, loaded, update, _terminal, _trainer_step, _pending = _stage(tmp_path)
    store.record_update_commit(
        sequence=0, admission_sha256=_sha("a"), update_receipt=update
    )
    terminal = _causal_terminal_receipt(update)

    store.record_campaign_terminal(
        sequence=0,
        admission_sha256=_sha("a"),
        terminal_receipt=terminal,
    )

    replayed = store.load(
        sequence=0, admission_sha256=_sha("a"), load_tensors=False
    )
    assert replayed is not None
    assert replayed.events[-1]["evidence"] == terminal
    assert loaded.pending_step["campaign_schedule_root_sha256"] == _sha("7")


def test_reconcile_recovers_external_publications_missed_before_local_marker(tmp_path):
    store, loaded, update, terminal, trainer_step, _pending = _stage(tmp_path)
    checkpoint = _checkpoint(loaded, trainer_step)
    observed = []

    result = store.reconcile(
        sequence=0,
        admission_sha256=_sha("a"),
        load_update_receipt=lambda: observed.append("update") or update,
        load_campaign_terminal=lambda: observed.append("terminal") or terminal,
        load_trainer_checkpoint=lambda: observed.append("checkpoint") or checkpoint,
    )

    assert observed == ["update", "terminal", "checkpoint"]
    assert result.classification == "after_trainer_checkpoint"
    assert [event["kind"] for event in store.load(
        sequence=0, admission_sha256=_sha("a"), load_tensors=False
    ).events] == ["update_commit", "campaign_terminal", "trainer_checkpoint"]


def test_publication_callbacks_run_in_order_and_are_idempotently_recorded(tmp_path):
    store, loaded, update, terminal, trainer_step, _pending = _stage(tmp_path)
    calls = []
    store.publish_update_commit(
        sequence=0,
        admission_sha256=_sha("a"),
        publish=lambda: calls.append("update") or update,
    )
    store.publish_campaign_terminal(
        sequence=0,
        admission_sha256=_sha("a"),
        publish=lambda: calls.append("terminal") or terminal,
    )
    store.publish_trainer_checkpoint(
        sequence=0,
        admission_sha256=_sha("a"),
        publish=lambda: calls.append("checkpoint") or _checkpoint(loaded, trainer_step),
    )
    assert calls == ["update", "terminal", "checkpoint"]


def test_published_generations_and_artifacts_are_read_only(tmp_path):
    _store, loaded, *_ = _stage(tmp_path)
    stage = loaded.transaction_dir / "generations" / "00000000-staged"
    assert stage.stat().st_mode & 0o777 == 0o500
    assert all(path.stat().st_mode & 0o777 == 0o400 for path in stage.iterdir())


def test_unpublished_temporary_stage_is_classified_before_stage(tmp_path):
    store = VerifiedTransitionTransactionStore.open(tmp_path / "transactions")
    transaction = store.transactions / f"seq-00000000-{_sha('a')}"
    generations = transaction / "generations"
    generations.mkdir(parents=True, mode=0o700)
    os.chmod(transaction, 0o700)
    os.chmod(generations, 0o700)
    partial = generations / (".tmp-00000000-staged-" + "d" * 32)
    partial.mkdir(mode=0o700)
    (partial / "adapter.safetensors").write_bytes(b"partial")

    result = store.reconcile(sequence=0, admission_sha256=_sha("a"))
    assert result.classification == "before_stage"
    assert store.inventory() == ()
    assert not transaction.exists()


def test_update_receipt_rebinding_is_rejected(tmp_path):
    store, _loaded, update, *_ = _stage(tmp_path)
    forged = copy.deepcopy(update)
    forged["group_admission_sha256"] = _sha("9")
    forged.pop("receipt_sha256")
    forged = _seal(forged)
    with pytest.raises(VerifiedTransitionTransactionError, match="binding_mismatch"):
        store.record_update_commit(
            sequence=0, admission_sha256=_sha("a"), update_receipt=forged
        )


def test_resealed_update_receipt_with_unknown_field_is_rejected(tmp_path):
    store, _loaded, update, *_ = _stage(tmp_path)
    forged = dict(update)
    forged.pop("receipt_sha256")
    forged["unreviewed_extension"] = True
    forged = _seal(forged)
    with pytest.raises(VerifiedTransitionTransactionError, match="schema_invalid"):
        store.record_update_commit(
            sequence=0, admission_sha256=_sha("a"), update_receipt=forged
        )


def test_campaign_terminal_rebinding_is_rejected(tmp_path):
    store, _loaded, update, terminal, *_ = _stage(tmp_path)
    store.record_update_commit(
        sequence=0, admission_sha256=_sha("a"), update_receipt=update
    )
    forged = copy.deepcopy(terminal)
    forged["reward_receipt_sha256"] = _sha("8")
    forged.pop("receipt_sha256")
    forged = _seal(forged)
    with pytest.raises(VerifiedTransitionTransactionError, match="binding_mismatch"):
        store.record_campaign_terminal(
            sequence=0, admission_sha256=_sha("a"), terminal_receipt=forged
        )


def test_trainer_checkpoint_must_bind_step_and_exact_staged_tensor_files(tmp_path):
    store, loaded, update, terminal, trainer_step, _pending = _stage(tmp_path)
    store.record_update_commit(
        sequence=0, admission_sha256=_sha("a"), update_receipt=update
    )
    store.record_campaign_terminal(
        sequence=0, admission_sha256=_sha("a"), terminal_receipt=terminal
    )
    checkpoint = _checkpoint(loaded, trainer_step)
    forged_document = copy.deepcopy(checkpoint.document)
    forged_document["adapter"]["sha256"] = _sha("9")
    forged = TrainerCheckpointEvidence(
        document=forged_document,
        artifact_sha256=hashlib.sha256(
            _canonical(forged_document, newline=True)
        ).hexdigest(),
    )
    with pytest.raises(VerifiedTransitionTransactionError, match="adapter_binding"):
        store.record_trainer_checkpoint(
            sequence=0, admission_sha256=_sha("a"), checkpoint=forged
        )


def test_trainer_checkpoint_requires_complete_verified_resume_state(tmp_path):
    store, loaded, update, terminal, trainer_step, _pending = _stage(tmp_path)
    store.record_update_commit(
        sequence=0, admission_sha256=_sha("a"), update_receipt=update
    )
    store.record_campaign_terminal(
        sequence=0, admission_sha256=_sha("a"), terminal_receipt=terminal
    )
    checkpoint = _checkpoint(loaded, trainer_step)
    incomplete = copy.deepcopy(checkpoint.document)
    incomplete.pop("protocol_sha256")
    forged = TrainerCheckpointEvidence(
        document=incomplete,
        artifact_sha256=hashlib.sha256(
            _canonical(incomplete, newline=True)
        ).hexdigest(),
    )

    with pytest.raises(
        VerifiedTransitionTransactionError,
        match="checkpoint_state_invalid",
    ):
        store.record_trainer_checkpoint(
            sequence=0,
            admission_sha256=_sha("a"),
            checkpoint=forged,
        )


@pytest.mark.parametrize("role", ["adapter", "optimizer"])
def test_tensor_byte_tamper_is_detected_before_restore(tmp_path, role):
    store, loaded, *_ = _stage(tmp_path)
    path = (
        loaded.transaction_dir
        / "generations"
        / "00000000-staged"
        / f"{role}.safetensors"
    )
    os.chmod(path, 0o600)
    path.write_bytes(path.read_bytes() + b"tamper")
    os.chmod(path, 0o400)
    with pytest.raises(VerifiedTransitionTransactionError, match=f"{role}_size_mismatch"):
        store.load(sequence=0, admission_sha256=_sha("a"))


def test_resealed_path_escape_is_rejected(tmp_path):
    store, loaded, *_ = _stage(tmp_path)
    generation = (
        loaded.transaction_dir
        / "generations"
        / "00000000-staged"
        / "generation.json"
    )
    _rewrite_sealed(
        generation,
        lambda document: document["adapter"].update({"path": "../adapter.safetensors"}),
    )
    with pytest.raises(VerifiedTransitionTransactionError, match="path_invalid"):
        store.load(sequence=0, admission_sha256=_sha("a"))


def test_symlinked_tensor_is_rejected_even_when_target_bytes_match(tmp_path):
    store, loaded, *_ = _stage(tmp_path)
    stage = loaded.transaction_dir / "generations" / "00000000-staged"
    adapter = stage / "adapter.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    replacement.write_bytes(adapter.read_bytes())
    os.chmod(stage, 0o700)
    adapter.unlink()
    adapter.symlink_to(replacement)
    os.chmod(stage, 0o500)
    try:
        with pytest.raises(VerifiedTransitionTransactionError, match="symlink_rejected"):
            store.load(sequence=0, admission_sha256=_sha("a"))
    finally:
        os.chmod(stage, 0o700)
        adapter.unlink(missing_ok=True)
        for artifact in stage.iterdir():
            os.chmod(artifact, 0o600)
        os.chmod(stage.parent, 0o700)
        os.chmod(loaded.transaction_dir, 0o700)


def test_event_generation_tamper_breaks_hash_chain(tmp_path):
    store, loaded, update, *_ = _stage(tmp_path)
    store.record_update_commit(
        sequence=0, admission_sha256=_sha("a"), update_receipt=update
    )
    generation = (
        loaded.transaction_dir
        / "generations"
        / "00000001-update-commit"
        / "generation.json"
    )
    os.chmod(generation, 0o600)
    document = json.loads(generation.read_text(encoding="ascii"))
    document["previous_generation_sha256"] = _sha("9")
    generation.write_bytes(_canonical(document))
    os.chmod(generation, 0o400)
    with pytest.raises(VerifiedTransitionTransactionError, match="digest_mismatch"):
        store.load(sequence=0, admission_sha256=_sha("a"), load_tensors=False)


def test_external_commit_without_durable_stage_fails_closed(tmp_path):
    store = VerifiedTransitionTransactionStore.open(tmp_path / "transactions")
    update = _update_receipt()
    with pytest.raises(
        VerifiedTransitionTransactionError, match="external_evidence_without_stage"
    ):
        store.reconcile(
            sequence=0,
            admission_sha256=_sha("a"),
            load_update_receipt=lambda: update,
        )


def test_external_terminal_without_observable_update_commit_fails_closed(tmp_path):
    store, _loaded, _update, terminal, *_ = _stage(tmp_path)
    with pytest.raises(
        VerifiedTransitionTransactionError, match="terminal_without_update"
    ):
        store.reconcile(
            sequence=0,
            admission_sha256=_sha("a"),
            load_update_receipt=lambda: None,
            load_campaign_terminal=lambda: terminal,
        )


def test_transaction_root_symlink_is_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(VerifiedTransitionTransactionError, match="symlink_component"):
        VerifiedTransitionTransactionStore.open(alias)


def test_source_contains_no_pickle_or_mutable_latest_pointer():
    source = inspect.getsource(transaction_module)
    assert "import pickle" not in source
    assert "pickle." not in source
    assert "latest.json" not in source
