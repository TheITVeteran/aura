"""Crash-point and tamper tests for rejected-group trainer custody."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from core.learning.grpo import group_advantages
from core.learning.verified_transition_rejection_transaction import (
    VerifiedTransitionRejectionTransactionCoordinator,
    VerifiedTransitionRejectionTransactionError,
    VerifiedTransitionRejectionTransactionStore,
    build_rejected_transaction_trainer_step,
    build_rejection_intent,
)
from core.learning.verified_transition_trainer import (
    validate_verified_transition_step_receipt,
)
from core.learning.verified_transition_transaction import build_trainer_step_static


def _sha(character: str) -> str:
    return character * 64


def _canonical(value: object, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return payload + (b"\n" if newline else b"")


def _seal(value: dict) -> dict:
    document = dict(value)
    document["receipt_sha256"] = hashlib.sha256(_canonical(document)).hexdigest()
    return document


def _static() -> dict:
    return build_trainer_step_static(
        samples=[
            {"schema": "sample.v1", "sample": 0},
            {"schema": "sample.v1", "sample": 1},
        ],
        structured_rewards=[-1.0, -0.5],
        optimizer_admission_reason="right_to_wrong_present",
        answer_channel={"correct": 0, "completions": 2},
        advantage_report=group_advantages([-1.0, -0.5]),
    )


def _intent() -> dict:
    return build_rejection_intent(
        sequence=0,
        trainer_step=1,
        task_id="task-0",
        trainer_sample_seed=7,
        execution_spec_sha256=_sha("6"),
        campaign_manifest_sha256=_sha("d"),
        campaign_schedule_root_sha256=_sha("7"),
        group_manifest_sha256=_sha("e"),
        reward_receipt_sha256=_sha("f"),
        policy_sha256=_sha("b"),
        trainer_step_static=_static(),
        created_at_unix_ns=1,
    )


def _terminal(*, finished_at_unix_ns: int | None = None) -> dict:
    return _seal(
        {
            "schema": "aura.verified_transition.causal_group_terminal.v1",
            "campaign_manifest_sha256": _sha("d"),
            "campaign_schedule_root_sha256": _sha("7"),
            "sequence": 0,
            "group_id": "group-0",
            "group_manifest_sha256": _sha("e"),
            "group_start_sha256": _sha("5"),
            "status": "rejected",
            "reward_receipt_sha256": _sha("f"),
            "group_admission_sha256": None,
            "update_receipt_sha256": None,
            "policy_before_sha256": _sha("b"),
            "policy_after_sha256": _sha("b"),
            "terminal_reason": "right_to_wrong_present",
            "finished_at_unix_ns": (
                time.time_ns()
                if finished_at_unix_ns is None
                else finished_at_unix_ns
            ),
        }
    )


def _checkpoint_directory(tmp_path: Path, trainer_step: dict) -> Path:
    directory = tmp_path / ("step-00000001-" + "1" * 32)
    directory.mkdir(mode=0o700)
    document = {
        "schema": "aura.grpo_checkpoint.v2",
        "checkpoint_id": directory.name,
        "created_unix": 1.0,
        "protocol_sha256": _sha("2"),
        "dataset_sha256": _sha("3"),
        "step": 1,
        "curriculum": {},
        "telemetry": {},
        "last_step_committed": True,
        "history": [],
        "baseline_eval": None,
        "calibration": None,
        "elapsed_training_s": 1.0,
        "invocation_count": 1,
        "rng_strategy": "stateless_sha256_step_seeded_v1",
        "optimizer_updates": 0,
        "last_step_kind": "verified_rejected_group",
        "execution_mode": "recurrent",
        "execution_spec_sha256": _sha("6"),
        "step_receipts": [trainer_step],
        "adapter": {
            "path": "adapter.safetensors",
            "sha256": _sha("8"),
            "size_bytes": 1,
        },
        "optimizer": {
            "path": "optimizer.safetensors",
            "sha256": _sha("9"),
            "size_bytes": 1,
        },
    }
    (directory / "complete.json").write_bytes(_canonical(document, newline=True))
    (directory / "complete.json").chmod(0o600)
    return directory


def _stage(tmp_path: Path):
    store = VerifiedTransitionRejectionTransactionStore.open(tmp_path / "transactions")
    intent = _intent()
    loaded = store.stage(intent)
    return store, loaded, intent


def test_rejection_chain_is_durable_and_reconstructs_exact_trainer_step(
    tmp_path: Path,
) -> None:
    store, staged, intent = _stage(tmp_path)
    assert staged.intent == intent
    assert staged.events == ()

    store.record_campaign_terminal(
        sequence=0,
        reward_sha256=_sha("f"),
        terminal_receipt=_terminal(),
    )
    terminalized = store.load(sequence=0, reward_sha256=_sha("f"))
    assert terminalized is not None
    trainer_step = validate_verified_transition_step_receipt(
        build_rejected_transaction_trainer_step(terminalized),
        group_size=2,
        execution_spec_sha256=_sha("6"),
    )
    checkpoint = _checkpoint_directory(tmp_path, trainer_step)
    store.record_trainer_checkpoint(
        sequence=0,
        reward_sha256=_sha("f"),
        checkpoint_dir=checkpoint,
    )

    sealed = store.load(sequence=0, reward_sha256=_sha("f"))
    assert sealed is not None
    assert [event["kind"] for event in sealed.events] == [
        "campaign_terminal",
        "trainer_checkpoint",
    ]
    assert store.inventory() == (sealed,)


def test_crash_after_intent_reloads_without_advancing_terminal(tmp_path: Path) -> None:
    store, _staged, intent = _stage(tmp_path)
    restarted = VerifiedTransitionRejectionTransactionStore.open(
        tmp_path / "transactions"
    )
    loaded = restarted.load(sequence=0, reward_sha256=_sha("f"))
    assert loaded is not None
    assert loaded.intent == intent
    assert loaded.events == ()


def test_checkpoint_cannot_precede_campaign_terminal(tmp_path: Path) -> None:
    store, staged, _intent_document = _stage(tmp_path)
    terminalized = type(staged)(
        staged.transaction_dir,
        staged.intent,
        (
            {
                "kind": "campaign_terminal",
                "evidence": _terminal(),
            },
        ),
    )
    trainer_step = build_rejected_transaction_trainer_step(terminalized)
    checkpoint = _checkpoint_directory(tmp_path, trainer_step)
    with pytest.raises(
        VerifiedTransitionRejectionTransactionError,
        match="terminal_event_missing",
    ):
        store.record_trainer_checkpoint(
            sequence=0,
            reward_sha256=_sha("f"),
            checkpoint_dir=checkpoint,
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("policy_after_sha256", _sha("c"), "terminal_binding_mismatch"),
        ("group_admission_sha256", _sha("a"), "terminal_binding_mismatch"),
        ("status", "updated", "terminal_binding_mismatch"),
    ],
)
def test_terminal_substitution_is_rejected(
    tmp_path: Path, field: str, value: object, error: str
) -> None:
    store, _staged, _intent_document = _stage(tmp_path)
    terminal = _terminal()
    terminal[field] = value
    terminal.pop("receipt_sha256")
    terminal = _seal(terminal)
    with pytest.raises(VerifiedTransitionRejectionTransactionError, match=error):
        store.record_campaign_terminal(
            sequence=0,
            reward_sha256=_sha("f"),
            terminal_receipt=terminal,
        )


def test_resealed_unknown_terminal_field_is_rejected(tmp_path: Path) -> None:
    store, _staged, _intent_document = _stage(tmp_path)
    terminal = _terminal()
    terminal.pop("receipt_sha256")
    terminal["untrusted"] = True
    terminal = _seal(terminal)
    with pytest.raises(
        VerifiedTransitionRejectionTransactionError,
        match="terminal_schema_invalid",
    ):
        store.record_campaign_terminal(
            sequence=0,
            reward_sha256=_sha("f"),
            terminal_receipt=terminal,
        )


def test_checkpoint_cannot_substitute_static_rejection_receipt(
    tmp_path: Path,
) -> None:
    store, _staged, _intent_document = _stage(tmp_path)
    store.record_campaign_terminal(
        sequence=0,
        reward_sha256=_sha("f"),
        terminal_receipt=_terminal(),
    )
    transaction = store.load(sequence=0, reward_sha256=_sha("f"))
    assert transaction is not None
    trainer_step = build_rejected_transaction_trainer_step(transaction)
    trainer_step["answer_channel"] = {"correct": 2, "completions": 2}
    trainer_step.pop("receipt_sha256")
    trainer_step["receipt_sha256"] = hashlib.sha256(
        _canonical(trainer_step, newline=True)
    ).hexdigest()
    checkpoint = _checkpoint_directory(tmp_path, trainer_step)
    with pytest.raises(
        VerifiedTransitionRejectionTransactionError,
        match="step_receipt_invalid",
    ):
        store.record_trainer_checkpoint(
            sequence=0,
            reward_sha256=_sha("f"),
            checkpoint_dir=checkpoint,
        )


def test_coordinator_orders_intent_before_terminal(tmp_path: Path) -> None:
    store = VerifiedTransitionRejectionTransactionStore.open(tmp_path / "transactions")
    coordinator = VerifiedTransitionRejectionTransactionCoordinator(
        store=store,
        sequence=0,
        trainer_step=1,
        task_id="task-0",
        trainer_sample_seed=7,
        execution_spec_sha256=_sha("6"),
        campaign_manifest_sha256=_sha("d"),
        campaign_schedule_root_sha256=_sha("7"),
        group_manifest_sha256=_sha("e"),
        reward_receipt_sha256=_sha("f"),
        trainer_step_static=_static(),
    )
    with pytest.raises(
        VerifiedTransitionRejectionTransactionError,
        match="coordinator_not_staged",
    ):
        coordinator.record_campaign_terminal(_terminal())
    coordinator.stage_rejection(policy_sha256=_sha("b"))
    coordinator.record_campaign_terminal(_terminal())
    assert len(store.inventory()[0].events) == 1


def test_peer_writable_intent_and_event_tampering_fail_closed(tmp_path: Path) -> None:
    store, staged, _intent_document = _stage(tmp_path)
    intent_path = staged.transaction_dir / "00000000-intent.json"
    intent_path.chmod(0o644)
    with pytest.raises(
        VerifiedTransitionRejectionTransactionError,
        match="not_private_owned_file",
    ):
        store.load(sequence=0, reward_sha256=_sha("f"))

    intent_path.chmod(0o600)
    store.record_campaign_terminal(
        sequence=0,
        reward_sha256=_sha("f"),
        terminal_receipt=_terminal(),
    )
    event_path = staged.transaction_dir / "00000001-campaign-terminal.json"
    document = json.loads(event_path.read_text(encoding="ascii"))
    document["previous_receipt_sha256"] = _sha("0")
    document.pop("receipt_sha256")
    document = _seal(document)
    event_path.write_bytes(_canonical(document))
    event_path.chmod(0o600)
    with pytest.raises(
        VerifiedTransitionRejectionTransactionError,
        match="lineage_invalid",
    ):
        store.load(sequence=0, reward_sha256=_sha("f"))


def test_peer_writable_transaction_directory_is_rejected(tmp_path: Path) -> None:
    store, staged, _intent_document = _stage(tmp_path)
    staged.transaction_dir.chmod(0o755)
    with pytest.raises(
        VerifiedTransitionRejectionTransactionError,
        match="not_private_owned_directory",
    ):
        store.load(sequence=0, reward_sha256=_sha("f"))


def test_symlinked_root_component_is_rejected(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir(mode=0o700)
    link = tmp_path / "linked"
    os.symlink(actual, link)
    with pytest.raises(
        VerifiedTransitionRejectionTransactionError,
        match="symlink_rejected",
    ):
        VerifiedTransitionRejectionTransactionStore.open(link / "transactions")


def test_empty_crash_directory_is_cleaned_by_inventory(tmp_path: Path) -> None:
    store = VerifiedTransitionRejectionTransactionStore.open(tmp_path / "transactions")
    orphan = store.transactions / f"seq-00000000-{_sha('f')}"
    orphan.mkdir(mode=0o700)
    assert store.inventory() == ()
    assert not orphan.exists()


def test_symlink_transaction_is_rejected(tmp_path: Path) -> None:
    store = VerifiedTransitionRejectionTransactionStore.open(tmp_path / "transactions")
    target = tmp_path / "target"
    target.mkdir()
    link = store.transactions / f"seq-00000000-{_sha('f')}"
    os.symlink(target, link)
    with pytest.raises(
        VerifiedTransitionRejectionTransactionError,
        match="directory_name_invalid",
    ):
        store.inventory()
