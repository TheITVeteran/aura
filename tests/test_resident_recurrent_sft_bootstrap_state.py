from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

from core.learning.resident_recurrent_sft_bootstrap_authority import SAMPLER_NAME
from core.learning.resident_recurrent_sft_bootstrap_state import (
    BINDING_ROLES,
    ResidentSFTBootstrapStateError,
    inspect_checkpoint,
    load_checkpoint,
    order_sha256,
    save_checkpoint,
    validate_checkpoint_state,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _bindings() -> dict[str, str]:
    return {role: _sha(role) for role in BINDING_ROLES}


def _state(
    *,
    sequence: int = 1,
    step: int = 0,
    terminal: bool = False,
) -> dict[str, Any]:
    count = 3
    epoch, cursor = divmod(step, count)
    order = [2, 0, 1] if epoch % 2 == 0 else [1, 2, 0]
    return {
        **_bindings(),
        "checkpoint_sequence": sequence,
        "step": step,
        "optimizer_updates": step,
        "epoch": epoch,
        "cursor": cursor,
        "order": order,
        "order_sha256": order_sha256(
            order=order,
            seed=2026080107,
            epoch=epoch,
        ),
        "sampler": SAMPLER_NAME,
        "seed": 2026080107,
        "train_example_count": count,
        "validation_example_count": 2,
        "elapsed_training_s": float(step + 1),
        "invocation_count": 1,
        "sample_history_sha256": _sha(f"history:{step}"),
        "initial_adapter_sha256": _sha("initial-adapter"),
        "adapter_topology_sha256": _sha("adapter-topology"),
        "loss_trail": [],
        "validation_trail": [],
        "pending_losses": [],
        "baseline_validation": {"mean_loss": 2.0, "examples": 2},
        "last_step_committed": True,
        "terminal": terminal,
        "halt_reason": "max_steps" if terminal else None,
    }


def _adapter(value: float = 1.0) -> dict[str, mx.array]:
    return {"adapter.weight": mx.array([[value, value + 1.0]])}


def _optimizer(value: float = 0.1) -> dict[str, mx.array]:
    return {"state.m": mx.array([value, value + 0.1])}


def _canonical_write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="ascii",
    )


def test_checkpoint_state_requires_without_replacement_epoch_order() -> None:
    state = _state()
    state["order"] = [0, 0, 1]
    state["order_sha256"] = order_sha256(
        order=state["order"],
        seed=state["seed"],
        epoch=state["epoch"],
    )

    with pytest.raises(
        ResidentSFTBootstrapStateError,
        match="order_not_without_replacement",
    ):
        validate_checkpoint_state(state)


def test_checkpoint_state_rejects_position_digest_and_partial_update() -> None:
    invalid_position = _state(step=1)
    invalid_position["cursor"] = 2
    with pytest.raises(ResidentSFTBootstrapStateError, match="sample_position"):
        validate_checkpoint_state(invalid_position)

    invalid_digest = _state()
    invalid_digest["order_sha256"] = "f" * 64
    with pytest.raises(ResidentSFTBootstrapStateError, match="order_digest"):
        validate_checkpoint_state(invalid_digest)

    partial = _state(step=1)
    partial["last_step_committed"] = False
    with pytest.raises(ResidentSFTBootstrapStateError, match="partial_update"):
        validate_checkpoint_state(partial)


def test_checkpoint_state_requires_terminal_reason_equivalence() -> None:
    terminal_without_reason = _state(terminal=True)
    terminal_without_reason["halt_reason"] = None
    with pytest.raises(ResidentSFTBootstrapStateError, match="terminal_reason"):
        validate_checkpoint_state(terminal_without_reason)

    running_with_reason = _state()
    running_with_reason["halt_reason"] = "interrupted"
    with pytest.raises(ResidentSFTBootstrapStateError, match="terminal_reason"):
        validate_checkpoint_state(running_with_reason)


def test_save_inspect_and_load_exact_complete_generation(tmp_path: Path) -> None:
    generation = save_checkpoint(
        tmp_path / "run",
        adapter_tensors=_adapter(),
        optimizer_tensors=_optimizer(),
        state=_state(),
    )

    assert generation.name.startswith("sequence-00000001-step-00000000-")
    inspected = inspect_checkpoint(
        tmp_path / "run",
        expected_bindings=_bindings(),
    )
    loaded = load_checkpoint(tmp_path / "run", expected_bindings=_bindings())

    assert inspected.state["step"] == 0
    assert loaded.complete_sha256 == inspected.complete_sha256
    assert mx.array_equal(
        loaded.adapter_tensors["adapter.weight"],
        _adapter()["adapter.weight"],
    ).item()
    assert mx.array_equal(
        loaded.optimizer_tensors["state.m"],
        _optimizer()["state.m"],
    ).item()


def test_inspection_rejects_protocol_binding_drift(tmp_path: Path) -> None:
    save_checkpoint(
        tmp_path / "run",
        adapter_tensors=_adapter(),
        optimizer_tensors=_optimizer(),
        state=_state(),
    )
    changed = _bindings()
    changed["model_identity_sha256"] = _sha("different-model")

    with pytest.raises(ResidentSFTBootstrapStateError, match="protocol_binding"):
        inspect_checkpoint(tmp_path / "run", expected_bindings=changed)


def test_inspection_rejects_adapter_tamper(tmp_path: Path) -> None:
    generation = save_checkpoint(
        tmp_path / "run",
        adapter_tensors=_adapter(),
        optimizer_tensors=_optimizer(),
        state=_state(),
    )
    (generation / "adapter.safetensors").write_bytes(b"tampered")

    with pytest.raises(ResidentSFTBootstrapStateError, match="commitment_mismatch"):
        inspect_checkpoint(tmp_path / "run", expected_bindings=_bindings())


def test_inspection_rejects_complete_and_pointer_tamper(tmp_path: Path) -> None:
    generation = save_checkpoint(
        tmp_path / "run",
        adapter_tensors=_adapter(),
        optimizer_tensors=_optimizer(),
        state=_state(),
    )
    complete = json.loads((generation / "complete.json").read_text(encoding="ascii"))
    complete["state"]["elapsed_training_s"] = 999.0
    _canonical_write(generation / "complete.json", complete)
    with pytest.raises(ResidentSFTBootstrapStateError, match="complete_commitment"):
        inspect_checkpoint(tmp_path / "run", expected_bindings=_bindings())

    run_two = tmp_path / "run-two"
    save_checkpoint(
        run_two,
        adapter_tensors=_adapter(),
        optimizer_tensors=_optimizer(),
        state=_state(),
    )
    pointer_path = run_two / "latest.json"
    pointer = json.loads(pointer_path.read_text(encoding="ascii"))
    pointer["checkpoint_sequence"] = 2
    _canonical_write(pointer_path, pointer)
    with pytest.raises(ResidentSFTBootstrapStateError, match="sequence_mismatch"):
        inspect_checkpoint(run_two, expected_bindings=_bindings())


@pytest.mark.parametrize(
    ("sequence", "step", "error"),
    [
        (3, 1, "checkpoint_sequence"),
        (2, 2, "nonmonotonic_transition"),
        (2, 0, "nonmonotonic_transition"),
    ],
)
def test_save_rejects_skipped_or_nonmonotonic_updates(
    tmp_path: Path,
    sequence: int,
    step: int,
    error: str,
) -> None:
    run = tmp_path / "run"
    save_checkpoint(
        run,
        adapter_tensors=_adapter(),
        optimizer_tensors=_optimizer(),
        state=_state(),
    )

    with pytest.raises(ResidentSFTBootstrapStateError, match=error):
        save_checkpoint(
            run,
            adapter_tensors=_adapter(),
            optimizer_tensors=_optimizer(),
            state=_state(sequence=sequence, step=step),
        )


def test_same_step_terminal_checkpoint_preserves_tensors(tmp_path: Path) -> None:
    run = tmp_path / "run"
    save_checkpoint(
        run,
        adapter_tensors=_adapter(),
        optimizer_tensors=_optimizer(),
        state=_state(),
    )
    terminal = _state(sequence=2, terminal=True)
    save_checkpoint(
        run,
        adapter_tensors=_adapter(),
        optimizer_tensors=_optimizer(),
        state=terminal,
    )
    assert inspect_checkpoint(run, expected_bindings=_bindings()).state["terminal"]

    second_run = tmp_path / "second-run"
    save_checkpoint(
        second_run,
        adapter_tensors=_adapter(),
        optimizer_tensors=_optimizer(),
        state=_state(),
    )
    with pytest.raises(ResidentSFTBootstrapStateError, match="terminal_tensor_drift"):
        save_checkpoint(
            second_run,
            adapter_tensors=_adapter(7.0),
            optimizer_tensors=_optimizer(),
            state=terminal,
        )


def test_checkpoint_root_symlink_is_forbidden(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (run / "checkpoints").symlink_to(external, target_is_directory=True)

    with pytest.raises(ResidentSFTBootstrapStateError, match="root_symlink"):
        save_checkpoint(
            run,
            adapter_tensors=_adapter(),
            optimizer_tensors=_optimizer(),
            state=_state(),
        )


def test_inspection_rejects_checkpoint_root_replaced_by_symlink(tmp_path: Path) -> None:
    run = tmp_path / "run"
    save_checkpoint(
        run,
        adapter_tensors=_adapter(),
        optimizer_tensors=_optimizer(),
        state=_state(),
    )
    original = run / "checkpoint-generations"
    (run / "checkpoints").rename(original)
    (run / "checkpoints").symlink_to(original, target_is_directory=True)

    with pytest.raises(ResidentSFTBootstrapStateError, match="symlink_forbidden"):
        inspect_checkpoint(run, expected_bindings=_bindings())


def test_binding_role_set_is_exact() -> None:
    state = _state()
    del state["runtime_identity_sha256"]
    with pytest.raises(ResidentSFTBootstrapStateError, match="schema_invalid"):
        validate_checkpoint_state(state)

    extra = copy.deepcopy(_state())
    extra["unbound"] = "f" * 64
    with pytest.raises(ResidentSFTBootstrapStateError, match="schema_invalid"):
        validate_checkpoint_state(extra)
