"""Contracts for teacher-removed recurrent neural transition tissue."""

from __future__ import annotations

from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from core.brain.llm.latent_cortex.neural_transition_tissue import (  # noqa: E402
    NeuralTransitionTissue,
    execute_neural_action_program,
    load_neural_transition_tissue,
)
from core.brain.llm.latent_cortex.objective_program_verifier import (  # noqa: E402
    _resident_neural_transition_tissue,
    solve_objective_program,
)
from core.brain.llm.latent_cortex.typed_action_compiler import (  # noqa: E402
    compile_public_transition_program,
)
from core.brain.llm.latent_cortex.typed_transition_executor import (  # noqa: E402
    CertifiedTransitionExecutor,
)
from core.learning.neural_transition_training import (  # noqa: E402
    build_certified_transition_batch,
    train_and_write_neural_transition_artifact,
    train_neural_transition_tissue,
    transition_training_metrics,
)
from core.learning.recurrence_curriculum import modular_chain, nested_boolean  # noqa: E402
from tools.verify_neural_transition_tissue import (  # noqa: E402
    verify_neural_transition_artifact,
)


@pytest.fixture(scope="module")
def trained() -> NeuralTransitionTissue:
    tissue, receipt = train_neural_transition_tissue()
    assert receipt["initial_metrics"]["exact_accuracy"] < 1.0
    assert receipt["final_metrics"]["exact_accuracy"] == 1.0
    return tissue


def test_training_learns_all_3842_certified_primitives(trained: NeuralTransitionTissue) -> None:
    batch = build_certified_transition_batch()
    assert len(batch.boolean_keys) == 14
    assert len(batch.modular_keys) == 3_828
    assert transition_training_metrics(trained, batch)["exact_accuracy"] == 1.0


def test_fresh_deep_programs_compose_after_teacher_is_removed(
    trained: NeuralTransitionTissue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("certified teacher reached teacher-removed evaluation")

    monkeypatch.setattr(CertifiedTransitionExecutor, "execute", forbidden)
    observed = 0
    for generator in (nested_boolean, modular_chain):
        for depth in (1, 2, 4, 8, 16, 32):
            for seed in (196_001, 196_002, 196_003, 196_004):
                task = generator(depth, seed)
                program = compile_public_transition_program(task.prompt)
                execution = execute_neural_action_program(program, trained)
                assert execution.terminal_state == task.transition_trace.states[-1]
                assert execution.receipt()["teacher_available"] is False
                observed += depth
    assert observed == 504


def test_recurrence_action_and_state_controls_are_causal(
    trained: NeuralTransitionTissue,
) -> None:
    changed_by_t1 = 0
    changed_by_shuffle = 0
    changed_by_reset = 0
    for seed in range(196_010, 196_030):
        task = modular_chain(8, seed)
        program = compile_public_transition_program(task.prompt)
        full = execute_neural_action_program(program, trained)
        t1 = execute_neural_action_program(program, trained, max_steps=1)
        shuffled_actions = tuple(reversed(program.actions))
        shuffled = execute_neural_action_program(program, trained, actions=shuffled_actions)
        reset_state = program.initial_state
        for action in program.actions:
            reset_state = trained.transition(
                family=program.family,
                depth=program.depth,
                field_names=program.field_names,
                state=(reset_state[0], program.initial_state[1], 0),
                action_field_names=program.action_field_names,
                action=action,
            ).next_state
        changed_by_t1 += int(t1.states[-1][1] != full.terminal_state[1])
        changed_by_shuffle += int(shuffled.terminal_state[1] != full.terminal_state[1])
        changed_by_reset += int(reset_state[1] != full.terminal_state[1])
    assert changed_by_t1 >= 15
    assert changed_by_shuffle >= 12
    assert changed_by_reset >= 15


def test_untrained_sham_does_not_match_trained_tissue(
    trained: NeuralTransitionTissue,
) -> None:
    sham = NeuralTransitionTissue()
    trained_correct = 0
    sham_correct = 0
    for seed in range(196_100, 196_140):
        task = modular_chain(8, seed)
        program = compile_public_transition_program(task.prompt)
        expected = task.transition_trace.states[-1]
        trained_correct += int(execute_neural_action_program(program, trained).terminal_state == expected)
        sham_correct += int(execute_neural_action_program(program, sham).terminal_state == expected)
    assert trained_correct == 40
    assert sham_correct <= 5


def test_artifact_is_hash_bound_reloadable_and_tamper_evident(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    manifest = train_and_write_neural_transition_artifact(artifact)
    loaded = load_neural_transition_tissue(artifact)
    assert loaded.tissue_sha256 == manifest["weights_sha256"]

    weights = artifact / "weights.safetensors"
    weights.write_bytes(weights.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="commitment differs"):
        load_neural_transition_tissue(artifact)


def test_checked_in_artifact_passes_independent_teacher_free_verification() -> None:
    artifact = (
        Path(__file__).resolve().parent.parent
        / "core"
        / "brain"
        / "llm"
        / "latent_cortex"
        / "assets"
        / "neural_transition_tissue_v1"
    )
    report = verify_neural_transition_artifact(artifact)
    assert report["verified"] is True
    assert report["teacher_available"] is False
    assert report["primitive_count"] == 3_842
    assert report["fresh_program_count"] == 192
    assert report["fresh_transition_count"] == 2_016


def test_invalid_neural_requests_fail_closed(trained: NeuralTransitionTissue) -> None:
    with pytest.raises(ValueError, match="support"):
        trained.transition(
            family="modular",
            depth=2,
            field_names=("pc", "residue", "done"),
            state=(0, 1, 0),
            action_field_names=("opcode", "operand", "modulus"),
            action=(0, 1, 29),
        )
    with pytest.raises(ValueError, match="invalid"):
        trained.transition(
            family="boolean",
            depth=1,
            field_names=("pc", "value", "done"),
            state=(1, 0, 1),
            action_field_names=("opcode", "operand", "has_operand"),
            action=(0, 0, 0),
        )


@pytest.mark.parametrize("generator", (nested_boolean, modular_chain))
def test_canonical_objective_producer_uses_teacher_removed_tissue(
    generator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _resident_neural_transition_tissue.cache_clear()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("certified teacher reached canonical neural producer")

    monkeypatch.setattr(CertifiedTransitionExecutor, "execute", forbidden)
    task = generator(16, 196_500)
    solved = solve_objective_program(task.prompt)
    assert solved is not None
    candidate, receipt = solved
    assert candidate.endswith(task.answer)
    assert receipt["execution"]["engine"] == "neural_transition_tissue.v1"
    assert receipt["execution"]["teacher_available"] is False
