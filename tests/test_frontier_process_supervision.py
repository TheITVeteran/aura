from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.brain.llm.latent_cortex.frontier_tasks import (
    CONTAMINATION_SAFE_REGISTRY_VERSION,
    CURRENT_REGISTRY_VERSION,
    FRONTIER_DOMAINS,
    generate_task,
)
from core.brain.llm.latent_cortex.objective_program_verifier import (
    solve_objective_program,
)
from core.learning.frontier_process_supervision import (
    FRONTIER_PROCESS_SCHEMA,
    compile_frontier_process_supervision,
    frontier_process_task_battery,
)
from core.learning.recurrent_action_schema import (
    ACTION_NULL,
    OP_FRONTIER_AUDIT,
    OP_FRONTIER_CALIBRATE,
    OP_FRONTIER_ENUMERATE,
    OP_FRONTIER_INFER,
    OP_FRONTIER_SCHEDULE,
    OP_FRONTIER_SIMULATE,
    OP_FRONTIER_TRAVERSE,
    action_targets_from_program,
)
from core.learning.recurrent_state_schema import state_targets_from_trace
from tools.unified_intrinsic_tokenization_contract import (
    freeze_source_dataset,
    load_source_dataset,
)

_EXPECTED_OPCODES = {
    "novel_algorithms": OP_FRONTIER_TRAVERSE,
    "mathematics": OP_FRONTIER_ENUMERATE,
    "coding": OP_FRONTIER_SIMULATE,
    "scientific_inference": OP_FRONTIER_INFER,
    "long_horizon_planning": OP_FRONTIER_SCHEDULE,
    "calibration": OP_FRONTIER_CALIBRATE,
    "misleading_premise": OP_FRONTIER_AUDIT,
}


@pytest.mark.parametrize("domain", FRONTIER_DOMAINS)
def test_every_frontier_domain_compiles_into_recurrent_process_targets(domain: str) -> None:
    for difficulty in (1, 2, 3):
        for seed in range(4):
            source = generate_task(domain, seed=seed, difficulty=difficulty)
            compiled = compile_frontier_process_supervision(source)
            training = compiled.to_training_task()
            assert training.prompt == source.public.prompt
            assert training.transition_trace is compiled.program.state_trace
            assert training.transition_program is compiled.program
            state_targets = state_targets_from_trace(
                training.transition_trace,
                training.depth + 1,
            )
            action_targets = action_targets_from_program(
                training.transition_program,
                training.depth + 1,
            )
            assert len(state_targets.values) == training.depth + 1
            assert len(action_targets.values) == training.depth + 1
            assert all(row[0] == _EXPECTED_OPCODES[domain] for row in action_targets.values[:-1])
            assert action_targets.values[-1] == (ACTION_NULL,) * len(action_targets.values[-1])


@pytest.mark.parametrize("domain", FRONTIER_DOMAINS)
@pytest.mark.parametrize("difficulty", (1, 2, 3))
def test_every_frontier_process_executes_exactly_without_its_teacher(
    domain: str,
    difficulty: int,
) -> None:
    mx = pytest.importorskip("mlx.core")
    from core.learning.unified_intrinsic_recurrence import (
        UnifiedRecurrenceConfig,
        UnifiedRecurrentController,
    )

    source = generate_task(domain, seed=91_700 + difficulty, difficulty=difficulty)
    program = compile_frontier_process_supervision(source).program
    targets = action_targets_from_program(program, program.state_trace.depth)
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(hidden_size=32, correction_rank=4)
    )
    current = program.state_trace.states[0]
    for step, expected in enumerate(program.state_trace.states[1:]):
        state_probabilities = controller.exact_probabilities(
            current,
            slots=controller.config.state_slots,
            cardinality=controller.config.state_cardinality,
        )
        action_probabilities = controller.exact_probabilities(
            targets.values[step],
            slots=controller.config.action_slots,
            cardinality=controller.config.action_cardinality,
        )
        logits, recognized = controller.microcode_transition_logits(
            state_probabilities,
            action_probabilities,
        )
        mx.eval(logits, recognized)
        produced = tuple(
            int(value) for value in mx.argmax(logits[0], axis=-1).tolist()
        )
        assert bool(recognized.item())
        assert produced == expected
        current = produced


def test_frontier_actions_execute_results_instead_of_copying_teacher_answers() -> None:
    novel = compile_frontier_process_supervision(
        generate_task("novel_algorithms", seed=51_901, difficulty=2)
    ).program
    audit = compile_frontier_process_supervision(
        generate_task("misleading_premise", seed=51_902, difficulty=2)
    ).program
    planning = compile_frontier_process_supervision(
        generate_task("long_horizon_planning", seed=51_903, difficulty=2)
    ).program

    novel_targets = action_targets_from_program(novel, novel.state_trace.depth)
    audit_targets = action_targets_from_program(audit, audit.state_trace.depth)
    planning_targets = action_targets_from_program(planning, planning.state_trace.depth)

    # Traversal carries selection plus value operands, never the teacher's
    # precomputed checksum. Premise audit carries a row plus score operands,
    # never the running winner/score. Planning masks fields not consumed by the
    # executable transition.
    assert all(row[4:7] == (ACTION_NULL,) * 3 for row in novel_targets.values)
    assert all(row[5:7] == (ACTION_NULL,) * 2 for row in audit_targets.values)
    assert tuple(row[1] for row in audit_targets.values) == tuple(
        range(audit.state_trace.depth)
    )
    assert all(
        row[3] == ACTION_NULL and row[5:7] == (ACTION_NULL,) * 2
        for row in planning_targets.values
    )


@pytest.mark.parametrize("domain", FRONTIER_DOMAINS)
def test_frontier_process_commitment_is_deterministic_and_value_free(domain: str) -> None:
    source = generate_task(domain, seed=8_173, difficulty=2)
    first = compile_frontier_process_supervision(source)
    second = compile_frontier_process_supervision(source)
    assert first.program == second.program
    assert first.public_commitment == second.public_commitment
    commitment_text = json.dumps(first.public_commitment, sort_keys=True)
    runtime = first.public_inference_request()
    runtime_text = json.dumps(runtime, sort_keys=True)
    verifier_payload = source.reveal_for_verifier()
    expected_text = json.dumps(
        verifier_payload["expected"],
        sort_keys=True,
        separators=(",", ":"),
    )
    assert first.public_commitment["schema"] == FRONTIER_PROCESS_SCHEMA
    assert first.public_commitment["private_state_action_values_exposed"] is False
    assert first.public_commitment["final_answer_exposed"] is False
    assert first.public_commitment["runtime_teacher_available"] is False
    assert runtime["runtime_teacher_available"] is False
    assert set(runtime) == {
        "schema",
        "public_prompt",
        "runtime_teacher_available",
    }
    assert "answer" not in runtime
    assert "program" not in runtime
    assert "state" not in runtime
    assert "action" not in runtime
    assert expected_text not in commitment_text
    assert expected_text not in runtime_text


@pytest.mark.parametrize(
    "registry_version",
    (CURRENT_REGISTRY_VERSION, CONTAMINATION_SAFE_REGISTRY_VERSION),
)
def test_objective_executor_accepts_both_versioned_planning_prompts(
    registry_version: str,
) -> None:
    task = generate_task(
        "long_horizon_planning",
        seed=817_231,
        difficulty=2,
        registry_version=registry_version,
    )
    solved = solve_objective_program(task.public.prompt)
    assert solved is not None
    candidate, receipt = solved
    assert task.score(candidate).correct is True
    assert receipt["family"] == "dependency_deadline_portfolio"


def test_frontier_process_refuses_non_task_sources() -> None:
    with pytest.raises(TypeError, match="wrong type"):
        compile_frontier_process_supervision(object())  # type: ignore[arg-type]


def test_frontier_process_battery_is_disjoint_and_cell_complete() -> None:
    domains = ("novel_algorithms", "coding", "calibration")
    first = frontier_process_task_battery(domains, (1, 3), 2, seed=10_000)
    second = frontier_process_task_battery(
        domains,
        (1, 3),
        2,
        seed=20_000,
        excluded_prompts={task.prompt for task in first},
    )
    assert len(first) == len(second) == 12
    assert {task.family for task in first} == {
        "frontier_novel_algorithms",
        "frontier_coding",
        "frontier_calibration",
    }
    assert {task.prompt for task in first}.isdisjoint(task.prompt for task in second)
    assert len({task.task_id for task in first + second}) == 24


def test_frontier_process_dataset_round_trip_preserves_private_programs(
    tmp_path: Path,
) -> None:
    domains = ("novel_algorithms", "coding", "calibration")
    train = frontier_process_task_battery(domains, (1, 2), 2, seed=31_000)
    holdout = frontier_process_task_battery(
        domains,
        (1, 2),
        1,
        seed=41_000,
        excluded_prompts={task.prompt for task in train},
    )
    path = tmp_path / "dataset.json"
    receipt = freeze_source_dataset(path, train, holdout)
    restored_train, restored_holdout = load_source_dataset(path)
    assert receipt["train_count"] == 12
    assert receipt["holdout_count"] == 6
    assert receipt["partition_overlap"] == 0
    assert restored_train == train
    assert restored_holdout == holdout
    assert path.stat().st_mode & 0o777 == 0o400
