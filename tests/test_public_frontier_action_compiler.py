from __future__ import annotations

import json

import pytest

from core.brain.llm.latent_cortex.frontier_tasks import generate_task
from core.learning.frontier_process_supervision import (
    compile_frontier_process_supervision,
)
from core.learning.public_frontier_action_compiler import (
    PUBLIC_FRONTIER_ACTION_SCHEMA,
    compile_public_frontier_actions,
)
from core.learning.recurrent_action_schema import action_targets_from_program

_SUPPORTED = (
    "mathematics",
    "coding",
    "scientific_inference",
    "calibration",
    "misleading_premise",
)


@pytest.mark.parametrize("domain", _SUPPORTED)
@pytest.mark.parametrize("difficulty", (1, 2, 3))
def test_public_actions_match_private_teaching_actions_without_answer_access(
    domain: str,
    difficulty: int,
) -> None:
    for seed in range(8):
        source = generate_task(domain, seed=81_000 + seed, difficulty=difficulty)
        private = compile_frontier_process_supervision(source).program
        expected = action_targets_from_program(private, private.state_trace.depth)
        public = compile_public_frontier_actions(
            source.public.prompt,
            private.state_trace.family,
        )
        assert public.values == expected.values
        receipt = public.receipt()
        assert receipt["schema"] == PUBLIC_FRONTIER_ACTION_SCHEMA
        assert receipt["verifier_answer_available"] is False
        assert receipt["private_state_trace_available"] is False
        assert receipt["derived_answer_fields_present"] is False
        assert receipt["correctness_authority"] is False


@pytest.mark.parametrize("domain", _SUPPORTED)
def test_public_action_receipt_is_deterministic_and_answer_free(domain: str) -> None:
    source = generate_task(domain, seed=81_701, difficulty=2)
    family = f"frontier_{domain}"
    first = compile_public_frontier_actions(source.public.prompt, family)
    second = compile_public_frontier_actions(source.public.prompt, family)
    assert first == second
    assert first.receipt() == second.receipt()
    receipt_text = json.dumps(first.receipt(), sort_keys=True)
    answer_text = json.dumps(
        source.reveal_for_verifier()["expected"],
        sort_keys=True,
        separators=(",", ":"),
    )
    assert answer_text not in receipt_text
    assert "answer" not in first.receipt()


def test_public_action_program_exposes_prefixes_and_post_terminal_padding() -> None:
    source = generate_task("coding", seed=81_703, difficulty=2)
    program = compile_public_frontier_actions(
        source.public.prompt,
        "frontier_coding",
    )
    assert len(program.values) > 1

    assert program.values_for_iterations(1) == program.values[:1]
    assert program.values_for_iterations(len(program.values)) == program.values
    padded = program.values_for_iterations(len(program.values) + 2)
    assert padded[: len(program.values)] == program.values
    assert padded[-2:] == ((32,) * 8,) * 2
    with pytest.raises(ValueError, match="iteration budget"):
        program.values_for_iterations(0)


@pytest.mark.parametrize(
    "family",
    (
        "frontier_novel_algorithms",
        "frontier_long_horizon_planning",
    ),
)
def test_unsupported_answer_bearing_family_fails_closed(family: str) -> None:
    with pytest.raises(ValueError, match="no answer-blind"):
        compile_public_frontier_actions("A public prompt.", family)


def test_public_compiler_rejects_answer_or_trace_parameters_by_api_shape() -> None:
    source = generate_task("calibration", seed=81_702, difficulty=2)
    with pytest.raises(TypeError):
        compile_public_frontier_actions(  # type: ignore[call-arg]
            source.public.prompt,
            "frontier_calibration",
            expected=source.reveal_for_verifier()["expected"],
        )
