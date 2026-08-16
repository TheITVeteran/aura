"""CP557 follow-up: the microcode had an opcode it could not run.

OP_CAUSAL_CHAIN was added to the schema and implemented in the reference
executor and in the semantic neural machine, but never in the categorical
microcode. Recognition there was a range check against MAX_RECURRENT_OPCODE, so
bumping the constant made the microcode declare it recognized the instruction:
it ran no branch, returned the registers unchanged, and reported success.

The microcode now executes the six-stage chain. These tests hold it against the
reference executor in core/learning/frontier_process_supervision.py — the same
inputs through both, step by step — because a second implementation of a state
machine is only worth having if something checks the two agree.
"""
from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.frontier_tasks import generate_task
from core.learning.frontier_process_supervision import (
    compile_frontier_process_supervision,
)
from core.learning.recurrent_action_schema import (
    ACTION_NULL,
    MAX_RECURRENT_OPCODE,
    OP_CAUSAL_CHAIN,
    action_targets_from_program,
)
from core.learning.recurrent_state_schema import (
    SEMANTIC_STATE_SLOT_NAMES,
    STATE_INVALID,
    state_targets_from_trace,
)
from core.learning.unified_intrinsic_recurrence import (
    MICROCODE_IMPLEMENTED_OPCODES,
)

mx = pytest.importorskip("mlx.core")


def _controller(state_width: int):
    from core.learning.unified_intrinsic_recurrence import (
        UnifiedRecurrenceConfig,
        UnifiedRecurrentController,
    )

    return UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=32, correction_rank=4, state_slots=state_width
        )
    )


def _run_microcode(program) -> list[tuple[int, ...]]:
    """Every state the microcode reaches, driven by the compiled actions."""
    depth = program.state_trace.depth
    targets = action_targets_from_program(program, depth)
    state_targets = state_targets_from_trace(
        program.state_trace, depth, state_slots=len(SEMANTIC_STATE_SLOT_NAMES)
    )
    controller = _controller(len(SEMANTIC_STATE_SLOT_NAMES))
    current = state_targets.initial_values
    produced_states: list[tuple[int, ...]] = []
    history = []
    for step in range(len(state_targets.values)):
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
        history.append(action_probabilities)
        logits, recognized = controller.microcode_transition_logits(
            state_probabilities,
            action_probabilities,
            action_probability_history=history,
        )
        mx.eval(logits, recognized)
        assert bool(recognized.item()), f"step {step} was not recognized"
        current = tuple(int(value) for value in mx.argmax(logits[0], axis=-1).tolist())
        produced_states.append(current)
    return produced_states


class TestOpcodeRegistration:
    def test_the_causal_chain_is_declared_implemented(self):
        assert OP_CAUSAL_CHAIN in MICROCODE_IMPLEMENTED_OPCODES

    def test_recognition_no_longer_follows_the_range(self):
        """The bug was that these two sets were assumed identical."""
        every_opcode = set(range(MAX_RECURRENT_OPCODE + 1))
        assert MICROCODE_IMPLEMENTED_OPCODES <= every_opcode
        assert ACTION_NULL not in MICROCODE_IMPLEMENTED_OPCODES


class TestAgreementWithTheReference:
    @pytest.mark.parametrize("difficulty", [1, 2, 3])
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_the_microcode_reproduces_the_reference_trace(self, difficulty, seed):
        """Same actions, same states — every step, not just the last one."""
        source = generate_task(
            "scientific_inference", seed=91_700 + seed, difficulty=difficulty
        )
        program = compile_frontier_process_supervision(source).program
        expected = [tuple(row) for row in program.state_trace.states[1:]]
        produced = _run_microcode(program)
        # A prefix comparison would pass on an empty trace, which is exactly
        # the shape a silently-skipped opcode produces.
        assert len(expected) >= 10
        assert produced == expected

    def test_the_chain_actually_runs_here(self):
        """Guard against the test passing because no causal action appeared."""
        source = generate_task("scientific_inference", seed=91_701, difficulty=2)
        program = compile_frontier_process_supervision(source).program
        targets = action_targets_from_program(program, program.state_trace.depth)
        opcodes = {row[0] for row in targets.values[:-1]}
        assert OP_CAUSAL_CHAIN in opcodes

    def test_every_stage_of_the_chain_is_exercised(self):
        """The six stages live in slot 9 of the semantic state."""
        source = generate_task("scientific_inference", seed=91_702, difficulty=3)
        program = compile_frontier_process_supervision(source).program
        stages = {row[9] for row in program.state_trace.states}
        assert stages >= {0, 1, 2, 3, 4, 5, 6}


class TestRefusals:
    def _states_and_actions(self):
        source = generate_task("scientific_inference", seed=91_703, difficulty=2)
        program = compile_frontier_process_supervision(source).program
        depth = program.state_trace.depth
        return (
            program,
            action_targets_from_program(program, depth),
            state_targets_from_trace(
                program.state_trace, depth, state_slots=len(SEMANTIC_STATE_SLOT_NAMES)
            ),
            len(SEMANTIC_STATE_SLOT_NAMES),
        )

    def _step(self, controller, state_values, action_values):
        state_probabilities = controller.exact_probabilities(
            state_values,
            slots=controller.config.state_slots,
            cardinality=controller.config.state_cardinality,
        )
        action_probabilities = controller.exact_probabilities(
            action_values,
            slots=controller.config.action_slots,
            cardinality=controller.config.action_cardinality,
        )
        logits, _ = controller.microcode_transition_logits(
            state_probabilities,
            action_probabilities,
            action_probability_history=[action_probabilities],
        )
        mx.eval(logits)
        return tuple(int(value) for value in mx.argmax(logits[0], axis=-1).tolist())

    def test_an_out_of_order_edge_produces_the_invalid_state(self):
        """The reference raises here; the microcode has no exception path, so
        it must reach the invalid sentinel rather than a plausible state."""
        _program, targets, state_targets, width = self._states_and_actions()
        controller = _controller(width)
        # Stage 5 with a first-edge action: the chain is already past its
        # opening move.
        state = list(state_targets.initial_values)
        state[9] = 5
        action = list(targets.values[0])
        action[0] = OP_CAUSAL_CHAIN
        action[1], action[2] = 0, 1
        action[6] = 0
        produced = self._step(controller, tuple(state), tuple(action))
        assert all(value == STATE_INVALID for value in produced)

    def test_a_first_edge_from_a_variable_to_itself_is_refused(self):
        _program, targets, state_targets, width = self._states_and_actions()
        controller = _controller(width)
        action = list(targets.values[0])
        action[0] = OP_CAUSAL_CHAIN
        action[1], action[2] = 1, 1
        action[6] = 0
        produced = self._step(
            controller, state_targets.initial_values, tuple(action)
        )
        assert all(value == STATE_INVALID for value in produced)
