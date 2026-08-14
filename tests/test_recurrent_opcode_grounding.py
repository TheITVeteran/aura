"""Contracts for tokenizer-bound canonical operation observations."""

from __future__ import annotations

import pytest

from core.learning.recurrent_action_schema import (
    ACTION_NULL,
    OP_ADD_MOD,
    OP_BOOL_AND,
    OP_BOOL_NOT,
    OP_BOOL_OR,
    OP_BOOL_XOR,
    OP_COPY_VALUE,
    OP_FRONTIER_AUDIT,
    OP_FRONTIER_CALIBRATE,
    OP_FRONTIER_ENUMERATE,
    OP_FRONTIER_INFER,
    OP_FRONTIER_SCHEDULE,
    OP_FRONTIER_SIMULATE,
    OP_FRONTIER_TRAVERSE,
    OP_MUL_MOD,
    OP_REGISTER_AFFINE,
    OP_SUB_MOD,
)
from core.learning.recurrent_literal_grounding import LiteralObservationContract
from core.learning.recurrent_opcode_grounding import (
    FrontierFamilyObservationContract,
    OpcodeObservationContract,
)

OPCODE_PATTERNS = (
    (OP_COPY_VALUE, (7,)),
    (OP_ADD_MOD, (8,)),
    (OP_MUL_MOD, (9,)),
    (OP_SUB_MOD, (10,)),
    (OP_BOOL_NOT, (11,)),
    (OP_BOOL_AND, (12,)),
    (OP_BOOL_OR, (13,)),
    (OP_BOOL_XOR, (14,)),
)
CONTEXT_PATTERNS = (
    ("graph", (40,)),
    ("graph_edges_start", (41,)),
    ("graph_edges_end", (42,)),
    ("modular_start", (43,)),
    ("modular_end", (44,)),
    ("boolean_start", (45,)),
    ("boolean_end", (46,)),
    ("register", (47,)),
    ("register_ops_start", (48,)),
    ("register_ops_end", (49,)),
)
FRONTIER_PATTERNS = tuple(
    (opcode, (100 + index, 200 + index))
    for index, opcode in enumerate(
        (
            OP_FRONTIER_TRAVERSE,
            OP_FRONTIER_ENUMERATE,
            OP_FRONTIER_SIMULATE,
            OP_FRONTIER_INFER,
            OP_FRONTIER_SCHEDULE,
            OP_FRONTIER_CALIBRATE,
            OP_FRONTIER_AUDIT,
        )
    )
)


def test_frontier_family_contract_routes_public_markers_only() -> None:
    contract = FrontierFamilyObservationContract(FRONTIER_PATTERNS)
    values, recognized = contract.observe(
        ((1, 100, 200, 2), (1, 106, 206, 2), (1, 2, 3))
    )
    assert values == (OP_FRONTIER_TRAVERSE, OP_FRONTIER_AUDIT, ACTION_NULL)
    assert recognized == (True, True, False)
    assert len(contract.contract_sha256) == 64


def test_frontier_family_contract_rejects_conflicting_public_markers() -> None:
    contract = FrontierFamilyObservationContract(FRONTIER_PATTERNS)
    with pytest.raises(ValueError, match="conflicting frontier families"):
        contract.observe(((100, 200, 101, 201),))


def test_opcode_contract_marks_every_exact_occurrence_without_assigning_relevance() -> None:
    contract = OpcodeObservationContract(
        OPCODE_PATTERNS,
        CONTEXT_PATTERNS,
    )
    values, masks = contract.observe(((40, 1, 7, 2, 7), (43, 1, 8, 2, 44)))
    assert values == (
        (0, 0, OP_COPY_VALUE, 0, OP_COPY_VALUE),
        (0, 0, OP_ADD_MOD, 0, 0),
    )
    assert masks == (
        (False, False, True, False, True),
        (False, False, True, False, False),
    )
    assert len(contract.contract_sha256) == 64


def test_register_context_suppresses_internal_arithmetic_markers() -> None:
    contract = OpcodeObservationContract(
        OPCODE_PATTERNS,
        tuple(
            (name, (50, 51) if name == "register" else pattern)
            for name, pattern in CONTEXT_PATTERNS
        ),
    )
    values, masks = contract.observe(((50, 51, 1, 8, 2, 8),))
    assert values == ((0, OP_REGISTER_AFFINE, 0, 0, 0, 0),)
    assert masks == ((False, True, False, False, False, False),)


def test_public_modular_grammar_decodes_initial_state_and_instruction() -> None:
    contract = OpcodeObservationContract(OPCODE_PATTERNS, CONTEXT_PATTERNS)
    literals = LiteralObservationContract(tuple(range(100, 110)))
    row = (105, 1, 107, 43, 8, 103, 44)
    states, state_known = contract.public_initial_states((row,), literals)
    instructions, instruction_known = contract.public_instructions(
        (row,),
        literals,
        states,
    )
    assert state_known == (True,)
    assert states == ((0, 7, 0, 0, 0),)
    assert instruction_known == (True,)
    assert instructions == (
        (OP_ADD_MOD, 3, 5, ACTION_NULL, ACTION_NULL, ACTION_NULL, ACTION_NULL, 1),
    )
