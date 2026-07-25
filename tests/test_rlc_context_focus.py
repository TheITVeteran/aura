from __future__ import annotations

import copy

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from core.brain.llm.latent_cortex.context_focus import (  # noqa: E402
    apply_context_focus,
    context_sources_for_action,
    source_matches_action,
    validate_context_focus_receipt,
)
from core.brain.llm.latent_cortex.epistemic_state import OperationKind  # noqa: E402

COGNITIVE_SLOTS = [
    {"slot": 1, "source": "memory"},
    {"slot": 2, "source": "reference"},
    {"slot": 3, "source": "goals"},
]


def _state():
    values = np.arange(1, 1 + 5 * 8, dtype=np.float32).reshape(1, 5, 8)
    values[:, 1, :] *= -1.0
    values[:, 2, :] = np.roll(values[:, 2, :], 3)
    return mx.array(values)


def test_source_classes_are_explicit_and_hybrid_one_shot_is_evidence():
    assert source_matches_action("memory", OperationKind.SEARCH_MEMORY)
    assert source_matches_action("memory.episodic.abc", "search_memory")
    assert source_matches_action("one_shot_memory", "search_memory")
    assert source_matches_action("one_shot_memory", "retrieve_evidence")
    assert not source_matches_action("reference", "search_memory")
    assert source_matches_action("reference", "retrieve_evidence")
    assert source_matches_action("tool_observation.web", "retrieve_evidence")
    assert not source_matches_action("goals", "retrieve_evidence")

    assert context_sources_for_action(
        COGNITIVE_SLOTS,
        OperationKind.SEARCH_MEMORY,
    ) == ((1,), ("memory",))
    assert context_sources_for_action(
        COGNITIVE_SLOTS,
        OperationKind.RETRIEVE_EVIDENCE,
    ) == ((2,), ("reference",))


@pytest.mark.parametrize(
    ("action", "expected_slot", "expected_class"),
    [
        (OperationKind.SEARCH_MEMORY, 1, "memory"),
        (OperationKind.RETRIEVE_EVIDENCE, 2, "evidence"),
    ],
)
def test_focus_is_causal_preserves_sources_and_validates(
    action,
    expected_slot,
    expected_class,
):
    state = _state()
    output, receipt = apply_context_focus(
        state,
        context_slots=COGNITIVE_SLOTS,
        action=action,
        branch_index=2,
        action_step=3,
    )

    assert receipt["source_class"] == expected_class
    assert receipt["source_slots"] == [expected_slot]
    assert receipt["external_retrieval_effect"] == "none"
    assert receipt["input_sha256"] != receipt["output_sha256"]
    assert receipt["source_sha256"] == receipt["preserved_source_sha256"]
    assert bool(mx.array_equal(output[:, 1:4, :], state[:, 1:4, :]))
    assert not bool(mx.array_equal(output[:, 0:1, :], state[:, 0:1, :]))
    assert validate_context_focus_receipt(
        receipt,
        cognitive_slots=COGNITIVE_SLOTS,
    ) == receipt


def test_memory_and_evidence_actions_produce_distinct_states():
    state = _state()
    memory, memory_receipt = apply_context_focus(
        state,
        context_slots=COGNITIVE_SLOTS,
        action=OperationKind.SEARCH_MEMORY,
        branch_index=0,
        action_step=0,
    )
    evidence, evidence_receipt = apply_context_focus(
        state,
        context_slots=COGNITIVE_SLOTS,
        action=OperationKind.RETRIEVE_EVIDENCE,
        branch_index=0,
        action_step=0,
    )

    assert not bool(mx.array_equal(memory, evidence))
    assert memory_receipt["source_sha256"] != evidence_receipt["source_sha256"]
    assert memory_receipt["output_sha256"] != evidence_receipt["output_sha256"]


def test_focus_refuses_absent_source_instead_of_running_generic_transition():
    with pytest.raises(ValueError, match="no matching admitted source"):
        apply_context_focus(
            _state(),
            context_slots=[{"slot": 1, "source": "goals"}],
            action=OperationKind.SEARCH_MEMORY,
            branch_index=0,
            action_step=0,
        )


def test_validator_rejects_source_inventory_tamper():
    _output, receipt = apply_context_focus(
        _state(),
        context_slots=COGNITIVE_SLOTS,
        action=OperationKind.SEARCH_MEMORY,
        branch_index=0,
        action_step=0,
    )
    tampered = copy.deepcopy(receipt)
    tampered["source_slots"] = [2]

    with pytest.raises(ValueError):
        validate_context_focus_receipt(
            tampered,
            cognitive_slots=COGNITIVE_SLOTS,
        )


def test_validator_rejects_false_external_retrieval_claim_even_when_rehashed():
    _output, receipt = apply_context_focus(
        _state(),
        context_slots=COGNITIVE_SLOTS,
        action=OperationKind.RETRIEVE_EVIDENCE,
        branch_index=0,
        action_step=0,
    )
    tampered = copy.deepcopy(receipt)
    tampered["external_retrieval_effect"] = "new_evidence_fetched"
    from core.brain.llm.latent_cortex.epistemic_state import canonical_sha256

    tampered["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "receipt_sha256"}
    )

    with pytest.raises(ValueError, match="execution metadata"):
        validate_context_focus_receipt(
            tampered,
            cognitive_slots=COGNITIVE_SLOTS,
        )
