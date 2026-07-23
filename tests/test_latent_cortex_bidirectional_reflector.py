"""Complete-trace, non-causal hidden reflection contracts."""

from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.bidirectional_reflector import (  # noqa: E402
    build_bidirectional_reflector_receipt,
    observe_reflector_vectors,
    validate_bidirectional_reflector_receipt,
)
from core.brain.llm.latent_cortex.engine import LatentCortexEngine  # noqa: E402
from core.brain.llm.latent_cortex.loop_core import canonical_sha256  # noqa: E402
from core.brain.llm.latent_cortex.types import (  # noqa: E402
    BranchConfig,
    ComputeBudget,
    CortexConfig,
    LatentOptConfig,
    RecurrenceConfig,
    WorkspaceConfig,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _trace(
    *,
    final_shift: float = 0.0,
    rejected_middle: bool = False,
) -> tuple[SimpleNamespace, dict]:
    observations = []
    transitions = []
    states = (
        ((0.0, 0.1, 0.2), (0.2, 0.2, 0.3)),
        ((0.2, 0.2, 0.3), (0.9, -0.5, 0.7)),
        (
            (0.2 if rejected_middle else 0.9, 0.2 if rejected_middle else -0.5, 0.3 if rejected_middle else 0.7),
            (1.0 + final_shift, 0.2 + final_shift, 0.8 - final_shift),
        ),
    )
    for step, (prior, proposal) in enumerate(states):
        accepted = not (rejected_middle and step == 1)
        admitted = proposal if accepted else prior
        prior_hash = _digest(f"prior:{step}:{prior}")
        proposal_hash = _digest(f"proposal:{step}:{proposal}")
        admitted_hash = proposal_hash if accepted else prior_hash
        observations.append(
            observe_reflector_vectors(
                prior,
                proposal,
                admitted,
                branch_index=0,
                branch_step=step,
                prior_state_sha256=prior_hash,
                proposal_state_sha256=proposal_hash,
                admitted_state_sha256=admitted_hash,
                accepted=accepted,
            )
        )
        transitions.append(
            {
                "branch_step": step,
                "prior_reasoning_sha256": prior_hash,
                "proposal_reasoning_sha256": proposal_hash,
                "admitted_reasoning_sha256": admitted_hash,
                "accepted": accepted,
            }
        )
    source_payload = {"branches": [{"transitions": transitions}]}
    source = {
        **source_payload,
        "receipt_sha256": canonical_sha256(source_payload),
    }
    return SimpleNamespace(index=0, reflector_trace=observations), source


def _receipt(*, final_shift: float = 0.0, rejected_middle: bool = False):
    branch, source = _trace(
        final_shift=final_shift,
        rejected_middle=rejected_middle,
    )
    return (
        build_bidirectional_reflector_receipt(
            branches=[branch],
            update_acceptance=source,
            selected_branch=0,
        ),
        source,
    )


def test_reflector_inspects_complete_hidden_trace_without_answer_or_authority():
    receipt, _source = _receipt()
    branch = receipt["branches"][0]
    assert receipt["complete_trace_inspected"] is True
    assert receipt["hidden_trace_only"] is True
    assert receipt["answer_text_consumed"] is False
    assert receipt["state_mutation_authorized"] is False
    assert receipt["selection_authorized"] is False
    assert receipt["repair_authorized"] is False
    assert receipt["attention_perturbation_authorized"] is False
    assert branch["trace_length"] == 3
    assert branch["reflections"][0]["uses_future_context"] is True
    assert branch["reflections"][-1]["uses_future_context"] is False
    assert branch["reflections"][-1]["uses_past_context"] is True


def test_future_hidden_conclusion_changes_earlier_reflection():
    baseline, _ = _receipt(final_shift=0.0)
    changed, _ = _receipt(final_shift=2.0)
    baseline_branch = baseline["branches"][0]
    changed_branch = changed["branches"][0]
    assert (
        baseline_branch["observations"][0]
        == changed_branch["observations"][0]
    )
    assert (
        baseline_branch["premise_sketch_sha256"]
        == changed_branch["premise_sketch_sha256"]
    )
    assert (
        baseline_branch["reflections"][0]["reflected_state_sha256"]
        != changed_branch["reflections"][0]["reflected_state_sha256"]
    )
    assert (
        baseline_branch["reflections"][0]["suffix_context_sha256"]
        != changed_branch["reflections"][0]["suffix_context_sha256"]
    )


def test_rejected_proposal_is_inspected_but_not_the_admitted_path():
    receipt, _source = _receipt(rejected_middle=True)
    middle = receipt["branches"][0]["observations"][1]
    assert middle["accepted"] is False
    assert middle["proposal_reasoning_sha256"] != middle["prior_reasoning_sha256"]
    assert middle["admitted_reasoning_sha256"] == middle["prior_reasoning_sha256"]
    metrics = receipt["branches"][0]["reflections"][1]["metrics"]
    assert metrics["local_proposal_delta_rms"] > 0.0
    assert metrics["local_admitted_delta_rms"] == 0.0


def test_rehashed_reflection_metric_lie_is_rejected():
    receipt, source = _receipt()
    forged = copy.deepcopy(receipt)
    forged["branches"][0]["reflections"][0]["metrics"][
        "proposal_to_conclusion_cosine"
    ] = 0.999
    forged["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in forged.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="branch reconstruction failed"):
        validate_bidirectional_reflector_receipt(
            forged,
            update_acceptance=source,
            expected_n_branches=1,
        )


def test_extreme_finite_proposal_is_bounded_for_escape_diagnostics():
    prior_hash = _digest("extreme-prior")
    proposal_hash = _digest("extreme-proposal")
    observation = observe_reflector_vectors(
        (0.0, 1.0, -1.0),
        (1e300, -1e300, 1e250),
        (1e300, -1e300, 1e250),
        branch_index=0,
        branch_step=0,
        prior_state_sha256=prior_hash,
        proposal_state_sha256=proposal_hash,
        admitted_state_sha256=proposal_hash,
        accepted=True,
    )
    assert all(
        abs(value) < 1_000.0 for value in observation["proposal_sketch"]
    )


def test_array_backed_position_sequences_are_unambiguous():
    prior_hash = _digest("array-prior")
    proposal_hash = _digest("array-proposal")
    positions = np.asarray(
        ((0.0, 0.1, 0.2), (0.3, 0.4, 0.5)),
        dtype=np.float64,
    )
    observation = observe_reflector_vectors(
        (0.15, 0.25, 0.35),
        (0.20, 0.30, 0.40),
        (0.20, 0.30, 0.40),
        branch_index=0,
        branch_step=0,
        prior_state_sha256=prior_hash,
        proposal_state_sha256=proposal_hash,
        admitted_state_sha256=proposal_hash,
        accepted=True,
        prior_positions=positions,
        proposal_positions=positions + 0.05,
        admitted_positions=positions + 0.05,
    )
    assert observation["position_count"] == 2
    assert len(observation["proposal_position_sketches"]) == 2


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(character) % 128 for character in text][:16]

    def decode(self, ids):
        return " ".join(str(item) for item in ids)


def test_real_tiny_qwen_reflects_every_transition_and_meters_work():
    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=32,
            num_hidden_layers=8,
            intermediate_size=64,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=128,
            num_key_value_heads=2,
            max_position_embeddings=512,
            rope_theta=10000.0,
        )
    )
    mx.eval(model.parameters())
    engine = LatentCortexEngine(
        model,
        _Tokenizer(),
        config=CortexConfig(
            workspace=WorkspaceConfig(n_slots=4, seed=19),
            recurrence=RecurrenceConfig(
                min_steps=1,
                max_steps=3,
                convergence_eps=1e-9,
                fixed_depth=True,
            ),
            branches=BranchConfig(n_branches=2),
            latent_opt=LatentOptConfig(enabled=False),
            decode_max_tokens=3,
            allow_vanilla_fallback=False,
        ),
    )
    result = engine.reason(
        token_ids=[5, 9, 17, 3, 42, 7],
        budget=ComputeBudget(),
    )
    assert result.ok is True
    receipt = result.receipt.bidirectional_reflector
    transitions = [
        transition
        for branch in result.receipt.update_acceptance["branches"]
        for transition in branch["transitions"]
    ]
    assert receipt["observation_count"] == len(transitions) >= 1
    operations = result.receipt.budget["resource_accounting"]["operations"]
    assert operations["bidirectional_reflector_capture"][
        "tensor_element_reads"
    ] >= len(transitions) * 96
    assert operations["bidirectional_reflector_review"]["host_scalar_ops"] > 0
    validate_bidirectional_reflector_receipt(
        receipt,
        update_acceptance=result.receipt.update_acceptance,
        expected_n_branches=2,
    )
