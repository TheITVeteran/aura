"""Live RLC hidden-state uncertainty wiring and receipt reconstruction."""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.engine import LatentCortexEngine  # noqa: E402
from core.brain.llm.latent_cortex.loop_core import canonical_sha256  # noqa: E402
from core.brain.llm.latent_cortex.neural_uncertainty import (  # noqa: E402
    LEARNED,
    NeuralUncertaintyRuntime,
    validate_neural_uncertainty_receipt,
)
from core.brain.llm.latent_cortex.types import (  # noqa: E402
    BranchConfig,
    ComputeBudget,
    CortexConfig,
    LatentOptConfig,
    RecurrenceConfig,
    WorkspaceConfig,
)
from core.learning.neural_uncertainty import (  # noqa: E402
    HiddenStateCorrectnessExample,
    NeuralUncertaintyHead,
)

HIDDEN = 32


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _examples(split: str):
    rng = np.random.default_rng(5 if split == "train" else 17)
    rows = []
    index = 0
    for signal, expected_rate in (
        (-3.0, 0.05),
        (-1.5, 0.20),
        (-0.5, 0.40),
        (0.5, 0.60),
        (1.5, 0.80),
        (3.0, 0.95),
    ):
        positives = round(48 * expected_rate)
        labels = [True] * positives + [False] * (48 - positives)
        rng.shuffle(labels)
        for correct in labels:
            hidden = rng.normal(0.0, 0.10, size=HIDDEN)
            hidden[0] = signal + rng.normal(0.0, 0.03)
            rows.append(
                HiddenStateCorrectnessExample(
                    example_id=f"{split}-{index}",
                    task_id=f"{split}-task-{index % 8}",
                    hidden_state=tuple(float(value) for value in hidden),
                    correct=correct,
                    state_sha256=_digest(f"{split}:state:{index}"),
                    outcome_receipt_sha256=_digest(f"{split}:outcome:{index}"),
                    outcome_verifier_id="independent-exact-grader-v1",
                )
            )
            index += 1
    return rows


def _head(tmp_path):
    head = NeuralUncertaintyHead.fit(
        _examples("train"),
        _examples("calibration"),
        hidden_width=8,
        seed=13,
        steps=500,
    )
    path = tmp_path / "uncertainty.json"
    digest = head.save(path)
    return head, {
        "mode": LEARNED,
        "head_path": str(path),
        "head_sha256": digest,
    }


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(character) % 128 for character in text][:16]

    def decode(self, ids):
        return " ".join(str(item) for item in ids)


@pytest.fixture(scope="module")
def tiny_model():
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=HIDDEN,
        num_hidden_layers=8,
        intermediate_size=64,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


def _config(
    uncertainty_head=None,
    *,
    allow_vanilla_fallback=True,
    n_branches=1,
):
    return CortexConfig(
        workspace=WorkspaceConfig(n_slots=4, seed=23),
        recurrence=RecurrenceConfig(
            min_steps=1,
            max_steps=3,
            convergence_eps=1e-9,
            fixed_depth=True,
        ),
        branches=BranchConfig(n_branches=n_branches),
        latent_opt=LatentOptConfig(enabled=False),
        decode_max_tokens=3,
        uncertainty_head=uncertainty_head,
        allow_vanilla_fallback=allow_vanilla_fallback,
    )


def test_train_live_pooling_matches_head_input_exactly(tmp_path):
    head, config = _head(tmp_path)
    runtime = NeuralUncertaintyRuntime.from_config(config)
    state = mx.reshape(mx.arange(2 * HIDDEN, dtype=mx.float32), (1, 2, HIDDEN))
    pooled = [
        round(float(item), 8)
        for item in mx.mean(state, axis=(0, 1)).tolist()
    ]
    row = runtime.observe(
        state,
        branch_index=0,
        branch_step=2,
        state_sha256="a" * 64,
    )
    assert row["pooled_hidden"] == pooled
    assert row["estimate"] == head.estimate(pooled)


def test_real_tiny_qwen_emits_one_objective_observation_per_transition(
    tiny_model,
    tmp_path,
):
    _head_value, head_config = _head(tmp_path)
    engine = LatentCortexEngine(
        tiny_model,
        _Tokenizer(),
        config=_config(head_config, n_branches=2),
    )
    result = engine.reason(
        token_ids=[5, 9, 17, 3, 42, 7],
        budget=ComputeBudget(),
    )
    assert result.ok is True
    receipt = result.receipt.neural_uncertainty
    transitions = [
        transition
        for branch in result.receipt.update_acceptance["branches"]
        for transition in branch["transitions"]
    ]
    assert receipt["mode"] == LEARNED
    assert receipt["observation_count"] == len(transitions) >= 1
    assert sum(
        len(branch["observations"]) for branch in receipt["branches"]
    ) == len(transitions)
    assert receipt["selection_eligible"] is True
    assert receipt["selection_causal"] is True
    assert receipt["selection_basis"] == "neural_uncertainty"
    expected_winner = max(
        range(2),
        key=lambda index: receipt["latest_supported_scores"][str(index)],
    )
    assert result.receipt.selected_branch == expected_winner
    assert (
        result.receipt.budget["resource_accounting"]["operations"][
            "neural_uncertainty_head"
        ]["tensor_element_reads"]
        >= len(transitions) * HIDDEN
    )
    validate_neural_uncertainty_receipt(
        receipt,
        expected_runtime=NeuralUncertaintyRuntime.from_config(head_config),
        update_acceptance=result.receipt.update_acceptance,
        expected_n_branches=2,
    )


def test_disabled_runtime_emits_no_confidence(tiny_model):
    result = LatentCortexEngine(
        tiny_model,
        _Tokenizer(),
        config=_config(),
    ).reason(
        token_ids=[5, 9, 17, 3, 42, 7],
        budget=ComputeBudget(),
    )
    receipt = result.receipt.neural_uncertainty
    assert receipt["mode"] == "unavailable"
    assert receipt["observation_count"] == 0
    assert receipt["supported_count"] == 0
    assert receipt["branches"][0]["observations"] == []


@pytest.mark.parametrize(
    "selection_basis",
    [
        "task_verifier_counterfactual_tiebreak",
        "task_verifier_generative_refutation_veto",
        "task_verifier_counterfactual_tiebreak_generative_refutation_veto",
    ],
)
def test_admitted_selection_pipeline_provenance_is_reconstructible(
    tiny_model,
    selection_basis,
):
    result = LatentCortexEngine(
        tiny_model,
        _Tokenizer(),
        config=_config(),
    ).reason(
        token_ids=[5, 9, 17, 3, 42, 7],
        budget=ComputeBudget(),
    )
    receipt = copy.deepcopy(result.receipt.neural_uncertainty)
    receipt["selection_basis"] = selection_basis
    receipt["selection_causal"] = False
    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = canonical_sha256(payload)

    validate_neural_uncertainty_receipt(
        receipt,
        expected_runtime=NeuralUncertaintyRuntime.from_config(None),
        update_acceptance=result.receipt.update_acceptance,
        expected_n_branches=1,
    )


@pytest.mark.parametrize(
    "selection_basis",
    [
        "neural_uncertainty_counterfactual_tiebreak",
        "process_verifier_generative_refutation_veto",
        "task_verifier_generative_refutation_veto_counterfactual_tiebreak",
        "task_verifier_counterfactual_tiebreak_counterfactual_tiebreak",
        "task_verifier_unknown_override",
    ],
)
def test_unproducible_selection_pipeline_provenance_is_rejected(
    tiny_model,
    selection_basis,
):
    result = LatentCortexEngine(
        tiny_model,
        _Tokenizer(),
        config=_config(),
    ).reason(
        token_ids=[5, 9, 17, 3, 42, 7],
        budget=ComputeBudget(),
    )
    receipt = copy.deepcopy(result.receipt.neural_uncertainty)
    receipt["selection_basis"] = selection_basis
    receipt["selection_causal"] = False
    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = canonical_sha256(payload)

    with pytest.raises(ValueError, match="aggregate evidence differs"):
        validate_neural_uncertainty_receipt(
            receipt,
            expected_runtime=NeuralUncertaintyRuntime.from_config(None),
            update_acceptance=result.receipt.update_acceptance,
            expected_n_branches=1,
        )


def test_rehashed_prediction_and_source_tampering_are_rejected(
    tiny_model,
    tmp_path,
):
    _head_value, head_config = _head(tmp_path)
    result = LatentCortexEngine(
        tiny_model,
        _Tokenizer(),
        config=_config(head_config),
    ).reason(
        token_ids=[5, 9, 17, 3, 42, 7],
        budget=ComputeBudget(),
    )
    runtime = NeuralUncertaintyRuntime.from_config(head_config)
    receipt = result.receipt.neural_uncertainty

    forged = copy.deepcopy(receipt)
    forged["branches"][0]["observations"][0]["estimate"][
        "correctness_probability"
    ] = 0.999
    with pytest.raises(ValueError):
        validate_neural_uncertainty_receipt(
            forged,
            expected_runtime=runtime,
            update_acceptance=result.receipt.update_acceptance,
            expected_n_branches=1,
        )

    altered_source = copy.deepcopy(result.receipt.update_acceptance)
    altered_source["branches"][0]["transitions"][0][
        "admitted_reasoning_sha256"
    ] = "b" * 64
    with pytest.raises(ValueError):
        validate_neural_uncertainty_receipt(
            receipt,
            expected_runtime=runtime,
            update_acceptance=altered_source,
            expected_n_branches=1,
        )


def test_hidden_state_width_mismatch_refuses_episode(tiny_model, tmp_path):
    narrow = [
        HiddenStateCorrectnessExample(
            example_id=f"narrow-train-{index}",
            task_id=f"narrow-train-task-{index % 8}",
            hidden_state=(2.0 if index % 2 == 0 else -2.0,) + (0.0,) * 14,
            correct=index % 2 == 0,
            state_sha256=_digest(f"narrow-train-state-{index}"),
            outcome_receipt_sha256=_digest(f"narrow-train-outcome-{index}"),
            outcome_verifier_id="grader",
        )
        for index in range(64)
    ]
    calibration = [
        HiddenStateCorrectnessExample(
            example_id=f"narrow-cal-{index}",
            task_id=f"narrow-cal-task-{index % 8}",
            hidden_state=(2.0 if index % 2 == 0 else -2.0,) + (0.0,) * 14,
            correct=index % 2 == 0,
            state_sha256=_digest(f"narrow-cal-state-{index}"),
            outcome_receipt_sha256=_digest(f"narrow-cal-outcome-{index}"),
            outcome_verifier_id="grader",
        )
        for index in range(64)
    ]
    head = NeuralUncertaintyHead.fit(
        narrow,
        calibration,
        hidden_width=4,
        steps=200,
    )
    path = tmp_path / "narrow.json"
    digest = head.save(path)
    result = LatentCortexEngine(
        tiny_model,
        _Tokenizer(),
        config=_config(
            {
                "mode": LEARNED,
                "head_path": str(path),
                "head_sha256": digest,
            },
            allow_vanilla_fallback=False,
        ),
    ).reason(
        token_ids=[5, 9, 17, 3, 42, 7],
        budget=ComputeBudget(),
    )
    assert result.ok is False
    assert "vanilla_fallback_disabled" in result.receipt.honest_flags
