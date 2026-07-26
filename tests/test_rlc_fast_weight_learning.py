from __future__ import annotations

import copy
import hashlib

import pytest

from core.brain.latent_cortex_service import LatentCortexService
from core.brain.llm.latent_cortex.fast_weight_learning import (
    build_fast_weight_admission,
    empty_learning_state,
    finalize_fast_weight_learning_receipt,
    token_sequence_sha256,
    unavailable_admission,
    validate_fast_weight_admission,
    validate_fast_weight_learning_receipt,
)
from core.brain.llm.latent_cortex.task_verifiers import EpisodeTaskVerifier


class _ByteTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(text.encode("utf-8"))


def _admission(candidate: str) -> tuple[dict, list[int]]:
    verifier = EpisodeTaskVerifier("Check the calculation.")
    evaluation = verifier.evaluate(candidate)
    return build_fast_weight_admission(
        evaluation,
        candidate=candidate,
        objective=verifier.objective,
        evaluation_index=0,
        tokenizer=_ByteTokenizer(),
    )


def test_exact_local_evidence_is_admitted_and_unknown_prose_is_excluded():
    candidate = "2 + 2 = 4. This sentence is explanatory prose."
    admission, target_tokens = _admission(candidate)

    assert validate_fast_weight_admission(admission)["admitted"] is True
    assert admission["evidence_atom_ids"] == ["a000"]
    assert bytes(target_tokens).decode("utf-8") == "2 + 2 = 4."
    assert admission["target_tokens_sha256"] == token_sequence_sha256(
        target_tokens
    )


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        ("2 + 2 = 5.", "deterministic_evidence_refuted"),
        (
            "2 + 2 = 4. According to a source, this is documented.",
            "unsupported_evidence_dependency",
        ),
        ("A fluent answer without an exact check.", "no_exact_local_evidence"),
    ],
)
def test_refuted_unsupported_and_unchecked_targets_are_not_admitted(
    candidate: str,
    reason: str,
):
    admission, target_tokens = _admission(candidate)
    assert admission["admitted"] is False
    assert admission["reason"] == reason
    assert target_tokens == []


def _accepted_learning_receipt() -> dict:
    admission, _target_tokens = _admission("2 + 2 = 4.")
    winner_sha = hashlib.sha256(b"winner").hexdigest()
    state = empty_learning_state(
        episode_id="episode-1",
        input_tokens_sha256=hashlib.sha256(b"input").hexdigest(),
        selected_branch=0,
        winner_state_sha256=winner_sha,
        admission=admission,
    )
    probe_sha = hashlib.sha256(b"identity-probe").hexdigest()
    state["lease"] = {
        "schema": "aura.rlc.fast_weight_model_lease.v1",
        "owner_sha256": hashlib.sha256(b"owner").hexdigest(),
        "model_sha256": hashlib.sha256(b"model").hexdigest(),
        "acquired": True,
        "released": True,
        "conflicts": 0,
    }
    state["attach_identity"] = {
        "measured": True,
        "pre_probe_sha256": probe_sha,
        "post_probe_sha256": probe_sha,
        "exact": True,
        "winner_state_before_sha256": winner_sha,
        "winner_state_after_sha256": winner_sha,
    }
    state["optimization"] = {
        "optimizer": "rms_normalized_sgd_backtracking_v1",
        "attempts": 1,
        "accepted_steps": 1,
        "rejected_steps": 0,
        "budget_exhausted": False,
        "loss_trail": [1.0, 0.5],
        "gradient_norm_trail": [0.25],
        "accepted_step_sizes": [0.01],
        "line_search_backtracks": 0,
    }
    state["controls"] = {
        "decision": "accepted",
        "capability_canaries": {"decision": "accepted"},
    }
    state["causal_probe"] = {
        "evaluated": True,
        "pre_tokens_sha256": token_sequence_sha256([1]),
        "post_tokens_sha256": token_sequence_sha256([2]),
        "pre_text_sha256": admission["source_sha256"],
        "post_text_sha256": hashlib.sha256(b"improved").hexdigest(),
        "pre_score": 0.5,
        "post_score": 0.75,
        "token_sequence_changed": True,
        "strict_improvement": True,
        "winner_state_before_sha256": winner_sha,
        "winner_state_after_sha256": winner_sha,
    }
    state["final_answer"] = {
        "decoded_under_adaptation": True,
        "tokens_sha256": token_sequence_sha256([9, 10]),
        "text_sha256": hashlib.sha256(b"answer").hexdigest(),
        "token_count": 2,
    }
    state["cleanup"] = {
        "required": True,
        "detached": True,
        "erase_proven": True,
        "lease_released": True,
        "conflicts": 0,
        "pre_probe_sha256": probe_sha,
        "post_probe_sha256": probe_sha,
        "erased_layer_ids": ["layers.0.o_proj"],
    }
    state["disposition"] = "accepted_causal_improvement"
    return finalize_fast_weight_learning_receipt(state)


def test_complete_learning_receipt_reconstructs_and_detects_tampering():
    receipt = _accepted_learning_receipt()
    assert validate_fast_weight_learning_receipt(receipt)["disposition"] == (
        "accepted_causal_improvement"
    )

    tampered = copy.deepcopy(receipt)
    tampered["causal_probe"]["post_score"] = 0.1
    with pytest.raises(ValueError, match="commitment mismatch"):
        validate_fast_weight_learning_receipt(tampered)


def test_rehashed_receipt_cannot_claim_equal_tokens_were_causal():
    receipt = _accepted_learning_receipt()
    state = {key: copy.deepcopy(value) for key, value in receipt.items() if key != "receipt_sha256"}
    state["causal_probe"]["post_tokens_sha256"] = state["causal_probe"][
        "pre_tokens_sha256"
    ]
    state["causal_probe"]["token_sequence_changed"] = False
    with pytest.raises(ValueError, match="causal improvement"):
        finalize_fast_weight_learning_receipt(state)


def test_rehashed_receipt_cannot_decouple_token_hashes_from_change_verdict():
    receipt = _accepted_learning_receipt()
    state = {
        key: copy.deepcopy(value)
        for key, value in receipt.items()
        if key != "receipt_sha256"
    }
    state["causal_probe"]["post_tokens_sha256"] = state["causal_probe"][
        "pre_tokens_sha256"
    ]
    with pytest.raises(ValueError, match="causal probe evidence"):
        finalize_fast_weight_learning_receipt(state)


def test_rehashed_receipt_rejects_bool_like_lease_and_disposition_lies():
    receipt = _accepted_learning_receipt()
    state = {
        key: copy.deepcopy(value)
        for key, value in receipt.items()
        if key != "receipt_sha256"
    }
    state["lease"]["acquired"] = 1
    with pytest.raises(ValueError, match="boolean types"):
        finalize_fast_weight_learning_receipt(state)

    state["lease"]["acquired"] = True
    state["disposition"] = "rejected_no_accepted_step"
    state["final_answer"]["decoded_under_adaptation"] = False
    with pytest.raises(ValueError, match="no-step"):
        finalize_fast_weight_learning_receipt(state)


def test_strict_gain_reconstructs_from_unrounded_decision_scores():
    receipt = _accepted_learning_receipt()
    state = {
        key: copy.deepcopy(value)
        for key, value in receipt.items()
        if key != "receipt_sha256"
    }
    state["causal_probe"]["pre_score"] = 0.50000049
    state["causal_probe"]["post_score"] = 0.50000151

    rebuilt = finalize_fast_weight_learning_receipt(state)
    assert rebuilt["causal_probe"]["strict_improvement"] is True


def test_service_reconstructs_active_learning_and_output_binding():
    learning = _accepted_learning_receipt()
    receipt = {
        "episode_id": "episode-1",
        "input_tokens_sha256": hashlib.sha256(b"input").hexdigest(),
        "fast_weight_learning": learning,
        "fast_weights_applied": True,
        "fast_weights_erased": True,
        "fast_weights_layers": 2,
        "fast_weight_optimization_attempts": 1,
        "fast_weight_optimized_steps": 1,
        "fast_weight_rejected_steps": 0,
        "fast_weight_budget_exhausted": False,
        "fast_weight_optimizer": "rms_normalized_sgd_backtracking_v1",
        "fast_weight_loss_trail": [1.0, 0.5],
        "fast_weight_gradient_norm_trail": [0.25],
        "fast_weight_accepted_step_sizes": [0.01],
        "fast_weight_line_search_backtracks": 0,
    }
    errors = LatentCortexService._receipt_contract_errors(
        receipt,
        {"fast_weights": True},
        output_tokens=[9, 10],
        output_text="answer",
    )
    assert "fast_weight_learning_receipt_unproven" not in errors

    errors = LatentCortexService._receipt_contract_errors(
        receipt,
        {"fast_weights": True},
        output_tokens=[9, 11],
        output_text="answer",
    )
    assert "fast_weight_learning_receipt_unproven" in errors


def test_service_accepts_proven_ineligibility_without_model_mutation():
    input_sha = hashlib.sha256(b"ineligible-input").hexdigest()
    winner_sha = hashlib.sha256(b"ineligible-winner").hexdigest()
    state = empty_learning_state(
        episode_id="episode-ineligible",
        input_tokens_sha256=input_sha,
        selected_branch=0,
        winner_state_sha256=winner_sha,
        admission=unavailable_admission(
            source_sha256=hashlib.sha256(b"").hexdigest(),
            objective_sha256=hashlib.sha256(b"query").hexdigest(),
            reason="verifier_provider_untrusted",
        ),
    )
    state["final_answer"] = {
        "decoded_under_adaptation": False,
        "tokens_sha256": token_sequence_sha256([3]),
        "text_sha256": hashlib.sha256(b"base answer").hexdigest(),
        "token_count": 1,
    }
    receipt = {
        "episode_id": "episode-ineligible",
        "input_tokens_sha256": input_sha,
        "fast_weight_learning": finalize_fast_weight_learning_receipt(state),
        "fast_weights_applied": False,
        "fast_weights_erased": None,
        "fast_weights_layers": 0,
        "fast_weight_optimization_attempts": 0,
        "fast_weight_optimized_steps": 0,
        "fast_weight_rejected_steps": 0,
    }
    errors = LatentCortexService._receipt_contract_errors(
        receipt,
        {"fast_weights": True},
        output_tokens=[3],
        output_text="base answer",
    )
    assert "fast_weight_learning_receipt_unproven" not in errors
    assert "fast_weight_ineligible_episode_mutated_model" not in errors
