from __future__ import annotations

import copy
import hashlib
import json
import re

import pytest

from core.brain.llm.latent_cortex.generative_verifier import (
    FRESH_CONTEXT_SCHEMA,
    bind_selection_effect,
    build_verification_prompt,
    parse_generation_result,
    run_generative_verifier,
    validate_generative_verifier_envelope,
)


def _context() -> dict:
    return {
        "schema": FRESH_CONTEXT_SCHEMA,
        "prompt_token_count": 41,
        "generated_token_count": 18,
        "termination": "contract_complete",
        "initial_cache_offsets": [0, 0, 0],
        "final_cache_offsets": [58, 58, 58],
        "all_initial_offsets_zero": True,
        "solver_context_imported": False,
        "parameter_relation": "shared_resident_checkpoint",
    }


def _generator(*, verdict: str, witness: str):
    def generate(prompt: str) -> dict:
        claim = re.search(r"ANONYMIZED_CLAIM_SHA256: ([0-9a-f]{64})", prompt)
        assert claim is not None
        payload = {
            "claim_sha256": claim.group(1),
            "verdict": verdict,
            "witness": witness,
        }
        return {
            "text": "FINAL_ANSWER: " + json.dumps(payload, separators=(",", ":")),
            "context": _context(),
        }

    return generate


def test_prompt_is_ownership_free_and_target_bound():
    prompt = build_verification_prompt(
        objective="Compute the total.",
        atom="2 + 2 = 5",
        atom_sha256="a" * 64,
    )
    assert "branch" not in prompt.lower()
    assert "previous answer" not in prompt.lower()
    assert "a" * 64 in prompt
    assert "2 + 2 = 5" in prompt


def test_strict_generation_contract_refuses_rationale_and_wrong_binding():
    valid = 'FINAL_ANSWER: {"claim_sha256":"' + "b" * 64 + '","verdict":"unknown","witness":""}'
    assert parse_generation_result(valid, claim_sha256="b" * 64)["verdict"] == "unknown"
    with pytest.raises(ValueError, match="contract"):
        parse_generation_result("I think so. " + valid, claim_sha256="b" * 64)
    with pytest.raises(ValueError, match="bound"):
        parse_generation_result(valid, claim_sha256="c" * 64)


def test_fresh_derivation_can_veto_exactly_refuted_arithmetic():
    receipt = run_generative_verifier(
        "The result is 2 + 2 = 5.",
        objective="Compute 2 + 2 exactly.",
        generate=_generator(verdict="refutes", witness="2 + 2 = 4"),
    )
    assert receipt["parameter_independence"] is False
    assert receipt["context_independence"] is True
    assert receipt["admitted"] == 1
    assert receipt["causal_refutation"] is True
    assert receipt["attempts"][0]["relation"]["actual"] == 4
    assert validate_generative_verifier_envelope(receipt) == receipt

    bound = bind_selection_effect(receipt, vetoed_branch=0, replacement_branch=1)
    assert bound["selection_effect"] == "winner_replaced"
    assert bound["replacement_branch"] == 1
    assert validate_generative_verifier_envelope(bound) == bound


def test_unsupported_prose_abstains_even_when_generator_claims_refutation():
    receipt = run_generative_verifier(
        "The ungrounded system claim is definitely true.",
        objective="Assess the system claim.",
        generate=_generator(verdict="refutes", witness="The claim is false."),
    )
    assert receipt["attempted"] == 1
    assert receipt["admitted"] == 0
    assert receipt["causal_refutation"] is False
    assert receipt["attempts"][0]["relation"]["reason"] == (
        "relation_not_machine_checkable"
    )
    assert validate_generative_verifier_envelope(receipt) == receipt


def test_malformed_model_contract_retains_context_evidence_but_no_authority():
    receipt = run_generative_verifier(
        "This disputed prose claim has no deterministic route.",
        objective="Assess the claim.",
        generate=lambda _prompt: {"text": "unstructured answer", "context": _context()},
    )
    attempt = receipt["attempts"][0]
    assert attempt["generation_status"] == "complete"
    assert attempt["context"]["all_initial_offsets_zero"] is True
    assert attempt["authority_admitted"] is False
    assert attempt["relation"]["reason"].startswith("contract_refused:")
    assert validate_generative_verifier_envelope(receipt) == receipt


def test_verified_atoms_do_not_spend_fresh_generation_budget():
    called = False

    def generate(_prompt: str) -> dict:
        nonlocal called
        called = True
        raise AssertionError("verified atom must not be regenerated")

    receipt = run_generative_verifier(
        "2 + 2 = 4",
        objective="Compute 2 + 2.",
        generate=generate,
    )
    assert receipt["attempted"] == 0
    assert called is False
    assert validate_generative_verifier_envelope(receipt) == receipt


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value["attempts"][0]["context"].update({"solver_context_imported": True}), "commitment"),
        (lambda value: value["attempts"][0]["relation"].update({"actual": 7}), "commitment"),
        (lambda value: value.update({"selection_effect": "winner_replaced"}), "commitment"),
    ],
)
def test_tampered_receipts_fail_closed(mutate, match):
    receipt = run_generative_verifier(
        "2 + 2 = 5",
        objective="Compute 2 + 2.",
        generate=_generator(verdict="refutes", witness="2 + 2 = 4"),
    )
    tampered = copy.deepcopy(receipt)
    mutate(tampered)
    with pytest.raises(ValueError, match=match):
        validate_generative_verifier_envelope(tampered)


def test_recommitted_imported_solver_context_still_fails_validation():
    receipt = run_generative_verifier(
        "2 + 2 = 5",
        objective="Compute 2 + 2.",
        generate=_generator(verdict="refutes", witness="2 + 2 = 4"),
    )
    forged = copy.deepcopy(receipt)
    forged["attempts"][0]["context"]["solver_context_imported"] = True
    payload = {key: value for key, value in forged.items() if key != "receipt_sha256"}
    forged["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="context isolation"):
        validate_generative_verifier_envelope(forged)


def test_recommitted_challenge_reordering_cannot_cherry_pick_atoms():
    receipt = run_generative_verifier(
        "2 + 2 = 5. This unrelated prose claim is certain.",
        objective="Check both claims.",
        generate=_generator(verdict="refutes", witness="2 + 2 = 4"),
        max_atoms=2,
    )
    assert receipt["attempted"] == 2
    forged = copy.deepcopy(receipt)
    forged["attempts"].reverse()
    payload = {key: value for key, value in forged.items() if key != "receipt_sha256"}
    forged["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="cherry-picked"):
        validate_generative_verifier_envelope(forged)


def test_service_reconstructs_causal_replacement_and_rejects_mismatch():
    from core.brain.latent_cortex_service import LatentCortexService

    base = run_generative_verifier(
        "2 + 2 = 5",
        objective="Compute 2 + 2.",
        generate=_generator(verdict="refutes", witness="2 + 2 = 4"),
    )
    bound = bind_selection_effect(base, vetoed_branch=0, replacement_branch=1)
    config = {"generative_verifier_enabled": True}
    accepted_errors = LatentCortexService._receipt_contract_errors(
        {"generative_verifier": bound, "selected_branch": 1},
        config,
    )
    assert "generative_verifier_unproven" not in accepted_errors

    rejected_errors = LatentCortexService._receipt_contract_errors(
        {"generative_verifier": bound, "selected_branch": 0},
        config,
    )
    assert "generative_verifier_unproven" in rejected_errors
