from __future__ import annotations

import copy
import hashlib
import json
import re

import pytest

from core.brain.llm.latent_cortex.counterfactual_verifier import (
    build_counterfactual_prompt,
    parse_counterfactual_result,
    run_counterfactual_verifier,
    validate_counterfactual_verifier_envelope,
)
from core.brain.llm.latent_cortex.generative_verifier import FRESH_CONTEXT_SCHEMA


def _context() -> dict:
    return {
        "schema": FRESH_CONTEXT_SCHEMA,
        "prompt_token_count": 47,
        "generated_token_count": 18,
        "termination": "contract_complete",
        "initial_cache_offsets": [0, 0, 0],
        "final_cache_offsets": [65, 65, 65],
        "all_initial_offsets_zero": True,
        "solver_context_imported": False,
        "parameter_relation": "shared_resident_checkpoint",
    }


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _receipt_sha(value: dict) -> str:
    payload = {key: value[key] for key in value if key != "receipt_sha256"}
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _generator(modes: dict[str, str]):
    calls: list[str] = []

    def generate(prompt: str) -> dict:
        calls.append(prompt)
        claim_sha = re.search(r"ANONYMIZED_CLAIM_SHA256: ([0-9a-f]{64})", prompt)
        intervention_sha = re.search(r"INTERVENTION_SHA256: ([0-9a-f]{64})", prompt)
        inputs = re.search(r"COUNTERFACTUAL_INPUT: (-?\d+) ([+\-*/]) (-?\d+)", prompt)
        claim_text = re.search(
            r"ANONYMIZED_CLAIM:\n(.*?)\nINTERVENTION_SHA256:",
            prompt,
            re.DOTALL,
        )
        assert claim_sha and intervention_sha and inputs and claim_text
        left, operator, right = int(inputs.group(1)), inputs.group(2), int(inputs.group(3))
        actual = {
            "+": left + right,
            "-": left - right,
            "*": left * right,
            "/": left // right,
        }[operator]
        original = re.search(r"=\s*(-?\d+)", claim_text.group(1))
        assert original
        mode = modes[_sha(claim_text.group(1))]
        predicted = actual if mode == "correct" else int(original.group(1))
        payload = {
            "claim_sha256": claim_sha.group(1),
            "intervention_sha256": intervention_sha.group(1),
            "prediction": f"{left} {operator} {right} = {predicted}",
        }
        return {
            "text": "FINAL_ANSWER: " + json.dumps(payload, separators=(",", ":")),
            "context": _context(),
        }

    return generate, calls


def test_prompt_is_branch_blind_and_binds_exact_intervention():
    intervention = {
        "family": "left_input_delta",
        "before": {
            "left": 2,
            "operator": "+",
            "right": 2,
            "claimed_result": 4,
            "actual_result": 4,
        },
        "after": {"left": 3, "operator": "+", "right": 2, "expected_result": 5},
        "expected_consequence_changed": True,
        "intervention_sha256": "b" * 64,
    }
    prompt = build_counterfactual_prompt(
        objective="Compute the total.",
        claim_text="2 + 2 = 4",
        claim_sha256="a" * 64,
        intervention=intervention,
    )
    assert "branch" not in prompt.lower()
    assert "solver state" in prompt.lower()
    assert "a" * 64 in prompt
    assert "b" * 64 in prompt
    assert "3 + 2" in prompt


def test_counterfactual_contract_rejects_prose_and_stale_bindings():
    valid = (
        'FINAL_ANSWER: {"claim_sha256":"'
        + "a" * 64
        + '","intervention_sha256":"'
        + "b" * 64
        + '","prediction":"3 + 2 = 5"}'
    )
    assert (
        parse_counterfactual_result(
            valid,
            claim_sha256="a" * 64,
            intervention_sha256="b" * 64,
        )["prediction"]
        == "3 + 2 = 5"
    )
    with pytest.raises(ValueError, match="contract"):
        parse_counterfactual_result(
            "because " + valid,
            claim_sha256="a" * 64,
            intervention_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="intervention"):
        parse_counterfactual_result(
            valid,
            claim_sha256="a" * 64,
            intervention_sha256="c" * 64,
        )


def test_complete_equal_score_counterfactual_can_replace_invariant_winner():
    candidates = {
        0: "The first result is 4 + 3 = 7.",
        1: "The second result is 4 + 3 = 7.",
    }
    generate, calls = _generator(
        {
            _sha(candidates[0]): "invariant",
            _sha(candidates[1]): "correct",
        }
    )
    receipt = run_counterfactual_verifier(
        candidates,
        objective="Compute 4 + 3 and explain sensitivity.",
        task_scores={0: 1.0, 1: 1.0},
        selected_branch=0,
        generate=generate,
    )
    assert len(calls) == 4
    assert receipt["all_tied_branches_covered"] is True
    assert receipt["selection_authority_admitted"] is True
    assert receipt["selection_effect"] == "winner_replaced"
    assert receipt["source_selected_branch"] == 0
    assert receipt["selected_branch"] == 1
    assert receipt["branches"][0]["invariant_failures"] == 2
    assert receipt["branches"][1]["correct_changes"] == 2
    assert validate_counterfactual_verifier_envelope(receipt) == receipt


def test_non_tied_scores_never_spend_generation_or_override_stronger_evidence():
    called = False

    def generate(_prompt: str) -> dict:
        nonlocal called
        called = True
        raise AssertionError("non-tied candidate must not be challenged")

    receipt = run_counterfactual_verifier(
        {0: "4 + 3 = 7", 1: "4 + 3 = 8"},
        objective="Compute 4 + 3.",
        task_scores={0: 1.0, 1: 0.0},
        selected_branch=0,
        generate=generate,
    )
    assert called is False
    assert receipt["tied_branches"] == [0]
    assert receipt["selection_authority_admitted"] is False
    assert receipt["selection_effect"] == "none"
    assert validate_counterfactual_verifier_envelope(receipt) == receipt


def test_incomplete_or_unsupported_tie_abstains_without_partial_ranking():
    generate, _calls = _generator(
        {
            _sha("4 + 3 = 7"): "correct",
            _sha("A prose-only answer."): "correct",
        }
    )
    receipt = run_counterfactual_verifier(
        {0: "4 + 3 = 7", 1: "A prose-only answer."},
        objective="Compute 4 + 3.",
        task_scores={0: 0.5, 1: 0.5},
        selected_branch=0,
        generate=generate,
    )
    assert receipt["branches"][0]["attempted"] == 2
    assert receipt["branches"][1]["attempted"] == 0
    assert receipt["all_tied_branches_covered"] is False
    assert receipt["selection_authority_admitted"] is False


def test_multiway_tie_remains_unresolved_when_best_evidence_is_still_tied():
    candidates = {
        0: "The first result is 4 + 3 = 7.",
        1: "The second result is 4 + 3 = 7.",
        2: "The third result is 4 + 3 = 7.",
    }
    generate, _calls = _generator(
        {
            _sha(candidates[0]): "correct",
            _sha(candidates[1]): "correct",
            _sha(candidates[2]): "invariant",
        }
    )
    receipt = run_counterfactual_verifier(
        candidates,
        objective="Compute 4 + 3 and explain sensitivity.",
        task_scores={0: 1.0, 1: 1.0, 2: 1.0},
        selected_branch=1,
        generate=generate,
    )
    assert receipt["all_tied_branches_covered"] is True
    assert receipt["selection_authority_admitted"] is False
    assert receipt["selection_effect"] == "none"
    assert receipt["selected_branch"] == 1
    assert validate_counterfactual_verifier_envelope(receipt) == receipt


def test_service_reconstructs_tiebreak_and_rejects_unapplied_selection():
    from core.brain.latent_cortex_service import LatentCortexService

    candidates = {0: "First: 4 + 3 = 7.", 1: "Second: 4 + 3 = 7."}
    generate, _calls = _generator(
        {_sha(candidates[0]): "invariant", _sha(candidates[1]): "correct"}
    )
    counterfactual = run_counterfactual_verifier(
        candidates,
        objective="Compute 4 + 3.",
        task_scores={0: 1.0, 1: 1.0},
        selected_branch=0,
        generate=generate,
    )
    receipt = {
        "counterfactual_verifier": counterfactual,
        "blind_review": {
            "rows": [
                {"branch": 0, "score": 1.0},
                {"branch": 1, "score": 1.0},
            ]
        },
        "selected_branch": 1,
    }
    config = {
        "counterfactual_verifier_enabled": True,
        "generative_verifier_enabled": False,
    }
    accepted = LatentCortexService._receipt_contract_errors(receipt, config)
    assert "counterfactual_verifier_unproven" not in accepted

    rejected = LatentCortexService._receipt_contract_errors(
        {**receipt, "selected_branch": 0},
        config,
    )
    assert "counterfactual_verifier_unproven" in rejected


def test_service_reconstructs_runtime_six_decimal_task_scores():
    from core.brain.latent_cortex_service import LatentCortexService

    candidates = {0: "First: 4 + 3 = 7.", 1: "Second: 4 + 3 = 7."}
    generate, _calls = _generator(
        {_sha(candidates[0]): "invariant", _sha(candidates[1]): "correct"}
    )
    counterfactual = run_counterfactual_verifier(
        candidates,
        objective="Compute 4 + 3.",
        task_scores={0: round(0.8123456123, 6), 1: round(0.8123456789, 6)},
        selected_branch=1,
        generate=generate,
    )
    errors = LatentCortexService._receipt_contract_errors(
        {
            "counterfactual_verifier": counterfactual,
            "blind_review": {
                "rows": [
                    {"branch": 0, "score": 0.8123456123},
                    {"branch": 1, "score": 0.8123456789},
                ]
            },
            "selected_branch": counterfactual["selected_branch"],
        },
        {
            "counterfactual_verifier_enabled": True,
            "generative_verifier_enabled": False,
        },
    )
    assert "counterfactual_verifier_unproven" not in errors


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["branches"][0]["attempts"][0]["context"].update(
                {"solver_context_imported": True}
            ),
            "context isolation",
        ),
        (
            lambda value: value["branches"][0]["attempts"][0]["intervention"]["after"].update(
                {"expected_result": 999}
            ),
            "attempt identity",
        ),
        (
            lambda value: value.update({"selected_branch": 0}),
            "selection effect",
        ),
    ],
)
def test_recommitted_counterfactual_tampering_fails_closed(mutation, match):
    candidates = {0: "First: 4 + 3 = 7.", 1: "Second: 4 + 3 = 7."}
    generate, _calls = _generator(
        {_sha(candidates[0]): "invariant", _sha(candidates[1]): "correct"}
    )
    receipt = run_counterfactual_verifier(
        candidates,
        objective="Compute 4 + 3.",
        task_scores={0: 1.0, 1: 1.0},
        selected_branch=0,
        generate=generate,
    )
    forged = copy.deepcopy(receipt)
    mutation(forged)
    forged["receipt_sha256"] = _receipt_sha(forged)
    with pytest.raises(ValueError, match=match):
        validate_counterfactual_verifier_envelope(forged)


def test_malformed_branch_rows_and_abstention_evidence_fail_closed():
    candidates = {0: "First: 4 + 3 = 7.", 1: "Second: 4 + 3 = 7."}
    generate, _calls = _generator(
        {_sha(candidates[0]): "invariant", _sha(candidates[1]): "correct"}
    )
    receipt = run_counterfactual_verifier(
        candidates,
        objective="Compute 4 + 3.",
        task_scores={0: 1.0, 1: 1.0},
        selected_branch=0,
        generate=generate,
    )

    malformed = copy.deepcopy(receipt)
    malformed["branches"][0] = "not-a-branch"
    malformed["receipt_sha256"] = _receipt_sha(malformed)
    with pytest.raises(ValueError, match="branch coverage"):
        validate_counterfactual_verifier_envelope(malformed)

    abstained = copy.deepcopy(receipt)
    attempt = abstained["branches"][0]["attempts"][0]
    attempt["outcome"] = "abstained"
    attempt["prediction_text"] = ""
    attempt["prediction_sha256"] = ""
    attempt["evidence"] = {"arbitrary": True}
    abstained["branches"][0]["admitted"] -= 1
    abstained["branches"][0]["invariant_failures"] -= 1
    abstained["branches"][0]["complete_coverage"] = False
    abstained["branches"][0]["robustness_score"] = 0.0
    abstained["receipt_sha256"] = _receipt_sha(abstained)
    with pytest.raises(ValueError, match="abstained counterfactual"):
        validate_counterfactual_verifier_envelope(abstained)
