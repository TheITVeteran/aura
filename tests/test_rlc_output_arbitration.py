"""Serving authority for RLC is conservative, bound, and replayable."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from itertools import product

import pytest

from core.brain.llm.latent_cortex.output_arbitration import (
    ArbitrationDecision,
    CandidateSource,
    OutputCandidate,
    assert_output_arbitration_no_regression,
    build_output_arbitration_receipt,
    validate_output_arbitration_receipt,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _candidate(
    source: CandidateSource,
    text: str,
    *,
    contract_valid: bool = True,
    product_valid: bool | None = None,
    independent: bool = False,
    full_span: bool = False,
    lower: float = 0.0,
    upper: float = 1.0,
    material_regression: bool = False,
) -> OutputCandidate:
    return OutputCandidate(
        source=source,
        text=text,
        tokens=tuple(text.encode("utf-8")) if text else (),
        request_sha256=_sha("request"),
        model_sha256=_sha("model"),
        seed=17,
        contract_sha256=_sha("contract"),
        contract_valid=contract_valid,
        product_valid=product_valid,
        verifier_independent=independent,
        full_span_coverage=full_span,
        quality_lower_bound=lower,
        quality_upper_bound=upper,
        verifier_receipt_sha256=_sha("verifier") if independent else "",
        material_regression=material_regression,
    )


def _verified(
    source: CandidateSource,
    text: str,
    *,
    lower: float,
    upper: float,
) -> OutputCandidate:
    return _candidate(
        source,
        text,
        product_valid=True,
        independent=True,
        full_span=True,
        lower=lower,
        upper=upper,
    )


def test_contract_complete_but_wrong_rlc_cannot_displace_vanilla() -> None:
    vanilla = _candidate(CandidateSource.VANILLA, "correct ordinary answer")
    rlc = _candidate(CandidateSource.RLC, "confidently wrong recurrent answer")

    receipt, text, _tokens = build_output_arbitration_receipt(vanilla, rlc)

    assert receipt["decision"] == ArbitrationDecision.RETAIN_VANILLA.value
    assert receipt["reason"] == "rlc_lacks_independent_full_span_authority"
    assert text == vanilla.text


def test_process_proxy_scores_have_no_serving_authority() -> None:
    vanilla = _candidate(CandidateSource.VANILLA, "incumbent", lower=0.0, upper=0.01)
    rlc = _candidate(CandidateSource.RLC, "high process score", lower=0.99, upper=1.0)

    receipt, text, _tokens = build_output_arbitration_receipt(vanilla, rlc)

    assert receipt["decision"] == ArbitrationDecision.RETAIN_VANILLA.value
    assert receipt["policy"]["proxy_score_authority"] == "none"
    assert text == "incumbent"


def test_independently_verified_dominance_can_select_rlc() -> None:
    vanilla = _verified(
        CandidateSource.VANILLA,
        "weak incumbent",
        lower=0.1,
        upper=0.3,
    )
    rlc = _verified(CandidateSource.RLC, "proved better", lower=0.8, upper=0.9)

    receipt, text, tokens = build_output_arbitration_receipt(vanilla, rlc)

    assert receipt["decision"] == ArbitrationDecision.SELECT_RLC.value
    assert receipt["reason"] == "rlc_authoritative_lower_bound_dominates_vanilla"
    assert text == rlc.text
    assert tokens == rlc.tokens


def test_authoritative_rlc_can_rescue_an_invalid_vanilla_output() -> None:
    vanilla = _candidate(
        CandidateSource.VANILLA,
        "malformed",
        contract_valid=False,
    )
    rlc = _verified(CandidateSource.RLC, "valid rescue", lower=0.8, upper=0.9)

    receipt, text, _tokens = build_output_arbitration_receipt(vanilla, rlc)

    assert receipt["decision"] == ArbitrationDecision.SELECT_RLC.value
    assert receipt["reason"] == "rlc_authoritative_rescue"
    assert text == "valid rescue"


def test_unverified_rlc_cannot_rescue_an_invalid_vanilla_output() -> None:
    vanilla = _candidate(CandidateSource.VANILLA, "malformed", contract_valid=False)
    rlc = _candidate(CandidateSource.RLC, "looks valid")

    receipt, text, tokens = build_output_arbitration_receipt(vanilla, rlc)

    assert receipt["decision"] == ArbitrationDecision.NO_ADMISSIBLE_OUTPUT.value
    assert text == ""
    assert tokens == ()


def test_material_regression_blocks_even_measured_rlc() -> None:
    vanilla = _candidate(CandidateSource.VANILLA, "incumbent")
    rlc = replace(
        _verified(CandidateSource.RLC, "regression", lower=1.0, upper=1.0),
        material_regression=True,
    )

    receipt, text, _tokens = build_output_arbitration_receipt(vanilla, rlc)

    assert receipt["reason"] == "rlc_material_regression"
    assert text == vanilla.text


def test_ties_and_equivalent_outputs_keep_the_incumbent() -> None:
    vanilla = _verified(CandidateSource.VANILLA, "same", lower=0.8, upper=0.9)
    rlc = _verified(CandidateSource.RLC, "same", lower=0.9, upper=1.0)

    receipt, text, _tokens = build_output_arbitration_receipt(vanilla, rlc)

    assert receipt["reason"] == "equivalent_output_keeps_incumbent"
    assert text == "same"


def test_candidate_bindings_must_refer_to_the_same_experiment() -> None:
    vanilla = _candidate(CandidateSource.VANILLA, "incumbent")
    rlc = replace(
        _candidate(CandidateSource.RLC, "challenger"),
        request_sha256=_sha("different request"),
    )

    with pytest.raises(ValueError, match="bindings differ"):
        build_output_arbitration_receipt(vanilla, rlc)


def test_measured_product_claim_requires_independent_evidence() -> None:
    with pytest.raises(ValueError, match="independent verifier"):
        _candidate(
            CandidateSource.RLC,
            "unsupported claim",
            product_valid=True,
            full_span=True,
        )


def test_positive_product_claim_requires_full_span_coverage() -> None:
    with pytest.raises(ValueError, match="full-span coverage"):
        _candidate(
            CandidateSource.RLC,
            "partial",
            product_valid=True,
            independent=True,
        )


def test_receipt_replay_binds_the_exact_selected_output() -> None:
    vanilla = _candidate(CandidateSource.VANILLA, "incumbent")
    rlc = _candidate(CandidateSource.RLC, "challenger")
    receipt, text, tokens = build_output_arbitration_receipt(vanilla, rlc)

    rebuilt = validate_output_arbitration_receipt(
        receipt,
        vanilla=vanilla,
        rlc=rlc,
        expected_output_text=text,
        expected_output_tokens=tokens,
    )

    assert rebuilt == receipt


def test_receipt_tampering_is_detected() -> None:
    vanilla = _candidate(CandidateSource.VANILLA, "incumbent")
    rlc = _candidate(CandidateSource.RLC, "challenger")
    receipt, _text, _tokens = build_output_arbitration_receipt(vanilla, rlc)
    tampered = {**receipt, "decision": ArbitrationDecision.SELECT_RLC.value}

    with pytest.raises(ValueError, match="reconstruction differs"):
        validate_output_arbitration_receipt(tampered, vanilla=vanilla, rlc=rlc)


def test_no_regression_assertion_rejects_forged_rlc_selection() -> None:
    vanilla = _candidate(CandidateSource.VANILLA, "incumbent")
    rlc = _candidate(CandidateSource.RLC, "challenger")
    receipt, _text, _tokens = build_output_arbitration_receipt(vanilla, rlc)
    forged = {
        **receipt,
        "decision": ArbitrationDecision.SELECT_RLC.value,
        "selected_source": CandidateSource.RLC.value,
    }

    with pytest.raises(ValueError, match="lacks independent full-span authority"):
        assert_output_arbitration_no_regression(forged)


@pytest.mark.parametrize(
    ("vanilla_contract", "rlc_contract", "rlc_measured", "rlc_regression"),
    product((False, True), repeat=4),
)
def test_property_rlc_selection_always_has_authority_or_rescues_invalid_incumbent(
    vanilla_contract: bool,
    rlc_contract: bool,
    rlc_measured: bool,
    rlc_regression: bool,
) -> None:
    vanilla = _candidate(
        CandidateSource.VANILLA,
        "vanilla",
        contract_valid=vanilla_contract,
        upper=0.2,
    )
    rlc = (
        _verified(CandidateSource.RLC, "rlc", lower=0.8, upper=0.9)
        if rlc_measured
        else _candidate(CandidateSource.RLC, "rlc")
    )
    rlc = replace(
        rlc,
        contract_valid=rlc_contract,
        material_regression=rlc_regression,
    )

    receipt, _text, _tokens = build_output_arbitration_receipt(vanilla, rlc)
    assert_output_arbitration_no_regression(receipt)
    if receipt["decision"] == ArbitrationDecision.SELECT_RLC.value:
        evidence = receipt["candidates"]["rlc"]
        assert evidence["authoritative"] is True
        assert evidence["material_regression"] is False
        assert vanilla.admissible is False or (
            evidence["quality_interval"]["lower_bound"]
            > receipt["candidates"]["vanilla"]["quality_interval"]["upper_bound"]
            + receipt["policy"]["margin"]
        )
