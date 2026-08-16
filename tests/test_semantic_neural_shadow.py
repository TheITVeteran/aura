from __future__ import annotations

import hashlib
import json

import pytest

from core.brain.llm.semantic_neural_shadow import (
    build_semantic_shadow_comparison,
    record_semantic_shadow_comparison,
)


def _admission():
    return {
        "family": "frontier_calibration",
        "parser_id": "semantic_calibration_canonical.v1",
        "receipt_sha256": "a" * 64,
    }


def _activation():
    return {
        "package_id": "cp568-resident-semantic-neural-shadow",
        "promotion_mode": "shadow",
        "activation_sha256": "b" * 64,
    }


def _receipt_sha(payload):
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    return hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def test_shadow_comparison_detects_match_without_retaining_content():
    objective = "private prompt text"
    qualified = 'FINAL_ANSWER: {"posterior_denominator":7,"posterior_numerator":3}'
    ordinary = (
        "I checked the update.\n"
        'FINAL_ANSWER: {"posterior_numerator":3,"posterior_denominator":7}'
    )
    comparison = build_semantic_shadow_comparison(
        objective=objective,
        qualified_text=qualified,
        ordinary_text=ordinary,
        admission_receipt=_admission(),
        activation_receipt=_activation(),
    )

    assert comparison["ordinary_answer_parsed"] is True
    assert comparison["answer_match"] is True
    assert comparison["qualified_gain_candidate"] is False
    assert comparison["ordinary_success_preserved"] is True
    assert comparison["receipt_sha256"] == _receipt_sha(comparison)
    wire = json.dumps(comparison, sort_keys=True)
    assert objective not in wire
    assert qualified not in wire
    assert ordinary not in wire


def test_shadow_comparison_detects_candidate_gain_over_unparsed_ordinary_answer():
    comparison = build_semantic_shadow_comparison(
        objective="bounded task",
        qualified_text='FINAL_ANSWER: {"residue":11}',
        ordinary_text="I am not sure.",
        admission_receipt=_admission(),
        activation_receipt=_activation(),
    )

    assert comparison["ordinary_answer_parsed"] is False
    assert comparison["answer_match"] is False
    assert comparison["qualified_gain_candidate"] is True
    assert comparison["ordinary_success_preserved"] is False


def test_shadow_comparison_refuses_noncanonical_qualified_authority():
    with pytest.raises(ValueError, match="not canonical"):
        build_semantic_shadow_comparison(
            objective="bounded task",
            qualified_text="maybe 11",
            ordinary_text='FINAL_ANSWER: {"residue":11}',
            admission_receipt=_admission(),
            activation_receipt=_activation(),
        )


@pytest.mark.asyncio
async def test_shadow_comparison_persists_one_privacy_bounded_record(tmp_path):
    ledger = tmp_path / "shadow.jsonl"
    comparison = await record_semantic_shadow_comparison(
        objective="do not retain this prompt",
        qualified_text='FINAL_ANSWER: {"residue":11}',
        ordinary_text='FINAL_ANSWER: {"residue":10}',
        admission_receipt=_admission(),
        activation_receipt=_activation(),
        ledger_path=ledger,
    )

    assert comparison["persisted"] is True
    records = ledger.read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    persisted = json.loads(records[0])
    assert persisted["receipt_sha256"] == comparison["receipt_sha256"]
    assert persisted["raw_prompt_retained"] is False
    assert persisted["raw_answers_retained"] is False
    assert "do not retain this prompt" not in records[0]
