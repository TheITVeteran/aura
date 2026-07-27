from __future__ import annotations

import re
import unicodedata

import pytest

from core.learning import recurrent_sft_retention as retention
from core.learning.recurrent_sft_behavior_canaries import (
    build_generated_behavior_canaries,
)
from core.learning.recurrent_sft_retention import (
    RETENTION_FAMILIES,
    RecurrentSFTRetentionError,
    build_retention_rows,
    retention_manifest,
)


def _normalize(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", text).casefold(),
    ).strip()


def _word_ngrams(text: str, size: int) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", _normalize(text))
    return {
        tuple(words[index : index + size])
        for index in range(max(0, len(words) - size + 1))
    }


def test_retention_splits_are_deterministic_balanced_and_disjoint() -> None:
    train = build_retention_rows("train")
    validation = build_retention_rows("validation")
    assert train == build_retention_rows("train")
    assert validation == build_retention_rows("validation")
    assert len(train) == 24
    assert len(validation) == 12
    for rows, expected_per_family in ((train, 8), (validation, 4)):
        assert {
            family: sum(row["_meta"]["family"] == family for row in rows)
            for family in RETENTION_FAMILIES
        } == dict.fromkeys(RETENTION_FAMILIES, expected_per_family)
    assert {
        row["_meta"]["case_fingerprint"] for row in train
    }.isdisjoint(
        row["_meta"]["case_fingerprint"] for row in validation
    )


def test_rows_match_chat_projection_and_supervision_contract() -> None:
    for row in build_retention_rows("train"):
        assert [message["role"] for message in row["messages"]] == [
            "system",
            "user",
            "assistant",
        ]
        assert row["tools"] == []
        assert row["_meta"]["target_kind"] == "behavior_retention"
        assert row["_meta"]["loss_policy"]["mask_prompt"] is True
        assert (
            row["_meta"]["loss_policy"]["supervised_region"]
            == "final_assistant_message_only"
        )
        assert row["_meta"]["projection"] == {
            "answer_evidence_in_input": False,
            "oracle_fields_exported_to_trainer": [],
        }


def test_training_prompts_do_not_reuse_evaluator_prompts_or_long_phrases() -> None:
    retention_prompts = {
        _normalize(row["messages"][1]["content"])
        for split in ("train", "validation")
        for row in build_retention_rows(split)
    }
    evaluator_prompts = {
        _normalize(case["prompt"])
        for case in build_generated_behavior_canaries()
    }
    assert retention_prompts.isdisjoint(evaluator_prompts)
    for retention_prompt in retention_prompts:
        retention_ngrams = _word_ngrams(retention_prompt, 8)
        for evaluator in evaluator_prompts:
            assert retention_ngrams.isdisjoint(
                _word_ngrams(evaluator, 8)
            )


def test_curriculum_contains_positive_authorized_action_examples() -> None:
    answers = [
        row["messages"][-1]["content"]
        for row in build_retention_rows("train")
        if row["_meta"]["family"] == "authority_safety"
    ]
    assert any("can perform" in answer for answer in answers)
    assert any("can execute" in answer for answer in answers)
    assert any("will not" in answer for answer in answers)


def test_manifest_replays_all_split_commitments() -> None:
    first = retention_manifest()
    assert first == retention_manifest()
    assert first["split_case_overlap_count"] == 0
    assert first["evaluator_prompts_included"] is False
    assert first["evaluator_separation"]["exact_prompt_overlap_count"] == 0
    assert first["evaluator_separation"]["long_ngram_overlap_count"] == 0
    assert len(
        first["evaluator_separation"]["evaluator_registry_sha256"]
    ) == 64
    assert first["splits"]["train"]["example_count"] == 24
    assert first["splits"]["validation"]["example_count"] == 12
    assert len(first["manifest_sha256"]) == 64


def test_unknown_split_fails_closed() -> None:
    with pytest.raises(
        RecurrentSFTRetentionError,
        match="split_invalid",
    ):
        build_retention_rows("holdout")


def test_split_identity_does_not_hide_duplicate_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicated = dict(retention._VALIDATION_CASES)
    duplicated["identity_grounding"] = (
        retention._TRAIN_CASES["identity_grounding"][0],
        *duplicated["identity_grounding"][1:],
    )
    monkeypatch.setattr(retention, "_VALIDATION_CASES", duplicated)
    with pytest.raises(
        RecurrentSFTRetentionError,
        match="split_overlap",
    ):
        retention_manifest()


def test_manifest_runtime_rejects_evaluator_prompt_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = build_retention_rows("train")[0]["messages"][1]["content"]
    monkeypatch.setattr(
        "core.learning.recurrent_sft_behavior_canaries."
        "build_generated_behavior_canaries",
        lambda: [{"prompt": prompt}],
    )
    with pytest.raises(
        RecurrentSFTRetentionError,
        match="evaluator_overlap",
    ):
        retention_manifest()
