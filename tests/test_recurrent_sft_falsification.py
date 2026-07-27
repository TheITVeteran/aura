from __future__ import annotations

import copy

import pytest

from core.learning.recurrent_sft_falsification import (
    ALL_ARMS,
    BASE_ARM,
    CONTROL_ARMS,
    TRAINED_ARM,
    RecurrentSFTFalsificationError,
    build_falsification_verdict,
    compare_observation_arms,
    exact_two_sided_sign_test,
    transform_control_rows,
)


def _rows() -> list[dict]:
    return [
        {
            "example_id": f"{index + 1:064x}",
            "family": "logic" if index % 2 == 0 else "tool",
            "target_kind": "answer",
            "prompt_tokens": [100 + index, 200 + index],
            "answer_tokens": [
                10 + index,
                20 + index,
                30 + index,
                40 + index,
                99,
            ],
            "full_token_count": 7,
        }
        for index in range(4)
    ]


def _observations(
    *,
    loss_shift: float,
    improve_tokens: bool,
) -> list[dict]:
    rows: list[dict] = []
    for index in range(8):
        before = [False, True, False, True]
        after = [True, True, True, True] if improve_tokens else before
        rows.append(
            {
                "example_id": f"{index + 1:064x}",
                "family": "logic" if index < 4 else "tool",
                "loss": 1.0 + loss_shift + index * 0.001,
                "target_top1": after,
                "generated_correct": bool(improve_tokens),
            }
        )
    return rows


@pytest.mark.parametrize("arm", CONTROL_ARMS)
def test_control_arms_are_deterministic_and_workload_equivalent(arm: str) -> None:
    original = _rows()
    surfaces = {
        token: ("+" if token in {20, 22, 40, 42, 99} else "word")
        for row in original
        for token in row["answer_tokens"]
    }
    first = transform_control_rows(
        original,
        arm=arm,
        seed=17,
        token_surfaces=surfaces,
        neutral_token_id=7,
    )
    second = transform_control_rows(
        copy.deepcopy(original),
        arm=arm,
        seed=17,
        token_surfaces=surfaces,
        neutral_token_id=7,
    )

    assert first == second
    assert first.manifest["changed_answer_tokens"] > 0
    assert first.manifest["prompt_tokens_unchanged"] is True
    assert first.manifest["answer_lengths_unchanged"] is True
    assert first.manifest["terminal_tokens_unchanged"] is True
    assert first.manifest["full_token_budget"] == 28
    for before, after in zip(original, first.rows, strict=True):
        assert after["prompt_tokens"] == before["prompt_tokens"]
        assert len(after["answer_tokens"]) == len(before["answer_tokens"])
        assert after["answer_tokens"][-1] == before["answer_tokens"][-1]
        assert after["full_token_count"] == before["full_token_count"]


def test_sham_labels_take_content_from_another_example() -> None:
    rows = _rows()
    transformed = transform_control_rows(
        rows,
        arm="sham_labels",
        seed=17,
    )
    assert transformed.rows[0]["answer_tokens"][:-1] == rows[1]["answer_tokens"][:-1]
    assert transformed.rows[-1]["answer_tokens"][:-1] == rows[0]["answer_tokens"][:-1]


def test_shuffled_trace_changes_order_but_preserves_multiset() -> None:
    row = _rows()[0]
    transformed = transform_control_rows(
        [row, _rows()[1]],
        arm="shuffled_traces",
        seed=23,
    ).rows[0]
    assert sorted(transformed["answer_tokens"][:-1]) == sorted(
        row["answer_tokens"][:-1]
    )
    assert transformed["answer_tokens"][:-1] != row["answer_tokens"][:-1]


def test_syntax_only_retains_punctuation_and_masks_alphanumerics() -> None:
    rows = _rows()
    surfaces = {
        token: ("+" if token in {20, 22, 40, 42, 99} else "value")
        for row in rows
        for token in row["answer_tokens"]
    }
    transformed = transform_control_rows(
        rows,
        arm="syntax_only",
        seed=17,
        token_surfaces=surfaces,
        neutral_token_id=7,
    ).rows[0]["answer_tokens"]
    assert transformed == [7, 20, 7, 40, 99]


def test_control_contract_rejects_bad_rows_and_missing_surfaces() -> None:
    malformed = _rows()
    malformed[0]["full_token_count"] += 1
    with pytest.raises(
        RecurrentSFTFalsificationError,
        match="control_row_identity_invalid",
    ):
        transform_control_rows(malformed, arm="sham_labels", seed=1)
    with pytest.raises(
        RecurrentSFTFalsificationError,
        match="syntax_token_surface_missing",
    ):
        transform_control_rows(
            _rows(),
            arm="syntax_only",
            seed=1,
            token_surfaces={99: "<eos>"},
            neutral_token_id=7,
        )


def test_exact_sign_test_excludes_ties_and_is_two_sided() -> None:
    result = exact_two_sided_sign_test([-1.0] * 8 + [0.0, 0.0])
    assert result == {
        "non_ties": 8,
        "improved": 8,
        "regressed": 0,
        "ties": 2,
        "two_sided_p_value": 0.0078125,
    }


def test_paired_comparison_records_transfer_and_error_transitions() -> None:
    base = _observations(loss_shift=0.0, improve_tokens=False)
    trained = _observations(loss_shift=-0.2, improve_tokens=True)
    comparison = compare_observation_arms(
        base,
        trained,
        reference_arm=BASE_ARM,
        candidate_arm=TRAINED_ARM,
    )
    assert comparison["passed"] is True
    assert comparison["overall"]["mean_loss_delta"] == -0.2
    assert comparison["overall"]["wrong_to_right"] == 16
    assert comparison["overall"]["right_to_wrong"] == 0
    assert comparison["overall"]["generated_wrong_to_right"] == 8
    assert comparison["negative_transfer_families"] == []


def test_paired_comparison_rejects_family_negative_transfer() -> None:
    base = _observations(loss_shift=0.0, improve_tokens=False)
    trained = _observations(loss_shift=-0.2, improve_tokens=True)
    for row in trained:
        if row["family"] == "tool":
            row["loss"] += 0.5
    comparison = compare_observation_arms(
        base,
        trained,
        reference_arm=BASE_ARM,
        candidate_arm=TRAINED_ARM,
    )
    assert comparison["passed"] is False
    assert comparison["negative_transfer_families"] == ["tool"]


def test_verdict_requires_trained_arm_to_beat_every_control() -> None:
    observations = {
        BASE_ARM: _observations(loss_shift=0.0, improve_tokens=False),
        TRAINED_ARM: _observations(loss_shift=-0.3, improve_tokens=True),
        "sham_labels": _observations(loss_shift=-0.05, improve_tokens=False),
        "shuffled_traces": _observations(loss_shift=-0.1, improve_tokens=False),
        "syntax_only": _observations(loss_shift=-0.15, improve_tokens=False),
    }
    assert set(observations) == set(ALL_ARMS)
    verdict = build_falsification_verdict(observations)
    assert verdict["heldout_transfer_proven"] is True
    assert verdict["reasoning_gain_proven"] is False
    assert verdict["frontier_performance_proven"] is False
    assert verdict["wow_signal"] is False

    observations["syntax_only"] = _observations(
        loss_shift=-0.5,
        improve_tokens=True,
    )
    failed = build_falsification_verdict(observations)
    assert failed["heldout_transfer_proven"] is False
    assert (
        failed["trained_vs_controls"]["syntax_only"]["overall"]["mean_loss_delta"]
        > 0.0
    )


def test_observation_alignment_is_fail_closed() -> None:
    base = _observations(loss_shift=0.0, improve_tokens=False)
    trained = _observations(loss_shift=-0.2, improve_tokens=True)
    trained[0]["example_id"] = "f" * 64
    with pytest.raises(
        RecurrentSFTFalsificationError,
        match="comparison_identity_mismatch",
    ):
        compare_observation_arms(
            base,
            trained,
            reference_arm=BASE_ARM,
            candidate_arm=TRAINED_ARM,
        )
