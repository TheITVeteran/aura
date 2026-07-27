from __future__ import annotations

import copy

import pytest

from core.learning.recurrent_sft_sampling import (
    FAMILY_BALANCED_SAMPLER,
    RecurrentSFTSamplingError,
    family_balance_receipt,
    family_balanced_epoch_order,
    sample_history,
    validate_family_balanced_order,
)


def _rows() -> list[dict]:
    families = ("structured", "structured", "code", "code", "code", "code", "retention")
    return [
        {
            "example_id": f"{index + 1:064x}",
            "family": family,
        }
        for index, family in enumerate(families)
    ]


def test_epoch_is_exactly_family_balanced_and_covers_every_row() -> None:
    rows = _rows()
    order = family_balanced_epoch_order(rows, seed=17, epoch=0)
    receipt = family_balance_receipt(rows, order)
    assert len(order) == 12
    assert receipt["sampler"] == FAMILY_BALANCED_SAMPLER
    assert receipt["family_exposures"] == {
        "code": 4,
        "retention": 4,
        "structured": 4,
    }
    assert receipt["exact_family_balance"] is True
    assert receipt["all_rows_covered"] is True
    assert receipt["missing_row_indices"] == []
    assert validate_family_balanced_order(
        rows,
        order,
        seed=17,
        epoch=0,
    ) == order


def test_order_is_reproducible_but_changes_between_epochs() -> None:
    rows = _rows()
    first = family_balanced_epoch_order(rows, seed=29, epoch=0)
    assert first == family_balanced_epoch_order(rows, seed=29, epoch=0)
    assert first != family_balanced_epoch_order(rows, seed=29, epoch=1)


def test_round_robin_prevents_family_streaks() -> None:
    rows = _rows()
    order = family_balanced_epoch_order(rows, seed=37, epoch=2)
    families = [rows[index]["family"] for index in order]
    assert all(
        current != following
        for current, following in zip(
            families,
            families[1:],
            strict=False,
        )
    )


def test_history_reconstructs_complete_and_partial_epochs() -> None:
    rows = _rows()
    history = sample_history(rows, seed=41, steps=25)
    epoch_zero = family_balanced_epoch_order(rows, seed=41, epoch=0)
    epoch_one = family_balanced_epoch_order(rows, seed=41, epoch=1)
    epoch_two = family_balanced_epoch_order(rows, seed=41, epoch=2)
    assert history["indices"] == epoch_zero + epoch_one + epoch_two[:1]
    assert history["epoch_boundaries"] == [
        {
            "epoch": 0,
            "history_start": 0,
            "history_end": 12,
            "epoch_order_length": 12,
        },
        {
            "epoch": 1,
            "history_start": 12,
            "history_end": 24,
            "epoch_order_length": 12,
        },
        {
            "epoch": 2,
            "history_start": 24,
            "history_end": 25,
            "epoch_order_length": 12,
        },
    ]


def test_replay_rejects_order_row_and_identity_tampering() -> None:
    rows = _rows()
    order = family_balanced_epoch_order(rows, seed=53, epoch=0)
    tampered = list(order)
    tampered[0], tampered[1] = tampered[1], tampered[0]
    with pytest.raises(
        RecurrentSFTSamplingError,
        match="order_drift",
    ):
        validate_family_balanced_order(
            rows,
            tampered,
            seed=53,
            epoch=0,
        )

    duplicate = copy.deepcopy(rows)
    duplicate[1]["example_id"] = duplicate[0]["example_id"]
    with pytest.raises(
        RecurrentSFTSamplingError,
        match="row_identity_invalid",
    ):
        family_balanced_epoch_order(duplicate, seed=53, epoch=0)


def test_zero_step_history_is_explicit_and_empty() -> None:
    assert sample_history(_rows(), seed=1, steps=0) == {
        "sampler": FAMILY_BALANCED_SAMPLER,
        "steps": 0,
        "indices": [],
        "epoch_boundaries": [],
    }


@pytest.mark.parametrize(
    ("operation", "kwargs"),
    [
        (family_balanced_epoch_order, {"seed": True, "epoch": 0}),
        (validate_family_balanced_order, {"seed": 1, "epoch": -1}),
        (sample_history, {"seed": "1", "steps": 0}),
    ],
)
def test_every_replay_surface_validates_seed_and_epoch(
    operation: object,
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(
        RecurrentSFTSamplingError,
        match="sampling_seed_invalid",
    ):
        if operation is validate_family_balanced_order:
            operation(_rows(), [0], **kwargs)
        else:
            operation(_rows(), **kwargs)


def test_epoch_size_is_rejected_before_unbounded_schedule_allocation() -> None:
    rows = [
        {
            "example_id": f"{index + 1:064x}",
            "family": "large" if index < 2_000 else f"singleton-{index}",
        }
        for index in range(2_501)
    ]
    with pytest.raises(
        RecurrentSFTSamplingError,
        match="epoch_too_large",
    ):
        family_balanced_epoch_order(rows, seed=1, epoch=0)
