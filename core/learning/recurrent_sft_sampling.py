"""Deterministic family-balanced sampling for recurrent SFT."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Final, Never

FAMILY_BALANCED_SAMPLER: Final = (
    "sha256_stateless_family_balanced_epoch.v1"
)
_MAX_ROWS: Final = 100_000
_MAX_STEPS: Final = 1_000_000


class RecurrentSFTSamplingError(ValueError):
    """A recurrent-SFT sample schedule is malformed or unbalanced."""


def _fail(code: str) -> Never:
    raise RecurrentSFTSamplingError(
        str(code or "recurrent_sft_sampling_invalid")
    )


def _validated_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes, bytearray))
        or not 1 <= len(rows) <= _MAX_ROWS
    ):
        _fail("recurrent_sft_sampling_rows_invalid")
    material: list[dict[str, Any]] = []
    buckets: dict[str, list[int]] = defaultdict(list)
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            _fail("recurrent_sft_sampling_row_invalid")
        example_id = row.get("example_id")
        family = row.get("family")
        if (
            not isinstance(example_id, str)
            or len(example_id) != 64
            or any(character not in "0123456789abcdef" for character in example_id)
            or example_id in seen_ids
            or not isinstance(family, str)
            or not family.strip()
        ):
            _fail("recurrent_sft_sampling_row_identity_invalid")
        seen_ids.add(example_id)
        material.append(dict(row))
        buckets[family].append(index)
    return material, dict(buckets)


def _digest(seed: int, epoch: int, role: str, value: str) -> bytes:
    return hashlib.sha256(
        f"{seed}:{epoch}:{role}:{value}".encode()
    ).digest()


def _validate_seed_epoch(*, seed: int, epoch: int) -> None:
    if (
        type(seed) is not int
        or seed < 0
        or type(epoch) is not int
        or epoch < 0
    ):
        _fail("recurrent_sft_sampling_seed_invalid")


def family_balanced_epoch_order(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    epoch: int,
) -> list[int]:
    """Interleave equal family exposure while covering every row."""

    material, _buckets = _validated_rows(rows)
    _validate_seed_epoch(seed=seed, epoch=epoch)
    order = _family_balanced_epoch_order_unchecked(
        material,
        seed=seed,
        epoch=epoch,
    )
    receipt = family_balance_receipt(material, order)
    if (
        not receipt["exact_family_balance"]
        or not receipt["all_rows_covered"]
    ):
        _fail("recurrent_sft_sampling_balance_invalid")
    return order


def family_balance_receipt(
    rows: Sequence[Mapping[str, Any]],
    order: Sequence[int],
) -> dict[str, Any]:
    """Summarize exact family and per-row exposure."""

    material, buckets = _validated_rows(rows)
    observed = list(order)
    if (
        not observed
        or len(observed) > _MAX_STEPS
        or any(
            type(index) is not int or not 0 <= index < len(material)
            for index in observed
        )
    ):
        _fail("recurrent_sft_sampling_order_invalid")
    family_counts = Counter(material[index]["family"] for index in observed)
    row_counts = Counter(observed)
    missing = sorted(set(range(len(material))) - set(row_counts))
    family_values = list(family_counts.values())
    body = {
        "sampler": FAMILY_BALANCED_SAMPLER,
        "row_count": len(material),
        "family_count": len(buckets),
        "epoch_update_count": len(observed),
        "family_exposures": dict(sorted(family_counts.items())),
        "family_exposure_min": min(family_values),
        "family_exposure_max": max(family_values),
        "row_exposure_min": min(row_counts.values()),
        "row_exposure_max": max(row_counts.values()),
        "missing_row_indices": missing,
        "all_rows_covered": not missing,
        "exact_family_balance": len(set(family_values)) == 1,
    }
    return {
        **body,
        "receipt_sha256": hashlib.sha256(
            json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest(),
    }


def validate_family_balanced_order(
    rows: Sequence[Mapping[str, Any]],
    order: Sequence[int],
    *,
    seed: int,
    epoch: int,
) -> list[int]:
    """Replay one exact family-balanced epoch."""

    material, _buckets = _validated_rows(rows)
    _validate_seed_epoch(seed=seed, epoch=epoch)
    observed = list(order)
    expected = _family_balanced_epoch_order_unchecked(
        material,
        seed=seed,
        epoch=epoch,
    )
    if observed != expected:
        _fail("recurrent_sft_sampling_order_drift")
    receipt = family_balance_receipt(material, observed)
    if (
        not receipt["exact_family_balance"]
        or not receipt["all_rows_covered"]
    ):
        _fail("recurrent_sft_sampling_balance_invalid")
    return observed


def sample_history(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    steps: int,
) -> dict[str, Any]:
    """Reconstruct the exact multi-epoch sample history."""

    material, _buckets = _validated_rows(rows)
    if type(steps) is not int or not 0 <= steps <= _MAX_STEPS:
        _fail("recurrent_sft_sampling_steps_invalid")
    _validate_seed_epoch(seed=seed, epoch=0)
    indices: list[int] = []
    epoch_boundaries: list[dict[str, int]] = []
    epoch = 0
    while len(indices) < steps:
        order = _family_balanced_epoch_order_unchecked(
            material,
            seed=seed,
            epoch=epoch,
        )
        take = min(len(order), steps - len(indices))
        start = len(indices)
        indices.extend(order[:take])
        epoch_boundaries.append(
            {
                "epoch": epoch,
                "history_start": start,
                "history_end": len(indices),
                "epoch_order_length": len(order),
            }
        )
        epoch += 1
    return {
        "sampler": FAMILY_BALANCED_SAMPLER,
        "steps": steps,
        "indices": indices,
        "epoch_boundaries": epoch_boundaries,
    }


def _family_balanced_epoch_order_unchecked(
    material: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    epoch: int,
) -> list[int]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(material):
        buckets[str(row["family"])].append(index)
    families = sorted(
        buckets,
        key=lambda family: (
            _digest(seed, epoch, "family", family),
            family,
        ),
    )
    shuffled = {
        family: sorted(
            indices,
            key=lambda index: (
                _digest(
                    seed,
                    epoch,
                    f"row:{family}",
                    str(material[index]["example_id"]),
                ),
                index,
            ),
        )
        for family, indices in buckets.items()
    }
    quota = max(len(indices) for indices in shuffled.values())
    if quota * len(families) > _MAX_STEPS:
        _fail("recurrent_sft_sampling_epoch_too_large")
    return [
        shuffled[family][round_index % len(shuffled[family])]
        for round_index in range(quota)
        for family in (
            families[round_index % len(families) :]
            + families[: round_index % len(families)]
        )
    ]


__all__ = [
    "FAMILY_BALANCED_SAMPLER",
    "RecurrentSFTSamplingError",
    "family_balance_receipt",
    "family_balanced_epoch_order",
    "sample_history",
    "validate_family_balanced_order",
]
