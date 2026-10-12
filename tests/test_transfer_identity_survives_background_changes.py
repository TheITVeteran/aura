"""Object identity crosses a background change without accepting a different occupant."""
import base64
from dataclasses import replace

import numpy as np
import pytest

from core.perception.observed_transfer import (
    ScreenReading,
    TransferReceipt,
    bind_transfer,
    finish_delivery,
    make_snapshot,
    revalidate_placement,
    verify_transfer,
)
from core.runtime.skill_contract import PredicateState

pytestmark = pytest.mark.unit
SOURCE, TARGET = (20, 30), (200, 130)


def picture(background, *locations, wrong=False):
    result = np.full((240, 320, 3), background, np.uint8)
    rows, cols = np.indices((32, 32))
    opaque = (rows >= 6) & (rows < 26) & (cols >= 6) & (cols < 26)
    colour = np.stack(((rows * 9 + cols * 3) % 251, (rows * 3 + cols * 11) % 241,
                       ((cols // 8 + rows // 8) % 2) * 160 + 30), axis=2).astype(np.uint8)
    if wrong:
        colour = 255 - colour
    for x, y in locations:
        region = result[y:y + 32, x:x + 32]
        region[opaque] = colour[opaque]
    return result


def reading(image, at, *, source=False, surface="canvas"):
    regions = [{"text": "object", "x": 20 / 320, "y": 30 / 240, "width": .1, "height": 32 / 240, "shape": True}] if source else []
    return ScreenReading({"capture_at": at, "_capture_epoch": f"frame-{at}", "owner": surface,
                          "bounds": [0, 0, 320, 240], "settled": True, "shapes": regions},
                         picture_snapshot=make_snapshot(image))


def transfer(*, copied=False, wrong=False, surface="canvas"):
    before = reading(picture(245, SOURCE), 1, source=True, surface=surface)
    bound = bind_transfer('drag "object" to "workspace"', before, source="object", destination="workspace",
                          source_at=(36 / 320, 46 / 240), destination_at=(216 / 320, 146 / 240))
    bound = finish_delivery(bound, started=1.1, completed=1.2, delivered=True)
    positions = (TARGET, SOURCE) if copied else (TARGET,)
    after = reading(picture(70, *positions, wrong=wrong), 2, surface=surface)
    return bound, after


@pytest.mark.parametrize("surface", ["diagram", "image editor", "inventory"])
@pytest.mark.parametrize("copied", [False, True])
def test_delivered_object_retains_identity_when_its_surrounding_pixels_change(surface, copied):
    bound, after = transfer(copied=copied, surface=surface)
    receipt = verify_transfer(bound, after)
    assert receipt.state is PredicateState.SATISFIED, receipt.reason
    assert receipt.effect == ("copied" if copied else "moved")
    assert revalidate_placement(receipt, reading(picture(180, TARGET), 3, surface=surface)) is PredicateState.SATISFIED


def test_different_foreground_does_not_become_the_source_when_background_is_ignored():
    bound, after = transfer(wrong=True)
    assert verify_transfer(bound, after).state is not PredicateState.SATISFIED


def test_only_changing_the_background_does_not_create_an_occupant():
    bound, _after = transfer()
    assert verify_transfer(bound, reading(picture(70), 2)).state is not PredicateState.SATISFIED


def test_two_matching_foregrounds_near_the_endpoint_remain_ambiguous():
    bound, _after = transfer()
    after = reading(picture(70, (184, 130), (216, 130)), 2)
    assert verify_transfer(bound, after).state is PredicateState.UNKNOWN


def test_existing_foreground_cannot_be_credited_as_new_work():
    before = reading(picture(245, SOURCE, TARGET), 1, source=True)
    bound = bind_transfer('drag "object" to "workspace"', before, source="object", destination="workspace",
                          source_at=(36 / 320, 46 / 240), destination_at=(216 / 320, 146 / 240))
    bound = finish_delivery(bound, started=1.1, completed=1.2, delivered=True)
    assert verify_transfer(bound, reading(picture(70, TARGET), 2)).state is PredicateState.UNSATISFIED


def test_owned_mask_survives_bounded_storage_and_independent_reobservation():
    bound, after = transfer()
    receipt = verify_transfer(bound, after)
    assert len(receipt.appearance_mask) == 256
    restored = TransferReceipt.from_memory(receipt.as_memory())
    assert restored == receipt
    assert revalidate_placement(restored, reading(picture(180, TARGET), 3)) is PredicateState.SATISFIED
    assert revalidate_placement(restored, reading(picture(180, TARGET, wrong=True), 3)) is not PredicateState.SATISFIED


@pytest.mark.parametrize("mask", [bytes(256), bytes([1] * 256), bytes([2] * 256), bytes([1] * 25), bytes(257)])
def test_untrusted_or_tiny_masks_cannot_restore_success(mask):
    bound, after = transfer()
    receipt = verify_transfer(bound, after)
    memory = {**receipt.as_memory(), "appearance_mask": base64.b64encode(mask).decode("ascii")}
    assert TransferReceipt.from_memory(memory) is None
    assert revalidate_placement(replace(receipt, appearance_mask=mask), reading(picture(180, TARGET), 3)) is PredicateState.UNKNOWN


def test_mask_requires_a_flat_measured_exterior_and_keeps_enclosed_highlights():
    from core.perception.foreground_appearance import _foreground_mask_invariant, foreground_mask

    assert _foreground_mask_invariant() == ()
    boundary_noise = np.indices((16, 16)).sum(axis=0) % 2 * 255
    patch = np.repeat(boundary_noise[:, :, None], 3, axis=2).astype(np.uint8).tobytes()
    assert foreground_mask(patch, 16) == b""
    for edge in (0, -1, 1000000, True):
        assert foreground_mask(b"", edge) == b""
