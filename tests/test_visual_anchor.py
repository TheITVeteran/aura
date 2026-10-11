"""Remembered labels are hypotheses bound to current visual measurements."""
import numpy as np
import pytest

from core.perception.observed_transfer import ScreenReading, make_snapshot
from core.perception.visual_anchor import _remembered_labels_invariant, reacquire, remember_label


def screen(epoch=1, *, offset=(0, 0), visible=True, faded=False, duplicate=False, flat=False):
    pixels = np.full((160, 240, 3), 30, dtype=np.uint8)
    patch = np.random.default_rng(18).integers(30, 210, (20, 24, 3), dtype=np.uint8)
    if flat:
        patch[:] = 90
    if faded:
        patch = (patch.astype(float) * .5 + 90).astype(np.uint8)
    dx, dy = offset
    if visible:
        pixels[60 + dy:80 + dy, 100 + dx:124 + dx] = patch
    pixels[:30, :40] = epoch * 20  # Animation elsewhere must not revoke a still local control.
    regions = [{"text": "Continue", "x": 100 / 240, "y": 60 / 160, "width": .1, "height": .125}] if epoch == 1 else []
    if duplicate:
        pixels[60:80, 180:204] = patch
        regions.append({"text": "", "x": .75, "y": .375, "width": .1, "height": .125, "shape": True})
    return ScreenReading({"layout": regions, "bounds": [0, 0, 240, 160], "surface_id": "editor",
                          "capture_at": float(epoch), "_capture_epoch": str(epoch), "settled": True},
                         picture_snapshot=make_snapshot(pixels))


def test_registered_invariant_is_measured():
    assert _remembered_labels_invariant() == ()


def test_registered_claim_runs_its_measurement_and_preserves_the_live_proof_boundary():
    from types import SimpleNamespace

    from core.organism.model_validation import Outcome, get_suite, install_runtime_validation

    install_runtime_validation()
    suite = get_suite()
    test = next(test for test in suite.tests() if test.name == "remembered_labels_require_current_pixels")
    result = test.run(SimpleNamespace(name="probe", capabilities=lambda: set()))
    assert result.score.outcome is Outcome.PASS
    claim = next(claim for claim in suite.claims() if claim.test == test.name)
    assert "live demo performance remains independently unproved" in claim.evidence_note


@pytest.mark.parametrize("offset,faded", [((0, 0), False), ((2, 1), False), ((0, 0), True), ((2, -1), True)])
def test_reacquisition_finds_current_extent_through_small_motion_and_fading(offset, faded):
    before = screen()
    anchor = remember_label("Continue", before, episode="request")
    assert anchor is not None
    before.picture_snapshot = None  # Immutable anchor cannot depend on a mutable observation.
    box = reacquire(anchor, screen(2, offset=offset, faded=faded), episode="request")
    assert box is not None
    assert box[:2] == pytest.approx(((100 + offset[0]) / 240, (60 + offset[1]) / 160))


@pytest.mark.parametrize("damage", ["absent", "flat", "duplicate", "surface", "viewport", "episode", "work_epoch",
                                   "old", "same_epoch", "expired", "occluded", "unsettled", "pixels", "changed_label"])
def test_unmeasured_or_ambiguous_current_control_stays_unknown(damage):
    anchor = remember_label("Continue", screen(), episode="request")
    now = screen(2, visible=damage != "absent", duplicate=damage == "duplicate", flat=damage == "flat")
    if damage == "surface":
        now["surface_id"] = "other app"
    elif damage == "viewport":
        now["bounds"] = [0, 0, 480, 320]
    elif damage == "work_epoch":
        now["_work_epoch"] = "reset"
    elif damage == "old":
        now["capture_at"] = .5
    elif damage == "same_epoch":
        now["_capture_epoch"] = "1"
    elif damage == "expired":
        now["capture_at"] = 122.
    elif damage == "occluded":
        now["occluded"] = True
    elif damage == "unsettled":
        now["settled"] = False
    elif damage == "pixels":
        now.picture_snapshot = None
    elif damage == "changed_label":
        now["layout"] = [{**screen()["layout"][0], "text": "Delete"}]
    assert reacquire(anchor, now, episode="other" if damage == "episode" else "request") is None


@pytest.mark.parametrize("damage", ["no_extent", "flat", "duplicate_label", "missing_context"])
def test_initial_memory_requires_a_unique_distinct_measured_region(damage):
    before = screen(flat=damage == "flat")
    if damage == "no_extent":
        before["layout"] = [{"text": "Continue", "center_x": .5, "center_y": .5}]
    elif damage == "duplicate_label":
        before["layout"].append({**before["layout"][0], "x": .75})
    elif damage == "missing_context":
        before["surface_id"] = ""
    assert remember_label("Continue", before) is None
