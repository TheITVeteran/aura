"""Obeying the file-delete rule produced the harm the rule prevents.

``file_delete`` is off by default and listed in ``_UNGRANTABLE_MODALITIES``:
Aura may not delete a person's files, and a session grant is the wrong
instrument for revisiting that. Correct.

The rule is keyed on the word "delete" appearing in an action summary, so it
also refused her removing the ephemeral screenshot she had taken herself two
seconds earlier in order to OCR it. LIVE, 2026-08-10, once capture started
working: ``~/.aura/data/ephemeral`` grew by a full-screen capture roughly
every seven seconds — 51MB in twenty minutes — and nothing could ever remove
one. Retention could not prune them either, for the same reason, which is why
``data/screenshots`` sat at 236 files against a 200-file limit.

An ephemeral capture that is never deleted is a permanent, growing record of
everything on the person's screen. That is precisely what the rule exists to
prevent, arrived at by obeying it.

The exemption is keyed on FACTS, never on a caller's claim about itself:
``local_internal_decision`` says outright that it "is not a substitute for
consequential action authorization", and a permission gate that can be talked
out of its decision is not a gate. Three independent conditions must all hold
— resolved containment inside Aura's own capture directories, an image
suffix, and a real extractable path.
"""
from __future__ import annotations

import json

import pytest

from core.capabilities.permission_model import (
    _UNGRANTABLE_MODALITIES,
    ModalityPermissions,
    PermissionRiskModel,
    _is_own_capture_artifact,
)
from core.runtime.state_ownership import state_root


def _summary(path: str, op: str = "delete") -> str:
    """The action summary shape the Will actually receives."""
    return "host_automation.ephemeral_ocr_cleanup params=" + json.dumps(
        {"op": op, "path": path}, sort_keys=True
    )


def _root() -> str:
    return str(state_root())


@pytest.mark.parametrize(
    "relative",
    ["data/ephemeral/screenshot_1.png", "data/screenshots/screen_1.png"],
)
def test_her_own_captures_are_hers_to_clean_up(relative):
    assert _is_own_capture_artifact(_summary(f"{_root()}/{relative}"))


@pytest.mark.parametrize(
    "path",
    [
        "/Users/bryan/Documents/taxes.pdf",
        "/Users/bryan/Pictures/wedding.png",
        "/Users/bryan/Desktop/Monthly Expenses.xlsx",
        "/etc/passwd",
    ],
)
def test_a_persons_own_files_are_never_covered(path):
    """The property the whole rule exists for. A photo is the sharp case:
    same suffix as a capture, and still not hers to delete."""
    assert not _is_own_capture_artifact(_summary(path))


def test_traversal_cannot_walk_out_of_the_capture_directory():
    """Containment is checked after resolution, not by prefix matching."""
    escape = f"{_root()}/data/ephemeral/../../../../etc/passwd"

    assert not _is_own_capture_artifact(_summary(escape))


def test_a_non_image_in_the_capture_directory_is_not_covered():
    """Otherwise the rule widens by dropping a file into the right folder."""
    assert not _is_own_capture_artifact(_summary(f"{_root()}/data/ephemeral/notes.txt"))


def test_her_own_non_capture_data_is_not_covered():
    """"Under her state root" is not the test — being a CAPTURE is."""
    assert not _is_own_capture_artifact(_summary(f"{_root()}/memory/episodes.db"))


def test_prose_mentioning_deletion_is_not_a_path():
    """The summary is not always structured; a sentence is not a target."""
    assert not _is_own_capture_artifact("please delete the ephemeral screenshots")
    assert not _is_own_capture_artifact("")


def test_the_modality_changes_only_for_her_own_captures():
    """The end-to-end behaviour, at the gate that actually decides."""
    model = PermissionRiskModel()

    own = model._detect_modality("file_write", _summary(f"{_root()}/data/ephemeral/s.png"))
    theirs = model._detect_modality("file_write", _summary("/Users/bryan/Documents/a.pdf"))

    assert own == "file_write"
    assert theirs == "file_delete"


def test_cleaning_her_own_capture_is_approved_and_a_user_file_is_not():
    model = PermissionRiskModel()

    mine = model.check_permission(
        "file_write", _summary(f"{_root()}/data/ephemeral/s.png")
    )
    yours = model.check_permission("file_write", _summary("/Users/bryan/Documents/a.pdf"))

    assert mine.modality == "file_write"
    assert yours.approved is False
    assert "file_delete" in yours.reason


def test_the_rule_itself_is_unchanged():
    """What this must NOT have done."""
    assert ModalityPermissions().file_delete is False
    assert "file_delete" in _UNGRANTABLE_MODALITIES
