"""Artifact integrity: nothing becomes running code unverified.

Clean-room adoption of TUF/SLSA's verify-before-install principle — without a
key hierarchy, because Aura has no self-updater and a full TUF client would be
speculative machinery for a path that does not exist. What does exist is the
moment a file becomes executable Python, and this is the shared gate for it.
"""
from __future__ import annotations

import hashlib

import pytest

from core.runtime.artifact_integrity import (
    IntegrityFailure,
    IntegrityLevel,
    verify_artifact,
)

pytestmark = pytest.mark.unit


def _artifact(root, name="kernel.py", *, body="x = 1\n", manifest=True, digest=None):
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(body)
    if manifest:
        (root / f"{name}.sha256").write_text(
            digest or hashlib.sha256(path.read_bytes()).hexdigest()
        )
    return path


# ── containment ────────────────────────────────────────────────────────────


def test_a_verified_artifact_passes(tmp_path):
    root = tmp_path / "backups"
    verdict = verify_artifact(_artifact(root), within=root)

    assert verdict.ok
    assert verdict.level is IntegrityLevel.DIGEST
    assert verdict.digest


def test_a_sibling_directory_cannot_masquerade_as_contained(tmp_path):
    """str(p).startswith(str(base)) accepts any sibling whose name merely
    begins with the base — 'backups_evil' passes a 'backups' prefix test."""
    root = tmp_path / "backups"
    root.mkdir()
    evil = _artifact(tmp_path / "backups_evil")

    verdict = verify_artifact(evil, within=root)

    assert not verdict.ok
    assert verdict.failure is IntegrityFailure.OUTSIDE_ROOT


def test_traversal_is_refused(tmp_path):
    root = tmp_path / "backups"
    root.mkdir()
    outside = _artifact(tmp_path / "elsewhere")

    assert verify_artifact(f"{root}/../elsewhere/kernel.py", within=root).failure \
        is IntegrityFailure.OUTSIDE_ROOT
    assert not verify_artifact(outside, within=root).ok


def test_a_missing_artifact_is_unresolvable_not_a_crash(tmp_path):
    root = tmp_path / "backups"
    root.mkdir()

    verdict = verify_artifact(root / "nope.py", within=root)

    assert verdict.failure is IntegrityFailure.UNRESOLVABLE


def test_a_directory_is_not_a_regular_file(tmp_path):
    root = tmp_path / "backups"
    (root / "subdir").mkdir(parents=True)

    verdict = verify_artifact(root / "subdir", within=root)

    assert verdict.failure is IntegrityFailure.NOT_REGULAR_FILE


# ── digest: absence is refusal, not a warning ──────────────────────────────


def test_unsigned_content_is_refused(tmp_path):
    """Anything that landed in the directory would otherwise become running
    code just because nobody supplied a signature."""
    root = tmp_path / "backups"
    verdict = verify_artifact(_artifact(root, manifest=False), within=root)

    assert verdict.failure is IntegrityFailure.MANIFEST_MISSING


def test_a_tampered_artifact_is_refused(tmp_path):
    root = tmp_path / "backups"
    path = _artifact(root)
    path.write_text("x = 2  # tampered after the manifest was written\n")

    verdict = verify_artifact(path, within=root)

    assert verdict.failure is IntegrityFailure.DIGEST_MISMATCH


def test_an_empty_manifest_is_refused(tmp_path):
    """An empty manifest is not "no expectation" — it is an unusable one."""
    root = tmp_path / "backups"
    path = _artifact(root)
    (root / f"{path.name}.sha256").write_text("   \n")

    assert verify_artifact(path, within=root).failure \
        is IntegrityFailure.MANIFEST_UNREADABLE


# ── it has to parse, or promotion bricks the target ────────────────────────


def test_unparseable_python_is_refused(tmp_path):
    """Restoring a syntactically broken file leaves the target unimportable —
    bricking the very thing the promotion was meant to rescue."""
    root = tmp_path / "backups"
    verdict = verify_artifact(_artifact(root, body="def (((:\n"), within=root)

    assert verdict.failure is IntegrityFailure.NOT_VALID_PYTHON
    assert "unimportable" in verdict.detail


def test_non_python_artifacts_can_skip_the_parse_check(tmp_path):
    root = tmp_path / "backups"
    blob = _artifact(root, name="weights.bin", body="\x00\x01not python")

    assert verify_artifact(blob, within=root, require_python=False).ok


# ── the verdict must not overclaim ─────────────────────────────────────────


def test_a_passing_verdict_says_digest_not_signed():
    """A sha256 manifest proves the bytes are the bytes that were recorded, NOT
    that a trusted party recorded them. Claiming otherwise would be exactly the
    overclaim this module exists to prevent."""
    assert IntegrityLevel.DIGEST.value == "digest"
    assert IntegrityLevel.SIGNED.value == "signed"


def test_verification_never_raises(tmp_path):
    """A gate that can itself explode is not a gate."""
    for bad in ("", "/dev/null/nope", tmp_path / "missing", "\x00invalid"):
        verdict = verify_artifact(bad, within=tmp_path)
        assert verdict.ok is False
        assert verdict.failure is not None


# ── consolidation: one gate, used by the live path ─────────────────────────


def test_the_immune_system_rollback_uses_the_shared_gate():
    """This logic was written in immune_system first; it now lives in one place
    so every promotion path gets the same checks."""
    import inspect

    import core.adaptation.immune_system as ims

    source = inspect.getsource(ims.ImmuneSystem.initiate_rollback)
    assert "verify_artifact" in source
    assert not hasattr(ims.ImmuneSystem, "_snapshot_integrity_ok"), \
        "the duplicated implementation should be gone"
