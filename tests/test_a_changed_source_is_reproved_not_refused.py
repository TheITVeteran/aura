"""Editing a skill file must not turn a person's request into a crash.

Live 2026-07-28: a skill file was edited while an instance was running, and
the next ordinary request — "open Notes and write me a note" — came back as

    CRITICAL SERVICE FAILURE: Subsystem 'capability_engine' failed with
    failure policy 'fail-closed'. Original error: RuntimeError: catalog
    source changed after validation; reload is required

The message names its own remedy and then nobody performs it. Nothing was
being protected: a changed digest is not evidence of tampering, it is
evidence of an edit.

What the digest stands for is that a skill's identity and authority were
proved against that exact content. So the rule these tests hold is narrow and
strict — a changed source is re-proved, and a source that fails the proof is
still refused, permanently rather than once.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from core.capability_engine import CapabilityEngine, SkillMetadata


@pytest.fixture()
def skill_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from core import capability_engine as module

    class _Paths:
        base_dir = str(tmp_path)

    class _Config:
        paths = _Paths()

    monkeypatch.setattr(module, "config", _Config())
    path = tmp_path / "skills" / "demo.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("original\n")
    return path


def _meta(path: Path, digest: str) -> SkillMetadata:
    return SkillMetadata(
        name="demo",
        description="demo",
        source_path="skills/demo.py",
        source_sha256=digest,
    )


def _digest_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_unchanged_source_reports_nothing_to_reprove(skill_file: Path) -> None:
    meta = _meta(skill_file, _digest_of(skill_file))
    assert CapabilityEngine._verify_catalog_source(meta) == ""


def test_a_changed_source_returns_the_new_digest_rather_than_raising(
    skill_file: Path,
) -> None:
    meta = _meta(skill_file, _digest_of(skill_file))
    skill_file.write_text("edited by a developer\n")
    observed = CapabilityEngine._verify_catalog_source(meta)
    assert observed == _digest_of(skill_file)
    # And critically it did NOT adopt the new digest by itself — that only
    # happens once the import-time checks have passed.
    assert meta.source_sha256 != observed


def test_an_unreadable_source_still_refuses(skill_file: Path) -> None:
    """Deleted, unreadable, or replaced by a directory: nothing can be
    checked, so nothing may be trusted. This is the case the old behaviour
    was right about, and it keeps the old behaviour."""
    meta = _meta(skill_file, _digest_of(skill_file))
    skill_file.unlink()
    with pytest.raises(RuntimeError, match="no longer readable"):
        CapabilityEngine._verify_catalog_source(meta)


def test_a_source_outside_the_tree_still_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A catalog entry pointing out of the repo is an attack shape, not an
    edit, and must never be re-proved into acceptance."""
    from core import capability_engine as module

    class _Paths:
        base_dir = str(tmp_path / "root")

    class _Config:
        paths = _Paths()

    (tmp_path / "root").mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("payload\n")
    monkeypatch.setattr(module, "config", _Config())

    meta = SkillMetadata(
        name="demo",
        description="demo",
        source_path="../outside.py",
        source_sha256="0" * 64,
    )
    with pytest.raises(RuntimeError, match="no longer readable"):
        CapabilityEngine._verify_catalog_source(meta)


def test_a_skill_with_no_declared_digest_is_not_second_guessed(
    skill_file: Path,
) -> None:
    meta = SkillMetadata(name="demo", description="demo")
    assert CapabilityEngine._verify_catalog_source(meta) == ""


def test_the_digest_is_only_adopted_after_the_identity_checks_pass() -> None:
    """The ordering is the guarantee.

    If the loader adopted the new digest before re-proving identity, a source
    that fails the checks would be refused once and then sail through on the
    next attempt — the checks would protect nothing. Read the loader and
    confirm the adoption sits after the last raise.
    """
    import inspect

    body = inspect.getsource(CapabilityEngine.execute)
    adopt_at = body.find("meta.source_sha256 = revalidated_digest")
    assert adopt_at > 0, "the loader no longer adopts a re-proved digest"

    preceding = body[:adopt_at]
    for guarantee in (
        "implementation no longer satisfies canonical BaseSkill",
        "implementation name changed after catalog validation",
        "implementation effect classification changed after validation",
    ):
        assert guarantee in preceding, (
            f"{guarantee!r} must be proved BEFORE the new digest is trusted"
        )
