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
    """The ordering is the guarantee, checked against the source that has it.

    If the loader adopted the new digest before re-proving identity, a tampered
    source would be refused once and then sail through on the next attempt —
    the second attempt would compare against the digest the first one just
    wrote, and the checks would protect nothing.

    This reads `_prepare_skill_instance`, which is where the loader lives. The
    previous version read `CapabilityEngine.execute` and matched three error
    messages verbatim; the loader moved and all three messages were reworded,
    so it failed with "the loader no longer adopts a re-proved digest" while
    the ordering it names was entirely intact. Structure is checked here;
    behaviour is checked in the test below, which is the one that would notice
    if the ordering actually inverted.
    """
    import inspect

    body = inspect.getsource(CapabilityEngine._prepare_skill_instance)
    adopt_at = body.find("metadata.source_sha256 = revalidated_digest")
    assert adopt_at > 0, (
        "the loader no longer adopts a re-proved digest in _prepare_skill_instance "
        "— find where adoption moved to and point this test at it"
    )

    preceding = body[:adopt_at]
    # Stages, not prose. The loader names each phase it is in, and every
    # identity phase must be entered before the digest is trusted. Renaming an
    # error message cannot break this; removing a check still does.
    for stage in ("catalog_identity", "source_identity", "implementation_contract"):
        assert f'stage = "{stage}"' in preceding, (
            f"the {stage!r} stage must run BEFORE the new digest is trusted"
        )
    # And the adoption must be inside the publication stage, which is the one
    # that re-checks the catalog generation before writing anything.
    assert 'stage = "publication"' in preceding


def test_a_failed_contract_check_leaves_the_recorded_digest_untouched() -> None:
    """Behavioural twin of the ordering test.

    A skill whose implementation does not satisfy the contract must come back
    with its recorded digest unchanged. If adoption ran first, the refusal
    would still have written the new digest, and the next attempt would accept
    the tampered source because it now matches what was recorded.
    """
    from types import SimpleNamespace

    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )

    original_digest = "sha256:the-digest-recorded-when-this-was-trusted"
    meta = SkillMetadata(
        name="demo",
        description="demo",
        skill_class=type("NotASkill", (), {"name": "demo"}),
        validation_state="valid",
    )
    meta.source_sha256 = original_digest

    engine._skills = {"demo": meta}
    engine._instances = {}
    engine._catalog_digest = "catalog-1"
    engine._skill_preflight_results = {}
    engine.skill_last_errors = {}

    # The source re-proves to a DIFFERENT digest: exactly the case where
    # adopting early would be fatal.
    engine._verify_catalog_source = lambda _meta: "sha256:the-tampered-source"

    receipt, instance = engine._prepare_skill_instance("demo", meta)

    assert receipt["ok"] is False
    assert instance is None
    assert meta.source_sha256 == original_digest, (
        "the loader recorded the re-proved digest even though the identity "
        "checks refused the implementation — a second attempt would now accept "
        "the tampered source"
    )
