"""The retention gate: nothing becomes a capability without having run.

These tests execute real code under the real kernel boundary. That is the point
— a verification gate proved with a mocked sandbox proves nothing about the gate
that will actually run, and the defect being closed here was precisely a chain of
checks that had never been exercised end to end.
"""

from __future__ import annotations

import pytest

from core.sandbox.untrusted_python import available_boundary
from core.skill_management.forged_artifact import (
    ADAPTER_TEMPLATE,
    VERIFIED_REGION_MARKER,
    ArtifactError,
    LedgerEntry,
    assemble,
    class_name_for,
    digest_of,
    load_verified_region,
    next_version_path,
    split,
)
from core.skill_management.skill_verification import (
    ALLOWED_IMPORTS,
    ENTRYPOINT,
    ContractError,
    Probe,
    SkillDraft,
    render_skill_module,
    screen_source,
    verify_draft,
)

needs_boundary = pytest.mark.skipif(
    not available_boundary(),
    reason="no kernel sandbox on this host; the gate refuses rather than running unconfined",
)


def _module(body: str, imports=()) -> str:
    return render_skill_module(
        name="probe_skill", description="A skill under test.", body=body, imports=imports
    )


def _draft(body: str, probes, *, deterministic: bool = True, imports=()) -> SkillDraft:
    return SkillDraft(
        name="probe_skill",
        description="A skill under test.",
        source=_module(body, imports),
        probes=tuple(probes),
        deterministic=deterministic,
    )


# ── the contract screen ──────────────────────────────────────────────────


def test_source_without_the_entrypoint_is_refused():
    with pytest.raises(ContractError, match="defines no run"):
        screen_source("def something_else(params):\n    return {}\n")


def test_an_async_entrypoint_is_refused_with_the_reason():
    with pytest.raises(ContractError, match="cannot await a coroutine"):
        screen_source("async def run(params):\n    return {}\n")


def test_the_entrypoint_must_take_exactly_one_argument():
    with pytest.raises(ContractError, match="exactly one"):
        screen_source("def run(params, extra):\n    return {}\n")
    with pytest.raises(ContractError, match="exactly one"):
        screen_source("def run():\n    return {}\n")


def test_imports_outside_the_allowlist_are_refused():
    with pytest.raises(ContractError, match="outside the skill contract"):
        screen_source("import os\n\ndef run(params):\n    return {}\n")


def test_importing_core_is_refused():
    with pytest.raises(ContractError, match="outside the skill contract"):
        screen_source(
            "from core.skills.base_skill import BaseSkill\n\ndef run(params):\n    return {}\n"
        )


def test_the_subclasses_escape_is_refused_without_any_import():
    with pytest.raises(ContractError, match="__subclasses__"):
        screen_source(
            "def run(params):\n    return {'ok': ().__class__.__mro__[1].__subclasses__()}\n"
        )


def test_eval_is_refused():
    with pytest.raises(ContractError, match="reference to eval"):
        screen_source("def run(params):\n    return {'ok': eval(params['x'])}\n")


def test_the_allowlist_holds_no_module_that_reaches_the_host():
    assert not (ALLOWED_IMPORTS & {"os", "sys", "subprocess", "socket", "shutil", "pathlib"})


def test_an_empty_body_is_refused_at_render():
    with pytest.raises(ContractError, match="empty body"):
        render_skill_module(name="x", description="d", body="   \n  ")


# ── verification ─────────────────────────────────────────────────────────


def test_a_draft_with_no_probes_is_refused_rather_than_passed():
    report = verify_draft(_draft('return {"ok": True}', ()))
    assert not report.passed
    assert report.stage == "contract"
    assert "nothing executed" in report.reason


@needs_boundary
def test_a_working_skill_passes_and_says_what_that_means():
    report = verify_draft(
        _draft(
            'text = str(params.get("text") or "")\nreturn {"ok": True, "count": len(text.split())}',
            [Probe.of({"text": "a b c"}, expect={"ok": True, "count": 3})],
        )
    )
    assert report.passed
    assert report.boundary in {"seatbelt", "bubblewrap"}
    assert report.executed == 1
    assert report.precommitted == 1
    assert "not that it is correct" in report.summary


@needs_boundary
def test_code_that_raises_is_rejected_and_the_traceback_is_the_feedback():
    report = verify_draft(_draft('return {"ok": True, "n": undefined_name}', [Probe.of({})]))
    assert not report.passed
    assert report.stage == "execution"
    assert "NameError" in report.feedback()


@needs_boundary
def test_a_precommitted_expectation_the_code_violates_fails():
    report = verify_draft(
        _draft(
            'return {"ok": True, "count": len(str(params.get("text") or "").split()) + 1}',
            [Probe.of({"text": "a b c"}, expect={"ok": True, "count": 3})],
        )
    )
    assert not report.passed
    assert report.stage == "probe"
    assert "precommitted expectation not met" in report.feedback()


@needs_boundary
def test_a_result_missing_the_ok_key_fails_the_contract():
    report = verify_draft(_draft('return {"count": 1}', [Probe.of({})]))
    assert not report.passed
    assert "missing declared key" in report.outcomes[0].reason


@needs_boundary
def test_a_non_object_result_fails_the_contract():
    report = verify_draft(_draft("return 42", [Probe.of({})]))
    assert not report.passed
    assert "contract is a JSON object" in report.outcomes[0].reason


@needs_boundary
def test_declared_determinism_is_rechecked_by_rerunning():
    report = verify_draft(
        _draft(
            'return {"ok": True, "n": random.random()}',
            [Probe.of({})],
            imports=("random",),
        )
    )
    assert not report.passed
    assert "two identical calls disagreed" in report.outcomes[0].reason


@needs_boundary
def test_a_skill_that_declares_itself_nondeterministic_is_not_rerun():
    report = verify_draft(
        _draft(
            'return {"ok": True, "n": random.random()}',
            [Probe.of({}, expect_keys=("n",))],
            deterministic=False,
            imports=("random",),
        )
    )
    assert report.passed


@needs_boundary
def test_expect_keys_asserts_structure_without_asserting_a_value():
    report = verify_draft(
        _draft('return {"ok": True, "at": 12345}', [Probe.of({}, expect_keys=("at",))])
    )
    assert report.passed
    assert report.precommitted == 1


@needs_boundary
def test_expecting_none_differs_from_expecting_nothing():
    """``expect=None`` must be a real assertion, not an absent one."""
    silent = Probe.of({})
    explicit = Probe.of({}, expect=None)
    assert not silent.has_expectation
    assert explicit.has_expectation
    report = verify_draft(_draft('return {"ok": True}', [explicit]))
    assert not report.passed


def test_probe_of_rejects_unknown_keywords():
    with pytest.raises(TypeError, match="unexpected keyword"):
        Probe.of({}, expectt=1)


@needs_boundary
def test_verification_refuses_rather_than_running_unconfined():
    report = verify_draft(
        _draft('return {"ok": True}', [Probe.of({})]), require_boundary=True
    )
    assert report.passed and report.outcomes[0].sandboxed


# ── the artifact and its binding to the evidence ─────────────────────────


def test_the_verified_region_survives_assembly_byte_for_byte():
    pure = _module('return {"ok": True}')
    artifact = assemble(pure, skill_name="probe_skill", description="A skill.")
    assert split(artifact.text)[0] == pure
    assert artifact.digest == digest_of(pure)


def test_the_adapter_is_exactly_the_template_and_never_model_authored():
    pure = _module('return {"ok": True}')
    artifact = assemble(pure, skill_name="probe_skill", description="A skill.")
    assert artifact.adapter == ADAPTER_TEMPLATE.format(
        class_name=artifact.class_name,
        skill_name="probe_skill",
        description="A skill.",
        entrypoint=ENTRYPOINT,
    )


def test_the_assembled_file_is_valid_python():
    import ast

    pure = _module('return {"ok": True}')
    artifact = assemble(pure, skill_name="probe_skill", description="A skill.")
    ast.parse(artifact.text)


def test_a_region_containing_the_marker_is_refused():
    with pytest.raises(ArtifactError, match="contains the region marker"):
        assemble(f"# {VERIFIED_REGION_MARKER}\n", skill_name="x", description="d")


def test_a_file_with_two_markers_is_refused_rather_than_guessed():
    pure = _module('return {"ok": True}')
    artifact = assemble(pure, skill_name="probe_skill", description="A skill.")
    with pytest.raises(ArtifactError, match="2 verified-region markers"):
        split(artifact.text + "\n" + VERIFIED_REGION_MARKER + "\n")


def test_a_file_with_no_marker_is_refused():
    with pytest.raises(ArtifactError, match="no verified-region marker"):
        split("def run(params):\n    return {}\n")


def test_class_names_survive_leading_digits_and_punctuation():
    assert class_name_for("word count") == "WordCountSkill"
    assert class_name_for("already_skill") == "AlreadySkill"
    assert class_name_for("3d_render") == "Forged3dRenderSkill"
    with pytest.raises(ArtifactError):
        class_name_for("!!!")


def test_every_generated_class_name_is_a_valid_identifier():
    for raw in ("3d_render", "word count", "a-b-c", "9", "__x__", "Skill", "émoji_name"):
        try:
            name = class_name_for(raw)
        except ArtifactError:
            continue
        assert name.isidentifier(), f"{raw!r} produced {name!r}"


def test_an_edited_file_no_longer_matches_its_evidence(tmp_path):
    pure = _module('return {"ok": True, "n": 1}')
    artifact = assemble(pure, skill_name="probe_skill", description="A skill.")
    target = tmp_path / "probe_skill.py"
    target.write_text(artifact.text, encoding="utf-8")
    assert load_verified_region(target, expected_digest=artifact.digest) == pure

    target.write_text(artifact.text.replace('"n": 1', '"n": 2'), encoding="utf-8")
    with pytest.raises(ArtifactError, match="no longer matches its verification evidence"):
        load_verified_region(target, expected_digest=artifact.digest)


def test_the_archive_path_steps_past_versions_that_already_exist(tmp_path):
    target = tmp_path / "skill.py"
    target.write_text("x", encoding="utf-8")
    first = next_version_path(target)
    assert first.name == "skill.v2.py"
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_text("x", encoding="utf-8")
    assert next_version_path(target).name == "skill.v3.py"


def test_the_archive_hides_from_the_catalog_walk(tmp_path):
    """A dot-directory, so superseded versions are not rediscovered as skills."""
    assert next_version_path(tmp_path / "skill.py").parent.name.startswith(".")


# ── the deployed artifact is what the catalog accepts ────────────────────


def test_the_assembled_artifact_is_discovered_by_the_real_catalog(tmp_path):
    from core.skills.discovery import SkillSourceRoot, build_skill_catalog

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    pure = _module('return {"ok": True}')
    artifact = assemble(pure, skill_name="probe_skill", description="A skill.")
    (skills_dir / "probe_skill.py").write_text(artifact.text, encoding="utf-8")

    catalog = build_skill_catalog(
        roots=(SkillSourceRoot(skills_dir, "skills", "project"),), try_rust=False
    )
    assert [d.name for d in catalog.accepted] == ["probe_skill"]
    assert not catalog.blocking_issues


def test_a_skill_without_an_effect_scope_is_rejected_by_the_catalog(tmp_path):
    """The old template's defect, pinned so it cannot come back."""
    from core.skills.discovery import SkillSourceRoot, build_skill_catalog

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "scopeless.py").write_text(
        "from core.skills.base_skill import BaseSkill\n\n\n"
        "class ScopelessSkill(BaseSkill):\n"
        '    """No scope."""\n\n'
        '    name = "scopeless"\n'
        '    description = "No scope."\n\n'
        "    async def execute(self, params, context=None):\n"
        '        return {"ok": True}\n',
        encoding="utf-8",
    )
    catalog = build_skill_catalog(
        roots=(SkillSourceRoot(skills_dir, "skills", "project"),), try_rust=False
    )
    assert not catalog.accepted
    assert any(i.code == "unclassified_effect" for i in catalog.blocking_issues)


# ── the ledger ───────────────────────────────────────────────────────────


def test_reliability_is_absent_until_something_has_run():
    entry = LedgerEntry(
        skill_name="s",
        digest="d",
        path="p",
        verified_at=0.0,
        boundary="seatbelt",
        probes_executed=1,
        probes_precommitted=1,
        summary="",
    )
    assert entry.reliability is None
    from dataclasses import replace

    assert replace(entry, successes=3, failures=1).reliability == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_a_new_digest_bumps_the_version(tmp_path):
    from core.skill_management.forged_artifact import ForgeLedger

    ledger = ForgeLedger(tmp_path / "ledger.json")
    base = dict(
        skill_name="s",
        path="p",
        verified_at=1.0,
        boundary="seatbelt",
        probes_executed=1,
        probes_precommitted=1,
        summary="",
    )
    await ledger.record_async(LedgerEntry(digest="aaa", **base))
    assert ledger.entry_for("s").version == 1
    await ledger.record_async(LedgerEntry(digest="bbb", **base))
    assert ledger.entry_for("s").version == 2
    await ledger.record_async(LedgerEntry(digest="bbb", **base))
    assert ledger.entry_for("s").version == 2


@pytest.mark.asyncio
async def test_re_verifying_unchanged_code_keeps_its_record(tmp_path):
    """Re-recording the same digest must not erase what the skill has done."""
    from core.skill_management.forged_artifact import ForgeLedger

    ledger = ForgeLedger(tmp_path / "ledger.json")
    base = dict(
        skill_name="s",
        path="p",
        verified_at=1.0,
        boundary="seatbelt",
        probes_executed=1,
        probes_precommitted=1,
        summary="",
    )
    await ledger.record_async(LedgerEntry(digest="aaa", **base))
    await ledger.record_outcome_async("s", succeeded=True)
    await ledger.record_outcome_async("s", succeeded=True)

    await ledger.record_async(LedgerEntry(digest="aaa", **base))
    entry = ledger.entry_for("s")
    assert (entry.version, entry.successes) == (1, 2)


@pytest.mark.asyncio
async def test_replacing_the_code_does_not_inherit_the_old_record(tmp_path):
    """Successes belonged to the implementation that was just replaced."""
    from core.skill_management.forged_artifact import ForgeLedger

    ledger = ForgeLedger(tmp_path / "ledger.json")
    base = dict(
        skill_name="s",
        path="p",
        verified_at=1.0,
        boundary="seatbelt",
        probes_executed=1,
        probes_precommitted=1,
        summary="",
    )
    await ledger.record_async(LedgerEntry(digest="aaa", **base))
    await ledger.record_outcome_async("s", succeeded=True)
    await ledger.record_async(LedgerEntry(digest="bbb", **base))
    entry = ledger.entry_for("s")
    assert (entry.version, entry.successes, entry.failures) == (2, 0, 0)


@pytest.mark.asyncio
async def test_outcomes_accumulate_against_the_entry(tmp_path):
    from core.skill_management.forged_artifact import ForgeLedger

    ledger = ForgeLedger(tmp_path / "ledger.json")
    await ledger.record_async(
        LedgerEntry(
            skill_name="s",
            digest="d",
            path="p",
            verified_at=1.0,
            boundary="seatbelt",
            probes_executed=1,
            probes_precommitted=1,
            summary="",
        )
    )
    await ledger.record_outcome_async("s", succeeded=True)
    await ledger.record_outcome_async("s", succeeded=False)
    entry = ledger.entry_for("s")
    assert (entry.successes, entry.failures) == (1, 1)


@pytest.mark.asyncio
async def test_an_outcome_for_an_unknown_skill_is_ignored(tmp_path):
    from core.skill_management.forged_artifact import ForgeLedger

    ledger = ForgeLedger(tmp_path / "ledger.json")
    await ledger.record_outcome_async("never_forged", succeeded=True)
    assert ledger.entries() == []


def test_a_corrupt_ledger_yields_no_entries_rather_than_raising(tmp_path):
    from core.skill_management.forged_artifact import ForgeLedger

    path = tmp_path / "ledger.json"
    path.write_text("{not json", encoding="utf-8")
    assert ForgeLedger(path).entries() == []
