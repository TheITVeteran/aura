from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from core.brain.llm.latent_cortex.campaign_journal import (
    CampaignJournal,
    CampaignJournalError,
    CampaignPlan,
    canonical_json_bytes,
)
from tests import detached_resume_harness as harness
from tools import run_detached_step
from tools import verify_latent_cortex_campaign_resume as verifier


def _assert_runner_accepts(verdict: Any) -> None:
    """Prove the verdict against the only process that ever consumes one.

    This verifier runs as its own process, so nothing binds it to the runner's
    contract at import time. Without this the schema and evidence digest can
    drift silently until a campaign actually resumes.
    """
    run_detached_step.validate_resume_verdict(
        verdict,
        plan_sha256="a" * 64,
        command_sha256="b" * 64,
        prior_attempt=1,
        prior_journal_head_sha256="c" * 64,
    )


def _bind_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    suffix: str = "first",
) -> None:
    monkeypatch.setenv("AURA_DETACHED_PLAN_SHA256", "a" * 64)
    monkeypatch.setenv("AURA_DETACHED_COMMAND_SHA256", "b" * 64)
    monkeypatch.setenv("AURA_DETACHED_PRIOR_ATTEMPT", "1")
    monkeypatch.setenv("AURA_DETACHED_PRIOR_JOURNAL_HEAD_SHA256", "c" * 64)
    monkeypatch.setenv("AURA_DETACHED_RESUME_EVIDENCE_TRANSPORT", "stdout-v3")


def _plan() -> CampaignPlan:
    return CampaignPlan.build(
        "resume-verifier-test",
        [{"domain": "mathematics", "seed": 7, "task_sha256": "c" * 64}],
        metadata={"profile": "test"},
    )


def _persist_plan(campaign_dir: Path, plan: CampaignPlan) -> None:
    campaign_dir.mkdir(parents=True, exist_ok=True)
    (campaign_dir / verifier.PLAN_FILE).write_bytes(
        canonical_json_bytes(plan.to_dict()) + b"\n"
    )


def _complete_cell(journal: CampaignJournal, cell_id: str) -> None:
    attempt_id = journal.start_cell(cell_id)
    journal.record_arm_result(cell_id, attempt_id, {"answer": "ok"})
    journal.record_verified(cell_id, attempt_id, {"accepted": True})
    journal.commit_cell(cell_id, attempt_id, {"raw_receipt_sha256": "d" * 64})


def test_missing_campaign_is_safe_to_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_environment(monkeypatch, tmp_path)
    verdict = verifier.verify_campaign(tmp_path / "missing")
    assert verdict["verdict"] == "safe_to_resume"
    assert verdict["evidence"]["reason"] == "campaign_not_started"
    assert verdict["prior_attempt"] == 1
    # Evidence rides inline on stdout-v3; no evidence file is written any more.
    assert "evidence_path" not in verdict
    assert not list(tmp_path.glob("*resume-evidence*"))
    _assert_runner_accepts(verdict)


def test_replayable_incomplete_journal_is_safe_to_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_environment(monkeypatch, tmp_path)
    plan = _plan()
    _persist_plan(tmp_path, plan)
    with CampaignJournal(tmp_path / verifier.JOURNAL_FILE, plan) as journal:
        journal.start_cell(plan.cell_ids[0])

    verdict = verifier.verify_campaign(tmp_path)
    assert verdict["verdict"] == "safe_to_resume"
    assert verdict["evidence"]["incomplete_count"] == 1
    assert verdict["evidence"]["reason"] == "journal_replay_allows_infrastructure_resume"
    _assert_runner_accepts(verdict)


def test_complete_bound_campaign_is_not_relaunched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_environment(monkeypatch, tmp_path)
    plan = _plan()
    _persist_plan(tmp_path, plan)
    with CampaignJournal(tmp_path / verifier.JOURNAL_FILE, plan) as journal:
        _complete_cell(journal, plan.cell_ids[0])
        manifest = journal.finalize(tmp_path / verifier.MANIFEST_FILE)
    grade_material = {
        "campaign_manifest_sha256": manifest["manifest_sha256"],
        "verdict": "incomplete_underpowered",
    }
    grade = {
        **grade_material,
        "grade_sha256": hashlib.sha256(canonical_json_bytes(grade_material)).hexdigest(),
    }
    (tmp_path / verifier.GRADE_FILE).write_bytes(canonical_json_bytes(grade) + b"\n")

    verdict = verifier.verify_campaign(tmp_path)
    assert verdict["verdict"] == "already_completed"
    assert verdict["evidence"]["committed_count"] == 1
    _assert_runner_accepts(verdict)


def test_orphan_or_tampered_artifacts_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_environment(monkeypatch, tmp_path)
    (tmp_path / verifier.JOURNAL_FILE).write_text("orphan\n", encoding="utf-8")
    verdict = verifier.verify_campaign(tmp_path)
    assert verdict["verdict"] == "indeterminate"
    _assert_runner_accepts(verdict)

    plan = _plan()
    _persist_plan(tmp_path, plan)
    _bind_environment(monkeypatch, tmp_path, suffix="second")
    with pytest.raises(CampaignJournalError):
        verifier.verify_campaign(tmp_path)


def test_verdict_is_accepted_by_the_real_detached_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner supplies the environment; nothing here sets it by hand.

    ``_bind_environment`` is deliberately not called: this proves the verifier
    against what ``run_detached_step`` actually exports, so the pair cannot
    drift apart the way they did at bd2961d3d.
    """
    verdict = harness.accepted_verdict(
        monkeypatch,
        lambda _environment: verifier.verify_campaign(tmp_path / "missing"),
        cwd=tmp_path,
    )
    assert verdict["verdict"] == "safe_to_resume"
    assert verdict["evidence"]["reason"] == "campaign_not_started"
    assert verdict["plan_sha256"] == harness.PLAN_SHA256
