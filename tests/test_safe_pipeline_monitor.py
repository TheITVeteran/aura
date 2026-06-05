from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace


def _proposal(target: str, before: str, after: str):
    from core.self_modification.safe_pipeline import PipelineProposal, Stage

    return PipelineProposal(
        proposal_id="SMP-test",
        drive="stability",
        intent="test post-deploy monitor",
        file_path=target,
        before_source=before,
        after_source=after,
        stages_completed=[Stage.STAGED_DEPLOY.value],
    )


def _pipeline(tmp_path, monkeypatch, guardian):
    import core.self_modification.safe_pipeline as safe_pipeline
    from core.container import ServiceContainer
    from core.self_modification.safe_pipeline import SafePipeline

    monkeypatch.setattr(safe_pipeline, "_LEDGER_PATH", tmp_path / "pipeline.jsonl")

    def fake_get(cls, name, default=None):
        if name == "stability_guardian":
            return guardian
        return default

    monkeypatch.setattr(ServiceContainer, "get", classmethod(fake_get))

    pipeline = SafePipeline()
    pipeline.POST_DEPLOY_MONITOR_S = 0.01
    pipeline.POST_DEPLOY_POLL_S = 0.001
    return pipeline


def _install_pipeline_approval_fakes(monkeypatch):
    import core.ethics.conscience as conscience
    import core.governance.will_client as will_client
    import core.self_modification.formal_verifier as verifier
    from core.ethics.conscience import Verdict
    from core.self_modification.safe_pipeline import SafePipeline

    monkeypatch.setattr(
        verifier,
        "verify_mutation",
        lambda **_kwargs: SimpleNamespace(
            ok=True,
            invariants_violated=[],
            invariants_satisfied=["unit_test"],
        ),
    )
    monkeypatch.setattr(
        conscience,
        "get_conscience",
        lambda: SimpleNamespace(
            evaluate=lambda **_kwargs: SimpleNamespace(
                verdict=Verdict.APPROVE,
                rule_id="unit_test",
            )
        ),
    )

    async def decide_async(_self, _req):
        return SimpleNamespace(
            receipt_id="will-unit-test",
            is_approved=lambda: True,
            reason="unit-approved",
        )

    async def shadow_ok(_self, _sandbox_file):
        return True, "shadow-ok"

    monkeypatch.setattr(will_client.WillClient, "decide_async", decide_async)
    monkeypatch.setattr(SafePipeline, "_run_shadow", shadow_ok)


def test_post_deploy_monitor_rolls_back_when_stability_guardian_missing(tmp_path, monkeypatch):
    from core.self_modification.safe_pipeline import Stage

    target = tmp_path / "module.py"
    before = "value = 1\n"
    after = "value = 2\n"
    target.write_text(after, encoding="utf-8")
    proposal = _proposal(str(target), before, after)

    pipeline = _pipeline(tmp_path, monkeypatch, guardian=None)
    asyncio.run(pipeline._post_deploy_monitor(proposal, target, before))

    assert target.read_text(encoding="utf-8") == before
    assert proposal.blocked_at == Stage.POST_DEPLOY_MONITOR.value
    assert "stability_guardian_unavailable" in (proposal.blocked_reason or "")
    assert Stage.POST_DEPLOY_MONITOR.value not in proposal.stages_completed


def test_post_deploy_monitor_rolls_back_without_health_report(tmp_path, monkeypatch):
    from core.self_modification.safe_pipeline import Stage

    target = tmp_path / "module.py"
    before = "value = 1\n"
    after = "value = 2\n"
    target.write_text(after, encoding="utf-8")
    proposal = _proposal(str(target), before, after)
    guardian = SimpleNamespace(
        get_latest_report=lambda: None,
        get_health_summary=lambda: {
            "status": "initializing",
            "healthy": True,
            "message": "legacy optimistic startup",
        },
    )

    pipeline = _pipeline(tmp_path, monkeypatch, guardian=guardian)
    asyncio.run(pipeline._post_deploy_monitor(proposal, target, before))

    assert target.read_text(encoding="utf-8") == before
    assert proposal.blocked_at == Stage.POST_DEPLOY_MONITOR.value
    assert "initializing" in (proposal.blocked_reason or "")
    assert Stage.POST_DEPLOY_MONITOR.value not in proposal.stages_completed


def test_post_deploy_monitor_rolls_back_on_unhealthy_report(tmp_path, monkeypatch):
    from core.self_modification.safe_pipeline import Stage

    target = tmp_path / "module.py"
    before = "value = 1\n"
    after = "value = 2\n"
    target.write_text(after, encoding="utf-8")
    proposal = _proposal(str(target), before, after)
    guardian = SimpleNamespace(
        get_latest_report=lambda: {
            "overall_healthy": False,
            "checks": [{"name": "kernel", "healthy": False}],
            "timestamp": 123.0,
        }
    )

    pipeline = _pipeline(tmp_path, monkeypatch, guardian=guardian)
    asyncio.run(pipeline._post_deploy_monitor(proposal, target, before))

    assert target.read_text(encoding="utf-8") == before
    assert proposal.blocked_at == Stage.POST_DEPLOY_MONITOR.value
    assert "degraded" in (proposal.blocked_reason or "")
    assert Stage.POST_DEPLOY_MONITOR.value not in proposal.stages_completed


def test_post_deploy_monitor_completes_with_positive_health_evidence(tmp_path, monkeypatch):
    from core.self_modification.safe_pipeline import Stage

    target = tmp_path / "module.py"
    before = "value = 1\n"
    after = "value = 2\n"
    target.write_text(after, encoding="utf-8")
    proposal = _proposal(str(target), before, after)
    guardian = SimpleNamespace(
        get_latest_report=lambda: {
            "overall_healthy": True,
            "checks": [{"name": "kernel", "healthy": True}],
            "timestamp": 123.0,
        }
    )

    pipeline = _pipeline(tmp_path, monkeypatch, guardian=guardian)
    asyncio.run(pipeline._post_deploy_monitor(proposal, target, before))

    assert target.read_text(encoding="utf-8") == after
    assert proposal.blocked_at is None
    assert Stage.POST_DEPLOY_MONITOR.value in proposal.stages_completed


def test_safe_pipeline_quarantines_approved_patch_without_supervised_deploy(
    tmp_path,
    monkeypatch,
):
    import core.self_modification.safe_pipeline as safe_pipeline
    from core.self_modification.safe_pipeline import SafePipeline, Stage

    _install_pipeline_approval_fakes(monkeypatch)
    monkeypatch.delenv("AURA_ALLOW_SUPERVISED_SELF_MODIFICATION", raising=False)
    monkeypatch.setattr(safe_pipeline, "_LEDGER_PATH", tmp_path / "pipeline.jsonl")
    monkeypatch.setattr(safe_pipeline, "_STAGING_DIR", tmp_path / "staged")
    target = tmp_path / "core" / "brain" / "example_patch.py"
    target.parent.mkdir(parents=True)
    target.write_text("VALUE = 1\n", encoding="utf-8")

    proposal = asyncio.run(
        SafePipeline().run(
            drive="stability",
            intent="unit test approved patch must quarantine by default",
            file_path=str(target),
            before_source="VALUE = 1\n",
            after_source="VALUE = 2\n",
            owner_approved=True,
        )
    )

    assert proposal.blocked_at == Stage.STAGED_DEPLOY.value
    assert "operator_promotion_required" in (proposal.blocked_reason or "")
    assert proposal.promotion_artifact_path
    artifact_path = Path(proposal.promotion_artifact_path)
    assert artifact_path.is_file()
    assert artifact_path.read_text(encoding="utf-8") == "VALUE = 2\n"
    assert target.read_text(encoding="utf-8") == "VALUE = 1\n"
