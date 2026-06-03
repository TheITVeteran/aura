from __future__ import annotations

import asyncio
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
