"""Internal bookkeeping writes must succeed under the live governance runtime.

Live incident (July 2026): with governance active, every cognitive-trace save
and every learning-example append was refused as a GOVERNANCE VIOLATION —
spawning incidents, inflating resilience frustration, and silently dropping
the write. Internal maintenance writers must establish their own
local_internal_governed_scope because they are invoked from arbitrary
contexts, including bare threads with no inherited scope.
"""
from __future__ import annotations

import contextlib
import json
import threading
import time

import pytest

from core.container import ServiceContainer


@contextlib.contextmanager
def _governance_runtime_forced_active(monkeypatch):
    monkeypatch.delenv("AURA_GOVERNANCE_MODE", raising=False)
    monkeypatch.delenv("AURA_REQUIRE_GOVERNANCE", raising=False)
    saved_services = dict(ServiceContainer._services)
    saved_aliases = dict(ServiceContainer._aliases)
    saved_locked = ServiceContainer._registration_locked
    try:
        ServiceContainer._services = {}
        ServiceContainer._aliases = {}
        ServiceContainer._registration_locked = True
        yield
    finally:
        ServiceContainer._services = saved_services
        ServiceContainer._aliases = saved_aliases
        ServiceContainer._registration_locked = saved_locked


def test_cognitive_trace_save_is_governed(monkeypatch, tmp_path):
    from core.governance_context import governance_runtime_active
    from core.meta.cognitive_trace import CognitiveTrace

    with _governance_runtime_forced_active(monkeypatch):
        assert governance_runtime_active() is True
        trace = CognitiveTrace(trace_id="governed-test")
        trace.log_dir = str(tmp_path)
        trace.record_step("reason", "governed write check")
        trace.save()

        saved = tmp_path / "trace_governed-test.json"
        assert saved.exists(), "trace save must not be refused under live governance"
        payload = json.loads(saved.read_text())
        assert payload["steps"][0]["content"] == "governed write check"


def test_learning_pipeline_record_is_governed_from_background_thread(monkeypatch, tmp_path):
    from core.governance_context import governance_runtime_active
    from core.learning.genuine_learning_pipeline import ExperienceBuffer

    with _governance_runtime_forced_active(monkeypatch):
        assert governance_runtime_active() is True
        pipeline = ExperienceBuffer(db_path=str(tmp_path / "examples.jsonl"))

        # The append runs on a fresh daemon thread; intercept thread start to
        # run it synchronously so the assertion window is deterministic while
        # still exercising the no-inherited-scope path.
        started: list[threading.Thread] = []
        real_thread = threading.Thread

        class ImmediateThread(real_thread):
            def start(self):
                started.append(self)
                self.run()

        monkeypatch.setattr(threading, "Thread", ImmediateThread)

        accepted = pipeline.record(
            system_prompt="sys",
            user_input="hello",
            response="a substantive reply that satisfies quality scoring here",
            quality_score=0.95,
        )

        assert accepted is True
        assert started, "background persist thread never started"
        assert pipeline.db_path.exists(), (
            "learning example append must not be refused under live governance"
        )
        line = pipeline.db_path.read_text().strip()
        record = json.loads(line)
        assert record["_meta"]["quality"] == pytest.approx(0.95)


def test_degradation_is_not_recorded_for_governed_bookkeeping(monkeypatch, tmp_path):
    """The live failure mode: refused writes spawned incidents every turn."""
    from core.meta.cognitive_trace import CognitiveTrace

    degradations: list[str] = []
    monkeypatch.setattr(
        "core.meta.cognitive_trace.record_degradation",
        lambda subsystem, exc, **kw: degradations.append(f"{subsystem}: {exc}"),
    )

    with _governance_runtime_forced_active(monkeypatch):
        trace = CognitiveTrace(trace_id=f"clean-{int(time.time())}")
        trace.log_dir = str(tmp_path)
        trace.save()

    assert degradations == [], f"governed save still degraded: {degradations}"
