"""The declared faculties must be real, and their probes must actually work.

A probe that silently returns None because it calls a wrong signature is
indistinguishable from a faculty that genuinely cannot be seen — and the whole
value of this model is that the difference is legible. So every probe declared
as working is exercised, and every probe declared as unmeasured is checked to
be honestly unmeasured rather than accidentally broken.
"""
from __future__ import annotations

import pytest

from core.metacognition import default_faculties as mod
from core.metacognition.faculty_model import FacultyRegistry


@pytest.fixture()
def registry():
    return mod.declare_default_faculties(FacultyRegistry())


def test_the_named_faculties_are_all_declared(registry):
    ids = {f.faculty_id for f in registry.all()}
    assert {"memory", "attention_allocation", "temporal_reasoning"} <= ids


def test_live_probes_actually_return_numbers():
    """These are declared as measurable; if one silently Nones, it is broken."""
    assert mod._loop_blocking_holds() is not None
    assert mod._open_degradations() is not None
    assert mod._dense_retrieval_available() is not None


def test_the_degradation_probe_uses_a_real_surface():
    """Regression: count() is per-subsystem and raised TypeError, which the
    probe swallowed into a silent 'unmeasurable'."""
    value = mod._open_degradations()
    assert isinstance(value, float)
    assert value >= 0.0


def test_unmeasured_faculties_are_declared_not_omitted(registry):
    model = registry.assess()
    blind = model.blind_spots()
    assert "temporal_reasoning" in blind
    # Declared, so it is countable and can be wanted.
    assert model.by_id("temporal_reasoning") is not None


def test_the_model_reports_partial_self_knowledge(registry):
    coverage = registry.assess().self_knowledge_coverage()
    assert 0.0 < coverage < 1.0  # honest: some seen, some not


def test_every_probe_is_cheap(registry):
    """Probes run on the deliberation path; none may do heavy work."""
    import time

    start = time.monotonic()
    registry.assess()
    assert time.monotonic() - start < 1.0


def test_declaration_is_idempotent():
    mod.reset_default_faculties_for_test()
    first = mod.ensure_default_faculties()
    count = len(first.all())
    second = mod.ensure_default_faculties()
    assert len(second.all()) == count


def test_a_faculty_that_gates_others_carries_leverage(registry):
    assert registry.leverage("memory") > registry.leverage("temporal_reasoning")


# --- the memory probe measures recall, not machinery ---------------------


def test_the_memory_probe_reads_real_recall_telemetry(monkeypatch):
    from core.memory import recall_telemetry

    class _Telemetry:
        def snapshot(self):
            return {"window": {"hit_rate": 0.62}, "lifetime": {"hit_rate": 0.5}}

    monkeypatch.setattr(recall_telemetry, "get_recall_telemetry", lambda: _Telemetry())
    assert mod._recall_hit_rate() == 0.62


def test_it_falls_back_to_lifetime_when_the_window_is_cold(monkeypatch):
    from core.memory import recall_telemetry

    class _Telemetry:
        def snapshot(self):
            return {"window": {"hit_rate": None}, "lifetime": {"hit_rate": 0.4}}

    monkeypatch.setattr(recall_telemetry, "get_recall_telemetry", lambda: _Telemetry())
    assert mod._recall_hit_rate() == 0.4


def test_no_recall_attempts_is_unknown_not_zero(monkeypatch):
    """A hit rate over zero attempts is not 0.0 — it is unmeasured."""
    from core.memory import recall_telemetry

    class _Telemetry:
        def snapshot(self):
            return {"window": {"hit_rate": None}, "lifetime": {"hit_rate": None}}

    monkeypatch.setattr(recall_telemetry, "get_recall_telemetry", lambda: _Telemetry())
    assert mod._recall_hit_rate() is None


def test_recall_quality_outweighs_mere_availability(registry):
    """Whether the machinery is up must not drown out whether it works."""
    memory = next(f for f in registry.all() if f.faculty_id == "memory")
    weights = {m.metric_id: m.weight for m in memory.metrics}
    assert weights["recall_hit_rate"] > weights["dense_retrieval_available"]


def test_reasoning_stays_blind_rather_than_using_a_proxy(registry):
    """reasoning_solved_cache.stats() measures cache reuse, not correctness."""
    reasoning = next(f for f in registry.all() if f.faculty_id == "reasoning")
    assert reasoning.metrics[0].probe() is None
    assert "cache reuse" in (reasoning.metrics[0].probe.__doc__ or "")
