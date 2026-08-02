"""Causal trace propagation: the machinery existed, nothing used it.

core/runtime/causal_trace.py has had inject_trace_carrier/extract_trace_carrier
for a while with ZERO call sites outside its own module — so a turn's trace
stopped dead at the IPC edge, and worker-side events could not be correlated
back to the conversation that caused them. The capability existing made it look
solved.
"""
from __future__ import annotations

import re

import pytest

from core.runtime.causal_trace import (
    extract_trace_carrier,
    inject_trace_carrier,
    new_trace,
    trace_scope,
)

pytestmark = pytest.mark.unit

W3C_TRACEPARENT = re.compile(r"00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}")


# ── W3C interop ────────────────────────────────────────────────────────────


def test_traceparent_is_a_valid_w3c_header():
    """A carrier that leaves Aura should be readable by any standard tool
    without a translation layer."""
    assert W3C_TRACEPARENT.fullmatch(new_trace("user_turn").traceparent())


def test_a_child_span_keeps_the_trace_id_and_changes_the_span_id():
    parent = new_trace("user_turn")
    child = parent.child("worker_request")

    _, p_trace, p_span, _ = parent.traceparent().split("-")
    _, c_trace, c_span, _ = child.traceparent().split("-")

    assert c_trace == p_trace, "one turn is one trace"
    assert c_span != p_span, "each step is its own span"


def test_an_unusual_id_still_renders_a_valid_header():
    """A non-hex or short id must not produce a header a parser rejects."""
    from core.runtime.causal_trace import TraceSpanContext

    weird = TraceSpanContext(trace_id="not-hex-at-all!", span_id="xyz")

    assert W3C_TRACEPARENT.fullmatch(weird.traceparent())


def test_traceparent_is_stable_for_the_same_span():
    span = new_trace("t")

    assert span.traceparent() == span.traceparent()


# ── the carrier survives a process boundary ────────────────────────────────


def test_a_job_carries_the_active_trace():
    span = new_trace("user_turn")
    with trace_scope(span):
        job = inject_trace_carrier({"id": "j1", "action": "generate"})

    assert job["trace_id"] == span.trace_id
    assert job["_causal_trace"]["traceparent"]


def test_the_far_side_rebuilds_the_context_from_the_job_alone():
    """This is what makes 'why did this turn fail?' answerable from one graph."""
    span = new_trace("user_turn")
    with trace_scope(span):
        job = inject_trace_carrier({"id": "j1"})

    rebuilt = extract_trace_carrier(job, fallback_name="mlx_worker")

    assert rebuilt is not None
    assert rebuilt.trace_id == span.trace_id


def test_no_active_trace_fabricates_no_correlation():
    """Inventing a trace id would be worse than having none — it would assert
    a causal link that does not exist."""
    job = inject_trace_carrier({"id": "j1"})

    assert "_causal_trace" not in job


def test_extraction_of_an_untraced_payload_returns_nothing():
    assert extract_trace_carrier({"id": "j1"}) is None


# ── wired at the real chokepoint ───────────────────────────────────────────


def _client():
    import core.brain.llm.mlx_client as mc

    for name in dir(mc):
        obj = getattr(mc, name)
        if isinstance(obj, type) and hasattr(obj, "_inject_causal_trace"):
            instance = obj.__new__(obj)
            instance._contract_key = None
            return instance
    raise AssertionError("no client exposes _inject_causal_trace")


def test_every_job_through_the_authorization_chokepoint_carries_the_trace():
    """Injecting at _authorize_job rather than at each job construction site
    means every job type is covered by one wiring, and no future job path can
    forget."""
    client = _client()
    span = new_trace("user_turn")

    with trace_scope(span):
        job = client._authorize_job({"id": "j", "action": "generate"},
                                    principal="test")

    assert job["trace_id"] == span.trace_id


def test_the_chokepoint_leaves_untraced_jobs_untouched():
    client = _client()

    job = client._authorize_job({"id": "j", "action": "generate"},
                                principal="test")

    assert job == {"id": "j", "action": "generate"}


def test_injection_never_breaks_a_job_when_tracing_is_unavailable(monkeypatch):
    """Observability must not be able to take down inference."""

    def _boom():
        raise RuntimeError("tracing subsystem down")

    monkeypatch.setattr("core.runtime.causal_trace.current_span", _boom)
    client = _client()

    job = client._authorize_job({"id": "j"}, principal="test")

    assert job == {"id": "j"}
