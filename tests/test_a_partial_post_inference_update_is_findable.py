"""Nine downstream systems, nine fail-open blocks, one invisible outcome.

`_post_inference_update` advances CRSM, HOT, the hedonic gradient and LoRA
bridge, credit assignment, homeostasis, the world model, synaptic plasticity
and temporal continuity — each in its own `try/except` that records a
degradation and continues. Fail-open is right here: a response has already
reached the person, and no downstream bookkeeping is worth taking that back.

What was wrong is that a PARTIAL update — some of the nine advanced by this
response, others not — was invisible. Eight of the blocks recorded the
identical action string, "skipped unavailable post-inference update hook
after response delivery", so the trail could not say which hook was skipped,
and nothing tied the records from one response together.

This does NOT add rollback, and these tests assert that it doesn't claim to.
CRSM, the world model and synaptic plasticity are not transactional stores;
there is nothing to roll back to. What exists now is a shared id, a stage
name on every record, and a receipt naming exactly which stages ran — so a
run of half-updates is a rate somebody can find rather than nine anecdotes
nobody connects.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _gate():
    import core.brain.inference_gate as ig

    cls = next(
        obj
        for obj in vars(ig).values()
        if inspect.isclass(obj) and hasattr(obj, "_post_inference_update")
    )
    gate = cls.__new__(cls)
    gate._last_successful_generation_at = 0.0
    return gate


# ─────────────────────────────── the receipt exists and is honest


def test_a_clean_sequence_reports_complete():
    gate = _gate()

    gate._post_inference_update("a real answer")
    receipt = gate.post_inference_receipt()

    assert receipt["complete"] is True
    assert receipt["skipped"] == []
    assert receipt["update_id"].startswith("pi_")


def test_a_failing_stage_is_named_in_the_receipt(monkeypatch):
    """The whole point: which of the nine did not run."""
    import core.container

    real_get = core.container.ServiceContainer.get

    def _broken(name, default=None):
        if name == "homeostasis":
            raise RuntimeError("homeostasis service is broken")
        return real_get(name, default)

    monkeypatch.setattr(core.container.ServiceContainer, "get", staticmethod(_broken))

    gate = _gate()
    gate._post_inference_update("a real answer")
    receipt = gate.post_inference_receipt()

    assert receipt["complete"] is False
    assert "homeostasis" in receipt["skipped"], (
        f"a failed stage is missing from the receipt: {receipt['skipped']}"
    )


def test_every_stage_the_receipt_names_is_declared():
    """A stage that appends a name not in the declared tuple would report a
    skip nobody can map back to anything."""
    import core.brain.inference_gate as ig

    source = inspect.getsource(ig)
    appended = set(
        line.split('_skipped_stages.append("')[1].split('"')[0]
        for line in source.splitlines()
        if "_skipped_stages.append(" in line
    )

    assert appended <= set(ig._POST_INFERENCE_STAGES), (
        f"these stages report skips but are not declared: "
        f"{appended - set(ig._POST_INFERENCE_STAGES)}"
    )


def test_every_declared_stage_can_report_a_skip():
    """The reverse: a declared stage with no `append` can fail silently and
    still be counted as having run."""
    import core.brain.inference_gate as ig

    source = inspect.getsource(ig)
    appended = set(
        line.split('_skipped_stages.append("')[1].split('"')[0]
        for line in source.splitlines()
        if "_skipped_stages.append(" in line
    )

    missing = set(ig._POST_INFERENCE_STAGES) - appended
    assert not missing, (
        f"these declared stages never record a skip, so a failure in them "
        f"still reports as a complete update: {missing}"
    )


# ─────────────────────── the degradation trail can be joined


def test_no_stage_uses_the_old_indistinguishable_action_string():
    """Eight blocks shared one sentence, so the trail could not say which
    hook was skipped."""
    source = (ROOT / "core" / "brain" / "inference_gate.py").read_text("utf-8")

    assert (
        "skipped unavailable post-inference update hook after response delivery"
        not in source
    ), "the generic post-inference action string is back"


def test_each_stage_record_carries_the_shared_update_id():
    source = (ROOT / "core" / "brain" / "inference_gate.py").read_text("utf-8")

    named = source.count('"update_id": _update_id')

    import core.brain.inference_gate as ig

    assert named >= len(ig._POST_INFERENCE_STAGES), (
        "not every stage record carries the correlation id, so records from "
        "one response cannot be joined"
    )


# ──────────────────────────── it does not claim to roll back


def test_the_partial_state_is_reported_as_unrecoverable():
    """Claiming rollback across non-transactional cognitive subsystems would
    be the overclaim this pass exists to remove."""
    source = (ROOT / "core" / "brain" / "inference_gate.py").read_text("utf-8")

    assert "no rollback exists for these subsystems" in source


def test_the_reasoning_stays_next_to_the_code():
    """The stricter-looking option — adding rollback — has to stay refused
    for a stated reason, or the next reader adds it."""
    source = (ROOT / "core" / "brain" / "inference_gate.py").read_text("utf-8")

    assert "not transactional stores" in source


def test_a_partial_update_records_a_degradation(monkeypatch):
    import core.brain.inference_gate as ig
    import core.container

    real_get = core.container.ServiceContainer.get

    def _broken(name, default=None):
        if name == "credit_assignment":
            raise RuntimeError("credit service is broken")
        return real_get(name, default)

    monkeypatch.setattr(core.container.ServiceContainer, "get", staticmethod(_broken))

    recorded: list[dict] = []
    monkeypatch.setattr(
        ig,
        "_record_inference_degradation",
        lambda exc, **kw: recorded.append(kw),
    )

    gate = _gate()
    gate._post_inference_update("a real answer")

    summaries = [
        entry
        for entry in recorded
        if "partial post-inference state" in str(entry.get("action", ""))
    ]
    assert summaries, "a partial update produced no summary record"
    assert summaries[0]["extra"]["complete"] is False


def test_an_empty_response_updates_nothing():
    """An empty response is not a success and must not advance anything."""
    gate = _gate()

    gate._post_inference_update("   ")

    assert gate.post_inference_receipt() == {}
