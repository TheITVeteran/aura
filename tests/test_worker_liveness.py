"""Worker liveness: wedged, or just thinking?

Clean-room adoption of the request-slot discipline mature inference servers
settled on — the unit of failure is one request, not the loaded model. Aura's
resident Cortex is ~20GB of wired memory, and killing it has historically
cascaded (cold reload, a second worker stacking beside the first), so these
tests pin the rule that proof of life outranks proof of bookkeeping staleness.
"""
from __future__ import annotations

import pytest

from core.runtime.worker_liveness import (
    LivenessPolicy,
    LivenessVerdict,
    WorkerEvidence,
    classify_worker,
    kill_is_justified,
)

pytestmark = pytest.mark.unit


POLICY = LivenessPolicy(heartbeat_timeout_s=90.0, job_stall_s=120.0,
                        job_wedged_s=600.0)


# ── proof of life outranks everything ──────────────────────────────────────


def test_recent_output_is_never_wedged_however_stale_the_bookkeeping():
    """A worker producing tokens is not wedged, no matter what any state
    machine says. This is the inversion that made a stale lane state destroy a
    working model."""
    verdict = classify_worker(
        WorkerEvidence(process_alive=True, last_heartbeat_age_s=10_000.0,
                       active_job=True, job_age_s=10_000.0, loop_stalled=True,
                       last_progress_age_s=2.0),
        POLICY,
    )

    assert verdict.verdict is LivenessVerdict.GENERATING
    assert verdict.kill_justified is False


def test_a_fresh_heartbeat_with_a_young_job_is_generating():
    verdict = classify_worker(
        WorkerEvidence(process_alive=True, last_heartbeat_age_s=1.0,
                       active_job=True, job_age_s=15.0),
        POLICY,
    )

    assert verdict.verdict is LivenessVerdict.GENERATING
    assert verdict.kill_justified is False


# ── the graded response ────────────────────────────────────────────────────


def test_a_stalled_job_cancels_the_request_rather_than_the_model():
    """The whole point of the borrowed discipline: the request is the failure
    unit. A live heartbeat proves the process runs its own loop, so the model
    is fine even when one job is not progressing."""
    verdict = classify_worker(
        WorkerEvidence(process_alive=True, last_heartbeat_age_s=2.0,
                       active_job=True, job_age_s=200.0, loop_stalled=True),
        POLICY,
    )

    assert verdict.verdict is LivenessVerdict.STALLED
    assert verdict.should_cancel_request is True
    assert verdict.kill_justified is False, "must not destroy 20GB over one request"


def test_a_job_stalled_far_past_escalation_is_wedged():
    """Escalation exists — a stall that never resolves is not merely slow."""
    verdict = classify_worker(
        WorkerEvidence(process_alive=True, last_heartbeat_age_s=2.0,
                       active_job=True, job_age_s=900.0, loop_stalled=True),
        POLICY,
    )

    assert verdict.verdict is LivenessVerdict.WEDGED
    assert verdict.kill_justified is True


def test_silence_beyond_tolerance_is_wedged():
    verdict = classify_worker(
        WorkerEvidence(process_alive=True, last_heartbeat_age_s=300.0),
        POLICY,
    )

    assert verdict.verdict is LivenessVerdict.WEDGED
    assert verdict.kill_justified is True


def test_a_dead_process_is_dead():
    verdict = classify_worker(WorkerEvidence(process_alive=False), POLICY)

    assert verdict.verdict is LivenessVerdict.DEAD
    assert verdict.kill_justified is True


def test_idle_worker_is_not_killable_for_staleness():
    verdict = classify_worker(
        WorkerEvidence(process_alive=True, last_heartbeat_age_s=3.0,
                       active_job=False),
        POLICY,
    )

    assert verdict.verdict is LivenessVerdict.IDLE
    assert verdict.kill_justified is False


# ── absent evidence must never license an irreversible action ──────────────


def test_a_worker_that_has_not_reported_yet_is_starting_not_dying():
    """None and 'stale' are different claims; collapsing them would kill every
    worker during its first load."""
    verdict = classify_worker(
        WorkerEvidence(process_alive=True, last_heartbeat_age_s=None), POLICY
    )

    assert verdict.verdict is LivenessVerdict.UNKNOWN
    assert verdict.kill_justified is False


def test_no_evidence_at_all_is_unknown_and_not_killable():
    verdict = classify_worker(WorkerEvidence(), POLICY)

    assert verdict.verdict is LivenessVerdict.UNKNOWN
    assert verdict.kill_justified is False


@pytest.mark.parametrize("evidence", [
    WorkerEvidence(),
    WorkerEvidence(process_alive=True),
    WorkerEvidence(process_alive=True, last_heartbeat_age_s=None,
                   active_job=True, job_age_s=9999.0, loop_stalled=True),
])
def test_kill_is_never_justified_without_positive_evidence(evidence):
    assert kill_is_justified(evidence, POLICY) is False


# ── policy hygiene ─────────────────────────────────────────────────────────


def test_policy_env_overrides_are_clamped(monkeypatch):
    monkeypatch.setenv("AURA_WORKER_HEARTBEAT_TIMEOUT_S", "0.0001")
    monkeypatch.setenv("AURA_WORKER_JOB_WEDGED_S", "not-a-number")

    policy = LivenessPolicy.from_env()

    assert policy.heartbeat_timeout_s >= 5.0
    assert policy.job_wedged_s == 600.0


def test_verdict_serialises_with_its_reason_and_evidence():
    """The decision must be auditable after the fact."""
    verdict = classify_worker(
        WorkerEvidence(process_alive=True, last_heartbeat_age_s=300.0,
                       source="mlx_lane:warming"),
        POLICY,
    )
    payload = verdict.to_dict()

    assert payload["verdict"] == "wedged"
    assert "no heartbeat" in payload["reason"]
    assert payload["kill_justified"] is True
    assert payload["evidence"]["source"] == "mlx_lane:warming"
