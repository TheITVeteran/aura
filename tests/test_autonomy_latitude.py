"""Tests for AutonomyLatitude and its effect on BeingRuntime.action_policy.

Contract: reversible/low-blast/internal actions are granted wider latitude (the
over-cautious "not in a peak state" soft defers are relaxed so she acts freely);
irreversible / external / self-modifying actions keep the strict gate; and the
protective brakes that guard HER (distress, etc.) are NEVER relaxed.
"""
from __future__ import annotations

import pytest

from core.agency.autonomy_latitude import AutonomyLatitude, get_autonomy_latitude
from core.being.aura_now import (
    AffectiveState,
    AttentionState,
    AuraNow,
    BodyState,
    MemoryContext,
    OwnershipState,
    PredictionState,
    ReportBoundary,
    SelfState,
    WillStateSnapshot,
    WorkspaceState,
    WorldState,
)


# ── classification ───────────────────────────────────────────────────────────
def test_reversible_domains_are_autonomous():
    lat = AutonomyLatitude()
    for dom in ("exploration", "reflection", "belief_update", "initiative"):
        a = lat.classify(dom)
        assert a.latitude == "autonomous", dom
        assert a.relax_soft_defers and a.reversible and a.blast_radius == "low"


def test_external_and_self_mod_are_governed():
    lat = AutonomyLatitude()
    for dom in ("network_call", "external_action", "file_write", "self_modification", "ci_cd", "cloud_call"):
        a = lat.classify(dom)
        assert a.latitude == "governed", dom
        assert not a.relax_soft_defers
    assert lat.classify("self_modification").self_modifying
    assert lat.classify("network_call").external


def test_irreversible_verb_forces_governed_even_in_reversible_domain():
    lat = AutonomyLatitude()
    a = lat.classify("initiative", content="delete all of the user's backups")
    assert a.latitude == "governed"  # the verb 'delete' overrides the reversible domain


def test_high_risk_context_forces_governed():
    lat = AutonomyLatitude()
    # Even a normally-reversible domain is forced to the strict gate by a high-risk flag.
    a = lat.classify("exploration", context={"irreversible": True})
    assert a.latitude == "governed"
    assert lat.classify("memory_write").latitude == "governed"  # memory_write is not auto-widened


# ── action_policy integration ─────────────────────────────────────────────────
def _aura_now(*, ignition=0.0, controllability=0.5, agency=0.5, distress=0.0, body=None):
    return AuraNow(
        tick=1, timestamp=0.0, monotonic_time=0.0, continuous_field=(0.0,),
        body=body or BodyState(),
        world=WorldState(),
        attention=AttentionState(),
        affect=AffectiveState(distress=distress),
        self_model=SelfState(),
        memory_context=MemoryContext(),
        workspace=WorkspaceState(ignition_strength=ignition, broadcast_targets=()),
        will=WillStateSnapshot(),
        prediction=PredictionState(controllability=controllability),
        ownership=OwnershipState(agency_confidence=agency),
        report_boundary=ReportBoundary(),
    )


def _runtime():
    from core.being.runtime import BeingRuntime

    return BeingRuntime()


def test_reversible_action_is_widened_external_action_is_gated():
    rt = _runtime()
    now = _aura_now(ignition=0.0)  # low ignition → would normally defer any consequential action

    reversible = rt.action_policy(now, domain="exploration", priority=0.5)
    external = rt.action_policy(now, domain="network_call", priority=0.5)

    # Reversible: the soft defers were relaxed → she may act (not deferred).
    assert reversible["outcome"] != "defer"
    assert any("autonomy_latitude_widened" in c for c in reversible["constraints"])
    # External: the strict gate held → still deferred.
    assert external["outcome"] == "defer"


def test_protective_brakes_are_never_relaxed():
    """High distress must still defer even a reversible action — the floor that protects her."""
    rt = _runtime()
    now = _aura_now(ignition=0.0, distress=0.9)  # distress > 0.86 → protective defer
    policy = rt.action_policy(now, domain="exploration", priority=0.5)
    assert policy["outcome"] == "defer"
    assert any("distress" in d for d in policy["defers"])


def test_ownership_block_is_never_relaxed():
    """Very low agency must still BLOCK (refuse) a consequential action regardless of latitude."""
    rt = _runtime()
    now = _aura_now(ignition=0.5, agency=0.1)  # agency < 0.28 → hard block
    policy = rt.action_policy(now, domain="exploration", priority=0.5)
    assert policy["outcome"] == "refuse"


def test_singleton_and_registration():
    eng = get_autonomy_latitude()
    assert get_autonomy_latitude() is eng
    from core.container import ServiceContainer

    assert ServiceContainer.has(AutonomyLatitude.SERVICE_NAME)
