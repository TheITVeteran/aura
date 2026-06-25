"""Nociception: operationally-grounded damage sensing and improvement/deterioration valence."""
from __future__ import annotations

import pytest

from core.affect.nociception import (
    DamageChannel,
    NociceptionEngine,
    channel_for_subsystem,
    get_nociception_engine,
)


def test_damage_raises_pressure_and_lowers_integrity():
    eng = NociceptionEngine()
    t = 1000.0
    assert eng.nociceptive_pressure(now=t) == 0.0
    eng.register_damage(DamageChannel.MEMORY_CORRUPTION, 0.9, now=t)
    assert eng.nociceptive_pressure(now=t) > 0.2
    assert eng.tissue_integrity(now=t) < 0.8


def test_pain_decays_over_time():
    eng = NociceptionEngine(half_life_s=10.0)
    t = 1000.0
    eng.register_damage(DamageChannel.FAILED_TOOL_USE, 1.0, now=t)
    p0 = eng.nociceptive_pressure(now=t)
    p_later = eng.nociceptive_pressure(now=t + 40.0)  # 4 half-lives
    assert p_later < p0 * 0.2  # mostly healed


def test_repeated_failures_escalate():
    eng = NociceptionEngine(half_life_s=1000.0, repeat_window_s=30.0)
    t = 1000.0
    single = NociceptionEngine(half_life_s=1000.0)
    single.register_damage(DamageChannel.FAILED_TOOL_USE, 0.2, now=t)
    one_off = single.nociceptive_pressure(now=t)

    for i in range(4):  # four hits in the repeat window
        eng.register_damage(DamageChannel.FAILED_TOOL_USE, 0.2, now=t + i)
    repeated = eng.nociceptive_pressure(now=t + 3)
    assert repeated > one_off  # repeated failed tool use hurts more


def test_valence_is_negative_while_deteriorating_positive_while_healing():
    eng = NociceptionEngine(half_life_s=15.0)
    t = 1000.0
    # rising damage → deteriorating → negative valence
    eng.register_damage(DamageChannel.GOVERNANCE_BREACH, 0.3, now=t)
    eng.register_damage(DamageChannel.GOVERNANCE_BREACH, 0.5, now=t + 1)
    eng.register_damage(DamageChannel.MEMORY_CORRUPTION, 0.6, now=t + 2)
    deteriorating = eng.grounded_valence(now=t + 2)
    assert deteriorating < 0

    # now let it heal with no new damage → improving → valence climbs
    healing = eng.grounded_valence(now=t + 60)
    assert healing > deteriorating


def test_subsystem_routing_to_channels():
    assert channel_for_subsystem("memory_facade") == DamageChannel.MEMORY_CORRUPTION
    assert channel_for_subsystem("self_awareness") == DamageChannel.IDENTITY_DISCONTINUITY
    assert channel_for_subsystem("governance_gateway") == DamageChannel.GOVERNANCE_BREACH
    assert channel_for_subsystem("desktop_tool") == DamageChannel.FAILED_TOOL_USE
    assert channel_for_subsystem("compute_resource") == DamageChannel.RESOURCE_EXHAUSTION
    assert channel_for_subsystem("something_unknown") == DamageChannel.GENERIC


def test_severity_intensity_ordering():
    crit = NociceptionEngine(half_life_s=1000.0)
    warn = NociceptionEngine(half_life_s=1000.0)
    t = 1000.0
    crit.ingest_degradation("memory", "critical", now=t)
    warn.ingest_degradation("memory", "warning", now=t)
    assert crit.nociceptive_pressure(now=t) > warn.nociceptive_pressure(now=t)


def test_record_degradation_feeds_nociception_live():
    """The canonical degradation sink must make damage felt."""
    from core.runtime.errors import record_degradation

    eng = get_nociception_engine()
    eng.reset()
    record_degradation("memory_facade", RuntimeError("index corrupted"), severity="critical")
    assert eng.nociceptive_pressure() > 0.0
    worst = eng.worst_channel()
    assert worst is not None and worst[0] == DamageChannel.MEMORY_CORRUPTION.value
    eng.reset()


def test_nociception_drives_phenomenal_body_error_pressure():
    """error_pressure (long dead — fed by a missing function) now reflects real damage."""
    from core.affect.phenomenal_integration import PhenomenalIntegrator

    eng = get_nociception_engine()
    eng.reset()
    integ = PhenomenalIntegrator()
    baseline = integ.collect_observations(orchestrator=None)
    # collect_observations(None) returns static defaults; exercise the real path via a
    # tiny orchestrator stub so the nociception branch runs.

    class _Orch:
        pass

    eng.register_damage(DamageChannel.MEMORY_CORRUPTION, 1.0)
    obs = integ.collect_observations(orchestrator=_Orch())
    assert obs["error_pressure"] > 0.2  # no longer pinned at the default
    assert obs["safety"] <= 0.8         # damage erodes felt safety
    eng.reset()
