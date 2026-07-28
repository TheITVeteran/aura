"""The immune population must be able to express Aura's failures.

CP126 fae09510: behavioural rules are generated over whatever sensors exist,
and the only ones that existed described ports, vessels and a warehouse. So
every B cell Aura evolved was an opinion about maritime logistics, and the
adaptive population began with no code, service, queue, model, network,
memory or UI repair grammar. The vocabulary was the ceiling on what the
immune system could ever learn to repair.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.adaptation.adaptive_immunity import (
    _live_rule_vocabulary,
    _mutate_behavioral_rule,
)
from core.sensors.sensor_registry import SensorRegistry

_MARITIME = ("port_", "vessel_", "warehouse_")


@pytest.fixture()
def registry():
    return SensorRegistry()


def test_runtime_subsystems_have_sensors(registry):
    names = set(registry.read_all())
    runtime = {n for n in names if not n.startswith(_MARITIME)}

    assert len(runtime) >= 10


@pytest.mark.parametrize(
    "domain",
    ["runtime_", "model_", "memory_", "network_", "interface_", "storage_"],
)
def test_each_repair_domain_is_expressible(registry, domain):
    """A domain with no sensor is a domain no rule can be about."""
    assert any(name.startswith(domain) for name in registry.read_all())


def test_the_maritime_sensors_are_not_the_whole_vocabulary(registry):
    names = set(registry.read_all())
    maritime = {n for n in names if n.startswith(_MARITIME)}

    assert len(names - maritime) > len(maritime)


def test_seeded_rules_reach_aura_subsystems():
    vocabulary = _live_rule_vocabulary()
    assert vocabulary is not None

    rng = np.random.default_rng(0)
    sensors = set()
    for _ in range(60):
        rule = _mutate_behavioral_rule(None, rng)
        if rule:
            sensors.add(rule["conditions"][0]["sensor"])

    runtime = {s for s in sensors if not s.startswith(_MARITIME)}
    assert runtime, "the seeded population can only talk about shipping"


def test_the_population_is_not_one_rule_family():
    """Every initial B cell used to receive the same maritime flow rule."""
    rng = np.random.default_rng(3)
    rules = [_mutate_behavioral_rule(None, rng) for _ in range(20)]
    sensors = {r["conditions"][0]["sensor"] for r in rules if r}

    assert len(sensors) > 1


def test_a_declared_sensor_with_no_reading_is_still_declared(registry):
    """Declaring it is what lets a rule be ABOUT it, and what makes a missing
    reading visible rather than the subsystem invisible."""
    assert "runtime_event_loop_lag" in registry.read_all()
