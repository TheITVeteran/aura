"""The immune system must learn about Aura, not about a logistics toy.

CP126 956ba926 — rule generation drew from a hardcoded maritime vocabulary
(port_east_load, Port_East -> Port_West), so every B-cell Aura evolved was an
opinion about a shipping simulation.

CP126 691b21ed — fitness scored Port_East/Port_West load imbalance. On any
runtime without those entities it raised KeyError, the broad except returned
0.0, and EVERY rule scored identically: evolution over noise, reported as
causal repair fitness.

CP126 3da4c199 — the lab swaps process-global singletons with no lock.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from core.adaptation import adaptive_immunity as mod


@pytest.fixture()
def rng():
    return np.random.default_rng(7)


# --- 956ba926: vocabulary comes from the live system ---------------------


def test_a_seeded_rule_uses_the_supplied_vocabulary(rng):
    vocab = {"sensors": ["memory.pressure"], "actuators": ["compact_memory"]}

    rule = mod._mutate_behavioral_rule(None, rng, vocabulary=vocab)

    assert rule["conditions"][0]["sensor"] == "memory.pressure"
    assert rule["actions"][0]["actuator"] == "compact_memory"


def test_no_vocabulary_means_no_rule_is_authored(rng):
    """A rule over sensors that do not exist can never fire; it would occupy a
    population slot pretending otherwise."""
    assert mod._mutate_behavioral_rule(None, rng, vocabulary=None) is None


def test_sensor_mutation_stays_inside_the_vocabulary(rng):
    vocab = {"sensors": ["a.sensor", "b.sensor"], "actuators": ["act"]}
    rule = {
        "conditions": [{"sensor": "a.sensor", "operator": ">", "value": 1.0}],
        "actions": [{"actuator": "act", "params": {}}],
    }

    for _ in range(30):
        rule = mod._mutate_behavioral_rule(rule, rng, vocabulary=vocab)
        assert rule["conditions"][0]["sensor"] in vocab["sensors"]


def test_mutation_without_vocabulary_keeps_the_existing_sensor(rng):
    """Substituting a hardcoded maritime name is the defect; inventing one is
    worse than leaving the rule alone."""
    rule = {
        "conditions": [{"sensor": "real.sensor", "operator": ">", "value": 1.0}],
        "actions": [{"actuator": "act", "params": {}}],
    }

    for _ in range(30):
        rule = mod._mutate_behavioral_rule(rule, rng, vocabulary=None)

    assert rule["conditions"][0]["sensor"] == "real.sensor"


def test_legacy_consequential_rule_is_replaced_from_bounded_grammar(rng):
    vocabulary = {
        "sensors": ["runtime_event_loop_lag"],
        "sensor_values": {"runtime_event_loop_lag": 0.2},
        "actuators": ["reallocate_flow"],
        "action_templates": {
            "reallocate_flow": {
                "source_id": "A",
                "target_id": "B",
                "amount": 10.0,
                "allow_partial": True,
            }
        },
    }
    legacy = {
        "conditions": [{"sensor": "runtime_event_loop_lag", "operator": ">", "value": 0.1}],
        "actions": [{"actuator": "git_operation", "params": {"allow_partial": -4.1}}],
    }

    normalized, migrated = mod._normalize_behavioral_rule(
        legacy,
        rng,
        vocabulary=vocabulary,
    )

    assert migrated is True
    assert normalized["actions"][0]["actuator"] == "reallocate_flow"
    assert normalized["actions"][0]["params"]["allow_partial"] is True


def test_boolean_action_parameters_are_not_numerically_mutated(rng):
    vocabulary = {
        "sensors": ["runtime_event_loop_lag"],
        "actuators": ["reallocate_flow"],
        "action_templates": {
            "reallocate_flow": {
                "source_id": "A",
                "target_id": "B",
                "amount": 10.0,
                "allow_partial": True,
            }
        },
    }
    rule = mod._mutate_behavioral_rule(None, rng, vocabulary=vocabulary)

    for _ in range(100):
        rule = mod._mutate_behavioral_rule(rule, rng, vocabulary=vocabulary)

    assert rule["actions"][0]["params"]["allow_partial"] is True


def test_the_hardcoded_maritime_vocabulary_is_gone():
    import inspect

    source = inspect.getsource(mod._mutate_behavioral_rule)
    for literal in ("Port_East", "Port_West", "vessel_alpha_speed", "warehouse_load"):
        assert literal not in source


def test_the_vocabulary_helper_fails_closed(monkeypatch):
    monkeypatch.setattr(
        mod, "_live_rule_vocabulary", mod._live_rule_vocabulary
    )
    # Both registries unreadable -> None, not a fabricated list.
    import core.sensors.sensor_registry as sr

    monkeypatch.setattr(
        sr, "get_sensor_registry",
        lambda: (_ for _ in ()).throw(RuntimeError("down")),
    )
    assert mod._live_rule_vocabulary() is None


# --- 691b21ed: the metric is general and honest --------------------------


def _world(entities):
    return SimpleNamespace(entities={e.entity_id: e for e in entities})


def _entity(eid, latency=0.0, load=0.0, capacity=100.0):
    return SimpleNamespace(
        entity_id=eid, latency=latency, load=load, capacity=capacity
    )


def test_pressure_is_defined_for_any_entity_set():
    """Not just ports — the old metric KeyError'd on anything else."""
    pressure = mod._system_pressure(_world([_entity("memory_pool", latency=2.0)]))

    assert pressure == pytest.approx(2.0)


def test_over_capacity_counts_as_pressure():
    pressure = mod._system_pressure(
        _world([_entity("q", load=150.0, capacity=100.0)])
    )

    assert pressure == pytest.approx(0.5)


def test_an_empty_world_is_unmeasurable_not_zero():
    """Zero pressure over zero entities is not a healthy system."""
    assert mod._system_pressure(_world([])) is None
    assert mod._system_pressure(None) is None


def test_non_finite_readings_do_not_poison_the_metric():
    pressure = mod._system_pressure(
        _world([_entity("x", latency=float("nan")), _entity("y", latency=1.0)])
    )

    assert pressure == pytest.approx(1.0)


def test_no_rule_is_unmeasurable_rather_than_zero():
    assert mod._evaluate_causal_fitness(None) is None


def test_unmeasurable_fitness_is_distinct_from_measured_zero():
    """The whole point: 0.0 must mean 'tested, no effect'."""
    import inspect

    source = inspect.getsource(mod._evaluate_causal_fitness)
    assert "return None" in source
    assert "return 0.0" in source


def test_the_sham_control_is_present():
    """Without it a rule is credited for improvement the world produced on
    its own."""
    import inspect

    source = inspect.getsource(mod._evaluate_causal_fitness)
    assert "control_model" in source
    assert "treatment_relief - control_relief" in source


def test_unmeasured_cells_are_not_scored_as_useless():
    import inspect

    source = inspect.getsource(mod.OfflineCoevolutionLab._objective)
    assert "if causal_fit is not None:" in source


# --- 714a9713: one objective, used for both selection and ranking --------


def test_selection_and_final_ranking_share_one_objective():
    """The lane used to select WITH causal fitness and then re-sort the
    survivors by receptor affinity alone, so it optimized one objective and
    shipped the winner of another."""
    import inspect

    evolve = inspect.getsource(mod.OfflineCoevolutionLab.evolve)
    assert evolve.count("self._objective(") >= 2
    # The affinity-only re-sort is gone.
    assert "key=lambda cell: sum(" not in evolve


def test_the_objective_includes_causal_fitness_for_repair_cells():
    import inspect

    source = inspect.getsource(mod.OfflineCoevolutionLab._objective)
    assert "_evaluate_causal_fitness" in source
    assert "CellKind.B" in source


# --- 3da4c199: the global swap is serialized ----------------------------


def test_the_isolation_flag_is_clear_when_no_lab_runs():
    assert mod.simulation_isolation_active() is False


def test_the_isolation_lock_serializes_labs():
    order = []

    def _hold():
        with mod._SIMULATION_ISOLATION_LOCK:
            order.append("in")
            order.append("out")

    threads = [threading.Thread(target=_hold) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    # Never interleaved: every "in" is immediately followed by its "out".
    assert order == ["in", "out"] * 4


def test_the_swap_is_taken_under_the_lock():
    import inspect

    source = inspect.getsource(mod._evaluate_causal_fitness)
    swap_at = source.index("wm._instance = treatment_model")
    lock_at = source.index("with _SIMULATION_ISOLATION_LOCK:")
    assert lock_at < swap_at


def test_the_singletons_are_restored_even_on_failure():
    import inspect

    source = inspect.getsource(mod._evaluate_causal_fitness)
    assert "finally:" in source
    tail = source.split("finally:", 1)[1]
    assert "wm._instance = original_model" in tail
    assert "_simulation_active.clear()" in tail
