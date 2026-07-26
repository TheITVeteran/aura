"""CP126 contract tests for the homeostatic inference modulator.

These parameters reach the sampler, so an absent organ or an unvalidated
reading changes what Aura actually says.
"""
from __future__ import annotations

import tempfile
import threading

import numpy as np
import pytest

from core.brain import homeostatic_modulator as module
from core.brain.homeostatic_modulator import (
    MAX_SCORED_TOKENS,
    PARAM_BOUNDS,
    SubstrateLogitProjection,
)


def _projection(dim=8) -> SubstrateLogitProjection:
    return SubstrateLogitProjection(substrate_dim=dim, save_path=tempfile.mktemp(suffix=".json"))


class _FHN:
    def __init__(self, arousal=0.5, fatigue=0.1):
        self.arousal, self.fatigue = arousal, fatigue


class _Precision:
    def __init__(self, arousal=0.5, fatigue=0.1, temperature=0.7, heads=None):
        self.fhn = _FHN(arousal, fatigue)
        self._temperature = temperature
        self.heads = np.ones(32, dtype=np.float32) if heads is None else heads

    def get_temperature(self):
        return self._temperature

    def get_head_weights(self):
        return self.heads


class _FreeEnergy:
    def __init__(self, fe=0.3, urgency=0.4):
        self._smoothed_fe = fe
        self._urgency = urgency

    def get_action_urgency(self):
        return self._urgency


class _Substrate:
    def __init__(self, vector=None, idx=0):
        self.x = np.zeros(8, dtype=np.float32) if vector is None else vector
        self.idx_frustration = idx
        self.sync_lock = threading.RLock()


@pytest.fixture()
def modulator(monkeypatch):
    """A modulator whose ServiceContainer lookups the test controls."""
    services: dict = {}
    from core.container import ServiceContainer

    # Patch only the lookup, so the rest of the container contract (failure
    # policy resolution, degradation routing) stays real.
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda cls, name, default=None: services.get(name, default)),
    )
    inst = module.HomeostaticModulator.__new__(module.HomeostaticModulator)
    inst.projection = _projection()
    return inst, services


# --- 59c7356b: absent organs must not fabricate live readings -------------


def test_missing_organs_are_declared_not_disguised(modulator):
    inst, _services = modulator

    result = inst.compute_modulation()
    snapshot = result.source_snapshot

    assert snapshot["fully_measured"] is False
    assert snapshot["availability"] == {
        "precision_engine": False,
        "free_energy_engine": False,
        "liquid_substrate": False,
    }
    assert snapshot["measured"]["fhn_arousal"] is False
    assert snapshot["measured"]["free_energy"] is False
    assert snapshot["head_weight_source"] == "default"


def test_present_organs_are_marked_measured(modulator):
    inst, services = modulator
    services["precision_engine"] = _Precision()
    services["free_energy_engine"] = _FreeEnergy()
    services["liquid_substrate"] = _Substrate()

    snapshot = inst.compute_modulation().source_snapshot

    assert snapshot["fully_measured"] is True
    assert snapshot["measured"]["fhn_arousal"] is True
    assert snapshot["measured"]["head_weights"] is True
    assert snapshot["captured_at"] > 0


def test_a_partially_available_runtime_is_reported_per_channel(modulator):
    inst, services = modulator
    services["precision_engine"] = _Precision()

    snapshot = inst.compute_modulation().source_snapshot

    assert snapshot["measured"]["fhn_arousal"] is True
    assert snapshot["measured"]["free_energy"] is False
    assert snapshot["fully_measured"] is False


# --- 2bcec133: every reading is validated before it reaches the sampler ---


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -50.0, 99.0])
def test_hostile_precision_readings_cannot_break_the_sampler(modulator, bad):
    inst, services = modulator
    services["precision_engine"] = _Precision(arousal=bad, fatigue=bad, temperature=bad)

    result = inst.compute_modulation()

    for name, (low, high) in PARAM_BOUNDS.items():
        value = getattr(result, name)
        assert np.isfinite(value)
        assert low <= value <= high


def test_hostile_free_energy_is_clamped(modulator):
    inst, services = modulator
    services["free_energy_engine"] = _FreeEnergy(fe=float("nan"), urgency=float("inf"))

    result = inst.compute_modulation()

    assert np.isfinite(result.top_p) and np.isfinite(result.urgency)
    assert 0.0 <= result.urgency <= 1.0


def test_a_nan_substrate_vector_is_neutralized(modulator):
    inst, services = modulator
    vector = np.array([float("nan")] * 8, dtype=np.float32)
    services["liquid_substrate"] = _Substrate(vector=vector)

    result = inst.compute_modulation()

    assert np.isfinite(result.repetition_penalty)
    assert result.source_snapshot["input_faults"]


def test_a_raising_engine_falls_back_without_crashing(modulator):
    class Exploding:
        @property
        def fhn(self):
            raise RuntimeError("engine died")

        def get_head_weights(self):
            raise RuntimeError("engine died")

    inst, services = modulator
    services["precision_engine"] = Exploding()

    result = inst.compute_modulation()

    assert np.isfinite(result.temperature)
    assert result.source_snapshot["head_weight_source"] == "default"


# --- 67d4e9a4: the substrate snapshot is coherent -------------------------


def test_frustration_and_vector_come_from_one_locked_read(modulator):
    """A writer holding the lock cannot interleave between the two reads."""
    inst, services = modulator
    substrate = _Substrate(vector=np.full(8, 0.5, dtype=np.float32), idx=0)
    services["liquid_substrate"] = substrate

    observed: list[tuple[float, float]] = []
    original_lock = substrate.sync_lock

    class _WatchingLock:
        def __enter__(self):
            original_lock.acquire()
            return self

        def __exit__(self, *exc):
            # Record what the reader saw while still holding the lock.
            observed.append((float(substrate.x[0]), float(substrate.x[-1])))
            original_lock.release()
            return False

    substrate.sync_lock = _WatchingLock()
    inst.compute_modulation()

    assert observed and observed[0][0] == observed[0][1] == 0.5


def test_the_returned_vector_is_a_copy(modulator):
    inst, services = modulator
    vector = np.full(8, 0.4, dtype=np.float32)
    services["liquid_substrate"] = _Substrate(vector=vector)

    frustration, snapshot, _ = inst._read_substrate(services["liquid_substrate"])
    snapshot[0] = 99.0

    assert vector[0] == pytest.approx(0.4)
    assert frustration == pytest.approx(0.4)


def test_an_out_of_range_frustration_index_is_safe(modulator):
    inst, services = modulator
    services["liquid_substrate"] = _Substrate(idx=999)

    result = inst.compute_modulation()

    assert np.isfinite(result.repetition_penalty)


# --- 1233aa4c: head weights are an ownership-safe copy --------------------


def test_head_weights_are_copied_not_aliased(modulator):
    inst, services = modulator
    engine_owned = np.full(32, 0.5, dtype=np.float32)
    services["precision_engine"] = _Precision(heads=engine_owned)

    result = inst.compute_modulation()

    assert result.head_weights is not engine_owned
    assert result.head_weights.flags.writeable is False
    with pytest.raises(ValueError):
        result.head_weights[0] = 9.0
    assert engine_owned[0] == pytest.approx(0.5)


def test_a_later_engine_mutation_does_not_change_the_snapshot(modulator):
    inst, services = modulator
    engine_owned = np.full(32, 0.5, dtype=np.float32)
    services["precision_engine"] = _Precision(heads=engine_owned)

    result = inst.compute_modulation()
    engine_owned[:] = 99.0

    assert result.head_weights[0] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "bad", [np.zeros((2, 2), dtype=np.float32), np.array([], dtype=np.float32),
            np.array([float("nan")] * 4, dtype=np.float32), "not an array"]
)
def test_malformed_head_weights_fall_back_to_uniform(modulator, bad):
    inst, services = modulator
    services["precision_engine"] = _Precision(heads=bad)

    result = inst.compute_modulation()

    assert result.head_weights.shape == (32,)
    assert result.source_snapshot["head_weight_source"] == "default"


# --- d22709dc: dimension coercion is reported ----------------------------


def test_a_truncated_projection_is_declared():
    projection = _projection(dim=4)
    projection.weights = {1: np.ones(4, dtype=np.float32)}

    projection.get_biases(np.ones(10, dtype=np.float32))

    assert projection.last_projection["kind"] == "truncated"


def test_a_zero_padded_projection_is_declared():
    projection = _projection(dim=10)
    projection.weights = {1: np.ones(10, dtype=np.float32)}

    projection.get_biases(np.ones(4, dtype=np.float32))

    assert projection.last_projection["kind"] == "zero_padded"


def test_an_exact_projection_is_not_flagged():
    projection = _projection(dim=8)
    projection.weights = {1: np.ones(8, dtype=np.float32)}

    projection.get_biases(np.ones(8, dtype=np.float32))

    assert projection.last_projection["kind"] == "exact"


def test_incompatible_stored_weights_are_counted_not_silent():
    projection = _projection(dim=8)
    projection.weights = {
        1: np.ones(8, dtype=np.float32),
        2: np.ones(3, dtype=np.float32),
    }

    projection.get_biases(np.ones(8, dtype=np.float32))

    assert projection.last_projection["skipped"] == 1


# --- 1eb6e7ee: the scan is bounded and off the lock -----------------------


def test_the_token_scan_is_bounded():
    projection = _projection(dim=4)
    projection.weights = {
        token: np.full(4, 0.5, dtype=np.float32) for token in range(MAX_SCORED_TOKENS + 500)
    }

    projection.get_biases(np.ones(4, dtype=np.float32))

    assert projection.last_projection["scored"] == MAX_SCORED_TOKENS
    assert projection.last_projection["truncated"] is True


def test_scoring_does_not_hold_the_projection_lock():
    projection = _projection(dim=4)
    projection.weights = {token: np.full(4, 0.5, dtype=np.float32) for token in range(200)}
    acquired: list[bool] = []

    original = projection._lock

    class _CountingLock:
        def __enter__(self):
            original.acquire()
            acquired.append(True)
            return self

        def __exit__(self, *exc):
            original.release()
            acquired.append(False)
            return False

    projection._lock = _CountingLock()
    projection.get_biases(np.ones(4, dtype=np.float32))

    # Exactly one acquire/release pair: the snapshot. Scoring happened after.
    assert acquired == [True, False]


def test_biases_are_clamped_to_the_safe_logit_range():
    projection = _projection(dim=4)
    projection.weights = {1: np.full(4, 1000.0, dtype=np.float32)}

    biases = projection.get_biases(np.full(4, 1000.0, dtype=np.float32))

    assert all(-2.0 <= value <= 2.0 for value in biases.values())


def test_an_empty_projection_returns_nothing():
    projection = _projection(dim=4)

    assert projection.get_biases(np.ones(4, dtype=np.float32)) == {}
    assert projection.last_projection["scored"] == 0


# --- learn_step input validation ------------------------------------------


def test_a_nan_reward_cannot_poison_persisted_weights():
    projection = _projection(dim=4)
    # Seed a real weight so the NaN update lands on existing state rather than
    # being pruned as a fresh near-zero row.
    projection.learn_step(np.ones(4, dtype=np.float32), [7], 1.0, 0.0, lr=0.5)
    assert 7 in projection.weights

    projection.learn_step(np.ones(4, dtype=np.float32), [7], float("nan"), 1.0)

    assert np.all(np.isfinite(projection.weights[7]))


def test_a_nan_surprise_cannot_poison_persisted_weights():
    projection = _projection(dim=4)

    projection.learn_step(np.ones(4, dtype=np.float32), [7], 0.5, float("nan"))

    assert np.all(np.isfinite(projection.weights[7]))


def test_a_hostile_learning_rate_is_clamped():
    projection = _projection(dim=4)

    projection.learn_step(np.ones(4, dtype=np.float32), [7], 0.5, 0.0, lr=float("inf"))

    assert np.all(np.isfinite(projection.weights[7]))
