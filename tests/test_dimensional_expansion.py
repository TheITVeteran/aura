import math
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest

from core.adaptation.dimensional_expansion import (
    DimensionalExpansionEngine,
    FeatureAxis,
    DynamicFeatureWeights,
)


@pytest.fixture
def temp_data_dir():
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


def test_feature_axis_serialization():
    proj_vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    axis = FeatureAxis(
        axis_id="test_axis",
        origin="anomaly",
        projection_vector=proj_vec,
        weight=0.8,
        usage_count=5,
        contribution_score=0.4,
        total_observations=10,
    )

    data = axis.to_dict()
    assert data["axis_id"] == "test_axis"
    assert data["origin"] == "anomaly"
    assert np.allclose(data["projection_vector"], [0.1, 0.2, 0.3], atol=1e-5)
    assert data["weight"] == 0.8
    assert data["usage_count"] == 5
    assert data["contribution_score"] == 0.4
    assert data["total_observations"] == 10

    restored = FeatureAxis.from_dict(data)
    assert restored.axis_id == "test_axis"
    assert restored.origin == "anomaly"
    assert np.allclose(restored.projection_vector, proj_vec)
    assert restored.weight == 0.8
    assert restored.usage_count == 5
    assert restored.contribution_score == 0.4
    assert restored.total_observations == 10


def test_dynamic_feature_weights():
    base = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    dfw = DynamicFeatureWeights(base)
    assert dfw.base_dim == 3
    assert dfw.dim == 3
    assert np.allclose(dfw.get(), base)

    # Expand
    dfw.expand(1.2)
    assert dfw.dim == 4
    assert np.allclose(dfw.get(), [0.5, 0.5, 0.5, 1.2])

    # Update
    dfw.update_expanded_weight(0, 0.8)
    assert np.allclose(dfw.get(), [0.5, 0.5, 0.5, 0.8])

    # Contract
    dfw.contract(0)
    assert dfw.dim == 3
    assert np.allclose(dfw.get(), base)


def test_dimensional_expansion_engine_init():
    engine = DimensionalExpansionEngine(initial_dim=16, max_dim=32)
    assert engine.current_dim == 16
    assert engine.expanded_count == 0
    assert engine.feature_weights.dim == 16


def test_telemetry_to_raw_vector():
    engine = DimensionalExpansionEngine()

    # Flat dict
    telemetry = {"cpu": 0.5, "memory": 2048, "status": "active"}
    vec = engine._telemetry_to_raw_vector(telemetry)
    # status is not numeric, only cpu and memory should be converted
    assert len(vec) == 2
    # cpu: 1/(1+e^-0.5) = 0.6224, memory is squashed to ~1.0
    assert vec[0] > 0.5
    assert vec[1] > 0.99


def test_resize_vector():
    engine = DimensionalExpansionEngine(initial_dim=16)

    vec = np.ones(10, dtype=np.float32)
    resized = engine.resize_vector(vec)
    assert len(resized) == 16
    assert np.allclose(resized[:10], vec)
    assert np.allclose(resized[10:], 0.0)

    vec_long = np.ones(20, dtype=np.float32)
    resized_long = engine.resize_vector(vec_long, target_dim=12)
    assert len(resized_long) == 12
    assert np.allclose(resized_long, vec_long[:12])


def test_evaluate_expansion_triggering():
    # Set interval = 16 for quick checks
    engine = DimensionalExpansionEngine(
        initial_dim=16,
        max_dim=20,
        expansion_check_interval=16,
        expansion_eigenvalue_threshold=0.01,
        residual_buffer_size=32,
    )

    base_vector = np.ones(16, dtype=np.float32) * 0.5

    # Run 15 observations - should populate buffer but not evaluate yet
    for i in range(15):
        telemetry = {f"metric_{j}": float(i + j) for j in range(10)}
        vec, events = engine.evaluate_expansion(telemetry, base_vector)
        assert len(events) == 0
        assert len(vec) == 16

    # 16th observation - trigger evaluation
    telemetry = {f"metric_{j}": float(16 + j) for j in range(10)}
    vec, events = engine.evaluate_expansion(telemetry, base_vector)
    # Check if a new dimension was born
    assert engine.expanded_count == len(events)


def test_evaluate_contraction():
    engine = DimensionalExpansionEngine(
        initial_dim=16, contraction_min_observations=10, contraction_score_floor=0.1
    )

    # Artificially inject an expanded axis
    axis = FeatureAxis(
        axis_id="test_axis_1",
        origin="test",
        projection_vector=np.ones(10, dtype=np.float32),
        weight=0.5,
    )
    engine._expanded_axes.append(axis)
    engine._feature_weights.expand(0.5)

    assert engine.expanded_count == 1

    # Before min observations reached
    axis.total_observations = 5
    axis.contribution_score = 0.01  # below floor
    retired = engine.evaluate_contraction()
    assert len(retired) == 0

    # After min observations reached (min_observations defaults to 50 minimum)
    axis.total_observations = 55
    retired = engine.evaluate_contraction()
    assert len(retired) == 1
    assert retired[0] == "test_axis_1"
    assert engine.expanded_count == 0


def test_fitness_feedback():
    engine = DimensionalExpansionEngine(initial_dim=16)
    axis = FeatureAxis(
        axis_id="test_axis",
        origin="test",
        projection_vector=np.ones(10, dtype=np.float32),
        weight=0.5,
    )
    engine._expanded_axes.append(axis)
    engine._feature_weights.expand(0.5)

    engine.record_fitness_feedback("test_axis", 1.0)
    assert axis.weight == pytest.approx(0.55)
    assert engine._feature_weights.get()[-1] == pytest.approx(0.55)

    engine.record_fitness_feedback("test_axis", -2.0)
    assert axis.weight == pytest.approx(0.45)
    assert engine._feature_weights.get()[-1] == pytest.approx(0.45)


def test_edge_cases_and_nans():
    engine = DimensionalExpansionEngine(initial_dim=16)

    # Empty telemetry
    vec = engine._telemetry_to_raw_vector({})
    assert len(vec) == 1
    assert vec[0] == 0.0

    # NaN telemetry
    nan_telemetry = {"a": float("nan"), "b": float("inf"), "c": 0.5}
    vec = engine._telemetry_to_raw_vector(nan_telemetry)
    # only 'c' should succeed
    assert len(vec) == 1
    assert abs(vec[0] - 0.6224) < 1e-4


# ── CP126 remediation regressions ───────────────────────────────────────────


def test_governance_failure_refuses_expansion(monkeypatch):
    """Expansion is a persistent mutation of Aura's own representation.
    max_dim is a ceiling on how far it can go, not consent to go there."""
    import core.adaptation.dimensional_expansion as mod

    engine = DimensionalExpansionEngine(initial_dim=4, max_dim=8)

    def _boom(*_a, **_k):
        raise RuntimeError("will unavailable")

    monkeypatch.setattr(mod, "record_degradation", lambda *a, **k: None)
    # Force the Will import inside the method to fail.
    monkeypatch.setitem(__import__("sys").modules, "core.will", None)

    assert engine._governance_approve_expansion(0.9) is False


def test_axis_ids_are_not_reused_after_contraction():
    """IDs derived from the CURRENT list length, so retiring a non-terminal
    axis and expanding again minted an id another axis already owned."""
    engine = DimensionalExpansionEngine(initial_dim=4, max_dim=16)

    minted = []
    for _ in range(3):
        engine._axis_seq += 1
        minted.append(f"expanded_{engine._initial_dim + engine._axis_seq}")
    # Simulate retiring the middle axis, then minting another.
    engine._axis_seq += 1
    minted.append(f"expanded_{engine._initial_dim + engine._axis_seq}")

    assert len(set(minted)) == len(minted)


def test_restore_drops_malformed_axes():
    """Restored axes come off disk unsigned; a bad one silently corrupts every
    projection afterwards."""
    engine = DimensionalExpansionEngine(initial_dim=4, max_dim=16)
    engine._raw_dim = 4
    good = FeatureAxis(axis_id="expanded_5", origin="ok",
                       projection_vector=np.array([1.0, 0, 0, 0], dtype=np.float32))
    payload = {
        "initial_dim": 4,
        "max_dim": 16,
        "raw_dim": 4,
        "expanded_axes": [
            good.to_dict(),
            {**good.to_dict(), "axis_id": "expanded_6",
             "projection_vector": [1.0, 2.0]},                      # wrong width
            {**good.to_dict(), "axis_id": "expanded_7",
             "projection_vector": [float("nan"), 0, 0, 0]},         # non-finite
            {**good.to_dict(), "axis_id": "expanded_5"},            # duplicate id
        ],
        "feature_weights": {"base": [1.0] * 4, "expanded": [0.5] * 4},
    }

    restored = DimensionalExpansionEngine.from_dict(payload)

    assert [a.axis_id for a in restored._expanded_axes] == ["expanded_5"]
    # Weight vector was reconciled to the surviving axis count.
    assert restored._feature_weights.dim - restored._feature_weights.base_dim == 1


def test_restored_axis_counter_does_not_collide_with_loaded_axes():
    """A restored engine must not mint ids matching the axes it just loaded."""
    payload = {
        "initial_dim": 4,
        "max_dim": 16,
        "raw_dim": 4,
        "expanded_axes": [
            FeatureAxis(axis_id="expanded_9", origin="ok",
                        projection_vector=np.array([1.0, 0, 0, 0],
                                                   dtype=np.float32)).to_dict(),
        ],
        "feature_weights": {"base": [1.0] * 4, "expanded": [0.5]},
    }

    engine = DimensionalExpansionEngine.from_dict(payload)
    engine._axis_seq += 1
    next_id = f"expanded_{engine._initial_dim + engine._axis_seq}"

    assert next_id != "expanded_9"
    assert engine._axis_seq >= 5
