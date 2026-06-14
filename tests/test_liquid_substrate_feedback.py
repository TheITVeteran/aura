import time

import numpy as np
import pytest
import torch

from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig


def test_substrate_accepts_feedback():
    config = SubstrateConfig(neuron_count=16)
    substrate = LiquidSubstrate(config=config)

    # Base values
    substrate.x[substrate.idx_valence] = 0.0
    substrate.x[substrate.idx_frustration] = 0.0
    substrate.x[substrate.idx_focus] = 0.5
    substrate.x[substrate.idx_curiosity] = 0.5
    substrate.x_torch = torch.from_numpy(substrate.x).to(substrate.device).float()

    # Apply positive coherence feedback (surprise=0.5, coherence=1.0)
    substrate.accept_inference_feedback(surprise=0.5, coherence=1.0)

    # 1. Valence should increase
    assert substrate.x[substrate.idx_valence] > 0.0

    # 2. Focus should increase
    assert substrate.x[substrate.idx_focus] > 0.5

    # 3. Frustration should stay low (mitigated by coherence)
    # df = 0.1 * 0.5 - 0.1 * 1.0 = -0.05 -> clipped at 0.0
    assert substrate.x[substrate.idx_frustration] == 0.0


def test_frustration_increases_with_surprise():
    config = SubstrateConfig(neuron_count=16)
    substrate = LiquidSubstrate(config=config)

    substrate.x[substrate.idx_frustration] = 0.0
    substrate.x_torch = torch.from_numpy(substrate.x).to(substrate.device).float()

    # Apply high surprise, low coherence (surprise=3.0, coherence=-1.0)
    substrate.accept_inference_feedback(surprise=3.0, coherence=-1.0)

    # Frustration must increase significantly
    assert substrate.x[substrate.idx_frustration] > 0.0


def test_wundt_curve_curiosity_peaks_at_optimal_surprise():
    config = SubstrateConfig(neuron_count=16)
    substrate = LiquidSubstrate(config=config)

    def evaluate_curiosity_delta(surprise):
        # Reset curiosity
        substrate.x[substrate.idx_curiosity] = 0.5
        substrate.x_torch = (
            torch.from_numpy(substrate.x).to(substrate.device).float()
        )
        substrate.accept_inference_feedback(surprise=surprise, coherence=0.0)
        return substrate.x[substrate.idx_curiosity] - 0.5

    # Optimal surprise is 0.75. Let's compare delta at 0.75, 0.0, and 3.0
    delta_optimal = evaluate_curiosity_delta(0.75)
    delta_low = evaluate_curiosity_delta(0.0)
    delta_high = evaluate_curiosity_delta(3.0)

    assert delta_optimal > delta_low
    assert delta_optimal > delta_high


def test_substrate_feedback_clamping():
    config = SubstrateConfig(neuron_count=16)
    substrate = LiquidSubstrate(config=config)

    # Push to extremes multiple times
    for _ in range(50):
        # coherence positive pushes valence positive, surprise positive and coherence negative/zero pushes frustration positive
        substrate.accept_inference_feedback(surprise=5.0, coherence=0.0)
    substrate.x[substrate.idx_valence] = 1.0  # manually ensure valence is also maxed out

    assert substrate.x[substrate.idx_valence] == 1.0
    assert substrate.x[substrate.idx_frustration] == 1.0
    assert substrate.x[substrate.idx_focus] == 0.0  # 0.15 * 0 - 0.05 * 5 = -0.25 -> 0
    assert substrate.x[substrate.idx_curiosity] == 0.0

    for _ in range(50):
        substrate.accept_inference_feedback(surprise=0.0, coherence=5.0)

    assert substrate.x[substrate.idx_frustration] == 0.0  # frustration is bounded [0, 1]


def test_substrate_torch_synchronization():
    config = SubstrateConfig(neuron_count=16)
    substrate = LiquidSubstrate(config=config)

    substrate.accept_inference_feedback(surprise=1.0, coherence=0.5)

    # x and x_torch must contain identical values
    x_from_torch = substrate.x_torch.cpu().numpy()
    assert np.allclose(substrate.x, x_from_torch)


def test_substrate_weight_cache_starts_from_numpy_connectome_and_resyncs_when_dirty():
    config = SubstrateConfig(neuron_count=16)
    substrate = LiquidSubstrate(config=config)

    assert np.allclose(substrate.W, substrate.W_torch.cpu().numpy(), atol=1e-6)

    with substrate.sync_lock:
        substrate.W = np.zeros_like(substrate.W)
        substrate._mark_weight_cache_dirty()

    substrate._step_torch_math(0.01)

    assert np.allclose(substrate.W_torch.cpu().numpy(), 0.0)
    assert substrate._weight_cache_dirty is False
    assert substrate._cached_connectivity_norm == pytest.approx(0.0)


def test_substrate_prompt_marks_stale_snapshot_as_historical():
    config = SubstrateConfig(neuron_count=16)
    substrate = LiquidSubstrate(config=config)
    substrate.last_update = time.time() - 30.0
    substrate.current_update_rate = 20.0

    prompt_fragment = substrate.format_for_prompt()

    assert "Snapshot is stale" in prompt_fragment
    assert "historical telemetry" in prompt_fragment
