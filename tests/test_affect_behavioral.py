"""Behavioral tests for the Affect Engine and Qualia caching.

Verifies the Damasio V2 oscillation detector, stuck valence watchdog,
and the QualiaSynthesizer's tick-based meta-qualia cache.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pytest
import asyncio
from unittest.mock import patch, MagicMock

from core.affect.damasio_v2 import AffectEngineV2, DamasioMarkers
from core.consciousness.qualia_synthesizer import QualiaSynthesizer, QualiaSnapshot


@pytest.fixture
def affect_engine():
    """Fresh AffectEngineV2 instance."""
    engine = AffectEngineV2()
    engine.markers = DamasioMarkers()
    return engine


@pytest.fixture
def qualia_synth():
    """Fresh QualiaSynthesizer instance."""
    return QualiaSynthesizer()


class TestAffectOscillationDetector:
    """Valence oscillations must trigger dampening momentum."""

    @pytest.mark.asyncio
    async def test_oscillation_dampens_momentum(self, affect_engine):
        """Rapid flipping between positive and negative valence should increase momentum."""
        initial_momentum = affect_engine.markers.momentum
        assert initial_momentum == 0.85
        
        # Inject rapid oscillations
        for _ in range(6):
            # Pos
            affect_engine.markers.emotions["joy"] = 1.0
            affect_engine.markers.emotions["fear"] = 0.0
            await affect_engine.pulse()
            
            # Neg
            affect_engine.markers.emotions["joy"] = 0.0
            affect_engine.markers.emotions["fear"] = 1.0
            await affect_engine.pulse()
            
        assert getattr(affect_engine, "_oscillation_flag", False) is True
        assert affect_engine.markers.momentum == 0.95, "Momentum did not dampen on oscillation."

    @pytest.mark.asyncio
    async def test_oscillation_recovery(self, affect_engine):
        """Stable valence should restore normal momentum."""
        affect_engine._oscillation_flag = True
        affect_engine.markers.momentum = 0.95
        affect_engine._valence_history = [0.8, -0.8, 0.8, -0.8] * 2  # Fake history
        
        # Inject stable state
        for _ in range(11):
            affect_engine.markers.emotions["joy"] = 0.8
            affect_engine.markers.emotions["fear"] = 0.0
            await affect_engine.pulse()
            
        assert getattr(affect_engine, "_oscillation_flag", False) is False
        assert affect_engine.markers.momentum == 0.85, "Momentum did not recover."


class TestPinnedValenceWatchdog:
    """Stuck highly negative valence should trigger a reset."""

    @pytest.mark.asyncio
    async def test_pinned_valence_resets(self, affect_engine):
        """Valence pinned at <= -0.95 should trigger a baseline reset."""
        affect_engine.markers.emotions["fear"] = 1.0
        affect_engine.markers.emotions["joy"] = 0.0
        
        # Pulse until just before threshold
        for _ in range(affect_engine._PINNED_RESET_AFTER - 1):
            await affect_engine.pulse()
            # Force pin back to 1.0 to defeat decay
            affect_engine.markers.emotions["fear"] = 1.0
            
        assert affect_engine.markers.emotions["fear"] == 1.0
        
        # Threshold tick
        await affect_engine.pulse()
        
        # Fear should be snapped halfway to baseline
        assert affect_engine.markers.emotions["fear"] < 1.0
        # Positive emotions should be nudged
        assert affect_engine.markers.emotions["joy"] > 0.0


class TestExpandedAffectiveDrivers:
    """Requested psychological drivers must affect runtime behavior, not just labels."""

    def test_requested_emotions_have_baselines_and_telemetry(self):
        markers = DamasioMarkers()
        expected = {
            "longing": 0.05,
            "upset": 0.02,
            "confused": 0.04,
            "loneliness": 0.05,
            "pride": 0.05,
            "frustration": 0.03,
            "curiosity": 0.10,
        }

        wheel = markers.get_wheel()

        for emotion, baseline in expected.items():
            assert emotion in markers.emotions
            assert markers.mood_baselines[emotion] == pytest.approx(baseline)
            assert emotion in wheel["experiential"]

    def test_error_stimulates_distress_and_physiology(self):
        markers = DamasioMarkers()

        markers.somatic_update("error", 1.0)

        assert markers.emotions["upset"] > 0.25
        assert markers.emotions["frustration"] > 0.25
        assert markers.emotions["confused"] > 0.25
        assert markers.cortisol > 10.0
        assert markers.heart_rate > 60.0
        assert markers.gsr > 1.5

    def test_interaction_resolves_longing_and_loneliness(self):
        markers = DamasioMarkers()
        markers.emotions["longing"] = 0.6
        markers.emotions["loneliness"] = 0.6
        markers.emotions["indifference"] = 0.4

        markers.somatic_update("interaction", 0.8)

        assert markers.emotions["longing"] < 0.6
        assert markers.emotions["loneliness"] < 0.6
        assert markers.emotions["indifference"] < 0.4
        assert markers.emotions["happiness"] > 0.0
        assert markers.emotions["trust"] > 0.0

    def test_temporal_pulse_builds_and_relieves_relational_absence(self):
        markers = DamasioMarkers()
        markers.last_interaction_time = time.time() - 400.0

        idle_deltas = markers.temporal_pulse()

        assert idle_deltas["loneliness"] > 0.0
        assert idle_deltas["longing"] > 0.0

        markers.temporal_texture = 0.9
        markers.last_interaction_time = time.time()

        fast_deltas = markers.temporal_pulse()

        assert fast_deltas["loneliness"] < 0.0
        assert fast_deltas["longing"] < 0.0
        assert fast_deltas["indifference"] < 0.0

    @pytest.mark.asyncio
    async def test_new_drivers_change_behavioral_modifiers(self, affect_engine):
        affect_engine.markers.emotions["confused"] = 0.8
        affect_engine.markers.emotions["curiosity"] = 0.7
        affect_engine.markers.emotions["upset"] = 0.6
        affect_engine.markers.emotions["frustration"] = 0.6
        affect_engine.markers.emotions["pride"] = 0.7

        modifiers = await affect_engine.get_behavioral_modifiers()

        assert modifiers["metacognition_depth"] > 1.7
        assert modifiers["risk_tolerance"] < 1.0
        assert modifiers["patience"] < 1.0
        assert modifiers["persistence"] > 1.3
        assert modifiers["creativity"] > 1.0

    def test_new_drivers_are_reflected_in_snapshot_status_and_legacy_state(self, affect_engine):
        affect_engine.markers.emotions["loneliness"] = 0.6
        affect_engine.markers.emotions["longing"] = 0.5
        affect_engine.markers.emotions["confused"] = 0.4
        affect_engine.markers.emotions["curiosity"] = 0.8
        affect_engine.markers.emotions["frustration"] = 0.7

        snapshot = affect_engine._snapshot_state()
        status = affect_engine.get_status()
        current = affect_engine.current

        assert snapshot.valence < 0.0
        assert status["loneliness"] == 60
        assert status["longing"] == 50
        assert status["confused"] == 40
        assert status["curiosity"] == 80
        assert status["frustration"] == 70
        assert status["experiential"]["curiosity"] == pytest.approx(0.8)
        assert current.curiosity == pytest.approx(0.8)
        assert current.frustration == pytest.approx(0.7)
        assert affect_engine._raw_state["curiosity_metric"] == pytest.approx(80.0)
        assert affect_engine._raw_state["frustration_metric"] == pytest.approx(70.0)

    def test_despair_spiral_releases_new_distress_states(self, affect_engine):
        affect_engine.markers.emotions["sadness"] = 0.95
        affect_engine.markers.emotions["fear"] = 0.85
        affect_engine.markers.emotions["joy"] = 0.0
        affect_engine.markers.emotions["upset"] = 0.8
        affect_engine.markers.emotions["frustration"] = 0.8
        affect_engine.markers.emotions["confused"] = 0.7
        affect_engine.markers.emotions["loneliness"] = 0.7
        affect_engine.markers.emotions["longing"] = 0.7

        affect_engine._check_for_despair_spiral()

        assert affect_engine.markers.emotions["upset"] < 0.8
        assert affect_engine.markers.emotions["frustration"] < 0.8
        assert affect_engine.markers.emotions["confused"] < 0.7
        assert affect_engine.markers.emotions["loneliness"] < 0.7
        assert affect_engine.markers.emotions["longing"] < 0.7

    def test_heuristic_appraisal_recognizes_new_affect_language(self):
        confused = AffectEngineV2._heuristic_appraisal(
            "I am confused and unclear about this failure",
            {"intensity": 0.8},
        )
        proud = AffectEngineV2._heuristic_appraisal(
            "The task succeeded and I feel pride in the result",
            {"intensity": 0.8},
        )

        assert confused["v"] < 0.0
        assert confused["a"] > 0.5
        assert confused["e"] > 0.5
        assert proud["v"] > 0.0


class TestQualiaCache:
    """Meta-qualia should cache per tick."""

    def test_meta_qualia_caches_per_tick(self, qualia_synth):
        """Calling compute_meta_qualia multiple times on same tick should return cached dict."""
        import numpy as np
        # Populate some history so it computes
        for _ in range(3):
            state = QualiaSnapshot(
                q_vector=np.random.rand(6),
                q_norm=0.5,
                pri=0.5,
                ual_profile={},
                is_attractor=False,
                dominant_dimension="visual",
                timestamp=time.time()
            )
            qualia_synth._history.append(state)
            
        qualia_synth._tick = 42
        
        # First call computes and caches
        meta1 = qualia_synth.compute_meta_qualia()
        
        # Tamper with the internal data to prove it doesn't recompute
        qualia_synth._history[-1].q_vector = np.zeros(6)
        
        # Second call should return exactly the identical dictionary object
        meta2 = qualia_synth.compute_meta_qualia()
        
        assert meta1 is meta2, "Cache was not used; recomputation occurred."

    def test_meta_qualia_cache_invalidates(self, qualia_synth):
        """Advancing the tick should invalidate the cache."""
        import numpy as np
        for _ in range(3):
            state = QualiaSnapshot(
                q_vector=np.random.rand(6),
                q_norm=0.5,
                pri=0.5,
                ual_profile={},
                is_attractor=False,
                dominant_dimension="visual",
                timestamp=time.time()
            )
            qualia_synth._history.append(state)
            
        qualia_synth._tick = 42
        meta1 = qualia_synth.compute_meta_qualia()
        
        # Advance tick
        qualia_synth._tick = 43
        meta2 = qualia_synth.compute_meta_qualia()
        
        assert meta1 is not meta2, "Cache did not invalidate on tick advance."


class TestQualiaEcho:
    """receive_qualia_echo must adjust Damasio emotions."""

    def test_qualia_echo_amplifies_dominant(self, affect_engine):
        """High qualia intensity should boost the dominant emotion."""
        affect_engine.markers.emotions["joy"] = 0.6
        affect_engine.markers.emotions["fear"] = 0.1
        
        affect_engine.receive_qualia_echo(q_norm=0.8, pri=0.5, trend=0.0)
        
        assert affect_engine.markers.emotions["joy"] > 0.6
        
    def test_qualia_echo_trend(self, affect_engine):
        """Trends should affect anticipation and sadness."""
        affect_engine.markers.emotions["anticipation"] = 0.5
        affect_engine.markers.emotions["sadness"] = 0.5
        
        # Rising trend
        affect_engine.receive_qualia_echo(q_norm=0.5, pri=0.5, trend=0.1)
        assert affect_engine.markers.emotions["anticipation"] > 0.5
        
        # Falling trend
        affect_engine.receive_qualia_echo(q_norm=0.5, pri=0.5, trend=-0.1)
        assert affect_engine.markers.emotions["sadness"] > 0.5
