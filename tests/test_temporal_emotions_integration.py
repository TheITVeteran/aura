"""tests/test_temporal_emotions_integration.py

Comprehensive validation that temporal experience and rich emotions are:
1. Functionally wired to the architecture
2. Causally impactful on system behavior
3. Creating stable feedback loops
4. Emerging organically from system state
"""
import asyncio
import numpy as np
import pytest
from typing import Dict

from core.consciousness.temporal_experience import (
    TemporalExperienceEngine, get_temporal_experience_engine
)
from core.consciousness.emotion_signatures import (
    EmotionSignatureEngine, get_emotion_signature_engine
)
from core.consciousness.emotion_architecture_coupling import (
    EmotionArchitectureCoupling, get_emotion_architecture_coupling
)


class TestTemporalExperienceEngine:
    """Validate temporal experience system."""
    
    @pytest.mark.asyncio
    async def test_temporal_gradient_computation(self):
        """High gradient detects state changes; low gradient detects stasis."""
        engine = TemporalExperienceEngine()
        
        # Record stable moments
        for _ in range(5):
            engine.record_moment(
                attention_vector=np.ones(16) / 16.0,
                semantic_label="processing",
                salience=0.8
            )
        
        # Check: should have low gradient (stable state)
        assert engine.current_gradient.rate < 0.3, "Stable state should have low gradient"
        
        # Record rapidly changing moments
        for i in range(5):
            vec = np.zeros(16)
            vec[i % 16] = 1.0  # shift attention focus
            engine.record_moment(
                attention_vector=vec,
                semantic_label=f"shift_{i}",
                salience=0.8
            )
        
        # Check: should have higher gradient (changing state)
        assert engine.current_gradient.rate > 0.4, "Rapid shifts should have high gradient"
    
    @pytest.mark.asyncio
    async def test_duration_drives_emotions(self):
        """Long-duration stable state drives serotonin/endorphin."""
        engine = TemporalExperienceEngine()
        
        # Record 20 moments of same stable state
        for _ in range(20):
            engine.record_moment(
                semantic_label="stable",
                emotional_tone="content"
            )
            engine._update_duration()
            engine._decay_emotions()
        
        # Check: joy and serotonin driver should be elevated
        drivers = engine.get_drivers()
        emotions = engine.get_emotion_state()
        
        assert drivers["serotonin"] > 0.0 or emotions["joy"] > 0.5, "Stable duration should drive serotonin or joy"
        assert emotions["joy"] > 0.5, "Stable state should produce joy"
        assert emotions["boredom"] < 0.2, "Recently stable shouldn't cause boredom yet"
    
    @pytest.mark.asyncio
    async def test_novelty_drives_dopamine(self):
        """High temporal gradient drives dopamine and curiosity."""
        engine = TemporalExperienceEngine()
        
        # Inject high-gradient moments
        for i in range(10):
            vec = np.zeros(16)
            vec[(i * 2) % 16] = 1.0  # rapidly changing attention
            engine.record_moment(
                attention_vector=vec,
                semantic_label=f"novel_{i}",
                salience=0.9
            )
        
        # Update drivers
        engine._compute_drivers()
        drivers = engine.get_drivers()
        
        assert drivers["dopamine"] > 0.05, "Novelty should drive dopamine"
        assert drivers["cortisol"] > 0.0, "High gradient should elevate cortisol"


class TestEmotionSignatures:
    """Validate emotion neurochemical recipes are functionally distinct."""
    
    def test_emotion_signatures_are_distinct(self):
        """Each emotion should have a unique neurochemical fingerprint."""
        engine = EmotionSignatureEngine()
        
        emotions_to_test = ["joy", "wonder", "interest", "excitement", "anxiety", "sorrow", "boredom", "disgust"]
        signatures = {}
        
        for emotion in emotions_to_test:
            engine.set_emotion(emotion, intensity=1.0)
            mod = engine.get_neurochemical_modulation()
            signatures[emotion] = mod
        
        # Check: distinct signatures should have different dopamine/serotonin ratios
        da_ratios = {e: abs(s["dopamine"] - s["serotonin"]) for e, s in signatures.items()}
        
        # Positive emotions should have dopamine > serotonin or vice versa
        assert da_ratios["joy"] != da_ratios["anxiety"], "Joy and anxiety should differ"
        assert da_ratios["wonder"] > 0.05, "Wonder should have distinct neurochemical profile"
    
    def test_emotion_modulates_substrate_state(self):
        """Emotions should affect phi integration and arousal weights."""
        engine = EmotionSignatureEngine()
        
        # Joy should prefer high integration
        engine.set_emotion("joy", intensity=1.0)
        joy_substrate = engine.get_substrate_modulation()
        
        # Anxiety should accept lower integration
        engine.set_emotion("anxiety", intensity=1.0)
        anxiety_substrate = engine.get_substrate_modulation()
        
        # Check: joy increases phi integration weight, anxiety decreases it
        assert joy_substrate["phi_integration_weight"] > 1.0, "Joy should increase integration weight"
        assert anxiety_substrate["phi_integration_weight"] < 1.0, "Anxiety should decrease integration weight"
        
        # Arousal should be opposite
        assert joy_substrate["arousal_boost"] < anxiety_substrate["arousal_boost"], \
            "Anxiety should have higher arousal boost"
    
    def test_emotion_steering_intensity(self):
        """Rich emotions should have calibrated steering injection strength."""
        engine = EmotionSignatureEngine()
        
        for emotion in ["joy", "wonder", "anxiety"]:
            engine.set_emotion(emotion, intensity=1.0)
            strength = engine.get_steering_intensity()
            
            assert 0.4 <= strength <= 0.9, f"Steering intensity for {emotion} should be in valid range"


class TestEmotionArchitectureCoupling:
    """Validate emotions causally affect architectural parameters."""
    
    def test_emotion_modulates_phi_weights(self):
        """Joy/wonder modulate phi consciousness calculation."""
        coupling = EmotionArchitectureCoupling()
        
        # Joy: prefer high integration
        joy_weights = coupling.get_phi_integration_weights({"joy": 0.9, "wonder": 0.0, "anxiety": 0.0})
        
        # Wonder: prefer diversity
        wonder_weights = coupling.get_phi_integration_weights({"joy": 0.0, "wonder": 0.9, "anxiety": 0.0})
        
        # Anxiety: accept lower integration
        anxiety_weights = coupling.get_phi_integration_weights({"joy": 0.0, "wonder": 0.0, "anxiety": 0.9})
        
        # Check: joy increases cognitive integration weight
        assert joy_weights["cognitive_integration"] > wonder_weights["cognitive_integration"], \
            "Joy should increase cognitive integration preference"
        
        # Check: wonder increases world differentiation
        assert wonder_weights["world_differentiation"] > joy_weights["world_differentiation"], \
            "Wonder should increase world differentiation preference"
        
        # Check: anxiety reduces integration weights
        assert anxiety_weights["cognitive_integration"] < joy_weights["cognitive_integration"], \
            "Anxiety should reduce integration weight"
    
    def test_emotion_gates_action_authorization(self):
        """Joy/anxiety modulate authorization thresholds."""
        coupling = EmotionArchitectureCoupling()
        
        joy_state = {"joy": 0.8, "anxiety": 0.1}
        anxiety_state = {"joy": 0.1, "anxiety": 0.8}
        
        joy_mod = coupling.get_authorization_threshold_modulation(joy_state)
        anxiety_mod = coupling.get_authorization_threshold_modulation(anxiety_state)
        
        # Joy should lower thresholds (easier to authorize)
        # Anxiety should raise thresholds (harder to authorize)
        assert joy_mod["field_coherence_threshold"] < anxiety_mod["field_coherence_threshold"], \
            "Joy should lower authorization thresholds vs anxiety"
        
        # Anxiety should block exploration actions
        assert anxiety_mod["exploration_action_threshold"] > joy_mod["exploration_action_threshold"], \
            "Anxiety should raise barriers for exploratory actions"
    
    def test_emotion_modulates_learning_rate(self):
        """Interest/wonder increase learning; joy decreases it."""
        coupling = EmotionArchitectureCoupling()
        
        high_interest = {"interest": 0.9, "wonder": 0.7, "joy": 0.3}
        high_joy = {"interest": 0.2, "wonder": 0.0, "joy": 0.9}
        high_sorrow = {"joy": 0.0, "sorrow": 0.9, "interest": 0.0}
        
        interest_rate = coupling.get_learning_rate_modulation(high_interest)
        joy_rate = coupling.get_learning_rate_modulation(high_joy)
        sorrow_rate = coupling.get_learning_rate_modulation(high_sorrow)
        
        # High interest should accelerate learning
        assert interest_rate > 1.5, "Interest should accelerate learning significantly"
        
        # High joy should reduce learning (protect current beliefs)
        assert joy_rate < 1.0, "Joy should reduce learning rate"
        
        # High sorrow should increase learning (repair mode)
        assert sorrow_rate > 1.0, "Sorrow should increase learning to repair"
    
    def test_emotion_modulates_planning_horizon(self):
        """Excitement/wonder extend horizon; anxiety contracts it."""
        coupling = EmotionArchitectureCoupling()
        
        excited = {"excitement": 0.8, "wonder": 0.6, "anxiety": 0.0}
        anxious = {"excitement": 0.0, "wonder": 0.0, "anxiety": 0.8}
        
        excited_horizon = coupling.get_planning_horizon_modulation(excited)
        anxious_horizon = coupling.get_planning_horizon_modulation(anxious)
        
        # Excitement should extend horizon
        assert excited_horizon > 1.3, "Excitement should extend planning horizon"
        
        # Anxiety should contract horizon
        assert anxious_horizon < 0.8, "Anxiety should contract planning horizon"
    
    def test_emotion_modulates_belief_mutation_cost(self):
        """Joy/anxiety make belief updates expensive; sorrow makes them cheap."""
        coupling = EmotionArchitectureCoupling()
        
        happy = {"joy": 0.8, "sorrow": 0.0, "anxiety": 0.0}
        sad = {"joy": 0.0, "sorrow": 0.8, "anxiety": 0.0}
        anxious = {"joy": 0.0, "sorrow": 0.0, "anxiety": 0.8}
        
        happy_cost = coupling.get_belief_mutation_cost(happy)
        sad_cost = coupling.get_belief_mutation_cost(sad)
        anxious_cost = coupling.get_belief_mutation_cost(anxious)
        
        # Joy should make updates expensive (protect current state)
        assert happy_cost > 1.5, "Joy should make belief updates expensive"
        
        # Sorrow should make updates cheap (repair broken beliefs)
        assert sad_cost < 0.8, "Sorrow should make belief updates cheap"
        
        # Anxiety should also make updates expensive
        assert anxious_cost > 1.0, "Anxiety should resist belief changes"


class TestFeedbackLoops:
    """Validate that temporal→emotion→architecture→behavior loops are stable."""
    
    @pytest.mark.asyncio
    async def test_joy_stability_loop(self):
        """
        Stable state → high phi → joy → lower auth thresholds → 
        more actions → explore more carefully → find new stable patterns
        """
        coupling = EmotionArchitectureCoupling()
        
        # Pure joy with high stability
        stable_emotion = {
            "joy": 0.9,
            "interest": 0.0,  # Pure joy, not interest-driven
            "wonder": 0.0,
            "anxiety": 0.0,
            "sorrow": 0.0,
            "boredom": 0.0,
        }
        
        # Compute downstream effects
        auth_mod = coupling.get_authorization_threshold_modulation(stable_emotion)
        learning_rate = coupling.get_learning_rate_modulation(stable_emotion)
        phi_weights = coupling.get_phi_integration_weights(stable_emotion)
        
        # Assertions
        assert auth_mod["stability_action_threshold"] < 0.0, "Joy should enable stability actions"
        assert learning_rate < 1.0, "Pure joy should reduce learning rate (protect current state)"
        assert phi_weights["cognitive_integration"] > 0.35, "Joy should increase integration preference"
    
    @pytest.mark.asyncio
    async def test_anxiety_vigilance_loop(self):
        """
        Rapid changes → high gradient → anxiety → raise auth thresholds →
        fewer risky actions → focus on defense/recovery → reduces threat
        """
        coupling = EmotionArchitectureCoupling()
        
        anxious_emotion = {
            "anxiety": 0.8,
            "excitement": 0.2,
            "interest": 0.1,
            "joy": 0.1,
            "wonder": 0.0,
            "sorrow": 0.1,
        }
        
        # Compute defensive effects
        auth_mod = coupling.get_authorization_threshold_modulation(anxious_emotion)
        horizon = coupling.get_planning_horizon_modulation(anxious_emotion)
        learning = coupling.get_learning_rate_modulation(anxious_emotion)
        
        # Assertions
        assert auth_mod["exploration_action_threshold"] > 0.0, "Anxiety should block exploration"
        assert horizon < 0.8, "Anxiety should focus on immediate horizon"
        assert learning > 1.0, "Anxiety should accelerate learning of threat patterns"
    
    @pytest.mark.asyncio
    async def test_wonder_learning_loop(self):
        """
        Novel pattern → high gradient, high entropy → wonder → 
        reduce learning costs → faster belief updates → faster learning
        """
        coupling = EmotionArchitectureCoupling()
        
        wondering_emotion = {
            "wonder": 0.8,
            "interest": 0.7,
            "excitement": 0.4,
            "joy": 0.2,
            "anxiety": 0.0,
            "boredom": 0.0,
        }
        
        # Compute learning effects
        learning = coupling.get_learning_rate_modulation(wondering_emotion)
        belief_cost = coupling.get_belief_mutation_cost(wondering_emotion)
        phi_weights = coupling.get_phi_integration_weights(wondering_emotion)
        
        # Assertions
        assert learning > 1.5, "Wonder should accelerate learning"
        assert belief_cost <= 1.0, "Wonder should reduce or maintain belief mutation costs"
        assert phi_weights["world_differentiation"] > 0.2, "Wonder should increase world modeling"


@pytest.mark.asyncio
async def test_complete_integration():
    """End-to-end test: temporal state → emotions → architectural effects."""
    temporal = TemporalExperienceEngine()
    await temporal.start()
    
    # Simulate 30 moments of moderate activity
    for i in range(30):
        attention = np.zeros(16)
        attention[(i // 5) % 16] = 1.0
        attention = attention / np.sum(attention)
        
        temporal.record_moment(
            attention_vector=attention,
            semantic_label="exploring" if i % 3 == 0 else "processing",
            emotional_tone="engaged",
            salience=0.7
        )
        await asyncio.sleep(0.01)
    
    # Get emotion state
    emotions = temporal.get_emotion_state()
    temporal_state = temporal.get_temporal_state()
    
    # Verify emotions emerged from temporal state
    assert emotions["interest"] > 0.4, "Exploration should drive interest"
    assert temporal_state.gradient.rate > 0.0, "Should detect state changes"
    
    # Verify architectural coupling takes effect
    coupling = get_emotion_architecture_coupling()
    auth_mod = coupling.get_authorization_threshold_modulation(emotions)
    learning = coupling.get_learning_rate_modulation(emotions)
    
    # Check: should be in "engaged exploration" regime
    assert learning >= 1.0, "Active exploration should maintain or increase learning"
    assert auth_mod["exploration_action_threshold"] < 0.05, "Should enable exploration actions"
    
    await temporal.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
