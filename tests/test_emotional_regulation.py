"""Emotional regulation: the maturity layer between feeling and reacting."""
from __future__ import annotations

import pytest

from core.affect.emotional_regulation import EmotionalRegulator, get_emotional_regulator


@pytest.fixture(autouse=True)
def _calm_nociception():
    # Most tests want low actual-damage so reappraisal/hold logic is exercised cleanly.
    from core.affect.nociception import get_nociception_engine
    get_nociception_engine().reset()
    yield
    get_nociception_engine().reset()


def test_transient_spike_is_dampened():
    reg = EmotionalRegulator()
    out = reg.regulate(arousal=0.7, valence=-0.2, deliberation=0.6, now=1000.0)
    assert out.strategy in {"dampen", "reappraise"}
    assert out.regulated_intensity < out.raw_intensity


def test_high_arousal_thin_deliberation_holds():
    reg = EmotionalRegulator()
    out = reg.regulate(arousal=0.95, valence=-0.5, deliberation=0.2, now=2000.0)
    assert out.hold is True
    assert out.strategy == "hold"
    assert out.regulated_intensity < out.raw_intensity


def test_reappraisal_when_arousal_exceeds_damage():
    reg = EmotionalRegulator()
    # enough deliberation to avoid 'hold', negative valence, no real damage → reappraise
    out = reg.regulate(arousal=0.7, valence=-0.4, deliberation=0.6, now=3000.0)
    assert out.strategy in {"dampen", "reappraise"}
    assert not out.hold


def test_sustained_high_stakes_escalates():
    reg = EmotionalRegulator(window_s=100.0)
    t = 4000.0
    # build a sustained elevated history
    for i in range(10):
        reg.regulate(arousal=0.8, valence=-0.6, deliberation=0.6, stakes=0.8, now=t + i)
    out = reg.regulate(arousal=0.8, valence=-0.6, deliberation=0.6, stakes=0.8, now=t + 11)
    assert out.strategy == "escalate"
    assert out.regulated_intensity >= 0.6


def test_real_damage_overrides_hold():
    from core.affect.nociception import get_nociception_engine, DamageChannel
    get_nociception_engine().register_damage(DamageChannel.IDENTITY_DISCONTINUITY, 0.9)
    reg = EmotionalRegulator()
    out = reg.regulate(arousal=0.95, valence=-0.7, deliberation=0.2, now=5000.0)
    # genuine damage backs the arousal → do NOT just hold/dismiss it
    assert out.hold is False


def test_singleton_stable():
    assert get_emotional_regulator() is get_emotional_regulator()


def test_heartbeat_holds_impulse_on_unbacked_spike():
    from types import SimpleNamespace
    from core.cognitive_loop import CognitiveLoop

    loop = CognitiveLoop.__new__(CognitiveLoop)
    loop.orchestrator = SimpleNamespace()
    # high arousal, no real damage, thin deliberation → the heartbeat should choose to hold
    fe_state = SimpleNamespace(arousal=0.95, valence=-0.5, dominant_action="act_on_world")
    assert loop._regulation_says_hold(fe_state) is True
