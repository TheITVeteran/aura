from pathlib import Path
import threading

import numpy as np
import pytest

from core.brain.homeostatic_modulator import InferenceModulation
from core.brain.inference_feedback import InferenceFeedbackLoop
from core.container import ServiceContainer


class LiquidSubstrateFixture:
    idx_valence = 0
    idx_arousal = 1

    def __init__(self, values):
        self.x = np.array(values, dtype=np.float32)
        self.sync_lock = threading.RLock()
        self.feedback_events = []

    def accept_inference_feedback(self, **kwargs):
        self.feedback_events.append(kwargs)


class FeedbackEngineFixture:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []

    def accept_surprise_signal(self, *args, **kwargs):
        if self.fail:
            raise AssertionError("feedback invariant broken")
        self.calls.append((args, kwargs))

    def accept_inference_feedback(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class ProjectionFixture:
    def __init__(self):
        self.learn_calls = []

    def learn_step(self, *args, **kwargs):
        self.learn_calls.append((args, kwargs))


def test_inference_feedback_source_uses_typed_recoverable_errors():
    source = Path("core/brain/inference_feedback.py").read_text(encoding="utf-8")

    assert "except Exception" not in source
    assert "_FEEDBACK_RECOVERABLE_ERRORS" in source


@pytest.fixture
def base_modulation():
    return InferenceModulation(
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        logit_bias={},
        head_weights=np.ones(32, dtype=np.float32),
        urgency=0.5,
    )


def test_surprise_calculation_with_logprobs(base_modulation):
    loop = InferenceFeedbackLoop(substrate_dim=10)

    # All highly probable tokens (logprobs close to 0) -> low surprise
    logprobs = [-0.01, -0.05, -0.02]
    metrics = loop.process_output(
        output_text="test message",
        token_ids=[1, 2, 3],
        logprobs=logprobs,
        modulation=base_modulation,
        modulator_projection=None,
    )
    assert metrics["surprise"] < 0.1

    # Highly improbable tokens (large negative logprobs) -> high surprise
    logprobs = [-3.5, -4.0, -3.8]
    metrics = loop.process_output(
        output_text="test message",
        token_ids=[1, 2, 3],
        logprobs=logprobs,
        modulation=base_modulation,
        modulator_projection=None,
    )
    # Clipped at 3.0
    assert metrics["surprise"] == 3.0


def test_surprise_calculation_lexical_fallback(base_modulation):
    loop = InferenceFeedbackLoop(substrate_dim=10)

    # Empty/repetitive text should yield different surprise than unique text
    metrics_rep = loop.process_output(
        output_text="test test test test",
        token_ids=[1, 1, 1, 1],
        logprobs=None,
        modulation=base_modulation,
        modulator_projection=None,
    )
    metrics_uniq = loop.process_output(
        output_text="this is a unique collection of words",
        token_ids=[1, 2, 3, 4, 5, 6, 7],
        logprobs=None,
        modulation=base_modulation,
        modulator_projection=None,
    )

    # Repetitive text has low unique ratio -> higher surprise fallback score
    assert metrics_rep["surprise"] > metrics_uniq["surprise"]


def test_coherence_calculation_valence_alignment(base_modulation, monkeypatch):
    loop = InferenceFeedbackLoop(substrate_dim=5)

    substrate = LiquidSubstrateFixture([0.8, 0.5, 0.0, 0.0, 0.0])

    def service_lookup(name, default=None):
        if name == "liquid_substrate":
            return substrate
        return default

    monkeypatch.setattr("core.brain.inference_feedback.get_runtime_service", service_lookup)
    # 1. Output text with positive valence words (should align -> high coherence)
    metrics_pos = loop.process_output(
        output_text="completed success stable resolved",
        token_ids=[1, 2],
        logprobs=None,
        modulation=base_modulation,
        modulator_projection=None,
    )
    assert metrics_pos["coherence"] > 0.0

    # 2. Output text with negative valence words (should conflict -> low coherence)
    metrics_neg = loop.process_output(
        output_text="failed error danger hazard broken",
        token_ids=[3, 4],
        logprobs=None,
        modulation=base_modulation,
        modulator_projection=None,
    )
    assert metrics_neg["coherence"] < 0.0


def test_engine_feedback_injection(base_modulation, monkeypatch):
    loop = InferenceFeedbackLoop(substrate_dim=5)

    substrate = LiquidSubstrateFixture([0.5, 0.5, 0.0, 0.0, 0.0])
    free_energy = FeedbackEngineFixture()
    precision = FeedbackEngineFixture()

    def service_lookup(name, default=None):
        if name == "liquid_substrate":
            return substrate
        if name == "free_energy_engine":
            return free_energy
        if name == "precision_engine":
            return precision
        return default

    monkeypatch.setattr("core.brain.inference_feedback.get_runtime_service", service_lookup)
    loop.process_output(
        output_text="success resolved",
        token_ids=[1, 2],
        logprobs=[-0.1, -0.1],
        modulation=base_modulation,
        modulator_projection=None,
    )

    assert len(free_energy.calls) == 1
    assert len(substrate.feedback_events) == 1
    assert len(precision.calls) == 1


def test_feedback_injection_surfaces_invariant_failures(base_modulation, monkeypatch):
    loop = InferenceFeedbackLoop(substrate_dim=5)
    free_energy = FeedbackEngineFixture(fail=True)

    def service_lookup(name, default=None):
        if name == "free_energy_engine":
            return free_energy
        return default

    monkeypatch.setattr("core.brain.inference_feedback.get_runtime_service", service_lookup)
    with pytest.raises(AssertionError, match="feedback invariant broken"):
        loop.process_output(
            output_text="success resolved",
            token_ids=[1, 2],
            logprobs=[-0.1, -0.1],
            modulation=base_modulation,
            modulator_projection=None,
        )


def test_hebbian_projection_updates(base_modulation, monkeypatch):
    loop = InferenceFeedbackLoop(substrate_dim=5)

    # arousal = substrate.x[1] = 0.5
    substrate = LiquidSubstrateFixture([0.5, 0.5, 0.0, 0.0, 0.0])

    projection = ProjectionFixture()

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(lambda name, default=None: substrate))
    loop.process_output(
        output_text="test resolved",
        token_ids=[42, 43],
        logprobs=[-0.05, -0.05],
        modulation=base_modulation,
        modulator_projection=projection,
    )

    # learning_rate = 0.002 * (1.0 + arousal) = 0.002 * 1.5 = 0.003
    assert len(projection.learn_calls) == 1
    _args, kwargs = projection.learn_calls[0]
    assert kwargs["lr"] == 0.003
    assert kwargs["token_ids"] == [42, 43]
