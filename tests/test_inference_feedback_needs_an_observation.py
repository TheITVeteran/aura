"""Feedback that nobody observed must not train anything.

CP126 on core/brain/inference_feedback.py, which closes the loop from LLM
output back into Aura's homeostatic substrate and trains the logit
projection. When the substrate service was absent it substituted a zero
vector, valence 0.0 and arousal 0.5, computed coherence from those
invented numbers — coming out at a perfect 1.0 — fed them to three
engines, and trained the projection on the synthetic vector.

Perfect alignment with a state nobody had observed, and it became durable
weights.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from core.brain.inference_feedback import InferenceFeedbackLoop


class _Projection:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def learn_step(self, **kwargs):
        self.calls.append(kwargs)


class _Substrate:
    """A liquid substrate with a controllable state vector."""

    def __init__(self, dim=512, valence=0.5, arousal=0.5) -> None:
        import threading

        self.sync_lock = threading.Lock()
        self.x = np.zeros(dim, dtype=np.float32)
        self.idx_valence, self.idx_arousal = 0, 1
        self.x[0], self.x[1] = valence, arousal
        self.fed: list[dict] = []

    def accept_inference_feedback(self, **kwargs):
        self.fed.append(kwargs)


@pytest.fixture
def loop():
    return InferenceFeedbackLoop(substrate_dim=512)


def _process(loop, projection, *, text="the repair completed and the system is stable", **over):
    kwargs = dict(
        output_text=text,
        token_ids=[1, 2, 3],
        logprobs=[-0.5, -0.4, -0.6],
        modulation=SimpleNamespace(temperature=0.8, top_p=0.9),
        modulator_projection=projection,
    )
    kwargs.update(over)
    return loop.process_output(**kwargs)


# ------------------------------------------------- the false observation


def test_an_absent_substrate_never_trains_the_projection(loop, monkeypatch):
    monkeypatch.setattr(
        "core.brain.inference_feedback.get_runtime_service",
        lambda name, default=None: default,
    )
    projection = _Projection()
    result = _process(loop, projection)

    assert projection.calls == [], (
        "the projection was trained on a synthetic zero vector standing in "
        "for a substrate nobody observed"
    )
    assert result["substrate_available"] is False
    assert result["projection_trained"] is False


def test_an_absent_substrate_does_not_produce_perfect_coherence(loop, monkeypatch):
    monkeypatch.setattr(
        "core.brain.inference_feedback.get_runtime_service",
        lambda name, default=None: default,
    )
    result = _process(loop, _Projection())
    assert result["coherence_grounded"] is False
    assert result["coherence"] != 1.0


def test_an_absent_substrate_reports_no_valence_rather_than_zero(loop, monkeypatch):
    """0.0 is a real valence. None is the absence of one."""
    monkeypatch.setattr(
        "core.brain.inference_feedback.get_runtime_service",
        lambda name, default=None: default,
    )
    assert _process(loop, _Projection())["substrate_valence"] is None


def test_ungrounded_coherence_never_doses_its_consumers(loop, monkeypatch):
    """Surprise and coherence are grounded independently.

    Surprise comes from the model's own logprobs and is a real measurement
    with or without a substrate. Only coherence needs an observed state —
    so only its consumers are withheld. Collapsing the two would have been
    the same mistake in the other direction: refusing a measurement that
    was actually made.
    """
    monkeypatch.setattr(
        "core.brain.inference_feedback.get_runtime_service",
        lambda name, default=None: default,
    )
    result = _process(loop, _Projection())
    assert result["engines_fed"]["coherence_consumers"] == (
        "skipped:coherence_never_observed"
    )
    assert "liquid_substrate" not in result["engines_fed"]
    assert "precision_engine" not in result["engines_fed"]


def test_a_grounded_surprise_still_reaches_the_free_energy_engine(loop, monkeypatch):
    """The control: withholding coherence must not suppress a real measurement."""
    fed: list[float] = []

    class _FreeEnergy:
        def accept_surprise_signal(self, value):
            fed.append(value)

    monkeypatch.setattr(
        "core.brain.inference_feedback.get_runtime_service",
        lambda name, default=None: _FreeEnergy() if name == "free_energy_engine" else default,
    )
    result = _process(loop, _Projection())
    assert result["engines_fed"]["free_energy_engine"] == "applied"
    assert fed and 0.0 <= fed[0] <= 1.0


def test_a_lexical_surprise_estimate_does_not_dose_the_free_energy_engine(loop, monkeypatch):
    """No logprobs means surprise is an estimate, not a measurement."""
    monkeypatch.setattr(
        "core.brain.inference_feedback.get_runtime_service",
        lambda name, default=None: default,
    )
    result = _process(loop, _Projection(), logprobs=None)
    assert result["engines_fed"]["free_energy_engine"] == "skipped:surprise_ungrounded"


def test_an_observed_substrate_does_train(loop, monkeypatch):
    """The control: the loop must still close when there IS an observation."""
    substrate = _Substrate()
    monkeypatch.setattr(
        "core.brain.inference_feedback.get_runtime_service",
        lambda name, default=None: substrate if name == "liquid_substrate" else default,
    )
    projection = _Projection()
    result = _process(loop, projection)

    assert result["substrate_available"] is True
    assert result["coherence_grounded"] is True
    assert result["projection_trained"] is True
    assert len(projection.calls) == 1
    assert substrate.fed, "the substrate received its feedback"


# ------------------------------------------------------------- atomicity


def test_a_retried_generation_does_not_dose_the_substrate_twice(loop, monkeypatch):
    substrate = _Substrate()
    monkeypatch.setattr(
        "core.brain.inference_feedback.get_runtime_service",
        lambda name, default=None: substrate if name == "liquid_substrate" else default,
    )
    _process(loop, _Projection())
    second = _process(loop, _Projection())

    assert len(substrate.fed) == 1, "the retry applied the same feedback again"
    assert second["engines_fed"].get("skipped") == "already_applied"


def test_a_different_generation_is_still_applied(loop, monkeypatch):
    """The dedupe must key on content, not block everything after the first."""
    substrate = _Substrate()
    monkeypatch.setattr(
        "core.brain.inference_feedback.get_runtime_service",
        lambda name, default=None: substrate if name == "liquid_substrate" else default,
    )
    _process(loop, _Projection(), text="the repair completed and the system is stable")
    _process(loop, _Projection(), text="a different stable and healthy outcome entirely")
    assert len(substrate.fed) == 2


# --------------------------------------------------------- numeric safety


@pytest.mark.parametrize("bad", [[float("nan")], [float("inf")], [-0.5, float("nan")]])
def test_non_finite_logprobs_never_reach_the_engines(loop, monkeypatch, bad):
    """np.clip does not remove NaN — it propagates it."""
    substrate = _Substrate()
    monkeypatch.setattr(
        "core.brain.inference_feedback.get_runtime_service",
        lambda name, default=None: substrate if name == "liquid_substrate" else default,
    )
    result = _process(loop, _Projection(), logprobs=bad)
    assert math.isfinite(result["surprise"])
    assert "logprobs_not_finite" in result["degraded"]
    for call in substrate.fed:
        assert math.isfinite(call["surprise"])
        assert math.isfinite(call["coherence"])


def test_a_non_finite_arousal_cannot_explode_the_learning_rate(loop, monkeypatch):
    substrate = _Substrate()
    substrate.x[substrate.idx_arousal] = float("nan")
    monkeypatch.setattr(
        "core.brain.inference_feedback.get_runtime_service",
        lambda name, default=None: substrate if name == "liquid_substrate" else default,
    )
    projection = _Projection()
    _process(loop, projection)
    for call in projection.calls:
        assert math.isfinite(call["lr"])
        assert 0.0 < call["lr"] <= 0.004


def test_an_extreme_arousal_is_bounded(loop, monkeypatch):
    substrate = _Substrate(arousal=1e9)
    monkeypatch.setattr(
        "core.brain.inference_feedback.get_runtime_service",
        lambda name, default=None: substrate if name == "liquid_substrate" else default,
    )
    projection = _Projection()
    _process(loop, projection)
    assert all(call["lr"] <= 0.004 for call in projection.calls)


def test_a_negative_channel_index_is_refused_not_wrapped(loop, monkeypatch):
    """Only indices ABOVE the length were rejected; negatives read from the end."""
    substrate = _Substrate()
    substrate.idx_valence = -3
    monkeypatch.setattr(
        "core.brain.inference_feedback.get_runtime_service",
        lambda name, default=None: substrate if name == "liquid_substrate" else default,
    )
    result = _process(loop, _Projection())
    assert result["substrate_available"] is False


def test_a_wrong_sized_substrate_vector_is_refused(loop, monkeypatch):
    substrate = _Substrate(dim=64)
    monkeypatch.setattr(
        "core.brain.inference_feedback.get_runtime_service",
        lambda name, default=None: substrate if name == "liquid_substrate" else default,
    )
    result = _process(loop, _Projection())
    assert result["substrate_available"] is False
    assert any("substrate_dim_mismatch" in reason for reason in result["degraded"])


def test_a_nonsense_substrate_dim_is_refused_at_construction():
    for bad in (0, -5, 1.5, float("nan"), "512"):
        with pytest.raises(ValueError):
            InferenceFeedbackLoop(substrate_dim=bad)


# ------------------------------------------------------------ availability


def test_a_broken_substrate_degrades_instead_of_aborting_the_response(loop, monkeypatch):
    """Lock/vector access ran outside every handler and could abort finalisation."""

    class _Broken:
        @property
        def sync_lock(self):
            raise RuntimeError("lock is gone")

    monkeypatch.setattr(
        "core.brain.inference_feedback.get_runtime_service",
        lambda name, default=None: _Broken() if name == "liquid_substrate" else default,
    )
    result = _process(loop, _Projection())
    assert result["substrate_available"] is False
    assert any("substrate_read_failed" in reason for reason in result["degraded"])


def test_an_engine_failure_is_reported_per_engine(loop, monkeypatch):
    substrate = _Substrate()

    class _BadEngine:
        def accept_surprise_signal(self, value):
            raise RuntimeError("engine down")

    def resolve(name, default=None):
        if name == "liquid_substrate":
            return substrate
        if name == "free_energy_engine":
            return _BadEngine()
        return default

    monkeypatch.setattr("core.brain.inference_feedback.get_runtime_service", resolve)
    result = _process(loop, _Projection())
    assert result["engines_fed"]["free_energy_engine"].startswith("failed:")
    assert result["engines_fed"]["liquid_substrate"] == "applied", (
        "one engine failing must not stop the others"
    )


# -------------------------------------------------------------- honesty


def test_the_surprise_method_is_named_not_called_perplexity(loop, monkeypatch):
    substrate = _Substrate()
    monkeypatch.setattr(
        "core.brain.inference_feedback.get_runtime_service",
        lambda name, default=None: substrate if name == "liquid_substrate" else default,
    )
    result = _process(loop, _Projection())
    assert result["surprise_method"] == "mean_negative_logprob_clipped_0_3"


def test_the_lexical_fallback_actually_examines_punctuation(loop, monkeypatch):
    """The docstring promised punctuation volatility; the code ignored it."""
    substrate = _Substrate()
    monkeypatch.setattr(
        "core.brain.inference_feedback.get_runtime_service",
        lambda name, default=None: substrate if name == "liquid_substrate" else default,
    )
    plain = _process(loop, _Projection(), logprobs=None, text="alpha beta gamma delta")
    punctuated = _process(
        loop, _Projection(), logprobs=None, text="alpha, beta; gamma! delta?"
    )
    assert plain["surprise_method"] == "lexical_repetition_and_punctuation"
    assert punctuated["surprise"] > plain["surprise"]


def test_text_with_no_valence_words_is_not_reported_as_measured_neutral(loop, monkeypatch):
    substrate = _Substrate()
    monkeypatch.setattr(
        "core.cognitive.sentiment_tracker._score_with_apple_natural_language",
        lambda _chunk: (None, "", "native_unsupported"),
    )
    monkeypatch.setattr(
        "core.brain.inference_feedback.get_runtime_service",
        lambda name, default=None: substrate if name == "liquid_substrate" else default,
    )
    result = _process(loop, _Projection(), text="the capital of france is paris")
    assert result["output_valence_grounded"] is False
    assert result["coherence_grounded"] is False


def test_negated_semantic_valence_drives_coherence_in_the_correct_direction(
    loop, monkeypatch
):
    substrate = _Substrate(valence=0.8)
    monkeypatch.setattr(
        "core.brain.inference_feedback.get_runtime_service",
        lambda name, default=None: substrate if name == "liquid_substrate" else default,
    )

    result = _process(loop, _Projection(), text="This is not remotely safe.")

    assert result["output_valence_grounded"] is True
    assert result["output_valence"] < 0.0
    assert result["coherence_grounded"] is True
    assert result["coherence"] < 0.0
    evidence = result["sentiment_evidence"]
    assert evidence["method"] in {
        "semantic_context_consensus_v1",
        "contextual_lexicon_sentiment_v2",
    }


def test_the_modulation_that_caused_the_generation_is_recorded(loop, monkeypatch):
    """It was accepted as an argument and never read, so nothing could attribute."""
    substrate = _Substrate()
    monkeypatch.setattr(
        "core.brain.inference_feedback.get_runtime_service",
        lambda name, default=None: substrate if name == "liquid_substrate" else default,
    )
    result = _process(loop, _Projection())
    assert result["modulation"]["temperature"] == pytest.approx(0.8)
    assert result["modulation"]["top_p"] == pytest.approx(0.9)


def test_the_receipt_distinguishes_calculation_from_applied_feedback(loop, monkeypatch):
    substrate = _Substrate()
    monkeypatch.setattr(
        "core.brain.inference_feedback.get_runtime_service",
        lambda name, default=None: substrate if name == "liquid_substrate" else default,
    )
    result = _process(loop, _Projection())
    for key in (
        "generation_id",
        "token_count",
        "logprobs_available",
        "substrate_available",
        "engines_fed",
        "projection_trained",
        "surprise_method",
        "degraded",
    ):
        assert key in result, f"the receipt cannot answer: {key}"


def test_the_existing_caller_contract_still_holds(loop, monkeypatch):
    """unified_inference reads feedback['surprise'] and ['coherence'] as floats."""
    substrate = _Substrate()
    monkeypatch.setattr(
        "core.brain.inference_feedback.get_runtime_service",
        lambda name, default=None: substrate if name == "liquid_substrate" else default,
    )
    result = _process(loop, _Projection())
    assert isinstance(result["surprise"], float)
    assert isinstance(result["coherence"], float)
    assert f"{result['surprise']:.4f}"
