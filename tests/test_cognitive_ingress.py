"""Contract tests: typed cognitive ingress for latent allocation.

Allocation must come from the organs that know — memory, body, goals, Will,
affect, self-model, world model — with per-signal receipts, and degrade to
stable baselines (not crashes, not zeros) when organs are absent.
"""
from __future__ import annotations

import asyncio
import gc
import threading
import warnings

import pytest

from core.brain.cognitive_ingress import (
    COGNITIVE_INGRESS_SCHEMA,
    assemble_cognitive_ingress,
    assemble_cognitive_ingress_async,
)


@pytest.fixture()
def registry(monkeypatch):
    """A controllable runtime-service registry."""
    services: dict[str, object] = {}

    import core.brain.cognitive_ingress as ingress_mod

    monkeypatch.setattr(
        ingress_mod, "_get_service", lambda name: services.get(name)
    )
    return services


def test_cold_registry_yields_baselines_with_absent_receipts(registry):
    ingress = assemble_cognitive_ingress(None, "explain the scheduler")
    receipt = ingress.to_receipt()
    assert receipt["schema"] == COGNITIVE_INGRESS_SCHEMA
    assert 0.0 < ingress.stakes < 1.0 and 0.0 < ingress.uncertainty < 1.0
    assert ingress.stakes == pytest.approx(receipt["base_stakes"], abs=1e-9)
    # Every organ is reported absent, none silently skipped.
    assert set(receipt["absent_sources"]) >= {
        "memory", "goals", "will", "affect", "world_model"
    }


def test_memory_familiarity_lowers_uncertainty_and_blankness_raises_it(registry):
    class FamiliarMemory:
        def search(self, objective, limit=4):
            return ["hit"] * 4

    class BlankMemory:
        def search(self, objective, limit=4):
            return []

    registry["memory_facade"] = FamiliarMemory()
    familiar = assemble_cognitive_ingress(None, "the scheduler design")
    registry["memory_facade"] = BlankMemory()
    blank = assemble_cognitive_ingress(None, "the scheduler design")
    assert familiar.uncertainty < blank.uncertainty
    familiar_signal = next(s for s in familiar.signals if s.source == "memory")
    assert familiar_signal.present and familiar_signal.value == 1.0


def test_memory_ingress_prefers_explicit_sync_facade_contract(registry):
    class HybridMemory:
        sync_calls = 0
        async_calls = 0

        def search_sync(self, objective, limit=4):
            self.sync_calls += 1
            return [{"content": "verified synchronous memory", "verified": True}]

        async def search(self, objective, limit=4):
            self.async_calls += 1
            return [{"content": "async path must not run"}]

    memory = HybridMemory()
    registry["memory_facade"] = memory

    ingress = assemble_cognitive_ingress(None, "the scheduler design")

    signal = next(s for s in ingress.signals if s.source == "memory")
    assert signal.present is True
    assert signal.detail.startswith("memory_facade.search_sync:")
    assert memory.sync_calls == 1
    assert memory.async_calls == 0


def test_memory_ingress_closes_hidden_awaitable_without_warning(registry):
    class AwaitableOnlyMemory:
        def search(self, objective, limit=4):
            async def hidden_search():
                return [{"content": "must not cross the sync boundary"}]

            return hidden_search()

    registry["memory_facade"] = AwaitableOnlyMemory()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ingress = assemble_cognitive_ingress(None, "the scheduler design")
        gc.collect()

    signal = next(s for s in ingress.signals if s.source == "memory")
    assert signal.present is False
    assert not any("was never awaited" in str(item.message) for item in caught)


def test_memory_ingress_cancels_hidden_cancelable_awaitable(registry):
    class HiddenAwaitable:
        def __init__(self):
            self.cancelled = False

        def __await__(self):
            async def value():
                return []

            return value().__await__()

        def cancel(self):
            self.cancelled = True

    hidden = HiddenAwaitable()

    class AwaitableOnlyMemory:
        def search(self, objective, limit=4):
            return hidden

    registry["memory_facade"] = AwaitableOnlyMemory()
    ingress = assemble_cognitive_ingress(None, "the scheduler design")

    signal = next(s for s in ingress.signals if s.source == "memory")
    assert signal.present is False
    assert hidden.cancelled is True


@pytest.mark.asyncio
async def test_async_ingress_keeps_blocking_memory_search_off_event_loop(registry):
    started = threading.Event()
    release = threading.Event()

    class BlockingMemory:
        def search_sync(self, objective, limit=4):
            started.set()
            assert release.wait(timeout=2.0)
            return [{"content": "recalled after worker release", "verified": True}]

    registry["memory_facade"] = BlockingMemory()
    assembly = asyncio.create_task(
        assemble_cognitive_ingress_async(None, "the scheduler design")
    )
    assert await asyncio.to_thread(started.wait, 1.0)
    ticker = asyncio.create_task(asyncio.sleep(0.01))
    done, _ = await asyncio.wait(
        {assembly, ticker},
        timeout=0.25,
        return_when=asyncio.FIRST_COMPLETED,
    )
    assert ticker in done
    assert assembly not in done
    release.set()
    ingress = await assembly
    signal = next(s for s in ingress.signals if s.source == "memory")
    assert signal.present is True


def test_goal_overlap_raises_stakes(registry):
    class Goals:
        def active_goals(self):
            return ["Ship the scheduler arbitration redesign safely"]

    registry["goal_engine"] = Goals()
    on_goal = assemble_cognitive_ingress(
        None, "How should the scheduler arbitration redesign handle deadlines?"
    )
    off_goal = assemble_cognitive_ingress(None, "What rhymes with orange?")
    assert on_goal.stakes > off_goal.stakes
    signal = next(s for s in on_goal.signals if s.source == "goals")
    assert signal.present and signal.value > 0


def test_affect_uncertainty_and_will_preference_flow_through(registry):
    class Affect:
        felt_uncertainty = 0.8

    class Will:
        deliberation_preference = 0.9

    registry["affect_engine"] = Affect()
    registry["volition"] = Will()
    ingress = assemble_cognitive_ingress(None, "resolve the conflict")
    baseline = ingress.to_receipt()["base_uncertainty"]
    assert ingress.uncertainty > baseline
    will_signal = next(s for s in ingress.signals if s.source == "will")
    assert will_signal.present and will_signal.stakes_delta > 0


def test_self_model_terms_raise_stakes(registry):
    identity = assemble_cognitive_ingress(
        None, "Should you change your own governance weights, Aura?"
    )
    mundane = assemble_cognitive_ingress(None, "Summarize the quarterly report")
    assert identity.stakes > mundane.stakes
    signal = next(s for s in identity.signals if s.source == "self_model")
    assert signal.present


def test_broken_organ_is_absent_not_fatal(registry):
    class ExplodingMemory:
        def search(self, objective, limit=4):
            raise RuntimeError("index corrupted")

    registry["memory_facade"] = ExplodingMemory()
    ingress = assemble_cognitive_ingress(None, "anything")
    signal = next(s for s in ingress.signals if s.source == "memory")
    assert signal.present is False


def test_receipt_proves_which_sources_moved_the_allocation(registry):
    class Goals:
        def active_goals(self):
            return ["Verify the latent cortex consolidation pipeline"]

    registry["goal_engine"] = Goals()
    ingress = assemble_cognitive_ingress(
        None, "Verify the latent cortex consolidation pipeline end to end"
    )
    receipt = ingress.to_receipt()
    moved = [s for s in receipt["signals"] if s["present"] and s["stakes_delta"] > 0]
    assert any(s["source"] == "goals" for s in moved)
    assert all("detail" in s for s in receipt["signals"])


def test_body_pressure_vector_names_strain_and_weights_anticipation(registry):
    """The body speaks as a vector: strained channels are named, forecast
    pressure weighs more per unit than current pressure, and the property-
    shaped total_pressure on the REAL BodyState is read correctly."""
    from core.brain import cognitive_ingress as ingress_mod

    class FakeBody:
        @property
        def total_pressure(self):
            return 0.5

        def pressure_vector(self):
            return {
                "cpu_pressure": 0.05,
                "memory_pressure": 0.71,
                "thermal_pressure": 0.44,
                "anticipatory_pressure": 0.40,
            }

    class FakeBodyState:
        @classmethod
        def from_aura_state(cls, state, **kwargs):
            return FakeBody()

    import core.being.aura_now as aura_now_mod

    from types import SimpleNamespace

    original = aura_now_mod.BodyState
    aura_now_mod.BodyState = FakeBodyState
    try:
        signal = ingress_mod._signal_body(SimpleNamespace(state=None))
    finally:
        aura_now_mod.BodyState = original
    assert ingress_mod._signal_body(None).present is False
    assert signal.present and signal.value == 0.5
    assert signal.stakes_delta == pytest.approx(0.10 * 0.5 + 0.05 * 0.40)
    assert "memory 0.71" in signal.context_text
    assert "thermal 0.44" in signal.context_text
    assert "cpu" not in signal.context_text  # below the strain threshold
    assert "forecast pressure 0.40" in signal.context_text
    assert "anticipatory=0.400" in signal.detail


def test_affect_distress_raises_stakes_and_renders_felt_quality(registry):
    class VadState:
        valence = -0.6
        arousal = 0.9
        label = "distressed"

    class Affect:
        felt_uncertainty = 0.7
        state = VadState()

    registry["affect_engine"] = Affect()
    ingress = assemble_cognitive_ingress(None, "decide under pressure")
    signal = next(s for s in ingress.signals if s.source == "affect")
    assert signal.present
    assert signal.value == pytest.approx(0.7)  # doubt keeps value semantics
    assert signal.stakes_delta == pytest.approx(
        min(0.13, 0.08 * 0.6 + 0.05 * (0.9 - 0.6) / 0.4)
    )
    assert signal.uncertainty_delta == pytest.approx(0.15 * 0.7)
    assert "valence -0.60" in signal.context_text
    assert "(distressed)" in signal.context_text


def test_affect_calm_positive_state_adds_no_stakes(registry):
    class VadState:
        valence = 0.5
        arousal = 0.3

    class Affect:
        felt_uncertainty = 0.1
        state = VadState()

    registry["affect_engine"] = Affect()
    ingress = assemble_cognitive_ingress(None, "an ordinary question")
    signal = next(s for s in ingress.signals if s.source == "affect")
    assert signal.present
    assert signal.stakes_delta == 0.0
    assert signal.context_text == ""  # nothing pronounced to seed


def test_self_model_semantic_match_without_keywords(registry):
    """An identity-relevant objective with ZERO keyword hits still raises
    stakes when the canonical-self embedding similarity is high."""

    class CanonicalSelf:
        def get_context_block(self):
            return "I value honesty, continuity of my commitments, and care."

    class Vectors:
        def embed(self, text):
            import numpy as np

            # Orthogonal-ish unless the text mentions commitments/honesty.
            base = np.zeros(4)
            lowered = text.lower()
            if "honesty" in lowered or "commitment" in lowered:
                base[0] = 1.0
            if "quarterly" in lowered:
                base[1] = 1.0
            base[2] = 0.1
            return base

    registry["canonical_self"] = CanonicalSelf()
    registry["vector_memory"] = Vectors()
    identity = assemble_cognitive_ingress(
        None, "Would abandoning a commitment for speed be acceptable?"
    )
    mundane = assemble_cognitive_ingress(None, "Summarize the quarterly report")
    identity_signal = next(
        s for s in identity.signals if s.source == "self_model"
    )
    mundane_signal = next(s for s in mundane.signals if s.source == "self_model")
    assert identity_signal.present
    assert identity_signal.stakes_delta > mundane_signal.stakes_delta
    assert "embedding_cosine_vs_canonical_self" in identity_signal.detail
    assert "matched my canonical self" in identity_signal.context_text


def test_interoceptive_context_reaches_slot_items(registry):
    from core.brain.cognitive_ingress import (
        CognitiveIngress,
        IngressSignal,
        cognitive_context_items,
    )

    ingress = CognitiveIngress(
        stakes=0.8,
        uncertainty=0.6,
        signals=[
            IngressSignal(
                source="body",
                present=True,
                value=0.55,
                context_text="My body: strained now: memory 0.71",
            ),
            IngressSignal(
                source="affect",
                present=True,
                value=0.7,
                context_text="How this feels right now: valence -0.60",
            ),
        ],
    )
    items = cognitive_context_items(ingress)
    assert [item["source"] for item in items] == ["interoception"]
    text = items[0]["text"]
    assert "strained now: memory 0.71" in text
    assert "valence -0.60" in text
    assert "Current felt state:" in text
    assert len(text) <= 400
