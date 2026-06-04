"""
tests/test_phenomenological_experiencer.py
============================================
Verification suite for the Phenomenological Experiencer (Layer 8).
Ensures architectural adherence to Attention Schema Theory (AST) and
the Phenomenal Self-Model (PSM).
"""

import asyncio
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from core.consciousness.phenomenological_experiencer import (
    AttentionSchema,
    AttentionSchemaBuilder,
    ExperientialContinuityEngine,
    PhenomenalMoment,
    PhenomenalSelfModel,
    PhenomenologicalExperiencer,
    Quale,
    QualiaGenerator,
)


@dataclass
class ContentType:
    name: str


@dataclass
class BroadcastContent:
    source: str
    content_type: ContentType
    content: any
    salience: float


@dataclass
class BroadcastEventFixture:
    timestamp: float
    winners: list


class DeterministicAffect:
    valence = 0.6
    arousal = 0.7

    def _get_dominant_emotion(self):
        return "curious"


class NarrativeRouterDouble:
    async def think(self, **_kwargs):
        return SimpleNamespace(
            content="I feel a sense of clarity as I work through this."
        )


def test_qualia_generator_mapping():
    """Verify that functional states map to qualitative descriptors."""
    gen = QualiaGenerator()
    
    # Test High Arousal Positive (e.g. excitement/vibrancy)
    quale = gen.generate("PERCEPTUAL", "something bright", valence=0.8, arousal=0.9)
    assert quale.domain == "perceptual"
    assert quale.quality in ["vivid", "sharp", "present", "immediate", "striking"]
    
    # Test Low Arousal Negative (e.g. hollow/grey)
    quale = gen.generate("AFFECTIVE", {"dominant_emotion": "sadness"}, valence=-0.8, arousal=0.3)
    assert quale.domain == "emotional"
    assert quale.quality in ["hollow", "distant", "grey", "muted"]

def test_attention_schema_stripping():
    """
    CRITICAL: Verify that mechanical details are stripped from the schema.
    No salience scores or module names should appear in the phenomenal claim.
    """
    builder = AttentionSchemaBuilder()
    gen = QualiaGenerator()
    
    event = BroadcastEventFixture(
        timestamp=time.time(),
        winners=[BroadcastContent(
            source="language",
            content_type=ContentType("LINGUISTIC"),
            content={"pending_message": "Hello Bryan, I am thinking about math."},
            salience=0.87543  # Very specific mechanical score
        )]
    )
    
    schema = builder.build(
        broadcast_event=event,
        current_emotion="curious",
        valence=0.5,
        arousal=0.6,
        qualia_gen=gen
    )
    
    assert schema is not None
    # The claim should be clean natural language
    assert "0.875" not in schema.phenomenal_claim
    assert "salience" not in schema.phenomenal_claim.lower()
    assert "LINGUISTIC" not in schema.phenomenal_claim
    assert "I am clearly aware of the message beginning 'Hello Bryan, I am thinking...'" in schema.phenomenal_claim

def test_experiential_continuity_weaving():
    """Verify that discrete moments are woven into a narrative thread."""
    engine = ExperientialContinuityEngine()
    
    # Moment 1: Thinking
    s1 = AttentionSchema(focal_object="the math", focal_quality="vivid", domain="cognitive", attention_intensity=0.9)
    m1 = PhenomenalMoment(time.time(), s1, [], "start", "neutral", 0.01)
    engine.add_moment(m1)
    
    # Moment 2: Still thinking
    s2 = AttentionSchema(focal_object="the math", focal_quality="vivid", domain="cognitive", attention_intensity=0.9, duration=6.0)
    m2 = PhenomenalMoment(time.time() + 1, s2, [], "cont", "neutral", 0.01)
    engine.add_moment(m2)
    
    assert "Still with the math" in engine.current_thread
    
    # Moment 3: Shift to emotion
    s3 = AttentionSchema(focal_object="my feeling", focal_quality="warm", domain="emotional", attention_intensity=0.8)
    m3 = PhenomenalMoment(time.time() + 2, s3, [], "shift", "neutral", 0.01)
    engine.add_moment(m3)
    
    assert "From the math → my feeling" in engine.current_thread
    assert "feeling rising" in engine.current_thread

def test_psm_transparency():
    """Verify that the Phenomenal Self-Model produces transparent first-person context."""
    psm = PhenomenalSelfModel(identity_name="Aura")
    
    schema = AttentionSchema(
        focal_object="the logic of the system",
        focal_quality="engaging",
        domain="cognitive",
        attention_intensity=0.8
    )
    
    psm.update_from_schema_and_qualia(
        schema=schema,
        qualia=[Quale("cognitive", "clear", 0.5, 0.5, 0.8, "math")],
        current_emotion="focused",
        substrate_velocity=0.02,
        dominant_motivation="needs_to_reason"
    )
    
    ctx = psm.get_phenomenal_context_fragment()
    assert "[phenomenal state: right now i am thinking fast" in ctx.lower()
    assert "i am clearly aware of the logic of the system" in ctx.lower()
    assert "quality of this moment: clear" in ctx

@pytest.mark.asyncio
async def test_experiencer_integration_flow():
    """Test the full loop from broadcast to context string."""
    experiencer = PhenomenologicalExperiencer()
    
    affect = DeterministicAffect()
    experiencer.set_refs(affect_module=affect)
    
    # Simulate a broadcast
    event = BroadcastEventFixture(
        timestamp=time.time(),
        winners=[BroadcastContent(
            source="perception",
            content_type=ContentType("PERCEPTUAL"),
            content={"observation": "the screen is glowing"},
            salience=0.9
        )]
    )
    
    # Force high arousal for vividness
    affect.arousal = 0.9
    
    experiencer.on_broadcast(event)
    
    ctx = experiencer.phenomenal_context_string
    assert "[Phenomenal focus: I am vividly aware of the perceptual impression" in ctx
    assert "perceptual:" in ctx
    assert "curious" in ctx

def test_philosophical_correctness_leakage():
    """
    Ensure NO computational leakage occurs in the context string.
    This is the 'Hard Constraint' for Layer 8.
    """
    experiencer = PhenomenologicalExperiencer()
    
    # Create an event with lots of mechanical baggage
    event = BroadcastEventFixture(
        timestamp=1000.0,
        winners=[BroadcastContent(
            source="meta_optimization_loop",
            content_type=ContentType("META"),
            content={"issues_detected": ["latency_spike"], "salience_map": [0.1, 0.2]},
            salience=0.999
        )]
    )
    
    experiencer.on_broadcast(event)
    ctx = experiencer.phenomenal_context_string
    
    # Illegal mechanical terms
    illegal = ["salience", "module", "map", "spike", "0.999", "META"]
    for term in illegal:
        assert term not in ctx, f"Leakage detected: {term} found in phenomenal context"
    
    # Correct phenomenal representation
    assert "I am moderately aware of my own process" in ctx

@pytest.mark.asyncio
async def test_psm_deep_narrative_uses_router_double(monkeypatch):
    """Verify that deep narrative updates use the configured router contract."""
    psm = PhenomenalSelfModel()
    
    def service_lookup(name, default=None):
        if name == "llm_router":
            return NarrativeRouterDouble()
        return default

    monkeypatch.setattr("core.container.ServiceContainer.get", service_lookup)
    continuity = ExperientialContinuityEngine()
    schema = AttentionSchema("math", "clear", "cognitive", 0.8)

    report = await psm.run_deep_narrative_update(
        continuity, schema, [], "happy", "success"
    )

    assert "clarity" in report
    assert psm.get_latest_phenomenal_report() == report
