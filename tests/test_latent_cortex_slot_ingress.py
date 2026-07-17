"""Cognitive-slot ingress: organs seed identifiable workspace slots.

Exit gate 2 of the RSL gap analysis: memory, goals, world hypotheses,
affect, body, self-model, and Will must be able to seed or modulate
IDENTIFIABLE workspace slots — not merely scale the compute budget. These
tests prove the chain end to end on a tiny real Qwen2: seeds become tail
slots with named roles, the receipt maps slot → organ, the seeded content
is CAUSAL on the answer distribution, each seeded slot is individually
ablatable, and every wire boundary validates the payload.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.engine import LatentCortexEngine  # noqa: E402
from core.brain.llm.latent_cortex.types import (  # noqa: E402
    BranchConfig,
    CortexConfig,
    RecurrenceConfig,
    WorkspaceConfig,
)
from core.brain.llm.latent_cortex.workspace import LatentWorkspace  # noqa: E402

N_LAYERS = 8
PROMPT_TOKENS = [5, 9, 17, 3, 42, 7, 11, 23, 2, 88]

MEMORY_ITEM = {
    "source": "memory",
    "text": "Recalled: the same failure appeared after the June restart.",
}
GOAL_ITEM = {
    "source": "goals",
    "text": "Active goal: keep the runtime stable through the demo.",
}


class FakeTokenizer:
    eos_token_id = None

    def encode(self, text, add_special_tokens=False):
        return [(ord(ch) % 96) + 8 for ch in str(text)][:48]

    def decode(self, tokens):
        return " ".join(f"t{int(t)}" for t in tokens)


def _model():
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=64,
        num_hidden_layers=N_LAYERS,
        intermediate_size=128,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def tiny_model():
    return _model()


def _config(n_slots=8, **overrides) -> CortexConfig:
    base = dict(
        workspace=WorkspaceConfig(n_slots=n_slots, seed=3),
        recurrence=RecurrenceConfig(max_steps=4, min_steps=2),
        branches=BranchConfig(n_branches=2, exchange_interval=2),
        prelude_frac=0.25,
        coda_frac=0.25,
        decode_max_tokens=6,
    )
    base.update(overrides)
    return CortexConfig(**base)


# ── Workspace-level seeding ──────────────────────────────────────────────


def test_context_seeds_become_identifiable_tail_slots():
    embeddings = mx.random.normal((1, 6, 64))
    vec = mx.random.normal((1, 1, 64))
    ws = LatentWorkspace.from_prompt_embeddings(
        embeddings,
        WorkspaceConfig(n_slots=8, seed=1),
        context_seeds=[("memory", vec), ("goals", vec)],
    )
    assert ws.context_slots == [
        {"slot": 6, "source": "goals"},
        {"slot": 7, "source": "memory"},
    ]
    assert ws.roles[7] == "context:memory"
    assert ws.roles[6] == "context:goals"
    # Comm slot 0 keeps its ordinary role.
    assert not ws.roles[0].startswith("context:")


def test_context_slot_cap_preserves_thought_slots():
    embeddings = mx.random.normal((1, 6, 64))
    vec = mx.random.normal((1, 1, 64))
    ws = LatentWorkspace.from_prompt_embeddings(
        embeddings,
        WorkspaceConfig(n_slots=4, seed=1),
        context_seeds=[("memory", vec), ("goals", vec), ("world_model", vec)],
    )
    # m=4 ⇒ at most 1 context slot (comm slot + free thought slots preserved).
    assert len(ws.context_slots) == 1
    assert ws.context_slots[0]["slot"] == 3


# ── Engine-level: receipted and CAUSAL ───────────────────────────────────


def test_receipt_maps_slots_to_organs(tiny_model):
    engine = LatentCortexEngine(tiny_model, FakeTokenizer(), config=_config())
    result = engine.reason(
        token_ids=PROMPT_TOKENS,
        cognitive_context=[MEMORY_ITEM, GOAL_ITEM],
    )
    assert result.ok
    slots = result.receipt.cognitive_slots
    assert {row["source"] for row in slots} == {"memory", "goals"}
    for row in slots:
        assert 0 < row["slot"] < 8
        assert row["text_chars"] > 0
        assert len(row["text_sha256"]) == 64


def test_context_seeding_is_causal_on_answer_distribution(tiny_model):
    plain = LatentCortexEngine(tiny_model, FakeTokenizer(), config=_config())
    seeded = LatentCortexEngine(tiny_model, FakeTokenizer(), config=_config())
    without = plain.reason(token_ids=PROMPT_TOKENS)
    with_ctx = seeded.reason(
        token_ids=PROMPT_TOKENS,
        cognitive_context=[MEMORY_ITEM],
    )
    assert without.ok and with_ctx.ok
    assert (
        without.receipt.first_logits_digest != with_ctx.receipt.first_logits_digest
    ), "organ-seeded slots must reach the answer distribution"


def test_seeded_slot_is_individually_ablatable(tiny_model):
    engine = LatentCortexEngine(tiny_model, FakeTokenizer(), config=_config())
    intact = engine.reason(
        token_ids=PROMPT_TOKENS, cognitive_context=[MEMORY_ITEM]
    )
    assert intact.ok
    slot = intact.receipt.cognitive_slots[0]["slot"]
    ablated = engine.reason(
        token_ids=PROMPT_TOKENS,
        cognitive_context=[MEMORY_ITEM],
        ablate_slot=slot,
    )
    assert ablated.ok
    assert any(
        flag.startswith(f"slot_ablated:{slot}") for flag in ablated.receipt.honest_flags
    )
    assert (
        intact.receipt.first_logits_digest != ablated.receipt.first_logits_digest
    ), "destroying the organ slot must change the answer distribution"


def test_malformed_context_is_rejected(tiny_model):
    engine = LatentCortexEngine(tiny_model, FakeTokenizer(), config=_config())
    with pytest.raises(ValueError):
        engine.reason(token_ids=PROMPT_TOKENS, cognitive_context="not a list")
    with pytest.raises(ValueError):
        engine.reason(
            token_ids=PROMPT_TOKENS, cognitive_context=[{"source": "", "text": "x"}]
        )


# ── Wire boundaries ──────────────────────────────────────────────────────


def test_worker_handler_rejects_malformed_context(tiny_model):
    from core.brain.llm.latent_cortex.worker_handler import handle_latent_reason

    body = handle_latent_reason(
        {
            "action": "latent_reason",
            "prompt": "why",
            "cognitive_context": [{"source": "memory", "text": "x" * 500}],
        },
        model=tiny_model,
        tokenizer=FakeTokenizer(),
        model_path="",
    )
    assert body["status"] == "error"
    assert "cognitive_context" in body["message"]


@pytest.mark.asyncio
async def test_service_validates_cognitive_context():
    from core.brain.latent_cortex_service import LatentCortexService

    service = LatentCortexService()
    result = await service.deep_reason(
        "why is the sky blue?",
        cognitive_context=[{"source": "memory"}],  # missing text
    )
    assert result["ok"] is False
    assert result["reason"] == "invalid_cognitive_context"


# ── Ingress items builder ────────────────────────────────────────────────


def test_cognitive_context_items_render_organ_content():
    from core.brain.cognitive_ingress import (
        CognitiveIngress,
        IngressSignal,
        cognitive_context_items,
    )

    ingress = CognitiveIngress(
        stakes=0.7,
        uncertainty=0.6,
        signals=[
            IngressSignal(
                source="memory",
                present=True,
                value=0.5,
                context_text="Recalled: prior restart fixed it.",
            ),
            IngressSignal(
                source="goals",
                present=True,
                value=0.4,
                context_text="Active goal: ship the demo.",
            ),
            IngressSignal(source="body", present=True, value=0.42),
            IngressSignal(source="will", present=True, value=0.80),
            IngressSignal(source="affect", present=True, value=0.30),
            IngressSignal(source="world_model", present=False),
        ],
    )
    items = cognitive_context_items(ingress)
    sources = [item["source"] for item in items]
    assert sources == ["memory", "goals", "interoception"]
    interoception = items[-1]["text"]
    assert "body pressure 0.42" in interoception
    assert "deliberation preference 0.80" in interoception
    assert "felt uncertainty 0.30" in interoception
    assert all(len(item["text"]) <= 400 for item in items)
