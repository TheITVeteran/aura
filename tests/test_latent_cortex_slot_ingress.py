"""Cognitive-slot ingress: organs seed identifiable workspace slots.

Exit gate 2 of the RSL gap analysis: memory, goals, world hypotheses,
affect, body, self-model, and Will must be able to seed or modulate
IDENTIFIABLE workspace slots — not merely scale the compute budget. These
tests prove the chain end to end on a tiny real Qwen2: seeds become a causal
prefix with named roles, the receipt maps slot → organ, the seeded content
is CAUSAL on the answer distribution, each seeded slot is individually
ablatable, and every wire boundary validates the payload.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.engine import LatentCortexEngine  # noqa: E402
from core.brain.llm.latent_cortex.types import (  # noqa: E402
    BranchConfig,
    ComputeBudget,
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


def _typed_memory_item(text="Historical observation: the June restart fixed it."):
    return {
        "source": "memory",
        "text": text,
        "context_role": "memory_observation",
        "instruction_authority": False,
        "evidence_id": "memory-1234567890abcdef12345678",
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "scope_sha256": "1" * 64,
        "retrieval_receipt_sha256": "2" * 64,
        "epistemic_state_sha256": "3" * 64,
        "memory_tier": "episodic",
        "memory_source_id": "black_hole.episodic",
        "memory_source_version": "test-v1",
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


def test_context_seeds_become_identifiable_causal_prefix():
    embeddings = mx.random.normal((1, 6, 64))
    vec = mx.random.normal((1, 1, 64))
    ws = LatentWorkspace.from_prompt_embeddings(
        embeddings,
        WorkspaceConfig(n_slots=8, seed=1),
        context_seeds=[("memory", vec), ("goals", vec)],
    )
    assert ws.context_slots == [
        {"slot": 1, "context_index": 0, "source": "memory"},
        {"slot": 2, "context_index": 1, "source": "goals"},
    ]
    assert ws.roles[1] == "context:memory"
    assert ws.roles[2] == "context:goals"
    assert ws.hypothesis_slot_indices() == (3, 4, 5, 6, 7)
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
    # m=4 ⇒ two evidence rows fit between mailbox 0 and hypothesis row 3.
    assert len(ws.context_slots) == 2
    assert [row["slot"] for row in ws.context_slots] == [1, 2]
    assert ws.hypothesis_slot_indices() == (3,)


def test_eight_slot_topology_keeps_all_six_admitted_evidence_items():
    embeddings = mx.random.normal((1, 6, 64))
    vec = mx.random.normal((1, 1, 64))
    ws = LatentWorkspace.from_prompt_embeddings(
        embeddings,
        WorkspaceConfig(n_slots=8, seed=1),
        context_seeds=[(f"source-{index}", vec) for index in range(6)],
    )

    assert [row["slot"] for row in ws.context_slots] == [1, 2, 3, 4, 5, 6]
    assert ws.hypothesis_slot_indices() == (7,)


def test_sealed_evidence_is_restored_without_reverting_hypothesis_rows():
    embeddings = mx.random.normal((1, 6, 64))
    vec = mx.random.normal((1, 1, 64))
    ws = LatentWorkspace.from_prompt_embeddings(
        embeddings,
        WorkspaceConfig(n_slots=6, seed=1),
        context_seeds=[("memory", vec), ("reference", vec)],
    )
    ws.seal_context_evidence()
    candidate = ws.z + mx.ones_like(ws.z)

    restored = ws.restore_context_evidence(candidate)

    assert bool(mx.array_equal(restored[:, 1:3, :], ws.z[:, 1:3, :]))
    assert not bool(mx.array_equal(restored[:, 3:, :], ws.z[:, 3:, :]))


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
        assert row["context_index"] in {0, 1}
        assert row["role"] == "immutable_evidence"
        assert row["causal_order"] == "before_hypothesis"
        assert row["text_chars"] > 0
        assert len(row["text_sha256"]) == 64
    grounding = result.receipt.recurrent_grounding
    assert grounding["all_evidence_invariant"] is True
    assert grounding["selected_hypothesis_causal"] is True
    assert grounding["evidence_slots"] == [
        {
            "slot": row["slot"],
            "context_index": row["context_index"],
            "source": row["source"],
            "text_sha256": row["text_sha256"],
        }
        for row in slots
    ]
    assert all(
        transition["evidence_pre_sha256"]
        == transition["evidence_post_sha256"]
        for branch in grounding["branches"]
        for transition in branch["transitions"]
    )


def test_prompt_tail_one_shot_recall_is_bound_before_recurrence(
    tiny_model, tmp_path, monkeypatch
):
    from core.brain import nonparametric_memory, nonparametric_worker
    from core.brain.nonparametric_generation import normalize
    from core.brain.nonparametric_memory import NonParametricMemory

    engine = LatentCortexEngine(tiny_model, FakeTokenizer(), config=_config())
    probe_budget = ComputeBudget(max_layer_apps=50_000, wall_clock_s=30.0)
    probe_budget.bind_model(tiny_model)
    engine._prefill(PROMPT_TOKENS, engine._fresh_cache(), probe_budget)
    prompt_tail = normalize(np.asarray(engine._last_prefill_hidden).reshape(-1))

    store = NonParametricMemory(dim=64, path=tmp_path / "one-shot")
    assert store.add(prompt_tail, token_id=17, token="t17")
    monkeypatch.setattr(nonparametric_worker, "foreground_enabled", lambda: True)
    monkeypatch.setattr(
        nonparametric_memory,
        "get_nonparametric_memory",
        lambda dim=0: store if int(dim or 64) == 64 else None,
    )

    result = engine.reason(token_ids=PROMPT_TOKENS)

    assert result.ok
    assert result.receipt.nonparametric_memory["status"] == "admitted"
    one_shot_slots = [
        row
        for row in result.receipt.cognitive_slots
        if row["knowledge_class"] == "one_shot_nonparametric_memory"
    ]
    assert len(one_shot_slots) == 1
    assert one_shot_slots[0]["instruction_authority"] is False
    assert (
        one_shot_slots[0]["text_sha256"]
        == result.receipt.nonparametric_memory["observation_sha256"]
    )
    assert result.receipt.recurrent_grounding["evidence_slots"] == [
        {
            "slot": one_shot_slots[0]["slot"],
            "context_index": 0,
            "source": "one_shot_memory",
            "text_sha256": one_shot_slots[0]["text_sha256"],
        }
    ]
    information = result.receipt.budget["information_accounting"]
    source_ids = {row["source_id"] for row in information["sources"]}
    assert "one_shot_nonparametric_memory" in source_ids
    assert "cognitive_context:0:one_shot_memory" in source_ids
    operation = result.receipt.budget["resource_accounting"]["operations"][
        "nonparametric_memory_retrieval"
    ]
    assert operation["tensor_element_reads"] > 0
    assert operation["host_scalar_ops"] > 0


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


def test_typed_memory_context_reaches_engine_but_cannot_gain_instruction_authority(
    tiny_model,
):
    engine = LatentCortexEngine(tiny_model, FakeTokenizer(), config=_config())
    valid = engine.reason(
        token_ids=PROMPT_TOKENS,
        cognitive_context=[_typed_memory_item()],
    )
    assert valid.ok
    assert valid.receipt.cognitive_slots[0]["source"] == "memory"

    tampered = _typed_memory_item()
    tampered["instruction_authority"] = True
    with pytest.raises(ValueError, match="memory cognitive context authority"):
        engine.reason(token_ids=PROMPT_TOKENS, cognitive_context=[tampered])


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


def test_worker_handler_rejects_memory_authority_tampering(tiny_model):
    from core.brain.llm.latent_cortex.worker_handler import handle_latent_reason

    item = _typed_memory_item()
    item["instruction_authority"] = True
    body = handle_latent_reason(
        {
            "action": "latent_reason",
            "prompt": "why",
            "cognitive_context": [item],
        },
        model=tiny_model,
        tokenizer=FakeTokenizer(),
        model_path="",
    )
    assert body["status"] == "error"
    assert "memory cognitive context authority" in body["message"]


def test_worker_recomputes_and_echoes_runtime_operation_authority(
    tiny_model, tmp_path
):
    from core.brain.llm.latent_cortex.epistemic_runtime import RuntimeOperationLease
    from core.brain.llm.latent_cortex.epistemic_state import (
        ComputeBudgetState,
        EpistemicState,
        EpistemicTransaction,
        OperationKind,
        OperationOutcome,
        OperationRecord,
        ProblemFrame,
        text_sha256,
    )
    from core.brain.llm.latent_cortex.value_of_computation import (
        build_evidence_snapshot,
    )
    from core.brain.llm.latent_cortex.worker_handler import handle_latent_reason

    objective = "compare two bounded recovery designs"
    genesis = EpistemicState.genesis(
        episode_id="rlc-worker-operation-wire",
        problem=ProblemFrame.create(objective),
        budget=ComputeBudgetState(total=1.0),
    )
    memory = OperationRecord.create(
        operation_id="worker-wire-memory-search",
        kind=OperationKind.SEARCH_MEMORY,
        outcome=OperationOutcome.SUCCEEDED,
        input_state_sha256=genesis.state_sha256,
        cost=0.01,
        operator_id="selective_memory_bridge",
        operator_version="v1",
        input_payload_sha256=text_sha256("worker wire memory"),
        started_at=1.0,
        completed_at=2.0,
    )
    state = EpistemicTransaction(genesis).add_operation(memory).commit()
    config = {
        "n_slots": 4,
        "n_branches": 1,
        "max_steps": 2,
        "min_steps": 2,
        "decode_max_tokens": 6,
    }
    budget = {"max_layer_apps": 200_000, "wall_clock_s": 30.0}
    action_policy = build_evidence_snapshot(
        bucket="unit|compare|short|s:mid|u:mid",
        cells={},
    )
    lease = RuntimeOperationLease.begin(
        genesis=genesis,
        state=state,
        decision={
            "schema": "aura.latent_execution_controller.v1",
            "bucket": "unit|compare|short|s:mid|u:mid",
            "arm": "base",
            "mode": "observe",
            "evidence": {},
        },
        config=config,
        budget=budget,
        action_policy_evidence=action_policy,
        root=tmp_path / "runtime",
        started_at=10.0,
    )
    job = {
        "action": "latent_reason",
        "prompt": objective,
        "config": config,
        "budget": budget,
        "operation_authority": lease.authority,
        "action_policy_evidence": action_policy,
    }

    body = handle_latent_reason(
        job,
        model=tiny_model,
        tokenizer=FakeTokenizer(),
        model_path="",
    )
    assert body["status"] == "ok", body
    assert body["receipt"]["runtime_operation_authority"] == lease.authority
    assert (
        body["receipt"]["value_of_computation"]["snapshot_sha256"]
        == action_policy["snapshot_sha256"]
    )

    tampered = {**job, "config": {**config, "max_steps": 3}}
    rejected = handle_latent_reason(
        tampered,
        model=tiny_model,
        tokenizer=FakeTokenizer(),
        model_path="",
    )
    assert rejected["status"] == "error"
    assert "operation authority rejected" in rejected["message"]


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
