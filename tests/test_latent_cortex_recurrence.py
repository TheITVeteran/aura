"""Contract tests: latent workspace + controlled recurrence.

These run REAL mlx_lm Qwen2 forward passes on a tiny random-weight model —
they prove the mechanics (KV rewind, RoPE stability, RMSMatch bounds,
fixed-point convergence, slot causality) hold on the genuine architecture,
which is exactly what transfers to the resident 32B. Capability claims are
NOT tested here; see the experiments harness.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen2 import Model, ModelArgs

from core.brain.llm.latent_cortex.recurrence import (
    HaltingController,
    WindowRunner,
    alpha_at,
    recurrence_step,
    relative_residual,
    rms_match,
)
from core.brain.llm.latent_cortex.types import ComputeBudget, RecurrenceConfig, WorkspaceConfig
from core.brain.llm.latent_cortex.workspace import LatentWorkspace, per_position_rms


N_LAYERS, P_END, C_START = 8, 2, 6


@pytest.fixture(scope="module")
def tiny_model():
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


def _prefill(model, prompt):
    from mlx_lm.models.base import create_attention_mask

    inner = model.model
    cache = [KVCache() for _ in inner.layers]
    h = inner.embed_tokens(prompt)
    mask = create_attention_mask(h, cache)
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask, cache[i])
    mx.eval(h)
    return cache


PROMPT = [[5, 9, 17, 3, 42, 7, 11, 23, 2, 88]]


def _seed_workspace(model, cache, n_slots=4, budget=None, branch_role=None):
    inner = model.model
    prompt = mx.array(PROMPT)
    emb = inner.embed_tokens(prompt)
    ws = LatentWorkspace.from_prompt_embeddings(
        emb, WorkspaceConfig(n_slots=n_slots, seed=3), branch_role=branch_role
    )
    runner = WindowRunner(inner, budget or ComputeBudget())
    ws.update(runner.run(ws.z, cache, 0, P_END, persist=True))
    return ws, runner


# ── Workspace ───────────────────────────────────────────────────────────


def test_workspace_seed_is_rms_matched_and_deterministic(tiny_model):
    inner = tiny_model.model
    emb = inner.embed_tokens(mx.array(PROMPT))
    cfg = WorkspaceConfig(n_slots=4, seed=3)
    ws1 = LatentWorkspace.from_prompt_embeddings(emb, cfg)
    ws2 = LatentWorkspace.from_prompt_embeddings(emb, cfg)
    assert ws1.z.shape == (1, 4, 64)
    assert bool(mx.allclose(ws1.z, ws2.z)), "workspace seeding must be deterministic"
    emb_rms = float(mx.mean(per_position_rms(emb)))
    ws_rms = float(mx.mean(per_position_rms(ws1.z)))
    assert abs(ws_rms - emb_rms) / emb_rms < 0.05, "seeds must be RMS-matched to embeddings"


def test_branch_roles_give_distinct_seeds(tiny_model):
    inner = tiny_model.model
    emb = inner.embed_tokens(mx.array(PROMPT))
    cfg = WorkspaceConfig(n_slots=4, seed=3)
    a = LatentWorkspace.from_prompt_embeddings(emb, cfg, branch_role="constructive")
    b = LatentWorkspace.from_prompt_embeddings(emb, cfg, branch_role="counterexample")
    assert not bool(mx.allclose(a.z, b.z)), "branch roles must produce distinct basins"


def test_ablation_restores_exactly(tiny_model):
    inner = tiny_model.model
    emb = inner.embed_tokens(mx.array(PROMPT))
    ws = LatentWorkspace.from_prompt_embeddings(emb, WorkspaceConfig(n_slots=4, seed=3))
    original = ws.z
    record = ws.ablate(2, mode="zero")
    assert float(mx.max(mx.abs(ws.z[:, 2, :]))) == 0.0
    assert not bool(mx.allclose(ws.z, original))
    ws.restore_ablation(record)
    assert bool(mx.allclose(ws.z, original)), "ablation restore must be exact"


# ── KV discipline ───────────────────────────────────────────────────────


def test_window_rewind_keeps_offsets_stable(tiny_model):
    cache = _prefill(tiny_model, mx.array(PROMPT))
    prompt_len = len(PROMPT[0])
    ws, runner = _seed_workspace(tiny_model, cache)
    # Prelude persisted slot KV at [0..P_END); window layers still prompt-only.
    for i in range(N_LAYERS):
        expected = prompt_len + 4 if i < P_END else prompt_len
        assert cache[i].offset == expected

    cfg = RecurrenceConfig(max_steps=6)
    z = ws.z
    for step in range(6):
        z = recurrence_step(z, runner, cache, P_END, C_START, cfg, step)
        for i in range(P_END, C_START):
            assert cache[i].offset == prompt_len, "recurrent pass must rewind slot KV"

    z_fin = runner.run(z, cache, P_END, C_START, persist=True)
    runner.run(z_fin, cache, C_START, N_LAYERS, persist=True)
    assert all(c.offset == prompt_len + 4 for c in cache), "final persist fills every layer"


# ── Controlled recurrence dynamics ──────────────────────────────────────


def test_recurrence_converges_to_fixed_point(tiny_model):
    cache = _prefill(tiny_model, mx.array(PROMPT))
    ws, runner = _seed_workspace(tiny_model, cache)
    cfg = RecurrenceConfig(max_steps=24, convergence_eps=0.02, min_steps=2)
    halting = HaltingController(config=cfg, baseline_rms=float(mx.mean(per_position_rms(ws.z))))

    z = ws.z
    reason = ""
    for step in range(cfg.max_steps):
        z_next = recurrence_step(z, runner, cache, P_END, C_START, cfg, step, anchor=ws.z)
        residual = relative_residual(z_next, z)
        decision = halting.observe(step, z_next, residual)
        z = z_next
        if decision.should_halt:
            reason = decision.reason
            break

    assert reason == "converged", f"expected fixed-point convergence, got {reason!r}"
    trail = halting.residual_trail
    assert trail[-1] < trail[0] / 3, f"residuals must contract: {trail}"
    assert bool(mx.all(mx.isfinite(z)))


def test_rms_match_bounds_norm_drift(tiny_model):
    z_ref = mx.random.normal((1, 4, 64), key=mx.random.key(0))
    z_wild = z_ref * 250.0
    matched = rms_match(z_wild, z_ref, clip_ratio=3.0)
    ratio = float(mx.mean(per_position_rms(matched)) / mx.mean(per_position_rms(z_ref)))
    assert ratio <= 3.01, "RMSMatch must clamp runaway norms"


def test_divergence_guard_trips_on_nonfinite():
    cfg = RecurrenceConfig()
    halting = HaltingController(config=cfg, baseline_rms=1.0)
    bad = mx.array([[[float("nan")] * 8]])
    decision = halting.observe(0, bad, 0.5)
    assert decision.should_halt and decision.reason == "diverged_nonfinite"


def test_overthinking_revert_returns_best_state():
    cfg = RecurrenceConfig(max_steps=5, min_steps=1, convergence_eps=1e-9)
    halting = HaltingController(config=cfg, baseline_rms=1.0)
    states = [mx.full((1, 2, 4), float(i + 1)) for i in range(5)]
    scores = [0.1, 0.9, 0.4, 0.2, 0.1]  # peak at step 1 — later steps overthink
    for step, (state, score) in enumerate(zip(states, scores)):
        halting.observe(step, state, residual=0.5, score=score)
    final, reverted = halting.final_state(states[-1])
    assert reverted is True
    assert float(final[0, 0, 0]) == 2.0, "must revert to the peak-score state"


def test_alpha_cosine_schedule_decays():
    cfg = RecurrenceConfig(max_steps=10, alpha=0.6, alpha_schedule="cosine")
    values = [alpha_at(cfg, t) for t in range(10)]
    assert values[0] == pytest.approx(0.6, abs=1e-6)
    assert values[-1] < values[0] / 2
    assert all(a >= cfg.alpha * 0.25 - 1e-9 for a in values)


def test_budget_charged_and_halts(tiny_model):
    cache = _prefill(tiny_model, mx.array(PROMPT))
    budget = ComputeBudget(max_layer_apps=4 * (C_START - P_END) * 2)  # two steps' worth
    ws, runner = _seed_workspace(tiny_model, cache, budget=budget)
    cfg = RecurrenceConfig(max_steps=50, convergence_eps=1e-9, min_steps=1)
    halting = HaltingController(config=cfg, baseline_rms=float(mx.mean(per_position_rms(ws.z))))

    z = ws.z
    reason = ""
    for step in range(cfg.max_steps):
        z_next = recurrence_step(z, runner, cache, P_END, C_START, cfg, step, anchor=ws.z)
        decision = halting.observe(
            step, z_next, relative_residual(z_next, z), budget=budget
        )
        z = z_next
        if decision.should_halt:
            reason = decision.reason
            break

    assert reason == "budget_exhausted"
    assert budget.spent_layer_apps >= budget.max_layer_apps


# ── Causality: slots must matter to decode ──────────────────────────────


def test_slot_ablation_changes_decode_logits(tiny_model):
    inner = tiny_model.model
    prompt = mx.array(PROMPT)
    prompt_len = prompt.shape[1]

    def episode(ablate: bool):
        from mlx_lm.models.base import create_attention_mask

        cache = _prefill(tiny_model, prompt)
        ws, runner = _seed_workspace(tiny_model, cache)
        cfg = RecurrenceConfig(max_steps=6)
        z = ws.z
        for step in range(cfg.max_steps):
            z = recurrence_step(z, runner, cache, P_END, C_START, cfg, step, anchor=ws.z)
        ws.update(z)
        if ablate:
            ws.ablate(1, mode="zero")
        z_fin = runner.run(ws.z, cache, P_END, C_START, persist=True)
        runner.run(z_fin, cache, C_START, N_LAYERS, persist=True)
        tok = mx.array([[1]])
        h = inner.embed_tokens(tok)
        mask = create_attention_mask(h, cache)
        for i, layer in enumerate(inner.layers):
            h = layer(h, mask, cache[i])
        h = inner.norm(h)
        logits = (
            tiny_model.lm_head(h) if hasattr(tiny_model, "lm_head") else inner.embed_tokens.as_linear(h)
        )
        mx.eval(logits)
        return logits

    base = episode(ablate=False)
    base_again = episode(ablate=False)
    ablated = episode(ablate=True)
    assert bool(mx.allclose(base, base_again)), "episodes must be deterministic"
    delta = float(mx.max(mx.abs(base - ablated)))
    assert delta > 1e-6, "ablating a thought slot must causally change decode logits"
