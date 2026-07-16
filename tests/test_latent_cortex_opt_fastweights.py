"""Contract tests: latent optimization + episode fast weights.

The heart of these contracts:
- proxy descent actually reduces the proxy loss (gradients are real);
- the control arm moves with matched magnitude but random direction;
- verifier hill-climbing never accepts a verifier-rejected state;
- fast weights are EXACT identity at attach, trainable to nonzero effect,
  and provably erased (bit-for-bit baseline restoration);
- consolidation export refuses unproven erases.
"""
from __future__ import annotations

import json

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen2 import Model, ModelArgs

from core.brain.llm.latent_cortex.fast_weights import EpisodicFastWeights
from core.brain.llm.latent_cortex.latent_opt import (
    LatentOptimizer,
    build_proxy_loss,
    prompt_token_distribution,
)
from core.brain.llm.latent_cortex.types import FastWeightsConfig, LatentOptConfig, WorkspaceConfig
from core.brain.llm.latent_cortex.workspace import LatentWorkspace

N_LAYERS, P_END, C_START = 8, 2, 6
PROMPT_TOKENS = [5, 9, 17, 3, 42, 7, 11, 23, 2, 88]


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


def _workspace(model, n_slots=4):
    emb = model.model.embed_tokens(mx.array([PROMPT_TOKENS]))
    return LatentWorkspace.from_prompt_embeddings(emb, WorkspaceConfig(n_slots=n_slots, seed=3))


# ── Latent optimization ─────────────────────────────────────────────────


def test_prompt_distribution_is_normalized_and_supported():
    dist = prompt_token_distribution(PROMPT_TOKENS, 128)
    assert float(mx.sum(dist)) == pytest.approx(1.0, abs=1e-5)
    assert float(dist[5]) > 0 and float(dist[6]) == 0.0


def test_proxy_descent_reduces_loss(tiny_model):
    ws = _workspace(tiny_model)
    cfg = LatentOptConfig(enabled=True, steps=6, lr=0.05)
    loss_fn = build_proxy_loss(tiny_model, ws.z, PROMPT_TOKENS, cfg)
    opt = LatentOptimizer(loss_fn, cfg, seed=1)
    z_out = opt.run(ws.z)
    trail = opt.trace.loss_trail
    assert len(trail) == 7  # 6 step losses + final
    assert trail[-1] < trail[0], f"proxy descent must reduce loss: {trail}"
    assert bool(mx.all(mx.isfinite(z_out)))


def test_control_arm_matches_magnitude_but_not_direction(tiny_model):
    ws = _workspace(tiny_model)
    base_cfg = dict(enabled=True, steps=1, lr=0.05)
    grad_opt = LatentOptimizer(
        build_proxy_loss(tiny_model, ws.z, PROMPT_TOKENS, LatentOptConfig(**base_cfg)),
        LatentOptConfig(**base_cfg),
        seed=1,
    )
    ctrl_opt = LatentOptimizer(
        build_proxy_loss(tiny_model, ws.z, PROMPT_TOKENS, LatentOptConfig(**base_cfg, control_mode=True)),
        LatentOptConfig(**base_cfg, control_mode=True),
        seed=1,
    )
    z_grad = grad_opt.step(ws.z, 0)
    z_ctrl = ctrl_opt.step(ws.z, 0)
    mag_grad = float(mx.linalg.norm(mx.reshape(z_grad - ws.z, (-1,))))
    mag_ctrl = float(mx.linalg.norm(mx.reshape(z_ctrl - ws.z, (-1,))))
    assert mag_grad == pytest.approx(mag_ctrl, rel=1e-4), "control must match step magnitude"
    direction_cos = float(
        mx.sum(mx.reshape(z_grad - ws.z, (-1,)) * mx.reshape(z_ctrl - ws.z, (-1,)))
        / (mag_grad * mag_ctrl)
    )
    assert abs(direction_cos) < 0.5, "control direction must not mirror the gradient"
    assert grad_opt.trace.mode == "gradient" and ctrl_opt.trace.mode == "control"


def test_verifier_hill_climb_never_accepts_rejected_states(tiny_model):
    ws = _workspace(tiny_model)
    cfg = LatentOptConfig(enabled=True, steps=5, lr=0.05)
    loss_fn = build_proxy_loss(tiny_model, ws.z, PROMPT_TOKENS, cfg)
    opt = LatentOptimizer(loss_fn, cfg, seed=2)

    z_out, score = opt.run_with_verifier(ws.z, lambda z: -1.0)  # verifier hates everything
    assert bool(mx.allclose(z_out, ws.z)), "all-rejected climb must return the original state"
    assert opt.trace.accepted == 0 and opt.trace.rejected == 5

    # A verifier that rewards drift accepts at least one proposal.
    opt2 = LatentOptimizer(loss_fn, cfg, seed=2)
    z_out2, _ = opt2.run_with_verifier(
        ws.z, lambda z: float(mx.linalg.norm(mx.reshape(z - ws.z, (-1,))))
    )
    assert opt2.trace.accepted >= 1
    assert not bool(mx.allclose(z_out2, ws.z))


# ── Fast weights ────────────────────────────────────────────────────────


def _probe(model):
    tokens = mx.array([PROMPT_TOKENS])
    cache = [KVCache() for _ in model.model.layers]
    from mlx_lm.models.base import create_attention_mask

    inner = model.model
    h = inner.embed_tokens(tokens)
    mask = create_attention_mask(h, cache)
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask, cache[i])
    h = inner.norm(h)
    out = model.lm_head(h) if hasattr(model, "lm_head") else inner.embed_tokens.as_linear(h)
    mx.eval(out)
    return out


def test_fast_weights_identity_at_attach(tiny_model):
    baseline = _probe(tiny_model)
    fw = EpisodicFastWeights(FastWeightsConfig(enabled=True, rank=2, target="o_proj"))
    wrapped = fw.attach(tiny_model.model, (P_END, C_START), seed_stat=0.4, episode_id="ep-t1")
    try:
        assert wrapped == C_START - P_END
        attached_out = _probe(tiny_model)
        assert bool(mx.allclose(attached_out, baseline)), "V=0 attach must be exact identity"
    finally:
        fw.detach()


def test_fast_weights_optimize_changes_function_then_erase_restores(tiny_model):
    baseline = _probe(tiny_model)
    fw = EpisodicFastWeights(
        FastWeightsConfig(enabled=True, rank=2, target="down_proj", opt_steps=4, lr=0.05)
    )
    fw.attach(tiny_model.model, (P_END, C_START), seed_stat=0.4, episode_id="ep-t2")

    target_token = 42

    def loss_fn():
        out = _probe(tiny_model)[0, -1]
        return -(out[target_token] - mx.logsumexp(out))

    before = float(loss_fn())
    fw.optimize(loss_fn)
    after = float(loss_fn())
    assert after < before, "fast-weight optimization must reduce its loss"
    assert fw.lifecycle.optimized_steps == 4

    trained_out = _probe(tiny_model)
    assert not bool(mx.allclose(trained_out, baseline)), "trained ΔW must change the function"

    fw.snapshot_for_export()
    fw.detach()
    assert fw.prove_erase(lambda: _probe(tiny_model), baseline) is True
    assert fw.lifecycle.erased and fw.lifecycle.erase_proven


def test_erase_proof_fails_honestly_if_model_left_dirty(tiny_model):
    baseline = _probe(tiny_model)
    fw = EpisodicFastWeights(FastWeightsConfig(enabled=True, rank=1, target="o_proj"))
    fw.attach(tiny_model.model, (P_END, P_END + 1), seed_stat=0.4, episode_id="ep-t3")
    # Sabotage: corrupt a base weight while attached, then detach.
    layer = tiny_model.model.layers[P_END]
    original = layer.self_attn.q_proj.weight
    layer.self_attn.q_proj.weight = original + 0.01
    fw.detach()
    try:
        assert fw.prove_erase(lambda: _probe(tiny_model), baseline) is False
    finally:
        layer.self_attn.q_proj.weight = original


def test_consolidation_export_requires_proven_erase(tiny_model, tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    baseline = _probe(tiny_model)
    fw = EpisodicFastWeights(FastWeightsConfig(enabled=True, rank=2, target="o_proj"))
    fw.attach(tiny_model.model, (P_END, C_START), seed_stat=0.4, episode_id="ep-t4")
    fw.snapshot_for_export()

    # Unproven erase ⇒ refused.
    assert fw.export_candidate(tmp_path / "queue", episode_id="ep-t4", evidence={}) is None

    fw.detach()
    assert fw.prove_erase(lambda: _probe(tiny_model), baseline) is True
    out_dir = fw.export_candidate(
        tmp_path / "queue", episode_id="ep-t4", evidence={"wins": 3, "domain": "unit"}
    )
    assert out_dir is not None
    assert (out_dir / "delta_weights.npz").exists()
    payload = json.loads((out_dir / "evidence.json").read_text())
    assert payload["evidence"]["wins"] == 3
    assert payload["lifecycle"]["erase_proven"] is True
