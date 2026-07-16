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

import hashlib
import json
import stat
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.cache import KVCache  # noqa: E402
from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.fast_weights import EpisodicFastWeights  # noqa: E402
from core.brain.llm.latent_cortex.latent_opt import (  # noqa: E402
    LatentOptimizer,
    build_proxy_loss,
    prompt_token_distribution,
)
from core.brain.llm.latent_cortex.types import (  # noqa: E402
    ComputeBudget,
    FastWeightsConfig,
    LatentOptConfig,
    WorkspaceConfig,
)
from core.brain.llm.latent_cortex.workspace import LatentWorkspace  # noqa: E402

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
    with pytest.raises(ValueError):
        prompt_token_distribution([], 128)
    with pytest.raises(ValueError):
        prompt_token_distribution([128], 128)


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
    assert opt.trace.accepted == opt.trace.steps_taken
    assert all(
        after < before
        for before, after in zip(trail, trail[1:], strict=False)
    )


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


def test_control_arm_uses_the_same_monotone_acceptance_rule(tiny_model):
    ws = _workspace(tiny_model)
    cfg = LatentOptConfig(enabled=True, steps=8, lr=0.05, control_mode=True)
    loss_fn = build_proxy_loss(tiny_model, ws.z, PROMPT_TOKENS, cfg)
    opt = LatentOptimizer(loss_fn, cfg, seed=7)

    opt.run(ws.z)

    assert opt.trace.accepted == opt.trace.steps_taken
    assert opt.trace.accepted + opt.trace.rejected == opt.trace.attempts
    assert all(
        after < before
        for before, after in zip(
            opt.trace.loss_trail,
            opt.trace.loss_trail[1:],
            strict=False,
        )
    )


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

    opt3 = LatentOptimizer(loss_fn, cfg, seed=2)
    with pytest.raises(RuntimeError, match="non-finite baseline"):
        opt3.run_with_verifier(ws.z, lambda z: float("nan"))


def test_latent_optimizer_budget_preserves_completion_reserve(tiny_model):
    ws = _workspace(tiny_model)
    cfg = LatentOptConfig(enabled=True, steps=6, lr=0.05)
    loss_fn = build_proxy_loss(tiny_model, ws.z, PROMPT_TOKENS, cfg)
    per_loss = ws.z.shape[1] * N_LAYERS
    completion_reserve = 2 * per_loss
    # Exactly one gradient/backward estimate (3x) and its fixed-cost line
    # search (12x) fit. A second proposal must be refused before compute.
    budget = ComputeBudget(
        max_layer_apps=completion_reserve + 15 * per_loss,
        wall_clock_s=30.0,
    )
    opt = LatentOptimizer(
        loss_fn,
        cfg,
        seed=4,
        budget=budget,
        layer_apps_per_loss=per_loss,
        reserve_layer_apps=completion_reserve,
    )

    z_out = opt.run(ws.z)

    assert opt.trace.attempts == 1
    assert opt.trace.accepted == opt.trace.steps_taken == 1
    assert opt.trace.budget_exhausted is True
    assert budget.spent_layer_apps == 15 * per_loss
    assert budget.remaining_layer_apps == completion_reserve
    assert bool(mx.all(mx.isfinite(z_out)))


def test_budgeted_latent_optimizer_requires_explicit_compute_cost(tiny_model):
    ws = _workspace(tiny_model)
    cfg = LatentOptConfig(enabled=True, steps=1)
    loss_fn = build_proxy_loss(tiny_model, ws.z, PROMPT_TOKENS, cfg)
    with pytest.raises(ValueError, match="positive loss-evaluation cost"):
        LatentOptimizer(loss_fn, cfg, budget=ComputeBudget(max_layer_apps=100))


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


def test_fast_weight_attach_is_transactional(tiny_model, monkeypatch):
    originals = [layer.self_attn.o_proj for layer in tiny_model.model.layers]
    target_parent = tiny_model.model.layers[P_END + 1].self_attn
    target_type = type(target_parent)
    original_setattr = target_type.__setattr__

    def guarded_setattr(self, name, value):
        if self is target_parent and name == "o_proj" and value is not originals[P_END + 1]:
            raise RuntimeError("injected partial-attach failure")
        return original_setattr(self, name, value)

    monkeypatch.setattr(target_type, "__setattr__", guarded_setattr)
    fw = EpisodicFastWeights(FastWeightsConfig(enabled=True, rank=2, target="o_proj"))
    with pytest.raises(RuntimeError, match="partial-attach"):
        fw.attach(
            tiny_model.model,
            (P_END, C_START),
            seed_stat=0.4,
            episode_id="ep-transaction",
        )
    assert not fw.handles
    assert [layer.self_attn.o_proj for layer in tiny_model.model.layers] == originals


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
    assert 1 <= fw.lifecycle.optimized_steps <= 4
    assert fw.lifecycle.optimization_attempts <= 4
    assert fw.lifecycle.rejected_steps in (0, 1)
    assert all(
        after_step < before_step
        for before_step, after_step in zip(
            fw.lifecycle.loss_trail,
            fw.lifecycle.loss_trail[1:],
            strict=False,
        )
    )

    trained_out = _probe(tiny_model)
    assert not bool(mx.allclose(trained_out, baseline)), "trained ΔW must change the function"

    fw.snapshot_for_export()
    fw.detach()
    assert fw.prove_erase(lambda: _probe(tiny_model), baseline) is True
    assert fw.lifecycle.erased and fw.lifecycle.erase_proven


def test_fast_weight_optimizer_charges_before_faulting_gradient(tiny_model):
    fw = EpisodicFastWeights(
        FastWeightsConfig(enabled=True, rank=1, target="down_proj", opt_steps=1)
    )
    fw.attach(tiny_model.model, (P_END, P_END + 1), seed_stat=0.4, episode_id="ep-cost")
    budget = ComputeBudget(max_layer_apps=400)

    def broken_loss():
        raise RuntimeError("faulted model operation")

    with pytest.raises(RuntimeError, match="faulted model operation"):
        fw.optimize(
            broken_loss,
            budget=budget,
            layer_apps_per_forward=100,
            reserve_layer_apps=100,
        )

    assert budget.spent_layer_apps == 300
    assert budget.remaining_layer_apps == 100
    assert fw.lifecycle.optimization_attempts == 1
    fw.detach()


def test_fast_weight_optimizer_preserves_completion_reserve(tiny_model):
    fw = EpisodicFastWeights(
        FastWeightsConfig(
            enabled=True,
            rank=2,
            target="down_proj",
            opt_steps=4,
            lr=0.05,
        )
    )
    fw.attach(
        tiny_model.model,
        (P_END, P_END + 1),
        seed_stat=0.4,
        episode_id="ep-reserve",
    )

    def loss_fn():
        out = _probe(tiny_model)[0, -1]
        return -(out[42] - mx.logsumexp(out))

    budget = ComputeBudget(max_layer_apps=450)
    fw.optimize(
        loss_fn,
        budget=budget,
        layer_apps_per_forward=100,
        reserve_layer_apps=50,
    )

    assert budget.spent_layer_apps == 400
    assert budget.remaining_layer_apps == 50
    assert fw.lifecycle.optimization_attempts == 1
    assert fw.lifecycle.budget_exhausted is True
    fw.detach()


def test_budgeted_fast_weight_optimizer_requires_explicit_compute_cost(tiny_model):
    fw = EpisodicFastWeights(
        FastWeightsConfig(enabled=True, rank=1, target="down_proj", opt_steps=1)
    )
    fw.attach(tiny_model.model, (P_END, P_END + 1), seed_stat=0.4, episode_id="ep-free")

    with pytest.raises(ValueError, match="requires a positive forward cost"):
        fw.optimize(
            lambda: mx.array(0.0),
            budget=ComputeBudget(max_layer_apps=100),
        )

    fw.detach()


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
    delta_path = out_dir / "delta_weights.npz"
    assert delta_path.exists()
    payload = json.loads((out_dir / "evidence.json").read_text())
    assert payload["schema"] == "aura.latent_cortex.fast_weight_candidate.v1"
    assert payload["evidence"]["wins"] == 3
    assert payload["lifecycle"]["erase_proven"] is True
    assert payload["lifecycle"]["exported"] is True
    assert payload["artifacts"]["delta_weights.npz"] == {
        "sha256": hashlib.sha256(delta_path.read_bytes()).hexdigest(),
        "size_bytes": delta_path.stat().st_size,
    }
    assert stat.S_IMODE(out_dir.stat().st_mode) == 0o700
    assert fw.last_export_receipt is not None
    assert fw.last_export_receipt["transaction_id"]
    assert set(fw.last_export_receipt["paths"]) == {
        str(delta_path),
        str(out_dir / "evidence.json"),
    }


def test_consolidation_export_rejects_tampered_batch_receipt(
    tiny_model, tmp_path, monkeypatch
):
    from core.brain.llm.latent_cortex import persistence

    baseline = _probe(tiny_model)
    fw = EpisodicFastWeights(FastWeightsConfig(enabled=True, rank=2, target="o_proj"))
    fw.attach(tiny_model.model, (P_END, C_START), seed_stat=0.4, episode_id="ep-tamper")
    fw.snapshot_for_export()
    fw.detach()
    assert fw.prove_erase(lambda: _probe(tiny_model), baseline) is True

    class TamperedPersistence:
        def publish_fast_weight_candidate(
            self, target_dir, *, delta_payload, evidence_payload
        ):
            delta_path = str(target_dir / "delta_weights.npz")
            evidence_path = str(target_dir / "evidence.json")
            return SimpleNamespace(
                transaction_id="forged",
                paths=(delta_path, evidence_path),
                sha256={
                    delta_path: hashlib.sha256(delta_payload).hexdigest(),
                    evidence_path: "0" * 64,
                },
            )

    monkeypatch.setattr(persistence, "_PERSISTENCE", TamperedPersistence())
    exported = fw.export_candidate(
        tmp_path / "queue", episode_id="ep-tamper", evidence={"wins": 1}
    )

    assert exported is None
    assert fw.lifecycle.exported is False
    assert fw.last_export_receipt is None
