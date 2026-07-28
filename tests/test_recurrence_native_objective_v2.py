"""Live-cache parity and learnability contracts for recurrence objective v2."""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.base import create_attention_mask  # noqa: E402
from mlx_lm.models.cache import KVCache  # noqa: E402
from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.execution_spec import (  # noqa: E402
    RLCExecutionSpec,
)
from core.brain.llm.latent_cortex.loop_core import (  # noqa: E402
    build_loop_core_contract,
)
from core.brain.llm.latent_cortex.recurrence import (  # noqa: E402
    WindowRunner,
    recurrence_step,
)
from core.brain.llm.latent_cortex.recurrence_adapter import (  # noqa: E402
    ScopedLoRALinear,
)
from core.brain.llm.latent_cortex.types import (  # noqa: E402
    ComputeBudget,
    RecurrenceConfig,
    WorkspaceConfig,
)
from core.brain.llm.latent_cortex.workspace import LatentWorkspace  # noqa: E402
from core.learning.recurrence_native_objective_v2 import (  # noqa: E402
    _exchange_and_decorrelate,
    depth_curriculum_loss_v2,
    detached_monotonicity_penalty,
    live_path_forward,
    live_path_loss,
    prepare_final_recurrent_transition,
    validate_final_recurrent_transition_receipt,
)
from core.learning.recurrent_grpo import (  # noqa: E402
    cortex_config_from_execution_spec,
    recurrent_policy_sha256,
)

PROMPT = [5, 9, 17, 3, 42]
ANSWER = [7, 11, 23]


def _model() -> Model:
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=32,
        num_hidden_layers=4,
        intermediate_size=64,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
    )
    model = Model(args)
    model.freeze()
    for index in (1, 2):
        parent = model.model.layers[index].self_attn
        wrapped = ScopedLoRALinear.from_base(parent.o_proj, r=2, scale=1.0)
        wrapped.lora_a = mx.ones_like(wrapped.lora_a) * 0.03
        wrapped.lora_b = mx.ones_like(wrapped.lora_b) * 0.03
        parent.o_proj = wrapped
    mx.eval(model.parameters())
    return model


def _spec(**changes) -> RLCExecutionSpec:
    values = RLCExecutionSpec(
        n_slots=3,
        branch_roles=("constructive_solution",),
        recurrent_steps=2,
        exchange_interval=1,
    ).to_dict()
    values.update(changes)
    return RLCExecutionSpec.from_dict(values)


def _live_cache_logits(model: Model, spec: RLCExecutionSpec):
    inner = model.model
    prompt = mx.array([PROMPT])
    cache = [KVCache() for _ in inner.layers]
    prompt_embeddings = inner.embed_tokens(prompt)
    hidden = prompt_embeddings
    mask = create_attention_mask(hidden, cache)
    for index, layer in enumerate(inner.layers):
        hidden = layer(hidden, mask, cache[index])

    workspace = LatentWorkspace.from_prompt_embeddings(
        prompt_embeddings,
        WorkspaceConfig(
            n_slots=spec.n_slots,
            seed=spec.slot_seed,
            roles=spec.slot_roles,
            anchor_scale=spec.anchor_scale,
        ),
        branch_role=spec.branch_roles[0],
    )
    runner = WindowRunner(inner, ComputeBudget())
    prelude_end = 1
    coda_start = 3
    anchor = runner.run(
        workspace.seed_z, cache, 0, prelude_end, persist=False
    )
    state = anchor
    recurrence = RecurrenceConfig(
        max_steps=spec.recurrent_steps,
        min_steps=1,
        alpha=spec.alpha,
        alpha_schedule=spec.alpha_schedule,
        rms_clip_ratio=spec.rms_clip_ratio,
    )
    for step in range(spec.recurrent_steps):
        state = recurrence_step(
            state,
            runner,
            cache,
            prelude_end,
            coda_start,
            recurrence,
            step,
            anchor=anchor,
        )

    runner.run(workspace.seed_z, cache, 0, prelude_end, persist=True)
    persisted = runner.run(state, cache, prelude_end, coda_start, persist=True)
    output = runner.run(persisted, cache, coda_start, len(inner.layers), persist=True)

    def logits(value):
        return inner.embed_tokens.as_linear(inner.norm(value))

    predictions = [logits(output[:, -1:, :])[:, -1, :]]
    for token in ANSWER[:-1]:
        token_hidden = inner.embed_tokens(mx.array([[token]]))
        token_mask = create_attention_mask(token_hidden, cache)
        for index, layer in enumerate(inner.layers):
            token_hidden = layer(token_hidden, token_mask, cache[index])
        predictions.append(logits(token_hidden)[:, -1, :])
    stacked = mx.stack(predictions, axis=1)
    mx.eval(stacked, state)
    return stacked, state


def test_execution_spec_round_trip_hash_and_strict_fields():
    spec = _spec()
    assert RLCExecutionSpec.from_dict(spec.to_dict()) == spec
    assert len(spec.sha256) == 64
    assert spec.with_depth(4).recurrent_steps == 4
    payload = spec.to_dict()
    payload["unknown"] = True
    with pytest.raises(ValueError, match="unknown"):
        RLCExecutionSpec.from_dict(payload)


def test_functional_executor_matches_live_cache_slots_and_logits():
    model = _model()
    spec = _spec()
    functional = live_path_forward(model, PROMPT, ANSWER, spec=spec)
    cached_logits, cached_state = _live_cache_logits(model, spec)
    functional_logits = functional.branch_logits[0]
    functional_state = functional.branch_states[0]
    mx.eval(functional_logits, functional_state)

    assert bool(mx.array_equal(functional_state, cached_state))
    # KV-cache and full causal-sequence attention use different kernels, so the
    # logits are numerically close rather than bit-identical on MLX.
    assert float(mx.max(mx.abs(functional_logits - cached_logits))) < 0.01


def test_final_transition_freezes_one_real_parent_child_edge() -> None:
    model = _model()
    spec = _spec(
        recurrent_steps=3,
        branch_roles=("constructive_solution", "critical_audit"),
    )
    policy_before = recurrent_policy_sha256(model, spec)

    transition = prepare_final_recurrent_transition(model, PROMPT, spec=spec)
    parent = live_path_forward(
        model,
        PROMPT,
        ANSWER,
        spec=spec.with_depth(2),
    )
    child = live_path_forward(model, PROMPT, ANSWER, spec=spec)

    assert transition.transition_index == 2
    assert transition.parent_branch_sha256s != transition.child_branch_sha256s
    assert len(transition.parent_states) == len(transition.child_states) == 2
    for observed, expected in zip(
        transition.parent_states, parent.branch_states, strict=True
    ):
        assert bool(mx.array_equal(observed, expected))
    for observed, expected in zip(
        transition.child_states, child.branch_states, strict=True
    ):
        assert bool(mx.array_equal(observed, expected))
    assert recurrent_policy_sha256(model, spec) == policy_before
    assert validate_final_recurrent_transition_receipt(
        transition.receipt()
    ) == transition.receipt()


def test_transition_receipt_rejects_resealed_noncausal_state_substitution() -> None:
    transition = prepare_final_recurrent_transition(
        _model(), PROMPT, spec=_spec(recurrent_steps=2)
    )
    attacked = transition.receipt()
    attacked["child_branch_sha256s"] = list(attacked["parent_branch_sha256s"])

    with pytest.raises(ValueError, match="identity|digest"):
        validate_final_recurrent_transition_receipt(attacked)


@pytest.mark.parametrize(
    ("schedule", "depth"),
    (("constant", 1), ("constant", 3), ("cosine", 4)),
)
def test_shared_loop_contract_and_cache_parity_across_schedules(schedule, depth):
    model = _model()
    spec = _spec(alpha_schedule=schedule, recurrent_steps=depth)
    functional = live_path_forward(model, PROMPT, ANSWER, spec=spec)
    cached_logits, cached_state = _live_cache_logits(model, spec)
    mx.eval(functional.branch_logits[0], functional.branch_states[0])

    assert bool(mx.array_equal(functional.branch_states[0], cached_state))
    assert float(mx.max(mx.abs(functional.branch_logits[0] - cached_logits))) < 0.01
    live_config = cortex_config_from_execution_spec(spec)
    expected = build_loop_core_contract(
        prelude_end=1,
        coda_start=3,
        max_steps=live_config.recurrence.max_steps,
        min_steps=live_config.recurrence.min_steps,
        alpha=live_config.recurrence.alpha,
        alpha_schedule=live_config.recurrence.alpha_schedule,
        rms_clip_ratio=live_config.recurrence.rms_clip_ratio,
        convergence_eps=live_config.recurrence.convergence_eps,
        divergence_ratio=live_config.recurrence.divergence_ratio,
        fixed_depth=live_config.recurrence.fixed_depth,
    )
    assert functional.loop_core == expected


def test_answer_loss_is_finite_and_recurrence_adapter_receives_gradient():
    model = _model()
    projection = model.model.layers[1].self_attn.o_proj

    def loss_for_b(lora_b):
        original = projection.lora_b
        projection.lora_b = lora_b
        try:
            return live_path_loss(model, PROMPT, ANSWER, spec=_spec())
        finally:
            projection.lora_b = original

    value, gradient = mx.value_and_grad(loss_for_b)(projection.lora_b)
    mx.eval(value, gradient)
    assert bool(mx.isfinite(value))
    assert float(mx.linalg.norm(mx.reshape(gradient, (-1,)))) > 0.0


def test_two_branch_exchange_and_depth_curriculum_are_live():
    model = _model()
    spec = _spec(
        branch_roles=["constructive_solution", "counterexample_search"]
    )
    forward = live_path_forward(model, PROMPT, ANSWER, spec=spec)
    assert len(forward.branch_logits) == 2
    assert forward.exchanges == spec.recurrent_steps
    assert not bool(mx.array_equal(forward.branch_states[0], forward.branch_states[1]))
    loss = depth_curriculum_loss_v2(
        model,
        PROMPT,
        ANSWER,
        spec=spec,
        depths=(1, 2),
    )
    mx.eval(loss)
    assert bool(mx.isfinite(loss))


def test_training_exchange_cannot_rebroadcast_mailbox_content():
    spec = _spec(
        n_slots=4,
        branch_roles=["constructive_solution", "counterexample_search"],
        exchange_interval=1,
        exchange_gamma=0.35,
        jitter_scale=0.0,
    )
    private_left = mx.ones((1, 3, 8))
    private_right = mx.ones((1, 3, 8)) * 2.0
    first = [
        mx.concatenate([mx.zeros((1, 1, 8)), private_left], axis=1),
        mx.concatenate([mx.ones((1, 1, 8)) * 10.0, private_right], axis=1),
    ]
    mailbox_perturbed = [
        mx.concatenate([mx.ones((1, 1, 8)) * 999.0, private_left], axis=1),
        mx.concatenate([mx.ones((1, 1, 8)) * -999.0, private_right], axis=1),
    ]
    exchanged = _exchange_and_decorrelate(first, spec, 1)
    perturbed = _exchange_and_decorrelate(mailbox_perturbed, spec, 1)

    def inferred_consensus(output, prior):
        return (
            output[:, :1, :] - (1.0 - spec.exchange_gamma) * prior[:, :1, :]
        ) / spec.exchange_gamma

    for index in range(2):
        left = inferred_consensus(exchanged[index], first[index])
        right = inferred_consensus(perturbed[index], mailbox_perturbed[index])
        assert bool(mx.allclose(left, right, atol=1e-3))


def test_monotonic_hinge_cannot_reward_shallow_damage():
    def penalty_for_shallow(shallow):
        return detached_monotonicity_penalty((shallow, mx.array(2.0)))

    def penalty_for_deep(deep):
        return detached_monotonicity_penalty((mx.array(1.0), deep))

    shallow_grad = mx.grad(penalty_for_shallow)(mx.array(1.0))
    deep_grad = mx.grad(penalty_for_deep)(mx.array(2.0))
    mx.eval(shallow_grad, deep_grad)
    assert float(shallow_grad) == 0.0
    assert float(deep_grad) == 1.0


def test_v2_rejects_unsupported_post_transforms_and_bridge_mismatch():
    payload = _spec().to_dict()
    payload["fast_weights_mode"] = "enabled"
    with pytest.raises(ValueError, match="fast weights"):
        RLCExecutionSpec.from_dict(payload)

    with pytest.raises(ValueError, match="bridge tokens supplied"):
        live_path_forward(_model(), PROMPT, ANSWER, spec=_spec(), bridge_tokens=[2])
