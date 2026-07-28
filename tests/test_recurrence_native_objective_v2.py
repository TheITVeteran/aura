"""Live-cache parity and learnability contracts for recurrence objective v2."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_lm")

from mlx.utils import tree_flatten  # noqa: E402
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
from core.learning.progressive_recurrent_objective import (  # noqa: E402
    progressive_objective_loss,
)
from core.learning.recurrence_native_objective_v2 import (  # noqa: E402
    EXACT_ADJOINT_INTERVENTION_RECEIPT_SCHEMA,
    EXACT_ADJOINT_TRAJECTORY_RECEIPT_SCHEMA,
    ExactAdjointInterventionConfig,
    ExactAdjointTrajectoryConfig,
    _advance_recurrent_states,
    _exchange_and_decorrelate,
    _persist_and_score,
    _prepare_live_path,
    depth_curriculum_loss_v2,
    detached_monotonicity_penalty,
    exact_adjoint_composite_live_path_value_and_grad,
    exact_adjoint_trajectory_live_path_value_and_grad,
    live_path_forward,
    live_path_loss,
    prepare_final_recurrent_transition,
    validate_exact_adjoint_live_path_receipt,
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
    anchor = runner.run(workspace.seed_z, cache, 0, prelude_end, persist=False)
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
    for observed, expected in zip(transition.parent_states, parent.branch_states, strict=True):
        assert bool(mx.array_equal(observed, expected))
    for observed, expected in zip(transition.child_states, child.branch_states, strict=True):
        assert bool(mx.array_equal(observed, expected))
    assert recurrent_policy_sha256(model, spec) == policy_before
    assert validate_final_recurrent_transition_receipt(transition.receipt()) == transition.receipt()


def test_transition_receipt_rejects_resealed_noncausal_state_substitution() -> None:
    transition = prepare_final_recurrent_transition(_model(), PROMPT, spec=_spec(recurrent_steps=2))
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


def test_bounded_exact_adjoint_matches_full_unroll_trajectory_gradient():
    """Trajectory terms must be mathematically identical, not merely finite."""

    mx.random.seed(20260728)
    monolithic = _model()
    mx.random.seed(20260728)
    streamed = _model()
    spec = _spec(recurrent_steps=3)
    config = ExactAdjointTrajectoryConfig(
        probe_steps=(1, 2, 3),
        improvement_weight=0.7,
        improvement_margin=0.2,
        displacement_weight=0.4,
        displacement_floor=0.99,
        oscillation_weight=0.3,
    )

    def full_unroll_loss(current_model):
        value, _telemetry = progressive_objective_loss(
            current_model,
            PROMPT,
            ANSWER,
            spec=spec,
            depth=3,
            probe_steps=config.probe_steps,
            final_weight=0.0,
            improvement_weight=config.improvement_weight,
            improvement_margin=config.improvement_margin,
            oscillation_weight=config.oscillation_weight,
            displacement_weight=config.displacement_weight,
            displacement_floor=config.displacement_floor,
        )
        return value

    full_value, full_gradients = nn.value_and_grad(monolithic, full_unroll_loss)(monolithic)
    exact = exact_adjoint_trajectory_live_path_value_and_grad(
        streamed,
        PROMPT,
        ANSWER,
        spec=spec,
        trajectory_config=config,
        policy_sha256=recurrent_policy_sha256(streamed, spec),
        token_loss_weights=(0.0,) * len(ANSWER),
    )
    mx.eval(full_value, full_gradients, exact.gradients)
    full_flat = dict(tree_flatten(full_gradients))
    exact_flat = dict(tree_flatten(exact.gradients))

    assert exact.receipt()["schema"] == EXACT_ADJOINT_TRAJECTORY_RECEIPT_SCHEMA
    assert exact.receipt()["branch_indices"] == [0]
    assert validate_exact_adjoint_live_path_receipt(exact.receipt()) == exact.receipt()
    assert exact.value == pytest.approx(float(full_value), abs=2e-5)
    assert set(full_flat) == set(exact_flat)
    for name in full_flat:
        difference = float(mx.max(mx.abs(full_flat[name] - exact_flat[name])))
        assert difference < 3e-4, name


def test_bounded_exact_adjoint_matches_causal_and_stopping_gradient():
    """Intervention terms must match one monolithic differentiable oracle."""

    mx.random.seed(20260729)
    monolithic = _model()
    mx.random.seed(20260729)
    streamed = _model()
    spec = _spec(recurrent_steps=3)
    config = ExactAdjointInterventionConfig(
        lesion_steps=(1, 3),
        causality_weight=0.7,
        causality_margin=2.0,
        stopping_steps=(1, 2, 3),
        stopping_weight=0.6,
        stopping_ponder_cost=0.02,
        stopping_temperature=0.3,
    )

    def full_unroll_loss(current_model):
        prepared = _prepare_live_path(
            current_model,
            PROMPT,
            ANSWER,
            spec=spec,
            bridge_tokens=(),
        )
        states = list(prepared.states)
        history = [tuple(states)]
        for step in range(spec.recurrent_steps):
            states = _advance_recurrent_states(
                current_model,
                prepared.prompts_at_window,
                states,
                prepared.anchors,
                spec,
                step,
                prepared.prelude_end,
                prepared.coda_start,
            )
            history.append(tuple(states))
        targets = mx.array(ANSWER)[None, :]

        def answer_loss(state):
            logits = _persist_and_score(
                current_model,
                prepared.prompt_embeddings,
                prepared.seeds[0],
                state,
                prepared.tail_embeddings,
                bridge_count=prepared.bridge_count,
                answer_count=prepared.answer_count,
                prelude_end=prepared.prelude_end,
                coda_start=prepared.coda_start,
            )
            return nn.losses.cross_entropy(
                logits.astype(mx.float32),
                targets,
                reduction="mean",
            )

        losses = [answer_loss(history[step][0]) for step in config.stopping_steps]
        intact = losses[-1]
        lesion_losses = []
        for lesion_step in config.lesion_steps:
            lesioned = list(history[lesion_step - 1])
            for replay_step in range(lesion_step, spec.recurrent_steps):
                lesioned = _advance_recurrent_states(
                    current_model,
                    prepared.prompts_at_window,
                    lesioned,
                    prepared.anchors,
                    spec,
                    replay_step,
                    prepared.prelude_end,
                    prepared.coda_start,
                )
            lesion_losses.append(answer_loss(lesioned[0]))
        causality = (
            config.causality_weight
            * mx.stack(
                [
                    mx.maximum(
                        intact - mx.stop_gradient(loss) + config.causality_margin,
                        0.0,
                    )
                    for loss in lesion_losses
                ]
            ).mean()
        )
        risks = mx.stack(
            [
                loss + config.stopping_ponder_cost * step
                for step, loss in zip(config.stopping_steps, losses, strict=True)
            ]
        )
        probabilities = mx.softmax(-mx.stop_gradient(risks) / config.stopping_temperature)
        stopping = config.stopping_weight * mx.sum(probabilities * risks)
        return causality + stopping

    full_value, full_gradients = nn.value_and_grad(monolithic, full_unroll_loss)(monolithic)
    exact = exact_adjoint_composite_live_path_value_and_grad(
        streamed,
        PROMPT,
        ANSWER,
        spec=spec,
        trajectory_config=None,
        intervention_config=config,
        policy_sha256=recurrent_policy_sha256(streamed, spec),
        token_loss_weights=(0.0,) * len(ANSWER),
    )
    mx.eval(full_value, full_gradients, exact.gradients)
    full_flat = dict(tree_flatten(full_gradients))
    exact_flat = dict(tree_flatten(exact.gradients))
    receipt = exact.receipt()

    assert receipt["schema"] == EXACT_ADJOINT_INTERVENTION_RECEIPT_SCHEMA
    assert receipt["intervention_config"] == config.to_dict()
    assert set(receipt["lesion_losses"]) == {"1", "3"}
    assert len(receipt["stopping_teacher_receipts"]) == 1
    assert validate_exact_adjoint_live_path_receipt(receipt) == receipt
    assert exact.value == pytest.approx(float(full_value), abs=2e-5)
    assert set(full_flat) == set(exact_flat)
    for name in full_flat:
        difference = float(mx.max(mx.abs(full_flat[name] - exact_flat[name])))
        assert difference < 3e-4, name


def test_disabled_intervention_terms_do_not_constrain_recurrent_depth():
    ExactAdjointInterventionConfig(
        lesion_steps=(1,),
        causality_weight=0.5,
        stopping_steps=(1, 2),
        stopping_weight=0.0,
    ).validate_depth(1)
    ExactAdjointInterventionConfig(
        lesion_steps=(1, 3),
        causality_weight=0.0,
        stopping_steps=(1, 2),
        stopping_weight=0.5,
    ).validate_depth(2)


def test_intervention_receipt_rejects_sequence_coercion_as_teacher_mapping():
    model = _model()
    spec = _spec(recurrent_steps=2)
    receipt = exact_adjoint_composite_live_path_value_and_grad(
        model,
        PROMPT,
        ANSWER,
        spec=spec,
        trajectory_config=None,
        intervention_config=ExactAdjointInterventionConfig(
            lesion_steps=(1,),
            causality_weight=0.0,
            stopping_steps=(1, 2),
            stopping_weight=0.5,
        ),
        policy_sha256=recurrent_policy_sha256(model, spec),
        token_loss_weights=(0.0,) * len(ANSWER),
    ).receipt()
    attacked = copy.deepcopy(receipt)
    attacked["stopping_teacher_receipts"][0] = list(
        attacked["stopping_teacher_receipts"][0].items()
    )
    payload = {key: value for key, value in attacked.items() if key != "receipt_sha256"}
    attacked["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()

    with pytest.raises(ValueError, match="teacher cardinality"):
        validate_exact_adjoint_live_path_receipt(attacked)


def test_intervention_receipt_rejects_resealed_measurement_boundary_change():
    model = _model()
    spec = _spec(recurrent_steps=2)
    receipt = exact_adjoint_composite_live_path_value_and_grad(
        model,
        PROMPT,
        ANSWER,
        spec=spec,
        trajectory_config=None,
        intervention_config=ExactAdjointInterventionConfig(
            lesion_steps=(1,),
            causality_weight=0.5,
            stopping_steps=(1, 2),
            stopping_weight=0.5,
        ),
        policy_sha256=recurrent_policy_sha256(model, spec),
        token_loss_weights=(0.0,) * len(ANSWER),
    ).receipt()
    attacked = copy.deepcopy(receipt)
    attacked["measurement_trust_boundary"] = "independently_replayed"
    payload = {key: value for key, value in attacked.items() if key != "receipt_sha256"}
    attacked["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()

    with pytest.raises(ValueError, match="measurement boundary is invalid"):
        validate_exact_adjoint_live_path_receipt(attacked)


def test_exact_adjoint_trajectory_selects_one_producing_branch():
    model = _model()
    spec = _spec(
        recurrent_steps=2,
        branch_roles=("constructive_solution", "counterexample_search"),
    )
    config = ExactAdjointTrajectoryConfig(
        probe_steps=(1, 2),
        improvement_weight=1.0,
        improvement_margin=0.5,
    )

    result = exact_adjoint_trajectory_live_path_value_and_grad(
        model,
        PROMPT,
        ANSWER,
        spec=spec,
        trajectory_config=config,
        policy_sha256=recurrent_policy_sha256(model, spec),
        branch_index=1,
        diversity_weight=0.4,
        diversity_target_cos=0.5,
        token_loss_weights=(0.0,) * len(ANSWER),
    )
    mx.eval(result.gradients)

    assert result.branch_indices == (1,)
    assert set(result.step_losses) == {1, 2}
    assert all(len(losses) == 1 for losses in result.step_losses.values())
    assert result.receipt()["trajectory_config"] == config.to_dict()
    assert result.receipt()["execution_branch_count"] == 2
    assert validate_exact_adjoint_live_path_receipt(result.receipt()) == result.receipt()


def test_exact_adjoint_composite_supports_diversity_without_fake_trajectory():
    model = _model()
    spec = _spec(
        recurrent_steps=2,
        branch_roles=("constructive_solution", "counterexample_search"),
    )

    result = exact_adjoint_composite_live_path_value_and_grad(
        model,
        PROMPT,
        ANSWER,
        spec=spec,
        trajectory_config=None,
        policy_sha256=recurrent_policy_sha256(model, spec),
        branch_index=None,
        diversity_weight=0.4,
        diversity_target_cos=0.5,
        token_loss_weights=(0.0,) * len(ANSWER),
    )

    assert result.branch_indices == (0, 1)
    assert result.trajectory_config is None
    assert result.terminal_value == pytest.approx(0.0, abs=1e-12)
    assert validate_exact_adjoint_live_path_receipt(result.receipt()) == result.receipt()


def test_exact_adjoint_receipt_binds_every_proof_input():
    model = _model()
    spec = _spec(
        recurrent_steps=2,
        branch_roles=("constructive_solution", "counterexample_search"),
        decode_bridge_policy="assistant_answer",
    )
    bridge = (19, 29)
    policy_sha256 = recurrent_policy_sha256(model, spec)
    result = exact_adjoint_composite_live_path_value_and_grad(
        model,
        PROMPT,
        ANSWER,
        spec=spec,
        trajectory_config=None,
        policy_sha256=policy_sha256,
        bridge_tokens=bridge,
        branch_index=0,
        token_loss_weights=(0.25, 0.5, 1.0),
    )
    receipt = result.receipt()

    def token_sha256(tokens):
        return hashlib.sha256(
            json.dumps(list(tokens), separators=(",", ":"), allow_nan=False).encode("ascii")
        ).hexdigest()

    assert receipt["policy_sha256"] == policy_sha256
    assert receipt["prompt_tokens_sha256"] == token_sha256(PROMPT)
    assert receipt["answer_tokens_sha256"] == token_sha256(ANSWER)
    assert receipt["bridge_tokens_sha256"] == token_sha256(bridge)
    assert receipt["token_loss_weights"] == [0.25, 0.5, 1.0]
    assert validate_exact_adjoint_live_path_receipt(receipt) == receipt

    attacked = copy.deepcopy(receipt)
    attacked["answer_tokens_sha256"] = "0" * 64
    attacked["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in attacked.items() if key != "receipt_sha256"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    with pytest.raises(ValueError, match="objective input commitment mismatch"):
        validate_exact_adjoint_live_path_receipt(attacked)


def test_exact_adjoint_proof_receipt_rejects_signed_token_weights():
    model = _model()
    spec = _spec()
    with pytest.raises(ValueError, match="must be non-negative"):
        exact_adjoint_composite_live_path_value_and_grad(
            model,
            PROMPT,
            ANSWER,
            spec=spec,
            trajectory_config=None,
            policy_sha256=recurrent_policy_sha256(model, spec),
            token_loss_weights=(0.0, -0.1, 0.0),
        )


def test_exact_adjoint_trajectory_receipt_rejects_resealed_false_arithmetic():
    model = _model()
    spec = _spec()
    config = ExactAdjointTrajectoryConfig(
        probe_steps=(1, 2),
        improvement_weight=1.0,
        displacement_weight=0.5,
        displacement_floor=0.99,
    )
    receipt = exact_adjoint_trajectory_live_path_value_and_grad(
        model,
        PROMPT,
        ANSWER,
        spec=spec,
        trajectory_config=config,
        policy_sha256=recurrent_policy_sha256(model, spec),
        token_loss_weights=(0.0,) * len(ANSWER),
    ).receipt()
    attacked = copy.deepcopy(receipt)
    attacked["trajectory_values"]["improvement"] += 1.0
    attacked["value"] += 1.0
    payload = {key: value for key, value in attacked.items() if key != "receipt_sha256"}
    attacked["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()

    with pytest.raises(ValueError, match="improvement term does not replay"):
        validate_exact_adjoint_live_path_receipt(attacked)


def test_exact_adjoint_trajectory_receipt_rejects_out_of_domain_atoms():
    model = _model()
    spec = _spec()
    receipt = exact_adjoint_trajectory_live_path_value_and_grad(
        model,
        PROMPT,
        ANSWER,
        spec=spec,
        trajectory_config=ExactAdjointTrajectoryConfig(
            probe_steps=(1, 2),
            improvement_weight=1.0,
            displacement_weight=0.5,
            displacement_floor=0.99,
            oscillation_weight=0.5,
        ),
        policy_sha256=recurrent_policy_sha256(model, spec),
        token_loss_weights=(0.0,) * len(ANSWER),
    ).receipt()

    def reseal(attacked):
        payload = {key: value for key, value in attacked.items() if key != "receipt_sha256"}
        attacked["receipt_sha256"] = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
        return attacked

    mutations = (
        (
            lambda attacked: attacked["step_losses"].update(
                {"01": attacked["step_losses"].pop("1")}
            ),
            "step-loss row",
        ),
        (
            lambda attacked: attacked["step_losses"]["1"].__setitem__(0, -0.1),
            "step loss is negative",
        ),
        (
            lambda attacked: attacked["displacements"].__setitem__(0, -0.1),
            "displacement is negative",
        ),
        (
            lambda attacked: attacked["oscillation_cosines"].__setitem__(0, 1.1),
            "outside cosine range",
        ),
    )
    for mutate, error in mutations:
        attacked = copy.deepcopy(receipt)
        mutate(attacked)
        with pytest.raises(ValueError, match=error):
            validate_exact_adjoint_live_path_receipt(reseal(attacked))


def test_two_branch_exchange_and_depth_curriculum_are_live():
    model = _model()
    spec = _spec(branch_roles=["constructive_solution", "counterexample_search"])
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
