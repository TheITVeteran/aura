"""Verifier gradients pass through the actual recurrent hidden-state path."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.execution_spec import (  # noqa: E402
    RLCExecutionSpec,
)
from core.learning.recurrence_native_objective_v2 import (  # noqa: E402
    LivePathForward,
    live_path_branch_answer_ce_trail,
)
from core.learning.recurrent_grpo import (  # noqa: E402
    RecurrentGroupClipAdmissionError,
    RecurrentGRPOConfig,
    RecurrentSamplingConfig,
    branch_token_logprobs,
    clipped_recurrent_grpo_objective,
    cortex_config_from_execution_spec,
    exact_adjoint_sampled_group_value_and_grad,
    exact_adjoint_verifier_group_value_and_grad,
    recurrent_completion_token_logprobs,
    recurrent_policy_sample_from_causal_pair,
    recurrent_sampling_rng_root_sha256,
    sample_final_recurrent_transition_pair,
    sample_recurrent_completion,
    validate_causal_recurrent_transition_pair_receipt,
    validate_recurrent_policy_sample_receipt,
    verifier_group_objective,
)
from tools.recurrence_native_train_v2 import _wrap_window_layers  # noqa: E402
from tools.train_grpo import sample_recurrent_group  # noqa: E402


def _model(seed: int = 311) -> Model:
    mx.random.seed(seed)
    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=16,
            num_hidden_layers=4,
            intermediate_size=32,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=32,
            num_key_value_heads=2,
            max_position_embeddings=64,
            rope_theta=10000.0,
        )
    )
    mx.eval(model.parameters())
    return model


def _prepared(seed: int = 311) -> Model:
    model = _model(seed)
    wrapped = _wrap_window_layers(
        model,
        rank=2,
        targets=("o_proj",),
        prelude_frac=0.25,
        coda_frac=0.25,
    )
    assert wrapped
    return model


def _set_adapter_delta(model: Model, value: float) -> None:
    for layer_index in (1, 2):
        adapter = model.model.layers[layer_index].self_attn.o_proj
        adapter.lora_b = mx.ones_like(adapter.lora_b) * value
    mx.eval(model.trainable_parameters())


def _spec(
    depth: int = 2,
    branch_roles: tuple[str, ...] = ("constructive_solution",),
) -> RLCExecutionSpec:
    return RLCExecutionSpec(
        n_slots=2,
        branch_roles=branch_roles,
        exchange_interval=1,
        recurrent_steps=depth,
        alpha=0.35,
        prelude_frac=0.25,
        coda_frac=0.25,
    )


def test_branch_logprobs_are_bound_to_exact_tokens_and_branch():
    logits = mx.zeros((1, 2, 8))
    forward = LivePathForward(
        branch_logits=(logits,),
        branch_states=(mx.zeros((1, 2, 4)),),
        exchanges=0,
        prompt_tokens=3,
        answer_tokens=2,
        bridge_tokens=0,
    )

    logprobs = branch_token_logprobs(forward, [2, 3], branch_index=0)

    assert logprobs.shape == (2,)
    assert all(float(value) == pytest.approx(-math.log(8.0)) for value in logprobs)
    with pytest.raises(ValueError, match="branch_index"):
        branch_token_logprobs(forward, [2, 3], branch_index=1)
    with pytest.raises(ValueError, match="do not align"):
        branch_token_logprobs(forward, [2], branch_index=0)


def test_clipped_objective_is_token_normalized_and_reference_anchored():
    old = [mx.array([-2.0, -2.0]), mx.array([-3.0])]
    policy = [old[0] + math.log(1.5), old[1] + math.log(0.5)]
    reference = [mx.array([-2.0, -2.0]), mx.array([-3.0])]
    objective = clipped_recurrent_grpo_objective(
        policy,
        old,
        [1.0, -1.0],
        reference_logprobs=reference,
        config=RecurrentGRPOConfig(clip_epsilon=0.2, kl_coefficient=0.1),
    )
    receipt = objective.receipt()

    assert receipt["policy_loss"] == pytest.approx(-0.2, rel=1e-5)
    assert receipt["clip_fraction"] == 1.0
    assert receipt["reference_kl"] > 0.0
    assert receipt["completion_count"] == 2
    assert receipt["token_count"] == 3


def test_adapter_disabled_reference_keeps_recurrence_but_changes_policy():
    model = _prepared()
    _set_adapter_delta(model, 0.05)

    on = recurrent_completion_token_logprobs(
        model, [5, 9, 17], [7, 11], spec=_spec(), branch_index=0
    )
    off = recurrent_completion_token_logprobs(
        model,
        [5, 9, 17],
        [7, 11],
        spec=_spec(),
        branch_index=0,
        adapters_on=False,
    )
    mx.eval(on, off)

    assert on.shape == off.shape == (2,)
    assert not bool(mx.array_equal(on, off))


def test_execution_spec_maps_to_fixed_live_training_graph():
    spec = _spec(
        depth=3,
        branch_roles=("constructive_solution", "critical_audit"),
    )
    config = cortex_config_from_execution_spec(
        spec,
        sampling=RecurrentSamplingConfig(max_tokens=4),
    )

    assert config.workspace.n_slots == spec.n_slots
    assert config.workspace.roles == spec.slot_roles
    assert config.recurrence.max_steps == spec.recurrent_steps
    assert config.recurrence.min_steps == spec.recurrent_steps
    assert config.recurrence.fixed_depth is True
    assert config.branches.roles == spec.branch_roles
    assert config.branches.exchange_interval == spec.exchange_interval
    assert config.latent_opt.enabled is False
    assert config.fast_weights.enabled is False
    assert config.allow_vanilla_fallback is False
    assert config.decode_temperature == 1.0
    assert config.decode_top_p == 1.0

    with pytest.raises(ValueError, match="temperature=1"):
        RecurrentSamplingConfig(temperature=0.8)


def test_cached_recurrent_sampler_is_admitted_by_differentiable_policy():
    model = _prepared(seed=919)
    _set_adapter_delta(model, 0.02)
    spec = _spec(
        depth=2,
        branch_roles=("constructive_solution", "critical_audit"),
    )

    sample = sample_recurrent_completion(
        model,
        [5, 9, 17],
        spec=spec,
        seed=117,
        sampling=RecurrentSamplingConfig(max_tokens=3),
        episode_id="engine-sample-test-117",
    )
    receipt = sample.receipt()
    validated = validate_recurrent_policy_sample_receipt(receipt)

    assert len(sample.tokens) == 3
    assert len(sample.behavior_logprobs) == len(sample.differentiable_logprobs) == 3
    assert sample.branch_index in (0, 1)
    assert sample.behavior_admitted is True
    assert receipt["behavior_admitted"] is True
    assert receipt["seed"] == 117
    assert len(receipt["prompt_tokens_sha256"]) == 64
    assert len(receipt["policy_sha256"]) == 64
    assert len(receipt["behavior_logprobs_sha256"]) == 64
    assert len(receipt["differentiable_logprobs_sha256"]) == 64
    assert receipt["cached_params_unchanged"] is True
    assert receipt["cached_nonparametric_memory_status"] == (
        "disabled_by_policy"
    )
    assert receipt["cached_recurrence_adapter"]["active"] is True
    assert validated["sample_kind"] == "engine_episode"
    assert validated["episode_id"] == "engine-sample-test-117"
    assert validated["episode_receipt"]["episode_id"] == sample.episode_id


def test_causal_pair_decodes_one_frozen_edge_under_matched_randomness():
    model = _prepared(seed=927)
    _set_adapter_delta(model, 0.02)
    spec = _spec(
        depth=3,
        branch_roles=("constructive_solution", "critical_audit"),
    )

    sampling = RecurrentSamplingConfig(
        max_tokens=3,
        max_clipped_token_fraction=1.0,
    )
    pair = sample_final_recurrent_transition_pair(
        model,
        [5, 9, 17],
        spec=spec,
        branch_index=1,
        seed=117,
        sampling=sampling,
        episode_id="causal-edge-test-117",
    )
    receipt = validate_causal_recurrent_transition_pair_receipt(pair.receipt())
    sample = recurrent_policy_sample_from_causal_pair(
        pair,
        sampling=sampling,
    )
    sample_receipt = validate_recurrent_policy_sample_receipt(sample.receipt())

    assert pair.transition.transition_index == 2
    assert pair.parent.depth == 2
    assert pair.child.depth == 3
    assert pair.parent.seed == pair.child.seed == 117
    assert len(pair.parent.tokens) == len(pair.child.tokens) == 3
    assert pair.parent.state_sha256 == pair.transition.parent_branch_sha256s[1]
    assert pair.child.state_sha256 == pair.transition.child_branch_sha256s[1]
    assert pair.child_behavior_admitted is True
    assert receipt["fixed_token_budget"] == 3
    assert receipt["episode_id"] == "causal-edge-test-117"
    assert receipt["runtime_integrity"]["verdict"][
        "engine_measurements_complete"
    ] is True
    assert receipt["recurrence_adapter"]["active"] is True
    assert sample_receipt["sample_kind"] == "causal_final_transition"
    assert sample.tokens == pair.child.tokens


def test_causal_sample_receipt_rejects_trace_substitution():
    pair = sample_final_recurrent_transition_pair(
        _prepared(seed=928),
        [5, 9, 17],
        spec=_spec(depth=2),
        branch_index=0,
        seed=43,
        sampling=RecurrentSamplingConfig(max_tokens=2),
        episode_id="causal-trace-tamper-43",
    )
    receipt = recurrent_policy_sample_from_causal_pair(pair).receipt()
    receipt["tokens"][0] = (receipt["tokens"][0] + 1) % 32

    with pytest.raises(ValueError, match="trace"):
        validate_recurrent_policy_sample_receipt(receipt)


def test_causal_samples_enter_the_exact_adjoint_without_identity_loss():
    model = _prepared(seed=930)
    _set_adapter_delta(model, 0.02)
    prompt = [5, 9, 17]
    spec = _spec(
        depth=2,
        branch_roles=("constructive_solution", "critical_audit"),
    )
    sampling = RecurrentSamplingConfig(
        max_tokens=2,
        max_clipped_token_fraction=1.0,
        max_old_policy_approx_kl=1.0,
    )
    samples = [
        recurrent_policy_sample_from_causal_pair(
            sample_final_recurrent_transition_pair(
                model,
                prompt,
                spec=spec,
                branch_index=branch,
                seed=seed,
                sampling=sampling,
                episode_id=f"causal-adjoint-{branch}-{seed}",
            ),
            sampling=sampling,
        )
        for branch, seed in ((0, 47), (1, 53))
    ]

    result = exact_adjoint_sampled_group_value_and_grad(
        model,
        prompt,
        samples,
        [1.0, 0.0],
        spec=spec,
        config=RecurrentGRPOConfig(
            kl_coefficient=0.0,
            max_initial_clip_fraction=1.0,
            max_initial_old_policy_approx_kl=1.0,
        ),
    )

    assert result.gradients is not None
    assert result.completion_count == 2
    assert result.branch_indices == (0, 1)


def test_causal_pair_rejects_branch_state_substitution():
    pair = sample_final_recurrent_transition_pair(
        _prepared(seed=929),
        [5, 9, 17],
        spec=_spec(depth=2),
        branch_index=0,
        seed=41,
        sampling=RecurrentSamplingConfig(max_tokens=2),
    )
    attacked = pair.receipt()
    attacked["child"]["state_sha256"] = attacked["parent"]["state_sha256"]

    with pytest.raises(ValueError, match="child"):
        validate_causal_recurrent_transition_pair_receipt(attacked)


def test_trainer_group_uses_tokenizer_and_distinct_bound_seeds():
    model = _prepared(seed=941)

    class Tokenizer:
        eos_token_id = None

        @staticmethod
        def apply_chat_template(messages, **_kwargs):
            assert messages[0]["content"].endswith("\n\nsolve")
            assert "FINAL_ANSWER: {JSON object}" in messages[0]["content"]
            return "rendered"

        @staticmethod
        def encode(text, **_kwargs):
            if text == "rendered":
                return [5, 9, 17]
            assert _kwargs.get("add_special_tokens") is False
            assert isinstance(text, str) and text
            return [1 + (sum(text.encode("utf-8")) % 61)]

        @staticmethod
        def decode(tokens):
            return " ".join(str(token) for token in tokens)

    class Task:
        task_id = "trainer-group-task"
        prompt = "solve"

    prompt, samples, completions = sample_recurrent_group(
        model,
        Tokenizer(),
        Task(),
        spec=_spec(depth=2),
        size=2,
        max_tokens=2,
        seed=51,
        sampling_config=RecurrentSamplingConfig(
            max_tokens=2,
            max_abs_logprob_drift=100.0,
            max_mean_abs_logprob_drift=100.0,
            clip_epsilon=1.0,
            max_clipped_token_fraction=1.0,
            max_old_policy_approx_kl=100.0,
        ),
    )

    assert prompt == [5, 9, 17]
    assert len(samples) == len(completions) == 2
    assert samples[0].seed != samples[1].seed
    assert all(sample.behavior_admitted for sample in samples)
    assert completions == [
        " ".join(str(token) for token in sample.tokens) for sample in samples
    ]


def test_trainer_executes_exact_pre_admitted_causal_group_without_retries():
    from core.learning.verified_transition_group_admission import (
        sampling_config_sha256,
    )
    from core.learning.verified_transition_trainer import (
        VerifiedTransitionSamplingEntry,
        VerifiedTransitionSamplingPlan,
    )

    model = _prepared(seed=943)
    spec = _spec(
        depth=2,
        branch_roles=("constructive_solution", "critical_audit"),
    )
    sampling = RecurrentSamplingConfig(
        max_tokens=2,
        max_clipped_token_fraction=1.0,
        max_old_policy_approx_kl=1.0,
    )

    class Tokenizer:
        @staticmethod
        def apply_chat_template(_messages, **_kwargs):
            return "rendered"

        @staticmethod
        def encode(_text, **_kwargs):
            return [5, 9, 17]

        @staticmethod
        def decode(tokens):
            return " ".join(str(token) for token in tokens)

    class Task:
        task_id = "signed-causal-task"
        prompt = "solve"

    class Provider:
        calls = 0

        def sampling_plan(
            self, *, sequence, task, prompt_tokens, policy_sha256
        ):
            self.calls += 1
            entries = []
            prompt_sha256 = hashlib.sha256(
                json.dumps(
                    list(prompt_tokens), separators=(",", ":")
                ).encode("ascii")
            ).hexdigest()
            for branch, seed in ((0, 61), (1, 67)):
                episode_id = f"signed-causal-{branch}-{seed}"
                template = SimpleNamespace(sampling_config=sampling)
                entries.append(
                    VerifiedTransitionSamplingEntry(
                        episode_id=episode_id,
                        rng_root_sha256=recurrent_sampling_rng_root_sha256(
                            episode_id=episode_id,
                            prompt_tokens_sha256=prompt_sha256,
                            policy_sha256=policy_sha256,
                            execution_spec_sha256=spec.sha256,
                            branch_index=branch,
                            seed=seed,
                            sampling_config=sampling,
                        ),
                        producing_branch_index=branch,
                        sample_seed=seed,
                        sampling_config_sha256=sampling_config_sha256(template),
                    )
                )
            return VerifiedTransitionSamplingPlan(
                campaign_sequence=sequence,
                group_manifest_sha256="a" * 64,
                task_id=task.task_id,
                policy_sha256=policy_sha256,
                prompt_tokens_sha256=prompt_sha256,
                execution_spec_sha256=spec.sha256,
                entries=tuple(entries),
                sampling_config=sampling.to_dict(),
            )

    provider = Provider()
    prompt, samples, _completions = sample_recurrent_group(
        model,
        Tokenizer(),
        Task(),
        spec=spec,
        size=2,
        max_tokens=2,
        seed=999,
        sampling_config=sampling,
        verified_group_provider=provider,
        campaign_sequence=0,
    )

    assert prompt == [5, 9, 17]
    assert provider.calls == 1
    assert [sample.episode_id for sample in samples] == [
        "signed-causal-0-61",
        "signed-causal-1-67",
    ]
    assert [sample.seed for sample in samples] == [61, 67]
    assert all(sample.sample_kind == "causal_final_transition" for sample in samples)


def test_live_path_branch_answer_ce_trail_scores_each_recurrent_step():
    model = _prepared(seed=949)
    spec = _spec(
        depth=3,
        branch_roles=("constructive_solution", "critical_audit"),
    )

    trail = live_path_branch_answer_ce_trail(
        model,
        [5, 9, 17],
        [7, 11],
        spec=spec,
        branch_index=1,
    )

    assert len(trail) == 3
    assert all(math.isfinite(value) and value >= 0.0 for value in trail)
    with pytest.raises(ValueError, match="branch_index"):
        live_path_branch_answer_ce_trail(
            model,
            [5, 9, 17],
            [7, 11],
            spec=spec,
            branch_index=2,
        )


def test_sampled_verifier_update_improves_preference_without_base_drift():
    model = _prepared(seed=977)
    _set_adapter_delta(model, 0.02)
    prompt = [5, 9, 17]
    spec = _spec(depth=2)
    sampling = RecurrentSamplingConfig(max_tokens=2)
    samples = []
    for seed in range(200, 216):
        candidate = sample_recurrent_completion(
            model,
            prompt,
            spec=spec,
            seed=seed,
            sampling=sampling,
        )
        if not samples or sum(candidate.tokens) != sum(samples[0].tokens):
            samples.append(candidate)
        if len(samples) == 2:
            break
    assert len(samples) == 2

    # A deterministic external program ranks lower token sums higher. The
    # production trainer substitutes each task's exact correctness verifier;
    # this synthetic rule keeps the contract test contamination-free.
    rewards = tuple(-float(sum(sample.tokens)) for sample in samples)
    assert rewards[0] != rewards[1]
    branches = tuple(sample.branch_index for sample in samples)
    completions = tuple(sample.tokens for sample in samples)

    def policy_scores(*, adapters_on: bool) -> tuple[float, float]:
        scores = []
        for completion, branch in zip(completions, branches, strict=True):
            values = recurrent_completion_token_logprobs(
                model,
                prompt,
                completion,
                spec=spec,
                branch_index=branch,
                adapters_on=adapters_on,
            )
            mx.eval(values)
            scores.append(float(mx.sum(values)))
        return tuple(scores)

    policy_before = policy_scores(adapters_on=True)
    reference_before = policy_scores(adapters_on=False)
    standard_before = model(mx.array([prompt]))
    mx.eval(standard_before)

    result = exact_adjoint_sampled_group_value_and_grad(
        model,
        prompt,
        samples,
        rewards,
        spec=spec,
        config=RecurrentGRPOConfig(kl_coefficient=0.0),
    )
    assert result.gradients is not None
    optimizer = optim.SGD(learning_rate=0.02)
    optimizer.update(model, result.gradients)
    mx.eval(model.parameters(), optimizer.state)

    policy_after = policy_scores(adapters_on=True)
    reference_after = policy_scores(adapters_on=False)
    standard_after = model(mx.array([prompt]))
    mx.eval(standard_after)
    winner = rewards.index(max(rewards))
    loser = 1 - winner

    assert (
        policy_after[winner] - policy_after[loser]
        > policy_before[winner] - policy_before[loser]
    )
    assert reference_after == pytest.approx(reference_before, abs=1e-7)
    assert bool(mx.array_equal(standard_after, standard_before))


def test_sampled_group_rejects_excess_clip_fraction_before_adjoint():
    model = _prepared(seed=983)
    prompt = [5, 9, 17]
    spec = _spec(depth=2)
    samples = [
        sample_recurrent_completion(
            model,
            prompt,
            spec=spec,
            seed=seed,
            sampling=RecurrentSamplingConfig(max_tokens=2),
        )
        for seed in (31, 37)
    ]
    shifted = []
    for sample in samples:
        behavior = list(sample.differentiable_logprobs)
        behavior[0] -= 0.19
        differences = [
            abs(left - right)
            for left, right in zip(
                behavior, sample.differentiable_logprobs, strict=True
            )
        ]
        changed = replace(
            sample,
            behavior_logprobs=tuple(behavior),
            max_abs_logprob_drift=max(differences),
            mean_abs_logprob_drift=sum(differences) / len(differences),
            max_abs_logprob_drift_token_index=differences.index(
                max(differences)
            ),
            clipped_token_fraction=0.5,
            old_policy_approx_kl=sum(
                (math.exp(target - cached) - 1.0) - (target - cached)
                for target, cached in zip(
                    sample.differentiable_logprobs,
                    behavior,
                    strict=True,
                )
            )
            / len(behavior),
            behavior_admitted=True,
            sampling_config=replace(
                sample.sampling_config,
                max_mean_abs_logprob_drift=0.1,
                max_clipped_token_fraction=1.0,
            ),
        )
        shifted.append(
            replace(
                changed,
                rng_root_sha256=recurrent_sampling_rng_root_sha256(
                    episode_id=changed.episode_id,
                    prompt_tokens_sha256=changed.prompt_tokens_sha256,
                    policy_sha256=changed.policy_sha256,
                    execution_spec_sha256=changed.execution_spec_sha256,
                    branch_index=changed.branch_index,
                    seed=changed.seed,
                    sampling_config=changed.sampling_config,
                ),
            )
        )

    with pytest.raises(RecurrentGroupClipAdmissionError) as captured:
        exact_adjoint_sampled_group_value_and_grad(
            model,
            prompt,
            shifted,
            [1.0, 0.0],
            spec=spec,
            config=RecurrentGRPOConfig(max_initial_clip_fraction=0.2),
        )

    assert captured.value.clip_fraction == pytest.approx(0.5)

    with pytest.raises(RuntimeError, match="PPO KL admission"):
        exact_adjoint_sampled_group_value_and_grad(
            model,
            prompt,
            shifted,
            [1.0, 0.0],
            spec=spec,
            config=RecurrentGRPOConfig(
                max_initial_clip_fraction=1.0,
                max_initial_old_policy_approx_kl=1e-6,
            ),
        )


def test_verifier_advantage_produces_finite_nonzero_recurrent_gradients():
    model = _prepared(seed=617)
    prompt = [5, 9, 17]
    completions = ([7, 11], [8, 12])
    spec = _spec(depth=2)
    old = [
        mx.stop_gradient(
            recurrent_completion_token_logprobs(
                model, prompt, tokens, spec=spec, branch_index=0
            )
        )
        for tokens in completions
    ]
    reference = [
        mx.stop_gradient(
            recurrent_completion_token_logprobs(
                model,
                prompt,
                tokens,
                spec=spec,
                branch_index=0,
                adapters_on=False,
            )
        )
        for tokens in completions
    ]
    mx.eval(old, reference)

    def loss_fn(current_model):
        policy = [
            recurrent_completion_token_logprobs(
                current_model, prompt, tokens, spec=spec, branch_index=0
            )
            for tokens in completions
        ]
        objective, report = verifier_group_objective(
            policy,
            old,
            [1.0, 0.0],
            reference_logprobs=reference,
            config=RecurrentGRPOConfig(kl_coefficient=0.04),
        )
        assert report["degenerate"] is False
        return objective.loss

    value, gradients = nn.value_and_grad(model, loss_fn)(model)
    flattened = [gradient for _name, gradient in tree_flatten(gradients)]
    mx.eval(value, gradients)

    assert math.isfinite(float(value))
    assert flattened
    assert all(bool(mx.all(mx.isfinite(gradient))) for gradient in flattened)
    assert any(float(mx.max(mx.abs(gradient))) > 0.0 for gradient in flattened)


def test_exact_adjoint_group_matches_monolithic_recurrent_grpo_gradient():
    monolithic = _prepared(seed=811)
    streamed = _prepared(seed=811)
    _set_adapter_delta(monolithic, 0.02)
    _set_adapter_delta(streamed, 0.02)
    prompt = [5, 9, 17]
    completions = ([7, 11], [8, 12])
    branches = (1, 0)
    rewards = (1.0, 0.0)
    spec = _spec(
        depth=2,
        branch_roles=("constructive_solution", "critical_audit"),
    )
    config = RecurrentGRPOConfig(kl_coefficient=0.04)

    old = [
        mx.stop_gradient(
            recurrent_completion_token_logprobs(
                monolithic, prompt, tokens, spec=spec, branch_index=branch
            )
        )
        for tokens, branch in zip(completions, branches, strict=True)
    ]
    behavior = [
        old[0] + mx.array([-0.25, 0.05]),
        old[1] + mx.array([0.0, 0.10]),
    ]
    reference = [
        mx.stop_gradient(
            recurrent_completion_token_logprobs(
                monolithic,
                prompt,
                tokens,
                spec=spec,
                branch_index=branch,
                adapters_on=False,
            )
        )
        for tokens, branch in zip(completions, branches, strict=True)
    ]
    mx.eval(old, behavior, reference)

    def loss_fn(current_model):
        policy = [
            recurrent_completion_token_logprobs(
                current_model, prompt, tokens, spec=spec, branch_index=branch
            )
            for tokens, branch in zip(completions, branches, strict=True)
        ]
        objective, _report = verifier_group_objective(
            policy,
            behavior,
            rewards,
            reference_logprobs=reference,
            config=config,
        )
        return objective.loss

    _value, monolithic_gradients = nn.value_and_grad(monolithic, loss_fn)(monolithic)
    streamed_result = exact_adjoint_verifier_group_value_and_grad(
        streamed,
        prompt,
        completions,
        branches,
        rewards,
        spec=spec,
        behavior_logprobs=[
            [float(value) for value in values] for values in behavior
        ],
        config=config,
    )
    mx.eval(monolithic_gradients, streamed_result.gradients)
    monolithic_flat = dict(tree_flatten(monolithic_gradients))
    streamed_flat = dict(tree_flatten(streamed_result.gradients))

    assert set(monolithic_flat) == set(streamed_flat)
    assert streamed_result.receipt()["has_gradient"] is True
    assert streamed_result.receipt()["clip_fraction"] == pytest.approx(0.25)
    for key in monolithic_flat:
        difference = float(
            mx.max(mx.abs(monolithic_flat[key] - streamed_flat[key]))
        )
        assert difference < 2e-4, key
